# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC ## 3.2b GNN Hyperparameter Tuning — Parallel GPU (Optuna + MlflowSparkStudy)
# MAGIC
# MAGIC Parallel Bayesian hyperparameter optimization for the RetweetGNN model.
# MAGIC Each trial runs on a separate Spark worker with its own GPU.
# MAGIC
# MAGIC **Search space** (same architecture, same epochs as 3.1):
# MAGIC - Learning rate (log-uniform)
# MAGIC - Weight decay (log-uniform)
# MAGIC - Batch size
# MAGIC - Dropout rate
# MAGIC - Drop-edge rate
# MAGIC - LR warmup epochs
# MAGIC - Gradient accumulation steps
# MAGIC - LR schedule (cosine vs linear decay)
# MAGIC
# MAGIC **Strategy**: Optuna TPE (Tree of Parzen Estimators) with `MlflowSparkStudy` for
# MAGIC automatic distribution across GPU workers. Each trial is logged to MLflow **immediately**
# MAGIC when it completes (real-time monitoring).
# MAGIC
# MAGIC **Why Optuna over Hyperopt?**
# MAGIC - Hyperopt is deprecated and removed from DBR > 16.4 LTS ML
# MAGIC - Optuna + MlflowSparkStudy logs each trial to MLflow as it finishes (no waiting for fmin() to return)
# MAGIC - Better pruning, visualization, and active community support

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --upgrade torch-geometric-signed-directed networkx optuna mlflow

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Parameters
import os

# ============================================================
# Parameters
# ============================================================
N_USERS = 300  # Total users to sample (proportionally across groups). None = all users.
EXPERIMENT_TAG = "v3_full"  # Must match 3.1 to reuse the same user sample

# Paths
sources_path = "/serafin/pcelayes/repos/sna_classifier/"
DATA_PATH = "/Workspace/Users/pablo.celayes@bolt.eu/learning/data/sna_classifier"
EMBEDDINGS_PATH = f"{DATA_PATH}/node_embeddings.pt"

# Derived experiment folder (same as 3.1 — reuses user_sample.json)
FINAL_TAG = f"{EXPERIMENT_TAG}_N{N_USERS}" if N_USERS else EXPERIMENT_TAG
EXPERIMENT_DIR = f"./experiments/{FINAL_TAG}"

# Hyperparameter tuning settings
EPOCHS = 5  # Same as 3.1 — fixed across all trials
MAX_EVALS = 20  # Total number of Optuna trials (reduced — narrower search space needs fewer)
PARALLELISM = 3  # Number of trials running in parallel (reduced for memory safety)
RESTART_TUNING = True  # If True, delete previous study and start fresh; if False, resume from existing trials
PATIENCE = 10  # Early stopping patience per trial (shorter than 3.1 to save time)
MAX_VAL_SAMPLES = 30_000  # Cap val samples (reduced from 50k to lower memory pressure)
TRAIN_CHUNK_SIZE = 10  # Users per chunk for on-the-fly GNN sample generation
TARGET_POS_RATE = 0.10  # Oversample positives in train loader to reach this rate. None = disabled.

# Output directory for tuning results
TUNING_DIR = f"./experiments/{FINAL_TAG}_hparam_tuning"
os.makedirs(TUNING_DIR, exist_ok=True)

print(f"Base experiment: {FINAL_TAG}")
print(f"Tuning output:  {TUNING_DIR}")
print(f"Trials: {MAX_EVALS} (parallelism={PARALLELISM})")
print(f"Epochs per trial: {EPOCHS}")

# COMMAND ----------

# DBTITLE 1,Imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import gc
import time
import logging
import warnings
from random import sample, shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data, Batch
import networkx as nx

import optuna
from mlflow.optuna import MlflowStorage
from mlflow.pyspark.optuna.study import MlflowSparkStudy
import mlflow

warnings.filterwarnings('ignore')
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

sys.path.insert(0, str(sources_path))

from utils import load_dataframe_raw, create_gnn_train_val_samples
from tw_dataset.settings import IG_GRAPH_PATH
from gnn_models import (
    PretrainedEmbeddingLookup, RetweetGNN,
    soft_f1_loss, combined_loss, evaluate, train_model,
)

print("Imports OK")

# COMMAND ----------

# DBTITLE 1,Load graph and users (reuse 3.1 sample)
# Load graph
graph = nx.read_graphml(IG_GRAPH_PATH)
print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Load user splits
with open(f"{DATA_PATH}/datasets/user_splits.json") as f:
    user_splits = json.load(f)

TRAIN_GROUPS = ["u_train", "au_train"]
TEST_GROUPS = [g for g in user_splits.keys() if g not in TRAIN_GROUPS]

# Load the saved sample from 3.1 (deterministic, same users)
SAMPLE_PATH = f"{EXPERIMENT_DIR}/user_sample.json"
assert os.path.exists(SAMPLE_PATH), (
    f"User sample not found at {SAMPLE_PATH}. Run notebook 3.1 first to generate it."
)

with open(SAMPLE_PATH) as f:
    saved_sample = json.load(f)
print(f"Loading user sample from {SAMPLE_PATH}")

user_data = {}  # group -> uid -> (X_tr, X_te, y_tr, y_te)
failed_users = []
for group, uids in saved_sample.items():
    user_data[group] = {}
    for uid in uids:
        try:
            data = load_dataframe_raw(uid, sparse=True)
            X_tr, X_te, y_tr, y_te = data
            if X_tr.shape[0] > 0 and X_te.shape[0] > 0 and y_tr.sum() > 0 and y_te.sum() > 0:
                user_data[group][uid] = (X_tr, X_te, y_tr, y_te)
            else:
                failed_users.append((group, uid, "empty data"))
        except Exception as e:
            failed_users.append((group, uid, str(e)))

print(f"Loaded users per group:")
for group in saved_sample:
    print(f"  {group}: {len(user_data[group])}/{len(saved_sample[group])}")
print(f"Total valid: {sum(len(user_data[g]) for g in user_data)}")
if failed_users:
    print(f"Failed: {len(failed_users)}")

# COMMAND ----------

# DBTITLE 1,Build train/val user assignments
# Train: all users from TRAIN_GROUPS (their "train" split)
train_user_items = []
for group in TRAIN_GROUPS:
    for uid in user_data.get(group, {}):
        train_user_items.append((group, uid))

# Val: train-group users' "test" split + test-group users' both splits
val_user_splits = []
for group in TRAIN_GROUPS:
    for uid in user_data.get(group, {}):
        val_user_splits.append((group, uid, "test"))
for group in TEST_GROUPS:
    for uid in user_data.get(group, {}):
        val_user_splits.append((group, uid, "train"))
        val_user_splits.append((group, uid, "test"))

print(f"Train users: {len(train_user_items)} (from {TRAIN_GROUPS})")
print(f"Val user/split combos: {len(val_user_splits)} (from {TRAIN_GROUPS} test + {TEST_GROUPS} both)")

# Compute class weights from train labels
label_counts = {}
for group, uid in train_user_items:
    _, _, y_tr, _ = user_data[group][uid]
    y_arr = np.asarray(y_tr).ravel()
    for label in y_arr:
        label_counts[int(label)] = label_counts.get(int(label), 0) + 1

# Adjust for oversampling target
if TARGET_POS_RATE is not None and 1 in label_counts and 0 in label_counts:
    n_neg = label_counts[0]
    target_n_pos = int(np.ceil(TARGET_POS_RATE * n_neg / (1.0 - TARGET_POS_RATE)))
    if target_n_pos > label_counts[1]:
        label_counts[1] = target_n_pos

total_train_samples = sum(label_counts.values())
n_classes = len(label_counts)
class_weights = torch.tensor(
    [total_train_samples / (n_classes * label_counts[i]) for i in range(n_classes)],
    dtype=torch.float32,
)
print(f"\nLabel counts: {label_counts}")
print(f"Class weights: {class_weights.tolist()}")
print(f"Total train samples: {total_train_samples}")

# COMMAND ----------

# DBTITLE 1,GNN sample helpers (same as 3.1)
from random import shuffle as _shuffle_list

def _sample_to_pyg_data(sample):
    """Convert a raw sample dict from create_gnn_train_val_samples to PyG Data."""
    central_id = int(sample["central_user_id"])
    neighbor_ids = (
        sample["neighbor_ids"].tolist()
        if hasattr(sample["neighbor_ids"], "tolist")
        else list(sample["neighbor_ids"])
    )
    all_ids = [central_id] + neighbor_ids
    num_nodes = len(all_ids)

    user_ids = torch.tensor(all_ids, dtype=torch.long)

    retweeted_raw = sample["retweeted_ids"]
    retweeted_set = set(
        int(r) for r in (retweeted_raw.tolist() if hasattr(retweeted_raw, "tolist") else retweeted_raw)
    )
    retweet_flag = torch.tensor(
        [1.0 if uid in retweeted_set else 0.0 for uid in all_ids],
        dtype=torch.float,
    ).unsqueeze(1)

    ei = sample["edge_index"]
    if hasattr(ei, "__len__") and len(ei) > 0:
        ei_arr = np.array(ei, dtype=np.int64) if not isinstance(ei, np.ndarray) else ei.astype(np.int64)
        edge_index = torch.from_numpy(ei_arr).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    label = torch.tensor(int(sample["label"]), dtype=torch.long)

    return Data(
        user_ids=user_ids,
        retweet_flag=retweet_flag,
        edge_index=edge_index,
        y=label,
        num_nodes=num_nodes,
        central_mask=torch.zeros(num_nodes, dtype=torch.bool).index_fill_(
            0, torch.tensor([0]), True
        ),
    )

print("GNN sample helpers defined.")

# COMMAND ----------

# DBTITLE 1,Objective function: train one trial
def objective(trial):
    """
    Optuna objective function. Each call trains a RetweetGNN from scratch
    with the given hyperparameters and returns -val_f1 (minimization).

    This runs on a Spark worker with its own GPU.
    Each trial is logged to MLflow immediately when it completes.
    """
    import gc
    import time
    import torch
    import torch.nn as nn
    import numpy as np
    from torch_geometric.data import Batch
    from sklearn.metrics import f1_score

    # Unpack broadcast variables (shared memory, not re-serialized per task)
    graph = _graph_bc.value
    user_data = _user_data_bc.value

    trial_start = time.time()

    # Suggest hyperparameters via Optuna
    # Narrowed search space based on experiment results:
    # - lr: 1e-5 to 5e-4 (top trials had lr ~5e-5 to 1e-4; higher lr always poor)
    # - batch_size: removed 64 (causes OOM with parallelism on g5.2xlarge)
    # - dropout: capped at 0.25 (>0.4 always poor in completed trials)
    # - drop_edge_rate: capped at 0.2 (top trials had very low values)
    # - grad_accum_steps: removed 16 (extreme combo slowdown, no quality gain)
    # - warmup_epochs: removed 8 (poor performance in completed trials)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    dropout = trial.suggest_float("dropout", 0.05, 0.25)
    drop_edge_rate = trial.suggest_float("drop_edge_rate", 0.0, 0.2)
    warmup_epochs = trial.suggest_categorical("warmup_epochs", [2, 3, 5])
    grad_accum_steps = trial.suggest_categorical("grad_accum_steps", [2, 4, 8])
    lr_schedule = trial.suggest_categorical("lr_schedule", ["cosine", "linear"])

    # Free stale GPU memory before building model (prevents OOM with parallelism)
    gc.collect()
    torch.cuda.empty_cache()

    # Select GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Nested MLflow run for per-trial training curves and artifacts.
    # Params are logged automatically by MlflowSparkStudy (no need to duplicate);
    # key metrics are also stored as trial user_attrs so they appear in the
    # study-managed run too (queryable without hunting for nested runs).
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        try:
            # Build model
            model = RetweetGNN(
                ff_hidden_dim=64,
                gcn_hidden_dim=64,
                transformer_dim=64,
                transformer_heads=4,
                embeddings_path=EMBEDDINGS_PATH,
                device=device,
                dropout=dropout,
                drop_edge_rate=drop_edge_rate,
            ).to(device)

            # Gate at 0.0 (balanced start)
            with torch.no_grad():
                model.gate_param.fill_(0.0)

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            # LR scheduler
            total_steps_est = (total_train_samples // (batch_size * grad_accum_steps)) * EPOCHS
            warmup_steps = (total_train_samples // (batch_size * grad_accum_steps)) * warmup_epochs

            if lr_schedule == "cosine":
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / max(warmup_steps, 1)
                    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
                    return 0.5 * (1.0 + np.cos(np.pi * progress))
            else:  # linear decay
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / max(warmup_steps, 1)
                    progress = (step - warmup_steps) / max(total_steps_est - warmup_steps, 1)
                    return max(0.01, 1.0 - progress)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            # Loss
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

            # --- Prepare validation samples (subset, cached in CPU memory) ---
            val_batches = []
            val_sample_count = 0
            for group, uid, split_name in val_user_splits:
                if val_sample_count >= MAX_VAL_SAMPLES:
                    break
                X_tr, X_te, y_tr, y_te = user_data[group][uid]
                X_split = X_te if split_name == "test" else X_tr
                y_split = y_te if split_name == "test" else y_tr
                try:
                    samples, _ = create_gnn_train_val_samples(
                        uid, graph, X_split, y_split, X_te, y_te
                    )
                    for s in samples:
                        val_batches.append(_sample_to_pyg_data(s))
                        val_sample_count += 1
                        if val_sample_count >= MAX_VAL_SAMPLES:
                            break
                except Exception:
                    continue

            # --- Training loop ---
            best_val_f1 = 0.0
            steps_without_improvement = 0
            global_step = 0
            history = []  # Per-epoch training output for inspection

            for epoch in range(EPOCHS):
                model.train()
                users = list(train_user_items)
                _shuffle_list(users)

                optimizer.zero_grad()
                accum_count = 0
                epoch_loss_sum = 0.0
                epoch_loss_count = 0

                for chunk_start in range(0, len(users), TRAIN_CHUNK_SIZE):
                    chunk_users = users[chunk_start : chunk_start + TRAIN_CHUNK_SIZE]
                    chunk_data_list = []

                    for grp, uid in chunk_users:
                        X_tr, X_te, y_tr, y_te = user_data[grp][uid]
                        try:
                            train_samples, _ = create_gnn_train_val_samples(
                                uid, graph, X_tr, y_tr, X_te, y_te
                            )
                            for s in train_samples:
                                chunk_data_list.append(_sample_to_pyg_data(s))
                        except Exception:
                            continue

                    if not chunk_data_list:
                        continue

                    # --- Oversample positives ---
                    if TARGET_POS_RATE is not None:
                        pos_idx = [i for i, d in enumerate(chunk_data_list) if d.y.item() == 1]
                        n_pos = len(pos_idx)
                        n_neg = len(chunk_data_list) - n_pos
                        if n_pos > 0 and n_neg > 0:
                            target_n_pos = int(np.ceil(
                                TARGET_POS_RATE * n_neg / (1.0 - TARGET_POS_RATE)
                            ))
                            n_extra = target_n_pos - n_pos
                            if n_extra > 0:
                                extra_idx = np.random.choice(pos_idx, size=n_extra, replace=True)
                                extra_samples = [chunk_data_list[i] for i in extra_idx]
                                chunk_data_list.extend(extra_samples)

                    _shuffle_list(chunk_data_list)

                    for start in range(0, len(chunk_data_list) - batch_size + 1, batch_size):
                        batch = Batch.from_data_list(
                            chunk_data_list[start : start + batch_size]
                        ).to(device)
                        out = model(batch)
                        loss = criterion(out, batch.y) / grad_accum_steps
                        loss.backward()
                        epoch_loss_sum += loss.item() * grad_accum_steps
                        epoch_loss_count += 1
                        accum_count += 1

                        if accum_count >= grad_accum_steps:
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            accum_count = 0
                            global_step += 1

                    del chunk_data_list
                    gc.collect()
                    torch.cuda.empty_cache()

                # --- End of epoch: validate ---
                gc.collect()
                torch.cuda.empty_cache()
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for i in range(0, len(val_batches), batch_size):
                        batch = Batch.from_data_list(
                            val_batches[i : i + batch_size]
                        ).to(device)
                        out = model(batch)
                        all_preds.extend(out.argmax(dim=1).cpu().tolist())
                        all_labels.extend(batch.y.cpu().tolist())

                val_f1 = f1_score(all_labels, all_preds)
                avg_train_loss = epoch_loss_sum / max(epoch_loss_count, 1)

                # Log per-epoch metrics to MLflow (step = epoch)
                mlflow.log_metrics(
                    {"epoch_train_loss": avg_train_loss, "epoch_val_f1": val_f1},
                    step=epoch,
                )
                history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_f1": val_f1})

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                # Early stopping
                if PATIENCE and steps_without_improvement >= PATIENCE:
                    break

            elapsed = time.time() - trial_start

            # Save per-epoch training history as artifact
            import json as _json
            mlflow.log_text(_json.dumps(history, indent=2), "training_history.json")

            # Training curves as MLflow artifact
            if history:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                epochs_x = [h["epoch"] for h in history]
                axes[0].plot(epochs_x, [h["train_loss"] for h in history], 'b-o', ms=3, label='Train loss')
                if "val_loss" in history[0]:
                    axes[0].plot(epochs_x, [h["val_loss"] for h in history], 'r-o', ms=3, label='Val loss')
                axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss')
                axes[0].legend(); axes[0].grid(True, alpha=0.3)
                if "train_f1" in history[0]:
                    axes[1].plot(epochs_x, [h["train_f1"] for h in history], 'b-o', ms=3, label='Train F1')
                axes[1].plot(epochs_x, [h["val_f1"] for h in history], 'r-o', ms=3, label='Val F1')
                axes[1].set(xlabel='Epoch', ylabel='F1', title='F1')
                axes[1].legend(); axes[1].grid(True, alpha=0.3)
                plt.suptitle(f'Trial {trial.number}', fontsize=11)
                plt.tight_layout()
                mlflow.log_figure(fig, "training_curves.png")
                plt.close(fig)

            # Log metrics to nested MLflow run (visible in experiment UI)
            mlflow.log_metrics({"best_val_f1": best_val_f1, "trial_time_s": elapsed})
            # Also store key metrics in Optuna trial attrs → accessible from
            # the MlflowSparkStudy-managed run without digging into nested runs
            trial.set_user_attr("best_val_f1", best_val_f1)
            trial.set_user_attr("trial_time_s", elapsed)

            # Cleanup
            del model, optimizer, scheduler, criterion, val_batches
            gc.collect()
            torch.cuda.empty_cache()

            return -best_val_f1

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            mlflow.log_param("error", str(e)[:200])
            mlflow.log_text(tb_str, "traceback.txt")
            trial.set_user_attr("error", str(e)[:200])
            gc.collect()
            torch.cuda.empty_cache()
            return 1.0


print("Objective function defined.")

# COMMAND ----------

# DBTITLE 1,Search space note
# MAGIC %md
# MAGIC ### Search space (defined inside objective function)
# MAGIC
# MAGIC With Optuna, the search space is defined inline via `trial.suggest_*()` calls inside the
# MAGIC objective function (cell above). No separate `search_space` dict is needed.
# MAGIC
# MAGIC | Parameter | Type | Range |
# MAGIC |-----------|------|-------|
# MAGIC | lr | log-float | [1e-4, 1e-2] |
# MAGIC | weight_decay | log-float | [1e-5, 1e-2] |
# MAGIC | batch_size | categorical | {128, 256, 512} |
# MAGIC | dropout | float | [0.05, 0.5] |
# MAGIC | drop_edge_rate | float | [0.0, 0.3] |
# MAGIC | warmup_epochs | categorical | {2, 3, 5, 8} |
# MAGIC | grad_accum_steps | categorical | {1, 2, 4, 8} |
# MAGIC | lr_schedule | categorical | {cosine, linear} |
# MAGIC
# MAGIC Key insight from 3.1 fine-tuning: aggressive LR destroys pretrained representations.
# MAGIC We search broadly around the working regime (3e-3 for from-scratch, 3e-4 for fine-tune).

# COMMAND ----------

# DBTITLE 1,Propagate credentials to executors
# Fix: MlflowSparkStudy distributes trials to Spark executors, which lack
# the driver's native Databricks auth context. Capture credentials here and
# inject them via the objective function closure so the Databricks SDK on
# executors can authenticate when logging artifacts to MLflow.
import os

_db_host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
_db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_experiment_name = f"/Users/pablo.celayes@bolt.eu/sna_classifier_gnn_hparam_tuning_{FINAL_TAG}"

_original_objective = objective

def objective(trial):
    """Wrapper that sets auth env vars and experiment context on executors before running the trial."""
    os.environ["DATABRICKS_HOST"] = _db_host
    os.environ["DATABRICKS_TOKEN"] = _db_token
    mlflow.set_experiment(_experiment_name)
    return _original_objective(trial)

# COMMAND ----------

# DBTITLE 1,Broadcast large objects to executors
# Broadcast graph and user_data so they are serialized once and shared
# across executors, instead of being re-serialized per task via closure capture.
_graph_bc = sc.broadcast(graph)
_user_data_bc = sc.broadcast(user_data)
print(f"Broadcast graph ({graph.number_of_nodes()} nodes) and user_data ({len(user_data)} groups) to executors")

# COMMAND ----------

# DBTITLE 1,Run Optuna with MlflowSparkStudy
# MLflow experiment for tracking all trials
mlflow.set_experiment(f"/Users/pablo.celayes@bolt.eu/sna_classifier_gnn_hparam_tuning_{FINAL_TAG}")

print(f"Starting Optuna search: {MAX_EVALS} trials, parallelism={PARALLELISM}")
print(f"Each trial trains for up to {EPOCHS} epochs with early stopping (patience={PATIENCE})")
print(f"RESTART_TUNING={RESTART_TUNING}")
print("=" * 60)

with mlflow.start_run(run_name="gnn_hparam_tuning_optuna") as run:
    experiment_id = run.info.experiment_id
    mlflow_storage = MlflowStorage(experiment_id=experiment_id)

    if RESTART_TUNING:
        # Delete existing study if present, then create fresh
        try:
            optuna.delete_study(study_name="gnn-hparam-tuning", storage=mlflow_storage)
            print("Deleted existing study — starting fresh.")
        except KeyError:
            print("No existing study found — starting fresh.")
    else:
        print("Resuming from existing study (if any).")

    mlflow_study = MlflowSparkStudy(
        study_name="gnn-hparam-tuning",
        storage=mlflow_storage,
    )
    # Callback prints trial results on the driver as they complete (real-time visibility)
    def trial_callback(study, trial):
        f1 = -trial.value if trial.value is not None else None
        dur = trial.duration.total_seconds() / 60 if trial.duration else 0
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        best_str = f"{-study.best_value:.4f}" if study.best_trial else "N/A"
        error = trial.user_attrs.get("error", "")
        status = f"ERROR: {error[:80]}" if error else f"val_f1={f1_str}"
        print(f"[Trial {trial.number:>3d}] {status}, "
              f"time={dur:.1f}min, best_so_far={best_str}")

    mlflow_study.optimize(objective, n_trials=MAX_EVALS, n_jobs=PARALLELISM, callbacks=[trial_callback])

    print("\n" + "=" * 60)
    print("Optuna search complete!")
    print("=" * 60)
    print(f"Best trial value: {mlflow_study.study.best_value:.4f}")
    print(f"Best val F1: {-mlflow_study.study.best_value:.4f}")
    print(f"Best params: {mlflow_study.study.best_params}")

# COMMAND ----------

# DBTITLE 1,Display failed trial errors
# Show errors from failed trials in the notebook
import optuna

all_trials = mlflow_study.study.trials
failed = [
    t for t in all_trials
    if t.state == optuna.trial.TrialState.COMPLETE and t.value == 1.0
]

if failed:
    print(f"\n{'='*60}")
    print(f"\u26a0\ufe0f  {len(failed)} / {len(all_trials)} TRIALS LIKELY FAILED (returned penalty value 1.0)")
    print(f"{'='*60}\n")
    print("Check MLflow for full tracebacks (logged as 'traceback.txt' artifact).")
    print("Trial errors (from MLflow params):")
    for t in failed:
        error = t.user_attrs.get("error", t.params.get("error", "unknown"))
        print(f"  Trial {t.number}: {error}")
    print()
else:
    print(f"\u2705 All {len(all_trials)} trials completed successfully (no penalty returns).")

# COMMAND ----------

# DBTITLE 1,Best hyperparameters and results
# Extract best trial from Optuna study
best_trial = mlflow_study.study.best_trial
best_params = best_trial.params
best_val_f1 = -best_trial.value  # We minimized -f1, so negate back

print("\n" + "=" * 60)
print("BEST HYPERPARAMETERS")
print("=" * 60)
for k, v in sorted(best_params.items()):
    print(f"  {k:20s}: {v}")

print(f"\nBest validation F1: {best_val_f1:.4f}")
print(f"Baseline SVC combined F1 from 3.1: 0.8520")
print(f"Improvement over baseline: {best_val_f1 - 0.8520:+.4f}")

# Save best params
import json
with open(f"{TUNING_DIR}/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2, default=str)
print(f"\nBest params saved to: {TUNING_DIR}/best_params.json")

# COMMAND ----------

# DBTITLE 1,Trial results analysis
import matplotlib.pyplot as plt

# Extract all trial results from Optuna study
trial_f1s = []
for t in mlflow_study.study.trials:
    if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
        trial_f1s.append(-t.value)  # Negate back from minimization
    else:
        trial_f1s.append(None)

valid_f1s = [f for f in trial_f1s if f is not None]

print(f"\nTrial summary:")
print(f"  Total trials:    {len(trial_f1s)}")
print(f"  Successful:      {len(valid_f1s)}")
print(f"  Failed:          {len(trial_f1s) - len(valid_f1s)}")
print(f"  Best val F1:     {max(valid_f1s):.4f}")
print(f"  Median val F1:   {np.median(valid_f1s):.4f}")
print(f"  Worst val F1:    {min(valid_f1s):.4f}")

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: trial F1 over time
ax = axes[0]
ax.scatter(range(len(valid_f1s)), valid_f1s, alpha=0.6, c='steelblue', s=30)
best_so_far = np.maximum.accumulate(valid_f1s)
ax.plot(best_so_far, 'r-', linewidth=2, label='Best so far')
ax.axhline(0.8520, color='gray', linestyle='--', alpha=0.7, label='Baseline SVC (0.8520)')
ax.set_xlabel('Trial')
ax.set_ylabel('Validation F1')
ax.set_title('Optuna Convergence')
ax.legend()

# Right: F1 distribution
ax = axes[1]
ax.hist(valid_f1s, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(max(valid_f1s), color='red', linestyle='--', label=f'Best: {max(valid_f1s):.4f}')
ax.axvline(np.median(valid_f1s), color='orange', linestyle='--', label=f'Median: {np.median(valid_f1s):.4f}')
ax.set_xlabel('Validation F1')
ax.set_ylabel('Count')
ax.set_title('Trial F1 Distribution')
ax.legend()

plt.tight_layout()
plt.savefig(f"{TUNING_DIR}/hparam_convergence.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to: {TUNING_DIR}/hparam_convergence.png")

# COMMAND ----------

# DBTITLE 1,Top-10 trials ranked by val F1
import pandas as pd

# Build a dataframe of all successful trials
trial_records = []
for t in mlflow_study.study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
        continue
    record = {
        "trial": t.number,
        "val_f1": -t.value,
        **t.params,
    }
    trial_records.append(record)

df_trials = pd.DataFrame(trial_records).sort_values("val_f1", ascending=False)

print("\nTop-10 trials:")
display(df_trials.head(10))

# Save full results
df_trials.to_csv(f"{TUNING_DIR}/all_trials.csv", index=False)
print(f"\nAll trial results saved to: {TUNING_DIR}/all_trials.csv")

# COMMAND ----------

# DBTITLE 1,Retrain best model and save weights
# Retrain the best model on the driver (full training with logging) and save weights.
# This uses the full train_model() function from gnn_models with the best hyperparameters.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build model with best hparams
model = RetweetGNN(
    ff_hidden_dim=64,
    gcn_hidden_dim=64,
    transformer_dim=64,
    transformer_heads=4,
    embeddings_path=EMBEDDINGS_PATH,
    device=device,
    dropout=best_params["dropout"],
    drop_edge_rate=best_params["drop_edge_rate"],
).to(device)

with torch.no_grad():
    model.gate_param.fill_(0.0)

print(f"Retraining best model with:")
for k, v in sorted(best_params.items()):
    print(f"  {k}: {v}")
print(f"\nDevice: {device}")
print(f"Output: {TUNING_DIR}/best_retweet_gnn_tuned.pt")

# Use the ChunkedGNNTrainLoader and CachedValLoader from 3.1 imports
from random import shuffle as _shuffle_list

# Inline training with best hparams (same loop as objective but with full logging)
model, history = train_model(
    model=model,
    train_loader=train_loader if 'train_loader' in dir() else None,  # may need to recreate
    val_loader=val_loader if 'val_loader' in dir() else None,
    experiment_dir=TUNING_DIR,
    class_weights=class_weights,
    epochs=EPOCHS,
    device=device,
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
    lr_warmup_epochs=best_params["warmup_epochs"],
    gradient_accumulation_steps=best_params["grad_accum_steps"],
    patience=20,  # Full patience for final training
    log_every_n_steps=200,
    mixed_precision=False,
    progress_every_n_steps=50,
)

# Save best model weights
torch.save(model.state_dict(), f"{TUNING_DIR}/best_retweet_gnn_tuned.pt")
print(f"\nBest model saved to: {TUNING_DIR}/best_retweet_gnn_tuned.pt")
print(f"Use INIT_WEIGHTS_PATH = '{TUNING_DIR}/best_retweet_gnn_tuned.pt' in 3.1 to fine-tune from this.")