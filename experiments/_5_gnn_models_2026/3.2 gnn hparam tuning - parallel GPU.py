# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC ## 3.2 GNN Hyperparameter Tuning — Parallel GPU (Hyperopt + SparkTrials)
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
# MAGIC **Strategy**: Hyperopt TPE (Tree of Parzen Estimators) with `SparkTrials` for
# MAGIC automatic distribution across GPU workers. MLflow logs every trial.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --upgrade torch-geometric-signed-directed networkx hyperopt

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
EPOCHS = 10  # Same as 3.1 — fixed across all trials
MAX_EVALS = 40  # Total number of Hyperopt trials
PARALLELISM = 8  # Number of trials running in parallel (= number of GPU workers)
PATIENCE = 10  # Early stopping patience per trial (shorter than 3.1 to save time)
MAX_VAL_SAMPLES = 50_000  # Cap val samples (same as 3.1)
TRAIN_CHUNK_SIZE = 10  # Users per chunk for on-the-fly GNN sample generation

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

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, SparkTrials, space_eval
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
def train_one_trial(params):
    """
    Hyperopt objective function. Each call trains a RetweetGNN from scratch
    with the given hyperparameters and returns -val_f1 (minimization).

    This runs on a Spark worker with its own GPU.
    """
    import gc
    import time
    import torch
    import torch.nn as nn
    import numpy as np
    from torch_geometric.data import Batch
    from sklearn.metrics import f1_score

    trial_start = time.time()

    # Extract hyperparameters
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = int(params["batch_size"])
    dropout = params["dropout"]
    drop_edge_rate = params["drop_edge_rate"]
    warmup_epochs = int(params["warmup_epochs"])
    grad_accum_steps = int(params["grad_accum_steps"])
    lr_schedule = params["lr_schedule"]  # "cosine" or "linear"

    # Select GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        history = []

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

            # --- End of epoch: validate ---
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
            mlflow.log_metrics({"epoch_train_loss": avg_train_loss, "epoch_val_f1": val_f1}, step=epoch)
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

        # Save history and training curves
        import json as _json
        mlflow.log_text(_json.dumps(history, indent=2), "training_history.json")
        if history:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            epochs_x = [h["epoch"] for h in history]
            axes[0].plot(epochs_x, [h["train_loss"] for h in history], 'b-o', ms=3, label='Train loss')
            axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss')
            axes[0].legend(); axes[0].grid(True, alpha=0.3)
            axes[1].plot(epochs_x, [h["val_f1"] for h in history], 'r-o', ms=3, label='Val F1')
            axes[1].set(xlabel='Epoch', ylabel='F1', title='F1')
            axes[1].legend(); axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            mlflow.log_figure(fig, "training_curves.png")
            plt.close(fig)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({"best_val_f1": best_val_f1, "trial_time_s": elapsed})

        # Cleanup
        del model, optimizer, scheduler, criterion, val_batches
        gc.collect()
        torch.cuda.empty_cache()

        return {"loss": -best_val_f1, "status": STATUS_OK, "best_val_f1": best_val_f1}

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return {"loss": 1.0, "status": STATUS_FAIL, "error": str(e), "traceback": tb_str}


print("Objective function defined.")

# COMMAND ----------

# DBTITLE 1,Define search space
import math

# Hyperparameter search space
# Key insight from 3.1 fine-tuning: aggressive LR destroys pretrained representations.
# We search broadly around the working regime (3e-3 for from-scratch, 3e-4 for fine-tune).
search_space = {
    # Learning rate: log-uniform from 1e-4 to 1e-2
    "lr": hp.loguniform("lr", math.log(1e-4), math.log(1e-2)),

    # Weight decay: log-uniform from 1e-5 to 1e-2
    "weight_decay": hp.loguniform("weight_decay", math.log(1e-5), math.log(1e-2)),

    # Batch size: discrete choices
    "batch_size": hp.choice("batch_size", [128, 256, 512]),

    # Dropout: uniform 0.05 to 0.5
    "dropout": hp.uniform("dropout", 0.05, 0.5),

    # Drop-edge rate: uniform 0.0 to 0.3
    "drop_edge_rate": hp.uniform("drop_edge_rate", 0.0, 0.3),

    # Warmup epochs: discrete
    "warmup_epochs": hp.choice("warmup_epochs", [2, 3, 5, 8]),

    # Gradient accumulation steps
    "grad_accum_steps": hp.choice("grad_accum_steps", [1, 2, 4, 8]),

    # LR schedule type
    "lr_schedule": hp.choice("lr_schedule", ["cosine", "linear"]),
}

print("Search space:")
for k, v in search_space.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,Run Hyperopt with SparkTrials
# MLflow experiment for tracking all trials
mlflow.set_experiment(f"/Users/pablo.celayes@bolt.eu/sna_classifier_gnn_hparam_tuning_{FINAL_TAG}")

# SparkTrials distributes trials across Spark workers (each with a GPU).
# `parallelism` controls how many trials run simultaneously.
spark_trials = SparkTrials(parallelism=PARALLELISM)

print(f"Starting Hyperopt search: {MAX_EVALS} trials, parallelism={PARALLELISM}")
print(f"Each trial trains for up to {EPOCHS} epochs with early stopping (patience={PATIENCE})")
print("="*60)

best_params_raw = fmin(
    fn=train_one_trial,
    space=search_space,
    algo=tpe.suggest,
    max_evals=MAX_EVALS,
    trials=spark_trials,
    rstate=np.random.default_rng(42),
)

print("\n" + "="*60)
print("Hyperopt search complete!")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Display failed trial errors
# Show errors from failed trials in the notebook
failed = [
    (i, r) for i, r in enumerate(spark_trials.results)
    if r.get("status") == STATUS_FAIL
]

if failed:
    print(f"\n{'='*60}")
    print(f"⚠️  {len(failed)} / {len(spark_trials.results)} TRIALS FAILED")
    print(f"{'='*60}\n")
    for trial_idx, result in failed:
        print(f"--- Trial {trial_idx} ---")
        print(f"Error: {result.get('error', 'unknown')}")
        if "traceback" in result:
            print(result["traceback"])
        print()
else:
    print(f"✅ All {len(spark_trials.results)} trials completed successfully.")

# COMMAND ----------

# DBTITLE 1,Best hyperparameters and results
# Decode best params (hp.choice returns indices, need to map back)
best_params = space_eval(search_space, best_params_raw)

print("\n" + "="*60)
print("BEST HYPERPARAMETERS")
print("="*60)
for k, v in sorted(best_params.items()):
    print(f"  {k:20s}: {v}")

# Get the best trial's val F1
best_trial = min(spark_trials.results, key=lambda r: r.get("loss", 0))
best_val_f1 = best_trial.get("best_val_f1", -best_trial["loss"])
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

# Extract all trial results
trial_f1s = []
trial_params_list = []
for i, result in enumerate(spark_trials.results):
    if result["status"] == STATUS_OK:
        trial_f1s.append(-result["loss"])
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
ax.set_title('Hyperopt Convergence')
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
for i, (trial, result) in enumerate(zip(spark_trials.trials, spark_trials.results)):
    if result["status"] != STATUS_OK:
        continue
    params = space_eval(search_space, {k: v[0] for k, v in trial["misc"]["vals"].items()})
    record = {
        "trial": i,
        "val_f1": -result["loss"],
        **params,
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