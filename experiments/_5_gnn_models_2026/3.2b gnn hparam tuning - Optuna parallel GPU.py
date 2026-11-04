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
N_USERS = None  # Total users to sample (proportionally across groups). None = all users.
EXPERIMENT_TAG = "v3_full"  # Must match 3.1 to reuse the same user sample

# Paths
sources_path = "/serafin/pcelayes/repos/sna_classifier/"
DATA_PATH = "/Workspace/Users/pablo.celayes@bolt.eu/learning/data/sna_classifier"
EMBEDDINGS_PATH = f"{DATA_PATH}/node_embeddings.pt"

# Derived experiment folder (same as 3.1 — reuses user_sample.json)
FINAL_TAG = f"{EXPERIMENT_TAG}_N{N_USERS}" if N_USERS else EXPERIMENT_TAG
EXPERIMENT_DIR = f"./experiments/{FINAL_TAG}"

# Hyperparameter tuning settings
EPOCHS = 1  # Same as 3.1 — fixed across all trials
MAX_EVALS = 25  # Total number of Optuna trials (reduced — narrower search space needs fewer)
PARALLELISM = 3  # Number of trials running in parallel (reduced for memory safety)
RESTART_TUNING = False  # If True, delete previous study and start fresh; if False, resume from existing trials
LOG_EVERY_N_STEPS = 2000  # Evaluate val F1 every N optimizer steps (step-based patience, matching 3.1)
PATIENCE_STEPS = 30_000  # Early stopping: optimizer steps without val F1 improvement. None = disabled.
PATIENCE = -(-PATIENCE_STEPS // LOG_EVERY_N_STEPS) if PATIENCE_STEPS else None  # Convert to checkpoints (rounded up)
MAX_VAL_SAMPLES = 50_000  # Cap val samples (increased for stable F1 estimates at full-data scale)
TRAIN_CHUNK_SIZE = 10  # Users per chunk for on-the-fly GNN sample generation
TARGET_POS_RATE = 0.10  # Oversample positives in train loader to reach this rate. None = disabled.

# Pre-trained weights initialization (set to a .pt file path to warm-start all trials)
# Example: "./experiments/v2_N20/best_retweet_gnn_general.pt"
# INIT_WEIGHTS_PATH = None  # None = train from scratch
INIT_WEIGHTS_PATH = "./experiments/v2_N20/best_retweet_gnn_general.pt"

# Output directory for tuning results
TUNING_DIR = f"./experiments/{FINAL_TAG}_hparam_tuning"
os.makedirs(TUNING_DIR, exist_ok=True)

# DBFS directory for per-trial history checkpoints (survives cancellation/cluster loss)
HISTORY_CHECKPOINT_DIR = f"/dbfs/tmp/gnn_hparam_history_{FINAL_TAG}"
os.makedirs(HISTORY_CHECKPOINT_DIR, exist_ok=True)

print(f"Base experiment: {FINAL_TAG}")
print(f"Tuning output:  {TUNING_DIR}")
print(f"Trials: {MAX_EVALS} (parallelism={PARALLELISM})")
print(f"Epochs per trial: {EPOCHS}")
if INIT_WEIGHTS_PATH:
    print(f"Pre-trained weights: {INIT_WEIGHTS_PATH}")
else:
    print("Training from scratch (no pre-trained weights).")

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

# ---------------------------------------------------------------------------
# Shared helper: proportional allocation with largest-remainder method
# (same as 3.1's CachedValLoader)
# ---------------------------------------------------------------------------
def _proportional_allocate(row_counts, max_samples):
    """Proportionally allocate max_samples across items using largest-remainder.

    Each item gets at least 1 sample. Returns a list of allocations
    (same length as row_counts). If total available <= max_samples or
    max_samples is None, returns row_counts as-is.
    """
    total_available = sum(row_counts)
    if not max_samples or total_available <= max_samples:
        return list(row_counts)

    sample_rate = max_samples / total_available
    raw = [n * sample_rate for n in row_counts]
    allocations = [max(1, int(r)) for r in raw]

    # Fill deficit by giving +1 to items that lost the most from int() truncation
    deficit = max_samples - sum(allocations)
    if deficit > 0:
        remainders = sorted(
            range(len(raw)),
            key=lambda i: raw[i] - int(raw[i]),
            reverse=True,
        )
        for i in remainders[:deficit]:
            allocations[i] += 1

    # Trim any excess from max(1, ...) bumps on tiny items
    while sum(allocations) > max_samples:
        max_idx = max(range(len(allocations)), key=lambda i: allocations[i])
        allocations[max_idx] -= 1

    return allocations


print("GNN sample helpers defined.")

# COMMAND ----------

# DBTITLE 1,Objective function: train one trial
def objective(trial):
    """
    Optuna objective function. Each call trains a RetweetGNN (optionally
    warm-started from INIT_WEIGHTS_PATH) with the given hyperparameters
    and returns -val_f1 (minimization).

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

    # Load graph and user_data from source files on the executor
    # (no broadcast — avoids JVM holding serialised copy in executor heap)
    import networkx as nx
    graph = nx.read_graphml(IG_GRAPH_PATH)

    user_data = {}
    for _grp, _uids in saved_sample.items():
        user_data[_grp] = {}
        for _uid in _uids:
            try:
                _X_tr, _X_te, _y_tr, _y_te = load_dataframe_raw(_uid, sparse=True)
                if _X_tr.shape[0] > 0 and _X_te.shape[0] > 0 and _y_tr.sum() > 0 and _y_te.sum() > 0:
                    user_data[_grp][_uid] = (_X_tr, _X_te, _y_tr, _y_te)
            except Exception:
                continue

    trial_start = time.time()

    # Suggest hyperparameters via Optuna
    # Narrowed search space based on experiment results:
    # - lr: 1e-5 to 5e-4 (top trials had lr ~5e-5 to 1e-4; higher lr always poor)
    # - batch_size: capped at 256 (512 caused OOM on 22GB GPU)
    # - dropout: capped at 0.25 (>0.4 always poor in completed trials)
    # - drop_edge_rate: capped at 0.2 (top trials had very low values)
    # - grad_accum_steps: increased to [4,8,16] to compensate for smaller batch sizes
    # - warmup_epochs: reduced to {1,2,3} (epochs are ~10x longer at full scale)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    dropout = trial.suggest_float("dropout", 0.05, 0.25)
    drop_edge_rate = trial.suggest_float("drop_edge_rate", 0.0, 0.2)
    warmup_epochs = trial.suggest_categorical("warmup_epochs", [1, 2, 3])
    grad_accum_steps = trial.suggest_categorical("grad_accum_steps", [4, 8, 16])
    lr_schedule = trial.suggest_categorical("lr_schedule", ["cosine", "linear"])

    # Free stale GPU memory before building model (prevents OOM with parallelism)
    gc.collect()
    torch.cuda.empty_cache()

    # Select GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # No MLflow on executors — results are logged driver-side after trial completes.
    # Using `if True:` to preserve indentation of the existing try/except block.
    if True:
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

            # Load pre-trained weights if INIT_WEIGHTS_PATH is set
            if INIT_WEIGHTS_PATH:
                state_dict = torch.load(INIT_WEIGHTS_PATH, map_location=device, weights_only=False)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                # Missing/unexpected keys are fine if architecture evolved
            else:
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

            # Loss — plain CE is equivalent to 3.1's combined_loss for EPOCHS<=10,
            # since combined_loss only blends soft_f1 after warmup_epochs=10.
            # With EPOCHS=1 per tuning trial, both produce the same gradients.
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

            # --- Validation metadata (proportional allocation, matching 3.1) ---
            # Only compute allocation counts here; actual PyG samples are built
            # lazily inside _eval_val() to avoid holding 50k Data objects in RAM.
            row_counts = []
            for group, uid, split_name in val_user_splits:
                X_tr, X_te, _, _ = user_data[group][uid]
                row_counts.append(X_tr.shape[0] if split_name == "train" else X_te.shape[0])
            _val_allocations = _proportional_allocate(row_counts, MAX_VAL_SAMPLES)
            _val_do_subsample = bool(MAX_VAL_SAMPLES) and sum(row_counts) > MAX_VAL_SAMPLES
            del row_counts

            # --- Training loop (step-based patience, matching 3.1) ---
            best_val_f1 = 0.0
            steps_without_improvement = 0
            global_step = 0
            running_loss = 0.0
            running_steps = 0
            _early_stop = False
            history = {"step": [], "train_loss": [], "val_f1": [],
                       "epoch_step": [], "epoch_train_loss": [], "epoch_val_f1": []}

            # Local helper: evaluate val F1 lazily (build-and-discard per user,
            # never holds more than batch_size PyG Data objects at once)
            def _eval_val():
                model.eval()
                _preds, _labels = [], []
                _buf = []
                with torch.no_grad():
                    for _vi, (_vg, _vu, _vs) in enumerate(val_user_splits):
                        _X_tr, _X_te, _y_tr, _y_te = user_data[_vg][_vu]
                        _n = _val_allocations[_vi]
                        try:
                            if _val_do_subsample:
                                if _vs == "test":
                                    _idx = np.random.choice(_X_te.shape[0], size=min(_n, _X_te.shape[0]), replace=False)
                                    _, _samps = create_gnn_train_val_samples(
                                        _vu, graph, _X_tr.iloc[:1], _y_tr.iloc[:1],
                                        _X_te.iloc[_idx], _y_te.iloc[_idx])
                                else:
                                    _idx = np.random.choice(_X_tr.shape[0], size=min(_n, _X_tr.shape[0]), replace=False)
                                    _samps, _ = create_gnn_train_val_samples(
                                        _vu, graph, _X_tr.iloc[_idx], _y_tr.iloc[_idx],
                                        _X_te.iloc[:1], _y_te.iloc[:1])
                            else:
                                _tr_s, _te_s = create_gnn_train_val_samples(
                                    _vu, graph, _X_tr, _y_tr, _X_te, _y_te)
                                _samps = _tr_s if _vs == "train" else _te_s
                            for _s in _samps:
                                _buf.append(_sample_to_pyg_data(_s))
                                if len(_buf) >= batch_size:
                                    _b = Batch.from_data_list(_buf).to(device)
                                    _out = model(_b)
                                    _preds.extend(_out.argmax(dim=1).cpu().tolist())
                                    _labels.extend(_b.y.cpu().tolist())
                                    del _b
                                    _buf = []
                        except Exception:
                            continue
                    # Flush remaining
                    if _buf:
                        _b = Batch.from_data_list(_buf).to(device)
                        _out = model(_b)
                        _preds.extend(_out.argmax(dim=1).cpu().tolist())
                        _labels.extend(_b.y.cpu().tolist())
                        del _b
                    _buf = []
                model.train()
                return f1_score(_labels, _preds)

            for epoch in range(EPOCHS):
                model.train()
                users = list(train_user_items)
                _shuffle_list(users)

                optimizer.zero_grad()
                accum_count = 0
                epoch_loss_sum = 0.0
                epoch_loss_count = 0

                for chunk_start in range(0, len(users), TRAIN_CHUNK_SIZE):
                    if _early_stop:
                        break

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
                        running_loss += loss.item() * grad_accum_steps
                        running_steps += 1
                        accum_count += 1

                        if accum_count >= grad_accum_steps:
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            accum_count = 0
                            global_step += 1

                            # --- Periodic checkpoint (step-based, matching 3.1) ---
                            if LOG_EVERY_N_STEPS and global_step % LOG_EVERY_N_STEPS == 0:
                                avg_loss = running_loss / max(running_steps, 1)
                                val_f1 = _eval_val()

                                if val_f1 > best_val_f1:
                                    best_val_f1 = val_f1
                                    steps_without_improvement = 0
                                else:
                                    steps_without_improvement += 1

                                history["step"].append(global_step)
                                history["train_loss"].append(avg_loss)
                                history["val_f1"].append(val_f1)

                                # Checkpoint history to DBFS (survives cancellation)
                                import json as _json_ckpt
                                _ckpt_path = f"{HISTORY_CHECKPOINT_DIR}/trial_{trial.number}_history.json"
                                with open(_ckpt_path, "w") as _hf:
                                    _json_ckpt.dump({"history": history, "params": trial.params,
                                                     "best_val_f1": best_val_f1, "global_step": global_step}, _hf)

                                running_loss = 0.0
                                running_steps = 0

                                # Step-based early stopping
                                if PATIENCE and steps_without_improvement >= PATIENCE:
                                    _early_stop = True
                                    break

                                gc.collect()
                                torch.cuda.empty_cache()
                                model.train()

                    del chunk_data_list
                    gc.collect()
                    torch.cuda.empty_cache()

                if _early_stop:
                    break

                # --- End of epoch: validate ---
                gc.collect()
                torch.cuda.empty_cache()
                val_f1 = _eval_val()
                avg_train_loss = epoch_loss_sum / max(epoch_loss_count, 1)

                history["epoch_step"].append(global_step)
                history["epoch_train_loss"].append(avg_train_loss)
                history["epoch_val_f1"].append(val_f1)

                # Checkpoint history to DBFS after epoch
                import json as _json_ckpt
                _ckpt_path = f"{HISTORY_CHECKPOINT_DIR}/trial_{trial.number}_history.json"
                with open(_ckpt_path, "w") as _hf:
                    _json_ckpt.dump({"history": history, "params": trial.params,
                                     "best_val_f1": best_val_f1, "global_step": global_step}, _hf)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    steps_without_improvement = 0

            elapsed = time.time() - trial_start

            # Final checkpoint to DBFS (captures complete history before return)
            import json as _json
            _ckpt_path = f"{HISTORY_CHECKPOINT_DIR}/trial_{trial.number}_history.json"
            with open(_ckpt_path, "w") as _hf:
                _json.dump({"history": history, "params": trial.params,
                            "best_val_f1": best_val_f1, "global_step": global_step,
                            "trial_time_s": elapsed, "completed": True}, _hf)

            # Store training history in trial user_attrs (logged to MLflow driver-side)
            trial.set_user_attr("history", _json.dumps(history))
            # Also store key metrics in Optuna trial attrs → accessible from
            # the MlflowSparkStudy-managed run without digging into nested runs
            trial.set_user_attr("best_val_f1", best_val_f1)
            trial.set_user_attr("trial_time_s", elapsed)

            # Cleanup
            del model, optimizer, scheduler, criterion
            del graph, user_data  # free per-trial loaded data
            gc.collect()
            torch.cuda.empty_cache()

            return -best_val_f1

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            trial.set_user_attr("error", str(e)[:500])
            trial.set_user_attr("error_type", type(e).__name__)
            trial.set_user_attr("traceback", tb_str[:3000])
            del graph, user_data  # free per-trial loaded data
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
# MAGIC | Parameter | Type | Range | Notes |
# MAGIC |-----------|------|-------|-------|
# MAGIC | lr | log-float | [1e-5, 5e-4] | Narrowed: top trials had lr \~5e-5 to 1e-4 |
# MAGIC | weight_decay | log-float | [1e-5, 5e-3] | |
# MAGIC | batch_size | categorical | {128, 256} | Capped at 256 (512 caused OOM on 22GB GPU) |
# MAGIC | dropout | float | [0.05, 0.25] | Capped: >0.4 always poor |
# MAGIC | drop_edge_rate | float | [0.0, 0.2] | Capped: top trials had low values |
# MAGIC | warmup_epochs | categorical | {1, 2, 3} | Reduced: epochs are \~10x longer at full scale |
# MAGIC | grad_accum_steps | categorical | {4, 8, 16} | Increased to compensate for smaller batch sizes |
# MAGIC | lr_schedule | categorical | {cosine, linear} | |
# MAGIC
# MAGIC Key insight from 3.1 fine-tuning: aggressive LR destroys pretrained representations.
# MAGIC We search broadly around the working regime (3e-3 for from-scratch, 3e-4 for fine-tune).

# COMMAND ----------

# DBTITLE 1,No broadcast — trials load from disk
# No broadcast — each trial loads graph + user_data from source files
# directly on the executor.  This avoids the JVM holding a serialised
# copy (~GB) in every executor's heap, which was the main cause of the
# CPU-RAM OOM killing workers on g6e.xlarge (32 GiB).
#
# graph:     loaded from IG_GRAPH_PATH (GraphML, shared storage)
# user_data: loaded per-user via load_dataframe_raw (same as cell 7)
# saved_sample (user ID lists) is captured via closure — tiny.
print("No broadcast — each trial will load graph + user_data from disk")
print(f"  Graph path:  {IG_GRAPH_PATH}")
print(f"  User sample: {len(saved_sample)} groups, "
      f"{sum(len(v) for v in saved_sample.values())} users")

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
        except Exception:
            print("No existing study found — starting fresh.")
    else:
        print("Resuming from existing study (if any).")

    mlflow_study = MlflowSparkStudy(
        study_name="gnn-hparam-tuning",
        storage=mlflow_storage,
    )
    # Callback: prints trial results AND logs to MLflow on the driver (no MLflow on executors)
    def trial_callback(study, trial):
        import json as _json
        f1 = -trial.value if trial.value is not None else None
        dur = trial.duration.total_seconds() / 60 if trial.duration else 0
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        best_str = f"{-study.best_value:.4f}" if study.best_trial else "N/A"
        error = trial.user_attrs.get("error", "")
        is_failure = error or (trial.value is not None and trial.value == 1.0)

        # --- Print to notebook output (real-time visibility) ---
        if is_failure:
            tb = trial.user_attrs.get("traceback", "")
            print(f"\u26a0\ufe0f  [Trial {trial.number:>3d}] FAILED: {error[:120]}")
            if tb:
                # Show last 3 lines of traceback for quick diagnosis
                tb_lines = tb.strip().splitlines()
                for line in tb_lines[-3:]:
                    print(f"    {line}")
            print(f"    time={dur:.1f}min, best_so_far={best_str}")
        else:
            print(f"\u2705 [Trial {trial.number:>3d}] val_f1={f1_str}, "
                  f"time={dur:.1f}min, best_so_far={best_str}")

        # --- Log to MLflow on the driver (nested run per trial) ---
        try:
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                # Log hyperparameters
                mlflow.log_params(trial.params)
                if f1 is not None:
                    mlflow.log_metric("best_val_f1", f1 if not is_failure else 0.0)
                mlflow.log_metric("trial_time_s", dur * 60)
                mlflow.log_metric("trial_failed", 1 if is_failure else 0)
                mlflow.log_param("trial_number", trial.number)
                if is_failure:
                    error_type = trial.user_attrs.get("error_type", "Unknown")
                    mlflow.set_tag("error_type", error_type)
                if error:
                    mlflow.log_param("error", error[:200])
                    tb = trial.user_attrs.get("traceback", "")
                    if tb:
                        mlflow.log_text(tb, "traceback.txt")
                # Log training history if available
                history_json = trial.user_attrs.get("history", "")
                if history_json:
                    mlflow.log_text(history_json, "training_history.json")
                    history = _json.loads(history_json)
                    # Log step-level metrics for charting
                    for i, step in enumerate(history.get("step", [])):
                        mlflow.log_metrics(
                            {"checkpoint_train_loss": history["train_loss"][i],
                             "checkpoint_val_f1": history["val_f1"][i]},
                            step=step,
                        )
        except Exception as log_err:
            print(f"    (MLflow logging failed for trial {trial.number}: {log_err})")

    mlflow_study.optimize(objective, n_trials=MAX_EVALS, n_jobs=PARALLELISM, callbacks=[trial_callback])

    # --- Post-optimize: clean up any stale RUNNING trials ---
    stale = [t for t in mlflow_study.study.trials
             if t.state == optuna.trial.TrialState.RUNNING]
    if stale:
        print(f"\n\u26a0\ufe0f  {len(stale)} trials still in RUNNING state (likely executor crashes):")
        for t in stale:
            err = t.user_attrs.get("error", "no error captured (executor killed?)")
            print(f"  Trial {t.number}: {err}")

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

# Save best params with metadata matching what 3.1 expects in hparams_from_gpu_tuning.json
import json
best_hparams = dict(best_params)
best_hparams["train_chunk_size"] = TRAIN_CHUNK_SIZE
best_hparams["epochs"] = EPOCHS
best_hparams["patience"] = PATIENCE
best_hparams["max_val_samples"] = MAX_VAL_SAMPLES
best_hparams["_source"] = {
    "experiment_id": run.info.experiment_id,
    "run_name": run.info.run_name,
    "run_id": run.info.run_id,
    "best_val_f1": best_val_f1,
}
with open(f"{TUNING_DIR}/best_params.json", "w") as f:
    json.dump(best_hparams, f, indent=2, default=str)
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
# Matches 3.1's training configuration: ChunkedGNNTrainLoader + CachedValLoader.

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

# --- Build loaders matching 3.1's approach ---
# Import the on-the-fly loader classes defined in 3.1's cell 18
# (These are defined in the "On-the-fly GNN DataLoaders" cell of 3.1)
from random import shuffle as _shuffle_list
from torch_geometric.data import Batch

_best_batch_size = int(best_params["batch_size"])

# Use ChunkedGNNTrainLoader from the notebook's earlier cells
# (train_user_items, user_data, graph are already available from cells 7-8)
print(f"\nBuilding train/val loaders (batch_size={_best_batch_size})...")

# For retrain, we reuse the train_user_items and val_user_splits from cell 8,
# and create simple loaders that match 3.1's ChunkedGNNTrainLoader interface.
# The full train_model() function expects loaders with __iter__, __len__, and total_samples.

class _SimpleTrainLoader:
    """Minimal wrapper matching ChunkedGNNTrainLoader interface for train_model()."""
    def __init__(self, user_items, user_data, graph, batch_size, chunk_size, target_pos_rate):
        self.user_items = list(user_items)
        self.user_data = user_data
        self.graph = graph
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.target_pos_rate = target_pos_rate
        self._total = sum(
            user_data[g][u][0].shape[0] for g, u in user_items
        )
        if target_pos_rate:
            self._total = int(self._total / (1 - target_pos_rate))

    @property
    def total_samples(self):
        return self._total

    def __len__(self):
        return self._total // self.batch_size

    def __iter__(self):
        users = list(self.user_items)
        _shuffle_list(users)
        for chunk_start in range(0, len(users), self.chunk_size):
            chunk_users = users[chunk_start : chunk_start + self.chunk_size]
            chunk_data_list = []
            for grp, uid in chunk_users:
                X_tr, X_te, y_tr, y_te = self.user_data[grp][uid]
                try:
                    train_samples, _ = create_gnn_train_val_samples(
                        uid, self.graph, X_tr, y_tr, X_te, y_te
                    )
                    for s in train_samples:
                        chunk_data_list.append(_sample_to_pyg_data(s))
                except Exception:
                    continue
            if not chunk_data_list:
                continue
            if self.target_pos_rate is not None:
                pos_idx = [i for i, d in enumerate(chunk_data_list) if d.y.item() == 1]
                n_pos = len(pos_idx)
                n_neg = len(chunk_data_list) - n_pos
                if n_pos > 0 and n_neg > 0:
                    target_n_pos = int(np.ceil(
                        self.target_pos_rate * n_neg / (1.0 - self.target_pos_rate)
                    ))
                    n_extra = target_n_pos - n_pos
                    if n_extra > 0:
                        extra_idx = np.random.choice(pos_idx, size=n_extra, replace=True)
                        chunk_data_list.extend([chunk_data_list[i] for i in extra_idx])
            _shuffle_list(chunk_data_list)
            for start in range(0, len(chunk_data_list) - self.batch_size + 1, self.batch_size):
                yield Batch.from_data_list(chunk_data_list[start : start + self.batch_size])
            del chunk_data_list
            gc.collect()

class _SimpleValLoader:
    """Minimal wrapper matching CachedValLoader interface for train_model()."""
    def __init__(self, val_batches, batch_size):
        self._samples = val_batches
        self.batch_size = batch_size

    @property
    def total_samples(self):
        return len(self._samples)

    def __len__(self):
        return -(-len(self._samples) // self.batch_size)

    def __iter__(self):
        for i in range(0, len(self._samples), self.batch_size):
            yield Batch.from_data_list(self._samples[i : i + self.batch_size])

# Build val samples (same as objective function)
print("Building validation samples...")
_val_batches = []
_val_count = 0
for group, uid, split_name in val_user_splits:
    if _val_count >= MAX_VAL_SAMPLES:
        break
    X_tr, X_te, y_tr, y_te = user_data[group][uid]
    X_split = X_te if split_name == "test" else X_tr
    y_split = y_te if split_name == "test" else y_tr
    try:
        samples, _ = create_gnn_train_val_samples(
            uid, graph, X_split, y_split, X_te, y_te
        )
        for s in samples:
            _val_batches.append(_sample_to_pyg_data(s))
            _val_count += 1
            if _val_count >= MAX_VAL_SAMPLES:
                break
    except Exception:
        continue
print(f"  Val samples: {len(_val_batches)}")

_retrain_train_loader = _SimpleTrainLoader(
    train_user_items, user_data, graph,
    batch_size=_best_batch_size,
    chunk_size=TRAIN_CHUNK_SIZE,
    target_pos_rate=TARGET_POS_RATE,
)
_retrain_val_loader = _SimpleValLoader(_val_batches, batch_size=_best_batch_size)

print(f"  Train loader: {_retrain_train_loader.total_samples} samples")
print(f"  Val loader: {_retrain_val_loader.total_samples} samples")

# Retrain with best hparams using train_model() from gnn_models
model, history = train_model(
    model=model,
    train_loader=_retrain_train_loader,
    val_loader=_retrain_val_loader,
    experiment_dir=TUNING_DIR,
    class_weights=class_weights,
    epochs=EPOCHS,
    device=device,
    lr=float(best_params["lr"]),
    weight_decay=float(best_params["weight_decay"]),
    lr_warmup_epochs=int(best_params["warmup_epochs"]),
    gradient_accumulation_steps=int(best_params["grad_accum_steps"]),
    patience=20,  # Full patience for final training
    log_every_n_steps=200,
    mixed_precision=False,
)

# Save best model weights
torch.save(model.state_dict(), f"{TUNING_DIR}/best_retweet_gnn_tuned.pt")
print(f"\nBest model saved to: {TUNING_DIR}/best_retweet_gnn_tuned.pt")
print(f"Use INIT_WEIGHTS_PATH = '{TUNING_DIR}/best_retweet_gnn_tuned.pt' in 3.1 to fine-tune from this.")

# COMMAND ----------

# DBTITLE 1,Query Optuna study results (live)
# Query the Optuna study directly from MlflowStorage — no need to wait for the job to finish.
import optuna
import mlflow
from mlflow.optuna import MlflowStorage
import pandas as pd

# Connect to the experiment and study
FINAL_TAG = "v3_full"
experiment = mlflow.get_experiment_by_name(
    f"/Users/pablo.celayes@bolt.eu/sna_classifier_gnn_hparam_tuning_{FINAL_TAG}"
)
assert experiment, "Experiment not found"

storage = MlflowStorage(experiment_id=experiment.experiment_id)
study = optuna.load_study(study_name="gnn-hparam-tuning", storage=storage)

# Summarize trials
trials = study.trials
completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

print(f"=== Optuna Study: {study.study_name} ===")
print(f"Total trials: {len(trials)}")
print(f"  Completed: {len(completed)}")
print(f"  Running:   {len(running)}")
print(f"  Failed:    {len(failed)}")

if completed:
    # Build results table
    records = []
    for t in completed:
        val_f1 = -t.value  # we minimized -f1
        duration_min = t.duration.total_seconds() / 60 if t.duration else None
        record = {
            "trial": t.number,
            "val_f1": round(val_f1, 4),
            "time_min": round(duration_min, 1) if duration_min else None,
            **{k: v for k, v in t.params.items()},
        }
        # Add user attrs if present
        if "best_val_f1" in t.user_attrs:
            record["best_val_f1_attr"] = round(t.user_attrs["best_val_f1"], 4)
        records.append(record)

    df = pd.DataFrame(records).sort_values("val_f1", ascending=False)
    print(f"\n=== Top completed trials (by val F1) ===")
    display(df.head(10))

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best val F1: {-study.best_value:.4f}")
    print(f"Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k}: {v}")
else:
    print("\nNo completed trials yet.")