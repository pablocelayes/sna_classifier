# Databricks notebook source
# MAGIC %pip install torch-geometric-signed-directed

# COMMAND ----------

# DBTITLE 1,Parameters
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

# ============================================================
# Parameters (notebook widgets)
# ============================================================
dbutils.widgets.text("tag", "v1", "Experiment tag")
dbutils.widgets.text("n_users", "20", "Number of users")

EXPERIMENT_TAG = dbutils.widgets.get("tag")
N_USERS = int(dbutils.widgets.get("n_users"))

# Derived experiment folder
FINAL_TAG = f"{EXPERIMENT_TAG}_N{N_USERS}"
EXPERIMENT_DIR = f"./experiments/{FINAL_TAG}"

print(f"Experiment: {FINAL_TAG}")
print(f"Output dir: {EXPERIMENT_DIR}")

# COMMAND ----------

# DBTITLE 1,Load training history
HISTORY_PATH = f"{EXPERIMENT_DIR}/training_history.pkl"

if not os.path.exists(HISTORY_PATH):
    raise FileNotFoundError(f"No history found at {HISTORY_PATH}")

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

print(f"Loaded history: {len(history['step'])} checkpoints, {len(history['epoch_step'])} epochs")
print(f"  Last checkpoint step: {history['step'][-1] if history['step'] else 'N/A'}")
print(f"  Best val F1 (checkpoint): {max(history['val_f1']):.4f}" if history['val_f1'] else "")
print(f"  Best val F1 (end-of-epoch): {max(history['epoch_val_f1']):.4f}" if history['epoch_val_f1'] else "")

# COMMAND ----------

# DBTITLE 1,Training curves: F1 and Loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Loss curves ---
ax = axes[0]
if history["step"]:
    ax.plot(history["step"], history["train_loss"], 'b-', alpha=0.6, label='Train loss (checkpoint)')
    ax.plot(history["step"], history["val_loss"], 'r-', alpha=0.6, label='Val loss (checkpoint)')
if history["epoch_step"]:
    ax.plot(history["epoch_step"], history["epoch_val_loss"], 'ro-', markersize=5, label='Val loss (end-of-epoch)')
ax.set_xlabel('Global Step')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Right: F1 curves ---
ax = axes[1]
if history["step"]:
    ax.plot(history["step"], history["val_f1"], 'r-', alpha=0.6, label='Val F1 (checkpoint)')
if history["epoch_step"]:
    ax.plot(history["epoch_step"], history["epoch_train_f1"], 'b^-', markersize=5, label='Train F1 (end-of-epoch)')
    ax.plot(history["epoch_step"], history["epoch_val_f1"], 'ro-', markersize=5, label='Val F1 (end-of-epoch)')
ax.set_xlabel('Global Step')
ax.set_ylabel('F1 Score')
ax.set_title('Training & Validation F1')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f'GNN Training Curves — {FINAL_TAG}', fontsize=13)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Model and evaluate definitions
from gnn_models import PretrainedEmbeddingLookup, RetweetDataset, RetweetGNN, evaluate

DATA_PATH = "/Workspace/Users/pablo.celayes@bolt.eu/learning/data/sna_classifier"
EMBEDDINGS_PATH = f"{DATA_PATH}/node_embeddings.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# COMMAND ----------

# DBTITLE 1,Load model and test data
# Load best model
BEST_MODEL_PATH = f"{EXPERIMENT_DIR}/best_retweet_gnn_general.pt"
assert os.path.exists(BEST_MODEL_PATH), f"No model found at {BEST_MODEL_PATH}"

model = RetweetGNN(
    ff_hidden_dim=64, gcn_hidden_dim=64, transformer_dim=64, transformer_heads=4,
    embeddings_path=EMBEDDINGS_PATH, device=device
).to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True, map_location=device))
model.eval()
print(f"Loaded model from {BEST_MODEL_PATH}")

# Load GNN samples cache
GNN_SAMPLES_PATH = f"{EXPERIMENT_DIR}/gnn_samples_cache.pkl"
assert os.path.exists(GNN_SAMPLES_PATH), f"No samples cache at {GNN_SAMPLES_PATH}"

with open(GNN_SAMPLES_PATH, "rb") as f:
    gnn_cache = pickle.load(f)
all_test_samples = gnn_cache["all_test_samples"]
user_test_samples = gnn_cache["user_test_samples"]
print(f"Loaded {len(all_test_samples)} test samples, {len(user_test_samples)} users")

# COMMAND ----------

# DBTITLE 1,GNN: Global F1 on combined test set
import gc
gc.collect()
torch.cuda.empty_cache()

# Evaluate on global test set
test_ds = RetweetDataset(all_test_samples)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

global_f1, global_preds, global_labels, _ = evaluate(model, test_loader, device)
print(f"=== GNN Global Test F1: {global_f1:.4f} ===")
print(f"  Total test samples: {len(all_test_samples)}")
print(f"  Positive rate: {global_labels.float().mean():.3f}")

# COMMAND ----------

# DBTITLE 1,GNN: Per-user F1 on test sets
# Compute F1 per user on their test samples
gnn_f1s = {}

for uid, samples in user_test_samples.items():
    if len(samples) == 0:
        continue
    user_ds = RetweetDataset(samples)
    user_loader = DataLoader(user_ds, batch_size=32, shuffle=False)
    user_f1, _, _, _ = evaluate(model, user_loader, device)
    gnn_f1s[uid] = user_f1

gnn_f1_values = list(gnn_f1s.values())

print(f"=== GNN — Per-user Test F1 Distribution ===")
print(f"  Mean:   {np.mean(gnn_f1_values):.4f}")
print(f"  Median: {np.median(gnn_f1_values):.4f}")
print(f"  Std:    {np.std(gnn_f1_values):.4f}")
print(f"  Min:    {np.min(gnn_f1_values):.4f}")
print(f"  Max:    {np.max(gnn_f1_values):.4f}")
print(f"  Users:  {len(gnn_f1_values)}")

# COMMAND ----------

# DBTITLE 1,GNN: Per-user F1 distribution plot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(gnn_f1_values, bins=20, edgecolor='black', alpha=0.7, color='darkorange')
ax.axvline(np.mean(gnn_f1_values), color='red', linestyle='--', label=f'Mean: {np.mean(gnn_f1_values):.3f}')
ax.axvline(np.median(gnn_f1_values), color='blue', linestyle='--', label=f'Median: {np.median(gnn_f1_values):.3f}')
ax.set_xlabel('Test F1 Score')
ax.set_ylabel('Count')
ax.set_title(f'GNN (General) — Per-user Test F1 Distribution — {FINAL_TAG}')
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Comparison: Baseline vs GNN
# Load baseline SVC results
BASELINE_RESULTS_PATH = f"{EXPERIMENT_DIR}/baseline_svc_results.pkl"

if not os.path.exists(BASELINE_RESULTS_PATH):
    print(f"No baseline results found at {BASELINE_RESULTS_PATH} — skipping comparison.")
else:
    with open(BASELINE_RESULTS_PATH, "rb") as f:
        baseline_saved = pickle.load(f)
    baseline_f1s = baseline_saved["baseline_f1s"]
    all_baseline_test_preds = baseline_saved["all_baseline_test_preds"]

    # Combined baseline F1
    all_bl_preds = np.concatenate([p for p, _ in all_baseline_test_preds])
    all_bl_labels = np.concatenate([l for _, l in all_baseline_test_preds])
    combined_f1 = f1_score(all_bl_labels, all_bl_preds)

    # Side-by-side comparison
    common_users = set(baseline_f1s.keys()) & set(gnn_f1s.keys())
    baseline_common = [baseline_f1s[u] for u in common_users]
    gnn_common = [gnn_f1s[u] for u in common_users]

    print(f"=== Comparison (on {len(common_users)} common users) ===")
    print(f"  Baseline SVC mean F1: {np.mean(baseline_common):.4f}")
    print(f"  GNN mean F1:          {np.mean(gnn_common):.4f}")
    print(f"  GNN wins: {sum(g > b for g, b in zip(gnn_common, baseline_common))}/{len(common_users)}")
    print(f"\n  Baseline combined F1 (pooled): {combined_f1:.4f}")
    print(f"  GNN global F1 (pooled):        {global_f1:.4f}")

    # Side-by-side histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    axes[0].hist(baseline_common, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(baseline_common), color='red', linestyle='--', label=f'Mean: {np.mean(baseline_common):.3f}')
    axes[0].set_xlabel('Test F1 Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Baseline SVC (RBF)')
    axes[0].legend()

    axes[1].hist(gnn_common, bins=20, edgecolor='black', alpha=0.7, color='darkorange')
    axes[1].axvline(np.mean(gnn_common), color='red', linestyle='--', label=f'Mean: {np.mean(gnn_common):.3f}')
    axes[1].set_xlabel('Test F1 Score')
    axes[1].set_title('GNN (General)')
    axes[1].legend()

    plt.suptitle(f'Per-user Test F1 Distribution: Baseline vs GNN — {FINAL_TAG}', fontsize=13)
    plt.tight_layout()
    plt.show()