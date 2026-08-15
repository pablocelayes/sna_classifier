# Databricks notebook source
# DBTITLE 1,Parameters
import os
import pickle
import matplotlib.pyplot as plt

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