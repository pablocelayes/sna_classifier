# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,HP Tuning Visualizer
# MAGIC %md
# MAGIC # 4.1 HP Tuning Visualizer
# MAGIC
# MAGIC Visualize training losses and validation F1 scores from GNN hyperparameter tuning runs (notebook 3.2b).
# MAGIC Reads per-trial checkpoint JSON files from DBFS — works with partial results while the job is still running.

# COMMAND ----------

# DBTITLE 1,Parameters
# ============================================================
# Parameters — adjust to match the tuning run you want to inspect
# ============================================================
import os

EXPERIMENT_TAG = "v3_full"  # Must match 3.2b's EXPERIMENT_TAG
N_USERS = None              # Must match 3.2b's N_USERS

FINAL_TAG = f"{EXPERIMENT_TAG}_N{N_USERS}" if N_USERS else EXPERIMENT_TAG

# Where 3.2b checkpoints per-trial history JSONs
HISTORY_CHECKPOINT_DIR = f"/dbfs/tmp/gnn_hparam_history_{FINAL_TAG}"

print(f"Looking for trial histories in: {HISTORY_CHECKPOINT_DIR}")
print(f"Exists: {os.path.isdir(HISTORY_CHECKPOINT_DIR)}")

# COMMAND ----------

# DBTITLE 1,Load trial histories from checkpoint dir
import json
import glob
from datetime import datetime

def load_trial_histories(checkpoint_dir):
    """
    Scan the checkpoint directory for trial_*_history.json files.
    Returns a list of dicts sorted by trial number, each containing:
      - trial_number, params, history, best_val_f1, global_step,
        completed (bool), trial_time_s (if completed), mtime
    Works with partial results (files are written incrementally during training).
    """
    pattern = os.path.join(checkpoint_dir, "trial_*_history.json")
    files = sorted(glob.glob(pattern))
    trials = []
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            # Extract trial number from filename
            fname = os.path.basename(fpath)
            trial_num = int(fname.replace("trial_", "").replace("_history.json", ""))
            data["trial_number"] = trial_num
            data["completed"] = data.get("completed", False)
            data["mtime"] = datetime.fromtimestamp(os.path.getmtime(fpath))
            trials.append(data)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  Skipping {fpath}: {e}")
            continue
    trials.sort(key=lambda t: t["trial_number"])
    return trials

trials = load_trial_histories(HISTORY_CHECKPOINT_DIR)

# Summary
n_completed = sum(1 for t in trials if t["completed"])
n_running = len(trials) - n_completed
print(f"Found {len(trials)} trial(s): {n_completed} completed, {n_running} in-progress")
for t in trials:
    h = t["history"]
    n_steps = len(h.get("step", []))
    n_epochs = len(h.get("epoch_step", []))
    status = "done" if t["completed"] else "running"
    best = t.get("best_val_f1", 0)
    if t.get("trial_time_s"):
        total_min = t["trial_time_s"] / 60
        hrs, mins = divmod(total_min, 60)
        time_str = f", {int(hrs)}h:{int(mins):02d}m" if hrs >= 1 else f", {total_min:.1f}min"
    else:
        time_str = ""
    print(f"  Trial {t['trial_number']:>2}: {n_steps} checkpoints, {n_epochs} epochs, "
          f"best_val_f1={best:.4f}, [{status}{time_str}] (updated {t['mtime']:%H:%M:%S})")

# COMMAND ----------

# DBTITLE 1,Per-trial training curves (plotly)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Key hparams to show in each plot title
TITLE_PARAMS = ["lr", "weight_decay", "batch_size", "dropout", "drop_edge_rate",
                "grad_accum_steps", "lr_schedule", "warmup_epochs"]

def format_param(k, v):
    """Format a single hparam value for the title."""
    if isinstance(v, float):
        if abs(v) < 0.01 or abs(v) >= 1000:
            return f"{k}={v:.1e}"
        return f"{k}={v:.4g}"
    return f"{k}={v}"

def _make_title(trial_num, completed, best_f1, params):
    status_tag = "DONE" if completed else "RUNNING"
    param_parts = [format_param(k, params[k]) for k in TITLE_PARAMS if k in params]
    line1 = f"Trial {trial_num} [{status_tag}] — best val F1: {best_f1:.4f}"
    line2 = "  |  ".join(param_parts) if param_parts else ""
    return f"{line1}<br><span style='font-size:11px'>{line2}</span>"

LAYOUT_COMMON = dict(
    height=400, width=1000, template="plotly_white",
    showlegend=True,
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    margin=dict(t=80, b=100),
)

for t in trials:
    h = t["history"]
    params = t.get("params", {})
    trial_num = t["trial_number"]
    completed = t["completed"]
    best_f1 = t.get("best_val_f1", 0)
    title_base = _make_title(trial_num, completed, best_f1, params)

    steps = h.get("step", [])
    epoch_steps = h.get("epoch_step", [])

    # ── Plot 1: Train Loss (checkpoint-level) ──
    train_loss = h.get("train_loss", [])
    if steps and train_loss:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps, y=train_loss, mode="lines", name="Train loss",
            line=dict(color="royalblue", width=1.5)))
        # overlay epoch-end markers if available
        epoch_train_loss = h.get("epoch_train_loss", [])
        if epoch_steps and epoch_train_loss:
            fig.add_trace(go.Scatter(
                x=epoch_steps, y=epoch_train_loss, mode="markers",
                name="Epoch end", marker=dict(size=8, symbol="diamond", color="dodgerblue")))
        fig.update_layout(
            title=dict(text=f"Train Loss — {title_base}", font=dict(size=13)),
            xaxis_title="Global Step", yaxis_title="Loss", **LAYOUT_COMMON)
        fig.show()

    # ── Plot 2: Val F1 (checkpoint-level) ──
    val_f1 = h.get("val_f1", [])
    if steps and val_f1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps, y=val_f1, mode="lines", name="Val F1",
            line=dict(color="crimson", width=1.5)))
        epoch_val_f1 = h.get("epoch_val_f1", [])
        if epoch_steps and epoch_val_f1:
            fig.add_trace(go.Scatter(
                x=epoch_steps, y=epoch_val_f1, mode="markers",
                name="Epoch end", marker=dict(size=8, symbol="diamond", color="red")))
        if best_f1 > 0:
            fig.add_hline(y=best_f1, line_dash="dash", line_color="green", opacity=0.5,
                          annotation_text=f"best={best_f1:.4f}", annotation_position="top left")
        fig.update_layout(
            title=dict(text=f"Val F1 — {title_base}", font=dict(size=13)),
            xaxis_title="Global Step", yaxis_title="F1 Score", **LAYOUT_COMMON)
        fig.show()

    # ── Plot 3: Val Loss (epoch-level, or checkpoint if available) ──
    val_loss_ckpt = h.get("val_loss", [])
    epoch_val_loss = h.get("epoch_val_loss", [])
    has_val_loss = (steps and val_loss_ckpt) or (epoch_steps and epoch_val_loss)
    if has_val_loss:
        fig = go.Figure()
        if steps and val_loss_ckpt:
            fig.add_trace(go.Scatter(
                x=steps, y=val_loss_ckpt, mode="lines", name="Val loss (ckpt)",
                line=dict(color="crimson", width=1.5)))
        if epoch_steps and epoch_val_loss:
            fig.add_trace(go.Scatter(
                x=epoch_steps, y=epoch_val_loss, mode="lines+markers",
                name="Val loss (epoch)", marker=dict(size=8, symbol="diamond", color="red"),
                line=dict(color="red", width=2, dash="dot")))
        fig.update_layout(
            title=dict(text=f"Val Loss — {title_base}", font=dict(size=13)),
            xaxis_title="Global Step", yaxis_title="Loss", **LAYOUT_COMMON)
        fig.show()

    # ── Plot 4: Train F1 (epoch-level, or checkpoint if available) ──
    train_f1_ckpt = h.get("train_f1", [])
    epoch_train_f1 = h.get("epoch_train_f1", [])
    has_train_f1 = (steps and train_f1_ckpt) or (epoch_steps and epoch_train_f1)
    if has_train_f1:
        fig = go.Figure()
        if steps and train_f1_ckpt:
            fig.add_trace(go.Scatter(
                x=steps, y=train_f1_ckpt, mode="lines", name="Train F1 (ckpt)",
                line=dict(color="royalblue", width=1.5)))
        if epoch_steps and epoch_train_f1:
            fig.add_trace(go.Scatter(
                x=epoch_steps, y=epoch_train_f1, mode="lines+markers",
                name="Train F1 (epoch)", marker=dict(size=8, symbol="diamond", color="dodgerblue"),
                line=dict(color="dodgerblue", width=2, dash="dot")))
        fig.update_layout(
            title=dict(text=f"Train F1 — {title_base}", font=dict(size=13)),
            xaxis_title="Global Step", yaxis_title="F1 Score", **LAYOUT_COMMON)
        fig.show()

# COMMAND ----------

# DBTITLE 1,Cross-trial comparison
# ============================================================
# Cross-trial overview: best val F1 per trial + convergence
# ============================================================
import plotly.express as px
import pandas as pd

if not trials:
    print("No trials found.")
else:
    # --- 1. Bar chart: best val F1 per trial ---
    records = []
    for t in trials:
        params = t.get("params", {})
        records.append({
            "trial": t["trial_number"],
            "best_val_f1": t.get("best_val_f1", 0),
            "status": "completed" if t["completed"] else "running",
            "lr": params.get("lr"),
            "weight_decay": params.get("weight_decay"),
            "batch_size": params.get("batch_size"),
            "dropout": params.get("dropout"),
            "drop_edge_rate": params.get("drop_edge_rate"),
            "grad_accum_steps": params.get("grad_accum_steps"),
            "lr_schedule": params.get("lr_schedule"),
            "warmup_epochs": params.get("warmup_epochs"),
            "n_checkpoints": len(t["history"].get("step", [])),
            "time_min": t.get("trial_time_s", 0) / 60 if t.get("trial_time_s") else None,
        })
    df = pd.DataFrame(records).sort_values("best_val_f1", ascending=False)

    fig_bar = px.bar(
        df, x="trial", y="best_val_f1", color="status",
        color_discrete_map={"completed": "steelblue", "running": "orange"},
        hover_data=["lr", "weight_decay", "batch_size", "dropout",
                    "drop_edge_rate", "grad_accum_steps", "lr_schedule"],
        title="Best Validation F1 per Trial",
        labels={"best_val_f1": "Best Val F1", "trial": "Trial #"},
    )
    fig_bar.update_layout(height=400, width=900, template="plotly_white",
                          xaxis=dict(dtick=1))
    fig_bar.show()

    # --- 2. All val F1 curves overlaid ---
    fig_overlay = go.Figure()
    for t in trials:
        h = t["history"]
        steps = h.get("step", [])
        val_f1 = h.get("val_f1", [])
        if steps and val_f1:
            params = t.get("params", {})
            lr_str = f"{params.get('lr', '?'):.1e}" if isinstance(params.get('lr'), float) else "?"
            label = f"T{t['trial_number']} (lr={lr_str})"
            fig_overlay.add_trace(
                go.Scatter(x=steps, y=val_f1, mode="lines", name=label,
                           opacity=0.7, line=dict(width=1.5))
            )
    fig_overlay.update_layout(
        title="All Trials — Val F1 vs Global Step",
        xaxis_title="Global Step", yaxis_title="Val F1",
        height=450, width=1000, template="plotly_white",
        legend=dict(font=dict(size=10)),
    )
    fig_overlay.show()

    # --- 3. Summary table ---
    print("\n=== Trial Summary (sorted by best val F1) ===")
    display(df)

# COMMAND ----------

# DBTITLE 1,Install optuna
# MAGIC %pip install optuna -q

# COMMAND ----------

# DBTITLE 1,Parallel coordinates: hparams vs val F1 (Optuna)
# ============================================================
# Parallel coordinates plot via Optuna's plot_parallel_coordinate
# Reconstructs an in-memory Optuna study from the saved trial JSONs.
# ============================================================
import optuna
from optuna.visualization import plot_parallel_coordinate

if not trials:
    print("No trials found.")
else:
    # Build an in-memory study and populate it with FrozenTrial objects
    study = optuna.create_study(direction="maximize", study_name="gnn_hparam_tuning")

    PARAM_KEYS = [
        "lr", "weight_decay", "batch_size", "dropout",
        "drop_edge_rate", "grad_accum_steps", "lr_schedule", "warmup_epochs",
    ]

    for t in trials:
        p = t.get("params", {})
        params = {k: p[k] for k in PARAM_KEYS if k in p}

        # Build distributions so Optuna knows each param's type
        distributions = {}
        for k, v in params.items():
            if isinstance(v, float):
                distributions[k] = optuna.distributions.FloatDistribution(v, v)
            elif isinstance(v, int):
                distributions[k] = optuna.distributions.IntDistribution(v, v)
            else:
                distributions[k] = optuna.distributions.CategoricalDistribution([v])

        frozen = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            values=[t.get("best_val_f1", 0)],
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(frozen)

    fig = plot_parallel_coordinate(study, params=PARAM_KEYS)
    fig.update_layout(
        title="Parallel Coordinates — Hyperparameters vs Best Val F1 (Optuna)",
        height=500, width=1100, template="plotly_white",
        margin=dict(t=80, b=40, l=80, r=40),
    )
    fig.show()

# COMMAND ----------

# DBTITLE 1,Auto-monitor: cancel tuning run after 6 trials
import time
from databricks.sdk import WorkspaceClient

JOB_ID = 81399685941470
RUN_ID = 3829506390716583
TARGET_COMPLETED = 16
CHECK_INTERVAL_MIN = 20

w = WorkspaceClient()

while True:
    trials_now = load_trial_histories(HISTORY_CHECKPOINT_DIR)
    n_completed = sum(1 for t in trials_now if t["completed"])
    n_total = len(trials_now)
    best_f1 = max((t.get("best_val_f1", 0) for t in trials_now), default=0)

    print(f"[{datetime.now():%H:%M:%S}] {n_completed}/{n_total} trials completed "
          f"(best F1: {best_f1:.4f})")

    if n_completed >= TARGET_COMPLETED:
        print(f"\n✓ {n_completed} trials completed — cancelling run {RUN_ID}...")
        w.jobs.cancel_run(RUN_ID)
        print("Run cancelled successfully!")
        break

    print(f"  Next check in {CHECK_INTERVAL_MIN} min...")
    time.sleep(CHECK_INTERVAL_MIN * 60)