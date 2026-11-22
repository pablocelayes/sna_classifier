#!/usr/bin/env python3
"""Script version of the GNN training part of notebook
"3.1 gnn model for general users - full training data.ipynb".

The baseline SVC step (notebook Step 1) is intentionally omitted.

Pipeline:
  1. (optional) reset training artifacts in the experiment dir
  2. load graph + per-user dataframes (deterministic user sample per tag)
  3. build on-the-fly train loader, cached val loader, sampled train-eval loader
  4. build RetweetGNN (optionally warm-started from pre-trained weights)
  5. train with checkpointing / resume, log to MLflow
  6. log summary metrics, model, history and training curves

Defaults match the notebook's parameter cells. Optional numeric/path
arguments accept 'none' to disable them.

Example:
  python train_gnn.py --experiment-tag v3_full --epochs 10 --reset-gnn-training
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import gc
import glob
import json
import logging
import random
import shutil
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT_HOME = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

import matplotlib

matplotlib.use("Agg")  # headless (cluster jobs have no display)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.data import Batch, Data

warnings.filterwarnings("ignore")

TRAIN_GROUPS = ["u_train", "au_train"]
BASE_LR = 3e-3
BASE_WD = 1e-3

# Model architecture (fixed in the notebook)
FF_HIDDEN_DIM = 64
GCN_HIDDEN_DIM = 64
TRANSFORMER_DIM = 64
TRANSFORMER_HEADS = 4


# =============================================================================
# CLI
# =============================================================================
def _none_or(cast):
    def _parse(value):
        if value is None or str(value).strip().lower() in {"none", "null", ""}:
            return None
        return cast(value)

    _parse.__name__ = f"none_or_{cast.__name__}"
    return _parse


none_or_int = _none_or(int)
none_or_float = _none_or(float)
none_or_str = _none_or(str)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100),
    )

    g = p.add_argument_group("experiment")
    g.add_argument("--experiment-tag", default="v3_full", help="Experiment identifier.")
    g.add_argument("--n-users", type=none_or_int, default=None,
                   help="Total users to sample proportionally across groups ('none' = all users).")
    g.add_argument("--experiments-root", default=os.path.join(SCRIPT_DIR, "experiments"),
                   help="Parent dir for experiment output folders.")
    g.add_argument("--reset-gnn-training", action=argparse.BooleanOptionalAction, default=True,
                   help="Clear checkpoints/logs and train from scratch. "
                        "--no-reset-gnn-training resumes from an existing checkpoint.")
    g.add_argument("--seed", type=none_or_int, default=None,
                   help="Global RNG seed for random/numpy/torch ('none' = unseeded, as in the notebook).")

    g = p.add_argument_group("paths")
    g.add_argument("--project-home", default=DEFAULT_PROJECT_HOME,
                   help="sna_classifier repo root (added to sys.path for tw_dataset).")
    g.add_argument("--data-path", default="/serafin/pcelayes/repos/sna_classifier/data",
                   help="Data dir containing datasets/user_splits.json.")
    g.add_argument("--embeddings-path", default=None,
                   help="Node embeddings .pt (default: <data-path>/node_embeddings.pt).")
    g.add_argument("--graph-path", default=None,
                   help="GraphML graph (default: tw_dataset.settings.IG_GRAPH_PATH).")
    g.add_argument("--hparams-file", type=none_or_str,
                   default=os.path.join(SCRIPT_DIR, "hparams_from_gpu_tuning_v2.json"),
                   help="Tuned hparams JSON overriding batch size, dropout, lr, etc. ('none' = disabled).")
    g.add_argument("--init-weights-path", type=none_or_str,
                   default=os.path.join(SCRIPT_DIR, "experiments/v2_N20/best_retweet_gnn_general.pt"),
                   help="Pre-trained weights for warm start ('none' = train from scratch).")

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=2)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--train-chunk-size", type=int, default=10,
                   help="Users per chunk for on-the-fly sample generation.")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--drop-edge-rate", type=float, default=0.1)
    g.add_argument("--log-every-n-steps", type=int, default=2000)
    g.add_argument("--train-f1-every-n-epochs", type=none_or_int, default=1)
    g.add_argument("--patience-steps", type=none_or_int, default=30_000,
                   help="Early stopping in steps without val F1 improvement ('none' = disabled).")
    g.add_argument("--gradient-accumulation-steps", type=none_or_int, default=8)
    g.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--target-pos-rate", type=none_or_float, default=0.10,
                   help="Oversample positives in train loader to this rate ('none' = disabled).")
    g.add_argument("--max-val-samples", type=none_or_int, default=25_000)
    g.add_argument("--max-train-eval-samples", type=none_or_int, default=25_000)
    g.add_argument("--finetune-lr-factor", type=float, default=0.05)
    g.add_argument("--finetune-wd-factor", type=float, default=0.2)
    g.add_argument("--finetune-warmup-epochs", type=int, default=2)

    g = p.add_argument_group("runtime")
    g.add_argument("--device", default="auto",
                   help="'auto' = GPU with most free memory (respects CUDA_VISIBLE_DEVICES), "
                        "or e.g. 'cuda:0', 'cpu'.")
    g.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=True,
                   help="Log params/metrics/artifacts to MLflow.")
    g.add_argument("--mlflow-tracking-uri", default=None,
                   help="MLflow tracking URI (default: MLFLOW_TRACKING_URI env or ./mlruns).")
    g.add_argument("--mlflow-experiment-name", default=None,
                   help="MLflow experiment name (default: '<project-home>/<final-tag>/', as in the notebook).")

    args = p.parse_args()

    args.final_tag = f"{args.experiment_tag}_N{args.n_users}" if args.n_users else args.experiment_tag
    args.experiment_dir = os.path.join(args.experiments_root, args.final_tag)
    if args.embeddings_path is None:
        args.embeddings_path = os.path.join(args.data_path, "node_embeddings.pt")
    args.patience = (
        -(-args.patience_steps // args.log_every_n_steps) if args.patience_steps else None
    )
    return args


# =============================================================================
# Setup helpers
# =============================================================================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)


def seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seeded RNGs with {seed}")


def reset_training_artifacts(experiment_dir):
    print("RESET_GNN_TRAINING=True — clearing checkpoints and training logs...")
    deleted = []
    for fname in [
        "best_retweet_gnn_general.pt",
        "training.log",
        "training_history.json",
        "training_history.pkl",
        "checkpoint.pt",
        "training_checkpoint.pt",
        "train_eval_metadata.json",
        "training_config.json",
    ]:
        fpath = os.path.join(experiment_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            deleted.append(fpath)
    for meta_file in glob.glob(os.path.join(experiment_dir, "train_eval_metadata_*.json")):
        os.remove(meta_file)
        deleted.append(meta_file)
    legacy_cache = os.path.join(experiment_dir, "train_eval_samples_cache")
    if os.path.isdir(legacy_cache):
        shutil.rmtree(legacy_cache)
        deleted.append(legacy_cache)
    print(f"  Cleared {len(deleted)} training artifacts. GNN training will start fresh.")
    for d in deleted:
        print(f"  - {d}")


def apply_hparam_overrides(args):
    """Override args from the tuned hparams JSON (same keys as the notebook)."""
    args.tuned_lr = None
    args.tuned_wd = None
    if not args.hparams_file:
        return
    with open(args.hparams_file) as f:
        tuned = json.load(f)
    args.batch_size = tuned.get("batch_size", args.batch_size)
    args.train_chunk_size = tuned.get("train_chunk_size", args.train_chunk_size)
    args.gradient_accumulation_steps = tuned.get("grad_accum_steps", args.gradient_accumulation_steps)
    args.dropout = tuned.get("dropout", args.dropout)
    args.drop_edge_rate = tuned.get("drop_edge_rate", args.drop_edge_rate)
    args.finetune_warmup_epochs = tuned.get("warmup_epochs", args.finetune_warmup_epochs)
    args.tuned_lr = tuned.get("lr")
    args.tuned_wd = tuned.get("weight_decay")
    print(f"Loaded tuned hparams from: {args.hparams_file}")


def select_device(spec):
    if spec != "auto":
        return torch.device(spec)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # Use torch (not nvidia-smi) so indices match CUDA_VISIBLE_DEVICES set by SLURM.
    free = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        f, t = torch.cuda.mem_get_info(i)
        free.append(f)
        print(f"GPU {i}: {props.name}, free={f / 1e9:.1f}GB / {t / 1e9:.1f}GB")
    best = max(range(len(free)), key=lambda i: free[i])
    print(f"Selected GPU {best}")
    return torch.device(f"cuda:{best}")


class TeeLogger:
    """Tee stdout to both the original stream and a timestamped log file."""

    def __init__(self, log_path, original_stdout):
        self._original = original_stdout
        self._file = open(log_path, "a", buffering=1)
        self._line_buffer = ""

    def write(self, msg):
        self._original.write(msg)
        for char in msg:
            if char == "\n":
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._file.write(f"[{ts}] {self._line_buffer}\n")
                self._line_buffer = ""
            else:
                self._line_buffer += char

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        if self._line_buffer:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._file.write(f"[{ts}] {self._line_buffer}\n")
            self._line_buffer = ""
        self._file.close()


@contextmanager
def tee_to_log(log_path):
    original_stdout = sys.stdout
    tee = TeeLogger(log_path, original_stdout)
    sys.stdout = tee
    try:
        yield log_path
    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"Training log saved to: {log_path}")


# =============================================================================
# Sample conversion / allocation helpers
# =============================================================================
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
        [1.0 if uid in retweeted_set else 0.0 for uid in all_ids], dtype=torch.float
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
        central_mask=torch.zeros(num_nodes, dtype=torch.bool).index_fill_(0, torch.tensor([0]), True),
    )


def _proportional_allocate(row_counts, max_samples):
    """Proportionally allocate max_samples across items (largest-remainder, min 1 each)."""
    total_available = sum(row_counts)
    if not max_samples or total_available <= max_samples:
        return list(row_counts)

    sample_rate = max_samples / total_available
    raw = [n * sample_rate for n in row_counts]
    allocations = [max(1, int(r)) for r in raw]

    deficit = max_samples - sum(allocations)
    if deficit > 0:
        remainders = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in remainders[:deficit]:
            allocations[i] += 1

    while sum(allocations) > max_samples:
        max_idx = max(range(len(allocations)), key=lambda i: allocations[i])
        allocations[max_idx] -= 1

    return allocations


# =============================================================================
# Loaders (identical logic to the notebook)
# =============================================================================
class ChunkedGNNTrainLoader:
    """Training loader that transforms GNN samples on-the-fly in user chunks.

    Each epoch shuffles users (seeded), processes them in chunks, oversamples
    positives per chunk if target_pos_rate is set, shuffles and yields batches.
    Supports deterministic mid-epoch resume via get_state()/set_resume_state().
    """

    def __init__(self, user_items, user_data, graph, batch_size, chunk_size=10, target_pos_rate=None):
        self.user_items = list(user_items)
        self.user_data = user_data
        self.graph = graph
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.target_pos_rate = target_pos_rate
        self._failed_users = []
        self._estimated_samples = None
        self._resume_state = None
        self._current_seed = None
        self._current_chunk_idx = 0

    @property
    def total_samples(self):
        if self._estimated_samples is None:
            self._estimated_samples = sum(self.get_label_counts().values())
        return self._estimated_samples

    def __len__(self):
        return self.total_samples // self.batch_size

    def get_state(self):
        return (self._current_seed, self._current_chunk_idx)

    def set_resume_state(self, epoch_seed, chunk_index):
        self._resume_state = (epoch_seed, chunk_index)

    def __iter__(self):
        if self._resume_state is not None:
            epoch_seed, start_chunk = self._resume_state
            self._resume_state = None
        else:
            epoch_seed = random.randint(0, 2**31 - 1)
            start_chunk = 0

        self._current_seed = epoch_seed

        users = list(self.user_items)
        random.Random(epoch_seed).shuffle(users)

        total_yielded = 0
        self._failed_users = []

        chunks = list(range(0, len(users), self.chunk_size))
        for chunk_idx, chunk_start in enumerate(chunks):
            self._current_chunk_idx = chunk_idx
            if chunk_idx < start_chunk:
                continue

            chunk_users = users[chunk_start: chunk_start + self.chunk_size]

            chunk_data_list = []
            for group, uid in chunk_users:
                X_tr, X_te, y_tr, y_te = self.user_data[group][uid]
                try:
                    train_samples, _ = create_gnn_train_val_samples(uid, self.graph, X_tr, y_tr, X_te, y_te)
                    for s in train_samples:
                        chunk_data_list.append(_sample_to_pyg_data(s))
                except Exception as e:
                    self._failed_users.append((group, uid, str(e)))
                    continue

            if not chunk_data_list:
                continue

            chunk_rng = random.Random(epoch_seed + chunk_idx + 1)
            chunk_np_rng = np.random.RandomState(epoch_seed + chunk_idx + 1)

            if self.target_pos_rate is not None:
                pos_idx = [i for i, d in enumerate(chunk_data_list) if d.y.item() == 1]
                n_pos = len(pos_idx)
                n_neg = len(chunk_data_list) - n_pos
                if n_pos > 0 and n_neg > 0:
                    target_n_pos = int(np.ceil(self.target_pos_rate * n_neg / (1.0 - self.target_pos_rate)))
                    n_extra = target_n_pos - n_pos
                    if n_extra > 0:
                        extra_idx = chunk_np_rng.choice(pos_idx, size=n_extra, replace=True)
                        chunk_data_list.extend(chunk_data_list[i] for i in extra_idx)

            chunk_rng.shuffle(chunk_data_list)

            for start in range(0, len(chunk_data_list) - self.batch_size + 1, self.batch_size):
                yield Batch.from_data_list(chunk_data_list[start: start + self.batch_size])
                total_yielded += self.batch_size

            del chunk_data_list
            gc.collect()

        if total_yielded > 0:
            self._estimated_samples = total_yielded

    def get_label_counts(self):
        """Count labels from y_tr, adjusted for oversampling if target_pos_rate is set."""
        counts = {}
        for group, uid in self.user_items:
            _, _, y_tr, _ = self.user_data[group][uid]
            for label in np.asarray(y_tr).ravel():
                counts[int(label)] = counts.get(int(label), 0) + 1
        if self.target_pos_rate is not None and 1 in counts and 0 in counts:
            target_n_pos = int(np.ceil(self.target_pos_rate * counts[0] / (1.0 - self.target_pos_rate)))
            if target_n_pos > counts[1]:
                counts[1] = target_n_pos
        return counts


class CachedValLoader:
    """Validation loader streaming pre-computed samples from disk, one chunk at a time."""

    CHUNK_SIZE = 1000  # samples per chunk file

    def __init__(self, val_user_splits, user_data, graph, batch_size, max_samples=None, cache_path=None):
        self.batch_size = batch_size
        self._cache_path = cache_path
        self._total_samples = 0
        self._label_counts = {}
        self._failed_users = []
        self._chunk_files = []

        if cache_path and os.path.isdir(cache_path):
            meta_path = os.path.join(cache_path, "metadata.json")
            chunk_files = sorted(
                f for f in os.listdir(cache_path) if f.startswith("chunk_") and f.endswith(".pt")
            )
            if chunk_files and os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self._chunk_files = [os.path.join(cache_path, cf) for cf in chunk_files]
                self._total_samples = meta["total_samples"]
                self._label_counts = {int(k): v for k, v in meta["label_counts"].items()}
                print(f"Loaded val cache metadata from: {cache_path}")
                print(f"  {self._total_samples} samples, {len(self._chunk_files)} chunks, {len(self)} batches")
                return

        assert cache_path, "cache_path is required for CachedValLoader (streaming mode)"
        os.makedirs(cache_path, exist_ok=True)

        n_total = len(val_user_splits)
        print(f"Pre-computing validation samples ({n_total} user/split combos)...")

        row_counts = []
        for group, uid, split_name in val_user_splits:
            X_tr, X_te, _, _ = user_data[group][uid]
            row_counts.append(X_tr.shape[0] if split_name == "train" else X_te.shape[0])
        total_available_rows = sum(row_counts)

        allocations = _proportional_allocate(row_counts, max_samples)
        do_subsample = bool(max_samples) and total_available_rows > max_samples
        if do_subsample:
            print(f"  Sampling enabled: {max_samples} target from {total_available_rows} available rows"
                  f" (allocated {sum(allocations)})")

        pending_samples = []
        chunk_idx = 0

        def _flush_chunk():
            nonlocal pending_samples, chunk_idx
            if not pending_samples:
                return
            chunk_path = os.path.join(cache_path, f"chunk_{chunk_idx:03d}.pt")
            torch.save(pending_samples, chunk_path)
            self._chunk_files.append(chunk_path)
            self._total_samples += len(pending_samples)
            for data in pending_samples:
                label = data.y.item()
                self._label_counts[label] = self._label_counts.get(label, 0) + 1
            chunk_idx += 1
            pending_samples = []
            gc.collect()

        progress_every = max(1, n_total // 20)
        for i, (group, uid, split_name) in enumerate(val_user_splits):
            if (i + 1) % progress_every == 0 or (i + 1) == n_total:
                print(f"  [{i + 1}/{n_total}] {self._total_samples + len(pending_samples)} samples, "
                      f"{chunk_idx} chunks saved...", flush=True)

            X_tr, X_te, y_tr, y_te = user_data[group][uid]
            n_keep = allocations[i]

            try:
                if do_subsample:
                    if split_name == "test":
                        idx = np.random.choice(X_te.shape[0], size=min(n_keep, X_te.shape[0]), replace=False)
                        _, samples = create_gnn_train_val_samples(
                            uid, graph, X_tr.iloc[:1], y_tr.iloc[:1], X_te.iloc[idx], y_te.iloc[idx]
                        )
                    else:
                        idx = np.random.choice(X_tr.shape[0], size=min(n_keep, X_tr.shape[0]), replace=False)
                        samples, _ = create_gnn_train_val_samples(
                            uid, graph, X_tr.iloc[idx], y_tr.iloc[idx], X_te.iloc[:1], y_te.iloc[:1]
                        )
                else:
                    train_samples, test_samples = create_gnn_train_val_samples(uid, graph, X_tr, y_tr, X_te, y_te)
                    samples = train_samples if split_name == "train" else test_samples

                for s in samples:
                    pending_samples.append(_sample_to_pyg_data(s))
                    if len(pending_samples) >= self.CHUNK_SIZE:
                        _flush_chunk()
            except Exception as e:
                self._failed_users.append((group, uid, str(e)))
                continue

        _flush_chunk()

        with open(os.path.join(cache_path, "metadata.json"), "w") as f:
            json.dump({"total_samples": self._total_samples, "label_counts": self._label_counts}, f)

        gc.collect()
        print(f"  Saved {self._total_samples} val samples in {len(self._chunk_files)} chunks to: {cache_path}"
              f" (from {n_total - len(self._failed_users)} user/split combos, "
              f"rows available: {total_available_rows}, failed: {len(self._failed_users)})")

    @property
    def total_samples(self):
        return self._total_samples

    def __len__(self):
        return -(-self._total_samples // self.batch_size)

    def __iter__(self):
        carry_over = []
        for chunk_path in self._chunk_files:
            chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
            samples = carry_over + chunk_data
            del chunk_data
            start = 0
            while start + self.batch_size <= len(samples):
                yield Batch.from_data_list(samples[start: start + self.batch_size])
                start += self.batch_size
            carry_over = samples[start:]
            del samples
            gc.collect()
        if carry_over:
            yield Batch.from_data_list(carry_over)

    def get_label_counts(self):
        return dict(self._label_counts)


class SampledTrainEvalLoader:
    """Train-eval loader over a fixed, persisted subset of train rows (generated on-the-fly)."""

    CHUNK_SIZE = 10  # users per chunk during iteration

    def __init__(self, train_user_items, user_data, graph, batch_size, max_samples, metadata_path):
        self.user_data = user_data
        self.graph = graph
        self.batch_size = batch_size
        self._failed_users = []
        self._total_samples = 0
        self._label_counts = {}

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            self._allocations = meta["allocations"]
            self._total_samples = meta["total_samples"]
            self._label_counts = {int(k): v for k, v in meta["label_counts"].items()}
            print(f"Loaded train-eval metadata from: {metadata_path}")
            print(f"  {self._total_samples} samples across {len(self._allocations)} users")
            return

        row_counts = [user_data[group][uid][0].shape[0] for group, uid in train_user_items]
        total_available = sum(row_counts)
        allocations = _proportional_allocate(row_counts, max_samples)

        self._allocations = []
        for i, (group, uid) in enumerate(train_user_items):
            X_tr, _, y_tr, _ = user_data[group][uid]
            n_available = X_tr.shape[0]
            idx = np.random.choice(n_available, size=min(allocations[i], n_available), replace=False)
            self._allocations.append({"group": group, "uid": uid, "row_indices": sorted(idx.tolist())})
            for label in np.asarray(y_tr).ravel()[idx]:
                self._label_counts[int(label)] = self._label_counts.get(int(label), 0) + 1

        self._total_samples = sum(len(a["row_indices"]) for a in self._allocations)

        with open(metadata_path, "w") as f:
            json.dump({
                "total_samples": self._total_samples,
                "label_counts": self._label_counts,
                "allocations": self._allocations,
            }, f)
        print(f"Created train-eval metadata: {metadata_path}")
        print(f"  {self._total_samples} samples from {total_available} available "
              f"({len(self._allocations)} users)")

    @property
    def total_samples(self):
        return self._total_samples

    def __len__(self):
        return -(-self._total_samples // self.batch_size)

    def get_label_counts(self):
        return dict(self._label_counts)

    def __iter__(self):
        self._failed_users = []
        buffer = []
        for chunk_start in range(0, len(self._allocations), self.CHUNK_SIZE):
            for alloc in self._allocations[chunk_start: chunk_start + self.CHUNK_SIZE]:
                group, uid, row_indices = alloc["group"], alloc["uid"], alloc["row_indices"]
                X_tr, X_te, y_tr, y_te = self.user_data[group][uid]
                try:
                    train_samples, _ = create_gnn_train_val_samples(
                        uid, self.graph, X_tr.iloc[row_indices], y_tr.iloc[row_indices],
                        X_te.iloc[:1], y_te.iloc[:1],
                    )
                    for s in train_samples:
                        buffer.append(_sample_to_pyg_data(s))
                except Exception as e:
                    self._failed_users.append((group, uid, str(e)))
                    continue
            while len(buffer) >= self.batch_size:
                yield Batch.from_data_list(buffer[: self.batch_size])
                buffer = buffer[self.batch_size:]
        if buffer:
            yield Batch.from_data_list(buffer)


# =============================================================================
# Data loading
# =============================================================================
def _load_valid_user(uid, group, failed_users, empty_reason):
    try:
        X_tr, X_te, y_tr, y_te = load_dataframe_raw(uid, sparse=True)
        if X_tr.shape[0] > 0 and X_te.shape[0] > 0 and y_tr.sum() > 0 and y_te.sum() > 0:
            return (X_tr, X_te, y_tr, y_te)
        failed_users.append((group, uid, empty_reason))
    except Exception as e:
        failed_users.append((group, uid, str(e)))
    return None


def _proportional_user_sample(user_data, user_splits, test_groups, n_users):
    """Sample n_users: keep train/test proportions; u_train first, then au_train."""
    group_sizes = {g: len(user_data[g]) for g in user_splits}
    total_available = sum(group_sizes.values())

    train_available = sum(group_sizes.get(g, 0) for g in TRAIN_GROUPS)
    train_slots = int(round(n_users * train_available / total_available))
    test_slots = n_users - train_slots

    n_u_train = min(group_sizes.get("u_train", 0), train_slots)
    n_au_train = min(group_sizes.get("au_train", 0), train_slots - n_u_train)
    raw_alloc = {"u_train": n_u_train, "au_train": n_au_train}

    total_test_available = sum(group_sizes.get(g, 0) for g in test_groups)
    for g in test_groups:
        if total_test_available > 0 and group_sizes.get(g, 0) > 0:
            raw_alloc[g] = int(round(test_slots * group_sizes[g] / total_test_available))
        else:
            raw_alloc[g] = 0
    test_diff = test_slots - sum(raw_alloc.get(g, 0) for g in test_groups)
    for g in sorted(test_groups, key=lambda g: group_sizes.get(g, 0), reverse=True):
        if test_diff == 0:
            break
        adjustment = 1 if test_diff > 0 else -1
        raw_alloc[g] = max(1, raw_alloc[g] + adjustment)
        test_diff -= adjustment

    sampled = {}
    for g in user_splits:
        if raw_alloc.get(g, 0) == 0:
            sampled[g] = {}
            continue
        uids = list(user_data[g].keys())
        chosen = random.sample(uids, min(raw_alloc[g], len(uids)))
        sampled[g] = {uid: user_data[g][uid] for uid in chosen}

    print(f"\nSampled {n_users} users (train priority: u_train first, then au_train):")
    for g in user_splits:
        print(f"  {g}: {len(sampled[g])} (target {raw_alloc.get(g, 0)})")
    print(f"  Total sampled: {sum(len(sampled[g]) for g in user_splits)}")
    return sampled


def load_users(args):
    with open(os.path.join(args.data_path, "datasets", "user_splits.json")) as f:
        user_splits = json.load(f)

    print(f"Groups in user_splits: {list(user_splits.keys())}")
    for k, v in user_splits.items():
        print(f"  {k}: {len(v)} users")

    test_groups = [g for g in user_splits if g not in TRAIN_GROUPS]
    print(f"\nTrain groups: {TRAIN_GROUPS}")
    print(f"Test groups: {test_groups}")

    sample_path = os.path.join(args.experiment_dir, "user_sample.json")
    user_data = {}
    failed_users = []

    if os.path.exists(sample_path):
        with open(sample_path) as f:
            saved_sample = json.load(f)
        print(f"\nLoading saved user sample from {sample_path}")
        for group, uids in saved_sample.items():
            user_data[group] = {}
            for uid in uids:
                d = _load_valid_user(uid, group, failed_users, "empty data on reload")
                if d is not None:
                    user_data[group][uid] = d
        print("  Loaded users per group:")
        for group in saved_sample:
            print(f"    {group}: {len(user_data[group])}/{len(saved_sample[group])}")
        print(f"  Total: {sum(len(user_data[g]) for g in user_data)}")
        if failed_users:
            print(f"  Failed on reload: {len(failed_users)}")
    else:
        print(f"\nNo saved sample for {args.final_tag}, loading all users...")
        t0 = time.time()
        for group, group_uids in user_splits.items():
            user_data[group] = {}
            for uid in group_uids:
                d = _load_valid_user(uid, group, failed_users, "empty data")
                if d is not None:
                    user_data[group][uid] = d
        print(f"\nLoaded users per group ({time.time() - t0:.0f}s):")
        for group in user_splits:
            print(f"  {group}: {len(user_data[group])}/{len(user_splits[group])} valid")
        print(f"  Total valid: {sum(len(user_data[g]) for g in user_splits)}")
        print(f"  Failed: {len(failed_users)}")

        if args.n_users is not None:
            user_data = _proportional_user_sample(user_data, user_splits, test_groups, args.n_users)

        with open(sample_path, "w") as f:
            json.dump({g: list(user_data[g].keys()) for g in user_data}, f, indent=2)
        print(f"  Saved user sample to {sample_path}")

    valid_users = [uid for g in TRAIN_GROUPS for uid in user_data.get(g, {})]
    print(f"\nTrain-group valid users: {len(valid_users)}")
    return user_data, test_groups, valid_users


def build_user_assignments(user_data, test_groups):
    # Train: train-group users' "train" split
    train_user_items = [(g, uid) for g in TRAIN_GROUPS for uid in user_data.get(g, {})]
    # Val: train-group users' "test" split + test-group users' both splits
    val_user_splits = [(g, uid, "test") for g in TRAIN_GROUPS for uid in user_data.get(g, {})]
    for g in test_groups:
        for uid in user_data.get(g, {}):
            val_user_splits.append((g, uid, "train"))
            val_user_splits.append((g, uid, "test"))

    print(f"Train users: {len(train_user_items)} (from {TRAIN_GROUPS})")
    print(f"Val user/split combos: {len(val_user_splits)} (from {TRAIN_GROUPS} test + {test_groups} both)")
    return train_user_items, val_user_splits


def build_loaders(args, graph, user_data, train_user_items, val_user_splits):
    train_loader = ChunkedGNNTrainLoader(
        train_user_items, user_data, graph,
        batch_size=args.batch_size, chunk_size=args.train_chunk_size,
        target_pos_rate=args.target_pos_rate,
    )
    print(f"\nTrain loader: {len(train_user_items)} users, "
          f"chunk_size={args.train_chunk_size}, batch_size={args.batch_size}")

    print("Computing class weights from train labels...")
    label_counts = train_loader.get_label_counts()
    total_train = sum(label_counts.values())
    classes = sorted(label_counts.keys())
    class_weights = torch.tensor(
        [total_train / (len(classes) * label_counts[c]) for c in classes], dtype=torch.float
    )
    print(f"  Label counts: {label_counts}")
    print(f"  Positive rate: {label_counts.get(1, 0) / max(total_train, 1):.4f} "
          f"({label_counts.get(1, 0)}/{total_train})")
    print(f"  Class weights: {class_weights.tolist()}")

    val_cache_path = (
        os.path.join(args.experiment_dir, f"val_samples_cache_{args.max_val_samples}")
        if args.max_val_samples
        else os.path.join(args.experiment_dir, "val_samples_cache_all")
    )
    val_loader = CachedValLoader(
        val_user_splits, user_data, graph,
        batch_size=args.batch_size, max_samples=args.max_val_samples, cache_path=val_cache_path,
    )
    lc = val_loader.get_label_counts()
    print(f"\nVal loader: {val_loader.total_samples} cached samples, {len(val_loader)} batches")
    print(f"  Positive rate: {lc.get(1, 0) / max(sum(lc.values()), 1):.4f} ({lc.get(1, 0)}/{sum(lc.values())})")

    train_eval_loader = SampledTrainEvalLoader(
        train_user_items, user_data, graph,
        batch_size=args.batch_size, max_samples=args.max_train_eval_samples,
        metadata_path=os.path.join(args.experiment_dir, f"train_eval_metadata_{args.max_train_eval_samples}.json"),
    )
    lc = train_eval_loader.get_label_counts()
    print(f"\nTrain-eval loader: {train_eval_loader.total_samples} samples, {len(train_eval_loader)} batches")
    print(f"  Positive rate: {lc.get(1, 0) / max(sum(lc.values()), 1):.4f} ({lc.get(1, 0)}/{sum(lc.values())})")

    return train_loader, class_weights, val_loader, train_eval_loader


# =============================================================================
# Model / hparams
# =============================================================================
def build_model(args, device):
    model = RetweetGNN(
        ff_hidden_dim=FF_HIDDEN_DIM,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        transformer_dim=TRANSFORMER_DIM,
        transformer_heads=TRANSFORMER_HEADS,
        embeddings_path=args.embeddings_path,
        device=device,
        dropout=args.dropout,
        drop_edge_rate=args.drop_edge_rate,
    ).to(device)

    # Only apply init weights on a fresh start (never when resuming this experiment).
    existing = os.path.join(args.experiment_dir, "best_retweet_gnn_general.pt")
    is_fresh_start = args.reset_gnn_training or not os.path.exists(existing)
    apply_init = bool(args.init_weights_path) and is_fresh_start

    if not is_fresh_start and args.init_weights_path:
        print(f"Existing checkpoint found at {existing} — skipping --init-weights-path "
              f"(use --reset-gnn-training to force re-initialization).")

    if apply_init:
        print(f"Loading pre-trained weights from: {args.init_weights_path}")
        state_dict = torch.load(args.init_weights_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (will use random init): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        print(f"  Weights loaded. gate_param = {model.gate_param.item():.4f}")
    else:
        with torch.no_grad():
            model.gate_param.fill_(0.0)  # sigmoid=0.5: balanced start
    return model, apply_init


def resolve_training_hparams(args, apply_init):
    """Return (lr, weight_decay, warmup_epochs, can_resume); persist config on fresh runs."""
    config_path = os.path.join(args.experiment_dir, "training_config.json")
    checkpoint_exists = os.path.exists(os.path.join(args.experiment_dir, "training_checkpoint.pt"))
    can_resume = not args.reset_gnn_training and checkpoint_exists

    if can_resume and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        lr, wd, warmup = cfg["lr"], cfg["weight_decay"], cfg["warmup_epochs"]
        print(f"Resuming with saved training config from {config_path}:")
        print(f"  lr={lr:.1e}, weight_decay={wd:.1e}, warmup_epochs={warmup}")
        if cfg.get("is_finetuning"):
            print(f"  (originally fine-tuned from {cfg.get('init_weights_path', '?')})")
        return lr, wd, warmup, can_resume

    if args.tuned_lr is not None:
        lr, wd, warmup = args.tuned_lr, args.tuned_wd, args.finetune_warmup_epochs
        print(f"Using tuned hparams: lr={lr:.2e}, wd={wd:.2e}, warmup={warmup}")
    else:
        lr = BASE_LR * args.finetune_lr_factor if apply_init else BASE_LR
        wd = BASE_WD * args.finetune_wd_factor if apply_init else BASE_WD
        warmup = args.finetune_warmup_epochs if apply_init else 5
        if apply_init:
            print(f"Fine-tuning mode: lr {BASE_LR:.1e} -> {lr:.1e}, wd {BASE_WD:.1e} -> {wd:.1e}, "
                  f"warmup 5 -> {warmup}")

    with open(config_path, "w") as f:
        json.dump({
            "lr": lr,
            "weight_decay": wd,
            "warmup_epochs": warmup,
            "is_finetuning": bool(apply_init),
            "init_weights_path": args.init_weights_path,
        }, f, indent=2)
    print(f"Training config saved to {config_path}")
    return lr, wd, warmup, can_resume


# =============================================================================
# MLflow (optional)
# =============================================================================
class _NoMLflow:
    """No-op stand-in so the training code doesn't branch on --no-mlflow."""

    def log_params(self, *a, **k): pass
    def log_metrics(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass
    def end_run(self, *a, **k): pass


def start_mlflow(args, n_train_users, lr, wd, warmup):
    if not args.mlflow:
        print("MLflow disabled.")
        return _NoMLflow(), None

    import mlflow
    from mlflow.tracking import MlflowClient

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    exp_name = args.mlflow_experiment_name or f"{args.project_home}/{args.final_tag}/"
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id if exp is not None else client.create_experiment(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)

    run_name = args.final_tag
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        run_name = f"{run_name}_slurm{slurm_job}"
    run = mlflow.start_run(run_name=run_name)

    mlflow.log_params({
        "experiment_tag": args.experiment_tag,
        "final_tag": args.final_tag,
        "n_users": args.n_users if args.n_users else "all",
        "n_train_users": n_train_users,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_chunk_size": args.train_chunk_size,
        "lr": lr,
        "weight_decay": wd,
        "warmup_epochs": warmup,
        "patience": args.patience if args.patience else "disabled",
        "gradient_accumulation_steps": args.gradient_accumulation_steps or 1,
        "mixed_precision": args.mixed_precision,
        "log_every_n_steps": args.log_every_n_steps,
        "train_f1_every_n_epochs": args.train_f1_every_n_epochs or "disabled",
        "max_val_samples": args.max_val_samples or "all",
        "max_train_eval_samples": args.max_train_eval_samples or "all",
        "ff_hidden_dim": FF_HIDDEN_DIM,
        "gcn_hidden_dim": GCN_HIDDEN_DIM,
        "transformer_dim": TRANSFORMER_DIM,
        "transformer_heads": TRANSFORMER_HEADS,
        "dropout": args.dropout,
        "drop_edge_rate": args.drop_edge_rate,
        "init_weights_path": args.init_weights_path or "none",
        "reset_gnn_training": args.reset_gnn_training,
        "slurm_job_id": slurm_job or "none",
    })
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment:   {exp_name}")
    print(f"MLflow run ID:       {run.info.run_id}")
    return mlflow, run


def make_checkpoint_callback(mlflow_mod):
    def _on_checkpoint(history, global_step, is_epoch_end):
        try:
            metrics = {}
            if not is_epoch_end and history["step"]:
                i = len(history["step"]) - 1
                metrics["checkpoint/train_loss"] = history["train_loss"][i]
                metrics["checkpoint/val_loss"] = history["val_loss"][i]
                metrics["checkpoint/val_f1"] = history["val_f1"][i]
            if is_epoch_end and history["epoch_step"]:
                i = len(history["epoch_step"]) - 1
                metrics["epoch/val_loss"] = history["epoch_val_loss"][i]
                metrics["epoch/val_f1"] = history["epoch_val_f1"][i]
                tf1 = history.get("epoch_train_f1", [])
                if i < len(tf1) and tf1[i] is not None:
                    metrics["epoch/train_f1"] = tf1[i]
            if metrics:
                mlflow_mod.log_metrics(metrics, step=global_step)
        except Exception as e:
            print(f"Warning: MLflow callback error: {e}")

    return _on_checkpoint


# =============================================================================
# Evaluation / reporting
# =============================================================================
def _evaluate_loader(model, loader, criterion, device, name):
    preds, labels, loss, n = [], [], 0.0, 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss += criterion(out, batch.y).item()
            n += 1
            preds.extend(out.argmax(dim=1).cpu().tolist())
            labels.extend(batch.y.cpu().tolist())
    el = time.time() - t0
    print(f"  {name:<5} — Loss: {loss / max(n, 1):.4f} | "
          f"Acc: {accuracy_score(labels, preds):.4f} | "
          f"F1: {f1_score(labels, preds):.4f} | "
          f"P: {precision_score(labels, preds, zero_division=0):.4f} | "
          f"R: {recall_score(labels, preds, zero_division=0):.4f} | "
          f"samples: {len(labels)} | took {int(el) // 60}m {int(el) % 60:02d}s")


def initial_evaluation(model, val_loader, train_eval_loader, class_weights, device):
    print("\n" + "=" * 60)
    print("INITIAL EVALUATION (pre-trained weights, before training)")
    print("=" * 60)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    _evaluate_loader(model, val_loader, criterion, device, "Val")
    _evaluate_loader(model, train_eval_loader, criterion, device, "Train")
    print("=" * 60 + "\n")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_resume_progress(args):
    import pickle

    print("\n" + "=" * 60)
    print("RESUMING FROM CHECKPOINT — progress so far")
    print("=" * 60)

    ckpt = torch.load(os.path.join(args.experiment_dir, "training_checkpoint.pt"),
                      map_location="cpu", weights_only=False)
    print(f"  Checkpoint epoch:  {ckpt.get('epoch', '?')}")
    print(f"  Global step:       {ckpt.get('global_step', '?')}")
    if ckpt.get("best_val_f1") is not None:
        print(f"  Best val F1:       {ckpt['best_val_f1']:.4f}")
    if ckpt.get("steps_since_improvement") is not None:
        print(f"  No improvement:    {ckpt['steps_since_improvement']}"
              + (f"/{args.patience}" if args.patience else ""))
    del ckpt

    hist_path = os.path.join(args.experiment_dir, "training_history.pkl")
    if os.path.exists(hist_path):
        with open(hist_path, "rb") as f:
            h = pickle.load(f)
        if h.get("step"):
            n = len(h["step"])
            show = min(10, n)
            best = max(h["val_f1"][: n - show], default=0.0)
            print(f"\n  Checkpoint-level history ({n} checkpoints):")
            if n > show:
                print(f"    ... ({n - show} earlier checkpoints omitted)")
            for i in range(n - show, n):
                best = max(best, h["val_f1"][i])
                vp = h["val_precision"][i] if i < len(h.get("val_precision", [])) else None
                vr = h["val_recall"][i] if i < len(h.get("val_recall", [])) else None
                pr = f" (P={vp:.4f} R={vr:.4f})" if vp is not None else ""
                print(f"    Step {h['step'][i]:>6d} | Loss: {h['train_loss'][i]:.4f} | "
                      f"Val Loss: {h['val_loss'][i]:.4f} | Val F1: {h['val_f1'][i]:.4f}{pr} | Best: {best:.4f}")
        if h.get("epoch_step"):
            best = 0.0
            print(f"\n  Epoch-level history ({len(h['epoch_step'])} epochs completed):")
            for i in range(len(h["epoch_step"])):
                best = max(best, h["epoch_val_f1"][i])
                line = f"    Epoch {i + 1:>3d} (step {h['epoch_step'][i]:>6d}) | "
                if i < len(h.get("epoch_train_loss", [])):
                    line += f"Train Loss: {h['epoch_train_loss'][i]:.4f} | "
                if i < len(h.get("epoch_train_f1", [])) and h["epoch_train_f1"][i] is not None:
                    line += f"Train F1: {h['epoch_train_f1'][i]:.4f} | "
                line += (f"Val F1: {h['epoch_val_f1'][i]:.4f} | "
                         f"Val Loss: {h['epoch_val_loss'][i]:.4f} | Best: {best:.4f}")
                print(line)
    print("=" * 60 + "\n")


def finalize(args, history, mlflow_mod):
    if history:
        summary = {}
        if history.get("val_f1"):
            summary["best_checkpoint_val_f1"] = max(history["val_f1"])
        if history.get("epoch_val_f1"):
            summary["best_epoch_val_f1"] = max(history["epoch_val_f1"])
        tf1 = [v for v in history.get("epoch_train_f1", []) if v is not None]
        if tf1:
            summary["best_epoch_train_f1"] = max(tf1)
        if history.get("train_loss"):
            summary["final_train_loss"] = history["train_loss"][-1]
        if history.get("val_loss"):
            summary["final_val_loss"] = history["val_loss"][-1]
        if summary:
            mlflow_mod.log_metrics(summary)
            for k, v in summary.items():
                print(f"  {k}: {v:.4f}")

    best_model_path = os.path.join(args.experiment_dir, "best_retweet_gnn_general.pt")
    if os.path.exists(best_model_path):
        mlflow_mod.log_artifact(best_model_path, artifact_path="model")
        print(f"Best model: {best_model_path}")

    history_pkl = os.path.join(args.experiment_dir, "training_history.pkl")
    if os.path.exists(history_pkl):
        mlflow_mod.log_artifact(history_pkl, artifact_path="history")

    if history and (history.get("step") or history.get("epoch_step")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        if history.get("step"):
            ax.plot(history["step"], history["train_loss"], "b-", alpha=0.6, label="Train loss (checkpoint)")
            ax.plot(history["step"], history["val_loss"], "r-", alpha=0.6, label="Val loss (checkpoint)")
        if history.get("epoch_step"):
            ax.plot(history["epoch_step"], history["epoch_val_loss"], "ro-", markersize=5,
                    label="Val loss (end-of-epoch)")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if history.get("step"):
            ax.plot(history["step"], history["val_f1"], "r-", alpha=0.6, label="Val F1 (checkpoint)")
        if history.get("epoch_step"):
            tf1 = history.get("epoch_train_f1", [])
            if tf1 and len(tf1) == len(history["epoch_step"]):
                ax.plot(history["epoch_step"], tf1, "b^-", markersize=5, label="Train F1 (end-of-epoch)")
            ax.plot(history["epoch_step"], history["epoch_val_f1"], "ro-", markersize=5,
                    label="Val F1 (end-of-epoch)")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("F1 Score")
        ax.set_title("Training & Validation F1")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"GNN Training Curves — {args.final_tag}", fontsize=13)
        plt.tight_layout()
        curves_path = os.path.join(args.experiment_dir, "training_curves.png")
        fig.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow_mod.log_artifact(curves_path, artifact_path="plots")
        print(f"Training curves: {curves_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    setup_logging()

    # Make project modules importable regardless of the cwd SLURM starts us in.
    for p in (SCRIPT_DIR, args.project_home):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(SCRIPT_DIR)

    global RetweetGNN, train_model, create_gnn_train_val_samples, load_dataframe_raw
    from gnn_models import RetweetGNN, train_model
    from utils import create_gnn_train_val_samples, load_dataframe_raw

    if args.graph_path is None:
        from tw_dataset.settings import IG_GRAPH_PATH
        args.graph_path = IG_GRAPH_PATH

    print(f"Script dir:   {SCRIPT_DIR}")
    print(f"Project home: {args.project_home}")
    print(f"Experiment:   {args.final_tag}")
    print(f"Output dir:   {args.experiment_dir}")
    if os.environ.get("SLURM_JOB_ID"):
        print(f"SLURM job:    {os.environ['SLURM_JOB_ID']} on {os.environ.get('SLURMD_NODENAME', '?')}")

    os.makedirs(args.experiment_dir, exist_ok=True)
    if args.reset_gnn_training:
        reset_training_artifacts(args.experiment_dir)
    else:
        print("RESET_GNN_TRAINING=False — resuming from existing checkpoint if available.")

    apply_hparam_overrides(args)
    seed_everything(args.seed)
    print(f"Init weights: {args.init_weights_path or 'none (training from scratch)'}")

    device = select_device(args.device)
    print(f"Device: {device}")

    t0 = time.time()
    graph = nx.read_graphml(args.graph_path)
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
          f"({time.time() - t0:.0f}s)")

    user_data, test_groups, valid_users = load_users(args)
    train_user_items, val_user_splits = build_user_assignments(user_data, test_groups)
    train_loader, class_weights, val_loader, train_eval_loader = build_loaders(
        args, graph, user_data, train_user_items, val_user_splits
    )

    model, apply_init = build_model(args, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lr, wd, warmup, can_resume = resolve_training_hparams(args, apply_init)
    mlflow_mod, run = start_mlflow(args, len(valid_users), lr, wd, warmup)

    history = None
    status = "FAILED"
    try:
        with tee_to_log(os.path.join(args.experiment_dir, "training.log")):
            if apply_init:
                initial_evaluation(model, val_loader, train_eval_loader, class_weights, device)
            if can_resume:
                print_resume_progress(args)

            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                experiment_dir=args.experiment_dir,
                class_weights=class_weights,
                epochs=args.epochs,
                device=device,
                lr=lr,
                log_every_n_steps=args.log_every_n_steps,
                patience=args.patience,
                lr_warmup_epochs=warmup,
                weight_decay=wd,
                resume=can_resume,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                mixed_precision=args.mixed_precision,
                train_f1_every_n_epochs=args.train_f1_every_n_epochs,
                train_eval_loader=train_eval_loader,
                on_checkpoint=make_checkpoint_callback(mlflow_mod),
            )
        finalize(args, history, mlflow_mod)
        status = "FINISHED"
    except KeyboardInterrupt:
        # SLURM sends SIGTERM->SIGKILL on timeout; Ctrl-C/SIGINT lands here.
        status = "KILLED"
        raise
    finally:
        if run is not None:
            mlflow_mod.end_run(status=status)
            print(f"MLflow run {run.info.run_id} ended with status {status}")

    print(f"\nTraining run completed. All results in: {args.experiment_dir}")


if __name__ == "__main__":
    main()
