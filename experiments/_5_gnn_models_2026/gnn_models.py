"""Shared GNN model definitions for the SNA classifier experiments.

Contains:
- PretrainedEmbeddingLookup: frozen embedding layer
- RetweetDataset: PyG dataset wrapping raw sample dicts
- RetweetGNN: full model architecture (MagNet + Transformer + shortcut gate)
- soft_f1_loss, combined_loss: training objectives
- evaluate: inference + F1 computation
- train_model: unified training loop with configurable early stopping and LR warmup
"""
import os
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric_signed_directed.nn.directed import MagNetConv


# ---------------------------------------------------------------------------
# Pretrained Embedding Lookup
# ---------------------------------------------------------------------------
class PretrainedEmbeddingLookup(nn.Module):
    """Maps global user IDs to their pretrained MagNet embeddings. Frozen."""
    def __init__(self, embeddings_path: str, device: str):
        super().__init__()
        pretrained = torch.load(embeddings_path, weights_only=True, map_location=device)
        self.register_buffer("embeddings", pretrained)
        self.embedding_dim = pretrained.shape[1]

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[user_ids]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RetweetDataset(Dataset):
    def __init__(self, raw_samples: list):
        super().__init__()
        self.samples = raw_samples

    def len(self):
        return len(self.samples)

    def get(self, idx):
        s = self.samples[idx]
        all_ids = [s["central_user_id"]] + list(s["neighbor_ids"])
        num_nodes = len(all_ids)
        user_ids = torch.tensor(all_ids, dtype=torch.long)
        retweeted_set = set(s["retweeted_ids"])
        retweet_flag = torch.tensor(
            [1.0 if uid in retweeted_set else 0.0 for uid in all_ids],
            dtype=torch.float
        ).unsqueeze(1)
        if len(s["edge_index"]) > 0:
            edge_index = torch.tensor(s["edge_index"], dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        label = torch.tensor(s["label"], dtype=torch.long)
        return Data(
            user_ids=user_ids,
            retweet_flag=retweet_flag,
            edge_index=edge_index,
            y=label,
            num_nodes=num_nodes,
            central_mask=torch.zeros(num_nodes, dtype=torch.bool).index_fill_(0, torch.tensor([0]), True)
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RetweetGNN(nn.Module):
    def __init__(self, embeddings_path, device, ff_hidden_dim=256, gcn_hidden_dim=128,
                 transformer_dim=128, transformer_heads=4, num_classes=2, dropout=0.3,
                 q=0.25, K=1, drop_edge_rate=0.2):
        super().__init__()
        self.drop_edge_rate = drop_edge_rate
        self.flag_scale = nn.Parameter(torch.tensor(10.0))
        self.lookup = PretrainedEmbeddingLookup(embeddings_path, device)
        embed_dim = self.lookup.embedding_dim
        ff_input_dim = embed_dim + 1

        self.ff = nn.Sequential(
            nn.Linear(ff_input_dim, ff_hidden_dim),
            nn.LayerNorm(ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, gcn_hidden_dim),
            nn.LayerNorm(gcn_hidden_dim),
            nn.GELU(),
        )
        self.magnet1 = MagNetConv(gcn_hidden_dim, gcn_hidden_dim, q=q, K=K, trainable_q=True)
        self.transformer = TransformerConv(
            in_channels=gcn_hidden_dim * 2,
            out_channels=transformer_dim // transformer_heads,
            heads=transformer_heads,
            edge_dim=1, dropout=dropout, concat=True,
        )
        self.post_transformer_norm = nn.LayerNorm(transformer_dim)
        self.gate_param = nn.Parameter(torch.tensor(0.0))  # start at 50/50 so GNN branch is used
        self.shortcut_head = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, num_classes),
        )
        self.gnn_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def _drop_edges(self, edge_index, edge_attr=None):
        """Randomly drop edges during training (DropEdge regularization)."""
        if not self.training or self.drop_edge_rate <= 0:
            return edge_index, edge_attr
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges, device=edge_index.device) > self.drop_edge_rate
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
        return edge_index, edge_attr

    def forward(self, data):
        user_ids = data.user_ids
        retweet_flag = data.retweet_flag
        edge_index = data.edge_index
        batch = data.batch
        central_mask = data.central_mask

        # DropEdge: randomly remove edges during training
        edge_index, _ = self._drop_edges(edge_index)

        with torch.no_grad():
            pretrained = self.lookup(user_ids)
        x = torch.cat([pretrained, self.flag_scale * retweet_flag], dim=-1)
        x = self.ff(x)

        x_real, x_imag = x, torch.zeros_like(x)
        x_real, x_imag = self.magnet1(x_real, x_imag, edge_index)
        x_real, x_imag = F.gelu(x_real), F.gelu(x_imag)
        x_real, x_imag = self.dropout(x_real), self.dropout(x_imag)

        x = torch.cat([x_real, x_imag], dim=-1)
        edge_attr = retweet_flag[edge_index[1]]  # recompute after DropEdge
        edge_index, edge_attr = self._drop_edges(edge_index, edge_attr)
        x = self.transformer(x, edge_index, edge_attr=edge_attr)
        x = F.gelu(x)
        x = self.post_transformer_norm(x)
        central_x = x[central_mask]

        num_graphs = data.batch.max().item() + 1
        non_central = ~central_mask
        nc_flags = retweet_flag[non_central].squeeze()
        nc_batch = batch[non_central]
        rt_sum = torch.zeros(num_graphs, device=x.device).scatter_add_(0, nc_batch, nc_flags)
        node_counts = torch.zeros(num_graphs, device=x.device).scatter_add_(0, nc_batch, torch.ones_like(nc_flags))
        rt_frac = (rt_sum / node_counts.clamp(min=1)).unsqueeze(-1)
        node_counts = (node_counts / 50.0).unsqueeze(-1)
        shortcuts = torch.cat([rt_frac, node_counts], dim=-1)

        gate = torch.sigmoid(self.gate_param)
        shortcut_logits = self.shortcut_head(shortcuts)
        gnn_logits = self.gnn_head(central_x)
        logits = (1 - gate) * shortcut_logits + gate * gnn_logits
        return logits


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def soft_f1_loss(logits, labels, eps=1e-8):
    probs = F.softmax(logits, dim=-1)[:, 1]
    tp = (probs * labels).sum()
    fp = (probs * (1 - labels)).sum()
    fn = ((1 - probs) * labels).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return 1 - f1


def combined_loss(logits, labels, class_weights, epoch, warmup_epochs=10):
    ce = F.cross_entropy(logits, labels.long(), weight=class_weights.to(logits.device))
    if epoch <= warmup_epochs:
        return ce
    sf1 = soft_f1_loss(logits, labels)
    return 0.5 * ce + 0.5 * sf1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, class_weights=None, epoch=None):
    """Evaluate model on loader. Returns (f1, preds, labels, avg_loss).
    If class_weights and epoch are provided, also computes average loss (single pass).
    Otherwise avg_loss is None.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    compute_loss = class_weights is not None and epoch is not None
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.cpu())
        if compute_loss:
            total_loss += combined_loss(logits, batch.y.float(), class_weights, epoch).item()
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader) if compute_loss else None
    return f1, all_preds, all_labels, avg_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, raw_train_samples, raw_val_samples, experiment_dir,
                epochs=50, batch_size=128, lr=1e-2, device="cuda",
                log_every_n_steps=100, patience=15, lr_warmup_epochs=0,
                weight_decay=1e-3, resume=False, gradient_accumulation_steps=None,
                mixed_precision=False):
    """Unified training loop with checkpointing and configurable settings.

    Args:
        model: RetweetGNN instance (already on device).
        raw_train_samples: list of sample dicts for training.
        raw_val_samples: list of sample dicts for validation.
        experiment_dir: path to experiment output directory.
        epochs: number of training epochs.
        batch_size: batch size for DataLoader.
        lr: peak learning rate (reached after warmup).
        device: torch device string.
        log_every_n_steps: how often to log checkpoint metrics.
        patience: early stopping — stop after this many checkpoints without
            val F1 improvement. None disables early stopping.
        lr_warmup_epochs: number of epochs for linear LR warmup (0.1x → 1x).
            0 means no warmup (plain cosine decay).
        weight_decay: L2 regularization strength for AdamW.
        resume: whether to resume from a previous checkpoint.
        gradient_accumulation_steps: if not None and > 1, accumulate gradients
            over this many micro-batches before each optimizer step. Effective
            batch size = batch_size * gradient_accumulation_steps. None or 1
            disables accumulation (default behavior).
        mixed_precision: if True, use automatic mixed precision (float16) with
            GradScaler for faster training on CUDA. Requires CUDA device.

    Returns:
        (model, history) — model loaded with best weights, and full training history dict.
    """
    import pickle
    import time
    from tqdm.auto import tqdm

    checkpoint_path = f"{experiment_dir}/training_checkpoint.pt"
    best_model_path = f"{experiment_dir}/best_retweet_gnn_general.pt"
    history_path = f"{experiment_dir}/training_history.pkl"

    y = np.array([s["label"] for s in raw_train_samples])
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    train_ds = RetweetDataset(raw_train_samples)
    val_ds = RetweetDataset(raw_val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # LR schedule: optional linear warmup → cosine decay
    if lr_warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=lr_warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - lr_warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[lr_warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    print(f"Training on {device} | {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps} | "
          f"Logging every {log_every_n_steps} steps")
    print(f"LR warmup: {lr_warmup_epochs} epochs | weight_decay: {weight_decay} | "
          f"patience: {patience}")

    _use_grad_accum = (gradient_accumulation_steps is not None and gradient_accumulation_steps > 1)
    if _use_grad_accum:
        print(f"Gradient accumulation: {gradient_accumulation_steps} steps | "
              f"Effective batch size: {batch_size * gradient_accumulation_steps}")

    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    _autocast_device = str(device).split(":")[0]  # "cuda:0" -> "cuda", handles torch.device too
    if mixed_precision:
        print(f"Mixed precision: enabled (float16 autocast + GradScaler)")

    best_val_f1 = 0
    global_step = 0
    running_loss = 0.0
    running_steps = 0
    start_epoch = 1
    steps_since_improvement = 0

    history = {
        "step": [], "train_loss": [], "val_loss": [], "val_f1": [],
        "epoch_step": [], "epoch_train_f1": [], "epoch_val_f1": [], "epoch_val_loss": [],
    }

    # Resume from checkpoint if available
    if resume and os.path.exists(checkpoint_path):
        print(f"  Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_f1 = ckpt["best_val_f1"]
        steps_since_improvement = ckpt.get("steps_since_improvement", 0)
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history = pickle.load(f)
            print(f"  Restored training history "
                  f"({len(history['step'])} checkpoints, {len(history['epoch_step'])} epochs)")
        print(f"  Resumed at epoch {start_epoch}, global_step {global_step}, "
              f"best_val_f1 {best_val_f1:.4f}")
    elif resume:
        print(f"  No checkpoint found at {checkpoint_path}, starting from scratch.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        pbar = tqdm(enumerate(train_loader, 1), total=steps_per_epoch,
                    desc=f"Epoch {epoch}/{epochs}", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}, {rate_fmt}]")

        if _use_grad_accum:
            optimizer.zero_grad()

        for step_in_epoch, batch in pbar:
            batch = batch.to(device)
            with torch.autocast(device_type=_autocast_device, dtype=torch.float16, enabled=mixed_precision):
                logits = model(batch)
                loss = combined_loss(logits, batch.y.float(), class_weights, epoch)
            if _use_grad_accum:
                scaler.scale(loss / gradient_accumulation_steps).backward()
            else:
                optimizer.zero_grad()
                scaler.scale(loss).backward()

            if not _use_grad_accum or step_in_epoch % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            running_loss += loss.item()
            running_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            # Periodic checkpoint
            if global_step % log_every_n_steps == 0:
                pbar.refresh()
                avg_loss = running_loss / running_steps
                print(f"\n  --- Checkpoint at step {global_step} "
                      f"(epoch {epoch}/{epochs}, step {step_in_epoch}/{steps_per_epoch}) ---")
                print(f"    Computing val F1 + val loss (single pass)...")
                t_val = time.time()
                val_f1, _, _, val_loss = evaluate(model, val_loader, device,
                                                  class_weights=class_weights, epoch=epoch)
                print(f"    Val eval took {time.time() - t_val:.1f}s")
                gate_val = torch.sigmoid(model.gate_param).item()

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    steps_since_improvement = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    steps_since_improvement += 1

                print(f"    Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val F1: {val_f1:.4f} | Best: {best_val_f1:.4f} | "
                      f"Gate: {gate_val:.4f} | "
                      f"No improvement: {steps_since_improvement}/{patience}")

                history["step"].append(global_step)
                history["train_loss"].append(avg_loss)
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)

                # Early stopping
                if patience and steps_since_improvement >= patience:
                    print(f"\n  Early stopping: no val F1 improvement for {patience} "
                          f"consecutive checkpoints. Best val F1: {best_val_f1:.4f}")
                    pbar.close()
                    with open(history_path, "wb") as f:
                        pickle.dump(history, f)
                    _save_checkpoint(checkpoint_path, epoch, global_step, model,
                                    optimizer, scheduler, best_val_f1,
                                    steps_since_improvement)
                    model.load_state_dict(torch.load(best_model_path, weights_only=True))
                    return model, history

                running_loss = 0.0
                running_steps = 0
                model.train()

        pbar.close()

        # Flush any remaining accumulated gradients at epoch boundary
        if _use_grad_accum and step_in_epoch % gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        # End-of-epoch evaluation
        print(f"\n  === End of epoch {epoch}/{epochs} ===")
        print(f"    Computing train F1...")
        t_train_f1 = time.time()
        train_f1, _, _, _ = evaluate(model, train_loader, device)
        print(f"    Train F1 took {time.time() - t_train_f1:.1f}s")

        print(f"    Computing val F1 + val loss (single pass)...")
        t_val = time.time()
        val_f1, _, _, val_loss = evaluate(model, val_loader, device,
                                          class_weights=class_weights, epoch=epoch)
        print(f"    Val eval took {time.time() - t_val:.1f}s")

        gate_val = torch.sigmoid(model.gate_param).item()
        print(f"    Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Best: {best_val_f1:.4f} | Gate: {gate_val:.4f}")

        history["epoch_step"].append(global_step)
        history["epoch_train_f1"].append(train_f1)
        history["epoch_val_f1"].append(val_f1)
        history["epoch_val_loss"].append(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            steps_since_improvement = 0
            torch.save(model.state_dict(), best_model_path)

        # Save checkpoint
        _save_checkpoint(checkpoint_path, epoch, global_step, model,
                        optimizer, scheduler, best_val_f1, steps_since_improvement)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)
        print(f"    Checkpoint saved (epoch {epoch}, step {global_step}, "
              f"best_val_f1 {best_val_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f} | Total steps: {global_step}")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, history


def _save_checkpoint(path, epoch, global_step, model, optimizer, scheduler,
                    best_val_f1, steps_since_improvement):
    """Helper to save a training checkpoint."""
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "steps_since_improvement": steps_since_improvement,
    }, path)
