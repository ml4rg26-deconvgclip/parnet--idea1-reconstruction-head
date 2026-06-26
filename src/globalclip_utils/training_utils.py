"""Training utilities for GlobalCLIP models.

Loss strategy
─────────────
Target: log-fold-enrichment  t = log(1 + signal) − log(1 + control)

  L_pearson   1 − mean Pearson r between pred and t         (shape match)
  L_nll       Multinomial NLL of pred on raw signal counts  (count fidelity)
  L_alpha     Mean mixing coefficient                        (sparsity)
  L_phase     Mean squared phase  (QLayer only)             (regularisation)

  L_total = L_pearson + λ_nll · L_nll + λ_alpha · L_alpha [+ λ_phase · L_phase]

The Pearson loss is the dominant term — it forces the model to learn the
*shape* of the enrichment profile.  L_nll adds an inductive bias that the
predictions should be proportional to read counts (multinomial distribution).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


# ── Loss helpers ──────────────────────────────────────────────────────────────


def compute_log_enrichment(
    signal: torch.Tensor, control: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """Log fold-enrichment: log(eps + signal) − log(eps + control).

    With eps=1 this is log1p(signal) − log1p(control).
    Positive values indicate enrichment over background.
    """
    return torch.log(eps + signal) - torch.log(eps + control)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 − mean Pearson r over the batch.

    Args:
        pred:   (B, 1, L) model output.
        target: (B, 1, L) target (log-enrichment).
    """
    p = pred.squeeze(1)      # (B, L)
    t = target.squeeze(1)    # (B, L)
    p = p - p.mean(-1, keepdim=True)
    t = t - t.mean(-1, keepdim=True)
    r = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + eps)
    return (1.0 - r).mean()


def multinomial_nll_loss(pred: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Multinomial NLL: treats position distribution as a categorical.

    Args:
        pred:   (B, 1, L) unnormalized logits.
        counts: (B, 1, L) raw read counts.
    """
    log_probs = F.log_softmax(pred.squeeze(1), dim=-1)   # (B, L)
    c = counts.squeeze(1)                                 # (B, L)
    total = c.sum(-1).clamp(min=1)
    return -(c * log_probs).sum(-1).div(total).mean()


# ── Lightning module ──────────────────────────────────────────────────────────


class GlobalCLIPLightningModule(pl.LightningModule):
    """Unified LightningModule for both GlobalCLIP model variants.

    Works with GlobalCLIPStandardModel and GlobalCLIPQLayerModel.
    Both models return (pred, alpha) from forward().

    Batch format expected from GlobalCLIPDataset:
        batch["sequence"]  (B, 4, L)  one-hot encoded sequence
        batch["signal"]    (B, 1, L)  GlobalCLIP read counts
        batch["control"]   (B, 1, L)  control read counts

    Args:
        model:          A GlobalCLIPStandardModel or GlobalCLIPQLayerModel.
        lr:             AdamW learning rate (default 1e-4).
        lambda_nll:     Weight on multinomial NLL term (default 0.3).
        lambda_alpha:   Weight on alpha sparsity penalty (default 5.0).
        lambda_phase:   Weight on QLayer phase L2 regularisation (default 0.01).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        lambda_nll: float = 0.3,
        lambda_alpha: float = 5.0,
        lambda_phase: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lambda_nll = lambda_nll
        self.lambda_alpha = lambda_alpha
        self.lambda_phase = lambda_phase
        self.save_hyperparameters(ignore=["model"])

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        seq     = batch["sequence"]   # (B, 4, L)
        signal  = batch["signal"]     # (B, 1, L)
        control = batch["control"]    # (B, 1, L)

        pred, alpha = self.model(seq)

        target = compute_log_enrichment(signal, control)       # (B, 1, L)

        loss_p = pearson_loss(pred, target)
        loss_n = multinomial_nll_loss(pred, signal)
        loss_a = alpha.mean()

        total = loss_p + self.lambda_nll * loss_n + self.lambda_alpha * loss_a

        if hasattr(self.model, "qlayer"):
            phase_reg = (self.model.qlayer.phase ** 2).mean()
            total = total + self.lambda_phase * phase_reg
            self.log(f"{prefix}/phase_reg", phase_reg)

        self.log(f"{prefix}/loss",       total,   prog_bar=True, on_epoch=True)
        self.log(f"{prefix}/pearson",    loss_p,  prog_bar=True, on_epoch=True)
        self.log(f"{prefix}/nll",        loss_n,  on_epoch=True)
        self.log(f"{prefix}/alpha_mean", loss_a,  on_epoch=True)
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if hasattr(self.model, "qlayer"):
            # Train phases ~10× faster so they converge alongside MLP weights
            param_groups = [
                {"params": self.model.mix_coeff.parameters(), "lr": self.lr},
                {"params": [self.model.log_scale],             "lr": self.lr},
                {"params": self.model.qlayer.parameters(),     "lr": self.lr * 10},
                {"params": self.model.cnn.parameters(),        "lr": self.lr},
            ]
            return torch.optim.AdamW(param_groups)
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


# ── Run-config persistence ────────────────────────────────────────────────────


def save_run_config(out_dir: Path, config: dict) -> None:
    """Save training hyperparameters to run_config.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))


def load_run_config(out_dir: Path) -> dict:
    """Load run_config.json written by save_run_config."""
    return json.loads((out_dir / "run_config.json").read_text())
