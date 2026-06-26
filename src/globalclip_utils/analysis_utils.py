"""Analysis and visualisation utilities for trained GlobalCLIP models.

Key analyses:
  - Protein ranking by mean mixing coefficient (alpha)
  - 223×223 alpha correlation matrix (which proteins co-activate?)
  - QLayer phase polar plot (cooperative vs. competitive clusters)
  - QLayer coupling matrix cos(φ_i − φ_j)
  - Per-sequence Pearson r evaluation on test set
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ── Collect mixing coefficients ───────────────────────────────────────────────


@torch.no_grad()
def collect_alpha(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> np.ndarray:
    """Collect mixing coefficient matrix (N, num_rbps) over the dataset.

    Args:
        model:       A GlobalCLIP model (returns (pred, alpha) from forward).
        dataloader:  DataLoader yielding batches with "sequence" key.
        device:      Device to run inference on.
        max_batches: Stop after this many batches (None = full dataset).

    Returns:
        alpha_matrix: float32 array (N, num_rbps).
    """
    model.eval()
    chunks = []
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        seq = batch["sequence"].to(device)
        _, alpha = model(seq)
        chunks.append(alpha.cpu().float().numpy())
    return np.concatenate(chunks, axis=0)


# ── Protein ranking ───────────────────────────────────────────────────────────


def rank_proteins(
    alpha_matrix: np.ndarray,
    rbp_names: list[str],
    log_scale: np.ndarray | None = None,
) -> pd.DataFrame:
    """Rank proteins by mean mixing coefficient.

    Args:
        alpha_matrix: (N, num_rbps) from collect_alpha().
        rbp_names:    List of protein names matching the 223 columns.
        log_scale:    Optional (num_rbps,) log_scale.exp() values to include.

    Returns:
        DataFrame sorted by mean_alpha descending with columns:
        rank, protein, mean_alpha, std_alpha[, log_scale].
    """
    mean_a = alpha_matrix.mean(0)
    std_a  = alpha_matrix.std(0)
    df = pd.DataFrame({"protein": rbp_names, "mean_alpha": mean_a, "std_alpha": std_a})
    if log_scale is not None:
        df["exp_log_scale"] = log_scale
        df["effective_weight"] = mean_a * log_scale
    df = df.sort_values("mean_alpha", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ── Correlation analysis ──────────────────────────────────────────────────────


def alpha_correlation_matrix(alpha_matrix: np.ndarray) -> np.ndarray:
    """Pearson r between all pairs of protein mixing coefficients.

    Returns (num_rbps, num_rbps) correlation matrix.
    Correlated proteins tend to co-activate on the same sequences.
    """
    return np.corrcoef(alpha_matrix.T)


# ── Test-set evaluation ───────────────────────────────────────────────────────


def _log_enrichment(signal: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """log(1 + signal) − log(1 + control) — local copy to avoid circular import."""
    return torch.log1p(signal) - torch.log1p(control)


@torch.no_grad()
def evaluate_pearson(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Compute Pearson r on log-enrichment for each test sequence.

    Returns:
        mean_r:  Mean Pearson r across all test sequences.
        all_r:   (N,) array of per-sequence Pearson r values.
    """
    model.eval()
    corrs = []
    for batch in dataloader:
        seq     = batch["sequence"].to(device)
        signal  = batch["signal"].to(device)
        control = batch["control"].to(device)

        pred, _ = model(seq)
        target = _log_enrichment(signal, control)

        p = pred.squeeze(1)
        t = target.squeeze(1)
        pz = p - p.mean(-1, keepdim=True)
        tz = t - t.mean(-1, keepdim=True)
        r = (pz * tz).sum(-1) / (pz.norm(dim=-1) * tz.norm(dim=-1) + 1e-8)
        corrs.extend(r.cpu().float().numpy().tolist())

    all_r = np.array(corrs)
    return float(np.mean(all_r)), all_r


# ── Visualisation helpers ─────────────────────────────────────────────────────


def plot_alpha_heatmap(
    corr_matrix: np.ndarray,
    rbp_names: list[str] | None = None,
    figsize: tuple[int, int] = (12, 10),
    title: str = "Protein mixing coefficient correlation (α_i vs α_j)",
) -> plt.Figure:
    """Plot 223×223 Pearson correlation heatmap of mixing coefficients.

    Clusters of correlated proteins tend to bind the same sequence contexts
    together (same RNP complex, same cell-type specificity, etc.).
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(title)
    if rbp_names and len(rbp_names) <= 40:
        ax.set_xticks(range(len(rbp_names)))
        ax.set_yticks(range(len(rbp_names)))
        ax.set_xticklabels(rbp_names, rotation=90, fontsize=5)
        ax.set_yticklabels(rbp_names, fontsize=5)
    plt.tight_layout()
    return fig


def plot_top_proteins(
    ranking_df: pd.DataFrame,
    top_n: int = 30,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Bar plot of top-N proteins by mean mixing coefficient."""
    top = ranking_df.head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["protein"][::-1], top["mean_alpha"][::-1],
            xerr=top["std_alpha"][::-1], color="steelblue", alpha=0.8)
    ax.set_xlabel("Mean mixing coefficient α")
    ax.set_title(f"Top-{top_n} proteins by mean α (GlobalCLIP contribution)")
    plt.tight_layout()
    return fig


def plot_phase_polar(
    phases: np.ndarray,
    rbp_names: list[str] | None = None,
    label_threshold: float = 0.0,
    alpha_values: np.ndarray | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Polar scatter of QLayer phases.

    Interpretation:
      - Proteins clustered together (similar φ) cooperate (constructive interference)
      - Proteins on opposite sides (Δφ ≈ π) compete or cancel (destructive)
      - After training on log-FE signal, background-noise proteins drift toward
        φ ≈ φ_signal + π and self-cancel.

    Args:
        phases:          (num_rbps,) phase values in radians.
        rbp_names:       Optional list of protein names for annotation.
        label_threshold: Only label proteins with mean_alpha >= this value.
        alpha_values:    Optional (num_rbps,) mean alpha for sizing/colour.
        figsize:         Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    sizes  = 20 + 200 * (alpha_values / alpha_values.max()) if alpha_values is not None else 30
    colors = alpha_values if alpha_values is not None else np.ones(len(phases))

    sc = ax.scatter(phases, np.ones_like(phases), c=colors, s=sizes,
                    cmap="viridis", alpha=0.7)
    if alpha_values is not None:
        plt.colorbar(sc, ax=ax, label="Mean α", shrink=0.6, pad=0.08)

    if rbp_names and alpha_values is not None:
        for phi, name, av in zip(phases, rbp_names, alpha_values):
            if av >= label_threshold:
                ax.annotate(name, (phi, 1.05), fontsize=5, ha="center", alpha=0.9)

    ax.set_rticks([])
    ax.set_title(
        "QLayer protein phases\n"
        "Clusters = cooperative binding  |  Opposite = competitive",
        pad=20,
    )
    plt.tight_layout()
    return fig


def plot_coupling_heatmap(
    coupling_matrix: np.ndarray,
    rbp_names: list[str] | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot QLayer cos(φ_i − φ_j) coupling matrix.

    Values > 0: proteins i and j have constructive interference → cooperate.
    Values < 0: destructive interference → compete or cancel each other.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(coupling_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="cos(φ_i − φ_j)")
    ax.set_title(
        "QLayer coupling matrix\n"
        "+1 = cooperative  |  −1 = competitive / noise suppression"
    )
    if rbp_names and len(rbp_names) <= 40:
        ax.set_xticks(range(len(rbp_names)))
        ax.set_yticks(range(len(rbp_names)))
        ax.set_xticklabels(rbp_names, rotation=90, fontsize=5)
        ax.set_yticklabels(rbp_names, fontsize=5)
    plt.tight_layout()
    return fig


def plot_pearson_distribution(
    all_r_standard: np.ndarray,
    all_r_qlayer: np.ndarray | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Histogram of per-sequence Pearson r values on the test set."""
    fig, ax = plt.subplots(figsize=figsize)
    kw = dict(bins=50, alpha=0.7, edgecolor="none")
    ax.hist(all_r_standard, label=f"Standard  (mean={np.mean(all_r_standard):.3f})",
            color="steelblue", **kw)
    if all_r_qlayer is not None:
        ax.hist(all_r_qlayer, label=f"QLayer    (mean={np.mean(all_r_qlayer):.3f})",
                color="darkorange", **kw)
    ax.axvline(np.mean(all_r_standard), color="steelblue", linestyle="--", linewidth=1.5)
    if all_r_qlayer is not None:
        ax.axvline(np.mean(all_r_qlayer), color="darkorange", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Pearson r (pred vs. log-enrichment target)")
    ax.set_ylabel("Number of sequences")
    ax.set_title("Test-set profile Pearson r distribution")
    ax.legend()
    plt.tight_layout()
    return fig
