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
        # control = batch["control"].to(device)  # not used

        pred, _ = model(seq)
        # target = _log_enrichment(signal, control)  # not used
        target = torch.log1p(signal)

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


def _rank_last_dim(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort(dim=-1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    arange = torch.arange(x.shape[-1], dtype=torch.float32, device=x.device).expand_as(x)
    ranks.scatter_(-1, order, arange)
    return ranks


@torch.no_grad()
def evaluate_spearman(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device,
) -> tuple[float, np.ndarray]:
    """Spearman rank correlation between prediction and log1p(signal), per sequence."""
    model.eval()
    corrs = []
    for batch in dataloader:
        seq, signal = batch["sequence"].to(device), batch["signal"].to(device)
        pred, _ = model(seq)
        target = torch.log1p(signal)
        p, t = _rank_last_dim(pred.squeeze(1)), _rank_last_dim(target.squeeze(1))
        pz, tz = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pz * tz).sum(-1) / (pz.norm(dim=-1) * tz.norm(dim=-1) + 1e-8)
        corrs.extend(r.cpu().float().numpy().tolist())
    all_r = np.array(corrs)
    return float(np.mean(all_r)), all_r


def _smooth_last_dim(x: torch.Tensor, n_window: int) -> torch.Tensor:
    if n_window <= 1:
        return x
    import torch.nn.functional as F
    pad = n_window // 2
    kernel = torch.ones(1, 1, n_window, device=x.device, dtype=x.dtype) / n_window
    x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
    smoothed = F.conv1d(x_padded, kernel).squeeze(1)
    return smoothed[..., : x.shape[-1]]


@torch.no_grad()
def evaluate_pearson_windowed(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, n_window: int,
) -> float:
    """Mean Pearson r after smoothing pred/target with an n_window moving average."""
    model.eval()
    corrs = []
    for batch in dataloader:
        seq, signal = batch["sequence"].to(device), batch["signal"].to(device)
        pred, _ = model(seq)
        target = torch.log1p(signal)
        p, t = _smooth_last_dim(pred.squeeze(1), n_window), _smooth_last_dim(target.squeeze(1), n_window)
        pz, tz = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pz * tz).sum(-1) / (pz.norm(dim=-1) * tz.norm(dim=-1) + 1e-8)
        corrs.extend(r.cpu().float().numpy().tolist())
    return float(np.mean(corrs))


@torch.no_grad()
def evaluate_pearson_subset(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, max_batches: int,
) -> float:
    """Mean Pearson r over at most `max_batches` batches (cheap overfitting check
    on the training split, without iterating the full training set)."""
    model.eval()
    corrs = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        seq, signal = batch["sequence"].to(device), batch["signal"].to(device)
        pred, _ = model(seq)
        target = torch.log1p(signal)
        p, t = pred.squeeze(1), target.squeeze(1)
        pz, tz = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pz * tz).sum(-1) / (pz.norm(dim=-1) * tz.norm(dim=-1) + 1e-8)
        corrs.extend(r.cpu().float().numpy().tolist())
    return float(np.mean(corrs))


class NaiveBaselineModel(nn.Module):
    """Sequence-agnostic baseline: uniform mean of the raw per-RBP eCLIP tracks."""

    def __init__(self, parnet_model: nn.Module, num_rbps: int = 223):
        super().__init__()
        self.backbone = parnet_model
        self.num_rbps = num_rbps

    @torch.no_grad()
    def forward(self, seq_onehot: torch.Tensor):
        x = self.backbone.stem(seq_onehot)
        x = self.backbone.body(x)
        if hasattr(self.backbone, "projection"):
            x = self.backbone.projection(x)
        rbp_tracks = self.backbone.head.head_target.pointwise_conv(x)
        pred = rbp_tracks.mean(dim=1, keepdim=True)
        alpha = torch.full((seq_onehot.shape[0], self.num_rbps),
                            1.0 / self.num_rbps, device=seq_onehot.device)
        return pred, alpha


def paired_significance(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Wilcoxon signed-rank test p-value for r_b vs. r_a (paired, per-sequence)."""
    from scipy.stats import wilcoxon
    _, p = wilcoxon(r_b, r_a)
    return float(p)


def bootstrap_mean_diff_ci(
    r_a: np.ndarray, r_b: np.ndarray, n_boot: int = 10000, seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on the mean difference (r_b - r_a). Returns (mean_diff, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    diffs = r_b - r_a
    n = len(diffs)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[boot_idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


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
