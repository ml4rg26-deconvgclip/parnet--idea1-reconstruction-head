"""Export GlobalCLIP evaluation results as CSV + PNG (no notebook / no plt.show()).

Same analyses as notebooks/globalclip/03_1_evaluate_and_analyze.py.ipynb, but
written to run headless as a plain script. Every figure is saved as a PNG,
and the numeric data behind every figure/table is additionally saved as a
CSV so results can be read without opening a plot.

All output goes to a single flat folder: results/data/

Usage:
    pixi run -e parnet-dev-cu12 python scripts/export_results_data.py
    pixi run -e parnet-dev-cu12 python scripts/export_results_data.py --standard-run-id v2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dotmap import DotMap
from scipy.stats import pearsonr, wilcoxon

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR / "src"))

from parnet_additional_utils import ParnetModelName, load_parnet_model
from globalclip_utils import (
    GlobalCLIPStandardModel,
    GlobalCLIPQLayerModel,
    GlobalCLIPDataset,
    collect_alpha,
    rank_proteins,
    alpha_correlation_matrix,
    evaluate_pearson,
    plot_alpha_heatmap,
    plot_top_proteins,
    plot_phase_polar,
    plot_coupling_heatmap,
    plot_pearson_distribution,
    load_run_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--standard-run-id",     default="globalclip.standard.v1")
    p.add_argument("--qlayer-run-id",       default="globalclip.qlayer.v1")
    p.add_argument("--gpu",                 type=int, default=0)
    p.add_argument("--batch-size",          type=int, default=128)
    p.add_argument("--num-workers",         type=int, default=4)
    p.add_argument("--n-profile-examples",  type=int, default=5)
    p.add_argument("--top-n-proteins",      type=int, default=30)
    p.add_argument("--max-batches-track",   type=int, default=20)
    p.add_argument("--max-batches-overfit", type=int, default=20)
    p.add_argument("--output-dir",          default="results/data")
    return p.parse_args()


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {path.name}")


def save_csv(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info(f"  saved {path.name}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        log.info(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        log.warning("No GPU — running on CPU.")

    _fp_cfg = yaml.safe_load((PROJECT_DIR / "config" / "filepaths.server.yaml").read_text())
    pretrained_model_name = ParnetModelName.PARNET_7M_0_0

    def _res(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else PROJECT_DIR / p

    fp = DotMap()
    fp.pretrained_model = _res(_fp_cfg["models"][pretrained_model_name.value])
    fp.standard_run_dir = PROJECT_DIR / _fp_cfg["results"]["standard_model"] / args.standard_run_id
    fp.qlayer_run_dir   = PROJECT_DIR / _fp_cfg["results"]["qlayer_model"]   / args.qlayer_run_id
    fp.rbp_names_file   = PROJECT_DIR / "results" / "globalclip" / "datasets" / "rbp_names.txt"

    out_dir = PROJECT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {out_dir}")

    std_cfg = load_run_config(fp.standard_run_dir)
    qlayer_cfg = load_run_config(fp.qlayer_run_dir)
    fp.dataset = Path(std_cfg["dataset_path"])

    # ── Data ──────────────────────────────────────────────────────────────────
    rbp_names = fp.rbp_names_file.read_text().strip().split("\n")
    log.info(f"Loaded {len(rbp_names)} RBP names.")

    test_ds = GlobalCLIPDataset(fp.dataset, split="test", seq_len=600, total_key="globalCLIP")
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    log.info(f"Test set: {len(test_ds)} samples")

    # ── Models ────────────────────────────────────────────────────────────────
    log.info("Loading pretrained PARNET backbone...")
    parnet = load_parnet_model(pretrained_model_name, fp.pretrained_model,
                               dtype=torch.float32, device=device)
    parnet.eval()

    model_std = GlobalCLIPStandardModel(
        parnet_model=parnet,
        num_rbps=std_cfg["params_num_rbps"],
        mix_hidden=std_cfg["params_mix_hidden"],
    ).to(device)
    model_std.load_state_dict(
        torch.load(fp.standard_run_dir / "model.statedict.pt", map_location=device)
    )
    model_std.eval()
    log.info(f"Standard model loaded from {fp.standard_run_dir}")

    model_ql = GlobalCLIPQLayerModel(
        parnet_model=parnet,
        num_rbps=qlayer_cfg["params_num_rbps"],
        mix_hidden=qlayer_cfg["params_mix_hidden"],
        cnn_channels=qlayer_cfg["params_cnn_channels"],
        cnn_kernel=qlayer_cfg["params_cnn_kernel"],
        cnn_layers=qlayer_cfg["params_cnn_layers"],
    ).to(device)
    model_ql.load_state_dict(
        torch.load(fp.qlayer_run_dir / "model.statedict.pt", map_location=device)
    )
    model_ql.eval()
    log.info(f"QLayer model loaded from {fp.qlayer_run_dir}")

    # ══ 1 — Test-set Pearson r (Standard vs QLayer) ═════════════════════════════
    log.info("[1] Evaluating Standard/QLayer Pearson r on test set...")
    mean_r_std, all_r_std = evaluate_pearson(model_std, test_loader, device)
    mean_r_ql,  all_r_ql  = evaluate_pearson(model_ql,  test_loader, device)
    log.info(f"Standard mean r = {mean_r_std:.4f}   QLayer mean r = {mean_r_ql:.4f}")

    fig = plot_pearson_distribution(all_r_std, all_r_ql)
    save_fig(fig, out_dir, "01_pearson_distribution")

    # ══ 2 — Protein ranking (Standard) ═══════════════════════════════════════════
    log.info("[2] Collecting alpha (Standard) and ranking proteins...")
    alpha_std = collect_alpha(model_std, test_loader, device)
    log_scale_std = model_std.log_scale.exp().detach().cpu().numpy()
    ranking_std = rank_proteins(alpha_std, rbp_names, log_scale=log_scale_std)
    save_csv(ranking_std, out_dir, "02_protein_ranking_standard")

    fig = plot_top_proteins(ranking_std, top_n=args.top_n_proteins)
    save_fig(fig, out_dir, "02_protein_ranking_standard")

    # ══ 3 — Alpha correlation matrix (Standard) ══════════════════════════════════
    log.info("[3] Alpha correlation matrix (Standard)...")
    corr_std = alpha_correlation_matrix(alpha_std)
    save_csv(pd.DataFrame(corr_std, index=rbp_names, columns=rbp_names).reset_index(names="protein"),
             out_dir, "03_alpha_correlation_standard_matrix")

    fig = plot_alpha_heatmap(corr_std, rbp_names=None)
    save_fig(fig, out_dir, "03_alpha_correlation_standard")

    idx = np.triu_indices(len(rbp_names), k=1)
    pairs = [(corr_std[i, j], rbp_names[i], rbp_names[j]) for i, j in zip(idx[0], idx[1])]
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    save_csv(pd.DataFrame(pairs[:50], columns=["pearson_r", "protein_a", "protein_b"]),
             out_dir, "03_alpha_correlation_standard_top_pairs")

    # ══ 4 — Prediction profiles on high-signal test sequences ═══════════════════
    log.info("[4] Selecting high-signal test sequences and predicting profiles...")
    signal_totals = []
    with torch.no_grad():
        for batch in test_loader:
            signal_totals.extend(batch["signal"].squeeze(1).sum(-1).tolist())
    signal_totals = np.array(signal_totals)
    top_indices = signal_totals.argsort()[::-1][: args.n_profile_examples]

    profile_rows = []
    with torch.no_grad():
        for sample_idx in top_indices:
            sample = test_ds[sample_idx]
            seq = sample["sequence"].unsqueeze(0).to(device)
            signal = sample["signal"]
            target = torch.log1p(signal).squeeze().numpy()

            pred_std_np = model_std(seq)[0].squeeze().cpu().numpy()
            pred_ql_np  = model_ql(seq)[0].squeeze().cpu().numpy()

            for pos in range(len(target)):
                profile_rows.append({
                    "sample_index": int(sample_idx), "position": pos,
                    "ground_truth_log1p_signal": float(target[pos]),
                    "pred_standard": float(pred_std_np[pos]),
                    "pred_qlayer": float(pred_ql_np[pos]),
                })
    save_csv(pd.DataFrame(profile_rows), out_dir, "04_profile_comparison")

    fig, axes = plt.subplots(args.n_profile_examples, 3,
                              figsize=(16, 3.5 * args.n_profile_examples), sharex=True)
    with torch.no_grad():
        for row, sample_idx in enumerate(top_indices):
            sample = test_ds[sample_idx]
            seq = sample["sequence"].unsqueeze(0).to(device)
            signal = sample["signal"]
            target = torch.log1p(signal).squeeze().numpy()
            signal_np = signal.squeeze().numpy()

            pred_std_np = model_std(seq)[0].squeeze().cpu().numpy()
            pred_ql_np  = model_ql(seq)[0].squeeze().cpu().numpy()
            r_std, _ = pearsonr(pred_std_np, target)
            r_ql,  _ = pearsonr(pred_ql_np, target)

            pos = np.arange(len(signal_np))
            kw = dict(linewidth=0, alpha=0.8)
            axes[row, 0].fill_between(pos, signal_np, color="black", **kw)
            axes[row, 0].set_ylabel(f"Sample {sample_idx}\ncounts", fontsize=8)
            if row == 0:
                axes[row, 0].set_title("Ground truth (GlobalCLIP signal)")
            axes[row, 1].fill_between(pos, pred_std_np, color="steelblue", **kw)
            axes[row, 1].set_ylabel(f"r={r_std:.3f}", fontsize=8, color="steelblue")
            if row == 0:
                axes[row, 1].set_title("Standard model prediction")
            axes[row, 2].fill_between(pos, pred_ql_np, color="darkorange", **kw)
            axes[row, 2].set_ylabel(f"r={r_ql:.3f}", fontsize=8, color="darkorange")
            if row == 0:
                axes[row, 2].set_title("QLayer model prediction")
    plt.suptitle("Prediction profiles on high-signal test sequences", y=1.01)
    plt.tight_layout()
    save_fig(fig, out_dir, "04_profile_comparison")

    # ══ 5 — QLayer phase analysis ═════════════════════════════════════════════
    log.info("[5] Collecting alpha (QLayer) and phase analysis...")
    alpha_ql = collect_alpha(model_ql, test_loader, device)
    phases = model_ql.qlayer.phase.detach().cpu().numpy()
    mean_alpha_ql = alpha_ql.mean(0)
    phase_label_thresh = float(np.percentile(mean_alpha_ql, 80))

    fig = plot_phase_polar(phases=phases, rbp_names=rbp_names,
                            label_threshold=phase_label_thresh, alpha_values=mean_alpha_ql)
    save_fig(fig, out_dir, "05_qlayer_phase_polar")

    phase_df = pd.DataFrame({
        "protein": rbp_names, "phase_rad": phases, "mean_alpha": mean_alpha_ql,
    }).sort_values("phase_rad").reset_index(drop=True)
    save_csv(phase_df, out_dir, "05_qlayer_phases")

    # ══ 6 — QLayer coupling matrix ════════════════════════════════════════════
    log.info("[6] QLayer coupling matrix...")
    coupling = model_ql.get_coupling_matrix().cpu().numpy()
    save_csv(pd.DataFrame(coupling, index=rbp_names, columns=rbp_names).reset_index(names="protein"),
             out_dir, "06_qlayer_coupling_matrix")

    fig = plot_coupling_heatmap(coupling)
    save_fig(fig, out_dir, "06_qlayer_coupling")

    idx = np.triu_indices(len(rbp_names), k=1)
    coup_pairs = [(coupling[i, j], rbp_names[i], rbp_names[j]) for i, j in zip(idx[0], idx[1])]
    coup_pairs.sort(key=lambda x: x[0], reverse=True)
    coup_df = pd.DataFrame(coup_pairs, columns=["cos_delta_phi", "protein_a", "protein_b"])
    save_csv(pd.concat([coup_df.head(25), coup_df.tail(25)]), out_dir, "06_qlayer_coupling_top_pairs")

    # ══ 7 — Alpha correlation comparison (Standard vs QLayer) ════════════════════
    log.info("[7] Alpha correlation matrix (QLayer) + comparison plot...")
    corr_ql = alpha_correlation_matrix(alpha_ql)
    save_csv(pd.DataFrame(corr_ql, index=rbp_names, columns=rbp_names).reset_index(names="protein"),
             out_dir, "07_alpha_correlation_qlayer_matrix")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, corr, name in [(axes[0], corr_std, "Standard model"), (axes[1], corr_ql, "QLayer model")]:
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Alpha correlation matrix\n{name}")
    plt.tight_layout()
    save_fig(fig, out_dir, "07_alpha_correlation_comparison")

    # ══ 8 — Naive baseline + statistical significance ════════════════════════════
    log.info("[8] Naive baseline (uniform mean of raw RBP tracks) + significance...")

    class NaiveBaselineModel(nn.Module):
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

    model_base = NaiveBaselineModel(parnet, num_rbps=len(rbp_names)).to(device)
    mean_r_base, all_r_base = evaluate_pearson(model_base, test_loader, device)
    log.info(f"Baseline mean r = {mean_r_base:.4f}")

    def paired_significance(name_a, r_a, name_b, r_b):
        stat, p = wilcoxon(r_b, r_a)
        return float(p)

    p_base_std = paired_significance("Baseline", all_r_base, "Standard", all_r_std)
    p_base_ql  = paired_significance("Baseline", all_r_base, "QLayer",   all_r_ql)
    p_std_ql   = paired_significance("Standard", all_r_std,  "QLayer",   all_r_ql)

    def bootstrap_mean_diff_ci(r_a, r_b, n_boot=10000, seed=42):
        rng = np.random.default_rng(seed)
        diffs = r_b - r_a
        n = len(diffs)
        boot_idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = diffs[boot_idx].mean(axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return diffs.mean(), lo, hi

    sig_rows = []
    for name_a, r_a, name_b, r_b, p in [
        ("baseline", all_r_base, "standard", all_r_std, p_base_std),
        ("baseline", all_r_base, "qlayer",   all_r_ql,  p_base_ql),
        ("standard", all_r_std,  "qlayer",   all_r_ql,  p_std_ql),
    ]:
        mean_diff, lo, hi = bootstrap_mean_diff_ci(r_a, r_b)
        sig_rows.append({
            "comparison": f"{name_b}_vs_{name_a}", "mean_delta_r": mean_diff,
            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
            "wilcoxon_p": p, "significant": bool(lo > 0 or hi < 0),
        })
    save_csv(pd.DataFrame(sig_rows), out_dir, "08_significance_tests")

    # per-sequence Pearson r for all three models — the core evaluation table
    per_seq_df = pd.DataFrame({
        "sequence_index": np.arange(len(all_r_base)),
        "pearson_r_baseline": all_r_base,
        "pearson_r_standard": all_r_std,
        "pearson_r_qlayer": all_r_ql,
    })
    save_csv(per_seq_df, out_dir, "01_08_pearson_r_per_sequence")

    fig, ax = plt.subplots(figsize=(8, 5))
    kw = dict(bins=50, alpha=0.6, edgecolor="none")
    for all_r, label, color in [
        (all_r_base, f"Baseline (mean={mean_r_base:.3f})", "gray"),
        (all_r_std,  f"Standard (mean={mean_r_std:.3f})",  "steelblue"),
        (all_r_ql,   f"QLayer   (mean={mean_r_ql:.3f})",   "darkorange"),
    ]:
        ax.hist(all_r, label=label, color=color, **kw)
        ax.axvline(all_r.mean(), color=color, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Pearson r (pred vs. log1p(signal))")
    ax.set_ylabel("Number of sequences")
    ax.set_title("Test-set Pearson r: baseline vs. learned combinations")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, out_dir, "08_pearson_distribution_with_baseline")

    # ══ 9 — Sequence-independent baseline (mean training profile) ═══════════════
    log.info("[9] Sequence-independent mean training profile baseline...")
    train_ds_profile = GlobalCLIPDataset(fp.dataset, split="train", seq_len=600, total_key="globalCLIP")
    train_loader_profile = torch.utils.data.DataLoader(
        train_ds_profile, batch_size=256, shuffle=False, num_workers=args.num_workers,
    )
    profile_sum = torch.zeros(600)
    n_train = 0
    for batch in train_loader_profile:
        target = torch.log1p(batch["signal"]).squeeze(1)
        profile_sum += target.sum(0)
        n_train += target.shape[0]
    mean_train_profile = (profile_sum / n_train).numpy()

    mp_tensor = torch.from_numpy(mean_train_profile)
    mp_z = mp_tensor - mp_tensor.mean()
    mp_norm = mp_z.norm()
    all_r_meanprofile = []
    for batch in test_loader:
        target = torch.log1p(batch["signal"]).squeeze(1)
        tz = target - target.mean(-1, keepdim=True)
        r = (tz * mp_z[None, :]).sum(-1) / (tz.norm(dim=-1) * mp_norm + 1e-8)
        all_r_meanprofile.extend(r.numpy().tolist())
    mean_r_meanprofile = float(np.mean(all_r_meanprofile))
    log.info(f"Mean-profile (no sequence) baseline mean r = {mean_r_meanprofile:.4f}")

    save_csv(pd.DataFrame({"position": np.arange(600), "mean_log1p_signal": mean_train_profile}),
             out_dir, "09_mean_train_profile")

    # ══ 10 — Best single raw RBP track ════════════════════════════════════════
    log.info("[10] Per-track Pearson r on a subsample of the test set...")
    track_r_sum = torch.zeros(len(rbp_names))
    n_track_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.max_batches_track:
                break
            seq, signal = batch["sequence"].to(device), batch["signal"].to(device)
            x = parnet.stem(seq)
            x = parnet.body(x)
            if hasattr(parnet, "projection"):
                x = parnet.projection(x)
            rbp_tracks = parnet.head.head_target.pointwise_conv(x)
            target = torch.log1p(signal)
            p = rbp_tracks - rbp_tracks.mean(-1, keepdim=True)
            t = target - target.mean(-1, keepdim=True)
            r = (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + 1e-8)
            track_r_sum += r.mean(0).cpu()
            n_track_batches += 1
    mean_track_r = (track_r_sum / n_track_batches).numpy()
    track_df = pd.DataFrame({"protein": rbp_names, "pearson_r": mean_track_r}) \
        .sort_values("pearson_r", ascending=False).reset_index(drop=True)
    save_csv(track_df, out_dir, "10_best_single_track_pearson_r")

    # ══ 11 — Train vs. test Pearson r (overfitting check) ════════════════════════
    log.info("[11] Train vs. test Pearson r (overfitting check)...")
    train_ds_eval = GlobalCLIPDataset(fp.dataset, split="train", seq_len=600, total_key="globalCLIP")
    train_loader_eval = torch.utils.data.DataLoader(
        train_ds_eval, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    def evaluate_pearson_subset(model, dataloader, device, max_batches):
        model.eval()
        corrs = []
        with torch.no_grad():
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

    mean_r_std_train = evaluate_pearson_subset(model_std, train_loader_eval, device, args.max_batches_overfit)
    mean_r_ql_train  = evaluate_pearson_subset(model_ql,  train_loader_eval, device, args.max_batches_overfit)
    save_csv(pd.DataFrame([
        {"model": "standard", "train_r": mean_r_std_train, "test_r": mean_r_std, "gap": mean_r_std_train - mean_r_std},
        {"model": "qlayer",   "train_r": mean_r_ql_train,  "test_r": mean_r_ql,  "gap": mean_r_ql_train - mean_r_ql},
    ]), out_dir, "11_train_test_overfitting_gap")

    # ══ 12 — Training dynamics: train vs. val loss across epochs ════════════════
    log.info("[12] Loading training curves from csv_logs...")

    def load_metrics_epoch_df(run_dir):
        metrics_paths = sorted((run_dir / "csv_logs").glob("version_*/metrics.csv"))
        if not metrics_paths:
            raise FileNotFoundError(f"No csv_logs/version_*/metrics.csv found under {run_dir}")
        return pd.read_csv(metrics_paths[-1]).groupby("epoch").last().reset_index()

    epoch_df_std = load_metrics_epoch_df(fp.standard_run_dir)
    epoch_df_ql  = load_metrics_epoch_df(fp.qlayer_run_dir)
    save_csv(epoch_df_std.assign(model="standard"), out_dir, "12_training_curves_standard")
    save_csv(epoch_df_ql.assign(model="qlayer"), out_dir, "12_training_curves_qlayer")

    metrics_to_plot = ["loss", "pearson", "nll", "alpha_mean"]
    fig, axes = plt.subplots(2, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 7), sharex=True)
    for row, (epoch_df, name) in enumerate([(epoch_df_std, "Standard"), (epoch_df_ql, "QLayer")]):
        for col, metric in enumerate(metrics_to_plot):
            ax = axes[row, col]
            train_col, val_col = f"train/{metric}_epoch", f"val/{metric}"
            if train_col in epoch_df.columns:
                ax.plot(epoch_df["epoch"], epoch_df[train_col], marker="o", markersize=3, label="train")
            if val_col in epoch_df.columns:
                ax.plot(epoch_df["epoch"], epoch_df[val_col], marker="o", markersize=3, label="val")
            ax.set_title(f"{name} — {metric}")
            ax.set_xlabel("epoch")
            ax.legend(fontsize=7)
    plt.tight_layout()
    save_fig(fig, out_dir, "12_train_val_curves_comparison")

    # ══ 13 — Spearman rank correlation ════════════════════════════════════════
    log.info("[13] Spearman rank correlation on test set...")

    def _rank_last_dim(x: torch.Tensor) -> torch.Tensor:
        order = x.argsort(dim=-1)
        ranks = torch.empty_like(order, dtype=torch.float32)
        arange = torch.arange(x.shape[-1], dtype=torch.float32, device=x.device).expand_as(x)
        ranks.scatter_(-1, order, arange)
        return ranks

    @torch.no_grad()
    def evaluate_spearman(model, dataloader, device):
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

    mean_rho_base, all_rho_base = evaluate_spearman(model_base, test_loader, device)
    mean_rho_std,  all_rho_std  = evaluate_spearman(model_std,  test_loader, device)
    mean_rho_ql,   all_rho_ql   = evaluate_spearman(model_ql,   test_loader, device)

    save_csv(pd.DataFrame([
        {"model": "baseline", "pearson_mean": all_r_base.mean(), "pearson_median": np.median(all_r_base),
         "spearman_mean": all_rho_base.mean(), "spearman_median": np.median(all_rho_base)},
        {"model": "standard", "pearson_mean": all_r_std.mean(), "pearson_median": np.median(all_r_std),
         "spearman_mean": all_rho_std.mean(), "spearman_median": np.median(all_rho_std)},
        {"model": "qlayer", "pearson_mean": all_r_ql.mean(), "pearson_median": np.median(all_r_ql),
         "spearman_mean": all_rho_ql.mean(), "spearman_median": np.median(all_rho_ql)},
    ]), out_dir, "13_pearson_vs_spearman_summary")

    # ══ 14 — Windowed (smoothed) correlation ═════════════════════════════════════
    log.info("[14] Windowed/smoothed Pearson r robustness check...")

    def smooth_last_dim(x: torch.Tensor, n_window: int) -> torch.Tensor:
        if n_window <= 1:
            return x
        pad = n_window // 2
        kernel = torch.ones(1, 1, n_window, device=x.device, dtype=x.dtype) / n_window
        x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        smoothed = F.conv1d(x_padded, kernel).squeeze(1)
        return smoothed[..., : x.shape[-1]]

    @torch.no_grad()
    def evaluate_pearson_windowed(model, dataloader, device, n_window):
        model.eval()
        corrs = []
        for batch in dataloader:
            seq, signal = batch["sequence"].to(device), batch["signal"].to(device)
            pred, _ = model(seq)
            target = torch.log1p(signal)
            p, t = smooth_last_dim(pred.squeeze(1), n_window), smooth_last_dim(target.squeeze(1), n_window)
            pz, tz = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
            r = (pz * tz).sum(-1) / (pz.norm(dim=-1) * tz.norm(dim=-1) + 1e-8)
            corrs.extend(r.cpu().float().numpy().tolist())
        return float(np.mean(corrs))

    window_rows = []
    for n_window in [1, 5, 10, 25, 50]:
        window_rows.append({
            "n_window": n_window,
            "baseline": evaluate_pearson_windowed(model_base, test_loader, device, n_window),
            "standard": evaluate_pearson_windowed(model_std, test_loader, device, n_window),
            "qlayer": evaluate_pearson_windowed(model_ql, test_loader, device, n_window),
        })
    save_csv(pd.DataFrame(window_rows), out_dir, "14_windowed_correlation")

    # ══ 15 — Integrated Gradients (sequence attribution) ═════════════════════════
    log.info("[15] Integrated Gradients attribution on top test sequence...")

    def forward_with_grad(model, seq_onehot):
        x = model.backbone.stem(seq_onehot)
        x = model.backbone.body(x)
        if hasattr(model.backbone, "projection"):
            x = model.backbone.projection(x)
        embedding = x
        rbp_tracks = model.backbone.head.head_target.pointwise_conv(x)
        alpha = model.mix_coeff(embedding)
        scale = model.log_scale.exp()
        scaled = rbp_tracks * scale[None, :, None]
        if hasattr(model, "qlayer"):
            interference = model.qlayer(scaled, alpha)
            pred = model.cnn(interference)
        else:
            pred = (scaled * alpha[:, :, None]).sum(1, keepdim=True)
        return pred, alpha

    def integrated_gradients(model, seq_onehot, steps=50, batch=10):
        baseline = torch.zeros_like(seq_onehot)
        ig_alphas = torch.linspace(0, 1, steps, device=seq_onehot.device).view(steps, 1, 1)
        interpolated = baseline + ig_alphas * (seq_onehot - baseline)
        grads = []
        for i in range(0, steps, batch):
            chunk = interpolated[i:i + batch].clone().detach().requires_grad_(True)
            pred, _ = forward_with_grad(model, chunk)
            target = pred.sum()
            g, = torch.autograd.grad(target, chunk)
            grads.append(g.detach())
        avg_grad = torch.cat(grads, dim=0).mean(0)
        ig = (seq_onehot.squeeze(0) - baseline.squeeze(0)) * avg_grad
        return ig.sum(0).cpu().numpy()

    ig_sample = test_ds[top_indices[0]]
    ig_seq = ig_sample["sequence"].unsqueeze(0).to(device)
    ig_signal = ig_sample["signal"].squeeze().numpy()
    ig_std = integrated_gradients(model_std, ig_seq)
    ig_ql  = integrated_gradients(model_ql,  ig_seq)

    save_csv(pd.DataFrame({
        "position": np.arange(len(ig_signal)), "ground_truth_signal": ig_signal,
        "ig_standard": ig_std, "ig_qlayer": ig_ql,
    }), out_dir, f"15_integrated_gradients_sample_{int(top_indices[0])}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    pos = np.arange(len(ig_signal))
    axes[0].fill_between(pos, ig_signal, color="black", linewidth=0, alpha=0.8)
    axes[0].set_title(f"Ground truth signal (test sample {int(top_indices[0])})")
    axes[1].plot(pos, ig_std, color="steelblue", linewidth=1)
    axes[1].set_title("Integrated Gradients — Standard")
    axes[2].plot(pos, ig_ql, color="darkorange", linewidth=1)
    axes[2].set_title("Integrated Gradients — QLayer")
    axes[2].set_xlabel("Position")
    plt.tight_layout()
    save_fig(fig, out_dir, "15_integrated_gradients_example")

    # ══ Summary ══════════════════════════════════════════════════════════════════
    summary_rows = [
        {"model": "baseline (naive mean of raw tracks)",
         "trainable_params": 0,
         "mean_pearson_r": mean_r_base, "median_pearson_r": float(np.median(all_r_base)),
         "mean_spearman_rho": mean_rho_base},
        {"model": "mean_train_profile (no sequence)",
         "trainable_params": 0,
         "mean_pearson_r": mean_r_meanprofile, "median_pearson_r": float("nan"),
         "mean_spearman_rho": float("nan")},
        {"model": "standard (MixCoeffHead + log_scale)",
         "trainable_params": sum(p.numel() for p in model_std.parameters() if p.requires_grad),
         "mean_pearson_r": mean_r_std, "median_pearson_r": float(np.median(all_r_std)),
         "mean_spearman_rho": mean_rho_std},
        {"model": "qlayer (QLayer + dilated CNN)",
         "trainable_params": sum(p.numel() for p in model_ql.parameters() if p.requires_grad),
         "mean_pearson_r": mean_r_ql, "median_pearson_r": float(np.median(all_r_ql)),
         "mean_spearman_rho": mean_rho_ql},
    ]
    save_csv(pd.DataFrame(summary_rows), out_dir, "00_summary")

    results_json = {
        "baseline": {"mean_r": mean_r_base, "median_r": float(np.median(all_r_base)), "std_r": float(all_r_base.std())},
        "mean_train_profile": {"mean_r": mean_r_meanprofile},
        "standard": {"mean_r": mean_r_std, "median_r": float(np.median(all_r_std)), "std_r": float(all_r_std.std())},
        "qlayer": {"mean_r": mean_r_ql, "median_r": float(np.median(all_r_ql)), "std_r": float(all_r_ql.std())},
        "significance_wilcoxon_p": {
            "standard_vs_baseline": p_base_std,
            "qlayer_vs_baseline": p_base_ql,
            "qlayer_vs_standard": p_std_ql,
        },
    }
    (out_dir / "00_summary.json").write_text(json.dumps(results_json, indent=2))

    log.info("=== DONE ===")
    log.info(f"All CSVs and PNGs saved to: {out_dir}")


if __name__ == "__main__":
    main()
