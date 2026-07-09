"""Full evaluation of the new experimental models (positional-alpha / CNN-only
/ hybrid ablations) on the GlobalCLIP test set.

Unlike export_results_data.py (which always loads one Standard + one QLayer
run together for a joint comparison), this script evaluates each given model
independently and writes its results into its own subfolder, so runs stay
clearly separated. As of this version it runs the *same full analysis suite*
as export_results_data.py for every model given, not just a bare Pearson r:

    <output-dir>/<model>/summary.json                  mean/median/std r, n
    <output-dir>/<model>/pearson_r_per_sequence.csv
    <output-dir>/<model>/pearson_distribution.png
    <output-dir>/<model>/significance_vs_baseline.json  vs. frozen-backbone-only baseline
    <output-dir>/<model>/spearman.json
    <output-dir>/<model>/windowed_correlation.csv       r after smoothing (n=1,5,10,25,50)
    <output-dir>/<model>/protein_ranking.csv/.png       mean_alpha, effective_weight per RBP
    <output-dir>/<model>/alpha_correlation_matrix.csv/.png
    <output-dir>/<model>/coupling_matrix.csv/.png       QLayer/CombiLayer only
    <output-dir>/<model>/coupling_top_pairs.csv         QLayer/CombiLayer only
    <output-dir>/<model>/channel_ablation.json          CombiLayer only
    <output-dir>/00_combined_summary.csv                all provided models side by side

Any of the four --*-run-id flags may be omitted to skip that model.

Usage:
    pixi run -e parnet-dev-cu12 python scripts/evaluate_new_models.py \
        --standard-run-id globalclip.standard.positional_v1 \
        --qlayer-run-id globalclip.qlayer.positional_v1 \
        --cnn-run-id ablation_cnn_only_v1 \
        --combilayer-run-id globalclip.combilayer.v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from parnet_additional_utils import ParnetModelName, load_parnet_model
from globalclip_utils import (
    GlobalCLIPStandardModel,
    GlobalCLIPQLayerModel,
    GlobalCLIPCNNModel,
    GlobalCLIPHybridModel,
    GlobalCLIPDataset,
    evaluate_pearson,
    evaluate_spearman,
    evaluate_pearson_windowed,
    evaluate_pearson_subset,
    collect_alpha,
    rank_proteins,
    alpha_correlation_matrix,
    plot_alpha_heatmap,
    plot_top_proteins,
    plot_coupling_heatmap,
    NaiveBaselineModel,
    paired_significance,
    bootstrap_mean_diff_ci,
    load_run_config,
)

WINDOW_SIZES = [1, 5, 10, 25, 50]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--standard-run-id", default=None, help="Run ID under results/globalclip/standard/")
    p.add_argument("--qlayer-run-id",   default=None, help="Run ID under results/globalclip/qlayer/")
    p.add_argument("--cnn-run-id",      default=None, help="Run ID under results/globalclip/cnn_only/")
    p.add_argument("--combilayer-run-id", default=None, help="Run ID under results/globalclip/combilayer/")
    p.add_argument("--gpu",             type=int, default=0)
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--num-workers",     type=int, default=4)
    p.add_argument("--output-dir",      default="results/new")
    return p.parse_args()


def _res(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_DIR / p


def _get_parnet(name: str, fp_cfg: dict, device, cache: dict):
    """Load (or reuse from cache) a PARNET backbone by ParnetModelName value,
    so different runs can use different backbone sizes (e.g. 7M vs 21M)."""
    if name not in cache:
        model_name = ParnetModelName(name)
        path = _res(fp_cfg["models"][model_name.value])
        parnet = load_parnet_model(model_name, path, dtype=torch.float32, device=device)
        parnet.eval()
        cache[name] = parnet
    return cache[name]


def _load_model(model_cls, run_dir: Path, fp_cfg: dict, device, parnet_cache: dict, kwargs_from_cfg):
    cfg = load_run_config(run_dir)
    pretrained_name = cfg.get("pretrained_model_name", "parnet.7m-0.0")
    parnet = _get_parnet(pretrained_name, fp_cfg, device, parnet_cache)
    kwargs = kwargs_from_cfg(cfg)
    kwargs.setdefault("embed_dim", cfg.get("params_embed_dim", 512))
    model = model_cls(parnet_model=parnet, **kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "model.statedict.pt", map_location=device))
    model.eval()
    return model, cfg, parnet


def _test_loader(dataset_path: str, batch_size: int, num_workers: int, max_total_signal: float | None = None):
    ds = GlobalCLIPDataset(Path(dataset_path), split="test", seq_len=600, total_key="globalCLIP",
                           max_total_signal=max_total_signal)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def _load_rbp_names() -> list[str] | None:
    path = PROJECT_DIR / "results" / "globalclip" / "datasets" / "rbp_names.txt"
    if not path.exists():
        return None
    return path.read_text().strip().split("\n")


@torch.no_grad()
def _evaluate_pearson_ablated(model, test_loader, device, ablate: str) -> float:
    """Like evaluate_pearson, but zeroes one of GlobalCLIPHybridModel's two
    input channels ("mixed" or "interference") to measure how much that
    pathway actually contributes to prediction accuracy."""
    model.eval()
    corrs = []
    for batch in test_loader:
        seq, signal = batch["sequence"].to(device), batch["signal"].to(device)
        pred, _ = model(seq, ablate=ablate)
        target = torch.log1p(signal)
        p, t = pred.squeeze(1), target.squeeze(1)
        pz, tz = p - p.mean(-1, keepdim=True), t - t.mean(-1, keepdim=True)
        r = (pz * tz).sum(-1) / (pz.norm(dim=-1) * tz.norm(dim=-1) + 1e-8)
        corrs.extend(r.cpu().float().numpy().tolist())
    return float(sum(corrs) / len(corrs))


def _full_analysis(
    name: str, model, test_loader, device, parnet, num_rbps: int, rbp_names: list[str] | None, out_dir: Path,
    dataset_path: str | None = None, max_total_signal: float | None = None,
    batch_size: int = 128, num_workers: int = 4, max_overfit_batches: int = 20,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Core Pearson r ──────────────────────────────────────────────────────
    mean_r, all_r = evaluate_pearson(model, test_loader, device)
    summary = {
        "model":       name,
        "mean_r":      float(mean_r),
        "median_r":    float(np.median(all_r)),
        "std_r":       float(all_r.std()),
        "n_sequences": int(len(all_r)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"sequence_index": np.arange(len(all_r)), "pearson_r": all_r}).to_csv(
        out_dir / "pearson_r_per_sequence.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_r, bins=50, alpha=0.75, color="steelblue", edgecolor="none")
    ax.axvline(mean_r, color="steelblue", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Pearson r (pred vs. log1p(signal))")
    ax.set_ylabel("Number of sequences")
    ax.set_title(f"{name} — test-set Pearson r (mean={mean_r:.3f})")
    plt.tight_layout()
    fig.savefig(out_dir / "pearson_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Baseline comparison (frozen backbone, no combination layer at all) ──
    baseline_model = NaiveBaselineModel(parnet, num_rbps=num_rbps).to(device)
    base_mean_r, base_all_r = evaluate_pearson(baseline_model, test_loader, device)
    p_value = paired_significance(base_all_r, all_r)
    mean_diff, ci_lo, ci_hi = bootstrap_mean_diff_ci(base_all_r, all_r)
    significance = {
        "baseline_mean_r": float(base_mean_r),
        "model_mean_r": float(mean_r),
        "mean_delta_r": mean_diff,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "wilcoxon_p": p_value,
        "significant": bool(ci_lo > 0 or ci_hi < 0),
    }
    (out_dir / "significance_vs_baseline.json").write_text(json.dumps(significance, indent=2))

    # ── Spearman ──────────────────────────────────────────────────────────
    mean_rho, all_rho = evaluate_spearman(model, test_loader, device)
    (out_dir / "spearman.json").write_text(json.dumps({
        "mean_spearman": float(mean_rho), "median_spearman": float(np.median(all_rho)),
    }, indent=2))
    # Per-sequence values, same layout as pearson_r_per_sequence.csv, needed
    # for count-coverage-quartile box plots (mean/median alone are not enough).
    pd.DataFrame({"sequence_index": np.arange(len(all_rho)), "spearman_r": all_rho}).to_csv(
        out_dir / "spearman_r_per_sequence.csv", index=False
    )

    # ── Windowed correlation (robustness to positional noise) ──────────────
    windowed = {n: evaluate_pearson_windowed(model, test_loader, device, n) for n in WINDOW_SIZES}
    pd.DataFrame([windowed]).to_csv(out_dir / "windowed_correlation.csv", index=False)

    # ── Train vs. test overfitting gap (cheap: subsample of train batches) ──
    if dataset_path is not None:
        train_ds = GlobalCLIPDataset(Path(dataset_path), split="train", seq_len=600, total_key="globalCLIP",
                                      max_total_signal=max_total_signal)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        train_r = evaluate_pearson_subset(model, train_loader, device, max_overfit_batches)
        pd.DataFrame([{
            "model": name, "train_r": train_r, "test_r": summary["mean_r"],
            "gap": train_r - summary["mean_r"],
        }]).to_csv(out_dir / "overfitting_gap.csv", index=False)

    # ── Protein ranking + alpha correlation (needs rbp_names) ───────────────
    if rbp_names is not None:
        alpha_matrix = collect_alpha(model, test_loader, device)
        log_scale = model.log_scale.exp().detach().cpu().numpy() if hasattr(model, "log_scale") else None
        ranking_df = rank_proteins(alpha_matrix, rbp_names, log_scale=log_scale)
        ranking_df.to_csv(out_dir / "protein_ranking.csv", index=False)
        fig = plot_top_proteins(ranking_df, top_n=30)
        fig.savefig(out_dir / "protein_ranking.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        corr = alpha_correlation_matrix(alpha_matrix)
        pd.DataFrame(corr, index=rbp_names, columns=rbp_names).reset_index(names="protein").to_csv(
            out_dir / "alpha_correlation_matrix.csv", index=False
        )
        fig = plot_alpha_heatmap(corr, rbp_names=None)
        fig.savefig(out_dir / "alpha_correlation_matrix.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── QLayer/CombiLayer coupling matrix ────────────────────────────────────
    if hasattr(model, "get_coupling_matrix") and rbp_names is not None:
        sample_batch = next(iter(test_loader))["sequence"].to(device)
        try:
            coupling = model.get_coupling_matrix(sample_batch).cpu().numpy()
        except TypeError:
            coupling = model.get_coupling_matrix(seq_onehot=sample_batch).cpu().numpy()
        pd.DataFrame(coupling, index=rbp_names, columns=rbp_names).reset_index(names="protein").to_csv(
            out_dir / "coupling_matrix.csv", index=False
        )
        idx = np.triu_indices(len(rbp_names), k=1)
        pairs = sorted(
            [(coupling[i, j], rbp_names[i], rbp_names[j]) for i, j in zip(idx[0], idx[1])],
            key=lambda x: x[0],
        )
        pd.DataFrame(pairs[:25] + pairs[-25:], columns=["cos_delta_phi", "protein_a", "protein_b"]).to_csv(
            out_dir / "coupling_top_pairs.csv", index=False
        )
        fig = plot_coupling_heatmap(coupling, rbp_names=None)
        fig.savefig(out_dir / "coupling_matrix.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── CombiLayer channel ablation ──────────────────────────────────────────
    if hasattr(model, "channel_weight_summary"):
        r_no_interference = _evaluate_pearson_ablated(model, test_loader, device, ablate="interference")
        r_no_mixed = _evaluate_pearson_ablated(model, test_loader, device, ablate="mixed")
        ablation = {
            "mean_r_full": summary["mean_r"],
            "mean_r_interference_zeroed": r_no_interference,
            "mean_r_mixed_zeroed": r_no_mixed,
            "interference_contribution": summary["mean_r"] - r_no_interference,
            **model.channel_weight_summary(),
        }
        (out_dir / "channel_ablation.json").write_text(json.dumps(ablation, indent=2))
        print(f"  [{name}] channel ablation: {ablation}")

    summary["mean_spearman"] = float(mean_rho)
    summary["mean_delta_r_vs_baseline"] = mean_diff
    summary["wilcoxon_p_vs_baseline"] = p_value
    print(f"[{name}] mean r = {mean_r:.4f}  rho = {mean_rho:.4f}  "
          f"Δr vs baseline = {mean_diff:.4f} (p={p_value:.2e})  -> {out_dir}")
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    _fp_cfg = yaml.safe_load((PROJECT_DIR / "config" / "filepaths.server.yaml").read_text())
    parnet_cache: dict = {}  # keyed by pretrained_model_name, so different runs can use different backbones

    rbp_names = _load_rbp_names()
    if rbp_names is None:
        print("WARNING: rbp_names.txt not found -- skipping protein ranking / coupling analyses.")

    out_root = PROJECT_DIR / args.output_dir
    summaries = []

    if args.standard_run_id:
        run_dir = PROJECT_DIR / _fp_cfg["results"]["standard_model"] / args.standard_run_id
        model, cfg, parnet = _load_model(
            GlobalCLIPStandardModel, run_dir, _fp_cfg, device, parnet_cache,
            lambda cfg: dict(
                num_rbps=cfg["params_num_rbps"],
                mix_hidden=cfg["params_mix_hidden"],
                positional_alpha=cfg.get("params_positional_alpha", False),
            ),
        )
        loader = _test_loader(cfg["dataset_path"], args.batch_size, args.num_workers,
                               max_total_signal=cfg.get("params_max_total_signal"))
        summaries.append(_full_analysis("standard", model, loader, device, parnet,
                                         cfg["params_num_rbps"], rbp_names, out_root / "standard",
                                         dataset_path=cfg["dataset_path"],
                                         max_total_signal=cfg.get("params_max_total_signal"),
                                         batch_size=args.batch_size, num_workers=args.num_workers))

    if args.qlayer_run_id:
        run_dir = PROJECT_DIR / _fp_cfg["results"]["qlayer_model"] / args.qlayer_run_id
        model, cfg, parnet = _load_model(
            GlobalCLIPQLayerModel, run_dir, _fp_cfg, device, parnet_cache,
            lambda cfg: dict(
                num_rbps=cfg["params_num_rbps"],
                mix_hidden=cfg["params_mix_hidden"],
                cnn_channels=cfg["params_cnn_channels"],
                cnn_kernel=cfg["params_cnn_kernel"],
                cnn_layers=cfg["params_cnn_layers"],
                positional_alpha=cfg.get("params_positional_alpha", False),
                positional_phase=cfg.get("params_positional_phase", False),
            ),
        )
        loader = _test_loader(cfg["dataset_path"], args.batch_size, args.num_workers,
                               max_total_signal=cfg.get("params_max_total_signal"))
        summaries.append(_full_analysis("qlayer", model, loader, device, parnet,
                                         cfg["params_num_rbps"], rbp_names, out_root / "qlayer",
                                         dataset_path=cfg["dataset_path"],
                                         max_total_signal=cfg.get("params_max_total_signal"),
                                         batch_size=args.batch_size, num_workers=args.num_workers))

    if args.cnn_run_id:
        run_dir = PROJECT_DIR / "results" / "globalclip" / "cnn_only" / args.cnn_run_id
        model, cfg, parnet = _load_model(
            GlobalCLIPCNNModel, run_dir, _fp_cfg, device, parnet_cache,
            lambda cfg: dict(
                num_rbps=cfg["params_num_rbps"],
                mix_hidden=cfg["params_mix_hidden"],
                cnn_channels=cfg["params_cnn_channels"],
                cnn_kernel=cfg["params_cnn_kernel"],
                cnn_layers=cfg["params_cnn_layers"],
                positional_alpha=cfg.get("params_positional_alpha", False),
            ),
        )
        loader = _test_loader(cfg["dataset_path"], args.batch_size, args.num_workers,
                               max_total_signal=cfg.get("params_max_total_signal"))
        summaries.append(_full_analysis("cnn_only", model, loader, device, parnet,
                                         cfg["params_num_rbps"], rbp_names, out_root / "cnn_only",
                                         dataset_path=cfg["dataset_path"],
                                         max_total_signal=cfg.get("params_max_total_signal"),
                                         batch_size=args.batch_size, num_workers=args.num_workers))

    if args.combilayer_run_id:
        run_dir = PROJECT_DIR / "results" / "globalclip" / "combilayer" / args.combilayer_run_id
        model, cfg, parnet = _load_model(
            GlobalCLIPHybridModel, run_dir, _fp_cfg, device, parnet_cache,
            lambda cfg: dict(
                num_rbps=cfg["params_num_rbps"],
                mix_hidden=cfg["params_mix_hidden"],
                cnn_channels=cfg["params_cnn_channels"],
                cnn_kernel=cfg["params_cnn_kernel"],
                cnn_layers=cfg["params_cnn_layers"],
                positional_alpha=cfg.get("params_positional_alpha", False),
                positional_phase=cfg.get("params_positional_phase", False),
            ),
        )
        loader = _test_loader(cfg["dataset_path"], args.batch_size, args.num_workers,
                               max_total_signal=cfg.get("params_max_total_signal"))
        summaries.append(_full_analysis("combilayer", model, loader, device, parnet,
                                         cfg["params_num_rbps"], rbp_names, out_root / "combilayer",
                                         dataset_path=cfg["dataset_path"],
                                         max_total_signal=cfg.get("params_max_total_signal"),
                                         batch_size=args.batch_size, num_workers=args.num_workers))

    if not summaries:
        print("No --*-run-id given, nothing to evaluate.")
        return

    combined = pd.DataFrame(summaries)
    out_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_root / "00_combined_summary.csv", index=False)
    print(f"\nCombined summary saved to {out_root / '00_combined_summary.csv'}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
