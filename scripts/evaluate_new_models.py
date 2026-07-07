"""Evaluate the new experimental models (positional-alpha / CNN-only ablation)
on the GlobalCLIP test set.

Unlike export_results_data.py (which always loads one Standard + one QLayer
run together for a joint comparison), this script evaluates each given model
independently and writes its results into its own subfolder, so runs stay
clearly separated:

    results/new/standard/summary.json
    results/new/qlayer/summary.json
    results/new/cnn_only/summary.json
    results/new/00_combined_summary.csv   (all provided models side by side)

Any of the three --*-run-id flags may be omitted to skip that model.

Usage:
    pixi run -e parnet-dev-cu12 python scripts/evaluate_new_models.py \
        --standard-run-id globalclip.standard.positional_v1 \
        --qlayer-run-id globalclip.qlayer.positional_v1 \
        --cnn-run-id ablation_cnn_only_v1
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
    GlobalCLIPDataset,
    evaluate_pearson,
    load_run_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--standard-run-id", default=None, help="Run ID under results/globalclip/standard/")
    p.add_argument("--qlayer-run-id",   default=None, help="Run ID under results/globalclip/qlayer/")
    p.add_argument("--cnn-run-id",      default=None, help="Run ID under results/globalclip/cnn_only/")
    p.add_argument("--gpu",             type=int, default=0)
    p.add_argument("--batch-size",      type=int, default=128)
    p.add_argument("--num-workers",     type=int, default=4)
    p.add_argument("--output-dir",      default="results/new")
    return p.parse_args()


def _res(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_DIR / p


def _load_model(model_cls, run_dir: Path, parnet, device, kwargs_from_cfg):
    cfg = load_run_config(run_dir)
    kwargs = kwargs_from_cfg(cfg)
    model = model_cls(parnet_model=parnet, **kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "model.statedict.pt", map_location=device))
    model.eval()
    return model, cfg


def _test_loader(dataset_path: str, batch_size: int, num_workers: int):
    ds = GlobalCLIPDataset(Path(dataset_path), split="test", seq_len=600, total_key="globalCLIP")
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


def _evaluate_and_save(name: str, model, test_loader, device, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_r, all_r = evaluate_pearson(model, test_loader, device)

    summary = {
        "model":       name,
        "mean_r":      float(mean_r),
        "median_r":    float(np.median(all_r)),
        "std_r":       float(all_r.std()),
        "n_sequences": int(len(all_r)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({
        "sequence_index": np.arange(len(all_r)),
        "pearson_r":      all_r,
    }).to_csv(out_dir / "pearson_r_per_sequence.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_r, bins=50, alpha=0.75, color="steelblue", edgecolor="none")
    ax.axvline(mean_r, color="steelblue", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Pearson r (pred vs. log1p(signal))")
    ax.set_ylabel("Number of sequences")
    ax.set_title(f"{name} — test-set Pearson r (mean={mean_r:.3f})")
    plt.tight_layout()
    fig.savefig(out_dir / "pearson_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"[{name}] mean r = {mean_r:.4f}  median r = {summary['median_r']:.4f}  -> {out_dir}")
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    _fp_cfg = yaml.safe_load((PROJECT_DIR / "config" / "filepaths.server.yaml").read_text())
    pretrained_model_name = ParnetModelName.PARNET_7M_0_0
    pretrained_path = _res(_fp_cfg["models"][pretrained_model_name.value])

    parnet = load_parnet_model(pretrained_model_name, pretrained_path, dtype=torch.float32, device=device)
    parnet.eval()

    out_root = PROJECT_DIR / args.output_dir
    summaries = []

    if args.standard_run_id:
        run_dir = PROJECT_DIR / _fp_cfg["results"]["standard_model"] / args.standard_run_id
        model, cfg = _load_model(
            GlobalCLIPStandardModel, run_dir, parnet, device,
            lambda cfg: dict(
                num_rbps=cfg["params_num_rbps"],
                mix_hidden=cfg["params_mix_hidden"],
                positional_alpha=cfg.get("params_positional_alpha", False),
            ),
        )
        loader = _test_loader(cfg["dataset_path"], args.batch_size, args.num_workers)
        summaries.append(_evaluate_and_save("standard", model, loader, device, out_root / "standard"))

    if args.qlayer_run_id:
        run_dir = PROJECT_DIR / _fp_cfg["results"]["qlayer_model"] / args.qlayer_run_id
        model, cfg = _load_model(
            GlobalCLIPQLayerModel, run_dir, parnet, device,
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
        loader = _test_loader(cfg["dataset_path"], args.batch_size, args.num_workers)
        summaries.append(_evaluate_and_save("qlayer", model, loader, device, out_root / "qlayer"))

    if args.cnn_run_id:
        run_dir = PROJECT_DIR / "results" / "globalclip" / "cnn_only" / args.cnn_run_id
        model, cfg = _load_model(
            GlobalCLIPCNNModel, run_dir, parnet, device,
            lambda cfg: dict(
                num_rbps=cfg["params_num_rbps"],
                mix_hidden=cfg["params_mix_hidden"],
                cnn_channels=cfg["params_cnn_channels"],
                cnn_kernel=cfg["params_cnn_kernel"],
                cnn_layers=cfg["params_cnn_layers"],
                positional_alpha=cfg.get("params_positional_alpha", False),
            ),
        )
        loader = _test_loader(cfg["dataset_path"], args.batch_size, args.num_workers)
        summaries.append(_evaluate_and_save("cnn_only", model, loader, device, out_root / "cnn_only"))

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
