#!/usr/bin/env python
"""Evaluate an Idea 1 globalCLIP reconstruction head and simple baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyTorch is required to evaluate the reconstruction head.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_globalclip_reconstruction_head import (  # noqa: E402
    GlobalCLIPReconstructionHead,
    _batch_indices,
    _condition_output_dir,
    _CONTROL_HANDLING,
    _default_config_path,
    _extract_parnet_profiles,
    _get_eval_split,
    _load_pretrained_parnet,
    _load_pt_or_ptgz,
    _make_batch,
    _profile_cross_entropy,
    _read_yaml,
    _requested_conditions,
    _resolve_globalclip_path_for_condition,
    _resolve_path,
    _resolve_pretrained_model_path,
    _set_random_seed,
    _target_to_probability_and_mask,
    _validate_parnet_profiles,
    _validate_reconstruction_output,
    _validate_sequence_batch,
    _validate_target_key,
    _write_diagnostic_profile_plots,
    _write_json,
)


def _load_head(
    checkpoint_path: Path,
    *,
    device: torch.device,
    fallback_num_tracks: int,
    fallback_seq_len: int,
) -> tuple[GlobalCLIPReconstructionHead, dict[str, Any]]:
    """Load a saved reconstruction head checkpoint."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise TypeError(f"Expected a reconstruction head payload in {checkpoint_path}")
    metadata = payload.get("metadata", {})
    num_tracks = int(metadata.get("num_parnet_tracks", fallback_num_tracks))
    seq_len = int(metadata.get("seq_length", fallback_seq_len))
    head = GlobalCLIPReconstructionHead(
        num_tracks=num_tracks,
        seq_len=seq_len,
        normalize_reconstructed_profile=True,
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, metadata


def _resolve_checkpoint_path_for_condition(checkpoint_path: Path, condition: str) -> Path:
    """Resolve a checkpoint file for one condition."""
    checkpoint_path = _resolve_path(checkpoint_path)
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / condition / "reconstruction_head.pt"
        if candidate.exists():
            return candidate
        candidate = checkpoint_path / "reconstruction_head.pt"
        if candidate.exists() and condition == "custom":
            return candidate
        raise FileNotFoundError(
            f"Could not find reconstruction_head.pt for condition {condition!r} under "
            f"{checkpoint_path}"
        )
    return checkpoint_path


def _normalize_profile(profile: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize one or more profile rows over sequence length."""
    return profile / profile.sum(dim=-1, keepdim=True).clamp_min(eps)


def _get_spearman_function():
    """Return scipy spearmanr if available."""
    try:
        from scipy.stats import spearmanr
    except ModuleNotFoundError:
        print("Warning: scipy is not available; Spearman correlation will be skipped.")
        return None
    return spearmanr


def _empty_metric_accumulator() -> dict[str, Any]:
    """Return a metric accumulator dict."""
    return {
        "cross_entropy_sum": 0.0,
        "mse_sum": 0.0,
        "valid_windows": 0,
        "zero_signal_windows": 0,
        "pearsons": [],
        "spearmans": [],
    }


def _pearson_values(pred: torch.Tensor, target: torch.Tensor, eps: float) -> list[float]:
    """Return per-row Pearson values."""
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=-1)
    denom = torch.sqrt(
        pred_centered.square().sum(dim=-1) * target_centered.square().sum(dim=-1)
    )
    valid = denom > eps
    if not valid.any():
        return []
    return (numerator[valid] / denom[valid]).detach().cpu().tolist()


def _update_metrics(
    accumulator: dict[str, Any],
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    zero_signal_windows: int,
    eps: float,
    spearman_function,
) -> None:
    """Update metrics for one method on a valid batch."""
    if pred.numel() == 0:
        accumulator["zero_signal_windows"] += zero_signal_windows
        return
    valid_count = pred.shape[0]
    ce = _profile_cross_entropy(pred, target, eps)
    mse = pred.sub(target).square().mean(dim=-1).mean()
    accumulator["cross_entropy_sum"] += float(ce.detach().cpu()) * valid_count
    accumulator["mse_sum"] += float(mse.detach().cpu()) * valid_count
    accumulator["valid_windows"] += valid_count
    accumulator["zero_signal_windows"] += zero_signal_windows
    accumulator["pearsons"].extend(_pearson_values(pred, target, eps))

    if spearman_function is None:
        return
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for pred_row, target_row in zip(pred_np, target_np, strict=False):
        result = spearman_function(pred_row, target_row)
        rho = float(result.correlation)
        if not math.isnan(rho):
            accumulator["spearmans"].append(rho)


def _finalize_metrics(
    accumulator: dict[str, Any],
    *,
    condition: str,
    split_name: str,
    method: str,
) -> dict[str, Any]:
    """Convert an accumulator to a CSV-ready metrics row."""
    valid_windows = int(accumulator["valid_windows"])
    pearsons = accumulator["pearsons"]
    spearmans = accumulator["spearmans"]
    return {
        "condition": condition,
        "split": split_name,
        "method": method,
        "profile_cross_entropy": (
            accumulator["cross_entropy_sum"] / valid_windows if valid_windows else float("nan")
        ),
        "mse": accumulator["mse_sum"] / valid_windows if valid_windows else float("nan"),
        "mean_pearson": float(sum(pearsons) / len(pearsons)) if pearsons else float("nan"),
        "mean_spearman": (
            float(sum(spearmans) / len(spearmans)) if spearmans else float("nan")
        ),
        "valid_nonzero_windows": valid_windows,
        "zero_signal_windows": int(accumulator["zero_signal_windows"]),
    }


def _compute_mean_observed_profile(
    *,
    source_split_data: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    eps: float,
) -> tuple[torch.Tensor, int, int]:
    """Compute a mean observed globalCLIP profile for the baseline."""
    batches = _batch_indices(
        len(source_split_data),
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        max_batches=args.max_mean_baseline_batches,
    )
    profile_sum = torch.zeros(args.seq_length, device=device)
    valid_windows = 0
    zero_signal_windows = 0
    for batch_indices in batches:
        batch = _make_batch(source_split_data, batch_indices, args.seq_length, args.globalclip_key)
        target = batch["outputs"]["total"].float().to(device)
        target_profile, valid_mask, _ = _target_to_probability_and_mask(
            target,
            args.seq_length,
            eps,
        )
        zero_signal_windows += int((~valid_mask).sum().item())
        if valid_mask.any():
            profile_sum += target_profile[valid_mask].sum(dim=0)
            valid_windows += int(valid_mask.sum().item())

    if valid_windows == 0:
        print("Warning: no nonzero windows for mean observed baseline; using a uniform profile.")
        return torch.full((args.seq_length,), 1.0 / args.seq_length, device=device), 0, zero_signal_windows
    return profile_sum / valid_windows, valid_windows, zero_signal_windows


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write evaluation metrics to CSV."""
    fieldnames = [
        "condition",
        "split",
        "method",
        "profile_cross_entropy",
        "mse",
        "mean_pearson",
        "mean_spearman",
        "valid_nonzero_windows",
        "zero_signal_windows",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_one_condition(
    args: argparse.Namespace,
    *,
    condition: str,
    data_path: Path,
    model_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> None:
    """Evaluate one condition-specific head and baselines."""
    if not data_path.exists():
        raise FileNotFoundError(f"globalCLIP file not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"pretrained model not found: {model_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"reconstruction head checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_target_key(args.globalclip_key)
    device = torch.device(args.device)
    spearman_function = _get_spearman_function()

    print(f"Evaluating condition: {condition}")
    print(f"Split: {args.split}")
    print(f"Loading globalCLIP data: {data_path}")
    data = _load_pt_or_ptgz(data_path)
    actual_split_name, split_data = _get_eval_split(data, args.split)
    mean_source_split = data.get("train", split_data)
    mean_profile, mean_valid_windows, mean_zero_windows = _compute_mean_observed_profile(
        source_split_data=mean_source_split,
        args=args,
        device=device,
        eps=args.eps,
    )
    print(
        "Mean observed baseline source: "
        f"valid_nonzero_windows={mean_valid_windows}, zero_signal_windows={mean_zero_windows}"
    )

    print(f"Loading frozen Parnet model: {model_path}")
    parnet_model = _load_pretrained_parnet(model_path, args.model_name, device)
    print(f"Loading reconstruction head: {checkpoint_path}")
    head, checkpoint_metadata = _load_head(
        checkpoint_path,
        device=device,
        fallback_num_tracks=args.num_parnet_tracks,
        fallback_seq_len=args.seq_length,
    )
    args.num_parnet_tracks = head.num_tracks
    args.seq_length = head.seq_len or args.seq_length

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    random_weights = torch.softmax(torch.randn(args.num_parnet_tracks, generator=generator), dim=0).to(
        device
    )

    methods = {
        "trained_reconstruction_head": _empty_metric_accumulator(),
        "uniform_track_mixture": _empty_metric_accumulator(),
        "mean_observed_profile": _empty_metric_accumulator(),
        "average_parnet_profile": _empty_metric_accumulator(),
        "random_softmax_weights": _empty_metric_accumulator(),
    }
    batches = _batch_indices(
        len(split_data),
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        max_batches=args.max_batches,
    )
    parnet_output_shape: list[int] | None = None
    reconstruction_output_shape: list[int] | None = None

    head.eval()
    with torch.no_grad():
        for batch_indices in batches:
            batch = _make_batch(split_data, batch_indices, args.seq_length, args.globalclip_key)
            sequence = batch["inputs"]["sequence"].float().to(device)
            target = batch["outputs"]["total"].float().to(device)
            _validate_sequence_batch(sequence, args.seq_length)

            rbp_profiles = _extract_parnet_profiles(parnet_model, sequence, args.profile_key)
            _validate_parnet_profiles(
                rbp_profiles,
                batch_size=sequence.shape[0],
                num_tracks=args.num_parnet_tracks,
                seq_len=args.seq_length,
            )
            parnet_output_shape = list(rbp_profiles.shape)
            model_pred, _ = head(rbp_profiles)
            _validate_reconstruction_output(
                model_pred,
                batch_size=sequence.shape[0],
                seq_len=args.seq_length,
            )
            reconstruction_output_shape = list(model_pred.shape)
            target_profile, valid_mask, _ = _target_to_probability_and_mask(
                target,
                args.seq_length,
                args.eps,
            )
            zero_signal_windows = int((~valid_mask).sum().item())
            if not valid_mask.any():
                for accumulator in methods.values():
                    accumulator["zero_signal_windows"] += zero_signal_windows
                continue

            uniform_pred = _normalize_profile(rbp_profiles.mean(dim=1), args.eps)
            random_pred = _normalize_profile(torch.einsum("btl,t->bl", rbp_profiles, random_weights), args.eps)
            mean_pred = mean_profile.unsqueeze(0).expand(sequence.shape[0], -1)
            average_parnet_pred = uniform_pred
            predictions = {
                "trained_reconstruction_head": model_pred,
                "uniform_track_mixture": uniform_pred,
                "mean_observed_profile": mean_pred,
                "average_parnet_profile": average_parnet_pred,
                "random_softmax_weights": random_pred,
            }
            valid_target = target_profile[valid_mask]
            for method, pred in predictions.items():
                _update_metrics(
                    methods[method],
                    pred[valid_mask],
                    valid_target,
                    zero_signal_windows=zero_signal_windows,
                    eps=args.eps,
                    spearman_function=spearman_function,
                )

    rows = [
        _finalize_metrics(accumulator, condition=condition, split_name=actual_split_name, method=method)
        for method, accumulator in methods.items()
    ]
    _write_metrics_csv(output_dir / "evaluation_metrics.csv", rows)
    _write_json(
        output_dir / "evaluation_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(_resolve_path(args.config)),
            "condition": condition,
            "split": actual_split_name,
            "globalclip_path": str(data_path),
            "pretrained_model_path": str(model_path),
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "target_key": args.globalclip_key,
            "profile_key": args.profile_key,
            "controls_used": False,
            "control_handling": _CONTROL_HANDLING,
            "loss_type": "profile_cross_entropy_on_nonzero_globalclip_windows",
            "metrics": [row["method"] for row in rows],
            "spearman_available": spearman_function is not None,
            "random_seed": args.seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "split_size": len(split_data),
            "eval_batches": len(batches),
            "mean_baseline_valid_nonzero_windows": mean_valid_windows,
            "mean_baseline_zero_signal_windows": mean_zero_windows,
            "parnet_frozen": True,
            "parnet_output_shape": parnet_output_shape,
            "reconstruction_output_shape": reconstruction_output_shape,
            "reconstruction_head_type": "GlobalCLIPReconstructionHead_softmax_track_mixture",
            "checkpoint_metadata": checkpoint_metadata,
        },
    )

    if not args.no_plots:
        plot_args = argparse.Namespace(**vars(args))
        plot_args.condition_label = condition
        _write_diagnostic_profile_plots(
            output_dir=output_dir,
            split_name=actual_split_name,
            split_data=split_data,
            parnet_model=parnet_model,
            head=head,
            args=plot_args,
            device=device,
            eps=args.eps,
        )

    print(f"Saved evaluation outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--condition", choices=("interphase", "lysate", "both"), default=None)
    parser.add_argument("--globalclip-path", type=Path, default=None)
    parser.add_argument("--pretrained-model-path", type=Path, default=None)
    parser.add_argument("--model-name", default="parnet.7m-0.0")
    parser.add_argument("--globalclip-key", default="globalCLIP")
    parser.add_argument("--control-key", default="control")
    parser.add_argument("--profile-key", default="total")
    parser.add_argument("--split", default="test", choices=("train", "valid", "validation", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-mean-baseline-batches", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--num-parnet-tracks", type=int, default=223)
    parser.add_argument("--seq-length", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--plot-top-n", type=int, default=6)
    parser.add_argument("--plot-random-n", type=int, default=6)
    parser.add_argument("--max-plot-candidates", type=int, default=512)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run evaluation for one or more conditions."""
    args = parse_args()
    _set_random_seed(args.seed)
    config = _read_yaml(args.config)
    conditions = _requested_conditions(args)
    model_path = _resolve_pretrained_model_path(args, config)
    checkpoint_base = _resolve_path(args.checkpoint_path)
    if len(conditions) > 1 and not checkpoint_base.is_dir():
        raise ValueError(
            "--condition both requires --checkpoint-path to be a run directory containing "
            "interphase/reconstruction_head.pt and lysate/reconstruction_head.pt."
        )
    if args.output_dir is not None:
        base_output_dir = _resolve_path(args.output_dir)
    elif checkpoint_base.is_dir():
        base_output_dir = checkpoint_base / f"evaluation_{args.split}"
    else:
        base_output_dir = checkpoint_base.parent / f"evaluation_{args.split}"
    append_condition = len(conditions) > 1 or (args.condition is not None and args.output_dir is not None)

    for condition in conditions:
        data_path = _resolve_globalclip_path_for_condition(args, config, condition)
        checkpoint_path = _resolve_checkpoint_path_for_condition(checkpoint_base, condition)
        output_dir = _condition_output_dir(base_output_dir, condition, append_condition)
        _evaluate_one_condition(
            args,
            condition=condition,
            data_path=data_path,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
