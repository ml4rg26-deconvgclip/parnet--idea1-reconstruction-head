#!/usr/bin/env python
"""Train the Idea 1 globalCLIP reconstruction head on frozen Parnet profiles.

Example:
    python scripts/train_globalclip_reconstruction_head.py \
        --globalclip-path /path/to/globalclip_interphase_600bp_signalfiltered.pt.gz \
        --pretrained-model-path /path/to/parnet.7m-0.0.pt \
        --output-dir runs/globalclip_reconstruction_head \
        --batch-size 16 --epochs 5 --lr 0.05

Only the small reconstruction head is trained and saved. The script never saves
Parnet predictions, input data, or a copy of the pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyTorch is required to train the reconstruction head.") from exc


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parnet_demo_utils import GlobalCLIPReconstructionHead  # noqa: E402


_BASE_TO_CHANNEL = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path)


def _load_pt_or_ptgz(path: Path) -> dict[str, Any]:
    """Load a torch dataset file, preferring a mmap-able companion .pt file."""
    if path.name.endswith(".pt.gz"):
        companion_pt = Path(str(path)[:-3])
        if companion_pt.exists():
            print(f"Loading companion .pt with mmap=True: {companion_pt}")
            return torch.load(companion_pt, mmap=True, weights_only=False)
        print(f"Loading gzip torch file: {path}")
        # A monolithic torch .pt.gz cannot be streamed sample-by-sample through
        # torch.load, but we only keep small per-batch tensors during training.
        with gzip.open(path, "rb") as handle:
            return torch.load(handle, weights_only=False)

    print(f"Loading torch file with mmap=True: {path}")
    return torch.load(path, mmap=True, weights_only=False)


def _fit_length_1d(values: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Pad or center-crop a 1D tensor to ``seq_len``."""
    if values.shape[0] == seq_len:
        return values
    if values.shape[0] > seq_len:
        start = (values.shape[0] - seq_len) // 2
        return values[start : start + seq_len]

    out = torch.zeros(seq_len, dtype=values.dtype)
    offset = (seq_len - values.shape[0]) // 2
    out[offset : offset + values.shape[0]] = values
    return out


def _pad_or_crop_sequence(seq: str, seq_len: int, meta: dict[str, Any] | None) -> str:
    """Return a DNA string of exactly ``seq_len`` using N-padding if needed."""
    if len(seq) == seq_len:
        return seq
    if len(seq) > seq_len:
        start = (len(seq) - seq_len) // 2
        return seq[start : start + seq_len]

    pad = seq_len - len(seq)
    pad_side = (meta or {}).get("pad_side")
    if pad_side == 1:
        return ("N" * pad) + seq
    if pad_side == 2:
        return seq + ("N" * pad)

    left = pad // 2
    right = pad - left
    return ("N" * left) + seq + ("N" * right)


def _sequence_to_onehot(
    sequence: str | bytes | torch.Tensor,
    seq_len: int,
    meta: dict[str, Any] | None,
) -> torch.Tensor:
    """Convert sequence storage to channels-first one-hot shape ``(4, seq_len)``."""
    if torch.is_tensor(sequence):
        tensor = sequence.float()
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D one-hot sequence, got {tuple(tensor.shape)}")
        if tensor.shape == (seq_len, 4):
            return tensor.T.contiguous()
        if tensor.shape == (4, seq_len):
            return tensor.contiguous()
        if tensor.shape[-1] == 4:
            channels_first = tensor.T.contiguous()
        elif tensor.shape[0] == 4:
            channels_first = tensor.contiguous()
        else:
            raise ValueError(f"Cannot infer one-hot sequence layout: {tuple(tensor.shape)}")

        if channels_first.shape[1] == seq_len:
            return channels_first
        fitted = [
            _fit_length_1d(channels_first[channel], seq_len)
            for channel in range(channels_first.shape[0])
        ]
        return torch.stack(fitted, dim=0)

    if isinstance(sequence, bytes):
        sequence = sequence.decode("utf-8")
    if not isinstance(sequence, str):
        raise TypeError(f"Unsupported sequence type: {type(sequence).__name__}")

    sequence = _pad_or_crop_sequence(sequence.upper(), seq_len, meta)
    onehot = torch.zeros(4, seq_len, dtype=torch.float32)
    for position, base in enumerate(sequence):
        channel = _BASE_TO_CHANNEL.get(base)
        if channel is not None:
            onehot[channel, position] = 1.0
    return onehot


def _sparse_or_dense_to_tensor(signal: Any) -> torch.Tensor:
    """Convert dense tensors or sparse COO dicts from .pt/.pt.gz datasets."""
    if torch.is_tensor(signal):
        return signal.float()
    if isinstance(signal, dict) and {"indices", "values", "size"} <= signal.keys():
        indices = torch.as_tensor(signal["indices"], dtype=torch.long)
        values = torch.as_tensor(signal["values"], dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, signal["size"]).to_dense()
    raise TypeError(f"Unsupported output signal type: {type(signal).__name__}")


def _sample_output_to_total(sample: dict[str, Any], output_key: str, seq_len: int) -> torch.Tensor:
    """Extract one observed profile from a sample and name it ``outputs['total']``."""
    outputs = sample.get("outputs", {})
    if output_key not in outputs:
        raise KeyError(
            f"Output key {output_key!r} not found; available keys: {list(outputs.keys())}"
        )

    signal = _sparse_or_dense_to_tensor(outputs[output_key])
    if signal.ndim == 1:
        return _fit_length_1d(signal, seq_len)

    if signal.ndim != 2:
        raise ValueError(f"Expected 1D or 2D signal, got {tuple(signal.shape)}")

    if signal.shape[-1] != seq_len and signal.shape[0] == seq_len:
        signal = signal.T.contiguous()
    if signal.shape[-1] != seq_len:
        signal = torch.stack(
            [_fit_length_1d(signal[track], seq_len) for track in range(signal.shape[0])],
            dim=0,
        )

    # globalCLIP has one observed track. If this is pointed at an eCLIP-style
    # multi-track key for debugging, use the first track so the task is still
    # one-profile reconstruction.
    if signal.shape[0] > 1:
        signal = signal[:1, :]
    return signal


def _make_batch(
    split_data: list[dict[str, Any]],
    indices: Iterable[int],
    seq_len: int,
    output_key: str,
) -> dict[str, dict[str, torch.Tensor]]:
    sequences = []
    totals = []
    for index in indices:
        sample = split_data[index]
        sequences.append(
            _sequence_to_onehot(
                sample["inputs"]["sequence"],
                seq_len,
                sample.get("meta", {}),
            )
        )
        totals.append(_sample_output_to_total(sample, output_key, seq_len))

    return {
        "inputs": {"sequence": torch.stack(sequences, dim=0)},
        "outputs": {"total": torch.stack(totals, dim=0)},
    }


def _batch_indices(
    n_samples: int,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    max_batches: int | None,
) -> list[list[int]]:
    order = list(range(n_samples))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)

    batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    if max_batches is not None:
        batches = batches[:max_batches]
    return [batch for batch in batches if batch]


def _get_valid_split(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    for split_name in ("valid", "validation", "val"):
        if split_name in data:
            return split_name, data[split_name]
    raise KeyError(
        "Could not find a validation split. Expected one of: valid, validation, val. "
        f"Available keys: {list(data.keys())}"
    )


def _load_pretrained_parnet(
    pretrained_model_path: Path,
    model_name: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load Parnet, falling back to direct torch.load for checkpoint compatibility."""
    model: torch.nn.Module | None = None
    try:
        from parnet_additional_utils import ParnetModelName, load_parnet_model
    except Exception as exc:
        print(
            "parnet_additional_utils model loader unavailable "
            f"({type(exc).__name__}: {exc}); falling back to direct torch.load."
        )
    else:
        try:
            model = load_parnet_model(
                ParnetModelName(model_name),
                pretrained_model_path,
                dtype=torch.float32,
                device=device,
            )
        except Exception as exc:
            print(
                "parnet_additional_utils.load_parnet_model failed "
                f"({type(exc).__name__}: {exc}); falling back to direct torch.load."
            )

    if model is None:
        # The VM may need a Parnet source that can unpickle NewAdditiveMix while
        # parnet_additional_utils expects a different parnet.data API. Direct
        # torch.load uses the installed checkpoint-compatible Parnet classes.
        model = torch.load(pretrained_model_path, map_location=device, weights_only=False)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Expected pretrained checkpoint to load as an nn.Module, "
            f"got {type(model).__name__}"
        )

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _extract_parnet_profiles(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    profile_key: str,
) -> torch.Tensor:
    """Run frozen Parnet and return selected probability profiles."""
    with torch.no_grad():
        # Checkpoint-compatible Parnet expects a dict and reads inputs["sequence"].
        output = model({"sequence": sequence})

    if isinstance(output, dict):
        if profile_key not in output:
            available = list(output.keys())
            print(f"Requested Parnet output key {profile_key!r} not found.")
            print(f"Available Parnet output keys: {available}")
            raise KeyError(
                f"Parnet output does not contain {profile_key!r}; available keys: {available}"
            )
        profile_logprob = output[profile_key]
    elif torch.is_tensor(output):
        profile_logprob = output
    else:
        raise TypeError(f"Unsupported Parnet output type: {type(output).__name__}")

    if not torch.is_tensor(profile_logprob):
        raise TypeError(
            "Expected selected Parnet profile output to be a tensor, "
            f"got {type(profile_logprob).__name__}"
        )
    if profile_logprob.ndim != 3:
        raise ValueError(
            f"Expected selected Parnet profile output to be 3D, "
            f"got {tuple(profile_logprob.shape)}"
        )
    return profile_logprob.exp()


def _target_to_probability(target: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize observed globalCLIP counts over sequence length."""
    target = target.float()
    if target.ndim == 3:
        if target.shape[1] > 1:
            target = target[:, 0, :]
        else:
            target = target[:, 0, :]
    elif target.ndim != 2:
        raise ValueError(f"Expected target shape (B, L) or (B, 1, L), got {tuple(target.shape)}")

    denom = target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return target / denom


def _profile_cross_entropy(
    reconstructed_profile: torch.Tensor,
    target_profile: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return -(target_profile * reconstructed_profile.clamp_min(eps).log()).sum(dim=-1).mean()


def _batch_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float) -> list[float]:
    """Return per-sample Pearson r values, skipping constant rows safely."""
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


def _parnet_has_grad(model: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and parameter.grad.detach().abs().sum().item() > 0
        for parameter in model.parameters()
    )


def _run_epoch(
    *,
    split_data: list[dict[str, Any]],
    batches: list[list[int]],
    parnet_model: torch.nn.Module,
    head: GlobalCLIPReconstructionHead,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    eps: float,
) -> tuple[float, float | None]:
    is_train = optimizer is not None
    head.train(is_train)
    losses = []
    pearsons: list[float] = []

    for batch_indices in batches:
        batch = _make_batch(
            split_data,
            batch_indices,
            args.seq_length,
            args.globalclip_key,
        )
        sequence = batch["inputs"]["sequence"].to(device)
        target = batch["outputs"]["total"].to(device)

        rbp_profiles = _extract_parnet_profiles(parnet_model, sequence, args.profile_key)
        reconstructed_profile, _ = head(rbp_profiles)
        target_profile = _target_to_probability(target, eps)
        loss = _profile_cross_entropy(reconstructed_profile, target_profile, eps)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if _parnet_has_grad(parnet_model):
                raise RuntimeError("Parnet is frozen but received gradients.")
            optimizer.step()

        losses.append(loss.detach().item())
        if not is_train:
            pearsons.extend(_batch_pearson(reconstructed_profile, target_profile, eps))

    mean_loss = float(sum(losses) / len(losses)) if losses else float("nan")
    mean_pearson = None
    if not is_train:
        mean_pearson = float(sum(pearsons) / len(pearsons)) if pearsons else float("nan")
    return mean_loss, mean_pearson


def _write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "valid_loss", "valid_pearson"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_weights(output_dir: Path, head: GlobalCLIPReconstructionHead) -> None:
    logits = head.weight_logits.detach().cpu()
    weights = torch.softmax(logits, dim=0)
    rows = [
        {
            "track_index": track_index,
            "weight": float(weights[track_index]),
            "weight_logit": float(logits[track_index]),
        }
        for track_index in range(weights.numel())
    ]

    weights_path = output_dir / "rbp_reconstruction_weights.csv"
    with weights_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_index", "weight", "weight_logit"])
        writer.writeheader()
        writer.writerows(rows)

    top_path = output_dir / "top_reconstruction_weights.csv"
    with top_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_index", "weight", "weight_logit"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["weight"], reverse=True)[:30])


def _save_head(
    output_dir: Path,
    head: GlobalCLIPReconstructionHead,
    args: argparse.Namespace,
    final_metrics: dict[str, float | int],
) -> None:
    payload = {
        "state_dict": head.state_dict(),
        "metadata": {
            "num_parnet_tracks": args.num_parnet_tracks,
            "seq_length": args.seq_length,
            "model_name": args.model_name,
            "profile_key": args.profile_key,
            "globalclip_key": args.globalclip_key,
            "pretrained_model_path": str(_resolve_path(args.pretrained_model_path)),
            "globalclip_path": str(_resolve_path(args.globalclip_path)),
            "final_metrics": final_metrics,
        },
    }
    torch.save(payload, output_dir / "reconstruction_head.pt")


def _write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    train_size: int,
    valid_split_name: str,
    valid_size: int,
) -> None:
    config = vars(args).copy()
    config.update(
        {
            "globalclip_path": str(_resolve_path(args.globalclip_path)),
            "pretrained_model_path": str(_resolve_path(args.pretrained_model_path)),
            "output_dir": str(_resolve_path(args.output_dir)),
            "device_resolved": str(device),
            "train_size": train_size,
            "valid_split_name": valid_split_name,
            "valid_size": valid_size,
        }
    )
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/train_globalclip_reconstruction_head.py \\\n"
            "    --globalclip-path /data/globalclip_interphase_600bp_signalfiltered.pt.gz \\\n"
            "    --pretrained-model-path /models/parnet.7m-0.0.pt \\\n"
            "    --output-dir runs/globalclip_head --epochs 5 --batch-size 16\n"
        ),
    )
    parser.add_argument("--globalclip-path", type=Path, required=True)
    parser.add_argument("--pretrained-model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="parnet.7m-0.0")
    parser.add_argument("--globalclip-key", default="globalCLIP")
    parser.add_argument("--control-key", default="control")
    parser.add_argument("--profile-key", default="total")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--num-parnet-tracks", type=int, default=223)
    parser.add_argument("--seq-length", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = _resolve_path(args.globalclip_path)
    model_path = _resolve_path(args.pretrained_model_path)
    if not data_path.exists():
        raise FileNotFoundError(f"globalCLIP file not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"pretrained model not found: {model_path}")

    print(f"Device: {device}")
    print(f"Loading globalCLIP data: {data_path}")
    data = _load_pt_or_ptgz(data_path)
    if "train" not in data:
        raise KeyError(f"Train split not found. Available keys: {list(data.keys())}")
    train_data = data["train"]
    valid_split_name, valid_data = _get_valid_split(data)
    print(f"Train samples: {len(train_data)}")
    print(f"Validation split: {valid_split_name} ({len(valid_data)} samples)")

    print(f"Loading frozen Parnet model: {model_path}")
    parnet_model = _load_pretrained_parnet(model_path, args.model_name, device)
    total_params = sum(parameter.numel() for parameter in parnet_model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in parnet_model.parameters() if parameter.requires_grad
    )
    print(f"Parnet parameters: {total_params:,}; trainable after freeze: {trainable_params:,}")

    head = GlobalCLIPReconstructionHead(
        num_tracks=args.num_parnet_tracks,
        seq_len=args.seq_length,
        normalize_reconstructed_profile=True,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    eps = 1e-8

    _write_run_config(
        output_dir,
        args,
        device,
        len(train_data),
        valid_split_name,
        len(valid_data),
    )

    metrics_rows: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_batches = _batch_indices(
            len(train_data),
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            max_batches=args.max_train_batches,
        )
        valid_batches = _batch_indices(
            len(valid_data),
            args.batch_size,
            shuffle=False,
            seed=args.seed,
            max_batches=args.max_valid_batches,
        )

        train_loss, _ = _run_epoch(
            split_data=train_data,
            batches=train_batches,
            parnet_model=parnet_model,
            head=head,
            optimizer=optimizer,
            args=args,
            device=device,
            eps=eps,
        )
        with torch.no_grad():
            valid_loss, valid_pearson = _run_epoch(
                split_data=valid_data,
                batches=valid_batches,
                parnet_model=parnet_model,
                head=head,
                optimizer=None,
                args=args,
                device=device,
                eps=eps,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_pearson": float("nan") if valid_pearson is None else valid_pearson,
        }
        metrics_rows.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"valid_loss={valid_loss:.6f} valid_pearson={row['valid_pearson']:.6f}"
        )

        _write_metrics(output_dir / "training_metrics.csv", metrics_rows)
        _write_weights(output_dir, head)
        _save_head(output_dir, head, args, row)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
