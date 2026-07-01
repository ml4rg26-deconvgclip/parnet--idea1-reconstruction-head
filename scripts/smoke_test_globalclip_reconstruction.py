#!/usr/bin/env python
"""Smoke test for the Idea 1 globalCLIP reconstruction baseline.

This script tries to use the repository's configured globalCLIP dataset and
pretrained Parnet checkpoint. If either optional resource is unavailable, it
falls back to synthetic tensors so the reconstruction head and gradient checks
can still run.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PyTorch is required. Run this inside one of the repo's pixi Parnet "
        "environments, for example: pixi run -e parnet-dev python "
        "scripts/smoke_test_globalclip_reconstruction.py"
    ) from exc


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


def _default_config_path() -> Path:
    for relative_path in (
        "config/filepaths.yaml",
        "config/filepaths.lambosaur-ms-01-2.yaml",
    ):
        path = PROJECT_DIR / relative_path
        if path.exists():
            return path
    return PROJECT_DIR / "config/filepaths.yaml"


def _resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"Config not found, continuing with CLI/default paths: {path}")
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to read the path config.") from exc
    return yaml.safe_load(path.read_text()) or {}


def _nested_get(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _synthetic_batch(batch_size: int, seq_len: int) -> dict[str, dict[str, torch.Tensor]]:
    base_idx = torch.randint(0, 4, (batch_size, seq_len))
    sequence = F.one_hot(base_idx, num_classes=4).permute(0, 2, 1).float()
    target_counts = torch.rand(batch_size, 1, seq_len).mul(20.0)
    return {
        "inputs": {"sequence": sequence},
        "outputs": {"total": target_counts},
    }


def _load_pt_or_ptgz(path: Path) -> dict[str, Any]:
    """Load a torch dataset file without parnet_additional_utils dataset classes."""
    if path.name.endswith(".pt.gz"):
        companion_pt = Path(str(path)[:-3])
        if companion_pt.exists():
            print(f"Loading companion .pt with mmap=True: {companion_pt}")
            return torch.load(companion_pt, mmap=True, weights_only=False)
        print(f"Loading gzip torch file: {path}")
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
    """Convert a sample sequence to channels-first one-hot shape ``(4, seq_len)``."""
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

    # The globalCLIP files have one track. If a caller points this smoke test at
    # an eCLIP-style multi-track key, keep the first track so the baseline still
    # reconstructs one observed profile.
    if signal.shape[0] > 1:
        print(
            f"Output key {output_key!r} has {signal.shape[0]} tracks; "
            "using the first track for this one-profile smoke test."
        )
        signal = signal[:1, :]
    return signal


def _load_batch_from_pt_file(
    dataset_path: Path,
    split: str,
    batch_size: int,
    seq_len: int,
    output_key: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load the first batch directly from a .pt.gz/.pt dataset file.

    This intentionally avoids parnet_additional_utils.GzListDataset. On the VM
    described for this task, the checkpoint-compatible Parnet source contains
    NewAdditiveMix but lacks parnet.data.datasets.ListDataset, which makes the
    parnet_additional_utils dataset imports fail. Direct torch.load keeps this
    smoke test independent from that dependency-version mismatch.
    """
    data = _load_pt_or_ptgz(dataset_path)
    if split not in data:
        available = [key for key in data if isinstance(data.get(key), list)]
        raise KeyError(f"Split {split!r} not found in {dataset_path}; available: {available}")

    split_data = data[split]
    if len(split_data) == 0:
        raise ValueError(f"Split {split!r} is empty in {dataset_path}")

    samples = split_data[:batch_size]
    sequences = []
    totals = []
    for sample in samples:
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


def _load_globalclip_batch(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path | None, bool]:
    configured_path = _nested_get(config, ("global_clip", args.globalclip_config_key))
    dataset_path = _resolve_path(args.globalclip_path or configured_path)

    if dataset_path is None or not dataset_path.exists():
        message = f"GlobalCLIP dataset not found: {dataset_path}"
        if args.strict_real_data:
            raise FileNotFoundError(message)
        print(f"{message}; using a synthetic batch.")
        return _synthetic_batch(args.batch_size, args.seq_length), dataset_path, False

    try:
        batch = _load_batch_from_pt_file(
            dataset_path,
            args.split,
            args.batch_size,
            args.seq_length,
            args.globalclip_key,
        )
    except Exception as exc:
        if args.strict_real_data:
            raise
        print(f"Could not load real globalCLIP batch ({exc}); using a synthetic batch.")
        return _synthetic_batch(args.batch_size, args.seq_length), dataset_path, False

    print(f"Loaded real globalCLIP batch from: {dataset_path}")
    print(f"  split={args.split}, batch_size={batch['inputs']['sequence'].shape[0]}")
    return batch, dataset_path, True


def _load_pretrained_parnet(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module | None, Path | None]:
    configured_path = _nested_get(config, ("models", args.model_name))
    model_path = _resolve_path(args.pretrained_model_path or configured_path)

    if model_path is None or not model_path.exists():
        message = f"Pretrained Parnet checkpoint not found: {model_path}"
        if args.strict_real_data:
            raise FileNotFoundError(message)
        print(f"{message}; using synthetic Parnet profiles.")
        return None, model_path

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
            model_name = ParnetModelName(args.model_name)
            model = load_parnet_model(
                model_name,
                model_path,
                dtype=torch.float32,
                device=device,
            )
        except Exception as exc:
            print(
                "parnet_additional_utils.load_parnet_model failed "
                f"({type(exc).__name__}: {exc}); falling back to direct torch.load."
            )

    if model is None:
        try:
            # This fallback avoids dependency-version mismatch between
            # parnet_additional_utils and the Parnet source needed to unpickle
            # the checkpoint, for example when the checkpoint needs
            # NewAdditiveMix but parnet_additional_utils expects
            # parnet.data.datasets.ListDataset.
            model = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as exc:
            if args.strict_real_data:
                raise
            print(
                "Direct torch.load of pretrained model failed "
                f"({type(exc).__name__}: {exc}); using synthetic Parnet profiles."
            )
            return None, model_path

        if not isinstance(model, torch.nn.Module):
            message = (
                "Direct checkpoint load did not return an nn.Module; "
                f"got {type(model).__name__}"
            )
            if args.strict_real_data:
                raise TypeError(message)
            print(f"{message}; using synthetic Parnet profiles.")
            return None, model_path

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Loaded pretrained Parnet model from: {model_path}")
    print(f"  parameters={total_params:,}, trainable_after_freeze={trainable_params:,}")
    return model, model_path


def _synthetic_rbp_profiles(
    batch_size: int,
    num_tracks: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    profiles = torch.rand(batch_size, num_tracks, seq_len, device=device)
    return profiles / profiles.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _extract_rbp_profiles(
    model: torch.nn.Module | None,
    sequence: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    if model is None:
        return _synthetic_rbp_profiles(
            sequence.shape[0],
            args.num_parnet_tracks,
            args.seq_length,
            device,
        )

    with torch.no_grad():
        # The checkpoint-compatible Parnet source expects dictionary inputs and
        # indexes inputs["sequence"] inside forward.
        # Parnet forward API differs across versions: some expect
        # a dict {"sequence": sequence}, while this VM's installed
        # parnet package expects the sequence tensor directly.
        try:
            output = model({"sequence": sequence})
        except TypeError as exc:
            if "conv1d()" in str(exc) or "invalid combination of arguments" in str(exc):
                output = model(sequence)
            else:
                raise
    print("Parnet output shapes:")
    if isinstance(output, dict):
        for key, value in output.items():
            if torch.is_tensor(value):
                print(f"  {key}: {tuple(value.shape)}")
            else:
                print(f"  {key}: {type(value).__name__}")

        if args.profile_key not in output:
            available = list(output.keys())
            print(f"Requested Parnet output key {args.profile_key!r} not found.")
            print(f"Available Parnet output keys: {available}")
            raise KeyError(
                f"Parnet output does not contain {args.profile_key!r}; "
                f"available keys: {available}"
            )
        profile_logprob = output[args.profile_key]
        output_label = f"out[{args.profile_key!r}]"
    elif torch.is_tensor(output):
        print(f"  tensor: {tuple(output.shape)}")
        profile_logprob = output
        output_label = "tensor output"
    else:
        raise TypeError(f"Unsupported Parnet output type: {type(output).__name__}")

    if not torch.is_tensor(profile_logprob):
        raise TypeError(
            f"Expected Parnet profile output to be a tensor, "
            f"got {type(profile_logprob).__name__}"
        )
    if profile_logprob.ndim != 3:
        raise ValueError(
            f"Expected Parnet {args.profile_key!r} output to be 3D, "
            f"got {tuple(profile_logprob.shape)}"
        )

    profile_prob = profile_logprob.exp()
    sums = profile_prob.sum(dim=-1)
    print(f"Using Parnet output: {output_label}.exp()")
    print(
        "  profile probability sums over length: "
        f"min={sums.min().item():.6f}, max={sums.max().item():.6f}"
    )
    return profile_prob


def _extract_target_profile(
    batch: dict[str, Any],
    device: torch.device,
    eps: float,
) -> torch.Tensor | None:
    outputs = batch.get("outputs", {})
    target = outputs.get("total") if isinstance(outputs, dict) else None
    if target is None:
        return None

    target = target.float().to(device)
    if target.ndim == 3:
        if target.shape[1] > 1:
            print(
                "Target has more than one track; using the first track for "
                f"this one-profile smoke test: {tuple(target.shape)}"
            )
        target = target[:, 0, :]
    elif target.ndim != 2:
        raise ValueError(f"Expected target shape (batch, seq_len), got {tuple(target.shape)}")

    total = target.sum(dim=-1, keepdim=True)
    if not torch.all(total > 0):
        print("Target contains zero-signal rows; skipping real target loss.")
        return None
    return target / total.clamp_min(eps)


def _profile_cross_entropy(
    reconstructed_profile: torch.Tensor,
    target_profile: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return -(target_profile * reconstructed_profile.clamp_min(eps).log()).sum(dim=-1).mean()


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and parameter.grad.detach().abs().sum().item() > 0
        for parameter in module.parameters()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--globalclip-path", type=Path, default=None)
    parser.add_argument("--pretrained-model-path", type=Path, default=None)
    parser.add_argument("--model-name", default="parnet.7m-0.0")
    parser.add_argument("--globalclip-config-key", default="data_interphase_control")
    parser.add_argument("--globalclip-key", default="globalCLIP")
    parser.add_argument("--control-key", default="control")
    parser.add_argument("--profile-key", default="total")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-length", type=int, default=600)
    parser.add_argument("--num-parnet-tracks", type=int, default=223)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Use CUDA only when available unless explicitly set.",
    )
    parser.add_argument(
        "--strict-real-data",
        action="store_true",
        help="Fail instead of falling back to synthetic data/profiles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("GlobalCLIP reconstruction smoke test")
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Config     : {args.config}")
    print(f"Device     : {device}")

    config = _read_yaml(_resolve_path(args.config) or args.config)
    batch, _, real_batch = _load_globalclip_batch(args, config)

    sequence = batch["inputs"]["sequence"].float().to(device)
    print(f"Input sequence shape        : {tuple(sequence.shape)}")
    print(f"Using real globalCLIP batch : {real_batch}")

    parnet_model, _ = _load_pretrained_parnet(args, config, device)
    rbp_profiles = _extract_rbp_profiles(parnet_model, sequence, args, device)
    print(f"RBP profile shape           : {tuple(rbp_profiles.shape)}")

    head = GlobalCLIPReconstructionHead(
        num_tracks=rbp_profiles.shape[1],
        seq_len=rbp_profiles.shape[2],
        normalize_reconstructed_profile=True,
        eps=args.eps,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    reconstructed_profile, weights = head(rbp_profiles)
    print(f"Reconstructed profile shape : {tuple(reconstructed_profile.shape)}")
    print(f"Weight shape                : {tuple(weights.shape)}")
    print(f"Weight min/max/sum          : {weights.min().item():.6f} / "
          f"{weights.max().item():.6f} / {weights.sum().item():.6f}")

    target_profile = _extract_target_profile(batch, device, args.eps)
    if target_profile is None:
        target_profile = torch.rand_like(reconstructed_profile)
        target_profile = target_profile / target_profile.sum(
            dim=-1, keepdim=True
        ).clamp_min(args.eps)
        print("Using synthetic target profile for gradient smoke test.")
    print(f"Target profile shape        : {tuple(target_profile.shape)}")

    loss = _profile_cross_entropy(reconstructed_profile, target_profile, args.eps)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    head_has_grad = _has_nonzero_grad(head)
    parnet_has_grad = _has_nonzero_grad(parnet_model) if parnet_model is not None else False

    print(f"Profile loss                : {loss.item():.6f}")
    print(f"Reconstruction head grads   : {head_has_grad}")
    if parnet_model is not None:
        print(f"Parnet grads after backward : {parnet_has_grad}")
    else:
        print("Parnet grads after backward : skipped (model not loaded)")

    if not head_has_grad:
        raise RuntimeError("Expected reconstruction head to receive gradients.")
    if parnet_has_grad:
        raise RuntimeError("Parnet is frozen but received gradients.")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
