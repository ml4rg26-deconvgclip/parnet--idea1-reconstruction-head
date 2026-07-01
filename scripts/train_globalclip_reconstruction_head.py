#!/usr/bin/env python
"""Train the Idea 1 globalCLIP reconstruction head on frozen Parnet profiles.

Example:
    python scripts/train_globalclip_reconstruction_head.py \
        --condition interphase \
        --config config/filepaths.lambosaur-ms-01-2.yaml \
        --pretrained-model-path resources/models/parnet.7m-0.0.pt \
        --output-dir runs/idea1 \
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
from collections import defaultdict
from datetime import datetime, timezone
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
_CONDITION_TO_CONFIG_KEY = {
    "interphase": "data_interphase_control",
    "lysate": "data_lysate_control",
}
_CONTROL_HANDLING = "ignored_for_idea1_reconstruction_head_v1"
_RBP_COLUMN_CANDIDATES = (
    "rbp",
    "rbp_name",
    "rbpname",
    "protein",
    "target",
    "target_name",
    "gene",
    "gene_name",
    "symbol",
)
_CELL_LINE_COLUMN_CANDIDATES = (
    "ct",
    "cell",
    "cell_line",
    "cellline",
    "cell_type",
    "celltype",
    "biosample",
)
_TRACK_COLUMN_CANDIDATES = (
    "rbp_ct",
    "track",
    "track_name",
    "experiment",
    "experiment_id",
    "accession",
    "encode_accession",
    "dataset",
)


def _default_config_path() -> Path:
    """Return the first existing filepaths config used by this repository."""
    for relative_path in (
        "config/filepaths.yaml",
        "config/filepaths.lambosaur-ms-01-2.yaml",
    ):
        path = PROJECT_DIR / relative_path
        if path.exists():
            return path
    return PROJECT_DIR / "config/filepaths.yaml"


def _resolve_path(path_like: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path)


def _resolve_optional_path(path_like: str | Path | None) -> Path | None:
    """Resolve an optional path relative to the project root."""
    if path_like is None:
        return None
    return _resolve_path(path_like)


def _read_yaml(path: Path | None) -> dict[str, Any]:
    """Read a YAML config, returning an empty dict when it is absent."""
    if path is None:
        return {}
    path = _resolve_path(path)
    if not path.exists():
        print(f"Warning: config file not found: {path}")
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to read the path config.") from exc
    return yaml.safe_load(path.read_text()) or {}


def _nested_get(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return a nested value from a dict, or None if any key is absent."""
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_random_seed(seed: int) -> None:
    """Seed Python and torch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_condition_from_path(path: Path) -> str:
    """Infer a condition label from a manually supplied globalCLIP path."""
    lowered = str(path).lower()
    if "lysate" in lowered:
        return "lysate"
    if "interphase" in lowered:
        return "interphase"
    return "custom"


def _requested_conditions(args: argparse.Namespace) -> list[str]:
    """Expand the requested condition setting into concrete condition labels."""
    if args.condition == "both":
        if args.globalclip_path is not None:
            raise ValueError(
                "--condition both requires config-based paths; pass no --globalclip-path "
                "so interphase and lysate are trained as separate runs."
            )
        return ["interphase", "lysate"]
    if args.condition is not None:
        return [args.condition]
    if args.globalclip_path is not None:
        return [_infer_condition_from_path(_resolve_path(args.globalclip_path))]
    return ["interphase"]


def _condition_output_dir(base_output_dir: Path, condition: str, append_condition: bool) -> Path:
    """Return the output directory for one condition run."""
    if append_condition:
        if base_output_dir.name == condition:
            return base_output_dir
        return base_output_dir / condition
    return base_output_dir


def _resolve_globalclip_path_for_condition(
    args: argparse.Namespace,
    config: dict[str, Any],
    condition: str,
) -> Path:
    """Resolve the globalCLIP dataset path for one condition."""
    explicit_path = _resolve_optional_path(args.globalclip_path)
    if explicit_path is not None:
        return explicit_path

    config_key = _CONDITION_TO_CONFIG_KEY.get(condition)
    if config_key is None:
        raise ValueError(
            f"Cannot resolve a config globalCLIP path for condition {condition!r}; "
            "pass --globalclip-path explicitly."
        )
    configured_path = _nested_get(config, ("global_clip", config_key))
    if configured_path is None:
        raise KeyError(
            f"Config does not define global_clip.{config_key} for condition {condition!r}."
        )
    return _resolve_path(configured_path)


def _resolve_pretrained_model_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    """Resolve the pretrained Parnet checkpoint path."""
    explicit_path = _resolve_optional_path(args.pretrained_model_path)
    if explicit_path is not None:
        return explicit_path
    configured_path = _nested_get(config, ("models", args.model_name))
    if configured_path is None:
        raise KeyError(
            f"Config does not define models.{args.model_name}; pass --pretrained-model-path."
        )
    return _resolve_path(configured_path)


def _resolve_metadata_path(args: argparse.Namespace, config: dict[str, Any]) -> Path | None:
    """Resolve the full RBP metadata table path if configured."""
    explicit_path = _resolve_optional_path(args.metadata_path)
    if explicit_path is not None:
        return explicit_path
    configured_path = _nested_get(config, ("metadata", "full_rbp_set"))
    if configured_path is None:
        return None
    return _resolve_path(configured_path)


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


def _validate_target_key(output_key: str) -> None:
    """Fail early if the target key looks like an eCLIP label."""
    if output_key.lower() == "eclip":
        raise ValueError(
            "--globalclip-key must point to observed globalCLIP, not eCLIP target labels."
        )
    if output_key != "globalCLIP":
        print(
            f"Warning: target key is {output_key!r}, not the default 'globalCLIP'. "
            "Make sure this is an observed globalCLIP target."
        )


def _sample_output_to_total(sample: dict[str, Any], output_key: str, seq_len: int) -> torch.Tensor:
    """Extract one observed globalCLIP profile from one sample."""
    outputs = sample.get("outputs", {})
    if output_key not in outputs:
        raise KeyError(
            f"Output key {output_key!r} not found; available keys: {list(outputs.keys())}"
        )

    signal = _sparse_or_dense_to_tensor(outputs[output_key])
    if signal.ndim == 1:
        return _fit_length_1d(signal, seq_len).unsqueeze(0)

    if signal.ndim != 2:
        raise ValueError(f"Expected 1D or 2D target signal, got {tuple(signal.shape)}")

    if signal.shape == (seq_len, 1):
        signal = signal.T.contiguous()
    elif signal.shape[-1] != seq_len:
        signal = torch.stack(
            [_fit_length_1d(signal[track], seq_len) for track in range(signal.shape[0])],
            dim=0,
        )

    if signal.shape != (1, seq_len):
        raise ValueError(
            f"Expected observed globalCLIP target shape (1, {seq_len}); "
            f"got {tuple(signal.shape)} for key {output_key!r}. "
            "This often means an eCLIP multi-track label was selected by mistake."
        )
    return signal.float()


def _make_batch(
    split_data: list[dict[str, Any]],
    indices: Iterable[int],
    seq_len: int,
    output_key: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build one in-memory batch from a list-like split."""
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
    """Return a deterministic list of mini-batch sample indices."""
    order = list(range(n_samples))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)

    batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    if max_batches is not None:
        batches = batches[:max_batches]
    return [batch for batch in batches if batch]


def _get_valid_split(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Return the validation split from a loaded dataset dict."""
    for split_name in ("valid", "validation", "val"):
        if split_name in data:
            return split_name, data[split_name]
    raise KeyError(
        "Could not find a validation split. Expected one of: valid, validation, val. "
        f"Available keys: {list(data.keys())}"
    )


def _get_eval_split(data: dict[str, Any], split_name: str) -> tuple[str, list[dict[str, Any]]]:
    """Return one split by name, accepting common validation aliases."""
    if split_name == "valid":
        return _get_valid_split(data)
    if split_name in data:
        return split_name, data[split_name]
    raise KeyError(f"Split {split_name!r} not found. Available keys: {list(data.keys())}")


def _available_output_keys(split_data: list[dict[str, Any]]) -> list[str]:
    """Return output keys from the first sample in a split."""
    if not split_data:
        return []
    outputs = split_data[0].get("outputs", {})
    if not isinstance(outputs, dict):
        return []
    return list(outputs.keys())


def _split_sizes(data: dict[str, Any]) -> dict[str, int]:
    """Return split sizes for list-like dataset entries."""
    return {
        key: len(value)
        for key, value in data.items()
        if isinstance(value, (list, tuple))
    }


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


def _assert_parnet_frozen(model: torch.nn.Module) -> None:
    """Raise if any Parnet parameter is trainable."""
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"Parnet is expected to be frozen; trainable parameters: {trainable[:10]}")


def _extract_parnet_profiles(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    profile_key: str,
) -> torch.Tensor:
    """Run frozen Parnet and return selected probability profiles."""
    model.eval()
    with torch.no_grad():
        # Checkpoint-compatible Parnet expects a dict and reads inputs["sequence"].
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


def _validate_sequence_batch(sequence: torch.Tensor, seq_len: int) -> None:
    """Validate Parnet sequence input shape."""
    if sequence.ndim != 3 or sequence.shape[1] != 4 or sequence.shape[2] != seq_len:
        raise ValueError(
            f"Expected sequence batch shape (B, 4, {seq_len}); got {tuple(sequence.shape)}"
        )


def _validate_parnet_profiles(
    rbp_profiles: torch.Tensor,
    *,
    batch_size: int,
    num_tracks: int,
    seq_len: int,
) -> None:
    """Validate frozen Parnet profile output shape."""
    expected = (batch_size, num_tracks, seq_len)
    if tuple(rbp_profiles.shape) != expected:
        raise ValueError(
            f"Expected Parnet profile output shape {expected}; got {tuple(rbp_profiles.shape)}"
        )


def _validate_reconstruction_output(
    reconstructed_profile: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
) -> None:
    """Validate reconstruction head output shape."""
    expected = (batch_size, seq_len)
    if tuple(reconstructed_profile.shape) != expected:
        raise ValueError(
            f"Expected reconstructed globalCLIP profile shape {expected}; "
            f"got {tuple(reconstructed_profile.shape)}"
        )


def _target_to_2d(target: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Convert a target tensor to shape ``(B, seq_len)``."""
    target = target.float()
    if target.ndim == 3:
        if target.shape[1] != 1 or target.shape[2] != seq_len:
            raise ValueError(
                f"Expected target shape (B, 1, {seq_len}); got {tuple(target.shape)}"
            )
        return target[:, 0, :]
    if target.ndim == 2:
        if target.shape[1] != seq_len:
            raise ValueError(
                f"Expected target shape (B, {seq_len}); got {tuple(target.shape)}"
            )
        return target
    raise ValueError(f"Expected target shape (B, L) or (B, 1, L), got {tuple(target.shape)}")


def _target_to_probability_and_mask(
    target: torch.Tensor,
    seq_len: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize observed globalCLIP counts and mark nonzero windows."""
    target_2d = _target_to_2d(target, seq_len)
    target_sums = target_2d.sum(dim=-1)
    valid_mask = target_sums > eps
    target_profile = torch.zeros_like(target_2d)
    if valid_mask.any():
        target_profile[valid_mask] = (
            target_2d[valid_mask] / target_sums[valid_mask].unsqueeze(-1)
        )
    return target_profile, valid_mask, target_sums


def _target_to_probability(target: torch.Tensor, eps: float) -> torch.Tensor:
    """Backward-compatible target normalization helper."""
    seq_len = target.shape[-1]
    target_profile, _, _ = _target_to_probability_and_mask(target, seq_len, eps)
    return target_profile


def _profile_cross_entropy(
    reconstructed_profile: torch.Tensor,
    target_profile: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return mean profile cross entropy across rows in a valid batch."""
    return -(target_profile * reconstructed_profile.clamp_min(eps).log()).sum(dim=-1).mean()


def _profile_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean per-window MSE."""
    return pred.sub(target).square().mean(dim=-1).mean()


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
    """Return True if frozen Parnet received gradients."""
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
) -> dict[str, Any]:
    """Run one training or validation epoch."""
    is_train = optimizer is not None
    head.train(is_train)
    loss_sum = 0.0
    valid_windows = 0
    zero_signal_windows = 0
    pearsons: list[float] = []
    parnet_output_shape: tuple[int, ...] | None = None
    reconstruction_output_shape: tuple[int, ...] | None = None

    for batch_indices in batches:
        batch = _make_batch(
            split_data,
            batch_indices,
            args.seq_length,
            args.globalclip_key,
        )
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
        parnet_output_shape = tuple(rbp_profiles.shape)

        reconstructed_profile, _ = head(rbp_profiles)
        _validate_reconstruction_output(
            reconstructed_profile,
            batch_size=sequence.shape[0],
            seq_len=args.seq_length,
        )
        reconstruction_output_shape = tuple(reconstructed_profile.shape)

        target_profile, valid_mask, _ = _target_to_probability_and_mask(
            target,
            args.seq_length,
            eps,
        )
        batch_zero_signal = int((~valid_mask).sum().item())
        batch_valid = int(valid_mask.sum().item())
        zero_signal_windows += batch_zero_signal
        if batch_valid == 0:
            continue

        valid_pred = reconstructed_profile[valid_mask]
        valid_target = target_profile[valid_mask]
        loss = _profile_cross_entropy(valid_pred, valid_target, eps)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if _parnet_has_grad(parnet_model):
                raise RuntimeError("Parnet is frozen but received gradients.")
            optimizer.step()

        loss_sum += loss.detach().item() * batch_valid
        valid_windows += batch_valid
        if not is_train:
            pearsons.extend(_batch_pearson(valid_pred, valid_target, eps))

    mean_loss = loss_sum / valid_windows if valid_windows else float("nan")
    mean_pearson = float(sum(pearsons) / len(pearsons)) if pearsons else float("nan")
    return {
        "loss": mean_loss,
        "pearson": mean_pearson,
        "valid_windows": valid_windows,
        "zero_signal_windows": zero_signal_windows,
        "batches": len(batches),
        "parnet_output_shape": list(parnet_output_shape) if parnet_output_shape else None,
        "reconstruction_output_shape": (
            list(reconstruction_output_shape) if reconstruction_output_shape else None
        ),
    }


def _write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write per-epoch training metrics."""
    fieldnames = [
        "epoch",
        "train_loss",
        "valid_loss",
        "valid_pearson",
        "train_valid_windows",
        "train_zero_signal_windows",
        "valid_valid_windows",
        "valid_zero_signal_windows",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalise_column_name(name: str) -> str:
    """Normalize a metadata column name for fuzzy matching."""
    return "".join(character.lower() for character in name if character.isalnum())


def _find_metadata_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find a metadata column by common aliases."""
    normalized_to_original = {
        _normalise_column_name(fieldname): fieldname for fieldname in fieldnames
    }
    for candidate in candidates:
        normalized = _normalise_column_name(candidate)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]
    for fieldname in fieldnames:
        normalized_field = _normalise_column_name(fieldname)
        if any(
            len(_normalise_column_name(candidate)) >= 4
            and _normalise_column_name(candidate) in normalized_field
            for candidate in candidates
        ):
            return fieldname
    return None


def _load_metadata_rows(
    metadata_path: Path | None,
    num_tracks: int,
) -> tuple[list[dict[str, str]] | None, list[str]]:
    """Load full RBP metadata rows, if available."""
    if metadata_path is None:
        print("Warning: metadata.full_rbp_set is not configured; saving track-index weights only.")
        return None, []
    if not metadata_path.exists():
        print(f"Warning: RBP metadata file not found: {metadata_path}")
        return None, []

    delimiter = "\t" if metadata_path.suffix.lower() in {".tsv", ".txt"} else ","
    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])

    if len(rows) < num_tracks:
        print(
            f"Warning: metadata file has {len(rows)} rows but reconstruction head has "
            f"{num_tracks} tracks. Missing tracks will keep track_index only."
        )
    elif len(rows) > num_tracks:
        print(
            f"Warning: metadata file has {len(rows)} rows but reconstruction head has "
            f"{num_tracks} tracks. Extra rows are ignored."
        )
    return rows, fieldnames


def _derive_from_track_name(track_name: str, part: str) -> str:
    """Derive RBP or cell line from common RBP_CellLine track names."""
    if "_" not in track_name:
        return ""
    rbp_name, cell_line = track_name.rsplit("_", 1)
    return rbp_name if part == "rbp" else cell_line


def _metadata_value(row: dict[str, str], column: str | None) -> str:
    """Return a stripped metadata value."""
    if column is None:
        return ""
    return str(row.get(column, "")).strip()


def _build_weight_rows(
    head: GlobalCLIPReconstructionHead,
    metadata_path: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build sorted track-level and RBP-aggregated reconstruction weight rows."""
    logits = head.weight_logits.detach().cpu()
    weights = torch.softmax(logits, dim=0)
    metadata_rows, metadata_fieldnames = _load_metadata_rows(metadata_path, weights.numel())
    rbp_column = _find_metadata_column(metadata_fieldnames, _RBP_COLUMN_CANDIDATES)
    cell_line_column = _find_metadata_column(metadata_fieldnames, _CELL_LINE_COLUMN_CANDIDATES)
    track_column = _find_metadata_column(metadata_fieldnames, _TRACK_COLUMN_CANDIDATES)
    if metadata_rows is not None:
        print(
            "Metadata columns used for weight export: "
            f"rbp={rbp_column}, cell_line={cell_line_column}, track={track_column}"
        )

    track_rows: list[dict[str, Any]] = []
    for track_index in range(weights.numel()):
        metadata_row = metadata_rows[track_index] if metadata_rows and track_index < len(metadata_rows) else {}
        track_name = _metadata_value(metadata_row, track_column)
        rbp_name = _metadata_value(metadata_row, rbp_column)
        if not rbp_name and track_name:
            rbp_name = _derive_from_track_name(track_name, "rbp")
        if not rbp_name:
            rbp_name = f"track_{track_index}"

        cell_line = _metadata_value(metadata_row, cell_line_column)
        if not cell_line and track_name:
            cell_line = _derive_from_track_name(track_name, "cell_line")

        row: dict[str, Any] = {
            "track_index": track_index,
            "weight": float(weights[track_index]),
            "weight_logit": float(logits[track_index]),
            "rbp_name": rbp_name,
            "cell_line": cell_line,
            "experiment_id_or_track_name": track_name,
        }
        for fieldname in metadata_fieldnames:
            metadata_key = f"metadata_{fieldname}"
            row[metadata_key] = metadata_row.get(fieldname, "")
        track_rows.append(row)

    track_rows = sorted(track_rows, key=lambda row: row["weight"], reverse=True)
    for rank, row in enumerate(track_rows, start=1):
        row["rank"] = rank

    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rbp_name": "",
            "weight": 0.0,
            "n_tracks": 0,
            "track_indices": [],
            "cell_lines": set(),
            "top_track_index": None,
            "top_track_weight": 0.0,
            "top_track_name": "",
        }
    )
    for row in track_rows:
        rbp_name = str(row["rbp_name"])
        group = grouped[rbp_name]
        group["rbp_name"] = rbp_name
        group["weight"] += float(row["weight"])
        group["n_tracks"] += 1
        group["track_indices"].append(str(row["track_index"]))
        if row.get("cell_line"):
            group["cell_lines"].add(str(row["cell_line"]))
        if float(row["weight"]) > float(group["top_track_weight"]):
            group["top_track_weight"] = float(row["weight"])
            group["top_track_index"] = row["track_index"]
            group["top_track_name"] = row.get("experiment_id_or_track_name", "")

    rbp_rows = []
    for group in grouped.values():
        rbp_rows.append(
            {
                "rbp_name": group["rbp_name"],
                "weight": group["weight"],
                "n_tracks": group["n_tracks"],
                "track_indices": ";".join(group["track_indices"]),
                "cell_lines": ";".join(sorted(group["cell_lines"])),
                "top_track_index": group["top_track_index"],
                "top_track_weight": group["top_track_weight"],
                "top_track_name": group["top_track_name"],
            }
        )
    rbp_rows = sorted(rbp_rows, key=lambda row: row["weight"], reverse=True)
    for rank, row in enumerate(rbp_rows, start=1):
        row["rank"] = rank

    return track_rows, rbp_rows


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with a stable header."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_weight_bars(
    output_dir: Path,
    track_rows: list[dict[str, Any]],
    rbp_rows: list[dict[str, Any]],
    *,
    top_n: int = 20,
) -> None:
    """Save top-weight bar plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Warning: matplotlib is not available; skipping weight bar plots.")
        return

    def _label_track(row: dict[str, Any]) -> str:
        if row.get("experiment_id_or_track_name"):
            return str(row["experiment_id_or_track_name"])
        if row.get("cell_line"):
            return f"{row['rbp_name']} ({row['cell_line']})"
        return f"{row['rbp_name']} [{row['track_index']}]"

    for rows, label_key, title, filename in (
        (
            track_rows[:top_n],
            _label_track,
            "Top reconstruction weights: track level",
            "top20_track_level_weights",
        ),
        (
            rbp_rows[:top_n],
            lambda row: str(row["rbp_name"]),
            "Top reconstruction weights: RBP aggregated",
            "top20_rbp_aggregated_weights",
        ),
    ):
        if not rows:
            continue
        plot_rows = list(reversed(rows))
        labels = [label_key(row) for row in plot_rows]
        values = [float(row["weight"]) for row in plot_rows]
        height = max(4.0, 0.32 * len(plot_rows) + 1.5)
        fig, ax = plt.subplots(figsize=(9, height))
        ax.barh(labels, values, color="#4c78a8")
        ax.set_xlabel("softmax reconstruction weight")
        ax.set_title(title)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{filename}.{suffix}", dpi=200)
        plt.close(fig)


def _write_weights(
    output_dir: Path,
    head: GlobalCLIPReconstructionHead,
    metadata_path: Path | None,
    *,
    write_plots: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Write metadata-aware reconstruction weights."""
    track_rows, rbp_rows = _build_weight_rows(head, metadata_path)
    track_fieldnames = [
        "rank",
        "track_index",
        "weight",
        "weight_logit",
        "rbp_name",
        "cell_line",
        "experiment_id_or_track_name",
    ]
    metadata_fieldnames = sorted(
        {
            key
            for row in track_rows
            for key in row
            if key.startswith("metadata_") and key not in track_fieldnames
        }
    )
    track_fieldnames.extend(metadata_fieldnames)
    rbp_fieldnames = [
        "rank",
        "rbp_name",
        "weight",
        "n_tracks",
        "track_indices",
        "cell_lines",
        "top_track_index",
        "top_track_weight",
        "top_track_name",
    ]

    _write_csv_rows(output_dir / "reconstruction_weights_track_level.csv", track_rows, track_fieldnames)
    _write_csv_rows(output_dir / "reconstruction_weights_rbp_aggregated.csv", rbp_rows, rbp_fieldnames)

    # Keep legacy filenames for users who already scripted against them.
    _write_csv_rows(output_dir / "rbp_reconstruction_weights.csv", track_rows, track_fieldnames)
    _write_csv_rows(output_dir / "top_reconstruction_weights.csv", track_rows[:30], track_fieldnames)

    if write_plots:
        _plot_weight_bars(output_dir, track_rows, rbp_rows)
    return track_rows, rbp_rows


def _save_head(
    output_dir: Path,
    head: GlobalCLIPReconstructionHead,
    args: argparse.Namespace,
    final_metrics: dict[str, float | int],
    run_metadata: dict[str, Any],
) -> None:
    """Save the trained reconstruction head and metadata."""
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
            "condition": args.condition_label,
            "controls_used": False,
            "control_handling": _CONTROL_HANDLING,
            "final_metrics": final_metrics,
            "run_metadata": run_metadata,
        },
    }
    torch.save(payload, output_dir / "reconstruction_head.pt")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _base_run_metadata(
    args: argparse.Namespace,
    *,
    data_path: Path,
    model_path: Path,
    metadata_path: Path | None,
    output_dir: Path,
    device: torch.device,
    split_sizes: dict[str, int],
    available_output_keys: list[str],
) -> dict[str, Any]:
    """Build metadata shared by a training run."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(_resolve_path(args.config)),
        "model_checkpoint_path": str(model_path),
        "reconstruction_head_checkpoint_path": str(output_dir / "reconstruction_head.pt"),
        "globalclip_path": str(data_path),
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "output_dir": str(output_dir),
        "condition": args.condition_label,
        "target_key": args.globalclip_key,
        "profile_key": args.profile_key,
        "controls_used": False,
        "control_key": args.control_key,
        "control_handling": _CONTROL_HANDLING,
        "available_output_keys": available_output_keys,
        "loss_type": "profile_cross_entropy_on_nonzero_globalclip_windows",
        "split_sizes": split_sizes,
        "random_seed": args.seed,
        "device": str(device),
        "parnet_frozen": True,
        "parnet_output_shape": None,
        "reconstruction_output_shape": None,
        "reconstruction_head_type": "GlobalCLIPReconstructionHead_softmax_track_mixture",
        "num_parnet_tracks": args.num_parnet_tracks,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_train_batches": args.max_train_batches,
        "max_valid_batches": args.max_valid_batches,
    }


def _write_run_config(
    output_dir: Path,
    run_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write run metadata and a compatibility run_config.json."""
    config = vars(args).copy()
    config.update(run_metadata)
    _write_json(output_dir / "run_metadata.json", run_metadata)
    _write_json(output_dir / "run_config.json", config)


def _collect_profile_examples(
    *,
    split_data: list[dict[str, Any]],
    parnet_model: torch.nn.Module,
    head: GlobalCLIPReconstructionHead,
    args: argparse.Namespace,
    device: torch.device,
    eps: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Collect valid observed/reconstructed profiles for plotting."""
    candidate_count = min(len(split_data), max_candidates)
    if candidate_count <= 0:
        return []
    examples = []
    batches = _batch_indices(
        candidate_count,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        max_batches=None,
    )
    head.eval()
    with torch.no_grad():
        for batch_indices in batches:
            batch = _make_batch(split_data, batch_indices, args.seq_length, args.globalclip_key)
            sequence = batch["inputs"]["sequence"].float().to(device)
            target = batch["outputs"]["total"].float().to(device)
            rbp_profiles = _extract_parnet_profiles(parnet_model, sequence, args.profile_key)
            reconstructed_profile, _ = head(rbp_profiles)
            target_profile, valid_mask, target_sums = _target_to_probability_and_mask(
                target,
                args.seq_length,
                eps,
            )
            for local_index, sample_index in enumerate(batch_indices):
                if not bool(valid_mask[local_index].item()):
                    continue
                examples.append(
                    {
                        "sample_index": sample_index,
                        "target_sum": float(target_sums[local_index].detach().cpu()),
                        "observed": target_profile[local_index].detach().cpu(),
                        "reconstructed": reconstructed_profile[local_index].detach().cpu(),
                    }
                )
    return examples


def _plot_profile_examples(
    examples: list[dict[str, Any]],
    output_base: Path,
    title: str,
    *,
    seq_len: int,
) -> None:
    """Save observed-vs-reconstructed profile grids."""
    if not examples:
        print(f"Warning: no valid examples for {output_base.name}; skipping profile plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Warning: matplotlib is not available; skipping profile plots.")
        return

    n_cols = 2
    n_rows = (len(examples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(3.0, 2.6 * n_rows)), squeeze=False)
    positions = list(range(seq_len))
    for ax, example in zip(axes.ravel(), examples, strict=False):
        observed = example["observed"].numpy()
        reconstructed = example["reconstructed"].numpy()
        ax.plot(positions, observed, label="observed globalCLIP", color="#d55e00", linewidth=1.4)
        ax.plot(positions, reconstructed, label="reconstructed", color="#0072b2", linewidth=1.2)
        ax.set_title(
            f"sample {example['sample_index']} | total signal {example['target_sum']:.1f}",
            fontsize=9,
        )
        ax.set_xlim(0, seq_len - 1)
        ax.set_ylabel("profile probability")
    for ax in axes.ravel()[len(examples) :]:
        ax.axis("off")
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def _write_diagnostic_profile_plots(
    *,
    output_dir: Path,
    split_name: str,
    split_data: list[dict[str, Any]],
    parnet_model: torch.nn.Module,
    head: GlobalCLIPReconstructionHead,
    args: argparse.Namespace,
    device: torch.device,
    eps: float,
) -> None:
    """Save top-signal and random observed-vs-reconstructed profile plots."""
    if args.no_plots:
        return
    examples = _collect_profile_examples(
        split_data=split_data,
        parnet_model=parnet_model,
        head=head,
        args=args,
        device=device,
        eps=eps,
        max_candidates=args.max_plot_candidates,
    )
    if not examples:
        print(f"Warning: no nonzero {split_name} examples available for diagnostic plots.")
        return

    top_examples = sorted(examples, key=lambda example: example["target_sum"], reverse=True)[
        : args.plot_top_n
    ]
    rng = random.Random(args.seed)
    random_examples = examples[:]
    rng.shuffle(random_examples)
    random_examples = random_examples[: args.plot_random_n]

    _plot_profile_examples(
        top_examples,
        output_dir / f"{split_name}_top_signal_observed_vs_reconstructed",
        f"{split_name}: top signal windows",
        seq_len=args.seq_length,
    )
    _plot_profile_examples(
        random_examples,
        output_dir / f"{split_name}_random_observed_vs_reconstructed",
        f"{split_name}: random nonzero windows",
        seq_len=args.seq_length,
    )


def _log_control_handling(available_output_keys: list[str], control_key: str) -> None:
    """Log current control handling policy."""
    print("Controls used: false")
    print(f"Control handling: {_CONTROL_HANDLING}")
    print(f"Available output keys: {available_output_keys}")
    if control_key in available_output_keys:
        print(f"Control key {control_key!r} is present but intentionally ignored for Idea 1 v1.")
    else:
        print(f"Control key {control_key!r} was not present in the first sample outputs.")


def _train_one_condition(
    args: argparse.Namespace,
    *,
    data_path: Path,
    model_path: Path,
    metadata_path: Path | None,
    output_dir: Path,
) -> None:
    """Train and save one condition-specific reconstruction head."""
    if not data_path.exists():
        raise FileNotFoundError(f"globalCLIP file not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"pretrained model not found: {model_path}")

    _validate_target_key(args.globalclip_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Condition: {args.condition_label}")
    print(f"Device: {device}")
    print(f"Loading globalCLIP data: {data_path}")
    data = _load_pt_or_ptgz(data_path)
    if "train" not in data:
        raise KeyError(f"Train split not found. Available keys: {list(data.keys())}")
    train_data = data["train"]
    valid_split_name, valid_data = _get_valid_split(data)
    available_output_keys = _available_output_keys(train_data)
    _log_control_handling(available_output_keys, args.control_key)
    print(f"Train samples: {len(train_data)}")
    print(f"Validation split: {valid_split_name} ({len(valid_data)} samples)")

    print(f"Loading frozen Parnet model: {model_path}")
    parnet_model = _load_pretrained_parnet(model_path, args.model_name, device)
    _assert_parnet_frozen(parnet_model)
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
    eps = args.eps

    run_metadata = _base_run_metadata(
        args,
        data_path=data_path,
        model_path=model_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        device=device,
        split_sizes=_split_sizes(data),
        available_output_keys=available_output_keys,
    )
    _write_run_config(output_dir, run_metadata, args)

    metrics_rows: list[dict[str, float | int]] = []
    last_row: dict[str, float | int] = {}
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

        train_stats = _run_epoch(
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
            valid_stats = _run_epoch(
                split_data=valid_data,
                batches=valid_batches,
                parnet_model=parnet_model,
                head=head,
                optimizer=None,
                args=args,
                device=device,
                eps=eps,
            )

        run_metadata["parnet_output_shape"] = (
            train_stats["parnet_output_shape"] or valid_stats["parnet_output_shape"]
        )
        run_metadata["reconstruction_output_shape"] = (
            train_stats["reconstruction_output_shape"]
            or valid_stats["reconstruction_output_shape"]
        )
        run_metadata["train_batches_per_epoch"] = train_stats["batches"]
        run_metadata["valid_batches_per_epoch"] = valid_stats["batches"]
        run_metadata["last_train_zero_signal_windows"] = train_stats["zero_signal_windows"]
        run_metadata["last_valid_zero_signal_windows"] = valid_stats["zero_signal_windows"]

        last_row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "valid_loss": valid_stats["loss"],
            "valid_pearson": valid_stats["pearson"],
            "train_valid_windows": train_stats["valid_windows"],
            "train_zero_signal_windows": train_stats["zero_signal_windows"],
            "valid_valid_windows": valid_stats["valid_windows"],
            "valid_zero_signal_windows": valid_stats["zero_signal_windows"],
        }
        metrics_rows.append(last_row)
        print(
            f"epoch={epoch} train_loss={train_stats['loss']:.6f} "
            f"valid_loss={valid_stats['loss']:.6f} "
            f"valid_pearson={valid_stats['pearson']:.6f} "
            f"train_zero={train_stats['zero_signal_windows']} "
            f"valid_zero={valid_stats['zero_signal_windows']}"
        )

        _write_metrics(output_dir / "training_metrics.csv", metrics_rows)
        _write_weights(output_dir, head, metadata_path, write_plots=not args.no_plots)
        run_metadata["final_metrics"] = last_row
        _write_run_config(output_dir, run_metadata, args)
        _save_head(output_dir, head, args, last_row, run_metadata)

    _write_diagnostic_profile_plots(
        output_dir=output_dir,
        split_name=valid_split_name,
        split_data=valid_data,
        parnet_model=parnet_model,
        head=head,
        args=args,
        device=device,
        eps=eps,
    )
    print(f"Saved outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train_globalclip_reconstruction_head.py \\\n"
            "    --condition interphase --config config/filepaths.lambosaur-ms-01-2.yaml \\\n"
            "    --output-dir runs/idea1 --epochs 5 --batch-size 16\n\n"
            "  python scripts/train_globalclip_reconstruction_head.py \\\n"
            "    --condition both --config config/filepaths.lambosaur-ms-01-2.yaml \\\n"
            "    --output-dir runs/idea1 --epochs 5 --batch-size 16\n"
        ),
    )
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--condition", choices=("interphase", "lysate", "both"), default=None)
    parser.add_argument("--globalclip-path", type=Path, default=None)
    parser.add_argument("--pretrained-model-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
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
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--plot-top-n", type=int, default=6)
    parser.add_argument("--plot-random-n", type=int, default=6)
    parser.add_argument("--max-plot-candidates", type=int, default=512)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run condition-aware training."""
    args = parse_args()
    _set_random_seed(args.seed)
    config = _read_yaml(args.config)
    conditions = _requested_conditions(args)
    model_path = _resolve_pretrained_model_path(args, config)
    metadata_path = _resolve_metadata_path(args, config)
    base_output_dir = _resolve_path(args.output_dir)
    append_condition = args.condition is not None

    for condition in conditions:
        data_path = _resolve_globalclip_path_for_condition(args, config, condition)
        output_dir = _condition_output_dir(base_output_dir, condition, append_condition)
        condition_args = argparse.Namespace(**vars(args))
        condition_args.condition_label = condition
        condition_args.condition = condition
        condition_args.globalclip_path = data_path
        condition_args.pretrained_model_path = model_path
        condition_args.metadata_path = metadata_path
        condition_args.output_dir = output_dir
        _train_one_condition(
            condition_args,
            data_path=data_path,
            model_path=model_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
