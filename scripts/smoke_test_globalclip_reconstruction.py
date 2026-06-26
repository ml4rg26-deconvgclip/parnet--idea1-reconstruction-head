#!/usr/bin/env python
"""Smoke test for the Idea 1 globalCLIP reconstruction baseline.

This script tries to use the repository's configured globalCLIP dataset and
pretrained Parnet checkpoint. If either optional resource is unavailable, it
falls back to synthetic tensors so the reconstruction head and gradient checks
can still run.
"""

from __future__ import annotations

import argparse
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
        from parnet_additional_utils import GzListDataset
    except ModuleNotFoundError as exc:
        if args.strict_real_data:
            raise
        print(f"parnet_additional_utils unavailable ({exc}); using a synthetic batch.")
        return _synthetic_batch(args.batch_size, args.seq_length), dataset_path, False

    dataset = GzListDataset(
        dataset_path,
        split=args.split,
        length=args.seq_length,
        total_key=args.globalclip_key,
        control_key=args.control_key,
        shuffle=False,
        return_meta=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    batch = next(iter(loader))
    print(f"Loaded real globalCLIP batch from: {dataset_path}")
    print(f"  split={args.split}, dataset_size={len(dataset)}")
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

    try:
        from parnet_additional_utils import ParnetModelName, load_parnet_model
    except ModuleNotFoundError as exc:
        if args.strict_real_data:
            raise
        print(f"parnet_additional_utils unavailable ({exc}); using synthetic profiles.")
        return None, model_path

    model_name = ParnetModelName(args.model_name)
    model = load_parnet_model(
        model_name,
        model_path,
        dtype=torch.float32,
        device=device,
    )
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
        output = model(sequence)

    print("Parnet output shapes:")
    for key, value in output.items():
        if torch.is_tensor(value):
            print(f"  {key}: {tuple(value.shape)}")
        else:
            print(f"  {key}: {type(value).__name__}")

    if args.profile_key not in output:
        raise KeyError(
            f"Parnet output does not contain {args.profile_key!r}; "
            f"available keys: {list(output.keys())}"
        )

    profile_logprob = output[args.profile_key]
    if profile_logprob.ndim != 3:
        raise ValueError(
            f"Expected Parnet {args.profile_key!r} output to be 3D, "
            f"got {tuple(profile_logprob.shape)}"
        )

    profile_prob = profile_logprob.exp()
    sums = profile_prob.sum(dim=-1)
    print(f"Using Parnet output: out[{args.profile_key!r}].exp()")
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
