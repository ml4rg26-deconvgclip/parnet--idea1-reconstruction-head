#!/usr/bin/env python
"""Inspect one sample from a globalCLIP .pt/.pt.gz dataset file.

The script prints dataset structure and signal shapes only. It does not copy
data into the repository and does not load the Parnet model unless
``--check-model-load`` is explicitly passed. ``--model-path`` is accepted so the
data/model pair used for a smoke run can be checked together.
"""

from __future__ import annotations

import argparse
import gzip
import traceback
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyTorch is required to inspect .pt/.pt.gz files.") from exc


def _load_pt_or_ptgz(path: Path) -> dict[str, Any]:
    """Load a torch dataset file, preferring a mmap-able companion .pt file."""
    if path.name.endswith(".pt.gz"):
        companion_pt = Path(str(path)[:-3])
        if companion_pt.exists():
            print(f"Loading companion .pt with mmap=True: {companion_pt}")
            return torch.load(companion_pt, mmap=True, weights_only=False)
        print(f"Loading gzip torch file: {path}")
        # A monolithic torch .pt.gz cannot be partially deserialised with
        # torch.load; after loading the container, this script inspects only one
        # sample and does not materialise extra tensors.
        with gzip.open(path, "rb") as handle:
            return torch.load(handle, weights_only=False)

    print(f"Loading torch file with mmap=True: {path}")
    return torch.load(path, mmap=True, weights_only=False)


def _is_split_value(value: Any) -> bool:
    """Return True for list-like split payloads."""
    return isinstance(value, (list, tuple))


def _shape_of(value: Any) -> str:
    """Return a compact shape/size description for tensors, strings, and sparse dicts."""
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, str):
        return f"len={len(value)}"
    if isinstance(value, bytes):
        return f"len={len(value)} bytes"
    if isinstance(value, dict):
        if "size" in value:
            return f"sparse_size={tuple(value['size'])}"
        return f"dict_keys={list(value.keys())}"
    if hasattr(value, "shape"):
        return str(tuple(value.shape))
    if hasattr(value, "__len__"):
        return f"len={len(value)}"
    return "shape=unknown"


def _select_split(data: dict[str, Any], requested_split: str | None) -> str:
    split_names = [key for key, value in data.items() if _is_split_value(value)]
    if not split_names:
        raise ValueError("No list-like splits found in dataset.")
    if requested_split is not None:
        if requested_split not in split_names:
            raise KeyError(
                f"Requested split {requested_split!r} not found; available: {split_names}"
            )
        return requested_split

    for preferred in ("train", "valid", "validation", "test"):
        if preferred in split_names:
            return preferred
    return split_names[0]


def _print_split_summary(data: dict[str, Any]) -> None:
    split_names = [key for key, value in data.items() if _is_split_value(value)]
    print(f"available splits: {split_names}")
    print("number of samples per split:")
    for split_name in split_names:
        print(f"  {split_name}: {len(data[split_name])}")


def _print_sample_summary(sample: dict[str, Any]) -> None:
    inputs = sample.get("inputs", {})
    outputs = sample.get("outputs", {})
    meta = sample.get("meta", {})

    print(f"sample['inputs'].keys(): {list(inputs.keys())}")
    print(f"sample['outputs'].keys(): {list(outputs.keys())}")
    if isinstance(meta, dict):
        print(f"sample['meta'].keys(): {list(meta.keys())}")
    else:
        print(f"sample['meta']: type={type(meta).__name__}")

    sequence = inputs.get("sequence")
    print(
        "sequence type and shape: "
        f"type={type(sequence).__name__}, {_shape_of(sequence)}"
    )

    print("output signal keys and shapes:")
    for key, value in outputs.items():
        print(f"  {key}: type={type(value).__name__}, {_shape_of(value)}")


def _check_model_load(model_path: Path) -> None:
    """Try loading the model on CPU and print either its type or the full error."""
    print("checking model load with torch.load(..., map_location='cpu')")
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        print("model load failed:")
        print(traceback.format_exc())
        return

    print(f"model load succeeded: type={type(model).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--globalclip-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to inspect. Defaults to train/valid/validation/test if present.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--check-model-load",
        action="store_true",
        help="Try loading --model-path on CPU and print the model type or full error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    globalclip_path = args.globalclip_path.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()

    if not globalclip_path.exists():
        raise FileNotFoundError(f"globalCLIP file not found: {globalclip_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    print(f"globalclip path: {globalclip_path}")
    print(f"model path     : {model_path}")
    print(f"model size     : {model_path.stat().st_size:,} bytes")
    if args.check_model_load:
        _check_model_load(model_path)

    data = _load_pt_or_ptgz(globalclip_path)
    _print_split_summary(data)

    split_name = _select_split(data, args.split)
    if len(data[split_name]) == 0:
        raise ValueError(f"Split {split_name!r} is empty.")
    if args.sample_index < 0 or args.sample_index >= len(data[split_name]):
        raise IndexError(
            f"Sample index {args.sample_index} out of range for split "
            f"{split_name!r} with {len(data[split_name])} samples."
        )

    print(f"inspected split: {split_name}")
    print(f"inspected sample index: {args.sample_index}")
    _print_sample_summary(data[split_name][args.sample_index])


if __name__ == "__main__":
    main()
