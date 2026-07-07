"""Dataset for loading GlobalCLIP .pt.gz files.

The data files use the ParnetDataElement format from parnet_demo_utils:
  elem["inputs"]["sequence"]       str, RNA/DNA sequence of length seq_len
  elem["outputs"]["globalCLIP"]    SparseTensorDict  (1, seq_len) read counts
  elem["outputs"]["control"]       SparseTensorDict  (1, seq_len) control counts

Each file is split into "train", "valid", "test" keys.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import torch
import torch.utils.data

try:
    from parnet_demo_utils.sparse_utils import torch_sparse_to_dense
except ImportError:
    # Fallback: minimal implementation for SparseTensorDict COO format
    def torch_sparse_to_dense(sparse_dict: dict) -> torch.Tensor:
        """Convert a SparseTensorDict (COO) to a dense torch.Tensor."""
        return torch.sparse_coo_tensor(
            sparse_dict["indices"], sparse_dict["values"], sparse_dict["size"]
        ).to_dense()


_BASE_MAP: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def _seq_to_onehot(seq: str, seq_len: int) -> torch.Tensor:
    """Convert a DNA/RNA string to a one-hot tensor (4, seq_len)."""
    onehot = torch.zeros(4, seq_len, dtype=torch.float32)
    for i, c in enumerate(seq[:seq_len]):
        idx = _BASE_MAP.get(c.upper(), -1)
        if idx >= 0:
            onehot[idx, i] = 1.0
    return onehot


class GlobalCLIPDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for GlobalCLIP signal prediction.

    Loads a .pt or .pt.gz file in PARNET ParnetDataElement format and returns
    dicts with float tensors ready for the GlobalCLIP models.

    Args:
        pt_path:          Path to the .pt or .pt.gz data file.
        split:            Dataset split: "train", "valid", or "test".
        seq_len:           Sequence length to use (default 600).
        total_key:         Key for the target signal in elem["outputs"] (default "globalCLIP").
        max_total_signal: If set, drop windows whose total signal (summed
                          over the whole 600nt window) exceeds this value.
                          Matches the outlier-capping preprocessing used for
                          the "filtered" dataset comparison in Idea 2 (cap of
                          1000, implemented by Lukas) -- used here so Idea 1
                          and Idea 2 can be compared on the identical dataset.
    """

    def __init__(
        self,
        pt_path: str | Path,
        split: str,
        seq_len: int = 600,
        total_key: str = "globalCLIP",
        max_total_signal: float | None = None,
    ):
        pt_path = Path(pt_path)
        if not pt_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {pt_path}")

        print(f"Loading {'(gz) ' if pt_path.suffix == '.gz' else ''}{pt_path.name} split='{split}'...", end=" ", flush=True)

        if pt_path.suffix == ".gz":
            with gzip.open(pt_path, "rb") as f:
                data = torch.load(io.BytesIO(f.read()), weights_only=False)
        else:
            data = torch.load(pt_path, mmap=True, weights_only=False)

        if split not in data:
            raise KeyError(
                f"Split '{split}' not found. Available splits: {list(data.keys())}"
            )

        self.samples = data[split]
        self.seq_len = seq_len
        self.total_key = total_key
        print(f"loaded {len(self.samples)} samples.")

        if max_total_signal is not None:
            n_before = len(self.samples)
            self.samples = [
                elem for elem in self.samples
                if float(elem["outputs"][total_key]["values"].sum()) <= max_total_signal
            ]
            n_removed = n_before - len(self.samples)
            print(
                f"Filtered {n_removed} of {n_before} '{split}' windows with "
                f"total signal > {max_total_signal} ({len(self.samples)} remaining)."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        elem = self.samples[idx]

        seq_onehot = _seq_to_onehot(elem["inputs"]["sequence"], self.seq_len)
        signal = torch_sparse_to_dense(elem["outputs"][self.total_key]).float().clone()    # (1, L)
        # control = torch_sparse_to_dense(elem["outputs"]["control"]).float()             # (1, L) — not used

        if signal.shape[-1] != self.seq_len:
            # Some (rare) windows near chromosome/contig boundaries have a
            # shorter recorded signal length than seq_len. Pad/crop to a
            # fixed length so DataLoader batching never sees mismatched
            # tensor sizes, regardless of which samples end up in the same
            # batch (e.g. after outlier filtering changes batch composition).
            fixed = torch.zeros(signal.shape[0], self.seq_len, dtype=signal.dtype)
            n = min(signal.shape[-1], self.seq_len)
            fixed[:, :n] = signal[:, :n]
            signal = fixed

        return {
            "sequence": seq_onehot,   # (4, L)
            "signal": signal,         # (1, L)
            # "control": control,     # (1, L) — not used
        }


def build_dataloaders(
    pt_path: str | Path,
    batch_size: int = 64,
    seq_len: int = 600,
    total_key: str = "globalCLIP",
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Build train / valid / test DataLoaders from a single pt.gz file.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_ds = GlobalCLIPDataset(pt_path, "train", seq_len, total_key)
    val_ds   = GlobalCLIPDataset(pt_path, "valid", seq_len, total_key)
    test_ds  = GlobalCLIPDataset(pt_path, "test",  seq_len, total_key)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
