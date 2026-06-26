"""Small reconstruction head for globalCLIP deconvolution experiments."""

from __future__ import annotations

import torch
from torch import nn


class GlobalCLIPReconstructionHead(nn.Module):
    """Reconstruct one globalCLIP profile from pretrained Parnet RBP profiles.

    The input is expected to be a batch of probability profiles with shape
    ``(batch, num_tracks, seq_len)``. For the current Parnet setup this is
    typically ``(batch, 223, 600)`` after converting ``out["total"]`` from
    log-probabilities with ``.exp()``.
    """

    def __init__(
        self,
        num_tracks: int = 223,
        seq_len: int | None = 600,
        *,
        normalize_reconstructed_profile: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """Initialise trainable non-negative mixture weights.

        Args:
            num_tracks: Number of Parnet RBP-cell-line tracks to combine.
            seq_len: Expected sequence length. Pass ``None`` to allow any length.
            normalize_reconstructed_profile: Renormalise the reconstructed
                profile along sequence length. Keep this enabled when inputs are
                probability distributions.
            eps: Small denominator clamp for sequence-length normalisation.
        """
        super().__init__()
        self.num_tracks = num_tracks
        self.seq_len = seq_len
        self.normalize_reconstructed_profile = normalize_reconstructed_profile
        self.eps = eps
        self.weight_logits = nn.Parameter(torch.zeros(num_tracks))

    def forward(self, rbp_profiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine RBP profiles into one reconstructed globalCLIP profile.

        Args:
            rbp_profiles: Tensor with shape ``(batch, num_tracks, seq_len)``.

        Returns:
            ``(reconstructed_profile, weights)`` where reconstructed_profile has
            shape ``(batch, seq_len)`` and weights has shape ``(num_tracks,)``.
        """
        if rbp_profiles.ndim != 3:
            raise ValueError(
                "rbp_profiles must have shape "
                f"(batch, num_tracks, seq_len); got {tuple(rbp_profiles.shape)}"
            )
        if rbp_profiles.shape[1] != self.num_tracks:
            raise ValueError(
                f"expected {self.num_tracks} tracks, got {rbp_profiles.shape[1]}"
            )
        if self.seq_len is not None and rbp_profiles.shape[2] != self.seq_len:
            raise ValueError(
                f"expected sequence length {self.seq_len}, got {rbp_profiles.shape[2]}"
            )

        weights = torch.softmax(self.weight_logits, dim=0)
        reconstructed_profile = torch.einsum("btl,t->bl", rbp_profiles, weights)

        if self.normalize_reconstructed_profile:
            denom = reconstructed_profile.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            reconstructed_profile = reconstructed_profile / denom

        return reconstructed_profile, weights

    def extra_repr(self) -> str:
        """Return concise module settings for ``print(module)``."""
        return (
            f"num_tracks={self.num_tracks}, seq_len={self.seq_len}, "
            f"normalize_reconstructed_profile={self.normalize_reconstructed_profile}"
        )
