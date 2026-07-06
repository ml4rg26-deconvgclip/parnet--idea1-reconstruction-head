"""GlobalCLIP prediction models.

Two architectures:
  GlobalCLIPStandardModel  -- MixCoeffHead + log_scale (best without QLayer)
  GlobalCLIPQLayerModel    -- QLayer quantum interference + CNN (experimental)

Both use a frozen PARNET backbone to extract (B, 512, L) embeddings and
(B, 223, L) per-RBP binding scores, then learn to combine those into a
single GlobalCLIP track (B, 1, L).
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ── Shared sub-modules ────────────────────────────────────────────────────────


class MixCoeffHead(nn.Module):
    """Sequence-dependent mixing coefficients.

    Mean-pools the PARNET embedding over positions, then projects to
    (B, num_tasks) sigmoid mixing weights.
    """

    def __init__(self, embed_dim: int = 512, num_tasks: int = 223, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_tasks)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (B, embed_dim, L) backbone feature map.

        Returns:
            alpha: (B, num_tasks) mixing coefficients in [0, 1].
        """
        x = embedding.mean(dim=-1)                          # (B, D)
        return torch.sigmoid(self.fc2(self.act(self.fc1(x))))  # (B, T)


class QLayer(nn.Module):
    """Quantum-inspired interference layer.

    Treats each RBP binding track as a complex wave:
        ψ_i(p) = A_i(p) · e^{i·φ_i}

    The superposition Ψ = Σ_i α_i · ψ_i gives an interference pattern:
        I(p) = |Ψ(p)|² = Σ_i A_i² + 2 Σ_{i<j} α_i·α_j·A_i·A_j·cos(φ_i−φ_j)

    Cross-terms 2·α_i·α_j·cos(φ_i−φ_j) naturally encode protein-protein
    interactions with only 223 learnable phase parameters (vs. 223² for attention).

    Interpretation:
        φ_i ≈ φ_j        → constructive interference (cooperative binding)
        φ_i ≈ φ_j + π    → destructive interference (competitive / noisy)
        Learnable φ after training on log-FE signal: proteins that behave like
        background noise will be pushed toward opposing phases.
    """

    def __init__(self, num_rbps: int = 223):
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(num_rbps))

    def forward(self, rbp_tracks: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rbp_tracks: (B, num_rbps, L) scaled RBP binding tracks.
            alpha:      (B, num_rbps) mixing coefficients.

        Returns:
            interference: (B, 1, L) |Ψ|² interference pattern.
        """
        amp = rbp_tracks * alpha[:, :, None]                       # (B, R, L)
        phi = self.phase                                            # (R,)
        real = (amp * torch.cos(phi)[None, :, None]).sum(1)        # (B, L)
        imag = (amp * torch.sin(phi)[None, :, None]).sum(1)        # (B, L)
        return (real ** 2 + imag ** 2).unsqueeze(1)                # (B, 1, L)

    def coupling_matrix(self) -> torch.Tensor:
        """Return (num_rbps, num_rbps) pairwise coupling: cos(φ_i − φ_j).

        Values > 0 mean the two proteins tend to cooperate (constructive);
        values < 0 mean they compete or anti-correlate (destructive).
        """
        phi = self.phase.detach()
        return torch.cos(phi[:, None] - phi[None, :])


# ── PARNET backbone helper ────────────────────────────────────────────────────


def _extract_parnet_features(
    parnet_model: nn.Module,
    seq_onehot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass sequence through frozen PARNET; return embedding and RBP tracks.

    Args:
        parnet_model: Pretrained PARNET (RBPNet) with 223-task head.
        seq_onehot:   (B, 4, L) one-hot encoded RNA/DNA sequence.

    Returns:
        embedding:  (B, 512, L) backbone feature map after stem + body.
        rbp_tracks: (B, 223, L) per-RBP log-probability scores from head_target.
    """
    x = parnet_model.stem(seq_onehot)           # (B, C_stem, L)
    x = parnet_model.body(x)                    # (B, 512, L)
    if hasattr(parnet_model, "projection"):
        x = parnet_model.projection(x)          # identity after load_parnet_model
    embedding = x
    rbp_tracks = parnet_model.head.head_target.pointwise_conv(x)  # (B, 223, L)
    return embedding, rbp_tracks


# ── Model 1: Standard CombiLayer (no QLayer) ─────────────────────────────────


class GlobalCLIPStandardModel(nn.Module):
    """GlobalCLIP prediction via learned weighted sum of PARNET RBP tracks.

    Architecture:
        PARNET backbone (frozen)
            → (B, 512, L) embedding  +  (B, 223, L) rbp_tracks
        MixCoeffHead
            → (B, 223) alpha          (sequence-dependent, sigmoid)
        log_scale                     (223 global amplitude corrections)
        Weighted sum
            → (B, 1, L) GlobalCLIP prediction

    The model learns:
      - *Which* proteins contribute most (global, via log_scale)
      - *How much* each protein contributes for *this* sequence (via alpha)

    Training target: log-fold-enrichment  log(1+signal) − log(1+control)
    """

    def __init__(
        self,
        parnet_model: nn.Module,
        num_rbps: int = 223,
        mix_hidden: int = 128,
    ):
        super().__init__()
        self.backbone = parnet_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.mix_coeff = MixCoeffHead(embed_dim=512, num_tasks=num_rbps, hidden=mix_hidden)
        self.log_scale = nn.Parameter(torch.zeros(num_rbps))

    def forward(
        self, seq_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_onehot: (B, 4, L) one-hot encoded sequence.

        Returns:
            pred:  (B, 1, L) predicted GlobalCLIP track.
            alpha: (B, num_rbps) mixing coefficients (for analysis / IG).
        """
        with torch.no_grad():
            embedding, rbp_tracks = _extract_parnet_features(self.backbone, seq_onehot)

        alpha = self.mix_coeff(embedding)                          # (B, 223)
        scale = self.log_scale.exp()                               # (223,)
        scaled = rbp_tracks * scale[None, :, None]                 # (B, 223, L)
        pred = (scaled * alpha[:, :, None]).sum(1, keepdim=True)   # (B, 1, L)
        return pred, alpha

    def effective_weights(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        """Effective contribution per protein: alpha * exp(log_scale).

        Returns (B, num_rbps) positive values; higher means more influence
        on the predicted GlobalCLIP signal for this sequence.
        """
        _, alpha = self.forward(seq_onehot)
        return (alpha * self.log_scale.exp()[None, :]).detach()


# ── Model 2: QLayer interference model ───────────────────────────────────────


class GlobalCLIPQLayerModel(nn.Module):
    """GlobalCLIP prediction via quantum-inspired interference + CNN refinement.

    Architecture:
        PARNET backbone (frozen)
            → (B, 512, L) embedding  +  (B, 223, L) rbp_tracks
        MixCoeffHead
            → (B, 223) alpha          (sequence-dependent)
        log_scale                     (223 global amplitude corrections)
        QLayer                        (223 learnable phases φ_i)
            → (B, 1, L) |Ψ|²         (interference pattern with cross-terms)
        Dilated CNN                   (refines local context)
            → (B, 1, L) GlobalCLIP prediction

    The QLayer cross-terms 2·α_i·α_j·cos(φ_i−φ_j) capture pairwise
    protein-protein interactions with only 223 phase parameters.

    Training target: log-fold-enrichment  log(1+signal) − log(1+control)
    Proteins behaving like background noise learn φ ≈ φ_noise + π
    (destructive interference → their contribution cancels out).
    """

    def __init__(
        self,
        parnet_model: nn.Module,
        num_rbps: int = 223,
        mix_hidden: int = 128,
        cnn_channels: int = 64,
        cnn_kernel: int = 9,
        cnn_layers: int = 3,
    ):
        super().__init__()
        self.backbone = parnet_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.mix_coeff = MixCoeffHead(embed_dim=512, num_tasks=num_rbps, hidden=mix_hidden)
        self.log_scale = nn.Parameter(torch.zeros(num_rbps))
        self.qlayer = QLayer(num_rbps=num_rbps)

        # Dilated CNN to refine the local context after interference
        layers: list[nn.Module] = []
        in_ch = 1
        for i in range(cnn_layers):
            dil = 2 ** i
            pad = dil * (cnn_kernel // 2)
            layers += [
                nn.Conv1d(in_ch, cnn_channels, kernel_size=cnn_kernel, padding=pad, dilation=dil),
                nn.ReLU(),
            ]
            in_ch = cnn_channels
        layers.append(nn.Conv1d(cnn_channels, 1, kernel_size=1))
        self.cnn = nn.Sequential(*layers)

    def forward(
        self, seq_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_onehot: (B, 4, L) one-hot encoded sequence.

        Returns:
            pred:  (B, 1, L) predicted GlobalCLIP track.
            alpha: (B, num_rbps) mixing coefficients (for analysis / IG).
        """
        with torch.no_grad():
            embedding, rbp_tracks = _extract_parnet_features(self.backbone, seq_onehot)

        alpha = self.mix_coeff(embedding)                         # (B, 223)
        scale = self.log_scale.exp()                              # (223,)
        scaled = rbp_tracks * scale[None, :, None]                # (B, 223, L)

        interference = self.qlayer(scaled, alpha)                 # (B, 1, L)
        pred = self.cnn(interference)                             # (B, 1, L)
        return pred, alpha

    def get_coupling_matrix(self) -> torch.Tensor:
        """(num_rbps, num_rbps) pairwise coupling cos(φ_i − φ_j)."""
        return self.qlayer.coupling_matrix()
