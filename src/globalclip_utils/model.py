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

    Two modes:
      - global (positional=False, original behaviour): mean-pools the
        PARNET embedding over all positions first, then projects to a
        single (B, num_tasks) sigmoid weight vector per sequence. This
        means the model can only pick one RBP mixture for the entire
        600bp window, even though the underlying binding signal is known
        to vary along the sequence.
      - positional (positional=True): applies the same MLP at every
        position independently (via 1x1 convs), producing
        (B, num_tasks, L) weights that can vary along the sequence --
        letting different sub-regions be dominated by different RBPs.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_tasks: int = 223,
        hidden: int = 128,
        positional: bool = False,
    ):
        super().__init__()
        self.positional = positional
        if positional:
            self.fc1 = nn.Conv1d(embed_dim, hidden, kernel_size=1)
            self.fc2 = nn.Conv1d(hidden, num_tasks, kernel_size=1)
        else:
            self.fc1 = nn.Linear(embed_dim, hidden)
            self.fc2 = nn.Linear(hidden, num_tasks)
        self.act = nn.ReLU()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (B, embed_dim, L) backbone feature map.

        Returns:
            alpha: (B, num_tasks) mixing coefficients in [0, 1] if
                   positional=False, or (B, num_tasks, L) if
                   positional=True.
        """
        if self.positional:
            return torch.sigmoid(self.fc2(self.act(self.fc1(embedding))))  # (B, T, L)
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

    def __init__(self, num_rbps: int = 223, embed_dim: int = 512, positional_phase: bool = False):
        super().__init__()
        self.positional_phase = positional_phase
        if positional_phase:
            # Phase becomes a function of local sequence context instead of a
            # single global value per protein: phi(p) = conv1x1(embedding)(p).
            # Cooperativity/competition between two proteins can then depend
            # on e.g. local secondary structure, not just protein identity.
            self.phase_net = nn.Conv1d(embed_dim, num_rbps, kernel_size=1)
        else:
            # NOTE: must NOT init all phases to exactly 0 — at phi=0 for every
            # protein, imag = sum(amp_i * sin(0)) = 0 identically, which makes
            # dI/dphi_j = 2*real*(-amp_j*sin(0)) + 2*imag*(amp_j*cos(0)) = 0 for
            # every j regardless of the data. That's an exact saddle point, so
            # phase never moves away from 0 during training. Small random init
            # breaks the symmetry so gradients can flow from step 0.
            self.phase = nn.Parameter(torch.randn(num_rbps) * 0.1)

    def _phase(self, embedding: torch.Tensor | None) -> torch.Tensor:
        if self.positional_phase:
            if embedding is None:
                raise ValueError("positional_phase=True requires `embedding` to be passed.")
            return self.phase_net(embedding)                        # (B, R, L)
        return self.phase                                           # (R,)

    def forward(
        self, rbp_tracks: torch.Tensor, alpha: torch.Tensor, embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            rbp_tracks: (B, num_rbps, L) scaled RBP binding tracks.
            alpha:      (B, num_rbps) or (B, num_rbps, L) mixing coefficients.
            embedding:  (B, embed_dim, L) backbone embedding. Required iff
                        positional_phase=True.

        Returns:
            interference: (B, 1, L) |Ψ|² interference pattern.
        """
        if alpha.dim() == 2:
            alpha = alpha[:, :, None]                               # (B, R) -> (B, R, 1)
        amp = rbp_tracks * alpha                                    # (B, R, L)
        phi = self._phase(embedding)
        if phi.dim() == 1:
            phi = phi[None, :, None]                                # (R,) -> (1, R, 1), broadcasts
        real = (amp * torch.cos(phi)).sum(1)                        # (B, L)
        imag = (amp * torch.sin(phi)).sum(1)                        # (B, L)
        return (real ** 2 + imag ** 2).unsqueeze(1)                # (B, 1, L)

    def coupling_matrix(self, embedding: torch.Tensor | None = None) -> torch.Tensor:
        """Return (num_rbps, num_rbps) pairwise coupling: cos(φ_i − φ_j).

        Values > 0 mean the two proteins tend to cooperate (constructive);
        values < 0 mean they compete or anti-correlate (destructive).

        If positional_phase=True, `embedding` (a batch of backbone
        embeddings) must be supplied; the returned matrix is then based on
        the phase averaged over batch and position (a single summary
        matrix, not a per-position one).
        """
        if self.positional_phase:
            if embedding is None:
                raise ValueError("positional_phase=True requires `embedding` to be passed.")
            phi = self.phase_net(embedding).detach().mean(dim=(0, 2))  # (R,)
        else:
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
        positional_alpha: bool = False,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.backbone = parnet_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.mix_coeff = MixCoeffHead(
            embed_dim=embed_dim, num_tasks=num_rbps, hidden=mix_hidden, positional=positional_alpha
        )
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
                   If the mixing head is positional, this is the
                   per-position weights averaged over L, so downstream
                   analysis code (ranking, correlation) keeps working
                   unchanged; the position-resolved weights are still
                   used internally to compute `pred`.
        """
        with torch.no_grad():
            embedding, rbp_tracks = _extract_parnet_features(self.backbone, seq_onehot)

        alpha = self.mix_coeff(embedding)                          # (B, 223) or (B, 223, L)
        scale = self.log_scale.exp()                               # (223,)
        scaled = rbp_tracks * scale[None, :, None]                 # (B, 223, L)
        alpha_bc = alpha[:, :, None] if alpha.dim() == 2 else alpha
        pred = (scaled * alpha_bc).sum(1, keepdim=True)             # (B, 1, L)
        alpha_out = alpha if alpha.dim() == 2 else alpha.mean(dim=-1)
        return pred, alpha_out

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
        positional_alpha: bool = False,
        positional_phase: bool = False,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.backbone = parnet_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.mix_coeff = MixCoeffHead(
            embed_dim=embed_dim, num_tasks=num_rbps, hidden=mix_hidden, positional=positional_alpha
        )
        self.log_scale = nn.Parameter(torch.zeros(num_rbps))
        self.qlayer = QLayer(num_rbps=num_rbps, embed_dim=embed_dim, positional_phase=positional_phase)

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

        alpha = self.mix_coeff(embedding)                         # (B, 223) or (B, 223, L)
        scale = self.log_scale.exp()                              # (223,)
        scaled = rbp_tracks * scale[None, :, None]                # (B, 223, L)

        interference = self.qlayer(scaled, alpha, embedding=embedding)  # (B, 1, L)
        pred = self.cnn(interference)                             # (B, 1, L)
        alpha_out = alpha if alpha.dim() == 2 else alpha.mean(dim=-1)
        return pred, alpha_out

    def get_coupling_matrix(self, seq_onehot: torch.Tensor | None = None) -> torch.Tensor:
        """(num_rbps, num_rbps) pairwise coupling cos(φ_i − φ_j).

        If the QLayer uses positional_phase, `seq_onehot` (a batch of
        sequences, e.g. from the test set) must be supplied so the
        embedding needed to compute phase(p) can be derived; the returned
        matrix is then the phase averaged over that batch and position.
        """
        if self.qlayer.positional_phase:
            if seq_onehot is None:
                raise ValueError("positional_phase=True requires `seq_onehot` to be passed.")
            with torch.no_grad():
                embedding, _ = _extract_parnet_features(self.backbone, seq_onehot)
            return self.qlayer.coupling_matrix(embedding=embedding)
        return self.qlayer.coupling_matrix()


# ── Model 3: CNN-only ablation (no QLayer interference) ──────────────────────


class GlobalCLIPCNNModel(nn.Module):
    """Ablation model: Standard weighted-sum mixing, refined directly by the
    same dilated CNN used in GlobalCLIPQLayerModel -- but with no QLayer
    interference step in between.

    Purpose: QLayer outperforms the Standard model mostly because of the
    dilated-CNN refinement stage, not the phase-interference mechanism
    (see Results). This model isolates that CNN contribution on its own,
    so it can be compared directly against both GlobalCLIPStandardModel
    (no CNN) and GlobalCLIPQLayerModel (CNN + interference) to determine
    how much of the improvement the interference step actually adds.
    """

    def __init__(
        self,
        parnet_model: nn.Module,
        num_rbps: int = 223,
        mix_hidden: int = 128,
        cnn_channels: int = 64,
        cnn_kernel: int = 9,
        cnn_layers: int = 3,
        positional_alpha: bool = False,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.backbone = parnet_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.mix_coeff = MixCoeffHead(
            embed_dim=embed_dim, num_tasks=num_rbps, hidden=mix_hidden, positional=positional_alpha
        )
        self.log_scale = nn.Parameter(torch.zeros(num_rbps))

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
        with torch.no_grad():
            embedding, rbp_tracks = _extract_parnet_features(self.backbone, seq_onehot)

        alpha = self.mix_coeff(embedding)                         # (B, 223) or (B, 223, L)
        scale = self.log_scale.exp()                              # (223,)
        scaled = rbp_tracks * scale[None, :, None]                # (B, 223, L)
        alpha_bc = alpha[:, :, None] if alpha.dim() == 2 else alpha
        mixed = (scaled * alpha_bc).sum(1, keepdim=True)          # (B, 1, L) — same as Standard model
        pred = self.cnn(mixed)                                    # (B, 1, L)
        alpha_out = alpha if alpha.dim() == 2 else alpha.mean(dim=-1)
        return pred, alpha_out


# ── Model 4: Hybrid CombiLayer (CNN-only path + QLayer path, both to CNN) ────


class GlobalCLIPHybridModel(nn.Module):
    """Hybrid "CombiLayer" model: feeds BOTH the plain weighted-sum signal
    (as in GlobalCLIPCNNModel) AND the QLayer interference pattern into the
    CNN as two separate input channels, instead of choosing one or the
    other.

    Motivation: across every prior comparison (global vs. positional alpha,
    global vs. positional phase), the interference pathway never improved
    on the plain CNN-only pathway, and sometimes made it slightly worse.
    Rather than picking a winner, let the CNN itself learn how much (if
    any) weight to give the interference channel -- so the model can never
    do worse than CNN-only (the CNN can learn to ignore channel 2), while
    the QLayer phase parameters still receive gradient and remain
    interpretable via `get_coupling_matrix()`, regardless of how much the
    CNN actually uses them for prediction.

    Architecture:
        PARNET backbone (frozen)
            -> (B, 512, L) embedding  +  (B, 223, L) rbp_tracks
        MixCoeffHead -> alpha, log_scale -> scaled tracks
        mixed        = sum_i(scaled_i * alpha_i)              (B, 1, L)
        interference = QLayer(scaled, alpha, embedding)       (B, 1, L)
        cnn_input    = concat([mixed, interference], dim=1)   (B, 2, L)
        Dilated CNN (2 input channels) -> (B, 1, L) prediction
    """

    def __init__(
        self,
        parnet_model: nn.Module,
        num_rbps: int = 223,
        mix_hidden: int = 128,
        cnn_channels: int = 64,
        cnn_kernel: int = 9,
        cnn_layers: int = 3,
        positional_alpha: bool = False,
        positional_phase: bool = False,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.backbone = parnet_model
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.mix_coeff = MixCoeffHead(
            embed_dim=embed_dim, num_tasks=num_rbps, hidden=mix_hidden, positional=positional_alpha
        )
        self.log_scale = nn.Parameter(torch.zeros(num_rbps))
        self.qlayer = QLayer(num_rbps=num_rbps, embed_dim=embed_dim, positional_phase=positional_phase)

        layers: list[nn.Module] = []
        in_ch = 2                                                  # mixed + interference
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
        self, seq_onehot: torch.Tensor, ablate: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            seq_onehot: (B, 4, L) one-hot encoded sequence.
            ablate: None (normal), "interference" (zero that channel), or
                    "mixed" (zero that channel) -- for measuring how much
                    each pathway actually contributes to predictions
                    (channel-ablation analysis).

        Returns:
            pred:  (B, 1, L) predicted GlobalCLIP track.
            alpha: (B, num_rbps) mixing coefficients.
        """
        with torch.no_grad():
            embedding, rbp_tracks = _extract_parnet_features(self.backbone, seq_onehot)

        alpha = self.mix_coeff(embedding)                         # (B, 223) or (B, 223, L)
        scale = self.log_scale.exp()                              # (223,)
        scaled = rbp_tracks * scale[None, :, None]                # (B, 223, L)
        alpha_bc = alpha[:, :, None] if alpha.dim() == 2 else alpha

        mixed = (scaled * alpha_bc).sum(1, keepdim=True)          # (B, 1, L)
        interference = self.qlayer(scaled, alpha, embedding=embedding)  # (B, 1, L)

        if ablate == "interference":
            interference = torch.zeros_like(interference)
        elif ablate == "mixed":
            mixed = torch.zeros_like(mixed)

        cnn_input = torch.cat([mixed, interference], dim=1)       # (B, 2, L)
        pred = self.cnn(cnn_input)                                # (B, 1, L)
        alpha_out = alpha if alpha.dim() == 2 else alpha.mean(dim=-1)
        return pred, alpha_out

    def get_coupling_matrix(self, seq_onehot: torch.Tensor | None = None) -> torch.Tensor:
        """(num_rbps, num_rbps) pairwise coupling cos(φ_i − φ_j). See
        GlobalCLIPQLayerModel.get_coupling_matrix for the positional_phase
        case (requires `seq_onehot`)."""
        if self.qlayer.positional_phase:
            if seq_onehot is None:
                raise ValueError("positional_phase=True requires `seq_onehot` to be passed.")
            with torch.no_grad():
                embedding, _ = _extract_parnet_features(self.backbone, seq_onehot)
            return self.qlayer.coupling_matrix(embedding=embedding)
        return self.qlayer.coupling_matrix()

    def channel_weight_summary(self) -> dict:
        """Quick, data-free proxy for how much the CNN attends to each
        input channel: L2 norm of the first conv layer's weights per
        channel. Not a substitute for the channel-ablation test in
        evaluate_new_models.py (which measures actual accuracy impact),
        but a cheap sanity check obtainable from the weights alone.
        """
        first_conv = self.cnn[0]
        w = first_conv.weight.detach()             # (out_channels, 2, kernel_size)
        mixed_norm = w[:, 0, :].norm().item()
        interference_norm = w[:, 1, :].norm().item()
        total = mixed_norm + interference_norm + 1e-8
        return {
            "mixed_weight_norm": mixed_norm,
            "interference_weight_norm": interference_norm,
            "interference_weight_fraction": interference_norm / total,
        }
