# GlobalCLIP CombiLayer — PARNET-based deconvolution

Predicts the combined **GlobalCLIP** signal (all RNA-protein interactions
measured at once, no single antibody) from RNA sequence, by learning to
recombine the 223 per-protein binding tracks of a frozen, pretrained
**PARNET** backbone. The learned mixing weights and (experimental)
interaction terms are then used to ask: *which proteins drive the signal,
and how do they interact?*

See `ideen.txt` for the full original design notes (architecture options,
loss functions, interaction-analysis methods). This README documents what
was actually built and how to run it.

## Project structure

```
project-parnet/
├── config/
│   └── filepaths.server.yaml   # server-only paths (models, datasets, results dirs)
├── src/globalclip_utils/
│   ├── model.py                # model architectures (see below)
│   ├── datasets.py              # GlobalCLIPDataset (loads .pt.gz files)
│   ├── training_utils.py        # loss functions + PyTorch Lightning module
│   └── analysis_utils.py        # ranking, correlation, plotting helpers
├── scripts/
│   ├── train_standard.py        # trains GlobalCLIPStandardModel
│   ├── train_qlayer.py          # trains GlobalCLIPQLayerModel
│   ├── train_cnn_only.py        # trains GlobalCLIPCNNModel (ablation)
│   ├── export_results_data.py   # full analysis (old pipeline, Standard+QLayer together)
│   └── evaluate_new_models.py   # lightweight per-model test-set evaluation
├── results/
│   ├── globalclip/{standard,qlayer,cnn_only}/<run-id>/   # trained checkpoints + training curves
│   ├── data/                                              # export_results_data.py output (old run)
│   ├── results_data_qlayer_v2/                            # export_results_data.py output (after phase-bug fix)
│   ├── results_new/                                       # evaluate_new_models.py output (first 3 ablations)
│   └── new_v2/                                            # evaluate_new_models.py output (4th ablation)
└── nohup.txt                    # copy-pasteable server commands (this README explains the "why")
```

## Data

GlobalCLIP `.pt.gz` files (600nt regions), same train/valid/test region split
(39,052 / 7,361 / 4,946) across all three control variants. Target used for
training: `log1p(signal)` (raw counts, log-transformed). The biological
`control` track exists in the raw data and dataset loader supports it in
principle, but is **intentionally unused** — supervisors assessed the
GlobalCLIP control tracks as unsuitable for normalization (see
`training_utils.py` / `datasets.py`, lines commented `# not used`).

## Models (`src/globalclip_utils/model.py`)

All three models share the same frozen PARNET backbone
(`_extract_parnet_features`): sequence → `(B, 512, L)` embedding +
`(B, 223, L)` per-RBP binding tracks. Only the layers described below are
trained.

### `MixCoeffHead`
Turns the embedding into per-protein mixing weights α (sigmoid, one value
per protein). Two modes:
- **global** (default): mean-pools the embedding over all 600 positions
  first → **one** α-vector per sequence.
- **positional** (`positional_alpha=True` / `--positional-alpha` flag):
  1×1-conv MLP applied at every position independently → α varies along
  the sequence, `(B, 223, L)`. This was the single biggest empirical lever
  found (see Results below) — the original global-pooling design couldn't
  express that different sub-regions of a 600bp window can be dominated by
  different RBPs.

  Either way, `forward()` always *also* returns the α averaged over
  position, `(B, 223)`, so existing ranking/correlation analysis code keeps
  working unchanged regardless of which mode was used for training.

### `GlobalCLIPStandardModel`
`rbp_tracks × α × exp(log_scale)`, summed over the 223 proteins →
`(B, 1, L)` prediction. No refinement step. `log_scale` is a learned
per-protein amplitude correction (compensates for eCLIP antibody-efficiency
bias across proteins of different sizes).

### `GlobalCLIPQLayerModel`
Same mixing, but routed through `QLayer` (a "quantum-inspired" interference
layer, see `ideen.txt` §15) before a dilated CNN refines it:

```
ψ_i(p) = α_i · rbp_track_i(p) · e^{iφ_i}      (φ_i = learnable phase per protein)
Ψ(p)   = Σ_i ψ_i(p)
I(p)   = |Ψ(p)|²                                → (B, 1, L), fed into the CNN
```

The cross-terms in `I(p)` encode pairwise protein-protein coupling
(`cos(φ_i − φ_j)`) with only 223 parameters instead of a 223×223 matrix.

**Known/fixed bug:** `QLayer.phase` was originally initialized to all-zero
(`torch.zeros`), following the "warm start = behaves like linear mixing"
idea in `ideen.txt` §"TRAININGS-STRATEGIE FÜR QLAYER". This is a real trap:
when *every* phase is exactly equal, `Im(Ψ) ≡ 0` regardless of the data,
which makes `∂I/∂φ_i = 0` for every protein — an exact saddle point with
zero gradient in every direction, so phases never move during training.
Fixed by initializing `torch.randn(num_rbps) * 0.1` instead (small random
values break the symmetry). Confirmed fixed: phases now differentiate
during training and the resulting coupling matrix is non-degenerate
(previously: `cos(Δφ)=1.0` for literally every protein pair).

### Positional phase (`QLayer.positional_phase`)
Extension of `QLayer`: instead of one global phase φ_i per protein
(`nn.Parameter(223,)`), phase becomes a function of local sequence context,
`φ_i(p) = Conv1d(embedding)(p)` — a `(B, 223, L)` tensor. Motivation: the
global→positional change was the biggest lever for α (see Results); this
tests the same idea for phase, since cooperativity between two proteins may
depend on local context (e.g. secondary structure) rather than being fixed
everywhere. `QLayer.forward`/`coupling_matrix` handle both the
`(223,)`-global and `(B,223,L)`-positional cases transparently; in the
positional case, `coupling_matrix()` returns a single summary matrix based
on phase averaged over the given batch/positions (needs a batch of
sequences passed in, since there's no single global phase to read off a
trained parameter).

### `GlobalCLIPCNNModel` (ablation)
Same as `GlobalCLIPStandardModel`'s mixing, but the weighted sum is fed
directly into the same dilated CNN used by QLayer — **no** interference
step in between. Added to answer: does QLayer's advantage come from the
CNN, or from the interference mechanism? (Answer: the CNN — see Results.)
This directly follows `ideen.txt`'s own documented ablation recipe ("Modell
ohne QLayer ... vs. mit QLayer").

## Scripts

| Script | Trains/evaluates | Key flags |
|---|---|---|
| `train_standard.py` | `GlobalCLIPStandardModel` | `--positional-alpha`, `--lambda-alpha`, `--lambda-nll`, `--run-id` |
| `train_qlayer.py` | `GlobalCLIPQLayerModel` | same + `--lambda-phase`, `--cnn-*` |
| `train_cnn_only.py` | `GlobalCLIPCNNModel` (ablation) | same as qlayer minus phase |
| `train_qlayer_positional_phase.py` | `GlobalCLIPQLayerModel` with **positional phase** always on | `--positional-alpha` (combine with positional α too), `--lambda-phase`, `--cnn-*` |
| `export_results_data.py` | Loads **one** Standard + **one** QLayer run together, full analysis: protein ranking, alpha correlation matrix, QLayer phase polar plot + coupling matrix, baselines, significance tests, Spearman, windowed correlation, Integrated Gradients. Writes ~15 CSVs + PNGs to a flat output dir. | `--standard-run-id`, `--qlayer-run-id`, `--output-dir` |
| `evaluate_new_models.py` | Evaluates **any subset** of {standard, qlayer, cnn_only} runs **independently** (each gets its own subfolder) — lighter-weight than `export_results_data.py`, just test-set Pearson r + distribution plot per model, no protein-ranking/IG/etc. | `--standard-run-id`, `--qlayer-run-id`, `--cnn-run-id` (any combination), `--output-dir` |

All training scripts save to `results/globalclip/<model-type>/<run-id>/`:
`model.statedict.pt`, `model.full.pt`, `run_config.json` (hyperparameters —
read back by both evaluation scripts to reconstruct the exact architecture),
`csv_logs/`, `checkpoints/`, `training_curves.png`.

**Important distinction:** training-time logs (`training_curves.png`,
`csv_logs/`) show train/val loss *during training* — they are not the same
as test-set Pearson r. You must run one of the two evaluation scripts
afterward to get an actual held-out test-set number.

## Results — what's in each folder

- **`results/globalclip/<model>/<run-id>/`** — raw training output (checkpoints
  + training curves). Input to both evaluation scripts, not itself a result.
- **`results/data/`** — `export_results_data.py` output for the *first*
  QLayer run (had the phase-init bug). Kept for reference/history.
- **`results/results_data_qlayer_v2/`** — `export_results_data.py` output
  after the phase-bug fix (Standard v3 + QLayer v2). Current full analysis.
- **`results/results_new/`** — `evaluate_new_models.py` output for the first
  three ablation experiments (standard+positional, qlayer+positional,
  cnn_only global-α).
- **`results/new_v2/`** — `evaluate_new_models.py` output for the fourth
  ablation (cnn_only + positional-α).

## Findings so far (test-set mean Pearson r, `log1p(signal)` vs. prediction)

| Model | Mixing | Refinement | Mean r | Notes |
|---|---|---|---|---|
| Baseline (naive track mean) | – | – | 0.189 | sequence-agnostic floor |
| Standard | global | – | 0.266 | v3, tuned λ |
| Standard | **positional** | – | **0.4315** | biggest single lever |
| CNN-only (ablation) | global | CNN only | 0.430 | ≈ same jump as positional-α |
| QLayer | global | interference + CNN | 0.407 | phase bug fixed; ≈ same as CNN-only *or slightly worse* |
| QLayer | positional | interference + CNN | **0.4451** | best so far |
| CNN-only (ablation) | positional | CNN only | **0.4458** | best overall — beats QLayer at both global and positional settings |

**Key conclusions:**
1. The global (one-per-sequence) mixing weight was the single biggest
   architectural bottleneck — positional resolution alone nearly matches
   what the full QLayer+CNN redesign achieved.
2. QLayer's phase/interference mechanism does **not** improve accuracy over
   a plain CNN on the same mixed signal (CNN-only ≥ QLayer at matched
   settings) — per `ideen.txt`'s own ablation criterion (Δr > 0.02 =
   meaningful, < 0.01 = interactions barely matter), this says the modeled
   interactions barely matter for prediction accuracy in this dataset.
3. The QLayer coupling analysis (protein clusters, `cos(Δφ)`) is still
   scientifically legitimate to report post-fix, but is an
   *interpretability* result, not an *accuracy* result — and known
   biological validation targets (spliceosome components) were not among
   the model's dominant proteins in this run, and no destructive/competitive
   couplings were found (only constructive, 0.77–1.0).
4. r² context: best raw (1bp) result so far is r≈0.445 → r²≈0.20. A
   Poisson counting-noise ceiling in the read-count data means very high r
   (e.g. 0.9) is not realistically achievable at 1bp resolution regardless
   of model size/training time — smoothing to ~50bp already lifts r to
   ~0.65–0.70 for the better models, which is closer to a practical ceiling.

## Quick reference: running an experiment end-to-end

```bash
cd /mnt/storage1/workspace/pgoldemund/parnet--idea1-reconstruction-head
export PIXI_CACHE_DIR="$HOME/.pixi-cache"

# 1) train (pick one)
pixi run -e parnet-dev-cu12 python scripts/train_standard.py --lambda-alpha 0.1 --lambda-nll 0.1 --positional-alpha --run-id <run-id> --gpu 0
pixi run -e parnet-dev-cu12 python scripts/train_qlayer.py   --lambda-alpha 0.1 --lambda-nll 0.1 --positional-alpha --run-id <run-id> --gpu 0
pixi run -e parnet-dev-cu12 python scripts/train_cnn_only.py --positional-alpha --run-id <run-id> --gpu 0

# 2) evaluate (any subset)
pixi run -e parnet-dev-cu12 python scripts/evaluate_new_models.py \
  --standard-run-id <run-id> --qlayer-run-id <run-id> --cnn-run-id <run-id> \
  --output-dir results/<some-new-folder>
```

See `nohup.txt` for ready-to-paste `nohup` background-run commands
(including the full training+eval chains actually used so far).
