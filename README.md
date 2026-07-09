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

### `GlobalCLIPHybridModel` ("CombiLayer")
Feeds **both** the plain weighted-sum signal (`mixed`, same as
`GlobalCLIPCNNModel`) and the QLayer interference pattern into the CNN as
two separate input channels (`(B, 2, L)` instead of `(B, 1, L)`), instead of
picking one architecture over the other. The CNN itself learns how much
weight to give each channel.

**Important caveat:** if the CNN learns to fully ignore the interference
channel (weight → 0), the gradient path that would otherwise teach the
phases φ_i to differentiate also disappears — only `λ_phase`
regularization remains, which pulls phases back toward 0. In that case the
coupling matrix could become uninformative again, this time because the
network genuinely found no useful interaction signal (not a bug). Always
check `channel_weight_summary()` / `channel_ablation.json` (written by
`evaluate_new_models.py --combilayer-run-id ...`) to see whether this
happened before interpreting the coupling matrix.

`evaluate_new_models.py --combilayer-run-id <id>` additionally reports:
- `channel_ablation.json`: test-set mean r with each channel zeroed out
  (`mean_r_interference_zeroed`, `mean_r_mixed_zeroed`) and the resulting
  `interference_contribution` (= accuracy drop from removing it) — the
  most direct measure of how much the interference pathway matters.
- `coupling_matrix.csv` / `.png` / `coupling_top_pairs.csv`: the
  protein-protein interaction structure, independent of whether the CNN
  uses it for prediction.

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
| `train_combilayer.py` | `GlobalCLIPHybridModel` ("CombiLayer": CNN-only path + QLayer path, both fed to CNN) | `--positional-alpha`, `--positional-phase`, `--lambda-phase`, `--cnn-*` |
| `export_results_data.py` | Loads **one** Standard + **one** QLayer run together, full analysis: protein ranking, alpha correlation matrix, QLayer phase polar plot + coupling matrix, baselines, significance tests, Spearman, windowed correlation, Integrated Gradients. Writes ~15 CSVs + PNGs to a flat output dir. | `--standard-run-id`, `--qlayer-run-id`, `--output-dir` |
| `evaluate_new_models.py` | Evaluates **any subset** of {standard, qlayer, cnn_only, combilayer} runs **independently** (each gets its own subfolder) — lighter-weight than `export_results_data.py`, just test-set Pearson r + distribution plot per model, no protein-ranking/IG/etc. | `--standard-run-id`, `--qlayer-run-id`, `--cnn-run-id`, `--combilayer-run-id` (any combination), `--output-dir` |
| `motif_validation.py` | Independent check of the protein ranking: runs Integrated Gradients directly on PARNET's own frozen per-protein tracks (not on any combi-layer prediction), then the same metamotif + RBP-database pipeline as Idea 2, and checks whether each protein's own recovered motif matches its own literature motif ("self-match"). Does not depend on any globalCLIP fine-tuning or on our mixing weights. | `--proteins` (list of RBP-cell-line track names), `--output-dir`, `--gpu` |

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
  This local checkout only has the *early* runs (pre-outlier-filtering:
  `standard.v1-v3`, `qlayer.v1`, two `cnn_only` ablations). The final,
  filtered-dataset runs referenced in the table below (the `*_filtered_v1`
  and `*_21m_filtered_v1` run-ids, plus `combilayer.filtered_v1`) were
  trained on the server and are not all present in this local `results/` —
  re-run the corresponding `nohup.txt` block if you need the checkpoints
  themselves; the numbers below are already-verified final results.
- **`results/data/`** — `export_results_data.py` output for the *first*
  QLayer run (had the phase-init bug). Kept for reference/history.
- **`results/results_data_qlayer_v2/`** — `export_results_data.py` output
  after the phase-bug fix (Standard v3 + QLayer v2).
- **`results/results_new/`**, **`results/new_v2/`**, **`new_v3/`**, **`new_v4/`** —
  `evaluate_new_models.py` output for the early (pre-filtering) ablation
  experiments, in the order they were run (see `nohup.txt` for which
  run-ids each corresponds to).
- **`results/results_new_filtered_{standard_global,standard_positional,
  cnn_global,cnn_positional,qlayer_global,qlayer_positional,qlayer_posphase,
  combilayer}/`** — `evaluate_new_models.py` output for the final,
  outlier-filtered runs (one folder per configuration). These are the
  source of the "Findings so far" table below. The 21M-backbone run
  (`cnn_only`, positional, 21M) was evaluated separately and is not present
  under this naming pattern in this local checkout.
- **`results/results_motif_validation/`** — `motif_validation.py` output:
  one subfolder per tested protein (`metamotif/` search + alignment
  results), plus `00_summary.json` with the self-match verdict per protein.

## Findings so far (test-set mean Pearson r, outlier-filtered dataset, `log1p(signal)` vs. prediction)

| Model | Mixing | Backbone | Mean r | Notes |
|---|---|---|---|---|
| Standard | global | 7M | 0.264 | |
| Standard | **positional** | 7M | **0.429** | biggest single lever (+0.165 over global) |
| CNN-only (ablation) | global | 7M | 0.427 | |
| CNN-only (ablation) | positional | 7M | 0.442 | |
| CNN-only (ablation) | positional | **21M** | **0.476** | best overall |
| QLayer | positional | 7M | 0.439 | interference + CNN, no measurable gain over CNN-only |
| QLayer | positional, positional phase | 7M | 0.436 | phase also made position-dependent — still no gain |
| CombiLayer (hybrid) | positional | 7M | 0.443 | CNN-only + QLayer signal fed together |

**Key conclusions:**
1. The global (one-per-sequence) mixing weight was the single biggest
   architectural bottleneck — switching to positional resolution alone
   (+0.165 Pearson for the Standard model) is the largest lever anywhere in
   this project, larger than any refinement architecture tested afterward.
2. QLayer's phase/interference mechanism does **not** improve accuracy over
   a plain CNN on the same mixed signal: CNN-only (0.442), CombiLayer
   (0.443), QLayer positional (0.439), and QLayer positional+phase (0.436)
   are all within **0.007 Pearson** of each other at matched (7M,
   positional) settings — statistically indistinguishable. This is why
   QLayer/CombiLayer results are not reported in the team's LaTeX writeup.
3. The QLayer coupling analysis (protein clusters, `cos(Δφ)`) is still
   scientifically legitimate to report, but is an *interpretability*
   result, not an *accuracy* result — and known biological validation
   targets (spliceosome components) were not among the model's dominant
   proteins in the run tested so far, and no destructive/competitive
   couplings were found (only constructive). **This analysis predates the
   outlier-filtering fix and should be re-run on the final checkpoints
   before drawing conclusions from it** — see Outlook below.
4. **Best accuracy is not best interpretability.** The 21M backbone gives
   the best reconstruction accuracy (0.476), but an independent check
   (`motif_validation.py`, self-match: does a protein's own recovered
   motif match its own literature motif?) found the 7M backbone
   self-matches 6 of its top 15 weighted proteins, versus only 2 of 15 for
   the 21M backbone — and the 21M backbone's motifs are more fragmented
   (3.27 vs. 2.33 consensus motifs per protein on average). The backbone
   that best reconstructs the mixed signal is not the one whose per-protein
   contributions are easiest to validate individually. See
   `latex3/sections/results.tex` (`sec:selfmatch-results`) for the full
   writeup.
5. r² context: best result is r≈0.476 → r²≈0.23. A Poisson counting-noise
   ceiling in the read-count data means very high r (e.g. 0.9) is not
   realistically achievable at 1bp resolution regardless of model size or
   training time — smoothing to ~50bp lifts r to ~0.65–0.70 for the better
   models, which is closer to a practical ceiling.

## Outlook / how to improve this further

If you're picking this up next, roughly in priority order:

1. **Diagnose the coupling matrix before trusting or discarding it.** The
   core limitation behind points 3–4 above is an identifiability problem:
   the reconstruction loss only ever sees the *aggregate* signal, never any
   individual protein's true contribution. Proteins with correlated
   Parnet-predicted tracks (e.g. PTBP1, PTBP2, PUF60 — all
   polypyrimidine-tract binders) can trade mixing weight between each other
   with almost no effect on the reconstructed signal, so the loss has no
   way to prefer a "correct" attribution over a merely equivalent one. This
   is exactly what QLayer's phase mechanism was meant to break via pairwise
   interference, but it's unproven whether it actually does. Cheap first
   step (no retraining): compute the raw correlation between Parnet's own
   223 per-protein tracks directly — independent of any mixing weights —
   and compare it against whatever coupling matrix a trained QLayer/
   CombiLayer model produces. Agreement suggests the phase mechanism
   captures genuine co-activity; disagreement means the aggregate-only loss
   needs an explicit interaction-supervision term (a more expensive, second
   step — only worth it if the diagnostic actually shows a mismatch).
2. **Re-run the QLayer/CombiLayer coupling-matrix analysis on the final
   filtered checkpoints.** The current coupling results (only cooperative
   couplings found, expected spliceosome validation proteins not among the
   dominant ones) predate the outlier-filtering fix and were never
   recomputed on `*.filtered_v1` runs.
3. **Test control-track normalization empirically, despite the "not
   established as suitable" caveat.** All models here train on raw
   `log1p(signal)`, never `log(1+signal) - log(1+control)`. An imperfect
   control estimate could still help separate protein-specific signal from
   shared background — this can only be settled by actually training a
   variant with it and comparing, not by assumption. `datasets.py` and
   `training_utils.py` already load/support `control`, just wired to be
   unused (`compute_log_enrichment` exists but is dead code) — flip it back
   on and compare.
4. **Stability selection / ensemble training for robust protein rankings.**
   `standard_global` and `standard_positional` share *no* common top-5
   protein despite differing only in mixing mode — a real instability, not
   noise. Train several seeds (or bootstrap-resampled data) per
   configuration and only trust a protein's contribution if it's
   consistently high-ranked across runs, rather than reading off a single
   training run's top-5.
5. **Biologically-informed grouping as a structural prior.** Instead of
   223 fully free mixing weights, group tracks by known RBP complex
   membership (spliceosome, EJC, hnRNP family — from STRING-DB or
   literature) and mix within-group first, then between-group. This
   reduces the degrees of freedom directly, which should make the
   attribution more identifiable rather than just hoping a bigger/better
   model sorts it out on its own.

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
