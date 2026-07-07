"""Validate Idea 1's dominant-protein ranking via Integrated Gradients + metamotif.

For each requested PARNET RBP track (e.g. the top-ranked proteins from our
Standard/QLayer/CombiLayer mixing-weight ranking), computes Integrated
Gradients attribution of that SPECIFIC protein's raw PARNET track output
(before any Idea-1 combi-layer mixing) across the test set, then runs the
same metamotif search + alignment + RBP-database-matching pipeline used for
Idea 2 (see 03_integrated_gradients.ipynb, shared by Lukas), to check
whether the recovered consensus motif matches known literature motifs for
that protein -- an independent validation of the protein ranking, since it
does not depend on our combi-layer's mixing weights at all, only on
PARNET's own (eCLIP-trained) per-protein detector.

Usage:
    pixi run -e parnet-dev-cu12 python scripts/motif_validation.py \
        --proteins PABPN1_HepG2 HNRNPC_K562 SND1_HepG2 DGCR8_K562 NONO_K562 LSM11_K562 \
        --output-dir results/motif_validation
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from parnet_additional_utils import ParnetModelName, load_parnet_model
from globalclip_utils import GlobalCLIPDataset
from globalclip_utils.model import _extract_parnet_features

import pylbsr.bio.motifs as motifs_lib
from pylbsr.bio.motifs.motif import convert_matrix_type

MOTIFS_DIR = Path("/mnt/storage1/ml4rg26-shared/rbp_binding_motifs")
DATABASES = ["mCrossBase", "ATtRACT", "RBPmap_1.2", "oRNAment", "RBPDB"]
NUCLEOTIDE_ARR = np.array(["A", "C", "G", "T"])

# Path to metamotif's bundled alignment script (found via pixi/uv cache).
# If this changes between environments, override with --align-script.
_DEFAULT_ALIGN_SCRIPT = Path(
    "/home/pgoldemund/.pixi-cache/uv-cache/git-v0/checkouts/9a8a60cb70338b5a"
    "/ac43d46/scripts/variable-length-seed-motif-alignment.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proteins", nargs="+", required=True,
                    help="RBP track names to validate, e.g. PABPN1_HepG2 HNRNPC_K562")
    p.add_argument("--dataset", default="globalclip_lysate_noNHS")
    p.add_argument("--n-steps", type=int, default=50, help="Integrated Gradients steps")
    p.add_argument("--max-sequences", type=int, default=None,
                    help="Limit number of test sequences (default: all)")
    p.add_argument("--sig-p", type=float, default=0.05)
    p.add_argument("--seed-size", type=int, default=3)
    p.add_argument("--max-size", type=int, default=20)
    p.add_argument("--min-support", type=int, default=20)
    p.add_argument("--max-motifs", type=int, default=15)
    p.add_argument("--align-script", default=str(_DEFAULT_ALIGN_SCRIPT))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", default="results/motif_validation")
    return p.parse_args()


def _res(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_DIR / p


@torch.no_grad()
def _load_rbp_names() -> list[str]:
    path = PROJECT_DIR / "results" / "globalclip" / "datasets" / "rbp_names.txt"
    return path.read_text().strip().split("\n")


def integrated_gradients_for_track(parnet, seq_onehot: torch.Tensor, track_idx: int, n_steps: int) -> np.ndarray:
    """IG attribution of a single PARNET RBP track's output w.r.t. input sequence.

    Args:
        parnet:     Frozen PARNET backbone.
        seq_onehot: (1, 4, L) one-hot sequence.
        track_idx:  Index into the 223 RBP tracks to attribute.
        n_steps:    Number of interpolation steps.

    Returns:
        (L,) attribution scores, summed over the 4 nucleotide channels.
    """
    baseline = torch.zeros_like(seq_onehot)
    alphas = torch.linspace(0, 1, n_steps, device=seq_onehot.device).view(n_steps, 1, 1)
    interpolated = (baseline + alphas * (seq_onehot - baseline)).squeeze(1)  # (n_steps, 4, L)
    interpolated.requires_grad_(True)

    _, rbp_tracks = _extract_parnet_features(parnet, interpolated)  # grad flows despite no_grad decorator absent here
    target = rbp_tracks[:, track_idx, :].sum()
    grads, = torch.autograd.grad(target, interpolated)

    avg_grad = grads.mean(0)                                    # (4, L)
    attribution = avg_grad * (seq_onehot.squeeze(0) - baseline.squeeze(0))
    return attribution.sum(0).detach().cpu().numpy()             # (L,)


def onehot_to_string(onehot: np.ndarray) -> str:
    idx = onehot.argmax(axis=0)
    return "".join(NUCLEOTIDE_ARR[idx])


def load_motif_ppm(f: Path):
    with open(f) as fh:
        ms = motifs_lib.read_motifs_transfac(fh)
    if not ms:
        return None
    m = ms[0]
    mt = f.stem.split(".")[1].upper()
    m.matrix_type = mt
    try:
        if mt in ("PCM", "PPM"):
            m = convert_matrix_type(m, "PPM")
        else:
            return None
    except Exception:
        return None
    return m


def load_all_rbp_motifs() -> dict:
    all_motifs: dict[str, list] = {}
    for db in DATABASES:
        preproc = MOTIFS_DIR / db / "preprocessed"
        if not preproc.exists():
            continue
        for rbp_dir in preproc.iterdir():
            if not rbp_dir.is_dir():
                continue
            rbp = rbp_dir.name
            for f in rbp_dir.glob("*.transfac"):
                m = load_motif_ppm(f)
                if m is None:
                    continue
                all_motifs.setdefault(rbp, []).append(m.matrix.to_numpy(dtype=float))
    return all_motifs


def best_pcc(query_ppm: np.ndarray, target_ppm: np.ndarray, min_overlap_frac: float = 0.8) -> float:
    L1, L2 = len(query_ppm), len(target_ppm)
    required_overlap = int(np.ceil(min_overlap_frac * min(L1, L2)))
    best = -np.inf
    for shift in range(-(L2 - required_overlap), L1 - required_overlap + 1):
        s1 = max(0, shift)
        e1 = min(L1, shift + L2)
        s2 = s1 - shift
        e2 = s2 + (e1 - s1)
        if e1 - s1 < required_overlap:
            continue
        a = query_ppm[s1:e1].flatten()
        b = target_ppm[s2:e2].flatten()
        if a.std() == 0 or b.std() == 0:
            continue
        r, _ = pearsonr(a, b)
        if np.isfinite(r):
            best = max(best, r)
    return best if best > -np.inf else np.nan


def load_consensus_ppm(tsv_path: Path) -> np.ndarray:
    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    mat = df[["A", "C", "G", "U"]].to_numpy(dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums


def validate_protein(
    protein: str, track_idx: int, parnet, test_ds, device, args, out_root: Path, all_motifs: dict,
) -> dict:
    print(f"\n=== {protein} (track {track_idx}) ===")
    out_dir = out_root / protein
    metamotif_dir = out_dir / "metamotif"
    metamotif_dir.mkdir(parents=True, exist_ok=True)

    n_sequences = len(test_ds) if args.max_sequences is None else min(args.max_sequences, len(test_ds))

    importances, seq_strings = [], []
    for i in tqdm(range(n_sequences), desc=protein):
        sample = test_ds[i]
        seq = sample["sequence"].unsqueeze(0).to(device)
        attr = integrated_gradients_for_track(parnet, seq, track_idx, args.n_steps)
        importances.append(attr)
        seq_strings.append(onehot_to_string(sample["sequence"].numpy()))

    importance_scores = np.stack(importances)  # (n_seq, L)

    fasta_path = metamotif_dir / "sequences.fasta"
    scores_path = metamotif_dir / "scores.npy"
    with open(fasta_path, "w") as fh:
        for i, seq in enumerate(seq_strings):
            fh.write(f">seq_{i}\n{seq}\n")
    np.save(scores_path, importance_scores)
    print(f"  Wrote {n_sequences} sequences + scores -> {metamotif_dir}")

    search_config = metamotif_dir / "search.config.gin"
    search_config.write_text(
        f"search.seed_size = {args.seed_size}\n"
        f"search.max_size = {args.max_size}\n"
        f"search.sig_p = {args.sig_p}\n"
    )

    kmers_path = metamotif_dir / "motifs.kmers.tsv"
    cmd = ["metamotif", "search", str(fasta_path), str(scores_path),
           "--config", str(search_config), "-o", str(kmers_path)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  metamotif search FAILED:\n{result.stderr}")
        return {"protein": protein, "status": "search_failed", "matches": []}

    motifs_out_dir = metamotif_dir / "aligned_motifs"
    if motifs_out_dir.exists():
        shutil.rmtree(motifs_out_dir)
    motifs_out_dir.mkdir(parents=True, exist_ok=True)

    align_cmd = ["python", args.align_script, str(kmers_path),
                 "--min-support", str(args.min_support), "--max-motifs", str(args.max_motifs),
                 "-o", str(motifs_out_dir)]
    print(f"  Running: {' '.join(align_cmd)}")
    result = subprocess.run(align_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Alignment FAILED:\n{result.stderr}")
        return {"protein": protein, "status": "alignment_failed", "matches": []}

    consensus_motifs = {}
    for tsv in sorted(motifs_out_dir.glob("motif-*.tsv")):
        consensus_motifs[tsv.stem] = load_consensus_ppm(tsv)

    match_results = {}
    for motif_name, query_ppm in consensus_motifs.items():
        rbp_best = {}
        for rbp, motif_list in all_motifs.items():
            best = max((best_pcc(query_ppm, m) for m in motif_list), default=np.nan)
            if np.isfinite(best):
                rbp_best[rbp] = best
        top_rbps = sorted(rbp_best.items(), key=lambda x: x[1], reverse=True)[:10]
        match_results[motif_name] = top_rbps

    match_out = {
        motif: [{"rbp": rbp, "pcc": round(float(score), 4)} for rbp, score in top_rbps]
        for motif, top_rbps in match_results.items()
    }
    (metamotif_dir / "rbp_matches.json").write_text(json.dumps(match_out, indent=2))

    self_match = None
    for motif_name, top_rbps in match_results.items():
        for rbp, score in top_rbps:
            if protein.split("_")[0].upper() in rbp.upper():
                self_match = {"motif": motif_name, "rbp": rbp, "pcc": round(float(score), 4)}
                break
        if self_match:
            break

    print(f"  Consensus motifs: {list(consensus_motifs.keys())}")
    print(f"  Self-match (own protein name in top-10): {self_match}")

    return {
        "protein": protein,
        "status": "ok",
        "n_consensus_motifs": len(consensus_motifs),
        "self_match": self_match,
        "matches": match_out,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    _fp_cfg = yaml.safe_load((PROJECT_DIR / "config" / "filepaths.server.yaml").read_text())
    pretrained_model_name = ParnetModelName.PARNET_7M_0_0
    pretrained_path = _res(_fp_cfg["models"][pretrained_model_name.value])

    parnet = load_parnet_model(pretrained_model_name, pretrained_path, dtype=torch.float32, device=device)
    parnet.eval()
    # NOTE: unlike other scripts, we do NOT wrap the backbone forward in
    # torch.no_grad() here, since IG needs gradients w.r.t. the input.

    dataset_path = _res(_fp_cfg["data"][args.dataset]["pt"])
    test_ds = GlobalCLIPDataset(dataset_path, split="test", seq_len=600, total_key="globalCLIP")

    rbp_names = _load_rbp_names()
    name_to_idx = {name: i for i, name in enumerate(rbp_names)}

    out_root = PROJECT_DIR / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading RBP motif databases...")
    all_motifs = load_all_rbp_motifs()
    print(f"Loaded motifs for {len(all_motifs)} RBPs")

    summary = []
    for protein in args.proteins:
        if protein not in name_to_idx:
            print(f"WARNING: '{protein}' not found in rbp_names.txt, skipping.")
            continue
        result = validate_protein(
            protein, name_to_idx[protein], parnet, test_ds, device, args, out_root, all_motifs,
        )
        summary.append(result)

    (out_root / "00_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out_root / '00_summary.json'}")
    for r in summary:
        print(f"  {r['protein']}: status={r['status']}, self_match={r.get('self_match')}")


if __name__ == "__main__":
    main()
