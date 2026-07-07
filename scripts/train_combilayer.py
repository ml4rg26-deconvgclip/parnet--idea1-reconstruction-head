"""Train GlobalCLIPHybridModel ("CombiLayer"): CNN-only path + QLayer
interference path, both fed to the same dilated CNN as separate channels.

Motivation: across every ablation so far (with/without interference, global
vs. positional phase), QLayer's interference pathway never beat the plain
CNN-only pathway on prediction accuracy. Instead of picking one or the
other, this model gives the CNN both signals and lets it learn how much (if
any) weight to give the interference channel -- it can never be worse than
CNN-only, while QLayer's phase parameters still get trained and remain
readable via get_coupling_matrix() for the protein-interaction analysis,
independent of whether the CNN ends up using them for prediction.

Usage:
    pixi run -e parnet-dev-cu12 python scripts/train_combilayer.py
    pixi run -e parnet-dev-cu12 python scripts/train_combilayer.py --positional-alpha --run-id v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
import lightning.pytorch as pl
from dotmap import DotMap
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR / "src"))

from parnet_additional_utils import ParnetModelName, load_parnet_model
from globalclip_utils import (
    GlobalCLIPDataset,
    GlobalCLIPLightningModule,
    GlobalCLIPHybridModel,
    save_run_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id",        default="globalclip.combilayer.v1")
    p.add_argument("--dataset",       default="globalclip_lysate_noNHS")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--max-epochs",    type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--mix-hidden",    type=int,   default=128)
    p.add_argument("--cnn-channels",  type=int,   default=64)
    p.add_argument("--cnn-kernel",    type=int,   default=9)
    p.add_argument("--cnn-layers",    type=int,   default=3)
    p.add_argument("--lambda-nll",    type=float, default=0.1)
    p.add_argument("--lambda-alpha",  type=float, default=0.1)
    p.add_argument("--lambda-phase",  type=float, default=0.01)
    p.add_argument("--patience",      type=int,   default=8)
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--seq-len",       type=int,   default=600)
    p.add_argument("--num-rbps",      type=int,   default=223)
    p.add_argument("--positional-alpha", action="store_true")
    p.add_argument("--positional-phase", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        log.info(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        log.warning("No GPU — running on CPU.")

    _fp_cfg = yaml.safe_load((PROJECT_DIR / "config" / "filepaths.server.yaml").read_text())
    pretrained_model_name = ParnetModelName.PARNET_7M_0_0

    def _res(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else PROJECT_DIR / p

    fp = DotMap()
    fp.pretrained_model = _res(_fp_cfg["models"][pretrained_model_name.value])
    fp.dataset          = _res(_fp_cfg["data"][args.dataset]["pt"])
    fp.output_dir       = PROJECT_DIR / "results" / "globalclip" / "combilayer" / args.run_id
    fp.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Run ID          : {args.run_id}")
    log.info(f"Dataset         : {fp.dataset}")
    log.info(f"Output dir      : {fp.output_dir}")
    log.info(f"Learning rate   : {args.lr}")
    log.info(f"Positional alpha: {args.positional_alpha}")
    log.info(f"Positional phase: {args.positional_phase}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = GlobalCLIPDataset(fp.dataset, split="train",
                                 seq_len=args.seq_len, total_key="globalCLIP")
    val_ds   = GlobalCLIPDataset(fp.dataset, split="valid",
                                 seq_len=args.seq_len, total_key="globalCLIP")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    log.info(f"Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    log.info(f"Valid: {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info(f"Loading PARNET from {fp.pretrained_model}")
    parnet = load_parnet_model(pretrained_model_name, fp.pretrained_model,
                               dtype=torch.float32, device=device)
    parnet.eval()

    model = GlobalCLIPHybridModel(
        parnet_model=parnet,
        num_rbps=args.num_rbps,
        mix_hidden=args.mix_hidden,
        cnn_channels=args.cnn_channels,
        cnn_kernel=args.cnn_kernel,
        cnn_layers=args.cnn_layers,
        positional_alpha=args.positional_alpha,
        positional_phase=args.positional_phase,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {trainable:,} trainable / {total:,} total")
    log.info(f"  MixCoeffHead : {sum(p.numel() for p in model.mix_coeff.parameters()):,}")
    log.info(f"  CNN (2ch in) : {sum(p.numel() for p in model.cnn.parameters()):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    lightning_model = GlobalCLIPLightningModule(
        model=model,
        lr=args.lr,
        lambda_nll=args.lambda_nll,
        lambda_alpha=args.lambda_alpha,
        lambda_phase=args.lambda_phase,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=fp.output_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss", mode="min", save_top_k=2,
        ),
        EarlyStopping(monitor="val/loss", patience=args.patience, mode="min"),
    ]
    loggers = [
        CSVLogger(str(fp.output_dir), name="csv_logs"),
        TensorBoardLogger(str(fp.output_dir), name="tb_logs"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.gpu] if torch.cuda.is_available() else 1,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=50,
        deterministic=True,
    )

    log.info("Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)
    log.info("Training complete.")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), fp.output_dir / "model.statedict.pt")
    torch.save(model, fp.output_dir / "model.full.pt")

    log.info(f"Channel weight summary: {model.channel_weight_summary()}")

    run_cfg = {
        "model_type":              "GlobalCLIPHybridModel",
        "pretrained_model_name":   pretrained_model_name.value,
        "control_dataset":         args.dataset,
        "params_seq_length":       args.seq_len,
        "params_batch_size":       args.batch_size,
        "params_num_rbps":         args.num_rbps,
        "params_mix_hidden":       args.mix_hidden,
        "params_cnn_channels":     args.cnn_channels,
        "params_cnn_kernel":       args.cnn_kernel,
        "params_cnn_layers":       args.cnn_layers,
        "params_positional_alpha": args.positional_alpha,
        "params_positional_phase": args.positional_phase,
        "params_lr":               args.lr,
        "params_max_epochs":       args.max_epochs,
        "params_lambda_nll":       args.lambda_nll,
        "params_lambda_alpha":     args.lambda_alpha,
        "params_lambda_phase":     args.lambda_phase,
        "dataset_path":            str(fp.dataset),
        "output_dir":              str(fp.output_dir),
    }
    save_run_config(fp.output_dir, run_cfg)
    log.info(f"Model and config saved to {fp.output_dir}")

    # ── Training curves ───────────────────────────────────────────────────────
    csv_log_dir = fp.output_dir / "csv_logs"
    metrics_paths = sorted(csv_log_dir.glob("version_*/metrics.csv"))
    if metrics_paths:
        df = pd.read_csv(metrics_paths[-1])
        epoch_df = df.groupby("epoch").last().reset_index()

        plots = [
            ({"train": "train/loss_epoch", "val": "val/loss"},       "Total loss"),
            ({"train": "train/pearson_epoch", "val": "val/pearson"}, "Pearson loss"),
            ({"phase_reg": "val/phase_reg"},                          "Phase L2 reg"),
        ]
        fig, axes = plt.subplots(1, len(plots), figsize=(15, 4))
        for (col_dict, title), ax in zip(plots, axes):
            for label, col in col_dict.items():
                if col in epoch_df.columns:
                    epoch_df.plot("epoch", col, ax=ax, label=label, marker="o", markersize=3)
            ax.set_title(title)
            ax.legend(fontsize=8)
        plt.suptitle(f"Training metrics — {args.run_id}")
        plt.tight_layout()
        plt.savefig(fp.output_dir / "training_curves.png", dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"Training curves saved to {fp.output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
