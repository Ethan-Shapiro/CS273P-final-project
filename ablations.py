"""
ISIC 2024 — Ablation Study

Runs four ablation experiments (A-D) that systematically disable or swap
components of the full multimodal baseline, then saves results and plots.

Experiments:
  Baseline : EfficientNet + MLP (tabular) + GeM   [already trained via train.py]
  Ablation A: Tabular MLP only (no image branch)
  Ablation B: EfficientNet + GeM only (no tabular branch)
  Ablation C: EfficientNet + MLP + AvgPool (GeM replaced with global avg pool)
  Ablation D: EfficientNet + MLP + GeM, but backbone is *frozen*

Usage:
    python ablations.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import (
    CONFIG, set_seed, engineer_tabular_features, TABULAR_FEATURE_COLS,
    build_tabular_tensor, data_transforms, ISICMultimodalDataset,
    GeM, criterion, compute_pauc,
    train_one_epoch, valid_one_epoch, run_training, evaluate_test,
)

import os, gc, copy, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import timm

# ---------------------------------------------------------------------------
# Model variants for each ablation
# ---------------------------------------------------------------------------

class TabularOnlyModel(nn.Module):
    """Ablation A — tabular MLP only, no image branch."""

    def __init__(self, n_tabular_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, tabular):
        return self.mlp(tabular).squeeze(1)


class ImageOnlyModel(nn.Module):
    """Ablation B — EfficientNet + GeM, no tabular branch."""

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        img_feat_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pool = GeM()
        self.head = nn.Sequential(
            nn.Linear(img_feat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, tabular):
        features = self.backbone(images)
        features = self.pool(features).flatten(1)
        return self.head(features).squeeze(1)


class MultimodalAvgPoolModel(nn.Module):
    """Ablation C — full multimodal, but AvgPool instead of GeM."""

    def __init__(self, model_name, n_tabular_features, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        img_feat_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        fusion_dim = img_feat_dim + 128
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, tabular):
        img_features = self.backbone(images)
        img_features = self.pool(img_features).flatten(1)
        tab_features = self.tab_mlp(tabular)
        combined = torch.cat([img_features, tab_features], dim=1)
        return self.head(combined).squeeze(1)


class MultimodalFrozenBackboneModel(nn.Module):
    """Ablation D — full multimodal with GeM, but backbone weights are frozen."""

    def __init__(self, model_name, n_tabular_features, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        img_feat_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.pool = GeM()

        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        fusion_dim = img_feat_dim + 128
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, tabular):
        with torch.no_grad():
            img_features = self.backbone(images)
        img_features = self.pool(img_features).flatten(1)
        tab_features = self.tab_mlp(tabular)
        combined = torch.cat([img_features, tab_features], dim=1)
        return self.head(combined).squeeze(1)


# ---------------------------------------------------------------------------
# Ablation registry
# ---------------------------------------------------------------------------
ABLATIONS = {
    "Ablation_A": {
        "desc": "Tabular Only (MLP)",
        "image_branch": "None",
        "tabular_branch": "MLP",
        "pooling": "N/A",
    },
    "Ablation_B": {
        "desc": "Image Only (EfficientNet + GeM)",
        "image_branch": "EfficientNet",
        "tabular_branch": "None",
        "pooling": "GeM",
    },
    "Ablation_C": {
        "desc": "Multimodal with AvgPool",
        "image_branch": "EfficientNet",
        "tabular_branch": "MLP",
        "pooling": "AvgPool",
    },
    "Ablation_D": {
        "desc": "Multimodal with Frozen Backbone",
        "image_branch": "EfficientNet (frozen)",
        "tabular_branch": "MLP",
        "pooling": "GeM",
    },
}


def build_model(name, n_tab_features):
    """Instantiate the right model variant for the given ablation name."""
    if name == "Ablation_A":
        return TabularOnlyModel(n_tab_features)
    elif name == "Ablation_B":
        return ImageOnlyModel(CONFIG["model_name"], pretrained=True)
    elif name == "Ablation_C":
        return MultimodalAvgPoolModel(CONFIG["model_name"], n_tab_features, pretrained=True)
    elif name == "Ablation_D":
        return MultimodalFrozenBackboneModel(CONFIG["model_name"], n_tab_features, pretrained=True)
    else:
        raise ValueError(f"Unknown ablation: {name}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_training_curves(all_histories, output_dir):
    """Per-experiment training curves: loss and AUC over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for name, hist in all_histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["val_loss"], marker="o", markersize=3, label=name)
        axes[1].plot(epochs, hist["val_auc"], marker="o", markersize=3, label=name)
        axes[2].plot(epochs, hist["val_pauc"], marker="o", markersize=3, label=name)

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Validation pAUC@80%TPR")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("pAUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "ablation_training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {path}")


def plot_test_comparison(results, output_dir):
    """Bar chart comparing final test AUC and pAUC across experiments."""
    names = list(results.keys())
    aucs = [results[n]["test_auc"] for n in names]
    paucs = [results[n]["test_pauc"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, aucs, width, label="AUC", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, paucs, width, label="pAUC@80%TPR", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Test Set Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = output_dir / "ablation_test_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved test comparison → {path}")


def print_results_table(results):
    """Pretty-print a summary table to the console."""
    header = f"{'Experiment':<20} {'Image Branch':<22} {'Tabular Branch':<18} {'Pooling':<10} {'Test AUC':>10} {'Test pAUC':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        info = ABLATIONS.get(name, {})
        print(
            f"{name:<20} "
            f"{info.get('image_branch', 'EfficientNet'):<22} "
            f"{info.get('tabular_branch', 'MLP'):<18} "
            f"{info.get('pooling', 'GeM'):<10} "
            f"{r['test_auc']:>10.4f} "
            f"{r['test_pauc']:>10.4f}"
        )
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = CONFIG["output_dir"] / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ISIC 2024 — Ablation Study")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data preparation (shared across all ablations)
    # ------------------------------------------------------------------
    print("\n[DATA] Loading metadata and engineering features ...")
    set_seed(CONFIG["seed"])
    df = engineer_tabular_features(CONFIG["train_metadata"])

    df["file_path"] = df["isic_id"].apply(
        lambda x: str(CONFIG["train_image_dir"] / f"{x}.jpg")
    )
    exists_mask = df["file_path"].apply(os.path.isfile)
    print(f"  Images on disk: {exists_mask.sum()} / {len(df)}")
    df = df[exists_mask].reset_index(drop=True)
    if len(df) == 0:
        print("[ERROR] No images found.")
        return

    # Balance
    df_pos = df[df["target"] == 1].reset_index(drop=True)
    df_neg = df[df["target"] == 0].reset_index(drop=True)
    n_neg = min(len(df_neg), len(df_pos) * CONFIG["pos_neg_ratio"])
    df_neg = df_neg.sample(n=n_neg, random_state=CONFIG["seed"]).reset_index(drop=True)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    print(f"  Balanced: {len(df)}  (pos={len(df_pos)}, neg={n_neg})")

    # Folds
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=CONFIG["n_fold"], shuffle=True, random_state=CONFIG["seed"])
    df["kfold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["target"], df["patient_id"])):
        df.loc[val_idx, "kfold"] = fold_idx

    val_fold, test_fold = 0, 1
    df_train = df[(df["kfold"] != val_fold) & (df["kfold"] != test_fold)].reset_index(drop=True)
    df_val   = df[df["kfold"] == val_fold].reset_index(drop=True)
    df_test  = df[df["kfold"] == test_fold].reset_index(drop=True)
    print(f"  Train: {len(df_train)}  |  Val: {len(df_val)}  |  Test: {len(df_test)}")

    # Tabular features
    tab_train = build_tabular_tensor(df_train, TABULAR_FEATURE_COLS)
    tab_val   = build_tabular_tensor(df_val, TABULAR_FEATURE_COLS)
    tab_test  = build_tabular_tensor(df_test, TABULAR_FEATURE_COLS)
    tab_mean = tab_train.mean(axis=0)
    tab_std  = tab_train.std(axis=0) + 1e-8
    tab_train = (tab_train - tab_mean) / tab_std
    tab_val   = (tab_val - tab_mean)   / tab_std
    tab_test  = (tab_test - tab_mean)  / tab_std
    n_tab_features = tab_train.shape[1]

    # Dataloaders
    train_ds = ISICMultimodalDataset(df_train, tab_train, transforms=data_transforms["train"], is_training=True)
    val_ds   = ISICMultimodalDataset(df_val, tab_val, transforms=data_transforms["valid"], is_training=False)
    test_ds  = ISICMultimodalDataset(df_test, tab_test, transforms=data_transforms["valid"], is_training=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["train_batch_size"],
                              num_workers=CONFIG["num_workers"], shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG["valid_batch_size"],
                              num_workers=CONFIG["num_workers"], shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=CONFIG["valid_batch_size"],
                              num_workers=CONFIG["num_workers"], shuffle=False, pin_memory=True)

    # ------------------------------------------------------------------
    # 2. Run each ablation
    # ------------------------------------------------------------------
    all_results = {}
    all_histories = {}

    for abl_name, abl_info in ABLATIONS.items():
        print("\n" + "#" * 60)
        print(f"# {abl_name}: {abl_info['desc']}")
        print("#" * 60)

        set_seed(CONFIG["seed"])

        model = build_model(abl_name, n_tab_features)
        model.to(CONFIG["device"])

        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {total_p:,} total | {train_p:,} trainable")

        T_max = (len(train_loader) * CONFIG["epochs"]) // CONFIG["n_accumulate"]
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"],
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=CONFIG["min_lr"])

        abl_save_dir = output_dir / abl_name

        model, history = run_training(
            model, optimizer, scheduler,
            train_loader, val_loader,
            CONFIG["device"], CONFIG["epochs"],
            save_dir=abl_save_dir,
        )

        test_auc, test_pauc = evaluate_test(model, test_loader, CONFIG["device"])

        all_results[abl_name] = {
            "test_auc": float(test_auc),
            "test_pauc": float(test_pauc),
            "best_val_auc": float(max(history["val_auc"])),
            "best_val_pauc": float(max(history["val_pauc"])),
            **abl_info,
        }
        all_histories[abl_name] = {k: [float(v) for v in vals] for k, vals in history.items()}

        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # 3. Save results & generate plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)

    print_results_table(all_results)

    results_path = output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {results_path}")

    histories_path = output_dir / "ablation_histories.json"
    with open(histories_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    print(f"Histories saved → {histories_path}")

    plot_training_curves(all_histories, output_dir)
    plot_test_comparison(all_results, output_dir)


if __name__ == "__main__":
    main()
