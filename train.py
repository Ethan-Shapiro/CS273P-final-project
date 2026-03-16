"""
ISIC 2024 Skin Cancer Detection — Multimodal (Image + Tabular) PyTorch Model

Combines a pretrained EfficientNet-B0 image encoder with a tabular feature
MLP, fusing both modalities before a final classification head.

Based on two Kaggle notebooks:
  - Image-only:  isic-pytorch-training-baseline-image-only.ipynb
  - Tabular-only: isic-2024-only-tabular-data.ipynb
"""

import os
import gc
import cv2
import copy
import time
import random
import itertools
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "seed": 42,
    "epochs": 15,
    "img_size": 224,
    "model_name": "tf_efficientnet_b0_ns",
    "num_workers": 2,
    "train_batch_size": 32,
    "valid_batch_size": 64,
    "learning_rate": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "min_lr": 1e-6,
    "weight_decay": 1e-6,
    "n_accumulate": 1,
    "n_fold": 5,
    "fold": 0,
    "pos_neg_ratio": 20,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    # Paths — adjust these to match your local layout
    "data_dir": Path("data"),
    "train_metadata": Path("data/train-metadata.csv"),
    "train_image_dir": Path("data/train-image/image"),
    "output_dir": Path("output"),
}

CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(CONFIG["seed"])

# ---------------------------------------------------------------------------
# Tabular feature engineering (ported from the tabular-only notebook)
# ---------------------------------------------------------------------------
ERR = 1e-5

NUM_COLS = [
    "age_approx",
    "clin_size_long_diam_mm",
    "tbp_lv_A", "tbp_lv_Aext",
    "tbp_lv_B", "tbp_lv_Bext",
    "tbp_lv_C", "tbp_lv_Cext",
    "tbp_lv_H", "tbp_lv_Hext",
    "tbp_lv_L", "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL",
    "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",
    "tbp_lv_norm_border", "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL", "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle",
    "tbp_lv_x", "tbp_lv_y", "tbp_lv_z",
]

NEW_NUM_COLS = [
    "lesion_size_ratio",
    "lesion_shape_index",
    "hue_contrast",
    "luminance_contrast",
    "lesion_color_difference",
    "border_complexity",
    "color_uniformity",
    "position_distance_3d",
    "perimeter_to_area_ratio",
    "area_to_perimeter_ratio",
    "lesion_visibility_score",
    "symmetry_border_consistency",
    "consistency_symmetry_border",
    "color_consistency",
    "consistency_color",
    "size_age_interaction",
    "hue_color_std_interaction",
    "lesion_severity_index",
    "shape_complexity_index",
    "color_contrast_index",
    "log_lesion_area",
    "normalized_lesion_size",
    "mean_hue_difference",
    "std_dev_contrast",
    "color_shape_composite_index",
    "lesion_orientation_3d",
    "overall_color_difference",
    "symmetry_perimeter_interaction",
    "comprehensive_lesion_index",
    "color_variance_ratio",
    "border_color_interaction",
    "border_color_interaction_2",
    "size_color_contrast_ratio",
    "age_normalized_nevi_confidence",
    "age_normalized_nevi_confidence_2",
    "color_asymmetry_index",
    "volume_approximation_3d",
    "color_range",
    "shape_color_consistency",
    "border_length_ratio",
    "age_size_symmetry_index",
    "index_age_size_symmetry",
]

CAT_COLS = [
    "sex",
    "anatom_site_general",
    "tbp_tile_type",
    "tbp_lv_location",
    "tbp_lv_location_simple",
    "attribution",
]


def engineer_tabular_features(path: Path) -> pd.DataFrame:
    """Read CSV with Polars and compute all engineered features, return pandas DF."""
    df = (
        pl.read_csv(str(path))
        .with_columns(
            pl.col("age_approx").cast(pl.String).replace("NA", None).cast(pl.Float64),
        )
        .with_columns(
            pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median()),
        )
        .with_columns(
            pl.col(pl.Float64).fill_null(pl.col(pl.Float64).median()),
        )
        # --- first batch of derived features ---
        .with_columns(
            lesion_size_ratio=(pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm")),
            lesion_shape_index=(pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2)),
            hue_contrast=((pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs()),
            luminance_contrast=((pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()),
            lesion_color_difference=(
                (pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2).sqrt()
            ),
            border_complexity=(pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis")),
            color_uniformity=(pl.col("tbp_lv_color_std_mean") / (pl.col("tbp_lv_radial_color_std_max") + ERR)),
        )
        .with_columns(
            position_distance_3d=((pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt()),
            perimeter_to_area_ratio=(pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2")),
            area_to_perimeter_ratio=(pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM")),
            lesion_visibility_score=(pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color")),
            symmetry_border_consistency=(pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border")),
            consistency_symmetry_border=(
                pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border"))
            ),
        )
        .with_columns(
            color_consistency=(pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext")),
            consistency_color=(
                pl.col("tbp_lv_stdL") * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext"))
            ),
            size_age_interaction=(pl.col("clin_size_long_diam_mm") * pl.col("age_approx")),
            hue_color_std_interaction=(pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean")),
            lesion_severity_index=(
                (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")) / 3
            ),
            shape_complexity_index=(pl.col("border_complexity") + pl.col("lesion_shape_index")),
            color_contrast_index=(
                pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL") + pl.col("tbp_lv_deltaLBnorm")
            ),
        )
        .with_columns(
            log_lesion_area=((pl.col("tbp_lv_areaMM2") + 1).log()),
            normalized_lesion_size=(pl.col("clin_size_long_diam_mm") / pl.col("age_approx")),
            mean_hue_difference=((pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2),
            std_dev_contrast=(
                ((pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2) / 3).sqrt()
            ),
            color_shape_composite_index=(
                (pl.col("tbp_lv_color_std_mean") + pl.col("tbp_lv_area_perim_ratio") + pl.col("tbp_lv_symm_2axis")) / 3
            ),
            lesion_orientation_3d=(pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x"))),
            overall_color_difference=(
                (pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")) / 3
            ),
        )
        .with_columns(
            symmetry_perimeter_interaction=(pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM")),
            comprehensive_lesion_index=(
                (pl.col("tbp_lv_area_perim_ratio") + pl.col("tbp_lv_eccentricity")
                 + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_symm_2axis")) / 4
            ),
            color_variance_ratio=(pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt")),
            border_color_interaction=(pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color")),
            border_color_interaction_2=(
                pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color"))
            ),
            size_color_contrast_ratio=(pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm")),
            age_normalized_nevi_confidence=(pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx")),
            age_normalized_nevi_confidence_2=(
                (pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2).sqrt()
            ),
            color_asymmetry_index=(pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis")),
        )
        .with_columns(
            volume_approximation_3d=(
                pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt()
            ),
            color_range=(
                (pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs()
            ),
            shape_color_consistency=(pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean")),
            border_length_ratio=(
                pl.col("tbp_lv_perimeterMM") / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt())
            ),
            age_size_symmetry_index=(
                pl.col("age_approx") * pl.col("clin_size_long_diam_mm") * pl.col("tbp_lv_symm_2axis")
            ),
            index_age_size_symmetry=(
                pl.col("age_approx") * pl.col("tbp_lv_areaMM2") * pl.col("tbp_lv_symm_2axis")
            ),
        )
    )

    return df.to_pandas()


# Which numeric columns to feed the tabular branch (raw + engineered)
TABULAR_FEATURE_COLS = NUM_COLS + NEW_NUM_COLS


def build_tabular_tensor(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Extract tabular features as a float32 numpy array, filling NaN with 0."""
    arr = df[feature_cols].values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ---------------------------------------------------------------------------
# Image augmentations (ported from image-only notebook)
# ---------------------------------------------------------------------------
data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Downscale(scale_range=(0.25, 0.25), p=0.25),
        A.Affine(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ], p=1.0),

    "valid": A.Compose([
        A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ], p=1.0),
}

# ---------------------------------------------------------------------------
# Dataset — returns image tensor, tabular tensor, and target
# ---------------------------------------------------------------------------
class ISICMultimodalDataset(Dataset):
    """Balanced training dataset: randomly samples a positive or negative
    example each call so the model sees ~50/50 class balance."""

    def __init__(self, df, tabular_array, transforms=None, is_training=True):
        self.transforms = transforms
        self.is_training = is_training

        if is_training:
            pos_mask = df["target"].values == 1
            neg_mask = ~pos_mask
            self.idx_pos = np.where(pos_mask)[0]
            self.idx_neg = np.where(neg_mask)[0]

        self.file_paths = df["file_path"].values
        self.targets = df["target"].values.astype(np.float32)
        self.tabular = tabular_array  # (N, n_features) float32

    def __len__(self):
        if self.is_training:
            return len(self.idx_pos) * 2
        return len(self.targets)

    def __getitem__(self, index):
        if self.is_training:
            if random.random() >= 0.5:
                index = self.idx_pos[index % len(self.idx_pos)]
            else:
                index = self.idx_neg[index % len(self.idx_neg)]

        img_path = self.file_paths[index]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((CONFIG["img_size"], CONFIG["img_size"], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        tab = torch.tensor(self.tabular[index], dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.float32)

        return img, tab, target


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class GeM(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)


class ISICMultimodalModel(nn.Module):
    """
    Multimodal architecture:
      Image branch  → EfficientNet-B0 + GeM → 1280-d embedding
      Tabular branch → MLP → 128-d embedding
      Fusion         → concat → MLP → sigmoid → 1
    """

    def __init__(self, model_name, n_tabular_features, pretrained=True, checkpoint_path=None):
        super().__init__()

        # --- Image encoder ---
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, checkpoint_path=checkpoint_path,
        )
        img_feat_dim = self.backbone.classifier.in_features  # 1280 for effnet-b0
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pool = GeM()

        # --- Tabular encoder ---
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

        # --- Fusion head ---
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
        # Image branch
        img_features = self.backbone(images)
        img_features = self.pool(img_features).flatten(1)   # (B, 1280)

        # Tabular branch
        tab_features = self.tab_mlp(tabular)                # (B, 128)

        # Fusion
        combined = torch.cat([img_features, tab_features], dim=1)
        return self.head(combined).squeeze(1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def criterion(outputs, targets):
    return nn.BCELoss()(outputs, targets)


# ---------------------------------------------------------------------------
# Competition metric: partial AUC (pAUC at 80 % TPR)
# ---------------------------------------------------------------------------
def compute_pauc(y_true, y_pred, min_tpr=0.80):
    max_fpr = 1.0 - min_tpr
    v_gt = np.abs(y_true - 1)
    v_pred = 1.0 - y_pred
    try:
        partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    except ValueError:
        return 0.0
    partial_auc = (
        0.5 * max_fpr ** 2
        + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    )
    return partial_auc


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    running_loss = 0.0
    dataset_size = 0
    all_targets, all_outputs = [], []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, tabular, targets) in bar:
        images = images.to(device, dtype=torch.float)
        tabular = tabular.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        batch_size = images.size(0)

        outputs = model(images, tabular)
        loss = criterion(outputs, targets) / CONFIG["n_accumulate"]
        loss.backward()

        if (step + 1) % CONFIG["n_accumulate"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size * CONFIG["n_accumulate"]
        dataset_size += batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Loss=f"{epoch_loss:.4f}",
                        LR=f"{optimizer.param_groups[0]['lr']:.2e}")

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    epoch_auc = roc_auc_score(all_targets, all_outputs) if len(np.unique(all_targets)) > 1 else 0.0
    epoch_pauc = compute_pauc(all_targets, all_outputs)

    gc.collect()
    return epoch_loss, epoch_auc, epoch_pauc


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_targets, all_outputs = [], []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, tabular, targets) in bar:
        images = images.to(device, dtype=torch.float)
        tabular = tabular.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        batch_size = images.size(0)

        outputs = model(images, tabular)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        all_targets.append(targets.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())

        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Val_Loss=f"{epoch_loss:.4f}")

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    epoch_auc = roc_auc_score(all_targets, all_outputs) if len(np.unique(all_targets)) > 1 else 0.0
    epoch_pauc = compute_pauc(all_targets, all_outputs)

    gc.collect()
    return epoch_loss, epoch_auc, epoch_pauc


# ---------------------------------------------------------------------------
# Full training driver
# ---------------------------------------------------------------------------
def run_training(model, optimizer, scheduler, train_loader, valid_loader, device, num_epochs, save_dir=None):
    if save_dir is None:
        save_dir = CONFIG["output_dir"]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("[INFO] Using CPU")

    start = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_val_auc = -np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_loss, train_auc, train_pauc = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch,
        )
        val_loss, val_auc, val_pauc = valid_one_epoch(
            model, valid_loader, device, epoch,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["train_pauc"].append(train_pauc)
        history["val_pauc"].append(val_pauc)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  AUC: {train_auc:.4f}  pAUC: {train_pauc:.4f} | "
            f"Val Loss: {val_loss:.4f}  AUC: {val_auc:.4f}  pAUC: {val_pauc:.4f}"
        )

        if val_auc > best_val_auc:
            print(f"  >> Validation AUC improved ({best_val_auc:.4f} -> {val_auc:.4f}). Saving model...")
            best_val_auc = val_auc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        print()

    elapsed = time.time() - start
    print(f"Training complete in {elapsed // 3600:.0f}h {(elapsed % 3600) // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best Validation AUC: {best_val_auc:.4f}")

    model.load_state_dict(best_wts)
    return model, history


# ---------------------------------------------------------------------------
# Evaluation on held-out test split
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_test(model, test_loader, device):
    model.eval()
    all_targets, all_outputs = [], []
    for images, tabular, targets in tqdm(test_loader, desc="Testing"):
        images = images.to(device, dtype=torch.float)
        tabular = tabular.to(device, dtype=torch.float)
        outputs = model(images, tabular)
        all_targets.append(targets.numpy())
        all_outputs.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)

    auc = roc_auc_score(all_targets, all_outputs) if len(np.unique(all_targets)) > 1 else 0.0
    pauc = compute_pauc(all_targets, all_outputs)
    print(f"\n{'='*60}")
    print(f"TEST RESULTS  —  AUC: {auc:.4f}  |  pAUC@80%TPR: {pauc:.4f}")
    print(f"{'='*60}\n")
    return auc, pauc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ISIC 2024 — Multimodal (Image + Tabular) Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & engineer tabular features
    # ------------------------------------------------------------------
    print("\n[1/6] Loading metadata and engineering tabular features ...")
    df = engineer_tabular_features(CONFIG["train_metadata"])

    # Build image file paths
    df["file_path"] = df["isic_id"].apply(
        lambda x: str(CONFIG["train_image_dir"] / f"{x}.jpg")
    )

    # Keep only rows whose images actually exist on disk
    exists_mask = df["file_path"].apply(os.path.isfile)
    print(f"  Images found on disk: {exists_mask.sum()} / {len(df)}")
    df = df[exists_mask].reset_index(drop=True)

    if len(df) == 0:
        print("[ERROR] No images found. Check CONFIG['train_image_dir'] path.")
        return

    print(f"  Total samples: {len(df)}  |  Positive: {df['target'].sum()}  |  Negative: {(df['target'] == 0).sum()}")

    # ------------------------------------------------------------------
    # 2. Balance the dataset (keep all positives, subsample negatives)
    # ------------------------------------------------------------------
    print("\n[2/6] Balancing dataset ...")
    df_pos = df[df["target"] == 1].reset_index(drop=True)
    df_neg = df[df["target"] == 0].reset_index(drop=True)
    n_neg = min(len(df_neg), len(df_pos) * CONFIG["pos_neg_ratio"])
    df_neg = df_neg.sample(n=n_neg, random_state=CONFIG["seed"]).reset_index(drop=True)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    print(f"  Balanced: {len(df)} samples  (pos={len(df_pos)}, neg={n_neg})")

    # ------------------------------------------------------------------
    # 3. Split into train / val / test using StratifiedGroupKFold
    #    We use fold 0 val as the validation set and fold 1 val as the
    #    held-out test set; remaining folds are training data.
    # ------------------------------------------------------------------
    print("\n[3/6] Splitting data (StratifiedGroupKFold) ...")
    sgkf = StratifiedGroupKFold(n_splits=CONFIG["n_fold"], shuffle=True, random_state=CONFIG["seed"])
    df["kfold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["target"], df["patient_id"])):
        df.loc[val_idx, "kfold"] = fold_idx

    test_fold = 1
    val_fold = CONFIG["fold"]  # 0

    df_train = df[(df["kfold"] != val_fold) & (df["kfold"] != test_fold)].reset_index(drop=True)
    df_val   = df[df["kfold"] == val_fold].reset_index(drop=True)
    df_test  = df[df["kfold"] == test_fold].reset_index(drop=True)

    print(f"  Train: {len(df_train)}  |  Val: {len(df_val)}  |  Test: {len(df_test)}")

    # ------------------------------------------------------------------
    # 4. Build tabular feature arrays & normalise
    # ------------------------------------------------------------------
    print("\n[4/6] Preparing tabular features ...")
    tab_train = build_tabular_tensor(df_train, TABULAR_FEATURE_COLS)
    tab_val   = build_tabular_tensor(df_val, TABULAR_FEATURE_COLS)
    tab_test  = build_tabular_tensor(df_test, TABULAR_FEATURE_COLS)

    # Z-score normalisation fitted on training set
    tab_mean = tab_train.mean(axis=0)
    tab_std  = tab_train.std(axis=0) + 1e-8
    tab_train = (tab_train - tab_mean) / tab_std
    tab_val   = (tab_val - tab_mean)   / tab_std
    tab_test  = (tab_test - tab_mean)  / tab_std

    n_tab_features = tab_train.shape[1]
    print(f"  Tabular features: {n_tab_features}")

    # ------------------------------------------------------------------
    # 5. Create datasets & dataloaders
    # ------------------------------------------------------------------
    print("\n[5/6] Building dataloaders ...")
    train_ds = ISICMultimodalDataset(df_train, tab_train, transforms=data_transforms["train"], is_training=True)
    val_ds   = ISICMultimodalDataset(df_val, tab_val, transforms=data_transforms["valid"], is_training=False)
    test_ds  = ISICMultimodalDataset(df_test, tab_test, transforms=data_transforms["valid"], is_training=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["train_batch_size"],
                              num_workers=CONFIG["num_workers"], shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG["valid_batch_size"],
                              num_workers=CONFIG["num_workers"], shuffle=False,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=CONFIG["valid_batch_size"],
                              num_workers=CONFIG["num_workers"], shuffle=False,
                              pin_memory=True)

    # ------------------------------------------------------------------
    # 6. Build model, optimizer, scheduler → train → evaluate
    # ------------------------------------------------------------------
    print("\n[6/6] Initialising model ...")
    model = ISICMultimodalModel(
        model_name=CONFIG["model_name"],
        n_tabular_features=n_tab_features,
        pretrained=True,
    )
    model.to(CONFIG["device"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total  |  {trainable_params:,} trainable")

    T_max = (len(train_loader) * CONFIG["epochs"]) // CONFIG["n_accumulate"]
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=CONFIG["min_lr"])

    print("\nStarting training ...\n")
    model, history = run_training(
        model, optimizer, scheduler, train_loader, val_loader,
        CONFIG["device"], CONFIG["epochs"],
    )

    # --- Final evaluation on held-out test fold ---
    print("\nEvaluating on held-out test split ...")
    test_auc, test_pauc = evaluate_test(model, test_loader, CONFIG["device"])

    # Save final model
    final_path = CONFIG["output_dir"] / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    # Save training history + test results for visualization
    import json
    baseline_results = {
        "test_auc": float(test_auc),
        "test_pauc": float(test_pauc),
        "best_val_auc": float(max(history["val_auc"])),
        "best_val_pauc": float(max(history["val_pauc"])),
        "desc": "Baseline (EfficientNet + MLP + GeM)",
        "image_branch": "EfficientNet",
        "tabular_branch": "MLP",
        "pooling": "GeM",
    }
    history_out = {k: [float(v) for v in vals] for k, vals in history.items()}

    with open(CONFIG["output_dir"] / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    with open(CONFIG["output_dir"] / "baseline_history.json", "w") as f:
        json.dump(history_out, f, indent=2)
    print(f"Baseline history saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
