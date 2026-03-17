"""
ISIC 2024 — Visualization & Analysis

Generates all comparison plots for the report:
  1. Bar chart: test AUC & pAUC across all experiments
  2. Training curves (loss, AUC, pAUC) across all experiments
  3. Test ROC curves across baseline + all ablations
  4. Tabular feature importance (first-layer weights of tabular MLP)
  5. Image branch first-conv filter analysis (W^T W gram matrix)

This version is a safer replacement for the original visualize.py:
  - uses a safer test DataLoader for ROC plotting (num_workers=0, pin_memory=False)
  - adds step-by-step logging for the held-out test split rebuild
  - adds robust device selection (cuda / mps / cpu)
  - adds torch.load fallback for older PyTorch versions
  - makes test prediction collection more shape-safe and easier to debug

Usage:
    python visualize_fixed.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, roc_auc_score

from train import (
    CONFIG,
    ISICMultimodalModel,
    TABULAR_FEATURE_COLS,
    engineer_tabular_features,
    build_tabular_tensor,
    data_transforms,
    ISICMultimodalDataset,
)
from ablations import build_model

PLOTS_DIR = CONFIG["output_dir"] / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SHORT_NAMES = {
    col: col.replace("tbp_lv_", "").replace("_", " ")
    for col in TABULAR_FEATURE_COLS
}

COLORS = {
    "Baseline": "#2ecc71",
    "Ablation_A": "#3498db",
    "Ablation_B": "#e74c3c",
    "Ablation_C": "#9b59b6",
    "Ablation_D": "#f39c12",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def safe_torch_load(path):
    """Load a checkpoint compatibly across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_device():
    """Choose a valid runtime device from CONFIG with safe fallbacks."""
    requested = str(CONFIG.get("device", "cpu")).lower()

    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"

    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"

    return "cpu"


# ---------------------------------------------------------------------------
# 1. Load saved results
# ---------------------------------------------------------------------------
def load_results():
    """Return (all_results dict, all_histories dict) combining baseline + ablations."""
    all_results = {}
    all_histories = {}

    baseline_res_path = CONFIG["output_dir"] / "baseline_results.json"
    baseline_hist_path = CONFIG["output_dir"] / "baseline_history.json"
    if baseline_res_path.exists():
        with open(baseline_res_path) as f:
            all_results["Baseline"] = json.load(f)
    else:
        print(f"[WARN] {baseline_res_path} not found. Run `python train.py` first.")

    if baseline_hist_path.exists():
        with open(baseline_hist_path) as f:
            all_histories["Baseline"] = json.load(f)

    abl_res_path = CONFIG["output_dir"] / "ablations" / "ablation_results.json"
    abl_hist_path = CONFIG["output_dir"] / "ablations" / "ablation_histories.json"
    if abl_res_path.exists():
        with open(abl_res_path) as f:
            abl_results = json.load(f)
        all_results.update(abl_results)
    else:
        print(f"[WARN] {abl_res_path} not found. Run `python ablations.py` first.")

    if abl_hist_path.exists():
        with open(abl_hist_path) as f:
            abl_histories = json.load(f)
        all_histories.update(abl_histories)

    return all_results, all_histories


# ---------------------------------------------------------------------------
# 2. Test performance comparison — grouped bar chart
# ---------------------------------------------------------------------------
def plot_test_comparison(results):
    names = list(results.keys())
    aucs = [results[n]["test_auc"] for n in names]
    paucs = [results[n]["test_pauc"] for n in names]

    labels = []
    for n in names:
        desc = results[n].get("desc", n)
        labels.append(f"{n}\n({desc})" if desc != n else n)

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(
        x - width / 2,
        aucs,
        width,
        label="AUC",
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        paucs,
        width,
        label="pAUC @ 80% TPR",
        color="#DD8452",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Set Performance Across Experiments", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(aucs), max(paucs)) * 1.15)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    path = PLOTS_DIR / "test_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 3. Training curves — val loss, val AUC, val pAUC over epochs
# ---------------------------------------------------------------------------
def plot_training_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    metrics = [
        ("val_loss", "Validation Loss (BCE)", "lower is better"),
        ("val_auc", "Validation AUC", "higher is better"),
        ("val_pauc", "Validation pAUC @ 80% TPR", "higher is better"),
    ]

    for ax, (key, title, note) in zip(axes, metrics):
        for name, hist in histories.items():
            if key in hist:
                epochs = range(1, len(hist[key]) + 1)
                color = COLORS.get(name, None)
                ax.plot(
                    epochs,
                    hist[key],
                    marker="o",
                    markersize=3,
                    linewidth=1.8,
                    label=name,
                    color=color,
                )

        ax.set_title(f"{title}\n({note})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("val_", "").upper())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "training_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 4. Train vs Val curves for each experiment (overfitting analysis)
# ---------------------------------------------------------------------------
def plot_overfit_analysis(histories):
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9), squeeze=False)

    for col, (name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        color = COLORS.get(name, "#333")

        axes[0][col].plot(epochs, hist["train_loss"], label="Train", linewidth=1.5, color=color)
        axes[0][col].plot(epochs, hist["val_loss"], label="Val", linewidth=1.5, color=color, linestyle="--")
        axes[0][col].set_title(f"{name}\nLoss", fontsize=10, fontweight="bold")
        axes[0][col].legend(fontsize=8)
        axes[0][col].grid(True, alpha=0.3)
        axes[0][col].set_xlabel("Epoch")

        axes[1][col].plot(epochs, hist["train_auc"], label="Train", linewidth=1.5, color=color)
        axes[1][col].plot(epochs, hist["val_auc"], label="Val", linewidth=1.5, color=color, linestyle="--")
        axes[1][col].set_title(f"{name}\nAUC", fontsize=10, fontweight="bold")
        axes[1][col].legend(fontsize=8)
        axes[1][col].grid(True, alpha=0.3)
        axes[1][col].set_xlabel("Epoch")

    plt.suptitle("Train vs Validation (Overfitting Analysis)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = PLOTS_DIR / "overfit_analysis.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 5. Test ROC curves across baseline + all ablations
# ---------------------------------------------------------------------------
def build_shared_test_loader():
    """Recreate the held-out test split used by train.py / ablations.py."""
    print("  Rebuilding held-out test split ...")

    print("    [1/7] Engineering tabular features")
    df = engineer_tabular_features(CONFIG["train_metadata"])

    print("    [2/7] Building image file paths")
    df["file_path"] = df["isic_id"].apply(
        lambda x: str(CONFIG["train_image_dir"] / f"{x}.jpg")
    )

    print("    [3/7] Filtering rows with existing image files")
    exists_mask = df["file_path"].apply(os.path.isfile)
    df = df[exists_mask].reset_index(drop=True)
    print(f"         Rows after file check: {len(df)}")
    if len(df) == 0:
        raise FileNotFoundError("No images found on disk. Check CONFIG['train_image_dir'].")

    print("    [4/7] Rebuilding balanced positive/negative split")
    df_pos = df[df["target"] == 1].reset_index(drop=True)
    df_neg = df[df["target"] == 0].reset_index(drop=True)
    print(f"         Positives: {len(df_pos)} | Negatives before subsample: {len(df_neg)}")

    n_neg = min(len(df_neg), len(df_pos) * CONFIG["pos_neg_ratio"])
    df_neg = df_neg.sample(n=n_neg, random_state=CONFIG["seed"]).reset_index(drop=True)
    df = pd_concat([df_pos, df_neg]).reset_index(drop=True)
    print(f"         Negatives after subsample: {len(df_neg)} | Total balanced rows: {len(df)}")

    print("    [5/7] Rebuilding folds with StratifiedGroupKFold")
    sgkf = StratifiedGroupKFold(
        n_splits=CONFIG["n_fold"],
        shuffle=True,
        random_state=CONFIG["seed"],
    )
    df["kfold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["target"], df["patient_id"])):
        df.loc[val_idx, "kfold"] = fold_idx

    test_fold = 1
    val_fold = CONFIG["fold"]
    df_train = df[(df["kfold"] != val_fold) & (df["kfold"] != test_fold)].reset_index(drop=True)
    df_test = df[df["kfold"] == test_fold].reset_index(drop=True)
    print(f"         Train rows: {len(df_train)} | Test rows: {len(df_test)}")
    if len(df_test) == 0:
        raise RuntimeError("Held-out test split is empty after fold reconstruction.")

    print("    [6/7] Building standardized tabular tensors")
    tab_train = build_tabular_tensor(df_train, TABULAR_FEATURE_COLS)
    tab_test = build_tabular_tensor(df_test, TABULAR_FEATURE_COLS)

    tab_mean = tab_train.mean(axis=0)
    tab_std = tab_train.std(axis=0) + 1e-8
    tab_test = (tab_test - tab_mean) / tab_std

    print("    [7/7] Creating dataset and safe DataLoader")
    test_ds = ISICMultimodalDataset(
        df_test,
        tab_test,
        transforms=data_transforms["valid"],
        is_training=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["valid_batch_size"],
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    print(f"         Test dataset size: {len(test_ds)}")
    print(f"         Number of test batches: {len(test_loader)}")
    return test_loader, len(TABULAR_FEATURE_COLS)


def get_checkpoint_paths():
    """Return experiment -> checkpoint path for baseline and all ablations."""
    checkpoints = {}

    baseline_path = CONFIG["output_dir"] / "final_model.pth"
    if baseline_path.exists():
        checkpoints["Baseline"] = baseline_path

    abl_root = CONFIG["output_dir"] / "ablations"
    for name in ["Ablation_A", "Ablation_B", "Ablation_C", "Ablation_D"]:
        p = abl_root / name / "best_model.pth"
        if p.exists():
            checkpoints[name] = p

    return checkpoints


@torch.inference_mode()
def collect_test_predictions(model, test_loader, device, model_name="Model"):
    model.eval()
    all_targets, all_outputs = [], []

    print(f"    Running inference for {model_name} on device={device}")
    total_batches = len(test_loader)

    for batch_idx, (images, tabular, targets) in enumerate(test_loader, start=1):
        print(f"      Batch {batch_idx}/{total_batches}")

        images = images.to(device, dtype=torch.float, non_blocking=False)
        tabular = tabular.to(device, dtype=torch.float, non_blocking=False)

        outputs = model(images, tabular)

        outputs = outputs.detach().float().view(-1).cpu().numpy()
        targets = targets.detach().view(-1).cpu().numpy()

        all_targets.append(targets)
        all_outputs.append(outputs)

    if len(all_targets) == 0:
        raise RuntimeError(f"No predictions collected for {model_name}. Test loader produced zero batches.")

    y_true = np.concatenate(all_targets)
    y_score = np.concatenate(all_outputs)
    print(f"    Finished {model_name}: collected {len(y_true)} predictions")
    return y_true, y_score


def plot_test_roc_curves(results):
    checkpoints = get_checkpoint_paths()
    if not checkpoints:
        print("  Skipped — no model checkpoints found for ROC plotting.")
        return

    test_loader, n_tab_features = build_shared_test_loader()
    device = resolve_device()
    print(f"  Using device for ROC inference: {device}")

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    plotted_any = False

    for name in ["Baseline", "Ablation_A", "Ablation_B", "Ablation_C", "Ablation_D"]:
        ckpt_path = checkpoints.get(name)
        if ckpt_path is None:
            print(f"  [WARN] Missing checkpoint for {name}; skipping ROC curve.")
            continue

        print(f"  Loading checkpoint for {name}: {ckpt_path}")

        if name == "Baseline":
            model = ISICMultimodalModel(
                model_name=CONFIG["model_name"],
                n_tabular_features=n_tab_features,
                pretrained=False,
            )
        else:
            model = build_model(name, n_tab_features)

        try:
            state = safe_torch_load(ckpt_path)
            model.load_state_dict(state)
            model.to(device)

            y_true, y_score = collect_test_predictions(
                model,
                test_loader,
                device,
                model_name=name,
            )

            if len(np.unique(y_true)) < 2:
                print(f"  [WARN] ROC skipped for {name}; test labels contain a single class.")
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_value = results.get(name, {}).get("test_auc", roc_auc_score(y_true, y_score))
            label = f"{name} (AUC = {auc_value:.4f})"
            ax.plot(fpr, tpr, linewidth=2.0, label=label, color=COLORS.get(name, None))
            plotted_any = True
            print(f"  ROC complete for {name}")

        except Exception as e:
            print(f"  [ERROR] Failed on {name}: {type(e).__name__}: {e}")
            continue

    if not plotted_any:
        plt.close(fig)
        print("  Skipped — unable to plot any ROC curves.")
        return

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray", label="Random")
    ax.set_title("Held-Out Test ROC Curves Across Baseline and Ablations", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = PLOTS_DIR / "test_roc_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 6. Tabular feature importance from the MLP's first layer
# ---------------------------------------------------------------------------
def plot_tabular_feature_importance(model_path, n_tabular_features):
    """
    Computes importance of each tabular input feature by looking at the
    L2 norm of each column of the first linear layer: ||W[:, j]||_2.
    """
    model = ISICMultimodalModel(
        model_name=CONFIG["model_name"],
        n_tabular_features=n_tabular_features,
        pretrained=False,
    )
    state = safe_torch_load(model_path)
    model.load_state_dict(state)
    model.eval()

    W = model.tab_mlp[0].weight.detach().cpu().numpy()
    importance = np.linalg.norm(W, axis=0)

    feature_names = [FEATURE_SHORT_NAMES.get(c, c) for c in TABULAR_FEATURE_COLS]
    order = np.argsort(importance)[::-1]
    top_k = min(30, len(order))
    top_idx = order[:top_k]

    fig, ax = plt.subplots(figsize=(10, 9))
    y_pos = np.arange(top_k)
    ax.barh(y_pos, importance[top_idx][::-1], color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1], fontsize=9)
    ax.set_xlabel("Weight L2 Norm (||W[:, j]||)", fontsize=11)
    ax.set_title(f"Top {top_k} Tabular Feature Importances\n(First MLP Layer Weights)", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "tabular_feature_importance.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    import pandas as pd
    df_imp = pd.DataFrame({
        "feature": TABULAR_FEATURE_COLS,
        "feature_short": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    csv_path = PLOTS_DIR / "tabular_feature_importance.csv"
    df_imp.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")

    return importance


# ---------------------------------------------------------------------------
# 7. First-conv Gram matrix (W^T W) analysis
# ---------------------------------------------------------------------------
def plot_first_conv_analysis(model_path, n_tab_features):
    model = ISICMultimodalModel(
        model_name=CONFIG["model_name"],
        n_tabular_features=n_tab_features,
        pretrained=False,
    )
    state = safe_torch_load(model_path)
    model.load_state_dict(state)
    model.eval()

    conv_weight = model.backbone.conv_stem.weight.detach().cpu().numpy()
    out_ch, in_ch, kH, kW = conv_weight.shape

    W = conv_weight.reshape(out_ch, -1)
    G = W.T @ W

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    vmax = np.abs(G).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = axes[0].imshow(G, cmap="RdBu_r", norm=norm, aspect="equal")
    axes[0].set_title("Gram Matrix  $W^T W$\n(First Conv Layer, 27×27)", fontsize=12, fontweight="bold")

    ch_names = ["R", "G", "B"]
    tick_labels = [
        f"{ch_names[c]}({ky},{kx})"
        for c in range(in_ch)
        for ky in range(kH)
        for kx in range(kW)
    ]
    axes[0].set_xticks(range(len(tick_labels)))
    axes[0].set_xticklabels(tick_labels, rotation=90, fontsize=5)
    axes[0].set_yticks(range(len(tick_labels)))
    axes[0].set_yticklabels(tick_labels, fontsize=5)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    grid_rows, grid_cols = 4, 8
    filter_img = np.zeros((grid_rows * (kH + 1) - 1, grid_cols * (kW + 1) - 1, 3))

    for i in range(out_ch):
        r, c = divmod(i, grid_cols)
        filt = conv_weight[i].transpose(1, 2, 0)
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        y0 = r * (kH + 1)
        x0 = c * (kW + 1)
        filter_img[y0:y0 + kH, x0:x0 + kW, :] = filt

    axes[1].imshow(filter_img)
    axes[1].set_title(f"First Conv Filters ({out_ch}× {kH}×{kW} RGB)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    path = PLOTS_DIR / "first_conv_analysis.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    filter_norms = np.linalg.norm(W, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(out_ch), filter_norms, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Filter Index", fontsize=11)
    ax.set_ylabel("L2 Norm", fontsize=11)
    ax.set_title("First Conv Layer — Per-Filter Weight Magnitude", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "first_conv_filter_norms.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 8. Summary table (saved as image for report embedding)
# ---------------------------------------------------------------------------
def plot_summary_table(results):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    columns = ["Experiment", "Image Branch", "Tabular Branch", "Pooling", "Test AUC", "Test pAUC"]
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            r.get("image_branch", "—"),
            r.get("tabular_branch", "—"),
            r.get("pooling", "—"),
            f"{r['test_auc']:.4f}",
            f"{r['test_pauc']:.4f}",
        ])

    best_auc_idx = np.argmax([r["test_auc"] for r in results.values()])
    best_pauc_idx = np.argmax([r["test_pauc"] for r in results.values()])

    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(len(columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        if i == best_auc_idx:
            table[i + 1, 4].set_facecolor("#d5f5e3")
            table[i + 1, 4].set_text_props(fontweight="bold")
        if i == best_pauc_idx:
            table[i + 1, 5].set_facecolor("#d5f5e3")
            table[i + 1, 5].set_text_props(fontweight="bold")

    ax.set_title("Ablation Study Results Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path = PLOTS_DIR / "results_table.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ISIC 2024 — Generating Report Visualizations")
    print("=" * 60)

    results, histories = load_results()

    if not results:
        print("[ERROR] No results found. Run train.py and ablations.py first.")
        sys.exit(1)

    print("\n[1/7] Test performance comparison ...")
    plot_test_comparison(results)

    print("\n[2/7] Training curves ...")
    if histories:
        plot_training_curves(histories)
    else:
        print("  Skipped — no training histories found.")

    print("\n[3/7] Overfitting analysis ...")
    if histories:
        plot_overfit_analysis(histories)
    else:
        print("  Skipped — no training histories found.")

    print("\n[4/7] Test ROC curves ...")
    plot_test_roc_curves(results)

    print("\n[5/7] Summary table ...")
    plot_summary_table(results)

    model_path = CONFIG["output_dir"] / "final_model.pth"
    if not model_path.exists():
        for candidate in ["Ablation_C", "Ablation_D"]:
            p = CONFIG["output_dir"] / "ablations" / candidate / "best_model.pth"
            if p.exists():
                model_path = p
                print(f"\n  [INFO] Using {candidate} checkpoint for feature analysis: {p}")
                break

    n_tab = len(TABULAR_FEATURE_COLS)

    if model_path.exists():
        print("\n[6/7] Tabular feature importance ...")
        plot_tabular_feature_importance(model_path, n_tab)

        print("\n[7/7] First conv layer analysis (W^T W) ...")
        plot_first_conv_analysis(model_path, n_tab)
    else:
        print("\n[6/7] Skipped — no model checkpoint found.")
        print("  Run `python train.py` to generate output/final_model.pth")
        print("[7/7] Skipped.")

    print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()
