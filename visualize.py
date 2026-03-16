"""
ISIC 2024 — Visualization & Analysis

Generates all comparison plots for the report:
  1. Bar chart: test AUC & pAUC across all experiments
  2. Training curves (loss, AUC, pAUC) across all experiments
  3. Tabular feature importance (first-layer weights of tabular MLP)
  4. Image branch first-conv filter analysis (W^T W gram matrix)

Prerequisites:
  - Run  python train.py      (creates output/baseline_*.json + output/final_model.pth)
  - Run  python ablations.py  (creates output/ablations/ablation_*.json)

Usage:
    python visualize.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

import torch

from train import (
    CONFIG, ISICMultimodalModel, TABULAR_FEATURE_COLS, NUM_COLS, NEW_NUM_COLS,
)

PLOTS_DIR = CONFIG["output_dir"] / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Readable short names for the long engineered feature column names
FEATURE_SHORT_NAMES = {col: col.replace("tbp_lv_", "").replace("_", " ") for col in TABULAR_FEATURE_COLS}


# ---------------------------------------------------------------------------
# 1. Load saved results
# ---------------------------------------------------------------------------
def load_results():
    """Return (all_results dict, all_histories dict) combining baseline + ablations."""
    all_results = {}
    all_histories = {}

    # Baseline
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

    # Ablations
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

    # Pretty labels
    labels = []
    for n in names:
        desc = results[n].get("desc", n)
        labels.append(f"{n}\n({desc})" if desc != n else n)

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width / 2, aucs, width, label="AUC", color="#4C72B0", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, paucs, width, label="pAUC @ 80% TPR", color="#DD8452", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Set Performance Across Experiments", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(aucs), max(paucs)) * 1.15)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    path = PLOTS_DIR / "test_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 3. Training curves — val loss, val AUC, val pAUC over epochs
# ---------------------------------------------------------------------------
COLORS = {
    "Baseline": "#2ecc71",
    "Ablation_A": "#3498db",
    "Ablation_B": "#e74c3c",
    "Ablation_C": "#9b59b6",
    "Ablation_D": "#f39c12",
}

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
                ax.plot(epochs, hist[key], marker="o", markersize=3, linewidth=1.8,
                        label=name, color=color)

        ax.set_title(f"{title}\n({note})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.replace("val_", "").upper())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "training_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
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

        # Loss
        axes[0][col].plot(epochs, hist["train_loss"], label="Train", linewidth=1.5, color=color)
        axes[0][col].plot(epochs, hist["val_loss"], label="Val", linewidth=1.5, color=color, linestyle="--")
        axes[0][col].set_title(f"{name}\nLoss", fontsize=10, fontweight="bold")
        axes[0][col].legend(fontsize=8)
        axes[0][col].grid(True, alpha=0.3)
        axes[0][col].set_xlabel("Epoch")

        # AUC
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
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 5. Tabular feature importance from the MLP's first layer
# ---------------------------------------------------------------------------
def plot_tabular_feature_importance(model_path, n_tab_features):
    """
    Computes importance of each tabular input feature by looking at the
    L2 norm of each column of the first linear layer: ||W[:, j]||_2.
    """
    model = ISICMultimodalModel(
        model_name=CONFIG["model_name"],
        n_tabular_features=n_tab_features,
        pretrained=False,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # tab_mlp.0 is the first nn.Linear(n_features, 256)
    W = model.tab_mlp[0].weight.detach().numpy()   # (256, n_features)
    importance = np.linalg.norm(W, axis=0)          # (n_features,)

    feature_names = [FEATURE_SHORT_NAMES.get(c, c) for c in TABULAR_FEATURE_COLS]

    # Sort descending
    order = np.argsort(importance)[::-1]
    top_k = 30
    top_idx = order[:top_k]

    fig, ax = plt.subplots(figsize=(10, 9))
    y_pos = np.arange(top_k)
    bars = ax.barh(y_pos, importance[top_idx][::-1], color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1], fontsize=9)
    ax.set_xlabel("Weight L2 Norm (||W[:, j]||)", fontsize=11)
    ax.set_title(f"Top {top_k} Tabular Feature Importances\n(First MLP Layer Weights)", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / "tabular_feature_importance.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    # Also save full importance to CSV
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
# 6. First-conv Gram matrix  (W^T W)  analysis
# ---------------------------------------------------------------------------
def plot_first_conv_analysis(model_path, n_tab_features):
    """
    Loads the model and extracts the first conv layer of EfficientNet-B0.
    The first conv has shape (out_ch, in_ch, kH, kW) = (32, 3, 3, 3).

    We reshape to W: (32, 27) and compute the Gram matrix G = W^T W (27×27)
    which captures correlations among the learned spatial-color features.

    We also visualise each of the 32 filters as 3×3 RGB patches.
    """
    model = ISICMultimodalModel(
        model_name=CONFIG["model_name"],
        n_tabular_features=n_tab_features,
        pretrained=False,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # EfficientNet-B0 first conv: backbone.conv_stem
    conv_weight = model.backbone.conv_stem.weight.detach().numpy()  # (32, 3, 3, 3)
    out_ch, in_ch, kH, kW = conv_weight.shape

    # --- Gram matrix ---
    W = conv_weight.reshape(out_ch, -1)     # (32, 27)
    G = W.T @ W                              # (27, 27)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Gram matrix heatmap
    vmax = np.abs(G).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = axes[0].imshow(G, cmap="RdBu_r", norm=norm, aspect="equal")
    axes[0].set_title("Gram Matrix  $W^T W$\n(First Conv Layer, 27×27)", fontsize=12, fontweight="bold")

    # Label the 27 dimensions as (channel, ky, kx)
    ch_names = ["R", "G", "B"]
    tick_labels = [f"{ch_names[c]}({ky},{kx})" for c in range(in_ch) for ky in range(kH) for kx in range(kW)]
    axes[0].set_xticks(range(len(tick_labels)))
    axes[0].set_xticklabels(tick_labels, rotation=90, fontsize=5)
    axes[0].set_yticks(range(len(tick_labels)))
    axes[0].set_yticklabels(tick_labels, fontsize=5)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # --- Filter visualisation: 32 filters as 3×3 RGB images ---
    grid_rows, grid_cols = 4, 8
    filter_img = np.zeros((grid_rows * (kH + 1) - 1, grid_cols * (kW + 1) - 1, 3))

    for i in range(out_ch):
        r, c = divmod(i, grid_cols)
        filt = conv_weight[i].transpose(1, 2, 0)  # (3,3,3) in RGB
        # Normalize each filter to [0,1] for display
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
    plt.close()
    print(f"  Saved → {path}")

    # --- Per-filter energy bar chart ---
    filter_norms = np.linalg.norm(W, axis=1)  # (32,)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(out_ch), filter_norms, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Filter Index", fontsize=11)
    ax.set_ylabel("L2 Norm", fontsize=11)
    ax.set_title("First Conv Layer — Per-Filter Weight Magnitude", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "first_conv_filter_norms.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 7. Summary table (saved as image for report embedding)
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

    # Highlight best AUC/pAUC
    best_auc_idx = np.argmax([r["test_auc"] for r in results.values()])
    best_pauc_idx = np.argmax([r["test_pauc"] for r in results.values()])

    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best rows
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
    plt.close()
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

    # --- Comparison plots ---
    print("\n[1/6] Test performance comparison ...")
    plot_test_comparison(results)

    print("\n[2/6] Training curves ...")
    if histories:
        plot_training_curves(histories)
    else:
        print("  Skipped — no training histories found.")

    print("\n[3/6] Overfitting analysis ...")
    if histories:
        plot_overfit_analysis(histories)
    else:
        print("  Skipped — no training histories found.")

    print("\n[4/6] Summary table ...")
    plot_summary_table(results)

    # --- Feature importance (requires a saved model checkpoint) ---
    model_path = CONFIG["output_dir"] / "final_model.pth"
    if not model_path.exists():
        # Try ablation checkpoints that include both branches
        for candidate in ["Ablation_C", "Ablation_D"]:
            p = CONFIG["output_dir"] / "ablations" / candidate / "best_model.pth"
            if p.exists():
                model_path = p
                print(f"\n  [INFO] Using {candidate} checkpoint for feature analysis: {p}")
                break

    n_tab = len(TABULAR_FEATURE_COLS)

    if model_path.exists():
        print("\n[5/6] Tabular feature importance ...")
        plot_tabular_feature_importance(model_path, n_tab)

        print("\n[6/6] First conv layer analysis (W^T W) ...")
        plot_first_conv_analysis(model_path, n_tab)
    else:
        print("\n[5/6] Skipped — no model checkpoint found.")
        print("  Run `python train.py` to generate output/final_model.pth")
        print("[6/6] Skipped.")

    print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()
