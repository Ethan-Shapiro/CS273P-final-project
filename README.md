# Multimodal Skin Cancer Detection: Fusing Image and Tabular Data

CS 273P Final Project — ISIC 2024 Skin Cancer Detection with 3D-TBP

## Project Overview

This project tackles the [ISIC 2024 Kaggle Challenge](https://www.kaggle.com/competitions/isic-2024-challenge) for binary skin lesion classification (malignant vs. benign). We implement a **multimodal deep learning architecture** in PyTorch that fuses:

- **Image features** from a fine-tuned EfficientNet-B0 with Generalized Mean (GeM) pooling
- **Tabular clinical features** (75 engineered numeric features) through a multi-layer perceptron

The two modality embeddings are concatenated and passed through a fusion head for final prediction. We conduct a systematic **ablation study** (5 experiments) to isolate the contribution of each component.

### Key Results

| Experiment | Test AUC | Test pAUC @ 80% TPR |
|---|---|---|
| Baseline (EfficientNet + MLP + GeM) | 0.9014 | 0.1264 |
| Ablation A — Tabular Only | 0.9283 | 0.1469 |
| Ablation B — Image Only | 0.9058 | 0.1311 |
| **Ablation C — Multimodal + AvgPool** | **0.9391** | **0.1563** |
| Ablation D — Frozen Backbone | 0.8817 | 0.1096 |

## Repository Structure

```
final-project/
├── train.py              # Baseline multimodal model training
├── ablations.py          # Ablation study (A-D) runner
├── visualize.py          # Generate all report figures
├── demo.ipynb            # Demo notebook (run code, test model, reproduce results)
├── create_sample_data.py # Generate small sample dataset for demo
├── report.md             # Full written report
├── requirements.txt      # Python dependencies
├── data/                 # Dataset (not included — see below)
│   ├── train-metadata.csv
│   ├── train-image/
│   │   └── image/
│   │       ├── ISIC_0082829.jpg
│   │       └── ...
│   └── sample/           # Small sample for demo (generated)
│       ├── sample_metadata.csv
│       └── images/
├── output/               # Training outputs (generated)
│   ├── final_model.pth
│   ├── baseline_results.json
│   ├── baseline_history.json
│   ├── plots/            # Report figures
│   │   ├── test_comparison.png
│   │   ├── training_curves.png
│   │   ├── overfit_analysis.png
│   │   ├── results_table.png
│   │   ├── tabular_feature_importance.png
│   │   ├── tabular_feature_importance.csv
│   │   ├── first_conv_analysis.png
│   │   └── first_conv_filter_norms.png
│   └── ablations/
│       ├── ablation_results.json
│       ├── ablation_histories.json
│       ├── Ablation_A/best_model.pth
│       ├── Ablation_B/best_model.pth
│       ├── Ablation_C/best_model.pth
│       └── Ablation_D/best_model.pth
├── isic-2024-only-tabular-data.ipynb          # Reference notebook (tabular)
└── isic-pytorch-training-baseline-image-only.ipynb  # Reference notebook (image)
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd final-project
```

### 2. Create a Python Environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

**With GPU (recommended):** Install PyTorch with CUDA first from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**CPU only:**

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the ISIC 2024 dataset from Kaggle:

1. Go to https://www.kaggle.com/competitions/isic-2024-challenge/data
2. Download `train-metadata.csv` and the `train-image` folder
3. Place them in the `data/` directory:

```
data/
├── train-metadata.csv
└── train-image/
    └── image/
        ├── ISIC_0082829.jpg
        ├── ISIC_0096034.jpg
        └── ... (401,059 images)
```

Alternatively, use the Kaggle CLI:

```bash
kaggle competitions download -c isic-2024-challenge -p data/
unzip data/isic-2024-challenge.zip -d data/
```

## How to Train the Model

### Train the Baseline (Multimodal: EfficientNet + MLP + GeM)

```bash
python train.py
```

This will:
- Load and engineer 75 tabular features from the metadata CSV
- Balance the dataset (all positives + 20x negatives)
- Split into train / validation / test using StratifiedGroupKFold by patient ID
- Train the multimodal model for 15 epochs
- Save the best model to `output/final_model.pth`
- Save training history to `output/baseline_history.json`

**Expected runtime:** ~15–30 minutes on a GPU, ~2–4 hours on CPU.

### Run the Ablation Study

```bash
python ablations.py
```

This trains four additional model variants sequentially:

| Ablation | Description |
|---|---|
| A | Tabular only (MLP, no images) |
| B | Image only (EfficientNet + GeM, no tabular) |
| C | Multimodal with AvgPool (instead of GeM) |
| D | Multimodal with frozen EfficientNet backbone |

Results are saved to `output/ablations/`.

**Expected runtime:** ~1–2 hours on GPU (4 experiments × 15 epochs each).

## How to Evaluate the Model

### Generate All Visualizations and Analysis

```bash
python visualize.py
```

This produces all report figures in `output/plots/`:

| File | Description |
|---|---|
| `test_comparison.png` | Bar chart comparing AUC and pAUC across all experiments |
| `training_curves.png` | Validation loss, AUC, and pAUC over training epochs |
| `overfit_analysis.png` | Train vs. validation curves per experiment |
| `results_table.png` | Publication-ready summary table |
| `tabular_feature_importance.png` | Top 30 features ranked by learned weight magnitude |
| `tabular_feature_importance.csv` | Full ranked feature importance list |
| `first_conv_analysis.png` | W^T W Gram matrix and filter visualization |
| `first_conv_filter_norms.png` | Per-filter weight magnitude |

### Evaluate a Trained Model Directly

The test set evaluation is performed automatically at the end of both `train.py` and `ablations.py`. Final test metrics (AUC and pAUC @ 80% TPR) are printed to the console and saved to JSON files.

## Expected Outputs

After running all three scripts, you should see:

**Console output from `train.py`:**
```
============================================================
ISIC 2024 — Multimodal (Image + Tabular) Training
============================================================
[1/6] Loading metadata and engineering tabular features ...
...
Epoch 15/15 | Train Loss: 0.3312  AUC: 0.9421  pAUC: 0.1548 | Val Loss: 0.2891  AUC: 0.8653  pAUC: 0.1075
...
============================================================
TEST RESULTS  —  AUC: 0.9014  |  pAUC@80%TPR: 0.1264
============================================================
```

**Console output from `ablations.py`:**
```
============================================================
ABLATION STUDY COMPLETE
============================================================
Experiment           Image Branch           Tabular Branch     Pooling    Test AUC  Test pAUC
---------------------------------------------------------------------------------------------
Ablation_A           None                   MLP                N/A          0.9283     0.1469
Ablation_B           EfficientNet           None               GeM          0.9058     0.1311
Ablation_C           EfficientNet           MLP                AvgPool      0.9391     0.1563
Ablation_D           EfficientNet (frozen)  MLP                GeM          0.8817     0.1096
```

**Generated files:**
```
output/
├── final_model.pth              (~21 MB)
├── baseline_results.json
├── baseline_history.json
├── plots/                       (8 PNG files + 1 CSV)
└── ablations/
    ├── ablation_results.json
    ├── ablation_histories.json
    └── Ablation_{A,B,C,D}/best_model.pth
```

## Configuration

All hyperparameters are set in the `CONFIG` dictionary at the top of `train.py`:

```python
CONFIG = {
    "seed": 42,
    "epochs": 15,
    "img_size": 224,
    "model_name": "tf_efficientnet_b0_ns",
    "train_batch_size": 32,
    "valid_batch_size": 64,
    "learning_rate": 1e-4,
    "pos_neg_ratio": 20,
    ...
}
```

Key paths to adjust if your data is in a different location:
- `CONFIG["train_metadata"]` — path to `train-metadata.csv`
- `CONFIG["train_image_dir"]` — path to the folder containing `.jpg` images

## Sample Dataset

The full ISIC 2024 dataset is ~50 GB and requires a Kaggle account. To make the demo notebook runnable without the full dataset, we provide a script that creates a small 50-sample subset:

```bash
# Requires the full dataset in data/ first
python create_sample_data.py
```

This copies 20 positive and 30 negative samples (images + metadata) into `data/sample/`. The demo notebook uses this sample by default.

If you cannot download the full dataset, the demo notebook can still:
- Load the trained model and display its architecture
- Show all ablation study results and plots from saved JSON files
- Display feature importance and conv filter analysis

## Demo Notebook

The `demo.ipynb` Jupyter notebook provides a self-contained walkthrough:

```bash
jupyter notebook demo.ipynb
```

The notebook covers:

1. Loading and visualizing sample images (malignant vs. benign)
2. Running the full feature engineering pipeline
3. Loading the trained model and running inference
4. Evaluating predictions (ROC curve, AUC, pAUC)
5. Displaying the full ablation study comparison (table + bar chart)
6. Training curve visualization across all experiments
7. Tabular feature importance analysis
8. First conv layer Gram matrix ($W^T W$) analysis
9. Model architecture summary

## Acknowledgments

This project builds on the following open-source resources:

- **EfficientNet-B0** via [`timm`](https://github.com/huggingface/pytorch-image-models) (Ross Wightman, Apache 2.0)
- **Tabular feature engineering** adapted from the Kaggle notebook [*ISIC 2024 — Only Tabular Data*](https://www.kaggle.com/competitions/isic-2024-challenge)
- **Image training pipeline** adapted from the Kaggle notebook [*ISIC PyTorch Training Baseline (Image Only)*](https://www.kaggle.com/competitions/isic-2024-challenge)

Our original contributions include the multimodal fusion architecture, ablation study framework, feature importance analysis, and all training/evaluation infrastructure.
