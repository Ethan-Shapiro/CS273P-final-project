# Multimodal Skin Cancer Detection: Fusing Image and Tabular Data with Deep Learning

## CS 273P — Final Project Report

---

## 1. Problem Definition and Motivation

Skin cancer is the most commonly diagnosed cancer worldwide, accounting for approximately one-third of all cancer diagnoses annually. Early detection is critical: when identified at an early stage, the five-year survival rate for melanoma exceeds 99%, but drops below 30% for late-stage detection. The clinical workflow for skin cancer screening relies heavily on visual inspection by dermatologists, a process that is both time-intensive and subject to inter-observer variability.

The **ISIC 2024 Challenge — Skin Cancer Detection with 3D Total Body Photography** (hosted on Kaggle) provides a large-scale dataset of dermoscopic images paired with rich clinical metadata. The competition objective is binary classification: given a skin lesion, predict whether it is **malignant** (target = 1) or **benign** (target = 0).

This project addresses the challenge through a **multimodal deep learning approach** that jointly leverages:

1. **Visual features** extracted from dermoscopic images via a convolutional neural network (EfficientNet-B0), and
2. **Tabular clinical features** including lesion geometry, color statistics, patient demographics, and 42 hand-engineered derived measurements, processed through a multi-layer perceptron (MLP).

The central research question is: **does fusing image and tabular modalities through a unified neural architecture improve classification performance over either modality alone?** We investigate this through a systematic ablation study that isolates the contribution of each component.

---

## 2. Related Work

**Deep learning for skin lesion classification.** Convolutional neural networks have been applied to dermoscopic image analysis since the early ImageNet era. Esteva et al. (2017) demonstrated dermatologist-level classification using Inception-v3. Subsequent work has employed increasingly sophisticated architectures including ResNets, DenseNets, and EfficientNets. The ISIC challenge series (2016–2024) has served as a primary benchmark, with state-of-the-art approaches consistently relying on transfer learning from ImageNet-pretrained models.

**EfficientNet and transfer learning.** Tan and Le (2019) introduced EfficientNet, a family of models that uniformly scale depth, width, and resolution using a compound scaling coefficient. EfficientNet-B0, the smallest variant, provides a strong accuracy-to-compute tradeoff and has become a standard backbone for medical image classification tasks. Fine-tuning pretrained EfficientNets on domain-specific data consistently outperforms training from scratch, particularly in low-data regimes.

**Generalized Mean (GeM) Pooling.** Radenovic et al. (2018) proposed GeM pooling as a learnable generalization of average and max pooling. The pooling operation is parameterized by a scalar $p$: when $p = 1$ it reduces to average pooling, and as $p \to \infty$ it approaches max pooling. Learning $p$ from data allows the network to adaptively emphasize high-activation regions — useful for localizing discriminative lesion features within an image.

**Tabular data approaches.** The Kaggle competition leaderboard for ISIC 2024 was dominated by gradient-boosted tree ensembles (LightGBM, XGBoost) applied to the tabular metadata, often outperforming image-only deep learning models. This is consistent with the broader finding that tree-based models remain highly competitive on structured tabular data (Grinsztajn et al., 2022).

**Multimodal fusion for medical AI.** Combining imaging data with electronic health records has shown promise across medical domains. Fusion strategies range from early fusion (concatenating raw inputs), to late fusion (combining modality-specific embeddings), to attention-based cross-modal fusion. In dermatology, Yap et al. (2018) demonstrated that incorporating patient metadata alongside images improved melanoma detection. Our work follows the late-fusion paradigm, concatenating learned embeddings from each modality before a shared classification head.

---

## 3. Dataset Description

### 3.1 Source

The dataset originates from the ISIC 2024 Kaggle Challenge and consists of two components:

- **Tabular metadata** (`train-metadata.csv`): 401,059 lesion records with 55 columns including patient demographics, lesion measurements, and color/shape statistics derived from 3D total body photography.
- **Dermoscopic images** (`train-image/image/`): One JPEG image per lesion, captured via close-up tile extraction from 3D body scans.

### 3.2 Class Distribution

The dataset exhibits extreme class imbalance:

| Class | Count | Percentage |
|-------|-------|------------|
| Benign (target = 0) | 400,666 | 99.90% |
| Malignant (target = 1) | 393 | 0.10% |

This 1:1,020 imbalance ratio presents a significant modeling challenge and necessitates careful handling during training.

### 3.3 Tabular Features

We use **75 numeric features** organized into three categories:

**Raw features (33 columns):** Patient age (`age_approx`), lesion diameter (`clin_size_long_diam_mm`), LAB color space measurements inside and outside the lesion (`tbp_lv_A`, `tbp_lv_Aext`, `tbp_lv_B`, `tbp_lv_Bext`, etc.), chroma values, hue, luminance, lesion area, perimeter, eccentricity, border regularity scores, color variation scores, symmetry measures, nevus confidence score, and 3D body coordinates.

**Engineered features (42 columns):** Derived from the raw features to capture higher-order relationships:

- **Geometric ratios:** lesion size ratio (minor axis / diameter), shape index (area / perimeter²), border length ratio
- **Color contrasts:** hue contrast, luminance contrast, lesion color difference (Euclidean distance in LAB delta space), color range
- **Composite indices:** lesion severity index, shape complexity index, comprehensive lesion index, color shape composite index
- **Interaction terms:** size-age interaction, hue-color std interaction, symmetry-perimeter interaction, age-size-symmetry index
- **Normalized features:** age-normalized nevi confidence, normalized lesion size
- **Transforms:** log lesion area, 3D position distance, lesion orientation (arctan2)

### 3.4 Data Balancing

To address the extreme imbalance, we retain all 393 positive samples and subsample negatives at a 1:20 ratio, yielding approximately 8,253 balanced samples. During training, the dataloader further enforces 50/50 class balance per batch by randomly drawing from separate positive and negative pools each iteration.

### 3.5 Train / Validation / Test Splits

Since the competition does not provide labeled test data, we partition the training set using **StratifiedGroupKFold** with 5 folds, grouping by `patient_id` to prevent data leakage (no patient appears in more than one split):

| Split | Folds Used | Samples | Purpose |
|-------|-----------|---------|---------|
| Training | 2, 3, 4 | ~4,950 | Model training |
| Validation | 0 | ~1,650 | Hyperparameter selection, early stopping |
| Test | 1 | ~1,650 | Final held-out evaluation |

---

## 4. Model Design

### 4.1 Architecture Overview

We propose a **late-fusion multimodal architecture** that processes images and tabular data through independent branches before combining their representations for classification.

```
                  ┌──────────────────────────────────────┐
                  │           Input Image (224×224×3)     │
                  └──────────────┬───────────────────────┘
                                 │
                  ┌──────────────▼───────────────────────┐
                  │  EfficientNet-B0 (pretrained, timm)  │
                  │  ─ conv_stem → blocks → conv_head ─  │
                  └──────────────┬───────────────────────┘
                                 │ Feature maps (1280×7×7)
                  ┌──────────────▼───────────────────────┐
                  │     GeM Pooling (learned p=3)        │
                  └──────────────┬───────────────────────┘
                                 │ 1280-d
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          │    ┌─────────────────┘                      │
          │    │                                        │
          │    │         ┌──────────────────────────┐   │
          │    │         │  Tabular Input (75-d)    │   │
          │    │         └────────────┬─────────────┘   │
          │    │                      │                 │
          │    │         ┌────────────▼─────────────┐   │
          │    │         │ Linear(75→256) + BN      │   │
          │    │         │ ReLU + Dropout(0.3)      │   │
          │    │         │ Linear(256→128) + BN     │   │
          │    │         │ ReLU + Dropout(0.2)      │   │
          │    │         └────────────┬─────────────┘   │
          │    │                      │ 128-d           │
          │    │                      │                 │
          │    └──────────┬───────────┘                 │
          │               │ Concatenate                 │
          │    ┌──────────▼────────────────────────┐    │
          │    │       Fusion: 1408-d              │    │
          │    │  Linear(1408→256) + BN + ReLU     │    │
          │    │  Dropout(0.3)                     │    │
          │    │  Linear(256→1) + Sigmoid          │    │
          │    └──────────┬────────────────────────┘    │
          │               │                             │
          │        P(malignant)                         │
          └─────────────────────────────────────────────┘
```

### 4.2 Image Branch

The image branch uses **EfficientNet-B0** pretrained on ImageNet, accessed via the `timm` library. We remove the original classification head and global average pooling, replacing them with:

- **GeM Pooling**: A learnable pooling layer parameterized by $p$ (initialized to 3). The operation computes:

$$\text{GeM}(x) = \left( \frac{1}{H \times W} \sum_{h,w} x_{h,w}^p \right)^{1/p}$$

This produces a 1280-dimensional embedding per image. When $p > 1$, GeM emphasizes high-activation spatial regions more than standard average pooling, which is useful for focusing on the lesion area within the image.

### 4.3 Tabular Branch

The tabular branch is a two-layer MLP that maps the 75-dimensional Z-score normalized feature vector to a 128-dimensional embedding:

$$\mathbf{h}_{\text{tab}} = \text{ReLU}(\text{BN}(W_2 \cdot \text{ReLU}(\text{BN}(W_1 \cdot \mathbf{x}_{\text{tab}}))))$$

Dropout is applied after each ReLU activation (0.3 and 0.2 respectively) to regularize the small tabular network.

### 4.4 Fusion Head

The 1280-d image embedding and 128-d tabular embedding are concatenated into a 1408-d vector and passed through a final MLP:

$$\hat{y} = \sigma(W_4 \cdot \text{ReLU}(\text{BN}(W_3 \cdot [\mathbf{h}_{\text{img}} \| \mathbf{h}_{\text{tab}}])))$$

The fusion head dimension (256) was chosen to be smaller than the image embedding to encourage information compression and prevent overfitting.

### 4.5 Design Rationale

Late fusion was chosen over early fusion (feeding raw tabular features alongside image pixels) because the two modalities have fundamentally different statistical properties: images are high-dimensional, spatially structured, and benefit from convolutional inductive biases, while tabular features are low-dimensional and unstructured. Allowing each branch to learn modality-specific representations before fusion prevents the tabular features from being overwhelmed by the image gradient signal.

---

## 5. Training Procedure

### 5.1 Optimization

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | $1 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-6}$ |
| LR scheduler | CosineAnnealingLR |
| Min LR | $1 \times 10^{-6}$ |
| Epochs | 15 |
| Batch size (train) | 32 |
| Batch size (val/test) | 64 |
| Gradient accumulation | 1 step |

### 5.2 Loss Function

We use **Binary Cross-Entropy (BCE) Loss**:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

where $y_i \in \{0, 1\}$ is the ground truth and $\hat{y}_i \in [0, 1]$ is the model's predicted probability of malignancy.

### 5.3 Data Augmentation

During training, images undergo the following augmentation pipeline (using the Albumentations library):

| Augmentation | Parameters | Probability |
|-------------|------------|-------------|
| Resize | 224 × 224 | 1.0 |
| RandomRotate90 | — | 0.5 |
| HorizontalFlip | — | 0.5 |
| VerticalFlip | — | 0.5 |
| Downscale | scale = 0.25 | 0.25 |
| Affine | shift=0.1, scale=0.15, rotate=60° | 0.5 |
| HueSaturationValue | shift=0.2 each | 0.5 |
| RandomBrightnessContrast | limit=0.1 each | 0.5 |
| Normalize | ImageNet mean/std | 1.0 |

Validation and test images receive only resize and normalization.

### 5.4 Tabular Preprocessing

All 75 tabular features are Z-score normalized using statistics computed exclusively on the training set:

$$x_{\text{norm}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$

where $\epsilon = 10^{-8}$ prevents division by zero. Missing values are imputed with 0 after normalization.

### 5.5 Class Balancing

Two levels of class balancing are applied:

1. **Dataset level**: The full dataset (401K samples) is subsampled to a 1:20 positive-to-negative ratio (~8,253 samples).
2. **Batch level**: The training dataloader randomly selects from the positive or negative pool with equal probability for each sample, ensuring each batch is approximately 50% positive and 50% negative regardless of the dataset ratio.

---

## 6. Evaluation Metrics

### 6.1 AUC-ROC

The **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)** measures the model's ability to discriminate between positive and negative classes across all decision thresholds. An AUC of 1.0 indicates perfect discrimination; 0.5 indicates random chance.

### 6.2 Partial AUC at 80% TPR (pAUC)

The ISIC 2024 competition's primary metric is the **partial AUC** evaluated in the clinically relevant operating region where the true positive rate (sensitivity) is at least 80%. This reflects the practical requirement that a screening tool must catch at least 80% of malignant lesions; the metric then measures how well the model minimizes false positives within that constraint.

The pAUC is computed as:

$$\text{pAUC} = \frac{1}{2}(\text{max\_fpr})^2 + \frac{\text{max\_fpr} - \frac{1}{2}(\text{max\_fpr})^2}{0.5} \cdot (\text{pAUC}_{\text{scaled}} - 0.5)$$

where $\text{max\_fpr} = 1 - \text{min\_tpr} = 0.20$ and $\text{pAUC}_{\text{scaled}}$ is the `roc_auc_score` restricted to the $[0, 0.20]$ false positive rate range.

---

## 7. Experimental Results

### 7.1 Ablation Study Design

To understand the contribution of each architectural component, we conduct four ablation experiments alongside the full baseline model. Each ablation modifies exactly one aspect of the architecture while keeping all other hyperparameters, data splits, and training procedures identical.

| Experiment | Image Branch | Tabular Branch | Pooling | What It Tests |
|------------|-------------|----------------|---------|---------------|
| **Baseline** | EfficientNet-B0 | MLP | GeM | Full multimodal architecture |
| **Ablation A** | None | MLP | N/A | Value of tabular data alone |
| **Ablation B** | EfficientNet-B0 | None | GeM | Value of images alone |
| **Ablation C** | EfficientNet-B0 | MLP | AvgPool | GeM vs. standard average pooling |
| **Ablation D** | EfficientNet-B0 (frozen) | MLP | GeM | Value of fine-tuning the backbone |

### 7.2 Test Set Results

All models are evaluated on the same held-out test fold (fold 1, ~1,650 samples). Results are summarized in Table 1 and Figure 1.

**Table 1: Test set performance across all experiments.**

| Experiment | Test AUC | Test pAUC @ 80% TPR | Best Val AUC |
|------------|----------|---------------------|-------------|
| Baseline (EfficientNet + MLP + GeM) | 0.9014 | 0.1264 | 0.8653 |
| Ablation A — Tabular Only (MLP) | 0.9283 | 0.1469 | 0.9237 |
| Ablation B — Image Only (EfficientNet + GeM) | 0.9058 | 0.1311 | 0.8681 |
| **Ablation C — Multimodal + AvgPool** | **0.9391** | **0.1563** | 0.8955 |
| Ablation D — Multimodal + Frozen Backbone | 0.8817 | 0.1096 | 0.8484 |

*Figure 1: `output/plots/test_comparison.png` — Grouped bar chart comparing AUC and pAUC across all experiments.*

*Figure 2: `output/plots/training_curves.png` — Validation loss, AUC, and pAUC over 15 training epochs.*

*Figure 3: `output/plots/overfit_analysis.png` — Train vs. validation curves showing overfitting behavior.*

### 7.3 Key Observations

1. **Best overall model**: Ablation C (multimodal with AvgPool) achieves the highest test AUC of **0.9391** and highest pAUC of **0.1563**, outperforming all other configurations.

2. **Tabular data dominates images**: Ablation A (tabular only, AUC = 0.9283) substantially outperforms Ablation B (image only, AUC = 0.9058), a difference of +0.0225 AUC. The pAUC gap is even larger: 0.1469 vs. 0.1311.

3. **Fusion improves over single modalities**: The best fusion model (Ablation C, 0.9391) outperforms the best single-modality model (Ablation A, 0.9283) by +0.0108 AUC, confirming that images provide complementary information beyond what tabular features capture.

4. **AvgPool outperforms GeM in fusion**: Comparing the Baseline (GeM, 0.9014) with Ablation C (AvgPool, 0.9391), AvgPool yields a +0.0377 AUC improvement. This is a surprising finding discussed further in Section 8.

5. **Fine-tuning is essential**: Ablation D (frozen backbone, 0.8817) performs worst overall, demonstrating that ImageNet features alone are insufficient for dermatology — the CNN must be fine-tuned on domain-specific data.

---

## 8. Analysis and Discussion

### 8.1 Why Tabular Features Outperform Images Alone

The strong performance of the tabular-only MLP (Ablation A) is consistent with the broader ISIC 2024 competition landscape, where gradient-boosted tree ensembles on tabular data dominated the leaderboard. The tabular features encode decades of dermatological domain knowledge: the 75 features include clinically validated discriminators such as lesion asymmetry, border irregularity, color variation, and diameter — all components of the clinical ABCDE rule for melanoma screening. Additionally, features like `nevi_confidence` (a pre-computed CNN score from a separate model) effectively provide a distilled prior from a larger training set.

In contrast, the image-only model must learn these patterns from scratch on a relatively small training set (~4,950 balanced samples), with only 15 epochs of fine-tuning. The extreme class imbalance further challenges the visual model, as it must distinguish subtle malignant patterns from the overwhelming variety of benign presentations.

### 8.2 AvgPool vs. GeM: The Surprising Result

The most counterintuitive finding is that replacing GeM pooling with standard average pooling improves the multimodal model by 3.8 AUC points. We hypothesize two contributing factors:

1. **Overfitting of the learned pooling parameter.** GeM introduces a learnable parameter $p$ that controls the sharpness of spatial attention. With only ~250 positive training examples, the gradient signal for optimizing $p$ is noisy and may cause the pooling to overfit — focusing on spurious spatial patterns in the training set rather than generalizable features.

2. **Interaction with tabular fusion.** When combined with a strong tabular branch, the image branch needs to provide complementary rather than dominant features. Average pooling produces a more diffuse, holistic summary of the image, which may compose better with tabular features than the sharper, more localized GeM representation.

This finding highlights the importance of ablation studies: a component that improves performance in one setting (image-only classification, where GeM was originally proposed) may hurt performance when integrated into a different architecture.

### 8.3 The Value of Fine-Tuning

Ablation D demonstrates that frozen ImageNet features are a poor match for dermoscopic images. The 1.97 AUC-point gap between Ablation D (frozen, 0.8817) and the Baseline (fine-tuned, 0.9014) shows that domain adaptation through backpropagation is essential. ImageNet features are optimized for natural images (objects, scenes, animals), which have fundamentally different statistical properties from close-up clinical photographs of skin lesions. Fine-tuning allows the early convolutional layers to adapt their filters to dermatology-relevant textures (pigment patterns, vascular structures, scaling).

### 8.4 Tabular Feature Importance

*Figure 4: `output/plots/tabular_feature_importance.png` — Top 30 features ranked by first-layer MLP weight magnitude.*

We analyze feature importance by computing the L2 norm of each column of the tabular MLP's first linear layer: $\|W_{:,j}\|_2$ for each feature $j$. This measures how strongly the learned transformation weights each input dimension. The top 10 most important features are:

| Rank | Feature | Importance (L2 norm) | Category |
|------|---------|---------------------|----------|
| 1 | `tbp_lv_areaMM2` | 1.136 | Lesion area |
| 2 | `index_age_size_symmetry` | 1.119 | Engineered (age × area × symmetry) |
| 3 | `lesion_size_ratio` | 1.102 | Engineered (minor axis / diameter) |
| 4 | `tbp_lv_Cext` | 1.101 | Chroma outside lesion |
| 5 | `hue_contrast` | 1.099 | Engineered (|H_in - H_out|) |
| 6 | `lesion_visibility_score` | 1.095 | Engineered (contrast + color norm) |
| 7 | `border_color_interaction_2` | 1.095 | Engineered (border × color / sum) |
| 8 | `tbp_lv_deltaLB` | 1.092 | Color contrast (LB) |
| 9 | `age_normalized_nevi_confidence_2` | 1.091 | Engineered (√(size² + age²)) |
| 10 | `tbp_lv_nevi_confidence` | 1.090 | Pre-computed CNN confidence |

Several clinically meaningful patterns emerge:

- **Lesion geometry** (area, size ratio) ranks highest, consistent with the clinical observation that larger and asymmetric lesions are more suspicious.
- **Color contrast** features (hue contrast, chroma, deltaLB) are heavily weighted, reflecting the importance of color irregularity in the ABCDE screening criteria.
- **Engineered interaction terms** (index_age_size_symmetry, border_color_interaction_2) appear in the top 10, validating that the feature engineering step captures meaningful higher-order relationships not present in raw features alone.
- **Nevi confidence** (a pre-computed CNN score) is among the most important features, effectively providing the tabular branch with a distilled visual signal.

### 8.5 First Convolutional Layer Analysis

*Figure 5: `output/plots/first_conv_analysis.png` — Left: Gram matrix (W^T W) of the first conv layer. Right: Visualization of the 32 learned 3×3 RGB filters.*

We examine the learned representations of the first convolutional layer (`conv_stem`) of EfficientNet-B0 after fine-tuning. This layer has 32 output channels with 3×3 kernels over 3 input channels (RGB), giving a weight tensor of shape (32, 3, 3, 3). Reshaping to $W \in \mathbb{R}^{32 \times 27}$ and computing the **Gram matrix** $G = W^T W \in \mathbb{R}^{27 \times 27}$ reveals correlations among the learned spatial-color features.

The Gram matrix exhibits:

- **Block-diagonal structure**: The three 9×9 blocks along the diagonal correspond to within-channel (R-R, G-G, B-B) spatial correlations, showing that the layer learns channel-specific spatial patterns.
- **Off-diagonal cross-channel correlations**: Non-zero entries between the R, G, and B blocks indicate that certain filters jointly attend to cross-channel color patterns — consistent with the need to detect pigmentation changes that manifest across color channels.

The individual filter visualizations show a variety of edge detectors, color-contrast detectors, and texture-sensitive patterns. Some filters have learned to respond specifically to warm (red-brown) hues typical of pigmented lesions, while others capture the blue-white gradients associated with certain malignant presentations.

### 8.6 Overfitting Analysis

Examination of the train-versus-validation curves reveals different overfitting behaviors:

- **Ablation A (Tabular only)**: Shows the smallest train-val gap, suggesting the MLP is well-regularized relative to the tabular feature space. Validation AUC continues improving through epoch 12 before plateauing.
- **Ablation B (Image only)**: Shows the largest train-val gap (~0.07 AUC by epoch 15), indicating the CNN has more capacity than the data can constrain. The training AUC reaches 0.94 while validation plateaus at 0.87.
- **Ablation C (Multimodal + AvgPool)**: Shows moderate overfitting with the train-val gap stabilizing around 0.05 AUC, suggesting the tabular branch provides a regularizing effect.
- **Ablation D (Frozen backbone)**: Shows the slowest convergence, with validation AUC still increasing at epoch 15 — the frozen backbone limits the model's capacity, preventing both overfitting and fast learning.

---

## 9. Limitations and Future Work

### 9.1 Limitations

**Data limitations:**
- The extreme class imbalance (393 positives out of 401,059 total) means the model has limited exposure to malignant diversity. Even after balancing, the ~250 positive training samples span a wide range of malignancy types (melanoma, basal cell carcinoma, squamous cell carcinoma), leaving very few examples per subtype.
- Only one fold split was used for final evaluation due to computational constraints. A full 5-fold cross-validation would provide confidence intervals and more robust performance estimates.

**Architectural limitations:**
- Image resolution was reduced to 224×224, potentially losing fine-grained details like dermatoscopic structures (pigment network, globules, streaks) that may be visible at higher resolutions.
- The fusion strategy is simple concatenation. No learned attention mechanism selects which modality to emphasize for a given sample.
- Hyperparameters (MLP widths, dropout rates, learning rate) were chosen heuristically rather than optimized through systematic search.

**Evaluation limitations:**
- We evaluate on a single train/val/test split rather than cross-validation, meaning our reported numbers have some variance.
- The test set is drawn from the same distribution as training data. Real-world deployment would face distribution shift from different imaging devices, patient populations, and acquisition protocols.

### 9.2 Future Work

1. **Attention-based fusion.** Replace concatenation with cross-attention between image and tabular embeddings, allowing the model to learn input-dependent modality weighting.

2. **Larger backbones.** Experiment with EfficientNet-B3 or B4, or Vision Transformers (ViT), which may capture more complex visual patterns given sufficient regularization.

3. **Higher image resolution.** Train at 384×384 or 512×512, which may reveal clinically relevant microstructures lost at 224×224.

4. **Full dataset training.** Use all 401K samples with focal loss or class-weighted BCE instead of subsampling, preserving the natural data distribution while addressing imbalance through the loss function.

5. **Ensemble methods.** Combine the multimodal neural network with a standalone LightGBM on tabular features, leveraging the complementary strengths of deep learning and gradient boosting.

6. **Test-time augmentation (TTA).** Average predictions over multiple augmented views of each test image to reduce variance.

7. **Interpretability.** Apply Grad-CAM or similar attribution methods to visualize which image regions the CNN attends to for malignant predictions, providing clinically actionable explanations.

---

## 10. Attribution of Borrowed Code

In accordance with the project guidelines, we clearly document all external code used:

| Component | Source | License | Our Modifications |
|-----------|--------|---------|-------------------|
| EfficientNet-B0 backbone | `timm` library (Ross Wightman) | Apache 2.0 | Removed classifier head; replaced pooling with GeM; integrated into multimodal architecture |
| Tabular feature engineering | Kaggle notebook: *"ISIC 2024 — Only Tabular Data"* | Public Kaggle notebook | Ported from Polars-based notebook; adapted column selection for PyTorch integration; removed LightGBM-specific preprocessing |
| Image augmentation pipeline | Kaggle notebook: *"ISIC PyTorch Training Baseline (Image Only)"* | Public Kaggle notebook | Updated to Albumentations v2 API; reduced image size from 384→224; integrated with multimodal dataset class |
| GeM Pooling | Adapted from Radenovic et al. (2018) | — | Standard implementation |

**Our original contributions:**
- Design and implementation of the multimodal late-fusion architecture combining image and tabular branches
- Custom multimodal dataset class that jointly loads images and tabular features with balanced sampling
- Complete ablation study framework with four systematic experiments (tabular-only, image-only, AvgPool variant, frozen-backbone variant)
- Feature importance analysis via first-layer weight norms
- First convolutional layer Gram matrix ($W^T W$) analysis
- All training infrastructure, evaluation pipeline, visualization code, and this report

---

## References

1. Esteva, A., et al. "Dermatologist-level classification of skin cancer with deep neural networks." *Nature* 542.7639 (2017): 115-118.

2. Tan, M., & Le, Q. "EfficientNet: Rethinking model scaling for convolutional neural networks." *ICML* (2019).

3. Radenovic, F., Tolias, G., & Chum, O. "Fine-tuning CNN image retrieval with no human annotation." *IEEE TPAMI* 41.7 (2018): 1655-1668.

4. Grinsztajn, L., Oyallon, E., & Varoquaux, G. "Why do tree-based models still outperform deep learning on typical tabular data?" *NeurIPS* (2022).

5. Yap, J., et al. "Multimodal skin lesion classification using deep learning." *Experimental Dermatology* 27.11 (2018): 1261-1267.

6. ISIC 2024 Challenge. "Skin Cancer Detection with 3D-TBP." https://www.kaggle.com/competitions/isic-2024-challenge
