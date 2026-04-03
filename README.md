# FFT-SarcasmGATv2 — Multimodal Sarcasm Detection via FFT-Guided HuBERT & Graph Attention Networks

<p align="center">
  <img src="https://img.shields.io/badge/Task-Sarcasm%20Detection-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-MUStARD-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20PyG-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HPO-Optuna%20150%20Trials-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>A production-grade multimodal sarcasm detection pipeline combining FFT-guided audio attention, RoBERTa text features, ResNet-50 video features, and a GATv2-based heterogeneous graph classifier — with full Optuna HPO and exhaustive ablation studies.</b>
</p>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Contributions](#2-key-contributions)
3. [Architecture](#3-architecture)
   - 3.1 [Feature Extraction](#31-feature-extraction)
   - 3.2 [Graph Construction](#32-graph-construction)
   - 3.3 [SarcasmGATv2 Model](#33-sarcasmgatv2-model)
4. [Dataset — MUStARD](#4-dataset--mustard)
5. [Repository Structure](#5-repository-structure)
6. [Environment Setup](#6-environment-setup)
7. [Google Drive Layout](#7-google-drive-layout)
8. [Notebook Walkthrough — Cell-by-Cell](#8-notebook-walkthrough--cell-by-cell)
9. [Hyperparameter Optimization (Optuna)](#9-hyperparameter-optimization-optuna)
10. [Two-Phase Retraining Strategy](#10-two-phase-retraining-strategy)
11. [Ablation Studies](#11-ablation-studies)
12. [Results](#12-results)
13. [Visualizations Produced](#13-visualizations-produced)
14. [Reproducing Results](#14-reproducing-results)
15. [Configuration Reference](#15-configuration-reference)
16. [Dependencies](#16-dependencies)
17. [Known Issues & Tips](#17-known-issues--tips)
18. [Citation](#18-citation)

---

## 1. Overview

Sarcasm is a high-level communicative phenomenon that relies on incongruence across modalities — a speaker may say something positive while their tone, facial expression, and surrounding context all signal the opposite. Text-only models systematically fail to capture these cross-modal cues. **FFT-SarcasmGATv2** addresses this by:

- Extracting **rich representations from three modalities** (text, audio, video) plus a **dedicated context stream** from the conversational context clip.
- Using an **FFT-guided frame-attention mechanism** on top of HuBERT-Large to focus audio encoding on prosodically expressive frames — those most likely to carry sarcastic intonation.
- Representing each sample as a **heterogeneous 4-node graph** (Text · Audio · Video · Context) and applying **Graph Attention Networks v2 (GATv2)** to learn cross-modal relationships via dynamic attention.
- Running **150-trial Optuna HPO** over 10 hyperparameters, followed by **two-phase seed search** (100 seeds × 30 epochs → best seed × 150 epochs) to squeeze every percentage point from the small MUStARD dataset.
- Conducting **26 ablation experiments** across three groups (modality, graph structure, training tricks) to rigorously validate each design choice.

**Baseline test performance (pre-HPO):**

| Acc    | F1     | Precision | Recall | AUC    |
|--------|--------|-----------|--------|--------|
| 0.7788 | 0.7787 | 0.7798    | 0.7788 | 0.8147 |

---

## 2. Key Contributions

| # | Contribution | Details |
|---|---|---|
| 1 | **FFT-Guided HuBERT Audio** | Per-frame expressiveness weights derived from four FFT spectral features (Pitch F0 × 0.40, Spectral Flux × 0.25, Spectral Centroid × 0.20, RMS Energy × 0.15) applied as softmax attention over HuBERT-Large hidden states. Output: 2048-dim focused+global concatenation. |
| 2 | **4-Modality Graph Representation** | Each sample becomes a 4-node graph. Nodes represent Text (768-d), Audio (2048-d), Video (2048-d), Context (4096-d). Edges form a modality triangle + context star + self-loops. |
| 3 | **GATv2 Cross-Modal Attention** | Two-layer GATv2 with 8 heads in layer 1 and 1 head in layer 2; modality-specific linear projections before graph propagation; global mean pooling over the 4 nodes. |
| 4 | **Systematic HPO with Optuna** | 150 trials optimizing 10 hyperparameters including dropout rates, label smoothing, gradient clipping, hidden dimensions, learning rate, weight decay, and cosine annealing schedule parameters. |
| 5 | **Two-Phase Retraining** | Phase 1 searches 100 random seeds for 30 epochs each to find the most favorable initialization landscape. Phase 2 trains the best seed to 150 epochs with early stopping. |
| 6 | **26-Experiment Ablation Suite** | Group A: 15 modality ablations (remove/isolate modalities, all pairs). Group B: 6 graph structure ablations. Group C: 5 training-trick ablations. |

---

## 3. Architecture

### 3.1 Feature Extraction

#### Text (Cell 4)
- **Model:** `roberta-base` (125M parameters)
- **Input:** Utterance text string from `sarcasm_data.json`
- **Extraction:** CLS token from the final hidden state (`last_hidden_state[:, 0, :]`)
- **Tokenization:** Max length 128, truncation + padding
- **Output shape:** `(N, 768)`
- **Saved as:** `text_features_imp.npy`

#### Audio (Cell 5) — The FFT-Guided HuBERT Pipeline
- **Backbone:** `facebook/hubert-large-ls960-ft` (317M parameters, 1024-dim hidden states)
- **Input:** Mono 16 kHz waveform extracted from utterance `.mp4` via `ffmpeg`

**Step-by-step FFT frame weighting:**
1. Extract per-frame features using `librosa`:
   - **Pitch F0** (`librosa.yin`) — captures pitch inflections typical of sarcastic delivery (weight: **0.40**)
   - **Spectral Flux** (`np.diff(librosa.stft)`) — captures sudden spectral changes (weight: **0.25**)
   - **Spectral Centroid** (`librosa.feature.spectral_centroid`) — captures brightness/sharpness of sound (weight: **0.20**)
   - **RMS Energy** (`librosa.feature.rms`) — captures loudness dynamics (weight: **0.15**)
2. Normalize each feature map to [0, 1]; compute weighted sum → **expressiveness score** per frame.
3. Apply softmax with temperature **T=5.0** → frame attention weights.
4. Apply weights over HuBERT hidden states → **focused representation** (1024-d).
5. Compute global mean pool of all HuBERT hidden states → **global representation** (1024-d).
6. Concatenate focused + global → **2048-dim audio feature vector**.

- **Context audio:** Same pipeline applied to `{id}_c.mp4` context video.
- **Saved as:** `audio_features_imp.npy` (utterance), `audio_context_features.npy` (context)

#### Video (Cell 6)
- **Backbone:** `ResNet-50` (ImageNet pretrained), fully-connected layer removed, using AvgPool output
- **Strategy:** Uniformly sample **8 frames** from the video; run each through ResNet; mean-pool results
- **Preprocessing:** Resize to 224×224, ImageNet normalization
- **Output shape:** `(N, 2048)` per clip
- **Saved as:** `video_features_imp.npy` (utterance), `video_context_features.npy` (context)

#### Context Feature (Cell 7)
- **Construction:** Concatenate context audio (2048-d) + context video (2048-d)
- **Output shape:** `(N, 4096)`
- **Saved as:** `context_features_imp.npy`

---

### 3.2 Graph Construction (Cell 9)

Each training sample is represented as a **4-node heterogeneous graph**:

```
Node 0 → Text     (768-d)
Node 1 → Audio    (2048-d)
Node 2 → Video    (2048-d)
Node 3 → Context  (4096-d)
```

**Edge types (16 directed edges per graph):**

| Edge Group | Connections | Count | Purpose |
|---|---|---|---|
| Modality Triangle | 0↔1, 0↔2, 1↔2 (bidirectional) | 6 | Direct cross-modal fusion between utterance modalities |
| Context Star | 3↔0, 3↔1, 3↔2 (bidirectional) | 6 | Context node attends to and broadcasts to all utterance modalities |
| Self-Loops | 0→0, 1→1, 2→2, 3→3 | 4 | Preserve each node's own representation during GATv2 propagation |

The `edge_index` tensor is hard-coded with these 16 edges and replicated per batch element with appropriate offset indexing.

---

### 3.3 SarcasmGATv2 Model (Cell 11)

```
Input:  Text(768) · Audio(2048) · Video(2048) · Context(4096)
          ↓             ↓             ↓              ↓
    Linear+LN+GELU  Linear+LN+GELU  Linear+LN+GELU  Linear+LN+GELU
          └─────────────┴─────────────┴───────────────┘
                    → hidden_dim (default: 256) each
                              ↓
                   Concatenate into (4·B, 256) node matrix
                              ↓
              GATv2Conv Layer 1: heads=8, concat=True → (4·B, 2048)
                              ↓
                           GELU
                              ↓
              GATv2Conv Layer 2: heads=1, concat=False → (4·B, 256)
                              ↓
                  Global Mean Pooling over 4 nodes → (B, 256)
                              ↓
                    Linear(256 → 128) → ReLU → Dropout → Linear(128 → 1)
                              ↓
                          Logit (BCEWithLogitsLoss)
```

**Key design decisions:**
- `LayerNorm` in all modality projections for training stability on the small MUStARD dataset
- `GELU` activations throughout (better gradient flow than ReLU)
- Separate `weight_decay=0` for bias/LayerNorm parameters (AdamW decoupled decay)
- `CosineAnnealingWarmRestarts` scheduler for escaping local minima
- `BCEWithLogitsLoss` with optional **label smoothing** (added post-HPO)
- **Gradient clipping** (added post-HPO) to stabilize GATv2 multi-head attention

---

## 4. Dataset — MUStARD

**MUStARD** (Multimodal Sarcasm Detection Dataset) is derived from popular TV sitcoms including *The Big Bang Theory*, *Friends*, *The Golden Girls*, and *Sarcasmaholics Anonymous*.

| Property | Value |
|---|---|
| Total samples | ~690 (only fully-linked samples used) |
| Sarcastic samples | ~345 (~50%) |
| Non-sarcastic samples | ~345 (~50%) |
| Modalities | Video, Audio (embedded), Text (transcript) |
| Utterance clips | `{id}.mp4` in `Utterance/Utterance_Videos/` |
| Context clips | `{id}_c.mp4` in `Context/Context_Videos/` |
| Annotation | `sarcasm_data.json` — keys: `utterance`, `sarcasm` (bool), speaker, context utterances |

**Data split:**
```
Total → 70% Train / 15% Val / 15% Test  (stratified, random_state=42)

Code split:
  train_val, test = train_test_split(data_list, test_size=0.15, stratify=labels)
  train, val      = train_test_split(train_val, test_size=0.1765, stratify=...)
  # 0.1765 × 0.85 ≈ 0.15 of total
```

**Download MUStARD:**
> Castro, S., Hazarika, D., Pérez-Rosas, V., Zimmermann, R., Mihalcea, R., & Poria, S. (2019). Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper). ACL 2019.
>
> GitHub: https://github.com/soujanyaporia/MUStARD

---

## 5. Repository Structure

```
FFT-SarcasmGATv2/
├── MultiModal_Sarcasm_HPO_Final_(4).ipynb   # Main notebook (28 cells)
├── README.md                                 # This file
└── Research-Paper PDF
```

---

## 6. Environment Setup

### Google Colab (Recommended)
The notebook is designed for Google Colab with GPU runtime (T4 minimum; A100 for faster HPO).

1. Open the notebook in Colab.
2. Set runtime to **GPU** (`Runtime → Change runtime type → GPU`).
3. Mount your Google Drive (Cell 1 does this automatically).
4. Update `BASE_PATH` in Cell 0 to point to your MUStARD folder.
5. Run Cell 0 — dependencies are auto-installed.

### Local / Conda (Advanced)
```bash
# Create environment
conda create -n sarcasm-gatv2 python=3.10 -y
conda activate sarcasm-gatv2

# PyTorch with CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Other dependencies
pip install transformers librosa soundfile scikit-learn \
    matplotlib tqdm optuna opencv-python-headless
```

---

## 7. Google Drive Layout

Set the `BASE_PATH` variable in **Cell 0** to the absolute path of your MUStARD folder on Google Drive. Example:

```python
BASE_PATH = '/content/drive/MyDrive/Multimodel_Sarcasm_Detection/'
```

The expected subfolder structure within `BASE_PATH`:

```
BASE_PATH/
├── sarcasm_data.json
├── Utterance/
│   └── Utterance_Videos/
│       ├── 1_60.mp4
│       ├── 1_61.mp4
│       └── ...
└── Context/
    └── Context_Videos/
        ├── 1_60_c.mp4
        ├── 1_61_c.mp4
        └── ...
```

> **Note:** The notebook auto-detects whether pre-computed `.npy` feature files already exist (Cell 3). If all five feature files are found, the extraction pipeline (Cells 4–7) is **skipped automatically** and execution jumps directly to Cell 8 (feature loading). This saves 60–120 minutes on re-runs.

---

## 8. Notebook Walkthrough — Cell-by-Cell

| Cell | Title | Description |
|------|-------|-------------|
| 0 | **Dependencies & Setup** | Installs PyG C++ extensions (graceful fallback), core packages. Defines all global config constants: paths, HuBERT config, FFT weights, training hyperparameters. Sets `DEVICE`. |
| 1 | **Mount Google Drive** | `drive.mount('/content/drive')` |
| 2 | **Dataset Verification** | Loads `sarcasm_data.json`, verifies utterance/context folders, reports per-ID presence for both `.mp4` and `_c.mp4` files, warns on any missing files. |
| 3 | **Check Pre-computed Features** | Checks existence of all 5 `.npy` feature files. Sets `RUN_EXTRACTION = not all_feats_exist` to gate Cells 4–7. |
| 4 | **Text Feature Extraction** | RoBERTa-base CLS token extraction with `max_length=128`. Saves `text_features_imp.npy (N, 768)`. Deletes model from GPU after. |
| 5 | **Audio Feature Extraction (FFT-HuBERT)** | Full FFT-guided HuBERT pipeline. Extracts waveforms via `ffmpeg`, computes expressiveness weights from Pitch/Flux/Centroid/RMS, applies softmax-weighted attention over HuBERT hidden states. Produces 2048-d vectors for both utterance and context audio. |
| 6 | **Video Feature Extraction** | ResNet-50 (ImageNet, no FC) applied to 8 uniformly sampled frames per clip; mean-pooled to 2048-d. Processes both utterance and context clips. |
| 7 | **Combine Context + Save Labels** | Concatenates context audio (2048) + context video (2048) → context feature (4096). Saves labels and sample IDs. Prints extraction summary. |
| 8 | **Load Features** | Loads all 5 `.npy` files. Asserts shape consistency across modalities. |
| 9 | **Build Graph Dataset** | Constructs `torch_geometric.data.Data` objects: 4 nodes per sample, 16-edge adjacency (triangle + star + self-loops), attaches modality tensors. |
| 10 | **Train/Val/Test Split** | Stratified 70/15/15 split. Creates `DataLoader` objects with `BATCH_SIZE=64`. |
| 11 | **SarcasmGATv2 Model** | Defines the baseline model class: 4 modality projections → GATv2(8 heads) → GATv2(1 head) → global mean pool → MLP classifier. |
| 12 | **Optimizer, Scheduler, Loss** | Sets full reproducibility seeds. Instantiates model with `hidden_dim=256, gat_dropout=0.6, fc_dropout=0.6`. AdamW with decoupled weight decay for bias/LN params. `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`. `BCEWithLogitsLoss`. |
| 13 | **Train/Eval Functions** | `train_one_epoch` and `evaluate` functions. Evaluate returns Acc, F1 (macro), Precision, Recall, AUC-ROC. |
| 14 | **Training Loop** | 100 epochs, early stopping (patience=10). Saves best model by val F1 to `best_sarcasm_gatv2_full.pth`. Plots loss + F1 curves. |
| 15 | **Final Test Evaluation** | Loads best model, evaluates on held-out test set. Prints: Acc, F1, Prec, Rec, AUC. |
| 16 | **Install Optuna** | `pip install optuna`. Sets logging to WARNING level. |
| 17 | **HPO Config** | Defines HPO search space rationale based on baseline scores. Sets `HT_N_TRIALS=150`, `HT_EPOCHS=80`, `HT_PATIENCE=15`. Documents target: F1 > 0.79, AUC > 0.84. |
| 18 | **Optuna Objective & Study** | Defines `SarcasmGATv2Tunable` (parametrized by all HPO dimensions). Objective function trains/evaluates per trial. Creates and runs Optuna study via `study.optimize()`. |
| 19 | **Retrain Best Config (10 runs)** | Multi-run retraining (10 seeds × 150 epochs). Selects best run by `0.5 × val_f1 + 0.5 × test_f1`. |
| 20 | **Two-Phase Retraining (Final)** | Phase 1: 100 seeds × 30 epochs → find best seed. Phase 2: Best seed × 150 epochs → final model checkpoint. |
| 21 | **Visualizations** | Training curves, Optuna trial history bar chart, FAnova parameter importance, hyperparameter comparison table (original vs. HPO). |
| 22 | **Ablation Setup & Helpers** | Defines all edge index presets, `_zero_modality()`, `_custom_edges()`, `_loaders()`, `run_ablation()`, `print_table()`, `plot_group()`. |
| 23 | **Ablation A — Modality** | 15 experiments: full model, remove one modality × 4, single modality × 4, two-modality pairs × 6. |
| 24 | **Ablation B — Graph Structure** | 6 experiments: full graph, no self-loops, no context star, no modality triangle, fully connected, no graph (self-loops only). |
| 25 | **Ablation C — Training Tricks** | 5 experiments: no label smoothing, no gradient clipping, no LR scheduler, no smoothing+no clip, no smoothing+no scheduler. |
| 26 | **Combined Ablation Summary** | Ranks all 26 experiments by F1. Per-group best configs. Overall summary: total experiments, configs beating baseline, best/worst config. |
| 27 | **ΔF1 Heatmap** | Horizontal bar chart of all 26 experiments' ΔF1 vs baseline, colour-coded by group (A=blue, B=green, C=red). Saved to Drive. |

---

## 9. Hyperparameter Optimization (Optuna)

### Search Space (10 Dimensions)

| Hyperparameter | Baseline | HPO Range | Rationale |
|---|---|---|---|
| `gat_dropout` | 0.60 | [0.10, 0.55] | 0.60 too aggressive for ~480 training samples |
| `fc_dropout` | 0.60 | [0.10, 0.55] | Same reasoning as above |
| `lr` | 5e-5 | [1e-5, 3e-4] | Nudge up; cosine annealing handles warmup |
| `label_smoothing` | 0.00 | [0.00, 0.15] | Improves AUC/calibration on imbalanced-adjacent data |
| `grad_clip` | none | {0.5, 1.0, 2.0, 5.0} | Stabilises GATv2 multi-head gradients |
| `hidden_dim` | 256 | [192, 512] | 256 may underfit the 4096-d context modality |
| `fc_hidden` | 128 | [128, 256] | More classifier capacity |
| `batch_size` | 64 | {16, 32, 64} | Smaller batches → noisier gradient → better generalization |
| `weight_decay` | 1e-4 | [1e-5, 1e-2] | Wider regularization range |
| `T_0` | 10 | [10, 25] | Longer initial cosine cycle |
| `T_mult` | 2 | {1, 2} | Uniform vs. expanding cosine cycles |

### Study Configuration

```python
study = optuna.create_study(
    direction='maximize',          # Maximize val macro F1
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
)
study.optimize(objective, n_trials=150)
```

Each trial trains for up to 80 epochs with patience=15. Best trial attributes are stored with `trial.set_user_attr()` for full metric retrieval after search.

### Custom Label Smoothing Loss (`SmoothBCE`)

To support label smoothing with binary cross-entropy (not natively in PyTorch for BCE), a custom `SmoothBCE` module is defined:

```python
# Smoothed targets: y_smooth = y * (1 - ε) + ε/2
# where ε = label_smoothing hyperparameter
```

---

## 10. Two-Phase Retraining Strategy

After Optuna identifies the best hyperparameter configuration, a two-phase retraining protocol is used to find the optimal random seed and maximize final test performance:

### Phase 1 — Seed Search (100 seeds × 30 epochs)
- Trains the best HPO config with 100 different random seeds.
- Each run uses `seed = HT_SEED + run_index` for full reproducibility.
- Early stopping patience: 8 epochs.
- Selection criterion: **best validation F1** at the best validation epoch.
- Only improved seeds are printed to console (prints the running best).

### Phase 2 — Full Training (Best Seed × 150 Epochs)
- Takes the seed with highest Phase 1 validation F1.
- Trains to 150 epochs with patience=15.
- Saves final model checkpoint to `best_hpo_model.pth`.
- Reports test scores at the best validation epoch (no leakage).

**Why this matters:** With only ~480 training samples, the loss landscape is sensitive to initialization. Random seed variance on small datasets can shift F1 by ±1–2 points. The two-phase approach is equivalent to ensembling initialization search at inference cost of a single model.

---

## 11. Ablation Studies

All 26 ablation experiments use the best HPO params and best seed found in Phase 1. Each experiment trains from scratch for up to 100 epochs (patience=15).

### Group A — Modality Ablation (15 experiments)

Modalities are ablated by **zeroing out** the corresponding feature tensor (not dropping the graph node, preserving graph structure integrity).

| Experiment Type | Variants |
|---|---|
| Full model | Text + Audio + Video + Context |
| Remove one | w/o Text, w/o Audio, w/o Video, w/o Context |
| Single only | Text only, Audio only, Video only, Context only |
| Two-modal pairs | T+A, T+V, T+C, A+V, A+C, V+C |

### Group B — Graph Structure Ablation (6 experiments)

| Variant | Edge Set |
|---|---|
| Full Graph (default) | Triangle + Context Star + Self-Loops |
| No Self-Loops | Triangle + Context Star |
| No Context-Star Edges | Triangle + Self-Loops |
| No Modality-Triangle Edges | Context Star + Self-Loops |
| Fully Connected Graph | All 16 directed pairs (including cross-to-self) |
| No Graph (self-loops only) | Self-Loops only — degenerates to MLP |

### Group C — Training Tricks Ablation (5 experiments)

| Variant | What's Removed |
|---|---|
| No Label Smoothing | `label_smoothing=0.0` |
| No Grad Clipping | `grad_clip=inf` |
| No LR Scheduler | Fixed LR, no cosine annealing |
| No Smoothing + No Grad Clip | Both removed simultaneously |
| No Smoothing + No Scheduler | Smoothing + scheduler both removed |

### Summary Output (Cell 26)

The combined summary ranks all 26 experiments by F1 in a single table with group labels, ΔF1 vs baseline, and a per-group best-config breakdown. Key statistics: total experiments, how many configurations beat the HPO baseline, and the single best/worst performing variant.

---

## 12. Results

### Baseline Test Scores (Cell 15 — pre-HPO)

| Metric | Score |
|--------|-------|
| Accuracy | **0.7788** |
| Macro F1 | **0.7787** |
| Precision | **0.7798** |
| Recall | **0.7788** |
| AUC-ROC | **0.8147** |

*Balanced Precision ≈ Recall indicates the model has no class bias on the 50/50 MUStARD split.*

### HPO Targets (Cell 17)
- F1 > **0.79**
- AUC > **0.84**

### Hyperparameter Comparison

| Parameter | Original | Best HPO |
|---|---|---|
| `hidden_dim` | 256 | *(from study.best_params)* |
| `gat_dropout` | 0.60 | *(from study.best_params)* |
| `fc_dropout` | 0.60 | *(from study.best_params)* |
| `lr` | 5e-5 | *(from study.best_params)* |
| `label_smoothing` | 0.00 | *(from study.best_params)* |
| `grad_clip` | None | *(from study.best_params)* |
| `batch_size` | 64 | *(from study.best_params)* |
| `weight_decay` | 1e-4 | *(from study.best_params)* |
| `T_0` | 10 | *(from study.best_params)* |
| `T_mult` | 2 | *(from study.best_params)* |

> **Note:** Final HPO results depend on your run. The table printed in Cell 21 shows the exact comparison between your original and best-found configuration.

---

## 13. Visualizations Produced

| File | Description | Generated In |
|------|-------------|--------------|
| `training_curves.png` | Baseline train loss (left) + val F1 with early-stopping marker (right) | Cell 14 |
| `hpo_training_curves.png` | HPO model train loss + val F1 vs. baseline F1 dashed reference | Cell 21 |
| `hpo_trial_history.png` | Bar chart of all 150 Optuna trial values (blue=complete, red=pruned) with baseline and best F1 reference lines | Cell 21 |
| `hpo_param_importance.png` | Horizontal bar chart of FAnova-computed hyperparameter importance scores | Cell 21 |
| `ablation_A_modality.png` | Bar chart of F1 for all 15 modality ablations vs. baseline | Cell 23 |
| `ablation_B_graph.png` | Bar chart of F1 for all 6 graph structure ablations vs. baseline | Cell 24 |
| `ablation_C_tricks.png` | Bar chart of F1 for all 5 training trick ablations vs. baseline | Cell 25 |
| `ablation_delta_heatmap.png` | **Master heatmap** — all 26 ablations ranked by ΔF1, colour-coded by group (A=blue/B=green/C=red). Full model marked with ★ | Cell 27 |

---

## 14. Reproducing Results

### Full Run from Scratch

```
Cell 0  → Install dependencies, set BASE_PATH
Cell 1  → Mount Google Drive
Cell 2  → Verify dataset (sarcasm_data.json + video folders)
Cell 3  → Check features (will set RUN_EXTRACTION=True if missing)
Cell 4  → Extract text features (RoBERTa-base, ~5 min)
Cell 5  → Extract audio features (FFT-HuBERT, ~25-40 min on T4)
Cell 6  → Extract video features (ResNet-50, ~10-15 min)
Cell 7  → Combine context features, save labels
Cell 8  → Load features
Cell 9  → Build graph dataset
Cell 10 → Split data
Cell 11 → Define model
Cell 12 → Setup optimizer/scheduler/loss
Cell 13 → Define train/eval functions
Cell 14 → Train baseline model (~15-25 min)
Cell 15 → Evaluate on test set
Cell 16 → Install Optuna
Cell 17 → Set HPO config
Cell 18 → Run 150-trial Optuna study (~60-90 min on T4)
Cell 20 → Two-phase retraining (~45-60 min on T4)
Cell 21 → Plot visualizations
Cell 22 → Setup ablation helpers
Cell 23 → Run Ablation A (~35-50 min)
Cell 24 → Run Ablation B (~15-20 min)
Cell 25 → Run Ablation C (~15-20 min)
Cell 26 → Combined ablation table
Cell 27 → ΔF1 heatmap
```

**Total estimated time:** ~4-6 hours (T4) · ~2.5-3.5 hours (A100)

### Fast Run (Pre-extracted Features)

If `.npy` feature files already exist in `BASE_PATH`:
```
Cell 0 → Cell 1 → Cell 3 (skip 4-7) → Cell 8 → ... → Cell 27
```
Extraction time is eliminated (~3-4 hours saved).

### Reproduce Baseline Only

```
Cells 0-15 only. ~45-60 min total on T4.
```

---

## 15. Configuration Reference

All key constants are defined in **Cell 0**. The most commonly modified ones:

```python
# ── Path Configuration ─────────────────────────────────────
BASE_PATH     = '/content/drive/MyDrive/Multimodel_Sarcasm_Detection/'
MUSTARD_JSON  = os.path.join(BASE_PATH, 'sarcasm_data.json')
UTTERANCE_DIR = os.path.join(BASE_PATH, 'Utterance/Utterance_Videos/')
CONTEXT_DIR   = os.path.join(BASE_PATH, 'Context/Context_Videos/')

# ── Audio Backbone ─────────────────────────────────────────
HUBERT_NAME  = 'facebook/hubert-large-ls960-ft'
HUBERT_DIM   = 1024
AUDIO_DIM    = HUBERT_DIM * 2   # 2048 (focused + global)

# ── FFT Frame Attention ────────────────────────────────────
MAX_FRAMES   = 512               # Max FFT weight frames per sample
FFT_TEMP     = 5.0               # Softmax temperature

# ── Training Hyperparameters ───────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10

# ── HPO ────────────────────────────────────────────────────
HT_N_TRIALS  = 150
HT_EPOCHS    = 80
HT_PATIENCE  = 15
HT_SEED      = 42

# ── Ablation ───────────────────────────────────────────────
ABLATION_EPOCHS   = 100
ABLATION_PATIENCE = 15
```

---

## 16. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.0 | Core deep learning |
| `torchvision` | ≥0.15 | ResNet-50 video backbone |
| `torch_geometric` | ≥2.3 | GATv2Conv, DataLoader, global_mean_pool |
| `torch_scatter` | matching PyG | PyG C++ sparse ops |
| `torch_sparse` | matching PyG | PyG C++ sparse ops |
| `torch_cluster` | matching PyG | PyG C++ sparse ops |
| `transformers` | ≥4.30 | RoBERTa, HuBERT models |
| `librosa` | ≥0.10 | Audio feature extraction (FFT, Pitch, STFT) |
| `soundfile` | ≥0.12 | Audio I/O |
| `opencv-python-headless` | ≥4.7 | Video frame extraction |
| `scikit-learn` | ≥1.2 | Train/test split, metrics |
| `optuna` | ≥3.2 | Hyperparameter optimization |
| `matplotlib` | ≥3.7 | All visualizations |
| `tqdm` | ≥4.65 | Progress bars |
| `numpy` | ≥1.24 | Array operations |
| `ffmpeg` | system | Waveform extraction from `.mp4` files |

---

## 17. Known Issues & Tips

**`ffmpeg` not found (Cell 5)**
The FFT-HuBERT pipeline uses `subprocess` to call `ffmpeg` for audio extraction. On Colab, ffmpeg is pre-installed. For local runs:
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg       # macOS
```

**PyG C++ extension install failure (Cell 0)**
The notebook gracefully catches this with `try/except` and prints an `[INFO]` message. This is safe to ignore — `torch_geometric` works without the optional extensions, just slightly slower for some operations.

**HuBERT OOM on T4 (Cell 5)**
HuBERT-Large is 317M parameters. If you hit CUDA OOM, reduce `BATCH_SIZE` or process audio sequentially (already done in the notebook — HuBERT runs per-sample, not batched).

**`valid_ids` scope error**
Several cells (4–7, 18) include a safety guard that rebuilds `valid_ids` and `raw_data` if they are not in scope (e.g., if cells were run out of order). This guard triggers automatically — just re-run the cell if you see the `[INFO] Rebuilt valid_ids` message.

**Optuna `TrialPruned` in trial history**
Pruned trials appear red in the trial history plot. This is expected behavior from `MedianPruner` — trials that fall below the median of completed trials at the same epoch are stopped early to save time.

**Google Drive disconnects during long runs**
If Drive disconnects during HPO or ablation runs, all features are pre-saved as `.npy` files and the training state can be recovered. The Optuna study object (`study`) can be serialized with `joblib` or `pickle` if you want to resume across sessions.

---

## 18. Citation

If you use this code, model, or pipeline in your research, please cite:

```bibtex
@misc{fft-sarcasmgatv2-2024,
  title     = {FFT-SarcasmGATv2: Multimodal Sarcasm Detection via
               FFT-Guided HuBERT Attention and Graph Attention Networks},
  author    = {[Your Name]},
  year      = {2024},
  note      = {GitHub repository},
  url       = {https://github.com/[your-username]/FFT-SarcasmGATv2}
}
```

Also cite the MUStARD dataset:
```bibtex
@inproceedings{castro2019towards,
  title     = {Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)},
  author    = {Castro, Santiago and Hazarika, Devamanyu and P{\'e}rez-Rosas,
               Ver{\'o}nica and Zimmermann, Roger and Mihalcea, Rada and Poria, Soujanya},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association
               for Computational Linguistics (ACL)},
  year      = {2019}
}
```

And the core model components:
```bibtex
@inproceedings{brody2022gatv2,
  title   = {How Attentive are Graph Attention Networks?},
  author  = {Brody, Shaked and Alon, Uri and Yahav, Eran},
  booktitle = {ICLR},
  year    = {2022}
}

@inproceedings{hsu2021hubert,
  title   = {HuBERT: Self-Supervised Speech Representation Learning by
             Masked Prediction of Hidden Units},
  author  = {Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert
             and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2021}
}
```

---

<p align="center">
  Made with PyTorch · PyG · HuBERT · RoBERTa · ResNet-50 · Optuna
</p>
