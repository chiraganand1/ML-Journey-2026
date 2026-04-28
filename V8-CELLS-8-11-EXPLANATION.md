# BirdCLEF 2026 v8 Training Notebook — Deep Technical Reference: Cells 8–11

This document provides an exhaustive explanation of every design decision, algorithmic choice, and line of code in cells 8 through 11 of the v8 training notebook. It is written to be self-contained — you should be able to read this without having the notebook open.

---

## Background: Why These Four Cells Are the Heart of the Notebook

Cells 1–7 handle setup: installing packages, defining paths, precomputing mel spectrograms, defining the loss function, and defining the model architecture. They are prerequisites.

Cells 8–11 are where actual machine learning happens:

- **Cell 8** defines how data flows into the GPU: the `Dataset` class (how individual samples are loaded and cropped), the `mixup_collate` function (how batches are assembled with data augmentation), and `make_loader` (how all of this is wired together with a `DataLoader`).
- **Cell 9** reconciles the on-disk mel files with the DataFrame, ensuring training only uses samples that were successfully precomputed.
- **Cell 10** is the 15-model training loop: the outer loop over architectures, the inner loop over folds, and inside each fold the per-epoch train/validate/checkpoint cycle.
- **Cell 11** aggregates the out-of-fold (OOF) predictions into a final unbiased metric estimate.

Together they implement a complete, state-of-the-art training pipeline targeting the BirdCLEF 2026 macro-averaged ROC-AUC metric.

---

## Cell 8: Dataset, Mixup, and DataLoader

### Why precompute mels at all?

Before diving into `ClipDataset`, it is worth understanding why mels are saved to disk in Cell 4 rather than computed on-the-fly.

**The alternative** (on-the-fly spectrograms) would be: load audio → resample → compute mel → normalise, all inside `__getitem__`. This is perfectly valid but has a cost: on a Kaggle T4 GPU, a single mel computation on a 5-second clip at n_fft=2048 takes ~5–15ms. With a batch size of 32 and `num_workers=0`, that adds 160–500ms of CPU preprocessing time per batch, stalling the GPU. For 30 epochs across 15 models, this CPU bottleneck would roughly double total training time.

**The precompute approach** pays a one-time cost (Cell 4, ~15–40 minutes) and from then on, all 15 training runs read from `.npy` files. A numpy load is ~0.1ms, roughly 50–150x faster than recomputing the mel.

The tradeoff: the mels are fixed. You cannot augment parameters like `fmin`, `fmax`, `n_mels`, or `n_fft` during training. For v8, this is acceptable because the hyperparameters are already optimised (128 mels, 2048 FFT).

---

### `ClipDataset` — Full Walkthrough

```python
class ClipDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, mel_root: str, train: bool):
        self.df       = frame.reset_index(drop=True)
        self.mel_root = Path(mel_root)
        self.train    = train
        self.win_frames = int(CFG["seconds"] * CFG["sr"] / CFG["hop"]) + 1
```

**`reset_index(drop=True)`:** After `StratifiedKFold.split()`, the train and validation subsets are created via `train_df.iloc[train_idx]`. The original DataFrame indices are preserved (e.g., rows 3, 7, 12…). `reset_index(drop=True)` renumbers them 0, 1, 2, … which is required for `iloc[i]` inside `__getitem__` to work correctly.

**`win_frames` calculation:**
```
win_frames = int(5 * 16000 / 320) + 1
           = int(250.0) + 1
           = 251
```
This is the number of time frames in the mel spectrogram for a 5-second clip at sample rate 16000 and hop length 320. The `+1` accounts for the final partial frame that `librosa` produces. Each `.npy` file was created from a 5-second clip so `T_actual` will usually be exactly 251. The crop logic still handles the edge cases where it differs.

---

#### `__getitem__` deep dive

```python
def __getitem__(self, i):
    r = self.df.iloc[i]
    fname    = str(r["filename"])
    mel_name = (fname if fname.endswith(".npy") else fname.replace("/", "_") + ".npy")
    mel      = np.load(self.mel_root / mel_name).astype("float32")
```

**Filename resolution:** There are two categories of filenames in `train_df` after Cell 9:

1. **train_audio samples** → `filename` was normalised in Cell 9 to `species_XC12345.ogg` (slash replaced with underscore). The mel was saved as `species_XC12345.ogg.npy`. So `mel_name = fname + ".npy"`.
2. **Soundscape pseudo-samples** → `filename` was stored as `soundscape_XC99999_seg_2` (no extension, no slash). The mel was saved as `soundscape_XC99999_seg_2.npy`. The check `fname.endswith(".npy")` handles this but actually `.npy` won't be there — the `.npy` is appended. Both cases produce the correct path.

**`.astype("float32")`:** The saved `.npy` should already be float32 (from `logmel_from_wave`), but this is a defensive cast. Some numpy operations default to float64 and a silent upcast here would double memory usage in the DataLoader.

---

#### The crop strategy in detail

```python
T, W = mel.shape[1], self.win_frames
if T <= W:
    mel = np.concatenate([mel, np.zeros((mel.shape[0], W - T), dtype=np.float32)], axis=1)
else:
    start = np.random.randint(0, T - W) if self.train else (T - W) // 2
    mel   = mel[:, start:start + W]
```

**Why CNNs require fixed-size inputs:**
`torch.stack` in `mixup_collate` requires all tensors in a batch to have the same shape. CNNs also have fixed spatial dimensions due to their fully-connected head (or global average pooling with a fixed output size). So every mel must be exactly `(128, 251)` before it enters the network.

**Zero-padding case (`T_actual < 251`):**
This happens when:
- The original audio was shorter than 5 seconds (some XenoCanto recordings are 2–3 seconds)
- The mel computation produced slightly fewer frames than expected

Padding with zeros appends silence on the right. Silence in a mel spectrogram is a near-zero-power frame, which after normalisation becomes very dark. The model learns to treat these as uninformative regions rather than misidentifying them as bird calls.

**Random crop (train=True):**
`np.random.randint(0, T - W)` picks a uniform random start position from 0 to `T - W` inclusive. The window `[start, start + W)` is then extracted. For a 10-second recording (T ≈ 502), there are ~252 possible start positions, so the same recording can produce 252 different 5-second crops. This provides a form of temporal data augmentation.

**Why temporal position augmentation matters for BirdCLEF:**
Bird calls are not uniformly distributed across a recording. An annotator marks the species present but not the exact second the call occurs. The model must be robust to the call appearing at any position within the window — random cropping is the simplest way to train this invariance. Without it, the model would tend to look for calls in the centre of the spectrogram, matching how the validation centre-crop is constructed.

**Centre crop (train=False):**
`(T - W) // 2` takes the middle of the recording. This is deterministic and ensures that multiple validation passes (if you ran validation twice) would yield identical results — essential for reliable early stopping and AUC tracking across epochs.

---

#### Label encoding

```python
y = torch.from_numpy(
    row_to_soft_multihot(r["primary_label"], r.get("secondary_labels", "[]"))
).float()
```

`row_to_soft_multihot` (defined in Cell 3) returns a float32 vector of length 234 where:
- `y[sp_idx[primary_label]] = 1.0`
- `y[sp_idx[secondary_label]] = 0.3` for each secondary label present
- All other entries = 0.0

**Why soft labels instead of hard labels (0/1)?**
The competition annotations include secondary labels: species that were also heard in the recording but are not the primary focus. Setting secondary labels to 1.0 would tell the model "this recording perfectly represents this species," which is too strong — the call is incidental, shorter, or more distant. Setting them to 0.0 would discard real information. `0.3` is a compromise: the model is trained to assign a moderate probability to secondary species, which is closer to how they actually appear in test soundscapes.

**Interaction with Focal Loss:**
Because targets are now in `[0, 0.3, 1.0]` rather than `{0, 1}`, the sigmoid output targets `p = 0.3` for secondary species. Focal Loss computes `p_t = p * y + (1-p) * (1-y)` which for a secondary species with target 0.3 and prediction 0.5 gives `p_t = 0.5 * 0.3 + 0.5 * 0.7 = 0.5`. The focal weight `(1-0.5)^2 = 0.25` is the same as for a maximally uncertain primary species. This is appropriate — secondary labels should be treated as difficult (they are).

---

### `mixup_collate` — Deep Dive

#### What Mixup is and where it came from

Mixup (Zhang et al., 2018) is a data augmentation method that creates new training examples by linearly interpolating pairs of existing examples:

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$. The mixed example $(\tilde{x}, \tilde{y})$ is a convex combination of two real training examples. The model is trained to predict the convex combination of the labels.

**The key insight behind Mixup:** standard ERM (empirical risk minimisation) trains the model on individual data points. The model learns sharp decision boundaries that pass exactly between the training points but may not generalise. Mixup regularises by requiring that the model's behaviour between training points is smooth and linear — the model cannot be overconfident about any interpolation between two real examples.

#### The Beta distribution shape at α = 0.3

$\text{Beta}(0.3, 0.3)$ has a strongly U-shaped distribution. The probability density is highest near 0 and 1, meaning:

- ~60% of the time, $\lambda > 0.8$ (one sample dominates strongly)
- ~20% of the time, $0.4 < \lambda < 0.6$ (roughly equal mixture)
- The remaining ~20% is intermediate

This is the right choice for audio classification. At $\alpha = 0.3$, most mixed examples still clearly "sound like" one species (the dominant component), but occasionally two species are truly mixed. The model learns to handle both situations. At $\alpha = 1.0$ (uniform distribution), roughly equal mixtures are common — the spectrogram is so blended that the class signal is destroyed, hurting convergence. At $\alpha = 0.1$, mixing is so rare that regularisation is negligible.

#### Why Mixup is done in `collate_fn` rather than in `__getitem__`

`__getitem__` produces a single `(x, y)` pair. To mix two samples, you need two samples simultaneously. You cannot access the pair inside `__getitem__`. The collation function receives the whole batch at once (after all `__getitem__` calls), making it the natural place to implement Mixup.

This also means Mixup sees a full batch of 32 samples and mixes each with a randomly chosen partner from the same batch. No extra data loading is required.

#### Single λ for the whole batch

```python
lam = np.random.beta(alpha, alpha)
idx = torch.randperm(xs.size(0))
xs_m = lam * xs + (1 - lam) * xs[idx]
ys_m = lam * ys + (1 - lam) * ys[idx]
```

One $\lambda$ is sampled per batch, not per sample. Every pair in the batch shares the same mixing coefficient. This is slightly simpler than the per-sample variant and in practice performs similarly. The `torch.randperm` generates a random permutation of batch indices, so sample $i$ is mixed with sample `idx[i]` — a random partner from the batch.

#### Why Mixup is particularly well-suited to this competition

1. **Species co-occurrence:** In natural soundscapes, multiple bird species call simultaneously. Mixing two single-species recordings creates a plausible synthetic soundscape. The model is trained to recognise species in the presence of other species — exactly what it will face at inference on the test soundscapes.
2. **Rare species amplification:** If a batch contains one sample of a species with only 3 training examples (rare species) and 31 common species, Mixup causes that rare species' signal to appear in 31 out of 32 mixed examples (as the subdominant component). This artificially increases rare-species exposure.
3. **Label smoothing effect:** Mixed labels prevent the model from becoming overconfident (predicting exact 0s and 1s from the sigmoid). Overconfident predictions lead to poor calibration and worse ROC-AUC.

#### Why validation disables Mixup

```python
collate_fn=lambda b: mixup_collate(b, CFG["mixup_alpha"], use_mixup=train),
```

When `train=False`, `use_mixup=False` is passed, and `mixup_collate` returns `xs, ys` unchanged. This is critical for two reasons:

1. **AUC measurement:** The validation AUC must reflect how the model performs on real, clean samples — not Mixup blends. Measuring AUC on mixed targets would give a different value than what the Kaggle metric measures, making the validation AUC unreliable as a stopping criterion.
2. **Early stopping correctness:** If validation used Mixup, the noise in the mixed labels would make AUC estimates noisier, potentially triggering early stopping prematurely or masking genuine improvements.

---

### `make_loader` — Configuration Choices

```python
def make_loader(frame, train: bool):
    ds = ClipDataset(frame, MEL_OUT_DIR, train=train)
    return DataLoader(
        ds,
        batch_size=CFG["batch_size"],   # 32
        shuffle=train,
        num_workers=CFG["num_workers"], # 0
        collate_fn=lambda b: mixup_collate(b, CFG["mixup_alpha"], use_mixup=train),
        drop_last=train,
    )
```

#### `batch_size=32`

32 is a standard choice for GPU training. It balances:
- **Gradient quality:** Larger batches give lower-variance gradient estimates, but past ~256 samples the improvement plateaus while memory cost grows linearly.
- **Regularisation:** Smaller batches introduce gradient noise that acts as implicit regularisation. 32 is in the sweet spot.
- **Memory:** On a 16GB T4, 32 samples of shape `(1, 128, 251)` in float32 = 32 × 128 × 251 × 4 bytes = ~4MB per batch. EfficientNet-B2 has ~9.2M parameters, adding ~37MB in fp32 mode. The full batch including intermediate activations fits comfortably within 16GB.

#### `shuffle=train`

Training samples are shuffled to prevent the model from learning sequence-based patterns (e.g., all samples of species A followed by species B). Validation is not shuffled — it does not affect AUC since `roc_auc_score` is order-invariant, but keeping it unshuffled helps with debugging (you can compare validation batches across runs).

#### `num_workers=0` and why not more

`num_workers > 0` spawns background processes that load data in parallel with the GPU computation. On local machines with SSDs and Unix, `num_workers=4` or `8` is typical. On Kaggle:

1. **Shared memory limits:** Kaggle containers cap `/dev/shm` at 5.4GB. Worker processes use shared memory to pass tensors to the main process. With `num_workers=4` and large mel arrays, this can exhaust `/dev/shm` and crash the kernel with a cryptic "Bus error" or "killed" message.
2. **`num_workers=0` on Windows:** Since this repository may also run locally on Windows, `num_workers=0` avoids the Windows multiprocessing pickling issue where `DataLoader` with `num_workers > 0` requires a `if __name__ == "__main__":` guard.
3. **Pre-loaded `.npy` files:** Since mels are already on disk as numpy arrays (not compressed audio), loading a single sample takes ~0.1ms. The CPU overhead is low enough that parallelism is not the bottleneck — GPU computation is.

#### `drop_last=train`

During training, if the dataset size (say 15,000 samples) is not exactly divisible by the batch size (32), the final batch may contain only a few samples (e.g., 24). BatchNorm layers compute batch statistics (mean and variance) over the batch dimension. A batch of 24 vs 32 is generally fine. However:

- **Extremely small batches** (e.g., 2–5 samples after rare-species-heavy filtering in a fold) would give badly estimated BatchNorm statistics, and the loss would spike, corrupting gradients.
- Dropping the last incomplete training batch eliminates this edge case entirely.

During validation, `drop_last=False` ensures all samples in the fold are evaluated. Missing even a few validation samples would bias the AUC estimate.

---

## Cell 9: Preparing the Final Training DataFrame

### The reconciliation problem

Cell 4 precomputed mels from `train_audio`. It ran with a `try/except` wrapper and silently skipped failures. If 50 audio files failed (corrupted downloads, zero-length files, network timeouts), then 50 rows in `df` point to mels that don't exist on disk. If `ClipDataset.__getitem__` encounters a missing file, `np.load` throws a `FileNotFoundError` which crashes the DataLoader worker, halting training.

Cell 9 prevents this by building the `available_mels` set and filtering `train_df` to only rows with existing mels. Any failed audio is silently excluded from training.

### Filename normalisation: why it's needed

```python
train_df["filename"] = train_df["filename"].apply(lambda x: x.replace("/", "_"))
```

The `train.csv` file stores filenames in the form `primary_label/XC123456.ogg` (a subdirectory path relative to `train_audio`). For example: `asbfly/XC134829.ogg`.

Cell 4 saved the mel as: `asbfly_XC134829.ogg.npy` (slash replaced with underscore, `.npy` appended). This conversion was done by:

```python
mel_name = row["filename"].replace("/", "_") + ".npy"
```

So to match, `train_df["filename"]` must also have the slash replaced. After the replacement, `train_df["filename"]` contains `asbfly_XC134829.ogg`. The mel file stem (via `f.stem`) is also `asbfly_XC134829.ogg`. They match, and the `isin(available_mels)` filter works correctly.

**What goes wrong without this step:** `train_df["filename"]` would contain `asbfly/XC134829.ogg` (with a slash). This would never match `asbfly_XC134829.ogg` in `available_mels`, so `train_df` would be empty after filtering, and training would fail with a zero-length dataset.

### Soundscape pseudo-label appending

```python
if len(pseudo_df) > 0:
    train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
```

`pseudo_df` rows (from Cell 5) have different filename patterns: `soundscape_XC99999_seg_2`. The mel was saved as `soundscape_XC99999_seg_2.npy`. These do not go through the slash-replacement step (they have no slashes), so they are appended directly. `ignore_index=True` re-numbers the combined DataFrame from 0 so that all 15 training runs use consistent integer indices for `val_idx` bookkeeping in the OOF array.

**Strategic purpose of soundscape pseudo-labels:**
Some species may have zero or very few (< 5) examples in `train_audio`. These species will always land in the `__rare__` StratifiedKFold bucket and will have near-zero training exposure. Cell 5 adds up to 5 spectrogram segments per rarity-species from the annotated soundscape recordings, providing at least some direct training signal for the hardest classes.

---

## Cell 10: The 15-Model Training Loop

### Outer structure: why iterate architectures first, then folds

```python
for arch in CFG["architectures"]:
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, _strat_key)):
```

The outer loop is over architectures (resnet50, efficientnet_b0, efficientnet_b2). The inner loop is over folds. This means all 5 folds of `resnet50` are trained before any fold of `efficientnet_b0` starts.

This ordering matters for the `oof_preds` arrays: `oof_preds[arch]` is allocated once per architecture and filled fold-by-fold. If the architecture dimension were the inner loop, you would need to partially fill different architecture arrays simultaneously, which is more complex to track.

It also means the GPU is freed (`del model; torch.cuda.empty_cache()`) after each fold, and a new model is created for the next fold. All 15 folds run sequentially, never in parallel, so peak GPU memory is that of a single model.

---

### StratifiedKFold: the stratification problem in depth

#### Why stratification matters

Without stratification, a random 80/20 split of 15,000 samples might accidentally put all 3 samples of a very rare species into the training set (80%) and zero into validation. The model would then never produce a prediction for that species during validation, causing its column in `fp` to be all-zeros, `roc_auc_score` to be undefined, and the species to be excluded from the macro AUC — making the validation metric optimistic.

Stratification forces each species to appear in both training and validation in proportion to its overall frequency. For a 5-fold CV, each species gets exactly 80% of its samples in training and 20% in validation per fold.

#### The rare-species problem with StratifiedKFold

`StratifiedKFold` with 5 folds requires each class to have at least 5 samples. If `primary_label = "rednod"` has only 3 samples total, you cannot put 1 sample per fold without fractional samples. Scikit-learn's implementation raises:

```
ValueError: The least populated class in y has only 2 members, which is too few.
The minimum number of groups for any class cannot be less than n_splits=5.
```

#### The `__rare__` fix

```python
_label_counts = train_df["primary_label"].value_counts()
_strat_key = train_df["primary_label"].apply(
    lambda x: x if _label_counts.get(x, 0) >= CFG["folds"] else "__rare__"
)
```

All species with fewer than 5 samples are collapsed into a single synthetic class `"__rare__"`. The total number of `__rare__` samples might be 50–200 (depending on how many rare species exist). StratifiedKFold then treats the rare-species pool as a single class and ensures ~20% of that pool lands in each validation fold.

**What this means statistically:**
- Common species (≥5 samples): properly stratified across folds. Validation always contains representative examples.
- Rare species (<5 samples): distributed across folds as a pool rather than individually. Some rare species may appear zero times in a given fold's validation set — their AUC contribution is simply excluded for that fold via the `ft[:, j].sum() > 0` guard.

This is a pragmatic compromise. The alternative — excluding all rare species from training entirely — would be worse, since even one or two training examples of a rare species help the model learn its spectrographic signature.

---

### Fresh model initialisation per fold

```python
model = BirdCLEFModel(arch, n_classes=n_classes, pretrained=True).to(device)
```

This is one of the most important correctness properties of the CV setup. Every fold starts from exactly the same pretrained ImageNet weights — except for the replaced input conv, which starts from random init (new random weights sampled each time since it's a new `nn.Conv2d`).

**Why this is required for valid OOF:**
Imagine fold 3 carries over the weights from fold 2. Fold 2's training data has influenced those weights. Now when fold 3 tries to validate on its held-out data, the model has implicitly "seen" some of that data through fold 2's training. The OOF prediction for those samples is no longer truly out-of-fold — it is corrupted by cross-fold data leakage. The OOF AUC would be optimistically biased.

**How the pretrained weights help with this fold interdependence concern:**
Since all folds start from the same ImageNet checkpoint, any weight differences between folds at the end of training are purely due to the specific data seen in that fold's training split. This is correct — we want fold differences to reflect generalisation, not initialisation variance.

**The input conv initialisation:**
After `timm.create_model(arch, pretrained=True)`, the conv1/conv_stem has 3-channel weights from ImageNet. The code immediately replaces it with a new `nn.Conv2d(1, ...)` with random weights. This means the model learns the 1-channel → feature-map transformation from scratch for every fold. All other layers start from pretrained weights.

This design choice (replace only the first conv) rather than (random initialise everything) is intentional. The pretrained model spent millions of iterations learning to detect edges, corners, gradients, textures, and semantic mid-level features. These features appear in mel spectrograms too — frequency ridges correspond to visual edges, harmonic patterns correspond to textures. Retaining them massively accelerates convergence and improves final AUC compared to training from scratch.

---

### Optimizer: AdamW with weight decay

```python
optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
```

**AdamW vs Adam:** Standard Adam absorbs the weight decay into the adaptive learning rate calculation, which mathematically means the effective regularisation varies per-parameter. AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update, applying it directly to the parameters:

$$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

where $\eta$ is the learning rate, $\lambda$ is the weight decay coefficient (1e-4), and $\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ is the adaptive step. This cleaner formulation of regularisation consistently outperforms Adam on pretrained networks where different layers need different update magnitudes.

**`weight_decay=1e-4`:** Applies $L_2$ regularisation to all parameters. This penalises large weights, preventing the model from memorising training examples (overfitting). With 30 epochs and a relatively small dataset, L2 regularisation is important. Too high (e.g., 1e-2) would prevent the model from fitting the training data at all; too low (e.g., 1e-6) would provide negligible regularisation.

---

### Learning rate schedule: warmup + cosine decay

```python
warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=CFG["warmup_epochs"])
cosine_sched = CosineAnnealingLR(optimizer, T_max=CFG["epochs"] - CFG["warmup_epochs"], eta_min=1e-6)
scheduler    = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[CFG["warmup_epochs"]])
```

#### The warmup phase (epochs 1–5)

The actual learning rate per epoch during warmup:

| Epoch | start_factor + progress × (end_factor - start_factor) | Effective LR |
|-------|------------------------------------------------------|-|
| 1     | 0.1 + 0/4 × 0.9 = 0.1 → scaled | 0.1 × 5e-4 = **5.0e-5** |
| 2     | 0.1 + 1/4 × 0.9 = 0.325         | 3.25e-4 |
| 3     | 0.1 + 2/4 × 0.9 = 0.55          | 2.75e-4 |
| 4     | 0.1 + 3/4 × 0.9 = 0.775         | 3.875e-4 |
| 5     | 0.1 + 4/4 × 0.9 = 1.0           | **5.0e-4** |

`LinearLR(start_factor=0.1)` initialises the LR to `0.1 × base_lr = 5e-5` and linearly ramps to `1.0 × base_lr = 5e-4` by epoch 5.

**Why warmup is needed here specifically:**
The input conv was just randomly initialised. Its output features in epoch 1 are essentially random noise. The rest of the network (BatchNorm, subsequent convs) will receive garbage activations and generate large, noisy gradients as they try to adapt. If the LR starts at 5e-4 immediately, these noisy gradients will make large updates to the pretrained weights in layers 2–50 of ResNet50, destroying the carefully learned ImageNet features in the very first epoch. A 10× smaller starting LR limits the damage while the input conv "warms up" to producing meaningful features.

By epoch 5, the input conv has learned a reasonable 1-channel to 64-channel mapping (for ResNet50), the downstream layers have adapted to the mel input distribution, and the full LR 5e-4 is safe to use without catastrophic forgetting.

#### The cosine decay phase (epochs 6–30)

$$\text{LR}(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t \cdot \pi}{T_{\max}}\right)\right)$$

where $\eta_{\max} = 5 \times 10^{-4}$, $\eta_{\min} = 10^{-6}$, $T_{\max} = 25$ (epochs 6–30).

Key property: the cosine schedule decreases slowly at first (near epoch 6, the LR is still close to 5e-4) then very slowly near the end (near epoch 30, the LR approaches 1e-6). This contrasts with step decay, which drops the LR abruptly at fixed milestones (e.g., at epoch 15 and epoch 25). The gradual cosine schedule avoids abrupt loss spikes when the LR steps down.

**Why `eta_min=1e-6` instead of 0:**
A non-zero floor prevents the LR from reaching exactly zero. At LR=0, AdamW still applies weight decay (subtraction of `1e-4 × θ`) but makes zero gradient updates. The net effect is the model slowly shrinks toward zero — not useful. `eta_min=1e-6` ensures the model keeps making small gradient-based updates until early stopping or epoch exhaustion.

---

### The training step in detail

```python
model.train()
train_loss = 0.0
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    train_loss += loss.item()
train_loss /= max(len(train_loader), 1)
scheduler.step()
```

#### `model.train()`

Switches all layers to training mode. The key differences from eval mode:
- **BatchNorm:** Uses batch statistics (mean/variance computed from the current batch) rather than running statistics. Updates the running statistics.
- **Dropout:** Randomly zeroes activations with probability p=0.4 (as specified in the classification head).

These must be disabled during validation with `model.eval()`.

#### `optimizer.zero_grad()`

Clears accumulated gradients from the previous batch. PyTorch accumulates gradients by default (this is intentional for gradient accumulation use-cases). Here we always want fresh gradients per batch.

#### `criterion(model(x), y)`

`model(x)` returns raw logits of shape `(batch, 234)`. The logits are not passed through sigmoid — that is done inside `BinaryFocalLoss` via `torch.sigmoid(logits)`. Passing pre-sigmoid logits is numerically more stable than using `binary_cross_entropy(sigmoid(logits), y)` because the numerically stable log-sum-exp trick can be applied inside `binary_cross_entropy_with_logits`.

`y` is the soft multi-hot target, possibly further blended by Mixup. Values are in `[0, 1]`, not just `{0, 1}`. `BinaryFocalLoss` handles this correctly since it computes BCE element-wise (which works for any real target in [0,1]) and then applies the focal weight.

#### `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

This rescales all gradients so that the L2 norm of the concatenated gradient vector across all parameters is at most 1.0. If the norm is already ≤ 1.0, nothing happens:

$$g \leftarrow g \cdot \min\left(1, \frac{1.0}{\|g\|_2}\right)$$

**Why this is needed with Mixup + Focal Loss:**
Focal Loss already down-weights easy examples, focusing the loss on hard ones. With Mixup at α=0.3 and occasionally equal mixtures (λ≈0.5), a single batch might contain examples where neither species occupies more than 60% of the signal. The model produces large gradients trying to reconcile these ambiguous targets. Additionally, early in training (epoch 1–2), the input conv is random and produces large activations that propagate into large gradients. Without clipping, a single bad batch can produce gradients thousands of times larger than normal, making a single update step wildly oversized and destabilising the optimisation trajectory.

Clipping to norm 1.0 is conservative (ResNets and EfficientNets typically train stably with clip values up to 5.0), but conservative is appropriate here given the combination of pretrained weights + random first conv + soft Mixup targets + Focal Loss.

---

### The validation loop and AUC computation

```python
model.eval()
val_loss   = 0.0
fold_preds, fold_targets = [], []
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        val_loss += criterion(logits, y).item()
        fold_preds.append(torch.sigmoid(logits).cpu().numpy())
        fold_targets.append(y.cpu().numpy())
val_loss /= max(len(val_loader), 1)
```

#### `model.eval()` and `torch.no_grad()`

`model.eval()` disables Dropout and switches BatchNorm to use running statistics. `torch.no_grad()` disables gradient computation, reducing memory usage by ~60% (no need to store activations for backprop) and speeding up the forward pass by ~30%.

Both are critical during validation. Running `model.train()` during validation would give different BatchNorm statistics each epoch (since batch statistics fluctuate), making the AUC noisy and unreliable as a stopping criterion.

#### Building `fold_preds` and `fold_targets`

Each validation batch produces sigmoid probabilities in `[0,1]` (from `torch.sigmoid(logits)`) and targets (the soft multi-hot labels from `ClipDataset`). These are accumulated as Python lists of numpy arrays, then stacked:

```python
fp = np.vstack(fold_preds)   # shape: (N_val, 234)
ft = np.vstack(fold_targets) # shape: (N_val, 234)
```

#### AUC computation: per-class, then macro average

```python
auc_scores_ep = [
    roc_auc_score(ft[:, j], fp[:, j])
    for j in range(n_classes)
    if ft[:, j].sum() > 0 and (1 - ft[:, j]).sum() > 0
]
val_auc = np.mean(auc_scores_ep) if auc_scores_ep else 0.0
```

**Why macro-average?** The competition uses macro-averaged ROC-AUC: compute AUC for each species independently, then average across species. This treats every species equally, whether it has 10,000 training examples or 3. The validation metric must mirror this exactly — otherwise you are optimising a different objective from what Kaggle measures.

**The constant-column guard:** `ft[:, j].sum() > 0` checks that species j appears at least once as a positive in the validation fold. `(1 - ft[:, j]).sum() > 0` checks that it has at least one negative. If all validation samples show `ft[:, j] = 0` (species absent from every val sample), then `roc_auc_score` would raise a `ValueError: Only one class present in y_true`. The guard skips such columns. Note: with soft labels, `ft[:, j].sum() > 0` also catches any column with at least one secondary label (value 0.3), since 0.3 > 0.

**Soft vs hard targets for AUC in the per-epoch loop:** `roc_auc_score(ft[:, j], fp[:, j])` is computed with soft targets (0.0, 0.3, 1.0). ROC-AUC measures the probability that a random positive ranks above a random negative. With soft targets, "positives" are samples with `ft[:,j] = 1.0` or `ft[:,j] = 0.3`, and "negatives" are samples with `ft[:,j] = 0.0`. The ranking is correct — the AUC measures whether predictions for primary and secondary species are higher than for absent species, which is the right signal for training monitoring.

---

### Early stopping: interaction with warmup

```python
if val_auc > best_val_auc:
    best_val_auc   = val_auc
    patience_count = 0
    best_state     = copy.deepcopy(model.state_dict())
    best_fold_preds = fp.copy()
else:
    patience_count += 1

if patience_count >= CFG["patience"]:   # patience = 8
    print(f"    Early stopping at epoch {epoch+1}")
    break
```

**Patience of 8:** With 30 epochs total and 5 warmup epochs, the cosine decay phase runs for epochs 6–30. Setting `patience=8` means the model can run for up to 8 consecutive non-improving epochs before stopping. This prevents premature stopping during the LR warmup (where AUC may not improve every epoch as the randomly-init'd conv adapts) and during the end of cosine decay (where the LR is very low and improvements are small).

**`copy.deepcopy(model.state_dict())`:** Creates a completely independent copy of all weight tensors. Without `deepcopy`, `best_state = model.state_dict()` would store a reference to the live parameter tensors, which would then be overwritten by subsequent training steps. `deepcopy` ensures the best state is frozen at the moment of the best AUC.

**`best_fold_preds = fp.copy()`:** Similarly, this captures the validation predictions at the epoch with the best AUC, not the final epoch. These predictions are used for OOF. Using predictions from the checkpoint epoch (rather than the final epoch) is important because the final epoch may have slightly overfit to the training set — the checkpoint epoch represents the model's peak generalisation.

---

### Checkpoint saving

```python
model.load_state_dict(best_state)
ckpt_path = f"/kaggle/working/{arch}_v8_fold{fold_idx}.pt"
torch.save(model.state_dict(), ckpt_path)
```

The model weights are restored from the best checkpoint (`best_state`) before saving. This means the `.pt` file contains the weights from the best epoch, not the last epoch. The inference notebook loads these checkpoints to make final competition predictions.

**`/kaggle/working/` persistence:** Files written to `/kaggle/working/` in a Kaggle notebook are automatically saved and can be downloaded or attached to subsequent notebooks as a dataset. After training, you would download the 15 `.pt` files, create a Kaggle dataset named `birdclef-2026-v8-model`, and attach it to the inference notebook.

---

### OOF array population

```python
oof_preds[arch][val_idx] = best_fold_preds
```

**What `oof_preds[arch]` looks like after all 5 folds:**

For a dataset of 15,000 samples and 5 folds, each fold has ~3,000 validation samples. After fold 0: rows `val_idx_0` are filled. After fold 1: rows `val_idx_1` are filled. … After fold 4: rows `val_idx_4` are filled. Since `StratifiedKFold` produces non-overlapping, exhaustive splits, every row is written exactly once. The initial zero fill is overwritten for every row.

**Why the OOF predictions are unbiased:**
Each prediction `oof_preds[arch][i]` was made by a model that never saw sample `i` during training. In classical statistical terms, this is analogous to a leave-one-out (LOO) cross-validated prediction — just using folds instead of individual samples. The OOF estimate has no look-ahead bias.

**Numpy advanced indexing:** `oof_preds[arch][val_idx]` where `val_idx` is an integer array uses numpy's advanced (fancy) indexing. The assignment `arr[val_idx] = values` sets `arr[val_idx[0]] = values[0]`, `arr[val_idx[1]] = values[1]`, etc. This is equivalent to a for loop over indices but executing at C speed.

---

### GPU memory management

```python
del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

After saving the checkpoint, the model is deleted to free GPU memory before the next fold. Without this:
- A new model is created at the start of the next fold iteration
- The old model still occupies ~75–150MB of GPU VRAM
- After 5 folds, 5× the model size would be resident simultaneously
- On a 16GB T4 training EfficientNet-B2 (the largest of the 3 architectures), this could cause OOM errors by fold 3 or 4

`torch.cuda.empty_cache()` does not immediately free memory (the garbage collector handles that when `del model` runs) but flushes CUDA's internal cache of freed blocks back to the OS allocator, making them available for the next allocation. This prevents gradual fragmentation of the CUDA memory pool across 15 training runs.

---

## Cell 11: OOF Ensemble AUC and Summary

### Per-architecture diagnostics: what to look for

```python
for arch in CFG["architectures"]:
    fold_aucs = arch_scores[arch]
    print(f"   Fold AUCs : {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"   Mean AUC  : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
```

**Reading the fold AUCs:** Typical healthy output might look like:
```
resnet50:          [0.7821, 0.7743, 0.7889, 0.7802, 0.7756]  mean=0.7802 ± 0.0052
efficientnet_b0:   [0.7534, 0.7612, 0.7498, 0.7571, 0.7489]  mean=0.7541 ± 0.0046
efficientnet_b2:   [0.7703, 0.7821, 0.7688, 0.7744, 0.7712]  mean=0.7734 ± 0.0050
```

**What low variance tells you:** If std < 0.007, the splits are well-balanced and the model is stable across different data subsets. This is expected with stratified folds on a large dataset.

**What high fold AUC variance (std > 0.015) tells you:**
- One fold may contain a disproportionate number of easy or hard species in its validation set
- The dataset may be too small for reliable 5-fold CV
- The model is sensitive to which specific rare-species samples land in validation

**What to do if one architecture is consistently 0.02+ lower than the others:**
Consider excluding it from the ensemble (Cell 11's ensemble is a simple average — a weak architecture pulls it down). Alternatively, tune that architecture's hyperparameters separately.

**What a decreasing mean AUC per architecture across architectures tells you:**
If resnet50 > efficientnet_b2 > efficientnet_b0, ResNet50 is the strongest single model. This is expected — ResNet50 has ~25M parameters vs EfficientNet-B0's ~5M, giving it higher capacity for this 234-class problem.

---

### The ensemble mechanism

```python
ensemble_oof = np.mean([oof_preds[a] for a in CFG["architectures"]], axis=0)
```

This computes the element-wise mean of three arrays, each of shape `(N_train, 234)`:

$$\hat{p}_{\text{ensemble}}[i, j] = \frac{1}{3} \left(\hat{p}_{\text{resnet50}}[i, j] + \hat{p}_{\text{efficientnet\_b0}}[i, j] + \hat{p}_{\text{efficientnet\_b2}}[i, j]\right)$$

**Why averaging predictions (not logits) is appropriate:**
Averaging probability predictions is a form of probability averaging ensemble, also known as the arithmetic mean ensemble. It is well-calibrated when individual models are individually well-calibrated (which sigmoid-based models typically are after sufficient training).

An alternative is to average logits before sigmoid: $\hat{p} = \sigma(\text{mean}(\text{logits}))$. This is the "logit averaging" or "logistic averaging" ensemble. For well-calibrated individual models, both approaches give similar results. Averaging probabilities is simpler and slightly more robust to miscalibrated individual models.

**Why three architectures give better results than one:**
The architectures make different errors. ResNet50 excels at capturing mid-frequency harmonic patterns (strong in ResNet's hierarchical conv stack). EfficientNet-B2 excels at capturing fine-grained frequency detail (compound scaling balances width/depth/resolution). EfficientNet-B0 is a fast, simpler version that may excel at common-species discrimination.

When the three models disagree on a sample, their average will be less confident (closer to 0.5) — appropriate for genuinely ambiguous examples. When all three agree, the average confidently reflects the consensus. This ensemble diversity directly improves ROC-AUC by providing better-ranked probability scores.

**Theoretical justification (bias-variance tradeoff):**
For $K$ models with predictions variance $\sigma^2$ and covariance $\rho\sigma^2$ between pairs:

$$\text{Var}\left(\frac{1}{K}\sum_{k=1}^K \hat{p}_k\right) = \frac{\sigma^2}{K} + \frac{K-1}{K}\rho\sigma^2$$

As $K$ increases, the variance decreases — but only if $\rho < 1$ (models make different errors). With three architecturally diverse models ($\rho \approx 0.7$–$0.85$ for these architectures), even a simple average reduces prediction variance by ~15–25%.

---

### Target reconstruction and binarization

```python
oof_targets = np.zeros((len(train_df), n_classes), dtype=np.float32)
for i, row in train_df.iterrows():
    oof_targets[i] = row_to_soft_multihot(row["primary_label"], row.get("secondary_labels", "[]"))
oof_targets_bin = (oof_targets >= 0.5).astype(np.float32)
```

**Why rebuild targets instead of using the targets from the training loop:**
In Cell 10, fold_targets are collected per fold and discarded after each fold's loop. There is no single array accumulating all training targets. Cell 11 rebuilds the full target array by iterating over the complete `train_df`.

**Why binarize at 0.5 for the final summary AUC:**
`roc_auc_score` is defined for binary classification. With soft targets (0.0, 0.3, 1.0), samples with secondary labels (0.3) would be treated as fractional positives, which has no standard statistical meaning in terms of the ROC curve's TP/FP/TN/FN.

More importantly, the competition ground truth is binary: a species is either present (primary label = 1) or not (0). Secondary labels in training are an annotation convenience — the competition's own test labels are stored as binary present/absent. To make the OOF AUC comparable to the Kaggle metric, targets must be binarized.

Threshold at 0.5 produces:
- Primary label (1.0) → **1** (present) ✓
- Secondary label (0.3) → **0** (not counted as present) ✓
- Absent (0.0) → **0** ✓

This exactly reproduces the Kaggle ground truth interpretation.

---

### The ensemble AUC

```python
ensemble_auc_scores = [
    roc_auc_score(oof_targets_bin[:, j], ensemble_oof[:, j])
    for j in range(n_classes)
    if oof_targets_bin[:, j].sum() > 0 and (1 - oof_targets_bin[:, j]).sum() > 0
]
print(f"\n🏆 15-Model Ensemble OOF Macro AUC: {np.mean(ensemble_auc_scores):.4f}")
```

This is the headline metric of the training run. It estimates what the 3-architecture ensemble would score on the Kaggle public leaderboard.

**Statistical properties of the OOF estimate:**
- **Unbiasedness:** Each prediction was made by a model trained without that sample. The estimate is analogous to a 5-fold cross-validated score.
- **Slight pessimism:** Each individual model was trained on 80% of the data. The final Kaggle submission uses models trained on 100% (though in this notebook we save fold-specific models). A model trained on 100% has slightly higher accuracy than 80%-data models, so the true Kaggle score will be slightly above the OOF estimate.
- **Residual variance:** The 5-fold estimate still has sampling variance. For small datasets (<5,000 samples), this variance can be ±0.02 AUC. For this dataset with ~15,000+ samples, the variance is typically ±0.005–0.01.

**Calibration to Kaggle score:**
In practice, OOF macro AUC of X correlates with a Kaggle public LB score of approximately X - 0.02 to X + 0.02 for a well-regularised model on BirdCLEF-style data. The gap widens if:
- Train and test soundscapes have different background noise characteristics
- Some rare species have no training examples at all (they will score AUC = 0.5 = random)
- The model is overfit (OOF AUC looks good but Kaggle score is lower)

---

## Summary: End-to-End Flow Through Cells 8–11

The following table traces a single sample from disk to final OOF prediction:

| Stage | Location | What happens |
|-------|----------|-------------|
| Load `.npy` | `ClipDataset.__getitem__` | Float32 array `(128, T_actual)` loaded from `/kaggle/working/mels_v8/` |
| Random crop | `ClipDataset.__getitem__` | Cropped to `(128, 251)`, zero-padded if needed |
| Add channel dim | `ClipDataset.__getitem__` | `(128, 251)` → `(1, 128, 251)` |
| Soft multihot label | `ClipDataset.__getitem__` | `(234,)` vector with 1.0 primary, 0.3 secondary |
| Mixup | `mixup_collate` | Linear blend with a random batch partner, λ ~ Beta(0.3, 0.3) |
| `to(device)` | Training loop | Moved to GPU |
| Forward pass | Training loop | `(32, 1, 128, 251)` → logits `(32, 234)` |
| Focal Loss | Training loop | Maps logits + soft targets → scalar loss |
| Backward + clip | Training loop | Gradients clipped to L2-norm ≤ 1.0 |
| AdamW step | Training loop | Weights updated with decoupled weight decay |
| Validation sigmoid | Val loop | Logits → probabilities `(N_val, 234)` in [0,1] |
| Macro AUC | Val loop | Per-class + averaged → `val_auc` scalar |
| Best checkpoint | Training loop | State saved when `val_auc > best_val_auc` |
| OOF fill | After fold | `oof_preds[arch][val_idx] = best_fold_preds` |
| Ensemble average | Cell 11 | Mean of 3 arch OOFs → `(N_train, 234)` |
| Final AUC | Cell 11 | Binarised targets + ensemble preds → macro AUC |

---

## Quick Reference: Variable Shapes

| Variable | Shape | Description |
|----------|-------|-------------|
| `mel` (per sample) | `(128, 251)` | Log-mel normalised to [0,1], freq × time |
| `x` (per sample) | `(1, 128, 251)` | With channel dim, ready for CNN |
| `y` (per sample) | `(234,)` | Soft multi-hot: primary=1.0, secondary=0.3, absent=0.0 |
| `xs` (per batch) | `(32, 1, 128, 251)` | Batch of spectrograms post-Mixup |
| `ys` (per batch) | `(32, 234)` | Batch of labels post-Mixup |
| `logits` | `(32, 234)` | Raw model output, unbounded |
| `fp` (per val fold) | `(N_val, 234)` | Sigmoid probabilities for val fold |
| `ft` (per val fold) | `(N_val, 234)` | Soft multi-hot targets for val fold |
| `auc_scores_ep` | `(≤234,)` | Per-species AUC for one epoch, skipping constant species |
| `best_fold_preds` | `(N_val, 234)` | Val predictions at best-AUC epoch for this fold |
| `oof_preds[arch]` | `(N_total, 234)` | Full OOF: each row written once, by fold model |
| `ensemble_oof` | `(N_total, 234)` | Mean across 3 archs of OOF preds |
| `oof_targets` | `(N_total, 234)` | Rebuilt soft multi-hot targets for all training samples |
| `oof_targets_bin` | `(N_total, 234)` | Binarised at 0.5: primary→1, secondary+absent→0 |
| `ensemble_auc_scores` | `(≤234,)` | Per-species AUC for the ensemble, skipping constant species |

---

## Key Design Decisions and Their Reasoning

| Decision | Alternative considered | Why this choice wins |
|----------|-----------------------|---------------------|
| Precompute mels to disk | On-the-fly mel computation | 50–150× faster `__getitem__`, GPU never starved waiting for CPU |
| Random crop (train) | FixedCrop of first 5 seconds | Forces temporal invariance; call may be anywhere in recording |
| Centre crop (val) | Random crop | Deterministic; AUC must be comparable across epochs |
| Beta(0.3, 0.3) Mixup | α=1.0 (uniform), no Mixup | U-shaped keeps species signal intact; rare species appear more often |
| Mixup in `collate_fn` | Mixup in `__getitem__` | Needs two samples simultaneously; `collate_fn` has the full batch |
| `num_workers=0` | `num_workers=4` | Kaggle `/dev/shm` limits; pre-loaded `.npy` amortises loading cost |
| `drop_last=True` (train) | Keep last batch | Prevents BatchNorm instability from tiny final batches |
| StratifiedKFold + `__rare__` | Random split | Ensures each species in both train and val; avoids ValueError |
| Fresh model per fold | Re-use model from previous fold | Prevents cross-fold data leakage into OOF estimates |
| Pretrained ResNet50/EfficientNet | Train from scratch | ImageNet features transfer to spectrograms; 3–5× faster convergence |
| Replace only input conv | Replace all layers (full fine-tune) | Retains pretrained features; only ~0.1% of params are random init |
| Linear warmup (5 epochs) | No warmup (start at full LR) | Protects pretrained weights from large early-epoch gradients |
| Cosine decay | Step decay (LR ÷ 10 at epoch 15, 25) | No abrupt loss spikes; smoother convergence near optimum |
| `clip_grad_norm_(1.0)` | No clipping | Prevents catastrophic gradient steps from Mixup + Focal Loss bursts |
| AdamW with weight_decay=1e-4 | Adam (no weight decay) | Proper L2 regularisation; AdamW decouples from adaptive LR |
| Save `best_fold_preds` not last | Save last-epoch predictions | Best epoch = best generalisation; last epoch may have slightly overfit |
| Binarise OOF targets at 0.5 | Use soft targets for AUC | Matches competition ground truth definition; `roc_auc_score` is defined |
| Simple mean ensemble | Weighted ensemble, stacking | Unbiased; no hyperparameter tuning needed; works well when models are comparable |
