# Phase 1 Improvements: 0.648 → 0.9 Journey

## Overview
Phase 1 implements two key improvements to boost BirdCLEF score from **0.648 to ~0.7+**:

1. **Light Augmentation** during training
2. **Per-Species Threshold Optimization** for inference

---

## 1. Light Augmentation (Training)

### What Changed
Updated `ClipDataset` class with gentle data augmentation applied only during training:

#### Time Masking
- Randomly masks **10-15% of mel-spectrogram frames**
- Applied 50% of the time
- Helps model learn robustness to temporal gaps
- Example: if 251 frames total, mask 25-38 frames at random position

#### Frequency Masking  
- Randomly masks **5-10 mel bins** (out of 64 total)
- Applied 50% of the time
- Helps model be robust to frequency-band loss
- Conservative range to avoid over-corrupting audio

### Why This Works
- **Problem**: Without augmentation, model overfits to training distribution
- **Solution**: Gentle augmentation teaches model temporal/frequency invariance
- **Key difference from before**: 0.648 used NO augmentation. This adds light regularization.
- **Expected gain**: +0.05 to +0.15 AUC

### Implementation
```python
def apply_light_augmentation(self, mel):
    if not self.train:
        return mel
    
    # Time masking: 10-15% of frames
    if np.random.rand() < 0.5:
        T = mel.shape[1]
        mask_frames = np.random.randint(int(0.1 * T), int(0.15 * T))
        mel[:, start:start + mask_frames] = 0.0
    
    # Frequency masking: 5-10 mel bins
    if np.random.rand() < 0.5:
        mask_mels = np.random.randint(5, 10)
        mel[start:start + mask_mels, :] = 0.0
    
    return mel
```

---

## 2. Per-Species Threshold Optimization (Inference)

### What Changed
Instead of using a uniform 0.5 threshold for all species, compute optimal thresholds for each species individually:

#### During Training
1. Collect validation predictions at each fold's best epoch
2. Combine all validation data across 5 folds
3. For each species:
   - Try thresholds from 0.1 to 0.9 (step=0.05)
   - For each threshold, compute F1 score
   - Save threshold that maximizes F1
4. Save all 206 thresholds to `optimal_thresholds.json`

#### During Inference
1. Load per-species thresholds
2. Apply species-specific threshold when building submission
3. This allows rare species to have higher thresholds, common species lower thresholds

### Why This Works
- **Common bird species**: May need lower threshold (0.3-0.4) to catch detections
- **Rare bird species**: May need higher threshold (0.6-0.8) to avoid false positives
- **Uniform 0.5 threshold**: One-size-fits-all approach misses optimality
- **Expected gain**: +0.05 to +0.10 AUC

### Implementation
```python
# During training - collect validation predictions
for sp_idx, sp_id in enumerate(species):
    y_true = combined_targets[:, sp_idx]
    y_pred = combined_preds[:, sp_idx]
    
    best_f1 = 0.0
    best_thresh = 0.5
    
    for thresh in np.linspace(0.1, 0.9, 17):
        y_pred_binary = (y_pred >= thresh).astype(int)
        tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    optimal_thresholds[sp_id] = best_thresh
```

---

## Files Changed

### New Training Notebook
**`birdclef2026-train-weights-v2.ipynb`** (clean version with Phase 1)
- Light augmentation in ClipDataset
- Collects validation predictions at best epoch for each fold
- Computes per-species optimal thresholds after all folds complete
- Saves thresholds to `/kaggle/working/optimal_thresholds.json`

### Updated Inference Notebook
**`birdclef2026-inference.ipynb`** (existing - now uses thresholds)
- Loads optimal thresholds from training
- Applies per-species thresholds instead of uniform 0.5
- Falls back to 0.5 if thresholds file not found

---

## Expected Impact

| Component | Baseline (0.648) | Phase 1 Expected | Gain |
|-----------|------------------|------------------|------|
| Light Augmentation | - | +0.05 to +0.15 | Regularization |
| Per-Species Thresholds | Uniform 0.5 | Optimized per-species | +0.05 to +0.10 |
| **Total Expected** | **0.648** | **~0.70-0.73** | **+0.05-0.08** |

---

## How to Run

### Step 1: Train with Phase 1
Run `birdclef2026-train-weights-v2.ipynb` on Kaggle:
- Trains 5 folds with light augmentation
- Computes per-species optimal thresholds
- Outputs:
  - `model_fold{0-4}.pt` (5 trained models)
  - `optimal_thresholds.json` (206 optimal thresholds)
  - `species.json` (species list)

### Step 2: Generate Submission with Phase 1
Run `birdclef2026-inference.ipynb`:
- Loads trained models + optimal thresholds
- Generates predictions with multi-window ensemble
- Applies per-species thresholds
- Outputs: `submission.csv`

### Step 3: Submit
Upload `submission.csv` to Kaggle competition

---

## Next Steps (Phase 2)

If Phase 1 achieves **~0.70-0.73**, consider Phase 2:

1. **Learning Rate Scheduler** 
   - Cosine annealing + warmup (1-2 epochs)
   - Could improve convergence (+0.02-0.05)

2. **Increase Epochs**
   - Try 20 epochs instead of 15 (if not overfitting)
   - (+0.02-0.05 potential)

3. **Ensemble Diversity**
   - Train with different random seeds
   - Combine predictions (could gain +0.02-0.08)

4. **Better Feature Engineering**
   - Experiment with different mel settings (n_mels, n_fft, hop)
   - Or additional features beyond mel-spectrograms

---

## Key Hyperparameters

```python
CFG = {
    "sr": 16000,              # Sample rate
    "n_mels": 64,             # Mel bins
    "n_fft": 1024,            # FFT size
    "hop": 320,               # Hop length
    "fmin": 60,               # Frequency minimum
    "seconds": 5,             # Audio window
    "batch_size": 32,         # Batch size
    "epochs": 15,             # Training epochs
    "lr": 1e-3,               # Learning rate
}

# Augmentation
TIME_MASK: 10-15% of frames
FREQ_MASK: 5-10 mel bins
AUGMENT_PROB: 0.5 each (50% of time)

# Threshold optimization
THRESHOLD_RANGE: 0.1 to 0.9 (step=0.05)
METRIC: F1 score on validation data
```

---

## Notes

- **Validation-based thresholds**: Computed on actual validation data (not training data) to avoid overfitting
- **Missing species**: Still use taxonomy-based proxy with `alpha=0.2` scaling
- **Multi-window ensemble**: Kept active (3 overlapping windows, 5 folds = 15 predictions per test window)
- **Early stopping**: Patience=5 epochs (stops if no improvement for 5 epochs)

