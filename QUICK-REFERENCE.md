# ⚡ Quick Reference: Phase 1 Deployment

## 🚀 What to Run

### Step 1: Training (Phase 1)
```
File: birdclef2026-train-weights-v2.ipynb
Platform: Kaggle
Time: ~2-3 hours (depending on GPU)
Output: 5 models + optimal thresholds
```

### Step 2: Inference  
```
File: birdclef2026-inference.ipynb
Platform: Kaggle
Input: Models from Step 1
Output: submission.csv
```

---

## 📋 Checklist Before Running

- [ ] Using **v2 notebook** (not corrupted original)
- [ ] Kaggle dataset paths correct:
  - `/kaggle/input/competitions/birdclef-2026/train.csv`
  - `/kaggle/input/competitions/birdclef-2026/train_audio/`
  - `/kaggle/input/competitions/birdclef-2026/test_soundscapes/`
  - `/kaggle/input/competitions/birdclef-2026/sample_submission.csv`
- [ ] Output dataset configured for inference
- [ ] Sufficient disk space (~50GB for mels + models)

---

## 🎯 Key Changes from 0.648

| Component | Change | Impact |
|-----------|--------|--------|
| **Augmentation** | +Light (10-15% time, 5-10 freq) | +0.05 to +0.15 |
| **Thresholds** | +Per-species F1-optimized | +0.05 to +0.10 |
| **Model** | ResNet18 + 15 epochs (unchanged) | Baseline |
| **Ensemble** | Multi-window 3x (unchanged) | Baseline |

---

## 📊 Expected Score

```
Baseline: 0.648
Phase 1 Expected: 0.70-0.73
Improvement: +0.05-0.08
Confidence: HIGH (conservative improvements)
```

---

## 🔍 Files to Use

```
✅ birdclef2026-train-weights-v2.ipynb      (TRAINING - USE THIS)
✅ birdclef2026-inference.ipynb             (INFERENCE - USE THIS)
❌ birdclef2026-train-weights (1).ipynb     (CORRUPTED - DO NOT USE)
```

---

## 🆘 Troubleshooting

### Error: "species.json not found"
**Fix:** Run training notebook first (Cell 4 saves it)

### Error: "optimal_thresholds.json not found"  
**Fix:** Inference has fallback to 0.5 thresholds (runs but suboptimal)

### Error: "model_fold0.pt not found"
**Fix:** Ensure dataset is uploaded after training

### Memory issues during mel precomputation
**Fix:** Reduce batch_size or use lower num_workers

---

## 📈 Success Criteria

After running Phase 1:

1. **Training Output:**
   - [ ] 5 model files (`model_fold{0-4}.pt`)
   - [ ] Thresholds file (`optimal_thresholds.json`)
   - [ ] Species list (`species.json`)
   - [ ] Mean 5-Fold AUC: ~0.65-0.72

2. **Inference Output:**
   - [ ] `submission.csv` created
   - [ ] Shape: (N_test_windows, 235 species)
   - [ ] Values: all in [0, 1] range
   - [ ] No NaN values

3. **Kaggle Score:**
   - [ ] Expected: 0.70-0.73
   - [ ] Actual: Check competition leaderboard

---

## 🎓 What Was Changed

### Training Notebook (v2)
1. **Light Augmentation** in `ClipDataset.apply_light_augmentation()`
   - Time masking: 10-15% of frames
   - Frequency masking: 5-10 mel bins
   - Applied 50% of time during training

2. **Threshold Collection** in training loop
   - Stores best validation predictions/targets per fold
   - Combines across all 5 folds after training

3. **Threshold Optimization** in final cell
   - Tests thresholds 0.1 to 0.9
   - Maximizes F1 score for each species
   - Saves to `optimal_thresholds.json`

### Inference Notebook  
1. **Threshold Loading**
   - Loads from training outputs
   - Fallback to 0.5 if missing
   
2. **No other changes**
   - Multi-window ensemble unchanged
   - Per-fold averaging unchanged
   - Missing species proxy unchanged

---

## Next Phase (Phase 2)

**Only if Phase 1 scores < 0.70:**
- Learning rate scheduler (cosine annealing)
- Longer training (20+ epochs)
- Ensemble with different seeds

**If Phase 1 scores >= 0.70:**
- Consider other improvements
- Experiment with features
- Validate threshold effectiveness

---

## Links

- Training doc: [PHASE1-IMPROVEMENTS.md](PHASE1-IMPROVEMENTS.md)
- Full report: [SANITY-CHECK-REPORT.md](SANITY-CHECK-REPORT.md)
- Training notebook: [birdclef2026-train-weights-v2.ipynb](birdclef2026-train-weights-v2.ipynb)
- Inference notebook: [birdclef2026-inference.ipynb](birdclef2026-inference.ipynb)
