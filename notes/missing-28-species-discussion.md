# Missing 28 Species — Problem Discussion

**Date:** 2026-04-06  
**Context:** BirdCLEF 2026 competition — `birdclef2026-inference.ipynb`

---

## Problem

The training dataset does not have data for 28 species that appear in the submission columns. The current inference notebook fills those columns with `0.5`, which is a confidently-wrong neutral prediction that actively hurts the ranking score.

---

## Options (Ranked by Effort)

### ✅ Option 1 — Change fill value: `0.5` → `0.0` (Quick Fix, ~5 min)

In the inference notebook, find the line:
```python
submission[col] = 0.5
```
Change it to:
```python
submission[col] = 0.0
```

**Why**: Most 5-second windows will not contain any given rare species. Predicting `0.0` (absent) is statistically safer and avoids penalizing the score with a high-confidence wrong answer.

---

### 🐦 Option 2 — BirdNET Pretrained Embeddings (Medium effort)

BirdNET (Cornell Lab) is a pretrained bird sound classifier covering 6,000+ species.

Steps:
1. Install: `pip install birdnetlib`
2. Extract audio embeddings from test `.ogg` files using BirdNET
3. Use BirdNET's species-level confidence scores for the 28 missing species
4. Blend with your ResNet18 predictions for the 206 trained species

**Why**: Free, domain-specific, high-quality features. Directly addresses the gap without new training data.

---

### 🌐 Option 3 — Xeno-canto External Data (High effort)

Download audio recordings for the 28 missing species from [xeno-canto.org](https://xeno-canto.org) (public domain).  
Train a small classifier head on those recordings separately and integrate into the pipeline.

**Why**: BirdCLEF competition rules typically allow xeno-canto data. Provides true training signal for the missing species.

---

### 🧬 Option 4 — Taxonomy Proxy (Medium effort)

For each missing species, find its closest trained relative (same genus or family).  
Use the trained species' prediction as a proxy for the missing one.

**Why**: Exploits phylogenetic similarity in vocalizations. No external data needed.

---

## Recommended Action Plan

| Priority | Action | Impact |
|---|---|---|
| 1 | Change `0.5` → `0.0` for missing species | Immediate score improvement |
| 2 | Integrate BirdNET scores for missing 28 | Meaningful LB gain |
| 3 | Xeno-canto data collection (if allowed) | Best long-term solution |

---

## Files to Modify

- `birdclef2026-inference.ipynb` — fill value for missing species columns
- `birdclef2026-train-weights (1).ipynb` — if retraining with new data (Option 3)
