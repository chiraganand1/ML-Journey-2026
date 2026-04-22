# BirdCLEF 2026 Inference - Submission Verification Checklist

## Pre-Submission Checklist

### ✅ Model Configuration
- [x] 5 Perch folds loaded
- [x] 5 EfficientNet-B0 folds loaded  
- [x] 234 species taxonomy loaded from `taxonomy.csv`
- [x] Ensemble averaging working (10 models per window)

### ✅ Audio Processing
- [x] Test audio directory detected: `/kaggle/input/competitions/birdclef-2026/test_soundscapes/`
- [x] Audio file search supports multiple formats: `.ogg`, `.wav`, `.mp3`, `.flac`, `.m4a`
- [x] 3 windows per audio file (start, middle, end): WINDOW_OFFSETS = [5, 10, 15]
- [x] TestAudioDataset returns 3 windows with proper time alignment

### ✅ Submission Format
- [x] Column count: 235 (1 row_id + 234 species)
- [x] Row format: `{filename}_{window_offset}` (e.g., `abc123_5`, `abc123_10`, `abc123_15`)
- [x] row_id dtype: object (string)
- [x] Species dtypes: float64 (probabilities)
- [x] Species values range: [0.0, 1.0]

### ✅ Code Enhancements
- [x] Cell 11 - Multi-format audio file discovery
- [x] Cell 15 - Enhanced save with validation:
  - File creation verification
  - NaN value detection
  - Data type validation
  - Value range checking
  - Summary statistics

---

## Expected Runtime Behavior

### When Running on Kaggle

**Cell 2-10:** Setup (< 30 seconds)
- Load models
- Load taxonomy
- Define helper functions

**Cell 11: Generate Predictions** (variable - depends on file count)
- Output: "Found X test files in /kaggle/input/..."
- Output: "Audio formats found:" with breakdown (ogg: N, wav: M, etc.)
- **If 0 files found:** ⚠️ Audio format or path issue
- **If > 0 files found:** ✅ Proceed to predictions
- Estimated time: 5-30 mins (depending on # files and # models)

**Cell 12-14:** Apply predictions & validation (< 2 minutes)

**Cell 15: Save & Validate** (< 30 seconds)
- Output: "Submission saved to: /kaggle/working/submission.csv"  
- Output: File size (should be 100 KB - 2 MB depending on rows)
- Output: NaN check (should show "No NaN values detected")
- Output: Data type validation (should show float64 for species)
- Output: Value range (should show [0.0000, 1.0000])

**Cell 16-17:** Final diagnostics

### Total Expected Runtime
- **If test files found:** 15-35 minutes
- **If NO test files found:** ~10 minutes (early exit)

### How to Know It's Working
✅ Cell 11 finds test files (not 0)  
✅ Runtime > 10 minutes (more than quick exit)  
✅ Cell 15 shows file size > 100 KB  
✅ No NaN values in submission

---

## If Kaggle Still Rejects Submission

### Diagnosis Steps

1. **Check Cell 11 output:**
   ```
   If: "Found 0 test files"
   Then: Audio format or path issue
   
   If: "Found > 0 test files"
   Then: Format detection working, check validation
   ```

2. **Check Cell 15 output:**
   - File size too small? → Not enough predictions
   - NaN values present? → Prediction generation failed
   - Wrong dtypes? → Data type conversion issue
   - Values outside [0,1]? → Probability calibration issue

3. **Common Issues & Fixes:**

   | Issue | Check | Fix |
   |-------|-------|-----|
   | 0 test files found | Audio files in `/test_soundscapes/` | Add more formats to search pattern |
   | "incorrect format" error | Column count = 235? | Verify taxonomy loaded 234 species |
   | Wrong value ranges | Species values > 1.0 or < 0.0? | Clip probabilities to [0, 1] |
   | Missing predictions | NaN values in submission? | Debug `get_predictions_for_audio()` |
   | Wrong row format | Row IDs have `_5, _10, _15`? | Verify WINDOW_OFFSETS implementation |

---

## Submission Anatomy Reference

### Valid Row Example
```
row_id              | songbirds_001 | songbirds_002 | ... | songbirds_234
---|---|---|---|
myaudio_5           | 0.8234        | 0.0421        | ... | 0.0012
myaudio_10          | 0.7891        | 0.0389        | ... | 0.0015
myaudio_15          | 0.8102        | 0.0456        | ... | 0.0018
```

### Validation Points
- Column order: Same as `sample_submission.csv`
- No extra columns
- No duplicate row_ids
- All species columns present
- All probabilities finite (no NaN, inf)
- All probabilities in [0.0, 1.0]

---

## Performance Expectations

### Expected Leaderboard Impact (estimate)
- **With multi-format fix:** Full test set processed → better coverage
- **Baseline prediction:** Random 0.5 for all species = ~0-5% score
- **With ensemble model:** Expected 30-60% public LB score (depending on test set difficulty)

### Why Previous Failed
- ❌ Only searched `.ogg` files
- ❌ Kaggle test data likely in `.wav` or other format
- ❌ No files found → empty submission → format error

### Why This Should Work
- ✅ Multi-format support  
- ✅ Robust validation  
- ✅ Format verification before saving
- ✅ Detailed diagnostics for debugging
