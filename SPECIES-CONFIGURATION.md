# Species Configuration

## Overview
The BirdCLEF 2026 models use **234 species** extracted from `taxonomy.csv`.

## Species List
- **File**: `species.json`
- **Source**: `taxonomy.csv` (primary_label column)
- **Total species**: 234
- **Format**: JSON array of species codes (strings)

## Species Composition
- **Numeric IDs**: 47 species (e.g., 1161364, 116570)
- **Alphanumeric codes**: 187 species (e.g., ashgre1, brnowl)
- **Sonotype variants**: 25 insect sonotypes (47158son01 through 47158son25)

### Species Categories
| Category | Count | Examples |
|----------|-------|----------|
| Birds (Aves) | 174 | ashgre1 (Ashy-headed Greenlet), brnowl (American Barn Owl) |
| Amphibians | 45 | 1161364, 22930, 65377 |
| Mammals | 10 | 209233 (Horse), 41970 (Jaguar) |
| Insects | 5 | 244024 (Giant Cicada), 760266 |
| Insect Sonotypes | 25 | 47158son01-47158son25 |

## Loading Species

### In Training Notebooks
```python
# Try to load from Kaggle dataset first, then local fallback
species_paths = [
    "/kaggle/input/datasets/chiragggg/birdclef-2026-input-model-species/species.json",
    "species.json"
]

species = None
for path in species_paths:
    try:
        with open(path, "r") as f:
            species = json.load(f)
        break
    except FileNotFoundError:
        continue

if species is None:
    raise FileNotFoundError("species.json not found")
```

### In Inference Notebooks
Same pattern - loads from Kaggle dataset or local `species.json`

## Important Notes

1. **Kaggle Submission Format**: Must have exactly 234 species columns + row_id column (235 total)
2. **Column Order**: Must match the order in `species.json`
3. **Data Type**: All values must be 0 or 1 (binary predictions)
4. **No NaN values**: All cells must have valid data

## Related Files
- `taxonomy.csv`: Reference taxonomy with all species metadata
- `species.json`: Generated from taxonomy.csv (should be committed to git)
- `birdclef2026-train-v6-perch-efficientnet.ipynb`: Uses species.json
- `birdclef2026-inference-v6-perch-efficientnet.ipynb`: Uses species.json

## How to Regenerate species.json

If you need to regenerate `species.json` from `taxonomy.csv`:

```python
import csv
import json

species = []
with open('taxonomy.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        species.append(row['primary_label'])

with open('species.json', 'w') as f:
    json.dump(species, f)
```
