# NeuroSynth

## Overview

NeuroSynth is a platform for large-scale, automated synthesis of functional magnetic resonance imaging (fMRI) data. It provides tools for automatically extracting coordinates and terms from thousands of published neuroimaging studies and performing meta-analyses across the entire neuroimaging literature. NeuroSynth enables data-driven discovery of brain-behavior relationships through text mining and coordinate-based meta-analysis.

**Website:** https://neurosynth.org/
**Platform:** Python/Web
**Language:** Python
**License:** MIT License

## Key Features

- Automated literature mining from PubMed
- Large-scale coordinate database (15,000+ studies)
- Automated term extraction and frequency analysis
- Coordinate-based meta-analysis (CBMA)
- Interactive web interface for exploration
- Python API for programmatic access
- Reverse inference and decoding
- Meta-analytic connectivity modeling (MACM)
- ROI-based analysis
- Downloadable meta-analytic maps
- Integration with other neuroimaging tools

## Installation

### Python Package

```bash
# Install via pip
pip install neurosynth

# Or install from source
git clone https://github.com/neurosynth/neurosynth.git
cd neurosynth
pip install -e .

# Verify installation
python -c "import neurosynth; print(neurosynth.__version__)"
```

### Dependencies

```bash
# Core dependencies (auto-installed)
pip install numpy scipy pandas nibabel scikit-learn

# Optional for advanced features
pip install nilearn matplotlib seaborn
```

### Download Database

```bash
# Download the latest NeuroSynth database
python -c "from neurosynth import Dataset; Dataset.download()"

# Or manually from website
# https://github.com/neurosynth/neurosynth-data/releases

# Database files:
# - database.txt (coordinates and metadata)
# - features.txt (term frequencies per study)
```

## Data Structure

### NeuroSynth Database

```python
import neurosynth as ns

# Load the database
dataset = ns.Dataset('data/database.txt')

# Load feature data (term frequencies)
dataset.add_features('data/features.txt')

# Database contains:
# - Study IDs and metadata
# - Peak coordinates (MNI space)
# - Term frequencies per study
# - Image metadata

# Inspect
print(f"Number of studies: {len(dataset.mappables)}")
print(f"Number of features: {len(dataset.feature_names)}")
print(f"Total coordinates: {dataset.activations.shape[0]}")
```

### Study Metadata

```python
# Access study information
studies = dataset.get_studies()

# Each study has:
# - PubMed ID
# - Authors
# - Title
# - Journal
# - Peak coordinates
# - Associated terms/features

# Example: Get studies for a specific term
emotion_studies = dataset.get_studies(features='emotion', threshold=0.001)
print(f"Studies about emotion: {len(emotion_studies)}")
```

## Web Interface Usage

### Online Meta-Analysis

```bash
# Visit https://neurosynth.org/

# Features:
# 1. Search for terms (e.g., "working memory", "emotion")
# 2. View meta-analytic activation maps
# 3. Download NIfTI images
# 4. Explore term co-occurrence
# 5. Generate custom meta-analyses
# 6. Decode uploaded images
```

### Searching Terms

```bash
# On NeuroSynth website:
# 1. Enter search term in box
# 2. View results:
#    - Forward inference map (where term is used)
#    - Reverse inference map (term specificity)
#    - Number of studies
#    - Associated terms
# 3. Download maps as NIfTI files
```

### Decoding Activation Maps

```bash
# Upload your own activation map
# 1. Go to "Locations" or "Custom" tab
# 2. Upload NIfTI file
# 3. Get ranked list of associated terms
# 4. Interpret your results

# Tells you: "Your activation pattern is associated with..."
```

## Python API - Basic Usage

### Load Dataset

```python
from neurosynth import Dataset
import os

# Set data directory
data_dir = '/path/to/neurosynth-data'

# Load database
dataset = Dataset(os.path.join(data_dir, 'database.txt'))

# Add features
dataset.add_features(os.path.join(data_dir, 'features.txt'))

# Save for faster loading next time
dataset.save(os.path.join(data_dir, 'dataset.pkl'))

# Load from pickle (faster)
# dataset = Dataset.load(os.path.join(data_dir, 'dataset.pkl'))
```

### Generate Meta-Analysis

```python
from neurosynth import meta

# Perform meta-analysis for a term
# Example: "working memory"
ids = dataset.get_studies(features='working memory', threshold=0.001)

# Create meta-analysis
ma = meta.MetaAnalysis(dataset, ids)

# Save results
ma.save_results('/output/working_memory')

# Generates:
# - association-test_z.nii.gz (forward inference)
# - association-test_z_FDR_0.01.nii.gz (thresholded)
# - uniformity-test_z.nii.gz (reverse inference)
```

### Forward vs. Reverse Inference

```python
# Forward inference: P(activation | term)
# "Given working memory, where is the brain active?"
ma_forward = meta.MetaAnalysis(dataset, ids, q=0.01)
ma_forward.save_results('/output/wm_forward', image_list=['pFgA_z'])

# Reverse inference: P(term | activation)
# "Given activation here, how likely is it working memory?"
ma_reverse = meta.MetaAnalysis(dataset, ids, q=0.01)
ma_reverse.save_results('/output/wm_reverse', image_list=['pAgF_z'])

# pFgA: P(activation | feature)
# pAgF: P(feature | activation) - more specific
```

## Advanced Meta-Analysis

### Multiple Term Analysis

```python
# Analyze conjunction of multiple terms
terms = ['emotion', 'regulation', 'cognitive']

# Get studies mentioning all terms
ids_all = dataset.get_studies(
    features=terms,
    threshold=0.001,
    func='sum'  # or 'min', 'max'
)

# Perform meta-analysis
ma = meta.MetaAnalysis(dataset, ids_all)
ma.save_results('/output/emotion_regulation_cognitive')
```

### Contrast Analysis

```python
# Compare two sets of studies
# Example: Emotion vs. Cognition

# Get studies for each term
emotion_ids = dataset.get_studies(features='emotion', threshold=0.001)
cognition_ids = dataset.get_studies(features='cognition', threshold=0.001)

# Perform contrast
ma_emotion = meta.MetaAnalysis(dataset, emotion_ids)
ma_cognition = meta.MetaAnalysis(dataset, cognition_ids)

# Compute contrast
from neurosynth.analysis import compare

contrast = compare.compare_images(
    ma_emotion.images['pFgA_z'],
    ma_cognition.images['pFgA_z']
)

# Save contrast map
import nibabel as nib
nib.save(nib.Nifti1Image(contrast, dataset.masker.mask_img.affine),
         '/output/emotion_vs_cognition.nii.gz')
```

### ROI-Based Analysis

```python
import nibabel as nib
from neurosynth.analysis import roi

# Load ROI mask
roi_img = nib.load('/path/to/roi_mask.nii.gz')

# Find studies with peaks in ROI
roi_ids = dataset.get_ids_by_mask(roi_img, threshold=0.0)

print(f"Studies with activations in ROI: {len(roi_ids)}")

# Perform meta-analysis of ROI studies
ma_roi = meta.MetaAnalysis(dataset, roi_ids)
ma_roi.save_results('/output/roi_meta_analysis')

# Decode ROI: What terms are associated?
decoder = decode.Decoder(dataset)
decoded = decoder.decode(roi_img, save='/output/roi_decoded.txt')

# Print top terms
for term, score in decoded.items()[:10]:
    print(f"{term}: {score:.3f}")
```

## Decoding Analysis

### Decode Activation Map

```python
from neurosynth.analysis import decode

# Load your activation map
my_activations = nib.load('/path/to/my_activation_map.nii.gz')

# Create decoder
decoder = decode.Decoder(dataset)

# Decode image
decoded = decoder.decode(my_activations, save='/output/decoded_results.txt')

# Get top associated terms
top_terms = sorted(decoded.items(), key=lambda x: x[1], reverse=True)[:20]

print("Top associated terms:")
for term, correlation in top_terms:
    print(f"  {term}: r = {correlation:.3f}")
```

### Decoding with Custom Method

```python
# Different decoding methods
decoder = decode.Decoder(dataset)

# Pearson correlation (default)
decoded_pearson = decoder.decode(my_activations, method='pearson')

# Pattern correlation
decoded_pattern = decoder.decode(my_activations, method='pattern')

# Naive Bayes
decoded_nb = decoder.decode(my_activations, method='nb')

# Compare methods
for method, results in [('Pearson', decoded_pearson),
                        ('Pattern', decoded_pattern),
                        ('Naive Bayes', decoded_nb)]:
    top = sorted(results.items(), key=lambda x: x[1], reverse=True)[0]
    print(f"{method}: {top[0]} ({top[1]:.3f})")
```

## Meta-Analytic Connectivity Modeling (MACM)

```python
# Connectivity-based meta-analysis
from neurosynth.analysis import network

# Define seed region
seed_mask = nib.load('/path/to/seed_mask.nii.gz')

# Get studies with peaks in seed
seed_ids = dataset.get_ids_by_mask(seed_mask)

print(f"Studies activating seed: {len(seed_ids)}")

# Find co-activations
macm = meta.MetaAnalysis(dataset, seed_ids)
macm.save_results('/output/macm_results')

# Result shows regions commonly co-activated with seed
```

## Batch Processing

### Generate Multiple Meta-Analyses

```python
# Analyze multiple terms in batch
terms_of_interest = [
    'working memory',
    'attention',
    'language',
    'emotion',
    'motor',
    'visual'
]

results = {}

for term in terms_of_interest:
    print(f"Processing: {term}")

    # Get studies
    ids = dataset.get_studies(features=term, threshold=0.001)

    if len(ids) < 10:  # Skip if too few studies
        print(f"  Skipping (only {len(ids)} studies)")
        continue

    print(f"  {len(ids)} studies found")

    # Meta-analysis
    ma = meta.MetaAnalysis(dataset, ids)

    # Save
    output_dir = f'/output/{term.replace(" ", "_")}'
    ma.save_results(output_dir)

    results[term] = {
        'n_studies': len(ids),
        'output': output_dir
    }

# Save summary
import json
with open('/output/batch_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Create Term Atlas

```python
# Create probabilistic atlas from multiple terms
import numpy as np

terms = ['motor', 'visual', 'auditory', 'language', 'memory']
atlas_data = []

for i, term in enumerate(terms, 1):
    ids = dataset.get_studies(features=term, threshold=0.001)
    ma = meta.MetaAnalysis(dataset, ids)

    # Get z-map
    z_img = ma.images['pFgA_z']
    z_data = z_img.get_fdata()

    # Threshold and label
    z_thresh = z_data > 3.0  # z > 3
    atlas_data.append(z_thresh * i)

# Combine (winner-take-all)
combined_atlas = np.zeros_like(atlas_data[0])
for atlas in atlas_data:
    mask = atlas > 0
    combined_atlas[mask] = atlas[mask]

# Save atlas
atlas_img = nib.Nifti1Image(combined_atlas, dataset.masker.mask_img.affine)
nib.save(atlas_img, '/output/functional_atlas.nii.gz')

# Create labels file
with open('/output/atlas_labels.txt', 'w') as f:
    for i, term in enumerate(terms, 1):
        f.write(f"{i}: {term}\n")
```

## Visualization

### Plot Meta-Analysis Results

```python
from nilearn import plotting
import matplotlib.pyplot as plt

# Load meta-analysis result
result_img = nib.load('/output/working_memory/association-test_z.nii.gz')

# Plot on glass brain
fig = plotting.plot_glass_brain(
    result_img,
    threshold=3.0,
    colorbar=True,
    title='Working Memory Meta-Analysis'
)
fig.savefig('/output/working_memory_glass.png', dpi=300)

# Plot on anatomical
anat = plotting.load_mni152_template()
fig = plotting.plot_stat_map(
    result_img,
    bg_img=anat,
    threshold=3.0,
    display_mode='z',
    cut_coords=6,
    title='Working Memory Meta-Analysis'
)
fig.savefig('/output/working_memory_slices.png', dpi=300)
```

### Create Term Co-occurrence Network

```python
import pandas as pd
import seaborn as sns

# Get feature matrix
features = dataset.get_feature_data()

# Calculate term correlations
term_subset = ['emotion', 'cognition', 'memory', 'attention',
               'language', 'motor', 'visual', 'pain']
feature_subset = features[term_subset]

# Compute correlation matrix
correlation_matrix = feature_subset.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True)
plt.title('Term Co-occurrence in NeuroSynth Database')
plt.tight_layout()
plt.savefig('/output/term_cooccurrence.png', dpi=300)
```

## Custom Database Creation

### Build Database from Coordinates

```python
from neurosynth import Dataset

# Create custom database from your own studies
# Format: Tab-separated file with columns:
# id, x, y, z, space

# Example data
data = """id\tx\ty\tz\tspace
study1\t-45\t12\t24\tMNI
study1\t45\t15\t21\tMNI
study2\t0\t-6\t52\tMNI
study2\t-12\t-85\t2\tMNI
"""

with open('/data/custom_database.txt', 'w') as f:
    f.write(data)

# Load custom database
custom_dataset = Dataset('/data/custom_database.txt')

# Add custom features
# Format: study_id, feature1, feature2, ...
features_data = """id\temotion\tcognition
study1\t0.8\t0.2
study2\t0.1\t0.9
"""

with open('/data/custom_features.txt', 'w') as f:
    f.write(features_data)

custom_dataset.add_features('/data/custom_features.txt')

# Now can perform meta-analyses on custom data
```

## Integration with Other Tools

### Export to NiMARE

```python
# NeuroSynth data can be used in NiMARE
from neurosynth import Dataset
import pandas as pd

# Load NeuroSynth dataset
dataset = Dataset('data/database.txt')

# Convert to NiMARE format
from nimare import io

# Get coordinates for specific studies
emotion_ids = dataset.get_studies(features='emotion', threshold=0.001)
coords_df = dataset.get_coordinates(ids=emotion_ids)

# Create NiMARE dataset
nimare_dset = io.convert_neurosynth_to_dict(
    coords_df,
    annotations_files='data/features.txt'
)

# Now use with NiMARE
```

### Combine with Nilearn

```python
from nilearn import datasets, plotting
from neurosynth import Dataset, meta

# Load NeuroSynth dataset
dataset = Dataset.load('data/dataset.pkl')

# Generate meta-analysis
ids = dataset.get_studies(features='motor', threshold=0.001)
ma = meta.MetaAnalysis(dataset, ids)

# Use Nilearn for visualization
from nilearn.plotting import view_img

# Interactive viewer
view = view_img(
    ma.images['pFgA_z'],
    threshold=3.0,
    title='Motor Meta-Analysis'
)
view.save_as_html('/output/motor_interactive.html')
```

## Quality Control

### Check Database Statistics

```python
# Database overview
print(f"Studies: {len(dataset.mappables)}")
print(f"Coordinates: {dataset.activations.shape[0]}")
print(f"Features: {len(dataset.feature_names)}")

# Studies per feature
features = dataset.get_feature_data()
studies_per_feature = (features > 0).sum()
print(f"\nAverage studies per feature: {studies_per_feature.mean():.1f}")
print(f"Median: {studies_per_feature.median():.1f}")

# Most common features
top_features = studies_per_feature.sort_values(ascending=False).head(10)
print("\nTop 10 features:")
print(top_features)
```

### Validate Meta-Analysis

```python
# Check meta-analysis quality
def validate_meta_analysis(ma, min_studies=10):
    """Validate meta-analysis quality."""

    n_studies = len(ma.ids)

    if n_studies < min_studies:
        print(f"WARNING: Only {n_studies} studies (minimum: {min_studies})")

    # Check result
    z_img = ma.images['pFgA_z']
    z_data = z_img.get_fdata()

    # Statistics
    print(f"Studies included: {n_studies}")
    print(f"Max Z-score: {z_data.max():.2f}")
    print(f"Voxels > Z=3: {(z_data > 3).sum()}")
    print(f"Voxels > Z=5: {(z_data > 5).sum()}")

    # Check for empty result
    if z_data.max() < 2:
        print("WARNING: No strong activations found")

# Example
ids = dataset.get_studies(features='working memory', threshold=0.001)
ma = meta.MetaAnalysis(dataset, ids)
validate_meta_analysis(ma)
```

## Integration with Claude Code

When helping users with NeuroSynth:

1. **Check Installation:**
   ```python
   import neurosynth
   print(neurosynth.__version__)
   ```

2. **Download Data:**
   ```python
   # Ensure database is downloaded
   from neurosynth import Dataset
   Dataset.download()
   ```

3. **Common Workflows:**
   - Term-based meta-analysis
   - Decoding activation patterns
   - ROI-based analysis
   - MACM connectivity analysis

4. **Best Practices:**
   - Use threshold=0.001 for term selection
   - Check number of studies before meta-analysis
   - Use reverse inference for specificity
   - Visualize results for quality check

## Troubleshooting

**Problem:** Dataset won't load
**Solution:** Download latest database, check file paths, verify format

**Problem:** Very few studies for term
**Solution:** Try alternative terms, lower threshold (carefully), check spelling

**Problem:** Empty meta-analysis results
**Solution:** Increase study threshold, check term frequency, verify coordinates

**Problem:** Memory errors
**Solution:** Process in batches, use smaller feature set, increase system RAM

**Problem:** Outdated database
**Solution:** Download latest release from GitHub, update regularly

## Best Practices

1. **Always check number of studies** before meta-analysis (≥20 recommended)
2. **Use reverse inference** for more specific associations
3. **Validate results** by comparing with literature
4. **Report database version** and date in publications
5. **Consider publication bias** inherent in coordinate databases
6. **Use appropriate thresholds** (FDR or cluster correction)
7. **Interpret automated results cautiously** - not ground truth

## Resources

- **Website:** https://neurosynth.org/
- **GitHub:** https://github.com/neurosynth/neurosynth
- **Documentation:** https://neurosynth.readthedocs.io/
- **Database:** https://github.com/neurosynth/neurosynth-data
- **Forum:** https://neurostars.org/tag/neurosynth
- **Tutorials:** https://neurosynth.org/tutorials/

## Citation

```bibtex
@article{yarkoni2011neurosynth,
  title={Large-scale automated synthesis of human functional neuroimaging data},
  author={Yarkoni, Tal and Poldrack, Russell A and Nichols, Thomas E and Van Essen, David C and Wager, Tor D},
  journal={Nature Methods},
  volume={8},
  number={8},
  pages={665--670},
  year={2011},
  publisher={Nature Publishing Group}
}
```

## Related Tools

- **NiMARE:** Comprehensive meta-analysis framework
- **NeuroQuery:** Similar to NeuroSynth with predictive modeling
- **GingerALE:** Activation likelihood estimation
- **AES-SDM:** Seed-based d mapping meta-analysis
- **Neurosynth-compose:** Create custom meta-analyses
- **BrainMap:** Curated coordinate database
