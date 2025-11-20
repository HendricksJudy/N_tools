# NiMARE

## Overview

NiMARE (Neuroimaging Meta-Analysis Research Environment) is a comprehensive Python package for performing coordinate-based and image-based meta-analyses of neuroimaging data. It provides a unified framework implementing multiple meta-analytic algorithms (ALE, MKDA, KDA, etc.), supports both coordinate and image inputs, includes modern statistical methods, and integrates seamlessly with the broader neuroimaging ecosystem.

**Website:** https://nimare.readthedocs.io/
**Platform:** Python
**Language:** Python
**License:** MIT License

## Key Features

- Multiple coordinate-based meta-analysis (CBMA) algorithms
- Image-based meta-analysis (IBMA) support
- Modern statistical corrections (FWE, FDR, cluster)
- Meta-analytic coactivation modeling (MACM)
- Functional decoding and annotation
- Study contrast analysis
- Jackknife and bootstrap analyses
- Integration with NeuroQuery and NeuroSynth
- Extensible architecture for new methods
- Comprehensive documentation and tutorials

## Installation

### Install via pip

```bash
# Install NiMARE
pip install nimare

# Install with all optional dependencies
pip install nimare[all]

# Verify installation
python -c "import nimare; print(nimare.__version__)"
```

### Install from source

```bash
# Clone repository
git clone https://github.com/neurostuff/NiMARE.git
cd NiMARE

# Install in development mode
pip install -e .[all]

# Run tests
pytest nimare/tests/
```

### Dependencies

```bash
# Core dependencies (auto-installed)
pip install numpy scipy pandas nibabel nilearn scikit-learn

# Optional for advanced features
pip install matplotlib seaborn statsmodels
```

## Data Structures

### Dataset Object

```python
from nimare import dataset

# NiMARE uses Dataset object to store:
# - Study metadata
# - Peak coordinates
# - Statistical images
# - Annotations/labels

# Load from file
dset = dataset.Dataset.load('my_dataset.pkl')

# Access data
print(f"Studies: {len(dset.ids)}")
print(f"Coordinates: {dset.coordinates.shape}")
print(f"Images: {dset.images.shape if dset.images is not None else 0}")
```

### Create Dataset from Coordinates

```python
import pandas as pd
from nimare import dataset

# Coordinate data format
coords_df = pd.DataFrame({
    'id': ['study1', 'study1', 'study2', 'study2', 'study3'],
    'x': [-42, 45, -12, 0, 38],
    'y': [15, 18, -85, -6, -52],
    'z': [24, 21, 2, 52, -18],
    'space': ['MNI'] * 5
})

# Create dataset
dset = dataset.Dataset(coords_df)

# Add metadata
metadata = pd.DataFrame({
    'id': ['study1', 'study2', 'study3'],
    'sample_size': [20, 25, 18],
    'contrast': ['emotion > baseline', 'WM > baseline', 'motor > baseline']
})
dset.metadata = metadata

# Save
dset.save('my_dataset.pkl')
```

## Coordinate-Based Meta-Analysis (CBMA)

### Activation Likelihood Estimation (ALE)

```python
from nimare.meta.cbma import ALE
from nimare import dataset

# Load dataset
dset = dataset.Dataset.load('neurosynth_dataset.pkl')

# Select subset (e.g., emotion studies)
emotion_ids = dset.get_studies_by_label('emotion')
dset_emotion = dset.slice(emotion_ids)

# Create ALE meta-analysis
ale = ALE(kernel__fwhm=15)  # 15mm FWHM kernel

# Fit the model
results = ale.fit(dset_emotion)

# Get statistical maps
z_map = results.get_map('z')
p_map = results.get_map('p')

# Save results
z_map.to_filename('emotion_ale_z.nii.gz')

# Apply multiple comparisons correction
from nimare.correct import FWECorrector

corrector = FWECorrector(method='montecarlo',
                          n_iters=10000,
                          n_cores=-1)
corrected_results = corrector.transform(results)

# Save corrected map
corrected_map = corrected_results.get_map('z_corr-FWE_method-montecarlo')
corrected_map.to_filename('emotion_ale_z_fwe.nii.gz')
```

### Multilevel Kernel Density Analysis (MKDA)

```python
from nimare.meta.cbma import MKDADensity

# MKDA meta-analysis
mkda = MKDADensity(kernel__r=10)  # 10mm radius kernel

# Fit
results = mkda.fit(dset_emotion)

# Get results
z_map = results.get_map('z')
z_map.to_filename('emotion_mkda_z.nii.gz')

# MKDA is less conservative than ALE
# Good for smaller datasets
```

### Kernel Density Analysis (KDA)

```python
from nimare.meta.cbma import KDA

# Standard KDA
kda = KDA(kernel__r=10)  # 10mm radius

# Fit
results = kda.fit(dset_emotion)

# Save
results.get_map('of').to_filename('emotion_kda_of.nii.gz')

# KDA provides occurrence frequency map
```

## Multiple Comparison Correction

### Family-Wise Error (FWE) Correction

```python
from nimare.correct import FWECorrector

# Monte Carlo FWE correction
fwe_montecarlo = FWECorrector(
    method='montecarlo',
    n_iters=10000,
    n_cores=-1
)
results_fwe = fwe_montecarlo.transform(results)

# Bonferroni FWE correction
fwe_bonferroni = FWECorrector(method='bonferroni')
results_fwe_bonf = fwe_bonferroni.transform(results)

# Save corrected map
fwe_map = results_fwe.get_map('z_corr-FWE_method-montecarlo')
fwe_map.to_filename('emotion_ale_fwe.nii.gz')
```

### False Discovery Rate (FDR) Correction

```python
from nimare.correct import FDRCorrector

# FDR correction
fdr = FDRCorrector(method='indep', alpha=0.05)
results_fdr = fdr.transform(results)

# Get FDR-corrected map
fdr_map = results_fdr.get_map('z_corr-FDR_method-indep')
fdr_map.to_filename('emotion_ale_fdr.nii.gz')
```

### Cluster-Level Correction

```python
from nimare.correct import FWECorrector

# Cluster-level FWE
cluster_fwe = FWECorrector(
    method='montecarlo',
    n_iters=10000,
    voxel_thresh=0.001,  # Cluster-forming threshold
    n_cores=-1
)
results_cluster = cluster_fwe.transform(results)

# Extract significant clusters
clusters_map = results_cluster.get_map('z_corr-FWE_method-montecarlo')
clusters_map.to_filename('emotion_ale_cluster_fwe.nii.gz')
```

## Contrast Analysis

### Two-Group Comparison

```python
from nimare.meta.cbma import ALESubtraction

# Load datasets for two groups
emotion_ids = dset.get_studies_by_label('emotion')
cognition_ids = dset.get_studies_by_label('cognition')

dset_emotion = dset.slice(emotion_ids)
dset_cognition = dset.slice(cognition_ids)

# ALE subtraction analysis
ale_sub = ALESubtraction(n_iters=10000, kernel__fwhm=15)

# Fit with both datasets
results_sub = ale_sub.fit(dset_emotion, dset_cognition)

# Get maps
# Emotion > Cognition
emo_gt_cog = results_sub.get_map('z_desc-group1MinusGroup2')
emo_gt_cog.to_filename('emotion_gt_cognition.nii.gz')

# Cognition > Emotion
cog_gt_emo = results_sub.get_map('z_desc-group2MinusGroup1')
cog_gt_emo.to_filename('cognition_gt_emotion.nii.gz')
```

### Conjunction Analysis

```python
from nimare.meta.cbma import ALEConjunction

# Find common activations across multiple contrasts
ale_conj = ALEConjunction(n_iters=10000, kernel__fwhm=15)

# Fit with multiple datasets
results_conj = ale_conj.fit(dset_emotion, dset_cognition)

# Get conjunction map
conj_map = results_conj.get_map('z')
conj_map.to_filename('emotion_and_cognition_conjunction.nii.gz')
```

## Image-Based Meta-Analysis (IBMA)

### Stouffer's Method

```python
from nimare.meta.ibma import Stouffers

# When you have full statistical maps (not just coordinates)
# Load dataset with images
dset_images = dataset.Dataset.load('dataset_with_images.pkl')

# Stouffer's Z meta-analysis
stouffers = Stouffers()

# Fit
results_stouffers = stouffers.fit(dset_images)

# Get combined Z-map
z_combined = results_stouffers.get_map('z')
z_combined.to_filename('stouffers_meta_z.nii.gz')

# Apply FWE correction
corrector = FWECorrector(method='montecarlo', n_iters=10000)
results_corrected = corrector.transform(results_stouffers)
```

### Weighted Least Squares (WLS)

```python
from nimare.meta.ibma import WeightedLeastSquares

# WLS accounts for sample size differences
wls = WeightedLeastSquares()

# Fit (requires sample sizes in metadata)
results_wls = wls.fit(dset_images)

# Get results
wls_z = results_wls.get_map('z')
wls_z.to_filename('wls_meta_z.nii.gz')
```

### Fisher's Method

```python
from nimare.meta.ibma import Fishers

# Fisher's combined probability test
fishers = Fishers()

# Fit
results_fishers = fishers.fit(dset_images)

# Results
fishers_z = results_fishers.get_map('z')
fishers_z.to_filename('fishers_meta_z.nii.gz')
```

## Meta-Analytic Connectivity Modeling (MACM)

```python
from nimare.meta.cbma import ALE
import nibabel as nib

# Define seed region
seed_mask = nib.load('seed_roi.nii.gz')

# Get studies with peaks in seed
seed_ids = dset.get_studies_by_mask(seed_mask)

print(f"Studies with activations in seed: {len(seed_ids)}")

# Create MACM dataset
dset_macm = dset.slice(seed_ids)

# Perform ALE on co-activations
ale_macm = ALE(kernel__fwhm=15)
results_macm = ale_macm.fit(dset_macm)

# Apply correction
corrector = FWECorrector(method='montecarlo', n_iters=10000)
results_macm_fwe = corrector.transform(results_macm)

# Save connectivity map
conn_map = results_macm_fwe.get_map('z_corr-FWE_method-montecarlo')
conn_map.to_filename('macm_connectivity.nii.gz')
```

## Functional Decoding

### Decode Activation Pattern

```python
from nimare.decode import continuous

# Load your activation map
my_img = nib.load('my_activation_map.nii.gz')

# Create decoder using NeuroSynth data
decoder = continuous.CorrelationDecoder(
    feature_group='terms',
    features=None  # Use all features
)

# Fit decoder on database
decoder.fit(dset)

# Decode your image
decoded_df = decoder.transform(my_img)

# Sort by correlation
decoded_sorted = decoded_df.sort_values('r', ascending=False)

# Print top terms
print("Top associated terms:")
print(decoded_sorted.head(20))

# Save results
decoded_sorted.to_csv('decoded_results.csv')
```

### Discrete Decoding (Chi-Square)

```python
from nimare.decode import discrete

# For binary/thresholded images
decoder_discrete = discrete.NeurosynthDecoder(
    frequency_threshold=0.001
)

# Fit on database
decoder_discrete.fit(dset)

# Decode thresholded map
decoded_df = decoder_discrete.transform(my_img)

# Results include p-values and effect sizes
print(decoded_df.sort_values('p', ascending=True).head(20))
```

## Diagnostics and Quality Control

### Jackknife Analysis

```python
from nimare.diagnostics import Jackknife

# Jackknife to assess robustness
jackknife = Jackknife(
    target_image='z',
    voxel_thresh=None
)

# Run jackknife
ale = ALE(kernel__fwhm=15)
jack_results = jackknife.transform(ale, dset_emotion)

# Get results
# Shows which voxels are consistently significant
consistency_map = jack_results.get_map('consistency')
consistency_map.to_filename('ale_jackknife_consistency.nii.gz')

# Identify influential studies
influential_studies = jack_results.get_influential_studies()
print("Influential studies:")
print(influential_studies)
```

### Funnel Plot (Publication Bias)

```python
from nimare.diagnostics import plot_funnel

# Check for publication bias
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
plot_funnel(results, ax=ax)
plt.title('Funnel Plot - Publication Bias Check')
plt.savefig('funnel_plot.png', dpi=300)

# Asymmetry suggests publication bias
```

### Forest Plot

```python
from nimare.diagnostics import plot_forest

# Effect size forest plot for ROI
roi_mask = nib.load('roi.nii.gz')

fig, ax = plt.subplots(figsize=(10, 8))
plot_forest(dset_emotion, roi_mask, ax=ax)
plt.title('Forest Plot - Effect Sizes in ROI')
plt.tight_layout()
plt.savefig('forest_plot.png', dpi=300)
```

## Data Import and Export

### Import from NeuroSynth

```python
from nimare import extract

# Download NeuroSynth dataset
ns_dset = extract.download_neurosynth(
    data_dir='neurosynth_data',
    version='7'
)

# Convert to NiMARE Dataset
dset = ns_dset

# Select subset by term
emotion_ids = dset.get_studies_by_label('emotion', threshold=0.001)
dset_emotion = dset.slice(emotion_ids)

# Save
dset_emotion.save('neurosynth_emotion.pkl')
```

### Import from NeuroQuery

```python
from nimare import extract

# Download NeuroQuery dataset
nq_dset = extract.download_neuroquery(
    data_dir='neuroquery_data',
    version='1'
)

# Use in NiMARE analyses
```

### Import from Sleuth (GingerALE)

```python
from nimare.io import convert_sleuth_to_dataset

# Convert Sleuth text file to NiMARE Dataset
dset = convert_sleuth_to_dataset('sleuth_file.txt')

# Now can use in NiMARE
ale = ALE()
results = ale.fit(dset)
```

### Export Results

```python
# Save all maps from results
results.save_maps(output_dir='results/', prefix='emotion_ale_')

# Individual maps
z_map = results.get_map('z')
z_map.to_filename('z_map.nii.gz')

# Export as table
peaks_df = results.get_table()
peaks_df.to_csv('peaks_table.csv', index=False)
```

## Batch Processing

### Multiple Meta-Analyses

```python
from nimare.meta.cbma import ALE
from nimare.correct import FWECorrector
import os

# Terms to analyze
terms = ['emotion', 'cognition', 'motor', 'visual', 'language']

# Setup
ale = ALE(kernel__fwhm=15)
corrector = FWECorrector(method='montecarlo', n_iters=10000, n_cores=-1)

results_dict = {}

for term in terms:
    print(f"Processing: {term}")

    # Get studies
    ids = dset.get_studies_by_label(term, threshold=0.001)

    if len(ids) < 17:  # ALE minimum
        print(f"  Skipping (only {len(ids)} studies)")
        continue

    # Create subset
    dset_term = dset.slice(ids)

    # Meta-analysis
    results = ale.fit(dset_term)

    # Correct
    results_fwe = corrector.transform(results)

    # Save
    output_dir = f'results/{term}'
    os.makedirs(output_dir, exist_ok=True)
    results_fwe.save_maps(output_dir=output_dir, prefix=f'{term}_')

    results_dict[term] = {
        'n_studies': len(ids),
        'results': results_fwe
    }

print(f"\nProcessed {len(results_dict)} terms")
```

## Visualization

### Plot ALE Results

```python
from nilearn import plotting
import matplotlib.pyplot as plt

# Load result
z_map = results.get_map('z')

# Glass brain
fig = plotting.plot_glass_brain(
    z_map,
    threshold=3.0,
    colorbar=True,
    title='ALE Meta-Analysis - Emotion',
    plot_abs=False
)
plt.savefig('emotion_ale_glass.png', dpi=300, bbox_inches='tight')

# Stat map
fig = plotting.plot_stat_map(
    z_map,
    threshold=3.0,
    display_mode='z',
    cut_coords=6,
    title='ALE Meta-Analysis - Emotion'
)
plt.savefig('emotion_ale_slices.png', dpi=300, bbox_inches='tight')
```

### Interactive Visualization

```python
from nilearn.plotting import view_img

# Create interactive HTML viewer
view = view_img(
    z_map,
    threshold=3.0,
    title='Interactive ALE Results'
)

# Save
view.save_as_html('ale_results_interactive.html')
```

## Integration with Claude Code

When helping users with NiMARE:

1. **Check Installation:**
   ```python
   import nimare
   print(nimare.__version__)
   ```

2. **Load Data:**
   ```python
   from nimare import dataset
   dset = dataset.Dataset.load('dataset.pkl')
   ```

3. **Common Workflow:**
   - Load/create dataset → Select studies → Run meta-analysis → Correct → Visualize

4. **Choose Algorithm:**
   - ALE: Standard, well-validated
   - MKDA: Better for small datasets
   - KDA: Simple density analysis
   - Use IBMA when full images available

## Troubleshooting

**Problem:** "Insufficient studies" error
**Solution:** Need minimum 17 experiments for ALE, reduce for MKDA/KDA

**Problem:** Memory errors during correction
**Solution:** Reduce n_iters, use fewer cores, process smaller datasets

**Problem:** Slow Monte Carlo correction
**Solution:** Use n_cores=-1 for parallel processing, reduce iterations for testing

**Problem:** Empty results
**Solution:** Check coordinate space (MNI/Talairach), verify study selection, lower threshold

**Problem:** Import errors from NeuroSynth
**Solution:** Update both packages, use extract module, check data paths

## Best Practices

1. **Always use multiple comparison correction** (FWE or FDR)
2. **Report all parameters** (kernel size, iterations, threshold)
3. **Check minimum study requirements** for each algorithm
4. **Visualize results** before interpreting
5. **Perform diagnostics** (jackknife, funnel plots)
6. **Document NiMARE version** in publications
7. **Share code and datasets** for reproducibility

## Resources

- **Documentation:** https://nimare.readthedocs.io/
- **GitHub:** https://github.com/neurostuff/NiMARE
- **Tutorials:** https://nimare.readthedocs.io/en/latest/auto_examples/
- **Forum:** https://neurostars.org/tag/nimare
- **NeuroStuff:** https://github.com/neurostuff
- **Paper:** https://doi.org/10.52294/31bb5c68-e7d3-45ee-bea7-e179f53d7b70

## Citation

```bibtex
@article{salo2023nimare,
  title={NiMARE: Neuroimaging Meta-Analysis Research Environment},
  author={Salo, Taylor and Yarkoni, Tal and Nichols, Thomas E and Poline, Jean-Baptiste and Kent, James D and Gorgolewski, Krzysztof J and Glerean, Enrico and Bottenhorn, Katherine L and Bilgel, Murat and Wright, Jessey and others},
  journal={Aperture Neuro},
  volume={3},
  year={2023},
  publisher={Organization for Human Brain Mapping}
}
```

## Related Tools

- **NeuroSynth:** Large-scale automated meta-analysis
- **NeuroQuery:** Predictive meta-analysis
- **GingerALE:** Activation likelihood estimation (standalone)
- **AES-SDM:** Seed-based d mapping
- **PyMARE:** Meta-analysis in Python (used by NiMARE)
- **NeuroVault:** Share statistical maps
