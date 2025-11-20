# Nilearn

## Overview

Nilearn is a Python library for fast and easy statistical learning on neuroimaging data. It leverages scikit-learn for machine learning, provides tools for fMRI data manipulation, statistical analysis, and visualization, making it ideal for predictive modeling, decoding, connectivity analysis, and statistical inference.

**Website:** https://nilearn.github.io/
**Platform:** Cross-platform (Linux/macOS/Windows)
**Language:** Python
**License:** BSD 3-Clause

## Key Features

- fMRI data manipulation and masking
- Statistical analysis (GLM, decoding, encoding)
- Machine learning for neuroimaging
- Functional connectivity analysis
- Parcellation and atlas manipulation
- Surface-based analysis
- Advanced visualization (glass brain, surface plots)
- BIDS integration
- Integration with scikit-learn

## Installation

```bash
# Using pip
pip install nilearn

# Using conda
conda install -c conda-forge nilearn

# Development version
pip install git+https://github.com/nilearn/nilearn.git
```

### Verify Installation

```python
import nilearn
print(nilearn.__version__)

# Check example data
from nilearn import datasets
print(datasets.get_data_dirs())
```

## Data Loading and Manipulation

### Loading fMRI Data

```python
from nilearn import datasets, image

# Load example data
haxby_dataset = datasets.fetch_haxby()
func_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask

# Load as 4D image
from nilearn.image import load_img
func_img = load_img(func_filename)
print(func_img.shape)  # (40, 64, 64, 1452)

# Get data array
data = func_img.get_fdata()
```

### Image Manipulation

```python
from nilearn import image

# Resampling
resampled_img = image.resample_to_img(
    source_img=func_img,
    target_img=anat_img,
    interpolation='continuous'
)

# Smoothing
smoothed_img = image.smooth_img(func_img, fwhm=6)

# Mean image
mean_img = image.mean_img(func_img)

# Math operations
scaled_img = image.math_img('img * 2', img=func_img)
thresholded_img = image.math_img('img * (img > 100)', img=mean_img)

# Clean signal
from nilearn.image import clean_img
cleaned_img = clean_img(
    func_img,
    detrend=True,
    standardize=True,
    high_pass=0.01,
    t_r=2.5,
    ensure_finite=True
)
```

## Masking and Feature Extraction

```python
from nilearn.maskers import NiftiMasker

# Create masker
masker = NiftiMasker(
    mask_img=mask_filename,
    standardize=True,
    detrend=True,
    high_pass=0.01,
    t_r=2.5,
    memory='nilearn_cache',  # Enable caching
    memory_level=1
)

# Fit masker and extract 2D signal array
fmri_masked = masker.fit_transform(func_img)
print(fmri_masked.shape)  # (n_timepoints, n_voxels)

# Inverse transform back to 4D image
fmri_img_reconstructed = masker.inverse_transform(fmri_masked)
```

### Multi-Subject Masker

```python
from nilearn.maskers import MultiNiftiMasker

# Process multiple subjects
multi_masker = MultiNiftiMasker(
    mask_strategy='epi',  # Compute mask from EPI images
    standardize=True,
    memory='nilearn_cache'
)

# Extract from multiple runs/subjects
fmri_data_list = [func_run1, func_run2, func_run3]
masked_data = multi_masker.fit_transform(fmri_data_list)
```

## General Linear Model (GLM)

```python
from nilearn.glm.first_level import FirstLevelModel
import pandas as pd

# Design matrix
events = pd.read_csv('events.tsv', sep='\t')

# First-level model
fmri_glm = FirstLevelModel(
    t_r=2.0,
    noise_model='ar1',
    standardize=False,
    hrf_model='spm',
    drift_model='cosine',
    high_pass=0.01
)

# Fit model
fmri_glm = fmri_glm.fit(func_img, events=events)

# Define contrasts
contrasts = {
    'active-rest': 'active - rest',
    'faces-houses': 'faces - houses'
}

# Compute contrasts
z_map = fmri_glm.compute_contrast('active-rest', output_type='z_score')
effect_map = fmri_glm.compute_contrast('active-rest', output_type='effect_size')
stat_map = fmri_glm.compute_contrast('active-rest', output_type='stat')

# Save results
from nilearn.image import save_img
save_img(z_map, 'z_map.nii.gz')
```

### Second-Level Analysis

```python
from nilearn.glm.second_level import SecondLevelModel

# Collect first-level contrast maps
contrast_maps = [sub01_contrast, sub02_contrast, sub03_contrast]

# Design matrix for second level
import numpy as np
design_matrix = pd.DataFrame(
    [1] * len(contrast_maps),
    columns=['intercept']
)

# Second-level model
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(
    contrast_maps,
    design_matrix=design_matrix
)

# Group-level statistical map
z_map = second_level_model.compute_contrast(output_type='z_score')

# Thresholded map
from nilearn.glm import threshold_stats_img
thresholded_map, threshold = threshold_stats_img(
    z_map,
    alpha=0.05,
    height_control='fdr',
    cluster_threshold=10
)
```

## Machine Learning and Decoding

### Classification

```python
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

# Load data and labels
fmri_data = [run1_img, run2_img, run3_img]
conditions = ['face', 'house', 'chair', 'scramble'] * 48

# Decoder with built-in feature selection
decoder = Decoder(
    estimator='svc',  # SVM classifier
    mask=mask_filename,
    standardize=True,
    screening_percentile=5,
    scoring='accuracy',
    cv=5
)

# Fit and score
decoder.fit(fmri_data, conditions)
print(f"Accuracy: {decoder.score(fmri_data, conditions)}")

# Get feature weights
coef_img = decoder.coef_img_[('face', 'house')]
```

### Searchlight Analysis

```python
from nilearn.decoding import SearchLight
from sklearn.svm import SVC

# Setup searchlight
searchlight = SearchLight(
    mask_img=mask_filename,
    estimator=SVC(kernel='linear'),
    radius=5.6,  # in mm
    n_jobs=-1,  # Use all CPUs
    verbose=1,
    cv=5
)

# Run searchlight
searchlight.fit(func_img, conditions)

# Get accuracy map
searchlight_img = image.new_img_like(mask_filename, searchlight.scores_)
```

### Encoding Models

```python
from nilearn.glm.first_level import make_first_level_design_matrix

# Build design matrix with multiple regressors
frame_times = np.arange(n_scans) * t_r

# Add behavioral regressors
design_matrix = pd.DataFrame({
    'trial_type': trial_types,
    'reaction_time': reaction_times,
    'accuracy': accuracy_scores
})

design_matrix = make_first_level_design_matrix(
    frame_times,
    events=design_matrix,
    hrf_model='glover',
    drift_model='cosine',
    high_pass=0.01
)

# Fit encoding model
from sklearn.linear_model import Ridge
encoder = Ridge(alpha=100)

# For each voxel, predict BOLD from behavioral variables
from nilearn.maskers import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename)
fmri_data = masker.fit_transform(func_img)

encoder.fit(design_matrix, fmri_data)
predicted_fmri = encoder.predict(design_matrix)
```

## Connectivity Analysis

### Seed-Based Correlation

```python
from nilearn.maskers import NiftiSpheresMasker

# Define seed coordinates (MNI space)
seed_coords = [(0, -52, 18)]  # PCC

# Extract seed time series
seed_masker = NiftiSpheresMasker(
    seed_coords,
    radius=6,
    detrend=True,
    standardize=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)

seed_time_series = seed_masker.fit_transform(func_img)

# Compute connectivity with all voxels
brain_masker = NiftiMasker(
    mask_img=mask_filename,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)

brain_time_series = brain_masker.fit_transform(func_img)

# Correlation
from scipy.stats import pearsonr
correlation_map = np.zeros(brain_time_series.shape[1])

for i in range(brain_time_series.shape[1]):
    correlation_map[i] = pearsonr(
        seed_time_series[:, 0],
        brain_time_series[:, i]
    )[0]

# Convert to image
correlation_img = brain_masker.inverse_transform(correlation_map)
```

### ROI-to-ROI Connectivity

```python
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

# Load atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = atlas.maps

# Extract ROI time series
labels_masker = NiftiLabelsMasker(
    labels_img=atlas_filename,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)

roi_time_series = labels_masker.fit_transform(func_img)

# Compute connectivity matrix
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([roi_time_series])[0]

# Plot matrix
from nilearn import plotting
plotting.plot_matrix(
    correlation_matrix,
    labels=atlas.labels,
    colorbar=True,
    vmax=1,
    vmin=-1
)
```

### Independent Component Analysis (ICA)

```python
from nilearn.decomposition import CanICA

# Multi-subject ICA
canica = CanICA(
    n_components=20,
    mask=mask_filename,
    smoothing_fwhm=6,
    memory='nilearn_cache',
    memory_level=2,
    threshold=3.0,
    n_jobs=-1,
    random_state=42
)

# Fit on multiple subjects
canica.fit(fmri_data_list)

# Get components
components_img = canica.components_img_

# Transform new subject
subject_scores = canica.transform(new_subject_img)
```

## Visualization

### Statistical Maps

```python
from nilearn import plotting

# Glass brain
plotting.plot_glass_brain(
    stat_map,
    threshold=3.0,
    colorbar=True,
    title='Statistical Map',
    plot_abs=False,
    display_mode='lyrz'
)

# Overlay on anatomical
plotting.plot_stat_map(
    stat_map,
    bg_img=anat_img,
    threshold=3.0,
    display_mode='ortho',
    cut_coords=(0, 0, 0),
    title='Activation Map'
)

# Interactive view
view = plotting.view_img(
    stat_map,
    threshold=3.0,
    bg_img=anat_img,
    cmap='cold_hot'
)
view.open_in_browser()
```

### Connectivity Visualization

```python
# Connectivity matrix
plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=roi_labels,
    vmax=1,
    vmin=-1,
    reorder=True
)

# Connectome on glass brain
coords = plotting.find_xyz_cut_coords(stat_map)

plotting.plot_connectome(
    correlation_matrix,
    coords,
    edge_threshold='80%',
    title='Functional Connectome',
    node_size=50
)
```

### Surfaceplotting

```python
from nilearn import surface

# Project volume to surface
fsaverage = datasets.fetch_surf_fsaverage()

texture_left = surface.vol_to_surf(
    stat_map,
    fsaverage.pial_left
)

# Plot on surface
plotting.plot_surf_stat_map(
    fsaverage.infl_left,
    texture_left,
    hemi='left',
    title='Left Hemisphere',
    threshold=2.0,
    bg_map=fsaverage.sulc_left
)

# Interactive surface view
view = plotting.view_surf(
    fsaverage.infl_left,
    texture_left,
    threshold=2.0,
    bg_map=fsaverage.sulc_left
)
view.open_in_browser()
```

## Atlases and Parcellations

```python
from nilearn import datasets

# Harvard-Oxford atlas
ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# AAL atlas
aal = datasets.fetch_atlas_aal()

# Schaefer parcellation
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400)

# BASC multiscale atlas
basc = datasets.fetch_atlas_basc_multiscale_2015()

# Probabilistic atlases
destrieux = datasets.fetch_atlas_surf_destrieux()

# Visualize atlas
from nilearn import plotting
plotting.plot_roi(
    schaefer.maps,
    title='Schaefer 400 Parcellation',
    cut_coords=(0, 0, 0)
)
```

## Integration with Claude Code

When helping users with Nilearn:

1. **Check Installation:**
   ```python
   import nilearn
   print(nilearn.__version__)
   ```

2. **Common Issues:**
   - Memory errors with large 4D images
   - Masking inconsistencies
   - Design matrix errors in GLM
   - Coordinate system mismatches

3. **Performance Tips:**
   - Enable memory caching
   - Use memory_level parameter
   - Reduce memory with `standardize='zscore_sample'`
   - Use `low_memory=True` for large datasets

4. **Best Practices:**
   - Always check image orientations and affines
   - Use maskers for consistent preprocessing
   - Cache intermediate results
   - Visualize data at each step

## Troubleshooting

**Problem:** "Memory error"
**Solution:** Use `low_memory=True`, reduce smoothing, or process in batches

**Problem:** "Mask and data have incompatible shapes"
**Solution:** Resample mask to data space with `image.resample_to_img()`

**Problem:** GLM fitting fails
**Solution:** Check design matrix rank, ensure enough timepoints, verify events file

## Resources

- Documentation: https://nilearn.github.io/
- Examples Gallery: https://nilearn.github.io/stable/auto_examples/index.html
- GitHub: https://github.com/nilearn/nilearn
- Discussion: https://neurostars.org/ (tag: nilearn)

## Related Tools

- **scikit-learn:** Machine learning library
- **NiBabel:** Neuroimaging file I/O
- **Nipype:** Workflow management
- **Nistats:** (deprecated, merged into Nilearn)

## Citation

```bibtex
@article{abraham2014nilearn,
  title={Machine learning for neuroimaging with scikit-learn},
  author={Abraham, Alexandre and Pedregosa, Fabian and Eickenberg, Michael and others},
  journal={Frontiers in Neuroinformatics},
  volume={8},
  pages={14},
  year={2014},
  doi={10.3389/fninf.2014.00014}
}
```
