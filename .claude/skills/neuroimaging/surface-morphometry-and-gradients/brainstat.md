# BrainStat - Statistical Analysis for Neuroimaging

## Overview

**BrainStat** is a unified statistical analysis toolbox for neuroimaging data, developed at the Montreal Neurological Institute (MNI). It provides a comprehensive framework for both surface-based and volume-based statistical analyses, implementing linear models, mixed effects models, multiple comparison correction, and advanced techniques like Random Field Theory. BrainStat integrates seamlessly with common neuroimaging outputs (FreeSurfer, fMRIPrep, HCP) and provides both vertex-wise and parcel-wise analysis capabilities with consistent syntax across Python and MATLAB.

BrainStat fills a critical gap by providing modern statistical methods in an accessible framework, handling the complexities of neuroimaging data structure (surfaces, volumes, longitudinal, multi-site) while maintaining statistical rigor through proper multiple comparison correction and spatial modeling.

**Key Features:**
- Linear models for neuroimaging data (GLM)
- Mixed effects models for hierarchical and longitudinal data
- Surface-based statistical analysis (FreeSurfer and GIFTI surfaces)
- Volume-based analysis (NIfTI images)
- Multiple comparison correction (FDR, FWE, cluster-based)
- Random Field Theory for cluster correction
- Context decoding using NeuroSynth integration
- Meta-analysis utilities
- Visualization of statistical maps on surfaces
- Python and MATLAB implementations with consistent API
- Integration with BrainSpace for gradient analysis
- Parallelization support for large datasets

**Primary Use Cases:**
- Group comparison studies (patients vs. controls)
- Longitudinal neuroimaging analysis
- Multi-site harmonization and analysis
- Correlation with behavioral/clinical measures
- Vertex-wise and voxel-wise association studies
- Meta-analytic conjunction analyses
- Surface-based morphometry statistics

**Official Documentation:** https://brainstat.readthedocs.io/

---

## Installation

### Python Installation

```bash
# Install via pip
pip install brainstat

# Or install from GitHub for latest version
pip install git+https://github.com/MICA-MNI/BrainStat.git

# Verify installation
python -c "import brainstat; print(brainstat.__version__)"
```

### MATLAB Installation

```matlab
% Download BrainStat for MATLAB
% Visit: https://github.com/MICA-MNI/BrainStat/releases

% Add to MATLAB path
addpath(genpath('/path/to/BrainStat/matlab'));

% Verify installation
help term
```

### Install Dependencies

```bash
# Python dependencies
pip install numpy scipy pandas nibabel nilearn statsmodels

# For visualization
pip install matplotlib seaborn brainspace

# For parallel processing
pip install joblib
```

---

## Basic Linear Models

### Vertex-Wise GLM on Cortical Surface

```python
from brainstat.stats.terms import Term
from brainstat.stats.SLM import SLM
import numpy as np
import pandas as pd

# Load surface data (e.g., cortical thickness from FreeSurfer)
# Shape: (n_subjects, n_vertices)
n_subjects = 50
n_vertices = 10242  # fsaverage5 per hemisphere

cortical_thickness = np.random.randn(n_subjects, n_vertices) + 2.5

# Create design matrix
age = np.random.uniform(20, 80, n_subjects)
sex = np.random.choice([0, 1], n_subjects)  # 0=female, 1=male
group = np.random.choice([0, 1], n_subjects)  # 0=control, 1=patient

# Organize covariates in DataFrame
covariates = pd.DataFrame({
    'age': age,
    'sex': sex,
    'group': group
})

# Define model: thickness ~ age + sex + group
model = Term(1) + Term('age') + Term('sex') + Term('group')

# Fit linear model
slm = SLM(
    model,
    -Term('group'),  # Contrast: patient - control (negative for control - patient)
    surf='fsaverage5',
    mask=None,
    correction=['fdr', 'rft'],  # Multiple comparison corrections
    cluster_threshold=0.001
)

slm.fit(cortical_thickness)

# Access results
t_stats = slm.t  # T-statistics at each vertex
p_values = slm.P  # Uncorrected p-values

print(f"T-statistics shape: {t_stats.shape}")
print(f"Max T-stat: {np.max(t_stats):.3f}")
print(f"Min T-stat: {np.min(t_stats):.3f}")
```

### Multiple Comparison Correction

```python
# FDR correction
p_fdr = slm.Q  # FDR-corrected p-values

# Cluster-based correction (Random Field Theory)
p_cluster = slm.P['clus']  # Cluster p-values

# Significant vertices (FDR < 0.05)
sig_vertices_fdr = p_fdr < 0.05
n_sig_fdr = np.sum(sig_vertices_fdr)

print(f"Significant vertices (FDR < 0.05): {n_sig_fdr}")

# Significant clusters
if hasattr(slm, 'clusid'):
    sig_clusters = np.unique(slm.clusid[slm.clusid > 0])
    print(f"Significant clusters: {len(sig_clusters)}")
```

---

## Surface-Based Analysis

### Load FreeSurfer Data

```python
from brainstat.datasets import fetch_tutorial_data
import nibabel as nib

# Load example FreeSurfer thickness data
# In practice, load from FreeSurfer subjects directory

# Example: load thickness for one hemisphere
thickness_file = '/path/to/freesurfer/sub-01/surf/lh.thickness'

# Read with nibabel
# thickness_data = nib.freesurfer.read_morph_data(thickness_file)

# For multiple subjects, collect in array
# thickness_array shape: (n_subjects, n_vertices)
```

### Bilateral Analysis (Both Hemispheres)

```python
# Combine left and right hemisphere data
thickness_lh = np.random.randn(n_subjects, 10242)
thickness_rh = np.random.randn(n_subjects, 10242)

# Concatenate hemispheres
thickness_bilateral = np.concatenate([thickness_lh, thickness_rh], axis=1)

# Fit model to bilateral data
slm_bilateral = SLM(
    model,
    -Term('group'),
    surf='fsaverage5',  # BrainStat handles bilateral surfaces
    mask=None,
    correction='fdr'
)

slm_bilateral.fit(thickness_bilateral)

print(f"Bilateral analysis: {thickness_bilateral.shape[1]} vertices")
```

---

## Mixed Effects Models

### Longitudinal Analysis (Repeated Measures)

```python
from brainstat.stats.terms import Term, FixedEffect, RandomEffect
from brainstat.stats.SLM import SLM
import numpy as np
import pandas as pd

# Longitudinal data: 2 timepoints per subject
n_subjects = 30
n_timepoints = 2
n_total = n_subjects * n_timepoints
n_vertices = 10242

# Thickness data
thickness_long = np.random.randn(n_total, n_vertices) + 2.5

# Design matrix
subject_id = np.repeat(range(n_subjects), n_timepoints)
timepoint = np.tile([0, 1], n_subjects)  # Baseline, follow-up
age_baseline = np.repeat(np.random.uniform(20, 60, n_subjects), n_timepoints)
age = age_baseline + timepoint * 2  # Age increases by 2 years

covariates_long = pd.DataFrame({
    'subject': subject_id,
    'timepoint': timepoint,
    'age': age
})

# Mixed effects model: random intercept for subject
# thickness ~ timepoint + age + (1 | subject)

model_mixed = Term(1) + Term('timepoint') + Term('age') + RandomEffect(subject_id)

slm_mixed = SLM(
    model_mixed,
    -Term('timepoint'),  # Test effect of time
    surf='fsaverage5',
    correction='fdr'
)

slm_mixed.fit(thickness_long)

# Regions showing significant change over time
sig_change = slm_mixed.Q < 0.05

print(f"Vertices with significant longitudinal change: {np.sum(sig_change)}")
```

### Multi-Site Analysis with Random Effects

```python
# Multi-site study: account for site effects

n_sites = 3
site = np.random.choice(range(n_sites), n_subjects)

covariates_multisite = pd.DataFrame({
    'age': age,
    'sex': sex,
    'group': group,
    'site': site
})

# Model with site as random effect
model_site = Term(1) + Term('age') + Term('sex') + Term('group') + RandomEffect(site)

slm_site = SLM(
    model_site,
    -Term('group'),
    surf='fsaverage5',
    correction='fdr'
)

slm_site.fit(cortical_thickness)

print("Multi-site analysis complete")
```

---

## Volume-Based Analysis

### Voxel-Wise GLM on NIfTI Data

```python
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import Term
import nibabel as nib
import numpy as np

# Load volumetric data (e.g., gray matter density from VBM)
# Shape: (n_subjects, n_voxels) or (n_subjects, x, y, z)

n_subjects = 40
volume_shape = (91, 109, 91)  # MNI152 2mm dimensions
n_voxels = np.prod(volume_shape)

# Flatten for analysis
gm_density = np.random.randn(n_subjects, n_voxels)

# Design matrix
age = np.random.uniform(20, 80, n_subjects)
group = np.random.choice([0, 1], n_subjects)

covariates_vol = pd.DataFrame({
    'age': age,
    'group': group
})

# Model: GM ~ age + group
model_vol = Term(1) + Term('age') + Term('group')

# Fit model
slm_vol = SLM(
    model_vol,
    -Term('group'),
    mask=gm_density.mean(axis=0) > 0.1,  # Mask low-intensity voxels
    correction='fdr'
)

slm_vol.fit(gm_density)

# Reshape t-statistics back to 3D
t_stats_3d = slm_vol.t.reshape(volume_shape)

print(f"Volumetric T-stats shape: {t_stats_3d.shape}")
```

---

## Visualization

### Plot Statistical Map on Surface

```python
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_conte69
import numpy as np

# Load surfaces
surf_lh, surf_rh = load_conte69()

# T-statistics from bilateral analysis
t_stats_lh = slm_bilateral.t[:10242]
t_stats_rh = slm_bilateral.t[10242:]

# Threshold t-statistics for visualization
t_threshold = 2.5
t_stats_lh_thresh = t_stats_lh.copy()
t_stats_lh_thresh[np.abs(t_stats_lh) < t_threshold] = 0

t_stats_rh_thresh = t_stats_rh.copy()
t_stats_rh_thresh[np.abs(t_stats_rh) < t_threshold] = 0

# Plot on surface
plot_hemispheres(
    surf_lh, surf_rh,
    array_name=[t_stats_lh_thresh, t_stats_rh_thresh],
    size=(800, 400),
    cmap='RdBu_r',
    color_bar=True,
    label_text=['Group Difference (T-statistic)'],
    zoom=1.25
)
```

### Create Publication Figure

```python
import matplotlib.pyplot as plt
from nilearn import plotting

# For volumetric results, use nilearn
from nibabel import Nifti1Image

# Create NIfTI image from t-statistics
affine = np.eye(4)
affine[:3, :3] = 2.0 * np.eye(3)  # 2mm resolution
t_stats_img = Nifti1Image(t_stats_3d, affine)

# Plot statistical map
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Axial slices
plotting.plot_stat_map(
    t_stats_img,
    threshold=2.5,
    cmap='RdBu_r',
    symmetric_cbar=True,
    display_mode='z',
    cut_coords=5,
    axes=axes[0],
    title='Group Differences (Axial)'
)

# Sagittal
plotting.plot_stat_map(
    t_stats_img,
    threshold=2.5,
    cmap='RdBu_r',
    symmetric_cbar=True,
    display_mode='x',
    cut_coords=5,
    axes=axes[1],
    title='Group Differences (Sagittal)'
)

# Coronal
plotting.plot_stat_map(
    t_stats_img,
    threshold=2.5,
    cmap='RdBu_r',
    symmetric_cbar=True,
    display_mode='y',
    cut_coords=5,
    axes=axes[2],
    title='Group Differences (Coronal)'
)

plt.tight_layout()
plt.savefig('statistical_map.png', dpi=300, bbox_inches='tight')
```

---

## Context Decoding

### NeuroSynth Meta-Analysis Integration

```python
# Context decoding: what is this activation pattern associated with?
# Uses NeuroSynth database for meta-analytic decoding

# Identify significant clusters
sig_coords = []  # List of MNI coordinates for significant voxels

# In practice, extract coordinates from significant clusters
# For example purposes:
sig_coords = np.array([
    [40, -20, 50],   # Example coordinate 1
    [-40, -20, 50],  # Example coordinate 2
    [0, 30, -10]     # Example coordinate 3
])

# Use neurosynth for decoding (requires neurosynth package)
try:
    from neurosynth.base.dataset import Dataset
    from neurosynth import Masker

    # Load NeuroSynth database (requires download)
    # dataset = Dataset.load('neurosynth_dataset.pkl')

    # Decode activation pattern
    # decoded_terms = dataset.decode(sig_coords, r=6)
    # print("Associated terms:")
    # print(decoded_terms[:10])

except ImportError:
    print("Install neurosynth for context decoding:")
    print("pip install neurosynth")
```

---

## Advanced Features

### Interaction Terms

```python
# Test for age × group interaction

model_interaction = Term(1) + Term('age') + Term('group') + Term('age') * Term('group')

slm_interaction = SLM(
    model_interaction,
    Term('age') * Term('group'),  # Test interaction contrast
    surf='fsaverage5',
    correction='fdr'
)

slm_interaction.fit(cortical_thickness)

# Significant interaction indicates different age effects between groups
sig_interaction = slm_interaction.Q < 0.05

print(f"Vertices with significant age × group interaction: {np.sum(sig_interaction)}")
```

### Categorical Variables

```python
# Multiple groups (e.g., controls, MCI, AD)

diagnosis = np.random.choice(['control', 'mci', 'ad'], n_subjects)

covariates_categorical = pd.DataFrame({
    'age': age,
    'sex': sex,
    'diagnosis': diagnosis
})

# Automatically handles categorical encoding
model_categorical = Term(1) + Term('age') + Term('sex') + Term('diagnosis')

# F-test for overall effect of diagnosis
slm_categorical = SLM(
    model_categorical,
    Term('diagnosis'),  # F-test across all levels
    surf='fsaverage5',
    correction='fdr'
)

slm_categorical.fit(cortical_thickness)

# Pairwise contrasts
# Control vs. MCI
contrast_mci = slm_categorical.contrast(-Term('diagnosis[T.mci]'))

# Control vs. AD
contrast_ad = slm_categorical.contrast(-Term('diagnosis[T.ad]'))
```

---

## Batch Processing and Automation

### Multi-Contrast Analysis

```python
# Test multiple contrasts in one workflow

contrasts = {
    'group': -Term('group'),
    'age': -Term('age'),
    'sex': -Term('sex')
}

results = {}

for contrast_name, contrast in contrasts.items():
    slm_temp = SLM(
        model,
        contrast,
        surf='fsaverage5',
        correction='fdr'
    )

    slm_temp.fit(cortical_thickness)

    results[contrast_name] = {
        't': slm_temp.t,
        'p': slm_temp.P,
        'q': slm_temp.Q
    }

    print(f"{contrast_name}: {np.sum(results[contrast_name]['q'] < 0.05)} significant vertices")
```

### Parallel Processing

```python
from joblib import Parallel, delayed

def run_analysis(data, model, contrast):
    slm = SLM(model, contrast, surf='fsaverage5', correction='fdr')
    slm.fit(data)
    return slm

# Run analyses in parallel
subjects_batches = [cortical_thickness[i:i+10] for i in range(0, n_subjects, 10)]

results_parallel = Parallel(n_jobs=4)(
    delayed(run_analysis)(batch, model, -Term('group'))
    for batch in subjects_batches
)

print(f"Processed {len(results_parallel)} batches in parallel")
```

---

## MATLAB Implementation

### Basic GLM in MATLAB

```matlab
% Load surface data
thickness = randn(50, 10242) + 2.5;  % 50 subjects, 10242 vertices

% Design matrix
age = rand(50, 1) * 60 + 20;
group = round(rand(50, 1));

% Create term
model = 1 + term(age) + term(group);

% Contrast
contrast = -term(group);

% Fit SLM
slm = SLM(model, contrast, 'surf', 'fsaverage5');
slm = slm.fit(thickness);

% Results
t_stats = slm.t;
p_values = slm.P;

fprintf('Max T-stat: %.3f\n', max(t_stats));
fprintf('Significant vertices (FDR<0.05): %d\n', sum(slm.Q < 0.05));
```

---

## Troubleshooting

### Model Fitting Errors

```python
# Check for common issues

# 1. Design matrix rank deficiency
# Ensure no collinearity
import pandas as pd
from numpy.linalg import matrix_rank

X = covariates[['age', 'sex', 'group']].values
rank = matrix_rank(X)
print(f"Design matrix rank: {rank} (should be {X.shape[1]})")

# 2. Missing values
print(f"Missing in covariates: {covariates.isna().sum()}")
print(f"Missing in data: {np.isnan(cortical_thickness).sum()}")

# 3. Data dimensions
print(f"Data shape: {cortical_thickness.shape}")
print(f"Covariates shape: {covariates.shape}")
assert cortical_thickness.shape[0] == covariates.shape[0]
```

### Memory Issues

```python
# For large datasets, process in chunks

from brainstat.stats.SLM import SLM
import numpy as np

# Split vertices into chunks
chunk_size = 5000
n_chunks = int(np.ceil(n_vertices / chunk_size))

t_stats_full = np.zeros(n_vertices)

for i in range(n_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, n_vertices)

    data_chunk = cortical_thickness[:, start_idx:end_idx]

    slm_chunk = SLM(model, -Term('group'), correction=None)
    slm_chunk.fit(data_chunk)

    t_stats_full[start_idx:end_idx] = slm_chunk.t

    print(f"Processed chunk {i+1}/{n_chunks}")

# Apply FDR correction to full results
from statsmodels.stats.multitest import multipletests
_, p_fdr_full, _, _ = multipletests(p_values_full, method='fdr_bh')
```

---

## Best Practices

### Model Specification

1. **Include relevant covariates:**
   - Age, sex as minimum
   - Scanner/site for multi-site
   - Total intracranial volume for regional volumes

2. **Check assumptions:**
   - Residuals normality (Q-Q plots)
   - Homoscedasticity
   - No multicollinearity

3. **Sample size:**
   - Minimum ~20 per group for basic contrasts
   - More for interactions and complex designs

### Multiple Comparison Correction

1. **Choose appropriate method:**
   - FDR: Good balance, controls false discovery rate
   - RFT/cluster: More power for spatially extended effects
   - Bonferroni: Very conservative, use for ROI analysis

2. **Report thresholds:**
   - State correction method
   - Report cluster-forming threshold
   - Provide both corrected and uncorrected results

---

## Resources and Further Reading

### Official Documentation

- **BrainStat Docs:** https://brainstat.readthedocs.io/
- **GitHub:** https://github.com/MICA-MNI/BrainStat
- **Tutorials:** https://brainstat.readthedocs.io/en/latest/tutorials/index.html
- **API Reference:** https://brainstat.readthedocs.io/en/latest/api/index.html

### Related Tools

- **BrainSpace:** Gradient analysis integration
- **FreeSurfer:** Surface generation
- **fMRIPrep:** Preprocessing
- **statsmodels:** Statistical models in Python

### Key Publications

```
Larivière, S., et al. (2021).
BrainStat: A toolbox for statistical analysis of neuroimaging data.
NeuroImage, 239, 118303.
```

---

## Summary

**BrainStat** provides a unified statistical framework for neuroimaging:

**Strengths:**
- Consistent API across Python and MATLAB
- Surface and volume analysis
- Mixed effects for longitudinal/hierarchical data
- Proper multiple comparison correction
- Integration with BrainSpace
- Modern statistical methods

**Best For:**
- Group comparison studies
- Longitudinal analysis
- Multi-site studies
- Vertex-wise/voxel-wise associations
- Surface-based morphometry

**Typical Workflow:**
1. Load neuroimaging data (surfaces or volumes)
2. Specify design matrix and contrasts
3. Fit linear/mixed model
4. Apply multiple comparison correction
5. Visualize significant results

BrainStat streamlines statistical analysis of neuroimaging data with a modern, rigorous framework that handles the complexities of brain data while maintaining ease of use.

## Citation

```bibtex
@article{bethlehem2022brainstat,
  title={BrainStat: a toolbox for brain-wide statistics and multimodal integration},
  author={Bethlehem, Richard A. I. and Lariviere, Samuel and Paquola, Casey and others},
  journal={bioRxiv},
  year={2022},
  doi={10.1101/2022.04.07.487420}
}
```
