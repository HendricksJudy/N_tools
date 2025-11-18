# Nilearn GLM - General Linear Model for fMRI Analysis

## Overview

Nilearn's GLM module provides a comprehensive framework for statistical analysis of functional MRI data in Python. It implements both first-level (single-subject) and second-level (group-level) general linear models with support for various experimental designs, hemodynamic response function (HRF) modeling, confound regression, and statistical inference. Unlike MATLAB-based alternatives (SPM, FSL), Nilearn GLM offers a pure Python solution with excellent integration into the scientific Python ecosystem, making it ideal for reproducible research, custom analysis pipelines, and educational purposes.

The GLM framework is the gold standard for analyzing task-based fMRI data, modeling BOLD signal as a linear combination of predicted responses to experimental conditions plus noise. Nilearn's implementation supports canonical HRF modeling, temporal/dispersion derivatives, parametric modulation, and sophisticated confound regression strategies. It integrates seamlessly with fMRIPrep outputs and provides comprehensive visualization tools for design matrices, statistical maps, and glass brain projections.

**Official Website:** https://nilearn.github.io/stable/glm/index.html
**Repository:** https://github.com/nilearn/nilearn
**Documentation:** https://nilearn.github.io/stable/modules/reference.html#module-nilearn.glm

### Key Features

- **First-Level GLM:** Single-subject analysis with flexible design matrix construction
- **Second-Level GLM:** Group-level statistical inference (t-tests, ANOVA)
- **HRF Modeling:** Canonical HRF, temporal/dispersion derivatives, FIR models
- **Parametric Modulation:** Model trial-by-trial variability in activation
- **Confound Regression:** Motion parameters, physiological noise, global signals
- **Contrast Computation:** T-contrasts, F-contrasts, conjunction analysis
- **Multiple Comparison Correction:** FWE, FDR, cluster-level inference
- **Design Matrix Visualization:** Publication-quality design matrix plots
- **Statistical Map Visualization:** Glass brains, statistical overlays, surface projections
- **fMRIPrep Integration:** Direct loading of fMRIPrep derivatives and confounds
- **Python Ecosystem:** Works with NumPy, Pandas, Matplotlib, scikit-learn

### Applications

- Task-based fMRI analysis (event-related, block, mixed designs)
- Single-subject and group-level statistical inference
- Parametric modulation and psychophysiological interaction (PPI)
- Custom analysis pipelines and method development
- Educational demonstrations of fMRI statistics
- Reproducible research with Python-based workflows

### Citation

```bibtex
@article{abraham2014machine,
  title={Machine learning for neuroimaging with scikit-learn},
  author={Abraham, Alexandre and Pedregosa, Fabian and Eickenberg, Michael and
          Gervais, Philippe and Mueller, Andreas and Kossaifi, Jean and
          Gramfort, Alexandre and Thirion, Bertrand and Varoquaux, Ga{\"e}l},
  journal={Frontiers in Neuroinformatics},
  volume={8},
  pages={14},
  year={2014},
  publisher={Frontiers}
}

@article{Brett2002GLM,
  title={The general linear model},
  author={Brett, Matthew and Penny, Will and Kiebel, Stefan},
  journal={Statistical Parametric Mapping: The Analysis of Functional Brain Images},
  pages={159--187},
  year={2007},
  publisher={Academic Press}
}
```

---

## Installation

### Installing Nilearn with GLM Dependencies

Nilearn GLM requires Python ≥3.8 and several scientific Python packages:

```bash
# Install via pip (recommended)
pip install nilearn

# Install with all dependencies for GLM analysis
pip install nilearn[plotting]

# Install development version from GitHub
pip install git+https://github.com/nilearn/nilearn.git
```

### Conda Installation

```bash
# Install from conda-forge
conda install -c conda-forge nilearn

# Create dedicated environment for neuroimaging
conda create -n neuro python=3.10 nilearn matplotlib pandas
conda activate neuro
```

### Verify Installation

```python
import nilearn
from nilearn import glm
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel

print(f"Nilearn version: {nilearn.__version__}")
print("GLM module successfully imported")
```

### Required Data Formats

- **Functional images:** NIfTI format (.nii, .nii.gz)
- **Event files:** TSV or CSV format (onset, duration, trial_type)
- **Confound files:** TSV format (motion parameters, physiological regressors)
- **Anatomical images:** NIfTI format for overlay visualization

### Download Example Dataset

```python
from nilearn import datasets

# Download task fMRI dataset (motor localizer)
data = datasets.fetch_localizer_button_task()
fmri_file = data.epi_img
print(f"Downloaded fMRI data: {fmri_file}")

# Access events information
print(f"Events file: {data.events}")
```

---

## First-Level Analysis

First-level GLM analyzes individual subject data, modeling the BOLD response to experimental conditions.

### Design Matrix Construction

The design matrix specifies how experimental conditions relate to the observed fMRI signal:

```python
import pandas as pd
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix

# Load events from TSV file
events = pd.read_csv('sub-01_task-faces_events.tsv', sep='\t')
print(events.head())
# Output: onset, duration, trial_type

# Get acquisition parameters
n_scans = 200
tr = 2.0  # repetition time in seconds
frame_times = np.arange(n_scans) * tr

# Create design matrix
design_matrix = make_first_level_design_matrix(
    frame_times=frame_times,
    events=events,
    hrf_model='spm',  # canonical HRF
    drift_model='cosine',  # high-pass filter
    high_pass=0.01  # 1/128 Hz cutoff
)

print(design_matrix.head())
print(f"Design matrix shape: {design_matrix.shape}")
```

### Event-Related Design

Model brief, isolated events (e.g., individual trials):

```python
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix

# Define events (onset, duration, trial_type)
events_df = pd.DataFrame({
    'onset': [10, 30, 50, 70, 90],
    'duration': [1, 1, 1, 1, 1],
    'trial_type': ['face', 'house', 'face', 'house', 'face']
})

# Initialize first-level model
fmri_glm = FirstLevelModel(
    t_r=2.0,
    noise_model='ar1',  # autoregressive noise model
    standardize=False,
    hrf_model='spm',
    drift_model='cosine',
    high_pass=0.01
)

# Fit GLM
fmri_glm = fmri_glm.fit('sub-01_task-faces_bold.nii.gz', events=events_df)

# Visualize design matrix
design_matrix = fmri_glm.design_matrices_[0]
plot_design_matrix(design_matrix)
```

### Block Design

Model sustained periods of activation:

```python
# Block design events (longer durations)
block_events = pd.DataFrame({
    'onset': [0, 30, 60, 90, 120],
    'duration': [20, 20, 20, 20, 20],
    'trial_type': ['rest', 'task', 'rest', 'task', 'rest']
})

# Fit GLM with block design
fmri_glm_block = FirstLevelModel(t_r=2.0, hrf_model='spm')
fmri_glm_block = fmri_glm_block.fit(
    'sub-01_task-block_bold.nii.gz',
    events=block_events
)
```

### Mixed Design (Events + Blocks)

Combine event-related and block conditions:

```python
# Mixed design: short events during task blocks
mixed_events = pd.DataFrame({
    'onset': [0, 15, 30, 45, 60, 75, 90],
    'duration': [1, 1, 30, 1, 1, 30, 1],
    'trial_type': ['cue', 'target', 'task_block', 'cue', 'target', 'task_block', 'cue']
})

fmri_glm_mixed = FirstLevelModel(t_r=2.0, hrf_model='spm + derivative')
fmri_glm_mixed = fmri_glm_mixed.fit(
    'sub-01_task-mixed_bold.nii.gz',
    events=mixed_events
)
```

### Confound Regression

Include nuisance regressors to remove non-neural variance:

```python
# Load confounds from fMRIPrep
confounds = pd.read_csv('sub-01_task-faces_desc-confounds_timeseries.tsv', sep='\t')

# Select confounds to include
confound_vars = [
    'trans_x', 'trans_y', 'trans_z',  # motion parameters
    'rot_x', 'rot_y', 'rot_z',
    'csf', 'white_matter',  # physiological noise
    'framewise_displacement'  # motion quality metric
]

selected_confounds = confounds[confound_vars]

# Fit GLM with confounds
fmri_glm_confound = FirstLevelModel(t_r=2.0, hrf_model='spm')
fmri_glm_confound = fmri_glm_confound.fit(
    'sub-01_task-faces_bold.nii.gz',
    events=events_df,
    confounds=selected_confounds
)
```

---

## HRF Modeling

The hemodynamic response function (HRF) models the BOLD signal's temporal dynamics.

### Canonical HRF

Standard SPM canonical HRF (double-gamma function):

```python
from nilearn.glm.first_level import FirstLevelModel

# Use SPM canonical HRF
fmri_glm = FirstLevelModel(
    t_r=2.0,
    hrf_model='spm',  # canonical HRF
    standardize=False
)

fmri_glm.fit('func.nii.gz', events=events)
```

### Temporal and Dispersion Derivatives

Account for inter-subject/region variability in HRF shape:

```python
# Include temporal derivative
fmri_glm_deriv = FirstLevelModel(
    t_r=2.0,
    hrf_model='spm + derivative',  # canonical + temporal derivative
)

fmri_glm_deriv.fit('func.nii.gz', events=events)

# Include both temporal and dispersion derivatives
fmri_glm_deriv2 = FirstLevelModel(
    t_r=2.0,
    hrf_model='spm + derivative + dispersion',
)

fmri_glm_deriv2.fit('func.nii.gz', events=events)
```

### FIR (Finite Impulse Response) Model

Estimate HRF shape without assumptions:

```python
# FIR model with 16-second window (8 TRs at TR=2s)
fmri_glm_fir = FirstLevelModel(
    t_r=2.0,
    hrf_model='fir',
    fir_delays=np.arange(0, 16, 2),  # delays from 0 to 14s
)

fmri_glm_fir.fit('func.nii.gz', events=events)
```

### Custom HRF

Provide your own HRF function:

```python
import numpy as np
from scipy.stats import gamma

# Define custom HRF (example: custom gamma function)
def custom_hrf(t):
    """Custom HRF peaking at 5 seconds."""
    return gamma.pdf(t, 6) - 0.35 * gamma.pdf(t, 12)

# Use custom HRF
fmri_glm_custom = FirstLevelModel(
    t_r=2.0,
    hrf_model=custom_hrf,
)

fmri_glm_custom.fit('func.nii.gz', events=events)
```

---

## Contrast Estimation

Contrasts test specific hypotheses about experimental conditions.

### T-Contrasts

Test directional hypotheses (e.g., condition A > condition B):

```python
from nilearn.plotting import plot_stat_map

# Define contrast: faces > houses
contrast_def = 'face - house'

# Compute contrast
z_map = fmri_glm.compute_contrast(contrast_def, output_type='z_score')

# Visualize
plot_stat_map(
    z_map,
    threshold=3.1,  # p < 0.001 uncorrected
    title='Faces > Houses',
    display_mode='ortho'
)
```

### Multiple Contrasts

Compute several contrasts efficiently:

```python
# Define multiple contrasts
contrasts = {
    'faces': 'face',
    'houses': 'house',
    'faces_vs_houses': 'face - house',
    'houses_vs_faces': 'house - face',
    'all_stimuli': '0.5*face + 0.5*house'
}

# Compute all contrasts
z_maps = {}
for name, contrast in contrasts.items():
    z_maps[name] = fmri_glm.compute_contrast(
        contrast,
        output_type='z_score'
    )
    print(f"Computed: {name}")
```

### F-Contrasts

Test for any effect across multiple conditions (omnibus test):

```python
# F-contrast: test if faces OR houses activate
# (tests both face and house regressors jointly)
import numpy as np

# Get design matrix columns
design_matrix = fmri_glm.design_matrices_[0]
n_conditions = 2  # face and house

# Create F-contrast matrix (2x2 identity for 2 conditions)
f_contrast_matrix = np.eye(n_conditions)

# Compute F-contrast
f_map = fmri_glm.compute_contrast(
    f_contrast_matrix,
    stat_type='F',
    output_type='stat'
)

plot_stat_map(f_map, threshold=5.0, title='F-contrast: Any Stimulus')
```

### Conjunction Analysis

Test for regions active in multiple contrasts:

```python
from nilearn.glm import expression_to_contrast_vector

# Compute individual contrasts
z_faces = fmri_glm.compute_contrast('face', output_type='z_score')
z_houses = fmri_glm.compute_contrast('house', output_type='z_score')

# Conjunction: minimum of z-scores (conservative)
from nilearn.image import math_img
z_conjunction = math_img(
    'np.minimum(img1, img2)',
    img1=z_faces,
    img2=z_houses
)

plot_stat_map(z_conjunction, threshold=3.1, title='Conjunction: Faces AND Houses')
```

---

## Second-Level Analysis

Second-level GLM performs group-level statistical inference across subjects.

### One-Sample T-Test

Test if a contrast is significantly different from zero across subjects:

```python
from nilearn.glm.second_level import SecondLevelModel

# Collect first-level contrast maps from all subjects
contrast_maps = [
    'sub-01_faces-vs-houses_zmap.nii.gz',
    'sub-02_faces-vs-houses_zmap.nii.gz',
    'sub-03_faces-vs-houses_zmap.nii.gz',
    # ... more subjects
]

# Create design matrix (intercept only for one-sample t-test)
import pandas as pd
n_subjects = len(contrast_maps)
design_matrix = pd.DataFrame({'intercept': [1] * n_subjects})

# Fit second-level model
second_level_model = SecondLevelModel(smoothing_fwhm=5.0)
second_level_model = second_level_model.fit(
    contrast_maps,
    design_matrix=design_matrix
)

# Compute group-level z-map
z_map_group = second_level_model.compute_contrast(output_type='z_score')

plot_stat_map(z_map_group, threshold=3.29, title='Group: Faces > Houses (p<0.001)')
```

### Two-Sample T-Test

Compare two independent groups:

```python
# Contrast maps from two groups
group1_maps = ['sub-01_zmap.nii.gz', 'sub-02_zmap.nii.gz', 'sub-03_zmap.nii.gz']
group2_maps = ['sub-04_zmap.nii.gz', 'sub-05_zmap.nii.gz', 'sub-06_zmap.nii.gz']

all_maps = group1_maps + group2_maps

# Create design matrix with group indicator
design_matrix = pd.DataFrame({
    'group1': [1, 1, 1, 0, 0, 0],
    'group2': [0, 0, 0, 1, 1, 1]
})

# Fit model
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(all_maps, design_matrix=design_matrix)

# Contrast: group1 > group2
z_map_diff = second_level_model.compute_contrast('group1 - group2', output_type='z_score')

plot_stat_map(z_map_diff, threshold=3.1, title='Group 1 > Group 2')
```

### Paired T-Test

Compare within-subject conditions (e.g., pre vs. post treatment):

```python
# Pre and post treatment contrast maps
pre_maps = ['sub-01_pre_zmap.nii.gz', 'sub-02_pre_zmap.nii.gz']
post_maps = ['sub-01_post_zmap.nii.gz', 'sub-02_post_zmap.nii.gz']

# For paired t-test, compute difference images first
from nilearn.image import math_img
diff_maps = []
for pre, post in zip(pre_maps, post_maps):
    diff = math_img('img2 - img1', img1=pre, img2=post)
    diff_maps.append(diff)

# One-sample t-test on differences
design_matrix_paired = pd.DataFrame({'intercept': [1] * len(diff_maps)})
second_level_paired = SecondLevelModel()
second_level_paired = second_level_paired.fit(diff_maps, design_matrix=design_matrix_paired)

z_map_paired = second_level_paired.compute_contrast(output_type='z_score')
plot_stat_map(z_map_paired, threshold=3.1, title='Post > Pre Treatment')
```

### ANOVA with Covariates

Include continuous or categorical covariates:

```python
# Design matrix with age as covariate
design_matrix_cov = pd.DataFrame({
    'intercept': [1, 1, 1, 1, 1],
    'age': [25, 30, 35, 40, 45],  # continuous covariate
    'group': [0, 0, 1, 1, 1]  # categorical covariate
})

contrast_maps_cov = ['sub-01_zmap.nii.gz', 'sub-02_zmap.nii.gz',
                     'sub-03_zmap.nii.gz', 'sub-04_zmap.nii.gz',
                     'sub-05_zmap.nii.gz']

second_level_cov = SecondLevelModel()
second_level_cov = second_level_cov.fit(contrast_maps_cov, design_matrix=design_matrix_cov)

# Test age effect
z_map_age = second_level_cov.compute_contrast('age', output_type='z_score')
plot_stat_map(z_map_age, threshold=3.1, title='Age Effect')
```

---

## Statistical Inference

### Family-Wise Error (FWE) Correction

Control false positive rate across all voxels:

```python
from nilearn.glm import threshold_stats_img

# Compute contrast
z_map = fmri_glm.compute_contrast('face - house', output_type='z_score')

# FWE correction via permutation
thresholded_map, threshold = threshold_stats_img(
    z_map,
    alpha=0.05,
    height_control='fwe',  # family-wise error
    cluster_threshold=10
)

plot_stat_map(thresholded_map, title=f'FWE-corrected (threshold={threshold:.2f})')
```

### False Discovery Rate (FDR) Correction

Control expected proportion of false positives:

```python
# FDR correction (less conservative than FWE)
thresholded_fdr, threshold_fdr = threshold_stats_img(
    z_map,
    alpha=0.05,
    height_control='fdr'  # false discovery rate
)

plot_stat_map(thresholded_fdr, title=f'FDR-corrected (threshold={threshold_fdr:.2f})')
```

### Cluster-Level Inference

Test spatial extent of activations:

```python
# Cluster-level FWE correction
thresholded_cluster, threshold_cluster = threshold_stats_img(
    z_map,
    alpha=0.05,
    height_control=None,
    cluster_threshold=50,  # minimum cluster size in voxels
    two_sided=False
)

plot_stat_map(
    thresholded_cluster,
    title='Cluster-level FWE (k>50 voxels)'
)
```

---

## Integration with fMRIPrep

Nilearn GLM works seamlessly with fMRIPrep outputs:

```python
from nilearn.glm.first_level import FirstLevelModel
import pandas as pd

# fMRIPrep outputs
fmri_file = 'sub-01/func/sub-01_task-faces_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
events_file = 'sub-01/func/sub-01_task-faces_events.tsv'
confounds_file = 'sub-01/func/sub-01_task-faces_desc-confounds_timeseries.tsv'

# Load data
events = pd.read_csv(events_file, sep='\t')
confounds = pd.read_csv(confounds_file, sep='\t')

# Select minimal confound set
confound_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
selected_confounds = confounds[confound_cols].fillna(0)

# Fit GLM
fmri_glm = FirstLevelModel(t_r=2.0, hrf_model='spm', smoothing_fwhm=5.0)
fmri_glm.fit(fmri_file, events=events, confounds=selected_confounds)

# Compute contrast
z_map = fmri_glm.compute_contrast('face - house', output_type='z_score')
z_map.to_filename('sub-01_faces-vs-houses_zmap.nii.gz')
```

---

## Advanced Features

### Parametric Modulation

Model trial-by-trial variability:

```python
# Events with parametric modulator (e.g., reaction time)
events_param = pd.DataFrame({
    'onset': [10, 30, 50, 70],
    'duration': [1, 1, 1, 1],
    'trial_type': ['target', 'target', 'target', 'target'],
    'modulation': [0.5, 1.2, 0.8, 1.5]  # parametric modulator
})

# GLM will create separate regressor for parametric effect
fmri_glm_param = FirstLevelModel(t_r=2.0, hrf_model='spm')
fmri_glm_param.fit('func.nii.gz', events=events_param)

# Contrast for parametric modulation effect
z_map_param = fmri_glm_param.compute_contrast('target*modulation', output_type='z_score')
```

### High-Pass Filtering

Remove slow drifts in the GLM:

```python
# High-pass filtering at 1/128 Hz (default in SPM)
fmri_glm_hp = FirstLevelModel(
    t_r=2.0,
    hrf_model='spm',
    drift_model='cosine',
    high_pass=1/128  # cutoff frequency in Hz
)

fmri_glm_hp.fit('func.nii.gz', events=events)
```

### AR(1) Autocorrelation Model

Account for temporal autocorrelation in residuals:

```python
# AR(1) noise model (recommended for accurate inference)
fmri_glm_ar1 = FirstLevelModel(
    t_r=2.0,
    hrf_model='spm',
    noise_model='ar1',  # autoregressive order 1
    standardize=False,
    minimize_memory=True
)

fmri_glm_ar1.fit('func.nii.gz', events=events)
```

---

## Troubleshooting

### Design Matrix Singularity

**Error:** "Design matrix is singular"

**Solution:** Check for linear dependencies between regressors
```python
# Examine correlation between regressors
design_matrix = fmri_glm.design_matrices_[0]
corr_matrix = design_matrix.corr()
print(corr_matrix)

# Remove highly correlated confounds
```

### Collinearity Issues

**Problem:** High correlation between experimental conditions and confounds

**Solution:** Orthogonalize regressors or use more selective confounds
```python
# Use less confounds if collinearity is problematic
minimal_confounds = confounds[['trans_x', 'trans_y', 'trans_z']]
```

### Memory Issues

**Error:** "MemoryError" with large datasets

**Solution:** Enable memory-efficient mode
```python
fmri_glm = FirstLevelModel(
    t_r=2.0,
    minimize_memory=True,  # reduce memory footprint
    n_jobs=1  # avoid parallel processing overhead
)
```

### Contrast Not Found

**Error:** "Contrast not found in design matrix"

**Solution:** Check regressor names
```python
# Print available regressors
print(fmri_glm.design_matrices_[0].columns.tolist())
```

---

## Best Practices

### Design Efficiency

- Use jittered inter-stimulus intervals for event-related designs
- Optimize stimulus timing for statistical power (use design optimization tools)
- Include sufficient baseline/rest periods

### Confound Selection

- **Minimal:** 6 motion parameters (translation + rotation)
- **Standard:** + white matter/CSF signals
- **Aggressive:** + global signal, high-pass filtering, motion outliers
- Avoid over-fitting with too many confounds

### Statistical Power

- Ensure sufficient sample size (N≥20 for typical effect sizes)
- Use appropriate smoothing kernel (typically 1.5-2x voxel size)
- Preregister analyses to avoid multiple comparison issues

### Result Interpretation

- Always apply multiple comparison correction (FWE or FDR)
- Report cluster extent in addition to peak coordinates
- Validate findings with ROI-based analyses
- Consider effect sizes, not just significance

---

## References

1. **Nilearn GLM:**
   - Abraham et al. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics*, 8:14.
   - https://nilearn.github.io/stable/glm/index.html

2. **GLM Theory:**
   - Brett, Penny, Kiebel (2007). The general linear model. In *Statistical Parametric Mapping*.
   - Friston et al. (1994). Statistical parametric maps in functional imaging. *Human Brain Mapping*, 2(4):189-210.

3. **HRF Modeling:**
   - Glover (1999). Deconvolution of impulse response in event-related BOLD fMRI. *NeuroImage*, 9(4):416-429.

4. **Multiple Comparisons:**
   - Eklund et al. (2016). Cluster failure: Why fMRI inferences for spatial extent have inflated false-positive rates. *PNAS*, 113(28):7900-7905.

5. **Integration:**
   - Esteban et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature Methods*, 16:111-116.

**Official Resources:**
- Tutorial: https://nilearn.github.io/stable/auto_examples/04_glm_first_level/index.html
- API Reference: https://nilearn.github.io/stable/modules/reference.html#module-nilearn.glm
- fMRIPrep Integration: https://nilearn.github.io/stable/auto_examples/04_glm_first_level/plot_fmriprep_lss.html
