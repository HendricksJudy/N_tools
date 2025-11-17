# fMRIPrep QC Tools

## Overview

fMRIPrep generates comprehensive visual quality control reports and confound metrics as part of its preprocessing workflow. This skill focuses specifically on **interpreting and utilizing fMRIPrep's QC outputs**, including HTML visual reports, confound files, quality metrics extraction, and strategies for making data inclusion/exclusion decisions. While fMRIPrep preprocessing has been covered elsewhere, this skill provides deep coverage of quality control aspects essential for ensuring reliable fMRI analyses.

**Website:** https://fmriprep.org/
**Documentation:** https://fmriprep.readthedocs.io/
**GitHub:** https://github.com/nipreps/fmriprep
**Platform:** Python (Docker/Singularity)
**License:** Apache License 2.0

## Key Features

- Individual subject HTML visual reports
- Anatomical preprocessing QC (brain extraction, segmentation, registration)
- Functional preprocessing QC (motion correction, distortion correction)
- Alignment visualization (BOLD-to-T1w, T1w-to-template)
- Carpet plots for temporal quality assessment
- Comprehensive confounds files (100+ metrics)
- Motion parameters and framewise displacement
- DVARS and other temporal quality metrics
- Surface reconstruction quality
- Registration quality assessment
- Group-level QC aggregation
- Integration with MRIQC

## Understanding fMRIPrep Reports

### Report Structure

```python
# fMRIPrep generates HTML reports at:
# <output_dir>/fmriprep/sub-<label>.html
# <output_dir>/fmriprep/sub-<label>/figures/

# Report sections:
# 1. Summary (data, software versions)
# 2. Anatomical processing (brain extraction, segmentation, surfaces, registration)
# 3. Functional processing (per-run: alignment, confounds, carpet plots)
# 4. Error messages and warnings
```

### Navigate Reports

```bash
# Open individual report in browser
firefox sub-01.html

# Or use Python to serve reports
cd derivatives/fmriprep
python -m http.server 8000

# Navigate to http://localhost:8000
# Click on subject reports
```

### Report Overview Example

```python
# Parse fMRIPrep report metadata
from bs4 import BeautifulSoup

with open('sub-01.html', 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')

# Extract summary information
summary = soup.find('div', {'id': 'summary'})
print("Subject Summary:")
print(summary.get_text())

# Find warnings
warnings = soup.find_all('div', {'class': 'warning'})
print(f"\nWarnings: {len(warnings)}")
for w in warnings:
    print(f"  - {w.get_text()}")
```

## Anatomical QC

### Brain Extraction Quality

```python
# Check brain extraction from report images
# Located at: sub-<label>/figures/sub-<label>_desc-brain_mask.svg

# Visual inspection checklist:
# ✓ Entire brain included
# ✓ No skull/dura included
# ✓ Eyes excluded
# ✓ Cerebellum included
# ✓ Smooth mask boundaries

# Automated check using mask volume
import nibabel as nib
import numpy as np

mask = nib.load('sub-01/anat/sub-01_desc-brain_mask.nii.gz')
mask_data = mask.get_fdata()

# Brain volume (voxels)
brain_voxels = np.sum(mask_data > 0)
total_voxels = np.prod(mask_data.shape)

brain_fraction = brain_voxels / total_voxels

print(f"Brain mask: {brain_voxels} voxels ({brain_fraction*100:.1f}%)")

# Typical range: 15-25% of image volume
if brain_fraction < 0.10 or brain_fraction > 0.30:
    print("WARNING: Brain mask may be incorrect")
```

### T1w-to-Template Registration

```python
# Check alignment quality
# Report shows: sub-<label>_space-MNI152NLin2009cAsym_T1w.svg

# Visual inspection:
# ✓ Cortical ribbon aligned
# ✓ Ventricles aligned
# ✓ Corpus callosum aligned
# ✓ No visible warping artifacts
# ✓ Symmetric registration

# Automated check using registration metrics
# Load transformation matrix
import json

with open('sub-01/anat/sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.txt', 'r') as f:
    transform = f.read()

# Check for extreme values (indicating poor registration)
# This would require parsing the specific transform format
```

### Tissue Segmentation

```python
# Check CSF, GM, WM segmentation
# Report shows: sub-<label>_dseg.svg

# Segmentation quality checklist:
# ✓ GM/WM boundary clear
# ✓ Ventricles properly segmented
# ✓ Subcortical structures labeled
# ✓ No obvious mis-segmentations

# Load segmentation
seg = nib.load('sub-01/anat/sub-01_desc-aparcaseg_dseg.nii.gz')
seg_data = seg.get_fdata()

# Count tissue volumes
csf_voxels = np.sum((seg_data >= 4) & (seg_data <= 5))  # Lateral ventricles
gm_voxels = np.sum((seg_data >= 1000) & (seg_data < 3000))  # Cortical GM
wm_voxels = np.sum((seg_data == 2) | (seg_data == 41))  # Cerebral WM

print(f"CSF: {csf_voxels} voxels")
print(f"GM: {gm_voxels} voxels")
print(f"WM: {wm_voxels} voxels")

# Typical GM/WM ratio: 1.2-1.8
gm_wm_ratio = gm_voxels / wm_voxels
print(f"GM/WM ratio: {gm_wm_ratio:.2f}")

if gm_wm_ratio < 1.0 or gm_wm_ratio > 2.0:
    print("WARNING: Unusual GM/WM ratio")
```

### Surface Reconstruction

```python
# For fMRIPrep with FreeSurfer surfaces
# Report shows: sub-<label>_desc-reconall_T1w.svg

# Check surface quality:
# ✓ Pial surface follows GM/CSF boundary
# ✓ White surface follows GM/WM boundary
# ✓ No surface defects (holes, handles)
# ✓ Smooth surface topology

# Load surface metrics
import pandas as pd

# FreeSurfer stats
aseg_stats = pd.read_csv('sub-01/anat/sub-01_space-T1w_desc-aseg_stats.tsv', sep='\t')

# Euler number (topology check)
# Euler = 2 indicates no topological defects
```

## Functional QC

### Motion Correction

```python
# Check motion correction quality
# Report shows: motion plots and reference volume

# Visual inspection:
# ✓ Small motion parameters (< 3mm, < 3°)
# ✓ No sudden spikes
# ✓ Smooth motion trajectories

# Load motion parameters from confounds
conf = pd.read_csv('sub-01/func/sub-01_task-rest_desc-confounds_timeseries.tsv', sep='\t')

# Extract motion parameters
trans_x = conf['trans_x'].values
trans_y = conf['trans_y'].values
trans_z = conf['trans_z'].values
rot_x = conf['rot_x'].values
rot_y = conf['rot_y'].values
rot_z = conf['rot_z'].values

# Plot motion
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Translations
axes[0].plot(trans_x, label='X')
axes[0].plot(trans_y, label='Y')
axes[0].plot(trans_z, label='Z')
axes[0].set_ylabel('Translation (mm)')
axes[0].set_title('Motion Parameters - Translation')
axes[0].legend()
axes[0].grid(True)

# Rotations (convert to degrees)
axes[1].plot(np.rad2deg(rot_x), label='X')
axes[1].plot(np.rad2deg(rot_y), label='Y')
axes[1].plot(np.rad2deg(rot_z), label='Z')
axes[1].set_ylabel('Rotation (degrees)')
axes[1].set_xlabel('Volume')
axes[1].set_title('Motion Parameters - Rotation')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('motion_params.png')

# Summary statistics
print(f"Max translation: {np.max(np.abs([trans_x, trans_y, trans_z])):.2f} mm")
print(f"Max rotation: {np.max(np.abs(np.rad2deg([rot_x, rot_y, rot_z]))):.2f}°")
```

### BOLD-to-T1w Registration

```python
# Check BOLD-to-anatomical alignment
# Report shows: sub-<label>_task-<label>_desc-coreg_bold.svg

# Visual inspection:
# ✓ BOLD aligned to T1w
# ✓ Cortical ribbon visible
# ✓ Tissue boundaries match
# ✓ No distortion artifacts

# Boundary-based registration (BBR) cost
# Lower cost = better registration
# Look for: BBR cost < 0.7 (typical)
```

### Susceptibility Distortion Correction

```python
# If using field maps (SDC - susceptibility distortion correction)
# Report shows: before/after distortion correction

# Check:
# ✓ Reduced geometric distortion
# ✓ Better alignment to anatomy
# ✓ Frontal/temporal poles improved
# ✓ No overcorrection artifacts

# Field map quality (if available)
# Check for reasonable field map values (-200 to 200 Hz typical)
```

### Carpet Plots

```python
# Carpet plots show temporal patterns
# Report shows: sub-<label>_task-<label>_desc-carpetplot_bold.svg

# Interpretation:
# - Vertical lines = motion spikes
# - Horizontal bands = spatial artifacts
# - Smooth gradients = physiological noise
# - Periodic patterns = respiratory/cardiac

# Visual checklist:
# ✓ No excessive vertical lines
# ✓ No large horizontal bands
# ✓ Relatively smooth temporal patterns
# ✓ Check correspondence with FD plot above carpet
```

## Motion Assessment

### Framewise Displacement (FD)

```python
# Load confounds
conf = pd.read_csv('sub-01_task-rest_desc-confounds_timeseries.tsv', sep='\t')

# Framewise displacement
fd = conf['framewise_displacement'].values

# Summary statistics
mean_fd = np.nanmean(fd)  # Skip first value (NaN)
max_fd = np.nanmax(fd)
std_fd = np.nanstd(fd)

print(f"Mean FD: {mean_fd:.3f} mm")
print(f"Max FD: {max_fd:.3f} mm")
print(f"Std FD: {std_fd:.3f} mm")

# Quality thresholds
fd_threshold = 0.5  # mm (common threshold)
high_motion_vols = np.sum(fd > fd_threshold)
high_motion_pct = high_motion_vols / len(fd) * 100

print(f"Volumes with FD > {fd_threshold}mm: {high_motion_vols} ({high_motion_pct:.1f}%)")

# Flag if too much motion
if mean_fd > 0.5 or high_motion_pct > 20:
    print("WARNING: High motion detected")

# Plot FD
plt.figure(figsize=(12, 4))
plt.plot(fd, linewidth=0.5)
plt.axhline(fd_threshold, color='r', linestyle='--', label=f'Threshold ({fd_threshold}mm)')
plt.xlabel('Volume')
plt.ylabel('Framewise Displacement (mm)')
plt.title(f'Framewise Displacement (Mean: {mean_fd:.3f}mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fd_plot.png', dpi=150)
```

### DVARS

```python
# Derivative of RMS Variance over Voxels
# Measures intensity changes between volumes

# Standard DVARS
dvars_std = conf['std_dvars'].values

mean_dvars = np.nanmean(dvars_std)
max_dvars = np.nanmax(dvars_std)

print(f"Mean DVARS: {mean_dvars:.2f}")
print(f"Max DVARS: {max_dvars:.2f}")

# DVARS spike detection
dvars_threshold = 1.5  # standardized units
dvars_spikes = np.sum(dvars_std > dvars_threshold)

print(f"DVARS spikes (>{dvars_threshold}): {dvars_spikes}")

# Plot DVARS
plt.figure(figsize=(12, 4))
plt.plot(dvars_std, linewidth=0.5)
plt.axhline(dvars_threshold, color='r', linestyle='--')
plt.xlabel('Volume')
plt.ylabel('Standardized DVARS')
plt.title(f'DVARS (Mean: {mean_dvars:.2f})')
plt.grid(True, alpha=0.3)
plt.savefig('dvars_plot.png', dpi=150)
```

### Motion Scrubbing Criteria

```python
# Identify volumes to censor/scrub based on motion

# Criteria: FD > 0.5mm OR DVARS > 1.5
motion_outliers = (fd > 0.5) | (dvars_std > 1.5)

# Also censor adjacent volumes (common practice)
# Expand mask to include ±1 volumes
motion_outliers_expanded = np.copy(motion_outliers)
for i in np.where(motion_outliers)[0]:
    if i > 0:
        motion_outliers_expanded[i-1] = True
    if i < len(motion_outliers) - 1:
        motion_outliers_expanded[i+1] = True

n_censored = np.sum(motion_outliers_expanded)
n_retained = len(motion_outliers_expanded) - n_censored

print(f"Volumes to censor: {n_censored}/{len(motion_outliers_expanded)} ({n_censored/len(motion_outliers_expanded)*100:.1f}%)")
print(f"Volumes retained: {n_retained}")

# Check if enough data remains
min_volumes_required = 150  # Example threshold
if n_retained < min_volumes_required:
    print(f"WARNING: Insufficient data after scrubbing ({n_retained} < {min_volumes_required})")

# Save censoring mask
np.savetxt('censoring_mask.txt', motion_outliers_expanded.astype(int), fmt='%d')
```

## Confounds Files

### Understanding Confound Regressors

```python
# Load confounds file
conf = pd.read_csv('sub-01_task-rest_desc-confounds_timeseries.tsv', sep='\t')

print(f"Confounds available: {conf.shape[1]} columns, {conf.shape[0]} timepoints")
print("\nConfound categories:")
print(conf.columns.tolist())

# Main categories:
# - Motion parameters: trans_x, trans_y, trans_z, rot_x, rot_y, rot_z
# - Motion derivatives: trans_x_derivative1, etc.
# - Motion power: trans_x_power2, trans_x_derivative1_power2, etc.
# - Framewise displacement: framewise_displacement
# - DVARS: dvars, std_dvars, dvars_derivative1
# - Global signals: global_signal, csf, white_matter
# - aCompCor: a_comp_cor_00, a_comp_cor_01, ...
# - tCompCor: t_comp_cor_00, t_comp_cor_01, ...
# - Cosine drift: cosine00, cosine01, ...
# - Outlier detection: motion_outlier00, motion_outlier01, ...
```

### Select Confound Regressors

```python
# Common confound strategies

# Strategy 1: Minimal (motion only)
confounds_minimal = conf[['trans_x', 'trans_y', 'trans_z',
                           'rot_x', 'rot_y', 'rot_z']].values

# Strategy 2: Motion + derivatives
motion_cols = [c for c in conf.columns if 'trans_' in c or 'rot_' in c]
confounds_motion = conf[motion_cols].fillna(0).values

# Strategy 3: Motion + CompCor
acompcor_cols = [c for c in conf.columns if 'a_comp_cor_' in c][:5]  # First 5
confounds_compcor = conf[motion_cols + acompcor_cols].fillna(0).values

# Strategy 4: Motion + global signals
confounds_global = conf[motion_cols + ['global_signal', 'csf', 'white_matter']].fillna(0).values

# Strategy 5: Custom selection
selected_confounds = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'framewise_displacement',
    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02',
    'cosine00', 'cosine01', 'cosine02'
]
confounds_custom = conf[selected_confounds].fillna(0).values

print(f"Minimal: {confounds_minimal.shape[1]} regressors")
print(f"Motion+derivatives: {confounds_motion.shape[1]} regressors")
print(f"Motion+CompCor: {confounds_compcor.shape[1]} regressors")
print(f"Motion+Global: {confounds_global.shape[1]} regressors")
print(f"Custom: {confounds_custom.shape[1]} regressors")
```

### aCompCor and tCompCor

```python
# Anatomical CompCor (aCompCor): PCA in WM/CSF
# Temporal CompCor (tCompCor): PCA in high-variance voxels

# Load masks used for CompCor
wm_mask = nib.load('sub-01_desc-WM_mask.nii.gz')
csf_mask = nib.load('sub-01_desc-CSF_mask.nii.gz')
combined_mask = nib.load('sub-01_desc-WM+CSF_mask.nii.gz')

# Number of components to use (rule of thumb)
# aCompCor: 5-10 components
# tCompCor: 5-10 components

acompcor_cols = [c for c in conf.columns if 'a_comp_cor_' in c]
tcompcor_cols = [c for c in conf.columns if 't_comp_cor_' in c]

print(f"aCompCor components available: {len(acompcor_cols)}")
print(f"tCompCor components available: {len(tcompcor_cols)}")

# Check variance explained
# (fMRIPrep includes metadata about variance explained)
```

## Quality Metrics Extraction

### Aggregate Across Subjects

```python
# Extract key metrics from all subjects
import glob
import os

subjects = ['sub-01', 'sub-02', 'sub-03']  # Or load from file
task = 'rest'

metrics = []

for sub in subjects:
    conf_file = f'{sub}/func/{sub}_task-{task}_desc-confounds_timeseries.tsv'

    if not os.path.exists(conf_file):
        print(f"Missing: {conf_file}")
        continue

    conf = pd.read_csv(conf_file, sep='\t')

    # Extract summary metrics
    metrics.append({
        'subject_id': sub,
        'mean_fd': np.nanmean(conf['framewise_displacement']),
        'max_fd': np.nanmax(conf['framewise_displacement']),
        'mean_dvars': np.nanmean(conf['std_dvars']),
        'max_dvars': np.nanmax(conf['std_dvars']),
        'n_volumes': len(conf),
        'high_motion_vols': np.sum(conf['framewise_displacement'] > 0.5)
    })

# Create DataFrame
df_metrics = pd.DataFrame(metrics)

# Summary
print(df_metrics.describe())

# Save
df_metrics.to_csv('group_motion_qc.csv', index=False)
```

### Identify Outliers

```python
# Identify subjects with poor quality

# Define thresholds
thresholds = {
    'mean_fd': 0.5,
    'max_fd': 3.0,
    'mean_dvars': 1.5,
    'high_motion_pct': 20
}

df_metrics['high_motion_pct'] = (df_metrics['high_motion_vols'] /
                                   df_metrics['n_volumes'] * 100)

# Flag failures
df_metrics['qc_fail'] = (
    (df_metrics['mean_fd'] > thresholds['mean_fd']) |
    (df_metrics['max_fd'] > thresholds['max_fd']) |
    (df_metrics['mean_dvars'] > thresholds['mean_dvars']) |
    (df_metrics['high_motion_pct'] > thresholds['high_motion_pct'])
)

# Report
failed = df_metrics[df_metrics['qc_fail']]
print(f"Failed QC: {len(failed)}/{len(df_metrics)} subjects")
print(failed)

# Save exclusion list
failed['subject_id'].to_csv('exclude_high_motion.txt', index=False, header=False)
```

### Visualize Group Quality

```python
# Plot distribution of quality metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Mean FD
axes[0, 0].hist(df_metrics['mean_fd'], bins=20)
axes[0, 0].axvline(thresholds['mean_fd'], color='r', linestyle='--')
axes[0, 0].set_xlabel('Mean FD (mm)')
axes[0, 0].set_title('Mean Framewise Displacement')

# Max FD
axes[0, 1].hist(df_metrics['max_fd'], bins=20)
axes[0, 1].axvline(thresholds['max_fd'], color='r', linestyle='--')
axes[0, 1].set_xlabel('Max FD (mm)')
axes[0, 1].set_title('Maximum Framewise Displacement')

# Mean DVARS
axes[1, 0].hist(df_metrics['mean_dvars'], bins=20)
axes[1, 0].axvline(thresholds['mean_dvars'], color='r', linestyle='--')
axes[1, 0].set_xlabel('Mean DVARS')
axes[1, 0].set_title('Mean DVARS')

# High motion percentage
axes[1, 1].hist(df_metrics['high_motion_pct'], bins=20)
axes[1, 1].axvline(thresholds['high_motion_pct'], color='r', linestyle='--')
axes[1, 1].set_xlabel('High Motion (%)')
axes[1, 1].set_title('Percentage High Motion Volumes')

plt.tight_layout()
plt.savefig('group_motion_qc.png', dpi=150)
```

## Decision Making

### Inclusion/Exclusion Criteria

```python
# Develop data-driven thresholds

# Approach 1: Percentile-based
fd_95 = np.percentile(df_metrics['mean_fd'], 95)
dvars_95 = np.percentile(df_metrics['mean_dvars'], 95)

print(f"95th percentile FD: {fd_95:.3f} mm")
print(f"95th percentile DVARS: {dvars_95:.2f}")

# Approach 2: MAD (Median Absolute Deviation)
from scipy.stats import median_abs_deviation

mad_fd = median_abs_deviation(df_metrics['mean_fd'])
median_fd = np.median(df_metrics['mean_fd'])

threshold_fd_mad = median_fd + 3 * mad_fd
print(f"FD threshold (median + 3*MAD): {threshold_fd_mad:.3f} mm")

# Approach 3: Literature-based
# Common thresholds from literature:
# - Mean FD < 0.2mm (very strict)
# - Mean FD < 0.5mm (moderate)
# - Mean FD < 1.0mm (lenient)
# - High motion volumes < 20%
```

### Sensitivity Analysis

```python
# Test how results change with different QC thresholds

thresholds_to_test = [0.2, 0.3, 0.5, 0.7, 1.0]  # Mean FD thresholds

for thresh in thresholds_to_test:
    n_excluded = np.sum(df_metrics['mean_fd'] > thresh)
    n_retained = len(df_metrics) - n_excluded
    pct_retained = n_retained / len(df_metrics) * 100

    print(f"Threshold {thresh:.1f}mm: {n_retained}/{len(df_metrics)} retained ({pct_retained:.1f}%)")

# Run analysis with different thresholds to test robustness
```

## Integration with Analysis

### Quality Metrics as Covariates

```python
# Include motion as covariate in group analysis

# Load analysis data
analysis_df = pd.read_csv('group_analysis_data.csv')

# Merge with QC metrics
merged_df = analysis_df.merge(df_metrics, on='subject_id')

# Include motion in statistical model
from statsmodels.formula.api import ols

model = ols('brain_measure ~ group + age + sex + mean_fd', data=merged_df).fit()
print(model.summary())

# Check if motion is a significant predictor
# If yes, controlling for it is important
```

### Quality-Based Sample Selection

```python
# Create matched samples based on motion

controls = merged_df[merged_df['group'] == 'control']
patients = merged_df[merged_df['group'] == 'patient']

# Match on motion
from scipy.spatial.distance import cdist

# For each patient, find control with similar motion
matched_controls = []

for _, patient in patients.iterrows():
    patient_fd = patient['mean_fd']

    # Find closest control
    distances = np.abs(controls['mean_fd'] - patient_fd)
    closest_idx = distances.idxmin()

    matched_controls.append(controls.loc[closest_idx, 'subject_id'])

print(f"Matched {len(matched_controls)} control-patient pairs")
```

## Automated QC Pipelines

### Batch Extract Metrics

```bash
#!/bin/bash
# extract_qc_metrics.sh

output_dir="derivatives/fmriprep"
qc_dir="qc_metrics"

mkdir -p ${qc_dir}

# Extract metrics for all subjects
python << EOF
import pandas as pd
import glob
import os
import numpy as np

conf_files = glob.glob('${output_dir}/sub-*/func/*_desc-confounds_timeseries.tsv')

metrics = []
for f in conf_files:
    sub = os.path.basename(f).split('_')[0]
    task = os.path.basename(f).split('_')[1].replace('task-', '')

    conf = pd.read_csv(f, sep='\t')

    metrics.append({
        'subject_id': sub,
        'task': task,
        'mean_fd': np.nanmean(conf['framewise_displacement']),
        'max_fd': np.nanmax(conf['framewise_displacement']),
        'mean_dvars': np.nanmean(conf['std_dvars']),
        'n_volumes': len(conf)
    })

df = pd.DataFrame(metrics)
df.to_csv('${qc_dir}/all_subjects_qc.csv', index=False)
print(f"Extracted metrics for {len(df)} runs")
EOF
```

## Troubleshooting

**Problem:** HTML report won't open
**Solution:** Use modern browser, check file permissions, serve with http.server

**Problem:** Missing confounds file
**Solution:** Check fMRIPrep logs, ensure preprocessing completed successfully

**Problem:** NaN values in confounds
**Solution:** Normal for first volume (derivatives), handle with fillna(0)

**Problem:** Unexpected high motion
**Solution:** Review raw data, check scanner issues, consider participant feedback

**Problem:** Registration looks poor
**Solution:** Check anatomical quality, try different template, review brain extraction

## Best Practices

1. **Systematic Review:**
   - Review all subjects' reports
   - Use consistent criteria
   - Document decisions
   - Blind reviewers to group

2. **Threshold Selection:**
   - Use data-driven thresholds when possible
   - Test sensitivity to thresholds
   - Report all thresholds used
   - Justify choices with literature

3. **Documentation:**
   - Record QC criteria
   - Save scripts for reproducibility
   - Version control QC code
   - Share QC reports with data

4. **Quality Metrics:**
   - Always extract FD and DVARS
   - Consider as covariates
   - Report in papers
   - Include in supplementary materials

5. **Integration:**
   - Combine with MRIQC
   - Use VisualQC for manual review
   - Aggregate metrics across runs
   - Track quality over time

## Resources

- **fMRIPrep Docs:** https://fmriprep.readthedocs.io/
- **Confounds:** https://fmriprep.readthedocs.io/en/stable/outputs.html#confounds
- **GitHub:** https://github.com/nipreps/fmriprep
- **Forum:** https://neurostars.org/tag/fmriprep
- **Paper:** Esteban et al. (2019). Nature Methods

## Citation

```bibtex
@article{esteban2019fmriprep,
  title={fMRIPrep: a robust preprocessing pipeline for functional MRI},
  author={Esteban, Oscar and Markiewicz, Christopher J and Blair, Ross W and Moodie, Craig A and Isik, A Ilkay and Erramuzpe, Asier and Kent, James D and Goncalves, Mathias and DuPre, Elizabeth and Snyder, Madeleine and others},
  journal={Nature methods},
  volume={16},
  number={1},
  pages={111--116},
  year={2019},
  publisher={Nature Publishing Group}
}
```

## Related Tools

- **MRIQC:** Pre-processing automated QC
- **fMRIPrep:** Preprocessing pipeline
- **XCP-D:** Post-processing QC
- **VisualQC:** Manual quality control
- **Nilearn:** Load and plot confounds
- **C-PAC:** Alternative preprocessing with QC
