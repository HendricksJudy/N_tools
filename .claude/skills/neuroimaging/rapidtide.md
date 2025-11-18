# RapidTide: Rapid Time Delay Analysis for Physiological Noise Detection

## Overview

**RapidTide** is a suite of Python tools for detecting, characterizing, and removing physiological noise sources in fMRI data, with particular emphasis on systemic low-frequency oscillations (sLFOs) related to blood flow, cardiac, and respiratory cycles. Unlike traditional approaches requiring external physiological recordings (ECG, respiration), RapidTide can detect these signals directly from the BOLD data itself using cross-correlation and spectral analysis, enabling retrospective data cleaning and physiological signal characterization.

### Key Features

- **Data-Driven Detection**: Find physiological signals without external recordings
- **Time-Delay Mapping**: Voxel-wise delay maps of blood arrival times
- **Systemic Low-Frequency Oscillations**: Detect and remove sLFO contamination
- **Automatic Regressor Generation**: Create nuisance regressors from BOLD data
- **Cerebrovascular Assessment**: Map blood arrival timing patterns
- **Multiple Filtering Options**: Cardiac, respiratory, and LFO band separation
- **GLM-Based Denoising**: Remove detected physiological noise
- **Quality Metrics**: Significance testing and correlation strength maps
- **Integration Ready**: Works with fMRIPrep and other preprocessing pipelines

### Scientific Foundation

RapidTide detects systemic physiological oscillations by:

1. **Identifying a probe signal** (global or regional average) containing physiological noise
2. **Computing time-lagged correlations** between probe and each voxel
3. **Mapping delay times** where correlation is maximized (blood arrival time)
4. **Generating regressors** based on temporally shifted probe signals
5. **Removing noise** via GLM regression

This approach leverages the fact that physiological signals (cardiac pulsation, respiration, blood flow) propagate through the brain with characteristic delays, creating spatially structured temporal patterns that can be detected and removed.

### Primary Use Cases

1. **Resting-state fMRI**: Improved functional connectivity accuracy
2. **Retrospective denoising**: Clean data without physiological recordings
3. **Cerebrovascular assessment**: Blood arrival time mapping
4. **Multi-site studies**: Harmonize data with inconsistent physio recording
5. **Clinical imaging**: Enhance data quality without extra hardware
6. **Low-frequency artifact removal**: Clean global signal contamination

---

## Installation

### Using pip (Recommended)

```bash
# Install rapidtide
pip install rapidtide

# Verify installation
rapidtide --version
rapidtide --help
```

### Using conda

```bash
# Create environment with rapidtide
conda create -n rapidtide-env python=3.9
conda activate rapidtide-env
pip install rapidtide

# Verify installation
rapidtide --version
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/bbfrederick/rapidtide.git
cd rapidtide

# Install in development mode
pip install -e .

# Run tests
pytest rapidtide
```

### Dependencies

RapidTide requires:
- Python ≥ 3.7
- NumPy, SciPy
- nibabel (neuroimaging I/O)
- matplotlib (visualization)
- scikit-learn (signal processing)
- pyfftw (fast FFT, optional)

---

## Basic Workflow

### Command-Line Usage

```bash
# Basic rapidtide command
rapidtide \
  sub-01_task-rest_bold.nii.gz \
  rapidtide_output \
  --searchrange -5 15 \
  --filterband lfo \
  --passes 3

# This creates multiple outputs:
# - lagtimes.nii.gz (delay map)
# - lagstrengths.nii.gz (correlation strength)
# - lagsigma.nii.gz (significance map)
# - filtereddata.nii.gz (denoised data)
# - outputregressors.txt (nuisance regressors)
```

### Understanding Key Parameters

```python
import subprocess

# Define rapidtide parameters
params = {
    'input': 'sub-01_task-rest_bold.nii.gz',
    'output_basename': 'rapidtide_output',
    'searchrange': '-5 15',  # Search delays from -5 to +15 seconds
    'filterband': 'lfo',  # Low-frequency oscillations (0.01-0.15 Hz)
    'passes': '3',  # Number of refinement passes
    'nprocs': '4'  # Parallel processing
}

# Build command
cmd = [
    'rapidtide',
    params['input'],
    params['output_basename'],
    '--searchrange', *params['searchrange'].split(),
    '--filterband', params['filterband'],
    '--passes', params['passes'],
    '--nprocs', params['nprocs']
]

# Run rapidtide
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✓ RapidTide completed successfully")
else:
    print(f"✗ RapidTide failed: {result.stderr}")
```

### Output Files Overview

```python
from pathlib import Path
import nibabel as nib

def inspect_rapidtide_outputs(output_basename):
    """Inspect RapidTide output files"""

    outputs = {
        'lagtimes': 'Delay map (blood arrival times)',
        'lagstrengths': 'Correlation strength map',
        'lagsigma': 'Statistical significance of delays',
        'MTT': 'Mean transit time map',
        'filtereddata': 'Denoised fMRI data',
        'reference': 'Probe regressor timeseries',
        'outputregressors': 'Nuisance regressors (text)',
        'p_lt_0p050_mask': 'Significant voxels mask (p<0.05)'
    }

    print(f"RapidTide Outputs ({output_basename}):\n")

    for key, description in outputs.items():
        # Check for NIfTI file
        nifti_file = f"{output_basename}_{key}.nii.gz"
        if Path(nifti_file).exists():
            img = nib.load(nifti_file)
            print(f"✓ {nifti_file}")
            print(f"  Shape: {img.shape}, Description: {description}")

        # Check for text file
        txt_file = f"{output_basename}_{key}.txt"
        if Path(txt_file).exists():
            print(f"✓ {txt_file}: {description}")

    print()

# Inspect outputs
inspect_rapidtide_outputs('rapidtide_output')
```

---

## Time-Delay Mapping

### Interpreting Delay Maps

```python
from nilearn import plotting, image
import matplotlib.pyplot as plt
import numpy as np

# Load delay (lag time) map
lagtime_img = image.load_img('rapidtide_output_lagtimes.nii.gz')
lagstrength_img = image.load_img('rapidtide_output_lagstrengths.nii.gz')

# Visualize delay map
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Delay times (arterial = negative, venous = positive)
plotting.plot_stat_map(
    lagtime_img,
    title='Blood Arrival Time Delays',
    cmap='RdBu_r',
    vmin=-5,
    vmax=15,
    cut_coords=(0, 0, 0),
    axes=axes[0],
    colorbar=True
)

# Correlation strength (quality of delay estimate)
plotting.plot_stat_map(
    lagstrength_img,
    title='Delay Correlation Strength',
    cmap='hot',
    vmin=0,
    vmax=0.8,
    cut_coords=(0, 0, 0),
    axes=axes[1],
    colorbar=True
)

plt.tight_layout()
plt.savefig('rapidtide_delay_maps.png', dpi=150)

print("Delay map interpretation:")
print("  Negative delays (blue): Arterial/early arriving blood")
print("  Positive delays (red): Venous/late arriving blood")
print("  Strong correlation: Reliable delay estimation")
```

### Extracting Regional Delays

```python
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import pandas as pd

# Load atlas
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
atlas_img = atlas['maps']
atlas_labels = atlas['labels']

# Extract regional delays
masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=False
)

# Get mean delay per region
regional_delays = masker.fit_transform(lagtime_img)[0]

# Get mean correlation strength per region
regional_strength = masker.fit_transform(lagstrength_img)[0]

# Create summary table
delay_table = pd.DataFrame({
    'region': [label.decode() if isinstance(label, bytes) else label for label in atlas_labels],
    'mean_delay_sec': regional_delays,
    'correlation_strength': regional_strength
})

# Sort by delay (arterial to venous)
delay_table_sorted = delay_table.sort_values('mean_delay_sec')

print("Regions by Blood Arrival Time:")
print(delay_table_sorted.head(10))  # Earliest (arterial)
print("\n...")
print(delay_table_sorted.tail(10))  # Latest (venous)
```

### Physiological Interpretation

```python
# Classify voxels by hemodynamic timing
lagtime_data = lagtime_img.get_fdata()
lagstrength_data = lagstrength_img.get_fdata()

# Define thresholds
significant_mask = lagstrength_data > 0.3  # Good correlation

arterial_mask = (lagtime_data < -1) & significant_mask
normal_mask = (lagtime_data >= -1) & (lagtime_data <= 5) & significant_mask
venous_mask = (lagtime_data > 5) & significant_mask

print("Hemodynamic Timing Classification:")
print(f"  Arterial (< -1s): {arterial_mask.sum()} voxels ({100*arterial_mask.sum()/significant_mask.sum():.1f}%)")
print(f"  Normal (-1 to 5s): {normal_mask.sum()} voxels ({100*normal_mask.sum()/significant_mask.sum():.1f}%)")
print(f"  Venous (> 5s): {venous_mask.sum()} voxels ({100*venous_mask.sum()/significant_mask.sum():.1f}%)")

# Save classified maps
arterial_img = image.new_img_like(lagtime_img, arterial_mask.astype(float))
nib.save(arterial_img, 'arterial_voxels.nii.gz')

venous_img = image.new_img_like(lagtime_img, venous_mask.astype(float))
nib.save(venous_img, 'venous_voxels.nii.gz')
```

---

## Systemic Low-Frequency Oscillations (sLFOs)

### Understanding sLFOs

```python
# sLFOs are physiological oscillations in the 0.01-0.15 Hz range
# Sources: respiration (~0.2-0.3 Hz and harmonics), cardiac (~1 Hz and harmonics),
#          vasomotion, CO2 fluctuations, Mayer waves

# Run rapidtide with LFO filter
subprocess.run([
    'rapidtide',
    'sub-01_task-rest_bold.nii.gz',
    'rapidtide_lfo',
    '--searchrange', '-10', '20',
    '--filterband', 'lfo',  # 0.01-0.15 Hz
    '--passes', '3',
    '--nprocs', '4'
])

print("LFO band: 0.01-0.15 Hz")
print("Captures: respiratory variation, vasomotion, Mayer waves")
```

### Separating Frequency Bands

```python
# Cardiac band (0.8-1.5 Hz)
subprocess.run([
    'rapidtide',
    'sub-01_task-rest_bold.nii.gz',
    'rapidtide_cardiac',
    '--searchrange', '-2', '5',
    '--filterband', 'cardiac',  # 0.8-1.5 Hz
    '--passes', '2'
])

# Respiratory band (0.15-0.4 Hz)
subprocess.run([
    'rapidtide',
    'sub-01_task-rest_bold.nii.gz',
    'rapidtide_resp',
    '--searchrange', '-5', '10',
    '--filterband', 'resp',  # 0.15-0.4 Hz
    '--passes', '2'
])

# Very low frequency (0.001-0.01 Hz)
subprocess.run([
    'rapidtide',
    'sub-01_task-rest_bold.nii.gz',
    'rapidtide_vlf',
    '--searchrange', '-15', '30',
    '--filterband', 'vlf',  # 0.001-0.01 Hz
    '--passes', '3'
])

print("✓ Separated physiological frequency bands")
```

### Custom Frequency Bands

```python
# Define custom filter band
subprocess.run([
    'rapidtide',
    'sub-01_task-rest_bold.nii.gz',
    'rapidtide_custom',
    '--searchrange', '-5', '15',
    '--filterfreqs', '0.02', '0.10',  # Custom: 0.02-0.10 Hz
    '--passes', '3'
])

print("Custom filter: 0.02-0.10 Hz (narrow LFO band)")
```

---

## Regressor Generation and Denoising

### Extract Nuisance Regressors

```python
import numpy as np

# Load RapidTide regressors
regressors = np.loadtxt('rapidtide_output_outputregressors.txt')

print(f"Regressor shape: {regressors.shape}")
print(f"  Timepoints: {regressors.shape[0]}")
print(f"  Regressors: {regressors.shape[1]}")

# Regressors include:
# - Probe signal (global/regional average with physiological noise)
# - Time-shifted versions based on voxel-wise delays
# - Derivatives (optional)

# Visualize regressors
fig, axes = plt.subplots(regressors.shape[1], 1, figsize=(12, 8), sharex=True)

for i in range(regressors.shape[1]):
    axes[i].plot(regressors[:, i])
    axes[i].set_ylabel(f'Reg {i+1}')
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Volume')
plt.suptitle('RapidTide Nuisance Regressors')
plt.tight_layout()
plt.savefig('rapidtide_regressors.png', dpi=150)
```

### Apply GLM-Based Denoising

```python
from nilearn import image
from nilearn.maskers import NiftiMasker
from sklearn.linear_model import LinearRegression

# Load original data
fmri_img = image.load_img('sub-01_task-rest_bold.nii.gz')

# Load RapidTide regressors
regressors = np.loadtxt('rapidtide_output_outputregressors.txt')

# Mask and extract timeseries
masker = NiftiMasker(
    mask_strategy='epi',
    standardize=False,
    detrend=False
)

fmri_data = masker.fit_transform(fmri_img)

# Regress out nuisance signals
glm = LinearRegression()
glm.fit(regressors, fmri_data)

# Get residuals (denoised data)
predicted_noise = glm.predict(regressors)
denoised_data = fmri_data - predicted_noise

# Transform back to 4D image
denoised_img = masker.inverse_transform(denoised_data)

# Save denoised data
denoised_img.to_filename('sub-01_task-rest_denoised_rapidtide.nii.gz')

print("✓ Applied GLM denoising with RapidTide regressors")
```

### Compare Before and After

```python
from nilearn.connectome import ConnectivityMeasure

# Load atlas for connectivity
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)

# Extract timeseries before denoising
masker_roi = NiftiLabelsMasker(
    labels_img=atlas['maps'],
    standardize=True
)

timeseries_before = masker_roi.fit_transform(fmri_img)
timeseries_after = masker_roi.fit_transform(denoised_img)

# Compute connectivity
correlation_measure = ConnectivityMeasure(kind='correlation')
conn_before = correlation_measure.fit_transform([timeseries_before])[0]
conn_after = correlation_measure.fit_transform([timeseries_after])[0]

# Compare global signal contamination
np.fill_diagonal(conn_before, 0)
np.fill_diagonal(conn_after, 0)

mean_conn_before = np.abs(conn_before).mean()
mean_conn_after = np.abs(conn_after).mean()

print(f"Mean connectivity before: {mean_conn_before:.3f}")
print(f"Mean connectivity after: {mean_conn_after:.3f}")
print(f"Change: {(mean_conn_after - mean_conn_before) / mean_conn_before * 100:.1f}%")

# Plot connectivity matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(conn_before, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[0].set_title('Before RapidTide')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(conn_after, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[1].set_title('After RapidTide')
plt.colorbar(im2, ax=axes[1])

plt.savefig('connectivity_rapidtide_comparison.png', dpi=150)
```

---

## Integration with Preprocessing Pipelines

### Use with fMRIPrep Outputs

```python
from bids import BIDSLayout

# Load BIDS dataset
layout = BIDSLayout('/data/bids_dataset', derivatives=[
    '/data/bids_dataset/derivatives/fmriprep'
])

# Get fMRIPrep preprocessed data
subject = '01'
task = 'rest'

fmriprep_bold = layout.get(
    subject=subject,
    task=task,
    desc='preproc',
    space='MNI152NLin2009cAsym',
    suffix='bold',
    extension='nii.gz',
    return_type='filename'
)[0]

# Get brain mask
fmriprep_mask = layout.get(
    subject=subject,
    task=task,
    desc='brain',
    suffix='mask',
    extension='nii.gz',
    return_type='filename'
)[0]

# Run RapidTide on fMRIPrep data
subprocess.run([
    'rapidtide',
    fmriprep_bold,
    f'derivatives/rapidtide/sub-{subject}/rapidtide_{task}',
    '--searchrange', '-5', '15',
    '--filterband', 'lfo',
    '--passes', '3',
    '--spatialfilt', '2.0',  # Spatial smoothing (mm)
    '--globalmeaninclude', fmriprep_mask  # Use brain mask
])

print(f"✓ Processed fMRIPrep output for sub-{subject}")
```

### Combine with fMRIPrep Confounds

```python
import pandas as pd

# Load fMRIPrep confounds
confounds_file = layout.get(
    subject=subject,
    task=task,
    desc='confounds',
    extension='tsv',
    return_type='filename'
)[0]

confounds_df = pd.read_csv(confounds_file, sep='\t')

# Select motion confounds
motion_confounds = confounds_df[[
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z'
]].values

# Load RapidTide regressors
rapidtide_regressors = np.loadtxt(
    f'derivatives/rapidtide/sub-{subject}/rapidtide_{task}_outputregressors.txt'
)

# Combine confounds
combined_confounds = np.hstack([motion_confounds, rapidtide_regressors])

print(f"Motion confounds: {motion_confounds.shape[1]} regressors")
print(f"RapidTide confounds: {rapidtide_regressors.shape[1]} regressors")
print(f"Combined: {combined_confounds.shape[1]} total regressors")

# Apply combined denoising with nilearn
from nilearn.image import clean_img

cleaned_img = clean_img(
    fmriprep_bold,
    confounds=combined_confounds,
    standardize=True,
    detrend=True,
    high_pass=0.008,  # 0.008 Hz = 125s
    t_r=2.0
)

cleaned_img.to_filename(
    f'derivatives/rapidtide/sub-{subject}/sub-{subject}_task-{task}_denoised_combined.nii.gz'
)
```

---

## Advanced Applications

### Cerebrovascular Assessment

```python
# Run RapidTide with extended delay range for CVR assessment
subprocess.run([
    'rapidtide',
    'sub-01_task-rest_bold.nii.gz',
    'rapidtide_cvr',
    '--searchrange', '-20', '40',  # Extended range
    '--filterband', 'vlf',  # Very low frequencies
    '--passes', '4',
    '--pickleft'  # Keep negative correlation peaks
])

# Load delay map
lagtime_img = image.load_img('rapidtide_cvr_lagtimes.nii.gz')
lagstrength_img = image.load_img('rapidtide_cvr_lagstrengths.nii.gz')

# Compute mean transit time (MTT)
lagtime_data = lagtime_img.get_fdata()
lagstrength_data = lagstrength_img.get_fdata()

# Mask significant voxels
sig_mask = lagstrength_data > 0.3
mtt = lagtime_data[sig_mask].mean()
mtt_std = lagtime_data[sig_mask].std()

print(f"Mean Transit Time: {mtt:.2f} ± {mtt_std:.2f} seconds")

# Identify delayed regions (potential perfusion deficits)
delayed_mask = (lagtime_data > mtt + 2*mtt_std) & sig_mask
delayed_img = image.new_img_like(lagtime_img, delayed_mask.astype(float))

plotting.plot_roi(
    delayed_img,
    bg_img=fmri_img,
    title='Delayed Perfusion Regions',
    cmap='autumn'
)
plt.savefig('delayed_perfusion.png', dpi=150)
```

### Multi-Pass Refinement

```python
# RapidTide can iteratively refine delay estimates
# More passes = more refined, but diminishing returns after 3-4

pass_results = {}

for n_passes in [1, 2, 3, 4]:
    output_dir = f'rapidtide_pass{n_passes}'

    subprocess.run([
        'rapidtide',
        'sub-01_task-rest_bold.nii.gz',
        output_dir,
        '--searchrange', '-5', '15',
        '--filterband', 'lfo',
        '--passes', str(n_passes)
    ])

    # Load results
    lagstrength_img = image.load_img(f'{output_dir}_lagstrengths.nii.gz')
    mean_strength = lagstrength_img.get_fdata().mean()

    pass_results[n_passes] = mean_strength
    print(f"Pass {n_passes}: Mean correlation = {mean_strength:.3f}")

# Plot improvement
plt.figure(figsize=(8, 5))
plt.plot(list(pass_results.keys()), list(pass_results.values()), 'o-')
plt.xlabel('Number of Passes')
plt.ylabel('Mean Correlation Strength')
plt.title('RapidTide Refinement Across Passes')
plt.grid(True, alpha=0.3)
plt.savefig('rapidtide_passes_comparison.png', dpi=150)
```

### Region-Specific Probe Signal

```python
# Instead of global mean, use region-specific probe
# Useful for isolating specific vascular territories

# Create gray matter mask for probe
from nilearn.masking import compute_epi_mask

gm_mask = compute_epi_mask(fmri_img)
nib.save(gm_mask, 'gm_mask.nii.gz')

# Run with custom probe region
subprocess.run([
    'rapidtide',
    'sub-01_task-rest_bold.nii.gz',
    'rapidtide_gm_probe',
    '--searchrange', '-5', '15',
    '--filterband', 'lfo',
    '--globalmeaninclude', 'gm_mask.nii.gz',  # Use only GM for probe
    '--passes', '3'
])

print("✓ Used gray matter-specific probe signal")
```

---

## Quality Metrics and Validation

### Assess Statistical Significance

```python
# Load significance map
lagsigma_img = image.load_img('rapidtide_output_lagsigma.nii.gz')
lagsigma_data = lagsigma_img.get_fdata()

# Load p-value mask (p < 0.05)
if Path('rapidtide_output_p_lt_0p050_mask.nii.gz').exists():
    sig_mask_img = image.load_img('rapidtide_output_p_lt_0p050_mask.nii.gz')
    sig_mask = sig_mask_img.get_fdata() > 0

    print(f"Significantly correlated voxels: {sig_mask.sum()}")
    print(f"Percentage of brain: {100 * sig_mask.sum() / (lagsigma_data > 0).sum():.1f}%")

    # Visualize significance
    plotting.plot_stat_map(
        sig_mask_img,
        title='Significant Voxels (p < 0.05)',
        cmap='Reds',
        threshold=0.5
    )
    plt.savefig('rapidtide_significance.png', dpi=150)
```

### Correlation Strength Distribution

```python
# Analyze distribution of correlation strengths
lagstrength_data = lagstrength_img.get_fdata()

# Mask brain voxels
brain_mask = lagstrength_data > 0

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(lagstrength_data[brain_mask], bins=50, edgecolor='black')
plt.xlabel('Correlation Strength')
plt.ylabel('Number of Voxels')
plt.title('Distribution of Delay Correlations')
plt.axvline(0.3, color='red', linestyle='--', label='Threshold = 0.3')
plt.legend()

plt.subplot(1, 2, 2)
# Cumulative distribution
sorted_strengths = np.sort(lagstrength_data[brain_mask])
cumulative = np.arange(1, len(sorted_strengths) + 1) / len(sorted_strengths)
plt.plot(sorted_strengths, cumulative)
plt.xlabel('Correlation Strength')
plt.ylabel('Cumulative Proportion')
plt.title('Cumulative Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_distribution.png', dpi=150)

# Summary statistics
print(f"Median correlation: {np.median(lagstrength_data[brain_mask]):.3f}")
print(f"90th percentile: {np.percentile(lagstrength_data[brain_mask], 90):.3f}")
print(f"Voxels > 0.3: {(lagstrength_data[brain_mask] > 0.3).sum()}")
```

---

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# batch_rapidtide.sh

BIDS_DIR=/data/bids_dataset
DERIV_DIR=$BIDS_DIR/derivatives

subjects=$(ls $BIDS_DIR | grep "^sub-" | sed 's/sub-//')

for subject in $subjects; do
    echo "Processing sub-$subject..."

    input_file=$BIDS_DIR/sub-$subject/func/sub-${subject}_task-rest_bold.nii.gz
    output_base=$DERIV_DIR/rapidtide/sub-$subject/rapidtide

    if [ ! -f "$input_file" ]; then
        echo "  ✗ Input file not found: $input_file"
        continue
    fi

    mkdir -p $DERIV_DIR/rapidtide/sub-$subject

    rapidtide \
        $input_file \
        $output_base \
        --searchrange -5 15 \
        --filterband lfo \
        --passes 3 \
        --nprocs 4

    if [ $? -eq 0 ]; then
        echo "  ✓ sub-$subject completed"
    else
        echo "  ✗ sub-$subject failed"
    fi
done

echo "Batch processing complete"
```

### Python Batch Processing with Error Handling

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_subject_rapidtide(subject_id, bids_dir, deriv_dir):
    """Process single subject with RapidTide"""

    input_file = Path(bids_dir) / f'sub-{subject_id}' / 'func' / f'sub-{subject_id}_task-rest_bold.nii.gz'
    output_dir = Path(deriv_dir) / 'rapidtide' / f'sub-{subject_id}'
    output_base = output_dir / 'rapidtide'

    if not input_file.exists():
        logger.error(f"sub-{subject_id}: Input file not found")
        return {'subject': subject_id, 'status': 'failed', 'reason': 'input not found'}

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'rapidtide',
        str(input_file),
        str(output_base),
        '--searchrange', '-5', '15',
        '--filterband', 'lfo',
        '--passes', '3',
        '--nprocs', '2'
    ]

    logger.info(f"Processing sub-{subject_id}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ sub-{subject_id} completed")
        return {'subject': subject_id, 'status': 'success'}

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ sub-{subject_id} failed: {e.stderr}")
        return {'subject': subject_id, 'status': 'failed', 'reason': e.stderr}

# Batch process
subjects = ['01', '02', '03', '04', '05']
bids_dir = '/data/bids_dataset'
deriv_dir = '/data/bids_dataset/derivatives'

results = []
with ProcessPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(process_subject_rapidtide, subj, bids_dir, deriv_dir)
        for subj in subjects
    ]

    for future in futures:
        results.append(future.result())

# Summary
successful = sum(1 for r in results if r['status'] == 'success')
print(f"\nBatch Summary: {successful}/{len(subjects)} subjects successful")
```

---

## Comparison with Other Methods

### RapidTide vs Global Signal Regression

```python
# Compare RapidTide to simple GSR

# 1. Global Signal Regression
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(mask_strategy='epi', standardize=True)
fmri_data = masker.fit_transform(fmri_img)

# Global signal is mean timeseries
global_signal = fmri_data.mean(axis=1, keepdims=True)

# Regress out global signal
from sklearn.linear_model import LinearRegression
glm_gsr = LinearRegression()
glm_gsr.fit(global_signal, fmri_data)
denoised_gsr = fmri_data - glm_gsr.predict(global_signal)

denoised_gsr_img = masker.inverse_transform(denoised_gsr)

# 2. RapidTide denoising (already computed)
denoised_rapidtide_img = image.load_img('rapidtide_output_filtereddata.nii.gz')

# 3. Compare connectivity
timeseries_gsr = masker_roi.fit_transform(denoised_gsr_img)
timeseries_rapidtide = masker_roi.fit_transform(denoised_rapidtide_img)

conn_gsr = correlation_measure.fit_transform([timeseries_gsr])[0]
conn_rapidtide = correlation_measure.fit_transform([timeseries_rapidtide])[0]

# Compare
np.fill_diagonal(conn_gsr, 0)
np.fill_diagonal(conn_rapidtide, 0)

print("Comparison: GSR vs RapidTide")
print(f"  Mean connectivity (GSR): {np.abs(conn_gsr).mean():.3f}")
print(f"  Mean connectivity (RapidTide): {np.abs(conn_rapidtide).mean():.3f}")
print("\nAdvantage of RapidTide:")
print("  - Preserves spatial structure of physiological signals")
print("  - Accounts for time delays across voxels")
print("  - Less risk of removing neural signals")
```

### Complementary Use with tedana

```python
# RapidTide and tedana address different noise sources:
# - tedana: TE-independent (motion, scanner) via multi-echo
# - RapidTide: Physiological (cardiac, respiration, sLFO)

# Workflow: tedana first, then RapidTide

# 1. Run tedana on multi-echo data (already done)
# 2. Run RapidTide on tedana output

subprocess.run([
    'rapidtide',
    'tedana_output/desc-denoised_bold.nii.gz',  # tedana output
    'rapidtide_after_tedana',
    '--searchrange', '-5', '15',
    '--filterband', 'lfo',
    '--passes', '3'
])

print("✓ Applied RapidTide to tedana-denoised data")
print("This removes residual physiological noise after tedana")
```

---

## Troubleshooting

### Common Issues

**Issue: Low correlation strengths**

```python
# Check if correlation strengths are too low (<0.2 globally)
lagstrength_data = image.load_img('rapidtide_output_lagstrengths.nii.gz').get_fdata()

if lagstrength_data[lagstrength_data > 0].mean() < 0.2:
    print("⚠ Warning: Low correlation strengths")
    print("Possible causes:")
    print("  - Insufficient physiological noise in data")
    print("  - Wrong frequency band selected")
    print("  - Data already heavily denoised")
    print("Solutions:")
    print("  - Try different --filterband (lfo, resp, cardiac)")
    print("  - Increase --searchrange")
    print("  - Check data quality")
```

**Issue: Unrealistic delay values**

```python
# Check for unrealistic delays
lagtime_data = image.load_img('rapidtide_output_lagtimes.nii.gz').get_fdata()

if lagtime_data.max() > 30 or lagtime_data.min() < -30:
    print("⚠ Warning: Unrealistic delay values detected")
    print(f"  Range: {lagtime_data.min():.1f} to {lagtime_data.max():.1f} seconds")
    print("Solutions:")
    print("  - Adjust --searchrange to physiological values (-10 to 20)")
    print("  - Check TR and timing information")
```

**Issue: Memory errors**

```bash
# For large datasets, use less intensive processing
rapidtide \
  input.nii.gz \
  output \
  --searchrange -5 15 \
  --filterband lfo \
  --passes 1 \  # Reduce passes
  --nprocs 1    # Reduce parallelization
```

---

## Best Practices

### Recommended Workflow

1. **Preprocessing First**
   - Motion correction (fMRIPrep recommended)
   - Distortion correction
   - Do NOT apply spatial smoothing before RapidTide

2. **RapidTide Parameters**
   - Start with --filterband lfo for most applications
   - Use 3 passes for good refinement
   - --searchrange -5 to 15 for typical brain perfusion

3. **Quality Control**
   - Inspect delay maps visually
   - Check correlation strength distributions
   - Verify significance maps

4. **Denoising Application**
   - Use generated regressors in GLM
   - Combine with motion confounds
   - Consider minimal additional smoothing

5. **Validation**
   - Compare connectivity before/after
   - Check if biologically plausible delays
   - Validate against ground truth if available

### Parameter Selection Guide

```python
parameter_guide = {
    'Resting-state fMRI': {
        'filterband': 'lfo',
        'searchrange': '-5 15',
        'passes': 3
    },
    'Cerebrovascular assessment': {
        'filterband': 'vlf',
        'searchrange': '-20 40',
        'passes': 4
    },
    'Cardiac artifact removal': {
        'filterband': 'cardiac',
        'searchrange': '-2 5',
        'passes': 2
    },
    'Respiratory artifact': {
        'filterband': 'resp',
        'searchrange': '-5 10',
        'passes': 2
    }
}

for application, params in parameter_guide.items():
    print(f"{application}:")
    for param, value in params.items():
        print(f"  --{param} {value}")
    print()
```

---

## References

### Key Publications

1. Tong, Y., & Frederick, B. D. (2014). "Studying the spatial distribution of physiological effects on BOLD signals using ultrafast fMRI." *Frontiers in Human Neuroscience*, 8, 196.

2. Frederick, B. D., et al. (2016). "An introduction to, and review of, the use of resting state correlation analysis to study the brain's fluctuating activity." *Frontiers in Neuroscience*, 10, 345.

3. Erdoğan, S. B., et al. (2016). "Correcting for blood arrival time in global mean regression enhances functional connectivity analysis of resting state fMRI-BOLD signals." *Frontiers in Human Neuroscience*, 10, 311.

### Documentation and Resources

- **Documentation**: https://rapidtide.readthedocs.io/
- **GitHub**: https://github.com/bbfrederick/rapidtide
- **Tutorials**: https://rapidtide.readthedocs.io/en/latest/usage.html
- **Example Data**: Available in repository

### Related Tools

- **tedana**: Multi-echo denoising
- **PhysIO**: Model-based physiological correction
- **CONN**: Functional connectivity with denoising options
- **fMRIPrep**: Comprehensive preprocessing
- **Nilearn**: Functional connectivity analysis

---

## See Also

- **tedana.md**: Multi-echo fMRI denoising
- **physio.md**: Model-based physiological correction
- **fmriprep.md**: Preprocessing pipeline
- **nilearn.md**: Functional connectivity analysis
- **conn.md**: Connectivity toolbox
