# tedana: TE Dependent ANAlysis for Multi-Echo fMRI Denoising

## Overview

**tedana** (TE Dependent ANAlysis) is a Python library for denoising multi-echo functional MRI data using independent component analysis (ICA). Based on the ME-ICA (Multi-Echo Independent Component Analysis) framework, tedana leverages the different T2* decay rates of BOLD (neural) signals versus non-BOLD (artifact) signals to automatically classify and remove noise components without requiring manual inspection or external physiological recordings.

### Key Features

- **Automatic Denoising**: Classify BOLD vs non-BOLD components without manual intervention
- **Multi-Echo Optimization**: Optimal combination of echoes for maximum tSNR
- **T2* Mapping**: Generate quantitative T2* and S0 parameter maps
- **Component Classification**: TE-dependent (BOLD) vs TE-independent (noise) separation
- **BIDS Integration**: Compatible with BIDS datasets and fMRIPrep outputs
- **Comprehensive Reports**: HTML reports with component visualizations and metrics
- **Flexible Workflows**: Command-line and Python API interfaces
- **Multiple Algorithms**: Kundu, minimal, and manual component selection

### Scientific Foundation

Multi-echo fMRI acquires data at multiple echo times (TEs), allowing separation of signals based on T2* decay characteristics:

- **BOLD signals** (neural activity): Scale with TE (TE-dependent, high Kappa)
- **Non-BOLD artifacts** (motion, scanner noise): Independent of TE (TE-independent, high Rho)

tedana uses ICA to decompose the signal into components, then classifies each component based on how it scales with TE, automatically removing noise while preserving neural signals.

### Primary Use Cases

1. **Resting-state fMRI**: Improved functional connectivity detection
2. **Task fMRI**: Enhanced activation sensitivity
3. **High-field imaging**: Better artifact removal at 7T+
4. **Clinical populations**: Robust denoising with motion artifacts
5. **Pharmacological fMRI**: Detect subtle BOLD changes
6. **Precision mapping**: Presurgical planning and layer fMRI

---

## Installation

### Using pip (Recommended)

```bash
# Install tedana
pip install tedana

# Verify installation
tedana --version
```

### Using conda

```bash
# Create environment with tedana
conda create -n tedana-env python=3.9
conda activate tedana-env
pip install tedana

# Verify installation
tedana --version
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/ME-ICA/tedana.git
cd tedana

# Install in development mode
pip install -e .

# Run tests
pytest tedana
```

### Dependencies

tedana requires:
- Python ≥ 3.8
- NumPy, SciPy
- nibabel (neuroimaging file I/O)
- nilearn (neuroimaging analysis)
- scikit-learn (machine learning/ICA)
- pandas (data manipulation)
- matplotlib (visualization)

---

## Multi-Echo fMRI Data Requirements

### Acquisition Parameters

**Minimum Requirements:**
- At least 3 echo times (TEs)
- Typical TEs: 12ms, 28ms, 44ms (example for 3T)
- Same TR, volumes, and resolution for all echoes

**Recommended:**
- 4-5 echoes for optimal denoising
- First TE: ~10-15ms
- Last TE: ~50-60ms (at 3T)
- TE spacing: ~10-15ms

```python
# Example multi-echo parameters for 3T
echoes = {
    'n_echoes': 4,
    'TEs': [12.0, 26.0, 40.0, 54.0],  # milliseconds
    'TR': 2000.0,  # milliseconds
    'n_volumes': 200,
    'voxel_size': [2.5, 2.5, 2.5]  # mm
}

print(f"Acquisition: {echoes['n_echoes']} echoes")
print(f"Echo times: {echoes['TEs']} ms")
print(f"Total scan time: {echoes['TR'] * echoes['n_volumes'] / 1000 / 60:.1f} min")
```

### BIDS Format

Multi-echo data in BIDS format:

```
sub-01/
  func/
    sub-01_task-rest_echo-1_bold.nii.gz
    sub-01_task-rest_echo-2_bold.nii.gz
    sub-01_task-rest_echo-3_bold.nii.gz
    sub-01_task-rest_echo-4_bold.nii.gz
    sub-01_task-rest_bold.json  # Contains EchoTime for each run
```

JSON sidecar example:

```json
{
  "EchoTime": 0.012,
  "RepetitionTime": 2.0,
  "TaskName": "rest",
  "EchoNumber": 1
}
```

---

## Basic Workflow

### Command-Line Usage

```bash
# Basic tedana command
tedana \
  -d sub-01_task-rest_echo-1_bold.nii.gz \
     sub-01_task-rest_echo-2_bold.nii.gz \
     sub-01_task-rest_echo-3_bold.nii.gz \
  -e 12 28 44 \
  --out-dir tedana_output \
  --verbose

# This creates:
# - desc-optcom_bold.nii.gz (optimally combined)
# - desc-denoised_bold.nii.gz (denoised output)
# - desc-PCA_metrics.tsv (PCA component metrics)
# - desc-ICA_metrics.tsv (ICA component metrics)
# - tedana_report.html (comprehensive report)
```

### Python API

```python
from tedana import workflows
import os

# Define inputs
data_files = [
    'sub-01_task-rest_echo-1_bold.nii.gz',
    'sub-01_task-rest_echo-2_bold.nii.gz',
    'sub-01_task-rest_echo-3_bold.nii.gz',
    'sub-01_task-rest_echo-4_bold.nii.gz'
]

echo_times = [12.0, 26.0, 40.0, 54.0]  # milliseconds
out_dir = 'tedana_output'
os.makedirs(out_dir, exist_ok=True)

# Run tedana workflow
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir=out_dir,
    tedpca='kundu',  # PCA component selection
    tedort=True,  # Orthogonalize rejected components
    verbose=True
)

print(f"✓ Denoised data saved to {out_dir}/desc-denoised_bold.nii.gz")
```

### Output Files

```python
import os
from pathlib import Path

def list_tedana_outputs(output_dir):
    """List and describe tedana outputs"""

    output_dir = Path(output_dir)

    outputs = {
        'desc-optcom_bold.nii.gz': 'Optimally combined echoes',
        'desc-denoised_bold.nii.gz': 'Denoised timeseries (use for analysis)',
        'desc-PCA_mixing.tsv': 'PCA component timeseries',
        'desc-ICA_mixing.tsv': 'ICA component timeseries',
        'desc-PCA_metrics.tsv': 'PCA component metrics',
        'desc-ICA_metrics.tsv': 'ICA component metrics and classifications',
        'desc-tedana_metrics.json': 'Overall denoising metrics',
        'T2starmap.nii.gz': 'T2* parameter map',
        'S0map.nii.gz': 'S0 parameter map',
        'tedana_report.html': 'Comprehensive HTML report'
    }

    print("tedana Outputs:")
    for filename, description in outputs.items():
        filepath = output_dir / filename
        exists = "✓" if filepath.exists() else "✗"
        print(f"  {exists} {filename}: {description}")

# Check outputs
list_tedana_outputs('tedana_output')
```

---

## Echo Combination Methods

### Optimal Combination (OC)

The default method weights echoes by their T2* signal:

```python
from tedana import combine
import nibabel as nib

# Load multi-echo data
echo_data = [nib.load(f).get_fdata() for f in data_files]
echo_times = [12.0, 26.0, 40.0, 54.0]

# Optimal combination
combined_data = combine.make_optcom(
    data=echo_data,
    tes=echo_times,
    combmode='t2s',  # T2*-weighted combination
    mask=None  # Will create mask automatically
)

print(f"Combined data shape: {combined_data.shape}")
print(f"Input shape per echo: {echo_data[0].shape}")
```

### T2* and S0 Map Generation

```python
from tedana import decay
import numpy as np

# Fit T2* decay model
t2s_map, s0_map, t2s_limited, t2s_full = decay.fit_decay(
    data=np.array(echo_data),
    tes=echo_times,
    mask=None,
    adaptive_mask=True
)

print(f"T2* map range: {np.nanmin(t2s_map):.1f} - {np.nanmax(t2s_map):.1f} ms")
print(f"S0 map range: {np.nanmin(s0_map):.1f} - {np.nanmax(s0_map):.1f}")

# Save maps
t2s_img = nib.Nifti1Image(t2s_map, affine=nib.load(data_files[0]).affine)
nib.save(t2s_img, 'T2star_map.nii.gz')

s0_img = nib.Nifti1Image(s0_map, affine=nib.load(data_files[0]).affine)
nib.save(s0_img, 'S0_map.nii.gz')
```

### Visualize T2* Maps

```python
from nilearn import plotting
import matplotlib.pyplot as plt

# Plot T2* map
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plotting.plot_stat_map(
    'T2star_map.nii.gz',
    title='T2* Map',
    cut_coords=(0, 0, 0),
    vmax=60,
    cmap='hot',
    axes=axes[0]
)

plotting.plot_stat_map(
    'S0_map.nii.gz',
    title='S0 Map',
    cut_coords=(0, 0, 0),
    cmap='gray',
    axes=axes[1]
)

plotting.plot_epi(
    'tedana_output/desc-optcom_bold.nii.gz',
    title='Optimally Combined',
    cut_coords=(0, 0, 0),
    axes=axes[2]
)

plt.tight_layout()
plt.savefig('tedana_maps.png', dpi=150)
```

---

## Component Classification

### TE-Dependent vs TE-Independent

tedana classifies components based on two key metrics:

- **Kappa**: TE-dependence (higher = more BOLD-like)
- **Rho**: TE-independence (higher = more artifact-like)

```python
import pandas as pd

# Load component metrics
comp_table = pd.read_csv('tedana_output/desc-ICA_metrics.tsv', sep='\t')

# View key metrics
metrics_to_show = ['Component', 'kappa', 'rho', 'variance explained', 'classification']
print(comp_table[metrics_to_show].head(10))

# Summary by classification
classification_summary = comp_table['classification'].value_counts()
print("\nComponent Classification Summary:")
print(classification_summary)

# Accepted (BOLD) components
accepted = comp_table[comp_table['classification'] == 'accepted']
print(f"\nAccepted components: {len(accepted)}")
print(f"Total variance explained: {accepted['variance explained'].sum():.1f}%")

# Rejected (noise) components
rejected = comp_table[comp_table['classification'] == 'rejected']
print(f"\nRejected components: {len(rejected)}")
print(f"Noise variance removed: {rejected['variance explained'].sum():.1f}%")
```

### Kundu Decision Tree

The default classification algorithm:

```python
# Run tedana with Kundu decision tree (default)
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_kundu',
    tedpca='kundu',  # Kundu PCA selection
    tree='kundu',  # Kundu decision tree
    verbose=True
)
```

### Minimal Classification

More conservative, accepts fewer components:

```python
# Run tedana with minimal tree
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_minimal',
    tedpca='kundu',
    tree='minimal',  # More conservative
    verbose=True
)
```

### Manual Classification

Override automatic classification:

```python
# First run tedana to get component metrics
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_manual',
    tedpca='kundu',
    tree='kundu',
    verbose=True
)

# Load metrics and manually adjust
comp_table = pd.read_csv('tedana_manual/desc-ICA_metrics.tsv', sep='\t')

# Manually set component 3 as rejected (if misclassified)
comp_table.loc[comp_table['Component'] == 3, 'classification'] = 'rejected'
comp_table.loc[comp_table['Component'] == 3, 'classification_tags'] = 'manual override'

# Save modified metrics
comp_table.to_csv('tedana_manual/desc-ICA_metrics_edited.tsv', sep='\t', index=False)

# Re-run denoising with manual classifications
# (requires using lower-level functions)
```

---

## Quality Reports

### HTML Report Interpretation

```python
import webbrowser

# Open HTML report
report_path = 'tedana_output/tedana_report.html'
webbrowser.open(f'file://{os.path.abspath(report_path)}')

print(f"Opening report: {report_path}")
print("\nKey sections in report:")
print("1. Optimal combination overview")
print("2. PCA component selection")
print("3. ICA decomposition")
print("4. Component classification metrics")
print("5. Component timeseries and spatial maps")
print("6. Denoising summary statistics")
```

### Component Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Load mixing matrix (component timeseries)
mixing = pd.read_csv('tedana_output/desc-ICA_mixing.tsv', sep='\t')
comp_table = pd.read_csv('tedana_output/desc-ICA_metrics.tsv', sep='\t')

# Plot accepted vs rejected component timeseries
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Accepted components
accepted_comps = comp_table[comp_table['classification'] == 'accepted']['Component'].values
for comp in accepted_comps[:5]:  # Plot first 5
    axes[0].plot(mixing.iloc[:, comp], alpha=0.7, label=f'IC {comp}')
axes[0].set_title('Accepted Components (BOLD)')
axes[0].set_xlabel('Volume')
axes[0].set_ylabel('Amplitude')
axes[0].legend(loc='upper right', fontsize=8)

# Rejected components
rejected_comps = comp_table[comp_table['classification'] == 'rejected']['Component'].values
for comp in rejected_comps[:5]:  # Plot first 5
    axes[1].plot(mixing.iloc[:, comp], alpha=0.7, label=f'IC {comp}')
axes[1].set_title('Rejected Components (Noise)')
axes[1].set_xlabel('Volume')
axes[1].set_ylabel('Amplitude')
axes[1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('component_timeseries.png', dpi=150)
```

### Kappa vs Rho Plot

```python
# Scatter plot of Kappa vs Rho
fig, ax = plt.subplots(figsize=(10, 8))

accepted = comp_table[comp_table['classification'] == 'accepted']
rejected = comp_table[comp_table['classification'] == 'rejected']

ax.scatter(accepted['kappa'], accepted['rho'],
           c='green', s=100, alpha=0.6, label='Accepted (BOLD)')
ax.scatter(rejected['kappa'], rejected['rho'],
           c='red', s=100, alpha=0.6, label='Rejected (Noise)')

# Add component numbers
for _, row in comp_table.iterrows():
    ax.annotate(str(row['Component']),
                (row['kappa'], row['rho']),
                fontsize=8, alpha=0.7)

ax.set_xlabel('Kappa (TE-dependence)', fontsize=12)
ax.set_ylabel('Rho (TE-independence)', fontsize=12)
ax.set_title('Component Classification: Kappa vs Rho', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('kappa_vs_rho.png', dpi=150)
```

---

## Integration with fMRIPrep

### Using fMRIPrep Multi-Echo Outputs

```python
from pathlib import Path
from bids import BIDSLayout

# Load BIDS dataset with derivatives
layout = BIDSLayout(
    '/data/bids_dataset',
    derivatives='/data/bids_dataset/derivatives/fmriprep'
)

# Get fMRIPrep preprocessed multi-echo data
subject = '01'
session = '01'
task = 'rest'

echo_files = layout.get(
    subject=subject,
    session=session,
    task=task,
    desc='preproc',
    suffix='bold',
    extension='nii.gz',
    return_type='filename'
)

# Get echo times from metadata
echo_times = []
for echo_file in echo_files:
    metadata = layout.get_metadata(echo_file)
    echo_times.append(metadata['EchoTime'] * 1000)  # Convert to ms

print(f"Found {len(echo_files)} echoes")
print(f"Echo times: {echo_times} ms")

# Run tedana on fMRIPrep outputs
workflows.tedana_workflow(
    data=echo_files,
    tes=echo_times,
    out_dir=f'derivatives/tedana/sub-{subject}/ses-{session}',
    mask='auto',  # Use fMRIPrep mask
    verbose=True
)
```

### Combining with fMRIPrep Confounds

```python
import pandas as pd
from nilearn import image

# Load fMRIPrep confounds
confounds_file = layout.get(
    subject=subject,
    session=session,
    task=task,
    desc='confounds',
    extension='tsv',
    return_type='filename'
)[0]

confounds_df = pd.read_csv(confounds_file, sep='\t')

# Select confounds (motion, CSF, WM)
selected_confounds = confounds_df[[
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'csf', 'white_matter'
]]

# Load tedana denoised data
denoised_img = image.load_img(
    f'derivatives/tedana/sub-{subject}/ses-{session}/desc-denoised_bold.nii.gz'
)

# Apply confound regression with nilearn
from nilearn.image import clean_img

cleaned_img = clean_img(
    denoised_img,
    confounds=selected_confounds.values,
    standardize=True,
    detrend=True,
    high_pass=0.01,
    t_r=2.0
)

# Save final cleaned data
cleaned_img.to_filename(
    f'derivatives/tedana/sub-{subject}/ses-{session}/desc-denoised_confounds_bold.nii.gz'
)

print("✓ Applied confound regression to tedana output")
```

---

## Advanced Options

### Custom PCA Component Selection

```python
# Manual PCA dimensionality
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_pca_manual',
    tedpca=20,  # Keep 20 PCA components (integer)
    verbose=True
)

# AIC-based selection (Akaike Information Criterion)
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_pca_aic',
    tedpca='aic',
    verbose=True
)

# KIC-based selection (Kullback-Leibler Information Criterion)
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_pca_kic',
    tedpca='kic',
    verbose=True
)
```

### Low vs High Motion Datasets

```python
# High-motion dataset (more aggressive denoising)
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_highmotion',
    tedpca='kundu',
    tedort=True,  # Orthogonalize rejected components (recommended)
    gscontrol='gsr',  # Global signal regression
    verbose=True
)

# Low-motion dataset (preserve more variance)
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_lowmotion',
    tedpca='kundu_stabilize',  # More stable PCA
    tedort=False,  # Don't orthogonalize
    gscontrol=None,  # No GSR
    verbose=True
)
```

### Memory Optimization

```python
# For large datasets, process in chunks
workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_lowmem',
    tedpca='kundu',
    low_mem=True,  # Low memory mode
    verbose=True
)
```

---

## Quality Control

### tSNR Improvement Assessment

```python
from nilearn import image, masking
import numpy as np

def compute_tsnr(img):
    """Compute temporal SNR"""
    data = img.get_fdata()
    mean_signal = np.mean(data, axis=-1)
    std_signal = np.std(data, axis=-1)
    tsnr = mean_signal / (std_signal + 1e-10)
    return tsnr

# Load optimal combination (before denoising)
optcom_img = image.load_img('tedana_output/desc-optcom_bold.nii.gz')
tsnr_before = compute_tsnr(optcom_img)

# Load denoised data
denoised_img = image.load_img('tedana_output/desc-denoised_bold.nii.gz')
tsnr_after = compute_tsnr(denoised_img)

# Compare tSNR
mask = image.load_img('tedana_output/desc-optcom_bold.nii.gz').get_fdata().mean(axis=-1) > 100
tsnr_improvement = (tsnr_after[mask].mean() - tsnr_before[mask].mean()) / tsnr_before[mask].mean() * 100

print(f"Mean tSNR before: {tsnr_before[mask].mean():.2f}")
print(f"Mean tSNR after: {tsnr_after[mask].mean():.2f}")
print(f"Improvement: {tsnr_improvement:.1f}%")

# Visualize tSNR improvement
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plotting.plot_stat_map(
    image.new_img_like(optcom_img, tsnr_before),
    title='tSNR Before Denoising',
    vmax=100,
    axes=axes[0]
)

plotting.plot_stat_map(
    image.new_img_like(denoised_img, tsnr_after),
    title='tSNR After Denoising',
    vmax=100,
    axes=axes[1]
)

plotting.plot_stat_map(
    image.new_img_like(denoised_img, tsnr_after - tsnr_before),
    title='tSNR Improvement',
    cmap='RdBu_r',
    axes=axes[2]
)

plt.savefig('tsnr_comparison.png', dpi=150)
```

### Connectivity Improvement

```python
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# Load atlas
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
atlas_img = atlas['maps']

# Extract timeseries before denoising
masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True
)

timeseries_before = masker.fit_transform(optcom_img)

# Extract timeseries after denoising
timeseries_after = masker.fit_transform(denoised_img)

# Compute connectivity
correlation_measure = ConnectivityMeasure(kind='correlation')
conn_before = correlation_measure.fit_transform([timeseries_before])[0]
conn_after = correlation_measure.fit_transform([timeseries_after])[0]

# Compare edge strength
np.fill_diagonal(conn_before, 0)
np.fill_diagonal(conn_after, 0)

print(f"Mean connectivity before: {np.abs(conn_before).mean():.3f}")
print(f"Mean connectivity after: {np.abs(conn_after).mean():.3f}")

# Plot connectivity matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(conn_before, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
axes[0].set_title('Before tedana')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(conn_after, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
axes[1].set_title('After tedana')
plt.colorbar(im2, ax=axes[1])

plt.savefig('connectivity_comparison.png', dpi=150)
```

---

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# batch_tedana.sh

BIDS_DIR=/data/bids_dataset
DERIV_DIR=$BIDS_DIR/derivatives/tedana

subjects=$(ls $BIDS_DIR | grep "^sub-" | sed 's/sub-//')

for subject in $subjects; do
    echo "Processing sub-$subject..."

    # Find multi-echo files
    echo1=$BIDS_DIR/sub-$subject/func/sub-${subject}_task-rest_echo-1_bold.nii.gz
    echo2=$BIDS_DIR/sub-$subject/func/sub-${subject}_task-rest_echo-2_bold.nii.gz
    echo3=$BIDS_DIR/sub-$subject/func/sub-${subject}_task-rest_echo-3_bold.nii.gz
    echo4=$BIDS_DIR/sub-$subject/func/sub-${subject}_task-rest_echo-4_bold.nii.gz

    # Run tedana
    tedana \
        -d $echo1 $echo2 $echo3 $echo4 \
        -e 12 26 40 54 \
        --out-dir $DERIV_DIR/sub-$subject \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✓ sub-$subject completed"
    else
        echo "✗ sub-$subject failed"
    fi
done
```

### Python Batch Processing

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_subject_tedana(subject_id, bids_dir, deriv_dir):
    """Process single subject with tedana"""

    func_dir = Path(bids_dir) / f'sub-{subject_id}' / 'func'

    # Find multi-echo files
    echo_files = sorted(func_dir.glob(f'sub-{subject_id}_task-rest_echo-*_bold.nii.gz'))

    if len(echo_files) < 3:
        logger.error(f"sub-{subject_id}: Need at least 3 echoes, found {len(echo_files)}")
        return {'subject': subject_id, 'status': 'failed', 'reason': 'insufficient echoes'}

    # Get echo times from JSON sidecars
    echo_times = []
    for echo_file in echo_files:
        json_file = echo_file.with_suffix('.json')
        with open(json_file) as f:
            metadata = json.load(f)
            echo_times.append(metadata['EchoTime'] * 1000)

    output_dir = Path(deriv_dir) / 'tedana' / f'sub-{subject_id}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing sub-{subject_id} with {len(echo_files)} echoes")

    try:
        workflows.tedana_workflow(
            data=[str(f) for f in echo_files],
            tes=echo_times,
            out_dir=str(output_dir),
            verbose=False
        )
        logger.info(f"✓ sub-{subject_id} completed")
        return {'subject': subject_id, 'status': 'success'}

    except Exception as e:
        logger.error(f"✗ sub-{subject_id} failed: {str(e)}")
        return {'subject': subject_id, 'status': 'failed', 'reason': str(e)}

# Batch process subjects
bids_dir = '/data/bids_dataset'
deriv_dir = '/data/bids_dataset/derivatives'

subjects = ['01', '02', '03', '04', '05']

results = []
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(process_subject_tedana, subj, bids_dir, deriv_dir)
        for subj in subjects
    ]

    for future in futures:
        results.append(future.result())

# Summary
successful = sum(1 for r in results if r['status'] == 'success')
print(f"\nBatch Summary: {successful}/{len(subjects)} subjects completed successfully")
```

---

## Statistical Analysis with Denoised Data

### GLM with Denoised Timeseries

```python
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
import pandas as pd

# Load denoised data
denoised_img = image.load_img('tedana_output/desc-denoised_bold.nii.gz')

# Load events (task fMRI)
events = pd.read_csv('sub-01_task-faces_events.tsv', sep='\t')

# Create GLM model
fmri_glm = FirstLevelModel(
    t_r=2.0,
    noise_model='ar1',
    standardize=False,
    hrf_model='spm',
    drift_model='cosine',
    high_pass=1./128
)

# Fit model
fmri_glm = fmri_glm.fit(denoised_img, events=events)

# Compute contrast
z_map = fmri_glm.compute_contrast('faces - baseline', output_type='z_score')

# Display results
plotting.plot_stat_map(
    z_map,
    threshold=3.1,
    title='Faces > Baseline (tedana denoised)',
    cut_coords=(0, 0, 0)
)
plt.savefig('task_activation_denoised.png', dpi=150)
```

### Compare Denoised vs Original

```python
# Fit GLM with optimal combination (before denoising)
optcom_img = image.load_img('tedana_output/desc-optcom_bold.nii.gz')

fmri_glm_before = FirstLevelModel(
    t_r=2.0,
    noise_model='ar1',
    standardize=False,
    hrf_model='spm'
)

fmri_glm_before = fmri_glm_before.fit(optcom_img, events=events)
z_map_before = fmri_glm_before.compute_contrast('faces - baseline', output_type='z_score')

# Fit GLM with denoised data
fmri_glm_after = FirstLevelModel(
    t_r=2.0,
    noise_model='ar1',
    standardize=False,
    hrf_model='spm'
)

fmri_glm_after = fmri_glm_after.fit(denoised_img, events=events)
z_map_after = fmri_glm_after.compute_contrast('faces - baseline', output_type='z_score')

# Compare activation extent
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plotting.plot_stat_map(z_map_before, threshold=3.1, axes=axes[0], title='Before tedana')
plotting.plot_stat_map(z_map_after, threshold=3.1, axes=axes[1], title='After tedana')

plt.savefig('activation_comparison.png', dpi=150)

# Count significant voxels
sig_before = (z_map_before.get_fdata() > 3.1).sum()
sig_after = (z_map_after.get_fdata() > 3.1).sum()

print(f"Significant voxels before: {sig_before}")
print(f"Significant voxels after: {sig_after}")
print(f"Change: {(sig_after - sig_before) / sig_before * 100:.1f}%")
```

---

## Troubleshooting

### Common Issues

**Issue: Not enough echoes**

```python
# Check number of echoes
import nibabel as nib

echo_files = ['echo-1.nii.gz', 'echo-2.nii.gz']  # Only 2 echoes!

if len(echo_files) < 3:
    print(f"⚠ Warning: Only {len(echo_files)} echoes found")
    print("tedana requires at least 3 echoes for optimal performance")
    print("Recommendation: Acquire at least 3-4 echoes")
```

**Issue: Suboptimal echo times**

```python
# Check TE spacing
echo_times = [10.0, 15.0, 60.0]  # Poor spacing!

te_diffs = np.diff(echo_times)
print(f"TE spacing: {te_diffs}")

if np.std(te_diffs) > 5:
    print("⚠ Warning: Uneven TE spacing")
    print("Recommendation: Use evenly spaced TEs (e.g., 12, 26, 40, 54 ms)")
```

**Issue: Memory errors**

```bash
# Use low-memory mode
tedana \
  -d echo-*.nii.gz \
  -e 12 26 40 54 \
  --out-dir tedana_output \
  --low-mem
```

**Issue: Too many/few components rejected**

```python
# Check classification summary
comp_table = pd.read_csv('tedana_output/desc-ICA_metrics.tsv', sep='\t')

accepted_variance = comp_table[comp_table['classification'] == 'accepted']['variance explained'].sum()
rejected_variance = comp_table[comp_table['classification'] == 'rejected']['variance explained'].sum()

print(f"Accepted variance: {accepted_variance:.1f}%")
print(f"Rejected variance: {rejected_variance:.1f}%")

# If too much rejected (>50%), try minimal tree
if rejected_variance > 50:
    print("⚠ Warning: >50% variance rejected")
    print("Consider using --tree minimal for more conservative denoising")
```

### Validation Checks

```python
def validate_tedana_outputs(output_dir):
    """Validate tedana outputs"""

    output_dir = Path(output_dir)

    checks = {
        'denoised_exists': (output_dir / 'desc-denoised_bold.nii.gz').exists(),
        'optcom_exists': (output_dir / 'desc-optcom_bold.nii.gz').exists(),
        'metrics_exists': (output_dir / 'desc-ICA_metrics.tsv').exists(),
        'report_exists': (output_dir / 'tedana_report.html').exists()
    }

    # Check component classification
    if checks['metrics_exists']:
        comp_table = pd.read_csv(output_dir / 'desc-ICA_metrics.tsv', sep='\t')
        n_accepted = (comp_table['classification'] == 'accepted').sum()
        n_rejected = (comp_table['classification'] == 'rejected').sum()

        checks['components_classified'] = n_accepted + n_rejected == len(comp_table)
        checks['reasonable_acceptance'] = 5 <= n_accepted <= 50

    # Report results
    all_passed = all(checks.values())

    print("tedana Output Validation:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")

    return all_passed

# Validate
validate_tedana_outputs('tedana_output')
```

---

## Best Practices

### Recommended Workflow

1. **Acquisition Planning**
   - Use 3-5 echoes
   - Even TE spacing (~10-15ms)
   - First TE: 10-15ms, Last TE: 50-60ms (at 3T)
   - Maintain good SNR for all echoes

2. **Preprocessing**
   - Run fMRIPrep first (motion correction, distortion correction)
   - Apply tedana to preprocessed echoes
   - Do NOT apply smoothing before tedana

3. **tedana Execution**
   - Start with default settings (Kundu tree)
   - Review HTML report carefully
   - Check kappa/rho scatter plot
   - Validate tSNR improvement

4. **Quality Control**
   - Inspect component classifications
   - Check for reasonable acceptance rate (10-30 components)
   - Verify tSNR improvement (>20%)
   - Compare connectivity or activation with/without denoising

5. **Statistical Analysis**
   - Use desc-denoised_bold.nii.gz for analysis
   - Can still apply minimal confound regression (motion)
   - Avoid aggressive additional denoising (already done)

### Performance Tips

```python
# Optimize tedana performance
optimal_settings = {
    'low_mem': True,  # For large datasets
    'tedpca': 'kundu',  # Good default
    'tedort': True,  # Recommended for most cases
    'gscontrol': None,  # Usually not needed after tedana
    'verbose': True  # For monitoring
}

workflows.tedana_workflow(
    data=data_files,
    tes=echo_times,
    out_dir='tedana_optimized',
    **optimal_settings
)
```

---

## References

### Key Publications

1. Kundu, P., et al. (2012). "Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI." *NeuroImage*, 60(3), 1759-1770.

2. Kundu, P., et al. (2013). "Integrated strategy for improving functional connectivity mapping using multiecho fMRI." *PNAS*, 110(40), 16187-16192.

3. DuPre, E., et al. (2021). "TE-dependent analysis of multi-echo fMRI with tedana." *Journal of Open Source Software*, 6(66), 3669.

### Documentation and Resources

- **Documentation**: https://tedana.readthedocs.io/
- **GitHub**: https://github.com/ME-ICA/tedana
- **ME-ICA Website**: https://me-ica.readthedocs.io/
- **Tutorials**: https://tedana.readthedocs.io/en/stable/usage.html
- **Example Data**: OpenNeuro datasets with multi-echo fMRI

### Related Tools

- **fMRIPrep**: Preprocessing before tedana
- **Nilearn**: Statistical analysis and connectivity
- **AFNI**: Alternative multi-echo processing (afni_proc.py)
- **RapidTide**: Complementary physiological denoising
- **CONN**: Functional connectivity with multi-echo support

---

## See Also

- **fmriprep.md**: Preprocessing multi-echo fMRI
- **nilearn.md**: Functional connectivity analysis
- **rapidtide.md**: Physiological noise detection
- **physio.md**: Model-based physiological correction
- **conn.md**: Functional connectivity toolbox

## Citation

```bibtex
@article{dupre2021tedana,
  title={TE-dependent analysis of multi-echo fMRI data: {tedana}},
  author={DuPre, Elizabeth and Salo, Taylor and Markiewicz, Christopher and others},
  journal={Journal of Neuroscience Methods},
  volume={350},
  pages={109017},
  year={2021},
  doi={10.1016/j.jneumeth.2021.109017}
}
```
