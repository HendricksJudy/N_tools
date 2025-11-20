# SynthSeg

## Overview

SynthSeg is a robust segmentation tool that uses domain randomization to segment brain MRI scans of any contrast (T1, T2, FLAIR, CT, etc.) without retraining. Developed at MGH/MIT, SynthSeg handles scans with artifacts, pathology, and varying quality by training on synthetic images with extreme augmentation, making it ideal for heterogeneous datasets, legacy data, and clinical scans.

**Website:** https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg
**Platform:** Python/TensorFlow
**Language:** Python
**License:** FreeSurfer License
**Integration:** Part of FreeSurfer 7.2+

## Key Features

- Works on **any contrast** (T1, T2, FLAIR, PD, CT, etc.)
- No retraining needed for different contrasts
- Robust to artifacts, motion, noise, pathology
- Automatic quality control scores
- Fast processing (~1-2 minutes per scan)
- FreeSurfer integration
- Handles clinical scans and legacy data
- Parcellation with 32 or 99 labels
- Volume estimation
- Posterior probability maps
- Uncertainty quantification
- Topology correction
- CPU and GPU support

## Installation

### Via FreeSurfer (Easiest)

```bash
# SynthSeg is included in FreeSurfer 7.2+
# Download FreeSurfer from: https://surfer.nmr.mgh.harvard.edu/

# Source FreeSurfer environment
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Verify SynthSeg is available
which mri_synthseg
```

### Via pip

```bash
# Install SynthSeg Python package
pip install SynthSeg

# Verify installation
python -c "import SynthSeg; print(SynthSeg.__version__)"
```

### From Source

```bash
# Clone repository
git clone https://github.com/BBillot/SynthSeg.git
cd SynthSeg

# Install dependencies
pip install -r requirements.txt

# Install SynthSeg
pip install .

# Download pre-trained models (automatic on first run)
```

### Docker

```bash
# Pull FreeSurfer Docker image (includes SynthSeg)
docker pull freesurfer/freesurfer:7.4.1

# Run with GPU support
docker run --gpus all -it \
  -v /path/to/data:/data \
  freesurfer/freesurfer:7.4.1
```

## Basic Usage

### Command-Line (via FreeSurfer)

```bash
# Basic segmentation
mri_synthseg --i input_scan.nii.gz \
             --o output_segmentation.nii.gz

# With volumes and QC
mri_synthseg --i input_scan.nii.gz \
             --o output_segmentation.nii.gz \
             --vol output_volumes.csv \
             --qc output_qc_scores.csv

# With posterior probabilities
mri_synthseg --i input_scan.nii.gz \
             --o output_segmentation.nii.gz \
             --post output_posteriors.nii.gz
```

### Python API

```python
from SynthSeg.predict import predict

# Basic segmentation
predict(
    path_images='input_scan.nii.gz',
    path_segmentations='output_segmentation.nii.gz'
)

# With all outputs
predict(
    path_images='input_scan.nii.gz',
    path_segmentations='output_segmentation.nii.gz',
    path_volumes='output_volumes.csv',
    path_qc_scores='output_qc_scores.csv',
    path_posteriors='output_posteriors.nii.gz'
)
```

## Multi-Contrast Support

### T1-Weighted

```bash
# Standard T1 scan
mri_synthseg --i T1_scan.nii.gz \
             --o T1_segmentation.nii.gz \
             --vol T1_volumes.csv
```

### T2-Weighted

```bash
# T2 scan (no retraining needed!)
mri_synthseg --i T2_scan.nii.gz \
             --o T2_segmentation.nii.gz \
             --vol T2_volumes.csv
```

### FLAIR

```bash
# FLAIR scan
mri_synthseg --i FLAIR_scan.nii.gz \
             --o FLAIR_segmentation.nii.gz \
             --vol FLAIR_volumes.csv
```

### CT Scans

```bash
# Even works on CT!
mri_synthseg --i CT_scan.nii.gz \
             --o CT_segmentation.nii.gz \
             --vol CT_volumes.csv \
             --ct  # Use CT mode
```

### Multiple Contrasts

```python
# Process multiple contrasts in batch
contrasts = {
    'T1': 'subject_T1.nii.gz',
    'T2': 'subject_T2.nii.gz',
    'FLAIR': 'subject_FLAIR.nii.gz'
}

for contrast_name, input_file in contrasts.items():
    predict(
        path_images=input_file,
        path_segmentations=f'seg_{contrast_name}.nii.gz',
        path_volumes=f'vol_{contrast_name}.csv'
    )
```

## Parcellation Options

### Standard 32-Class Parcellation (Default)

```bash
# 32 anatomical structures
mri_synthseg --i scan.nii.gz \
             --o segmentation.nii.gz \
             --parc

# Labels include:
# - Left/Right cerebral white matter
# - Left/Right cortex
# - Subcortical structures (thalamus, caudate, putamen, etc.)
# - Cerebellum
# - Brainstem
```

### Cortical Parcellation (99 Labels)

```bash
# Detailed cortical parcellation
mri_synthseg --i scan.nii.gz \
             --o segmentation.nii.gz \
             --parc \
             --robust  # More robust parcellation

# 99 labels including:
# - 32 standard structures
# - Detailed cortical regions (FreeSurfer DKT atlas)
# - Left/Right hippocampus subfields
```

### Label List

```python
# Get label names
from SynthSeg.labels_table import get_labels

# Standard labels
labels = get_labels()
print(labels)

# Example labels:
# 0: Background
# 2: Left Cerebral White Matter
# 3: Left Cerebral Cortex
# 4: Left Lateral Ventricle
# 10: Left Thalamus
# 11: Left Caudate
# 12: Left Putamen
# ...
```

## Quality Control

### Automatic QC Scores

```bash
# Generate QC scores
mri_synthseg --i scan.nii.gz \
             --o segmentation.nii.gz \
             --qc qc_scores.csv

# QC scores indicate:
# - Overall quality (0-1, higher is better)
# - Per-structure quality
# - Confidence in segmentation
```

### Interpret QC Scores

```python
import pandas as pd

# Load QC scores
qc = pd.read_csv('qc_scores.csv')
print(qc)

# QC interpretation:
# > 0.9: Excellent quality
# 0.7-0.9: Good quality
# 0.5-0.7: Acceptable, review recommended
# < 0.5: Poor quality, manual check needed

# Flag low-quality scans
low_quality = qc[qc['qc_score'] < 0.7]
if not low_quality.empty:
    print("Warning: Low quality scans detected")
    print(low_quality)
```

### Visual QC

```bash
# Overlay segmentation on original scan
freeview -v scan.nii.gz \
         segmentation.nii.gz:colormap=lut:opacity=0.4

# Check alignment and accuracy
# Look for:
# - Proper tissue classification
# - No major misalignments
# - Reasonable structure boundaries
```

## Robust Mode

### Handle Artifacts and Pathology

```bash
# Robust mode for challenging scans
mri_synthseg --i noisy_scan.nii.gz \
             --o segmentation.nii.gz \
             --robust

# Robust mode:
# - More resistant to noise and artifacts
# - Better handling of pathology
# - Slightly slower (~2x)
# - Use for clinical scans or data with lesions
```

### Topology Correction

```bash
# Ensure topologically correct segmentation
mri_synthseg --i scan.nii.gz \
             --o segmentation.nii.gz \
             --topology

# Topology correction:
# - Fixes holes and disconnections
# - Ensures anatomically plausible segmentation
# - Important for surface reconstruction
```

## Fast Mode

```bash
# Fast processing for quick results
mri_synthseg --i scan.nii.gz \
             --o segmentation.nii.gz \
             --fast

# Fast mode:
# - ~30 seconds per scan
# - Slightly lower quality
# - Good for initial QC or large-scale screening
```

## Volume Estimation

### Extract Volumes

```bash
# Generate volume measurements
mri_synthseg --i scan.nii.gz \
             --o segmentation.nii.gz \
             --vol volumes.csv

# volumes.csv contains:
# - Structure names
# - Volumes in mm³
# - Normalized volumes (% of ICV)
```

### Analyze Volumes

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load volumes
volumes = pd.read_csv('volumes.csv', index_col=0)

# Display key structures
structures = ['Left-Hippocampus', 'Right-Hippocampus',
              'Left-Thalamus', 'Right-Thalamus']

for structure in structures:
    if structure in volumes.index:
        vol = volumes.loc[structure, 'Volume_mm3']
        print(f"{structure}: {vol:.1f} mm³")

# Plot volumes
volumes['Volume_mm3'].plot(kind='bar', figsize=(12, 6))
plt.ylabel('Volume (mm³)')
plt.title('Brain Structure Volumes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('volumes_plot.png', dpi=300)
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# Batch process all subjects

INPUT_DIR="/data/scans"
OUTPUT_DIR="/data/derivatives/synthseg"

mkdir -p $OUTPUT_DIR

# Process all NIfTI files
for scan in ${INPUT_DIR}/*.nii.gz; do
    # Extract filename
    basename=$(basename $scan .nii.gz)

    echo "Processing: $basename"

    # Run SynthSeg
    mri_synthseg \
        --i $scan \
        --o ${OUTPUT_DIR}/${basename}_seg.nii.gz \
        --vol ${OUTPUT_DIR}/${basename}_vol.csv \
        --qc ${OUTPUT_DIR}/${basename}_qc.csv \
        --post ${OUTPUT_DIR}/${basename}_post.nii.gz \
        --robust

    echo "Completed: $basename"
done

echo "Batch processing complete!"
```

### Parallel Processing

```python
from SynthSeg.predict import predict
from multiprocessing import Pool
import os

def process_subject(scan_path):
    """Process single subject."""
    basename = os.path.basename(scan_path).replace('.nii.gz', '')
    output_dir = '/data/derivatives/synthseg'

    predict(
        path_images=scan_path,
        path_segmentations=f'{output_dir}/{basename}_seg.nii.gz',
        path_volumes=f'{output_dir}/{basename}_vol.csv',
        path_qc_scores=f'{output_dir}/{basename}_qc.csv'
    )

    return basename

# Get all scans
scans = [f'/data/scans/{f}' for f in os.listdir('/data/scans')
         if f.endswith('.nii.gz')]

# Process in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_subject, scans)

print(f"Processed {len(results)} subjects")
```

## Uncertainty Quantification

### Posterior Probabilities

```bash
# Save probability maps for each structure
mri_synthseg --i scan.nii.gz \
             --o segmentation.nii.gz \
             --post posteriors.nii.gz

# posteriors.nii.gz is 4D:
# - Dim 4: probability for each label
# - Values: 0-1 (probability)
```

### Analyze Uncertainty

```python
import nibabel as nib
import numpy as np

# Load posteriors
post = nib.load('posteriors.nii.gz')
post_data = post.get_fdata()

# Calculate uncertainty (entropy)
epsilon = 1e-10
entropy = -np.sum(post_data * np.log(post_data + epsilon), axis=3)

# High entropy = high uncertainty
uncertainty_threshold = 1.0
uncertain_voxels = entropy > uncertainty_threshold

print(f"Uncertain voxels: {np.sum(uncertain_voxels)} / {np.prod(entropy.shape)}")

# Save uncertainty map
uncertainty_img = nib.Nifti1Image(entropy, post.affine)
nib.save(uncertainty_img, 'uncertainty_map.nii.gz')
```

## Integration with FreeSurfer

### SynthSeg in FreeSurfer Pipeline

```bash
# Use SynthSeg as part of FreeSurfer analysis
recon-all -subject sub01 \
          -i T1_scan.nii.gz \
          -all \
          -synthseg

# Or run SynthSeg separately and use results
mri_synthseg --i T1_scan.nii.gz \
             --o aseg.auto.mgz \
             --parc

# Copy to FreeSurfer subject directory
cp aseg.auto.mgz $SUBJECTS_DIR/sub01/mri/
```

### Convert SynthSeg Output

```bash
# Convert to FreeSurfer format (.mgz)
mri_convert segmentation.nii.gz segmentation.mgz

# Use in FreeSurfer stats
mri_segstats --seg segmentation.mgz \
             --sum stats.txt \
             --i original_scan.mgz
```

## Clinical Applications

### Lesion/Pathology Handling

```bash
# Segment scans with lesions or pathology
mri_synthseg --i scan_with_lesion.nii.gz \
             --o segmentation.nii.gz \
             --robust \
             --topology

# SynthSeg trained to handle:
# - Tumors
# - Stroke lesions
# - Multiple sclerosis plaques
# - Atrophy
# - Surgical resections
```

### Multi-Site Studies

```python
# Process heterogeneous multi-site data
sites = ['site1', 'site2', 'site3']

all_volumes = []

for site in sites:
    scans = os.listdir(f'/data/{site}')

    for scan in scans:
        # SynthSeg handles scanner/protocol differences
        predict(
            path_images=f'/data/{site}/{scan}',
            path_segmentations=f'/output/{site}_{scan}_seg.nii.gz',
            path_volumes=f'/output/{site}_{scan}_vol.csv'
        )

        # Collect volumes
        vol = pd.read_csv(f'/output/{site}_{scan}_vol.csv')
        vol['site'] = site
        vol['subject'] = scan
        all_volumes.append(vol)

# Combine all sites
combined = pd.concat(all_volumes, ignore_index=True)
combined.to_csv('multi_site_volumes.csv', index=False)
```

### Longitudinal Studies

```bash
# Process longitudinal scans
for timepoint in baseline followup_1year followup_2year; do
    mri_synthseg \
        --i sub01_${timepoint}.nii.gz \
        --o sub01_${timepoint}_seg.nii.gz \
        --vol sub01_${timepoint}_vol.csv \
        --robust
done

# Compare volumes across timepoints
```

## GPU Acceleration

### Enable GPU

```bash
# SynthSeg automatically uses GPU if available
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU (if needed)
export CUDA_VISIBLE_DEVICES=""
mri_synthseg --i scan.nii.gz --o seg.nii.gz --cpu
```

### Batch Processing on GPU

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

from SynthSeg.predict import predict

# Process multiple scans efficiently
scans = ['scan1.nii.gz', 'scan2.nii.gz', 'scan3.nii.gz']

for scan in scans:
    predict(
        path_images=scan,
        path_segmentations=scan.replace('.nii.gz', '_seg.nii.gz'),
        gpu=True  # Use GPU
    )
```

## Advanced Options

### Custom Segmentation Parameters

```python
from SynthSeg.predict import predict

predict(
    path_images='scan.nii.gz',
    path_segmentations='segmentation.nii.gz',
    robust=True,  # Robust mode
    fast=False,  # Not fast mode
    v1=False,  # Use latest model (v2)
    crop=True,  # Crop to brain (faster)
    topology_classes=None,  # Auto topology
    n_neutral_labels=1  # Background label
)
```

### Resample to Different Resolution

```bash
# Specify output resolution
mri_synthseg --i scan.nii.gz \
             --o segmentation.nii.gz \
             --resample output_resolution.nii.gz

# Use scan at different resolution as target
# Segmentation will be resampled to match
```

## Comparison with Other Methods

### SynthSeg vs. FreeSurfer

```python
import nibabel as nib
import numpy as np

def compare_segmentations(synthseg_file, freesurfer_file):
    """Compare SynthSeg and FreeSurfer segmentations."""

    synthseg = nib.load(synthseg_file).get_fdata()
    freesurfer = nib.load(freesurfer_file).get_fdata()

    # Overall agreement (Dice)
    intersection = np.sum((synthseg > 0) & (freesurfer > 0))
    dice = 2 * intersection / (np.sum(synthseg > 0) + np.sum(freesurfer > 0))

    print(f"Overall Dice: {dice:.3f}")

    # Processing time comparison
    print("SynthSeg: ~1-2 minutes")
    print("FreeSurfer: ~5-24 hours")

    # Advantages:
    # SynthSeg: Much faster, robust to contrasts/artifacts
    # FreeSurfer: More detailed cortical parcellation, surfaces

# Run comparison
compare_segmentations('synthseg_aseg.mgz', 'freesurfer_aseg.mgz')
```

## Integration with Claude Code

When helping users with SynthSeg:

1. **Check Installation:**
   ```bash
   which mri_synthseg
   python -c "import SynthSeg"
   ```

2. **Verify Input:**
   ```bash
   # Check image format
   mri_info scan.nii.gz
   ```

3. **Common Workflow:**
   - Basic: Input → SynthSeg → Segmentation + Volumes
   - Clinical: Use --robust for pathology
   - QC: Always check QC scores

4. **Best Practices:**
   - Use --robust for clinical/noisy data
   - Check QC scores (< 0.7 = review)
   - Works on any contrast (no preprocessing)
   - Fast enough for interactive QC

## Troubleshooting

**Problem:** Segmentation looks incorrect
**Solution:** Check input quality, use --robust mode, verify QC scores

**Problem:** "Out of memory" error
**Solution:** Use --fast mode, process on GPU, or close other applications

**Problem:** Very low QC scores
**Solution:** Check input scan quality, try --robust, inspect for artifacts/pathology

**Problem:** Different results from FreeSurfer
**Solution:** Expected - different algorithms; SynthSeg optimized for robustness

**Problem:** TensorFlow warnings
**Solution:** Usually harmless; update TensorFlow if persistent

## Best Practices

1. **Always check QC scores** before using results
2. **Use --robust mode** for clinical data or scans with pathology
3. **Works on any contrast** - no need to convert or preprocess
4. **Batch process efficiently** - SynthSeg is fast
5. **Save posteriors** for uncertainty quantification
6. **Visual QC critical** for important analyses
7. **Compare with ground truth** on subset when available
8. **Document input contrast** (T1, T2, FLAIR, etc.)
9. **Use topology correction** if generating surfaces
10. **GPU recommended** but CPU works fine (slightly slower)

## Resources

- **FreeSurfer Wiki:** https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg
- **GitHub:** https://github.com/BBillot/SynthSeg
- **Paper:** https://arxiv.org/abs/2107.09559
- **FreeSurfer:** https://surfer.nmr.mgh.harvard.edu/
- **Forum:** https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/

## Citation

```bibtex
@article{billot2023synthseg,
  title={SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining},
  author={Billot, Benjamin and Greve, Douglas N and Puonti, Oula and Thielscher, Axel and Van Leemput, Koen and Fischl, Bruce and Dalca, Adrian V and Iglesias, Juan Eugenio},
  journal={Medical Image Analysis},
  volume={86},
  pages={102789},
  year={2023},
  publisher={Elsevier}
}
```

## Related Tools

- **FreeSurfer:** Comprehensive cortical analysis suite
- **FastSurfer:** Fast deep learning parcellation
- **nnU-Net:** General segmentation framework
- **MONAI:** Medical imaging DL framework
- **CAT12:** VBM preprocessing
- **SAMSEG:** FreeSurfer's Bayesian segmentation
- **FreeView:** Visualization (part of FreeSurfer)
