# TractSeg - Automated White Matter Tract Segmentation

## Overview

TractSeg is a deep learning-based tool for fast and accurate segmentation of white matter fiber tracts directly from diffusion MRI data. Using convolutional neural networks trained on extensive manual annotations, TractSeg can automatically identify and segment 72 major white matter tracts, tract orientation maps (TOMs), and tract endings in minutes without requiring manual intervention or tractography. It's particularly valuable for large-scale studies, clinical applications, and ensuring reproducible tract delineation.

**Website:** https://github.com/MIC-DKFZ/TractSeg
**Platform:** Linux/macOS/Windows
**Language:** Python (PyTorch)
**License:** Apache-2.0

## Key Features

- Automated segmentation of 72 white matter tracts
- Deep learning-based (no manual seeding required)
- Tract Orientation Maps (TOMs) for fiber tracking
- Bundle-specific tractography
- Tract ending segmentation
- Uncertainty estimation
- Works with standard DTI or multi-shell data
- Fast processing (1-2 minutes per subject)
- Pre-trained models (no training required)
- Integration with existing pipelines
- MNI space or native space output
- Robust to varying data quality

## Installation

### Via pip (Recommended)

```bash
# Install TractSeg
pip install TractSeg

# Install PyTorch (required dependency)
# CPU version
pip install torch torchvision

# GPU version (CUDA 11.3)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113

# Verify installation
TractSeg --help
```

### Via Docker

```bash
# Pull Docker image
docker pull wasserth/tractseg_container:master

# Run TractSeg
docker run -v /path/to/data:/data wasserth/tractseg_container:master \
  TractSeg -i /data/peaks.nii.gz -o /data/tractseg_output
```

### Download Pre-trained Weights

```bash
# Download models (required on first run)
# Automatically downloaded to ~/.tractseg/

# Or manually download
mkdir -p ~/.tractseg
cd ~/.tractseg
wget https://zenodo.org/record/6481434/files/pretrained_weights_tractSeg_72_bundles.zip
unzip pretrained_weights_tractSeg_72_bundles.zip
```

## Prerequisites

### Input Data Requirements

```bash
# TractSeg requires fiber orientation distributions (FODs/peaks)
# Options:
# 1. CSD peaks from MRtrix3 (recommended)
# 2. DTI peaks from FSL
# 3. Multi-shell peaks from DIPY

# Preprocessing needed:
# - Motion/eddy correction
# - Registration to MNI space (or use native space mode)
# - Peak estimation
```

### Generate Peaks from MRtrix3

```bash
# Complete preprocessing pipeline
# 1. Preprocess DWI
dwidenoise dwi.mif dwi_denoised.mif
mrdegibbs dwi_denoised.mif dwi_degibbs.mif
dwifslpreproc dwi_degibbs.mif dwi_preproc.mif -rpe_none -pe_dir AP

# 2. Estimate response function
dwi2response dhollander dwi_preproc.mif wm.txt gm.txt csf.txt

# 3. Estimate FODs
dwi2fod msmt_csd dwi_preproc.mif wm.txt wm_fod.mif gm.txt gm_fod.mif csf.txt csf_fod.mif

# 4. Extract peaks for TractSeg
sh2peaks wm_fod.mif peaks.nii.gz -num 3

# 5. Register to MNI space
# TractSeg expects MNI152 space by default
mrregister wm_fod.mif $MRTRIX/share/mrtrix3/labelconvert/mni152.mif \
  -nl_warp warp.mif

# Apply warp to peaks
mrtransform peaks.nii.gz peaks_MNI.nii.gz -warp warp.mif
```

### Generate Peaks from FSL

```bash
# Using FSL bedpostx output
# 1. Run bedpostx
bedpostx dwi_folder

# 2. Extract peaks
# TractSeg can use bedpostx outputs directly
# Or convert to peak format

# 3. Register to MNI
flirt -in nodif_brain.nii.gz \
  -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz \
  -omat to_MNI.mat

# Apply to peaks
flirt -in peaks.nii.gz -ref MNI152_T1_1mm.nii.gz \
  -applyxfm -init to_MNI.mat -out peaks_MNI.nii.gz
```

## Basic Usage

### Tract Segmentation

```bash
# Segment all 72 tracts
TractSeg -i peaks.nii.gz -o tractseg_output

# Output: tractseg_output/bundle_segmentations/
# Contains binary masks for each tract:
# - AF_left.nii.gz (Arcuate Fasciculus)
# - CST_left.nii.gz (Corticospinal Tract)
# - ILF_left.nii.gz (Inferior Longitudinal Fasciculus)
# ... and 69 more tracts

# Specify output directory
TractSeg -i peaks.nii.gz -o /results/sub-01/tractseg
```

### Tract Orientation Maps (TOMs)

```bash
# Generate TOMs for bundle-specific tracking
TractSeg -i peaks.nii.gz -o tractseg_output --output_type TOM

# Output: tractseg_output/TOM/
# Peak orientations within each tract
# Used for improved tractography
```

### Tract Endings

```bash
# Segment tract start and end regions
TractSeg -i peaks.nii.gz -o tractseg_output --output_type endings_segmentation

# Output: tractseg_output/endings_segmentations/
# Binary masks for tract beginnings and endings
# Useful for seeding tractography
```

### All Outputs

```bash
# Generate all outputs (segmentations, TOMs, endings)
TractSeg -i peaks.nii.gz -o tractseg_output --output_type tract_segmentation
TractSeg -i peaks.nii.gz -o tractseg_output --output_type TOM
TractSeg -i peaks.nii.gz -o tractseg_output --output_type endings_segmentation
TractSeg -i peaks.nii.gz -o tractseg_output --output_type TOM_trackings

# Or use script to run all
for output_type in tract_segmentation TOM endings_segmentation; do
    TractSeg -i peaks.nii.gz -o tractseg_output --output_type ${output_type}
done
```

## Bundle-Specific Tractography

### Using TractSeg TOMs for Tracking

```bash
# 1. Generate TOMs
TractSeg -i peaks.nii.gz -o tractseg_output --output_type TOM

# 2. Generate tract segmentations (for filtering)
TractSeg -i peaks.nii.gz -o tractseg_output --output_type tract_segmentation

# 3. Perform bundle-specific tracking
Tracking -i peaks.nii.gz \
  -o tractseg_output \
  --tracking_format tck

# Output: tractseg_output/TOM_trackings/*.tck
# Streamlines for each tract
```

### MRtrix3 Integration

```bash
# Use TractSeg masks with MRtrix3 tckgen

# 1. Get TractSeg segmentation
TractSeg -i peaks.nii.gz -o tractseg_output

# 2. Track within specific tract
tckgen wm_fod.mif CST_left.tck \
  -seed_image tractseg_output/bundle_segmentations/CST_left.nii.gz \
  -include tractseg_output/bundle_segmentations/CST_left.nii.gz \
  -algorithm iFOD2 \
  -select 5000

# 3. Or use endings as seed/target
tckgen wm_fod.mif CST_left_seeded.tck \
  -seed_image tractseg_output/endings_segmentations/CST_left_b.nii.gz \
  -include tractseg_output/endings_segmentations/CST_left_e.nii.gz
```

## Extract Tract Metrics

### Calculate Mean FA/MD per Tract

```bash
# Extract metrics within tract masks
TractSeg -i peaks.nii.gz -o tractseg_output

# Calculate mean FA for each tract
python << EOF
import nibabel as nib
import numpy as np
import os

# Load FA map
fa = nib.load('FA.nii.gz').get_fdata()

# Tract list
tracts = ['AF_left', 'AF_right', 'CST_left', 'CST_right', 'ILF_left', 'ILF_right']

results = {}
for tract in tracts:
    mask_file = f'tractseg_output/bundle_segmentations/{tract}.nii.gz'
    if os.path.exists(mask_file):
        mask = nib.load(mask_file).get_fdata()
        mean_fa = np.mean(fa[mask > 0])
        results[tract] = mean_fa
        print(f'{tract}: FA = {mean_fa:.3f}')

# Save results
import pandas as pd
df = pd.DataFrame.from_dict(results, orient='index', columns=['Mean_FA'])
df.to_csv('tract_metrics.csv')
EOF
```

### Along-Tract Statistics

```bash
# Sample metrics along tract length
# Using TractSeg trackings

python << EOF
import nibabel as nib
from dipy.tracking.streamline import set_number_of_points
from dipy.io.streamline import load_tractogram
import numpy as np

# Load tract
tractogram = load_tractogram('tractseg_output/TOM_trackings/CST_left.tck', 'same')
streamlines = tractogram.streamlines

# Resample to 100 points
streamlines_resampled = set_number_of_points(streamlines, 100)

# Load FA map
fa = nib.load('FA.nii.gz')
fa_data = fa.get_fdata()

# Sample FA along streamlines
fa_profile = np.zeros((len(streamlines_resampled), 100))
for i, sl in enumerate(streamlines_resampled):
    for j, point in enumerate(sl):
        # Convert to voxel coordinates
        vox = np.round(point).astype(int)
        if all(vox >= 0) and all(vox < fa_data.shape):
            fa_profile[i, j] = fa_data[vox[0], vox[1], vox[2]]

# Mean profile across streamlines
mean_profile = np.mean(fa_profile, axis=0)
std_profile = np.std(fa_profile, axis=0)

# Save
np.savetxt('CST_left_FA_profile.txt', mean_profile)
EOF
```

## Advanced Options

### Different Models

```bash
# Use specific pre-trained model
TractSeg -i peaks.nii.gz -o tractseg_output --nr_cpus 4

# Available models:
# - 72 bundles (default)
# - 20 bundles (faster, major tracts only)
# - Custom trained models

# Single bundle mode (faster if only specific tracts needed)
TractSeg -i peaks.nii.gz -o tractseg_output --single_orientation
```

### Uncertainty Estimation

```bash
# Generate uncertainty maps
TractSeg -i peaks.nii.gz -o tractseg_output --get_probabilities

# Output: Probability maps instead of binary masks
# Values 0-1 indicating confidence
```

### GPU Acceleration

```bash
# Use GPU for faster processing
TractSeg -i peaks.nii.gz -o tractseg_output --use_gpu

# Specify GPU
CUDA_VISIBLE_DEVICES=0 TractSeg -i peaks.nii.gz -o tractseg_output --use_gpu
```

### Custom Peak Number

```bash
# If peaks have different number of orientations
TractSeg -i peaks.nii.gz -o tractseg_output --nr_of_peaks 3

# Default expects 3 peaks per voxel (9 values: 3 peaks × 3 coordinates)
```

## Working in Native Space

### Skip MNI Registration

```bash
# Process in native DWI space (no MNI registration needed)
TractSeg -i peaks.nii.gz -o tractseg_output --raw_diffusion_input

# Requires:
# - Brain mask: brain_mask.nii.gz
# - bvals: bvals.txt
# - bvecs: bvecs.txt

# TractSeg will handle registration internally
```

### Full Native Space Pipeline

```bash
# 1. Preprocess in native space
dwi2response dhollander dwi.mif wm.txt gm.txt csf.txt
dwi2fod msmt_csd dwi.mif wm.txt wm_fod.mif gm.txt gm_fod.mif csf.txt csf_fod.mif
sh2peaks wm_fod.mif peaks.nii.gz -num 3

# 2. Run TractSeg in raw mode
TractSeg -i peaks.nii.gz \
  -o tractseg_output \
  --raw_diffusion_input \
  --brain_mask brain_mask.nii.gz

# Outputs will be in native space
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# Batch TractSeg processing

subjects=(sub-01 sub-02 sub-03 sub-04)

for subj in "${subjects[@]}"; do
    echo "Processing ${subj}..."

    # Input peaks
    peaks="${subj}/peaks_MNI.nii.gz"
    output="${subj}/tractseg"

    # Segmentations
    TractSeg -i ${peaks} -o ${output} --output_type tract_segmentation

    # TOMs
    TractSeg -i ${peaks} -o ${output} --output_type TOM

    # Endings
    TractSeg -i ${peaks} -o ${output} --output_type endings_segmentation

    # Tractography
    Tracking -i ${peaks} -o ${output} --tracking_format tck

    echo "${subj} complete"
done
```

### Parallel Processing

```bash
# GNU Parallel
parallel -j 4 \
  'TractSeg -i {}/peaks.nii.gz -o {}/tractseg' \
  ::: sub-*/

# SLURM array
#!/bin/bash
#SBATCH --array=1-50
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)

TractSeg -i ${SUBJECT}/peaks.nii.gz \
  -o ${SUBJECT}/tractseg \
  --use_gpu
```

## Visualization

### View Segmentations

```bash
# FSLeyes
fsleyes peaks.nii.gz \
  tractseg_output/bundle_segmentations/CST_left.nii.gz -cm red -a 50 \
  tractseg_output/bundle_segmentations/CST_right.nii.gz -cm blue -a 50

# MRView (MRtrix3)
mrview peaks.nii.gz -overlay.load tractseg_output/bundle_segmentations/CST_left.nii.gz
```

### View Tractography

```bash
# MRView
mrview peaks.nii.gz \
  -tractography.load tractseg_output/TOM_trackings/CST_left.tck

# TrackVis
trackvis tractseg_output/TOM_trackings/CST_left.tck
```

## Quality Control

### Check Segmentation Quality

```bash
# Visual inspection
for tract in AF_left CST_left ILF_left; do
    fsleyes peaks.nii.gz \
      tractseg_output/bundle_segmentations/${tract}.nii.gz &
done

# Check tract volumes
python << EOF
import nibabel as nib
import os

seg_dir = 'tractseg_output/bundle_segmentations'
for file in os.listdir(seg_dir):
    if file.endswith('.nii.gz'):
        img = nib.load(os.path.join(seg_dir, file))
        voxels = (img.get_fdata() > 0).sum()
        volume_mm3 = voxels * abs(img.affine[0,0] * img.affine[1,1] * img.affine[2,2])
        print(f'{file}: {voxels} voxels, {volume_mm3:.1f} mm³')
EOF
```

## Available Tracts

### List of 72 Segmented Tracts

```python
# Major association tracts
- Arcuate Fasciculus (AF): left, right
- Cingulum (CG): left, right
- Inferior Longitudinal Fasciculus (ILF): left, right
- Inferior Fronto-Occipital Fasciculus (IFO): left, right
- Superior Longitudinal Fasciculus I, II, III (SLF): left, right (×3)
- Uncinate Fasciculus (UF): left, right

# Projection tracts
- Corticospinal Tract (CST): left, right
- Corona Radiata (CR): anterior, superior, posterior × left, right

# Commissural tracts
- Corpus Callosum: 7 segments
- Anterior Commissure (AC)
- Fornix (FX): left, right, body

# Brainstem
- Medial Lemniscus (ML): left, right
- Superior Cerebellar Peduncle (SCP): left, right
- Middle Cerebellar Peduncle (MCP)

# And 40+ more...
```

## Integration with Claude Code

When helping users with TractSeg:

1. **Check Installation:**
   ```bash
   TractSeg --help
   python -c "import torch; print(torch.__version__)"
   ```

2. **Common Issues:**
   - PyTorch not installed or wrong version
   - Peaks not in expected format (9 values per voxel)
   - Data not in MNI space (use --raw_diffusion_input)
   - Insufficient memory (reduce batch size)
   - GPU out of memory (use CPU mode)

3. **Best Practices:**
   - Use MRtrix3 CSD peaks for best results
   - Register to MNI152 template before TractSeg
   - Visual QC of all segmentations
   - Use TOMs for bundle-specific tracking
   - Generate endings for robust seeding
   - Extract metrics from segmentations
   - Document TractSeg version used

4. **Quality Checks:**
   - Verify peak extraction quality
   - Check tract volumes are reasonable
   - Visual inspection of major tracts
   - Compare left/right hemisphere symmetry
   - Validate with known anatomy

## Troubleshooting

**Problem:** "RuntimeError: Expected 4D input"
**Solution:** Verify peaks.nii.gz is 4D with shape (x, y, z, 9), reshape if needed

**Problem:** Segmentations look wrong
**Solution:** Check MNI registration quality, verify peak directions correct, ensure proper preprocessing

**Problem:** Missing tracts in output
**Solution:** Some tracts may not be detected if data quality insufficient, check model confidence with --get_probabilities

**Problem:** Out of memory
**Solution:** Use CPU instead of GPU, reduce batch size, process fewer subjects at once

**Problem:** Slow processing
**Solution:** Use GPU with --use_gpu, reduce number of peaks, use lighter model

## Resources

- GitHub: https://github.com/MIC-DKFZ/TractSeg
- Paper: Wasserthal et al. (2018) NeuroImage
- Documentation: https://github.com/MIC-DKFZ/TractSeg/blob/master/README.md
- Pre-trained weights: https://zenodo.org/record/6481434
- Tutorial: https://github.com/MIC-DKFZ/TractSeg/blob/master/examples/

## Citation

```bibtex
@article{wasserthal2018tractseg,
  title={TractSeg-Fast and accurate white matter tract segmentation},
  author={Wasserthal, Jakob and Neher, Peter and Maier-Hein, Klaus H},
  journal={NeuroImage},
  volume={183},
  pages={239--253},
  year={2018},
  publisher={Elsevier}
}
```

## Related Tools

- **MRtrix3:** FOD estimation and tractography
- **DSI Studio:** Alternative tract segmentation
- **RecoBundles:** DIPY bundle recognition
- **TRACULA:** FreeSurfer tract reconstruction
- **AutoPtx:** FSL automated probabilistic tractography
- **Diffusion Toolkit:** TrackVis integration
