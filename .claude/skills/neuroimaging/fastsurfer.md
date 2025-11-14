# FastSurfer

## Overview

FastSurfer is a deep learning-based neuroimaging pipeline that provides rapid and accurate cortical parcellation and segmentation of brain MRI scans. Developed as a faster alternative to FreeSurfer's recon-all, FastSurfer achieves comparable accuracy in approximately 1 hour compared to FreeSurfer's 5-24 hours, making it ideal for large-scale studies and clinical applications while maintaining FreeSurfer-compatible outputs.

**Website:** https://fastsurfer.github.io/
**Platform:** Python/PyTorch with Docker/Singularity
**Language:** Python
**License:** FreeSurfer License (requires FreeSurfer license for surface generation)

## Key Features

- ~100x faster than FreeSurfer recon-all (1 hour vs. 5-24 hours)
- FreeSurfer-compatible outputs and directory structure
- GPU and CPU support (GPU highly recommended)
- Deep learning-based segmentation (FastSurferCNN)
- Optional surface reconstruction
- BIDS-compatible processing
- Batch processing capabilities
- Docker and Singularity containers
- Quality control tools
- Comparable accuracy to FreeSurfer
- Lower memory requirements
- Easy integration with existing FreeSurfer workflows

## Installation

### Docker Installation (Recommended)

```bash
# Pull official Docker image
docker pull deepmi/fastsurfer:latest

# Verify installation
docker run --rm deepmi/fastsurfer:latest --version

# GPU-enabled Docker (NVIDIA GPU required)
# Install nvidia-docker2 first
docker run --gpus all --rm deepmi/fastsurfer:latest --version
```

### Singularity Installation

```bash
# Pull Singularity image
singularity pull docker://deepmi/fastsurfer:latest

# This creates fastsurfer_latest.sif

# Verify
singularity exec fastsurfer_latest.sif /fastsurfer/run_fastsurfer.sh --version

# GPU support (automatic with Singularity 3.0+)
singularity exec --nv fastsurfer_latest.sif /fastsurfer/run_fastsurfer.sh --help
```

### Source Installation

```bash
# Clone repository
git clone https://github.com/Deep-MI/FastSurfer.git
cd FastSurfer

# Create conda environment
conda env create -f environment.yml
conda activate fastsurfer

# Install FastSurfer
python setup.py install

# Download network checkpoints
cd checkpoints
./get_checkpoints.sh

# FreeSurfer license
# Place license.txt in $FREESURFER_HOME or current directory
cp /path/to/license.txt $FREESURFER_HOME/
```

## FreeSurfer License

```bash
# FastSurfer requires FreeSurfer license for surface generation
# Free registration at: https://surfer.nmr.mgh.harvard.edu/registration.html

# Place license
export FREESURFER_HOME=/path/to/freesurfer
cp license.txt $FREESURFER_HOME/license.txt

# Or specify with --fs_license flag
```

## Basic Usage

### Segmentation Only (No Surfaces)

```bash
# Docker - Segmentation only (fastest, ~1-5 min on GPU)
docker run --gpus all \
  -v /path/to/input:/data \
  -v /path/to/output:/output \
  -v /path/to/license.txt:/fs_license/license.txt \
  --rm --user $(id -u):$(id -g) \
  deepmi/fastsurfer:latest \
  --t1 /data/subject_T1.nii.gz \
  --sid subject01 \
  --sd /output \
  --seg_only

# Output: aparc.DKTatlas+aseg.deep.mgz (FreeSurfer parcellation)
```

### Full Pipeline with Surfaces

```bash
# Complete pipeline (~1 hour on GPU)
docker run --gpus all \
  -v /path/to/input:/data \
  -v /path/to/output:/output \
  -v /path/to/license.txt:/fs_license/license.txt \
  --rm --user $(id -u):$(id -g) \
  deepmi/fastsurfer:latest \
  --t1 /data/subject_T1.nii.gz \
  --sid subject01 \
  --sd /output \
  --fs_license /fs_license/license.txt

# Full outputs: segmentation + surfaces + stats
```

### CPU Mode

```bash
# Without GPU (much slower but works)
docker run \
  -v /path/to/input:/data \
  -v /path/to/output:/output \
  deepmi/fastsurfer:latest \
  --t1 /data/subject_T1.nii.gz \
  --sid subject01 \
  --sd /output \
  --seg_only \
  --device cpu
```

## FastSurferCNN Segmentation

### Segmentation Modes

```bash
# Viewagg (default, best quality, requires more memory)
docker run --gpus all -v /input:/data -v /output:/output \
  deepmi/fastsurfer:latest \
  --t1 /data/T1.nii.gz --sid sub01 --sd /output \
  --seg_only --viewagg_device auto

# Fast mode (single-view, faster but lower quality)
docker run --gpus all -v /input:/data -v /output:/output \
  deepmi/fastsurfer:latest \
  --t1 /data/T1.nii.gz --sid sub01 --sd /output \
  --seg_only --no_viewagg

# Specify batch size (adjust for GPU memory)
docker run --gpus all -v /input:/data -v /output:/output \
  deepmi/fastsurfer:latest \
  --t1 /data/T1.nii.gz --sid sub01 --sd /output \
  --seg_only --batch_size 8
```

### Output Segmentation

```bash
# Main output: aparc.DKTatlas+aseg.deep.mgz
# Contains 95 cortical and subcortical labels
# Compatible with FreeSurfer's DKT atlas

# Segmentation labels include:
# - Cortical parcellation (DKT atlas)
# - Subcortical structures
# - White matter
# - Cerebellum
# - Brainstem
```

## Surface Reconstruction

### With FastSurfer Surfaces

```bash
# Full pipeline with surface generation
docker run --gpus all \
  -v /input:/data \
  -v /output:/output \
  -v /license:/fs_license \
  deepmi/fastsurfer:latest \
  --t1 /data/T1.nii.gz \
  --sid subject01 \
  --sd /output \
  --fs_license /fs_license/license.txt \
  --parallel  # Use parallel processing

# Generates:
# - White matter surfaces (lh/rh.white)
# - Pial surfaces (lh/rh.pial)
# - Inflated surfaces (lh/rh.inflated)
# - Sphere registration (lh/rh.sphere.reg)
# - Curvature files
# - Surface statistics
```

### Hybrid: FastSurfer Segmentation + FreeSurfer Surfaces

```bash
# Step 1: Run FastSurfer segmentation only
docker run --gpus all -v /input:/data -v /output:/output \
  deepmi/fastsurfer:latest \
  --t1 /data/T1.nii.gz --sid sub01 --sd /output --seg_only

# Step 2: Run FreeSurfer surface pipeline using FastSurfer segmentation
recon-all -subjid sub01 -sd /output \
  -autorecon2 -autorecon3 \
  -no-isrunning
```

## Batch Processing

### Multiple Subjects Script

```bash
#!/bin/bash
# Batch process multiple subjects

INPUT_DIR="/data/raw"
OUTPUT_DIR="/data/derivatives/fastsurfer"
LICENSE="/path/to/license.txt"

# Find all T1 images
for t1_file in ${INPUT_DIR}/sub-*/anat/*_T1w.nii.gz; do
    # Extract subject ID
    subj=$(basename $(dirname $(dirname $t1_file)))

    echo "Processing: $subj"

    # Run FastSurfer
    docker run --gpus all \
      -v ${INPUT_DIR}:/input \
      -v ${OUTPUT_DIR}:/output \
      -v ${LICENSE}:/license/license.txt \
      --rm --user $(id -u):$(id -g) \
      deepmi/fastsurfer:latest \
      --t1 /input/${subj}/anat/${subj}_T1w.nii.gz \
      --sid ${subj} \
      --sd /output \
      --fs_license /license/license.txt \
      --parallel

    echo "Completed: $subj"
done

echo "Batch processing complete!"
```

### Parallel Processing

```bash
# Process multiple subjects in parallel (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0
docker run --gpus all ... --sid sub01 &

export CUDA_VISIBLE_DEVICES=1
docker run --gpus all ... --sid sub02 &

# Wait for all to complete
wait
```

## BIDS Integration

### BIDS-Compatible Processing

```bash
# BIDS dataset structure
bids_dataset/
├── sub-01/
│   └── anat/
│       └── sub-01_T1w.nii.gz
├── sub-02/
│   └── anat/
│       └── sub-02_T1w.nii.gz
└── dataset_description.json

# Process BIDS dataset
docker run --gpus all \
  -v /path/to/bids_dataset:/data \
  -v /path/to/derivatives:/output \
  -v /path/to/license.txt:/license/license.txt \
  deepmi/fastsurfer:latest \
  --t1 /data/sub-01/anat/sub-01_T1w.nii.gz \
  --sid sub-01 \
  --sd /output \
  --fs_license /license/license.txt

# Derivatives structure (BIDS-compatible)
derivatives/
└── fastsurfer/
    ├── sub-01/
    │   ├── mri/
    │   ├── surf/
    │   ├── stats/
    │   └── scripts/
    └── dataset_description.json
```

### Create BIDS Derivatives

```python
import json
from pathlib import Path

# Create dataset_description.json for derivatives
derivatives_info = {
    "Name": "FastSurfer Derivatives",
    "BIDSVersion": "1.6.0",
    "GeneratedBy": [{
        "Name": "FastSurfer",
        "Version": "2.0",
        "Container": {
            "Type": "docker",
            "Tag": "deepmi/fastsurfer:latest"
        }
    }],
    "SourceDatasets": [{
        "URL": "path/to/source/dataset"
    }]
}

derivatives_path = Path("/output/derivatives/fastsurfer")
derivatives_path.mkdir(parents=True, exist_ok=True)

with open(derivatives_path / "dataset_description.json", 'w') as f:
    json.dump(derivatives_info, f, indent=2)
```

## Quality Control

### Visual QC with FreeView

```bash
# FreeSurfer's FreeView for QC
freeview -v ${SUBJECTS_DIR}/sub01/mri/T1.mgz \
         -v ${SUBJECTS_DIR}/sub01/mri/aparc.DKTatlas+aseg.deep.mgz:colormap=lut \
         -f ${SUBJECTS_DIR}/sub01/surf/lh.pial:edgecolor=red \
         -f ${SUBJECTS_DIR}/sub01/surf/rh.pial:edgecolor=red \
         -f ${SUBJECTS_DIR}/sub01/surf/lh.white:edgecolor=yellow \
         -f ${SUBJECTS_DIR}/sub01/surf/rh.white:edgecolor=yellow
```

### Automated QC

```python
import nibabel as nib
import numpy as np

def qc_segmentation(seg_file):
    """Basic QC checks for FastSurfer segmentation."""

    seg = nib.load(seg_file)
    seg_data = seg.get_fdata()

    qc_results = {}

    # Check for reasonable brain volume
    brain_voxels = np.sum(seg_data > 0)
    brain_volume_mm3 = brain_voxels * np.prod(seg.header.get_zooms())
    qc_results['brain_volume_ml'] = brain_volume_mm3 / 1000

    # Typical adult brain: 1000-1500 ml
    if 800 < qc_results['brain_volume_ml'] < 1800:
        qc_results['volume_check'] = 'PASS'
    else:
        qc_results['volume_check'] = 'FAIL'

    # Check for left/right balance
    midline = seg_data.shape[0] // 2
    left_voxels = np.sum(seg_data[:midline, :, :] > 0)
    right_voxels = np.sum(seg_data[midline:, :, :] > 0)
    asymmetry = abs(left_voxels - right_voxels) / (left_voxels + right_voxels)
    qc_results['hemisphere_asymmetry'] = asymmetry

    if asymmetry < 0.15:  # Less than 15% asymmetry
        qc_results['asymmetry_check'] = 'PASS'
    else:
        qc_results['asymmetry_check'] = 'WARNING'

    return qc_results

# Run QC
qc = qc_segmentation('sub01/mri/aparc.DKTatlas+aseg.deep.mgz')
print(qc)
```

## Comparison with FreeSurfer

### Speed Comparison

```bash
# FastSurfer (GPU): ~1 hour
# FastSurfer (CPU): ~4-6 hours
# FreeSurfer: ~5-24 hours

# Benchmark test
time docker run --gpus all ... # FastSurfer
time recon-all -all -i T1.nii.gz -s subject # FreeSurfer
```

### Accuracy Comparison

```python
import nibabel as nib
import numpy as np

def compare_segmentations(fastsurfer_seg, freesurfer_seg):
    """Compare FastSurfer and FreeSurfer segmentations."""

    fs_fast = nib.load(fastsurfer_seg).get_fdata()
    fs_free = nib.load(freesurfer_seg).get_fdata()

    # Calculate Dice coefficient
    intersection = np.sum((fs_fast > 0) & (fs_free > 0))
    dice = 2 * intersection / (np.sum(fs_fast > 0) + np.sum(fs_free > 0))

    print(f"Overall Dice coefficient: {dice:.4f}")

    # Per-label comparison
    labels = np.unique(fs_free)
    for label in labels:
        if label == 0:  # Skip background
            continue

        mask_fast = fs_fast == label
        mask_free = fs_free == label

        if np.sum(mask_free) == 0:  # Label not in FreeSurfer
            continue

        inter = np.sum(mask_fast & mask_free)
        label_dice = 2 * inter / (np.sum(mask_fast) + np.sum(mask_free))

        if label_dice < 0.8:  # Flag low agreement
            print(f"Label {int(label)}: Dice = {label_dice:.3f} (LOW)")

# Run comparison
compare_segmentations(
    'fastsurfer/sub01/mri/aparc.DKTatlas+aseg.deep.mgz',
    'freesurfer/sub01/mri/aparc.DKTatlas+aseg.mgz'
)
```

## Extract Statistics

### Volume and Thickness Measurements

```bash
# FastSurfer generates stats files compatible with FreeSurfer
ls ${SUBJECTS_DIR}/sub01/stats/

# aseg.stats - Subcortical volumes
# lh.aparc.DKTatlas.stats - Left hemisphere cortical stats
# rh.aparc.DKTatlas.stats - Right hemisphere cortical stats

# View statistics
cat ${SUBJECTS_DIR}/sub01/stats/aseg.stats
```

### Extract Volumes Programmatically

```python
import pandas as pd

def parse_aseg_stats(stats_file):
    """Extract volumes from aseg.stats file."""

    volumes = {}

    with open(stats_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split()
            if len(parts) >= 5:
                structure = parts[4]
                volume_mm3 = float(parts[3])
                volumes[structure] = volume_mm3

    return pd.DataFrame.from_dict(volumes, orient='index',
                                  columns=['Volume_mm3'])

# Extract volumes for all subjects
subjects = ['sub01', 'sub02', 'sub03']
all_volumes = {}

for subj in subjects:
    stats_file = f'{SUBJECTS_DIR}/{subj}/stats/aseg.stats'
    all_volumes[subj] = parse_aseg_stats(stats_file)

# Combine into DataFrame
volumes_df = pd.concat(all_volumes, axis=1)
volumes_df.to_csv('fastsurfer_volumes.csv')
```

## Advanced Options

### Custom Checkpoints

```bash
# Use custom trained checkpoint
docker run --gpus all \
  -v /input:/data \
  -v /output:/output \
  -v /checkpoints:/ckpt \
  deepmi/fastsurfer:latest \
  --t1 /data/T1.nii.gz \
  --sid sub01 \
  --sd /output \
  --checkpoint /ckpt/custom_model.pkl \
  --seg_only
```

### Specify GPU

```bash
# Use specific GPU
docker run --gpus '"device=1"' \
  -v /input:/data -v /output:/output \
  deepmi/fastsurfer:latest \
  --t1 /data/T1.nii.gz --sid sub01 --sd /output --seg_only

# Environment variable
export CUDA_VISIBLE_DEVICES=2
docker run --gpus all ...
```

### Partial Processing

```bash
# Run only specific hemispheres
docker run --gpus all ... --hemi lh  # Left hemisphere only
docker run --gpus all ... --hemi rh  # Right hemisphere only

# Skip certain steps
docker run --gpus all ... --no_surfreg  # Skip surface registration
```

## Integration with Other Tools

### Export to Other Formats

```bash
# Convert .mgz to .nii.gz
mri_convert aparc.DKTatlas+aseg.deep.mgz \
            aparc.DKTatlas+aseg.deep.nii.gz

# Use with FSL, ANTs, etc.
```

### Integration with Statistical Analysis

```python
import pandas as pd
from scipy import stats

# Load volumes from multiple subjects
controls = pd.read_csv('fastsurfer_volumes_controls.csv', index_col=0)
patients = pd.read_csv('fastsurfer_volumes_patients.csv', index_col=0)

# Statistical comparison
for structure in controls.index:
    ctrl_vols = controls.loc[structure]
    pat_vols = patients.loc[structure]

    t, p = stats.ttest_ind(ctrl_vols, pat_vols)

    if p < 0.05:
        print(f"{structure}: t={t:.2f}, p={p:.4f}")
```

## Singularity Usage (HPC)

```bash
# On HPC cluster
export SINGULARITY_CACHEDIR=/scratch/singularity
singularity pull docker://deepmi/fastsurfer:latest

# Submit job
#!/bin/bash
#SBATCH --job-name=fastsurfer
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G

module load singularity

singularity exec --nv \
  -B /data:/data \
  -B /output:/output \
  fastsurfer_latest.sif \
  /fastsurfer/run_fastsurfer.sh \
  --t1 /data/sub01_T1w.nii.gz \
  --sid sub01 \
  --sd /output \
  --seg_only
```

## Integration with Claude Code

When helping users with FastSurfer:

1. **Check Installation:**
   ```bash
   docker run --rm deepmi/fastsurfer:latest --version
   ```

2. **Verify GPU:**
   ```bash
   nvidia-smi
   docker run --gpus all --rm nvidia/cuda:11.0-base nvidia-smi
   ```

3. **Common Workflow:**
   - Segmentation only (fast): 1-5 min on GPU
   - Full pipeline: ~1 hour on GPU
   - Use --parallel for surface generation

4. **Best Practices:**
   - Use Docker/Singularity for reproducibility
   - GPU highly recommended (100x faster)
   - Segmentation-only if surfaces not needed
   - Compare with FreeSurfer on subset first

## Troubleshooting

**Problem:** CUDA out of memory
**Solution:** Reduce batch size (--batch_size 4), use --no_viewagg, or process on CPU

**Problem:** Surfaces don't look correct
**Solution:** Check input T1 quality, verify FreeSurfer license, inspect segmentation first

**Problem:** Docker permission denied
**Solution:** Use --user $(id -u):$(id -g) flag, check volume mount permissions

**Problem:** Very different from FreeSurfer
**Solution:** Expected slight differences; check input quality, validate on multiple subjects

**Problem:** Segmentation has holes/artifacts
**Solution:** Check input scan quality, try different viewagg settings, consider preprocessing

## Best Practices

1. **Use GPU** whenever possible for acceptable processing times
2. **Start with segmentation only** to verify quality before surfaces
3. **Use Docker/Singularity** for reproducibility and easy deployment
4. **QC all outputs** visually with FreeView
5. **Compare with FreeSurfer** on representative subset
6. **Keep FreeSurfer license** accessible for surface generation
7. **Use parallel processing** (--parallel) for surface reconstruction
8. **Document versions** (Docker tag, Git commit) for reproducibility
9. **Batch process** with scripts for large studies
10. **Extract statistics** consistently across subjects

## Resources

- **Website:** https://fastsurfer.github.io/
- **GitHub:** https://github.com/Deep-MI/FastSurfer
- **Docker Hub:** https://hub.docker.com/r/deepmi/fastsurfer
- **Paper:** https://www.sciencedirect.com/science/article/pii/S1053811920304985
- **FreeSurfer Wiki:** https://surfer.nmr.mgh.harvard.edu/
- **Issues:** https://github.com/Deep-MI/FastSurfer/issues

## Citation

```bibtex
@article{henschel2020fastsurfer,
  title={FastSurfer-A fast and accurate deep learning based neuroimaging pipeline},
  author={Henschel, Leonie and Conjeti, Sailesh and Estrada, Santiago and Diers, Kersten and Fischl, Bruce and Reuter, Martin},
  journal={NeuroImage},
  volume={219},
  pages={117012},
  year={2020},
  publisher={Elsevier}
}
```

## Related Tools

- **FreeSurfer:** Original cortical reconstruction pipeline
- **nnU-Net:** General segmentation framework
- **SynthSeg:** Robust multi-contrast segmentation
- **CAT12:** VBM preprocessing alternative
- **CIVET:** Alternative cortical pipeline
- **FreeView:** Visualization tool (part of FreeSurfer)
- **ITK-SNAP:** Manual inspection and editing
