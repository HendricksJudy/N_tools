# QSIPrep

## Overview

QSIPrep (Q-Space Imaging Preprocessing) is a robust preprocessing pipeline for diffusion MRI (dMRI) data, built on the same framework as fMRIPrep. It provides state-of-the-art preprocessing for single-shell, multi-shell, and Cartesian q-space sampling schemes, with comprehensive quality control reports and outputs in multiple spaces.

**Website:** https://qsiprep.readthedocs.io/
**Platform:** Linux/macOS (Docker/Singularity containers)
**Language:** Python (Nipype-based)
**License:** BSD 3-Clause

## Key Features

- BIDS-compliant input and output
- Single-shell and multi-shell diffusion preprocessing
- Cartesian (DSI, GQI) q-space sampling support
- Distortion correction (fieldmap-based and fieldmap-less)
- Motion and eddy current correction
- Gradient nonlinearity correction
- Denoising (MP-PCA, MPPCA)
- Gibbs unringing
- Output to multiple standard spaces
- Reconstruction workflows (DTI, CSD, MAPMRI, etc.)
- Comprehensive HTML quality reports
- Integration with DSI Studio, MRtrix3, DIPY

## Installation

### Using Docker (Recommended)

```bash
# Pull latest version
docker pull pennbbl/qsiprep:latest

# Or specific version
docker pull pennbbl/qsiprep:0.19.0

# Check version
docker run -it --rm pennbbl/qsiprep:latest --version
```

### Using Singularity

```bash
# Build from Docker image
singularity build qsiprep-0.19.0.sif docker://pennbbl/qsiprep:0.19.0

# Run
singularity run qsiprep-0.19.0.sif --version
```

### FreeSurfer License

```bash
# QSIPrep requires FreeSurfer license for recon-all
# Get free license from: https://surfer.nmr.mgh.harvard.edu/registration.html
# Place license.txt in your home directory or specify with --fs-license-file
```

## Basic Usage

### BIDS Dataset Structure

```
bids_dataset/
├── dataset_description.json
├── participants.tsv
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   └── dwi/
│       ├── sub-01_dwi.nii.gz
│       ├── sub-01_dwi.bval
│       ├── sub-01_dwi.bvec
│       └── sub-01_dwi.json
└── sub-02/
    ├── anat/
    └── dwi/
```

### Running QSIPrep with Docker

```bash
# Basic command
docker run -ti --rm \
    -v /path/to/bids_dataset:/data:ro \
    -v /path/to/output:/out \
    -v /path/to/freesurfer/license.txt:/opt/freesurfer/license.txt:ro \
    pennbbl/qsiprep:latest \
    /data /out \
    participant \
    --participant-label 01 \
    --fs-license-file /opt/freesurfer/license.txt

# With more options
docker run -ti --rm \
    -v /path/to/bids_dataset:/data:ro \
    -v /path/to/output:/out \
    -v /path/to/work:/work \
    -v /path/to/freesurfer/license.txt:/opt/freesurfer/license.txt:ro \
    pennbbl/qsiprep:latest \
    /data /out \
    participant \
    --participant-label 01 02 03 \
    --fs-license-file /opt/freesurfer/license.txt \
    --work-dir /work \
    --output-resolution 1.5 \
    --denoise-method dwidenoise \
    --unringing-method mrdegibbs \
    --dwi-denoise-window 5 \
    --nthreads 8 \
    --omp-nthreads 4 \
    --mem-mb 16000 \
    --output-space T1w
```

### Running with Singularity

```bash
# Basic run
singularity run --cleanenv \
    -B /path/to/bids_dataset:/data:ro \
    -B /path/to/output:/out \
    -B /path/to/freesurfer/license.txt:/opt/freesurfer/license.txt:ro \
    qsiprep-0.19.0.sif \
    /data /out \
    participant \
    --participant-label 01 \
    --fs-license-file /opt/freesurfer/license.txt

# On HPC with SLURM
#!/bin/bash
#SBATCH --job-name=qsiprep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

singularity run --cleanenv \
    -B /data:/data:ro \
    -B /output:/out \
    -B /work:/work \
    qsiprep.sif \
    /data /out participant \
    --participant-label ${SUBJECT_ID} \
    --fs-license-file /opt/freesurfer/license.txt \
    --work-dir /work \
    --nthreads 8 \
    --omp-nthreads 4
```

## Preprocessing Options

### Denoising

```bash
# Marchenko-Pastur PCA denoising (default)
--denoise-method dwidenoise

# Patch2Self denoising
--denoise-method patch2self

# No denoising
--denoise-method none

# Denoising window size
--dwi-denoise-window 5  # 5x5x5 window (default: 5)
```

### Gibbs Unringing

```bash
# MRtrix3 method (default)
--unringing-method mrdegibbs

# No unringing
--unringing-method none
```

### Distortion Correction

```bash
# Use fieldmaps (if available in BIDS)
# QSIPrep automatically detects fieldmaps

# Fieldmap-less correction (SyN-based)
--use-syn-sdc

# Force specific distortion correction
--force-syn

# Ignore fieldmaps
--ignore fieldmaps
```

### Motion and Eddy Correction

```bash
# Eddy correction method
--hmc-model eddy  # FSL eddy (default, recommended)
--hmc-model 3dSHORE  # Experimental

# Eddy options
--eddy-config /path/to/eddy_config.json
```

### Output Spaces

```bash
# Output spaces
--output-space T1w MNI152NLin2009cAsym

# Available spaces:
# - T1w: Native anatomical space
# - template: Template space (MNI152NLin2009cAsym default)
# - Custom template

# Output resolution
--output-resolution 1.5  # mm isotropic
--output-resolution native  # Keep original resolution
```

## Reconstruction Workflows

### Built-in Reconstruction

```bash
# Specify reconstruction workflow
--recon-spec mrtrix_multishell_msmt_ACT-hsvs

# Available workflows:
# - mrtrix_multishell_msmt
# - mrtrix_multishell_msmt_ACT-hsvs
# - mrtrix_multishell_msmt_ACT-fast
# - dsi_studio_gqi
# - dipy_mapmri
# - dipy_dki
# - csdsi_3dshore
# - amico_noddi
# - tortoise

# Custom reconstruction spec
--recon-spec /path/to/custom_spec.json
```

### Example: MRtrix3 Multi-Shell Reconstruction

```bash
docker run -ti --rm \
    -v /data:/data:ro \
    -v /output:/out \
    pennbbl/qsiprep:latest \
    /data /out \
    participant \
    --participant-label 01 \
    --recon-spec mrtrix_multishell_msmt_ACT-hsvs \
    --recon-only  # Skip preprocessing if already done
```

### Example: DSI Studio Reconstruction

```bash
docker run -ti --rm \
    -v /data:/data:ro \
    -v /output:/out \
    pennbbl/qsiprep:latest \
    /data /out \
    participant \
    --participant-label 01 \
    --recon-spec dsi_studio_gqi \
    --output-resolution 2.0
```

## Understanding Outputs

### Directory Structure

```
qsiprep/
├── dataset_description.json
├── sub-01.html                              # QC report
├── sub-01/
│   ├── anat/
│   │   ├── sub-01_desc-preproc_T1w.nii.gz
│   │   └── sub-01_dseg.nii.gz
│   └── dwi/
│       ├── sub-01_space-T1w_desc-preproc_dwi.nii.gz
│       ├── sub-01_space-T1w_desc-preproc_dwi.bval
│       ├── sub-01_space-T1w_desc-preproc_dwi.bvec
│       ├── sub-01_space-T1w_dwiref.nii.gz
│       ├── sub-01_space-T1w_desc-brain_mask.nii.gz
│       └── sub-01_desc-confounds_timeseries.tsv
├── figures/
└── logs/
```

### Key Output Files

1. **Preprocessed DWI**: `*_space-*_desc-preproc_dwi.nii.gz`
2. **Gradient table**: `*_desc-preproc_dwi.bval/bvec`
3. **Brain mask**: `*_desc-brain_mask.nii.gz`
4. **Confounds**: `*_desc-confounds_timeseries.tsv`
5. **QC report**: `sub-*.html`

### Reconstruction Outputs

```
qsiprep/
└── sub-01/
    └── dwi/
        ├── sub-01_space-T1w_model-CSD_ODF.nii.gz
        ├── sub-01_space-T1w_model-CSD_WM.nii.gz
        ├── sub-01_space-T1w_model-DTI_FA.nii.gz
        ├── sub-01_space-T1w_model-DTI_MD.nii.gz
        └── sub-01_space-T1w_tractography.tck
```

## Working with QSIPrep Outputs

### Load Preprocessed Data in Python

```python
import nibabel as nib
import numpy as np

# Load preprocessed DWI
dwi_img = nib.load('sub-01_space-T1w_desc-preproc_dwi.nii.gz')
dwi_data = dwi_img.get_fdata()

# Load gradient table
bvals = np.loadtxt('sub-01_space-T1w_desc-preproc_dwi.bval')
bvecs = np.loadtxt('sub-01_space-T1w_desc-preproc_dwi.bvec')

# Load brain mask
mask_img = nib.load('sub-01_space-T1w_desc-brain_mask.nii.gz')
mask = mask_img.get_fdata().astype(bool)

# DIPY analysis
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

gtab = gradient_table(bvals, bvecs)
ten_model = TensorModel(gtab)
ten_fit = ten_model.fit(dwi_data, mask=mask)

fa = ten_fit.fa
```

### Tractography with MRtrix3

```bash
# If you used mrtrix_multishell_msmt reconstruction
# Outputs include FODs and 5TT

# Tractography
tckgen sub-01_space-T1w_model-CSD_WM.nii.gz \
    tracks.tck \
    -act sub-01_space-T1w_5tt.nii.gz \
    -seed_dynamic sub-01_space-T1w_model-CSD_WM.nii.gz \
    -select 10M

# SIFT filtering
tcksift2 tracks.tck \
    sub-01_space-T1w_model-CSD_WM.nii.gz \
    sift_weights.txt
```

## Quality Control

### Visual QC Reports

```bash
# Open HTML report for each subject
firefox qsiprep/sub-01.html

# Check:
# - Alignment between T1w and DWI
# - Motion parameters
# - Susceptibility distortion correction
# - Brain mask quality
# - Signal dropout
```

### Extract QC Metrics

```python
import pandas as pd

# Load confounds
confounds = pd.read_csv(
    'sub-01_desc-confounds_timeseries.tsv',
    sep='\t'
)

# Check motion
fd = confounds['framewise_displacement']
mean_fd = fd.mean()
print(f"Mean FD: {mean_fd:.3f} mm")

# Identify high-motion volumes
high_motion = (fd > 0.5).sum()
print(f"Volumes with FD > 0.5mm: {high_motion} ({100*high_motion/len(fd):.1f}%)")
```

## Common Workflows

### Multi-Shell Analysis Pipeline

```bash
# 1. Preprocess
docker run -ti --rm \
    -v /data:/data:ro \
    -v /output:/out \
    pennbbl/qsiprep:latest \
    /data /out participant \
    --participant-label 01 \
    --denoise-method dwidenoise \
    --unringing-method mrdegibbs \
    --output-space T1w \
    --fs-license-file /opt/freesurfer/license.txt

# 2. Reconstruct with MRtrix3
docker run -ti --rm \
    -v /output:/out \
    pennbbl/qsiprep:latest \
    /out /out participant \
    --participant-label 01 \
    --recon-spec mrtrix_multishell_msmt_ACT-hsvs \
    --recon-only \
    --fs-license-file /opt/freesurfer/license.txt

# 3. Post-processing with MRtrix3
# Extract outputs and run tractography, connectivity, etc.
```

### Single-Shell DTI Pipeline

```bash
# Preprocess and compute DTI
docker run -ti --rm \
    -v /data:/data:ro \
    -v /output:/out \
    pennbbl/qsiprep:latest \
    /data /out participant \
    --participant-label 01 \
    --output-space T1w \
    --recon-spec dsi_studio_gqi \
    --fs-license-file /opt/freesurfer/license.txt
```

## Advanced Options

### Custom Preprocessing

```bash
# Skip specific steps
--skip-bids-validation
--ignore slicetiming
--ignore fieldmaps

# Combine multiple runs
--combine-all-dwis

# Use specific b0 threshold
--b0-threshold 50  # Treat b<50 as b0

# Motion correction
--hmc-model eddy
--hmc-transform Rigid

# Use pre-computed brain mask
--brain-mask-path /path/to/mask.nii.gz
```

### Recon-Only Mode

```bash
# If preprocessing already done, run only reconstruction
docker run -ti --rm \
    -v /qsiprep_output:/out \
    pennbbl/qsiprep:latest \
    /out /out participant \
    --participant-label 01 \
    --recon-spec mrtrix_multishell_msmt \
    --recon-only \
    --fs-license-file /opt/freesurfer/license.txt
```

## Troubleshooting

**Problem:** Out of memory errors
**Solution:** Reduce resolution (`--output-resolution 2`), increase `--mem-mb`, or use `--low-mem`

**Problem:** Eddy correction fails
**Solution:** Check gradient table format, try `--force-syn`, or adjust `--hmc-model`

**Problem:** Distortion correction poor results
**Solution:** Verify fieldmap metadata, try `--use-syn-sdc`, check phase encoding direction

**Problem:** Reconstruction fails
**Solution:** Check shell coverage, verify b-values, try different `--recon-spec`

## Integration with Claude Code

When helping users with QSIPrep:

1. **Check Installation:**
   ```bash
   docker run -it --rm pennbbl/qsiprep:latest --version
   ```

2. **Verify BIDS:**
   ```bash
   docker run -it --rm -v /data:/data:ro \
       bids/validator /data
   ```

3. **Common Issues:**
   - BIDS validation errors (missing .json, wrong naming)
   - Gradient table orientation issues
   - Insufficient b-value coverage for multi-shell
   - Memory/disk space limitations

4. **Best Practices:**
   - Always review HTML QC reports
   - Check gradient tables before and after preprocessing
   - Use work directory on fast storage
   - Keep detailed processing logs
   - Validate outputs with DIPY/MRtrix3

## Resources

- Documentation: https://qsiprep.readthedocs.io/
- GitHub: https://github.com/PennLINC/qsiprep
- NeuroStars: https://neurostars.org/ (tag: qsiprep)
- Reconstruction Specs: https://qsiprep.readthedocs.io/en/latest/reconstruction.html
- Docker Hub: https://hub.docker.com/r/pennbbl/qsiprep

## Citation

```bibtex
@article{cieslak2021qsiprep,
  title={QSIPrep: an integrative platform for preprocessing and reconstructing diffusion MRI data},
  author={Cieslak, Matthew and Cook, Philip A and He, Xiaosong and others},
  journal={Nature methods},
  volume={18},
  number={7},
  pages={775--778},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Related Tools

- **fMRIPrep:** fMRI preprocessing (same framework)
- **MRtrix3:** Advanced diffusion analysis
- **DIPY:** Python diffusion toolkit
- **DSI Studio:** Diffusion imaging analysis
- **TORTOISE:** Alternative diffusion preprocessing
