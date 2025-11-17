# fMRIPrep

## Overview

fMRIPrep is a robust, state-of-the-art preprocessing pipeline for functional MRI data. Built using Nipype, it provides minimal preprocessing pipelines while maintaining transparency, reproducibility, and versatility. fMRIPrep generates comprehensive visual reports and outputs data in multiple standard spaces, making it a gold standard for fMRI preprocessing.

**Website:** https://fmriprep.org/
**Platform:** Linux/macOS (Docker/Singularity containers)
**Language:** Python (Nipype-based)
**License:** Apache 2.0

## Key Features

- BIDS-compliant input and output
- Robust anatomical preprocessing (brain extraction, surface reconstruction)
- Functional preprocessing with motion correction
- Susceptibility distortion correction (SDC)
- Multi-echo fMRI support
- Output to multiple standard spaces (MNI, fsaverage)
- Comprehensive HTML quality reports
- FreeSurfer integration
- ICA-AROMA for denoising
- Confound extraction for nuisance regression

## Installation

### Using Docker (Recommended)

```bash
# Pull latest version
docker pull nipreps/fmriprep:latest

# Or specific version
docker pull nipreps/fmriprep:23.2.0

# Check version
docker run -it --rm nipreps/fmriprep:latest --version
```

### Using Singularity

```bash
# Build from Docker image
singularity build fmriprep-23.2.0.simg docker://nipreps/fmriprep:23.2.0

# Run
singularity run fmriprep-23.2.0.simg --version
```

### FreeSurfer License

```bash
# fMRIPrep requires FreeSurfer license
# Get free license from: https://surfer.nmr.mgh.harvard.edu/registration.html
# Place license.txt in your home directory or specify with --fs-license-file
```

## Basic Usage

### BIDS Dataset Structure

fMRIPrep expects BIDS-formatted data:

```
bids_dataset/
├── dataset_description.json
├── participants.tsv
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   └── func/
│       ├── sub-01_task-rest_bold.nii.gz
│       └── sub-01_task-rest_bold.json
└── sub-02/
    ├── anat/
    └── func/
```

### Running fMRIPrep with Docker

```bash
# Basic command
docker run -ti --rm \
    -v /path/to/bids_dataset:/data:ro \
    -v /path/to/output:/out \
    -v /path/to/freesurfer/license.txt:/opt/freesurfer/license.txt:ro \
    nipreps/fmriprep:latest \
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
    nipreps/fmriprep:latest \
    /data /out \
    participant \
    --participant-label 01 02 03 \
    --fs-license-file /opt/freesurfer/license.txt \
    --work-dir /work \
    --output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage5 \
    --use-aroma \
    --nthreads 8 \
    --omp-nthreads 4 \
    --mem-mb 16000 \
    --skip-bids-validation
```

### Running with Singularity

```bash
# Basic run
singularity run --cleanenv \
    -B /path/to/bids_dataset:/data:ro \
    -B /path/to/output:/out \
    -B /path/to/freesurfer/license.txt:/opt/freesurfer/license.txt:ro \
    fmriprep-23.2.0.simg \
    /data /out \
    participant \
    --participant-label 01 \
    --fs-license-file /opt/freesurfer/license.txt

# On HPC with SLURM
#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

export SINGULARITYENV_TEMPLATEFLOW_HOME=/path/to/templateflow

singularity run --cleanenv \
    -B /data:/data:ro \
    -B /output:/out \
    -B /work:/work \
    -B /templateflow:/templateflow \
    fmriprep.simg \
    /data /out participant \
    --participant-label ${SUBJECT_ID} \
    --fs-license-file /opt/freesurfer/license.txt \
    --work-dir /work \
    --nthreads 8 \
    --omp-nthreads 4
```

## Output Spaces

```bash
# Specify output spaces
--output-spaces MNI152NLin2009cAsym:res-2 \
                MNI152NLin6Asym:res-native \
                anat \
                fsaverage5 \
                fsnative

# Common spaces:
# - MNI152NLin2009cAsym (MNI template)
# - MNI152NLin6Asym (alternative MNI)
# - anat (native anatomical space)
# - fsaverage5/fsaverage6 (FreeSurfer surfaces)
# - fsnative (subject's native surface)
```

## Common Options

### Resource Management

```bash
# CPU and memory
--nthreads 16          # Total threads for fMRIPrep
--omp-nthreads 8       # OpenMP threads per process
--mem-mb 32000         # Memory limit in MB
--low-mem              # Reduce memory usage

# Work directory
--work-dir /scratch/work  # Temporary working directory
--clean-workdir            # Remove working directory after completion
```

### Preprocessing Options

```bash
# Susceptibility distortion correction
--use-syn-sdc          # Use SyN-based SDC (no fieldmap required)
--ignore fieldmaps     # Skip fieldmap-based SDC

# ICA-AROMA denoising
--use-aroma            # Apply ICA-AROMA

# Slice timing correction
--ignore slicetiming   # Skip slice-timing correction

# FreeSurfer
--fs-no-reconall       # Skip FreeSurfer reconstruction
--fs-subjects-dir /path/to/freesurfer/subjects  # Use existing FreeSurfer

# Surface sampling
--cifti-output         # Generate CIFTI files (91k or 170k)
```

### Quality Control

```bash
# Skip validation
--skip-bids-validation

# Force rerun
--force-reindex
--force-reconall

# Debug
--verbose count
--debug compcor
```

## Understanding Outputs

### Directory Structure

```
fmriprep/
├── dataset_description.json
├── sub-01.html                              # Visual QC report
├── sub-01/
│   ├── anat/
│   │   ├── sub-01_desc-preproc_T1w.nii.gz  # Preprocessed T1
│   │   ├── sub-01_desc-brain_mask.nii.gz   # Brain mask
│   │   ├── sub-01_dseg.nii.gz              # Tissue segmentation
│   │   └── sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
│   ├── func/
│   │   ├── sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
│   │   ├── sub-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
│   │   ├── sub-01_task-rest_desc-confounds_timeseries.tsv
│   │   └── sub-01_task-rest_from-scanner_to-MNI152NLin2009cAsym_mode-image_xfm.h5
│   └── figures/
│       ├── sub-01_desc-summary_T1w.svg
│       ├── sub-01_task-rest_desc-carpet_bold.svg
│       └── ...
└── logs/
```

### Key Output Files

1. **Preprocessed BOLD**: `*_space-*_desc-preproc_bold.nii.gz`
2. **Brain mask**: `*_space-*_desc-brain_mask.nii.gz`
3. **Confounds**: `*_desc-confounds_timeseries.tsv`
4. **Transforms**: `*_from-*_to-*_mode-image_xfm.h5`
5. **HTML report**: `sub-*.html`

## Working with Confounds

```python
import pandas as pd
import numpy as np

# Load confounds
confounds = pd.read_csv('sub-01_task-rest_desc-confounds_timeseries.tsv', sep='\t')

# Available confounds include:
# - Motion parameters (trans_x/y/z, rot_x/y/z)
# - Framewise displacement (framewise_displacement)
# - DVARS (std_dvars)
# - CompCor components (a_comp_cor_*, t_comp_cor_*)
# - Cosine drift (cosine*)
# - Non-steady state volumes (non_steady_state_outlier*)

# Select confounds for regression
confound_cols = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'csf', 'white_matter',
    'framewise_displacement'
]

# Add cosine drift regressors
cosine_cols = [col for col in confounds.columns if 'cosine' in col]
confound_cols.extend(cosine_cols)

# Handle NaNs (first volume for FD)
confounds_subset = confounds[confound_cols].fillna(0)

# Apply confound regression
from nilearn.image import clean_img

cleaned_img = clean_img(
    'sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz',
    confounds=confounds_subset.values,
    detrend=False,  # Already done by fMRIPrep
    standardize='zscore',
    high_pass=None,  # Already done by fMRIPrep
    t_r=2.0
)
```

## Scrubbing High-Motion Volumes

```python
# Identify high-motion volumes
fd_threshold = 0.5  # mm
fd = confounds['framewise_displacement'].fillna(0)
high_motion = fd > fd_threshold

print(f"Volumes with FD > {fd_threshold}: {high_motion.sum()}")
print(f"Percentage: {100 * high_motion.sum() / len(fd):.1f}%")

# Scrubbing (removal)
from nilearn.image import index_img

good_volumes = ~high_motion
cleaned_img = index_img('preprocessed_bold.nii.gz', good_volumes)

# Or use as confound regressor
confounds_subset['motion_outlier'] = high_motion.astype(int)
```

## Multi-Echo fMRI

```bash
# fMRIPrep automatically detects multi-echo data
# BIDS structure:
# sub-01_task-rest_echo-1_bold.nii.gz
# sub-01_task-rest_echo-2_bold.nii.gz
# sub-01_task-rest_echo-3_bold.nii.gz

# Run normally - fMRIPrep handles multi-echo
docker run ... \
    --me-output-echos  # Output individual echoes
```

## Template Flow

fMRIPrep uses TemplateFlow for standard templates:

```bash
# Set TemplateFlow directory
export TEMPLATEFLOW_HOME=/path/to/templateflow

# Or bind in Docker
-v /path/to/templateflow:/templateflow

# Pre-download templates
python -c "from templateflow import api; \
           api.get('MNI152NLin2009cAsym', resolution=2)"
```

## Parallel Processing

```bash
# Process multiple subjects in parallel with GNU parallel
parallel -j 4 \
    docker run ... \
    --participant-label {} \
    ::: 01 02 03 04 05 06 07 08

# Or on SLURM
for subj in 01 02 03; do
    sbatch --export=SUBJECT=${subj} fmriprep_job.sh
done
```

## Troubleshooting

### Memory Issues

```bash
# Reduce memory usage
--low-mem
--mem-mb 16000
--omp-nthreads 1  # Reduce parallelization
```

### FreeSurfer Failures

```bash
# Use pre-run FreeSurfer
--fs-subjects-dir /existing/freesurfer/subjects

# Skip FreeSurfer
--fs-no-reconall

# Debug FreeSurfer
--debug freesurfer
```

### SDC Issues

```bash
# Try SyN-based SDC
--use-syn-sdc

# Or skip SDC
--ignore fieldmaps
```

### Checking Logs

```bash
# Work directory contains detailed logs
ls /work/fmriprep_wf/

# Check crash files
ls /work/fmriprep_wf/crash-*

# View log
cat /output/fmriprep/logs/CITATION.md
```

## Quality Control

### Visual Reports

Open the HTML file for each subject:

```bash
firefox fmriprep/sub-01.html
```

Check:
- Brain extraction quality
- Registration quality
- Motion parameters (FD plot)
- Carpet plot for artifacts
- BOLD-T1w alignment

### MRIQC Integration

Run MRIQC before fMRIPrep for detailed QC:

```bash
docker run -it --rm \
    -v /data:/data:ro \
    -v /output/mriqc:/out \
    nipreps/mriqc:latest \
    /data /out participant
```

## Integration with Claude Code

When helping users with fMRIPrep:

1. **Check BIDS Validity:**
   ```bash
   docker run -it --rm -v /data:/data:ro \
       bids/validator /data
   ```

2. **Verify Docker/Singularity:**
   ```bash
   docker --version
   singularity --version
   ```

3. **Common Issues:**
   - BIDS validation errors
   - FreeSurfer license missing
   - Insufficient memory/disk space
   - Fieldmap issues
   - Motion artifacts

4. **Best Practices:**
   - Always review HTML reports
   - Use work directory on fast storage
   - Clean work directory after success
   - Keep detailed logs
   - Use specific version tags

## Resources

- Documentation: https://fmriprep.org/
- Outputs: https://fmriprep.org/en/stable/outputs.html
- Docker: https://fmriprep.org/en/stable/docker.html
- Singularity: https://fmriprep.org/en/stable/singularity.html
- NeuroStars: https://neurostars.org/ (tag: fmriprep)
- GitHub: https://github.com/nipreps/fmriprep

## Citation

```bibtex
@article{fmriprep,
  title={fMRIPrep: a robust preprocessing pipeline for functional MRI},
  author={Esteban, Oscar and Markiewicz, Christopher J and Blair, Ross W and others},
  journal={Nature Methods},
  volume={16},
  pages={111--116},
  year={2019},
  publisher={Nature Publishing Group}
}
```

## Related Tools

- **MRIQC:** Quality control metrics
- **QSIPrep:** Diffusion MRI preprocessing (similar to fMRIPrep)
- **NiPyPe:** Underlying workflow engine
- **Nibabies:** fMRIPrep for infant data
- **FreeSurfer:** Surface reconstruction
