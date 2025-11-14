# C-PAC - Configurable Pipeline for the Analysis of Connectomes

## Overview

C-PAC (Configurable Pipeline for the Analysis of Connectomes) is an open-source, automated preprocessing and analysis pipeline for functional MRI data. Built on Nipype, C-PAC provides a comprehensive, configurable framework for preprocessing anatomical and functional MRI data, computing functional connectivity metrics, and performing quality control. It's designed for large-scale studies and supports both individual-level and group-level analyses with extensive customization options.

**Website:** https://fcp-indi.github.io/
**Platform:** Docker/Singularity (Linux/macOS/Windows)
**Language:** Python (Nipype-based)
**License:** LGPL-3.0

## Key Features

- Comprehensive fMRI preprocessing pipeline
- Anatomical and functional data processing
- Multiple registration strategies (ANTs, FSL)
- Nuisance regression with multiple strategies
- Functional connectivity metrics (seed-based, ROI-to-ROI)
- Network centrality measures
- ALFF/fALFF, ReHo, VMHC
- Quality control and visualization
- BIDS compatibility
- Flexible configuration system
- Parallel processing support
- Container-based deployment (Docker/Singularity)
- Integration with AWS and HPC clusters
- Extensive documentation and presets

## Installation

### Docker (Recommended)

```bash
# Pull C-PAC Docker image
docker pull fcpindi/c-pac:latest

# Or specific version
docker pull fcpindi/c-pac:v1.8.6

# Verify installation
docker run -it fcpindi/c-pac:latest --version

# Run C-PAC
docker run -i --rm \
  -v /path/to/data:/data:ro \
  -v /path/to/output:/output \
  fcpindi/c-pac:latest \
  /data /output participant
```

### Singularity

```bash
# Build Singularity image from Docker
singularity build c-pac.sif docker://fcpindi/c-pac:latest

# Run with Singularity
singularity run \
  -B /path/to/data:/data:ro \
  -B /path/to/output:/output \
  c-pac.sif \
  /data /output participant
```

### Python (Advanced)

```bash
# Install via pip (for developers)
pip install cpac

# Or from source
git clone https://github.com/FCP-INDI/C-PAC.git
cd C-PAC
pip install -e .

# Requires: FSL, AFNI, ANTs installed separately
```

## Data Preparation

### BIDS Format

```bash
# C-PAC works best with BIDS-formatted data
bids_dataset/
├── dataset_description.json
├── participants.tsv
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   └── func/
│       ├── sub-01_task-rest_bold.nii.gz
│       └── sub-01_task-rest_bold.json
├── sub-02/
│   ├── anat/
│   │   └── sub-02_T1w.nii.gz
│   └── func/
│       ├── sub-02_task-rest_bold.nii.gz
│       └── sub-02_task-rest_bold.json
...
```

### Non-BIDS Data

```bash
# C-PAC can work with custom layouts
# Define data_config.yml to specify file paths

data_config.yml:
  subjects:
    - subject_id: sub-01
      unique_id: session_1
      anat: /data/sub-01/T1w.nii.gz
      func: /data/sub-01/rest.nii.gz
    - subject_id: sub-02
      unique_id: session_1
      anat: /data/sub-02/T1w.nii.gz
      func: /data/sub-02/rest.nii.gz
```

## Basic Usage

### Quick Start with Default Pipeline

```bash
# Process BIDS dataset with default settings
docker run -i --rm \
  -v /data/bids_dataset:/bids_dir:ro \
  -v /data/outputs:/output \
  -v /tmp:/tmp \
  fcpindi/c-pac:latest \
  /bids_dir /output participant

# Parameters:
#   /bids_dir: Input BIDS directory
#   /output: Output directory
#   participant: Analysis level (participant or group)
```

### Process Specific Subjects

```bash
# Process only specific participants
docker run -i --rm \
  -v /data/bids_dataset:/bids_dir:ro \
  -v /data/outputs:/output \
  fcpindi/c-pac:latest \
  /bids_dir /output participant \
  --participant-label sub-01 sub-02 sub-03

# Use multiple cores
docker run -i --rm \
  -v /data/bids_dataset:/bids_dir:ro \
  -v /data/outputs:/output \
  fcpindi/c-pac:latest \
  /bids_dir /output participant \
  --n_cpus 8 \
  --mem_gb 32
```

## Pipeline Configuration

### Using Presets

```bash
# C-PAC includes several presets:
# - default: Standard preprocessing
# - fmriprep-options: fMRIPrep-like configuration
# - benchmark-ANTS: ANTs-based registration
# - benchmark-FNIRT: FSL FNIRT registration
# - preproc: Preprocessing only (no derivatives)
# - monkey: Non-human primate optimized
# - nhp-macaque: Macaque-specific
# - rodent: Rodent-optimized

# Use preset
docker run -i --rm \
  -v /data/bids_dataset:/bids_dir:ro \
  -v /data/outputs:/output \
  fcpindi/c-pac:latest \
  /bids_dir /output participant \
  --preconfig fmriprep-options
```

### Custom Pipeline Configuration

```bash
# Generate default configuration
docker run -i --rm \
  fcpindi/c-pac:latest \
  --save-config /tmp/default_config.yml

# Edit configuration file (pipeline_config.yml)
# Key sections:
#   - anatomical_preproc: T1 processing
#   - functional_preproc: fMRI preprocessing
#   - registration: Registration strategies
#   - nuisance_corrections: Denoising
#   - timeseries_extraction: ROI time series
#   - seed_based_correlation: FC analysis

# Run with custom config
docker run -i --rm \
  -v /data/bids_dataset:/bids_dir:ro \
  -v /data/outputs:/output \
  -v /configs/pipeline_config.yml:/config.yml:ro \
  fcpindi/c-pac:latest \
  /bids_dir /output participant \
  --pipeline-file /config.yml
```

## Pipeline Components

### Anatomical Preprocessing

```yaml
# Configuration: anatomical_preproc

# Brain extraction
brain_extraction:
  run: [On]
  method: [AFNI]  # or ANTs, FSL, UNet, niworkflows-ants

# Tissue segmentation
segmentation:
  run: [On]
  tissue_segmentation: [FSL-FAST]  # or ANTs Prior-Based

# Registration
registration:
  anatomical_template_resolution: [1mm]
  template_brain_only_for_anat: [Off]
  template_for_resample: /path/to/MNI152_T1_1mm_brain.nii.gz
```

### Functional Preprocessing

```yaml
# Configuration: functional_preproc

# Slice timing correction
slice_timing_correction:
  run: [On]
  first_timepoint: 0
  last_timepoint: null
  pattern: alt+z  # or seq+z, seq-z, alt+z2, etc.

# Motion correction
motion_estimates_and_correction:
  run: [On]
  motion_correction_reference: mean  # or median, selected_volume
  motion_correction_tool: [3dvolreg]  # or mcflirt

# Distortion correction
distortion_correction:
  run: [Off]  # Requires fieldmap data

# Despiking
despiking:
  run: [Off]

# Smoothing
spatial_smoothing:
  run: [On]
  kernel_FWHM: [4, 6]  # Can run multiple smoothing kernels

# Intensity normalization
intensity_normalization:
  run: [On]
```

### Registration Workflows

```yaml
# Registration to standard space

registration_workflows:
  anatomical_registration:
    resolution_for_anat: 1mm
    T1_template: /templates/MNI152_T1_1mm.nii.gz
    T1_template_brain: /templates/MNI152_T1_1mm_brain.nii.gz

  functional_registration:
    coregistration_method: [FSL]  # or ANTs
    func_registration_to_template: [T1_template]  # or EPI_template

  registration_method:
    FSL-FNIRT:
      run: [On]
      config_file: T1_2_MNI152_2mm
    ANTs:
      run: [Off]
      interpolation: LanczosWindowedSinc
```

### Nuisance Regression

```yaml
# Nuisance signal regression strategies

nuisance_corrections:
  2-nuisance_regression:
    run: [On]

    Regressors:
      - Name: Regressor-1
        Motion:
          include_delayed: Off
          include_squared: Off
          include_delayed_squared: Off
        aCompCor:
          summary:
            method: DetrendPC
            components: 5
          tissues: [WhiteMatter, CerebrospinalFluid]
          extraction_resolution: 2
        GlobalSignal:
          summary: Mean

      - Name: Regressor-2-CompCor-36P
        Motion:
          include_delayed: On
          include_squared: On
          include_delayed_squared: On
        aCompCor:
          summary:
            method: DetrendPC
            components: 5
          tissues: [WhiteMatter, CerebrospinalFluid]
```

### Functional Connectivity

```yaml
# Seed-based correlation analysis

seed_based_correlation_analysis:
  run: [On]
  sca_roi_paths:
    - /path/to/seeds/PCC_seed.nii.gz
    - /path/to/seeds/motor_seed.nii.gz

# ROI timeseries extraction
timeseries_extraction:
  run: [On]
  roi_paths:
    - /path/to/atlases/AAL_116.nii.gz
    - /path/to/atlases/CC200.nii.gz
  outputs_within_mask: [False]
```

## Derivatives and Outputs

### Output Structure

```bash
# C-PAC outputs organized by pipeline and strategy
output/
├── pipeline_default/
│   ├── sub-01_ses-01/
│   │   ├── anat/
│   │   │   ├── sub-01_desc-brain_T1w.nii.gz
│   │   │   ├── sub-01_desc-preproc_T1w.nii.gz
│   │   │   └── sub-01_space-MNI152_desc-brain_T1w.nii.gz
│   │   ├── func/
│   │   │   ├── sub-01_desc-preproc_bold.nii.gz
│   │   │   ├── sub-01_desc-brain_mask.nii.gz
│   │   │   └── sub-01_desc-motion_timeseries.1D
│   │   ├── alff/
│   │   │   ├── sub-01_alff.nii.gz
│   │   │   └── sub-01_falff.nii.gz
│   │   ├── reho/
│   │   │   └── sub-01_reho.nii.gz
│   │   ├── sca_roi/
│   │   │   └── sub-01_PCC_connectivity.nii.gz
│   │   └── qc/
│   │       ├── qc_motion.html
│   │       └── qc_registration.html
│   └── sub-02_ses-01/
│       └── ...
└── log/
    └── pipeline_default/
        └── sub-01_ses-01/
            └── cpac_individual_timing_sub-01.csv
```

### Quality Control Reports

```bash
# C-PAC generates comprehensive QC reports

# Motion QC
# - Framewise displacement plots
# - DVARS
# - Mean FD statistics
# - Outlier detection

# Registration QC
# - Anatomical-to-template overlay
# - Functional-to-anatomical overlay
# - Tissue segmentation overlay

# Access QC reports
open output/pipeline_default/sub-01/qc/qc_motion.html
open output/pipeline_default/sub-01/qc/qc_registration.html
```

## Advanced Features

### Group-Level Analysis

```bash
# After individual preprocessing, run group analysis
docker run -i --rm \
  -v /data/outputs:/output \
  fcpindi/c-pac:latest \
  /output group \
  --pipeline-file /config.yml
```

### Custom Data Configuration

```yaml
# data_config.yml for non-BIDS data

participants:
  - subject_id: sub-01
    unique_id: ses-01
    anat: /data/sub-01/anat/T1w.nii.gz
    func:
      rest:
        scan: /data/sub-01/func/rest.nii.gz
        scan_parameters:
          tr: 2.0
          acquisition: alt+z
          reference_frame: 0
  - subject_id: sub-02
    unique_id: ses-01
    anat: /data/sub-02/anat/T1w.nii.gz
    func:
      rest:
        scan: /data/sub-02/func/rest.nii.gz

# Run with data config
docker run -i --rm \
  -v /data:/data:ro \
  -v /data/outputs:/output \
  -v /configs/data_config.yml:/data_config.yml:ro \
  fcpindi/c-pac:latest \
  /data_config.yml /output participant
```

### Crash Recovery

```bash
# C-PAC can recover from crashes and continue processing

# Enable crash recovery in config
working_directory:
  path: /tmp/cpac_working
  remove_working_dir: Off  # Keep for crash recovery

# Resume after crash
docker run -i --rm \
  -v /data/bids_dataset:/bids_dir:ro \
  -v /data/outputs:/output \
  -v /tmp/cpac_working:/tmp \
  fcpindi/c-pac:latest \
  /bids_dir /output participant \
  --pipeline-file /config.yml
```

## Performance Optimization

### Parallel Processing

```bash
# Utilize multiple CPUs and memory
docker run -i --rm \
  -v /data/bids_dataset:/bids_dir:ro \
  -v /data/outputs:/output \
  fcpindi/c-pac:latest \
  /bids_dir /output participant \
  --n_cpus 16 \
  --mem_gb 64 \
  --num_ants_threads 4

# Control resource usage per participant
pipeline_setup:
  system_config:
    max_cores_per_participant: 4
    num_ants_threads: 2
    num_participants_at_once: 4
```

### HPC and Cloud Deployment

```bash
# AWS S3 integration
docker run -i --rm \
  -v ~/.aws:/root/.aws:ro \
  fcpindi/c-pac:latest \
  s3://bucket/bids_dataset s3://bucket/outputs participant

# SLURM job submission
#!/bin/bash
#SBATCH --job-name=cpac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

singularity run \
  -B /scratch/data:/data:ro \
  -B /scratch/outputs:/output \
  c-pac.sif \
  /data /output participant \
  --n_cpus 16 \
  --mem_gb 64 \
  --participant-label ${SUBJECT}
```

## Integration with Claude Code

When helping users with C-PAC:

1. **Check Installation:**
   ```bash
   docker run -it fcpindi/c-pac:latest --version
   # or
   singularity exec c-pac.sif cpac --version
   ```

2. **Common Issues:**
   - BIDS validation errors (run bids-validator first)
   - Memory errors (increase --mem_gb)
   - Permissions issues (check volume mounts)
   - Missing templates (ensure downloaded)
   - Registration failures (check image quality)

3. **Best Practices:**
   - Start with default or fmriprep-options preset
   - Always validate BIDS format first
   - Run on subset first to test configuration
   - Review QC reports for all subjects
   - Use crash recovery for large studies
   - Document pipeline configuration
   - Version control config files
   - Keep working directory for debugging

4. **Parameter Selection:**
   - Smoothing: 4-6mm FWHM typical
   - Band-pass: 0.01-0.1 Hz for resting-state
   - Nuisance regression: Start with CompCor-36P
   - Registration: ANTs for best quality, FSL for speed
   - Brain extraction: Try multiple if one fails

## Troubleshooting

**Problem:** "BIDS validation failed"
**Solution:** Run bids-validator separately, fix dataset structure, or use data_config.yml

**Problem:** Brain extraction failure
**Solution:** Try different method (AFNI, ANTs, UNet), adjust parameters, or manually provide mask

**Problem:** Registration poor quality
**Solution:** Check anatomical quality, try ANTs instead of FSL, adjust registration config

**Problem:** Out of memory
**Solution:** Increase --mem_gb, reduce num_participants_at_once, or run fewer strategies

**Problem:** Container cannot access files
**Solution:** Check volume mounts (-v), ensure read/write permissions, use absolute paths

## Resources

- Website: https://fcp-indi.github.io/
- Documentation: https://fcp-indi.github.io/docs/latest
- GitHub: https://github.com/FCP-INDI/C-PAC
- Forum: https://neurostars.org/tag/cpac
- User Guide: https://fcp-indi.github.io/docs/latest/user
- Configuration Guide: https://fcp-indi.github.io/docs/latest/user/pipelines/pipeline_config
- Docker Hub: https://hub.docker.com/r/fcpindi/c-pac

## Citation

```bibtex
@article{craddock2013c,
  title={Towards automated analysis of connectomes: The configurable pipeline for the analysis of connectomes (C-PAC)},
  author={Craddock, Cameron and Sikka, Sharad and Cheung, Brian and Khanuja, Ranjeet and Ghosh, Satrajit S and Yan, Chaogan and Li, Qingyang and Lurie, Daniel and Vogelstein, Joshua and Burns, Randal and others},
  journal={Front Neuroinform},
  volume={42},
  year={2013}
}
```

## Related Tools

- **fMRIPrep:** Alternative preprocessing pipeline
- **DPABI:** MATLAB-based pipeline
- **CONN:** Functional connectivity toolbox
- **xcpEngine:** Post-fMRIPrep processing
- **Nipype:** Underlying workflow engine
- **BIDS Apps:** Other BIDS-compatible pipelines
