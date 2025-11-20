# xcpEngine - Postprocessing and Quality Assessment for fMRI

## Overview

xcpEngine is a modular, flexible postprocessing framework for functional MRI data, designed to work downstream of minimal preprocessing pipelines like fMRIPrep. It specializes in advanced denoising, confound regression, quality control, and deriving functional connectivity measures. xcpEngine's strength lies in its extensive library of denoising strategies, allowing researchers to apply and compare different approaches to minimize motion artifacts and physiological noise while preserving signal of interest.

**Website:** https://xcpengine.readthedocs.io/
**Platform:** Linux (Docker/Singularity)
**Language:** Bash/Python
**License:** MIT

## Key Features

- Modular postprocessing pipeline
- 40+ validated denoising strategies
- Works with fMRIPrep, C-PAC, or other minimal preprocessing
- Confound regression (motion, CompCor, global signal)
- Temporal filtering (band-pass, high-pass, low-pass)
- Censoring/scrubbing high-motion frames
- Quality control metrics (QC-FC, QCFC-BOLD)
- Functional connectivity matrices
- Network analysis and centrality
- ALFF/ReHo computation
- Seed-based correlation
- Extensive quality control reports
- Parallel processing support
- Docker/Singularity containers

## Installation

### Docker (Recommended)

```bash
# Pull xcpEngine Docker image
docker pull pennbbl/xcpengine:latest

# Or specific version
docker pull pennbbl/xcpengine:1.2.4

# Verify installation
docker run -it pennbbl/xcpengine:latest xcpEngine --version
```

### Singularity

```bash
# Build from Docker Hub
singularity build xcpengine.sif docker://pennbbl/xcpengine:latest

# Run
singularity run xcpengine.sif xcpEngine --version
```

### Native Installation (Advanced)

```bash
# Clone repository
git clone https://github.com/PennBBL/xcpEngine.git
cd xcpEngine

# Install dependencies (requires FSL, AFNI, ANTs)
./install.sh

# Add to PATH
export XCPEDIR=/path/to/xcpEngine
export PATH=${XCPEDIR}:${PATH}
```

## Prerequisites

### Input from fMRIPrep

```bash
# xcpEngine expects fMRIPrep-style outputs
fmriprep_output/
├── sub-01/
│   ├── anat/
│   │   ├── sub-01_desc-preproc_T1w.nii.gz
│   │   ├── sub-01_desc-brain_mask.nii.gz
│   │   └── sub-01_dseg.nii.gz
│   └── func/
│       ├── sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
│       ├── sub-01_task-rest_desc-confounds_timeseries.tsv
│       └── sub-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
```

### Cohort File

```bash
# Create cohort file listing all subjects/sessions
# Format: id0,img,mask

cat > cohort.csv << EOF
id0,img
sub-01,/data/fmriprep/sub-01/func/sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
sub-02,/data/fmriprep/sub-02/func/sub-02_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
sub-03,/data/fmriprep/sub-03/func/sub-03_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
EOF

# With explicit masks
cat > cohort_with_mask.csv << EOF
id0,img,mask
sub-01,/data/fmriprep/sub-01/func/sub-01_task-rest_desc-preproc_bold.nii.gz,/data/fmriprep/sub-01/func/sub-01_desc-brain_mask.nii.gz
EOF
```

## Design Files (Pipelines)

### Built-in Designs

```bash
# xcpEngine includes pre-configured designs:
# - fc-36p: 36-parameter nuisance regression
# - fc-24p: 24-parameter regression
# - fc-aroma: ICA-AROMA denoising
# - fc-acompcor: aCompCor (5 components)
# - fc-scrub: Volume censoring
# - qc-fc: Quality control focused

# Location of designs
ls ${XCPEDIR}/designs/
```

### Design File Structure

```bash
# Design files define processing modules

# Example: fc-36p.dsn
#!/usr/bin/env bash

###############################################################
# Design: 36-parameter confound regression + bandpass
###############################################################

# Processing sequence
sequence=prestats,confound2,regress,fcon,reho,alff,seed,roiquant,norm,qcfc

# Module: prestats (preprocessing)
prestats_process[sub-01]=BOOL
prestats_rerun[sub-01]=N

# Module: confound (confound regression)
confound2_rps[sub-01]=1
confound2_rms[sub-01]=1
confound2_gm[sub-01]=0
confound2_wm[sub-01]=1
confound2_csf[sub-01]=1
confound2_gsr[sub-01]=1
confound2_acompcor[sub-01]=5
confound2_tcompcor[sub-01]=0
confound2_aroma[sub-01]=0
confound2_past[sub-01]=1
confound2_dx[sub-01]=1
confound2_sq[sub-01]=1

# Module: regress (apply regression)
regress_tmpf[sub-01]=butterworth
regress_hipass[sub-01]=0.01
regress_lopass[sub-01]=0.08
regress_censor[sub-01]=0
regress_censor_contig[sub-01]=0
regress_framewise[sub-01]=fds:0.167,dv:2

# Module: fcon (functional connectivity)
fcon_atlas[sub-01]=power264,gordon333,schaefer400

# Module: reho (regional homogeneity)
reho_nhood[sub-01]=vertices

# Module: alff (ALFF/fALFF)
alff_hipass[sub-01]=0.01
alff_lopass[sub-01]=0.08

# Module: seed (seed connectivity)
seed_lib[sub-01]=${XCPEDIR}/seeds/PCC.nii.gz
```

## Basic Usage

### Single Subject

```bash
# Run xcpEngine for one subject
docker run --rm \
  -v /data/fmriprep:/data:ro \
  -v /data/xcp_output:/output \
  -v /tmp:/tmp \
  pennbbl/xcpengine:latest \
  -d /xcpEngine/designs/fc-36p.dsn \
  -c /data/cohort.csv \
  -o /output \
  -i sub-01 \
  -t 1

# Parameters:
#   -d: Design file
#   -c: Cohort file
#   -o: Output directory
#   -i: Subject ID
#   -t: Number of threads
```

### Multiple Subjects (Parallel)

```bash
# Process all subjects in cohort
docker run --rm \
  -v /data/fmriprep:/data:ro \
  -v /data/xcp_output:/output \
  pennbbl/xcpengine:latest \
  -d /xcpEngine/designs/fc-36p.dsn \
  -c /data/cohort.csv \
  -o /output \
  -t 8

# Use job scheduler for large cohorts
for subj in sub-01 sub-02 sub-03; do
    sbatch run_xcp.sh ${subj}
done
```

## Denoising Strategies

### 36-Parameter Model

```bash
# Friston 24-parameter + derivatives + squared terms
# - 6 motion parameters
# - 6 temporal derivatives
# - 12 squared terms
# + WM, CSF, global signal (optional)

# Use design: fc-36p.dsn
docker run --rm \
  -v /data:/data \
  -v /output:/output \
  pennbbl/xcpengine:latest \
  -d /xcpEngine/designs/fc-36p.dsn \
  -c /data/cohort.csv \
  -o /output \
  -t 4
```

### aCompCor Strategy

```bash
# Anatomical CompCor (principal components from WM/CSF)
# No global signal regression

# Use design: fc-acompcor.dsn
# Or customize:

confound2_wm[sub-01]=1
confound2_csf[sub-01]=1
confound2_gsr[sub-01]=0
confound2_acompcor[sub-01]=5  # 5 components
```

### ICA-AROMA

```bash
# Use ICA-AROMA classified components

# Requires fMRIPrep with ICA-AROMA option
fmriprep --use-aroma ...

# Use design: fc-aroma.dsn
confound2_aroma[sub-01]=1
```

### Scrubbing (Censoring)

```bash
# Remove high-motion volumes

# Power scrubbing (FD > 0.5mm, DVARS outliers)
regress_censor[sub-01]=1
regress_framewise[sub-01]=fds:0.5,dv:5
regress_censor_contig[sub-01]=5  # Require 5 contiguous volumes

# Interpolate censored frames (not recommended)
regress_censor_interpolate[sub-01]=1
```

## Functional Connectivity

### ROI-to-ROI Connectivity

```bash
# Compute connectivity matrices

# Specify atlases in design file
fcon_atlas[sub-01]=power264,gordon333,schaefer400

# Or custom atlas
fcon_atlas[sub-01]=/path/to/my_atlas.nii.gz

# Output: Correlation matrices
# - <subject>_<atlas>_network.txt
# - Z-scored Fisher transformed correlations
```

### Seed-Based Connectivity

```bash
# Define seed regions

# In design file
seed_lib[sub-01]=${XCPEDIR}/seeds/PCC.nii.gz,${XCPEDIR}/seeds/motor.nii.gz

# Or custom seeds
seed_lib[sub-01]=/my_seeds/seed1.nii.gz,/my_seeds/seed2.nii.gz

# Output: Seed correlation maps
# - <subject>_<seed>_connectivity.nii.gz
```

## Quality Control

### QC Metrics

```bash
# xcpEngine computes extensive QC metrics:

# Motion:
# - Mean framewise displacement (FD)
# - Mean relative RMS displacement
# - Percent volumes with FD > threshold
# - Number of contiguous volumes

# Quality-connectivity relationships:
# - QC-FC correlation
# - Distance-dependent QC effects

# Signal quality:
# - tSNR (temporal signal-to-noise ratio)
# - Mean DVARS
# - Outlier frames
```

### QC Reports

```bash
# Output structure
xcp_output/
├── sub-01/
│   ├── fcon/
│   │   ├── sub-01_power264_network.txt
│   │   └── sub-01_schaefer400_network.txt
│   ├── qcfc/
│   │   ├── sub-01_qcfc.csv
│   │   └── sub-01_qcDistanceDependence.txt
│   ├── regress/
│   │   ├── sub-01_preprocessed.nii.gz
│   │   └── sub-01_confmat.1D
│   └── sub-01_quality.csv
└── group/
    ├── n36_quality.csv
    └── n36_qc-fc_summary.txt

# View QC CSV
# Contains: meanFD, relMeanRMS, nVolumeCensored, tSNR, etc.
```

### Visualize Quality Metrics

```bash
# Generate QC summary plots
python << EOF
import pandas as pd
import matplotlib.pyplot as plt

# Load QC data
qc = pd.read_csv('xcp_output/group/n36_quality.csv')

# Plot FD distribution
plt.figure(figsize=(10, 6))
plt.hist(qc['meanFD'], bins=30, edgecolor='black')
plt.xlabel('Mean Framewise Displacement (mm)')
plt.ylabel('Number of Subjects')
plt.title('Motion Distribution')
plt.axvline(0.2, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.savefig('motion_distribution.png')

# Identify high-motion subjects
high_motion = qc[qc['meanFD'] > 0.5]
print(f'High motion subjects: {list(high_motion["id0"])}')
EOF
```

## Custom Designs

### Create Custom Design

```bash
# Start from existing design
cp ${XCPEDIR}/designs/fc-36p.dsn my_custom.dsn

# Edit modules and parameters
vim my_custom.dsn

# Key decisions:
# 1. Confound strategy
# 2. Temporal filtering
# 3. Censoring threshold
# 4. Atlases for connectivity
# 5. Additional derivatives (ALFF, ReHo, etc.)

# Example custom strategy
confound2_rps[cxt]=1        # Realignment parameters
confound2_rms[cxt]=0        # RMS displacement
confound2_gm[cxt]=0         # Gray matter
confound2_wm[cxt]=1         # White matter
confound2_csf[cxt]=1        # CSF
confound2_gsr[cxt]=0        # No global signal regression
confound2_acompcor[cxt]=5   # 5 aCompCor components
confound2_past[cxt]=1       # Include derivatives
confound2_dx[cxt]=1         # Include derivatives
confound2_sq[cxt]=1         # Include squared terms

regress_tmpf[cxt]=butterworth
regress_hipass[cxt]=0.008   # 0.008 Hz high-pass
regress_lopass[cxt]=0.10    # 0.10 Hz low-pass
regress_censor[cxt]=1       # Enable censoring
regress_framewise[cxt]=fds:0.25,dv:3  # FD > 0.25mm or DVARS outlier
```

## Batch Processing

### SLURM Array Job

```bash
#!/bin/bash
#SBATCH --job-name=xcpengine
#SBATCH --array=1-50%10  # 50 subjects, max 10 concurrent
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=4:00:00

# Get subject from cohort
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)

# Run xcpEngine
singularity run \
  -B /data:/data \
  -B /scratch:/scratch \
  xcpengine.sif \
  -d /designs/my_design.dsn \
  -c /data/cohort.csv \
  -o /scratch/xcp_output \
  -i ${SUBJECT} \
  -t 4
```

### Compare Multiple Strategies

```bash
# Run multiple denoising strategies for comparison
strategies=("fc-36p" "fc-acompcor" "fc-aroma" "fc-scrub")

for strategy in "${strategies[@]}"; do
    echo "Running ${strategy}..."

    docker run --rm \
      -v /data:/data \
      -v /output/${strategy}:/output \
      pennbbl/xcpengine:latest \
      -d /xcpEngine/designs/${strategy}.dsn \
      -c /data/cohort.csv \
      -o /output \
      -t 8

    echo "${strategy} complete"
done

# Compare QC-FC relationships
for strategy in "${strategies[@]}"; do
    echo "=== ${strategy} ==="
    cat /output/${strategy}/group/*_qc-fc_summary.txt
done
```

## Integration with Claude Code

When helping users with xcpEngine:

1. **Check Installation:**
   ```bash
   docker run -it pennbbl/xcpengine:latest xcpEngine --version
   singularity exec xcpengine.sif xcpEngine --version
   ```

2. **Common Issues:**
   - Cohort file formatting errors
   - Missing fMRIPrep confounds file
   - Insufficient temporal filtering buffer
   - Module dependencies not met
   - Path mismatches in cohort file

3. **Best Practices:**
   - Test on single subject first
   - Choose denoising strategy based on literature
   - Compare multiple strategies for your data
   - Check QC-FC relationships
   - Examine censored volume counts
   - Keep intermediate files for debugging
   - Document design file used
   - Version control design files

4. **Parameter Recommendations:**
   - FD threshold: 0.2-0.5mm (stricter for development)
   - Temporal filter: 0.008-0.1 Hz for resting-state
   - Minimum contiguous volumes: 5
   - aCompCor components: 5
   - Censoring: Enable for high-motion cohorts

## Troubleshooting

**Problem:** "Cohort file malformed"
**Solution:** Check CSV format, ensure no spaces in paths, verify file paths exist

**Problem:** "Insufficient degrees of freedom"
**Solution:** Reduce number of confound regressors, or increase scan length

**Problem:** Module fails but others succeed
**Solution:** Check module dependencies, verify atlas files exist, check log files in output/sub-XX/log/

**Problem:** High QC-FC correlation
**Solution:** Try stricter censoring, different confound strategy, or report as limitation

**Problem:** All volumes censored
**Solution:** Loosen FD threshold, check motion parameters, verify preprocessing quality

## Resources

- Website: https://xcpengine.readthedocs.io/
- Documentation: https://xcpengine.readthedocs.io/
- GitHub: https://github.com/PennBBL/xcpEngine
- Docker Hub: https://hub.docker.com/r/pennbbl/xcpengine
- Forum: https://neurostars.org/ (tag: xcpengine)
- Paper: Ciric et al. (2017) Benchmarking denoising strategies

## Citation

```bibtex
@article{ciric2017benchmarking,
  title={Benchmarking of participant-level confound regression strategies for the control of motion artifact in studies of functional connectivity},
  author={Ciric, Rastko and Wolf, Daniel H and Power, Jonathan D and Roalf, David R and Baum, Graham L and Ruparel, Kosha and Shinohara, Russell T and Elliott, Mark A and Eickhoff, Simon B and Davatzikos, Christos and others},
  journal={Neuroimage},
  volume={154},
  pages={174--187},
  year={2017}
}
```

## Related Tools

- **fMRIPrep:** Minimal preprocessing (upstream)
- **C-PAC:** Alternative full pipeline
- **CONN:** Functional connectivity toolbox
- **ICA-AROMA:** Artifact removal (can be integrated)
- **DPABI:** MATLAB-based alternative
- **Nipype:** Underlying workflow engine
