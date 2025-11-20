# ASLPrep: BIDS-Compliant ASL Preprocessing Pipeline

## Overview

ASLPrep is a robust and easy-to-use pipeline for preprocessing Arterial Spin Labeling (ASL) MRI data. Built on the fMRIPrep framework, ASLPrep provides a BIDS-compliant workflow for ASL preprocessing, cerebral blood flow (CBF) quantification, and quality control reporting.

**Key Features:**
- **BIDS Compliance**: Native support for BIDS-formatted ASL datasets
- **Robust Preprocessing**: Anatomical and ASL-specific preprocessing workflows
- **CBF Quantification**: Multiple methods for CBF computation with proper calibration
- **Quality Control**: Comprehensive visual reports with quality metrics
- **Minimal User Input**: Sensible defaults with extensive customization options
- **Containerized**: Available as Docker/Singularity containers for reproducibility

**Website:** https://aslprep.readthedocs.io/

**Citation:** Adebimpe, A., et al. (2022). ASLPrep: A Platform for Processing of Arterial Spin Labeled MRI and Quantification of Regional Brain Perfusion. *Nature Methods*.

## Installation

### Using Docker (Recommended)

```bash
# Pull the latest ASLPrep Docker image
docker pull pennlinc/aslprep:latest

# Verify installation
docker run --rm pennlinc/aslprep:latest --version
```

### Using Singularity

```bash
# Build Singularity image from Docker Hub
singularity build aslprep-latest.simg docker://pennlinc/aslprep:latest

# Verify installation
singularity run aslprep-latest.simg --version
```

### Using pip (Development)

```bash
# Install from PyPI
pip install aslprep

# Or install from GitHub for latest development version
pip install git+https://github.com/PennLINC/aslprep.git
```

### Requirements

- **Input**: BIDS-formatted ASL dataset with structural (T1w/T2w) scans
- **FreeSurfer License**: Required for cortical surface reconstruction
- **Computational Resources**: Recommended 16GB RAM, 4+ CPU cores

## Basic Usage

### Preparing a BIDS Dataset

ASLPrep expects data organized according to BIDS specifications:

```
my_dataset/
├── dataset_description.json
├── participants.tsv
├── sub-01/
│   ├── anat/
│   │   ├── sub-01_T1w.nii.gz
│   │   └── sub-01_T1w.json
│   └── perf/
│       ├── sub-01_asl.nii.gz
│       ├── sub-01_asl.json
│       ├── sub-01_aslcontext.tsv
│       └── sub-01_m0scan.nii.gz
└── sub-02/
    └── ...
```

**aslcontext.tsv** (critical for ASL sequence specification):

```
volume_type
control
label
control
label
m0scan
```

### Running ASLPrep with Docker

```bash
# Basic ASLPrep command
docker run --rm -it \
  -v /path/to/data:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/freesurfer_license.txt:/opt/freesurfer/license.txt:ro \
  pennlinc/aslprep:latest \
  /data /out participant \
  --participant-label 01 \
  --fs-license-file /opt/freesurfer/license.txt

# Multi-subject processing
docker run --rm -it \
  -v /path/to/data:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/license.txt:/opt/freesurfer/license.txt:ro \
  pennlinc/aslprep:latest \
  /data /out participant \
  --fs-license-file /opt/freesurfer/license.txt \
  --nthreads 8 \
  --omp-nthreads 4
```

### Running ASLPrep with Singularity

```bash
# Basic Singularity command
singularity run --cleanenv \
  -B /path/to/data:/data:ro \
  -B /path/to/output:/out \
  -B /path/to/license.txt:/opt/freesurfer/license.txt:ro \
  aslprep-latest.simg \
  /data /out participant \
  --participant-label 01 \
  --fs-license-file /opt/freesurfer/license.txt

# HPC cluster with job scheduler
singularity run --cleanenv \
  -B ${BIDS_DIR}:/data:ro \
  -B ${OUTPUT_DIR}:/out \
  -B ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
  ${SINGULARITY_IMG} \
  /data /out participant \
  --participant-label ${SUBJECT_ID} \
  --fs-license-file /opt/freesurfer/license.txt \
  --mem-mb 16000 \
  --nthreads ${NSLOTS}
```

## ASL-Specific Preprocessing

### Specifying ASL Acquisition Parameters

Critical ASL parameters should be in the `*_asl.json` sidecar:

```json
{
  "MagneticFieldStrength": 3,
  "MRAcquisitionType": "3D",
  "ArterialSpinLabelingType": "PCASL",
  "PostLabelingDelay": 1.8,
  "LabelingDuration": 1.8,
  "BackgroundSuppression": true,
  "M0Type": "Separate",
  "M0Estimate": 1.0,
  "TotalAcquiredPairs": 40,
  "RepetitionTimePreparation": 4.0,
  "FlipAngle": 90,
  "EchoTime": 0.014,
  "VascularCrushing": true,
  "VascularCrushingVENC": 2
}
```

### Multi-PLD (Post-Labeling Delay) Processing

For multi-timepoint ASL acquisitions:

**aslcontext.tsv:**

```
volume_type
control
label
control
label
control
label
```

**asl.json with multiple PLDs:**

```json
{
  "ArterialSpinLabelingType": "PCASL",
  "PostLabelingDelay": [1.0, 1.5, 2.0, 2.5],
  "LabelingDuration": 1.8,
  "M0Type": "Separate",
  "RepetitionTimePreparation": 5.0
}
```

### CBF Quantification Methods

```bash
# Specify CBF computation method
docker run --rm -it \
  -v ${BIDS_DIR}:/data:ro \
  -v ${OUTPUT_DIR}:/out \
  -v ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
  pennlinc/aslprep:latest \
  /data /out participant \
  --participant-label 01 \
  --fs-license-file /opt/freesurfer/license.txt \
  --cbf-computation generalkinetic \
  --m0-scale 1.0

# Available methods:
# - buxton: Single-compartment Buxton model (default)
# - generalkinetic: General kinetic model for multi-PLD
# - wang: Wang et al. 2002 method
# - juttukonda: Multi-delay model with ATT estimation
```

## Quality Control and Outputs

### Understanding ASLPrep Outputs

```
output/
├── aslprep/
│   ├── dataset_description.json
│   ├── sub-01/
│   │   ├── anat/
│   │   │   ├── sub-01_desc-preproc_T1w.nii.gz
│   │   │   ├── sub-01_desc-brain_mask.nii.gz
│   │   │   └── sub-01_dseg.nii.gz
│   │   └── perf/
│   │       ├── sub-01_space-T1w_cbf.nii.gz
│   │       ├── sub-01_space-T1w_mean_cbf.nii.gz
│   │       ├── sub-01_space-T1w_att.nii.gz
│   │       ├── sub-01_space-MNI152NLin2009cAsym_cbf.nii.gz
│   │       ├── sub-01_desc-preproc_asl.nii.gz
│   │       └── sub-01_desc-confounds_timeseries.tsv
│   └── sub-01.html
├── freesurfer/
│   └── sub-01/
└── work/
    └── ...
```

### Key Output Files

**CBF Maps:**
- `*_cbf.nii.gz`: 4D CBF timeseries (ml/100g/min)
- `*_mean_cbf.nii.gz`: Mean CBF across all volumes
- `*_att.nii.gz`: Arterial transit time map (seconds)

**Preprocessed ASL:**
- `*_desc-preproc_asl.nii.gz`: Preprocessed ASL timeseries
- `*_desc-confounds_timeseries.tsv`: Nuisance regressors

### Visual Quality Assessment

```bash
# Open HTML report for subject
firefox aslprep/sub-01.html

# Key sections in the report:
# - Summary: Dataset and preprocessing parameters
# - Anatomical: T1w processing and segmentation
# - ASL: Registration quality, CBF maps, and quality metrics
# - CBF: Mean CBF distribution and regional values
# - About: Software versions and citations
```

### Quantitative QC Metrics

```python
import pandas as pd
import nibabel as nib
import numpy as np

# Load CBF map
cbf_img = nib.load('aslprep/sub-01/perf/sub-01_space-T1w_mean_cbf.nii.gz')
cbf_data = cbf_img.get_fdata()

# Load gray matter mask
gm_mask = nib.load('aslprep/sub-01/anat/sub-01_label-GM_probseg.nii.gz')
gm_data = gm_mask.get_fdata() > 0.5

# Calculate mean GM CBF
mean_gm_cbf = np.mean(cbf_data[gm_data])
print(f"Mean GM CBF: {mean_gm_cbf:.2f} ml/100g/min")

# Expected range: 40-80 ml/100g/min for healthy adults
if mean_gm_cbf < 30 or mean_gm_cbf > 100:
    print("Warning: CBF values outside expected range")

# Load confounds
confounds = pd.read_csv('aslprep/sub-01/perf/sub-01_desc-confounds_timeseries.tsv',
                        sep='\t')
print(f"Framewise displacement mean: {confounds['framewise_displacement'].mean():.3f} mm")
```

## Advanced Preprocessing Options

### Custom Output Spaces

```bash
# Multiple output spaces
docker run --rm -it \
  -v ${BIDS_DIR}:/data:ro \
  -v ${OUTPUT_DIR}:/out \
  -v ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
  pennlinc/aslprep:latest \
  /data /out participant \
  --participant-label 01 \
  --fs-license-file /opt/freesurfer/license.txt \
  --output-spaces MNI152NLin2009cAsym:res-2 \
                  MNI152NLin6Asym:res-2 \
                  T1w \
                  fsnative \
                  fsaverage5

# Template specification with resolution
# Format: <template>:res-<resolution>
# Templates: MNI152NLin2009cAsym, MNI152NLin6Asym, MNI152NLin2009aAsym
# Native spaces: T1w, asl
# Surface spaces: fsnative, fsaverage, fsaverage5, fsaverage6
```

### Fieldmap-Based Distortion Correction

```bash
# ASLPrep automatically detects and uses fieldmaps
# Ensure fieldmaps are in BIDS format:
# sub-01/fmap/sub-01_dir-AP_epi.nii.gz
# sub-01/fmap/sub-01_dir-AP_epi.json (with "IntendedFor" field)

docker run --rm -it \
  -v ${BIDS_DIR}:/data:ro \
  -v ${OUTPUT_DIR}:/out \
  -v ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
  pennlinc/aslprep:latest \
  /data /out participant \
  --participant-label 01 \
  --fs-license-file /opt/freesurfer/license.txt \
  --use-syn-sdc  # Use SyN-based SDC if no fieldmaps available
```

### Slice-Timing Correction

```bash
# Enable slice-timing correction for 2D ASL
docker run --rm -it \
  -v ${BIDS_DIR}:/data:ro \
  -v ${OUTPUT_DIR}:/out \
  -v ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
  pennlinc/aslprep:latest \
  /data /out participant \
  --participant-label 01 \
  --fs-license-file /opt/freesurfer/license.txt \
  --ignore slicetiming  # Skip slice-timing if not needed (3D acquisitions)
```

### Motion Correction and Scrubbing

```bash
# Configure motion correction parameters
docker run --rm -it \
  -v ${BIDS_DIR}:/data:ro \
  -v ${OUTPUT_DIR}:/out \
  -v ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
  pennlinc/aslprep:latest \
  /data /out participant \
  --participant-label 01 \
  --fs-license-file /opt/freesurfer/license.txt \
  --fd-threshold 0.5  # Framewise displacement threshold (mm)
```

## Batch Processing

### Processing Multiple Subjects on HPC

**submit_aslprep.sh** (SLURM example):

```bash
#!/bin/bash
#SBATCH --job-name=aslprep
#SBATCH --array=1-50%10
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/aslprep_%A_%a.out

# Load Singularity module
module load singularity

# Get subject ID from participants.tsv
SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" participants.tsv | cut -f1 | sed 's/sub-//')

# Set paths
BIDS_DIR=/project/data/bids
OUTPUT_DIR=/project/data/derivatives/aslprep
WORK_DIR=/scratch/${USER}/aslprep_work
SINGULARITY_IMG=/project/containers/aslprep-latest.simg
FS_LICENSE=/project/software/freesurfer/license.txt

# Create work directory
mkdir -p ${WORK_DIR}

# Run ASLPrep
singularity run --cleanenv \
  -B ${BIDS_DIR}:/data:ro \
  -B ${OUTPUT_DIR}:/out \
  -B ${WORK_DIR}:/work \
  -B ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
  ${SINGULARITY_IMG} \
  /data /out participant \
  --participant-label ${SUBJECT_ID} \
  --fs-license-file /opt/freesurfer/license.txt \
  --work-dir /work \
  --nthreads ${SLURM_CPUS_PER_TASK} \
  --omp-nthreads 4 \
  --mem-mb 16000 \
  --stop-on-first-crash

# Submit job array
sbatch submit_aslprep.sh
```

### Monitoring Progress

```bash
# Check running jobs
squeue -u $USER

# Monitor specific subject
tail -f logs/aslprep_12345_10.out

# Check for errors
grep -i "error\|exception" logs/*.out
```

### Restarting Failed Jobs

```bash
# ASLPrep caches completed steps
# Simply rerun with same work directory
singularity run --cleanenv \
  -B ${BIDS_DIR}:/data:ro \
  -B ${OUTPUT_DIR}:/out \
  -B ${WORK_DIR}:/work \
  -B ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
  ${SINGULARITY_IMG} \
  /data /out participant \
  --participant-label ${SUBJECT_ID} \
  --fs-license-file /opt/freesurfer/license.txt \
  --work-dir /work \
  --notrack  # Disable anonymous usage tracking
```

## Integration with Other Tools

### Integration with BASIL

```python
import subprocess
from pathlib import Path

def run_basil_on_aslprep_output(subject, aslprep_dir, basil_dir):
    """
    Run BASIL on ASLPrep preprocessed data for advanced Bayesian analysis
    """
    # Paths
    asl_file = aslprep_dir / subject / 'perf' / f'{subject}_desc-preproc_asl.nii.gz'
    t1_file = aslprep_dir / subject / 'anat' / f'{subject}_desc-preproc_T1w.nii.gz'
    mask_file = aslprep_dir / subject / 'anat' / f'{subject}_desc-brain_mask.nii.gz'

    output_dir = basil_dir / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read ASL parameters from JSON
    import json
    json_file = asl_file.with_suffix('').with_suffix('.json')
    with open(json_file) as f:
        params = json.load(f)

    # Run BASIL
    cmd = [
        'oxford_asl',
        '-i', str(asl_file),
        '-o', str(output_dir),
        '--iaf=tc',
        f'--tis={params["PostLabelingDelay"]}',
        f'--bolus={params["LabelingDuration"]}',
        '--casl' if params["ArterialSpinLabelingType"] == "PCASL" else '--pasl',
        '--spatial',
        '--inferart',
        '--pvcorr',
        '-c', str(t1_file),
        '-m', str(mask_file)
    ]

    subprocess.run(cmd, check=True)
    print(f"BASIL completed for {subject}")

# Process all subjects
aslprep_dir = Path('derivatives/aslprep')
basil_dir = Path('derivatives/basil')
for subject_dir in aslprep_dir.glob('sub-*'):
    subject = subject_dir.name
    run_basil_on_aslprep_output(subject, aslprep_dir, basil_dir)
```

### Integration with fMRIPrep Outputs

```python
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

def visualize_aslprep_fmriprep_overlay(subject, aslprep_dir, fmriprep_dir):
    """
    Overlay ASLPrep CBF on fMRIPrep structural outputs
    """
    # Load fMRIPrep structural
    t1_file = fmriprep_dir / subject / 'anat' / f'{subject}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

    # Load ASLPrep CBF in same space
    cbf_file = aslprep_dir / subject / 'perf' / f'{subject}_space-MNI152NLin2009cAsym_mean_cbf.nii.gz'

    # Create overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    display = plotting.plot_stat_map(
        str(cbf_file),
        bg_img=str(t1_file),
        threshold=30,
        vmax=80,
        cmap='hot',
        title='Mean CBF (ml/100g/min)',
        axes=axes[0],
        cut_coords=(-20, 0, 20)
    )

    plt.savefig(f'{subject}_cbf_overlay.png', dpi=300, bbox_inches='tight')
    print(f"Saved overlay for {subject}")
```

### Extracting Regional CBF Values

```python
import pandas as pd
import nibabel as nib
from nilearn import datasets, image
import numpy as np

def extract_regional_cbf(subject, aslprep_dir, atlas='aal'):
    """
    Extract mean CBF values from brain regions using an atlas
    """
    # Load CBF map in MNI space
    cbf_file = aslprep_dir / subject / 'perf' / f'{subject}_space-MNI152NLin2009cAsym_mean_cbf.nii.gz'
    cbf_img = nib.load(cbf_file)

    # Load atlas
    if atlas == 'aal':
        atlas_data = datasets.fetch_atlas_aal()
        atlas_img = nib.load(atlas_data.maps)
        labels = atlas_data.labels
    elif atlas == 'harvard_oxford':
        atlas_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = nib.load(atlas_data.maps)
        labels = atlas_data.labels

    # Resample atlas to CBF space
    atlas_resampled = image.resample_to_img(atlas_img, cbf_img, interpolation='nearest')
    atlas_data_arr = atlas_resampled.get_fdata()
    cbf_data = cbf_img.get_fdata()

    # Extract regional values
    results = []
    for roi_idx, roi_name in enumerate(labels, start=1):
        roi_mask = atlas_data_arr == roi_idx
        if np.sum(roi_mask) > 0:
            roi_cbf = np.mean(cbf_data[roi_mask])
            results.append({
                'subject': subject,
                'region': roi_name,
                'mean_cbf': roi_cbf,
                'voxels': np.sum(roi_mask)
            })

    df = pd.DataFrame(results)
    output_file = f'{subject}_regional_cbf_{atlas}.csv'
    df.to_csv(output_file, index=False)
    print(f"Regional CBF extracted for {subject} using {atlas} atlas")
    return df

# Process multiple subjects
subjects = ['sub-01', 'sub-02', 'sub-03']
all_results = []
for subject in subjects:
    df = extract_regional_cbf(subject, Path('derivatives/aslprep'), atlas='aal')
    all_results.append(df)

# Combine results
combined_df = pd.concat(all_results, ignore_index=True)
combined_df.to_csv('all_subjects_regional_cbf.csv', index=False)
```

## Troubleshooting

### Common Issues and Solutions

**1. FreeSurfer License Error**

```
Error: FreeSurfer license file not found
```

**Solution:**
```bash
# Obtain license from https://surfer.nmr.mgh.harvard.edu/registration.html
# Ensure license file is mounted correctly
docker run --rm -it \
  -v /path/to/license.txt:/opt/freesurfer/license.txt:ro \
  pennlinc/aslprep:latest \
  --fs-license-file /opt/freesurfer/license.txt
```

**2. Missing M0 Image**

```
Error: M0Type is 'Separate' but no M0 image found
```

**Solution:**
```bash
# Ensure M0 scan is in BIDS format
# sub-01/perf/sub-01_m0scan.nii.gz
# Or update JSON to use estimate
echo '{"M0Type": "Estimate", "M0Estimate": 1.0}' >> sub-01_asl.json
```

**3. Invalid aslcontext.tsv**

```
Error: Number of volumes in aslcontext.tsv does not match ASL data
```

**Solution:**
```python
import nibabel as nib

# Check number of volumes
asl_img = nib.load('sub-01_asl.nii.gz')
n_volumes = asl_img.shape[3]
print(f"Number of volumes: {n_volumes}")

# aslcontext.tsv must have n_volumes rows (excluding header)
# For 40 control-label pairs:
with open('sub-01_aslcontext.tsv', 'w') as f:
    f.write('volume_type\n')
    for i in range(40):
        f.write('control\n')
        f.write('label\n')
```

**4. Out of Memory Errors**

```
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Increase memory allocation
docker run --rm -it \
  --memory=32g \
  --memory-swap=32g \
  pennlinc/aslprep:latest \
  --mem-mb 32000 \
  --low-mem  # Enable low-memory mode
```

**5. Registration Failures**

```
Error: ANTs registration failed for sub-01
```

**Solution:**
```bash
# Use SyN-based SDC fallback
docker run --rm -it \
  pennlinc/aslprep:latest \
  --use-syn-sdc \
  --force-syn  # Force SyN even if fieldmaps exist

# Or skip problematic steps
--skip-asl-registration  # Use anatomical-to-template only
```

### Debugging with Work Directory

```bash
# Keep work directory for debugging
docker run --rm -it \
  -v ${WORK_DIR}:/work \
  pennlinc/aslprep:latest \
  --work-dir /work \
  --notrack \
  --write-graph \
  --stop-on-first-crash

# Inspect workflow graph
ls work/aslprep_wf/
# Shows: graph.svg, graph.png, report.rst

# Check intermediate outputs
ls work/aslprep_wf/single_subject_01_wf/asl_preproc_wf/
```

### Checking Software Versions

```bash
# Version information
docker run --rm pennlinc/aslprep:latest --version

# Full environment
docker run --rm pennlinc/aslprep:latest --help

# Dependencies
docker run --rm --entrypoint /bin/bash pennlinc/aslprep:latest -c \
  "pip list | grep -E 'nipype|niworkflows|fmriprep'"
```

## Best Practices

### 1. BIDS Validation

```bash
# Always validate BIDS dataset before running ASLPrep
docker run --rm -v /path/to/data:/data:ro \
  bids/validator /data

# Fix common BIDS errors
# - Missing dataset_description.json
# - Inconsistent file naming
# - Missing sidecar JSONs
```

### 2. Quality Control Workflow

```bash
# 1. Run on single subject first
# 2. Review HTML report thoroughly
# 3. Check CBF values are physiologically plausible (40-80 ml/100g/min)
# 4. Inspect registration quality
# 5. Process full dataset
# 6. Aggregate QC metrics across subjects
```

### 3. Resource Optimization

```bash
# Use appropriate resources
# Single subject: 4 cores, 8GB RAM, ~4-8 hours
# Adjust based on data:
# - Fewer volumes → less time
# - Higher resolution → more RAM
# - Surface processing → more time

docker run --rm -it \
  --cpus=4 \
  --memory=8g \
  pennlinc/aslprep:latest \
  --nthreads 4 \
  --omp-nthreads 2
```

## References

- **ASLPrep Documentation**: https://aslprep.readthedocs.io/
- **BIDS Specification (ASL)**: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#arterial-spin-labeling-perfusion-data
- **Adebimpe et al. (2022)**: ASLPrep: A Platform for Processing of Arterial Spin Labeled MRI. *Nature Methods*
- **Alsop et al. (2015)**: Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications. *Magnetic Resonance in Medicine*, 73(1), 102-116
- **fMRIPrep**: https://fmriprep.org/ (foundation for ASLPrep)
- **ASL White Paper**: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4190138/

## Related Tools

- **BASIL:** Bayesian ASL modeling in FSL
- **ExploreASL:** Multi-center ASL preprocessing
- **Quantiphyse:** Interactive ASL/qMRI workflows
- **fMRIPrep:** General fMRI preprocessing (non-ASL)
