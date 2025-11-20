# Clinica

## Overview

**Clinica** is a comprehensive software platform for clinical neuroimaging research, developed by the Aramis Lab at Inria Paris and ICM (Brain and Spine Institute). It provides end-to-end pipelines for processing and analyzing multimodal brain imaging data, with a particular focus on neurodegenerative diseases such as Alzheimer's disease and Parkinson's disease.

Clinica standardizes data organization using the Brain Imaging Data Structure (BIDS) for raw data and the ClinicA Processed Structure (CAPS) for processed derivatives. It integrates numerous neuroimaging tools (FreeSurfer, FSL, SPM, ANTs, MRtrix3) into unified, reproducible workflows designed for clinical research and multi-center studies.

**Key Use Cases:**
- Clinical neuroimaging studies with standardized pipelines
- Alzheimer's disease and Parkinson's disease biomarker extraction
- Multi-modal imaging analysis (T1-MRI, PET, DWI, fMRI)
- Machine learning on neuroimaging features
- Multi-center study harmonization
- Longitudinal disease progression tracking

**Official Website:** https://www.clinica.run/
**Documentation:** https://aramislab.paris.inria.fr/clinica/docs/public/latest/
**Source Code:** https://github.com/aramis-lab/clinica

---

## Key Features

- **BIDS and CAPS Compliance:** Automatic conversion to BIDS format and structured processed data organization
- **Disease-Specific Pipelines:** Pre-built workflows optimized for Alzheimer's and Parkinson's research
- **Multi-Modal Support:** Integrated processing for T1-MRI, PET, DWI, fMRI, and other modalities
- **Clinical Focus:** Biomarker extraction, statistical analysis, and machine learning for clinical research
- **Tool Integration:** Seamless integration with FreeSurfer, FSL, SPM12, ANTs, MRtrix3, and more
- **Reproducible Workflows:** Container support (Docker, Singularity) and version-controlled pipelines
- **Quality Control:** Built-in QC checks and visualization at each processing step
- **HPC Support:** Cluster execution with SLURM, PBS, and other schedulers
- **Machine Learning Module:** Classification pipelines with cross-validation and feature importance
- **Statistical Analysis:** Surface-based statistics, ROI analysis, and mass-univariate testing
- **Longitudinal Processing:** Template creation and change quantification for disease progression
- **PET Quantification:** SUVR computation, partial volume correction, and reference region extraction
- **Connectomics:** Structural and functional connectivity analysis
- **Extensive Documentation:** Comprehensive tutorials, examples, and troubleshooting guides
- **Active Development:** Regular updates with latest methodological advances and bug fixes

---

## Installation

### Using Conda (Recommended)

```bash
# Create dedicated Conda environment
conda create -n clinica python=3.9
conda activate clinica

# Install Clinica from conda-forge
conda install -c conda-forge -c aramislab clinica

# Verify installation
clinica --version
```

### Using Docker

```bash
# Pull the official Clinica Docker image
docker pull aramislab/clinica:latest

# Run Clinica in Docker
docker run -it \
  -v /path/to/data:/data \
  aramislab/clinica:latest \
  clinica --version
```

### Using Singularity (for HPC)

```bash
# Build Singularity container from Docker image
singularity build clinica.sif docker://aramislab/clinica:latest

# Run Clinica with Singularity
singularity exec \
  -B /path/to/data:/data \
  clinica.sif \
  clinica --version
```

### Installing Third-Party Software

Clinica requires external tools depending on the pipeline used:

```bash
# FreeSurfer (required for many pipelines)
# Download from https://surfer.nmr.mgh.harvard.edu/
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# FSL (for diffusion and functional pipelines)
# Download from https://fsl.fmrib.ox.ac.uk/
export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh

# ANTs (for registration pipelines)
conda install -c conda-forge ants

# SPM12 (for PET and statistical analyses)
# Requires MATLAB or SPM Standalone
# Download from https://www.fil.ion.ucl.ac.uk/spm/

# MRtrix3 (for diffusion pipelines)
conda install -c mrtrix3 mrtrix3
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/aramis-lab/clinica.git
cd clinica

# Install in development mode
pip install -e .[dev]

# Run tests
pytest
```

---

## Basic Usage

### Data Organization with BIDS

Clinica requires input data in BIDS format. Convert your data using BIDS converters:

```bash
# Example: Convert ADNI data to BIDS
clinica convert adni-to-bids \
  /path/to/adni/dataset \
  /path/to/clinical/data.csv \
  /path/to/bids/output

# Validate BIDS dataset
clinica iotools check-bids-folder /path/to/bids/output
```

### Running a Basic T1 Pipeline

```bash
# Linear processing of T1-weighted MRI
clinica run t1-linear \
  /path/to/bids \
  /path/to/caps \
  --participant_label sub-01 sub-02

# Check outputs in CAPS directory
ls /path/to/caps/subjects/sub-01/ses-M000/t1_linear/
```

### Multi-Subject Batch Processing

```bash
# Process all subjects in BIDS dataset
clinica run t1-linear \
  /path/to/bids \
  /path/to/caps

# Process specific sessions
clinica run t1-linear \
  /path/to/bids \
  /path/to/caps \
  --sessions_label M000 M012

# Use working directory for intermediate files
clinica run t1-linear \
  /path/to/bids \
  /path/to/caps \
  --working_directory /scratch/clinica_work
```

### Quality Control

```bash
# Generate QC report for T1-linear pipeline
clinica iotools check-missing-modalities \
  /path/to/caps \
  t1-linear

# View subject-specific outputs
clinica iotools check-missing-processing \
  /path/to/caps \
  t1-linear \
  --participant_label sub-01
```

---

## Anatomical Pipelines

### T1-Linear Pipeline

Affine registration to MNI space with tissue segmentation:

```bash
# Run T1-linear pipeline
clinica run t1-linear \
  /path/to/bids \
  /path/to/caps \
  --uncropped_image  # Keep original FOV

# Outputs:
# - Affine-registered T1 in MNI space
# - Brain mask
# - Tissue probability maps (GM, WM, CSF)
```

### T1-Volume Pipeline

Atlas-based segmentation and regional volumetry:

```bash
# Run T1-volume pipeline (requires t1-linear)
clinica run t1-volume \
  /path/to/caps \
  --atlas neuromorphometrics hammers \
  --modulate on  # For VBM-style analysis

# Extract regional volumes
clinica run statistics-volume \
  /path/to/caps \
  --atlas neuromorphometrics \
  --group_label ADvsHC
```

**Python API for Volume Extraction:**

```python
from clinica.pipelines.statistics_volume import StatisticsVolumeCLI
from pathlib import Path

# Load volumetric data
caps_dir = Path("/path/to/caps")
atlas = "neuromorphometrics"

# Get ROI volumes for all subjects
volumes = {}
for subject_dir in caps_dir.glob("subjects/sub-*/ses-*/t1/spm/segmentation/"):
    # Read atlas statistics
    stats_file = subject_dir / f"*_space-Ixi549Space_map-{atlas}_statistics.tsv"
    # Parse volumes (implementation depends on Clinica version)
```

### T1-FreeSurfer Pipeline

Surface-based morphometry with FreeSurfer:

```bash
# Run FreeSurfer recon-all through Clinica
clinica run t1-freesurfer \
  /path/to/bids \
  /path/to/caps \
  --recon_all_args "-qcache -measure thickness"

# Run on HPC cluster with SLURM
clinica run t1-freesurfer \
  /path/to/bids \
  /path/to/caps \
  --n_procs 8 \
  --working_directory /scratch/clinica_work
```

**Template Creation for Longitudinal Studies:**

```bash
# Create subject-specific template
clinica run t1-freesurfer-longitudinal \
  /path/to/bids \
  /path/to/caps \
  --participant_label sub-01 \
  --sessions_label M000 M012 M024

# Analyze longitudinal changes
clinica run statistics-surface \
  /path/to/caps \
  --feature_label thickness \
  --group_label longitudinal
```

### PET-Volume Pipeline

PET quantification in volumetric space:

```bash
# PET partial volume correction and SUVR
clinica run pet-volume \
  /path/to/bids \
  /path/to/caps \
  --acq_label av45 \
  --suvr_reference_region pons \
  --pvc_psf_tsv /path/to/psf.tsv

# Multiple tracers
clinica run pet-volume \
  /path/to/bids \
  /path/to/caps \
  --acq_label fdg \
  --suvr_reference_region pons cerebellumPons
```

**SUVR Computation Options:**

```python
# Reference regions for different PET tracers
reference_regions = {
    "fdg": ["pons", "cerebellumPons"],
    "av45": ["cerebellumPons", "pons", "pons2", "cerebellumPons2"],
    "pib": ["cerebellumPons"],
    "fbb": ["cerebellumPons"],
    "flute": ["inferiorCerebellum"]
}
```

---

## Diffusion MRI Pipelines

### DWI Preprocessing

```bash
# Comprehensive DWI preprocessing
clinica run dwi-preprocessing-using-t1 \
  /path/to/bids \
  /path/to/caps \
  --use_cuda  # GPU acceleration if available

# Outputs:
# - Denoised, corrected DWI
# - Brain mask
# - B0 image
# - Gradient table
```

### DTI Pipeline

```bash
# Compute DTI metrics
clinica run dwi-dti \
  /path/to/caps

# Outputs in CAPS:
# subjects/sub-*/ses-*/dwi/dti_based_processing/
#   - FA, MD, AD, RD maps
#   - Colored FA map
#   - Registered to T1 space
```

### DWI Connectome

```bash
# Generate structural connectome
clinica run dwi-connectome \
  /path/to/caps \
  --n_tracks 10000000 \
  --atlas desikan destrieux

# Outputs:
# - Tractography streamlines
# - Connectivity matrices (ROI x ROI)
# - Network metrics
```

**Extract Connectivity Matrix:**

```python
import numpy as np
from pathlib import Path

# Load connectivity matrix
caps_dir = Path("/path/to/caps")
subject = "sub-01"
session = "ses-M000"
atlas = "desikan"

matrix_file = (
    caps_dir / f"subjects/{subject}/{session}/dwi/connectome/"
    f"{subject}_{session}_atlas-{atlas}_connMatrix.npy"
)

conn_matrix = np.load(matrix_file)
print(f"Connectivity matrix shape: {conn_matrix.shape}")
print(f"Mean connectivity: {conn_matrix.mean():.4f}")
```

---

## PET Surface Pipeline

### PET-Surface Processing

```bash
# Project PET data onto FreeSurfer surfaces
clinica run pet-surface \
  /path/to/bids \
  /path/to/caps \
  --acq_label av45 \
  --suvr_reference_region pons \
  --pvc_psf_tsv /path/to/psf.tsv

# Smooth on surface
clinica run pet-surface \
  /path/to/bids \
  /path/to/caps \
  --acq_label av45 \
  --suvr_reference_region pons \
  --smooth 8  # 8mm FWHM smoothing
```

**Surface-Based SUVR Analysis:**

```python
# Read surface PET data (using nibabel)
import nibabel as nib
from pathlib import Path

caps_dir = Path("/path/to/caps")
subject = "sub-01"
session = "ses-M000"
tracer = "av45"
hemi = "lh"

surf_file = (
    caps_dir / f"subjects/{subject}/{session}/pet/surface/"
    f"{subject}_{session}_trc-{tracer}_hemi-{hemi}_fwhm-8_suvr-pons_pet.gii"
)

# Load surface data
img = nib.load(surf_file)
suvr_data = img.darrays[0].data
print(f"Surface vertices: {len(suvr_data)}")
print(f"Mean SUVR: {suvr_data.mean():.3f}")
```

---

## Machine Learning

### Classification Pipeline

```bash
# Train SVM classifier for AD vs CN
clinica run machinelearning-prepare-spatial-svm \
  /path/to/caps \
  --group_label ADvsCN \
  --image_type T1w \
  --atlas AAL2

# Run classification with cross-validation
clinica run machinelearning-classification \
  /path/to/caps \
  --group_label ADvsCN \
  --image_type T1w \
  --atlas AAL2 \
  --cv RepeatedKFold \
  --n_iterations 100
```

**Python API for Custom Classification:**

```python
from clinica.pipelines.machine_learning import (
    RB_RepHoldOut_DualSVM,
    VB_RepHoldOut_DualSVM
)
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

# Load features and labels
caps_dir = "/path/to/caps"
atlas = "AAL2"
feature_tsv = f"{caps_dir}/groups/group-ADvsCN/machine_learning/features_{atlas}.tsv"
df = pd.read_csv(feature_tsv, sep='\t')

# Prepare data
X = df.drop(['participant_id', 'session_id', 'diagnosis'], axis=1).values
y = df['diagnosis'].map({'AD': 1, 'CN': 0}).values

# Region-based SVM
classifier = RB_RepHoldOut_DualSVM()
results = classifier.evaluate(X, y)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"AUC: {results['auc']:.3f}")
print(f"Sensitivity: {results['sensitivity']:.3f}")
print(f"Specificity: {results['specificity']:.3f}")
```

### Feature Extraction

```bash
# Extract regional features for ML
clinica run machinelearning-prepare-spatial-svm \
  /path/to/caps \
  --group_label custom_group \
  --image_type PET \
  --acq_label fdg \
  --suvr_reference_region pons \
  --atlas neuromorphometrics

# Extract surface-based features
clinica run machinelearning-prepare-spatial-svm \
  /path/to/caps \
  --group_label custom_group \
  --image_type PET \
  --acq_label av45 \
  --surface_based
```

**Feature Importance Analysis:**

```python
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

# Train linear SVM to get feature weights
svm = LinearSVC(C=1.0, penalty='l2')
svm.fit(X, y)

# Get feature importance (absolute weights)
feature_importance = np.abs(svm.coef_[0])

# Map to ROI names
roi_names = df.drop(['participant_id', 'session_id', 'diagnosis'], axis=1).columns
importance_df = pd.DataFrame({
    'ROI': roi_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("Top 10 discriminative regions:")
print(importance_df.head(10))
```

---

## Statistical Analysis

### Surface-Based Statistics

```bash
# GLM analysis on cortical surfaces
clinica run statistics-surface \
  /path/to/caps \
  --group_label ADvsCN \
  --orig_input_data t1-freesurfer \
  --glm_type group_comparison \
  --contrast AD - CN \
  --feature_label thickness

# Include covariates
clinica run statistics-surface \
  /path/to/caps \
  --group_label ADvsCN \
  --orig_input_data pet-surface \
  --acq_label av45 \
  --suvr_reference_region pons \
  --glm_type group_comparison \
  --covariates age sex apoe4
```

### ROI-Based Statistics

```python
import pandas as pd
from scipy import stats
import numpy as np

# Load ROI statistics
caps_dir = "/path/to/caps"
atlas = "neuromorphometrics"

# Collect volumes for all subjects
volumes_list = []
for subj_dir in Path(caps_dir).glob("subjects/sub-*/ses-*/t1/spm/segmentation/"):
    stats_file = list(subj_dir.glob(f"*_{atlas}_statistics.tsv"))[0]
    df = pd.read_csv(stats_file, sep='\t')
    # Add metadata
    subject = stats_file.parent.parent.parent.parent.parent.name
    session = stats_file.parent.parent.parent.parent.name
    df['subject'] = subject
    df['session'] = session
    volumes_list.append(df)

all_volumes = pd.concat(volumes_list, ignore_index=True)

# Load clinical data with diagnoses
participants = pd.read_csv(f"{caps_dir}/participants.tsv", sep='\t')
data = all_volumes.merge(participants, left_on='subject', right_on='participant_id')

# Compare hippocampal volume between AD and CN
ad_hipp = data[data['diagnosis'] == 'AD']['Right-Hippocampus'].values
cn_hipp = data[data['diagnosis'] == 'CN']['Right-Hippocampus'].values

t_stat, p_value = stats.ttest_ind(ad_hipp, cn_hipp)
print(f"Right Hippocampus: t={t_stat:.3f}, p={p_value:.4f}")
```

### Mass-Univariate Testing

```python
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np

# Get all ROI columns
roi_columns = [col for col in all_volumes.columns if 'Left-' in col or 'Right-' in col]

# Run t-tests for all ROIs
results = []
for roi in roi_columns:
    ad_vals = data[data['diagnosis'] == 'AD'][roi].values
    cn_vals = data[data['diagnosis'] == 'CN'][roi].values

    t_stat, p_val = stats.ttest_ind(ad_vals, cn_vals)
    cohen_d = (ad_vals.mean() - cn_vals.mean()) / np.sqrt((ad_vals.std()**2 + cn_vals.std()**2) / 2)

    results.append({
        'ROI': roi,
        't_statistic': t_stat,
        'p_value': p_val,
        'cohen_d': cohen_d
    })

results_df = pd.DataFrame(results)

# FDR correction
_, results_df['p_fdr'], _, _ = multipletests(
    results_df['p_value'],
    method='fdr_bh'
)

# Significant regions after correction
sig_regions = results_df[results_df['p_fdr'] < 0.05].sort_values('p_fdr')
print(f"Significant regions (FDR < 0.05): {len(sig_regions)}")
print(sig_regions[['ROI', 'cohen_d', 'p_fdr']].head(10))
```

---

## Disease-Specific Workflows

### Alzheimer's Disease Biomarkers

```bash
# Complete AD biomarker pipeline
# 1. Structural MRI
clinica run t1-volume /path/to/caps --atlas hippocampus_subfields

# 2. Amyloid PET
clinica run pet-volume \
  /path/to/bids /path/to/caps \
  --acq_label av45 \
  --suvr_reference_region cerebellumPons

# 3. FDG PET (metabolism)
clinica run pet-volume \
  /path/to/bids /path/to/caps \
  --acq_label fdg \
  --suvr_reference_region pons

# 4. Machine learning classification
clinica run machinelearning-classification \
  /path/to/caps \
  --group_label ADvsCN \
  --image_type T1w PET
```

**Multi-Modal Biomarker Integration:**

```python
import pandas as pd
from pathlib import Path

caps_dir = Path("/path/to/caps")

# Extract multiple biomarkers per subject
biomarkers = []
for subject_dir in caps_dir.glob("subjects/sub-*"):
    subject_id = subject_dir.name

    # Hippocampal volume
    hipp_vol = extract_hippocampal_volume(subject_dir)

    # Amyloid burden (from AV45 PET)
    amyloid_suvr = extract_amyloid_burden(subject_dir, atlas="desikan")

    # FDG metabolism
    fdg_suvr = extract_fdg_metabolism(subject_dir, atlas="desikan")

    biomarkers.append({
        'subject_id': subject_id,
        'hippocampal_volume': hipp_vol,
        'amyloid_burden': amyloid_suvr,
        'fdg_metabolism': fdg_suvr
    })

biomarker_df = pd.DataFrame(biomarkers)

# Merge with clinical data
participants = pd.read_csv(caps_dir / "participants.tsv", sep='\t')
full_data = biomarker_df.merge(participants, on='subject_id')

# Save multi-modal biomarker table
full_data.to_csv(caps_dir / "multimodal_biomarkers.csv", index=False)
```

### Parkinson's Disease Analysis

```bash
# Substantia nigra analysis with neuromelanin-sensitive MRI
clinica run t1-volume \
  /path/to/caps \
  --atlas substantia_nigra

# DAT-SPECT processing (if available)
clinica run pet-volume \
  /path/to/bids /path/to/caps \
  --acq_label datspect \
  --suvr_reference_region occipital

# Connectivity analysis
clinica run dwi-connectome \
  /path/to/caps \
  --atlas basal_ganglia
```

---

## Integration with Claude Code

Clinica integrates seamlessly with Claude Code for automated clinical neuroimaging workflows:

```python
# clinica_workflow.py - Automated multi-subject processing

from pathlib import Path
import subprocess
import logging

def run_clinica_pipeline(bids_dir, caps_dir, pipeline, **kwargs):
    """Run Clinica pipeline with error handling."""

    cmd = [
        "clinica", "run", pipeline,
        str(bids_dir),
        str(caps_dir)
    ]

    # Add optional arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])

    logging.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("Pipeline completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline failed: {e.stderr}")
        raise

def alzheimers_workflow(bids_dir, caps_dir, n_procs=4):
    """Complete Alzheimer's disease processing workflow."""

    # T1 processing
    run_clinica_pipeline(
        bids_dir, caps_dir,
        "t1-linear"
    )

    run_clinica_pipeline(
        caps_dir, caps_dir,
        "t1-volume",
        atlas="neuromorphometrics"
    )

    # FreeSurfer
    run_clinica_pipeline(
        bids_dir, caps_dir,
        "t1-freesurfer",
        n_procs=n_procs
    )

    # Amyloid PET
    run_clinica_pipeline(
        bids_dir, caps_dir,
        "pet-volume",
        acq_label="av45",
        suvr_reference_region="cerebellumPons"
    )

    # Quality control
    check_pipeline_outputs(caps_dir)

    logging.info("Alzheimer's workflow complete!")

# Usage in Claude Code
if __name__ == "__main__":
    alzheimers_workflow(
        Path("/data/ADNI_BIDS"),
        Path("/data/ADNI_CAPS"),
        n_procs=8
    )
```

**Batch Processing Script:**

```python
#!/usr/bin/env python3
# batch_process_clinica.py

import argparse
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def process_subject(subject_id, bids_dir, caps_dir, pipeline):
    """Process single subject with Clinica."""
    cmd = f"""
    clinica run {pipeline} \
        {bids_dir} {caps_dir} \
        --participant_label {subject_id}
    """
    subprocess.run(cmd, shell=True, check=True)
    return subject_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bids_dir", type=Path)
    parser.add_argument("caps_dir", type=Path)
    parser.add_argument("--pipeline", default="t1-linear")
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    # Get subject list
    subjects = [d.name for d in args.bids_dir.glob("sub-*") if d.is_dir()]

    # Parallel processing
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = [
            executor.submit(process_subject, subj, args.bids_dir, args.caps_dir, args.pipeline)
            for subj in subjects
        ]

        for future in futures:
            completed = future.result()
            print(f"Completed: {completed}")

if __name__ == "__main__":
    main()
```

---

## Integration with Other Tools

### FreeSurfer Integration

```bash
# Use FreeSurfer outputs in Clinica
export SUBJECTS_DIR=/path/to/caps/subjects

# Clinica automatically finds FreeSurfer outputs
clinica run pet-surface \
  /path/to/bids /path/to/caps \
  --acq_label fdg
```

### SPM Integration

Clinica uses SPM12 for VBM and PET processing:

```python
# Clinica handles SPM batch generation internally
# Access SPM outputs in CAPS structure
from pathlib import Path

caps_dir = Path("/path/to/caps")
spm_outputs = caps_dir / "subjects/sub-01/ses-M000/t1/spm/segmentation"

# Tissue probability maps
gm_map = spm_outputs / "*_space-Ixi549Space_map-graymatter_probability.nii.gz"
```

### ANTs Integration

```python
# Clinica uses ANTs for registration
# Access transformation files
transforms_dir = (
    caps_dir / "subjects/sub-01/ses-M000/t1/spm/segmentation/"
    "normalized_space"
)

# Apply transformations with ANTs directly
import subprocess
subprocess.run([
    "antsApplyTransforms",
    "-i", "input.nii.gz",
    "-r", "template.nii.gz",
    "-t", str(transforms_dir / "transform.mat"),
    "-o", "output.nii.gz"
])
```

### PyBIDS Integration

```python
from bids import BIDSLayout

# Index BIDS dataset
layout = BIDSLayout("/path/to/bids")

# Query specific files
t1w_files = layout.get(
    subject='01',
    session='M000',
    datatype='anat',
    suffix='T1w',
    extension='nii.gz'
)

# Feed to Clinica
for t1w in t1w_files:
    print(f"Processing: {t1w.path}")
```

### NiBabel Integration

```python
import nibabel as nib
from pathlib import Path

# Load Clinica outputs
caps_dir = Path("/path/to/caps")

# T1 in MNI space
t1_mni = nib.load(
    caps_dir / "subjects/sub-01/ses-M000/t1_linear/"
    "sub-01_ses-M000_space-MNI152NLin2009cSym_T1w.nii.gz"
)

# Tissue segmentation
gm_prob = nib.load(
    caps_dir / "subjects/sub-01/ses-M000/t1/spm/segmentation/"
    "sub-01_ses-M000_space-Ixi549Space_map-graymatter_probability.nii.gz"
)

print(f"T1 shape: {t1_mni.shape}")
print(f"GM probability mean: {gm_prob.get_fdata().mean():.3f}")
```

---

## Troubleshooting

### Problem 1: FreeSurfer License Missing

**Symptoms:** `recon-all` fails with license error

**Solution:**
```bash
# Copy FreeSurfer license file
cp license.txt $FREESURFER_HOME/license.txt

# Or set environment variable
export FS_LICENSE=/path/to/license.txt

# Verify
clinica run t1-freesurfer --help
```

### Problem 2: Insufficient Memory for FreeSurfer

**Symptoms:** FreeSurfer crashes or produces corrupted outputs

**Solution:**
```bash
# Increase memory limit (if using Docker)
docker run -m 16g aramislab/clinica ...

# Or split processing
clinica run t1-freesurfer \
  /path/to/bids /path/to/caps \
  --participant_label sub-01  # Process one at a time
```

### Problem 3: BIDS Validation Errors

**Symptoms:** Clinica rejects input dataset

**Solution:**
```bash
# Run BIDS validator
bids-validator /path/to/bids

# Common fixes:
# - Add dataset_description.json
# - Fix participant_id formatting (must start with 'sub-')
# - Add required metadata to JSON sidecars
# - Verify file naming conventions
```

### Problem 4: Missing PET Metadata

**Symptoms:** PET pipeline fails to read tracer information

**Solution:**
```json
// Add to PET JSON sidecar (sub-01_ses-M000_pet.json)
{
  "Manufacturer": "Siemens",
  "TracerName": "AV45",
  "InjectedRadioactivity": 370,
  "InjectedRadioactivityUnits": "MBq",
  "InjectedMass": 1.5,
  "InjectedMassUnits": "ug",
  "MolarActivity": 1000,
  "MolarActivityUnits": "GBq/umol",
  "TimeZero": "10:00:00",
  "ScanStart": 0,
  "InjectionStart": 0
}
```

### Problem 5: Cluster Execution Failures

**Symptoms:** Jobs fail on HPC cluster

**Solution:**
```bash
# Use working directory on scratch filesystem
clinica run t1-freesurfer \
  /path/to/bids /path/to/caps \
  --working_directory /scratch/$USER/clinica_work \
  --n_procs 1  # Start with serial processing

# Check SLURM logs
ls -lh /scratch/$USER/clinica_work/
```

---

## Best Practices

### 1. Data Organization

- **Always use BIDS:** Convert raw data to BIDS before processing
- **Validate BIDS dataset:** Run `bids-validator` and Clinica's BIDS checker
- **Consistent naming:** Use standardized participant IDs and session labels
- **Metadata completeness:** Include all required JSON sidecar information
- **Version control:** Track BIDS dataset version and Clinica version

### 2. Pipeline Execution

- **Test on subset:** Run pipelines on 1-2 subjects before full dataset
- **Working directory:** Use fast scratch filesystem for intermediate files
- **Resource allocation:** Match CPU/memory to pipeline requirements
- **Container usage:** Prefer containers for reproducibility
- **Resume capability:** Keep working directories to resume failed runs

### 3. Quality Control

- **Visual inspection:** Review outputs for each subject
- **Automated QC metrics:** Use Clinica's built-in QC tools
- **Outlier detection:** Flag subjects with extreme values
- **Missing data:** Track which subjects completed each pipeline
- **Regular backups:** Archive processed data incrementally

### 4. Multi-Center Studies

- **Harmonization:** Account for scanner differences in statistical models
- **Traveling subjects:** Use them to estimate inter-scanner variability
- **Consistent protocols:** Maintain uniform acquisition parameters
- **Site-specific QC:** Track quality metrics per site
- **ComBat correction:** Apply for voxel-based analyses if needed

### 5. Reproducibility

- **Document versions:** Record Clinica and dependency versions
- **Container snapshots:** Archive container images used
- **Configuration files:** Save all pipeline parameters
- **Random seeds:** Set for machine learning pipelines
- **Provenance tracking:** Use CAPS structure to trace processing steps

### 6. Performance Optimization

- **Parallel processing:** Use `--n_procs` appropriate for your system
- **Cluster submission:** Leverage HPC for large datasets
- **Disk I/O:** Use local fast storage for working directories
- **Caching:** Reuse intermediate results when possible
- **Batch size:** Balance between parallelism and resource usage

---

## Resources

### Official Documentation

- **Clinica Homepage:** https://www.clinica.run/
- **Full Documentation:** https://aramislab.paris.inria.fr/clinica/docs/public/latest/
- **GitHub Repository:** https://github.com/aramis-lab/clinica
- **Tutorial Notebooks:** https://github.com/aramis-lab/clinica/tree/dev/docs/Notebooks
- **Issue Tracker:** https://github.com/aramis-lab/clinica/issues

### Publications and Citations

- **Clinica Paper:** Routier et al. (2021) "Clinica: An Open-Source Software Platform for Reproducible Clinical Neuroscience Studies" *Frontiers in Neuroinformatics*
- **CAPS Format:** Routier et al. (2021) describing the ClinicA Processed Structure
- **AD Pipelines:** Samper-González et al. (2018) "Reproducible evaluation of classification methods in Alzheimer's disease"

### Tutorials and Examples

- **Getting Started Guide:** https://aramislab.paris.inria.fr/clinica/docs/public/latest/InteractingWithClinica/
- **Pipeline Tutorials:** https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/
- **BIDS Conversion:** https://aramislab.paris.inria.fr/clinica/docs/public/latest/Converters/
- **Machine Learning Tutorial:** https://aramislab.paris.inria.fr/clinica/docs/public/latest/MachineLearning/

### Community and Support

- **Google Group:** https://groups.google.com/g/clinica-user
- **Gitter Chat:** https://gitter.im/aramis-lab/clinica (if available)
- **Email:** Contact Aramis Lab team through their website

### Related Tools and Datasets

- **BIDS Specification:** https://bids-specification.readthedocs.io/
- **ADNI Database:** https://adni.loni.usc.edu/ (Alzheimer's Disease Neuroimaging Initiative)
- **PPMI Database:** https://www.ppmi-info.org/ (Parkinson's Progression Markers Initiative)
- **OpenNeuro:** https://openneuro.org/ (open neuroimaging datasets)

---

## Citation

```bibtex
@article{routier2021clinica,
  title={Clinica: An Open-Source Software Platform for Reproducible Clinical Neuroscience Studies},
  author={Routier, Alexandre and Burgos, Ninon and D{\'\i}az, Mauricio and Bacci, Michael and Bottani, Simona and El-Rifai, Omar and Fontanella, Sabrina and Gori, Pietro and Guillon, Jérémy and Guyot, Alexis and others},
  journal={Frontiers in Neuroinformatics},
  volume={15},
  pages={689675},
  year={2021},
  publisher={Frontiers Media SA},
  doi={10.3389/fninf.2021.689675}
}
```

---

## Related Tools

- **fMRIPrep:** Complementary fMRI preprocessing (see `fmriprep.md`)
- **QSIPrep:** Alternative diffusion preprocessing (see `qsiprep.md`)
- **FreeSurfer:** Cortical surface reconstruction (see `freesurfer.md`)
- **SPM12:** Statistical parametric mapping (see `spm.md`)
- **ANTs:** Advanced normalization tools (see `ants.md`)
- **MRtrix3:** Diffusion MRI processing (see `mrtrix3.md`)
- **Pydra:** Workflow engine (can wrap Clinica pipelines, see `pydra.md`)
- **Snakebids:** BIDS workflow framework (alternative approach, see `snakebids.md`)
- **TractoFlow:** Dedicated diffusion pipeline (see `tractoflow.md`)
- **NiPype:** Pipeline framework (Clinica is built on top of it)

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**Clinica Version Covered:** 0.7.x - 0.8.x
**Maintainer:** Claude Code Neuroimaging Skills
