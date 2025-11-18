# Nibabies: Neuroimaging Baby BIDS Application for Infant Examination Suite

## Overview

Nibabies (NeuroImaging Baby BIDS Application for Infant Examination Suite) is a specialized preprocessing pipeline designed specifically for infant and pediatric neuroimaging data. As an extension of fMRIPrep, Nibabies addresses the unique challenges of processing developing brains from birth through early childhood (0-24 months), where rapid anatomical changes, lower tissue contrast, incomplete myelination, and different skull characteristics require fundamentally different preprocessing approaches than adult neuroimaging.

The developing infant brain presents distinct preprocessing challenges: tissue contrast changes dramatically during the first two years of life (particularly the "dark period" at 6-12 months when gray and white matter have similar intensities), fontanelles and incomplete skull ossification complicate brain extraction, rapid volumetric growth requires age-specific templates, and motion artifacts from naturally sleeping or sedated infants require robust correction strategies.

**Key Features:**
- Age-specific infant brain templates (0-24 months)
- Adapted tissue segmentation algorithms for developing brains
- Robust motion correction optimized for infant data
- T1w and T2w co-registration for improved segmentation
- Surface reconstruction adapted for infant cortex
- Age-appropriate spatial normalization strategies
- BIDS-compatible inputs and fMRIPrep-style derivatives
- Comprehensive visual QC reports with age-specific metrics
- Integration with fMRIPrep ecosystem and tools

**Primary Use Cases:**
- Baby Connectome Project and similar infant cohorts
- Neurodevelopmental disorder detection and characterization
- Early brain development trajectory studies
- Effects of prenatal and perinatal factors on brain development
- Longitudinal infant studies (0-24 months)
- Pediatric clinical neuroimaging research

**Citation:**
```
Goncalves, M., Poldrack, R. A., & Satterthwaite, T. D. (in preparation).
Nibabies: A robust preprocessing workflow for infant fMRI data.
```

## Installation

### Docker Installation (Recommended)

Docker provides the most reliable installation method with all dependencies pre-configured.

```bash
# Pull the latest Nibabies Docker image
docker pull nipreps/nibabies:latest

# Verify installation
docker run -it --rm nipreps/nibabies:latest --version

# Expected output: nibabies v23.1.0
```

### Singularity Installation (for HPC)

```bash
# Build Singularity container from Docker image
singularity build nibabies-latest.sif docker://nipreps/nibabies:latest

# Test the container
singularity run nibabies-latest.sif --version
```

### FreeSurfer License

Nibabies requires a FreeSurfer license (free for academic use):

```bash
# Download license from: https://surfer.nmr.mgh.harvard.edu/registration.html
# Place license.txt in accessible location

# When running Nibabies, specify:
--fs-license-file /path/to/license.txt
```

### Age-Specific Templates

Nibabies uses infant-specific brain templates that must match the age of your participants:

```bash
# Templates are automatically downloaded on first use
# Supported templates:
# - MNIInfant:cohort-1 (0-2 months)
# - MNIInfant:cohort-2 (3-5 months)
# - MNIInfant:cohort-3 (6-8 months)
# - MNIInfant:cohort-4 (9-11 months)
# - MNIInfant:cohort-5 (12-14 months)
# - MNIInfant:cohort-6 (15-18 months)
# - MNIInfant:cohort-7 (19-24 months)
# - UNC:0-1-2 (birth, 1 year, 2 years from UNC)
```

### Python Package Installation (Advanced)

```bash
# Only for development or when containers unavailable
pip install nibabies

# Requires manual installation of:
# - FreeSurfer 7.3.2+
# - ANTs 2.3.3+
# - FSL 6.0.5+
```

## BIDS Organization for Infant Data

Proper BIDS organization is critical for Nibabies processing. Infant studies often include both T1w and T2w for improved segmentation.

### Basic BIDS Structure

```
infant_study_bids/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── sub-001/
│   ├── ses-01month/
│   │   └── anat/
│   │       ├── sub-001_ses-01month_T1w.nii.gz
│   │       ├── sub-001_ses-01month_T1w.json
│   │       ├── sub-001_ses-01month_T2w.nii.gz
│   │       └── sub-001_ses-01month_T2w.json
│   ├── ses-03month/
│   ├── ses-06month/
│   ├── ses-12month/
│   └── ses-24month/
└── sub-002/
```

**Example 1: Creating participants.tsv with Age Metadata**

```bash
# participants.tsv - critical for age-appropriate template selection
cat participants.tsv
```
```
participant_id  sex  gestational_age_weeks  birth_weight_kg
sub-001         M    40                     3.5
sub-002         F    38                     3.2
sub-003         M    41                     3.8
```

**Example 2: Session-Level Age Information**

```json
// sub-001/ses-06month/sub-001_ses-06month_scans.json
{
  "age_months": 6.2,
  "postnatal_age_weeks": 26.8,
  "scan_date": "2023-06-15",
  "sedation": "none",
  "sleep_state": "natural_sleep"
}
```

**Example 3: Longitudinal Infant Study BIDS**

```bash
# Organize longitudinal data with consistent session naming
for subject in sub-001 sub-002 sub-003; do
  for age in 01month 03month 06month 12month 24month; do
    mkdir -p ${subject}/ses-${age}/anat
    mkdir -p ${subject}/ses-${age}/func
  done
done

# BIDS validation
bids-validator /path/to/infant_study_bids --verbose
```

## Anatomical Preprocessing

### T1w and T2w Processing

Infant brain segmentation benefits greatly from combining T1w and T2w images, especially during the low-contrast period (6-12 months).

**Example 4: Basic Nibabies Anatomical Processing**

```bash
# Minimal command for anatomical-only preprocessing
docker run -it --rm \
  -v /path/to/bids:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/freesurfer_license.txt:/license.txt:ro \
  nipreps/nibabies:latest \
  /data /out participant \
  --participant-label 001 \
  --age-months 6 \
  --fs-license-file /license.txt \
  --output-spaces MNIInfant:cohort-3:res-2 anat \
  --skip-bids-validation

# Key parameters:
# --age-months: Critical for template selection
# --output-spaces: Infant template + native anatomical space
# MNIInfant:cohort-3: 6-8 month template
```

**Example 5: T1w + T2w Joint Processing**

```bash
# Recommended: Use both T1w and T2w for better segmentation
docker run -it --rm \
  -v /path/to/bids:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/license.txt:/license.txt:ro \
  -v /path/to/work:/work \
  nipreps/nibabies:latest \
  /data /out participant \
  --participant-label 001 002 003 \
  --age-months 12 \
  --anat-modality T1w T2w \
  --fs-license-file /license.txt \
  --output-spaces MNIInfant:cohort-5 anat \
  --work-dir /work \
  --n_cpus 8 \
  --mem_mb 16000

# T2w provides crucial contrast when T1w contrast is poor
# Particularly important at 6-12 months ("dark period")
```

**Example 6: Age-Specific Template Selection**

```python
# Python script to automatically determine appropriate template cohort
import pandas as pd

def select_infant_template(age_months):
    """Select appropriate MNIInfant cohort based on age."""
    if age_months < 2.5:
        return "MNIInfant:cohort-1"
    elif age_months < 5.5:
        return "MNIInfant:cohort-2"
    elif age_months < 8.5:
        return "MNIInfant:cohort-3"
    elif age_months < 11.5:
        return "MNIInfant:cohort-4"
    elif age_months < 14.5:
        return "MNIInfant:cohort-5"
    elif age_months < 18.5:
        return "MNIInfant:cohort-6"
    else:
        return "MNIInfant:cohort-7"

# Load participant ages
participants = pd.read_csv('participants.tsv', sep='\t')
scans = pd.read_csv('sub-001/ses-06month/sub-001_ses-06month_scans.tsv', sep='\t')

age = scans['age_months'].iloc[0]
template = select_infant_template(age)
print(f"Recommended template: {template}")
# Output: Recommended template: MNIInfant:cohort-3
```

**Example 7: Infant Brain Extraction and Segmentation**

```bash
# Nibabies uses age-adapted algorithms for brain extraction
# View brain mask quality in outputs:
# out/nibabies/sub-001/ses-06month/anat/
#   sub-001_ses-06month_desc-brain_mask.nii.gz
#   sub-001_ses-06month_desc-preproc_T1w.nii.gz
#   sub-001_ses-06month_dseg.nii.gz  # Tissue segmentation

# Tissue classes:
# 1: CSF (includes large ventricles in infants)
# 2: Gray matter (cortex + subcortical)
# 3: White matter (unmyelinated + myelinated)
```

### Surface Reconstruction

**Example 8: FreeSurfer Infant Surface Reconstruction**

```bash
# Enable surface reconstruction (optional, computationally intensive)
docker run -it --rm \
  -v /path/to/bids:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/license.txt:/license.txt:ro \
  nipreps/nibabies:latest \
  /data /out participant \
  --participant-label 001 \
  --age-months 12 \
  --fs-license-file /license.txt \
  --output-spaces fsaverage5 MNIInfant:cohort-5 \
  --derivatives /path/to/smriprep_outputs  # If available

# Outputs FreeSurfer surfaces suitable for:
# - Cortical thickness analysis
# - Surface-based functional analysis
# - Sulcal pattern characterization
```

## fMRI Preprocessing

### Motion Correction for Infant Data

Infant fMRI presents unique motion challenges: spontaneous movements during sleep, high-frequency jittering, sudden awakenings, and greater head motion amplitude relative to brain size.

**Example 9: Basic rs-fMRI Preprocessing**

```bash
# Preprocess resting-state fMRI from sleeping infant
docker run -it --rm \
  -v /path/to/bids:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/license.txt:/license.txt:ro \
  -v /path/to/work:/work \
  nipreps/nibabies:latest \
  /data /out participant \
  --participant-label 001 \
  --age-months 6 \
  --task-id rest \
  --fs-license-file /license.txt \
  --output-spaces MNIInfant:cohort-3:res-2 \
  --work-dir /work \
  --fd-spike-threshold 0.5 \
  --dummy-scans 5

# --fd-spike-threshold: More lenient for infants (0.5mm vs. 0.3mm adults)
# --dummy-scans: Remove initial volumes for T1 equilibration
```

**Example 10: Multi-Echo fMRI for Infants**

```bash
# Multi-echo improves BOLD sensitivity and motion robustness
docker run -it --rm \
  -v /path/to/bids:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/license.txt:/license.txt:ro \
  nipreps/nibabies:latest \
  /data /out participant \
  --participant-label 001 \
  --age-months 12 \
  --task-id rest \
  --echo-idx 1 2 3 \
  --fs-license-file /license.txt \
  --output-spaces MNIInfant:cohort-5:res-2 \
  --me-output-echos  # Preserve individual echoes

# Multi-echo allows T2* modeling and denoising
# Particularly useful for infant data with motion
```

**Example 11: Handling High-Motion Infant Data**

```python
# Post-Nibabies: Assess and handle motion
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load confounds
confounds = pd.read_csv(
    'out/nibabies/sub-001/ses-06month/func/'
    'sub-001_ses-06month_task-rest_desc-confounds_timeseries.tsv',
    sep='\t'
)

# Calculate framewise displacement (FD)
fd = confounds['framewise_displacement'].values

# Identify high-motion volumes
fd_threshold = 0.5  # mm, more lenient for infants
high_motion_vols = np.where(fd > fd_threshold)[0]
percent_high_motion = (len(high_motion_vols) / len(fd)) * 100

print(f"High-motion volumes: {percent_high_motion:.1f}%")

# Common thresholds for infant data:
# - Exclude run if >20% high-motion volumes
# - Scrubbing: remove high-motion volumes + 1 preceding, 2 following

if percent_high_motion > 20:
    print("WARNING: Consider excluding this run")
else:
    print("Acceptable motion for infant fMRI")

# Visualize motion
plt.figure(figsize=(12, 4))
plt.plot(fd)
plt.axhline(fd_threshold, color='r', linestyle='--', label=f'Threshold ({fd_threshold}mm)')
plt.xlabel('Volume')
plt.ylabel('FD (mm)')
plt.title(f'Framewise Displacement (Mean: {np.nanmean(fd):.3f}mm)')
plt.legend()
plt.savefig('motion_plot.png', dpi=150)
```

## Spatial Normalization

**Example 12: Age-Appropriate Normalization**

```bash
# Normalize to multiple infant template spaces
docker run -it --rm \
  -v /path/to/bids:/data:ro \
  -v /path/to/output:/out \
  -v /path/to/license.txt:/license.txt:ro \
  nipreps/nibabies:latest \
  /data /out participant \
  --participant-label 001 \
  --age-months 6 \
  --fs-license-file /license.txt \
  --output-spaces \
    MNIInfant:cohort-3:res-2 \
    UNC:0-1-2:res-1 \
    anat \
  --skull-strip-t1w auto

# Multiple output spaces:
# - MNIInfant:cohort-3: Age-matched group template
# - UNC:0-1-2: UNC infant atlas
# - anat: Native anatomical space (for ROI analysis)
```

**Example 13: Longitudinal Template Alignment**

```python
# For longitudinal studies, ensure age-appropriate templates at each visit
import json

sessions = {
    'ses-01month': {'age_months': 1, 'template': 'MNIInfant:cohort-1'},
    'ses-03month': {'age_months': 3, 'template': 'MNIInfant:cohort-2'},
    'ses-06month': {'age_months': 6, 'template': 'MNIInfant:cohort-3'},
    'ses-12month': {'age_months': 12, 'template': 'MNIInfant:cohort-5'},
    'ses-24month': {'age_months': 24, 'template': 'MNIInfant:cohort-7'}
}

# Generate commands for each session
for ses, info in sessions.items():
    cmd = f"""docker run -it --rm \\
  -v /data:/data:ro -v /out:/out \\
  nipreps/nibabies:latest \\
  /data /out participant \\
  --participant-label 001 \\
  --session-id {ses.replace('ses-', '')} \\
  --age-months {info['age_months']} \\
  --output-spaces {info['template']}:res-2 \\
  --fs-license-file /license.txt
"""
    print(cmd)
```

## Quality Control

### Automated QC Metrics

**Example 14: Reviewing Nibabies HTML Reports**

```bash
# Nibabies generates comprehensive HTML reports
# Open in browser: out/nibabies/sub-001.html

# Key sections to review:
# 1. Summary: Number of T1w/T2w, BOLD runs, warnings
# 2. Anatomical: Brain mask, tissue segmentation, registration
# 3. Functional: Alignment to T1w, motion parameters, carpet plot
# 4. About: Software versions, runtime, errors

# Focus areas for infant data:
# - Brain extraction quality (check fontanelle regions)
# - Gray/white matter segmentation (especially 6-12 months)
# - Functional-anatomical registration (infants move more)
# - Motion parameters (higher baseline than adults)
```

**Example 15: Batch QC for Infant Cohorts**

```python
import pandas as pd
import glob
import json

# Extract QC metrics from all subjects
qc_metrics = []

for subject_html in glob.glob('out/nibabies/sub-*.html'):
    subject_id = subject_html.split('/')[-1].replace('.html', '')

    # Parse confounds for motion metrics
    confounds_file = f'out/nibabies/{subject_id}/ses-*/func/*desc-confounds_timeseries.tsv'
    confounds_files = glob.glob(confounds_file)

    for conf_file in confounds_files:
        conf = pd.read_csv(conf_file, sep='\t')

        metrics = {
            'subject_id': subject_id,
            'mean_fd': conf['framewise_displacement'].mean(),
            'max_fd': conf['framewise_displacement'].max(),
            'percent_fd_05': (conf['framewise_displacement'] > 0.5).mean() * 100,
            'mean_dvars': conf['std_dvars'].mean() if 'std_dvars' in conf else None
        }
        qc_metrics.append(metrics)

qc_df = pd.DataFrame(qc_metrics)

# Flag subjects for manual review
qc_df['needs_review'] = (qc_df['percent_fd_05'] > 20) | (qc_df['max_fd'] > 2.0)

print(f"Subjects needing review: {qc_df['needs_review'].sum()}")
print(qc_df[qc_df['needs_review']])

# Save QC report
qc_df.to_csv('infant_cohort_qc.csv', index=False)
```

**Example 16: Visual Inspection of Segmentation**

```bash
# Use fsleyes to visually inspect critical outputs
fsleyes \
  out/nibabies/sub-001/ses-06month/anat/sub-001_ses-06month_desc-preproc_T1w.nii.gz \
  out/nibabies/sub-001/ses-06month/anat/sub-001_ses-06month_desc-brain_mask.nii.gz \
    -cm red -a 50 \
  out/nibabies/sub-001/ses-06month/anat/sub-001_ses-06month_dseg.nii.gz \
    -cm random -a 50

# Check for:
# 1. Brain mask includes all brain tissue (including cerebellum)
# 2. Brain mask excludes skull and dura
# 3. Fontanelles not incorrectly included in brain mask
# 4. Segmentation distinguishes GM/WM (challenging at 6-12 months)
```

## Troubleshooting Common Infant-Specific Issues

### Low Tissue Contrast (6-12 Months)

**Example 17: Handling Low-Contrast Period**

```bash
# For 6-12 month data with poor T1w contrast:

# 1. Ensure T2w is provided (critical for this age)
docker run -it --rm \
  -v /data:/data:ro -v /out:/out -v /license.txt:/license.txt:ro \
  nipreps/nibabies:latest \
  /data /out participant \
  --participant-label 001 \
  --age-months 9 \
  --anat-modality T1w T2w \
  --fs-license-file /license.txt

# 2. If segmentation still fails, use custom brain mask
# Create mask manually in ITK-SNAP, then:
--derivatives /path/to/custom_masks

# custom_masks/sub-001/ses-09month/anat/
#   sub-001_ses-09month_desc-brain_mask.nii.gz
```

### Brain Extraction Failures

**Example 18: Custom Brain Extraction**

```python
# If Nibabies brain extraction fails, create custom mask

from niworkflows.interfaces.ants import AI, BrainExtraction
from nipype.interfaces import ants

# Option 1: Use antsBrainExtraction with infant template
brain_extract = BrainExtraction(
    anatomical_image='sub-001_ses-06month_T1w.nii.gz',
    brain_template='MNIInfant_cohort3_T1.nii.gz',
    brain_probability_mask='MNIInfant_cohort3_brainmask.nii.gz',
    extraction_registration_mask='MNIInfant_cohort3_head_mask.nii.gz',
    out_prefix='sub-001_'
)
result = brain_extract.run()

# Option 2: Use HD-BET (deep learning)
# pip install HD-BET
# hd-bet -i sub-001_ses-06month_T1w.nii.gz -o sub-001_brain_mask.nii.gz -mode fast -tta 0

# Provide custom mask to Nibabies via derivatives
```

### Incomplete Skull Coverage

**Example 19: Handling Fontanelles**

```bash
# Fontanelles (skull gaps) can confuse brain extraction
# Modern Nibabies versions handle this, but if issues persist:

# Check raw data FOV
# Ensure acquisition covers:
# - Anterior fontanelle (closes ~18 months)
# - Posterior fontanelle (closes ~2 months)
# - Lateral fontanelles

# If brain mask incorrectly includes fontanelle regions:
# - Manual editing in ITK-SNAP
# - Morphological operations to close gaps

# Python example: closing holes in brain mask
from scipy import ndimage
import nibabel as nib

mask = nib.load('sub-001_desc-brain_mask.nii.gz')
mask_data = mask.get_fdata()

# Close small holes
mask_closed = ndimage.binary_closing(mask_data, iterations=3)

# Save corrected mask
mask_corrected = nib.Nifti1Image(mask_closed.astype('uint8'), mask.affine)
nib.save(mask_corrected, 'sub-001_desc-brain_mask_corrected.nii.gz')
```

## Advanced Workflows

**Example 20: Using Custom Infant Templates**

```bash
# Use institutional or study-specific infant template
docker run -it --rm \
  -v /data:/data:ro \
  -v /out:/out \
  -v /templates:/templates:ro \
  -v /license.txt:/license.txt:ro \
  nipreps/nibabies:latest \
  /data /out participant \
  --participant-label 001 \
  --age-months 12 \
  --output-spaces \
    /templates/study_specific_12month_template.nii.gz \
  --fs-license-file /license.txt

# Custom template must include:
# - T1w template image
# - Brain probability mask
# - Tissue priors (GM, WM, CSF)
```

**Example 21: Integration with Downstream Connectivity Analysis**

```python
# Post-Nibabies: Functional connectivity in infant brain

from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np

# Load preprocessed infant fMRI
func_file = 'out/nibabies/sub-001/ses-06month/func/sub-001_ses-06month_task-rest_space-MNIInfant_cohort-3_desc-preproc_bold.nii.gz'
confounds_file = 'out/nibabies/sub-001/ses-06month/func/sub-001_ses-06month_task-rest_desc-confounds_timeseries.tsv'

# Infant-appropriate brain parcellation (coarser than adult)
infant_atlas = 'path/to/infant_parcellation_n50.nii.gz'  # 50 ROIs

# Extract timeseries with denoising
import pandas as pd
confounds = pd.read_csv(confounds_file, sep='\t')

# Minimal confound model for infants
confound_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                 'csf', 'white_matter']
confounds_subset = confounds[confound_vars].fillna(0)

masker = NiftiLabelsMasker(
    labels_img=infant_atlas,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)

timeseries = masker.fit_transform(func_file, confounds=confounds_subset.values)

# Compute connectivity matrix
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([timeseries])[0]

# Visualize
plotting.plot_matrix(
    correlation_matrix,
    title='6-Month Infant Functional Connectivity',
    colorbar=True,
    vmax=1,
    vmin=-1
)
plt.savefig('infant_connectivity_matrix.png', dpi=300)
```

**Example 22: Longitudinal Within-Subject Analysis**

```python
# Track individual infant brain development

import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

# Load tissue volumes across timepoints
sessions = ['01month', '03month', '06month', '12month', '24month']
volumes = {'session': [], 'gm_vol': [], 'wm_vol': [], 'csf_vol': []}

for ses in sessions:
    dseg_file = f'out/nibabies/sub-001/ses-{ses}/anat/sub-001_ses-{ses}_dseg.nii.gz'

    if os.path.exists(dseg_file):
        dseg = nib.load(dseg_file)
        dseg_data = dseg.get_fdata()

        voxel_volume_mm3 = np.prod(dseg.header.get_zooms())

        volumes['session'].append(ses)
        volumes['gm_vol'].append(np.sum(dseg_data == 2) * voxel_volume_mm3 / 1000)  # cm³
        volumes['wm_vol'].append(np.sum(dseg_data == 3) * voxel_volume_mm3 / 1000)
        volumes['csf_vol'].append(np.sum(dseg_data == 1) * voxel_volume_mm3 / 1000)

vol_df = pd.DataFrame(volumes)

# Plot developmental trajectory
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(vol_df['session'], vol_df['gm_vol'], marker='o', label='Gray Matter')
ax.plot(vol_df['session'], vol_df['wm_vol'], marker='s', label='White Matter')
ax.plot(vol_df['session'], vol_df['csf_vol'], marker='^', label='CSF')
ax.set_xlabel('Age')
ax.set_ylabel('Volume (cm³)')
ax.set_title('Individual Brain Development Trajectory: sub-001')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('longitudinal_volumes.png', dpi=300)

# Expected pattern:
# - GM volume increases then plateaus
# - WM volume steadily increases (myelination)
# - CSF decreases as brain grows
```

## HPC and Parallel Processing

**Example 23: SLURM Array Job**

```bash
#!/bin/bash
#SBATCH --job-name=nibabies
#SBATCH --array=1-50
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/nibabies_%A_%a.out

# Load Singularity module
module load singularity

# Get subject ID from array
SUBJECTS=($(ls -d /data/bids/sub-* | xargs -n 1 basename))
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID-1]}
SUBJECT_ID=${SUBJECT#sub-}

# Read age from participants.tsv
AGE=$(grep "^${SUBJECT}" /data/bids/participants.tsv | cut -f 4)

# Run Nibabies
singularity run --cleanenv \
  -B /data:/data \
  -B /output:/output \
  -B /work:/work \
  -B /license.txt:/license.txt \
  /path/to/nibabies.sif \
  /data/bids /output participant \
  --participant-label ${SUBJECT_ID} \
  --age-months ${AGE} \
  --output-spaces MNIInfant:cohort-auto:res-2 \
  --fs-license-file /license.txt \
  --work-dir /work/${SUBJECT} \
  --n_cpus 8 \
  --mem_mb 30000 \
  --stop-on-first-crash

# Clean up work directory
rm -rf /work/${SUBJECT}
```

**Example 24: Monitoring Progress**

```bash
# Check Nibabies progress across HPC jobs
for log in logs/nibabies_*.out; do
  if grep -q "FINISHED SUCCESSFULLY" "$log"; then
    echo "$(basename $log): COMPLETE"
  elif grep -q "ERROR" "$log"; then
    echo "$(basename $log): FAILED"
  else
    echo "$(basename $log): RUNNING"
  fi
done

# Count completed subjects
completed=$(grep -l "FINISHED SUCCESSFULLY" logs/nibabies_*.out | wc -l)
total=$(ls logs/nibabies_*.out | wc -l)
echo "Progress: $completed / $total"
```

## Best Practices for Infant Neuroimaging

**Acquisition Protocols:**
- T1w: MPRAGE, 1mm isotropic, 5-7 minutes
- T2w: TSE, 1mm isotropic, crucial for 0-12 months
- rs-fMRI: TR=2s, multiband if available, 10+ minutes for sleep
- Natural sleep preferred over sedation when safe
- Scanner mock sessions for behavioral desensitization

**Quality Standards:**
- Manually review all brain masks (automated QC insufficient)
- Age-specific motion thresholds (more lenient than adults)
- Expect higher exclusion rates (20-30% vs. <10% adults)
- Require both T1w and T2w for 0-12 month data

**Longitudinal Design:**
- Use age-appropriate templates at each visit
- Consider within-subject template for longitudinal analysis
- Account for scanner upgrades in multi-year studies
- Balance developmental trajectories vs. cross-sectional comparisons

**Ethical Considerations:**
- Minimize scan time (infant safety and comfort)
- Natural sleep preferred over sedation
- Family-centered scanning environment
- Clear informed consent for research vs. clinical use

## Integration with Neuroimaging Ecosystem

**fMRIPrep Compatibility:**
Nibabies derivatives are fMRIPrep-compatible, enabling:
- Integration with XCPEngine for connectivity
- Use of fMRIPrep-compatible tools (fMRIDenoise, Nilearn pipelines)
- Standard BIDS derivatives format

**FreeSurfer Integration:**
- Infant-adapted recon-all algorithms
- Surfaces compatible with FreeSurfer analysis tools
- Longitudinal FreeSurfer can be run on Nibabies anatomical outputs

**Analysis Tools:**
- Nilearn: Functional connectivity analysis
- Conn Toolbox: Advanced connectivity (with infant parcellations)
- Custom Python/R: Developmental trajectory modeling

## Infant-Specific Atlases and Parcellations

**Available Infant Atlases:**
- UNC 0-1-2 Year Atlas (newborn, 1yr, 2yr)
- MNI Infant Templates (cohort-1 through cohort-7)
- Melbourne Children's Regional Infant Brain (M-CRIB)
- BCP Infant Parcellations (functional networks)

**Selecting Parcellations:**
- Use coarser parcellations than adults (50-100 ROIs vs. 200-400)
- Age-appropriate functional network definitions
- Consider developmental stage when interpreting connectivity

## References

**Nibabies and Infant Preprocessing:**
- Goncalves et al. (in prep). Nibabies: A robust preprocessing workflow for infant fMRI data.
- Esteban et al. (2019). fMRIPrep: A robust preprocessing pipeline for fMRI data. *Nature Methods*, 16(1), 111-116.

**Infant Brain Development:**
- Knickmeyer et al. (2008). A structural MRI study of human brain development from birth to 2 years. *Journal of Neuroscience*, 28(47), 12176-12182.
- Gilmore et al. (2012). Longitudinal development of cortical and subcortical gray matter from birth to 2 years. *Cerebral Cortex*, 22(11), 2478-2485.

**Infant Templates:**
- Shi et al. (2011). Infant brain atlases from neonates to 1- and 2-year-olds. *PLoS ONE*, 6(4), e18746.
- Fonov et al. (2011). Unbiased average age-appropriate atlases for pediatric studies. *NeuroImage*, 54(1), 313-327.

**Motion in Infant fMRI:**
- Power et al. (2012). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *NeuroImage*, 59(3), 2142-2154.
- Adapted thresholds for infant populations.

**Infant Functional Connectivity:**
- Gao et al. (2015). Functional connectivity of the infant human brain: Plastic and modifiable. *Neuroscientist*, 23(2), 169-184.
- Smyser et al. (2010). Longitudinal analysis of neural network development in preterm infants. *Cerebral Cortex*, 20(12), 2852-2862.

**Online Resources:**
- Nibabies Documentation: https://nibabies.readthedocs.io/
- fMRIPrep Documentation: https://fmriprep.org/
- Baby Connectome Project: https://nda.nih.gov/edit_collection.html?id=2848
- Developing Human Connectome Project: http://www.developingconnectome.org/
