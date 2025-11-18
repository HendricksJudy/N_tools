# Clinica: Clinical Neuroimaging Platform

## Overview

Clinica is an open-source software platform designed specifically for clinical neuroimaging research, with a primary focus on neurodegenerative diseases such as Alzheimer's disease (AD), Parkinson's disease (PD), and frontotemporal dementia (FTD). It provides standardized, reproducible pipelines for processing multi-modal neuroimaging data and extracting clinically relevant biomarkers.

Clinica addresses the unique challenges of clinical neuroimaging studies: longitudinal data analysis, multi-modal biomarker integration, heterogeneous datasets from different sites, and the need for reproducible processing in clinical trials. The platform integrates established neuroimaging tools (FreeSurfer, SPM, FSL, ANTs, PETPVC) into unified, BIDS-compatible workflows specifically designed for clinical research applications.

**Key Features:**
- BIDS-compatible clinical neuroimaging pipelines
- Longitudinal anatomical, diffusion, PET, and functional MRI analysis
- Multi-modal biomarker extraction (cortical thickness, hippocampal volume, amyloid/tau PET quantification)
- Machine learning classification pipelines with cross-validation
- Support for large clinical datasets (ADNI, AIBL, OASIS, UK Biobank)
- Statistical analysis and visualization tools
- Containerized execution (Docker, Singularity) for reproducibility
- Quality control and harmonization for multi-site studies

**Primary Use Cases:**
- Alzheimer's disease progression studies and clinical trials
- Parkinson's disease biomarker development
- Multi-modal imaging analysis in neurodegenerative diseases
- Longitudinal cohort studies (ADNI, OASIS, UK Biobank)
- Pharmaceutical clinical trials requiring imaging endpoints
- Predictive modeling for disease conversion (e.g., MCI to AD)

**Citation:**
```
Routier, A., Burgos, N., Díaz, M., Bacci, M., Bottani, S., El-Rifai, O., ... & Colliot, O. (2021).
Clinica: An open-source software platform for reproducible clinical neuroscience studies.
Frontiers in Neuroinformatics, 15, 689675.
```

## Installation

### Prerequisites

Clinica requires Python 3.7+ and depends on several neuroimaging software packages. You can install dependencies separately or use containerized versions.

**Required Dependencies:**
- FreeSurfer 6.0+ (for cortical reconstruction)
- SPM12 + MATLAB Runtime (for statistical analysis)
- FSL 6.0+ (for registration and preprocessing)
- ANTs 2.3+ (for normalization)
- PETPVC (for PET partial volume correction)

### Installation via Conda (Recommended)

```bash
# Create a dedicated conda environment
conda create -n clinica python=3.8
conda activate clinica

# Install Clinica
conda install -c Aramislab -c conda-forge clinica

# Verify installation
clinica --version
```

### Installation via Pip

```bash
# Create virtual environment
python -m venv clinica_env
source clinica_env/bin/activate

# Install Clinica
pip install clinica

# Install optional dependencies for specific pipelines
pip install clinica[ml]  # Machine learning dependencies
```

### Docker Installation (Fully Containerized)

```bash
# Pull Clinica Docker image (includes all dependencies)
docker pull aramislab/clinica:latest

# Run Clinica in container
docker run -it --rm \
  -v /path/to/data:/data \
  aramislab/clinica:latest \
  clinica --version
```

### Singularity Installation (for HPC)

```bash
# Build Singularity container
singularity build clinica.sif docker://aramislab/clinica:latest

# Run Clinica with Singularity
singularity exec clinica.sif clinica --version
```

### Installing Third-Party Software

```bash
# Set FreeSurfer environment
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Set FSL environment
export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh

# Set ANTs environment
export ANTSPATH=/usr/local/ants/bin
export PATH=$ANTSPATH:$PATH
```

### Testing Installation

```bash
# Download example dataset
clinica download nifd

# Test with example data
clinica run t1-linear nifd_bids nifd_caps -tsv subjects.tsv
```

## BIDS Conversion for Clinical Data

Clinica uses the BIDS (Brain Imaging Data Structure) standard to organize clinical neuroimaging data. For major clinical datasets like ADNI, Clinica provides dedicated conversion tools.

### Converting ADNI Data to BIDS

**Example 1: ADNI to BIDS Conversion**

```bash
# Download ADNI data (requires ADNI credentials)
# Organize downloaded data in ADNI format

# Convert ADNI to BIDS
clinica convert adni-to-bids \
  /path/to/adni/raw \
  /path/to/adni_bids \
  --clinical_data /path/to/adni/clinical.csv

# This converts:
# - T1w MRI (MPRAGE)
# - T2w, FLAIR, DWI
# - FDG, Amyloid, Tau PET
# - Participant demographics and clinical scores
```

### BIDS Structure for Longitudinal Clinical Data

```
adni_bids/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── sub-ADNI001/
│   ├── ses-M00/          # Baseline
│   │   ├── anat/
│   │   │   ├── sub-ADNI001_ses-M00_T1w.nii.gz
│   │   │   └── sub-ADNI001_ses-M00_T1w.json
│   │   ├── dwi/
│   │   │   ├── sub-ADNI001_ses-M00_dwi.nii.gz
│   │   │   └── sub-ADNI001_ses-M00_dwi.bval
│   │   ├── pet/
│   │   │   └── sub-ADNI001_ses-M00_trc-18FAV45_pet.nii.gz
│   │   └── func/
│   │       └── sub-ADNI001_ses-M00_task-rest_bold.nii.gz
│   ├── ses-M12/         # 12-month follow-up
│   └── ses-M24/         # 24-month follow-up
└── sub-ADNI002/
```

**Example 2: Converting OASIS Data**

```bash
# Convert OASIS-3 to BIDS
clinica convert oasis-to-bids \
  /path/to/oasis/raw \
  /path/to/oasis_bids \
  --subjects sub-OAS30001 sub-OAS30002

# Includes clinical metadata: CDR, MMSE, neuropsych scores
```

**Example 3: Creating Clinical Metadata**

```bash
# Create participants.tsv with clinical information
cat participants.tsv
# participant_id  age  sex  diagnosis  mmse  apoe4
# sub-ADNI001     72   F    AD         18    2
# sub-ADNI002     68   M    MCI        26    1
# sub-ADNI003     65   F    CN         30    0

# This metadata is used in statistical analyses and machine learning
```

**Example 4: Custom BIDS Conversion**

```python
# For custom clinical datasets
from clinica.iotools.converters.factory import get_converter

# Initialize converter
converter = get_converter('custom-to-bids')

# Define conversion mapping
mapping = {
    'anat': {'T1': 'T1w', 'T2': 'T2w'},
    'pet': {'PIB': 'trc-11CPIB', 'FDG': 'trc-18FFDG'},
    'dwi': {'DTI': 'dwi'}
}

# Run conversion
converter.convert(
    input_dir='/path/to/custom/data',
    output_dir='/path/to/bids',
    mapping=mapping,
    clinical_tsv='/path/to/clinical.tsv'
)
```

## Anatomical Pipelines

### T1w Linear Processing

The T1-linear pipeline performs volume-based morphometry using SPM12, extracting regional volumes and tissue segmentation without surface reconstruction.

**Example 5: Basic T1w Linear Pipeline**

```bash
# Run T1w linear processing (volume-based analysis)
clinica run t1-linear \
  /path/to/bids \
  /path/to/caps \
  -tsv participants.tsv

# Outputs:
# - Dartel template creation
# - Tissue segmentation (GM, WM, CSF)
# - Regional volumes (AAL2, LPBA40, Neuromorphometrics atlases)
# - Normalized images in MNI space
```

**Example 6: Extracting Hippocampal Volumes**

```bash
# T1-linear with atlas-based volume extraction
clinica run t1-linear \
  /path/to/bids \
  /path/to/caps \
  -tsv participants.tsv

# Extract regional volumes from results
clinica iotools merge-tsv \
  /path/to/caps \
  /path/to/regional_volumes.tsv \
  --pipelines t1-linear

# regional_volumes.tsv contains:
# participant_id  session_id  l_hippocampus  r_hippocampus  l_amygdala ...
# sub-ADNI001     M00         3421.2         3389.5         1876.3
```

### FreeSurfer Cross-Sectional and Longitudinal

**Example 7: FreeSurfer Cross-Sectional**

```bash
# Run FreeSurfer cortical reconstruction (cross-sectional)
clinica run t1-freesurfer \
  /path/to/bids \
  /path/to/caps \
  -tsv participants.tsv \
  --recon-all-options "-notal-check -cw256"

# Outputs:
# - Cortical thickness maps
# - Surface area and volume
# - Subcortical segmentation
# - Cortical parcellations (Desikan-Killiany, Destrieux)
```

**Example 8: FreeSurfer Longitudinal for Disease Progression**

```bash
# Create longitudinal FreeSurfer processing
clinica run t1-freesurfer-longitudinal \
  /path/to/bids \
  /path/to/caps \
  -tsv longitudinal_participants.tsv

# For each subject with multiple timepoints:
# 1. Creates unbiased within-subject template
# 2. Processes each timepoint relative to template
# 3. Computes atrophy rates over time

# Extract longitudinal cortical thickness
clinica iotools merge-tsv \
  /path/to/caps \
  /path/to/thickness_trajectories.tsv \
  --pipelines t1-freesurfer-longitudinal \
  --measure thickness
```

**Example 9: Regional Volume and Cortical Thickness Analysis**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load FreeSurfer outputs
thickness = pd.read_csv('caps/thickness_trajectories.tsv', sep='\t')

# Compute annual atrophy rate
thickness['time_delta'] = (thickness['session_id'].str.extract('M(\d+)')[0].astype(int) / 12)
thickness['thickness_change'] = thickness.groupby('participant_id')['entorhinal_thickness'].diff()
thickness['atrophy_rate'] = (thickness['thickness_change'] / thickness['time_delta']) * 100

# Group by diagnosis
ad_atrophy = thickness[thickness['diagnosis'] == 'AD']['atrophy_rate'].mean()
mci_atrophy = thickness[thickness['diagnosis'] == 'MCI']['atrophy_rate'].mean()
cn_atrophy = thickness[thickness['diagnosis'] == 'CN']['atrophy_rate'].mean()

print(f"Entorhinal cortex atrophy rate:")
print(f"  AD:  {ad_atrophy:.2f}%/year")
print(f"  MCI: {mci_atrophy:.2f}%/year")
print(f"  CN:  {cn_atrophy:.2f}%/year")
# Expected: AD: -4.5%/year, MCI: -2.2%/year, CN: -0.8%/year
```

## DWI Processing

**Example 10: DWI Preprocessing Pipeline**

```bash
# Run DWI preprocessing with T1w registration
clinica run dwi-preprocessing-using-t1 \
  /path/to/bids \
  /path/to/caps \
  -tsv participants.tsv

# Pipeline steps:
# 1. Head motion and eddy current correction
# 2. Susceptibility distortion correction
# 3. Registration to T1w
# 4. Resampling to 1mm isotropic
```

**Example 11: DTI Model Fitting**

```bash
# Fit DTI model to preprocessed DWI
clinica run dwi-dti \
  /path/to/bids \
  /path/to/caps \
  -tsv participants.tsv

# Outputs DTI metrics:
# - FA (fractional anisotropy)
# - MD (mean diffusivity)
# - RD (radial diffusivity)
# - AD (axial diffusivity)
```

**Example 12: White Matter Tract Analysis in AD**

```python
# Extract FA values in major white matter tracts
clinica iotools merge-tsv \
  /path/to/caps \
  /path/to/fa_values.tsv \
  --pipelines dwi-dti \
  --measure FA

# Atlas-based white matter tract extraction
import pandas as pd
import seaborn as sns

fa_data = pd.read_csv('fa_values.tsv', sep='\t')

# Hippocampal cingulum FA in AD vs. controls
cingulum_fa_ad = fa_data[fa_data['diagnosis'] == 'AD']['l_cingulum_fa'].mean()
cingulum_fa_cn = fa_data[fa_data['diagnosis'] == 'CN']['l_cingulum_fa'].mean()

print(f"Left cingulum FA: AD={cingulum_fa_ad:.3f}, CN={cingulum_fa_cn:.3f}")
# Expected: reduced FA in AD (~0.45) vs. CN (~0.58)
```

**Example 13: NODDI Microstructure Modeling**

```bash
# Fit NODDI model for microstructure analysis
clinica run dwi-noddi \
  /path/to/bids \
  /path/to/caps \
  -tsv participants.tsv

# Outputs NODDI metrics:
# - ICVF (intracellular volume fraction)
# - ISOVF (isotropic volume fraction)
# - ODI (orientation dispersion index)

# These metrics are sensitive to neurodegeneration
```

## PET Processing

### Amyloid PET Quantification

**Example 14: Amyloid PET SUVR Computation**

```bash
# Process amyloid PET (e.g., Florbetapir, Pittsburgh Compound B)
clinica run pet-linear \
  /path/to/bids \
  /path/to/caps \
  --acq_label av45 \
  --suvr_reference_region pons \
  -tsv participants.tsv

# Pipeline:
# 1. Co-register PET to T1w
# 2. Intensity normalization (SUVR)
# 3. Partial volume correction (optional)
# 4. Spatial normalization to MNI
# 5. Regional SUVR extraction
```

**Example 15: Centiloid Scale Conversion**

```bash
# Convert amyloid PET to Centiloid scale (standardized units)
clinica run pet-linear \
  /path/to/bids \
  /path/to/caps \
  --acq_label pib \
  --suvr_reference_region cerebellum-gm \
  --centiloid

# Extract Centiloid values
clinica iotools merge-tsv \
  /path/to/caps \
  /path/to/centiloid_values.tsv \
  --pipelines pet-linear \
  --measure centiloid

# Centiloid interpretation:
# < 12: Amyloid negative
# 12-24: Borderline
# > 24: Amyloid positive
```

**Example 16: Tau PET Quantification**

```bash
# Process tau PET (e.g., Flortaucipir, AV-1451)
clinica run pet-linear \
  /path/to/bids \
  /path/to/caps \
  --acq_label av1451 \
  --suvr_reference_region cerebellar-gm \
  --pvc rbv  # Partial volume correction

# Extract regional tau SUVR
clinica iotools merge-tsv \
  /path/to/caps \
  /path/to/tau_suvr.tsv \
  --pipelines pet-linear \
  --measure suvr
```

**Example 17: FDG PET Metabolism Analysis**

```bash
# Process FDG PET for glucose metabolism
clinica run pet-linear \
  /path/to/bids \
  /path/to/caps \
  --acq_label fdg \
  --suvr_reference_region pons \
  -tsv participants.tsv

# Typical AD pattern: reduced metabolism in:
# - Temporoparietal cortex
# - Posterior cingulate
# - Precuneus
```

**Example 18: Multi-Tracer PET Integration**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load multi-modal PET data
amyloid = pd.read_csv('centiloid_values.tsv', sep='\t')
tau = pd.read_csv('tau_suvr.tsv', sep='\t')
fdg = pd.read_csv('fdg_suvr.tsv', sep='\t')

# Merge datasets
pet_data = amyloid.merge(tau, on=['participant_id', 'session_id'])
pet_data = pet_data.merge(fdg, on=['participant_id', 'session_id'])

# A-T-N framework (Amyloid, Tau, Neurodegeneration)
pet_data['A_positive'] = pet_data['centiloid'] > 24
pet_data['T_positive'] = pet_data['entorhinal_tau_suvr'] > 1.3
pet_data['N_positive'] = pet_data['temporoparietal_fdg_suvr'] < 1.2

# Classify participants
pet_data['ATN_status'] = (
    pet_data['A_positive'].astype(int).astype(str) +
    pet_data['T_positive'].astype(int).astype(str) +
    pet_data['N_positive'].astype(int).astype(str)
)

print(pet_data['ATN_status'].value_counts())
# 000: Normal biomarkers
# 100: Amyloid only (preclinical AD)
# 110: Amyloid + Tau
# 111: Full AD signature
```

## fMRI Processing

**Example 19: Resting-State fMRI Preprocessing**

```bash
# Preprocess resting-state fMRI
clinica run fmri-preprocessing \
  /path/to/bids \
  /path/to/caps \
  -tsv participants.tsv \
  --working_directory /scratch/fmri_work

# Pipeline includes:
# - Slice-timing correction
# - Motion correction
# - Susceptibility distortion correction
# - Registration to T1w
# - Spatial normalization to MNI
# - Confound extraction (motion, CSF, WM)
```

**Example 20: Default Mode Network Connectivity in MCI**

```python
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np

# Load preprocessed fMRI and DMN atlas
dmn_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# Extract timeseries from DMN regions
masker = NiftiLabelsMasker(
    labels_img=dmn_atlas.maps,
    standardize=True,
    memory='nilearn_cache'
)

# For each subject
for subject in ['sub-ADNI001', 'sub-ADNI002']:
    func_file = f'caps/subjects/{subject}/ses-M00/fmri/preprocessing/{subject}_ses-M00_task-rest_space-MNI_desc-preproc_bold.nii.gz'
    timeseries = masker.fit_transform(func_file)

    # Compute correlation matrix
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([timeseries])[0]

    # Focus on posterior cingulate connectivity (DMN hub)
    pcc_connectivity = correlation_matrix[8, :]  # PCC index in atlas
    print(f"{subject} PCC connectivity: {np.mean(pcc_connectivity):.3f}")
```

**Example 21: Seed-Based Connectivity Analysis**

```bash
# Run seed-based connectivity using Clinica
clinica run fmri-connectome \
  /path/to/bids \
  /path/to/caps \
  --atlas schaefer2018 \
  --n_parcels 400 \
  -tsv participants.tsv

# Outputs connectivity matrices for each subject
# Dimensions: 400x400 (pairwise ROI correlations)
```

## Machine Learning Pipelines

**Example 22: SVM Classification for AD Diagnosis**

```bash
# Train SVM classifier: AD vs. CN using cortical thickness
clinica run machinelearning-classification \
  /path/to/caps \
  t1-freesurfer \
  AD_vs_CN_classification \
  --group_label AD CN \
  --participants_tsv participants.tsv \
  --feature_type cortical_thickness \
  --cross_validation 10 \
  --n_iterations 100

# Outputs:
# - Classification accuracy, sensitivity, specificity
# - Feature weights (important brain regions)
# - ROC curves
# - Cross-validation performance
```

**Example 23: Multi-Modal Feature Integration**

```python
from clinica.pipelines.machine_learning import RB_RepeatedHoldOut_DualSVM
import pandas as pd

# Load multi-modal features
thickness = pd.read_csv('caps/t1_freesurfer/cortical_thickness.tsv', sep='\t')
volumes = pd.read_csv('caps/t1_linear/regional_volumes.tsv', sep='\t')
pet_suvr = pd.read_csv('caps/pet_linear/amyloid_suvr.tsv', sep='\t')

# Merge features
features = thickness.merge(volumes, on=['participant_id', 'session_id'])
features = features.merge(pet_suvr, on=['participant_id', 'session_id'])

# Prepare data for machine learning
X = features.drop(['participant_id', 'session_id', 'diagnosis'], axis=1)
y = (features['diagnosis'] == 'AD').astype(int)

# Train classifier with repeated holdout
classifier = RB_RepeatedHoldOut_DualSVM(
    n_iterations=100,
    test_size=0.2,
    grid_search_folds=10
)

classifier.fit(X, y)
print(f"Balanced accuracy: {classifier.best_score_:.3f}")
print(f"Top features: {classifier.top_features_[:10]}")
```

**Example 24: Predicting MCI-to-AD Conversion**

```bash
# Predict conversion from MCI to AD using baseline features
clinica run machinelearning-classification \
  /path/to/caps \
  t1-freesurfer \
  MCI_converter_prediction \
  --group_label MCI_converter MCI_stable \
  --participants_tsv mci_participants.tsv \
  --feature_type cortical_thickness \
  --cross_validation 5 \
  --n_iterations 250

# Clinical relevance: Identify MCI patients at high risk
# Baseline features predict 3-year conversion with ~75% accuracy
```

**Example 25: Feature Importance Visualization**

```python
import matplotlib.pyplot as plt
import numpy as np

# Load classifier weights from Clinica ML output
weights = pd.read_csv('AD_vs_CN_classification/weights.tsv', sep='\t')

# Top 20 discriminative regions
top_regions = weights.nlargest(20, 'abs_weight')

plt.figure(figsize=(10, 8))
plt.barh(top_regions['region'], top_regions['weight'])
plt.xlabel('SVM Weight')
plt.ylabel('Brain Region')
plt.title('Top Discriminative Regions for AD vs. CN')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)

# Typically shows: hippocampus, entorhinal, temporal cortex
```

## Statistical Analysis and Quality Control

**Example 26: Group-Level Statistical Analysis**

```python
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Load longitudinal data
data = pd.read_csv('caps/longitudinal_measures.tsv', sep='\t')

# Mixed-effects model: cortical thickness ~ time * diagnosis + age + sex
from statsmodels.formula.api import mixedlm

model = mixedlm(
    "entorhinal_thickness ~ time * diagnosis + age + sex",
    data,
    groups=data["participant_id"]
)
result = model.fit()
print(result.summary())

# Test time × diagnosis interaction (differential atrophy rates)
# Significant interaction indicates faster decline in AD group
```

**Example 27: Automated Quality Control**

```bash
# Generate QC reports for all pipelines
clinica iotools check-missing-modalities \
  /path/to/bids \
  /path/to/caps \
  /path/to/qc_report.tsv

# Review T1w preprocessing quality
clinica iotools check-quality \
  /path/to/caps \
  --pipeline t1-freesurfer \
  --output qc_freesurfer.html

# Flags subjects with:
# - High motion artifacts
# - Segmentation failures
# - Registration errors
```

## HPC and Batch Processing

**Example 28: SLURM Cluster Execution**

```bash
# Run Clinica on HPC with SLURM
clinica run t1-freesurfer \
  /path/to/bids \
  /path/to/caps \
  -tsv participants.tsv \
  --n_procs 8 \
  --working_directory /scratch/$USER/clinica_work

# Create SLURM submission script
cat > submit_clinica.sh << EOF
#!/bin/bash
#SBATCH --job-name=clinica
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=1-100

# Load modules
module load freesurfer/7.1.1
module load fsl/6.0.4

# Activate Clinica environment
source activate clinica

# Process subjects in parallel
clinica run t1-freesurfer \\
  /data/bids \\
  /data/caps \\
  -tsv subjects_batch_\${SLURM_ARRAY_TASK_ID}.tsv \\
  --n_procs 8
EOF

sbatch submit_clinica.sh
```

## Troubleshooting

**Common Issues and Solutions:**

**FreeSurfer Segmentation Failures:**
```bash
# Check for skull-stripping issues
# Manually inspect: caps/subjects/sub-*/ses-*/t1/freesurfer_cross_sectional/*/mri/brainmask.mgz

# Rerun with custom brain mask
clinica run t1-freesurfer \
  --custom_mask /path/to/better_mask.nii.gz
```

**Memory Issues on Large Datasets:**
```bash
# Reduce parallel processing
clinica run t1-linear --n_procs 1

# Use working directory on fast disk
--working_directory /scratch/temp
```

**BIDS Validation Errors:**
```bash
# Validate BIDS before processing
bids-validator /path/to/bids

# Common fixes:
# - Ensure .json sidecars for all imaging files
# - Check participants.tsv format
# - Verify session naming (ses-M00, not ses-baseline)
```

**PET Processing Issues:**
```bash
# Ensure PET tracer is correctly specified
--acq_label av45  # Not AV45 or Amyloid

# Check reference region availability in atlas
# Some atlases may not include all reference regions
```

## Best Practices

**Clinical Data Organization:**
- Maintain original data and metadata separately from BIDS
- Use version control for participants.tsv and clinical scores
- Document all data cleaning and exclusion criteria
- Keep detailed logs of pipeline versions used

**Longitudinal Study Design:**
- Use consistent session naming (ses-M00, ses-M06, ses-M12, etc.)
- Process all timepoints with same pipeline version
- Use FreeSurfer longitudinal pipeline for unbiased estimates
- Account for scanner upgrades in statistical models

**Multi-Site Harmonization:**
- Document scanner models and acquisition parameters
- Use ComBat harmonization for multi-site data
- Include site as covariate in statistical models
- Perform QC separately for each site

**Reproducibility:**
- Always use containerized versions (Docker/Singularity) for production
- Document exact Clinica version: `clinica --version`
- Save all command-line arguments in scripts
- Archive processed data with provenance information

**Clinical Trial Endpoints:**
- Pre-register analysis pipelines before trial unblinding
- Use FDA-approved atlases when required (e.g., FreeSurfer)
- Validate automated segmentations with manual review
- Report processing failures and exclusions

## Integration with Other Tools

Clinica integrates seamlessly with the broader neuroimaging ecosystem:

**Preprocessing Integration:**
- Use fMRIPrep outputs for advanced fMRI denoising
- Combine with QSIPrep for advanced diffusion modeling
- Import FreeSurfer surfaces to Connectome Workbench

**Statistical Analysis:**
- Export data to R for advanced mixed-effects modeling
- Use Python pandas/statsmodels for custom analyses
- Integration with PALM for non-parametric statistics

**Visualization:**
- FreeSurfer's FreeView for cortical maps
- FSLeyes for volumetric overlays
- Nilearn for connectivity matrices
- BrainNet Viewer for network graphs

**Machine Learning:**
- Export features to scikit-learn, PyTorch, TensorFlow
- Integration with NiMARE for meta-analyses
- Combine with clinical data for multi-modal predictions

## References

**Primary Clinica Papers:**
- Routier et al. (2021). Clinica: An open-source software platform for reproducible clinical neuroscience studies. *Frontiers in Neuroinformatics*, 15, 689675.

**Clinical Neuroimaging Standards:**
- Jack et al. (2018). NIA-AA Research Framework: Toward a biological definition of Alzheimer's disease. *Alzheimer's & Dementia*, 14(4), 535-562.
- Klunk et al. (2015). The Centiloid Project: Standardizing quantitative amyloid plaque estimation by PET. *Alzheimer's & Dementia*, 11(1), 1-15.

**ADNI Publications:**
- Weiner et al. (2015). The Alzheimer's Disease Neuroimaging Initiative 3: Continued innovation for clinical trial improvement. *Alzheimer's & Dementia*, 11(5), 561-571.

**Longitudinal FreeSurfer:**
- Reuter et al. (2012). Within-subject template estimation for unbiased longitudinal image analysis. *NeuroImage*, 61(4), 1402-1418.

**Machine Learning in AD:**
- Rathore et al. (2017). A review on neuroimaging-based classification studies and associated feature extraction methods for Alzheimer's disease and its prodromal stages. *NeuroImage*, 155, 530-548.

**Related Tools:**
- FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
- SPM: https://www.fil.ion.ucl.ac.uk/spm/
- FSL: https://fsl.fmrib.ox.ac.uk/
- ANTs: http://stnava.github.io/ANTs/

**Online Resources:**
- Clinica Documentation: https://aramislab.paris.inria.fr/clinica/docs/
- BIDS Specification: https://bids-specification.readthedocs.io/
- ADNI Data Access: http://adni.loni.usc.edu/
- Clinica GitHub: https://github.com/aramis-lab/clinica
