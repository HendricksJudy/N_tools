# CIFTIFY: FreeSurfer to CIFTI Conversion and Quality Control

## Overview

CIFTIFY is a comprehensive toolbox for converting FreeSurfer-processed neuroimaging data to the CIFTI (Connectivity Informatics Technology Initiative) format used by the Human Connectome Project (HCP). CIFTI is a next-generation neuroimaging file format that efficiently stores both cortical surface and subcortical volume data in a single file, enabling integrated analysis of the whole brain. CIFTIFY not only performs format conversion but also provides extensive quality control tools, visualization capabilities, and integration with HCP-style analysis pipelines.

The Human Connectome Project established CIFTI as a standard format for "grayordinates" - a data representation that treats cortical surface vertices and subcortical voxels uniformly. This enables seamless analysis across cortical and subcortical structures, consistent handling of left and right hemispheres, and integration with the extensive HCP analysis ecosystem. CIFTIFY makes these powerful HCP tools accessible to researchers processing their own data with standard FreeSurfer workflows.

**Key Features:**
- Conversion from FreeSurfer to HCP-compatible CIFTI format
- Automated resampling to fsLR (HCP's standard surface space)
- Quality control HTML reports with visual inspection tools
- Integration with Connectome Workbench for visualization
- Support for anatomical and functional data
- BIDS-compatible workflows (ciftify_recon_all)
- Subcortical structure mapping to grayordinates
- Batch processing for large datasets

**Primary Use Cases:**
- Converting FreeSurfer outputs to HCP-style CIFTI format
- Preparing data for HCP pipelines and Connectome Workbench
- Quality control of surface reconstructions
- Multi-modal surface-based analysis (anatomical + functional)
- Standardizing data across FreeSurfer and HCP processing streams
- Large-scale neuroimaging studies requiring unified format

**Citation:**
```
Dickie, E. W., Anticevic, A., Smith, D. E., Coalson, T. S., Manogaran, M.,
Calarco, N., ... & Voineskos, A. N. (2019). Ciftify: A framework for
surface-based analysis of legacy MR acquisitions. NeuroImage, 197, 818-826.
```

## Installation

### Docker Installation (Recommended)

Docker provides the most reliable installation with all dependencies pre-configured:

```bash
# Pull CIFTIFY Docker image
docker pull tigrlab/fmriprep_ciftify:latest

# Verify installation
docker run --rm tigrlab/fmriprep_ciftify:latest ciftify_recon_all --version

# Expected output: ciftify version 2.3.3
```

### Singularity Installation (for HPC)

```bash
# Build Singularity container
singularity build ciftify.sif docker://tigrlab/fmriprep_ciftify:latest

# Test container
singularity exec ciftify.sif ciftify_recon_all --version
```

### Python Package Installation

```bash
# Install CIFTIFY via pip
pip install ciftify

# Install dependencies
# - FreeSurfer 6.0+ (required)
# - Connectome Workbench 1.3+ (required)
# - MSM (multimodal surface matching, optional but recommended)

# Verify installation
ciftify_recon_all --help
```

### Setting Up Environment

```bash
# FreeSurfer must be configured
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Set Connectome Workbench path
export CIFTIFY_WORKBENCH=/usr/local/workbench/bin_linux64

# Set CIFTIFY data directory
export CIFTIFY_DATA=/path/to/ciftify_data

# Verify paths
which wb_command  # Should point to Connectome Workbench
```

### Downloading Required Templates

```bash
# Download HCP templates and atlases
ciftify_download_templates

# Downloads to $CIFTIFY_DATA:
# - fsLR surface meshes (32k and 164k resolution)
# - HCP atlas files
# - Standard subcortical structures
# - MSM configuration files (if using MSM)
```

## FreeSurfer to CIFTI Conversion

### Basic Conversion Workflow

**Example 1: Convert Single Subject from FreeSurfer to CIFTI**

```bash
# Prerequisites: FreeSurfer recon-all already completed
# Input: $SUBJECTS_DIR/sub-001
# Output: CIFTI files in ciftify output directory

ciftify_recon_all \
  --fs-subjects-dir /path/to/freesurfer/subjects \
  --ciftify-work-dir /path/to/ciftify/output \
  sub-001

# Processing steps performed:
# 1. Resample FreeSurfer surfaces to 32k fsLR mesh
# 2. Create midthickness surfaces
# 3. Map cortical data to grayordinates
# 4. Generate subcortical segmentation
# 5. Create CIFTI dense timeseries (.dtseries.nii)
# 6. Quality control metric generation
```

**Example 2: Conversion with MSM Surface Registration**

```bash
# MSM provides superior surface alignment compared to FreeSurfer's spherical registration
# Requires MSM binary installed

ciftify_recon_all \
  --fs-subjects-dir /path/to/freesurfer/subjects \
  --ciftify-work-dir /path/to/ciftify/output \
  --surf-reg MSMSulc \
  sub-001

# --surf-reg options:
# - FS: FreeSurfer's spherical registration (default, fast)
# - MSMSulc: MSM using sulcal depth (recommended, slower but more accurate)
# - MSMAll: MSM using multiple features (best alignment, slowest)
```

**Example 3: Batch Processing Multiple Subjects**

```bash
# Process all subjects in FreeSurfer directory
for subject in $(ls /path/to/freesurfer/subjects); do
  if [ -d "/path/to/freesurfer/subjects/$subject/surf" ]; then
    echo "Processing $subject"
    ciftify_recon_all \
      --fs-subjects-dir /path/to/freesurfer/subjects \
      --ciftify-work-dir /path/to/ciftify/output \
      --surf-reg MSMSulc \
      $subject
  fi
done
```

### Understanding CIFTI Output Structure

**Example 4: Exploring CIFTIFY Output Directory**

```bash
# CIFTIFY output structure
ciftify_output/
├── sub-001/
│   ├── MNINonLinear/
│   │   ├── fsaverage_LR32k/
│   │   │   ├── sub-001.L.midthickness.32k_fs_LR.surf.gii
│   │   │   ├── sub-001.R.midthickness.32k_fs_LR.surf.gii
│   │   │   ├── sub-001.L.thickness.32k_fs_LR.shape.gii
│   │   │   ├── sub-001.R.thickness.32k_fs_LR.shape.gii
│   │   │   ├── sub-001.sulc.32k_fs_LR.dscalar.nii  # Sulcal depth
│   │   │   └── sub-001.corrThickness.32k_fs_LR.dscalar.nii
│   │   ├── Native/
│   │   │   └── sub-001.aparc.32k_fs_LR.dlabel.nii  # Parcellation
│   │   ├── ROIs/
│   │   │   ├── Atlas_ROIs.2.nii.gz  # Subcortical structures
│   │   │   └── ROIs.2.nii.gz
│   │   └── sub-001.164k_fs_LR.wb.spec  # Workbench spec file
│   └── T1w/
│       └── sub-001/
│           ├── surf/  # Original FreeSurfer surfaces
│           └── label/

# Key file types:
# - .surf.gii: Surface geometry (vertices, triangles)
# - .shape.gii: Scalar data on surface (thickness, curvature)
# - .dscalar.nii: Dense scalar CIFTI (cortical + subcortical)
# - .dlabel.nii: Dense label CIFTI (parcellations)
# - .dtseries.nii: Dense timeseries CIFTI (functional data)
```

## Quality Control

### Automated QC Report Generation

**Example 5: Generate Quality Control HTML Report**

```bash
# Generate comprehensive QC report for single subject
cifti_vis_recon_all \
  --ciftify-work-dir /path/to/ciftify/output \
  subject \
  sub-001

# Output: /path/to/ciftify/output/qc_recon_all/sub-001/qc.html

# QC report includes:
# - Surface reconstruction overlays on T1w
# - Pial and white surface boundaries
# - Subcortical segmentation
# - Surface mesh quality metrics
# - Cortical thickness maps
# - Registration quality to fsLR
```

**Example 6: Batch QC for All Subjects**

```bash
# Generate QC reports for all subjects
cifti_vis_recon_all snaps \
  --ciftify-work-dir /path/to/ciftify/output

# Creates index.html with all subjects
# Navigate to: /path/to/ciftify/output/qc_recon_all/index.html

# Allows rapid visual inspection of:
# - Surface reconstruction quality
# - Segmentation accuracy
# - Registration quality
# - Outlier detection
```

**Example 7: Quality Control Metrics Extraction**

```python
# Extract QC metrics for statistical analysis
import pandas as pd
import glob

qc_files = glob.glob('/path/to/ciftify/output/*/qc_recon_all.csv')

qc_data = []
for qc_file in qc_files:
    df = pd.read_csv(qc_file)
    qc_data.append(df)

qc_df = pd.concat(qc_data, ignore_index=True)

# QC metrics include:
# - Surface smoothness
# - Euler number (topology defects)
# - Mean curvature
# - Thickness statistics
# - Registration correlation

# Flag subjects needing manual review
flagged = qc_df[
    (qc_df['lh_euler'] < -50) |  # Topology issues
    (qc_df['rh_euler'] < -50) |
    (qc_df['mean_thickness'] < 2.0) |  # Unrealistic thickness
    (qc_df['mean_thickness'] > 3.5)
]

print(f"Subjects flagged for review: {len(flagged)}")
print(flagged[['subject_id', 'lh_euler', 'rh_euler', 'mean_thickness']])
```

## Functional Data Processing

### Converting Functional MRI to CIFTI

**Example 8: Project fMRI to Surface and Create Dense Timeseries**

```bash
# Convert volumetric fMRI to CIFTI dense timeseries
# Prerequisites: FreeSurfer recon-all and ciftify_recon_all completed

ciftify_subject_fmri \
  --ciftify-work-dir /path/to/ciftify/output \
  --surf-reg MSMSulc \
  sub-001 \
  /path/to/func/sub-001_task-rest_bold.nii.gz \
  sub-001_task-rest

# Output:
# sub-001/MNINonLinear/Results/sub-001_task-rest/
#   sub-001_task-rest_Atlas.dtseries.nii  # Dense timeseries CIFTI
#   sub-001_task-rest.L.native.func.gii   # Left hemisphere surface
#   sub-001_task-rest.R.native.func.gii   # Right hemisphere surface

# Dense timeseries includes:
# - Cortical surface data (32k vertices per hemisphere)
# - Subcortical voxels (thalamus, caudate, putamen, etc.)
# - Unified grayordinates format
```

**Example 9: Smoothing on Surface**

```bash
# Apply smoothing to CIFTI dense timeseries
# Uses geodesic smoothing on surface (better than volumetric)

wb_command -cifti-smoothing \
  sub-001_task-rest_Atlas.dtseries.nii \
  5.0 \  # Surface smoothing FWHM (mm)
  5.0 \  # Volume smoothing FWHM (mm)
  COLUMN \
  sub-001_task-rest_Atlas_s5.dtseries.nii \
  -left-surface sub-001.L.midthickness.32k_fs_LR.surf.gii \
  -right-surface sub-001.R.midthickness.32k_fs_LR.surf.gii

# Geodesic smoothing respects cortical topology
# Prevents smoothing across sulci
```

**Example 10: Extracting ROI Timeseries from CIFTI**

```python
# Extract ROI average timeseries using Connectome Workbench
import subprocess
import nibabel as nib

# Apply parcellation to extract ROI timeseries
cmd = [
    'wb_command', '-cifti-parcellate',
    'sub-001_task-rest_Atlas.dtseries.nii',
    'Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii',  # HCP MMP1.0 parcellation
    'COLUMN',
    'sub-001_task-rest_parcellated.ptseries.nii'
]
subprocess.run(cmd)

# Load parcellated timeseries
img = nib.load('sub-001_task-rest_parcellated.ptseries.nii')
data = img.get_fdata()  # Shape: (timepoints, 360 ROIs for MMP1.0)

print(f"Timeseries shape: {data.shape}")
print(f"Parcels: {data.shape[1]}")

# Compute connectivity matrix
import numpy as np
corr_matrix = np.corrcoef(data.T)
print(f"Connectivity matrix: {corr_matrix.shape}")
```

## Integration with HCP Pipelines

**Example 11: Using CIFTIFY Output with HCP Analysis Tools**

```bash
# CIFTIFY outputs are compatible with HCP processing scripts
# Example: ICA-FIX for artifact removal

# 1. Run melodic ICA on CIFTI data
wb_command -cifti-reduce \
  sub-001_task-rest_Atlas.dtseries.nii \
  MEAN \
  sub-001_task-rest_mean.dscalar.nii

# 2. Run ICA using FSL melodic
melodic \
  -i sub-001_task-rest_Atlas.dtseries.nii \
  -o sub-001_task-rest.ica \
  --dim=200 \
  --tr=2.0

# 3. Classify components with FIX
# (Requires trained FIX classifier)

# 4. Clean data with ICA-FIX
# Output: cleaned dense timeseries
```

**Example 12: Surface-Based Group Analysis**

```bash
# Prepare group analysis with PALM (Permutation Analysis of Linear Models)

# 1. Merge individual CIFTI scalar files
wb_command -cifti-merge \
  group_thickness.dscalar.nii \
  -cifti sub-001.thickness.32k_fs_LR.dscalar.nii \
  -cifti sub-002.thickness.32k_fs_LR.dscalar.nii \
  -cifti sub-003.thickness.32k_fs_LR.dscalar.nii
  # ... (all subjects)

# 2. Run PALM for group statistics
palm \
  -i group_thickness.dscalar.nii \
  -d design.mat \
  -t contrasts.con \
  -n 10000 \  # Permutations
  -o palm_output

# 3. Visualize results in Connectome Workbench
wb_view palm_output_tfce_tstat.dscalar.nii
```

## BIDS Integration

**Example 13: CIFTIFY with BIDS Datasets**

```bash
# CIFTIFY supports BIDS-formatted datasets
# Assumes fMRIPrep or FreeSurfer processing

# Convert BIDS derivatives to CIFTI
ciftify_recon_all \
  --fs-subjects-dir /bids/derivatives/freesurfer \
  --ciftify-work-dir /bids/derivatives/ciftify \
  --hcp-data-dir /bids/derivatives/ciftify \
  sub-001

# Process functional data with BIDS naming
for run in 1 2 3; do
  ciftify_subject_fmri \
    --ciftify-work-dir /bids/derivatives/ciftify \
    sub-001 \
    /bids/derivatives/fmriprep/sub-001/func/sub-001_task-rest_run-${run}_space-T1w_desc-preproc_bold.nii.gz \
    task-rest_run-${run}
done
```

## Visualization with Connectome Workbench

**Example 14: Viewing CIFTI Files in Workbench**

```bash
# Launch Connectome Workbench GUI
wb_view

# Or load specific files from command line
wb_view \
  sub-001.L.midthickness.32k_fs_LR.surf.gii \
  sub-001.R.midthickness.32k_fs_LR.surf.gii \
  sub-001.thickness.32k_fs_LR.dscalar.nii

# Workbench allows:
# - Interactive surface visualization
# - Overlaying scalar data (thickness, activation)
# - Viewing parcellations
# - Comparing multiple maps
# - Creating publication-quality figures
```

**Example 15: Creating Scene Files for Batch Visualization**

```python
# Automate Workbench visualization with scene files
import os

scene_template = """<?xml version="1.0" encoding="UTF-8"?>
<SceneFile Version="1">
  <Scene Name="{subject}_thickness">
    <SceneClass Name="Surface">
      <MapOne Index="0" ScalarFile="{subject}.thickness.32k_fs_LR.dscalar.nii"/>
    </SceneClass>
  </Scene>
</SceneFile>
"""

subjects = ['sub-001', 'sub-002', 'sub-003']

for subject in subjects:
    scene_content = scene_template.format(subject=subject)
    with open(f'{subject}_scene.scene', 'w') as f:
        f.write(scene_content)

    # Load scene in Workbench
    os.system(f'wb_view -scene-load {subject}_scene.scene 1')
```

## Troubleshooting

### Common Issues and Solutions

**FreeSurfer Segmentation Errors:**
```bash
# If ciftify_recon_all fails due to FreeSurfer issues:

# 1. Check FreeSurfer completion
ls $SUBJECTS_DIR/sub-001/surf/lh.pial  # Should exist

# 2. Verify FreeSurfer version compatibility
recon-all -version  # Should be 6.0 or higher

# 3. Re-run FreeSurfer if necessary
recon-all -all -s sub-001 -i /path/to/T1.nii.gz

# 4. Then retry CIFTIFY
ciftify_recon_all --fs-subjects-dir $SUBJECTS_DIR \
  --ciftify-work-dir /output sub-001
```

**MSM Registration Failures:**
```bash
# If MSM fails during surface registration:

# 1. Check MSM installation
which msm  # Should return path to MSM binary

# 2. Fall back to FreeSurfer registration
ciftify_recon_all \
  --surf-reg FS \  # Use FreeSurfer instead of MSM
  --fs-subjects-dir $SUBJECTS_DIR \
  --ciftify-work-dir /output \
  sub-001

# 3. Or install MSM separately
# Download from: https://github.com/ecr05/MSM_HOCR
```

**Memory Issues:**
```bash
# For large datasets or high-resolution surfaces:

# Limit parallel processing
export OMP_NUM_THREADS=1

# Use lower resolution (164k instead of 32k)
ciftify_recon_all \
  --resample-to-164k \
  --fs-subjects-dir $SUBJECTS_DIR \
  --ciftify-work-dir /output \
  sub-001
```

**CIFTI File Corruption:**
```bash
# Validate CIFTI files
wb_command -file-information sub-001.dscalar.nii

# If corrupted, regenerate
ciftify_recon_all \
  --fs-subjects-dir $SUBJECTS_DIR \
  --ciftify-work-dir /output \
  --rerun \  # Force reprocessing
  sub-001
```

## Advanced Applications

**Example 16: Multi-Modal Surface Analysis**

```python
# Combine multiple surface metrics in CIFTI format
import nibabel as nib
import numpy as np

# Load different metrics
thickness = nib.load('sub-001.thickness.32k_fs_LR.dscalar.nii')
myelin = nib.load('sub-001.MyelinMap.32k_fs_LR.dscalar.nii')
curvature = nib.load('sub-001.curvature.32k_fs_LR.dscalar.nii')

thickness_data = thickness.get_fdata().squeeze()
myelin_data = myelin.get_fdata().squeeze()
curvature_data = curvature.get_fdata().squeeze()

# Compute correlations
corr_thickness_myelin = np.corrcoef(thickness_data, myelin_data)[0, 1]
print(f"Thickness-Myelin correlation: {corr_thickness_myelin:.3f}")

# Create composite metric
composite = (thickness_data * myelin_data) / (curvature_data + 1)

# Save as new CIFTI
composite_img = nib.Cifti2Image(composite.reshape(1, -1), header=thickness.header)
nib.save(composite_img, 'sub-001.composite.32k_fs_LR.dscalar.nii')
```

**Example 17: Longitudinal Analysis**

```bash
# Process longitudinal data with CIFTIFY

# Timepoint 1
ciftify_recon_all \
  --fs-subjects-dir $SUBJECTS_DIR \
  --ciftify-work-dir /output \
  sub-001_tp1

# Timepoint 2
ciftify_recon_all \
  --fs-subjects-dir $SUBJECTS_DIR \
  --ciftify-work-dir /output \
  sub-001_tp2

# Compute change in thickness
wb_command -cifti-math '(tp2 - tp1) / tp1 * 100' \
  sub-001_thickness_change.dscalar.nii \
  -var tp1 sub-001_tp1.thickness.32k_fs_LR.dscalar.nii \
  -var tp2 sub-001_tp2.thickness.32k_fs_LR.dscalar.nii

# Visualize change map
wb_view sub-001_thickness_change.dscalar.nii
```

**Example 18: Custom Parcellation Application**

```bash
# Apply custom parcellation to CIFTI data

# 1. Create custom parcellation in CIFTI format
# (Assume parcellation created in FreeSurfer space)

# 2. Convert to CIFTI label file
wb_command -label-to-volume-mapping \
  custom_parcellation.label.gii \
  sub-001.L.midthickness.32k_fs_LR.surf.gii \
  custom_parcellation.32k_fs_LR.dlabel.nii

# 3. Extract ROI timeseries
wb_command -cifti-parcellate \
  sub-001_task-rest_Atlas.dtseries.nii \
  custom_parcellation.32k_fs_LR.dlabel.nii \
  COLUMN \
  sub-001_task-rest_custom_parcels.ptseries.nii

# 4. Analyze parcellated data
# (e.g., compute connectivity matrix)
```

## Best Practices

**Data Organization:**
- Maintain separate directories for FreeSurfer and CIFTIFY outputs
- Use consistent naming conventions across subjects
- Keep QC reports with processed data
- Version control processing scripts

**Quality Control:**
- Always review QC HTML reports before analysis
- Check surface reconstruction quality (pial, white matter boundaries)
- Verify registration to fsLR template
- Inspect subcortical segmentation
- Flag and manually correct problematic subjects

**Processing Recommendations:**
- Use MSM registration when possible (better alignment)
- Generate QC reports immediately after processing
- Process functional and anatomical data together
- Keep original FreeSurfer outputs for reprocessing if needed

**HPC Optimization:**
- Parallelize across subjects, not within subjects
- Allocate sufficient memory (8-16 GB per subject)
- Use local scratch space for temporary files
- Clean up intermediate files after successful completion

## Integration with Analysis Ecosystem

**FreeSurfer:**
- CIFTIFY requires FreeSurfer recon-all outputs
- Preserves FreeSurfer parcellations in CIFTI format
- Can reconvert from CIFTI back to FreeSurfer space

**Connectome Workbench:**
- Primary visualization tool for CIFTI files
- All Workbench commands work with CIFTIFY outputs
- Scene files enable batch visualization

**HCP Pipelines:**
- CIFTIFY outputs compatible with HCP minimal preprocessing
- Can feed into HCP-style ICA-FIX, MSMAll
- Integrates with HCP group analysis tools

**Python Ecosystem:**
- Nibabel for CIFTI I/O
- Nilearn for connectivity analysis
- HCPpipelines for advanced processing

## References

**CIFTIFY:**
- Dickie et al. (2019). Ciftify: A framework for surface-based analysis of legacy MR acquisitions. *NeuroImage*, 197, 818-826.

**CIFTI Format:**
- Marcus et al. (2013). Human Connectome Project informatics: Quality control, database services, and data visualization. *NeuroImage*, 80, 202-219.

**HCP Processing:**
- Glasser et al. (2013). The minimal preprocessing pipelines for the Human Connectome Project. *NeuroImage*, 80, 105-124.

**MSM Registration:**
- Robinson et al. (2014). MSM: A new flexible framework for multimodal surface matching. *NeuroImage*, 100, 414-426.

**Surface-Based Analysis:**
- Glasser et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature*, 536(7615), 171-178.

**Online Resources:**
- CIFTIFY Documentation: https://edickie.github.io/ciftify/
- CIFTIFY GitHub: https://github.com/edickie/ciftify
- Connectome Workbench: https://www.humanconnectome.org/software/connectome-workbench
- HCP Wiki: https://wiki.humanconnectome.org/
- FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
