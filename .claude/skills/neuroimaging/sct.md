# Spinal Cord Toolbox (SCT): Comprehensive Spinal Cord MRI Analysis

## Overview

Spinal Cord Toolbox (SCT) is a comprehensive, open-source software for processing and analyzing spinal cord MRI data. It addresses the unique challenges of spinal cord imaging—small anatomical structure, susceptibility to motion artifacts, and partial volume effects—through automated segmentation, registration to standardized atlases, and extraction of quantitative metrics.

**Key Features:**
- **Automated Segmentation**: Deep learning-based spinal cord and gray matter segmentation
- **Template Registration**: Alignment to PAM50 spinal cord atlas
- **Vertebral Labeling**: Automatic detection of vertebral levels
- **Quantitative Metrics**: Cross-sectional area (CSA), DTI metrics, MTR, lesion quantification
- **Multi-Parametric Analysis**: Support for T1w, T2w, T2*w, DWI, MTR, fMRI
- **Quality Control**: Comprehensive QC reports with visualization
- **BIDS Compatible**: Native support for BIDS datasets

**Website:** https://spinalcordtoolbox.com/

**Citation:** De Leener, B., et al. (2017). SCT: Spinal Cord Toolbox, an open-source software for processing spinal cord MRI data. *NeuroImage*, 145, 24-43.

## Installation

### Using Conda (Recommended)

```bash
# Create conda environment
conda create -n sct python=3.9

# Activate environment
conda activate sct

# Install SCT
pip install spinalcordtoolbox

# Test installation
sct_check_dependencies
```

### Using pip

```bash
# Install from PyPI
pip install spinalcordtoolbox

# Run installer to complete setup
sct_download_data -d sct_testing_data

# Verify installation
sct_version
```

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git
cd spinalcordtoolbox

# Install
./install_sct

# Add to PATH
export PATH="${PATH}:/path/to/spinalcordtoolbox/bin"

# Test installation
sct_check_dependencies
```

### Requirements

- **Operating System**: Linux, macOS, Windows (WSL2)
- **Python**: 3.7+
- **FSLeyes**: Recommended for visualization and manual corrections
- **FSL**: Optional, for advanced preprocessing
- **ANTs**: Included with SCT

## Basic Spinal Cord Segmentation

### Automatic Segmentation with DeepSeg

```bash
# Segment spinal cord from T2w image
sct_deepseg_sc -i t2.nii.gz -c t2 -o t2_seg.nii.gz

# Segment from T1w image
sct_deepseg_sc -i t1.nii.gz -c t1 -o t1_seg.nii.gz

# Segment with quality control
sct_deepseg_sc -i t2.nii.gz -c t2 -qc qc_folder
```

**Supported Contrasts:**
- `t1`: T1-weighted
- `t2`: T2-weighted
- `t2s`: T2*-weighted
- `dwi`: Diffusion-weighted

### Viewing Segmentation Results

```bash
# Open in FSLeyes for visual inspection
fsleyes t2.nii.gz t2_seg.nii.gz -cm red -a 50

# Or use SCT's viewer
sct_viewer -i t2.nii.gz -s t2_seg.nii.gz
```

### Manual Correction of Segmentation

```bash
# Open for manual editing in FSLeyes
# Install SCT FSLeyes plugin first
fsleyes t2.nii.gz t2_seg.nii.gz -cm red

# After manual edits, smooth the segmentation
sct_maths -i t2_seg_manual.nii.gz -smooth 1,1,0 -o t2_seg_smooth.nii.gz
```

### Batch Segmentation

```bash
# Segment multiple subjects
for subject in sub-01 sub-02 sub-03; do
    echo "Processing ${subject}"
    sct_deepseg_sc \
        -i ${subject}/anat/${subject}_T2w.nii.gz \
        -c t2 \
        -o ${subject}/anat/${subject}_T2w_seg.nii.gz \
        -qc qc_segmentation
done
```

## Gray and White Matter Segmentation

### Gray Matter Segmentation from T2*w

```bash
# Segment gray matter
sct_deepseg_gm -i t2s.nii.gz -o t2s_gmseg.nii.gz

# Quality control
sct_deepseg_gm -i t2s.nii.gz -o t2s_gmseg.nii.gz -qc qc_folder

# The output contains:
# - Gray matter segmentation
# - White matter (by subtraction from cord segmentation)
```

### White Matter Tract Parcellation

```bash
# First, register to template (see next section)
# Then use template-based white matter atlas

# Warp atlas to native space
sct_warp_template \
    -d t2.nii.gz \
    -w warp_template2anat.nii.gz \
    -a 1 \
    -o white_matter_atlas

# Extract specific tracts
# 0=dorsal columns, 1=lateral CST, 2=ventral CST, etc.
```

### Computing Cross-Sectional Areas

```bash
# Compute CSA of gray matter per vertebral level
sct_process_segmentation \
    -i t2s_gmseg.nii.gz \
    -vertfile vertebral_labeling.nii.gz \
    -perslice 0 \
    -vert 2:5 \
    -o gm_csa.csv

# Compute white matter CSA
sct_process_segmentation \
    -i wm_seg.nii.gz \
    -vertfile vertebral_labeling.nii.gz \
    -perslice 0 \
    -o wm_csa.csv
```

## Vertebral Labeling

### Automatic Vertebral Level Detection

```bash
# Detect vertebral levels automatically
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2

# Output: vertebral_labeling.nii.gz with labels 1-24
# C1=1, C2=2, ..., C7=7, T1=8, ..., L5=24

# Create labels for specific levels only
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 \
    -initlabel 2 -initz 50  # Initialize at C2, z-coordinate 50
```

### Manual Vertebral Labeling

```bash
# Create manual labels using sct_label_utils
sct_label_utils -i t2.nii.gz -create-viewer 3,4,5 -o labels_disc.nii.gz

# Labels correspond to intervertebral discs
# 3 = C2-C3, 4 = C3-C4, 5 = C4-C5, etc.

# Then generate vertebral labeling from disc labels
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -discfile labels_disc.nii.gz
```

### Verifying Vertebral Levels

```bash
# Generate QC report
sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d vertebral_labeling.nii.gz \
    -p sct_label_vertebrae -qc qc_folder

# View in FSLeyes
fsleyes t2.nii.gz vertebral_labeling.nii.gz -cm random
```

## Registration to PAM50 Template

### Basic Template Registration

```bash
# Register subject to PAM50 template
sct_register_to_template \
    -i t2.nii.gz \
    -s t2_seg.nii.gz \
    -l vertebral_labeling.nii.gz \
    -c t2 \
    -qc qc_registration

# Output files:
# - warp_template2anat.nii.gz: Template → Native
# - warp_anat2template.nii.gz: Native → Template
# - PAM50/template/: Template files in native space
```

### Advanced Registration Options

```bash
# Registration with custom parameters
sct_register_to_template \
    -i t2.nii.gz \
    -s t2_seg.nii.gz \
    -l vertebral_labeling.nii.gz \
    -c t2 \
    -param step=1,type=seg,algo=centermassrot:step=2,type=im,algo=syn,slicewise=1,iter=5 \
    -qc qc_registration

# Multi-channel registration (using T1 + T2)
sct_register_multimodal \
    -i t1.nii.gz \
    -d t2.nii.gz \
    -iseg t1_seg.nii.gz \
    -dseg t2_seg.nii.gz \
    -param step=1,type=seg,algo=centermass:step=2,type=im,algo=syn,iter=5
```

### Warping Data to Template Space

```bash
# Warp anatomical image to template
sct_apply_transfo \
    -i t2.nii.gz \
    -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz \
    -w warp_anat2template.nii.gz \
    -o t2_template.nii.gz

# Warp metric map (e.g., FA) to template
sct_apply_transfo \
    -i dti_FA.nii.gz \
    -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz \
    -w warp_anat2template.nii.gz \
    -o dti_FA_template.nii.gz \
    -x nn  # nearest neighbor for discrete values
```

### Straightening the Spinal Cord

```bash
# Straighten spinal cord along centerline
sct_straighten_spinalcord \
    -i t2.nii.gz \
    -s t2_seg.nii.gz \
    -o t2_straight.nii.gz

# Output:
# - t2_straight.nii.gz: Straightened image
# - straight2curved.nii.gz: Transformation to original space
# - curve2straight.nii.gz: Transformation to straight space
```

## Metric Extraction

### Cross-Sectional Area (CSA)

```bash
# Compute CSA per vertebral level
sct_process_segmentation \
    -i t2_seg.nii.gz \
    -vertfile vertebral_labeling.nii.gz \
    -perslice 0 \
    -vert 2:5 \
    -o csa_per_level.csv

# Output CSV contains:
# VertLevel, MEAN(area), STD(area)
# 2, 75.3, 2.1  (C2)
# 3, 73.8, 2.3  (C3)
# etc.

# Compute CSA per slice
sct_process_segmentation \
    -i t2_seg.nii.gz \
    -perslice 1 \
    -o csa_per_slice.csv
```

### DTI Metrics Extraction

```bash
# First, compute DTI metrics using sct_dmri_moco and sct_dmri_compute_dti
# Motion correction
sct_dmri_moco -i dwi.nii.gz -bvec bvecs.txt -o dwi_moco.nii.gz

# Compute DTI
sct_dmri_compute_dti \
    -i dwi_moco.nii.gz \
    -bval bvals.txt \
    -bvec bvecs_moco.txt

# Extract FA in white matter tracts
sct_extract_metric \
    -i dti_FA.nii.gz \
    -f atlas/PAM50_wm.nii.gz \
    -l 0,1,2 \
    -vert 2:5 \
    -vertfile vertebral_labeling.nii.gz \
    -o fa_in_wm.csv \
    -method wa  # weighted average

# Tracts: 0=dorsal columns, 1=lateral CST, 2=ventral CST
```

### Magnetization Transfer Ratio (MTR)

```bash
# Compute MTR from MT-on and MT-off images
sct_compute_mtr -mt0 mt0.nii.gz -mt1 mt1.nii.gz -o mtr.nii.gz

# Extract MTR in spinal cord per level
sct_extract_metric \
    -i mtr.nii.gz \
    -f t2_seg.nii.gz \
    -method wa \
    -vert 2:7 \
    -vertfile vertebral_labeling.nii.gz \
    -o mtr_per_level.csv
```

### Shape Metrics

```bash
# Compute shape metrics (AP diameter, RL diameter, etc.)
sct_process_segmentation \
    -i t2_seg.nii.gz \
    -vertfile vertebral_labeling.nii.gz \
    -perslice 0 \
    -vert 2:5 \
    -o shape_metrics.csv \
    -shape 1  # Include shape metrics

# Output includes:
# - CSA
# - AP (anterior-posterior) diameter
# - RL (right-left) diameter
# - Eccentricity, orientation, etc.
```

## Atlas-Based Analysis

### Using the PAM50 White Matter Atlas

```bash
# After registration to template, warp atlas to native space
sct_warp_template \
    -d t2.nii.gz \
    -w warp_template2anat.nii.gz \
    -a 1 \
    -o atlas_native

# Atlas files:
# - PAM50_wm.nii.gz: White matter tracts
# - PAM50_gm.nii.gz: Gray matter regions
# - PAM50_levels.nii.gz: Vertebral levels

# Extract metrics per tract
sct_extract_metric \
    -i dti_FA.nii.gz \
    -f atlas_native/PAM50_wm.nii.gz \
    -l 0:15 \
    -method wa \
    -vert 2:5 \
    -vertfile vertebral_labeling.nii.gz \
    -o fa_per_tract.csv
```

### Tract-Specific Analysis

```bash
# Define white matter tracts (PAM50 atlas):
# 0: Dorsal columns
# 1,2: Lateral corticospinal tracts (left, right)
# 3,4: Ventral corticospinal tracts
# 5,6: Spinothalamic tracts
# 15,16: Dorsal columns (fasciculus cuneatus)

# Extract FA in dorsal columns (C2-C5)
sct_extract_metric \
    -i dti_FA.nii.gz \
    -f atlas_native/PAM50_wm.nii.gz \
    -l 0 \
    -method wa \
    -vert 2:5 \
    -vertfile vertebral_labeling.nii.gz \
    -o fa_dorsal_columns.csv

# Compare left vs. right CST
sct_extract_metric \
    -i dti_FA.nii.gz \
    -f atlas_native/PAM50_wm.nii.gz \
    -l 1,2 \
    -method wa \
    -vert 2:5 \
    -vertfile vertebral_labeling.nii.gz \
    -o fa_cst_laterality.csv
```

## Lesion Analysis

### Lesion Segmentation

```bash
# Manual lesion segmentation
sct_label_utils -i t2.nii.gz -create-seg -o lesion_manual.nii.gz

# Automated lesion detection (MS lesions)
sct_deepseg_lesion -i t2.nii.gz -c t2 -o lesion_auto.nii.gz

# Combine manual corrections
fslmaths lesion_auto.nii.gz -add lesion_manual.nii.gz -bin lesion_final.nii.gz
```

### Lesion Quantification

```bash
# Compute lesion volume and distribution
sct_analyze_lesion \
    -m lesion_final.nii.gz \
    -s t2_seg.nii.gz \
    -vertfile vertebral_labeling.nii.gz \
    -o lesion_analysis.pkl

# Extract lesion metrics
sct_extract_metric \
    -i lesion_final.nii.gz \
    -f t2_seg.nii.gz \
    -method bin \
    -vert 2:7 \
    -vertfile vertebral_labeling.nii.gz \
    -o lesion_per_level.csv

# Output: lesion volume per vertebral level
```

### Lesion Load in White Matter Tracts

```bash
# Compute lesion overlap with white matter tracts
sct_extract_metric \
    -i lesion_final.nii.gz \
    -f atlas_native/PAM50_wm.nii.gz \
    -l 0:15 \
    -method bin \
    -vert 2:7 \
    -vertfile vertebral_labeling.nii.gz \
    -o lesion_in_tracts.csv

# Identify which tracts are affected
```

## Quality Control and Visualization

### Generating QC Reports

```bash
# QC for segmentation
sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc -qc qc_folder

# QC for vertebral labeling
sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d vertebral_labeling.nii.gz \
    -p sct_label_vertebrae -qc qc_folder

# QC for registration
sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d PAM50/template/PAM50_t2.nii.gz \
    -p sct_register_to_template -qc qc_folder

# Open QC report
firefox qc_folder/index.html
```

### FSLeyes Plugin for Manual Corrections

```bash
# Install FSLeyes SCT plugin
pip install fsleyes-plugin-sct

# Open with SCT plugin active
fsleyes t2.nii.gz t2_seg.nii.gz -cm red -a 50

# Manual correction workflow:
# 1. Edit segmentation in FSLeyes
# 2. Save corrected mask
# 3. Re-run subsequent steps
```

### Creating Custom Visualizations

```bash
# Create sagittal view with segmentation overlay
sct_create_mask -i t2.nii.gz -p centerline,t2_seg.nii.gz -size 40 -o mask_vis.nii.gz

# Generate mid-sagittal slice
sct_extract_metric -i t2.nii.gz -o t2_midsag.nii.gz -slice-projection 0

# Overlay segmentation for publication
# (Use external tools like matplotlib/nilearn)
```

## Batch Processing

### BIDS-Compatible Workflow

```bash
#!/bin/bash
# Process all subjects in BIDS dataset

BIDS_DIR=/path/to/bids
OUTPUT_DIR=/path/to/derivatives/sct

for subject in ${BIDS_DIR}/sub-*; do
    sub=$(basename $subject)
    echo "Processing ${sub}"

    # Input files
    T2=${subject}/anat/${sub}_T2w.nii.gz

    # Output directory
    OUT=${OUTPUT_DIR}/${sub}/anat
    mkdir -p ${OUT}

    # Segmentation
    sct_deepseg_sc -i ${T2} -c t2 -o ${OUT}/${sub}_T2w_seg.nii.gz -qc ${OUTPUT_DIR}/qc

    # Vertebral labeling
    sct_label_vertebrae -i ${T2} -s ${OUT}/${sub}_T2w_seg.nii.gz -c t2 \
        -o ${OUT}/${sub}_T2w_labels.nii.gz -qc ${OUTPUT_DIR}/qc

    # Registration
    sct_register_to_template -i ${T2} -s ${OUT}/${sub}_T2w_seg.nii.gz \
        -l ${OUT}/${sub}_T2w_labels.nii.gz -c t2 -ofolder ${OUT} \
        -qc ${OUTPUT_DIR}/qc

    # CSA extraction
    sct_process_segmentation -i ${OUT}/${sub}_T2w_seg.nii.gz \
        -vertfile ${OUT}/${sub}_T2w_labels.nii.gz \
        -perslice 0 -vert 2:5 -o ${OUT}/${sub}_csa.csv
done

# Combine results
python -c "
import pandas as pd
import glob

dfs = []
for f in glob.glob('${OUTPUT_DIR}/*/anat/*_csa.csv'):
    sub = f.split('/')[5]
    df = pd.read_csv(f)
    df['subject'] = sub
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('${OUTPUT_DIR}/group_csa.csv', index=False)
print('Combined CSA results saved')
"
```

### HPC Parallel Processing

```bash
#!/bin/bash
#SBATCH --job-name=sct_batch
#SBATCH --array=1-50
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Load modules
module load sct/5.8

# Get subject ID
SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subject_list.txt)

# Process single subject
bash process_single_subject.sh ${SUBJECT_ID}
```

### Multi-Contrast Processing

```bash
#!/bin/bash
# Process multiple contrasts for single subject

SUB=$1
BIDS_DIR=/path/to/bids
OUT_DIR=/path/to/derivatives/sct/${SUB}

# T1w processing
sct_deepseg_sc -i ${BIDS_DIR}/${SUB}/anat/${SUB}_T1w.nii.gz -c t1 \
    -o ${OUT_DIR}/${SUB}_T1w_seg.nii.gz

# T2w processing
sct_deepseg_sc -i ${BIDS_DIR}/${SUB}/anat/${SUB}_T2w.nii.gz -c t2 \
    -o ${OUT_DIR}/${SUB}_T2w_seg.nii.gz

# T2*w and gray matter
sct_deepseg_sc -i ${BIDS_DIR}/${SUB}/anat/${SUB}_T2star.nii.gz -c t2s \
    -o ${OUT_DIR}/${SUB}_T2star_seg.nii.gz

sct_deepseg_gm -i ${BIDS_DIR}/${SUB}/anat/${SUB}_T2star.nii.gz \
    -o ${OUT_DIR}/${SUB}_T2star_gmseg.nii.gz

# Register T1 to T2 (reference)
sct_register_multimodal \
    -i ${OUT_DIR}/${SUB}_T1w.nii.gz \
    -d ${OUT_DIR}/${SUB}_T2w.nii.gz \
    -iseg ${OUT_DIR}/${SUB}_T1w_seg.nii.gz \
    -dseg ${OUT_DIR}/${SUB}_T2w_seg.nii.gz \
    -param step=1,type=seg,algo=centermass:step=2,type=im,algo=syn,iter=3 \
    -o ${OUT_DIR}/${SUB}_T1w_reg.nii.gz
```

## Integration with Other Tools

### SCT + FSL Pipeline

```bash
# Preprocessing with FSL, analysis with SCT

# 1. FSL: Motion correction (if needed)
mcflirt -in t2.nii.gz -out t2_moco.nii.gz

# 2. SCT: Segmentation and registration
sct_deepseg_sc -i t2_moco.nii.gz -c t2 -o t2_seg.nii.gz
sct_label_vertebrae -i t2_moco.nii.gz -s t2_seg.nii.gz -c t2

# 3. SCT: Extract metrics
sct_process_segmentation -i t2_seg.nii.gz \
    -vertfile vertebral_labeling.nii.gz -perslice 0 -o csa.csv

# 4. FSL: Statistical analysis (group level)
# Use extracted metrics in FSL randomise or PALM
```

### SCT + ANTs for Custom Registration

```bash
# Use ANTs for advanced registration, SCT for metrics

# 1. SCT segmentation
sct_deepseg_sc -i t2.nii.gz -c t2 -o t2_seg.nii.gz

# 2. ANTs registration with lesion cost masking
antsRegistrationSyN.sh -d 3 -f template.nii.gz -m t2.nii.gz \
    -x lesion_mask.nii.gz -o t2_to_template_

# 3. Apply transform to segmentation
antsApplyTransforms -d 3 -i t2_seg.nii.gz -r template.nii.gz \
    -t t2_to_template_1Warp.nii.gz -t t2_to_template_0GenericAffine.mat \
    -o t2_seg_template.nii.gz -n NearestNeighbor

# 4. SCT metrics in template space
sct_process_segmentation -i t2_seg_template.nii.gz -o csa_template.csv
```

### Python Scripting with SCT

```python
import os
from spinalcordtoolbox.scripts import sct_deepseg_sc, sct_process_segmentation
import pandas as pd

def process_subject_sct(t2_file, output_dir):
    """
    Process single subject with SCT
    """
    # Segmentation
    seg_file = os.path.join(output_dir, 'seg.nii.gz')
    sct_deepseg_sc.main([
        '-i', t2_file,
        '-c', 't2',
        '-o', seg_file
    ])

    # CSA extraction
    csa_file = os.path.join(output_dir, 'csa.csv')
    sct_process_segmentation.main([
        '-i', seg_file,
        '-perslice', '1',
        '-o', csa_file
    ])

    # Load results
    df = pd.read_csv(csa_file)
    mean_csa = df['MEAN(area)'].mean()

    return mean_csa

# Process multiple subjects
subjects = ['sub-01', 'sub-02', 'sub-03']
results = []

for sub in subjects:
    t2 = f'/data/{sub}/anat/{sub}_T2w.nii.gz'
    out = f'/derivatives/sct/{sub}'
    os.makedirs(out, exist_ok=True)

    csa = process_subject_sct(t2, out)
    results.append({'subject': sub, 'mean_csa': csa})

# Save group results
df_results = pd.DataFrame(results)
df_results.to_csv('group_csa_results.csv', index=False)
```

## Troubleshooting

### Segmentation Failures

**Problem:** Segmentation fails or has large gaps

```bash
# Solution 1: Try different contrast
sct_deepseg_sc -i t1.nii.gz -c t1  # Instead of t2

# Solution 2: Initialize with manual centerline
sct_get_centerline -i t2.nii.gz -c t2 -o centerline.nii.gz
sct_deepseg_sc -i t2.nii.gz -c t2 -centerline centerline.nii.gz

# Solution 3: Manual correction in FSLeyes
# Edit failed regions, smooth result
```

**Problem:** Segmentation extends beyond cord (includes CSF)

```bash
# Erode segmentation slightly
sct_maths -i t2_seg.nii.gz -erode 1 -o t2_seg_eroded.nii.gz

# Or use manual editing to refine boundaries
```

### Registration Issues

**Problem:** Registration to template fails

```bash
# Check vertebral labeling first
fsleyes t2.nii.gz vertebral_labeling.nii.gz -cm random

# Fix incorrect labels
sct_label_utils -i vertebral_labeling.nii.gz -remove 5 -o labels_fixed.nii.gz
sct_label_utils -i labels_fixed.nii.gz -create 3,100,200,10 -o labels_fixed.nii.gz

# Retry registration with fixed labels
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz \
    -l labels_fixed.nii.gz -c t2
```

**Problem:** Template warping creates artifacts

```bash
# Use less aggressive deformation
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l labels.nii.gz \
    -param step=1,type=seg,algo=centermassrot:step=2,type=im,algo=syn,iter=3 \
    -c t2
```

### Motion and Artifacts

**Problem:** Severe motion artifacts

```bash
# Apply motion correction before SCT
mcflirt -in t2.nii.gz -out t2_moco.nii.gz

# Or use sct_dmri_moco for DWI
sct_dmri_moco -i dwi.nii.gz -bvec bvecs.txt
```

**Problem:** CSF pulsation artifacts

```bash
# Use cardiac gating during acquisition if possible
# For existing data: average multiple acquisitions
fslmaths scan1.nii.gz -add scan2.nii.gz -div 2 t2_avg.nii.gz
```

### Memory Issues

**Problem:** Out of memory errors

```bash
# Reduce image resolution temporarily
sct_resample -i t2.nii.gz -mm 1x1x2 -o t2_lowres.nii.gz

# Process low-res, upsample results
sct_resample -i result_lowres.nii.gz -ref t2.nii.gz -o result.nii.gz
```

## Best Practices

### Acquisition Protocols

**Recommended Sequences:**
- **T2w**: 0.5-1mm in-plane, 1-2mm through-plane, sagittal
- **T2*w**: 0.25-0.5mm in-plane (for GM segmentation), axial
- **T1w**: 1mm isotropic for anatomical reference
- **DWI**: 1.25mm in-plane, 30+ directions, b=1000 s/mm²
- **MTR**: Matched MT-on and MT-off sequences

**Quality Considerations:**
- Minimize motion with patient restraints
- Use cardiac gating for T2*w (reduces CSF pulsation)
- Cover full region of interest (include vertebral bodies)
- Ensure adequate SNR (especially for DTI)

### Quality Control Workflow

```bash
# 1. Check raw data quality
fsleyes t2.nii.gz

# 2. Verify segmentation
sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc -qc qc

# 3. Verify vertebral labels
sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d labels.nii.gz \
    -p sct_label_vertebrae -qc qc

# 4. Verify registration
sct_qc -i t2.nii.gz -d PAM50/template/PAM50_t2.nii.gz \
    -p sct_register_to_template -qc qc

# 5. Review all QC reports
firefox qc/index.html
```

### Reporting Standards

Include in methods:
- SCT version: `sct_version`
- Specific commands and parameters used
- Quality control procedures
- Vertebral levels analyzed
- Atlas version (PAM50)
- Manual correction procedures (if any)

### Multi-Site Harmonization

```bash
# Ensure consistent processing across sites

# 1. Standardize to template space
sct_register_to_template -i t2.nii.gz -s seg.nii.gz -l labels.nii.gz -c t2

# 2. Extract metrics in template space (more comparable)
sct_extract_metric -i metric_template.nii.gz -f PAM50_wm.nii.gz \
    -method wa -vert 2:5 -o results.csv

# 3. Account for site effects in statistical models
# Use site as covariate in analysis
```

## References

- **SCT Main Paper**: De Leener, B., et al. (2017). SCT: Spinal Cord Toolbox. *NeuroImage*, 145, 24-43.
- **PAM50 Template**: De Leener, B., et al. (2018). PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. *NeuroImage*, 165, 170-179.
- **Gray Matter Segmentation**: Gros, C., et al. (2019). Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions. *NeuroImage*, 184, 901-915.
- **Documentation**: https://spinalcordtoolbox.com/
- **Forum**: https://forum.spinalcordmri.org/
- **GitHub**: https://github.com/spinalcordtoolbox/spinalcordtoolbox
