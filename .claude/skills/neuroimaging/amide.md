# AMIDE (Amide's a Medical Image Data Examiner)

## Overview

**AMIDE** (Amide's a Medical Image Data Examiner) is a free, open-source medical imaging viewer and analysis tool with comprehensive support for PET, SPECT, CT, and MRI data. Developed primarily for nuclear medicine applications, AMIDE provides powerful visualization capabilities, ROI (Region of Interest) analysis, image registration, SUV calculation, time-activity curve extraction, and kinetic modeling. Its cross-platform GUI makes it accessible for clinical researchers, nuclear medicine physicians, and multi-modal imaging studies.

AMIDE supports multiple medical imaging formats (DICOM, NIFTI, ANALYZE, Interfile) and provides both interactive visualization and scriptable batch processing through XML-based automation. It is particularly valuable for PET quantification workflows including amyloid imaging, FDG metabolism studies, and longitudinal PET analysis.

**Key Features:**
- Multi-format medical image support (DICOM, NIFTI, ANALYZE, Interfile)
- PET, SPECT, CT, and MRI visualization
- 3D volume rendering and multi-planar reformatting (MPR)
- ROI drawing, editing, and statistical analysis
- Image registration and multi-modal fusion
- SUV (Standardized Uptake Value) calculation
- Time-activity curve (TAC) analysis for dynamic PET/SPECT
- Kinetic modeling capabilities
- DICOM import/export with metadata handling
- XML-based scripting for batch processing
- Cross-platform support (Linux, macOS, Windows)

**Primary Use Cases:**
- Clinical PET/SPECT quantification
- Amyloid and tau PET analysis for Alzheimer's disease
- FDG PET metabolism studies
- PET-MR and PET-CT multi-modal imaging
- ROI-based quantification across timepoints
- Dynamic PET/SPECT analysis
- Teaching and demonstration of nuclear medicine concepts

**Official Documentation:** http://amide.sourceforge.net/

---

## Installation

### Linux (Ubuntu/Debian)

```bash
# Install from package manager (if available)
sudo apt-get update
sudo apt-get install amide

# Or install from source
sudo apt-get install build-essential libgtk-3-dev libgsl-dev \
  libvolpack1-dev libxml2-dev dcmtk libdcmtk-dev

# Download and compile AMIDE
wget https://sourceforge.net/projects/amide/files/amide/1.0.6/amide-1.0.6.tar.bz2
tar -xjf amide-1.0.6.tar.bz2
cd amide-1.0.6

./configure --prefix=/usr/local
make
sudo make install
```

### macOS

```bash
# Using Homebrew
brew install amide

# Or download pre-compiled .dmg from:
# http://amide.sourceforge.net/

# Verify installation
amide --version
```

### Windows

```powershell
# Download Windows installer from:
# http://amide.sourceforge.net/packages.html

# Run installer: amide-1.0.6-win32.exe
# Follow installation wizard

# Launch from Start Menu or:
& "C:\Program Files\AMIDE\amide.exe"
```

### Verify Installation

```bash
# Check AMIDE version
amide --version

# Test with help
amide --help

# Launch GUI
amide &
```

---

## Basic Usage

### Load and Visualize PET Data

```bash
# Launch AMIDE GUI
amide

# From GUI:
# File → Import → DICOM or File → Import → File (NIFTI/ANALYZE)
# Select PET image

# Or load from command line
amide pet_fdg.nii.gz

# Load with specific threshold
amide --threshold 0.5 pet_fdg.nii.gz
```

### Load Multiple Modalities (PET + MRI)

```bash
# Load PET and MRI together
amide pet_amyloid.nii.gz t1w_mri.nii.gz

# AMIDE will open both in separate data sets
# Use View → Multiple to see side-by-side
```

### Export Screenshots

```bash
# In GUI: View → Export → Image
# Select format: PNG, JPEG, TIFF, BMP

# Configure resolution and quality
# Export current view
```

### Save AMIDE Project

```bash
# File → Save Project As → project_name.amide
# Saves all loaded data, ROIs, and settings

# Load existing project
amide project_name.amide
```

---

## ROI Analysis for PET Quantification

### Draw ROIs Manually

```bash
# In AMIDE GUI:
# 1. Load PET image
amide pet_fdg.nii.gz

# 2. Tools → ROI → New ROI
# 3. Select ROI shape:
#    - Ellipsoid
#    - Cylinder
#    - Box
#    - Isocontour (threshold-based)
#    - Freehand

# 4. Draw ROI on image
# 5. Adjust position and size
# 6. ROI statistics automatically calculated
```

### Create Isocontour ROIs (Threshold-Based)

```bash
# In GUI:
# Tools → ROI → New ROI → Isocontour

# Set threshold value (e.g., SUV > 1.5)
# AMIDE creates ROI for all voxels above threshold

# View ROI statistics in ROI panel:
# - Mean value
# - Max value
# - Standard deviation
# - Volume
# - Integrated activity
```

### Copy ROIs Across Timepoints

```bash
# For longitudinal studies:
# 1. Draw ROI on baseline PET
# 2. Load follow-up PET
# 3. Register images (see registration section)
# 4. Edit → Copy ROI
# 5. Select target data set
# 6. Edit → Paste ROI

# ROI is transferred to registered space
```

### Export ROI Statistics

```bash
# In GUI:
# ROI → Export Statistics
# Select output format: CSV, TXT

# Example output:
# ROI_Name, Mean, StdDev, Max, Volume_mm3, Total_Activity
# Precuneus, 1.45, 0.23, 2.1, 8450.2, 12252.8
# Frontal, 1.38, 0.19, 1.9, 7320.5, 10102.3
```

---

## SUV Calculation and Normalization

### Configure SUV Parameters

```bash
# In AMIDE GUI:
# Data Set → Properties → SUV Settings

# Enter parameters:
# - Injected dose (MBq or mCi)
# - Injection time
# - Scan time
# - Patient weight (kg)
# - Decay correction half-life

# AMIDE automatically converts activity to SUV
```

### Calculate SUV from PET Image

```python
# For automation, use external Python with NiBabel
import nibabel as nib
import numpy as np

# Load PET image (activity concentration in Bq/mL)
pet_img = nib.load('pet_fdg.nii.gz')
pet_data = pet_img.get_fdata()

# SUV parameters
injected_dose_mbq = 370  # MBq
patient_weight_kg = 70   # kg
decay_factor = 0.85      # Pre-calculated decay correction

# Calculate SUV
injected_dose_bq = injected_dose_mbq * 1e6
body_weight_g = patient_weight_kg * 1000
suv_data = (pet_data / injected_dose_bq) * body_weight_g / decay_factor

# Save SUV image
suv_img = nib.Nifti1Image(suv_data, pet_img.affine, pet_img.header)
nib.save(suv_img, 'pet_fdg_suv.nii.gz')

# Load in AMIDE
# amide pet_fdg_suv.nii.gz
```

### Calculate SUVr (SUV Ratio)

```bash
# In AMIDE:
# 1. Draw target ROI (e.g., precuneus for amyloid)
# 2. Draw reference ROI (e.g., cerebellum)
# 3. Note mean SUV values:
#    Target: 1.85
#    Reference: 1.15
# 4. Calculate SUVr = 1.85 / 1.15 = 1.61
```

```python
# Automated SUVr calculation
import nibabel as nib
import numpy as np

# Load SUV image and masks
suv_img = nib.load('pet_amyloid_suv.nii.gz')
suv_data = suv_img.get_fdata()

target_mask = nib.load('precuneus_mask.nii.gz').get_fdata()
ref_mask = nib.load('cerebellum_mask.nii.gz').get_fdata()

# Calculate mean SUV in each region
target_suv = np.mean(suv_data[target_mask > 0])
ref_suv = np.mean(suv_data[ref_mask > 0])

# Calculate SUVr
suvr = target_suv / ref_suv
print(f"Target SUV: {target_suv:.2f}")
print(f"Reference SUV: {ref_suv:.2f}")
print(f"SUVr: {suvr:.2f}")

# Create SUVr image
suvr_data = suv_data / ref_suv
suvr_img = nib.Nifti1Image(suvr_data, suv_img.affine, suv_img.header)
nib.save(suvr_img, 'pet_amyloid_suvr.nii.gz')
```

---

## Multi-Modal Image Registration

### Register PET to MRI

```bash
# In AMIDE GUI:
# 1. Load PET and MRI
amide pet.nii.gz t1w_mri.nii.gz

# 2. View → Multiple Data Sets
# 3. Edit → Register
# 4. Select:
#    - Source: PET
#    - Target: MRI
# 5. Choose registration method:
#    - Manual (interactive)
#    - Automatic (mutual information)
# 6. Click "Register"

# 7. Verify alignment visually
# 8. Apply transformation

# 9. Export registered PET
# File → Export → Data Set → registered_pet.nii.gz
```

### Manual Registration

```bash
# For manual fine-tuning:
# 1. Edit → Register → Manual
# 2. Use controls:
#    - Translation (X, Y, Z sliders)
#    - Rotation (pitch, yaw, roll)
#    - Scaling (if needed)
# 3. Toggle overlay to check alignment
# 4. Adjust iteratively
# 5. Accept transformation
```

### Automated Registration (External)

```python
# Use ANTs for registration, then load in AMIDE
import os

# Register PET to MRI using ANTs
os.system("""
antsRegistrationSyN.sh \
  -d 3 \
  -f t1w_mri.nii.gz \
  -m pet_fdg.nii.gz \
  -o pet_to_mri_ \
  -t r
""")

# This creates: pet_to_mri_Warped.nii.gz

# Load registered images in AMIDE
os.system("amide t1w_mri.nii.gz pet_to_mri_Warped.nii.gz")
```

### PET-CT Fusion

```bash
# Load PET and CT
amide pet_fdg.nii.gz ct.nii.gz

# CT and PET often pre-registered from scanner
# If alignment needed:
# Edit → Register → Automatic
# Mutual information works well for PET-CT

# Create fused display:
# View → Overlay Mode → Blend
# Adjust opacity sliders for optimal visualization
```

---

## Time-Activity Curve (TAC) Analysis

### Extract TAC from Dynamic PET

```bash
# Load dynamic PET (4D data)
amide dynamic_fdg_4d.nii.gz

# AMIDE recognizes 4D data and shows time controls

# Draw ROI on desired region
# Tools → ROI → New ROI → Ellipsoid
# Position over region of interest

# View TAC:
# Analysis → Time-Activity Curve
# Select ROI
# AMIDE plots activity vs. time

# Export TAC data:
# TAC window → Export → CSV
```

### Multi-Region TAC Comparison

```bash
# In AMIDE:
# 1. Draw multiple ROIs:
#    - ROI_1: Myocardium
#    - ROI_2: Blood pool
#    - ROI_3: Liver
# 2. Analysis → Time-Activity Curve
# 3. Select all ROIs
# 4. AMIDE plots all TACs on same graph
# 5. Export for kinetic modeling
```

### TAC Processing with Python

```python
# Process TAC data exported from AMIDE
import pandas as pd
import matplotlib.pyplot as plt

# Load exported TAC
tac_data = pd.read_csv('tac_export.csv')
# Columns: Time_min, ROI_1_Bq_mL, ROI_2_Bq_mL

time = tac_data['Time_min'].values
roi1 = tac_data['ROI_1_Bq_mL'].values
roi2 = tac_data['ROI_2_Bq_mL'].values

# Plot TACs
plt.figure(figsize=(10, 6))
plt.plot(time, roi1, 'o-', label='Myocardium')
plt.plot(time, roi2, 's-', label='Blood Pool')
plt.xlabel('Time (min)')
plt.ylabel('Activity (Bq/mL)')
plt.title('Time-Activity Curves')
plt.legend()
plt.grid(True)
plt.savefig('tac_plot.png', dpi=300)
```

---

## Kinetic Modeling

### Simple Logan Plot Analysis

```python
# Logan plot for reversible tracer (e.g., FDG)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# TAC data from AMIDE export
time = np.array([0, 2, 5, 10, 20, 30, 45, 60])  # minutes
tissue = np.array([0.5, 2.1, 3.8, 5.2, 6.1, 6.5, 6.7, 6.8])  # kBq/mL
blood = np.array([5.2, 4.1, 3.2, 2.5, 1.8, 1.4, 1.1, 0.9])  # kBq/mL

# Calculate cumulative integrals
time_hours = time / 60
cum_tissue = np.cumsum(tissue[:-1] + tissue[1:]) * np.diff(time_hours) / 2
cum_blood = np.cumsum(blood[:-1] + blood[1:]) * np.diff(time_hours) / 2

# Logan plot (for t > t*)
t_star_idx = 3  # Start after 10 min
x = np.append([0], cum_blood[t_star_idx:] / tissue[t_star_idx+1:])
y = np.append([0], cum_tissue[t_star_idx:] / tissue[t_star_idx+1:])

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(x[1:], y[1:])

# Distribution volume (slope of Logan plot)
dv = slope
print(f"Distribution Volume: {dv:.3f}")
print(f"R² = {r_value**2:.3f}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', markersize=8)
plt.plot(x, slope*x + intercept, 'r-', label=f'DV = {dv:.3f}')
plt.xlabel('∫Cp(τ)dτ / Ct(t)')
plt.ylabel('∫Ct(τ)dτ / Ct(t)')
plt.title('Logan Plot Analysis')
plt.legend()
plt.grid(True)
plt.savefig('logan_plot.png', dpi=300)
```

### Two-Tissue Compartment Model

```python
# Simplified 2-tissue compartment model
from scipy.optimize import curve_fit

def two_tissue_model(t, K1, k2, k3, k4, vb):
    """
    Two-tissue compartment model for PET
    K1: Blood to tissue transfer constant
    k2: Tissue to blood rate constant
    k3: Free to bound transition rate
    k4: Bound to free transition rate
    vb: Blood volume fraction
    """
    # Simplified implementation (requires input function)
    # This is a placeholder - full implementation needs blood input
    return K1 * (1 - np.exp(-k2 * t)) + vb

# Fit model to tissue TAC
# (Requires arterial blood input function)
time = np.array([0, 2, 5, 10, 20, 30, 45, 60])
tissue = np.array([0.5, 2.1, 3.8, 5.2, 6.1, 6.5, 6.7, 6.8])

# Initial parameter guess
p0 = [0.5, 0.1, 0.05, 0.02, 0.05]

# Fit (simplified)
# popt, pcov = curve_fit(two_tissue_model, time, tissue, p0=p0)
# print(f"K1 = {popt[0]:.3f} mL/min/mL")
```

---

## 3D Visualization and Volume Rendering

### Configure 3D Volume Rendering

```bash
# In AMIDE GUI:
# View → 3D Rendering

# Configure rendering parameters:
# - Rendering mode: Volume rendering
# - Transfer function: Adjust opacity curve
# - Color map: Hot, Rainbow, Grayscale
# - Lighting: Ambient, diffuse, specular

# Rotate 3D view:
# - Click and drag to rotate
# - Scroll to zoom
# - Right-click drag to pan

# Export 3D view:
# View → Export → 3D Image
```

### Create Surface Rendering

```bash
# View → 3D Rendering → Surface Mode

# Set isosurface threshold
# For PET: Choose SUV threshold (e.g., 2.5)

# Adjust surface properties:
# - Smoothing
# - Transparency
# - Color

# Useful for tumor visualization
```

### Multi-Modal 3D Fusion

```bash
# Load and register PET + MRI
amide pet.nii.gz t1w_mri.nii.gz

# After registration:
# View → 3D Rendering
# Enable both data sets
# Set PET to volume rendering (color)
# Set MRI to surface rendering (transparent gray)

# Creates anatomical context for PET hotspots
```

---

## Batch Processing with XML Scripts

### Create XML Script for Batch SUV Calculation

```xml
<!-- amide_batch_suv.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<amide_script version="1.0">
  <load_data_set file="pet_fdg.nii.gz" type="nifti"/>

  <set_suv_parameters>
    <injected_dose units="MBq">370</injected_dose>
    <patient_weight units="kg">70</patient_weight>
    <injection_time>2023-10-15T10:00:00</injection_time>
    <scan_time>2023-10-15T11:00:00</scan_time>
  </set_suv_parameters>

  <convert_to_suv/>

  <export_data_set file="pet_fdg_suv.nii.gz" type="nifti"/>

  <quit/>
</amide_script>
```

```bash
# Run batch script
amide --script amide_batch_suv.xml
```

### Batch ROI Analysis Script

```xml
<!-- roi_analysis.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<amide_script version="1.0">
  <load_data_set file="pet_amyloid_suv.nii.gz" type="nifti"/>

  <!-- Load predefined ROIs -->
  <load_roi file="precuneus_roi.xml"/>
  <load_roi file="frontal_roi.xml"/>
  <load_roi file="cerebellum_roi.xml"/>

  <!-- Calculate statistics -->
  <calculate_roi_statistics/>

  <!-- Export results -->
  <export_roi_statistics file="roi_results.csv" format="csv"/>

  <quit/>
</amide_script>
```

```bash
# Run batch ROI analysis
amide --script roi_analysis.xml
```

### Automated Multi-Subject Processing

```bash
#!/bin/bash
# batch_process_pet.sh - Process multiple PET scans

SUBJECTS=("sub-01" "sub-02" "sub-03" "sub-04")
INJECTED_DOSE=370  # MBq
WEIGHT=70          # kg

for SUBJ in "${SUBJECTS[@]}"; do
  echo "Processing $SUBJ..."

  # Create subject-specific XML script
  cat > ${SUBJ}_script.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<amide_script version="1.0">
  <load_data_set file="data/${SUBJ}_pet.nii.gz" type="nifti"/>
  <set_suv_parameters>
    <injected_dose units="MBq">${INJECTED_DOSE}</injected_dose>
    <patient_weight units="kg">${WEIGHT}</patient_weight>
  </set_suv_parameters>
  <convert_to_suv/>
  <load_roi file="templates/standard_rois.xml"/>
  <calculate_roi_statistics/>
  <export_roi_statistics file="results/${SUBJ}_roi_stats.csv" format="csv"/>
  <export_data_set file="results/${SUBJ}_suv.nii.gz" type="nifti"/>
  <quit/>
</amide_script>
EOF

  # Run AMIDE with script
  amide --script ${SUBJ}_script.xml

  # Clean up
  rm ${SUBJ}_script.xml
done

echo "Batch processing complete!"
```

---

## Integration with Neuroimaging Pipelines

### Use FreeSurfer ROIs for PET Quantification

```bash
#!/bin/bash
# Use FreeSurfer parcellation for PET ROI analysis

SUBJECT=sub-01
FREESURFER_DIR=$SUBJECTS_DIR/$SUBJECT

# Convert FreeSurfer parcellation to PET space
mri_vol2vol \
  --mov pet_fdg_suv.nii.gz \
  --targ $FREESURFER_DIR/mri/aparc+aseg.mgz \
  --regheader \
  --o aparc_in_pet.nii.gz \
  --nearest

# Extract specific regions (e.g., precuneus = label 1025/2025)
fslmaths aparc_in_pet.nii.gz -thr 1025 -uthr 1025 -bin precuneus_left.nii.gz
fslmaths aparc_in_pet.nii.gz -thr 2025 -uthr 2025 -bin precuneus_right.nii.gz
fslmaths precuneus_left.nii.gz -add precuneus_right.nii.gz precuneus.nii.gz

# Calculate mean SUV in precuneus
fslstats pet_fdg_suv.nii.gz -k precuneus.nii.gz -M

# Load in AMIDE for visualization
amide pet_fdg_suv.nii.gz precuneus.nii.gz
```

### Integration with FSL

```bash
# Preprocess PET with FSL, visualize in AMIDE

# Smooth PET
fslmaths pet_fdg.nii.gz -s 4 pet_fdg_smooth.nii.gz

# Threshold to remove noise
fslmaths pet_fdg_smooth.nii.gz -thr 0.1 pet_fdg_clean.nii.gz

# Load in AMIDE
amide pet_fdg_clean.nii.gz
```

### Use ANTs for PET-MR Registration

```bash
# High-quality PET-MR registration with ANTs
antsRegistrationSyN.sh \
  -d 3 \
  -f t1w_mri.nii.gz \
  -m pet_amyloid.nii.gz \
  -o pet_to_mri_ \
  -t r

# Load registered PET and MRI in AMIDE
amide t1w_mri.nii.gz pet_to_mri_Warped.nii.gz

# Or apply FreeSurfer ROIs to registered PET
antsApplyTransforms \
  -d 3 \
  -i aparc+aseg.nii.gz \
  -r pet_amyloid.nii.gz \
  -t [pet_to_mri_0GenericAffine.mat,1] \
  -o aparc_in_pet.nii.gz \
  -n NearestNeighbor
```

---

## Clinical PET Applications

### Amyloid PET Analysis (Alzheimer's Disease)

```bash
#!/bin/bash
# Amyloid PET SUVr calculation pipeline

SUBJECT=sub-AD001

# 1. Calculate SUV
# (Assume SUV image already created)

# 2. Register to MRI and FreeSurfer space
antsRegistrationSyN.sh \
  -d 3 \
  -f ${SUBJECT}_T1w.nii.gz \
  -m ${SUBJECT}_amyloid.nii.gz \
  -o ${SUBJECT}_pet_to_mri_

# 3. Apply FreeSurfer parcellation to PET
antsApplyTransforms \
  -d 3 \
  -i aparc+aseg.nii.gz \
  -r ${SUBJECT}_pet_to_mri_Warped.nii.gz \
  -t [${SUBJECT}_pet_to_mri_0GenericAffine.mat,1] \
  -o aparc_in_pet.nii.gz \
  -n NearestNeighbor

# 4. Extract cerebellum gray matter (reference region)
# Labels: 8 (left cerebellar cortex), 47 (right cerebellar cortex)
fslmaths aparc_in_pet.nii.gz -thr 8 -uthr 8 -bin cereb_left.nii.gz
fslmaths aparc_in_pet.nii.gz -thr 47 -uthr 47 -bin cereb_right.nii.gz
fslmaths cereb_left.nii.gz -add cereb_right.nii.gz cerebellum_gm.nii.gz

# 5. Calculate reference region mean SUV
REF_SUV=$(fslstats ${SUBJECT}_pet_to_mri_Warped.nii.gz -k cerebellum_gm.nii.gz -M)
echo "Reference SUV: $REF_SUV"

# 6. Create SUVr image
fslmaths ${SUBJECT}_pet_to_mri_Warped.nii.gz -div $REF_SUV ${SUBJECT}_amyloid_suvr.nii.gz

# 7. Calculate global cortical SUVr (composite)
# Extract cortical regions
fslmaths aparc_in_pet.nii.gz -thr 1000 -uthr 1035 -bin cortex_left.nii.gz
fslmaths aparc_in_pet.nii.gz -thr 2000 -uthr 2035 -bin cortex_right.nii.gz
fslmaths cortex_left.nii.gz -add cortex_right.nii.gz cortex_composite.nii.gz

GLOBAL_SUVR=$(fslstats ${SUBJECT}_amyloid_suvr.nii.gz -k cortex_composite.nii.gz -M)
echo "Global cortical SUVr: $GLOBAL_SUVR"

# 8. Load in AMIDE for visualization
amide ${SUBJECT}_T1w.nii.gz ${SUBJECT}_amyloid_suvr.nii.gz cortex_composite.nii.gz
```

### FDG PET Metabolism Study

```python
# FDG PET analysis with SUVR normalization
import nibabel as nib
import numpy as np

# Load SUV image
suv_img = nib.load('sub-01_fdg_suv.nii.gz')
suv_data = suv_img.get_fdata()

# Load pons reference mask (typical for FDG)
pons_mask = nib.load('pons_mask.nii.gz').get_fdata()

# Calculate pons mean SUV
pons_suv = np.mean(suv_data[pons_mask > 0])
print(f"Pons reference SUV: {pons_suv:.2f}")

# Create SUVr image
suvr_data = suv_data / pons_suv
suvr_img = nib.Nifti1Image(suvr_data, suv_img.affine, suv_img.header)
nib.save(suvr_img, 'sub-01_fdg_suvr.nii.gz')

# Load ROI masks for regions of interest
regions = {
    'frontal': 'frontal_mask.nii.gz',
    'parietal': 'parietal_mask.nii.gz',
    'temporal': 'temporal_mask.nii.gz',
    'occipital': 'occipital_mask.nii.gz'
}

# Calculate regional SUVr
results = {}
for region, mask_file in regions.items():
    mask = nib.load(mask_file).get_fdata()
    mean_suvr = np.mean(suvr_data[mask > 0])
    results[region] = mean_suvr
    print(f"{region}: SUVr = {mean_suvr:.2f}")

# Visualize in AMIDE
import subprocess
subprocess.run(['amide', 't1w.nii.gz', 'sub-01_fdg_suvr.nii.gz'])
```

### Longitudinal PET Analysis

```bash
#!/bin/bash
# Compare PET scans across timepoints

SUBJECT=sub-01
TP1="baseline"
TP2="12month"

# Calculate SUVr for both timepoints
for TP in $TP1 $TP2; do
  echo "Processing $TP..."

  # Convert to SUV (assume parameters known)
  # Then calculate SUVr

  # Extract ROI statistics
  fslstats ${SUBJECT}_${TP}_suvr.nii.gz -k precuneus.nii.gz -M > ${SUBJECT}_${TP}_precuneus.txt
done

# Compare values
BL_SUVR=$(cat ${SUBJECT}_${TP1}_precuneus.txt)
FU_SUVR=$(cat ${SUBJECT}_${TP2}_precuneus.txt)

echo "Baseline SUVr: $BL_SUVR"
echo "Follow-up SUVr: $FU_SUVR"

# Calculate percent change
python3 << EOF
bl = float($BL_SUVR)
fu = float($FU_SUVR)
pct_change = ((fu - bl) / bl) * 100
print(f"Percent change: {pct_change:.1f}%")
EOF

# Load both in AMIDE for visual comparison
amide ${SUBJECT}_${TP1}_suvr.nii.gz ${SUBJECT}_${TP2}_suvr.nii.gz
```

---

## Advanced Features

### DICOM Handling

```bash
# Import DICOM series
# File → Import → DICOM
# Navigate to DICOM directory
# AMIDE scans and lists all series
# Select PET series to import

# Export to DICOM
# File → Export → DICOM
# Configure DICOM tags
# Specify output directory
```

### Multi-Planar Reformatting (MPR)

```bash
# In AMIDE GUI:
# View → Multi-Planar Reformatting

# Displays:
# - Axial view
# - Coronal view
# - Sagittal view
# - Optional: Oblique views

# Synchronize views:
# Click in one view, crosshairs update in all views

# Useful for:
# - Precise ROI placement
# - Anatomical localization
# - Quality control
```

### Color Maps and Windowing

```bash
# View → Color Map
# Select from:
# - Hot (common for PET)
# - Rainbow
# - Grayscale
# - Spectrum
# - Custom

# Adjust window/level:
# View → Threshold
# Set min/max display values
# Optimize for specific SUV range
```

---

## Troubleshooting

### AMIDE Won't Launch

```bash
# Check installation
which amide

# Check dependencies
ldd `which amide` | grep "not found"

# Install missing libraries
sudo apt-get install libgtk-3-0 libgsl23 libvolpack1

# Check permissions
chmod +x /usr/local/bin/amide

# Run with verbose output
amide --verbose
```

### Cannot Load DICOM Files

```bash
# Ensure DCMTK is installed
sudo apt-get install dcmtk

# Test DICOM file
dcmdump pet_dicom_file.dcm

# Convert DICOM to NIFTI if needed
dcm2niix -o output_dir dicom_directory

# Load NIFTI in AMIDE
amide output_dir/pet.nii.gz
```

### Registration Fails

```bash
# Use external registration (ANTs, FSL)
# Then load pre-registered images

# Check image orientations
fslhd pet.nii.gz
fslhd mri.nii.gz

# Reorient if needed
fslreorient2std pet.nii.gz pet_reorient.nii.gz
fslreorient2std mri.nii.gz mri_reorient.nii.gz

# Try manual registration in AMIDE first
```

### Memory Issues with Large 4D Data

```bash
# Reduce data size before loading
# Extract specific time frames

fslroi dynamic_pet.nii.gz early_frames.nii.gz 0 10

# Load subset in AMIDE
amide early_frames.nii.gz

# Or increase available memory
# Close other applications
# Use 64-bit version of AMIDE
```

---

## Best Practices

### PET Quantification Workflow

1. **Quality Control:**
   - Check for motion artifacts
   - Verify injection parameters
   - Confirm scan timing

2. **SUV Calculation:**
   - Use accurate patient weight and injected dose
   - Apply decay correction
   - Document all parameters

3. **Reference Region Selection:**
   - Choose appropriate reference (cerebellum, pons, white matter)
   - Ensure anatomical accuracy with registration
   - Use standardized ROI definitions

4. **Statistical Analysis:**
   - Extract ROI statistics systematically
   - Use consistent thresholds
   - Document processing steps

### Multi-Modal PET-MR Analysis

1. **Registration:**
   - Use high-quality registration (ANTs, FSL FLIRT)
   - Verify alignment visually
   - Check multiple anatomical landmarks

2. **ROI Definition:**
   - Use MRI-based anatomical parcellations (FreeSurfer)
   - Transform ROIs to PET space accurately
   - Verify ROI placement on fused images

3. **Quantification:**
   - Extract PET metrics in anatomical ROIs
   - Consider partial volume effects
   - Apply corrections if needed

### Data Organization

```bash
# Organize PET data systematically
project/
├── sub-01/
│   ├── pet/
│   │   ├── raw_pet.nii.gz
│   │   ├── pet_suv.nii.gz
│   │   └── pet_suvr.nii.gz
│   ├── anat/
│   │   └── t1w.nii.gz
│   ├── rois/
│   │   ├── precuneus.nii.gz
│   │   └── cerebellum.nii.gz
│   └── results/
│       └── roi_statistics.csv
└── sub-02/
    └── ...
```

---

## Resources and Further Reading

### Official Resources

- **AMIDE Homepage:** http://amide.sourceforge.net/
- **User Manual:** http://amide.sourceforge.net/documentation.html
- **SourceForge Project:** https://sourceforge.net/projects/amide/

### PET Imaging Tutorials

- **SUV and SUVr Calculation:** Understanding standardized uptake values
- **Amyloid PET Quantification:** Centiloid scale and interpretation
- **FDG PET Analysis:** Metabolism patterns in neurodegenerative diseases
- **Dynamic PET:** Kinetic modeling and compartmental analysis

### Related Tools

- **NiftyPET:** GPU-accelerated PET reconstruction
- **SIRF:** Synergistic PET-MR reconstruction framework
- **STIR:** Tomographic image reconstruction library
- **FreeSurfer:** Anatomical segmentation for PET ROIs
- **FSL:** Image processing and registration
- **SPM:** Statistical parametric mapping with PET support

### Citations

If you use AMIDE in your research, please cite:

```
Loening, A. M., & Gambhir, S. S. (2003).
AMIDE: A free software tool for multimodality medical image analysis.
Molecular Imaging, 2(3), 131-137.
```

### Community and Support

- **Mailing List:** Check SourceForge for user discussions
- **Bug Reports:** https://sourceforge.net/p/amide/bugs/
- **Feature Requests:** https://sourceforge.net/p/amide/feature-requests/

---

## Summary

AMIDE is a versatile, cross-platform tool for PET, SPECT, and multi-modal medical image analysis. Its strengths include:

**Pros:**
- Free and open-source
- Cross-platform GUI (Linux, macOS, Windows)
- Comprehensive format support (DICOM, NIFTI, ANALYZE)
- Integrated SUV calculation and ROI analysis
- Time-activity curve and kinetic modeling support
- XML-based batch processing
- Good for teaching and clinical workflows

**Cons:**
- Limited advanced reconstruction capabilities (use NiftyPET, SIRF, STIR)
- Registration less robust than ANTs or FSL
- 3D rendering less advanced than specialized tools
- Smaller user community than major packages

**Best Used For:**
- Clinical PET/SPECT quantification
- Amyloid and tau PET analysis
- FDG metabolism studies
- Multi-modal PET-MR/PET-CT visualization
- Teaching nuclear medicine concepts
- Quick ROI-based analysis
- Interactive visualization and exploration

For PET reconstruction from raw data, consider **NiftyPET** or **SIRF**. For advanced registration, use **ANTs**. For statistical analysis, use **SPM** or **FSL**. AMIDE excels as a user-friendly viewer and quantification tool for clinical PET research.
