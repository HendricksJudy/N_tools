# BASIL: Bayesian Arterial Spin Labeling Analysis

## Overview

**BASIL** (Bayesian Arterial Spin Labeling) is FSL's tool for quantitative analysis of ASL MRI data using Bayesian inference. BASIL provides advanced modeling of ASL data including estimation of perfusion (CBF), arterial transit time (ATT), arterial blood volume (aBV), and partial volume correction. Using variational Bayesian inference, BASIL provides probabilistic estimates with uncertainty quantification, making it particularly suitable for multi-PLD ASL and research requiring rigorous statistical modeling.

### Key Features

- **Bayesian Inference**: Probabilistic estimation with uncertainty quantification
- **Multi-PLD Support**: Single and multi-delay ASL analysis
- **Perfusion Quantification**: Absolute CBF in mL/100g/min
- **Hemodynamic Parameters**: ATT, aBV, dispersion modeling
- **Partial Volume Correction**: Separate GM and WM CBF
- **Spatial Regularization**: Smooth parameter estimates
- **Model Selection**: Compare different kinetic models
- **FSL Integration**: Seamless workflow with BET, FAST, FLIRT, FNIRT
- **BASIL GUI**: Interactive analysis interface
- **Command-Line**: Scriptable for batch processing
- **Uncertainty Maps**: Posterior variance for all parameters

### Scientific Foundation

BASIL uses a general kinetic model for ASL:

- **ΔM(t)**: Perfusion-weighted signal at time t
- **CBF**: Cerebral blood flow (tissue perfusion)
- **ATT**: Arterial transit time (bolus arrival time)
- **τ**: Bolus duration (labeling duration)
- **T1b**: T1 of arterial blood

Bayesian inference provides:
1. **Prior distributions**: Incorporate physiological constraints
2. **Likelihood**: Data fit to kinetic model
3. **Posterior distributions**: Parameter estimates with uncertainty
4. **Variational Bayes**: Efficient approximation for voxel-wise fitting

### Primary Use Cases

1. **Multi-PLD ASL**: Estimate ATT and CBF simultaneously
2. **Clinical Research**: Stroke, dementia, tumors with uncertainty
3. **Vascular Studies**: ATT mapping in stenosis, collaterals
4. **Method Development**: Compare kinetic models
5. **High-Quality Estimates**: Spatial regularization for noisy data
6. **Research Studies**: When uncertainty quantification is important

---

## Installation and Setup

### FSL Installation

```bash
# BASIL is part of FSL (version 6.0 or later)
# Download FSL from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

# After installing FSL, set up environment
export FSLDIR=/usr/local/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh

# Verify FSL installation
fsl

# Check BASIL availability
which oxford_asl

# Expected output: /usr/local/fsl/bin/oxford_asl
```

### Python Environment

```bash
# BASIL uses Python for some components
# FSL includes necessary Python packages

# Verify Python environment
fslpython --version

# Check FSL Python packages
fslpython -c "import fsl; print(fsl.__version__)"
```

### Verify BASIL Installation

```bash
# Test BASIL command
oxford_asl --help

# This should display help text for oxford_asl

# Check BASIL GUI availability
Asl_gui &

# GUI should launch (requires X11/display)
```

---

## BASIL Basics

### Bayesian Inference for ASL

```bash
# BASIL models ASL data as:
# ΔM = M_control - M_label
#
# Kinetic model:
# ΔM(t) = 2 · M0b · CBF · ATT_function(t, τ, ATT, T1b)
#
# Bayesian approach:
# 1. Define prior distributions P(θ) for parameters θ = {CBF, ATT, ...}
# 2. Calculate likelihood P(data|θ) from kinetic model
# 3. Compute posterior P(θ|data) ∝ P(data|θ) · P(θ)
# 4. Use variational Bayes for efficient estimation

echo "Bayesian ASL Analysis"
echo "  - Incorporates physiological priors"
echo "  - Provides uncertainty estimates"
echo "  - Handles multiple PLDs"
echo "  - Spatial regularization option"
```

### Kinetic Models

```bash
# BASIL supports multiple kinetic models:

# 1. Standard model (single-compartment)
#    - Assumes instantaneous exchange
#    - Default for most applications

# 2. Two-compartment model
#    - Models tissue and blood compartments
#    - More complex, requires more data

# 3. Dispersion model
#    - Accounts for bolus dispersion
#    - Important for long transit times

echo "Kinetic Model Selection:"
echo "  Standard: Most common, single PLD OK"
echo "  Two-compartment: Research, multi-PLD"
echo "  Dispersion: Long ATT, vascular pathology"
```

---

## BASIL GUI

### Launch GUI

```bash
# Start BASIL GUI
Asl_gui &

# GUI has 4 main tabs:
# 1. Data - Input files and parameters
# 2. Structure - T1 and registration
# 3. Calibration - M0 for absolute quantification
# 4. Analysis - Model options and run
```

### Load Data (GUI)

```bash
# In GUI:
# 1. Data tab -> Input Image: Browse to ASL file
# 2. Set data format:
#    - Multi-phase/multi-TI: Single file with all dynamics
#    - Data order: Label-control pairs or control-label
# 3. Number of repeats/phases
# 4. Set acquisition parameters:
#    - Labeling: pCASL, CASL, or PASL
#    - Bolus duration (s)
#    - PLDs (s) - comma separated for multi-PLD
#    - Readout: 2D multi-slice or 3D

echo "Example parameters for pCASL:"
echo "  Labeling: pCASL"
echo "  Bolus duration: 1.8 s"
echo "  PLD: 1.8 s (or 0.25,0.5,1.0,1.5,2.0 for multi-PLD)"
echo "  Readout: 2D multi-slice"
```

### Configure Analysis (GUI)

```bash
# In GUI Analysis tab:

# Basic options:
# - Spatial regularization: ON (recommended)
# - Infer arterial transit time: ON for multi-PLD
# - Macro vascular component: ON for multi-PLD
# - Partial volume correction: ON if T1 available

# Advanced options:
# - Model: Standard (default)
# - T1/T1b values (default 1.3/1.65 s @ 3T)
# - Labeling efficiency (0.85 for pCASL)

echo "Run analysis:"
echo "  Click 'Go' button"
echo "  Output saved to specified directory"
```

---

## Command-Line Processing

### Basic Single-PLD Analysis

```bash
# Basic pCASL analysis with oxford_asl
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_output \
  --iaf=tc \
  --ibf=tis \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --spatial \
  --slicedt=0.045

# Parameters:
#   -i: Input ASL data
#   -o: Output directory
#   --iaf: Input ASL format (tc=tag-control pairs)
#   --ibf: Input block format (tis=PLDs/TIs)
#   --tis: Post-labeling delay (s)
#   --bolus: Labeling duration (s)
#   --casl: pCASL labeling
#   --spatial: Spatial regularization
#   --slicedt: Time between slices (s) for 2D readout
```

### Multi-PLD Analysis

```bash
# Multi-PLD pCASL with ATT estimation
oxford_asl \
  -i asl_multi_pld.nii.gz \
  -o basil_multi_pld \
  --iaf=tc \
  --ibf=tis \
  --tis=0.25,0.5,1.0,1.5,2.0,2.5 \
  --bolus=1.8 \
  --casl \
  --spatial \
  --inferart \
  --artoff \
  --slicedt=0.045

# Additional parameters for multi-PLD:
#   --tis: Multiple PLDs (comma-separated)
#   --inferart: Estimate arterial transit time
#   --artoff: Infer arterial (macrovascular) component
```

### With Structural Image

```bash
# Include structural T1w for registration and PVC
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_with_struct \
  -c struct/T1w.nii.gz \
  --iaf=tc \
  --ibf=tis \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --spatial \
  --pvcorr \
  --slicedt=0.045

# Additional parameters:
#   -c: Structural image (T1w)
#   --pvcorr: Partial volume correction
```

### With M0 Calibration

```bash
# Absolute quantification with M0 image
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_calibrated \
  -c struct/T1w.nii.gz \
  -m m0_scan.nii.gz \
  --iaf=tc \
  --ibf=tis \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --tr=4.0 \
  --cgain=1.0 \
  --spatial \
  --pvcorr

# Calibration parameters:
#   -m: M0 calibration image
#   --tr: TR of ASL sequence (s)
#   --cgain: Calibration gain (relative M0 intensity)
```

---

## Output Files

### Main Outputs

```bash
# BASIL creates numerous output files
output_dir="basil_output"

# Primary outputs:
echo "Main Output Files:"
echo "  native_space/perfusion.nii.gz - CBF map (relative)"
echo "  native_space/perfusion_calib.nii.gz - CBF (absolute, if M0 provided)"
echo "  native_space/arrival.nii.gz - Arterial transit time (if estimated)"
echo "  native_space/aCBV.nii.gz - Arterial cerebral blood volume"

# Quality and auxiliary:
echo "  native_space/perfusion_var.nii.gz - CBF uncertainty"
echo "  native_space/mask.nii.gz - Analysis mask"
echo "  native_space/mean_ftiss.nii.gz - Mean perfusion-weighted signal"

# Partial volume corrected (if --pvcorr):
echo "  native_space/perfusion_calib_gm_masked.nii.gz - GM CBF"
echo "  native_space/perfusion_calib_wm_masked.nii.gz - WM CBF"
```

### View Results

```bash
# View CBF map with fsleyes
fsleyes basil_output/native_space/perfusion_calib.nii.gz \
  -cm hot -dr 0 80

# View with overlays
fsleyes struct/T1w.nii.gz \
  basil_output/native_space/perfusion_calib.nii.gz \
  -cm hot -dr 0 80 -a 70

# View ATT map (if available)
if [ -f basil_output/native_space/arrival.nii.gz ]; then
    fsleyes basil_output/native_space/arrival.nii.gz \
      -cm cool -dr 0 2
fi
```

### Extract Summary Statistics

```bash
# Get mean CBF values
cbf_file="basil_output/native_space/perfusion_calib.nii.gz"

if [ -f "$cbf_file" ]; then
    # Mean CBF across brain
    mean_cbf=$(fslstats $cbf_file -M)
    echo "Mean CBF: $mean_cbf mL/100g/min"

    # Gray matter CBF (if available)
    gm_cbf_file="basil_output/native_space/perfusion_calib_gm_masked.nii.gz"
    if [ -f "$gm_cbf_file" ]; then
        gm_cbf=$(fslstats $gm_cbf_file -M)
        echo "Gray matter CBF: $gm_cbf mL/100g/min"
    fi

    # White matter CBF (if available)
    wm_cbf_file="basil_output/native_space/perfusion_calib_wm_masked.nii.gz"
    if [ -f "$wm_cbf_file" ]; then
        wm_cbf=$(fslstats $wm_cbf_file -M)
        echo "White matter CBF: $wm_cbf mL/100g/min"
    fi
fi
```

---

## Acquisition Parameters

### Labeling Schemes

```bash
# pCASL (Pseudo-continuous ASL) - RECOMMENDED
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_pcasl \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --slicedt=0.045

# PASL (Pulsed ASL)
oxford_asl \
  -i asl_pasl.nii.gz \
  -o basil_pasl \
  --iaf=tc \
  --tis=0.8 \
  --bolus=0.7 \
  --slicedt=0.045

# Note: No --casl flag for PASL

# CASL (Continuous ASL) - rare
oxford_asl \
  -i asl_casl.nii.gz \
  -o basil_casl \
  --iaf=tc \
  --tis=1.5 \
  --bolus=2.0 \
  --casl \
  --slicedt=0.045
```

### Readout Parameters

```bash
# 2D multi-slice (most common)
# Specify slice timing
oxford_asl \
  -i asl_2d.nii.gz \
  -o basil_2d \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --slicedt=0.045 \
  --sliceband=10

# Parameters:
#   --slicedt: Time per slice (TR/Nslices)
#   --sliceband: Number of slices per band (for SMS)

# 3D readout
oxford_asl \
  -i asl_3d.nii.gz \
  -o basil_3d \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl

# No --slicedt for 3D acquisition
```

---

## Perfusion Quantification

### Absolute CBF Calculation

```bash
# With M0 calibration
# CBF = λ · ΔM / (2 · α · M0 · T1b · (exp(-PLD/T1b) - exp(-(τ+PLD)/T1b)))

# Set physiological parameters
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_quant \
  -m m0.nii.gz \
  -c struct.nii.gz \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --bat=1.3 \
  --t1=1.3 \
  --t1b=1.65 \
  --alpha=0.85

# Physiological parameters:
#   --bat: Bolus arrival time (default ATT, s)
#   --t1: Tissue T1 (s) - 1.3 @ 3T, 1.0 @ 1.5T
#   --t1b: Blood T1 (s) - 1.65 @ 3T, 1.35 @ 1.5T
#   --alpha: Labeling efficiency (0.85 pCASL, 0.98 PASL)
```

### Spatial Regularization

```bash
# Enable spatial smoothing of parameter estimates
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_spatial \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --spatial

# Spatial regularization:
# - Reduces noise in parameter estimates
# - Assumes spatially smooth perfusion fields
# - Recommended for single-PLD
# - Essential for noisy data

# Without spatial regularization (noisier but unbiased)
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_no_spatial \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl

# Compare results
echo "With spatial regularization: smoother but potentially biased"
echo "Without spatial regularization: noisier but unbiased"
```

---

## Multi-PLD Analysis

### Arterial Transit Time Estimation

```bash
# Multi-PLD for ATT mapping
oxford_asl \
  -i asl_multi_pld.nii.gz \
  -o basil_att \
  --iaf=tc \
  --tis=0.25,0.5,1.0,1.5,2.0 \
  --bolus=1.8 \
  --casl \
  --inferart \
  --spatial

# Output includes:
#   arrival.nii.gz - Arterial transit time map
#   arrival_var.nii.gz - ATT uncertainty

# View ATT map
fsleyes basil_att/native_space/arrival.nii.gz \
  -cm cool -dr 0 2

# Extract mean ATT
mean_att=$(fslstats basil_att/native_space/arrival.nii.gz -M)
echo "Mean ATT: $mean_att seconds"
```

### Macrovascular Component

```bash
# Estimate arterial blood volume (macrovascular signal)
oxford_asl \
  -i asl_multi_pld.nii.gz \
  -o basil_macro \
  --iaf=tc \
  --tis=0.25,0.5,1.0,1.5,2.0 \
  --bolus=1.8 \
  --casl \
  --inferart \
  --artoff \
  --spatial

# Additional output:
#   aCBV.nii.gz - Arterial cerebral blood volume

# Macrovascular component helps:
# - Remove arterial signal from tissue perfusion
# - More accurate CBF in regions near large vessels
# - Important for vascular pathology
```

### Dispersion Modeling

```bash
# Model bolus dispersion (advanced)
oxford_asl \
  -i asl_multi_pld.nii.gz \
  -o basil_dispersion \
  --iaf=tc \
  --tis=0.25,0.5,1.0,1.5,2.0 \
  --bolus=1.8 \
  --casl \
  --inferart \
  --infert1 \
  --artoff \
  --spatial

# Additional parameters:
#   --infert1: Infer tissue T1 (requires multi-PLD)
#   Dispersion modeled implicitly in multi-PLD fit
```

---

## Partial Volume Correction

### Enable PVC

```bash
# Partial volume correction requires structural image
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_pvc \
  -c struct/T1w.nii.gz \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --pvcorr \
  --spatial

# PVC process:
# 1. Segment T1w into GM, WM, CSF (using FAST)
# 2. Register ASL to T1w
# 3. Partial volume correct perfusion estimates
# 4. Output separate GM and WM CBF maps
```

### Extract PVC Results

```bash
# Gray matter CBF
gm_cbf="basil_pvc/native_space/perfusion_calib_gm_masked.nii.gz"
wm_cbf="basil_pvc/native_space/perfusion_calib_wm_masked.nii.gz"

if [ -f "$gm_cbf" ]; then
    # Statistics
    gm_mean=$(fslstats $gm_cbf -M)
    wm_mean=$(fslstats $wm_cbf -M)

    echo "Partial Volume Corrected CBF:"
    echo "  Gray matter: $gm_mean mL/100g/min"
    echo "  White matter: $wm_mean mL/100g/min"

    # Calculate GM/WM ratio
    ratio=$(echo "scale=2; $gm_mean / $wm_mean" | bc)
    echo "  GM/WM ratio: $ratio"
    echo "  (Expected: ~2.5-3.0 for healthy adults)"
fi
```

### Visualize PVC Results

```bash
# Overlay GM and WM CBF on structural
fsleyes struct/T1w.nii.gz \
  basil_pvc/native_space/perfusion_calib_gm_masked.nii.gz \
  -cm hot -dr 0 80 -a 70 \
  basil_pvc/native_space/perfusion_calib_wm_masked.nii.gz \
  -cm cool -dr 0 30 -a 70
```

---

## Advanced Options

### Custom Prior Distributions

```bash
# BASIL uses Bayesian priors for parameters
# Defaults are physiologically reasonable
# Can be customized for specific populations

# Example: Modify tissue T1 for pediatric data
oxford_asl \
  -i asl_pediatric.nii.gz \
  -o basil_pediatric \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --t1=1.5 \
  --t1b=1.65 \
  --spatial

# Tissue T1 varies with age and pathology:
#   Neonates: ~2.0 s
#   Children: ~1.5 s
#   Adults: ~1.3 s
#   Elderly: ~1.2 s
#   Edema/tumor: higher
```

### Model Comparison

```bash
# Compare different kinetic models
# Standard model
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_standard \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --spatial

# With ATT estimation (if multi-PLD)
oxford_asl \
  -i asl_multi_pld.nii.gz \
  -o basil_with_att \
  --iaf=tc \
  --tis=0.5,1.0,1.5,2.0 \
  --bolus=1.8 \
  --casl \
  --inferart \
  --spatial

# Compare model fits via free energy (lower is better)
# Check logs in output directories
```

---

## Integration with FSL

### Preprocessing with FSL

```bash
# Typical preprocessing before BASIL

# 1. Brain extraction
bet asl_data.nii.gz asl_data_brain.nii.gz -f 0.5 -m

# 2. Motion correction
mcflirt -in asl_data_brain.nii.gz -out asl_data_mcf -plots

# 3. Use motion-corrected data for BASIL
oxford_asl \
  -i asl_data_mcf.nii.gz \
  -o basil_output \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --spatial
```

### Registration

```bash
# BASIL uses FLIRT for registration
# Can customize registration

# Register ASL to high-res T1w
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_output \
  -c struct/T1w_brain.nii.gz \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --regfrom=asl_calibration.nii.gz

# Parameters:
#   --regfrom: Use specific volume for registration
#              (e.g., M0 or calibration image)
```

### Segmentation for PVC

```bash
# BASIL uses FSL FAST for segmentation
# Segment structural image manually if needed

# Segment T1w
fast -t 1 -n 3 -o struct/T1w_seg struct/T1w_brain.nii.gz

# This creates:
#   T1w_seg_pve_0.nii.gz - CSF
#   T1w_seg_pve_1.nii.gz - GM
#   T1w_seg_pve_2.nii.gz - WM

# BASIL will use these automatically if present
```

---

## Quality Control

### Visual Inspection

```bash
# Check key quality indicators

# 1. View raw ASL data
fsleyes asl_data.nii.gz

# Check for:
# - Adequate SNR
# - No severe motion
# - Label-control alternation correct

# 2. View perfusion map
fsleyes basil_output/native_space/perfusion_calib.nii.gz \
  -cm hot -dr 0 80

# Check for:
# - Realistic CBF values (GM: 40-60, WM: 15-25 mL/100g/min)
# - No extreme artifacts
# - Good gray-white contrast

# 3. View ATT map (if available)
if [ -f basil_output/native_space/arrival.nii.gz ]; then
    fsleyes basil_output/native_space/arrival.nii.gz \
      -cm cool -dr 0 2

    # Check for:
    # - Realistic ATT (0.5-1.5 s typical)
    # - Spatial consistency
    # - Delayed regions (pathology)
fi
```

### Quantitative QC

```bash
# Extract QC metrics
function qc_basil() {
    output_dir=$1

    # Mean CBF
    cbf_mean=$(fslstats $output_dir/native_space/perfusion_calib.nii.gz -M)

    # CBF variability (coefficient of variation)
    cbf_std=$(fslstats $output_dir/native_space/perfusion_calib.nii.gz -S)
    cbf_cov=$(echo "scale=3; $cbf_std / $cbf_mean" | bc)

    # Mean uncertainty
    if [ -f $output_dir/native_space/perfusion_var.nii.gz ]; then
        var_mean=$(fslstats $output_dir/native_space/perfusion_var.nii.gz -M)
        uncertainty=$(echo "scale=1; sqrt($var_mean)" | bc)
    else
        uncertainty="N/A"
    fi

    echo "QC Metrics:"
    echo "  Mean CBF: $cbf_mean mL/100g/min"
    echo "  CBF CoV: $cbf_cov"
    echo "  Mean uncertainty: $uncertainty"

    # Flag potential issues
    if (( $(echo "$cbf_mean < 20" | bc -l) )); then
        echo "  ⚠ Low CBF detected"
    fi
    if (( $(echo "$cbf_cov > 0.5" | bc -l) )); then
        echo "  ⚠ High variability detected"
    fi
}

# Run QC
qc_basil basil_output
```

---

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# batch_basil.sh

# Directory structure:
# subjects/
#   sub-01/
#     asl.nii.gz
#     T1w.nii.gz
#     m0.nii.gz
#   sub-02/
#   ...

subjects_dir="subjects"
output_base="basil_results"

for subject_dir in $subjects_dir/sub-*; do
    subject=$(basename $subject_dir)
    echo "Processing $subject..."

    # Input files
    asl_file="$subject_dir/asl.nii.gz"
    struct_file="$subject_dir/T1w.nii.gz"
    m0_file="$subject_dir/m0.nii.gz"

    # Output directory
    output_dir="$output_base/$subject"

    # Check files exist
    if [ ! -f "$asl_file" ]; then
        echo "  ✗ ASL file not found"
        continue
    fi

    # Run BASIL
    oxford_asl \
      -i $asl_file \
      -o $output_dir \
      -c $struct_file \
      -m $m0_file \
      --iaf=tc \
      --tis=1.8 \
      --bolus=1.8 \
      --casl \
      --spatial \
      --pvcorr

    if [ $? -eq 0 ]; then
        echo "  ✓ $subject completed"
    else
        echo "  ✗ $subject failed"
    fi
done

echo "Batch processing complete"
```

### GNU Parallel Processing

```bash
# Process subjects in parallel
ls subjects/sub-*/asl.nii.gz | parallel -j 4 \
  'subject=$(basename $(dirname {})); \
   oxford_asl \
     -i {} \
     -o basil_results/$subject \
     -c subjects/$subject/T1w.nii.gz \
     --iaf=tc --tis=1.8 --bolus=1.8 --casl --spatial'
```

---

## Troubleshooting

### Common Issues

```bash
# Issue 1: Segmentation failed
# Check T1w image quality
# Solution: Pre-run FAST separately

fast -t 1 -n 3 -o struct/T1w_seg struct/T1w_brain.nii.gz

# Then run oxford_asl with --fastsrc pointing to segmentation

# Issue 2: Registration failed
# Solution: Use better reference for registration
oxford_asl \
  --regfrom=m0.nii.gz \
  ...

# Issue 3: Very low/high CBF values
# Check:
# - M0 calibration correct
# - Labeling efficiency parameter
# - PLD appropriate for population

# Issue 4: High uncertainty
# - Increase spatial regularization (default)
# - Check data quality (SNR, motion)
# - Use multi-PLD for better ATT estimation
```

### Debug Mode

```bash
# Enable verbose output
oxford_asl \
  -i asl_data.nii.gz \
  -o basil_debug \
  --iaf=tc \
  --tis=1.8 \
  --bolus=1.8 \
  --casl \
  --debug

# This saves intermediate files and logs
# Check basil_debug/logfile for detailed output
```

---

## Best Practices

### Recommended Workflow

```bash
echo "BASIL Best Practices:"
echo "1. Use multi-PLD if available (>3 PLDs)"
echo "2. Always include M0 for absolute quantification"
echo "3. Enable spatial regularization for single-PLD"
echo "4. Use PVC for cortical CBF analysis"
echo "5. Check ATT in clinical populations"
echo "6. Visually inspect all outputs"
echo "7. Report FSL/BASIL version"
echo "8. Document all parameters used"
```

### Parameter Selection

```bash
# pCASL @ 3T (recommended settings)
echo "pCASL @ 3T:"
echo "  PLD: 1.8 s (single), 0.25-2.5 s (multi)"
echo "  Bolus: 1.8 s"
echo "  T1b: 1.65 s"
echo "  Labeling efficiency: 0.85"

# PASL @ 3T
echo "PASL @ 3T:"
echo "  PLD: 0.8-1.2 s"
echo "  Bolus: 0.7 s"
echo "  Labeling efficiency: 0.98"
```

---

## References

### Key Publications

1. Chappell, M. A., et al. (2009). "Variational Bayesian inference for a nonlinear forward model." *IEEE Transactions on Signal Processing*, 57(1), 223-236.

2. Groves, A. R., et al. (2009). "Combined spatial and non-spatial prior for inference on MRI time-series." *NeuroImage*, 45(3), 795-809.

3. Alsop, D. C., et al. (2015). "Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications." *Magnetic Resonance in Medicine*, 73(1), 102-116.

### Documentation and Resources

- **FSL Wiki**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BASIL
- **Tutorial**: https://fsl.fmrib.ox.ac.uk/fslcourse/lectures/practicals/ASLpractical/
- **Examples**: Included with FSL installation
- **Support**: FSL mailing list

### Related Tools

- **ExploreASL**: MATLAB multi-center ASL pipeline
- **ASLPrep**: BIDS-compliant ASL preprocessing
- **FSL**: Broader neuroimaging tools
- **oxford_asl**: Main command-line tool
- **Asl_gui**: Interactive GUI

---

## See Also

- **exploreasl.md**: Multi-center ASL processing
- **aslprep.md**: BIDS ASL preprocessing
- **fsl.md**: FSL neuroimaging tools
- **bet.md**: Brain extraction
- **fast.md**: Tissue segmentation
- **flirt.md**: Image registration
