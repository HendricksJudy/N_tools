# AES-SDM (Seed-based d Mapping)

## Overview

AES-SDM (Anisotropic Effect Size Seed-based d Mapping), formerly known as Signed Differential Mapping (SDM), is a sophisticated statistical technique for meta-analysis of neuroimaging studies. Unlike coordinate-based methods that only use peak coordinates, SDM reconstructs full effect-size maps from published data, allowing for more nuanced analyses including correlations with clinical/demographic variables and assessment of heterogeneity.

**Website:** https://www.sdmproject.com/
**Platform:** Windows (with MATLAB Runtime)
**Language:** MATLAB (standalone version available)
**License:** Free for academic use

## Key Features

- Effect-size based meta-analysis
- Reconstruction of full brain maps from coordinates
- Meta-regression with continuous moderators
- Categorical moderator analysis
- Between-study heterogeneity assessment
- Multiple imputation of unreported effects
- Anisotropic smoothing kernels
- Family-wise error correction
- Leave-one-out sensitivity analysis
- Funnel plot and publication bias tests
- Both voxel-wise and ROI-based analyses

## Installation

### Download SDM Software

```bash
# Download from website
# https://www.sdmproject.com/software/

# Windows standalone (no MATLAB required)
# Extract SDM_6.21.zip to desired location
# C:\Program Files\SDM\

# Launch SDM.exe
```

### MATLAB Version

```bash
# If you have MATLAB
# Download SDM MATLAB scripts
# Add to MATLAB path

# In MATLAB:
addpath('C:\path\to\SDM');
sdm_gui; # Launch GUI
```

### Verify Installation

```bash
# Launch SDM
# Help → About should show version info
# File menu should be accessible
```

## Data Format

### SDM Table Format

```text
# Create studies table (tab-separated)
# Columns: study, subjects, contrast, coordinate_space

study1	20	Depression > Controls	MNI
study2	25	Patients > Healthy	MNI
study3	18	Schizophrenia > Controls	MNI

# Save as: studies.txt
```

### Coordinates File

```text
# For each study: study_name_coordinates.txt
# Format: x  y  z  t-value

# Example: study1_coordinates.txt
-42	15	24	4.52
45	18	21	3.89
-12	-85	2	5.12
0	-6	52	4.23

# Columns:
# x, y, z: MNI or Talairach coordinates
# t-value: Can use t, z, or p-values
# If only peaks reported, SDM reconstructs full maps
```

### Peak Coordinates Format

```text
# Minimal format when only peaks available
# study1_peaks.txt

x	y	z
-42	15	24
45	18	21
-12	-85	2

# SDM will impute effect sizes
```

## Basic Workflow (GUI)

### Step 1: Create New Meta-Analysis

```bash
# In SDM GUI:
# File → New Meta-Analysis
# Select working directory
# Name your project
# Save as .sdm file

# Project structure:
# project.sdm (main file)
# studies.txt (study information)
# study1_coordinates.txt (coordinates per study)
# study2_coordinates.txt
# ...
```

### Step 2: Enter Study Information

```bash
# Studies → Add Study

# For each study enter:
# - Study ID (unique identifier)
# - Sample size
# - Contrast description
# - Coordinate space (MNI/Talairach)
# - Coordinates file path

# Or import from text file:
# Studies → Import from Table
```

### Step 3: Preprocess Data

```bash
# Analysis → Preprocessing

# Options:
# - Anisotropic kernel (recommended)
# - FWHM: 20mm (default)
# - Voxel size: 2mm
# - Mask: gray matter mask

# Click "Preprocess"

# Creates reconstructed effect-size maps
# for each study
```

### Step 4: Mean Analysis

```bash
# Analysis → Mean Analysis

# Computes:
# - Mean effect-size map across studies
# - Standard error
# - Z-scores
# - P-values (corrected and uncorrected)

# Parameters:
# - FWE correction (permutation-based)
# - Cluster threshold: default
# - Number of permutations: 50 (quick) to 500 (publication)

# Results saved automatically
```

### Step 5: View Results

```bash
# Results → View Maps

# Display options:
# - Overlay on template
# - Threshold settings
# - Cluster table
# - Export options

# Results → Export

# Export formats:
# - NIfTI images
# - Text tables
# - CSV files
```

## Mean Meta-Analysis

### Standard Mean Analysis

```bash
# After preprocessing:
# Analysis → Mean Analysis

# Fixed-effects or Random-effects model
# (SDM uses random-effects by default)

# Output:
# - SDM_MeanAnalysis_p_FWE.nii: Corrected p-values
# - SDM_MeanAnalysis_z.nii: Z-scores
# - SDM_MeanAnalysis_clusters.txt: Cluster table

# Threshold typically: p < 0.05 FWE-corrected
```

### Jackknife Sensitivity Analysis

```bash
# Test robustness of results

# Analysis → Jackknife Analysis

# Performs leave-one-out:
# - Remove each study one at a time
# - Repeat mean analysis
# - Check which results remain

# Output:
# - Jackknife map (voxels significant in all iterations)
# - Table showing impact of each study

# Interpretation:
# - High replicability = robust findings
# - Low replicability = driven by single studies
```

### Heterogeneity Analysis

```bash
# Assess between-study variability

# Analysis → Heterogeneity

# Computes Q statistic per voxel:
# - Tests whether variability > expected by chance
# - I² statistic (percentage of true heterogeneity)

# Output:
# - Heterogeneity map (Q statistic)
# - I² map
# - Regions with significant heterogeneity

# High heterogeneity suggests:
# - Need for moderator analyses
# - Study differences (methods, populations)
```

## Meta-Regression

### Continuous Moderators

```bash
# Test correlation with continuous variables
# Example: Age, symptom severity, illness duration

# 1. Add moderator to studies table:
# Studies → Edit Variables
# Add column: age (or other variable)
# Enter values for each study

# 2. Run meta-regression:
# Analysis → Meta-Regression
# Select moderator variable
# Set permutations: 500

# Output:
# - Beta map (regression slope per voxel)
# - Z-map (significance)
# - P-map (FWE-corrected)

# Interpretation:
# - Positive beta: activation increases with moderator
# - Negative beta: activation decreases
```

### Example: Age Correlation

```text
# studies.txt with age variable

study	n	contrast	space	age
study1	20	Depression>Control	MNI	35.2
study2	25	Depression>Control	MNI	42.8
study3	18	Depression>Control	MNI	38.5
study4	22	Depression>Control	MNI	45.3

# Meta-regression tests:
# - Where does activation correlate with age?
# - Positive or negative relationship?
```

### Categorical Moderators

```bash
# Compare subgroups
# Example: Medication status, task type

# 1. Add categorical variable:
# medicated (yes=1, no=0)

# 2. Subgroup analysis:
# Analysis → Subgroup Analysis
# Select grouping variable
# Performs separate meta-analyses per group

# 3. Direct comparison:
# Analysis → Comparison
# Tests: Group1 > Group2 and Group2 > Group1

# Output:
# - Separate maps per group
# - Contrast maps
# - Tables
```

## Advanced Features

### Multiple Imputation

```bash
# SDM's strength: handles missing data

# Many studies don't report:
# - All significant clusters
# - Negative findings
# - Effect sizes

# SDM imputes unreported effects:
# - Based on spatial patterns
# - Correlation with reported findings
# - Multiple imputations (reduces bias)

# Automatically done during preprocessing
# Parameters in: Analysis → Preprocessing → Advanced
```

### Publication Bias Assessment

```bash
# Check for publication bias

# Analysis → Funnel Plot

# Produces:
# - Funnel plot (effect size vs. precision)
# - Egger's test (asymmetry)
# - Begg's test

# Interpretation:
# - Symmetric funnel = low bias
# - Asymmetric = potential publication bias
# - Statistical tests confirm visual inspection
```

### ROI Analysis

```bash
# Extract values from regions of interest

# Tools → ROI Analysis

# 1. Load ROI mask (NIfTI format)
# 2. Select analysis (mean, meta-regression)
# 3. Extract values per study

# Output:
# - Mean effect size per ROI
# - Standard error
# - Test statistics
# - Table format

# Useful for:
# - Hypothesis testing in specific regions
# - Comparing with coordinates from other studies
# - Meta-analytic connectivity
```

### Coordinate-Based ROI Analysis

```bash
# Define ROI from coordinates

# Tools → Sphere ROI
# Enter: x, y, z, radius
# Creates spherical ROI

# Or use anatomical atlas:
# Tools → Atlas ROI
# Select region from AAL, Harvard-Oxford, etc.

# Then run ROI analysis as above
```

## Comparison with Other Methods

### SDM vs. ALE

```text
# ALE (GingerALE):
# + Well-established, widely used
# + Fast computation
# - Only uses peak coordinates
# - Cannot do meta-regression
# - Binary (activated or not)

# SDM:
# + Uses effect sizes (full maps)
# + Meta-regression possible
# + Handles unreported findings
# - More complex
# - Requires more information
# - Slower computation

# When to use SDM:
# - Have effect sizes or t-values
# - Want meta-regression
# - Need to assess heterogeneity
# - Interested in moderator effects
```

## Output Files

### Main Results

```bash
# After mean analysis:
# SDM_MeanAnalysis_z.nii           # Z-scores
# SDM_MeanAnalysis_p.nii           # Uncorrected p
# SDM_MeanAnalysis_p_FWE.nii       # FWE-corrected p
# SDM_MeanAnalysis_effectsize.nii  # Effect sizes (d)
# SDM_MeanAnalysis_clusters.txt    # Cluster table

# Use in standard neuroimaging software
```

### Cluster Tables

```text
# Example cluster table:
Cluster	Voxels	MNI_x	MNI_y	MNI_z	SDM-Z	p-value	Region
1	 1248	-42	15	24	5.67	<0.001	L IFG
2	 896	45	18	21	5.21	<0.001	R IFG
3	 654	-6	24	50	4.89	0.002	Medial PFC
4	 432	-38	-52	-18	4.52	0.005	L Fusiform
```

## Visualization

### Built-in Viewer

```bash
# Results → View Results

# Options:
# - Slice view (axial, sagittal, coronal)
# - Overlay on MNI template
# - Adjust threshold
# - Toggle clusters
# - Navigation

# Export slices:
# View → Export Image
# Save as PNG or TIFF
```

### External Visualization

```bash
# Export NIfTI and use other tools

# FSLeyes
fsleyes MNI152_T1_2mm.nii.gz \
        SDM_MeanAnalysis_p_FWE.nii.gz \
        -cm red-yellow -dr 0.95 1

# Nilearn (Python)
from nilearn import plotting
plotting.plot_glass_brain(
    'SDM_MeanAnalysis_p_FWE.nii.gz',
    threshold=0.95,
    colorbar=True,
    title='SDM Meta-Analysis'
)

# MRIcron
mricron template.nii -o SDM_MeanAnalysis_p_FWE.nii.gz
```

## Best Practices

### Study Selection

```bash
# Include:
# - Whole-brain analyses
# - Reported effect sizes or t/z/p values
# - Independent samples
# - Consistent contrasts

# Optimal:
# - Effect sizes (Cohen's d, Hedges' g)
# - T-values or Z-scores
# - Sample sizes

# Can work with just:
# - Peak coordinates
# - Sample sizes
# (SDM will impute, but less precise)
```

### Sample Size Requirements

```bash
# Minimum recommendations:
# - 10 studies (absolute minimum)
# - 15+ studies (better)
# - 20+ studies (ideal)

# Meta-regression:
# - 10 studies per moderator tested
# - More studies = more power
```

### Effect Size Extraction

```bash
# From papers, extract:
# 1. Peak coordinates (x, y, z)
# 2. T-values or Z-scores at peaks
# 3. Sample sizes per group
# 4. P-values if no t/z available

# Conversion if needed:
# t-to-z, p-to-z converters
# Available in SDM: Tools → Convert Statistics
```

### Statistical Rigor

```bash
# Recommended settings:
# - Anisotropic kernel: FWHM 20mm
# - FWE correction via permutation
# - Permutations: 500 for publication
# - Threshold: p < 0.05 FWE
# - Always run jackknife analysis
# - Check heterogeneity
# - Assess publication bias
```

## Reporting Results

### Methods Section Template

```text
"Meta-analysis was performed using SDM version X.XX.
Studies were preprocessed using an anisotropic kernel
with FWHM=20mm. Mean effect size analysis was conducted
using random-effects model. Statistical significance was
assessed using family-wise error correction based on
500 permutations (p < 0.05 FWE-corrected). Heterogeneity
was assessed using Q statistics. Jackknife sensitivity
analysis confirmed robustness of findings. Meta-regression
tested correlations with [moderator variable]."
```

### Results Reporting

```text
# For each significant cluster report:
# - Peak coordinates (MNI x, y, z)
# - SDM-Z value
# - Cluster size (voxels or mm³)
# - P-value (FWE-corrected)
# - Anatomical label

# Example:
"Significant clusters were identified in left inferior
frontal gyrus (MNI: -42, 15, 24; SDM-Z = 5.67;
k = 1248 voxels; p < 0.001 FWE) and right IFG
(MNI: 45, 18, 21; SDM-Z = 5.21; k = 896 voxels;
p < 0.001 FWE)."
```

## Integration with Claude Code

When helping users with SDM:

1. **Check Installation:**
   - Launch SDM.exe successfully
   - Verify version in Help → About

2. **Data Preparation:**
   - Create studies.txt table
   - Collect coordinates files
   - Extract effect sizes from papers

3. **Standard Workflow:**
   - Import studies → Preprocess → Mean analysis → Jackknife → Export

4. **Meta-Regression:**
   - Add moderator variables
   - Check for multicollinearity
   - Interpret regression slopes

## Troubleshooting

**Problem:** SDM won't launch
**Solution:** Install/update MATLAB Runtime, check Windows compatibility mode

**Problem:** Preprocessing fails
**Solution:** Check coordinate format, verify study files exist, reduce voxel resolution

**Problem:** No significant results
**Solution:** Check input quality, verify effect sizes, consider more liberal threshold

**Problem:** Extreme heterogeneity
**Solution:** Review study selection, check for outliers, consider moderator analysis

**Problem:** Inconsistent with published meta-analysis
**Solution:** Verify same studies included, check SDM version, compare parameters

## Resources

- **Website:** https://www.sdmproject.com/
- **Manual:** https://www.sdmproject.com/manual/
- **Forum:** https://www.sdmproject.com/forum/
- **Tutorials:** Included in software download
- **Publications:** https://www.sdmproject.com/publications/

## Citation

```bibtex
@article{radua2012sdm,
  title={A new meta-analytic method for neuroimaging studies that combines reported peak coordinates and statistical parametric maps},
  author={Radua, Joaquim and Mataix-Cols, David and Phillips, Mary L and El-Hage, Wissam and Kronhaus, Daniel M and Cardoner, Narcis and Surguladze, Simon},
  journal={European Psychiatry},
  volume={27},
  number={8},
  pages={605--611},
  year={2012},
  publisher={Elsevier}
}

@article{albajes2019anisotropic,
  title={Anisotropic kernels for coordinate-based meta-analyses of neuroimaging studies},
  author={Albajes-Eizagirre, Anton and Solanes, Aleix and Vieta, Eduard and Radua, Joaquim},
  journal={Frontiers in Psychiatry},
  volume={10},
  pages={439},
  year={2019},
  publisher={Frontiers}
}
```

## Related Tools

- **GingerALE:** Activation likelihood estimation
- **NiMARE:** Python meta-analysis framework (includes SDM)
- **NeuroSynth:** Automated large-scale meta-analysis
- **Meta-Analyst:** Alternative meta-analysis software
- **BrainMap:** Coordinate database
