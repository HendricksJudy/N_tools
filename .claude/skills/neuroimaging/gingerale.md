# GingerALE

## Overview

GingerALE (formerly just ALE) is a standalone Java application for performing coordinate-based meta-analyses of neuroimaging data using the Activation Likelihood Estimation (ALE) algorithm. Developed by the BrainMap project, GingerALE provides a user-friendly interface for conducting rigorous meta-analyses with modern statistical corrections, making it the gold standard for ALE meta-analysis.

**Website:** https://brainmap.org/ale/
**Platform:** Windows/Linux (Java-based)
**Language:** Java
**License:** Free for academic use

## Key Features

- Activation Likelihood Estimation (ALE) algorithm
- Family-wise error (FWE) correction via permutation
- Cluster-level thresholding
- Contrast and conjunction analyses
- Meta-Analytic Connectivity Modeling (MACM)
- Failsafe N calculation
- Heterogeneity analysis
- Both GUI and command-line interfaces
- Integration with BrainMap database
- Sleuth text file format
- Cross-platform Java implementation

## Installation

### Requirements

```bash
# Java Runtime Environment 8 or later
java -version

# Should show version 1.8.0 or higher
```

### Download and Install

```bash
# Download from BrainMap website
# https://brainmap.org/ale/

# Linux/macOS
unzip GingerALE_3.0.2.zip
cd GingerALE_3.0.2

# Make executable
chmod +x GingerALE.sh

# Launch
./GingerALE.sh

# Windows
# Extract zip file
# Double-click GingerALE.bat
```

### Verify Installation

```bash
# Launch GingerALE
java -jar GingerALE.jar

# GUI should appear
# File menu should be accessible
```

## Data Format (Sleuth Text Files)

### File Structure

```text
// Reference=MNI
// Subjects in Experiment 1=20
//
// Working Memory > Baseline
-42  15  24
45   18  21
-12  -85  2

// Subjects in Experiment 2=25
//
// Working Memory > Baseline
-44  12  27
-38  -52  -18
0    -6  52
```

### Format Specifications

```text
# Key elements:
# 1. Reference space (MNI or Talairach)
# 2. Subject count per experiment
# 3. Contrast name
# 4. Coordinates (x, y, z)

# Rules:
# - Comments start with //
# - Each experiment separated by blank line with subject count
# - Coordinates: Tab or space-separated
# - Can include multiple contrasts per study
```

### Example Sleuth File

```text
// Reference=MNI
// GingerALE Meta-Analysis: Emotion Processing
// Created: 2024-01-15
//
// Subjects in Experiment 1=18
// Ochsner et al. (2004)
// Emotion Regulation > Baseline
-6   24  50
-44  14  26
44   16  24
-10  -52  16

// Subjects in Experiment 2=22
// Phan et al. (2005)
// Emotion > Neutral
-4   2   48
-42  22  -8
42   20  -6
-24  -6  -18
24   -4  -16

// Subjects in Experiment 3=20
// Kober et al. (2008)
// Negative Emotion > Baseline
-6   10  46
0    24  40
-40  16  -2
```

## Basic Workflow (GUI)

### Step 1: Create Sleuth File

```bash
# In GingerALE GUI:
# File → New

# Or manually create .txt file following format above
# Save with .txt extension
```

### Step 2: Load Data

```bash
# File → Open
# Select your .txt file

# GingerALE displays:
# - Number of experiments
# - Total coordinates
# - Coordinate space
# - Subject counts
```

### Step 3: Run ALE Analysis

```bash
# Compute → Run ALE Analysis

# Set parameters:
# - Cluster-forming threshold (p-value)
# - Correction method (FWE, FDR, uncorrected)
# - Number of permutations (5000-10000)
# - Minimum cluster volume (mm³)

# Click "Compute"
# Wait for processing (seconds to minutes)
```

### Step 4: View Results

```bash
# Results panel shows:
# - Thresholded ALE map
# - Cluster table (coordinates, volume, labels)
# - Statistical values

# Export results:
# - File → Save Results → NIfTI image
# - File → Save Results → Cluster table
```

## Command-Line Usage

### Basic ALE Analysis

```bash
# Run from command line
java -Xmx4g -cp GingerALE.jar org.brainmap.meta.getALE2 \
  input_file.txt \
  output_prefix \
  5000 \
  0.001 \
  cluster

# Parameters:
# -Xmx4g: Allocate 4GB memory
# input_file.txt: Sleuth format coordinates
# output_prefix: Name for output files
# 5000: Number of permutations
# 0.001: Cluster-forming p-threshold
# cluster: Perform cluster-level correction
```

### Batch Processing

```bash
#!/bin/bash
# Batch process multiple meta-analyses

# Array of input files
files=(
    "emotion.txt"
    "cognition.txt"
    "motor.txt"
    "visual.txt"
)

# Process each
for file in "${files[@]}"; do
    echo "Processing $file..."

    # Extract base name
    base=$(basename "$file" .txt)

    # Run ALE
    java -Xmx4g -cp GingerALE.jar org.brainmap.meta.getALE2 \
        "$file" \
        "results/${base}" \
        10000 \
        0.001 \
        cluster

    echo "  Completed $file"
done

echo "Batch processing complete!"
```

## ALE Algorithm Parameters

### Cluster-Forming Threshold

```bash
# Voxel-level p-value for initial thresholding
# Typical values:
# - 0.001 (recommended)
# - 0.0001 (more conservative)
# - 0.01 (more liberal, use with caution)

# Set in GUI:
# Preferences → Cluster-forming threshold → 0.001
```

### Number of Permutations

```bash
# More permutations = more precise p-values
# Recommendations:
# - 5000: Minimum for publication
# - 10000: Standard (recommended)
# - 20000: High precision

# Trade-off: computation time vs. precision

# Set in GUI:
# Preferences → Permutations → 10000
```

### Minimum Cluster Size

```bash
# Minimum volume (mm³) for significant clusters
# Typical values:
# - 100 mm³ (less conservative)
# - 200 mm³ (standard)
# - 500 mm³ (more conservative)

# Helps reduce false positives from small clusters

# Set in GUI:
# Preferences → Minimum cluster volume → 200
```

## Contrast Analysis

### Two-Group Subtraction

```bash
# Compare two groups of studies
# Example: Emotion > Cognition

# 1. Create two separate Sleuth files:
#    - emotion.txt
#    - cognition.txt

# 2. Load first file in GingerALE

# 3. Compute → Subtraction Analysis

# 4. Select second file as contrast

# 5. Set parameters:
#    - Permutations: 10000
#    - Cluster threshold: 0.001

# 6. Run analysis

# Results:
# - Group1 > Group2
# - Group2 > Group1
# - Both saved separately
```

### Conjunction Analysis

```bash
# Find common activations across analyses

# 1. Load multiple Sleuth files

# 2. Compute → Conjunction Analysis

# 3. Select files to include

# 4. Set parameters

# 5. Run

# Result: Voxels significant in all analyses
```

## Meta-Analytic Connectivity Modeling (MACM)

```bash
# Find brain regions co-activated with seed region

# Requires access to BrainMap database
# (License needed for local BrainMap)

# 1. Define seed region (ROI)

# 2. Query BrainMap for studies with peaks in seed

# 3. Extract coordinates from those studies

# 4. Create Sleuth file with co-activation coordinates

# 5. Run standard ALE on co-activations

# Result: Connectivity map for seed region
```

## Quality Control and Diagnostics

### Failsafe N

```bash
# Calculate failsafe N (robustness measure)
# "How many null studies needed to eliminate significance?"

# In GUI:
# Compute → Failsafe N

# Interpretation:
# - Higher = more robust
# - Rule of thumb: 5k + 10, where k = # studies
# - If Failsafe N > this, results robust
```

### Heterogeneity Analysis

```bash
# Test for between-study heterogeneity

# Compute → Heterogeneity Analysis

# Examines whether:
# - Studies show consistent activations
# - Variability beyond sampling error

# Results:
# - Chi-square test per voxel
# - Map showing heterogeneous regions

# High heterogeneity suggests:
# - Methodological differences
# - Population differences
# - Need for moderator analyses
```

### Contribution Analysis

```bash
# Assess individual study contributions

# For each study:
# - Run ALE with all studies
# - Run ALE excluding that study
# - Compare results

# Identifies influential studies

# In GUI:
# Compute → Contribution Analysis

# Time-consuming (N analyses for N studies)
```

## Output Files

### NIfTI Images

```bash
# GingerALE generates:
# - {prefix}_ALE.nii: Raw ALE values
# - {prefix}_ALE_z.nii: Z-scores
# - {prefix}_ALE_p.nii: P-values
# - {prefix}_ALE_FWE.nii: Corrected (thresholded)

# Use in any neuroimaging software
# MNI152 space (2mm resolution)
```

### Cluster Tables

```bash
# CSV or text table with:
# - Cluster ID
# - Volume (mm³)
# - Peak coordinates (x, y, z)
# - Peak ALE value
# - Peak Z-score
# - Cluster-level p-value (FWE-corrected)
# - Anatomical labels

# Example:
# Cluster, Volume, X, Y, Z, ALE, Z, p(FWE), Label
# 1, 1248, -42, 15, 24, 0.032, 5.67, <0.001, L IFG
# 2, 896, 45, 18, 21, 0.028, 5.21, <0.001, R IFG
```

### Summary Report

```bash
# Text file with:
# - Analysis parameters
# - Number of experiments/coordinates
# - Threshold information
# - Cluster statistics
# - Warnings/notes

# Useful for methods section
```

## Integration with Other Tools

### Export to Other Software

```bash
# NIfTI files can be used in:
# - SPM: Overlay on anatomical, extract values
# - FSL: FSLeyes visualization, cluster tools
# - AFNI: 3dClusterize, viewer
# - Nilearn: Python visualization
# - NiMARE: Further analyses

# Example: Visualize in FSLeyes
fsleyes $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz \
         emotion_ALE_FWE.nii.gz -cm red-yellow -dr 3 10
```

### Import from Literature

```bash
# Extract coordinates from papers

# 1. Manual entry:
#    - Copy coordinates from tables
#    - Format as Sleuth file

# 2. Semi-automated:
#    - Neurosynth export
#    - Convert to Sleuth format

# 3. From databases:
#    - BrainMap Sleuth
#    - Export coordinates
```

### Convert from Other Formats

```bash
# NiMARE to GingerALE
# In Python:
from nimare.io import convert_dataset_to_sleuth
convert_dataset_to_sleuth(dset, 'output.txt')

# Now load in GingerALE
```

## Best Practices

### Study Selection

```bash
# Include:
# - Whole-brain analyses (not ROI)
# - Consistent contrast across studies
# - Independent samples
# - Proper statistical thresholding

# Exclude:
# - ROI-only studies
# - Duplicate samples/data
# - Small volume correction only
# - Different contrasts
```

### Minimum Study Requirements

```bash
# ALE meta-analysis recommendations:
# - Minimum: 17-20 experiments
# - Ideal: 25+ experiments
# - More experiments = more power

# Fewer studies:
# - Results less reliable
# - Underpowered
# - Consider MKDA instead
```

### Coordinate Space

```bash
# Be consistent:
# - All MNI or all Talairach
# - Don't mix without conversion

# If mixed in literature:
# - Convert Talairach to MNI
# - Use tal2icbm_spm or similar
# - Document conversion

# GingerALE expects:
# Reference=MNI (preferred)
# Reference=Talairach (if necessary)
```

### Statistical Thresholds

```bash
# Recommendations:
# 1. Cluster-forming: p < 0.001
# 2. Cluster-level FWE: p < 0.05
# 3. Minimum cluster: 200 mm³
# 4. Permutations: 10000

# Report in paper:
# "ALE meta-analysis with cluster-forming threshold
#  p < 0.001, cluster-level FWE correction p < 0.05,
#  10000 permutations, minimum cluster volume 200 mm³"
```

## Visualization

### In GingerALE

```bash
# Built-in viewer:
# - View → Display Results
# - Overlay on MNI template
# - Navigate slices
# - Click clusters for info

# Basic but functional
# For publication figures, export and use other tools
```

### External Visualization

```bash
# Nilearn (Python)
from nilearn import plotting
plotting.plot_glass_brain('emotion_ALE_FWE.nii.gz',
                           threshold=3.0,
                           colorbar=True)

# MRIcron
mricron template.nii.gz -o emotion_ALE_FWE.nii.gz

# FSLeyes
fsleyes MNI152_T1_2mm.nii.gz emotion_ALE_FWE.nii.gz
```

## Common Issues and Solutions

**Problem:** Java heap space error
**Solution:** Increase memory with `-Xmx4g` or `-Xmx8g`

**Problem:** No significant clusters
**Solution:** Check study quality, consider more liberal threshold, verify coordinates

**Problem:** Too many clusters (noisy results)
**Solution:** Use more conservative threshold, increase minimum cluster size

**Problem:** Sleuth file won't load
**Solution:** Check format (spaces vs tabs), verify reference space, check for special characters

**Problem:** Results differ from published ALE studies
**Solution:** Verify same ALE version, check parameters, ensure same studies included

## Integration with Claude Code

When helping users with GingerALE:

1. **Check Java:**
   ```bash
   java -version
   ```

2. **Validate Sleuth File:**
   ```bash
   # Check format
   head -20 input.txt
   ```

3. **Standard Parameters:**
   - Cluster-forming: p < 0.001
   - Permutations: 10000
   - Min cluster: 200 mm³
   - FWE correction

4. **Common Workflow:**
   - Literature search → Extract coordinates → Create Sleuth file → Run ALE → Visualize

## Troubleshooting

**Problem:** "Cannot load file"
**Solution:** Check file encoding (UTF-8), remove special characters, verify format

**Problem:** Analysis very slow
**Solution:** Reduce permutations for testing, increase Java memory, close other programs

**Problem:** Unexpected activation patterns
**Solution:** Review input coordinates, check coordinate space, verify study selection criteria

**Problem:** Empty cluster table
**Solution:** Lower threshold, check input data quality, verify enough studies

## Resources

- **Website:** https://brainmap.org/ale/
- **BrainMap:** https://brainmap.org/
- **Manual:** Included in download
- **Forum:** https://groups.google.com/g/brainmap-forum
- **Papers:** https://brainmap.org/pubs/

## Citation

```bibtex
@article{eickhoff2012ale,
  title={Activation likelihood estimation meta-analysis revisited},
  author={Eickhoff, Simon B and Bzdok, Danilo and Laird, Angela R and Kurth, Florian and Fox, Peter T},
  journal={Neuroimage},
  volume={59},
  number={3},
  pages={2349--2361},
  year={2012},
  publisher={Elsevier}
}

@article{turkeltaub2012minimizing,
  title={Minimizing within-experiment and within-group effects in activation likelihood estimation meta-analyses},
  author={Turkeltaub, Peter E and Eickhoff, Simon B and Laird, Angela R and Fox, Mick and Wiener, Martin and Fox, Peter},
  journal={Human brain mapping},
  volume={33},
  number={1},
  pages={1--13},
  year={2012},
  publisher={Wiley Online Library}
}
```

## Related Tools

- **NiMARE:** Python implementation of ALE and other CBMA methods
- **NeuroSynth:** Large-scale automated meta-analysis
- **AES-SDM:** Alternative meta-analysis method
- **BrainMap:** Coordinate database (source of GingerALE data)
- **Sleuth:** BrainMap database query tool
- **MKDA:** Alternative to ALE for smaller datasets
