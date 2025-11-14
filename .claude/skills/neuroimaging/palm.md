# PALM (Permutation Analysis of Linear Models)

## Overview

PALM (Permutation Analysis of Linear Models) is a versatile tool for permutation-based inference in neuroimaging. It provides a powerful framework for statistical testing with proper control of family-wise error rate (FWER) or false discovery rate (FDR) using permutation and sign-flipping methods. PALM can handle complex experimental designs including repeated measures, multiple modalities, and non-exchangeable data.

**Website:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM
**Platform:** MATLAB/Octave or standalone (Linux/macOS)
**Language:** MATLAB/Octave
**License:** Apache License 2.0

## Key Features

- Permutation and sign-flipping for nonparametric inference
- Support for complex GLM designs
- Repeated measures and within-subject factors
- Multiple modalities (VBM, TBSS, fMRI, surface data)
- Accelerated permutation methods (tail approximation)
- Multiple comparison correction (FWER, FDR, cluster)
- Threshold-Free Cluster Enhancement (TFCE)
- Variance group analysis
- Multi-modal CCA (Canonical Correlation Analysis)
- NPC (Non-Parametric Combination) for joint inference

## Installation

### Download PALM

```bash
# Download from FSL website
wget https://fsl.fmrib.ox.ac.uk/fsldownloads/palm/palm-latest.tar.gz

# Extract
tar -xzf palm-latest.tar.gz
cd palm-alpha115/

# Make executable
chmod +x palm
```

### MATLAB/Octave Version

```matlab
% Add PALM to MATLAB path
addpath('/path/to/palm');
savepath;

% Verify installation
which palm
```

### Standalone Version

```bash
# Add to PATH
export PATH=/path/to/palm:$PATH

# Test
palm --version
```

## Basic Concepts

### Permutation Testing

- **Null hypothesis:** No difference between groups/conditions
- **Method:** Randomly permute group labels, recompute test statistics
- **Distribution:** Build null distribution from permutations
- **P-value:** Proportion of permutations with statistic ≥ observed

### When to Use Sign-Flipping

- **Paired data:** Within-subject designs
- **One-sample tests:** Testing against zero
- **Method:** Randomly flip signs of differences

### Exchangeability Blocks

- Define which observations can be permuted
- Critical for repeated measures designs
- Incorrect blocks → invalid inference

## Basic Usage

### Two-Sample T-Test (Unpaired)

```bash
# Simplest case: compare two groups
# Group 1: 20 subjects
# Group 2: 20 subjects

# Create design matrix (-d)
# Create contrast (-c)
# Input data (-i)
# Mask (-m)
# Number of permutations (-n)

palm \
  -i merged_4D_data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -o results_twosample
```

### Design Matrix for Two-Group Comparison

```matlab
% Create design.mat
% 20 subjects in Group 1, 20 in Group 2

% Design matrix: intercept + group effect
design = [
    ones(20, 1),  ones(20, 1);   % Group 1
    ones(20, 1),  zeros(20, 1)   % Group 2
];

% Save in FSL format
save('design.mat', 'design', '-ascii');

% Contrast: Group1 > Group2
contrast = [0, 1];
save('design.con', 'contrast', '-ascii');
```

### One-Sample T-Test (Paired)

```bash
# Test if values differ from zero
# E.g., pre-post difference scores

# Create design (single column of ones)
echo "1" > design_ones.mat
for i in {2..20}; do echo "1" >> design_ones.mat; done

# Contrast
echo "1" > design_ones.con

# Run PALM with sign-flipping
palm \
  -i difference_scores.nii.gz \
  -d design_ones.mat \
  -t design_ones.con \
  -m mask.nii.gz \
  -n 10000 \
  -within \
  -o results_onesample
```

## Advanced Designs

### Repeated Measures (Within-Subject Design)

```bash
# Example: 3 time points per subject, 15 subjects
# Total: 45 observations

# Create exchangeability blocks file
# Format: one line per observation
# Same number = same exchangeability block

# Exchange blocks (EB): subjects
# 1 1 1  (subject 1, 3 timepoints)
# 2 2 2  (subject 2, 3 timepoints)
# ...
# 15 15 15 (subject 15, 3 timepoints)

# Create EB file
awk 'BEGIN{for(s=1;s<=15;s++) for(t=1;t<=3;t++) print s}' > subjects.csv

# Design matrix: model timepoints
# [Intercept, Time1, Time2, Time3]
# Or use within-subject contrast

palm \
  -i data_4D.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -eb subjects.csv \
  -n 10000 \
  -o results_repeated
```

### ANCOVA (Covariate Analysis)

```bash
# Control for age and sex
# Design matrix: [Intercept, Group, Age, Sex]

# Example design.mat:
# 1  1  25  1
# 1  1  30  0
# 1  0  28  1
# 1  0  32  0
# ...

# Contrast for group effect (controlling covariates)
# [0, 1, 0, 0]

palm \
  -i data.nii.gz \
  -d design_with_covariates.mat \
  -t contrast_group.con \
  -m mask.nii.gz \
  -n 10000 \
  -demean \
  -o results_ancova
```

### F-Test (Multiple Contrasts)

```bash
# Test multiple contrasts simultaneously
# Example: main effect of diagnosis (3 groups)

# Create F-contrast file (.fts)
# Lists which contrasts to combine

# contrasts.con:
# 1 -1  0  (HC > Patient1)
# 0  1 -1  (Patient1 > Patient2)

# contrasts.fts:
# 1
# 1

palm \
  -i data.nii.gz \
  -d design.mat \
  -t contrasts.con \
  -f contrasts.fts \
  -m mask.nii.gz \
  -n 10000 \
  -o results_ftest
```

## Threshold-Free Cluster Enhancement (TFCE)

```bash
# TFCE: combines height and extent without arbitrary threshold
# More powerful than cluster-based inference

palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -tfce 2.0 \
  -o results_tfce

# TFCE parameters (usually defaults work well):
# H (height exponent): 2.0
# E (extent exponent): 0.5 (volumetric), 1.0 (surface)
# dh (step size): 0.1

# Custom TFCE parameters:
palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -tfce 2.0 0.5 0.1 \
  -o results_tfce_custom
```

## Multiple Comparison Correction

### Voxel-Wise FWE Correction

```bash
# Control family-wise error rate at voxel level
palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -voxelwise \
  -corrmod \
  -o results_fwe
```

### Cluster-Based Correction

```bash
# Cluster forming threshold: t = 2.3 (p < 0.01)
# Then test cluster size/mass

palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -C 2.3 \
  -corrcon \
  -o results_cluster

# Multiple cluster-forming thresholds
palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -C 2.0,2.3,2.6,3.0 \
  -corrcon \
  -o results_multicluster
```

### False Discovery Rate (FDR)

```bash
# Control FDR instead of FWER
palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -fdr \
  -o results_fdr
```

## Accelerated Permutations

### Tail Approximation

```bash
# Speed up inference for large datasets
# Fits Gamma distribution to permutation tail
# Requires fewer permutations

palm \
  -i large_data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 1000 \
  -accel tail \
  -o results_tail

# Can achieve equivalent power with ~1000 permutations
# vs. 10000 without acceleration
```

### NPC (Non-Parametric Combination)

```bash
# Combine p-values across modalities/contrasts
palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -npc \
  -o results_npc
```

## Specialized Applications

### TBSS (Tract-Based Spatial Statistics)

```bash
# Analyze DTI skeleton data

# TBSS preprocessing creates:
# - all_FA_skeletonised.nii.gz
# - mean_FA_skeleton_mask.nii.gz

palm \
  -i all_FA_skeletonised.nii.gz \
  -d design.mat \
  -t design.con \
  -m mean_FA_skeleton_mask.nii.gz \
  -n 10000 \
  -tfce 2.0 1.0 \
  -corrcon \
  -o tbss_results

# Typical TBSS TFCE parameters:
# H = 2.0, E = 1.0 (higher extent for skeleton)
```

### Surface-Based Analysis

```bash
# Analyze cortical thickness, area, etc.
# Input: GIFTI or ASCII format

palm \
  -i lh.thickness.mgh \
  -d design.mat \
  -t design.con \
  -s surf/lh.white.avg.area.mgh \
  -n 10000 \
  -tfce 2.0 1.0 \
  -corrcon \
  -o surf_results

# -s: surface file for TFCE computation
```

### Multi-Modal CCA

```bash
# Canonical Correlation Analysis
# Find relationships between imaging and behavior

palm \
  -i imaging_data.nii.gz \
  -d behavioral_data.mat \
  -m mask.nii.gz \
  -n 10000 \
  -cca \
  -o cca_results

# Outputs canonical variates and correlations
```

### Variance Group Analysis

```bash
# Test for differences in variance between groups
# (not just mean differences)

palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -vg \
  -o results_variance
```

## Working with Design Matrices

### Create Design in MATLAB

```matlab
% Complex design example
% 2 groups × 2 conditions (mixed design)

% 20 subjects per group
% 2 measurements per subject (condition A, B)

n_per_group = 20;

% Design matrix
% [Intercept, Group, Condition, Group×Condition]
design = [
    % Group 1, Condition A
    ones(n_per_group, 1), ones(n_per_group, 1), ones(n_per_group, 1), ones(n_per_group, 1);
    % Group 1, Condition B
    ones(n_per_group, 1), ones(n_per_group, 1), zeros(n_per_group, 1), zeros(n_per_group, 1);
    % Group 2, Condition A
    ones(n_per_group, 1), zeros(n_per_group, 1), ones(n_per_group, 1), zeros(n_per_group, 1);
    % Group 2, Condition B
    ones(n_per_group, 1), zeros(n_per_group, 1), zeros(n_per_group, 1), zeros(n_per_group, 1)
];

% Save
save('design_mixed.mat', 'design', '-ascii');

% Contrasts
% Main effect of group
con_group = [0, 1, 0, 0];
% Main effect of condition
con_condition = [0, 0, 1, 0];
% Interaction
con_interaction = [0, 0, 0, 1];

save('contrasts.con', 'con_group', 'con_condition', 'con_interaction', '-ascii');
```

### Exchangeability Blocks for Repeated Measures

```matlab
% 15 subjects, 3 time points each
n_subjects = 15;
n_timepoints = 3;

% Create EB file
eb = repelem(1:n_subjects, n_timepoints)';

% Save
writematrix(eb, 'eb_subjects.csv');

% Whole-block structure
% 1 1 1
% 2 2 2
% ...
% 15 15 15
```

## Interpreting Results

### Output Files

```bash
# PALM generates multiple output files:

# *_tstat.nii.gz        - T-statistics
# *_vox_tstat_fwep.nii.gz - Voxel-wise FWE-corrected p-values
# *_tfce_tstat_fwep.nii.gz - TFCE FWE-corrected p-values
# *_clustere_tstat_fwep.nii.gz - Cluster extent FWE p-values
# *_clusterm_tstat_fwep.nii.gz - Cluster mass FWE p-values

# View results
fsleyes \
  mean_FA.nii.gz \
  results_tfce_tstat_fwep.nii.gz \
  -cm red-yellow -dr 0.95 1
```

### Extract Significant Clusters

```bash
# Threshold at p < 0.05 (FWE-corrected)
fslmaths results_tfce_tstat_fwep.nii.gz \
  -uthr 0.05 \
  -bin \
  sig_mask.nii.gz

# Get cluster information
cluster \
  --in=sig_mask.nii.gz \
  --thresh=0.5 \
  --mm \
  > cluster_info.txt
```

## Batch Processing

```bash
#!/bin/bash
# Batch PALM for multiple contrasts

contrasts=(
    "HC_vs_Patient1"
    "HC_vs_Patient2"
    "Patient1_vs_Patient2"
)

for contrast in "${contrasts[@]}"; do
    echo "Running PALM for: $contrast"

    palm \
      -i data.nii.gz \
      -d design.mat \
      -t ${contrast}.con \
      -m mask.nii.gz \
      -n 10000 \
      -tfce 2.0 \
      -corrcon \
      -o results_${contrast}

    echo "Completed: $contrast"
done

echo "All contrasts completed!"
```

## Integration with FSL Tools

### Combine with Randomise

```bash
# PALM is more flexible than randomise
# But randomise can be faster for simple designs

# Equivalent randomise command:
randomise \
  -i data.nii.gz \
  -o randomise_results \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -T

# Convert to PALM (more options):
palm \
  -i data.nii.gz \
  -d design.mat \
  -t design.con \
  -m mask.nii.gz \
  -n 10000 \
  -tfce 2.0 \
  -corrcon \
  -o palm_results
```

### VBM Pipeline Integration

```bash
# After FSL VBM preprocessing

# Merge all subject GM images
fslmerge -t all_GM_mod sub*_GM_mod.nii.gz

# Create mask
fslmaths all_GM_mod -Tmean -thr 0.2 -bin GM_mask

# Run PALM
palm \
  -i all_GM_mod.nii.gz \
  -d design.mat \
  -t design.con \
  -m GM_mask.nii.gz \
  -n 10000 \
  -tfce 2.0 \
  -corrcon \
  -o vbm_results
```

## Integration with Claude Code

When helping users with PALM:

1. **Verify Design Matrix:**
   ```matlab
   % Check dimensions
   design = load('design.mat');
   fprintf('Design: %d observations, %d regressors\n', size(design));

   % Visualize
   imagesc(design);
   colorbar;
   title('Design Matrix');
   ```

2. **Check Exchangeability Blocks:**
   ```matlab
   % Load EB
   eb = readmatrix('eb.csv');

   % Verify length matches data
   assert(length(eb) == size(design, 1));

   % Check structure
   unique(eb)  % Should match number of subjects
   ```

3. **Monitor Progress:**
   ```bash
   # PALM prints progress to console
   # For long runs, use screen or tmux
   screen -S palm_analysis
   palm -i data.nii.gz ... -n 50000 -o results
   # Ctrl+A, D to detach
   ```

4. **Common Issues:**
   - Design matrix rank deficiency → remove collinear columns
   - Wrong EB structure → invalid permutations
   - Too few permutations → inaccurate p-values
   - Memory issues → process in chunks or use tail approximation

## Troubleshooting

**Problem:** "Design matrix is rank deficient"
**Solution:** Check for collinearity, mean-center covariates, remove redundant columns

**Problem:** "Permutations exhausted"
**Solution:** Exact number of possible permutations is limited; use sign-flipping or increase sample size

**Problem:** Very long runtime
**Solution:** Use tail approximation (-accel tail), reduce permutations initially for testing

**Problem:** No significant results
**Solution:** Check effect size, increase sample size, verify preprocessing quality, try TFCE

**Problem:** Results differ from randomise
**Solution:** PALM uses different permutation scheme; results should be similar but not identical

## Best Practices

1. **Permutations:**
   - Use ≥5000 for initial analysis
   - Use 10000+ for publication
   - Use tail approximation for large datasets

2. **Multiple Comparison Correction:**
   - TFCE recommended for most applications
   - Cluster-based useful for large, spatially extended effects
   - Report correction method clearly

3. **Design Specification:**
   - Double-check exchangeability blocks
   - Mean-center continuous covariates
   - Verify contrast coding

4. **Reporting:**
   - Report number of permutations used
   - Report correction method (TFCE, cluster, voxel-wise)
   - Report TFCE parameters if modified
   - Show uncorrected and corrected results

## Resources

- **FSL Wiki:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM
- **User Guide:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM/UserGuide
- **Examples:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM/Examples
- **Mailing List:** https://www.jiscmail.ac.uk/FSL

## Citation

```bibtex
@article{winkler2014permutation,
  title={Permutation inference for the general linear model},
  author={Winkler, Anderson M and Ridgway, Gerard R and Webster, Matthew A and Smith, Stephen M and Nichols, Thomas E},
  journal={Neuroimage},
  volume={92},
  pages={381--397},
  year={2014},
  publisher={Elsevier}
}

@article{winkler2016faster,
  title={Faster permutation inference in brain imaging},
  author={Winkler, Anderson M and Webster, Matthew A and Brooks, Jonathan C and Tracey, Irene and Smith, Stephen M and Nichols, Thomas E},
  journal={Neuroimage},
  volume={141},
  pages={502--516},
  year={2016},
  publisher={Elsevier}
}
```

## Related Tools

- **Randomise (FSL):** Simpler permutation testing tool
- **SnPM:** Statistical nonparametric mapping in SPM
- **NBS:** Network-based statistic for connectivity
- **TFCE:** Threshold-free cluster enhancement
- **FSL GLM:** General linear model setup
