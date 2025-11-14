# ICA-AROMA - Automatic Removal of Motion Artifacts

## Overview

ICA-AROMA (ICA-based Automatic Removal Of Motion Artifacts) is a data-driven method for identifying and removing motion-related artifacts from fMRI data using Independent Component Analysis (ICA). It automatically classifies ICA components as motion or non-motion using a set of predefined features, enabling non-aggressive denoising that preserves the signal of interest while effectively removing motion artifacts. ICA-AROMA is widely used as a preprocessing step and is integrated into fMRIPrep.

**Website:** https://github.com/maartenmennes/ICA-AROMA
**Platform:** Python (Linux/macOS)
**Language:** Python
**License:** Apache-2.0

## Key Features

- Automatic motion artifact detection
- ICA-based component classification
- Non-aggressive denoising (preserves signal)
- Four classification features (motion-related characteristics)
- Works with minimally preprocessed fMRI data
- Integrated into fMRIPrep pipeline
- Outputs cleaned data and component classifications
- Visual reports for quality control
- Compatible with FSL and other tools
- Single-subject processing
- No manual intervention required

## Installation

### Via pip

```bash
# Install ICA-AROMA
pip install ica-aroma

# Or from GitHub
pip install git+https://github.com/maartenmennes/ICA-AROMA.git

# Verify installation
python -m ICA_AROMA.ICA_AROMA --version
```

### Dependencies

```bash
# Required dependencies
pip install numpy scipy nibabel

# FSL required for preprocessing steps
# Download from: https://fsl.fmrib.ox.ac.uk/

# Verify FSL installation
which fsl
fslinfo
```

### Via fMRIPrep (Recommended)

```bash
# fMRIPrep includes ICA-AROMA
fmriprep /data /output participant --use-aroma

# This is the easiest way to use ICA-AROMA
```

## Prerequisites

### Required Inputs

```bash
# ICA-AROMA requires:
# 1. Preprocessed fMRI data (motion-corrected, registered)
# 2. Motion parameters (from realignment)
# 3. Brain mask

# Preprocessing steps needed:
# - Motion correction
# - Smoothing (6-8mm recommended, though optional)
# - Registration to MNI space (optional but recommended)
# - No temporal filtering (ICA-AROMA does this)
```

### Data Organization

```bash
# Organize input data
subject/
├── func_preprocessed.nii.gz    # Motion-corrected fMRI
├── motion_parameters.txt       # 6 columns (3 rotation, 3 translation)
└── brain_mask.nii.gz           # Brain mask in same space
```

## Basic Usage

### Single Subject

```bash
# Run ICA-AROMA
python -m ICA_AROMA.ICA_AROMA \
  -in sub-01_task-rest_bold.nii.gz \
  -out /output/sub-01_ICA-AROMA \
  -mc sub-01_motion.txt \
  -m sub-01_brain_mask.nii.gz

# Parameters:
#   -in: Input fMRI (4D NIfTI)
#   -out: Output directory
#   -mc: Motion parameters (FSL format: 6 columns)
#   -m: Brain mask
```

### With Automatic Mask Generation

```bash
# ICA-AROMA can generate mask automatically
python -m ICA_AROMA.ICA_AROMA \
  -in sub-01_task-rest_bold.nii.gz \
  -out /output/sub-01_ICA-AROMA \
  -mc sub-01_motion.txt \
  -den nonaggr

# -den: Denoising strategy
#   nonaggr: Non-aggressive (recommended)
#   aggr: Aggressive (removes full components)
#   no: No denoising (classification only)
```

### Advanced Options

```bash
# Full command with all options
python -m ICA_AROMA.ICA_AROMA \
  -in sub-01_bold.nii.gz \
  -out /output/ICA-AROMA \
  -mc motion_params.txt \
  -m brain_mask.nii.gz \
  -den nonaggr \
  -dim 0 \
  -tr 2.0

# Options:
#   -dim: Dimensionality reduction (0 = automatic, or specific number)
#   -tr: Repetition time (seconds)
#   -den: Denoising type (nonaggr, aggr, no)
#   -overwrite: Overwrite existing output
```

## Understanding the Method

### ICA-AROMA Workflow

```
1. Melodic ICA
   └─> Decompose fMRI into independent components

2. Feature Extraction (4 features per component)
   ├─> Maximum correlation with realignment parameters
   ├─> High-frequency content
   ├─> CSF fraction (edge component)
   └─> Edge fraction (edge component)

3. Classification
   └─> Motion vs. non-motion using feature thresholds

4. Denoising
   ├─> Non-aggressive: Regress out motion components
   └─> Aggressive: Remove motion components entirely
```

### Classification Features

```python
# Four features used for classification:

# 1. Motion correlation (MC)
# - Max correlation of component timecourse with motion parameters
# - Threshold: > 0.3 suggests motion artifact

# 2. High-frequency content (HFC)
# - Ratio of high-frequency (>0.35 Hz) to total power
# - Threshold: > 0.35 suggests motion artifact

# 3. CSF fraction
# - Overlap with CSF segmentation
# - Threshold: > 0.10 suggests edge artifact

# 4. Edge fraction
# - Overlap with brain edge
# - Threshold: > 0.10 suggests edge artifact

# Component classified as motion if:
#   MC > 0.3 AND HFC > 0.35
#   OR CSF > 0.10
#   OR Edge > 0.10
```

## Output Files

### ICA-AROMA Output Structure

```bash
output/ICA-AROMA/
├── denoised_func_data_nonaggr.nii.gz  # Non-aggressively denoised data
├── denoised_func_data_aggr.nii.gz     # Aggressively denoised data (if -den aggr)
├── melodic.ica/                        # Melodic ICA directory
│   ├── melodic_IC.nii.gz              # ICA component spatial maps
│   ├── melodic_mix                     # ICA component timecourses
│   └── melodic_FTmix                   # Frequency content
├── classified_motion_ICs.txt           # List of motion component IDs
├── feature_scores.csv                  # Feature values for each component
└── classification_overview.txt         # Summary statistics
```

### Interpreting Results

```bash
# View motion components
cat output/ICA-AROMA/classified_motion_ICs.txt
# Output: 3,7,12,18,25
# These component numbers are classified as motion

# Check feature scores
cat output/ICA-AROMA/feature_scores.csv
# Shows MC, HFC, CSF, and Edge scores for each component

# Classification summary
cat output/ICA-AROMA/classification_overview.txt
# Shows total components and number classified as motion
```

## Quality Control

### Visual Inspection

```bash
# View components in FSLview/FSLeyes
fsleyes \
  output/ICA-AROMA/melodic.ica/melodic_IC.nii.gz \
  -cm red-yellow -dr 3 10

# Check component timecourses
fsleyes \
  output/ICA-AROMA/melodic.ica/melodic_mix

# Motion components typically show:
# - Spatial maps at brain edges or ventricles
# - Timecourses correlated with motion
# - High-frequency oscillations
```

### Compare Before/After

```bash
# Compare original vs denoised
fsleyes \
  sub-01_bold.nii.gz \
  output/ICA-AROMA/denoised_func_data_nonaggr.nii.gz

# Calculate difference
fslmaths sub-01_bold.nii.gz \
  -sub output/ICA-AROMA/denoised_func_data_nonaggr.nii.gz \
  removed_signal.nii.gz

# Visualize what was removed
fsleyes removed_signal.nii.gz
```

### Temporal SNR

```bash
# Calculate tSNR before and after
python << EOF
import nibabel as nib
import numpy as np

# Load data
original = nib.load('sub-01_bold.nii.gz').get_fdata()
denoised = nib.load('output/ICA-AROMA/denoised_func_data_nonaggr.nii.gz').get_fdata()

# Calculate tSNR
tsnr_orig = np.mean(original, axis=3) / np.std(original, axis=3)
tsnr_denoised = np.mean(denoised, axis=3) / np.std(denoised, axis=3)

# Mean tSNR in brain
mask = nib.load('brain_mask.nii.gz').get_fdata()
print(f'Original tSNR: {np.mean(tsnr_orig[mask > 0]):.2f}')
print(f'Denoised tSNR: {np.mean(tsnr_denoised[mask > 0]):.2f}')
EOF
```

## Integration with Pipelines

### With fMRIPrep

```bash
# Run fMRIPrep with ICA-AROMA
fmriprep /data/bids_dataset /output participant \
  --use-aroma \
  --output-spaces MNI152NLin2009cAsym \
  --mem-mb 16000 \
  --nthreads 8

# Outputs include:
# *_desc-smoothAROMAnonaggr_bold.nii.gz
# *_AROMAnoiseICs.csv (motion component IDs)
```

### With FSL FEAT

```bash
# Use AROMA-denoised data in FEAT
# 1. Run ICA-AROMA preprocessing
# 2. Use denoised_func_data_nonaggr.nii.gz as input to FEAT
# 3. Skip motion correction in FEAT (already done)

# Or integrate AROMA into FEAT design
# feat design.fsf
# Set filtered_func_data to AROMA output
```

### With xcpEngine

```bash
# xcpEngine can use AROMA components
# In design file:
confound2_aroma[sub-01]=1

# Or use fMRIPrep AROMA outputs directly
# xcpEngine will read *_AROMAnoiseICs.csv
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# Batch ICA-AROMA processing

subjects=(sub-01 sub-02 sub-03 sub-04 sub-05)

for subj in "${subjects[@]}"; do
    echo "Processing ${subj}..."

    # Input files
    func_file="/data/${subj}/func/${subj}_task-rest_bold.nii.gz"
    mc_file="/data/${subj}/func/${subj}_motion.txt"
    mask_file="/data/${subj}/func/${subj}_brain_mask.nii.gz"
    output_dir="/output/${subj}/ICA-AROMA"

    # Run ICA-AROMA
    python -m ICA_AROMA.ICA_AROMA \
      -in ${func_file} \
      -out ${output_dir} \
      -mc ${mc_file} \
      -m ${mask_file} \
      -den nonaggr \
      -tr 2.0

    # Check results
    n_components=$(wc -l < ${output_dir}/classified_motion_ICs.txt)
    echo "${subj}: ${n_components} motion components identified"

    echo "${subj} complete"
done
```

### Parallel Processing

```bash
# GNU Parallel
parallel -j 4 \
  'python -m ICA_AROMA.ICA_AROMA \
    -in {}/func/bold.nii.gz \
    -out {}/ICA-AROMA \
    -mc {}/func/motion.txt \
    -den nonaggr' \
  ::: sub-*/

# Or with job scheduler
for subj in sub-*; do
    sbatch run_aroma.sh ${subj}
done
```

## Downstream Analysis

### Use Denoised Data

```bash
# Non-aggressive denoising recommended for most analyses
denoised_file="output/ICA-AROMA/denoised_func_data_nonaggr.nii.gz"

# Apply temporal filtering
fslmaths ${denoised_file} \
  -bptf 25 250 \
  denoised_filtered.nii.gz
# -bptf: band-pass temporal filter (sigma for high-pass, low-pass)

# Extract ROI timeseries
fslmeants -i ${denoised_file} \
  -o roi_timeseries.txt \
  -m roi_mask.nii.gz
```

### Connectivity Analysis

```bash
# After AROMA denoising, compute connectivity
python << EOF
import nibabel as nib
import numpy as np

# Load denoised data
img = nib.load('output/ICA-AROMA/denoised_func_data_nonaggr.nii.gz')
data = img.get_fdata()

# Load atlas
atlas = nib.load('atlas.nii.gz').get_fdata()
n_rois = int(atlas.max())

# Extract ROI timeseries
timeseries = np.zeros((data.shape[3], n_rois))
for roi in range(1, n_rois + 1):
    mask = atlas == roi
    timeseries[:, roi-1] = data[mask].mean(axis=0)

# Compute correlation
from scipy.stats import pearsonr
correlation_matrix = np.corrcoef(timeseries.T)

# Save
np.savetxt('connectivity_matrix.txt', correlation_matrix)
EOF
```

## Best Practices

### Recommendations

```bash
# 1. Preprocessing before AROMA
# - Motion correction: Required
# - Smoothing: 6-8mm FWHM recommended
# - Spatial normalization: Recommended (MNI space)
# - Temporal filtering: NO (AROMA expects unfiltered)

# 2. Denoising strategy
# - Use non-aggressive for most analyses
# - Aggressive only for highly motion-corrupted data
# - Consider additional confound regression after AROMA

# 3. Quality control
# - Always inspect classified components
# - Check number of motion components (typically 10-30%)
# - Compare before/after tSNR
# - Verify edge/CSF components removed

# 4. Documentation
# - Report number of components identified
# - Report percentage of variance removed
# - Cite ICA-AROMA paper
```

## Integration with Claude Code

When helping users with ICA-AROMA:

1. **Check Installation:**
   ```bash
   python -m ICA_AROMA.ICA_AROMA --version
   which fsl
   ```

2. **Common Issues:**
   - FSL not in PATH
   - Motion parameters wrong format (must be 6 columns)
   - Data already temporally filtered
   - Missing brain mask
   - Insufficient timepoints (< 100 volumes)

3. **Best Practices:**
   - Use with fMRIPrep (easiest integration)
   - Non-aggressive denoising preferred
   - Always inspect component classifications
   - Compare metrics before/after
   - Document number of motion components
   - Consider combining with CompCor
   - Keep original data for comparison

4. **Quality Checks:**
   - Motion components: 10-40% typical
   - Visual inspection of spatial maps
   - Check timecourse correlations
   - Verify tSNR improvement
   - Ensure edge components removed

## Troubleshooting

**Problem:** "FSL not found"
**Solution:** Install FSL, add to PATH: `export FSLDIR=/usr/local/fsl; source ${FSLDIR}/etc/fslconf/fsl.sh`

**Problem:** Few or no motion components identified
**Solution:** Check motion parameters format, verify preprocessing quality, consider data may be clean

**Problem:** Too many components classified as motion
**Solution:** May indicate poor data quality, check preprocessing, inspect component visually

**Problem:** ICA fails or produces empty results
**Solution:** Check input data dimensions, verify sufficient timepoints (>100), ensure mask is valid

**Problem:** Denoised data looks wrong
**Solution:** Compare aggressive vs non-aggressive, visually inspect removed components, check input quality

## Resources

- GitHub: https://github.com/maartenmennes/ICA-AROMA
- Paper: Pruim et al. (2015) NeuroImage
- fMRIPrep docs: https://fmriprep.org/en/stable/workflows.html#ica-aroma
- FSL Wiki: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
- Tutorial: https://github.com/maartenmennes/ICA-AROMA/wiki

## Citation

```bibtex
@article{pruim2015ica,
  title={ICA-AROMA: A robust ICA-based strategy for removing motion artifacts from fMRI data},
  author={Pruim, Raimon HR and Mennes, Maarten and van Rooij, Daan and Llera, Alberto and Buitelaar, Jan K and Beckmann, Christian F},
  journal={Neuroimage},
  volume={112},
  pages={267--277},
  year={2015},
  publisher={Elsevier}
}
```

## Related Tools

- **fMRIPrep:** Includes ICA-AROMA integration
- **FIX (FSL):** FSL's ICA-based denoising
- **xcpEngine:** Can use AROMA outputs
- **FSL MELODIC:** Underlying ICA method
- **CompCor:** Alternative artifact removal
- **CONN:** Can integrate AROMA denoising
