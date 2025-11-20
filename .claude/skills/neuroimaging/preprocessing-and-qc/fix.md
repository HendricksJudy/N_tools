# FIX - FMRIB's ICA-based X-noiseifier

## Overview

FIX (FMRIB's ICA-based X-noiseifier) is FSL's advanced automatic classification and removal of ICA components representing artifacts in fMRI data. Using machine learning trained on manually labeled datasets, FIX identifies noise components with high accuracy, enabling automated, robust denoising of both individual and group-level fMRI data. FIX is particularly powerful for multi-subject studies and is used extensively in the Human Connectome Project (HCP) pipelines.

**Website:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIX
**Platform:** Linux/macOS (part of FSL)
**Language:** MATLAB/Octave, R, Bash
**License:** Part of FSL

## Key Features

- Machine learning-based component classification
- Pre-trained classifiers for common acquisition protocols
- Custom classifier training on your own data
- Single-subject and multi-subject denoising
- Aggressive and non-aggressive cleanup options
- Integration with FSL MELODIC ICA
- Hierarchical fusion of classifiers
- Motion confound removal
- High-pass filtering options
- Works with task and resting-state fMRI
- HCP pipeline integration
- Group-PCA decomposition support

## Installation

### Part of FSL

```bash
# FIX is included with FSL 6.0+
# Install FSL from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

# Verify FSL installation
which fsl
echo $FSLDIR

# Check FIX is available
which fix

# FIX location
ls $FSLDIR/fix/
```

### Dependencies

```bash
# Required:
# - FSL (including MELODIC)
# - MATLAB or Octave
# - R with required packages

# Install R packages
R -e "install.packages(c('kernlab', 'ROCR', 'class', 'party', 'e1071', 'randomForest'))"

# Verify FIX setup
fix -h
```

## Prerequisites

### MELODIC ICA Decomposition

```bash
# FIX requires MELODIC ICA first
melodic -i filtered_func_data.nii.gz \
  -o filtered_func_data.ica \
  --dim=0 \
  --tr=2.0 \
  --nobet \
  --report \
  --mmthresh=0.5

# Output: filtered_func_data.ica/
# This contains ICA components FIX will classify
```

### Standard Space Data

```bash
# FIX works best with standard space data
# Use FEAT or manual registration to MNI152

# Example registration
flirt -in filtered_func_data.nii.gz \
  -ref $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
  -out filtered_func_data_to_standard.nii.gz \
  -omat func_to_standard.mat \
  -dof 12
```

## Using Pre-trained Classifiers

### List Available Classifiers

```bash
# Check pre-trained classifiers
ls $FSLDIR/fix/training_files/

# Common classifiers:
# - Standard.RData: General purpose
# - HCP_hp2000.RData: HCP-style data
# - UKBiobank.RData: UK Biobank protocol
# - WhII_MB6.RData: Multi-band EPI

# View classifier info
fix -t $FSLDIR/fix/training_files/Standard.RData
```

### Run FIX with Pre-trained Classifier

```bash
# Basic FIX classification
fix filtered_func_data.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20

# Parameters:
#   filtered_func_data.ica: MELODIC output directory
#   Standard.RData: Trained classifier
#   20: Threshold (typically 10-20, lower = more aggressive)

# Output:
# - fix4melview_Standard_thr20.txt: Component classifications
# - Classified as signal [1-100] or noise [0-100] probability
```

### Apply FIX Cleanup

```bash
# Clean data using classifications
fix -a filtered_func_data.ica/fix4melview_Standard_thr20.txt

# Or run classification and cleanup together
fix filtered_func_data.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20 \
  -m -h 100

# Options:
#   -m: Apply cleanup
#   -h 100: High-pass filter cutoff (s), 100s = 0.01 Hz
#   -A: Aggressive cleanup (vs. non-aggressive default)
```

## Cleanup Methods

### Non-Aggressive vs Aggressive

```bash
# Non-aggressive (default, recommended)
# - Regresses out noise components from data
# - Preserves variance, better for connectivity
fix filtered_func_data.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20 -m -h 100

# Output: filtered_func_data_clean.nii.gz

# Aggressive cleanup
# - Removes noise components entirely
# - More conservative, may remove signal
fix filtered_func_data.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20 -m -h 100 -A

# Output: filtered_func_data_clean.nii.gz
```

### Motion Confound Removal

```bash
# Include motion parameter regression
fix filtered_func_data.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20 -m -h 100 -x mc

# Motion options:
#   -x mc: Add 6 motion parameters
#   -x mc24: Add 24 motion parameters (derivatives + squared)
```

## Training Custom Classifiers

### Manual Component Labeling

```bash
# 1. Run MELODIC on training data
for subj in sub-01 sub-02 sub-03; do
    melodic -i ${subj}/filtered_func_data.nii.gz \
      -o ${subj}/filtered_func_data.ica \
      --dim=0 --tr=2.0 --report
done

# 2. Manually label components
# Open in MELODIC report
firefox sub-01/filtered_func_data.ica/report.html

# Create hand_labels_noise.txt
# List noise component numbers (one per line)
cat > sub-01/filtered_func_data.ica/hand_labels_noise.txt << EOF
1
5
8
12
15
EOF

# Signal components are implicitly those NOT listed
```

### Train Classifier

```bash
# Collect training datasets
mkdir training_data
cd training_data

# Link or copy MELODIC directories with hand labels
ln -s /data/sub-01/filtered_func_data.ica ./
ln -s /data/sub-02/filtered_func_data.ica ./
ln -s /data/sub-03/filtered_func_data.ica ./
# ... (10-20 subjects recommended)

# Train FIX classifier
fix -t training_data -l MyCustom

# Output: training_data/MyCustom.RData

# Use custom classifier
fix test_subject.ica training_data/MyCustom.RData 20 -m -h 100
```

### Hierarchical Classifier

```bash
# Combine multiple classifiers for robustness
fix -t training_data1 training_data2 training_data3 \
  -l HierarchicalCustom

# Uses ensemble of classifiers
```

## Multi-Subject (Group) FIX

### Group-ICA with FIX

```bash
# 1. Create temporal concatenation
fslmerge -t group_concat.nii.gz sub-*_filtered_func_data.nii.gz

# 2. Run group MELODIC
melodic -i group_concat.nii.gz \
  -o group.ica \
  --dim=0 \
  --tr=2.0 \
  --report

# 3. Run FIX on group data
fix group.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20

# 4. Apply cleanup to individual subjects
fix -a group.ica subject_list.txt

# subject_list.txt format (one subject per line):
# sub-01/filtered_func_data.nii.gz
# sub-02/filtered_func_data.nii.gz
```

## Output and Quality Control

### FIX Output Files

```bash
# After running FIX:
filtered_func_data.ica/
├── filtered_func_data_clean.nii.gz  # Denoised data
├── fix4melview_Standard_thr20.txt   # Classifications
├── fix/
│   ├── features.csv                 # Component features
│   └── log.txt                      # FIX log
└── report.html                       # MELODIC report

# Classification file format:
# [Component_ID, Signal_Probability, Noise_Probability, Final_Classification]
# 1, 98.5, 1.5, Signal
# 2, 15.2, 84.8, Noise
```

### Visual Inspection

```bash
# View MELODIC components
fsleyes filtered_func_data.ica/melodic_IC.nii.gz

# Check classified noise components
cat filtered_func_data.ica/fix4melview_Standard_thr20.txt | grep Noise

# View classification probabilities
cut -d',' -f1,2,3,4 filtered_func_data.ica/fix/features.csv | head -20

# Compare before/after
fsleyes filtered_func_data.nii.gz filtered_func_data_clean.nii.gz
```

### Quality Metrics

```bash
# Calculate tSNR improvement
python << EOF
import nibabel as nib
import numpy as np

# Load data
original = nib.load('filtered_func_data.nii.gz').get_fdata()
cleaned = nib.load('filtered_func_data_clean.nii.gz').get_fdata()
mask = nib.load('mask.nii.gz').get_fdata()

# Calculate tSNR
tsnr_orig = np.mean(original, axis=3) / np.std(original, axis=3)
tsnr_clean = np.mean(cleaned, axis=3) / np.std(cleaned, axis=3)

# Mean in brain
print(f'Original tSNR: {np.mean(tsnr_orig[mask > 0]):.2f}')
print(f'Cleaned tSNR: {np.mean(tsnr_clean[mask > 0]):.2f}')
print(f'Improvement: {np.mean(tsnr_clean[mask > 0]) - np.mean(tsnr_orig[mask > 0]):.2f}')
EOF
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# Batch FIX processing

subjects=(sub-01 sub-02 sub-03 sub-04)
classifier=$FSLDIR/fix/training_files/Standard.RData
threshold=20

for subj in "${subjects[@]}"; do
    echo "Processing ${subj}..."

    ica_dir="${subj}/filtered_func_data.ica"

    # Check MELODIC exists
    if [ ! -d "${ica_dir}" ]; then
        echo "  Running MELODIC..."
        melodic -i ${subj}/filtered_func_data.nii.gz \
          -o ${ica_dir} \
          --dim=0 --tr=2.0 --report
    fi

    # Run FIX
    echo "  Running FIX classification..."
    fix ${ica_dir} ${classifier} ${threshold} -m -h 100

    # Check results
    n_noise=$(grep -c "Noise" ${ica_dir}/fix4melview_Standard_thr${threshold}.txt)
    n_total=$(wc -l < ${ica_dir}/melodic_IC.nii.gz | awk '{print $5}')
    echo "  ${subj}: ${n_noise} / ${n_total} components classified as noise"

    echo "${subj} complete"
done
```

### SLURM Job Array

```bash
#!/bin/bash
#SBATCH --job-name=fix
#SBATCH --array=1-50
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=2:00:00

# Get subject from list
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)

# Run FIX
fix ${SUBJECT}/filtered_func_data.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20 -m -h 100
```

## Integration with Pipelines

### With FEAT

```bash
# Run FEAT, then apply FIX
feat design.fsf

# FIX on FEAT output
fix design.feat/filtered_func_data.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20 -m -h 100

# Use cleaned data for higher-level analysis
```

### With HCP Pipelines

```bash
# HCP minimal preprocessing includes FIX option
# Uses HCP-trained classifier

# Manual HCP-style FIX
fix rfMRI_REST.ica \
  $FSLDIR/fix/training_files/HCP_hp2000.RData \
  20 -m -h 2000
# Note: hp2000 = 2000s high-pass = 0.0005 Hz
```

### With fMRIPrep

```bash
# After fMRIPrep, run MELODIC + FIX
# 1. MELODIC on fMRIPrep output
melodic -i sub-01_desc-preproc_bold.nii.gz \
  -o sub-01.ica \
  --dim=0 --tr=2.0 --report

# 2. FIX cleanup
fix sub-01.ica \
  $FSLDIR/fix/training_files/Standard.RData \
  20 -m -h 100
```

## Advanced Usage

### Feature Extraction Only

```bash
# Extract features without classification
# Useful for developing custom classifiers
fix -f filtered_func_data.ica

# Output: filtered_func_data.ica/fix/features.csv
# 90+ features per component:
# - Spatial features (edge, CSF)
# - Temporal features (frequency content)
# - Motion correlation
# - And many more
```

### Manual Override

```bash
# Manually specify components to remove
cat > manual_noise_list.txt << EOF
1
3
7
12
18
EOF

# Apply manual classification
fix -c filtered_func_data.ica manual_noise_list.txt -m -h 100
```

### Threshold Optimization

```bash
# Try multiple thresholds to find optimal
for thr in 5 10 15 20 25 30; do
    fix filtered_func_data.ica \
      $FSLDIR/fix/training_files/Standard.RData \
      ${thr} -m -h 100

    mv filtered_func_data_clean.nii.gz \
       filtered_func_data_clean_thr${thr}.nii.gz
done

# Compare tSNR or connectivity for each threshold
```

## Best Practices

### Recommendations

```bash
# 1. Preprocessing before FIX
# - Motion correction: Required
# - Registration to standard: Recommended
# - Smoothing: After FIX (preserves component independence)
# - Temporal filtering: Done by FIX

# 2. Classifier selection
# - Use pre-trained if matches your protocol
# - Train custom for novel acquisition
# - Use 10-20 subjects for training

# 3. Threshold selection
# - 20 is typical starting point
# - Lower (10-15) for more aggressive
# - Higher (25-30) for conservative

# 4. Cleanup method
# - Non-aggressive for connectivity studies
# - Aggressive for task activation (more conservative)

# 5. Quality control
# - Always inspect classifications
# - Check before/after tSNR
# - Verify edge/motion components removed
# - Compare connectivity metrics
```

## Integration with Claude Code

When helping users with FIX:

1. **Check Installation:**
   ```bash
   which fix
   fix -h
   echo $FSLDIR
   ```

2. **Common Issues:**
   - R packages not installed
   - MATLAB/Octave not found
   - MELODIC not run first
   - Wrong FSL version (need 6.0+)
   - Insufficient training data for custom classifier

3. **Best Practices:**
   - Run MELODIC with automatic dimensionality
   - Use appropriate pre-trained classifier
   - Non-aggressive cleanup for most cases
   - Visual inspection of classifications
   - Document threshold used
   - Keep original data
   - Train custom classifier for novel protocols

4. **Quality Checks:**
   - Noise components: 20-50% typical
   - Visual inspection of spatial maps
   - Check motion-correlated removed
   - Verify tSNR improvement
   - Compare connectivity before/after

## Troubleshooting

**Problem:** "R package not found"
**Solution:** Install required R packages: `R -e "install.packages(c('kernlab', 'ROCR', 'class'))"`

**Problem:** FIX classification seems wrong
**Solution:** Check MELODIC quality, try different threshold, consider training custom classifier

**Problem:** All or no components classified as noise
**Solution:** Verify classifier appropriate for data, check MELODIC dimensionality, inspect features.csv

**Problem:** Cleaned data looks worse
**Solution:** Try non-aggressive if using aggressive, check threshold, verify preprocessing quality

**Problem:** Training fails
**Solution:** Need 10+ subjects, check hand_labels_noise.txt format, verify MELODIC outputs consistent

## Resources

- FSL Wiki: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIX
- User Guide: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIX/UserGuide
- HCP FIX: https://github.com/Washington-University/HCPpipelines
- FSL Course: https://fsl.fmrib.ox.ac.uk/fslcourse/
- Mailing List: https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=FSL

## Citation

```bibtex
@article{griffanti2014ica,
  title={ICA-based artefact removal and accelerated fMRI acquisition for improved resting state network imaging},
  author={Griffanti, Ludovica and Salimi-Khorshidi, Gholamreza and Beckmann, Christian F and Auerbach, Edward J and Douaud, Gwena{\"e}lle and Sexton, Claire E and Zsoldos, Enikő and Ebmeier, Klaus P and Filippini, Nicola and Mackay, Clare E and others},
  journal={Neuroimage},
  volume={95},
  pages={232--247},
  year={2014}
}

@article{salimi2014automatic,
  title={Automatic denoising of functional MRI data: combining independent component analysis and hierarchical fusion of classifiers},
  author={Salimi-Khorshidi, Gholamreza and Douaud, Gwena{\"e}lle and Beckmann, Christian F and Glasser, Matthew F and Griffanti, Ludovica and Smith, Stephen M},
  journal={Neuroimage},
  volume={90},
  pages={449--468},
  year={2014}
}
```

## Related Tools

- **FSL MELODIC:** ICA decomposition (prerequisite)
- **ICA-AROMA:** Python-based alternative
- **fMRIPrep:** Minimal preprocessing pipeline
- **HCP Pipelines:** Uses FIX extensively
- **FSL FEAT:** First-level analysis with FIX integration
- **xcpEngine:** Postprocessing with artifact removal
