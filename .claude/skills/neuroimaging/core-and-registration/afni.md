# AFNI (Analysis of Functional NeuroImages)

## Overview

AFNI is a comprehensive suite of programs for processing, analyzing, and displaying functional MRI data. Developed by the Scientific and Statistical Computing Core at NIMH, AFNI is known for its powerful visualization capabilities and extensive command-line tools.

**Website:** https://afni.nimh.nih.gov/
**Platform:** Linux/macOS
**Language:** C, Python, R, Shell
**License:** GNU GPL

## Key Features

- Comprehensive fMRI preprocessing and analysis
- Interactive 3D/4D visualization
- Volume and surface-based analysis
- Advanced statistical modeling
- Quality control (QC) reports
- Scripting and automation with `uber` scripts
- Real-time fMRI capabilities
- Group analysis and meta-analysis
- Extensive graphing and plotting tools

## Installation

### Linux (Recommended)

```bash
# Download and install
cd ~
curl -O https://afni.nimh.nih.gov/pub/dist/bin/misc/@update.afni.binaries
tcsh @update.afni.binaries -package linux_ubuntu_16_64 -do_extras

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=$PATH:~/abin
```

### macOS

```bash
# Install via the AFNI package
cd ~
curl -O https://afni.nimh.nih.gov/pub/dist/bin/misc/@update.afni.binaries
tcsh @update.afni.binaries -package macos_10.12_local -do_extras

export PATH=$PATH:~/abin
```

### Check Installation

```bash
afni -ver
afni_system_check.py -check_all
```

## Common Workflows

### Complete Preprocessing Pipeline

```bash
# Use afni_proc.py to generate preprocessing script
afni_proc.py \
    -subj_id sub01 \
    -dsets func_run*.nii.gz \
    -blocks despike tshift align tlrc volreg blur mask scale regress \
    -copy_anat anat.nii.gz \
    -anat_has_skull yes \
    -tcat_remove_first_trs 2 \
    -volreg_align_to MIN_OUTLIER \
    -volreg_align_e2a \
    -volreg_tlrc_warp \
    -blur_size 6.0 \
    -regress_stim_times stim*.1D \
    -regress_stim_labels task1 task2 \
    -regress_basis 'BLOCK(2,1)' \
    -regress_censor_motion 0.3 \
    -regress_motion_per_run \
    -regress_opts_3dD \
        -jobs 4 \
        -gltsym 'SYM: task1 -task2' -glt_label 1 'task1-task2' \
    -execute

# This creates proc.sub01 script which you can then run
tcsh -xef proc.sub01 |& tee output.proc.sub01
```

### Motion Correction

```bash
# 3dvolreg - volume registration (motion correction)
3dvolreg -verbose -zpad 1 -base 3 \
         -1Dfile motion.1D \
         -prefix func_volreg \
         input_func.nii.gz

# View motion parameters
1dplot motion.1D
```

### Skull Stripping

```bash
# 3dSkullStrip - brain extraction
3dSkullStrip -input anat.nii.gz \
             -prefix anat_brain.nii.gz \
             -push_to_edge

# Or use @SSwarper for simultaneous skull-strip and normalization
@SSwarper -input anat.nii.gz \
          -base MNI152_2009_template_SSW.nii.gz \
          -subid sub01 \
          -odir sub01_warp
```

### Spatial Normalization

```bash
# @auto_tlrc - Talairach transformation
@auto_tlrc -base TT_N27+tlrc \
           -input anat_brain.nii.gz \
           -no_ss

# Apply transformation to functional data
3dAllineate -base anat+tlrc \
            -input func.nii.gz \
            -prefix func_aligned.nii.gz \
            -1Dmatrix_apply func_to_anat.1D
```

### Smoothing

```bash
# 3dmerge - spatial smoothing
3dmerge -1blur_fwhm 6.0 -doall \
        -prefix func_smooth \
        input_func+orig

# Or use 3dBlurToFWHM for target FWHM
3dBlurToFWHM -input func.nii.gz \
             -prefix func_smooth \
             -FWHM 6.0 \
             -mask mask.nii.gz
```

### GLM Analysis

```bash
# 3dDeconvolve - general linear model
3dDeconvolve -input func_preproc.nii.gz \
             -polort A \
             -num_stimts 2 \
             -stim_times 1 task1.1D 'BLOCK(2,1)' \
             -stim_label 1 task1 \
             -stim_times 2 task2.1D 'BLOCK(2,1)' \
             -stim_label 2 task2 \
             -gltsym 'SYM: task1 -task2' \
             -glt_label 1 'contrast' \
             -fout -tout -x1D X.xmat.1D \
             -xjpeg X.jpg \
             -x1D_uncensored X.nocensor.xmat.1D \
             -fitts fitts \
             -errts errts \
             -bucket stats

# Estimate smoothness for multiple comparison correction
3dFWHMx -detrend -mask mask+orig errts+orig

# Multiple comparison correction
3dClustSim -athr 0.05 -pthr 0.001 \
           -mask mask+orig \
           -acf ACF_values
```

### Group Analysis

```bash
# 3dttest++ - group t-test
3dttest++ -setA subj*.nii.gz \
          -prefix group_ttest \
          -mask group_mask+tlrc

# ANOVA with 3dMVM (multivariate modeling)
3dMVM -prefix group_mvm \
      -jobs 4 \
      -mask group_mask+tlrc \
      -bsVars 'Group' \
      -wsVars 'Condition*Time' \
      -dataTable @data_table.txt
```

### Region of Interest Analysis

```bash
# Extract mean time series from ROI
3dmaskave -mask roi_mask+orig \
          -quiet func+orig > roi_timeseries.1D

# Calculate correlation between ROIs
3dROIstats -mask roi_atlas+tlrc \
           func+tlrc > roi_values.txt
```

## Visualization

```bash
# Launch AFNI GUI
afni &

# View overlay
afni -niml &
3dVol2Surf -spec surface_spec \
           -surf_A smoothwm \
           -sv anat+orig \
           -grid_parent stats+orig \
           -map_func ave \
           -out_niml stats_surf.niml.dset

# Generate images for publication
@chauffeur_afni -ulay anat+tlrc \
                -olay stats+tlrc \
                -prefix images/activation \
                -set_subbricks 0 1 2 \
                -pbar_posonly \
                -func_range 5
```

## Quality Control

```bash
# Generate QC HTML report
gen_ss_review_scripts.py \
    -mot_limit 0.3 \
    -out_limit 0.1 \
    -exit0

# View QC report in browser
@ss_review_html
```

## Integration with Claude Code

When helping users with AFNI:

1. **Check Setup:**
   ```bash
   afni_system_check.py -check_all
   ```

2. **File Formats:** AFNI uses BRIK/HEAD or NIfTI formats

3. **Dataset Notation:**
   - `+orig`: Original space
   - `+tlrc`: Talairach space
   - `+acpc`: AC-PC aligned

4. **Scripting:** AFNI excels at batch processing via shell scripts

5. **Common Issues:**
   - Environment not properly set
   - Missing R packages for some programs
   - Coordinate system confusion

## Best Practices

- Use `afni_proc.py` for standardized preprocessing
- Generate QC reports for every analysis
- Keep detailed processing logs
- Use descriptive dataset names
- Leverage AFNI's extensive documentation
- Visually inspect results at each step
- Use `-help` for detailed command info

## Useful Commands

```bash
# Dataset information
3dinfo dataset+orig

# Convert to NIfTI
3dAFNItoNIFTI dataset+orig

# Calculator for image math
3dcalc -a input1+orig -b input2+orig \
       -expr 'a+b' \
       -prefix output

# Resample to different resolution
3dresample -dxyz 2.0 2.0 2.0 \
           -prefix resampled \
           -input input+orig

# Threshold and cluster
3dClusterize -inset stats+tlrc'[1]' \
             -idat stats+tlrc'[0]' \
             -ithr 1 \
             -NN 1 \
             -clust_nvox 20 \
             -pref_map cluster_map
```

## Troubleshooting

**Problem:** "Command not found"
**Solution:** Check PATH includes ~/abin directory

**Problem:** X11/Display errors
**Solution:** Ensure X11 forwarding is enabled, or use `-noplugins` flag

**Problem:** R errors in group analysis
**Solution:** Install required R packages: `rPkgsInstall -pkgs ALL`

## Resources

- AFNI Bootcamp Materials: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/educational/bootcamp.html
- Message Board: https://discuss.afni.nimh.nih.gov/
- YouTube Tutorials: https://www.youtube.com/c/AFNIvideo
- AFNI Academy: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/index.html
- Handbook: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/handbook/main_toc.html

## Related Tools

- **SUMA:** Surface mapping and visualization
- **FATCAT:** Diffusion analysis tools
- **3dMEMA:** Meta-analysis tools
- **@SSwarper:** Skull-stripping and warping
- **InstaCorr:** Real-time correlation analysis

## Citation

```bibtex
@article{cox1996afni,
  title={AFNI: software for analysis and visualization of functional magnetic resonance neuroimages},
  author={Cox, Robert W.},
  journal={Computers and Biomedical Research},
  volume={29},
  number={3},
  pages={162--173},
  year={1996},
  doi={10.1006/cbmr.1996.0014}
}
```
