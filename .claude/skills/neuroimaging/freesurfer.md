# FreeSurfer

## Overview

FreeSurfer is a suite of tools for the analysis and visualization of structural and functional neuroimaging data from cross-sectional or longitudinal studies. It is particularly renowned for cortical surface reconstruction, subcortical segmentation, and cortical parcellation.

**Website:** https://surfer.nmr.mgh.harvard.edu/
**Platform:** Linux/macOS
**Language:** C, C++, Python, Shell
**License:** FreeSurfer Software License (free for research)

## Key Features

- Automated cortical surface reconstruction
- Subcortical segmentation
- Cortical parcellation and labeling
- Cortical thickness analysis
- Longitudinal processing stream
- Group analysis and statistics
- Surface-based registration
- Functional MRI analysis on surfaces
- Tractography and connectivity analysis
- Integration with other tools (FSL, AFNI, SPM)

## Installation

### Linux/macOS

```bash
# Download FreeSurfer from the website
# Extract and set environment variables

# Add to ~/.bashrc or ~/.zshrc
export FREESURFER_HOME=/usr/local/freesurfer/7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Set subjects directory
export SUBJECTS_DIR=/path/to/your/subjects
```

### Get License

```bash
# Request free license from:
# https://surfer.nmr.mgh.harvard.edu/registration.html
# Place license.txt in $FREESURFER_HOME
```

### Verify Installation

```bash
recon-all -version
mri_info --version
```

## Core Workflow: Cortical Reconstruction

### Complete Anatomical Processing

```bash
# Full recon-all pipeline (6-24 hours per subject)
recon-all -subject sub01 \
          -i /path/to/T1.nii.gz \
          -all \
          -parallel \
          -openmp 4

# Process multiple timepoints
recon-all -i t1_tp1.nii.gz -i t1_tp2.nii.gz \
          -subject sub01 \
          -all

# Longitudinal processing
# 1. Process each timepoint
recon-all -subject sub01_tp1 -i tp1.nii.gz -all
recon-all -subject sub01_tp2 -i tp2.nii.gz -all

# 2. Create within-subject template
recon-all -base sub01_template \
          -tp sub01_tp1 \
          -tp sub01_tp2 \
          -all

# 3. Process timepoints using template
recon-all -long sub01_tp1 sub01_template -all
recon-all -long sub01_tp2 sub01_template -all
```

### Processing Stages

```bash
# Run specific processing stages
recon-all -subject sub01 -autorecon1  # Skull strip, motion correct
recon-all -subject sub01 -autorecon2  # Surface reconstruction
recon-all -subject sub01 -autorecon3  # Surface refinement, parcellation

# Resume from failure
recon-all -subject sub01 -make all
```

## Analysis Commands

### Cortical Thickness

```bash
# Extract thickness values from a label
mris_anatomical_stats -f sub01.lh.thickness.txt \
                       -a lh.aparc.annot \
                       sub01 lh

# Get mean thickness for a specific ROI
mris_anatomical_stats -l lh.precentral.label \
                       sub01 lh
```

### Subcortical Volumes

```bash
# Extract subcortical segmentation statistics
asegstats2table --subjects sub01 sub02 sub03 \
                --meas volume \
                --tablefile aseg_stats.txt

# Specific structures
mri_segstats --seg $SUBJECTS_DIR/sub01/mri/aseg.mgz \
             --sum stats.txt \
             --id 17 # Left hippocampus
```

### Surface-based Analysis

```bash
# Resample thickness data to fsaverage
mris_preproc --target fsaverage \
             --hemi lh \
             --meas thickness \
             --out lh.thickness.mgh \
             --s sub01 --s sub02 --s sub03

# Smooth on surface
mris_fwhm --smooth-only --s fsaverage \
          --hemi lh \
          --i lh.thickness.mgh \
          --o lh.thickness.sm10.mgh \
          --fwhm 10

# GLM analysis
mri_glmfit --y lh.thickness.sm10.mgh \
           --fsgd group_design.fsgd \
           --C contrast.mtx \
           --surf fsaverage lh \
           --cortex \
           --glmdir lh.thickness.glmdir

# Cluster correction
mri_glmfit-sim --glmdir lh.thickness.glmdir \
               --cache 2.0 abs \
               --cwp 0.05
```

### ROI Analysis

```bash
# Create custom label/ROI
mri_cor2label --i $SUBJECTS_DIR/sub01/mri/orig.mgz \
              --id 1 \
              --surf sub01 lh \
              --l lh.custom_roi.label

# Extract values from label
mri_segstats --i thickness.mgh \
             --label lh.custom_roi.label \
             --sum roi_stats.txt
```

### Volume Registration

```bash
# Register to MNI305 space
mri_convert -at $SUBJECTS_DIR/sub01/mri/transforms/talairach.xfm \
            $SUBJECTS_DIR/sub01/mri/brain.mgz \
            brain_mni305.nii.gz

# Register functional to anatomical
bbregister --s sub01 \
           --mov func.nii.gz \
           --init-fsl \
           --reg func2anat.dat \
           --t2

# Apply registration
mri_vol2vol --mov func.nii.gz \
            --targ $SUBJECTS_DIR/sub01/mri/orig.mgz \
            --reg func2anat.dat \
            --o func_in_anat_space.nii.gz
```

### Surface Mapping

```bash
# Sample volume data onto surface
mri_vol2surf --mov func.nii.gz \
             --reg func2anat.dat \
             --hemi lh \
             --projfrac 0.5 \
             --o lh.func.mgh

# Resample to common space
mri_surf2surf --srcsubject sub01 \
              --trgsubject fsaverage \
              --hemi lh \
              --sval lh.func.mgh \
              --tval lh.func.fsaverage.mgh
```

## Visualization

```bash
# FreeView - modern viewer
freeview -v $SUBJECTS_DIR/sub01/mri/T1.mgz \
         -v $SUBJECTS_DIR/sub01/mri/aseg.mgz:colormap=lut \
         -f $SUBJECTS_DIR/sub01/surf/lh.pial:edgecolor=red \
         -f $SUBJECTS_DIR/sub01/surf/lh.white:edgecolor=yellow

# View surfaces with overlay
freeview -f $SUBJECTS_DIR/sub01/surf/lh.inflated:overlay=lh.thickness.mgh \
         -colorscale

# TkSurfer (legacy but useful)
tksurfer sub01 lh inflated -overlay lh.thickness.mgh

# Create publication images
tksurfer sub01 lh inflated -overlay lh.thickness.mgh \
         -tcl scripts/make_images.tcl
```

## Quality Control

```bash
# Check processing errors
less $SUBJECTS_DIR/sub01/scripts/recon-all.log

# Common QC checks:
# 1. Skull strip quality
freeview -v $SUBJECTS_DIR/sub01/mri/brainmask.mgz

# 2. White matter surface
freeview -v $SUBJECTS_DIR/sub01/mri/T1.mgz \
         -f $SUBJECTS_DIR/sub01/surf/lh.white \
         -f $SUBJECTS_DIR/sub01/surf/rh.white

# 3. Pial surface
freeview -v $SUBJECTS_DIR/sub01/mri/T1.mgz \
         -f $SUBJECTS_DIR/sub01/surf/lh.pial:edgecolor=red \
         -f $SUBJECTS_DIR/sub01/surf/rh.pial:edgecolor=red

# 4. Segmentation
freeview -v $SUBJECTS_DIR/sub01/mri/aseg.mgz:colormap=lut:opacity=0.4 \
         -v $SUBJECTS_DIR/sub01/mri/T1.mgz
```

## Manual Corrections

```bash
# Edit skull strip
freeview -v $SUBJECTS_DIR/sub01/mri/brainmask.mgz \
         -v $SUBJECTS_DIR/sub01/mri/T1.mgz

# Edit white matter
freeview -v $SUBJECTS_DIR/sub01/mri/wm.mgz \
         -v $SUBJECTS_DIR/sub01/mri/T1.mgz

# After editing, continue processing
recon-all -subject sub01 -autorecon2 -autorecon3
```

## Integration with Claude Code

When helping users with FreeSurfer:

1. **Check Environment:**
   ```bash
   echo $FREESURFER_HOME
   echo $SUBJECTS_DIR
   ```

2. **Processing Time:** Warn users that recon-all takes 6-24 hours

3. **File Organization:** FreeSurfer has specific directory structure in $SUBJECTS_DIR

4. **Common Issues:**
   - License file missing or incorrect
   - Insufficient disk space (need ~2GB per subject)
   - Poor skull stripping
   - Motion artifacts in input

5. **Parallelization:** Use `-parallel -openmp N` for faster processing

## Best Practices

- Always check input quality before processing
- Use high-resolution T1 images (1mm isotropic preferred)
- Monitor processing logs for errors
- Perform quality control at multiple stages
- Use longitudinal stream for follow-up studies
- Keep FreeSurfer version consistent within studies
- Document any manual edits

## Troubleshooting

**Problem:** "Could not open license file"
**Solution:** Copy license.txt to $FREESURFER_HOME

**Problem:** Poor skull stripping
**Solution:** Use `-gcut` flag or manually edit brainmask.mgz

**Problem:** Surface defects
**Solution:** Edit white matter segmentation (wm.mgz) or use control points

**Problem:** "No space left on device"
**Solution:** Each subject needs ~2GB, clean up or expand storage

## Resources

- FreeSurfer Wiki: https://surfer.nmr.mgh.harvard.edu/fswiki
- Tutorial: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial
- Mailing List: freesurfer@nmr.mgh.harvard.edu
- YouTube Course: https://www.youtube.com/c/freesurfersoftware

## Related Tools

- **FreeView:** Modern visualization tool
- **TkSurfer:** Surface visualization (legacy)
- **TkMedit:** Volume editor (legacy)
- **QDEC:** Query, Design, Estimate, Contrast
- **Tracula:** Automated tractography
