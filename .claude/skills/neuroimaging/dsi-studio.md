# DSI Studio - Diffusion MRI Analysis and Tractography

## Overview

DSI Studio is a comprehensive, cross-platform tool for analyzing diffusion MRI data, performing fiber tracking (tractography), and conducting connectivity analysis. It supports multiple reconstruction methods including DTI, DSI, GQI, QSDR, and provides advanced features for automatic tract recognition, connectometry analysis, and quantitative anisotropy mapping. With its intuitive GUI and extensive visualization capabilities, DSI Studio is widely used for both clinical and research applications.

**Website:** https://dsi-studio.labsolver.org/
**Platform:** Windows/macOS/Linux
**Language:** C++/Qt
**License:** Free for academic use

## Key Features

- Multiple reconstruction methods (DTI, DSI, GQI, QSDR)
- Deterministic and probabilistic tractography
- Automatic fiber tract recognition (>80 tracts)
- Connectometry for group studies
- Differential tractography
- Native space and template space analysis
- High-quality 3D visualization
- Quantitative anisotropy (QA) mapping
- Integration with structural MRI
- Region-based and ROI-based tracking
- Tract-specific analysis
- Cross-subject normalization (QSDR)
- Batch processing and automation
- Export to multiple formats

## Installation

### Download and Install

```bash
# Download from: https://dsi-studio.labsolver.org/

# Windows
# Run installer .exe

# macOS
# Open .dmg and drag to Applications
# May need to allow in Security & Privacy

# Linux
# Extract and run
tar -xzf dsi_studio_*.tar.gz
cd dsi_studio
./dsi_studio

# Add to PATH (optional)
export PATH="/path/to/dsi_studio:$PATH"

# Verify installation
dsi_studio --version

# Download atlas/templates
# Available in: Help → Download Templates
# Or from: https://brain.labsolver.org/hcp_template.html
```

## Data Preparation

### Required Input

```bash
# DSI Studio works with:
# 1. Raw DWI data (DICOM or NIfTI)
# 2. bval and bvec files (gradient information)
# 3. Optional: T1-weighted structural image

# Data structure example:
subject/
├── dwi.nii.gz         # 4D diffusion data
├── dwi.bval           # b-values
├── dwi.bvec           # gradient directions
└── t1.nii.gz          # T1-weighted image (optional)
```

### BIDS Compatibility

```bash
# BIDS format automatically recognized:
sub-01/
└── dwi/
    ├── sub-01_dwi.nii.gz
    ├── sub-01_dwi.bval
    ├── sub-01_dwi.bvec
    └── sub-01_dwi.json
```

## Basic Workflow

### 1. Create Source File (.src.gz)

```bash
# GUI: Step T1 - Open Source Images
# DICOM Import or Load 4D NIfTI

# Command line
dsi_studio --action=src \
  --source=dwi.nii.gz \
  --bval=dwi.bval \
  --bvec=dwi.bvec \
  --output=subject.src.gz

# Options:
#   --up_sampling=1    # Upsampling factor
#   --other_image=t1.nii.gz  # T1 for registration
```

### 2. Reconstruction (.fib.gz)

```bash
# GUI: Step T2 - Reconstruction
# Select method and parameters

# GQI reconstruction (recommended for most cases)
dsi_studio --action=rec \
  --source=subject.src.gz \
  --method=4 \
  --param0=1.25 \
  --output=subject.gqi.fib.gz

# Methods:
#   0 = DSI
#   1 = DTI
#   3 = QBI
#   4 = GQI (Generalized Q-sampling Imaging)
#   7 = QSDR (Q-space diffeomorphic reconstruction)

# Parameters:
#   --param0: Diffusion sampling length (typical: 1.25)
#   --mask: Brain mask
#   --align_acpc: Align to AC-PC
```

### 3. Fiber Tracking

```bash
# GUI: Step T3 - Fiber Tracking
# Open .fib.gz file and define tracking parameters

# Whole brain tracking
dsi_studio --action=trk \
  --source=subject.gqi.fib.gz \
  --method=0 \
  --seed_count=100000 \
  --fa_threshold=0.05 \
  --turning_angle=60 \
  --step_size=1.0 \
  --smoothing=0.5 \
  --min_length=30 \
  --max_length=300 \
  --output=subject.trk.gz

# Parameters:
#   --method: 0=streamline, 1=probabilistic, 2=RK4
#   --seed_count: Number of seeds
#   --fa_threshold: Anisotropy threshold (0.05-0.2)
#   --turning_angle: Max angle between steps (degrees)
#   --step_size: Step size (mm)
#   --min_length: Min tract length (mm)
#   --max_length: Max tract length (mm)
```

### 4. Fiber Tract Analysis

```bash
# GUI: Open .fib.gz in Step T3, load tracks
# Use ROI tools to select specific tracts

# Automatic tract recognition
dsi_studio --action=atk \
  --source=subject.gqi.fib.gz \
  --track=subject.trk.gz \
  --output=recognized_tracts

# Outputs individual tract files and statistics
```

## Reconstruction Methods

### DTI (Diffusion Tensor Imaging)

```bash
# Classic tensor model
dsi_studio --action=rec \
  --source=subject.src.gz \
  --method=1 \
  --output=subject.dti.fib.gz

# Generates:
# - FA (Fractional Anisotropy)
# - MD (Mean Diffusivity)
# - AD (Axial Diffusivity)
# - RD (Radial Diffusivity)
# - Principal eigenvector

# Good for: Single-shell data, quick processing
# Limitations: Cannot resolve crossing fibers
```

### GQI (Generalized Q-Sampling Imaging)

```bash
# Model-free reconstruction
dsi_studio --action=rec \
  --source=subject.src.gz \
  --method=4 \
  --param0=1.25 \
  --output=subject.gqi.fib.gz

# Generates:
# - QA (Quantitative Anisotropy)
# - Fiber ODFs (Orientation Distribution Functions)
# - ISO (Isotropic component)

# Good for: Multi-shell data, crossing fibers
# Default choice for modern acquisition
```

### QSDR (Template Space Reconstruction)

```bash
# Reconstruction in template space
dsi_studio --action=rec \
  --source=subject.src.gz \
  --method=7 \
  --param0=1.25 \
  --template=ICBM152 \
  --output=subject.qsdr.fib.gz

# Advantages:
# - Direct template space analysis
# - No post-hoc registration needed
# - Better cross-subject comparison

# Use for: Group studies, atlas-based analysis
```

## ROI-Based Tracking

### Define ROI Regions

```bash
# In GUI:
# 1. Open .fib.gz file
# 2. Regions → Load region
# 3. Or draw manually: Regions → New Region
# 4. Types:
#    - ROI (Seed): Starting points
#    - ROI2 (End): Target regions
#    - ROA (Avoid): Exclusion regions
#    - Terminative: Stop tracking here

# Track between two ROIs
dsi_studio --action=trk \
  --source=subject.gqi.fib.gz \
  --roi=motor_cortex.nii.gz \
  --roi2=spinal_cord.nii.gz \
  --output=corticospinal_tract.trk.gz
```

### Automatic Fiber Tract Recognition

```bash
# Recognize known tracts automatically
dsi_studio --action=atk \
  --source=subject.gqi.fib.gz \
  --track=whole_brain.trk.gz \
  --tolerance=22 \
  --output=auto_tracts

# Recognized tracts include:
# - Corpus Callosum
# - Corticospinal Tract
# - Arcuate Fasciculus
# - Cingulum
# - Superior Longitudinal Fasciculus
# - Inferior Longitudinal Fasciculus
# - Uncinate Fasciculus
# - And 70+ more

# Each tract saved separately with metrics
```

## Tract-Based Analysis

### Extract Tract Metrics

```bash
# GUI: Open tract, View → Statistics

# Command line: Export metrics
dsi_studio --action=ana \
  --source=subject.gqi.fib.gz \
  --track=tract.trk.gz \
  --export=stat

# Outputs:
# - Mean FA/QA along tract
# - Tract volume
# - Tract length
# - Shape metrics
# - Profile plots
```

### Tract Profile Analysis

```bash
# Along-tract analysis
dsi_studio --action=ana \
  --source=subject.gqi.fib.gz \
  --track=tract.trk.gz \
  --export=profile \
  --output=profile.txt

# Samples FA/QA at equidistant points along tract
# Useful for identifying regional differences
```

## Connectometry Analysis

### Group Comparison

```bash
# Create connectivity database
dsi_studio --action=cnt \
  --source=subjects_folder/*.fib.gz \
  --demo=demographics.txt \
  --output=connectivity.db.fz

# Demographics file format (tab-separated):
# subject_id  group  age  sex
# sub-01      0      25   1
# sub-02      1      30   0
# ...

# Run connectometry
dsi_studio --action=cnt \
  --source=connectivity.db.fz \
  --variable=group \
  --permutation=2000 \
  --threshold=0.6 \
  --output=connectometry_results
```

### Differential Tractography

```bash
# Compare baseline vs. follow-up
dsi_studio --action=trk \
  --source=baseline.fib.gz \
  --other_slices=followup.fib.gz \
  --threshold=20 \
  --output=differential_tracts.trk.gz

# Shows increased/decreased connectivity
# Useful for: Longitudinal studies, treatment response
```

## Visualization

### 3D Rendering

```
GUI visualization controls:

Mouse:
- Left drag: Rotate
- Right drag: Zoom
- Middle drag: Pan
- Wheel: Zoom

View options:
- Slice views: Axial, Coronal, Sagittal
- 3D view: Volume rendering
- Tract rendering: Line, tube, end point

Display settings:
- Color scheme: Direction, FA, tract-specific
- Transparency
- Tube diameter
- Lighting
```

### Export Visualizations

```bash
# Save screenshot
# File → Save → Screen (PNG, JPG)

# Export tract video
# [Tracts] → Save → Movie (rotate 360°)

# Export to other formats
# - TrackVis (.trk)
# - VTK (.vtk)
# - TRK.GZ (compressed)
# - ROI as NIfTI
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# Complete pipeline for multiple subjects

subjects=(sub-01 sub-02 sub-03 sub-04)

for subj in "${subjects[@]}"; do
    echo "Processing ${subj}..."

    # 1. Create source
    dsi_studio --action=src \
        --source=${subj}/dwi/${subj}_dwi.nii.gz \
        --bval=${subj}/dwi/${subj}_dwi.bval \
        --bvec=${subj}/dwi/${subj}_dwi.bvec \
        --output=${subj}/${subj}.src.gz

    # 2. Reconstruction
    dsi_studio --action=rec \
        --source=${subj}/${subj}.src.gz \
        --method=4 \
        --param0=1.25 \
        --output=${subj}/${subj}.gqi.fib.gz

    # 3. Whole brain tracking
    dsi_studio --action=trk \
        --source=${subj}/${subj}.gqi.fib.gz \
        --seed_count=1000000 \
        --fa_threshold=0.06 \
        --output=${subj}/${subj}_tracks.trk.gz

    # 4. Automatic tract recognition
    dsi_studio --action=atk \
        --source=${subj}/${subj}.gqi.fib.gz \
        --track=${subj}/${subj}_tracks.trk.gz \
        --output=${subj}/auto_tracts

    echo "${subj} complete"
done
```

## Quality Control

### Check Reconstruction Quality

```bash
# Visual QC in GUI:
# 1. Open .fib.gz file
# 2. Check slice views for artifacts
# 3. Verify fiber ODFs look reasonable
# 4. Check QA map (should be high in WM)

# Quantitative QC
dsi_studio --action=ana \
  --source=subject.gqi.fib.gz \
  --export=qa_map \
  --output=qa.nii.gz

# Mean QA in white matter should be > 0.05
```

### Validate Tracking Results

```bash
# Check tract counts and distributions
# Expected whole brain: 10k-500k streamlines

# GUI: Load tracts
# View → Statistics
# - Total count
# - Length distribution
# - FA/QA along tracts

# Remove spurious tracts
# [Tracts] → Delete → By Length (min/max)
# [Tracts] → Delete → By FA/QA threshold
```

## Integration with Claude Code

When helping users with DSI Studio:

1. **Check Installation:**
   ```bash
   dsi_studio --version
   which dsi_studio
   ```

2. **Common Issues:**
   - Missing bval/bvec files
   - Incorrect gradient orientation (check flip)
   - Memory errors (large datasets)
   - Template files not downloaded
   - Tracking parameters too restrictive

3. **Best Practices:**
   - Use GQI for most analyses (handles crossing fibers)
   - QA threshold: 0.05-0.06 (lower than FA threshold)
   - Seed count: 100k-1M for whole brain
   - Min length: 30mm (filter spurious tracts)
   - Always do visual QC of reconstructions
   - Use automatic tract recognition for consistency
   - Export metrics to CSV for statistical analysis
   - Save intermediate files (.src.gz, .fib.gz)

4. **Parameter Guidelines:**
   - **FA threshold:** 0.15-0.20 (DTI), 0.05-0.06 (GQI QA)
   - **Turning angle:** 35-60 degrees
   - **Step size:** 0.5-1.0 mm
   - **Smoothing:** 0-0.5
   - **Seed count:** Scale with brain size

## Troubleshooting

**Problem:** No tracts generated
**Solution:** Lower QA threshold, increase turning angle, check ROI placement

**Problem:** Too many spurious tracts
**Solution:** Increase QA threshold, reduce turning angle, filter by length

**Problem:** Reconstruction fails
**Solution:** Check gradient table (bval/bvec), verify data quality, try different method

**Problem:** Slow tracking
**Solution:** Reduce seed count, use streamline instead of probabilistic, smaller ROI

**Problem:** Cross-subject tracts don't align
**Solution:** Use QSDR reconstruction, or register .fib.gz files to template

## Resources

- Website: https://dsi-studio.labsolver.org/
- Documentation: https://dsi-studio.labsolver.org/doc/
- Forum: https://groups.google.com/g/dsi-studio
- YouTube: DSI Studio Tutorial Channel
- HCP Template: https://brain.labsolver.org/hcp_template.html
- Atlas Browser: https://brain.labsolver.org/

## Citation

```bibtex
@article{yeh2013generalized,
  title={Generalized $q$-sampling imaging},
  author={Yeh, Fang-Cheng and Wedeen, Van J and Tseng, Wen-Yih Isaac},
  journal={IEEE transactions on medical imaging},
  volume={29},
  number={9},
  pages={1626--1635},
  year={2010}
}

@article{yeh2018automatic,
  title={Automatic removal of false connections in diffusion MRI tractography using topology-informed pruning (TIP)},
  author={Yeh, Fang-Cheng and Panesar, Sandip and Barrios, Jesus and Fernandes, David and Abhinav, Kumar and Meola, Antonio and Fernandez-Miranda, Juan C},
  journal={Neurotherapeutics},
  volume={16},
  number={1},
  pages={52--58},
  year={2019}
}
```

## Related Tools

- **MRtrix3:** Advanced diffusion analysis
- **DIPY:** Python diffusion imaging
- **FSL (BEDPOSTX/ProbtrackX):** FSL tractography
- **TrackVis:** Tract visualization
- **TractSeg:** Automated tract segmentation
- **TORTOISE:** DTI preprocessing
- **Connectome Workbench:** Surface-based visualization
