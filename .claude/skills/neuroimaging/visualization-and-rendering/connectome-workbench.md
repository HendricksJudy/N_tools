# Connectome Workbench - HCP Surface Analysis and Visualization

## Overview

Connectome Workbench is an open-source visualization and analysis tool developed by the Human Connectome Project (HCP) for exploring neuroimaging data on cortical and subcortical surfaces. It provides specialized support for CIFTI files (Connectivity Informatics Technology Initiative), enabling integrated analysis of volumetric and surface data, multi-modal visualization, and advanced connectivity analyses across the cerebral cortex, subcortical structures, and cerebellum.

**Website:** https://www.humanconnectome.org/software/connectome-workbench
**Platform:** Windows/macOS/Linux
**Language:** C++ (Qt GUI)
**License:** GPLv2

## Key Features

- Native CIFTI (dense/parcellated) file support
- High-quality surface rendering and visualization
- Multi-modal data integration (fMRI, DTI, MEG, anatomy)
- Time series and connectivity visualization
- Scene creation for reproducible figures
- Parcellation and ROI analysis tools
- Geodesic distance calculations
- Volume and surface registration
- Command-line tools (wb_command) for batch processing
- Border and focus drawing tools
- Metric smoothing and resampling
- Integration with FreeSurfer, FSL, and HCP Pipelines

## Installation

### Download and Install

```bash
# Download from: https://www.humanconnectome.org/software/get-connectome-workbench

# Linux
unzip workbench-linux64-*.zip
cd workbench/bin_linux64
./wb_view  # GUI
./wb_command -version  # Command-line tools

# macOS
# Open .dmg and drag to Applications
/Applications/workbench/wb_view

# Windows
# Run installer
# Launch from Start Menu

# Add to PATH (Linux/macOS)
export PATH="/path/to/workbench/bin_linux64:$PATH"
export PATH="/path/to/workbench/bin_macosx64:$PATH"  # macOS

# Verify installation
wb_command -version
```

### Required Data Files

```bash
# Download HCP surface templates
# From: https://balsa.wustl.edu/

# Standard meshes:
# - 32k_fs_LR (standard resolution)
# - 164k_fs_LR (high resolution)
# - 59k_fs_LR (mid resolution)

# Typical file structure:
templates/
├── S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii
├── S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii
├── S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii
└── S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii
```

## File Formats

### CIFTI Files (.dtseries.nii, .dscalar.nii, .dlabel.nii)

```bash
# CIFTI = Connectivity Informatics Technology Initiative
# Combines cortical surface + subcortical volume data

# Common CIFTI types:
# .dtseries.nii - Dense time series (fMRI data)
# .dscalar.nii - Dense scalar (statistical maps, thickness)
# .dlabel.nii - Dense label (parcellations, ROIs)
# .dconn.nii - Dense connectivity (correlation matrices)
# .ptseries.nii - Parcellated time series
# .pscalar.nii - Parcellated scalar
# .pconn.nii - Parcellated connectivity

# View CIFTI file info
wb_command -file-information mydata.dscalar.nii

# Convert GIFTI surfaces to CIFTI
wb_command -cifti-create-dense-scalar \
  output.dscalar.nii \
  -left-metric left.func.gii \
  -roi-left left.roi.shape.gii \
  -right-metric right.func.gii \
  -roi-right right.roi.shape.gii
```

### GIFTI Surface Files (.surf.gii, .shape.gii, .func.gii)

```bash
# Surface geometry
*.surf.gii - Surface coordinates and triangles

# Surface data
*.shape.gii - Shape measurements (curvature, sulcal depth)
*.func.gii - Functional data (activity, connectivity)
*.label.gii - Label/ROI data

# View GIFTI info
wb_command -file-information surface.surf.gii
```

## GUI Usage (wb_view)

### Opening Files

```bash
# Launch wb_view
wb_view

# Load spec file (collection of related files)
# File → Open Spec File

# Or load individual files
wb_view mydata.dscalar.nii

# Load multiple modalities
wb_view timeseries.dtseries.nii overlay.dscalar.nii
```

### Navigation

```
Mouse controls:
- Left drag: Rotate surface
- Middle drag: Pan
- Scroll: Zoom
- Shift + left drag: Rotate clipping plane
- Ctrl + left drag: Select vertex/node

Keyboard shortcuts:
- Tab: Cycle through surfaces (left/right hemisphere)
- M: Toggle montage view
- V: Cycle view modes (lateral, medial, etc.)
- S: Take screenshot
- Spacebar: Play/pause time series
```

### View Modes

```bash
# Surface views:
# - Lateral
# - Medial
# - Ventral
# - Dorsal
# - Anterior
# - Posterior

# Special views:
# - Flat surface
# - Inflated
# - Very inflated
# - Sphere

# Multi-view layouts:
# - Montage (multiple views simultaneously)
# - All view
# - Grid
```

### Color Maps and Palettes

```
Display settings:
1. Select overlay in Overlay Toolbox
2. Choose palette (color scheme):
   - Gray_Interp (grayscale)
   - ROY-BIG-BL (blue-red)
   - videen_style
   - FSL (FSL colormaps)
   - JET256
   - Clear_Brain

3. Set thresholds:
   - Min/Max values
   - Positive/negative coloring
   - Show mapped/unmapped data

4. Opacity and blending
```

### Creating Scenes

```
Save reproducible views:
1. Set up desired view (surface orientation, overlays, colors)
2. Window → Create and Edit Scenes
3. Click "Add Scene"
4. Name and save scene
5. Load later: Select scene from list

Use cases:
- Publication figures
- Consistent views across subjects
- Quality control reviews
- Presentations
```

## Command-Line Tools (wb_command)

### File Information

```bash
# Display file metadata
wb_command -file-information input.dscalar.nii

# Show CIFTI structure
wb_command -cifti-export-dense-mapping input.dscalar.nii COLUMN -volume-all

# List available wb_command operations
wb_command -list-commands

# Help for specific command
wb_command -metric-smoothing -help
```

### Surface Smoothing

```bash
# Smooth metric data on surface
wb_command -metric-smoothing \
  surface.surf.gii \
  input.func.gii \
  5.0 \
  output_smoothed.func.gii \
  -fwhm

# Parameters:
# - surface: Surface geometry file
# - input: Data to smooth
# - 5.0: Smoothing kernel (mm FWHM)
# - output: Smoothed output
# - -fwhm: Interpret kernel as FWHM (vs sigma)

# Smooth CIFTI
wb_command -cifti-smoothing \
  input.dscalar.nii \
  5.0 \
  5.0 \
  COLUMN \
  output_smoothed.dscalar.nii \
  -left-surface left.midthickness.surf.gii \
  -right-surface right.midthickness.surf.gii

# 5.0 5.0 = surface FWHM, volume FWHM
```

### Surface Resampling

```bash
# Resample data between surface spaces
wb_command -metric-resample \
  input.32k_fs_LR.func.gii \
  source_sphere.surf.gii \
  target_sphere.surf.gii \
  ADAP_BARY_AREA \
  output.164k_fs_LR.func.gii

# Methods:
# ADAP_BARY_AREA - Adaptive barycentric (recommended)
# BARYCENTRIC - Simple barycentric
# NEAREST - Nearest neighbor (for labels)

# Resample CIFTI
wb_command -cifti-resample \
  input.32k.dscalar.nii \
  COLUMN \
  source_sphere.surf.gii \
  target_sphere.surf.gii \
  ADAP_BARY_AREA \
  output.164k.dscalar.nii \
  -left-spheres source_L.sphere.surf.gii target_L.sphere.surf.gii \
  -right-spheres source_R.sphere.surf.gii target_R.sphere.surf.gii
```

### ROI Analysis

```bash
# Create ROI from parcellation
wb_command -cifti-label-to-roi \
  parcellation.dlabel.nii \
  output_roi.dscalar.nii \
  -name "V1"

# Extract data within ROI
wb_command -cifti-roi-average \
  input.dtseries.nii \
  1.0 \
  roi.dscalar.nii \
  -out-text average_timeseries.txt

# Calculate mean within label
wb_command -cifti-label-statistics \
  input.dscalar.nii \
  parcellation.dlabel.nii \
  -column 0 \
  -text-out statistics.txt
```

### Parcellation Operations

```bash
# Apply parcellation to extract ROI time series
wb_command -cifti-parcellate \
  input.dtseries.nii \
  parcellation.dlabel.nii \
  COLUMN \
  output.ptseries.nii \
  -method MEAN

# Methods:
# MEAN - Average within parcel
# MEDIAN - Median value
# MAX - Maximum value
# MIN - Minimum value
# STDEV - Standard deviation

# Create connectivity matrix from parcellated data
wb_command -cifti-correlation \
  input.ptseries.nii \
  output.pconn.nii \
  -fisher-z
```

### Volume to Surface Mapping

```bash
# Map volume data to surface
wb_command -volume-to-surface-mapping \
  volume.nii.gz \
  surface.surf.gii \
  output.func.gii \
  -ribbon-constrained \
  white_surface.surf.gii \
  pial_surface.surf.gii

# Methods:
# -ribbon-constrained: Sample within cortical ribbon (best for fMRI)
# -trilinear: Simple trilinear interpolation
# -enclosing-voxel: Nearest voxel
# -cubic: Cubic interpolation
```

### Surface to Volume Mapping

```bash
# Create volume from surface data
wb_command -metric-to-volume-mapping \
  input.func.gii \
  surface.surf.gii \
  volume_template.nii.gz \
  output.nii.gz \
  -ribbon-constrained \
  white_surface.surf.gii \
  pial_surface.surf.gii
```

## HCP Pipeline Integration

### Working with HCP Data

```bash
# HCP data structure:
subject/
├── MNINonLinear/
│   ├── fsaverage_LR32k/
│   │   ├── subject.L.midthickness.32k_fs_LR.surf.gii
│   │   ├── subject.R.midthickness.32k_fs_LR.surf.gii
│   │   ├── subject.L.thickness.32k_fs_LR.shape.gii
│   │   └── subject.R.thickness.32k_fs_LR.shape.gii
│   ├── Results/
│   │   └── rfMRI_REST/
│   │       └── rfMRI_REST_Atlas.dtseries.nii
│   └── T1w.nii.gz

# Load HCP subject
wb_view \
  ${SUBJECT}/MNINonLinear/fsaverage_LR32k/${SUBJECT}.32k_fs_LR.wb.spec
```

### Converting FreeSurfer to Workbench

```bash
# FreeSurfer to GIFTI
mris_convert lh.pial lh.pial.surf.gii
mris_convert rh.pial rh.pial.surf.gii

# FreeSurfer overlay to GIFTI
mris_convert -c lh.thickness lh.pial lh.thickness.shape.gii

# Create CIFTI from FreeSurfer surfaces
wb_command -cifti-create-dense-scalar \
  output.dscalar.nii \
  -left-metric lh.thickness.shape.gii \
  -right-metric rh.thickness.shape.gii
```

## Connectivity Analysis

### Seed-Based Connectivity

```bash
# Extract seed time series
wb_command -cifti-roi-average \
  rest.dtseries.nii \
  1.0 \
  seed_roi.dscalar.nii \
  -out-text seed_timeseries.txt

# Calculate seed-based correlation
wb_command -cifti-correlation-gradient \
  rest.dtseries.nii \
  seed_roi.dscalar.nii \
  seed_connectivity.dscalar.nii
```

### Dense Connectome

```bash
# Calculate full correlation matrix
wb_command -cifti-correlation \
  rest.dtseries.nii \
  output.dconn.nii \
  -fisher-z \
  -mem-limit 8

# Dense connectome is large! (32k × 32k = 1 billion connections)
# Consider parcellation first

# Parcellate then correlate
wb_command -cifti-parcellate \
  rest.dtseries.nii \
  parcellation.dlabel.nii \
  COLUMN \
  parcellated.ptseries.nii

wb_command -cifti-correlation \
  parcellated.ptseries.nii \
  connectivity.pconn.nii \
  -fisher-z
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# Batch smooth and resample surfaces

subjects=(sub-01 sub-02 sub-03 sub-04)
template_dir=/templates/32k_fs_LR

for subj in "${subjects[@]}"; do
    echo "Processing ${subj}..."

    # Smooth functional data
    wb_command -metric-smoothing \
        ${subj}/surf/lh.midthickness.surf.gii \
        ${subj}/func/lh.${subj}.func.gii \
        5.0 \
        ${subj}/func/lh.${subj}_smooth5mm.func.gii \
        -fwhm

    wb_command -metric-smoothing \
        ${subj}/surf/rh.midthickness.surf.gii \
        ${subj}/func/rh.${subj}.func.gii \
        5.0 \
        ${subj}/func/rh.${subj}_smooth5mm.func.gii \
        -fwhm

    # Resample to standard mesh
    wb_command -metric-resample \
        ${subj}/func/lh.${subj}_smooth5mm.func.gii \
        ${subj}/surf/lh.sphere.reg.surf.gii \
        ${template_dir}/S1200.L.sphere.32k_fs_LR.surf.gii \
        ADAP_BARY_AREA \
        ${subj}/func/lh.${subj}_smooth5mm.32k.func.gii

    echo "${subj} complete"
done
```

### Create Group Average

```bash
# Merge individual CIFTI files
wb_command -cifti-merge \
  group_average.dscalar.nii \
  -cifti sub-01.dscalar.nii \
  -cifti sub-02.dscalar.nii \
  -cifti sub-03.dscalar.nii

# Calculate mean across subjects
wb_command -cifti-reduce \
  group_average.dscalar.nii \
  MEAN \
  group_mean.dscalar.nii
```

## Advanced Visualization

### Time Series Visualization

```bash
# Load time series in wb_view
wb_view rfMRI_REST_Atlas.dtseries.nii

# Enable time series chart:
# View → Show Time Series Chart
# Click on surface vertex to see time series plot

# Time course controls:
# - Play/pause animation
# - Scrub through volumes
# - Select time range
# - Export time series data
```

### Border and Focus Drawing

```
Create regions of interest:
1. Load surface and data
2. Borders → Create New Border
3. Draw border on surface (click to place points)
4. Close border
5. Convert border to ROI:
   wb_command -border-to-rois
```

### Scene Capture for Publications

```bash
# Create high-resolution scene images
# 1. Set up view in wb_view
# 2. Create scene
# 3. Export scene to image:

wb_command -show-scene \
  scenes.scene \
  1 \
  output_image.png \
  1920 1080 \
  -use-window-size

# Batch render all scenes
for scene_num in {1..5}; do
    wb_command -show-scene \
        my_scenes.scene \
        ${scene_num} \
        figure${scene_num}.png \
        2400 2400
done
```

## Integration with Claude Code

When helping users with Connectome Workbench:

1. **Check Installation:**
   ```bash
   wb_command -version
   which wb_view
   ```

2. **Common Issues:**
   - Missing surface files (need HCP templates)
   - CIFTI structure mismatches
   - Memory issues with dense connectomes
   - Incorrect resampling spheres
   - File permission errors on macOS

3. **Best Practices:**
   - Use 32k_fs_LR mesh for standard analyses
   - Always smooth on surface (not in volume first)
   - Use -fwhm flag for FWHM specification
   - Fisher z-transform correlations
   - Parcellate before dense connectivity
   - Save scenes for reproducibility
   - Use ADAP_BARY_AREA for resampling metrics
   - Document all wb_command parameters

4. **File Organization:**
   - Keep surface geometry separate from data
   - Use descriptive CIFTI filenames
   - Maintain hemisphere consistency (L/R)
   - Version control scenes and scripts
   - Archive raw before smoothing

## Troubleshooting

**Problem:** "Structure mismatch" error with CIFTI
**Solution:** Verify left/right hemisphere consistency, check surface vs. volume brainordinates match

**Problem:** Slow rendering with time series
**Solution:** Reduce time points loaded, disable time series chart when not needed, lower surface resolution

**Problem:** Can't see overlay data
**Solution:** Check palette settings, adjust thresholds, verify data range, ensure proper CIFTI structure

**Problem:** Resampling produces artifacts
**Solution:** Use correct sphere registrations, check for holes in surfaces, use ADAP_BARY_AREA method

**Problem:** wb_command out of memory
**Solution:** Use -mem-limit flag, reduce data resolution, parcellate before operations, close other applications

## Resources

- Website: https://www.humanconnectome.org/software/connectome-workbench
- Tutorial: https://www.humanconnectome.org/software/workbench-command
- BALSA Database: https://balsa.wustl.edu/
- HCP Pipelines: https://github.com/Washington-University/HCPpipelines
- Forum: https://www.mail-archive.com/hcp-users@humanconnectome.org/
- YouTube: HCP Connectome Workbench Tutorials

## Citation

```bibtex
@article{marcus2011informatics,
  title={Informatics and data mining tools and strategies for the human connectome project},
  author={Marcus, Daniel S and Harwell, John and Olsen, Timothy and Hodge, Michael and Glasser, Matthew F and Prior, Fred and Jenkinson, Mark and Laumann, Timothy and Curtiss, Sandra W and Van Essen, David C},
  journal={Frontiers in neuroinformatics},
  volume={5},
  pages={4},
  year={2011}
}

@article{glasser2013minimal,
  title={The minimal preprocessing pipelines for the Human Connectome Project},
  author={Glasser, Matthew F and Sotiropoulos, Stamatios N and Wilson, J Anthony and Coalson, Timothy S and Fischl, Bruce and Andersson, Jesper L and Xu, Junqian and Jbabdi, Saad and Webster, Matthew and Polimeni, Jonathan R and others},
  journal={Neuroimage},
  volume={80},
  pages={105--124},
  year={2013}
}
```

## Related Tools

- **FreeSurfer:** Cortical reconstruction (integrates with Workbench)
- **FSL:** fMRI preprocessing and analysis
- **HCP Pipelines:** Automated HCP-style processing
- **CIFTIFY:** Convert non-HCP data to CIFTI
- **MSM:** Multimodal surface matching
- **fMRIPrep:** Preprocessing pipeline with CIFTI output
