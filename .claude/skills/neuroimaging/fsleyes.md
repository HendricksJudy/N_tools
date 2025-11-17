# FSLeyes - FSL Image Viewer

## Overview

FSLeyes (FSL-Eyes) is the modern, feature-rich neuroimaging data visualization tool from FSL. It replaces the older FSLView and provides an intuitive interface for viewing, exploring, and overlaying 3D and 4D medical imaging data. FSLeyes supports multiple image formats, offers advanced rendering options, includes lightbox and 3D viewing modes, and integrates seamlessly with FSL analysis tools.

**Website:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes
**Platform:** Linux/macOS/Windows
**Language:** Python (wxPython GUI)
**License:** Apache 2.0

## Key Features

- View 2D, 3D, and 4D neuroimaging data
- Multiple overlay support with transparency control
- Orthographic and lightbox display modes
- 3D rendering and surface visualization
- Time series plotting for fMRI data
- Histogram and intensity profiling
- ROI drawing and editing tools
- Integration with FSL tools (FEAT, MELODIC, TBSS)
- Mesh and surface support (FreeSurfer, GIFTI)
- Customizable color maps and lookup tables
- Screenshot and movie generation
- Command-line interface for scripting
- Plugin system for extensibility

## Installation

### Via FSL Installation

```bash
# FSLeyes is included with FSL 6.0+
# If you have FSL installed:
fsleyes

# Update FSLeyes
conda activate fslpython
pip install --upgrade fsleyes
```

### Standalone Installation

```bash
# Using conda (recommended)
conda create -n fsleyes python=3.9
conda activate fsleyes
conda install -c conda-forge fsleyes

# Using pip
pip install fsleyes

# On Ubuntu/Debian
sudo apt-get install fsleyes

# Verify installation
fsleyes --version
```

### Install with Additional Features

```bash
# Install with all optional dependencies
pip install fsleyes[extras]

# Development version from GitHub
pip install git+https://github.com/pauldmccarthy/fsleyes.git
```

## Basic Usage

### Launch FSLeyes

```bash
# Launch GUI
fsleyes

# Load image on startup
fsleyes brain.nii.gz

# Load multiple overlays
fsleyes T1.nii.gz overlay.nii.gz stat.nii.gz

# Load with specific display settings
fsleyes T1.nii.gz -cm greyscale -dr 0 100

# Open FEAT results
fsleyes feat_dir.feat
```

### Command-Line Options

```bash
# Display help
fsleyes --help

# Verbose output
fsleyes --verbose

# Specify initial scene
fsleyes brain.nii.gz -vl 45 30 25  # Set view location

# Set display range
fsleyes brain.nii.gz -dr 0 1000

# Set colormap
fsleyes brain.nii.gz -cm hot

# Load with specific overlay type
fsleyes brain.nii.gz stat.nii.gz -ot volume
```

## Viewing Images

### Basic Navigation

```bash
# Keyboard shortcuts:
# Arrow keys: Move through slices
# Shift + arrows: Move faster
# Ctrl + arrows: Change view orientation
# Mouse wheel: Zoom in/out
# Middle click + drag: Pan
# Right click: Context menu

# View modes:
# Ortho: Standard orthographic view (sagittal, coronal, axial)
# Lightbox: Multiple slices in grid
# 3D: Volume rendering
# Scene: Custom layout
```

### Loading and Overlaying Images

```bash
# Load structural image
fsleyes T1.nii.gz

# Add functional overlay
fsleyes T1.nii.gz filtered_func_data.nii.gz

# Add statistical map
fsleyes T1.nii.gz zstat1.nii.gz -cm red-yellow -dr 2.3 8

# Add mask overlay
fsleyes T1.nii.gz brain_mask.nii.gz -a 50 -cm blue

# Options:
# -cm: colormap
# -dr: display range (min max)
# -a: alpha/transparency (0-100)
# -ot: overlay type
```

## Display Modes

### Orthographic View

```bash
# Standard ortho view
fsleyes brain.nii.gz

# Set initial location (voxel coordinates)
fsleyes brain.nii.gz -vl 45 54 36

# Set initial location (world coordinates)
fsleyes brain.nii.gz -wl 0 20 30

# Link cursor position across views
# (default behavior, can toggle in GUI)
```

### Lightbox Mode

```bash
# Switch to lightbox view
fsleyes brain.nii.gz -s lightbox

# Customize lightbox
fsleyes brain.nii.gz -s lightbox \
  -lz z \          # Slice through z-axis
  -ls 10 \         # Number of slices
  -lr 50 70 \      # Range (slice 50 to 70)
  -lh 100          # Highlight slice

# Create mosaic of all slices
fsleyes brain.nii.gz -s lightbox -lz z -ls 50
```

### 3D View

```bash
# Launch with 3D rendering
fsleyes brain.nii.gz -s 3d

# 3D with surface
fsleyes T1.nii.gz lh.pial.gii rh.pial.gii -s 3d

# Options in GUI:
# - Lighting: Adjust lighting angle and intensity
# - Clipping: Clip volumes to reveal internal structures
# - Quality: Adjust rendering quality
```

## Working with Statistical Maps

### Viewing FEAT Results

```bash
# Open FEAT directory
fsleyes design.feat

# Automatically loads:
# - Filtered func data (4D fMRI)
# - Registration images
# - Statistical maps (zstat, cope, etc.)
# - Cluster masks

# Manual loading
fsleyes \
  design.feat/filtered_func_data.nii.gz \
  design.feat/thresh_zstat1.nii.gz -cm red-yellow -dr 2.3 8
```

### Thresholding Statistical Maps

```bash
# Load with threshold
fsleyes bg.nii.gz zstat1.nii.gz -dr 2.3 8 -cm red-yellow

# Cluster thresholding in GUI:
# 1. Select overlay
# 2. Right-click → "Overlay information"
# 3. Set "Min" threshold value
# 4. Choose colormap (red-yellow, hot, cool, etc.)

# Show only positive values
fsleyes T1.nii.gz zstat.nii.gz -dr 2.3 10 -nc blue-lightblue -pc red-yellow

# Options:
# -nc: negative colormap
# -pc: positive colormap
```

### FSL Cluster Output

```bash
# View cluster masks
fsleyes T1.nii.gz \
  cluster_mask_zstat1.nii.gz -cm random -a 50

# Random colormap assigns unique color to each cluster
# Useful for visualizing discrete regions
```

## Time Series and 4D Data

### Viewing fMRI Time Series

```bash
# Load 4D fMRI data
fsleyes filtered_func_data.nii.gz

# Time series controls:
# - Play/pause button
# - Slider to scrub through volumes
# - Movie mode for continuous playback

# Plot time series at cursor
# Right-click → "Plot time series"
# Shows intensity over time at selected voxel
```

### Time Series Plotting

```bash
# CLI: Create time series plot
fsleyes render \
  --scene ortho \
  --outfile timeseries.png \
  filtered_func_data.nii.gz \
  -w 1200 -h 800

# In GUI:
# Tools → Time series
# - Click voxels to add to plot
# - Multiple ROI plotting
# - Export data to CSV
```

## ROI Tools

### Drawing ROIs

```bash
# Create new mask/ROI
# In GUI:
# 1. File → New image → Mask
# 2. Tools → Edit mode
# 3. Draw on slices
# 4. File → Save

# Selection tools:
# - Pen: Freehand drawing
# - Fill: Fill connected region
# - Select: Select and fill
# - Erase: Remove voxels

# Keyboard shortcuts in edit mode:
# 1-9: Change brush size
# C: Clear selection
# F: Fill
# Shift+click: 3D fill
```

### Editing Masks

```bash
# Load and edit existing mask
fsleyes T1.nii.gz roi_mask.nii.gz

# Enable edit mode:
# Settings → Ortho View → Edit mode
# Or: Tools → Edit mode

# Operations:
# - Add: Paint voxels (value 1)
# - Erase: Remove voxels (value 0)
# - Select: Define regions
# - Fill: Flood fill region
# - Copy/paste across slices
```

## Surface Visualization

### FreeSurfer Surfaces

```bash
# Load FreeSurfer surfaces
fsleyes \
  T1.nii.gz \
  surf/lh.pial surf/rh.pial \
  -s 3d

# Load with overlay data
fsleyes \
  surf/lh.inflated \
  -d surf/lh.thickness \
  -cm hot

# Common FreeSurfer files:
# - lh/rh.pial: Pial surface
# - lh/rh.white: White matter surface
# - lh/rh.inflated: Inflated surface
# - lh/rh.sphere: Spherical surface
# - lh/rh.thickness: Cortical thickness overlay
# - lh/rh.curv: Curvature
```

### GIFTI Surfaces

```bash
# Load GIFTI surface
fsleyes surface.surf.gii

# With data overlay
fsleyes \
  surface.surf.gii \
  -d data.shape.gii \
  -cm hot -dr 0 5

# Multiple hemispheres
fsleyes lh.pial.gii rh.pial.gii -s 3d
```

## Screenshots and Movies

### Taking Screenshots

```bash
# Via command line
fsleyes render \
  --scene ortho \
  --outfile screenshot.png \
  --size 1200 800 \
  T1.nii.gz overlay.nii.gz -cm hot -dr 2 8

# Lightbox screenshot
fsleyes render \
  --scene lightbox \
  --outfile mosaic.png \
  --sliceSpacing 5 \
  --zaxis 2 \
  --numSlices 20 \
  brain.nii.gz

# High-resolution render
fsleyes render \
  --scene 3d \
  --outfile render_3d.png \
  --size 2400 2400 \
  brain.nii.gz
```

### Creating Movies

```bash
# Create movie through time (4D data)
fsleyes render \
  --scene ortho \
  --outfile movie \
  fmri_4d.nii.gz

# Creates individual frames: movie_0000.png, movie_0001.png, ...

# Convert to video with ffmpeg
ffmpeg -framerate 10 -i movie_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4

# Rotating 3D movie
fsleyes render \
  --scene 3d \
  --outfile frames/frame \
  --rotate 360 0 0 \
  --rotateSteps 72 \
  brain.nii.gz

# Creates 72 frames (5° rotation each)
```

## Python API

### Basic Scripting

```python
#!/usr/bin/env python
import fsleyes

# Load FSLeyes programmatically
from fsleyes.main import main
main(['T1.nii.gz', 'overlay.nii.gz'])
```

### Advanced Scripting

```python
# Access FSLeyes overlay list
from fsleyes.views.orthopanel import OrthoPanel
from fsl.data.image import Image

# Load image
img = Image('brain.nii.gz')

# Create overlay
overlay = overlayList.append(img)

# Set display properties
display = displayCtx.getDisplay(overlay)
display.brightness = 50
display.contrast = 50
display.alpha = 100

# Set overlay type specific properties
opts = displayCtx.getOpts(overlay)
opts.cmap = 'hot'
opts.displayRange = (2.3, 8)
```

## FSL Integration

### MELODIC Results

```bash
# View MELODIC ICA results
fsleyes melodic.ica

# Loads:
# - Background image
# - Component spatial maps
# - Time courses
# - Frequency spectra

# Navigate components with slider
# Identify artifacts vs. signal
```

### TBSS Results

```bash
# View TBSS skeleton and statistics
fsleyes \
  $FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz \
  all_FA_skeletonised.nii.gz -cm green \
  tbss_tfce_corrp_tstat1.nii.gz -cm red-yellow -dr 0.95 1
```

### Registration Quality Check

```bash
# Check registration
fsleyes \
  standard.nii.gz \
  example_func2standard.nii.gz -a 50

# Overlay with transparency
# Use fade slider to check alignment
```

## Advanced Features

### Custom Color Maps

```bash
# Use built-in colormaps
fsleyes brain.nii.gz -cm greyscale  # or hot, cool, red-yellow, blue-lightblue

# Create custom colormap (in GUI)
# Settings → Overlay → Color map → Custom
# Define colors at specific intensity values

# Load LUT (lookup table)
fsleyes brain.nii.gz -l custom_lut.txt

# LUT format:
# 0    0   0   0   Background
# 1  255   0   0   Region1
# 2    0 255   0   Region2
```

### Plugins

```bash
# Install plugins
pip install fsleyes-plugin-myplugin

# Load plugin
fsleyes --plugin myplugin

# Develop custom plugins
# See: https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/plugins/
```

## Integration with Claude Code

When helping users with FSLeyes:

1. **Check Installation:**
   ```bash
   fsleyes --version
   which fsleyes
   ```

2. **Common Issues:**
   - Display errors on remote systems (X11 forwarding)
   - OpenGL compatibility issues
   - Memory usage with large 4D data
   - Slow rendering on older hardware
   - Missing FSL environment variables

3. **Best Practices:**
   - Use appropriate display range for data type
   - Choose colormap based on data (sequential vs. diverging)
   - Save screenshots at high resolution
   - Use lightbox for quick slice review
   - Enable GPU acceleration when available
   - Use command-line rendering for batch operations
   - Save custom views/layouts for reproducibility

4. **Quick Tips:**
   - Right-click for context menus
   - Use keyboard shortcuts for navigation
   - Middle-click drag to pan
   - Mouse wheel to zoom
   - Create custom layouts: View → Layouts → Save
   - Export time series data for analysis

## Troubleshooting

**Problem:** FSLeyes won't start on remote system
**Solution:** Enable X11 forwarding: `ssh -X user@host`, or use VNC/RDP

**Problem:** "OpenGL version too low" error
**Solution:** Update graphics drivers, or use software rendering: `LIBGL_ALWAYS_SOFTWARE=1 fsleyes`

**Problem:** Slow performance with 4D data
**Solution:** Downsample data temporally, increase memory allocation, or view single volumes

**Problem:** Can't see overlays
**Solution:** Check display range, adjust transparency (alpha), verify overlay is enabled in list

**Problem:** Screenshots have wrong colors
**Solution:** Check colormap settings, verify display range, ensure proper overlay order

## Resources

- Website: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes
- Documentation: https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/
- GitHub: https://github.com/pauldmccarthy/fsleyes
- User Group: https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=FSL
- Tutorial: https://fsl.fmrib.ox.ac.uk/fslcourse/

## Citation

```bibtex
@misc{fsleyes,
  title={FSLeyes},
  author={McCarthy, Paul},
  year={2023},
  howpublished={\url{https://zenodo.org/record/8404024}},
  doi={10.5281/zenodo.8404024}
}
```

## Related Tools

- **FSLView:** Older FSL viewer (deprecated)
- **MRIcron/MRIcroGL:** Alternative lightweight viewer
- **ITK-SNAP:** Segmentation-focused viewer
- **3D Slicer:** Comprehensive medical imaging platform
- **AFNI viewer:** AFNI's built-in visualization
- **Mango:** Multi-platform image viewer
