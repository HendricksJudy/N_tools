# MRIcron and MRIcroGL - Medical Image Viewers

## Overview

MRIcron and MRIcroGL are lightweight, fast medical image viewers developed by Chris Rorden. MRIcron is the classic cross-platform viewer with extensive format support, while MRIcroGL is the modern OpenGL-powered successor offering advanced 3D rendering, tractography visualization, and mesh display. Both tools are designed for quick data inspection, creating publication-quality figures, and basic image manipulation without the overhead of larger neuroimaging packages.

**Website:** https://www.nitrc.org/projects/mricron (MRIcron), https://www.nitrc.org/projects/mricrogl (MRIcroGL)
**Platform:** Windows/macOS/Linux
**Language:** Pascal (MRIcron), C/Pascal (MRIcroGL)
**License:** BSD

## Key Features

### MRIcron
- Fast, lightweight image viewer
- Extensive format support (NIfTI, DICOM, ANALYZE, etc.)
- ROI drawing and editing
- NPM statistical mapping
- dcm2nii DICOM converter (predecessor to dcm2niix)
- Cross-platform compatibility
- Batch processing scripts

### MRIcroGL
- GPU-accelerated rendering
- Advanced 3D visualization
- Tractography and fiber visualization (TrackVis .trk)
- Mesh display (FreeSurfer, GIFTI, OBJ, STL)
- Volume rendering with lighting effects
- Multi-planar reconstruction
- Python scripting interface
- Screenshot and movie generation
- Cross-section view with depth cues

## Installation

### MRIcron

```bash
# Linux
# Download from: https://www.nitrc.org/projects/mricron
# Extract and run
tar -xzf mricron_linux.tar.gz
cd mricron
./mricron

# macOS
# Download dmg installer
# Drag MRIcron to Applications

# Windows
# Download installer and run
# Or portable version
```

### MRIcroGL

```bash
# Linux
# Download from: https://github.com/rordenlab/MRIcroGL/releases
unzip MRIcroGL_linux.zip
cd MRIcroGL
./mricrogl

# macOS
# Download dmg
open MRIcroGL.dmg
# Drag to Applications

# Windows
# Download and extract portable version
# Or use installer

# Verify installation
./mricrogl --version
```

### dcm2niix (Included)

```bash
# dcm2niix is included with both tools
# Standalone installation:
# Linux
sudo apt-get install dcm2niix

# macOS
brew install dcm2niix

# Or download from: https://github.com/rordenlab/dcm2niix/releases
```

## Basic Usage - MRIcron

### Launch and Load Images

```bash
# Start MRIcron
mricron

# Load image from command line
mricron brain.nii.gz

# Load overlay
mricron T1.nii.gz -o zstat1.nii.gz -d 2.3 -h 8

# Options:
# -o: overlay image
# -d: display minimum
# -h: display maximum
```

### GUI Navigation

```
File → Open: Load background image
Overlay → Add: Load statistical overlay
Draw → ROI: Create regions of interest

Keyboard shortcuts:
- Arrow keys: Navigate slices
- +/-: Zoom in/out
- C: Toggle crosshair
- Space: Next overlay
- O: Overlay visibility
- R: Reset view
```

### Drawing ROIs

```bash
# In MRIcron GUI:
# 1. Draw → ROI
# 2. Select brush size
# 3. Draw on slices
# 4. File → Save VOI (Volume of Interest)

# ROI tools:
# - Pen: Freehand drawing
# - 2D/3D fill: Fill regions
# - Eraser: Remove voxels
# - Undo/Redo: Edit history

# Save ROI as NIfTI
# Draw → Save VOI → brain_roi.nii.gz
```

### Statistical Maps

```bash
# Load statistical overlay
mricron T1.nii.gz -o zstat.nii.gz

# Adjust thresholds:
# Overlay → Overlay (or Ctrl+O)
# - Min threshold slider
# - Max threshold slider
# - Color scheme dropdown
# - Opacity slider

# Common color schemes:
# - Red-Yellow
# - Blue-Green
# - Hot
# - Winter
# - Rainbow
```

## Basic Usage - MRIcroGL

### Launch and Navigation

```bash
# Start MRIcroGL
mricrogl

# Load image
mricrogl brain.nii.gz

# Load with overlay
mricrogl T1.nii.gz zstat.nii.gz

# Load tractography
mricrogl T1.nii.gz -tract fibers.trk

# Command-line options
mricrogl brain.nii.gz -c "savebitmap('output.png')"
```

### Display Modes

```bash
# Viewing modes (GUI):
# - Axial only: Single slice view
# - Axial + Coronal + Sagittal: Multi-planar
# - Multi-planar + 3D: 4-panel view
# - 3D render only: Volume rendering
# - Multi-tile: Mosaic view

# Keyboard shortcuts:
# A: Axial view
# C: Coronal view
# S: Sagittal view
# R: Render mode
# M: Mosaic view
# Space: Cycle through views
```

### 3D Rendering

```bash
# Enable 3D rendering
# View → Render

# Rendering options:
# - Clip plane: Slice through volume
# - Azimuth/Elevation: Rotate view
# - Depth: Clipping distance
# - Quality: Render samples
# - Lighting: Light position/intensity

# Presets:
# - CT Bone
# - CT Soft Tissue
# - MRI Brain
# - MRI Contrast
```

### Overlay Settings

```bash
# Advanced overlay options
# Overlay → Overlay 1

# Settings:
# - Color map: Red-Yellow, Blue-Green, Hot, etc.
# - Minimum: Lower threshold
# - Maximum: Upper threshold
# - Opacity: Transparency
# - Smooth: Edge smoothing
# - Outline: Show only edges
```

## Screenshots and Movies

### MRIcron Screenshots

```bash
# GUI: File → Save bitmap
# Saves current view as PNG

# Script for batch screenshots
#!/bin/bash
for file in *.nii.gz; do
    mricron "$file" -c "savebmp('${file%.nii.gz}.png')"
done
```

### MRIcroGL Screenshots

```bash
# High-quality screenshot
mricrogl brain.nii.gz -c "savebitmap('output.png')"

# With specific view
mricrogl brain.nii.gz -c "azimuth(45); elevation(25); savebitmap('rotated.png')"

# Batch screenshots from Python
python mricrogl_script.py
```

### Creating Movies

```bash
# Rotating movie script
mricrogl brain.nii.gz -c "
    for i = 0:5:360
        azimuth(i);
        savebitmap(sprintf('frame_%03d.png', i/5));
    end
"

# Convert to video
ffmpeg -framerate 30 -i frame_%03d.png -c:v libx264 rotation.mp4

# Slice-through movie
mricrogl brain.nii.gz -c "
    for i = 1:numSlices
        setSlice(i);
        savebitmap(sprintf('slice_%03d.png', i));
    end
"
```

## Python Scripting (MRIcroGL)

### Basic Python Interface

```python
#!/usr/bin/env python
import os
import subprocess

# Script for MRIcroGL
script = """
azimuth(45);
elevation(25);
zoom(1.5);
savebitmap('output.png');
"""

# Save script
with open('script.txt', 'w') as f:
    f.write(script)

# Execute
subprocess.run(['mricrogl', 'brain.nii.gz', '-s', 'script.txt'])
```

### Advanced Scripting

```python
#!/usr/bin/env python
"""
Generate publication-quality figures with MRIcroGL
"""

def create_overlay_figure(background, overlay, output, threshold=(2.3, 8)):
    """Create figure with statistical overlay"""

    script = f"""
    loadimage('{background}');
    overlayload('{overlay}');
    overlayminmax(1, {threshold[0]}, {threshold[1]});
    overlaycolorname(1, 'red-yellow');
    azimuth(90);
    savebitmap('{output}');
    """

    # Write and execute
    with open('temp_script.txt', 'w') as f:
        f.write(script)

    os.system(f'mricrogl -s temp_script.txt')
    os.remove('temp_script.txt')

# Usage
create_overlay_figure('T1.nii.gz', 'zstat1.nii.gz', 'figure1.png')
```

### Batch Processing

```python
#!/usr/bin/env python
import glob

subjects = glob.glob('sub-*/anat/*_T1w.nii.gz')

for subj_file in subjects:
    subj_id = subj_file.split('/')[0]
    output = f'screenshots/{subj_id}.png'

    script = f"""
    loadimage('{subj_file}');
    azimuth(90);
    elevation(15);
    savebitmap('{output}');
    """

    with open('temp.txt', 'w') as f:
        f.write(script)

    os.system('mricrogl -s temp.txt')
```

## Tractography Visualization

### Loading Fiber Tracts

```bash
# Load DTI tractography
mricrogl T1.nii.gz -tract fibers.trk

# Supported formats:
# - .trk (TrackVis)
# - .tck (MRtrix3)
# - .vtk (VTK polydata)

# GUI controls:
# Tract → Load Tract
# - Color by direction
# - Color by FA/MD
# - Tube vs. line rendering
# - Tract density/thinning
```

### Tract Visualization Options

```bash
# Scripting tract display
script="
    loadimage('T1.nii.gz');
    loadtrack('fibers.trk');
    trackprefs(1); // Tube rendering
    trackwidth(0.5); // Tube width
    savebitmap('tracts.png');
"

mricrogl -c "$script"
```

## Mesh Visualization

### Loading Surfaces

```bash
# Load FreeSurfer surface
mricrogl -m lh.pial rh.pial

# Load with overlay
mricrogl lh.inflated -d lh.thickness

# Supported mesh formats:
# - FreeSurfer (.pial, .white, .inflated)
# - GIFTI (.gii, .surf.gii)
# - Wavefront OBJ (.obj)
# - STL (.stl)
# - VTK (.vtk)

# Display options:
# - Lighting: Adjust material properties
# - Transparency: See-through rendering
# - Wire frame: Show mesh structure
# - Culling: Hide back faces
```

### Mesh Overlay Data

```bash
# Load surface with data overlay
mricrogl lh.pial -d lh.thickness -c "
    meshcolorname('hot');
    meshdatarange(1.5, 4.0);
    savebitmap('thickness.png');
"

# Color schemes for mesh data:
# - Hot, Cool, Jet, Rainbow
# - Brain colors (Freesurfer LUT)
```

## DICOM Conversion

### Using dcm2niix

```bash
# Convert DICOM to NIfTI
dcm2niix -o output_dir -f sub-01_T1w dicom_dir/

# Options:
# -o: output directory
# -f: filename format
# -z y: compress output (.nii.gz)
# -b y: create BIDS sidecar JSON
# -ba n: don't anonymize

# Batch conversion
dcm2niix -o nifti/ -f %p_%s -z y -b y dicom/

# Format strings:
# %p: protocol name
# %s: series number
# %d: description
# %t: time
```

### BIDS Conversion

```bash
# Create BIDS-compatible structure
dcm2niix -o output -f sub-%s_%p -b y -ba n dicom/

# Manual organization for BIDS:
# Rename and move files to:
# sub-01/
#   anat/
#     sub-01_T1w.nii.gz
#     sub-01_T1w.json
#   func/
#     sub-01_task-rest_bold.nii.gz
#     sub-01_task-rest_bold.json
```

## Image Processing

### NPM (Nonparametric Mapping)

```bash
# In MRIcron:
# Statistics → NPM

# Voxel-wise permutation testing
# - Load 4D file or multiple 3D files
# - Set design (one-sample, two-sample, paired)
# - Run permutations (5000-10000)
# - Threshold results
# - Save statistical maps
```

### Image Math

```bash
# Using MRIcron's calculator
# Tools → Image Math

# Operations:
# - Add, subtract, multiply, divide
# - Threshold
# - Smooth
# - Mask

# Command-line equivalent (using fslmaths or similar)
fslmaths img1.nii.gz -add img2.nii.gz result.nii.gz
```

## Scripting Reference

### MRIcroGL Script Commands

```python
# Core commands
"""
loadimage('file.nii.gz')         # Load volume
overlayload('overlay.nii.gz')    # Load overlay
overlayminmax(1, min, max)       # Set overlay range
overlaycolorname(1, 'red-yellow') # Set colormap
azimuth(angle)                    # Rotate azimuth
elevation(angle)                  # Rotate elevation
zoom(factor)                      # Zoom level
clip(0.1)                         # Clip plane
savebitmap('file.png')            # Save screenshot

# Mesh commands
loadmesh('surface.gii')           # Load surface
meshdatarange(min, max)           # Set overlay range
meshcolorname('hot')              # Set colormap
meshload('overlay.shape.gii')     # Load mesh data

# Track commands
loadtrack('fibers.trk')           # Load tractography
trackprefs(1)                     # Tube rendering
trackwidth(0.5)                   # Tube width
"""
```

## Integration with Claude Code

When helping users with MRIcron/MRIcroGL:

1. **Check Installation:**
   ```bash
   # MRIcron
   which mricron
   mricron --version

   # MRIcroGL
   which mricrogl
   mricrogl --version

   # dcm2niix
   dcm2niix --version
   ```

2. **Common Issues:**
   - OpenGL errors (update graphics drivers)
   - Slow rendering (reduce quality settings)
   - DICOM conversion errors (check file format)
   - Permission errors on macOS (Security & Privacy settings)
   - Display issues on Linux (missing libraries)

3. **Best Practices:**
   - Use MRIcroGL for modern systems (better performance)
   - Use MRIcron for older systems or specific features (NPM)
   - dcm2niix is the gold standard for DICOM conversion
   - Save scripts for reproducible figures
   - Use appropriate color schemes for data type
   - Export high-resolution screenshots (2x-4x native)
   - Validate DICOM conversions with multiple tools

4. **Quick Tips:**
   - Right-click for context menus
   - Use keyboard shortcuts for navigation
   - Script repetitive tasks (screenshots, conversions)
   - MRIcroGL supports drag-and-drop
   - Check dcm2niix JSON output for metadata
   - Use TrackVis format for tractography compatibility

## Troubleshooting

**Problem:** MRIcroGL won't start (OpenGL error)
**Solution:** Update graphics drivers, or use software rendering: `LIBGL_ALWAYS_SOFTWARE=1 mricrogl`

**Problem:** dcm2niix creates multiple files
**Solution:** Normal for multi-echo or multi-volume acquisitions. Check JSON descriptions to identify series.

**Problem:** Overlays not visible
**Solution:** Adjust threshold sliders, check overlay opacity, verify spatial alignment

**Problem:** Poor quality 3D rendering
**Solution:** Increase quality settings (Render → Quality), update OpenGL drivers

**Problem:** DICOM conversion fails
**Solution:** Check DICOM validity, try `dcm2niix -v y` for verbose output, update to latest version

## Resources

### MRIcron
- Website: https://www.nitrc.org/projects/mricron
- Forum: https://www.nitrc.org/forum/?group_id=152
- Manual: Included with installation (Help → Manual)

### MRIcroGL
- Website: https://www.nitrc.org/projects/mricrogl
- GitHub: https://github.com/rordenlab/MRIcroGL
- Documentation: https://www.mccauslandcenter.sc.edu/mricrogl/

### dcm2niix
- GitHub: https://github.com/rordenlab/dcm2niix
- Documentation: https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage
- BIDS Apps: https://github.com/bids-apps/

## Citation

```bibtex
@article{rorden2007mricron,
  title={MRIcron},
  author={Rorden, Chris and Brett, Matthew},
  year={2000},
  url={https://www.nitrc.org/projects/mricron}
}

@article{li2016dcm2niix,
  title={The first step for neuroimaging data analysis: DICOM to NIfTI conversion},
  author={Li, Xiangrui and Morgan, Paul S and Ashburner, John and Smith, Jolinda and Rorden, Christopher},
  journal={Journal of neuroscience methods},
  volume={264},
  pages={47--56},
  year={2016}
}
```

## Related Tools

- **FSLeyes:** FSL's modern viewer
- **ITK-SNAP:** Advanced segmentation tool
- **3D Slicer:** Comprehensive medical imaging platform
- **Mango:** Multi-image analysis GUI
- **TrackVis:** Dedicated tractography viewer
- **Freeview:** FreeSurfer's viewer
