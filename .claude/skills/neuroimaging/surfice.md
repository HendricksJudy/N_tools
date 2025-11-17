# Surfice

## Overview

**Surfice** is a powerful surface and volume visualization tool developed by Chris Rorden for creating publication-quality brain renderings. Built on modern OpenGL technology, Surfice provides high-performance rendering of cortical surfaces, tractography streamlines, and volumetric overlays with advanced lighting, transparency, and customization options. It's particularly popular in lesion-symptom mapping studies and for creating professional figures and videos for scientific publications.

Surfice supports multiple mesh formats (FreeSurfer, GIfTI, OBJ, PLY), volume formats (NIFTI, ANALYZE), and tractography formats (TRK, TCK). The tool offers both an interactive GUI for exploratory visualization and a Python-based scripting interface for automated batch processing and reproducible figure generation.

**Key Use Cases:**
- Publication-quality brain surface renderings
- Lesion overlap and subtraction mapping
- Tractography visualization on transparent surfaces
- Statistical map overlay on cortical surfaces
- Multi-subject composite figures
- Video creation for presentations
- Quality control of surface reconstructions

**Official Website:** https://www.nitrc.org/projects/surfice/
**Documentation:** https://www.nitrc.org/plugins/mwiki/index.php/surfice:MainPage
**Source Code:** https://github.com/neurolabusc/surf-ice

---

## Key Features

- **High-Performance Rendering:** Modern OpenGL for smooth real-time visualization
- **Multi-Format Support:** FreeSurfer, GIfTI, OBJ, PLY, BrainVoyager, VTK meshes
- **Volume Overlay:** NIFTI, ANALYZE, DICOM volume rendering on surfaces
- **Tractography Display:** TRK, TCK fiber bundle visualization
- **Lesion Mapping:** Overlap, subtraction, and cluster analysis tools
- **Advanced Materials:** Phong shading, transparency, matcap rendering
- **Custom Colormaps:** Built-in and custom color lookup tables
- **Camera Control:** Azimuth, elevation, distance for precise positioning
- **Quality Export:** High-resolution PNG, bitmap sequences for videos
- **Python Scripting:** Automate workflows and batch processing
- **Cross-Platform:** Windows, macOS, Linux binaries
- **No Dependencies:** Standalone executable, no installation required
- **Fast Loading:** Optimized mesh and volume loading
- **Multi-Panel Layouts:** Mosaic views for comparative figures
- **Clipping Planes:** Cut-away views for internal structures

---

## Installation

### Download Binaries

Surfice is distributed as standalone executables:

```bash
# Linux
wget https://github.com/neurolabusc/surf-ice/releases/latest/download/Surfice_linux.zip
unzip Surfice_linux.zip
cd Surfice
chmod +x surfice
./surfice

# macOS
# Download from: https://github.com/neurolabusc/surf-ice/releases/latest/download/Surfice_macOS.dmg
# Open DMG and drag Surfice to Applications

# Windows
# Download: https://github.com/neurolabusc/surf-ice/releases/latest/download/Surfice_windows.zip
# Extract and run surfice.exe
```

### Command-Line Access

```bash
# Add to PATH for command-line access (Linux/macOS)
export PATH=$PATH:/path/to/Surfice
echo 'export PATH=$PATH:/path/to/Surfice' >> ~/.bashrc

# Test installation
surfice --version

# Launch GUI
surfice

# Run script
surfice myscript.py
```

### Python Scripting Setup

```python
# Surfice uses its own embedded Python
# Scripts can be run via: surfice myscript.py

# Example script structure
import gl
gl.resetdefaults()
gl.meshload('lh.pial')
gl.shaderadjust('BrightnessMesh', 0.5)
```

---

## Basic Usage

### Launch and Load Surface

```bash
# Launch Surfice GUI
surfice

# Load surface from command line
surfice lh.pial

# Load with overlay
surfice lh.pial -o lh.thickness.nii.gz
```

**Interactive GUI Controls:**
- **Rotate:** Left mouse drag
- **Zoom:** Mouse wheel or right mouse drag
- **Pan:** Middle mouse drag or Shift + left drag
- **Reset view:** Press 'r' key
- **Toggle info:** Press 'i' key

### Load FreeSurfer Surfaces

```python
# load_freesurfer.py
import gl

# Reset to default state
gl.resetdefaults()

# Load FreeSurfer pial surface
gl.meshload('/path/to/freesurfer/subject/surf/lh.pial')

# Adjust brightness
gl.shaderadjust('BrightnessMesh', 0.6)

# Set background color (RGB 0-1)
gl.backcolor(255, 255, 255)  # White background

# Save screenshot
gl.savebmp('freesurfer_pial.png')
```

### Load Multiple Surfaces

```python
# load_bilateral.py
import gl

gl.resetdefaults()

# Load both hemispheres
gl.meshload('lh.pial')
gl.meshload('rh.pial')

# Adjust view
gl.azimuth(90)  # Rotate 90 degrees
gl.elevation(10)  # Tilt view

gl.savebmp('bilateral_surface.png')
```

---

## Surface Visualization

### Inflated and Flattened Surfaces

```python
# inflated_surface.py
import gl

gl.resetdefaults()

# Load inflated surface
gl.meshload('lh.inflated')

# Apply smooth shading
gl.shaderadjust('BrightnessMesh', 0.5)
gl.shaderadjust('BoundMesh', 0.0)  # Remove boundary highlighting

# Set view
gl.azimuth(180)
gl.elevation(0)

gl.savebmp('inflated_left.png')
```

### Different Surface Types

```python
# surface_types.py
import gl

surfaces = ['lh.pial', 'lh.white', 'lh.inflated', 'lh.sphere']
output_names = ['pial', 'white', 'inflated', 'sphere']

for surf, name in zip(surfaces, output_names):
    gl.resetdefaults()
    gl.meshload(surf)
    gl.azimuth(90)
    gl.savebmp(f'surface_{name}.png')
```

### Custom Mesh Colors

```python
# custom_colors.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')

# Set mesh color (RGB 0-255)
gl.meshcolor(0, 200, 100, 200)  # RGBA: green with alpha

# Or use predefined colors
gl.meshload('rh.pial')
gl.meshcolor(1, 255, 150, 100, 200)  # Orange for right hemisphere

gl.savebmp('colored_surfaces.png')
```

---

## Volume Overlays

### Overlay Statistical Maps

```python
# overlay_stats.py
import gl

gl.resetdefaults()

# Load surface
gl.meshload('lh.pial')

# Load overlay (e.g., activation map)
gl.overlayload('lh.activation.nii.gz')

# Set overlay transparency (0-1)
gl.overlayloadsmooth(0)  # No smoothing

# Set color range
gl.overlayminmax(0, 2, 5)  # Overlay 0: min=2, max=5

# Set colormap
gl.colorname(0, '5red')  # Red-yellow hot colormap

gl.savebmp('activation_overlay.png')
```

### Multiple Overlays

```python
# multiple_overlays.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')

# Load multiple statistical maps
gl.overlayload('positive_activation.nii.gz')
gl.overlayload('negative_activation.nii.gz')

# Configure first overlay (positive)
gl.overlayminmax(0, 2.3, 5.0)
gl.colorname(0, '5red')  # Red for positive

# Configure second overlay (negative)
gl.overlayminmax(1, -5.0, -2.3)
gl.colorname(1, '3blue')  # Blue for negative

gl.savebmp('bidirectional_activation.png')
```

### FreeSurfer Thickness Maps

```python
# thickness_overlay.py
import gl

gl.resetdefaults()

# Load inflated surface for better visualization
gl.meshload('lh.inflated')

# Load cortical thickness
gl.overlayload('lh.thickness')

# Set appropriate range for thickness (mm)
gl.overlayminmax(0, 1.5, 3.5)

# Use viridis colormap
gl.colorname(0, 'viridis')

# Adjust lighting
gl.shaderadjust('BrightnessMesh', 0.3)

gl.savebmp('cortical_thickness.png')
```

### Custom Color Maps

```python
# custom_colormap.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')
gl.overlayload('activation.nii.gz')

# Available colormaps:
# 'gray', 'red', 'green', 'blue', 'viridis', 'plasma',
# 'inferno', 'magma', '3blue', '5red', 'actc', 'BCGWHw'

gl.colorname(0, 'plasma')
gl.overlayminmax(0, 2.3, 5.0)

gl.savebmp('plasma_colormap.png')
```

---

## Lesion Mapping

### Single Lesion Visualization

```python
# single_lesion.py
import gl

gl.resetdefaults()

# Load template surface
gl.meshload('BrainMesh_ICBM152.lh.mz3')

# Load lesion mask
gl.overlayload('patient_lesion.nii.gz')

# Binary threshold for lesion
gl.overlayminmax(0, 1, 1)
gl.colorname(0, 'red')

# Make brain semi-transparent
gl.shaderadjust('Alpha', 0.7)

gl.savebmp('patient_lesion.png')
```

### Lesion Overlap Mapping

```python
# lesion_overlap.py
import gl

gl.resetdefaults()

# Load MNI template surface
gl.meshload('BrainMesh_ICBM152.mz3')

# Load lesion overlap map (voxel-wise count)
gl.overlayload('lesion_overlap_n50.nii.gz')

# Set colormap for overlap density
gl.overlayminmax(0, 1, 25)  # 1 to 25 patients
gl.colorname(0, 'warm')

# Add transparency to see surface
gl.overlayloadsmooth(1)

gl.savebmp('lesion_overlap_n50.png')
```

### Lesion Subtraction Analysis

```python
# lesion_subtraction.py
import gl

# Visualize difference between two patient groups
gl.resetdefaults()
gl.meshload('BrainMesh_ICBM152.lh.mz3')

# Load subtraction map (GroupA - GroupB)
gl.overlayload('subtraction_map.nii.gz')

# Positive values (more in GroupA) = red
# Negative values (more in GroupB) = blue
gl.overlayminmax(0, 5, 20)
gl.colorname(0, 'redyell')

gl.overlayload('subtraction_map.nii.gz')
gl.overlayminmax(1, -20, -5)
gl.colorname(1, 'bluegrn')

gl.savebmp('lesion_subtraction.png')
```

---

## Tractography Visualization

### Basic Fiber Tract Display

```python
# load_tractography.py
import gl

gl.resetdefaults()

# Load brain surface (transparent)
gl.meshload('lh.pial')
gl.meshload('rh.pial')
gl.shaderadjust('Alpha', 0.2)  # Very transparent

# Load tractography
gl.tractload('whole_brain.trk')

# Adjust tract rendering
gl.tractthick(0.3)  # Line thickness

gl.azimuth(90)
gl.savebmp('whole_brain_tracts.png')
```

### Specific Fiber Bundle

```python
# fiber_bundle.py
import gl

gl.resetdefaults()

# Load semi-transparent surfaces
gl.meshload('lh.pial')
gl.meshload('rh.pial')
gl.meshcolor(0, 200, 200, 200, 50)
gl.meshcolor(1, 200, 200, 200, 50)

# Load specific tract (e.g., arcuate fasciculus)
gl.tractload('arcuate_left.trk')

# Color by direction (default)
# Or set single color
# gl.tractcolor(255, 0, 0)  # Red tracts

gl.tractthick(0.5)

gl.azimuth(180)
gl.elevation(10)

gl.savebmp('arcuate_fasciculus.png')
```

### Multiple Tract Bundles

```python
# multiple_bundles.py
import gl

gl.resetdefaults()

# Load transparent brain
gl.meshload('BrainMesh_ICBM152.mz3')
gl.shaderadjust('Alpha', 0.15)

# Load multiple bundles
bundles = [
    'CST_left.trk',   # Corticospinal tract
    'CST_right.trk',
    'arcuate_left.trk',
    'uncinate_left.trk'
]

for bundle in bundles:
    gl.tractload(bundle)

# Adjust rendering
gl.tractthick(0.4)

gl.savebmp('multiple_bundles.png')
```

### MRtrix3 TCK Format

```python
# load_tck.py
import gl

gl.resetdefaults()

# Load surfaces
gl.meshload('lh.pial')
gl.meshload('rh.pial')
gl.shaderadjust('Alpha', 0.25)

# Load MRtrix3 .tck file
gl.tractload('tracks.tck')

# Set tract properties
gl.tractthick(0.3)

# Save from multiple angles
for angle in [0, 90, 180, 270]:
    gl.azimuth(angle)
    gl.savebmp(f'tracts_az{angle}.png')
```

---

## Camera and Lighting

### Camera Positioning

```python
# camera_control.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')

# Azimuth: horizontal rotation (0-360)
gl.azimuth(45)

# Elevation: vertical tilt (-90 to 90)
gl.elevation(15)

# Distance: zoom (0.5 = closer, 2.0 = farther)
gl.distance(1.5)

# Clip planes (near, far)
gl.clipplane(0.1, 2.0)

gl.savebmp('custom_view.png')
```

### Standard Views

```python
# standard_views.py
import gl

views = {
    'lateral': {'az': 90, 'el': 0},
    'medial': {'az': 270, 'el': 0},
    'dorsal': {'az': 0, 'el': 90},
    'ventral': {'az': 0, 'el': -90},
    'anterior': {'az': 180, 'el': 0},
    'posterior': {'az': 0, 'el': 0}
}

for view_name, angles in views.items():
    gl.resetdefaults()
    gl.meshload('lh.pial')
    gl.azimuth(angles['az'])
    gl.elevation(angles['el'])
    gl.savebmp(f'view_{view_name}.png')
```

### Lighting Adjustments

```python
# lighting_control.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')

# Adjust shader properties
gl.shaderadjust('BrightnessMesh', 0.5)  # Overall brightness
gl.shaderadjust('AmbientMesh', 0.3)     # Ambient light
gl.shaderadjust('DiffuseMesh', 0.6)     # Diffuse reflection
gl.shaderadjust('SpecularMesh', 0.2)    # Specular highlights
gl.shaderadjust('ShineMesh', 10)        # Shininess

gl.savebmp('custom_lighting.png')
```

### Matcap Rendering

```python
# matcap_rendering.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')

# Use matcap (material capture) shading
# Provides pre-baked lighting for realistic appearance
gl.matcapload('Cortex.png')  # Built-in matcap

# Alternative matcaps: 'Bone.png', 'Clay.png', 'Metal.png'

gl.savebmp('matcap_render.png')
```

---

## High-Quality Export

### High-Resolution Screenshots

```python
# high_res_export.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')
gl.overlayload('activation.nii.gz')
gl.overlayminmax(0, 2.3, 5.0)

gl.azimuth(90)

# Set output resolution (width x height)
gl.bmpzoom(3)  # 3x current window size

# Save high-resolution bitmap
gl.savebmp('high_res_figure.png')
```

### Multi-Panel Figures

```python
# multi_panel.py
import gl

# Create 2x2 mosaic
gl.resetdefaults()
gl.mosaic("A L H V")  # Anterior, Lateral, Horizontal, Ventral

gl.meshload('lh.pial')
gl.overlayload('activation.nii.gz')
gl.overlayminmax(0, 2.3, 5.0)

# Mosaic automatically shows multiple views
gl.savebmp('multi_panel_figure.png')
```

### Video Creation (Frame Sequence)

```python
# create_video_frames.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')
gl.overlayload('activation.nii.gz')

# Create rotating animation (360 frames)
for frame in range(360):
    gl.azimuth(frame)
    gl.savebmp(f'frame_{frame:04d}.png')

# Then use ffmpeg to create video:
# ffmpeg -r 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p rotation.mp4
```

### Transparent Backgrounds

```python
# transparent_background.py
import gl

gl.resetdefaults()

# Set background alpha to 0 (transparent)
gl.backcolor(0, 0, 0, 0)  # RGBA with alpha=0

gl.meshload('lh.pial')
gl.overlayload('activation.nii.gz')

# Save as PNG (supports transparency)
gl.savebmp('transparent_brain.png')
```

---

## Batch Processing

### Process Multiple Subjects

```python
# batch_subjects.py
import gl
import os

# List of subjects
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
fs_dir = '/path/to/freesurfer/subjects'

for subject in subjects:
    gl.resetdefaults()

    # Load subject's surface
    surf_file = os.path.join(fs_dir, subject, 'surf', 'lh.pial')
    gl.meshload(surf_file)

    # Load thickness
    thick_file = os.path.join(fs_dir, subject, 'surf', 'lh.thickness')
    gl.overlayload(thick_file)
    gl.overlayminmax(0, 1.5, 3.5)
    gl.colorname(0, 'viridis')

    # Standard view
    gl.azimuth(90)

    # Save
    gl.savebmp(f'{subject}_thickness.png')

    print(f"Processed {subject}")
```

### Batch Lesion Visualization

```python
# batch_lesions.py
import gl
import glob

# Load template once
gl.resetdefaults()
gl.meshload('BrainMesh_ICBM152.mz3')
gl.shaderadjust('Alpha', 0.6)

# Process all lesion files
lesion_files = glob.glob('lesions/patient_*.nii.gz')

for lesion_file in lesion_files:
    # Clear previous overlay
    gl.overlayclose(0)

    # Load new lesion
    gl.overlayload(lesion_file)
    gl.overlayminmax(0, 1, 1)
    gl.colorname(0, 'red')

    # Extract patient ID from filename
    patient_id = lesion_file.split('_')[1].replace('.nii.gz', '')

    # Save
    gl.savebmp(f'lesion_{patient_id}.png')
```

### Automated Multi-View Export

```python
# multi_view_batch.py
import gl

def render_all_views(surface_file, overlay_file, subject_id):
    """Render all standard views for a subject."""

    views = {
        'lateral': (90, 0),
        'medial': (270, 0),
        'dorsal': (0, 90),
        'ventral': (0, -90)
    }

    for view_name, (az, el) in views.items():
        gl.resetdefaults()
        gl.meshload(surface_file)

        if overlay_file:
            gl.overlayload(overlay_file)
            gl.overlayminmax(0, 2.3, 5.0)
            gl.colorname(0, '5red')

        gl.azimuth(az)
        gl.elevation(el)

        output_file = f'{subject_id}_{view_name}.png'
        gl.savebmp(output_file)

# Use function
render_all_views('lh.pial', 'lh.activation.nii.gz', 'sub-01')
```

---

## Integration with Claude Code

Surfice integrates seamlessly with automated neuroimaging workflows:

```python
# surfice_pipeline.py - Automated visualization pipeline

import subprocess
import os
from pathlib import Path

class SurficeRenderer:
    """Wrapper for Surfice rendering in automated pipelines."""

    def __init__(self, surfice_path='surfice'):
        self.surfice_path = surfice_path

    def run_script(self, script_file):
        """Execute a Surfice Python script."""
        cmd = [self.surfice_path, script_file]

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        return result

    def render_activation(self, surface, overlay, output, vmin=2.3, vmax=5.0):
        """Render activation map on surface."""

        script_content = f"""
import gl
gl.resetdefaults()
gl.meshload('{surface}')
gl.overlayload('{overlay}')
gl.overlayminmax(0, {vmin}, {vmax})
gl.colorname(0, '5red')
gl.azimuth(90)
gl.savebmp('{output}')
gl.quit()
"""

        # Write temporary script
        script_file = 'temp_render.py'
        with open(script_file, 'w') as f:
            f.write(script_content)

        # Execute
        self.run_script(script_file)

        # Cleanup
        os.remove(script_file)

        print(f"Rendered: {output}")

# Usage in pipeline
renderer = SurficeRenderer()

# Process fMRIPrep outputs
subjects_dir = Path('/data/derivatives/fmriprep')

for subject_dir in subjects_dir.glob('sub-*'):
    subject_id = subject_dir.name

    # Paths
    surface = subject_dir / 'anat' / f'{subject_id}_hemi-L_pial.surf.gii'
    stat_map = subject_dir / 'func' / f'{subject_id}_task-rest_zstat.nii.gz'
    output = f'figures/{subject_id}_activation.png'

    if surface.exists() and stat_map.exists():
        renderer.render_activation(
            str(surface),
            str(stat_map),
            output
        )
```

**Quality Control Visualization:**

```python
# qc_visualization.py
import subprocess
from pathlib import Path

def create_qc_figure(freesurfer_dir, subject, output_dir):
    """Create QC figure showing surface reconstruction."""

    script = f"""
import gl
gl.resetdefaults()

# Load both hemispheres
gl.meshload('{freesurfer_dir}/{subject}/surf/lh.pial')
gl.meshload('{freesurfer_dir}/{subject}/surf/rh.pial')

# Add original T1 as overlay
gl.overlayload('{freesurfer_dir}/{subject}/mri/T1.mgz')
gl.overlayminmax(0, 20, 120)
gl.colorname(0, 'gray')

# Multi-panel view
gl.mosaic("A L M")

# Save
gl.savebmp('{output_dir}/{subject}_qc.png')
gl.quit()
"""

    script_file = f'qc_{subject}.py'
    with open(script_file, 'w') as f:
        f.write(script)

    subprocess.run(['surfice', script_file], check=True)
    os.remove(script_file)

# Generate QC for all subjects
fs_dir = '/path/to/freesurfer'
subjects = ['sub-01', 'sub-02', 'sub-03']

for subject in subjects:
    create_qc_figure(fs_dir, subject, 'qc_figures')
```

---

## Integration with Other Tools

### FreeSurfer Integration

```python
# freesurfer_surfaces.py
import gl

# FreeSurfer subject directory
subject_dir = '/path/to/freesurfer/subjects/sub-01'

gl.resetdefaults()

# Load FreeSurfer surfaces
gl.meshload(f'{subject_dir}/surf/lh.pial')
gl.meshload(f'{subject_dir}/surf/rh.pial')

# Load FreeSurfer overlays
gl.overlayload(f'{subject_dir}/surf/lh.curv')
gl.overlayminmax(0, -0.5, 0.5)
gl.colorname(0, 'gray')

# Or load thickness
gl.overlayload(f'{subject_dir}/surf/lh.thickness')
gl.overlayminmax(0, 1.0, 4.0)
gl.colorname(0, 'viridis')

gl.savebmp('freesurfer_output.png')
```

### MRtrix3 Tractography

```python
# mrtrix_tracts.py
import gl

gl.resetdefaults()

# Load surfaces
gl.meshload('lh.pial')
gl.meshload('rh.pial')
gl.shaderadjust('Alpha', 0.2)

# Load MRtrix3 tractogram
gl.tractload('tracks_10M.tck')

# Can also load TCK files filtered by TractSeg
gl.tractload('CST_left.tck')
gl.tractload('CST_right.tck')

gl.tractthick(0.4)
gl.savebmp('mrtrix_tractography.png')
```

### FSL Statistical Maps

```python
# fsl_stats.py
import gl

gl.resetdefaults()

# Load MNI template surface
gl.meshload('BrainMesh_ICBM152.mz3')

# Load FSL z-stat map
gl.overlayload('zstat1.nii.gz')

# Set threshold (typically z > 2.3)
gl.overlayminmax(0, 2.3, 5.0)
gl.colorname(0, '5red')

# Add cluster mask (binary)
gl.overlayload('cluster_mask.nii.gz')
gl.overlayminmax(1, 1, 1)
gl.colorname(1, 'blue')

gl.savebmp('fsl_activation.png')
```

### SPM Results

```python
# spm_results.py
import gl

gl.resetdefaults()
gl.meshload('BrainMesh_ICBM152.mz3')

# Load SPM T-map
gl.overlayload('spmT_0001.nii')

# Threshold at p < 0.001 uncorrected (typically T > 3.1)
gl.overlayminmax(0, 3.1, 10.0)
gl.colorname(0, 'actc')  # SPM activation colormap

gl.azimuth(90)
gl.savebmp('spm_tmap.png')
```

---

## Advanced Techniques

### Clipping Planes

```python
# clipping_demo.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')
gl.meshload('rh.pial')

# Enable clipping plane
gl.clipplane(0.01, 0.5)  # Clip near half

# Useful for showing internal structures
gl.overlayload('subcortex.nii.gz')

gl.savebmp('clipped_view.png')
```

### Custom Node Colors

```python
# vertex_colors.py
import gl
import numpy as np

gl.resetdefaults()
gl.meshload('lh.pial')

# Surfice can load per-vertex color files
# Create custom vertex colors (NumPy array)
n_vertices = 163842  # FreeSurfer standard mesh

# Example: color gradient
colors = np.linspace(0, 1, n_vertices)
np.savetxt('custom_colors.txt', colors)

gl.overlayload('custom_colors.txt')
gl.colorname(0, 'plasma')

gl.savebmp('vertex_colors.png')
```

### Scripted Animations

```python
# rotation_animation.py
import gl

gl.resetdefaults()
gl.meshload('lh.pial')
gl.overlayload('activation.nii.gz')
gl.overlayminmax(0, 2.3, 5.0)

# Create smooth rotation
n_frames = 120
for i in range(n_frames):
    angle = (i / n_frames) * 360
    gl.azimuth(angle)
    gl.savebmp(f'anim/frame_{i:04d}.png')

# Convert to video with ffmpeg:
# ffmpeg -r 30 -i anim/frame_%04d.png -c:v libx264 rotation.mp4
```

---

## Troubleshooting

### Problem 1: Surfice Won't Launch

**Symptoms:** Application fails to start or crashes immediately

**Solution:**
```bash
# Linux: Check OpenGL support
glxinfo | grep OpenGL

# Minimum requirement: OpenGL 3.3
# Update graphics drivers if needed

# macOS: May need to allow in Security & Privacy settings
# System Preferences > Security & Privacy > General

# Windows: Install latest graphics drivers
# Download from NVIDIA/AMD/Intel website
```

### Problem 2: Mesh Not Loading

**Symptoms:** "Unable to load mesh" error

**Solution:**
```python
# Verify file format
# Supported: .pial, .white, .inflated, .gii, .obj, .ply, .mz3

# Check file path (use absolute paths)
import gl
gl.meshload('/full/path/to/surface.gii')

# For FreeSurfer, ensure surf directory structure
# subjects/sub-01/surf/lh.pial
```

### Problem 3: Overlay Not Visible

**Symptoms:** Overlay loads but doesn't appear

**Solution:**
```python
# Check overlay range
gl.overlayminmax(0, min_val, max_val)

# Ensure min < max
# Check data range in your volume

# Verify colormap
gl.colorname(0, '5red')

# Increase contrast
gl.overlayload('map.nii.gz')
gl.overlayminmax(0, 1, 10)  # Wider range
```

### Problem 4: Poor Image Quality

**Symptoms:** Jagged edges, pixelated output

**Solution:**
```python
# Increase output resolution
gl.bmpzoom(3)  # 3x window size

# Enable anti-aliasing (if available)
# Resize window before saving

# Use PNG format (better than BMP)
gl.savebmp('output.png')

# For publications, use very high zoom
gl.bmpzoom(5)
```

### Problem 5: Tractography Not Displaying

**Symptoms:** Tracts load but don't appear

**Solution:**
```python
# Increase tract thickness
gl.tractthick(1.0)  # Thicker lines

# Check tract file format (TRK or TCK)

# Ensure tracts are in same space as surface

# Make surface more transparent
gl.shaderadjust('Alpha', 0.1)

# Zoom out if tracts are outside view
gl.distance(2.0)
```

---

## Best Practices

### 1. Publication Figures

- **High resolution:** Use `gl.bmpzoom(3)` or higher
- **Clean backgrounds:** White (`gl.backcolor(255,255,255)`) or transparent
- **Consistent views:** Use same azimuth/elevation across subjects
- **Clear colormaps:** Choose perceptually uniform maps (viridis, plasma)
- **Appropriate thresholds:** Match statistical significance levels

### 2. Lesion Mapping

- **Template surfaces:** Use MNI152 template for group studies
- **Transparency:** Balance surface and overlay visibility
- **Color choice:** High-contrast colors for lesions (red on gray brain)
- **Multiple views:** Show lateral, medial, and dorsal views
- **Quantification:** Include overlap counts or percentages

### 3. Tractography

- **Transparency:** Keep surfaces very transparent (alpha < 0.3)
- **Tract thickness:** Adjust for visibility (0.3-0.6 typical)
- **Bundle selection:** Show specific bundles rather than whole brain
- **Color by direction:** Default RGB coloring aids interpretation
- **Multiple angles:** Provide different viewpoints

### 4. Scripting

- **Start with `gl.resetdefaults()`:** Ensure clean state
- **Use absolute paths:** Avoid path-related errors
- **End with `gl.quit()`:** Proper script termination
- **Comment code:** Document parameter choices
- **Parameterize:** Use variables for easy adjustment

### 5. Batch Processing

- **Consistent parameters:** Same color scales across subjects
- **Error handling:** Check file existence before loading
- **Progress tracking:** Print subject IDs as processed
- **Organized output:** Use descriptive filenames
- **Validation:** Visually check subset of outputs

---

## Resources

### Official Documentation

- **NITRC Project Page:** https://www.nitrc.org/projects/surfice/
- **Wiki Documentation:** https://www.nitrc.org/plugins/mwiki/index.php/surfice:MainPage
- **GitHub Repository:** https://github.com/neurolabusc/surf-ice
- **Issue Tracker:** https://github.com/neurolabusc/surf-ice/issues

### Tutorials and Examples

- **Scripting Examples:** Included in Surfice distribution (`/scripts` folder)
- **Lesion Mapping Tutorial:** https://www.nitrc.org/plugins/mwiki/index.php/surfice:lesion
- **Video Tutorials:** Available on Neuroimaging YouTube channels

### Related Tools by Chris Rorden

- **MRIcroGL:** Volume visualization (see `mricron.md`)
- **dcm2niix:** DICOM conversion
- **MRIcron:** Legacy viewer

### Community Support

- **NeuroStars Forum:** Tag questions with `surfice`
- **GitHub Discussions:** For feature requests and bugs

---

## Citation

```bibtex
@software{surfice2023,
  author = {Rorden, Christopher},
  title = {Surfice: Surface and Volume Visualization},
  year = {2023},
  url = {https://www.nitrc.org/projects/surfice/},
  note = {Version 1.0}
}
```

When using Surfice for publications, cite the software and any specific methods implemented:

```bibtex
@article{rorden2012age,
  title={Age-specific CT and MRI templates for spatial normalization},
  author={Rorden, Christopher and Bonilha, Leonardo and Fridriksson, Julius and Bender, Benjamin and Karnath, Hans-Otto},
  journal={Neuroimage},
  volume={61},
  number={4},
  pages={957--965},
  year={2012},
  publisher={Elsevier}
}
```

---

## Related Tools

- **MRIcroGL:** 3D volume rendering by same author (see `mricron.md`)
- **FSLeyes:** FSL's viewer (see `fsleyes.md`)
- **Connectome Workbench:** HCP visualization (see `connectome-workbench.md`)
- **PyCortex:** Web-based cortical visualization (see `pycortex.md`)
- **Brainrender:** Python 3D rendering (see `brainrender.md`)
- **BrainNet Viewer:** Network visualization (see `brainnet-viewer.md`)
- **FreeSurfer:** Surface reconstruction (provides input meshes)
- **MRtrix3:** Tractography generation (provides fiber bundles)
- **ITK-SNAP:** Segmentation viewer (see `itksnap.md`)

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**Surfice Version Covered:** 1.0.x
**Maintainer:** Claude Code Neuroimaging Skills
