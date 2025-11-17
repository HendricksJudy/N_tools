# PyCortex

## Overview

**PyCortex** is a Python library developed by the Gallant Lab at UC Berkeley for creating interactive, web-based visualizations of cortical surface data. Using modern WebGL technology, PyCortex generates stunning 3D brain visualizations that can be manipulated in real-time directly in a web browser or Jupyter notebook. The tool excels at displaying fMRI activations, retinotopic maps, and any cortical surface data with full interactivity including rotation, zooming, transparency adjustment, and anatomical layer selection.

PyCortex integrates seamlessly with FreeSurfer-processed anatomical data, provides powerful volume-to-surface mapping capabilities, and can export standalone HTML files for sharing visualizations with collaborators or as publication supplements. The library is particularly popular in visual neuroscience for creating retinotopic maps and in cognitive neuroscience for interactive exploration of fMRI results.

**Key Use Cases:**
- Interactive web-based fMRI activation visualization
- Retinotopic mapping and visual field analysis
- Cortical flatmap creation and display
- Publication supplements with interactive 3D brains
- Teaching materials with explorable neuroimaging data
- Group analysis visualization and comparison
- ROI definition on cortical surfaces

**Official Website:** https://gallantlab.github.io/pycortex/
**Documentation:** https://gallantlab.github.io/pycortex/docs/
**Source Code:** https://github.com/gallantlab/pycortex

---

## Key Features

- **Web-Based Visualization:** Interactive 3D rendering in browsers using WebGL
- **FreeSurfer Integration:** Direct import of FreeSurfer subjects and surfaces
- **Multiple Surface Views:** Inflated, fiducial, flat, sphere, and custom surfaces
- **Volume-to-Surface Mapping:** Automatic projection of volumetric data onto cortex
- **Flatmap Visualization:** Unfolded cortical surfaces for detailed inspection
- **Interactive Overlays:** Multiple data layers with adjustable transparency
- **Custom Colormaps:** Full matplotlib colormap support plus custom maps
- **Jupyter Integration:** Inline visualization in notebooks
- **Standalone HTML Export:** Share visualizations without requiring PyCortex installation
- **ROI Management:** Define, edit, and save regions of interest
- **Retinotopy Tools:** Specialized functions for visual field mapping
- **Subject Database:** Manage multiple subjects and surfaces
- **Vertex and Voxel Data:** Support for both surface and volume representations
- **High-Quality Rendering:** Anti-aliased, smooth visualizations
- **Python API:** Programmatic control for reproducible workflows

---

## Installation

### Using Conda (Recommended)

```bash
# Create conda environment
conda create -n pycortex python=3.9
conda activate pycortex

# Install pycortex from conda-forge
conda install -c conda-forge pycortex

# Verify installation
python -c "import cortex; print(cortex.__version__)"
```

### Using Pip

```bash
# Install pycortex
pip install pycortex

# Install optional dependencies
pip install matplotlib numpy scipy h5py pillow lxml nibabel tornado

# For Jupyter integration
pip install jupyter ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/gallantlab/pycortex.git
cd pycortex

# Install in development mode
pip install -e .

# Run tests
pytest
```

### Configuration

```bash
# Initialize pycortex configuration
python -c "import cortex; cortex.database.default_filestore = '/path/to/pycortex_store'"

# Or set environment variable
export CORTEX_DATA=/path/to/pycortex_store

# Create default configuration file
python -c "import cortex; cortex.options.config.save_default()"
```

---

## Basic Usage

### Import and Setup

```python
import cortex
import numpy as np
import matplotlib.pyplot as plt

# Set filestore location (where subject data is stored)
cortex.database.default_filestore = '/path/to/pycortex_store'

# List available subjects
subjects = cortex.database.subjects
print(f"Available subjects: {subjects}")
```

### Load and Display Surface

```python
# Simple surface display
subject = 'S1'

# Create a simple vertex array (random data for demo)
num_vertices = cortex.db.get_surfinfo(subject).left.shape[0]
data = np.random.randn(num_vertices)

# Create Vertex object
vertex_data = cortex.Vertex(data, subject, cmap='viridis')

# Display in web browser
cortex.webshow(vertex_data)
```

### Display in Jupyter Notebook

```python
# In Jupyter notebook
import cortex

# Create data
data = np.random.randn(cortex.db.get_surfinfo('S1').left.shape[0])
vertex_data = cortex.Vertex(data, 'S1', vmin=-2, vmax=2, cmap='RdBu_r')

# Display inline
cortex.webshow(vertex_data, open_browser=False, port=8888)
```

---

## Subject Database

### Import FreeSurfer Subject

```python
import cortex

# Import FreeSurfer subject
subject_id = 'sub-01'
freesurfer_subject_dir = '/path/to/freesurfer/subjects/sub-01'

cortex.freesurfer.import_subj(
    subject=subject_id,
    sname=subject_id,  # Name in pycortex database
    freesurfer_subject_dir=freesurfer_subject_dir
)

print(f"Imported {subject_id} into pycortex database")
```

### Import with Custom Options

```python
# Import with specific options
cortex.freesurfer.import_subj(
    subject='sub-01',
    sname='sub-01',
    freesurfer_subject_dir='/path/to/fs/sub-01',
    whitematter_surf='smoothwm',  # Use smoothed white matter surface
    aseg_name='aseg.mgz'  # Segmentation file
)

# Import flat surface if available
cortex.freesurfer.import_flat(
    subject='sub-01',
    patch='/path/to/freesurfer/sub-01/surf/lh.full.flat.patch.3d',
    hemis='lh'
)
```

### List and Manage Subjects

```python
# List all subjects
subjects = cortex.database.subjects
print(f"Subjects: {subjects}")

# Get surface information
surfinfo = cortex.db.get_surfinfo('sub-01')
print(f"Left hemisphere vertices: {surfinfo.left.shape[0]}")
print(f"Right hemisphere vertices: {surfinfo.right.shape[0]}")

# Remove subject (if needed)
# cortex.database.db.remove_subject('sub-01')
```

### Create Custom Subject

```python
# Create subject from custom surfaces
import nibabel as nib

# Load custom surfaces
lh_pial = nib.load('lh.pial.gii')
rh_pial = nib.load('rh.pial.gii')

# Add to pycortex database
cortex.database.db.save_xfm(
    'custom_subject',
    'identity',
    np.eye(4)  # Identity transform
)
```

---

## Volume-to-Surface Mapping

### Basic Volume Mapping

```python
import cortex
import nibabel as nib

# Load volumetric data
volume_file = 'zstat1.nii.gz'
volume_img = nib.load(volume_file)
volume_data = volume_img.get_fdata()

# Map to surface
subject = 'sub-01'
transform = 'identity'  # Or name of transform

# Create Volume object
volume_obj = cortex.Volume(
    volume_data,
    subject,
    transform,
    vmin=2.3,
    vmax=6.0,
    cmap='hot'
)

# Display
cortex.webshow(volume_obj)
```

### Map with Custom Transform

```python
# Register functional to anatomical and create transform
import numpy as np

# Load or create transform matrix (4x4)
# This would typically come from registration software
transform_matrix = np.eye(4)

# Save transform to pycortex database
cortex.database.db.save_xfm(
    'sub-01',
    'func_to_anat',
    transform_matrix,
    xfmtype='coord'
)

# Use transform for mapping
volume = cortex.Volume(
    volume_data,
    'sub-01',
    'func_to_anat',
    vmin=0,
    vmax=10,
    cmap='viridis'
)

cortex.webshow(volume)
```

### Automatic Alignment

```python
# Automatically align functional to anatomical
from cortex import align

# Align functional mean to anatomy
mean_func = nib.load('mean_func.nii.gz').get_fdata()

# Create alignment
aligned_xfm = align.automatic(
    'sub-01',
    'func_to_anat',
    mean_func,
    noclean=True
)

# Now use aligned transform
stat_map = nib.load('zstat1.nii.gz').get_fdata()
volume = cortex.Volume(
    stat_map,
    'sub-01',
    'func_to_anat',
    vmin=2.3,
    vmax=5
)

cortex.webshow(volume)
```

---

## Vertex Data Visualization

### Create Vertex Data

```python
# Create data for each hemisphere separately
import cortex
import numpy as np

subject = 'sub-01'

# Get number of vertices per hemisphere
left_vertices = cortex.db.get_surfinfo(subject).left.shape[0]
right_vertices = cortex.db.get_surfinfo(subject).right.shape[0]

# Create separate data for each hemisphere
left_data = np.random.randn(left_vertices)
right_data = np.random.randn(right_vertices)

# Create Vertex2D object
vertex_data = cortex.Vertex2D(
    left_data,
    right_data,
    subject=subject,
    vmin=-2,
    vmax=2,
    cmap='RdBu_r'
)

cortex.webshow(vertex_data)
```

### Single Hemisphere Data

```python
# Visualize only one hemisphere
# Set other hemisphere to NaN

left_data = np.random.randn(left_vertices)
right_data = np.full(right_vertices, np.nan)  # NaN for right

vertex_data = cortex.Vertex(
    left_data,
    right_data,
    subject,
    cmap='plasma'
)

cortex.webshow(vertex_data)
```

### Thresholded Vertex Data

```python
# Apply threshold to vertex data
data = np.random.randn(left_vertices + right_vertices)

# Threshold: only show values > 2
data_thresholded = data.copy()
data_thresholded[np.abs(data) < 2] = np.nan

vertex_data = cortex.Vertex(
    data_thresholded,
    subject,
    vmin=2,
    vmax=5,
    cmap='hot'
)

cortex.webshow(vertex_data)
```

---

## Flat Maps

### Create Flat Map Visualization

```python
# Display data on flattened cortex
import cortex

subject = 'sub-01'
data = np.random.randn(cortex.db.get_surfinfo(subject).left.shape[0])

vertex_data = cortex.Vertex(data, subject, cmap='viridis')

# Show flatmap view
cortex.webgl.show(
    vertex_data,
    recache=False,
    template='flatmap'  # Use flatmap template
)
```

### Create Custom Flatmap Cuts

```python
# Define flatmap cuts manually
import cortex.polyutils as polyutils

# Create cuts for flatmap
cuts = polyutils.Surface(subject, 'fiducial', hemisphere='lh')

# Manual cut definition (specify vertices to cut)
# This is typically done interactively
cortex.db.save_cuts(subject, cuts)
```

### Flatmap with Multiple Datasets

```python
# Show multiple datasets on flatmap
dataset1 = cortex.Vertex(data1, subject, cmap='Reds', vmin=0, vmax=1)
dataset2 = cortex.Vertex(data2, subject, cmap='Blues', vmin=0, vmax=1)

# Create VertexRGB for multi-channel display
rgb_data = cortex.VertexRGB(
    data1,  # Red channel
    data2,  # Green channel
    np.zeros_like(data1),  # Blue channel
    subject=subject
)

cortex.webshow(rgb_data)
```

---

## Colormaps and Styling

### Custom Colormaps

```python
# Use matplotlib colormaps
import matplotlib.pyplot as plt

# Any matplotlib colormap
vertex_data = cortex.Vertex(
    data,
    subject,
    cmap='viridis',  # 'plasma', 'inferno', 'magma', 'twilight', etc.
    vmin=-3,
    vmax=3
)

# Create custom colormap
from matplotlib.colors import LinearSegmentedColormap

colors = ['darkblue', 'lightblue', 'white', 'yellow', 'darkred']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

vertex_data = cortex.Vertex(data, subject, cmap=cmap)
cortex.webshow(vertex_data)
```

### RGB Visualization

```python
# Create RGB vertex data (multi-channel)
red_channel = np.random.rand(left_vertices + right_vertices)
green_channel = np.random.rand(left_vertices + right_vertices)
blue_channel = np.random.rand(left_vertices + right_vertices)

rgb_vertex = cortex.VertexRGB(
    red_channel,
    green_channel,
    blue_channel,
    subject=subject
)

cortex.webshow(rgb_vertex)
```

### Two-Sided Colormap (Diverging)

```python
# Diverging colormap for positive and negative values
pos_neg_data = np.random.randn(left_vertices + right_vertices)

vertex_data = cortex.Vertex(
    pos_neg_data,
    subject,
    vmin=-5,
    vmax=5,
    cmap='RdBu_r'  # Red-Blue reversed (red=positive, blue=negative)
)

cortex.webshow(vertex_data)
```

---

## Retinotopic Mapping

### Polar Angle and Eccentricity

```python
# Visualize retinotopic maps
import cortex

subject = 'sub-01'

# Load polar angle and eccentricity data (in radians and degrees)
polar_angle = np.load('polar_angle.npy')  # Range: -pi to pi
eccentricity = np.load('eccentricity.npy')  # Range: 0 to max_ecc

# Create angle map with hsv colormap
angle_vertex = cortex.Vertex(
    polar_angle,
    subject,
    vmin=-np.pi,
    vmax=np.pi,
    cmap='hsv'
)

# Create eccentricity map
ecc_vertex = cortex.Vertex(
    eccentricity,
    subject,
    vmin=0,
    vmax=10,  # degrees of visual angle
    cmap='hot'
)

# Show angle map
cortex.webshow(angle_vertex)
```

### Combined Retinotopy Visualization

```python
# RGB visualization: angle as hue, eccentricity as saturation/value
from matplotlib.colors import hsv_to_rgb

# Normalize eccentricity
ecc_norm = eccentricity / eccentricity.max()

# Convert polar angle to hue (0-1)
hue = (polar_angle + np.pi) / (2 * np.pi)

# Create HSV array
hsv = np.zeros((len(hue), 3))
hsv[:, 0] = hue
hsv[:, 1] = ecc_norm  # Saturation
hsv[:, 2] = ecc_norm  # Value

# Convert to RGB
rgb = hsv_to_rgb(hsv)

# Create RGB vertex
retino_rgb = cortex.VertexRGB(
    rgb[:, 0],
    rgb[:, 1],
    rgb[:, 2],
    subject=subject
)

cortex.webshow(retino_rgb)
```

### Visual Field Sign Map

```python
# Calculate visual field sign from gradients
import cortex.polyutils as polyutils

# Compute gradients (requires surface geometry)
surface = polyutils.Surface(subject, 'fiducial')

# Visual field sign calculation (simplified)
# In practice, use proper gradient calculation
vfs = np.random.choice([-1, 1], size=left_vertices + right_vertices)

vfs_vertex = cortex.Vertex(
    vfs,
    subject,
    vmin=-1,
    vmax=1,
    cmap='RdBu_r'
)

cortex.webshow(vfs_vertex)
```

---

## ROI Definition and Management

### Define ROI Interactively

```python
# Launch interactive ROI defin ition
import cortex

subject = 'sub-01'

# Start webgl viewer
cortex.webgl.show(subject)

# ROIs can be drawn interactively in the browser
# Save ROIs using the interface

# Programmatically load saved ROIs
rois = cortex.db.get_overlay(subject, 'rois')
```

### Create ROI Programmatically

```python
# Create ROI from vertex indices
import cortex

# Define vertices in ROI
roi_vertices = [100, 101, 102, 150, 151, 200]

# Create binary mask
num_vertices = cortex.db.get_surfinfo(subject).left.shape[0]
roi_mask = np.zeros(num_vertices)
roi_mask[roi_vertices] = 1

# Save ROI
cortex.db.save_mask(subject, 'my_roi', roi_mask, hemisphere='left')

# Load and visualize ROI
roi_data = cortex.db.get_mask(subject, 'my_roi')
roi_vertex = cortex.Vertex(roi_data, subject, cmap='Reds')
cortex.webshow(roi_vertex)
```

### Extract Data from ROI

```python
# Get data within ROI
roi_mask = cortex.db.get_mask(subject, 'V1_left')

# Apply mask to data
vertex_data = np.random.randn(left_vertices + right_vertices)
roi_data_values = vertex_data[roi_mask.astype(bool)]

# Compute statistics
mean_value = roi_data_values.mean()
std_value = roi_data_values.std()
print(f"ROI mean: {mean_value:.3f}, std: {std_value:.3f}")
```

---

## Multiple Overlays

### Layer Multiple Datasets

```python
# Create dataset with multiple overlays
import cortex

# Create base anatomical display
anat = cortex.Vertex.empty(subject)

# Create activation overlay
activation = cortex.Vertex(
    activation_data,
    subject,
    vmin=2.3,
    vmax=5,
    cmap='hot'
)

# Create ROI overlay
roi = cortex.Vertex(
    roi_mask,
    subject,
    vmin=0,
    vmax=1,
    cmap='Reds'
)

# Combine in dataset
dataset = cortex.Dataset(
    anatomical=anat,
    activation=activation,
    roi=roi
)

cortex.webshow(dataset)
```

### Adjust Layer Transparency

```python
# Control transparency of overlays
# Transparency is controlled in the web viewer
# Use alpha parameter in Vertex object

vertex_data = cortex.Vertex(
    data,
    subject,
    cmap='hot',
    vmin=0,
    vmax=1
)

# Transparency is adjusted interactively in webviewer
cortex.webshow(vertex_data)
```

---

## Export and Sharing

### Export Standalone HTML

```python
# Create standalone HTML visualization
import cortex

subject = 'sub-01'
data = np.random.randn(cortex.db.get_surfinfo(subject).left.shape[0])

vertex_data = cortex.Vertex(data, subject, cmap='viridis')

# Export to HTML file
cortex.webgl.make_static(
    outpath='brain_visualization.html',
    data=vertex_data,
    recache=False
)

print("Exported to brain_visualization.html")
# This file can be shared and opened in any browser
```

### Save High-Quality Images

```python
# Save static images from web viewer
# This requires running the viewer and using browser screenshot
# Or use matplotlib for 2D projections

import matplotlib.pyplot as plt

# Create flatmap figure
fig = cortex.quickflat.make_figure(
    vertex_data,
    with_curvature=True,
    with_sulci=True,
    with_labels=False
)

plt.savefig('flatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Create Publication Figures

```python
# Generate publication-quality flatmap
import cortex.quickflat as qf

# Create detailed flatmap
fig = qf.make_figure(
    vertex_data,
    with_curvature=True,
    with_sulci=True,
    with_roi=True,
    with_labels=True,
    height=2048,
    labelsize='14pt',
    dpi=300
)

plt.savefig('publication_flatmap.png', dpi=300, bbox_inches='tight')
```

---

## Integration with Claude Code

PyCortex integrates seamlessly with neuroimaging pipelines:

```python
# pycortex_pipeline.py - Automated fMRI visualization

import cortex
import nibabel as nib
import numpy as np
from pathlib import Path

class PyCortexVisualizer:
    """Wrapper for pycortex in automated workflows."""

    def __init__(self, subject, pycortex_db='/path/to/pycortex_store'):
        self.subject = subject
        cortex.database.default_filestore = pycortex_db

    def import_freesurfer(self, fs_subject_dir):
        """Import FreeSurfer subject."""
        cortex.freesurfer.import_subj(
            subject=self.subject,
            sname=self.subject,
            freesurfer_subject_dir=fs_subject_dir
        )

    def visualize_activation(self, zstat_file, output_html, vmin=2.3, vmax=6.0):
        """Visualize fMRI activation."""

        # Load statistical map
        zstat_img = nib.load(zstat_file)
        zstat_data = zstat_img.get_fdata()

        # Create volume
        volume = cortex.Volume(
            zstat_data,
            self.subject,
            'identity',
            vmin=vmin,
            vmax=vmax,
            cmap='hot'
        )

        # Export to HTML
        cortex.webgl.make_static(
            outpath=output_html,
            data=volume
        )

        print(f"Saved: {output_html}")

    def batch_visualize(self, contrast_files, output_dir):
        """Batch visualize multiple contrasts."""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        for contrast_file in contrast_files:
            contrast_name = Path(contrast_file).stem
            output_html = output_dir / f'{self.subject}_{contrast_name}.html'

            self.visualize_activation(
                contrast_file,
                str(output_html)
            )

# Usage in pipeline
visualizer = PyCortexVisualizer('sub-01')

# Import subject if needed
# visualizer.import_freesurfer('/path/to/freesurfer/sub-01')

# Visualize contrasts
contrasts = [
    '/data/derivatives/sub-01/zstat1.nii.gz',
    '/data/derivatives/sub-01/zstat2.nii.gz'
]

visualizer.batch_visualize(contrasts, 'visualizations')
```

**Group Analysis Visualization:**

```python
# group_visualization.py
import cortex
import nibabel as nib
import numpy as np
from scipy import stats

def visualize_group_ttest(subject_files, output_html):
    """Visualize group-level t-test on surface."""

    # Load data for all subjects (assumes already mapped to surface)
    subject_data = []
    for subj_file in subject_files:
        data = np.load(subj_file)  # Vertex data
        subject_data.append(data)

    subject_data = np.array(subject_data)

    # One-sample t-test against zero
    t_stats, p_values = stats.ttest_1samp(subject_data, 0, axis=0)

    # Threshold by significance
    t_stats[p_values > 0.05] = np.nan

    # Create vertex object
    group_result = cortex.Vertex(
        t_stats,
        'fsaverage',  # Use template subject
        vmin=3,
        vmax=10,
        cmap='hot'
    )

    # Export
    cortex.webgl.make_static(outpath=output_html, data=group_result)

# Use function
subject_files = [f'/data/sub-{i:02d}_surface_data.npy' for i in range(1, 21)]
visualize_group_ttest(subject_files, 'group_ttest.html')
```

---

## Integration with Other Tools

### Nilearn Integration

```python
# Use Nilearn for analysis, PyCortex for visualization
from nilearn import datasets, surface
import cortex

# Fetch example data
motor_images = datasets.fetch_neurovault_motor_task()
stat_img = motor_images.images[0]

# Project to FreeSurfer surface using Nilearn
texture_left = surface.vol_to_surf(stat_img, 'fsaverage5', hemi='left')
texture_right = surface.vol_to_surf(stat_img, 'fsaverage5', hemi='right')

# Combine hemispheres
texture = np.concatenate([texture_left, texture_right])

# Visualize with PyCortex
vertex_data = cortex.Vertex(
    texture,
    'fsaverage5',
    vmin=0,
    vmax=10,
    cmap='cold_hot'
)

cortex.webshow(vertex_data)
```

### NiBabel Integration

```python
# Load GIfTI surfaces with NiBabel
import nibabel as nib
import cortex

# Load surface data
gifti_file = 'lh.func.gii'
gifti_img = nib.load(gifti_file)
surface_data = gifti_img.darrays[0].data

# Visualize with PyCortex
vertex_data = cortex.Vertex(
    surface_data,
    'sub-01',
    cmap='viridis'
)

cortex.webshow(vertex_data)
```

### fMRIPrep Integration

```python
# Visualize fMRIPrep surface outputs
import cortex
import nibabel as nib

# fMRIPrep surface functional data
fmriprep_dir = '/data/derivatives/fmriprep/sub-01'

# Load surface functional data (GIfTI)
func_left = nib.load(
    f'{fmriprep_dir}/func/sub-01_task-rest_hemi-L_space-fsaverage5_bold.func.gii'
)
func_right = nib.load(
    f'{fmriprep_dir}/func/sub-01_task-rest_hemi-R_space-fsaverage5_bold.func.gii'
)

# Extract mean across time
mean_left = func_left.darrays[0].data.mean(axis=1)
mean_right = func_right.darrays[0].data.mean(axis=1)

# Combine and visualize
mean_func = np.concatenate([mean_left, mean_right])

vertex_data = cortex.Vertex(mean_func, 'fsaverage5', cmap='gray')
cortex.webshow(vertex_data)
```

---

## Advanced Techniques

### Animate Time Series

```python
# Create animation of time series data
import cortex
import numpy as np

# Load 4D data (time series)
timeseries = np.random.randn(100, left_vertices + right_vertices)  # 100 timepoints

# Create movie
cortex.dataset.save(
    'timeseries.hdf',
    timeseries=cortex.Vertex(timeseries.T, subject, cmap='RdBu_r')
)

# Export to video
cortex.webgl.make_video(
    'timeseries.hdf',
    'brain_timeseries.mp4',
    fps=10
)
```

### Custom Surface Shapes

```python
# Use custom surface inflation or transformation
import cortex

# Modify surface vertices (for custom inflation, etc.)
surface = cortex.polyutils.Surface(subject, 'fiducial')

# Get vertices and faces
vertices, faces = surface.coords, surface.polys

# Apply custom transformation
vertices_transformed = vertices * 1.1  # Simple scaling example

# Create new surface type
cortex.db.save_surf(
    subject,
    'custom_inflated',
    vertices_transformed,
    faces
)

# Use custom surface
data = np.random.randn(len(vertices))
vertex_data = cortex.Vertex(data, subject, cmap='viridis')
cortex.webshow(vertex_data, surface='custom_inflated')
```

### Searchlight Analysis Visualization

```python
# Visualize searchlight decoding accuracy
import cortex
import numpy as np

# Load searchlight results (accuracy per searchlight center)
searchlight_acc = np.load('searchlight_accuracy.npy')

# Map to vertices
vertex_accuracy = cortex.Vertex(
    searchlight_acc,
    subject,
    vmin=0.4,
    vmax=0.8,
    cmap='hot'
)

cortex.webshow(vertex_accuracy)
```

---

## Troubleshooting

### Problem 1: Subject Import Fails

**Symptoms:** Error importing FreeSurfer subject

**Solution:**
```python
# Ensure FreeSurfer subject directory is complete
import os

fs_dir = '/path/to/freesurfer/sub-01'
required_files = [
    'surf/lh.pial',
    'surf/rh.pial',
    'surf/lh.white',
    'surf/rh.white',
    'surf/lh.inflated',
    'surf/rh.inflated',
    'mri/T1.mgz'
]

for f in required_files:
    assert os.path.exists(os.path.join(fs_dir, f)), f"Missing: {f}"

# Try import again
cortex.freesurfer.import_subj('sub-01', 'sub-01', fs_dir)
```

### Problem 2: WebGL Viewer Won't Load

**Symptoms:** Browser shows blank page or errors

**Solution:**
```python
# Check port availability
cortex.webshow(data, port=8080)  # Try different port

# Clear cache
cortex.webgl.show(data, recache=True)

# Check browser console for errors
# Enable WebGL in browser settings if disabled

# For Jupyter, ensure widgets are enabled
# jupyter nbextension enable --py widgetsnbextension
```

### Problem 3: Volume Alignment Issues

**Symptoms:** Volumetric data misaligned with surface

**Solution:**
```python
# Check transform
transforms = cortex.db.get_xfms(subject)
print(f"Available transforms: {transforms}")

# Create new alignment manually
from cortex import align

# Use interactive alignment tool
align.manual('sub-01', 'func_to_anat', mean_func_volume)

# Or use automatic alignment
align.automatic('sub-01', 'func_to_anat', mean_func_volume)
```

### Problem 4: Flatmap Not Available

**Symptoms:** Cannot display flatmap view

**Solution:**
```python
# Create flatmap cuts
import cortex

# Import flatmap from FreeSurfer (if available)
cortex.freesurfer.import_flat(
    subject,
    patch='/path/to/fs/subject/surf/lh.full.flat.patch.3d',
    hemis='lh'
)

# Or create cuts interactively
cortex.db.save_view(subject)  # Save current viewpoint as flatmap
```

### Problem 5: Out of Memory with Large Datasets

**Symptoms:** Python crashes or hangs with large data

**Solution:**
```python
# Reduce surface resolution
# Use fsaverage5 instead of fsaverage (or subject-specific)

# Process hemispheres separately
left_data = cortex.Vertex(left_values, None, subject=subject)
right_data = cortex.Vertex(None, right_values, subject=subject)

# Downsample time series before visualization
timeseries_downsampled = timeseries[::2]  # Every other timepoint
```

---

## Best Practices

### 1. Data Organization

- **Use FreeSurfer organization:** Keep FreeSurfer subject directories intact
- **Separate pycortex store:** Maintain dedicated directory for pycortex database
- **Version control transforms:** Save alignment matrices for reproducibility
- **Document ROIs:** Use descriptive names for regions of interest

### 2. Visualization Design

- **Choose appropriate colormaps:** Perceptually uniform for continuous data
- **Threshold properly:** Set vmin/vmax based on data distribution
- **Use transparency:** Make anatomical features visible under overlays
- **Multiple views:** Provide lateral, medial, and flatmap views
- **Consistent scaling:** Use same color range across subjects for comparison

### 3. Interactive Exploration

- **Start with web viewer:** Explore data interactively before finalizing
- **Save viewpoints:** Document camera angles and settings
- **Layer data strategically:** Most important information on top
- **Test in target browsers:** Verify HTML exports work for intended audience

### 4. Publication Figures

- **High DPI:** Use 300 DPI for publication-quality images
- **Clear colorbars:** Include scales with appropriate labels
- **Minimal clutter:** Remove unnecessary elements
- **Consistent style:** Match formatting across figures
- **Include legends:** Explain colormaps and thresholds in captions

### 5. Reproducibility

- **Script everything:** Automate visualization pipeline
- **Version pycortex:** Record library version used
- **Save parameters:** Document all colormap and threshold choices
- **Archive HTML:** Keep standalone visualizations as supplements
- **Test on fresh install:** Verify scripts work in clean environment

---

## Resources

### Official Documentation

- **Website:** https://gallantlab.github.io/pycortex/
- **Documentation:** https://gallantlab.github.io/pycortex/docs/
- **GitHub:** https://github.com/gallantlab/pycortex
- **Issue Tracker:** https://github.com/gallantlab/pycortex/issues

### Tutorials and Examples

- **Quickstart:** https://gallantlab.github.io/pycortex/quickstart.html
- **Example Gallery:** https://gallantlab.github.io/pycortex/auto_examples/
- **Retinotopy Tutorial:** Available in documentation
- **Jupyter Notebooks:** Example notebooks in GitHub repository

### Publications

- **Original Paper:** Gao et al. (2015) "Pycortex: an interactive surface visualizer for fMRI" *Frontiers in Neuroinformatics*

### Community Support

- **GitHub Discussions:** For questions and feature requests
- **NeuroStars:** Tag questions with `pycortex`
- **Mailing List:** Available through project website

---

## Citation

```bibtex
@article{gao2015pycortex,
  title={Pycortex: an interactive surface visualizer for fMRI},
  author={Gao, James S and Huth, Alexander G and Lescroart, Mark D and Gallant, Jack L},
  journal={Frontiers in Neuroinformatics},
  volume={9},
  pages={23},
  year={2015},
  publisher={Frontiers},
  doi={10.3389/fninf.2015.00023}
}
```

---

## Related Tools

- **Nilearn:** Python neuroimaging analysis (see `nilearn.md`)
- **FreeSurfer:** Surface reconstruction (provides input surfaces)
- **fMRIPrep:** Preprocessing pipeline (provides surface data)
- **Surfice:** Alternative surface visualization (see `surfice.md`)
- **Connectome Workbench:** HCP visualization (see `connectome-workbench.md`)
- **Brainrender:** Python 3D rendering (see `brainrender.md`)
- **FSLeyes:** FSL viewer (see `fsleyes.md`)
- **Matplotlib:** Python plotting (colormap source)
- **NiBabel:** Python neuroimaging I/O

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**PyCortex Version Covered:** 1.3.x
**Maintainer:** Claude Code Neuroimaging Skills
