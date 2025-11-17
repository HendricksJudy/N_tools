# ITK-SNAP - Medical Image Segmentation Tool

## Overview

ITK-SNAP is a specialized medical image segmentation software that provides an intuitive interface for manual and semi-automatic segmentation of anatomical structures in 3D medical images. Developed at the Penn Image Computing and Science Laboratory, it offers advanced segmentation algorithms including active contours (snakes), manual painting tools, and support for multi-label segmentations, making it ideal for creating training data, validating automated methods, and detailed anatomical labeling.

**Website:** http://www.itksnap.org/
**Platform:** Windows/macOS/Linux
**Language:** C++ (ITK, VTK, Qt)
**License:** GPL-3.0

## Key Features

- Semi-automatic segmentation with active contours (snakes)
- Manual segmentation with paintbrush tools
- Multi-label segmentation support
- 3D mesh extraction and visualization
- Multi-modal image support with linked cursors
- Automatic and adaptive segmentation preprocessing
- Snake evolution with level sets
- Support for anisotropic voxel spacing
- Undo/redo functionality
- Import/export multiple formats (NIfTI, DICOM, etc.)
- Quantitative measurements (volume, statistics)
- Registration and resampling tools

## Installation

### Windows

```bash
# Download installer from: http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads
# Run .exe installer
# Follow installation wizard
# Launch from Start Menu
```

### macOS

```bash
# Download dmg from website
# Open dmg file
# Drag ITK-SNAP to Applications
# Launch from Applications

# If security warning appears:
# System Preferences → Security & Privacy → Open Anyway
```

### Linux

```bash
# Ubuntu/Debian
# Download .tar.gz from website
tar -xzf itksnap-*.tar.gz
cd itksnap-*
./itksnap

# Create desktop shortcut
sudo cp itksnap /usr/local/bin/

# Or use Snap
sudo snap install itksnap

# Verify installation
itksnap --version
```

## Basic Workflow

### 1. Load Image

```bash
# Command line
itksnap image.nii.gz

# GUI
# File → Open Main Image
# Select NIfTI, DICOM, or other supported format

# Load DICOM series
# File → Open Main Image → DICOM
# Select directory containing DICOM files
# Select series from list
```

### 2. Navigate Image

```
Keyboard shortcuts:
- Arrow keys: Navigate slices
- Scroll wheel: Zoom in/out
- Middle mouse: Pan
- Ctrl + arrow: Move through time (4D)
- +/-: Adjust zoom level
- R: Reset view
- 1,2,3: Toggle axial, coronal, sagittal views

View controls:
- Crosshair: Linked cursor across all views
- Zoom: Individual or linked zoom
- Window/Level: Adjust contrast
```

### 3. Manual Segmentation

```
Basic steps:
1. Segmentation → Manual (Paintbrush mode)
2. Select label from label list
3. Choose brush size and shape
4. Paint on slices
5. Use 3D view to verify
6. Save segmentation

Tools:
- Paintbrush: Paint voxels
- Polygon: Draw polygons, auto-fill
- Annotation: Add labels/markers
- Adaptive brush: Intensity-based painting
```

## Manual Segmentation

### Paintbrush Tool

```
Steps:
1. Segmentation → Paintbrush
2. Label → Select or create label
3. Brush settings:
   - Size: 1-50 voxels
   - Shape: Round or square
   - Mode: Paint or erase
   - Adaptive: On/off (follows intensity)

Shortcuts:
- P: Paintbrush mode
- E: Eraser
- [ ]: Decrease/increase brush size
- Shift + click: Paint line between points
- Ctrl + Z: Undo
- Ctrl + Y: Redo
```

### Polygon Tool

```
Workflow:
1. Segmentation → Polygon
2. Click to place vertices
3. Close polygon (double-click or right-click)
4. Fill polygon automatically
5. Move to next slice and repeat

Advanced:
- Edit vertices after creation
- Copy/paste across slices
- Interpolate between slices
- 3D interpolation between distant slices
```

### 3D Segmentation

```
3D brush:
1. Segmentation → Bubble (3D brush)
2. Click and drag to define sphere
3. Release to apply
4. Useful for volumetric regions

3D editing:
- View segmentation as 3D mesh
- Sculpt mode for mesh editing
- Smooth operations
- Fill holes
```

## Semi-Automatic Segmentation

### Active Contour (Snake) Segmentation

```
Workflow:
1. Preprocess image
   - Segmentation → Automatic Segmentation
   - Select preprocessing method

2. Initialize seeds
   - Draw inside target structure (foreground)
   - Draw outside structure (background)
   - Use different labels for each

3. Run active contour
   - Evolution → Active contour
   - Adjust parameters (balloon force, curvature, etc.)
   - Iterate evolution
   - Accept when satisfied

Parameters:
- Balloon force: Expansion/contraction strength
- Curvature: Smoothness of boundary
- Advection: Image edge attraction
- Iterations: Number of evolution steps
```

### Snake Parameters

```
Key parameters to adjust:

Balloon force (-1 to 1):
- Positive: Expands contour outward
- Negative: Contracts contour inward
- Start with small values (±0.1)

Curvature (0-1):
- Controls smoothness
- Higher = smoother boundary
- Typical: 0.2-0.3

Advection (0-2):
- Edge attraction strength
- Higher = stronger edge adherence
- Typical: 1.0

Propagation weight:
- Image-based speed term
- Depends on preprocessing
```

### Preprocessing for Snakes

```
Options:

1. Edge-based:
   - Segmentation → Preprocessing → Edge
   - Detects boundaries based on gradient
   - Good for high-contrast structures

2. Region-based:
   - Segmentation → Preprocessing → Region Competition
   - Based on intensity statistics
   - Good for uniform regions

3. Threshold:
   - Segmentation → Preprocessing → Thresholding
   - Simple intensity-based
   - Fast, for well-separated structures

4. Cluster:
   - Segmentation → Preprocessing → Clustering
   - K-means or EM clustering
   - Multi-tissue segmentation
```

## Multi-Label Segmentation

### Create Label Description

```
Steps:
1. Segmentation → Labels → Label Editor
2. Add new label
3. Set properties:
   - ID: Unique integer
   - Description: Anatomical name
   - Color: RGB values
   - Visible: Show/hide in 3D
   - Mesh: Export as mesh

Example labels:
1: Background (0,0,0)
2: Gray Matter (128,128,128)
3: White Matter (255,255,255)
4: CSF (0,128,255)
5: Hippocampus (255,128,0)
```

### Load/Save Label Definitions

```bash
# Save label description file
# Segmentation → Labels → Save Label Descriptions
# Saves as .txt file

# Example format:
##################################################
# ITK-SNAP Label Description File
##################################################
    0     0    0    0        0  0  0    "Clear Label"
    1   255    0    0      255  0  0    "Gray Matter"
    2     0  255    0        0  255  0  "White Matter"
    3     0    0  255        0  0  255  "CSF"

# Load existing labels
# Segmentation → Labels → Load Label Descriptions
```

## Image Registration and Resampling

### Image Overlay

```
Add overlays for multi-modal visualization:

1. Segmentation → Overlays → Add Image
2. Select overlay image (e.g., fMRI, PET)
3. Adjust:
   - Opacity: Blend with main image
   - Color map: Hot, cool, jet, etc.
   - Display range: Min/max values
   - Sticky mode: Link to main image transforms

Use cases:
- Overlay functional on structural
- Compare pre/post-treatment
- Multi-sequence MRI (T1, T2, FLAIR)
```

### Image Resampling

```
Workflow:
1. Tools → Resample Image
2. Options:
   - Change voxel size (e.g., 1x1x1mm)
   - Change dimensions
   - Interpolation method (NN, Linear, Cubic)

3. Apply and save

Use for:
- Matching resolutions across modalities
- Creating isotropic voxels
- Downsampling for faster processing
```

### Registration

```
# ITK-SNAP has basic registration via external tools

Using ANTS with ITK-SNAP:
1. Register images externally
   antsRegistrationSyN.sh -d 3 -f fixed.nii.gz -m moving.nii.gz -o output

2. Load transformation in ITK-SNAP
   File → Load Transformation → output0GenericAffine.mat

3. Apply to segmentation
   Segmentation → Transform → Apply Transform

Or use FSL FLIRT:
flirt -in moving.nii.gz -ref fixed.nii.gz -out registered.nii.gz -omat transform.mat
```

## 3D Visualization and Mesh Export

### 3D Mesh View

```
Enable 3D view:
1. View → 3D View
2. Adjust rendering:
   - Transparency: See-through structures
   - Lighting: Adjust light angle
   - Mesh quality: Low, medium, high

Interaction:
- Left drag: Rotate
- Right drag: Zoom
- Middle drag: Pan
- Ctrl + drag: Adjust clipping plane
```

### Export 3D Meshes

```
Export to VTK/STL:
1. Segmentation → Export as Surface Mesh
2. Select label(s) to export
3. Choose format:
   - VTK (.vtk): For Paraview, MeshLab
   - STL (.stl): For 3D printing
   - BYU (.byu): Legacy format

4. Mesh smoothing options:
   - Gaussian smoothing
   - Decimation (reduce polygons)

Use for:
- 3D printing anatomical models
- Advanced visualization (Blender, Paraview)
- Computational modeling (FEA, CFD)
```

## Quantitative Analysis

### Volume Measurements

```
Calculate volumes:
1. Segmentation → Statistics
2. Shows for each label:
   - Number of voxels
   - Volume (mm³, ml)
   - Percentage of image

Export to CSV:
- Save statistics to file
- Import into spreadsheet/R/Python for analysis

Applications:
- Hippocampal volumetry
- Lesion quantification
- Tumor volume tracking
- Atrophy measurements
```

### Intensity Statistics

```
Measure intensity in ROIs:
1. Load image
2. Load segmentation as overlay
3. Segmentation → Statistics → Intensity Statistics

Provides:
- Mean intensity per label
- Standard deviation
- Min/max values
- Volume-weighted statistics

Use for:
- Signal quantification in fMRI/PET
- Tissue characterization
- Quality control
```

## Advanced Features

### Layer Inspector

```
Manage multiple images:
- Tools → Layer Inspector
- View all loaded layers
- Adjust order, visibility, properties
- Link/unlink layers
- Save multi-layer workspace
```

### Automation with Command Line

```bash
# Command-line options
itksnap -g main.nii.gz -s segmentation.nii.gz

# Options:
# -g: Grayscale (main) image
# -s: Segmentation overlay
# -o: Additional overlay
# -l: Label description file
# -w: Load workspace

# Batch processing example
for subj in sub-*; do
    itksnap -g ${subj}/anat/${subj}_T1w.nii.gz \
            -s ${subj}/segmentation/${subj}_seg.nii.gz \
            -l labels.txt
done
```

### Workspace Management

```
Save/load complete workspace:

Save:
- File → Save Workspace
- Saves:
  - All loaded images
  - Segmentation state
  - View settings
  - Label definitions
  - Paths (relative or absolute)

Load:
- File → Open Workspace
- Restores entire session
- Useful for:
  - Resuming work
  - Consistent viewing across subjects
  - Quality control reviews
```

## Best Practices

### Segmentation Strategy

```
Efficient workflow:

1. Start with automatic methods
   - Try thresholding or clustering first
   - Provides initial rough segmentation

2. Refine with semi-automatic snakes
   - Initialize with manual seeds
   - Tune parameters iteratively
   - Work slice-by-slice if needed

3. Manual corrections
   - Fix snake errors with paintbrush
   - Use polygon tool for planar boundaries
   - 3D interpolation between key slices

4. Quality control
   - Review in all three views
   - Check 3D mesh for artifacts
   - Verify boundary accuracy
```

### Reproducibility

```
Document your workflow:
- Save label description files
- Record snake parameters used
- Save workspaces at key stages
- Take screenshots of results
- Export meshes for archival
- Keep processing log

Standard operating procedure:
1. Load standardized label file
2. Apply consistent preprocessing
3. Use same snake parameters across subjects
4. Have second rater review
5. Calculate inter-rater reliability
```

## Integration with Claude Code

When helping users with ITK-SNAP:

1. **Check Installation:**
   ```bash
   which itksnap
   itksnap --version
   ```

2. **Common Issues:**
   - Segmentation not visible (check label opacity/visibility)
   - Snake doesn't evolve (adjust parameters, check preprocessing)
   - Slow performance (reduce image resolution, close 3D view)
   - Can't load DICOM (check series selection, try NIfTI)
   - Undo history limited (save intermediate versions)

3. **Best Practices:**
   - Always work on copies of original data
   - Save frequently (segmentation can be time-consuming)
   - Use consistent label IDs across subjects
   - Test snake parameters on sample slices first
   - Use adaptive brush for intensity-guided segmentation
   - Verify segmentation in 3D view
   - Export meshes for backup and visualization

4. **Performance Tips:**
   - Downsample large images for initial segmentation
   - Close unused views and overlays
   - Use polygon tool for large planar regions
   - Work in smaller ROIs when possible
   - Batch similar structures together

## Troubleshooting

**Problem:** Snake doesn't evolve or evolves incorrectly
**Solution:** Adjust preprocessing method, modify balloon force, add more seeds, or increase iterations

**Problem:** Can't see segmentation overlay
**Solution:** Check Labels panel, ensure label is visible, adjust opacity in 3D view settings

**Problem:** Segmentation leaks into adjacent structures
**Solution:** Add background seeds, reduce balloon force, increase curvature for smoother boundaries

**Problem:** Slow performance with large images
**Solution:** Resample to lower resolution, work on subregions, close 3D view during editing

**Problem:** Can't load DICOM series
**Solution:** Convert to NIfTI first using dcm2niix, or try File → Open DICOM → Select correct series

## Resources

- Website: http://www.itksnap.org/
- Documentation: http://www.itksnap.org/pmwiki/pmwiki.php?n=Documentation.SNAP3
- Tutorials: http://www.itksnap.org/pmwiki/pmwiki.php?n=Documentation.TutorialSectionIntroduction
- User Forum: https://sourceforge.net/p/itk-snap/discussion/
- YouTube: ITK-SNAP Tutorial Videos
- GitHub: https://github.com/pyushkevich/itksnap

## Citation

```bibtex
@article{yushkevich2006itksnap,
  title={User-guided 3D active contour segmentation of anatomical structures: significantly improved efficiency and reliability},
  author={Yushkevich, Paul A and Piven, Joseph and Hazlett, Heather Cody and Smith, Rachel Gimpel and Ho, Sean and Gee, James C and Gerig, Guido},
  journal={Neuroimage},
  volume={31},
  number={3},
  pages={1116--1128},
  year={2006},
  publisher={Elsevier}
}
```

## Related Tools

- **3D Slicer:** Comprehensive medical imaging platform
- **FSLeyes:** FSL's image viewer with editing
- **MRIcron/MRIcroGL:** Lightweight viewers with ROI tools
- **MITK:** Medical imaging toolkit
- **FreeSurfer:** Automated cortical segmentation
- **ANTs:** Advanced registration and segmentation
