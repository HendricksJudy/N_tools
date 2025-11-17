# NiTransforms - Spatial Transforms for Neuroimaging

## Overview

NiTransforms is a Python library for handling spatial transformations in neuroimaging. Part of the NIPY (Neuroimaging in Python) ecosystem, it provides a unified interface for reading, writing, manipulating, and applying spatial transformations from various neuroimaging tools (AFNI, ANTs, FreeSurfer, FSL, ITK). NiTransforms enables researchers to work with transforms programmatically, compose transformations, convert between formats, and apply them to images and coordinates without relying on external command-line tools.

**Website:** https://github.com/nipy/nitransforms
**Platform:** Python (Cross-platform)
**License:** Apache 2.0
**Key Application:** Transform manipulation, format conversion, coordinate mapping

### Why NiTransforms?

Neuroimaging tools use different transform formats:
- **ANTs:** `.mat` (affine), `.h5` (composite)
- **FSL:** `.mat` (FLIRT affine)
- **FreeSurfer:** `.lta`, `.m3z`
- **ITK:** `.txt`, `.tfm`, `.h5`
- **AFNI:** `.1D`

NiTransforms provides:
- **Unified interface** - Single API for all formats
- **Python integration** - No subprocess calls
- **Format conversion** - Convert between tool formats
- **Programmatic control** - Build pipelines in Python
- **Coordinate transforms** - Map points between spaces

## Key Features

- **Multiple format support** - ANTs, FSL, FreeSurfer, ITK, AFNI
- **Linear transforms** - Affine, rigid, translation
- **Non-linear transforms** - B-spline, displacement fields
- **Transform composition** - Chain multiple transforms
- **Inverse transforms** - Compute and apply inverses
- **Coordinate mapping** - Transform points between spaces
- **Image resampling** - Apply transforms to images
- **Format conversion** - Save in different tool formats
- **NumPy integration** - Work with arrays directly
- **NiBabel compatibility** - Seamless image I/O
- **Lightweight** - Pure Python, minimal dependencies
- **Well-tested** - Extensive test coverage
- **Active development** - Part of NIPY ecosystem

## Installation

### Install via pip

```bash
# Install NiTransforms
pip install nitransforms

# Or install with all dependencies
pip install nitransforms[all]

# For development version
pip install git+https://github.com/nipy/nitransforms.git
```

### Install via conda

```bash
# Create environment with nitransforms
conda create -n neuro python=3.9
conda activate neuro
conda install -c conda-forge nitransforms
```

### Dependencies

```bash
# Core dependencies (installed automatically)
pip install numpy scipy nibabel h5py

# Optional for full functionality
pip install transforms3d  # For advanced transformations
```

### Verify Installation

```python
import nitransforms as nt
print(f"NiTransforms version: {nt.__version__}")

# Test basic functionality
from nitransforms import linear as ntl
affine = ntl.Affine()
print(f"Identity affine:\n{affine.matrix}")
```

## Loading Transforms

### From ANTs

```python
import nitransforms as nt

# Load ANTs affine (.mat)
xfm = nt.load('ants_affine_0GenericAffine.mat')
print(f"Transform type: {type(xfm)}")
print(f"Matrix:\n{xfm.matrix}")

# Load ANTs composite (.h5)
composite = nt.load('ants_composite_Composite.h5')
# Contains affine + deformable components

# Load displacement field
warp = nt.load('ants_1Warp.nii.gz', fmt='ants')
```

### From FSL

```python
# Load FSL FLIRT affine
xfm = nt.load('flirt.mat', fmt='fsl')

# Specify reference space if needed
xfm = nt.load(
    'flirt.mat',
    fmt='fsl',
    reference='target_image.nii.gz',
    moving='source_image.nii.gz'
)
```

### From ITK

```python
# Load ITK transform (.txt, .tfm, .h5)
xfm = nt.load('itk_transform.txt', fmt='itk')

# Load from file object
with open('transform.tfm', 'r') as f:
    xfm = nt.load(f, fmt='itk')
```

### From FreeSurfer

```python
# Load FreeSurfer LTA
xfm = nt.load('register.lta', fmt='fs')

# Load FreeSurfer nonlinear (.m3z)
warp = nt.load('morph.m3z', fmt='fs')
```

### From AFNI

```python
# Load AFNI 1D affine
xfm = nt.load('affine.1D', fmt='afni')
```

## Creating Transforms

### Identity Transform

```python
from nitransforms import linear as ntl

# Create identity affine
identity = ntl.Affine()
print(identity.matrix)
# [[1, 0, 0, 0],
#  [0, 1, 0, 0],
#  [0, 0, 1, 0],
#  [0, 0, 0, 1]]
```

### From Matrix

```python
import numpy as np

# Create affine from 4x4 matrix
matrix = np.array([
    [0.9, 0.1, 0.0, 10.0],
    [-0.1, 0.9, 0.0, 5.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

affine = ntl.Affine(matrix=matrix)
```

### Translation

```python
# Create translation transform
translation = ntl.Affine.from_params(
    'translation',
    translation=[10, 5, -3],  # x, y, z in mm
)
```

### Rotation

```python
# Create rotation (Euler angles in radians)
rotation = ntl.Affine.from_params(
    'rotation',
    angles=[0.1, 0.2, 0.0],  # rotations around x, y, z axes
)

# Or specify center of rotation
rotation_centered = ntl.Affine.from_params(
    'rotation',
    angles=[0.1, 0.2, 0.0],
    center=[128, 128, 90]  # Rotate around this point
)
```

### Scaling

```python
# Create scaling transform
scaling = ntl.Affine.from_params(
    'scaling',
    scales=[1.1, 1.1, 1.0],  # Scale factors for x, y, z
)
```

## Applying Transforms

### Transform Coordinates

```python
import numpy as np

# Points in source space (Nx3 array)
points_source = np.array([
    [100, 100, 50],
    [120, 110, 55],
    [90, 95, 48]
])

# Apply transform
points_target = xfm.map(points_source)

print("Source points:")
print(points_source)
print("\nTransformed points:")
print(points_target)
```

### Transform Images

```python
import nibabel as nib

# Load reference and moving images
reference = nib.load('target.nii.gz')
moving = nib.load('source.nii.gz')

# Apply transform to moving image
resampled = xfm.apply(
    moving,
    reference=reference,
    order=3  # Interpolation order (0=nearest, 1=linear, 3=cubic)
)

# Save result
nib.save(resampled, 'resampled.nii.gz')
```

### Interpolation Options

```python
# Nearest neighbor (for labels)
labels_resampled = xfm.apply(
    moving_labels,
    reference=reference,
    order=0
)

# Linear interpolation
linear_resampled = xfm.apply(moving, reference=reference, order=1)

# Cubic interpolation (smooth, good for anatomical)
cubic_resampled = xfm.apply(moving, reference=reference, order=3)
```

## Transform Composition

### Concatenate Transforms

```python
# Load two transforms
xfm1 = nt.load('rigid.mat', fmt='ants')
xfm2 = nt.load('affine.mat', fmt='ants')

# Compose: apply xfm1 first, then xfm2
composed = nt.TransformChain([xfm1, xfm2])

# Or use operator
composed = xfm2 + xfm1  # Note: right-to-left application

# Apply composed transform
points_final = composed.map(points)
```

### Chain Linear and Non-Linear

```python
# Affine + deformable registration
affine = nt.load('affine.mat', fmt='ants')
warp = nt.load('warp.nii.gz', fmt='ants')

# Chain them
full_transform = nt.TransformChain([affine, warp])

# Apply to image
result = full_transform.apply(moving, reference=reference)
```

## Transform Inversion

### Invert Affine

```python
# Load affine transform
forward = nt.load('forward_affine.mat')

# Compute inverse
inverse = ~forward  # Using ~ operator

# Or explicitly
inverse = forward.inv()

# Apply inverse
original_space = inverse.map(transformed_points)
```

### Invert Non-Linear (Approximate)

```python
# For displacement fields, inversion is approximate
warp_forward = nt.load('forward_warp.nii.gz')

# Compute approximate inverse
warp_inverse = warp_forward.inv()

# Note: May not be exact due to numerical approximation
```

## Format Conversion

### ANTs to FSL

```python
# Load ANTs affine
ants_xfm = nt.load('ants_affine.mat', fmt='ants')

# Save as FSL format
ants_xfm.to_filename('fsl_affine.mat', fmt='fsl')
```

### FSL to ITK

```python
# Load FSL transform
fsl_xfm = nt.load('flirt.mat', fmt='fsl',
                   reference='ref.nii.gz', moving='mov.nii.gz')

# Save as ITK
fsl_xfm.to_filename('itk_transform.txt', fmt='itk')
```

### Any to ANTs

```python
# Load from any supported format
xfm = nt.load('input.mat', fmt='fsl')  # or 'itk', 'fs', etc.

# Save as ANTs
xfm.to_filename('output.mat', fmt='ants')
```

## Integration with Registration Workflows

### ANTs Registration Pipeline

```python
import nibabel as nib
import nitransforms as nt

# After running ANTs registration via command line
# Load all transforms
affine = nt.load('output_0GenericAffine.mat')
warp = nt.load('output_1Warp.nii.gz')

# Combine transforms
full_xfm = nt.TransformChain([affine, warp])

# Apply to additional images
seg_moving = nib.load('segmentation.nii.gz')
reference = nib.load('target.nii.gz')

seg_registered = full_xfm.apply(
    seg_moving,
    reference=reference,
    order=0  # Nearest neighbor for labels
)

nib.save(seg_registered, 'segmentation_registered.nii.gz')
```

### FSL FLIRT Integration

```python
# After FSL FLIRT registration
# Load FLIRT output
xfm = nt.load(
    'flirt_output.mat',
    fmt='fsl',
    reference='target.nii.gz',
    moving='source.nii.gz'
)

# Apply to functional data
func_data = nib.load('functional_4d.nii.gz')

# Transform each volume
import numpy as np
func_array = func_data.get_fdata()
n_vols = func_array.shape[-1]

registered_vols = []
for vol_idx in range(n_vols):
    vol_img = nib.Nifti1Image(
        func_array[..., vol_idx],
        func_data.affine
    )
    vol_registered = xfm.apply(vol_img, reference=reference)
    registered_vols.append(vol_registered.get_fdata())

# Stack into 4D
func_registered = np.stack(registered_vols, axis=-1)
func_registered_img = nib.Nifti1Image(func_registered, reference.affine)
nib.save(func_registered_img, 'func_registered.nii.gz')
```

### FreeSurfer Integration

```python
# Load FreeSurfer registration
lta = nt.load('register.lta', fmt='fs')

# Transform MNI coordinates to subject space
mni_coords = np.array([[0, 0, 0], [10, 20, 30]])  # MNI coordinates
subject_coords = (~lta).map(mni_coords)  # Invert to go from MNI to subject

print("MNI coords:", mni_coords)
print("Subject coords:", subject_coords)
```

## Coordinate System Handling

### RAS vs LPS Conversions

```python
# NiTransforms handles coordinate system conversions
# Most neuroimaging tools use RAS (Right-Anterior-Superior)
# Some use LPS (Left-Posterior-Superior)

# Load transform with explicit coordinate system
xfm = nt.load('transform.mat', fmt='ants')  # ANTs uses LPS internally

# Apply to RAS coordinates (automatic conversion)
ras_coords = np.array([[100, 120, 80]])
transformed = xfm.map(ras_coords)  # NiTransforms handles conversion
```

### Voxel to World Coordinates

```python
import nibabel as nib

# Load image to get affine
img = nib.load('brain.nii.gz')
voxel_to_world = img.affine

# Voxel coordinates
voxel_coords = np.array([[50, 60, 70]])

# Convert to world coordinates
world_coords = nib.affines.apply_affine(voxel_to_world, voxel_coords)

# Apply transform in world space
transformed_world = xfm.map(world_coords)

# Convert back to voxel coordinates (in reference space)
world_to_voxel = np.linalg.inv(reference.affine)
transformed_voxel = nib.affines.apply_affine(world_to_voxel, transformed_world)
```

## Advanced Usage

### Custom Transform Pipelines

```python
from nitransforms import linear as ntl
import numpy as np

# Build multi-step pipeline
# 1. Center image
center_translation = ntl.Affine.from_params(
    'translation',
    translation=[-128, -128, -90]
)

# 2. Rotate 15 degrees around z-axis
rotation = ntl.Affine.from_params(
    'rotation',
    angles=[0, 0, np.radians(15)]
)

# 3. Translate back
uncenter = ntl.Affine.from_params(
    'translation',
    translation=[128, 128, 90]
)

# 4. Additional translation
final_shift = ntl.Affine.from_params(
    'translation',
    translation=[10, 5, 0]
)

# Compose all
pipeline = center_translation + rotation + uncenter + final_shift

# Apply
transformed = pipeline.map(points)
```

### Batch Transform Application

```python
from pathlib import Path
import nibabel as nib

# Load transform
xfm = nt.load('registration_composite.h5')

# Reference space
reference = nib.load('template.nii.gz')

# Apply to multiple images
input_dir = Path('/data/subjects/sub-01/anat')
output_dir = Path('/data/registered/sub-01')
output_dir.mkdir(parents=True, exist_ok=True)

images_to_transform = [
    'T1w.nii.gz',
    'T2w.nii.gz',
    'segmentation.nii.gz'
]

for img_name in images_to_transform:
    print(f"Transforming {img_name}...")

    img = nib.load(input_dir / img_name)

    # Use appropriate interpolation
    if 'segmentation' in img_name:
        order = 0  # Nearest neighbor for labels
    else:
        order = 3  # Cubic for anatomical

    transformed = xfm.apply(img, reference=reference, order=order)

    # Save
    output_path = output_dir / img_name
    nib.save(transformed, output_path)

print("Batch transformation complete!")
```

## Quality Control

### Visualize Transform Effect

```python
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Load images
fixed = nib.load('fixed.nii.gz').get_fdata()
moving = nib.load('moving.nii.gz').get_fdata()
registered = nib.load('registered.nii.gz').get_fdata()

# Select middle slice
slice_idx = fixed.shape[2] // 2

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(moving[:, :, slice_idx].T, cmap='gray', origin='lower')
axes[0].set_title('Moving (Original)')
axes[0].axis('off')

axes[1].imshow(registered[:, :, slice_idx].T, cmap='gray', origin='lower')
axes[1].set_title('Registered')
axes[1].axis('off')

axes[2].imshow(fixed[:, :, slice_idx].T, cmap='gray', origin='lower')
axes[2].set_title('Fixed (Target)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('registration_qc.png')
```

### Check Transform Parameters

```python
# Inspect affine matrix
print("Affine matrix:")
print(xfm.matrix)

# Decompose into components
# (requires transforms3d package)
try:
    from transforms3d import affines
    T, R, Z, S = affines.decompose44(xfm.matrix)
    print(f"\nTranslation: {T}")
    print(f"Rotation matrix:\n{R}")
    print(f"Zoom (scaling): {Z}")
    print(f"Shear: {S}")
except ImportError:
    print("Install transforms3d for decomposition")
```

## Integration with Claude Code

NiTransforms enables Python-based transform workflows:

### Pipeline Automation

```markdown
**Prompt to Claude:**
"Create Python script using NiTransforms that:
1. Loads ANTs registration outputs (affine + warp)
2. Applies to T1w, T2w, FLAIR, and segmentation
3. Uses appropriate interpolation for each
4. Saves in standard space
5. Generates QC images
Include error handling and logging."
```

### Format Conversion Utility

```markdown
**Prompt to Claude:**
"Write NiTransforms utility to convert transforms:
- Input: Directory of various transform formats
- Auto-detect format (ANTs, FSL, ITK)
- Convert all to ANTs format
- Generate conversion report
- Handle errors gracefully
Provide CLI interface."
```

### Coordinate Mapping Tool

```markdown
**Prompt to Claude:**
"Build NiTransforms tool for coordinate transformation:
- Load transformation chain
- Read coordinates from CSV (MNI space)
- Transform to subject space
- Save transformed coordinates
- Support batch processing
Include visualization of point mappings."
```

## Integration with Other Tools

### With NiBabel

```python
import nibabel as nib
import nitransforms as nt

# NiTransforms works seamlessly with NiBabel

# Load images with NiBabel
img1 = nib.load('source.nii.gz')
img2 = nib.load('target.nii.gz')

# Load transform with NiTransforms
xfm = nt.load('transform.mat')

# Apply transform
result = xfm.apply(img1, reference=img2)

# Save with NiBabel
nib.save(result, 'output.nii.gz')
```

### With NumPy

```python
import numpy as np

# Work directly with NumPy arrays
matrix = np.eye(4)
matrix[:3, 3] = [10, 5, 0]  # Translation

# Create transform from matrix
xfm = nt.linear.Affine(matrix=matrix)

# Transform points (NumPy array)
points = np.random.rand(100, 3) * 100
transformed = xfm.map(points)
```

### With Scipy

```python
from scipy.ndimage import affine_transform
import numpy as np

# NiTransforms can export parameters for scipy

# Get inverse matrix for scipy (uses different convention)
matrix_inv = np.linalg.inv(xfm.matrix[:3, :3])
offset = -matrix_inv @ xfm.matrix[:3, 3]

# Apply with scipy
from scipy import ndimage
transformed_array = ndimage.affine_transform(
    image_array,
    matrix_inv,
    offset=offset,
    order=3
)
```

## Troubleshooting

### Problem 1: Format Auto-Detection Fails

**Symptoms:** Cannot load transform file

**Solutions:**
```python
# Explicitly specify format
xfm = nt.load('transform.mat', fmt='ants')  # or 'fsl', 'itk', 'fs'

# For FSL, provide reference images
xfm = nt.load('flirt.mat', fmt='fsl',
              reference='ref.nii.gz',
              moving='mov.nii.gz')
```

### Problem 2: Coordinate System Confusion

**Symptoms:** Transformed coordinates in wrong location

**Solutions:**
```python
# Ensure using world coordinates, not voxel indices
# Convert voxel to world first
img = nib.load('image.nii.gz')
voxel_coords = np.array([[50, 60, 70]])
world_coords = nib.affines.apply_affine(img.affine, voxel_coords)

# Then transform
transformed = xfm.map(world_coords)
```

### Problem 3: Interpolation Artifacts

**Symptoms:** Blocky or blurry resampled images

**Solutions:**
```python
# Use appropriate interpolation order
# For anatomical images: order=3 (cubic)
result = xfm.apply(moving, reference=ref, order=3)

# For labels/segmentations: order=0 (nearest neighbor)
result_seg = xfm.apply(seg, reference=ref, order=0)
```

### Problem 4: Memory Issues with Large Images

**Symptoms:** Out of memory during transform application

**Solutions:**
```python
# Process image in chunks (manual implementation needed)
# Or reduce resolution temporarily
from scipy.ndimage import zoom

# Downsample
downsampled = zoom(image_array, 0.5, order=1)

# Transform
# ... 

# Upsample result
upsampled = zoom(transformed, 2.0, order=3)
```

## Best Practices

1. **Always specify format** - Don't rely on auto-detection for critical pipelines
2. **Use world coordinates** - Not voxel indices for coordinate transforms
3. **Match interpolation to data** - Nearest neighbor for labels, cubic for anatomical
4. **Chain transforms** - Compose before applying to avoid repeated resampling
5. **Test on small data** - Verify transform before batch processing
6. **Document conversions** - Note which tool format you're using
7. **Visual QC** - Always inspect results

## Resources

### Official Documentation

- **GitHub:** https://github.com/nipy/nitransforms
- **Documentation:** https://nitransforms.readthedocs.io/
- **API Reference:** https://nitransforms.readthedocs.io/en/latest/api.html

### Key Publications

- **NIPY Project:** https://nipy.org/
- **NiBabel:** Brett et al. "NiBabel: Access a cacophony of neuro-imaging file formats" (used alongside NiTransforms)

### Learning Resources

- **Examples:** https://github.com/nipy/nitransforms/tree/master/examples
- **Tutorials:** https://nitransforms.readthedocs.io/en/latest/tutorials.html

### Community Support

- **GitHub Issues:** https://github.com/nipy/nitransforms/issues
- **NeuroStars:** https://neurostars.org/ (tag: nitransforms)
- **NIPY Mailing List:** nipy-devel@python.org

## Citation

```bibtex
@software{nitransforms,
  title = {NiTransforms: Spatial transforms for neuroimaging},
  author = {{NiPy Developers}},
  year = {2020},
  url = {https://github.com/nipy/nitransforms},
  note = {Python package}
}
```

## Related Tools

- **NiBabel** - Neuroimaging file I/O (essential companion)
- **ANTs** - Registration framework (generates transforms)
- **FSL** - FMRIB Software Library (FLIRT/FNIRT transforms)
- **SimpleITK** - ITK transforms in Python
- **Nilearn** - Machine learning for neuroimaging
- **DIPY** - Diffusion imaging (includes transform utilities)
- **transforms3d** - General 3D transformations

---

**Skill Type:** Transform Manipulation
**Difficulty Level:** Intermediate
**Prerequisites:** Python, NumPy, Basic neuroimaging knowledge
**Typical Use Cases:** Transform format conversion, coordinate mapping, programmatic transform manipulation, Python-based registration pipelines
