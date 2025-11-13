# NiBabel

## Overview

NiBabel is the fundamental Python library for reading and writing neuroimaging file formats. It provides access to NIfTI, ANALYZE, MINC, MGH, DICOM, and many other formats, serving as the foundation for most Python neuroimaging tools. NiBabel handles the low-level details of file formats, allowing researchers to focus on data analysis.

**Website:** https://nipy.org/nibabel/
**Platform:** Cross-platform (Linux/macOS/Windows)
**Language:** Python
**License:** MIT

## Key Features

- Read/write NIfTI-1 and NIfTI-2 formats
- DICOM support
- FreeSurfer formats (MGH, surface files, annotations)
- GIFTI surface data
- CIFTI-2 for HCP data
- Tractography formats (TRK, TCK, TRX)
- Affine transformations and coordinate systems
- Memory-mapped array access for large files
- BIDS-compliant file handling

## Installation

```bash
# Using pip
pip install nibabel

# With all optional dependencies
pip install nibabel[all]

# Using conda
conda install -c conda-forge nibabel

# Development version
pip install git+https://github.com/nipy/nibabel.git
```

### Verify Installation

```python
import nibabel as nib
print(nib.__version__)

# Check available formats
print(nib.imageclasses.all_image_classes)
```

## Basic File I/O

### Reading Images

```python
import nibabel as nib

# Load NIfTI file
img = nib.load('brain.nii.gz')

# Get image properties
print(f"Shape: {img.shape}")
print(f"Data type: {img.get_data_dtype()}")
print(f"Affine:\n{img.affine}")
print(f"Header:\n{img.header}")

# Access data array
data = img.get_fdata()  # Recommended: returns float64
# Or
data = img.get_fdata(dtype=np.float32)  # Specify dtype

# For memory-mapped access (doesn't load full array)
dataobj = img.dataobj
slice_data = dataobj[:, :, 50]  # Access single slice
```

### Writing Images

```python
import numpy as np
import nibabel as nib

# Create new image from array
data = np.random.rand(64, 64, 40)
affine = np.eye(4)
img = nib.Nifti1Image(data, affine)

# Save image
nib.save(img, 'output.nii.gz')

# Create from existing image (preserving header)
template = nib.load('template.nii.gz')
new_data = np.zeros_like(template.get_fdata())
new_img = nib.Nifti1Image(new_data, template.affine, template.header)
nib.save(new_img, 'new_image.nii.gz')
```

## Working with Headers

```python
# Access header information
img = nib.load('brain.nii.gz')
header = img.header

# Get voxel dimensions
voxel_sizes = header.get_zooms()
print(f"Voxel sizes: {voxel_sizes} mm")

# Get TR (for 4D fMRI)
tr = header.get_zooms()[3]
print(f"TR: {tr} seconds")

# Modify header
new_header = header.copy()
new_header.set_xyzt_units('mm', 'sec')
new_header['descrip'] = 'Processed data'

# Check header validity
print(header.get_best_affine())

# Get qform and sform
qform = img.get_qform()
sform = img.get_sform()
print(f"Qform code: {img.get_qform(coded=True)[1]}")
print(f"Sform code: {img.get_sform(coded=True)[1]}")
```

## Affine Transformations

```python
import nibabel as nib
import numpy as np

# Understanding the affine matrix
img = nib.load('brain.nii.gz')
affine = img.affine

# Affine maps voxel coordinates (i,j,k) to world coordinates (x,y,z)
# [x]   [affine]   [i]
# [y] = [      ] × [j]
# [z]   [      ]   [k]
# [1]   [      ]   [1]

# Convert voxel to world coordinates
voxel_coords = np.array([32, 32, 20, 1])  # Homogeneous coordinates
world_coords = affine @ voxel_coords
print(f"World coordinates: {world_coords[:3]}")

# Apply transformation
def apply_affine(coords, affine):
    """Apply affine transformation to coordinates."""
    coords_h = np.column_stack([coords, np.ones(len(coords))])
    transformed = coords_h @ affine.T
    return transformed[:, :3]

voxels = np.array([[10, 20, 30], [15, 25, 35]])
world = apply_affine(voxels, affine)
```

## Resampling and Reorientation

```python
from nibabel.processing import resample_to_output, resample_from_to

# Resample to new voxel size
resampled = resample_to_output(img, voxel_sizes=[2.0, 2.0, 2.0])

# Resample to match another image
target_img = nib.load('target.nii.gz')
resampled = resample_from_to(img, target_img, order=3)

# Reorient to canonical orientation (RAS+)
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from nibabel.orientations import apply_orientation

# Get current orientation
orig_ornt = io_orientation(img.affine)
print(f"Original orientation: {nib.aff2axcodes(img.affine)}")

# Target orientation (RAS)
target_ornt = axcodes2ornt('RAS')

# Get transform
transform = ornt_transform(orig_ornt, target_ornt)

# Apply to data
data_reoriented = apply_orientation(img.get_fdata(), transform)

# Create new image
affine_reoriented = img.affine @ nib.orientations.inv_ornt_aff(transform, img.shape)
img_reoriented = nib.Nifti1Image(data_reoriented, affine_reoriented)
```

## Working with 4D fMRI Data

```python
# Load 4D fMRI
fmri_img = nib.load('func.nii.gz')
print(f"Shape: {fmri_img.shape}")  # (64, 64, 40, 200)

# Extract single volume
vol_50 = fmri_img.slicer[:, :, :, 50]
vol_50_img = nib.Nifti1Image(vol_50.get_fdata(), fmri_img.affine)

# Extract time series for a voxel
voxel_ts = fmri_img.get_fdata()[32, 32, 20, :]

# Compute mean over time
mean_data = fmri_img.get_fdata().mean(axis=3)
mean_img = nib.Nifti1Image(mean_data, fmri_img.affine)

# Slice objects for efficient access
from nibabel import funcs
mean_img = funcs.four_to_three(fmri_img)[0]  # First volume

# Concatenate volumes
vol1 = nib.load('run1.nii.gz')
vol2 = nib.load('run2.nii.gz')
concatenated = nib.concat_images([vol1, vol2])
```

## FreeSurfer Files

```python
# Load MGH/MGZ format
mgh_img = nib.load('brain.mgz')
data = mgh_img.get_fdata()

# Load FreeSurfer surface
from nibabel.freesurfer import read_geometry, write_geometry
coords, faces = read_geometry('lh.pial')
print(f"Vertices: {coords.shape}, Faces: {faces.shape}")

# Load annotation (parcellation)
from nibabel.freesurfer import read_annot, write_annot
labels, ctab, names = read_annot('lh.aparc.annot')

# Load morphometry data (thickness, curvature)
from nibabel.freesurfer import read_morph_data, write_morph_data
thickness = read_morph_data('lh.thickness')

# Load label
from nibabel.freesurfer import read_label, write_label
label_array = read_label('lh.precentral.label')
```

## GIFTI Surface Files

```python
# Load GIFTI
gii = nib.load('lh.pial.gii')

# Access data arrays
coords = gii.darrays[0].data  # Coordinates
faces = gii.darrays[1].data   # Triangles

# Load surface data (e.g., thickness)
thickness_gii = nib.load('lh.thickness.gii')
thickness = thickness_gii.darrays[0].data

# Create GIFTI file
from nibabel.gifti import GiftiDataArray, GiftiImage

darray = GiftiDataArray(
    data=thickness,
    intent='NIFTI_INTENT_SHAPE',
    datatype='NIFTI_TYPE_FLOAT32'
)

gii_img = GiftiImage(darrays=[darray])
nib.save(gii_img, 'output.gii')
```

## CIFTI Files (HCP Data)

```python
# Load CIFTI-2 file
cifti = nib.load('tfMRI.dtseries.nii')

# Get brain models (surface + volume)
brain_models = cifti.header.get_axis(1)

# Access data
data = cifti.get_fdata()
print(f"Shape: {data.shape}")  # (timepoints, grayordinates)

# Extract surface data
for bm in brain_models.iter_structures():
    print(f"Structure: {bm}")

# Get specific structure
lh_cortex_data = cifti.get_fdata()[:, brain_models.cortex_left]
```

## Tractography Files

```python
# Load TRK file (TrackVis)
from nibabel.streamlines import load, save

tractogram = load('tractography.trk')
streamlines = tractogram.streamlines
header = tractogram.header

print(f"Number of streamlines: {len(streamlines)}")
print(f"Total points: {sum(len(s) for s in streamlines)}")

# Access individual streamlines
first_streamline = streamlines[0]
print(f"Points in first streamline: {len(first_streamline)}")

# Create new tractogram
from nibabel.streamlines import Tractogram, TrkFile

new_tractogram = Tractogram(
    streamlines=streamlines,
    affine_to_rasmm=np.eye(4)
)

TrkFile(new_tractogram, header=header).save('output.trk')
```

## DICOM Support

```python
# Read DICOM
dcm = nib.load('slice_001.dcm')

# Read DICOM series
from nibabel.nicom import dicomreaders
dcm_img = dicomreaders.mosaic_to_nii('series_001.dcm')

# Get DICOM metadata
import pydicom
ds = pydicom.dcmread('slice_001.dcm')
print(ds.PatientName)
print(ds.StudyDate)
```

## Memory-Efficient Operations

```python
# Memory-mapped access (doesn't load full array into RAM)
img = nib.load('large_file.nii.gz')
data_proxy = img.dataobj

# Access subsets without loading full array
slice_data = data_proxy[:, :, 50]  # Just one slice
roi_data = data_proxy[20:40, 20:40, 20:40]  # Small ROI

# Iterate over volumes in 4D image
fmri = nib.load('func.nii.gz')
for vol_idx in range(fmri.shape[3]):
    vol_data = fmri.dataobj[:, :, :, vol_idx]
    # Process volume
    del vol_data  # Free memory

# Use caching for repeated access
from nibabel.arrayproxy import ArrayProxy
# Set cache size (in bytes)
img.dataobj._cache = ArrayProxy._cache_cls(1000000)
```

## Data Type Handling

```python
# Check data type
img = nib.load('brain.nii.gz')
print(f"Data type: {img.get_data_dtype()}")

# Convert data type
data = img.get_fdata()  # Always returns float64
data_float32 = img.get_fdata(dtype=np.float32)

# Create image with specific dtype
data_int16 = np.random.randint(0, 1000, (64, 64, 40), dtype=np.int16)
img_int16 = nib.Nifti1Image(data_int16, np.eye(4))
print(f"Stored as: {img_int16.get_data_dtype()}")

# Preserve dtype when saving
img_int16.header.set_data_dtype(np.int16)
nib.save(img_int16, 'output_int16.nii.gz')
```

## Coordinate Systems and Spaces

```python
# Check coordinate system codes
img = nib.load('brain.nii.gz')

# Qform (scanner coordinates)
qform, qform_code = img.get_qform(coded=True)
print(f"Qform code: {qform_code}")  # 1=scanner, 2=aligned, 3=talairach, 4=mni

# Sform (standard space coordinates)
sform, sform_code = img.get_sform(coded=True)
print(f"Sform code: {sform_code}")

# Set coordinate system
img.set_qform(qform, code=2)  # Set as aligned
img.set_sform(sform, code=4)  # Set as MNI

# Get axis codes
print(nib.aff2axcodes(img.affine))  # e.g., ('R', 'A', 'S')
```

## Integration with Claude Code

When helping users with NiBabel:

1. **Check Installation:**
   ```python
   import nibabel as nib
   print(nib.__version__)
   ```

2. **Common Issues:**
   - Affine/orientation mismatches
   - Memory errors with large files
   - Incorrect data types causing precision loss
   - Qform/sform confusion

3. **Best Practices:**
   - Always use `get_fdata()` instead of deprecated `get_data()`
   - Check orientation codes before processing
   - Use memory mapping for large files
   - Preserve headers when creating new images
   - Validate affines match when combining images

4. **Memory Tips:**
   - Use `img.dataobj` for memory-mapped access
   - Specify dtype in `get_fdata()` to reduce memory
   - Process data in chunks for large files
   - Delete arrays after use

## Troubleshooting

**Problem:** "MemoryError when loading large files"
**Solution:** Use memory-mapped access with `img.dataobj` instead of `get_fdata()`

**Problem:** "Images don't align after resampling"
**Solution:** Check affine matrices match, verify coordinate system codes

**Problem:** "Data values changed after saving"
**Solution:** Check data type - may need to scale or use appropriate dtype

**Problem:** "Can't read DICOM series"
**Solution:** Install pydicom: `pip install pydicom`

## Resources

- Documentation: https://nipy.org/nibabel/
- API Reference: https://nipy.org/nibabel/reference/
- GitHub: https://github.com/nipy/nibabel
- Coordinate Systems Guide: https://nipy.org/nibabel/coordinate_systems.html
- DICOM Guide: https://nipy.org/nibabel/dicom/dicom_intro.html

## Related Tools

- **Nilearn:** High-level neuroimaging analysis
- **DIPY:** Diffusion imaging (uses NiBabel)
- **Nipype:** Workflows (uses NiBabel)
- **PyDicom:** DICOM file handling
