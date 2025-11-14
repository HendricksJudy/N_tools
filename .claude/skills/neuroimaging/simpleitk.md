# SimpleITK - Simplified Interface to ITK

## Overview

SimpleITK is a simplified programming interface to the Insight Toolkit (ITK), designed to make medical image analysis more accessible to researchers and developers. While ITK is a powerful C++ library for image processing and registration, it has a steep learning curve. SimpleITK provides a simplified API with support for multiple programming languages (Python, R, Java, C#, Lua, Ruby, TCL) while maintaining access to ITK's robust algorithms. It's particularly valuable for rapid prototyping, education, and integrating medical image analysis into Python-based scientific workflows.

**Website:** https://simpleitk.org/
**Platform:** C++/Python/R/Java (Cross-platform)
**License:** Apache 2.0
**Key Application:** Medical image processing, registration, segmentation, filtering

### Why SimpleITK?

- **Easy to learn** - Clean, intuitive API vs. complex ITK templates
- **Multi-language** - Python, R, Java, C# support
- **No compilation** - Install via pip/conda, no C++ build needed
- **Jupyter-friendly** - Works seamlessly in notebooks
- **Production-ready** - Built on battle-tested ITK algorithms
- **Well-documented** - Extensive examples and tutorials
- **Active community** - Regular updates and support

## Key Features

- **Image I/O** - Read/write DICOM, NIfTI, MetaImage, and 30+ formats
- **Registration** - Rigid, affine, deformable (B-spline, demons)
- **Similarity metrics** - Mutual information, correlation, mean squares
- **Optimizers** - Gradient descent, evolutionary, exhaustive search
- **Transforms** - Translation, rotation, affine, B-spline, displacement field
- **Resampling** - Flexible interpolation and spatial transformations
- **Segmentation** - Thresholding, region growing, level sets, watersheds
- **Filtering** - Smoothing, edge detection, morphology, denoising
- **DICOM support** - Read DICOM series, access metadata
- **Elastix integration** - SimpleElastix for advanced registration
- **N-dimensional** - 2D, 3D, 4D, and higher dimensions
- **Efficient** - C++ backend for performance
- **Type-safe** - Automatic pixel type handling
- **GPU support** - Select algorithms with GPU acceleration

## Installation

### Python Installation (Recommended)

```bash
# Install via pip
pip install SimpleITK

# Or with conda
conda install -c conda-forge simpleitk

# For Jupyter notebook support
pip install SimpleITK matplotlib jupyter

# Verify installation
python -c "import SimpleITK as sitk; print(sitk.Version.VersionString())"
```

### Install with Elastix (SimpleElastix)

```bash
# Install pre-built SimpleElastix wheels
pip install SimpleITK-SimpleElastix

# Or build from source for latest features
git clone https://github.com/SuperElastix/SimpleElastix
cd SimpleElastix
mkdir build && cd build
cmake ../SuperBuild
make -j4
```

### R Installation

```r
# Install from CRAN
install.packages("SimpleITK")

# Load library
library(SimpleITK)
```

### Test Installation

```python
import SimpleITK as sitk
import numpy as np

# Create simple test image
img = sitk.GaussianSource(sitk.sitkFloat32, size=[100, 100])
print(f"Created image: {img.GetSize()}")
print(f"SimpleITK version: {sitk.Version.VersionString()}")
```

## Image I/O

### Reading Images

```python
import SimpleITK as sitk

# Read NIfTI file
image = sitk.ReadImage('brain_T1.nii.gz')

# Get image properties
print(f"Size: {image.GetSize()}")
print(f"Spacing: {image.GetSpacing()}")
print(f"Origin: {image.GetOrigin()}")
print(f"Direction: {image.GetDirection()}")
print(f"Pixel type: {image.GetPixelIDTypeAsString()}")

# Read specific pixel type
image_float = sitk.ReadImage('data.nii', sitk.sitkFloat32)
```

### Reading DICOM Series

```python
# Read DICOM series from directory
reader = sitk.ImageSeriesReader()
dicom_dir = '/path/to/dicom/series'

# Get series UIDs
series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
print(f"Found {len(series_IDs)} series")

# Read first series
if series_IDs:
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
    reader.SetFileNames(series_file_names)

    # Optional: Load metadata
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    # Read image
    image = reader.Execute()
    print(f"Loaded DICOM: {image.GetSize()}")

    # Access metadata
    for k in reader.GetMetaDataKeys(0):
        v = reader.GetMetaData(0, k)
        print(f"{k}: {v}")
```

### Writing Images

```python
# Write image
sitk.WriteImage(image, 'output.nii.gz')

# Write with specific pixel type
sitk.WriteImage(sitk.Cast(image, sitk.sitkInt16), 'output_int16.nii.gz')

# Write DICOM series
writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()

for i in range(image.GetDepth()):
    slice_img = image[:, :, i]
    writer.SetFileName(f'output_series/slice_{i:04d}.dcm')
    writer.Execute(slice_img)
```

### Converting to/from NumPy

```python
import numpy as np

# SimpleITK to NumPy
image = sitk.ReadImage('brain.nii')
array = sitk.GetArrayFromImage(image)  # Returns numpy array

# Note: Axes are reversed (ITK: x,y,z; NumPy: z,y,x)
print(f"ITK size: {image.GetSize()}")      # (256, 256, 180)
print(f"NumPy shape: {array.shape}")        # (180, 256, 256)

# NumPy to SimpleITK
new_array = np.random.rand(100, 100, 100)
new_image = sitk.GetImageFromArray(new_array)

# Set spacing and origin
new_image.SetSpacing([1.0, 1.0, 1.0])
new_image.SetOrigin([0.0, 0.0, 0.0])
```

## Basic Image Operations

### Arithmetic Operations

```python
# Element-wise operations
img1 = sitk.ReadImage('image1.nii')
img2 = sitk.ReadImage('image2.nii')

# Addition
sum_img = img1 + img2
sum_img = sitk.Add(img1, img2)

# Subtraction
diff_img = img1 - img2

# Multiplication
product = img1 * img2

# Division (with zero protection)
quotient = sitk.Divide(img1, img2)

# Scalar operations
scaled = img1 * 2.0
offset = img1 + 100
```

### Statistical Operations

```python
# Compute statistics
stats = sitk.StatisticsImageFilter()
stats.Execute(image)

print(f"Mean: {stats.GetMean()}")
print(f"Std Dev: {stats.GetSigma()}")
print(f"Min: {stats.GetMinimum()}")
print(f"Max: {stats.GetMaximum()}")
print(f"Sum: {stats.GetSum()}")

# Label statistics (per-region)
label_img = sitk.ReadImage('segmentation.nii')
label_stats = sitk.LabelStatisticsImageFilter()
label_stats.Execute(image, label_img)

for label in label_stats.GetLabels():
    print(f"Label {label}:")
    print(f"  Mean: {label_stats.GetMean(label)}")
    print(f"  Volume: {label_stats.GetCount(label)}")
```

### Thresholding

```python
# Binary threshold
binary = sitk.BinaryThreshold(image,
                               lowerThreshold=100,
                               upperThreshold=500,
                               insideValue=1,
                               outsideValue=0)

# Otsu thresholding (automatic)
otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)
otsu_filter.SetOutsideValue(1)
binary_otsu = otsu_filter.Execute(image)
threshold_value = otsu_filter.GetThreshold()
print(f"Otsu threshold: {threshold_value}")
```

## Image Filtering

### Smoothing Filters

```python
# Gaussian smoothing
sigma = 2.0  # Standard deviation in physical units (mm)
smooth = sitk.SmoothingRecursiveGaussian(image, sigma)

# Median filter (edge-preserving)
radius = [2, 2, 2]  # Radius in voxels
median = sitk.Median(image, radius)

# Bilateral filter (edge-preserving, intensity-aware)
bilateral = sitk.Bilateral(image,
                           domainSigma=2.0,
                           rangeSigma=50.0)

# Anisotropic diffusion (Perona-Malik)
diffusion = sitk.CurvatureAnisotropicDiffusion(image,
                                               timeStep=0.0625,
                                               conductanceParameter=3.0,
                                               numberOfIterations=5)
```

### Edge Detection

```python
# Canny edge detector
edges = sitk.CannyEdgeDetection(image,
                                lowerThreshold=10,
                                upperThreshold=50,
                                variance=[1.0, 1.0, 1.0])

# Sobel edge detection
sobel = sitk.SobelEdgeDetection(image)

# Gradient magnitude
gradient = sitk.GradientMagnitude(image)
```

### Morphological Operations

```python
# Create structuring element
kernel_radius = [3, 3, 3]
kernel_type = sitk.sitkBall  # or sitkBox, sitkCross

# Erosion (shrink)
eroded = sitk.BinaryErode(binary_image, kernel_radius, kernel_type)

# Dilation (expand)
dilated = sitk.BinaryDilate(binary_image, kernel_radius, kernel_type)

# Opening (erosion then dilation - removes small objects)
opened = sitk.BinaryMorphologicalOpening(binary_image, kernel_radius, kernel_type)

# Closing (dilation then erosion - fills small holes)
closed = sitk.BinaryMorphologicalClosing(binary_image, kernel_radius, kernel_type)

# Fill holes
filled = sitk.BinaryFillhole(binary_image)
```

## Image Registration

### Rigid Registration

```python
import SimpleITK as sitk

# Read images
fixed = sitk.ReadImage('fixed_T1.nii', sitk.sitkFloat32)
moving = sitk.ReadImage('moving_T1.nii', sitk.sitkFloat32)

# Initialize transform
initial_transform = sitk.CenteredTransformInitializer(
    fixed, moving,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

# Setup registration
registration = sitk.ImageRegistrationMethod()

# Similarity metric
registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration.SetMetricSamplingStrategy(registration.RANDOM)
registration.SetMetricSamplingPercentage(0.01)

# Interpolator
registration.SetInterpolator(sitk.sitkLinear)

# Optimizer
registration.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
registration.SetOptimizerScalesFromPhysicalShift()

# Setup transform
registration.SetInitialTransform(initial_transform, inPlace=False)

# Multi-resolution
registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Execute registration
print("Starting registration...")
final_transform = registration.Execute(fixed, moving)

# Print results
print(f"Final metric value: {registration.GetMetricValue()}")
print(f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
print(f"Iterations: {registration.GetOptimizerIteration()}")

# Apply transform
resampled = sitk.Resample(moving, fixed, final_transform,
                          sitk.sitkLinear, 0.0, moving.GetPixelID())

# Save result
sitk.WriteImage(resampled, 'registered.nii.gz')
```

### Affine Registration

```python
# Similar to rigid, but use affine transform
initial_transform = sitk.CenteredTransformInitializer(
    fixed, moving,
    sitk.AffineTransform(3),  # 3D affine
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

# Rest of setup same as rigid registration
registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# ... (same as above)

final_transform = registration.Execute(fixed, moving)
```

### Deformable Registration (B-spline)

```python
# B-spline deformable registration
transformDomainMeshSize = [10] * moving.GetDimension()
initial_transform = sitk.BSplineTransformInitializer(
    fixed,
    transformDomainMeshSize
)

registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsMattesMutualInformation(50)
registration.SetOptimizerAsLBFGSB(
    gradientConvergenceTolerance=1e-5,
    numberOfIterations=100,
    maximumNumberOfCorrections=5,
    maximumNumberOfFunctionEvaluations=1000
)
registration.SetOptimizerScalesFromPhysicalShift()
registration.SetInitialTransform(initial_transform, inPlace=True)

registration.SetShrinkFactorsPerLevel([4, 2, 1])
registration.SetSmoothingSigmasPerLevel([2, 1, 0])

# Execute
final_transform = registration.Execute(fixed, moving)

# Apply
registered = sitk.Resample(moving, fixed, final_transform)
```

### Demons Registration

```python
# Diffeomorphic demons
# Cast to appropriate type
fixed = sitk.Cast(fixed, sitk.sitkFloat32)
moving = sitk.Cast(moving, sitk.sitkFloat32)

# Match histograms first
matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(1024)
matcher.SetNumberOfMatchPoints(7)
matcher.ThresholdAtMeanIntensityOn()
moving_matched = matcher.Execute(moving, fixed)

# Demons filter
demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
demons.SetNumberOfIterations(50)
demons.SetStandardDeviations(1.0)

# Multi-resolution
for level in [4, 2, 1]:
    # Downsample
    fixed_smooth = sitk.SmoothingRecursiveGaussian(fixed, level)
    moving_smooth = sitk.SmoothingRecursiveGaussian(moving_matched, level)

    fixed_ds = sitk.Shrink(fixed_smooth, [level]*3)
    moving_ds = sitk.Shrink(moving_smooth, [level]*3)

    # Register at this level
    displacement_field = demons.Execute(fixed_ds, moving_ds)

    # Upsample for next level
    if level > 1:
        # Continue...
        pass

# Apply deformation
transform = sitk.DisplacementFieldTransform(displacement_field)
registered = sitk.Resample(moving, fixed, transform)
```

## Segmentation

### Region Growing

```python
# Confidence connected segmentation
seed_point = [128, 128, 90]  # x, y, z coordinates

seg = sitk.ConfidenceConnected(image,
                               seedList=[seed_point],
                               numberOfIterations=5,
                               multiplier=2.5,
                               initialNeighborhoodRadius=1,
                               replaceValue=1)

# Or neighborhood connected
seg = sitk.NeighborhoodConnected(image,
                                 seedList=[seed_point],
                                 lower=100,
                                 upper=200,
                                 radius=[2, 2, 2],
                                 replaceValue=1)
```

### Watershed Segmentation

```python
# Gradient-based watershed
# 1. Compute gradient magnitude
gradient = sitk.GradientMagnitude(image)

# 2. Smooth gradient
gradient_smooth = sitk.SmoothingRecursiveGaussian(gradient, sigma=1.0)

# 3. Watershed
watershed = sitk.MorphologicalWatershed(gradient_smooth,
                                        level=0.01,
                                        markWatershedLine=True,
                                        fullyConnected=False)

# 4. Relabel to merge small regions
relabeled = sitk.RelabelComponent(watershed, minimumObjectSize=100)
```

### Level Set Segmentation

```python
# Geodesic active contours
# Initialize with threshold
initial_seg = sitk.BinaryThreshold(image, lowerThreshold=150, upperThreshold=255)

# Distance map initialization
initial_distance_map = sitk.SignedMaurerDistanceMap(
    initial_seg,
    insideIsPositive=True,
    squaredDistance=False,
    useImageSpacing=True
)

# Feature image (edge-based)
feature_image = sitk.GradientMagnitude(image)
feature_image = sitk.SmoothingRecursiveGaussian(feature_image, sigma=1.0)

# Level set filter
level_set = sitk.GeodesicActiveContourLevelSet(
    initial_distance_map,
    feature_image,
    propagationScaling=1.0,
    curvatureScaling=1.0,
    advectionScaling=1.0,
    numberOfIterations=100,
    reverseExpansionDirection=False
)

# Threshold to get segmentation
final_seg = sitk.BinaryThreshold(level_set, lowerThreshold=-1000, upperThreshold=0)
```

### Connected Components

```python
# Find connected components
cc = sitk.ConnectedComponent(binary_image)

# Get statistics for each component
label_stats = sitk.LabelShapeStatisticsImageFilter()
label_stats.Execute(cc)

print(f"Number of components: {label_stats.GetNumberOfLabels()}")

for label in label_stats.GetLabels():
    print(f"Label {label}:")
    print(f"  Volume: {label_stats.GetNumberOfPixels(label)}")
    print(f"  Centroid: {label_stats.GetCentroid(label)}")
    print(f"  Bounding box: {label_stats.GetBoundingBox(label)}")
    print(f"  Roundness: {label_stats.GetRoundness(label)}")
```

## Transforms and Resampling

### Creating Transforms

```python
# Translation
translation = sitk.TranslationTransform(3)
translation.SetOffset([10.0, -5.0, 3.0])

# Rotation (Euler angles)
rotation = sitk.Euler3DTransform()
rotation.SetCenter([128, 128, 90])  # Rotate around center
rotation.SetRotation(0.1, 0.2, 0.0)  # radians around x, y, z

# Affine
affine = sitk.AffineTransform(3)
matrix = [1.0, 0.1, 0.0,
          0.1, 1.0, 0.0,
          0.0, 0.0, 1.0]
affine.SetMatrix(matrix)
affine.SetTranslation([5, 10, 0])

# Composite (chain multiple transforms)
composite = sitk.CompositeTransform(3)
composite.AddTransform(translation)
composite.AddTransform(rotation)
```

### Resampling Images

```python
# Resample moving image to fixed space
resampled = sitk.Resample(
    moving_image,           # Image to resample
    fixed_image,            # Reference (determines output grid)
    transform,              # Transformation
    sitk.sitkLinear,       # Interpolation method
    0.0,                    # Default pixel value
    moving_image.GetPixelID()  # Output pixel type
)

# Custom output grid
resampler = sitk.ResampleImageFilter()
resampler.SetSize([256, 256, 180])
resampler.SetOutputSpacing([1.0, 1.0, 1.0])
resampler.SetOutputOrigin([0.0, 0.0, 0.0])
resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
resampler.SetInterpolator(sitk.sitkBSpline)
resampler.SetTransform(transform)

resampled = resampler.Execute(image)
```

### Interpolation Methods

```python
# Available interpolators:
interpolators = {
    'nearest': sitk.sitkNearestNeighbor,  # For labels
    'linear': sitk.sitkLinear,             # Fast, decent quality
    'bspline': sitk.sitkBSpline,           # Smooth, slower
    'gaussian': sitk.sitkGaussian,         # Very smooth
    'hamming_sinc': sitk.sitkHammingWindowedSinc,  # High quality
    'lanczos': sitk.sitkLanczosWindowedSinc  # High quality
}

# Use for different data types
resampled_labels = sitk.Resample(labels, fixed, transform,
                                 sitk.sitkNearestNeighbor, 0, labels.GetPixelID())

resampled_image = sitk.Resample(image, fixed, transform,
                                sitk.sitkBSpline, 0.0, image.GetPixelID())
```

## Integration with Elastix

### Using SimpleElastix

```python
import SimpleITK as sitk

# Read images
fixed = sitk.ReadImage('fixed.nii')
moving = sitk.ReadImage('moving.nii')

# Create elastix filter
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixed)
elastixImageFilter.SetMovingImage(moving)

# Set parameter maps
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('rigid'))
elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('bspline'))

# Execute
elastixImageFilter.Execute()

# Get result
result_image = elastixImageFilter.GetResultImage()
transform_params = elastixImageFilter.GetTransformParameterMap()

# Save
sitk.WriteImage(result_image, 'registered.nii')

# Apply to other images with transformix
transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetTransformParameterMap(transform_params)
transformixImageFilter.SetMovingImage(moving_label)
result_label = transformixImageFilter.Execute()
```

### Custom Elastix Parameters

```python
# Modify parameter map
param_map = sitk.GetDefaultParameterMap('bspline')
param_map['MaximumNumberOfIterations'] = ['512']
param_map['FinalGridSpacingInPhysicalUnits'] = ['8.0']

elastixImageFilter.SetParameterMap(param_map)
```

## Batch Processing

### Process Multiple Subjects

```python
from pathlib import Path
import SimpleITK as sitk

def register_to_template(fixed_path, moving_path, output_path):
    """Register moving image to fixed template."""

    # Read images
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    # Initialize transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Registration setup
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial_transform)
    registration.SetInterpolator(sitk.sitkLinear)

    # Execute
    final_transform = registration.Execute(fixed, moving)

    # Resample
    resampled = sitk.Resample(moving, fixed, final_transform)

    # Save
    sitk.WriteImage(resampled, str(output_path))

    return registration.GetMetricValue()

# Batch process
template = Path('/data/template/MNI152_T1_1mm.nii.gz')
input_dir = Path('/data/subjects')
output_dir = Path('/data/registered')
output_dir.mkdir(exist_ok=True)

for moving_file in input_dir.glob('sub-*.nii.gz'):
    print(f"Processing {moving_file.name}...")

    output_file = output_dir / moving_file.name

    try:
        metric = register_to_template(template, moving_file, output_file)
        print(f"  Success! Final metric: {metric:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
```

## Visualization

### In Jupyter Notebooks

```python
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# Display 2D slice
def show_slice(image, slice_idx=None, title=''):
    """Display middle slice of 3D image."""

    if image.GetDimension() == 3:
        if slice_idx is None:
            slice_idx = image.GetSize()[2] // 2
        img_slice = image[:, :, slice_idx]
    else:
        img_slice = image

    array = sitk.GetArrayFromImage(img_slice)

    plt.figure(figsize=(8, 8))
    plt.imshow(array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Show image
image = sitk.ReadImage('brain.nii')
show_slice(image, title='Brain MRI')

# Compare before/after
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

slice_idx = fixed.GetSize()[2] // 2

# Original
axes[0].imshow(sitk.GetArrayFromImage(moving[:,:,slice_idx]), cmap='gray')
axes[0].set_title('Moving (Original)')
axes[0].axis('off')

# Registered
axes[1].imshow(sitk.GetArrayFromImage(registered[:,:,slice_idx]), cmap='gray')
axes[1].set_title('Registered')
axes[1].axis('off')

# Fixed
axes[2].imshow(sitk.GetArrayFromImage(fixed[:,:,slice_idx]), cmap='gray')
axes[2].set_title('Fixed (Target)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## Integration with Claude Code

SimpleITK integrates naturally into Claude-assisted workflows:

### Pipeline Development

```markdown
**Prompt to Claude:**
"Create a SimpleITK pipeline that:
1. Reads DICOM series from multiple folders
2. Registers all to a common template using rigid + affine
3. Applies skull stripping using Otsu + morphology
4. Segments white matter lesions using region growing
5. Computes volume and saves statistics to CSV
Include error handling and progress tracking."
```

### Registration Optimization

```markdown
**Prompt to Claude:**
"Optimize SimpleITK registration for pediatric brains:
- Compare MI vs correlation metrics
- Test different optimizer settings
- Evaluate on 5 subjects with manual landmarks
- Report TRE (target registration error)
Provide complete analysis script."
```

### Algorithm Comparison

```markdown
**Prompt to Claude:**
"Compare 3 segmentation methods in SimpleITK:
1. Otsu thresholding + morphology
2. Region growing
3. Watershed
Evaluate on brain lesions with Dice coefficient.
Generate comparison plots and statistical table."
```

## Integration with Other Tools

### NiBabel

```python
import nibabel as nib
import SimpleITK as sitk

# NiBabel to SimpleITK
nib_img = nib.load('brain.nii.gz')
nib_data = nib_img.get_fdata()
nib_affine = nib_img.affine

sitk_img = sitk.GetImageFromArray(nib_data.T)  # Transpose for axis order
sitk_img.SetSpacing(nib_img.header.get_zooms())
# Set origin and direction from affine...

# SimpleITK to NiBabel
sitk_img = sitk.ReadImage('brain.nii.gz')
sitk_data = sitk.GetArrayFromImage(sitk_img).T
sitk_spacing = sitk_img.GetSpacing()

# Create NIfTI
nib_img = nib.Nifti1Image(sitk_data, affine=np.eye(4))
nib.save(nib_img, 'output.nii.gz')
```

### ANTs (via file I/O)

```python
# Use SimpleITK for preprocessing, ANTs for registration
import SimpleITK as sitk
import subprocess

# Preprocess with SimpleITK
img = sitk.ReadImage('raw.nii')
smoothed = sitk.SmoothingRecursiveGaussian(img, 1.0)
sitk.WriteImage(smoothed, 'preprocessed.nii.gz')

# Register with ANTs
subprocess.run([
    'antsRegistrationSyN.sh',
    '-d', '3',
    '-f', 'fixed.nii.gz',
    '-m', 'preprocessed.nii.gz',
    '-o', 'ants_'
])

# Post-process with SimpleITK
registered = sitk.ReadImage('ants_Warped.nii.gz')
# Further processing...
```

## Troubleshooting

### Problem 1: Registration Fails

**Symptoms:** Poor alignment, high metric value

**Solutions:**
```python
# Better initialization
initial_transform = sitk.CenteredTransformInitializer(
    fixed, moving,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.MOMENTS  # Try MOMENTS instead of GEOMETRY
)

# Increase iterations
registration.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=500,  # Increase from 100
    estimateLearningRate=registration.EachIteration
)

# Use more samples
registration.SetMetricSamplingPercentage(0.1)  # 10% instead of 1%
```

### Problem 2: Memory Errors

**Symptoms:** Out of memory with large images

**Solutions:**
```python
# Downsample for registration
def downsample(image, factors=[2, 2, 2]):
    return sitk.Shrink(image, factors)

fixed_ds = downsample(fixed)
moving_ds = downsample(moving)

# Register downsampled
final_transform = registration.Execute(fixed_ds, moving_ds)

# Apply to full resolution
registered_full = sitk.Resample(moving, fixed, final_transform)
```

### Problem 3: Slow Processing

**Symptoms:** Registration takes too long

**Solutions:**
```python
# Reduce sampling
registration.SetMetricSamplingPercentage(0.01)  # 1% of voxels

# Use fewer resolution levels
registration.SetShrinkFactorsPerLevel([4, 2])  # Only 2 levels

# Switch to faster metric
registration.SetMetricAsMeanSquares()  # Faster than MI
```

## Best Practices

1. **Always visualize results** - Check registration quality
2. **Use physical coordinates** - Not voxel indices
3. **Handle exceptions** - Image I/O can fail
4. **Document parameters** - For reproducibility
5. **Test on subset** - Before batch processing
6. **Use appropriate interpolation** - Nearest neighbor for labels, linear/BSpline for intensities
7. **Multi-resolution** - Robust and fast
8. **Initialize transforms** - Better convergence

## Resources

### Official Documentation

- **SimpleITK Website:** https://simpleitk.org/
- **API Documentation:** https://simpleitk.readthedocs.io/
- **Jupyter Notebooks:** https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks
- **GitHub:** https://github.com/SimpleITK/SimpleITK

### Learning Resources

- **Tutorial:** https://simpleitk.readthedocs.io/en/master/fundamentals.html
- **Examples:** https://github.com/SimpleITK/SimpleITK/tree/master/Examples
- **Workshop Materials:** https://github.com/InsightSoftwareConsortium/SimpleITK-MICCAI-2018-Tutorial

### Community

- **Discourse Forum:** https://discourse.itk.org/
- **GitHub Issues:** https://github.com/SimpleITK/SimpleITK/issues

## Citation

```bibtex
@article{Lowekamp2013,
  title = {The Design of SimpleITK},
  author = {Lowekamp, Bradley C and Chen, David T and Ib{\'a}{\~n}ez, Luis and Blezek, Daniel},
  journal = {Frontiers in Neuroinformatics},
  volume = {7},
  pages = {45},
  year = {2013},
  doi = {10.3389/fninf.2013.00045}
}
```

## Related Tools

- **ITK** - Insight Toolkit (underlying C++ library)
- **Elastix** - Advanced registration framework
- **SimpleElastix** - Python interface to elastix
- **ANTs** - Alternative registration framework
- **NiBabel** - Python neuroimaging I/O
- **scikit-image** - General image processing
- **OpenCV** - Computer vision library

---

**Skill Type:** Image Processing & Registration
**Difficulty Level:** Beginner to Intermediate
**Prerequisites:** Python, Basic image processing knowledge
**Typical Use Cases:** Image registration, segmentation, filtering, medical image analysis, research prototyping
