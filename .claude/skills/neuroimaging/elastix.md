# Elastix - Image Registration Toolbox

## Overview

Elastix is a powerful, open-source toolbox for intensity-based medical image registration. Developed at the Leiden University Medical Center (LUMC), it implements a wide variety of registration algorithms including rigid, affine, and non-rigid (B-spline, diffeomorphic) transformations. Elastix uses a parameter file-based approach that makes it highly flexible and reproducible, supporting numerous image similarity metrics, optimizers, and transformation models. Its companion tool Transformix applies computed transformations to images and point sets.

**Website:** https://elastix.lumc.nl/
**Platform:** C++ (Linux/macOS/Windows)
**License:** Apache 2.0
**Key Application:** Medical image registration, atlas construction, motion correction

### Registration Concepts

Image registration finds the spatial transformation that aligns two images:
- **Fixed image:** Target/reference image (stays in place)
- **Moving image:** Image to be transformed (warped to match fixed)
- **Transformation:** Spatial mapping from moving to fixed space
- **Metric:** Similarity measure (mutual information, correlation, etc.)
- **Optimizer:** Algorithm to find optimal transformation
- **Interpolator:** Method for sampling moving image

Elastix supports hierarchical multi-resolution registration for robustness and speed.

## Key Features

- **Multiple transformation models** - Rigid, affine, B-spline, diffeomorphic
- **Extensive similarity metrics** - Mutual information, normalized correlation, sum of squared differences
- **Advanced optimizers** - Adaptive stochastic gradient descent (ASGD), quasi-Newton methods
- **Multi-resolution framework** - Pyramidal approach for robustness
- **Parameter file system** - Highly configurable, reproducible
- **Automatic parameter selection** - Default parameter sets for common tasks
- **Multi-metric registration** - Combine multiple similarity measures
- **Groupwise registration** - Register multiple images simultaneously
- **Point-based registration** - Incorporate landmark constraints
- **Transformix tool** - Apply transformations to images and points
- **Mask support** - Define regions of interest
- **Command-line interface** - Easy scripting and batch processing
- **Cross-platform** - Windows, Linux, macOS binaries
- **Well-documented** - Comprehensive manual and examples
- **Actively maintained** - Regular updates and community support

## Installation

### Binary Installation (Recommended)

**Linux:**
```bash
# Download latest release
cd ~/software
wget https://github.com/SuperElastix/elastix/releases/download/5.1.0/elastix-5.1.0-Linux.tar.bz2

# Extract
tar -xf elastix-5.1.0-Linux.tar.bz2
cd elastix-5.1.0-Linux

# Add to PATH
echo 'export PATH=$HOME/software/elastix-5.1.0-Linux/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**macOS:**
```bash
# Download macOS release
cd ~/software
curl -L -O https://github.com/SuperElastix/elastix/releases/download/5.1.0/elastix-5.1.0-mac.zip
unzip elastix-5.1.0-mac.zip

# Add to PATH
echo 'export PATH=$HOME/software/elastix-5.1.0-mac/bin:$PATH' >> ~/.bash_profile
source ~/.bash_profile
```

**Windows:**
```cmd
# Download Windows installer from:
# https://github.com/SuperElastix/elastix/releases
# Run the installer and add to PATH
```

### Verify Installation

```bash
# Check elastix version
elastix --version

# Check transformix
transformix --version

# Should output version 5.x.x
```

### Python Interface (SimpleElastix)

```bash
# Install SimpleElastix for Python access
pip install SimpleITK-SimpleElastix

# Or build from source for latest features
```

## Basic Registration

### Rigid Registration

Align images using translation and rotation only:

```bash
# Simple rigid registration
elastix \
  -f fixed_image.nii \
  -m moving_image.nii \
  -out output_rigid/ \
  -p parameters_Rigid.txt

# Output files in output_rigid/:
# - result.0.nii.gz (registered moving image)
# - TransformParameters.0.txt (transformation)
```

### Default Parameter File (Rigid)

```bash
# Use built-in parameter sets
# Download from: https://elastix.lumc.nl/modelzoo/

# Or create parameters_Rigid.txt:
cat > parameters_Rigid.txt << 'EOF'
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

// Main components
(Registration "MultiResolutionRegistration")
(Transform "EulerTransform")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "AdaptiveStochasticGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")

// Pyramid settings
(NumberOfResolutions 4)
(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1)

// Optimizer settings
(MaximumNumberOfIterations 500)
(AutomaticParameterEstimation "true")
(AutomaticScalesEstimation "true")

// Output settings
(WriteResultImage "true")
(ResultImagePixelType "short")
(ResultImageFormat "nii.gz")
EOF

# Run with custom parameters
elastix -f fixed.nii -m moving.nii -out output/ -p parameters_Rigid.txt
```

### Affine Registration

Add scaling and shearing:

```bash
# Affine transformation (12 DOF in 3D)
elastix \
  -f fixed_image.nii \
  -m moving_image.nii \
  -out output_affine/ \
  -p parameters_Affine.txt

# Affine after rigid (two-stage)
elastix \
  -f fixed_image.nii \
  -m moving_image.nii \
  -out output_twoStage/ \
  -p parameters_Rigid.txt \
  -p parameters_Affine.txt
```

### Non-Rigid (B-spline) Registration

Deformable registration:

```bash
# Three-stage registration: rigid → affine → B-spline
elastix \
  -f fixed_image.nii \
  -m moving_image.nii \
  -out output_nonrigid/ \
  -p par_Rigid.txt \
  -p par_Affine.txt \
  -p par_BSpline.txt

# Results:
# - result.0.nii.gz (after rigid)
# - result.1.nii.gz (after affine)
# - result.2.nii.gz (after B-spline, final)
# - TransformParameters.0/1/2.txt
```

## Parameter Files in Detail

### Key Parameters

```bash
# Essential transformation parameters

// Transform type
(Transform "EulerTransform")        // Rigid (translation + rotation)
(Transform "AffineTransform")       // Affine (12 DOF)
(Transform "BSplineTransform")      // Non-rigid B-spline
(Transform "SplineKernelTransform") // Thin-plate spline
(Transform "WeightedCombinationTransform") // Combined transforms

// Similarity metric
(Metric "AdvancedMattesMutualInformation")  // For multi-modal (T1-T2, MRI-CT)
(Metric "AdvancedNormalizedCorrelation")    // For same modality
(Metric "AdvancedMeanSquares")              // Sum of squared differences

// Optimizer
(Optimizer "AdaptiveStochasticGradientDescent")  // Robust, adaptive
(Optimizer "StandardGradientDescent")             // Classic gradient descent
(Optimizer "ConjugateGradient")                   // Conjugate gradient

// Multi-resolution
(NumberOfResolutions 4)
(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1)  // Downsampling factors
```

### B-spline Parameters

```bash
# Deformable registration settings
cat > parameters_BSpline.txt << 'EOF'
(Transform "BSplineTransform")
(FinalGridSpacingInPhysicalUnits 16)  // Control point spacing (mm)

// For finer deformation:
(FinalGridSpacingInPhysicalUnits 8)   // Smaller = more flexible

// Grid spacing schedule (per resolution)
(GridSpacingSchedule 4 2 1)

// B-spline order (3 = cubic, smooth)
(BSplineTransformSplineOrder 3)

// Regularization
(Metric0Weight 1.0)                   // Similarity term
(Metric1Weight 0.1)                   // Regularization (if using bending energy)
EOF
```

## Using Masks

### Fixed Image Mask

Restrict registration to specific region:

```bash
# Create binary mask (1 = include, 0 = exclude)
# E.g., brain mask excluding skull

# Use mask in registration
elastix \
  -f fixed_brain.nii \
  -m moving_brain.nii \
  -fMask fixed_brain_mask.nii \
  -out output/ \
  -p parameters.txt

# Only voxels where mask = 1 contribute to metric
```

### Moving Image Mask

```bash
# Mask on moving image
elastix \
  -f fixed.nii \
  -m moving.nii \
  -mMask moving_mask.nii \
  -out output/ \
  -p parameters.txt

# Use both masks
elastix \
  -f fixed.nii -m moving.nii \
  -fMask fixed_mask.nii \
  -mMask moving_mask.nii \
  -out output/ -p parameters.txt
```

## Transformix - Applying Transformations

### Transform Image

Apply computed transformation to new image:

```bash
# After registration with elastix
# Apply transformation to another image (e.g., segmentation)

transformix \
  -in moving_segmentation.nii \
  -out output_transformed/ \
  -tp output/TransformParameters.0.txt

# Output: result.nii.gz (transformed segmentation)
```

### Transform Multiple Images

```bash
# Transform several images with same transformation
for img in moving_t1.nii moving_t2.nii moving_label.nii; do
    transformix \
      -in ${img} \
      -out transformed_$(basename ${img} .nii)/ \
      -tp TransformParameters.0.txt
done
```

### Transform Point Sets

```bash
# Create input points file: inputpoints.txt
# Format:
# point
# N_points
# x1 y1 z1
# x2 y2 z2
# ...

cat > inputpoints.txt << EOF
point
3
50.0 100.0 80.0
60.0 110.0 85.0
55.0 105.0 82.0
EOF

# Transform points
transformix \
  -def inputpoints.txt \
  -out output/ \
  -tp TransformParameters.0.txt

# Output: outputpoints.txt (transformed coordinates)
```

### Generate Deformation Field

```bash
# Create deformation field for visualization
transformix \
  -def all \
  -out deformation_field/ \
  -tp TransformParameters.0.txt

# Output: deformationField.nii.gz (vector field)
# Can visualize in ITK-SNAP, 3D Slicer, etc.
```

## Advanced Features

### Multi-Metric Registration

Combine multiple similarity measures:

```bash
# Use both mutual information and bending energy
cat > parameters_MultiMetric.txt << 'EOF'
(Registration "MultiMetricMultiResolutionRegistration")
(Transform "BSplineTransform")

// Multiple metrics
(Metric "AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty")
(Metric0Weight 1.0)   // Data term
(Metric1Weight 0.5)   // Regularization term

(Optimizer "AdaptiveStochasticGradientDescent")
// ... rest of parameters
EOF
```

### Groupwise Registration

Register multiple images simultaneously:

```bash
# Create average template from multiple subjects
# Useful for atlas construction

# Requires specific parameter file for groupwise
# See elastix manual section 6.1.14

elastix \
  -f0 subject01.nii \
  -f1 subject02.nii \
  -f2 subject03.nii \
  -m0 subject01.nii \
  -m1 subject02.nii \
  -m2 subject03.nii \
  -out groupwise_output/ \
  -p parameters_Groupwise.txt
```

### Point-based Registration

Incorporate landmark constraints:

```bash
# Create corresponding points file
# fixedPoints.txt and movingPoints.txt

cat > parameters_Landmark.txt << 'EOF'
(Registration "MultiResolutionRegistration")
(Transform "BSplineTransform")
(Metric "CorrespondingPointsEuclideanDistanceMetric")

// Point sets
(FixedImagePointSetName "fixedPoints.txt")
(MovingImagePointSetName "movingPoints.txt")
EOF

elastix \
  -f fixed.nii -m moving.nii \
  -out output/ \
  -p parameters_Landmark.txt
```

### Initial Transform

Start from existing transformation:

```bash
# First registration
elastix -f fixed.nii -m moving.nii -out stage1/ -p par_Rigid.txt

# Second registration using first as initialization
elastix \
  -f fixed.nii \
  -m moving.nii \
  -out stage2/ \
  -p par_BSpline.txt \
  -t0 stage1/TransformParameters.0.txt

# Useful for:
# - Multi-stage refinement
# - Template-based initialization
# - Prior knowledge integration
```

## Metrics and Optimizers

### Similarity Metrics

**Mutual Information (MI):**
```bash
(Metric "AdvancedMattesMutualInformation")
(NumberOfHistogramBins 32)  # Typical: 32-64

# Best for:
# - Multi-modal registration (T1-T2, MRI-CT)
# - Robust to intensity differences
```

**Normalized Correlation:**
```bash
(Metric "AdvancedNormalizedCorrelation")

# Best for:
# - Same modality (T1-T1, T2-T2)
# - Faster than MI
# - Assumes linear intensity relationship
```

**Mean Squares:**
```bash
(Metric "AdvancedMeanSquares")

# Best for:
# - Same modality with similar intensity
# - Fastest
# - Requires good initialization
```

### Optimizer Settings

```bash
# Adaptive Stochastic Gradient Descent (recommended)
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations 500)
(AutomaticParameterEstimation "true")
(AutomaticScalesEstimation "true")
(NumberOfSpatialSamples 2048)  # Random samples per iteration

# Increase samples for better accuracy (slower)
(NumberOfSpatialSamples 8192)

# Maximum step length
(MaximumStepLength 1.0)  # Adjust based on image scale
```

## Multi-Resolution Strategy

### Pyramid Schedules

```bash
# 4-level pyramid
(NumberOfResolutions 4)

# Schedule defines downsampling at each level
(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1)
# Level 0: 8× downsampled (coarse)
# Level 1: 4× downsampled
# Level 2: 2× downsampled
# Level 3: Full resolution (fine)

# Smoothing at each level
(ImagePyramidSmoothingSigmaSchedule 4 2 1 0)  # Gaussian sigma in voxels

# Iterations per level
(MaximumNumberOfIterations 500 500 500 500)

# Or adaptive based on level
(MaximumNumberOfIterations 250 250 500 1000)  # More at finer levels
```

## Batch Processing

### Process Multiple Subject Pairs

```bash
#!/bin/bash
# batch_elastix.sh

# Input directories
fixed_dir="/data/templates"
moving_dir="/data/subjects"
output_base="/data/registered"

# Fixed template
template="${fixed_dir}/MNI152_T1_1mm.nii.gz"

# Loop through subjects
for moving in ${moving_dir}/sub-*.nii.gz; do
    subj=$(basename ${moving} .nii.gz)
    echo "Registering ${subj}..."

    outdir="${output_base}/${subj}"
    mkdir -p ${outdir}

    # Three-stage registration
    elastix \
      -f ${template} \
      -m ${moving} \
      -out ${outdir} \
      -p parameters_Rigid.txt \
      -p parameters_Affine.txt \
      -p parameters_BSpline.txt \
      -threads 4

    echo "${subj} complete"
done
```

### Parallel Processing with GNU Parallel

```bash
# Create list of subject pairs
ls /data/subjects/*.nii.gz > moving_list.txt

# Process in parallel (4 jobs at once)
parallel -j 4 --colsep ' ' \
  'elastix -f /data/template.nii -m {} -out /data/output/{/.}/ -p par.txt' \
  :::: moving_list.txt
```

### Python Wrapper Script

```python
import os
import subprocess
from pathlib import Path

def run_elastix(fixed, moving, output_dir, param_files):
    """Run elastix registration."""

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        'elastix',
        '-f', str(fixed),
        '-m', str(moving),
        '-out', str(output_dir),
    ]

    # Add parameter files
    for param in param_files:
        cmd.extend(['-p', str(param)])

    # Run
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Return path to final result
    return output_dir / 'result.2.nii.gz'  # Assuming 3 stages

# Example usage
fixed_img = '/data/template/MNI152.nii'
moving_img = '/data/subjects/sub-01_T1w.nii'
output = '/data/registered/sub-01'
params = ['par_Rigid.txt', 'par_Affine.txt', 'par_BSpline.txt']

registered = run_elastix(fixed_img, moving_img, output, params)
print(f"Registered image: {registered}")
```

## Quality Control

### Visual Inspection

```bash
# Check registration quality
# Use ITK-SNAP, FSLeyes, or similar

# Create checkerboard for comparison
# (Requires ImageMagick or Python)

# Or use elastix's built-in metric value
grep "Final metric value" output/elastix.log
# Lower is better for most metrics
```

### Quantitative Assessment

```python
import nibabel as nib
import numpy as np
from scipy import ndimage

def compute_dice(seg1, seg2):
    """Compute Dice coefficient between binary segmentations."""
    intersection = np.sum(seg1 * seg2)
    dice = 2 * intersection / (np.sum(seg1) + np.sum(seg2))
    return dice

# Load segmentations
fixed_seg = nib.load('fixed_seg.nii').get_fdata()
moving_seg_transformed = nib.load('result_seg.nii').get_fdata()

# Compute overlap
dice = compute_dice(fixed_seg > 0, moving_seg_transformed > 0)
print(f"Dice coefficient: {dice:.3f}")

# Dice > 0.7 typically indicates good registration
```

### Deformation Field Analysis

```bash
# Generate deformation field
transformix -def all -out defField/ -tp TransformParameters.0.txt

# Analyze deformation magnitude
# Large deformations may indicate overfitting or poor registration
```

## Integration with Claude Code

Elastix integrates well with Claude-assisted neuroimaging workflows:

### Pipeline Automation

```markdown
**Prompt to Claude:**
"Create an elastix pipeline for registering 100 subjects to MNI space:
1. Rigid → affine → B-spline registration
2. Apply transforms to T1w, T2w, and segmentation
3. Compute registration quality metrics (Dice, Jacobian)
4. Generate QC report with before/after images
5. Use GNU parallel for speed
Include error handling and logging."
```

### Parameter Optimization

```markdown
**Prompt to Claude:**
"Optimize elastix parameters for registering pediatric brains (age 6-12):
- Starting from default B-spline parameters
- Adjust for smaller brain size
- Increase regularization to prevent overfitting
- Test on 5 subjects and report metrics
Suggest optimal NumberOfSpatialSamples and FinalGridSpacing."
```

### Multi-Modal Registration

```markdown
**Prompt to Claude:**
"Set up elastix for T1w to T2w FLAIR registration:
- Choose appropriate metric (MI vs correlation)
- Configure multi-resolution pyramid
- Handle different intensity ranges
- Apply transform to lesion masks
Provide complete parameter file and bash script."
```

## Integration with Other Tools

### ANTs

Compare or combine with ANTs registration:

```bash
# ANTs registration for comparison
antsRegistrationSyN.sh \
  -d 3 \
  -f fixed.nii \
  -m moving.nii \
  -o ants_

# Compare Elastix vs ANTs results
# Both are excellent - Elastix offers more control via parameters
```

### FSL

```bash
# Use FSL-preprocessed images
# Brain extract first
bet moving_T1.nii moving_brain.nii -f 0.5

# Register with elastix
elastix -f template_brain.nii -m moving_brain.nii -out reg/ -p par.txt

# Convert elastix to FSL format (if needed)
# Note: Use transformix, not FSL tools, to apply elastix transforms
```

### FreeSurfer

```bash
# Register to FreeSurfer space

# 1. Extract brain from FreeSurfer
mri_convert $SUBJECTS_DIR/subject01/mri/brain.mgz brain_fs.nii.gz

# 2. Register external image to FreeSurfer
elastix \
  -f brain_fs.nii.gz \
  -m external_T1.nii \
  -out to_freesurfer/ \
  -p par_Rigid.txt par_Affine.txt

# 3. Apply to functional data
transformix \
  -in functional_data.nii \
  -out func_to_fs/ \
  -tp to_freesurfer/TransformParameters.1.txt
```

### Python (SimpleITK/SimpleElastix)

```python
import SimpleITK as sitk

# Read images
fixed = sitk.ReadImage('fixed.nii')
moving = sitk.ReadImage('moving.nii')

# Set up elastix filter
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixed)
elastixImageFilter.SetMovingImage(moving)

# Use default parameter maps
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('rigid'))
elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('bspline'))

# Execute
elastixImageFilter.Execute()

# Get result
result = elastixImageFilter.GetResultImage()
sitk.WriteImage(result, 'registered.nii')

# Get transform parameters
transform_params = elastixImageFilter.GetTransformParameterMap()
```

## Troubleshooting

### Problem 1: Registration Fails to Converge

**Symptoms:** Large metric values, poor alignment

**Solutions:**
```bash
# Increase iterations
(MaximumNumberOfIterations 2000)

# Use more spatial samples
(NumberOfSpatialSamples 8192)

# Better initialization
# Manually align first, or use center of mass alignment:
(AutomaticTransformInitialization "true")
(AutomaticTransformInitializationMethod "CenterOfGravity")

# Reduce step size
(MaximumStepLength 0.5)
```

### Problem 2: Over-Regularization or Under-Regularization

**Symptoms:** Too smooth (over) or checkerboard artifacts (under)

**Solutions:**
```bash
# For over-regularization (too smooth):
# Decrease grid spacing
(FinalGridSpacingInPhysicalUnits 8)  # Was 16

# Reduce bending energy weight
(Metric1Weight 0.01)  # Was 0.1

# For under-regularization (artifacts):
# Increase grid spacing
(FinalGridSpacingInPhysicalUnits 20)

# Add or increase bending energy
(Metric "AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty")
(Metric1Weight 1.0)
```

### Problem 3: Out of Memory

**Symptoms:** Process killed, memory errors

**Solutions:**
```bash
# Reduce spatial samples
(NumberOfSpatialSamples 2048)  # Was 8192

# Use fewer resolutions
(NumberOfResolutions 3)  # Was 4

# Process in lower resolution
# Downsample images first:
c3d input.nii -resample 50% -o input_downsampled.nii
```

### Problem 4: Wrong Final Image Size

**Symptoms:** Registered image has different dimensions than fixed

**Solutions:**
```bash
# In parameter file, ensure:
(UseDirectionCosines "true")

# Manually set output size/spacing to match fixed:
# (Size X Y Z)
# (Spacing sx sy sz)

# Or let elastix inherit from fixed image (default)
```

### Problem 5: Multi-Modal Registration Poor Results

**Symptoms:** T1-T2 or MRI-CT registration fails

**Solutions:**
```bash
# Use mutual information
(Metric "AdvancedMattesMutualInformation")
(NumberOfHistogramBins 64)  # Try 32, 64, or 128

# Increase samples for more robust MI estimation
(NumberOfSpatialSamples 4096)

# Use masks to exclude non-brain regions
elastix -f fixed.nii -m moving.nii \
  -fMask brain_mask.nii -mMask brain_mask.nii \
  -out output/ -p par.txt
```

### Problem 6: Transformix Produces Black Image

**Symptoms:** Transformed image is all zeros

**Solutions:**
```bash
# Check transform parameter file path
transformix -in image.nii -out output/ -tp TransformParameters.0.txt

# Ensure using FINAL transform parameters (highest number)
transformix -tp TransformParameters.2.txt  # Not 0.txt

# Check interpolator
# In TransformParameters.txt:
(ResampleInterpolator "FinalBSplineInterpolator")

# For binary masks, use nearest neighbor:
(ResampleInterpolator "FinalNearestNeighborInterpolator")
```

## Best Practices

### Parameter Selection

1. **Start with defaults** - Use parameter database examples
2. **Multi-stage registration** - Rigid → affine → B-spline typical
3. **Use masks** - Improve accuracy and speed
4. **Match metrics to modality** - MI for multi-modal, correlation for same modality
5. **Test on subset** - Optimize parameters on few subjects first

### Registration Workflow

1. **Preprocessing** - Bias correction, skull stripping before registration
2. **Visual QC** - Always inspect results
3. **Quantitative validation** - Use Dice or other overlap metrics
4. **Document parameters** - Save parameter files with results
5. **Version control** - Track elastix version and parameters

### Performance

1. **Use multi-resolution** - Faster and more robust
2. **Parallel processing** - Multiple subjects simultaneously
3. **Appropriate spatial samples** - 2048-4096 usually sufficient
4. **Reduce resolution if needed** - For very large images
5. **GPU acceleration** - Not available in elastix; consider ANTs if needed

### Quality Assurance

1. **Check metric convergence** - Plot from elastix.log
2. **Inspect deformation fields** - Look for unrealistic warps
3. **Validate on landmarks** - If available
4. **Compare methods** - Cross-validate with ANTs/FSL
5. **Clinical review** - For medical applications

## Resources

### Official Documentation

- **Elastix Website:** https://elastix.lumc.nl/
- **Manual:** https://elastix.lumc.nl/download/elastix-5.0.1-manual.pdf
- **Parameter Database:** https://elastix.lumc.nl/modelzoo/
- **GitHub:** https://github.com/SuperElastix/elastix
- **FAQ:** https://elastix.lumc.nl/faq.php

### Key Publications

- **Elastix:** Klein et al. (2010) "elastix: A toolbox for intensity-based medical image registration" IEEE TMI
- **ASGD Optimizer:** Klein et al. (2009) "Adaptive stochastic gradient descent optimisation for image registration" IJCV
- **Evaluation:** Klein et al. (2009) "Evaluation of 14 nonlinear deformation algorithms" Medical Image Analysis

### Learning Resources

- **Tutorial:** https://elastix.lumc.nl/doxygen/index.html
- **Example Scripts:** https://github.com/SuperElastix/elastix/tree/main/examples
- **Workshops:** Regular at MICCAI and SPIE Medical Imaging

### Community Support

- **Forum:** https://groups.google.com/g/elastix-imageregistration
- **GitHub Issues:** https://github.com/SuperElastix/elastix/issues
- **Stack Overflow:** Questions tagged 'elastix'

## Citation

```bibtex
@article{Klein2010,
  title = {elastix: A toolbox for intensity-based medical image registration},
  author = {Klein, Stefan and Staring, Marius and Murphy, Keelin and
            Viergever, Max A and Pluim, Josien PW},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {29},
  number = {1},
  pages = {196--205},
  year = {2010},
  doi = {10.1109/TMI.2009.2035616}
}
```

## Related Tools

- **Transformix** - Companion tool for applying transformations (included with elastix)
- **SimpleElastix** - Python/R/Java interface to elastix
- **ANTs** - Alternative registration framework with different algorithms
- **FSL FLIRT/FNIRT** - FSL's registration tools
- **SPM** - Statistical Parametric Mapping registration
- **ITK** - Insight Toolkit (elastix is built on ITK)
- **SimpleITK** - Simplified ITK interface with elastix integration
- **Demons** - Diffeomorphic demons registration algorithm

---

**Skill Type:** Image Registration
**Difficulty Level:** Intermediate to Advanced
**Prerequisites:** Basic neuroimaging knowledge, command-line experience, understanding of registration concepts
**Typical Use Cases:** Spatial normalization, multi-modal registration, atlas construction, longitudinal analysis, motion correction
