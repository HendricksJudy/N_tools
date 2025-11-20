# ANTs (Advanced Normalization Tools)

## Overview

ANTs (Advanced Normalization Tools) is a state-of-the-art medical image registration and segmentation toolkit. It excels at diffeomorphic image registration, segmentation, and quantifying biological shape and appearance. ANTs has consistently ranked as one of the top-performing registration methods in evaluation challenges.

**Website:** http://stnava.github.io/ANTs/
**Platform:** C++/Python (Linux/macOS/Windows)
**Language:** C++, with Python wrappers (ANTsPy)
**License:** Apache 2.0

## Key Features

- Diffeomorphic image registration (SyN)
- Multi-modal registration
- Template building
- Brain extraction
- Tissue segmentation
- Cortical thickness estimation
- Atropos segmentation
- Landmarking and shape analysis
- Joint label fusion

## Installation

### From Binary (Recommended)

```bash
# Download pre-compiled binaries
# Linux
cd /opt
wget https://github.com/ANTsX/ANTs/releases/download/v2.5.0/ants-2.5.0-ubuntu-20.04-X64-gcc.tar.gz
tar xzf ants-2.5.0-ubuntu-20.04-X64-gcc.tar.gz

# Add to PATH
export ANTSPATH=/opt/ants-2.5.0/bin/
export PATH=${ANTSPATH}:$PATH
```

### From Source

```bash
git clone https://github.com/ANTsX/ANTs.git
mkdir build install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ../ANTs
make -j 4
cd ANTS-build
make install
```

### Python Interface (ANTsPy)

```bash
pip install antspyx
```

### Verify Installation

```bash
antsRegistration --version
antsBrainExtraction.sh -h
```

## Common Workflows

### Brain Extraction

```bash
# Using antsBrainExtraction.sh
antsBrainExtraction.sh \
    -d 3 \
    -a input_T1.nii.gz \
    -e template_brain.nii.gz \
    -m template_brainmask.nii.gz \
    -o output_prefix

# With multimodal data
antsBrainExtraction.sh \
    -d 3 \
    -a input_T1.nii.gz \
    -t input_T2.nii.gz \
    -e template_brain.nii.gz \
    -m template_brainmask.nii.gz \
    -o output_prefix
```

### Image Registration

```bash
# Basic registration with SyN
antsRegistrationSyN.sh \
    -d 3 \
    -f fixed_image.nii.gz \
    -m moving_image.nii.gz \
    -o output_prefix \
    -t s  # s=SyN, sr=SyN with rigid, a=affine, r=rigid

# Quick rigid registration
antsRegistrationSyN.sh \
    -d 3 \
    -f fixed.nii.gz \
    -m moving.nii.gz \
    -o output_ \
    -t r

# Multimodal registration
antsRegistrationSyNQuick.sh \
    -d 3 \
    -f fixed_T1.nii.gz \
    -m moving_T2.nii.gz \
    -t s \
    -o output_
```

### Advanced Registration

```bash
# Full control with antsRegistration
antsRegistration \
    --dimensionality 3 \
    --float 0 \
    --output [output_prefix,output_warped.nii.gz] \
    --interpolation Linear \
    --winsorize-image-intensities [0.005,0.995] \
    --use-histogram-matching 0 \
    --initial-moving-transform [fixed.nii.gz,moving.nii.gz,1] \
    --transform Rigid[0.1] \
    --metric MI[fixed.nii.gz,moving.nii.gz,1,32,Regular,0.25] \
    --convergence [1000x500x250x100,1e-6,10] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    --transform Affine[0.1] \
    --metric MI[fixed.nii.gz,moving.nii.gz,1,32,Regular,0.25] \
    --convergence [1000x500x250x100,1e-6,10] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    --transform SyN[0.1,3,0] \
    --metric CC[fixed.nii.gz,moving.nii.gz,1,4] \
    --convergence [100x70x50x20,1e-6,10] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox
```

### Apply Transformations

```bash
# Apply warp to image
antsApplyTransforms \
    -d 3 \
    -i moving.nii.gz \
    -r fixed.nii.gz \
    -o warped_output.nii.gz \
    -t output_1Warp.nii.gz \
    -t output_0GenericAffine.mat

# Apply to label image (use nearest neighbor)
antsApplyTransforms \
    -d 3 \
    -i labels.nii.gz \
    -r fixed.nii.gz \
    -o warped_labels.nii.gz \
    -n NearestNeighbor \
    -t output_1Warp.nii.gz \
    -t output_0GenericAffine.mat

# Invert transformation
antsApplyTransforms \
    -d 3 \
    -i fixed.nii.gz \
    -r moving.nii.gz \
    -o inverse_warped.nii.gz \
    -t [output_0GenericAffine.mat,1] \
    -t output_1InverseWarp.nii.gz
```

### Tissue Segmentation (Atropos)

```bash
# N4 bias correction first
N4BiasFieldCorrection \
    -d 3 \
    -i input.nii.gz \
    -o corrected.nii.gz

# Atropos segmentation
Atropos \
    -d 3 \
    -a corrected.nii.gz \
    -i KMeans[3] \
    -o [segmentation.nii.gz,posteriors%02d.nii.gz] \
    -m [0.2,1x1x1] \
    -c [5,0] \
    -p Socrates[1] \
    -x brain_mask.nii.gz

# With priors
Atropos \
    -d 3 \
    -a corrected.nii.gz \
    -i PriorProbabilityImages[3,priors%02d.nii.gz,0.5] \
    -o [segmentation.nii.gz,posteriors%02d.nii.gz] \
    -m [0.2,1x1x1] \
    -c [5,0] \
    -x brain_mask.nii.gz
```

### Cortical Thickness

```bash
# antsCorticalThickness.sh
antsCorticalThickness.sh \
    -d 3 \
    -a input_T1.nii.gz \
    -e template_brain.nii.gz \
    -m template_brainmask.nii.gz \
    -p template_priors%d.nii.gz \
    -o output_prefix

# Output: cortical thickness maps, segmentation, etc.
```

### Template Building

```bash
# Build population template
antsMultivariateTemplateConstruction2.sh \
    -d 3 \
    -o output_template \
    -i 4 \
    -g 0.2 \
    -j 4 \
    -c 2 \
    -k 1 \
    -w 1 \
    -f 8x4x2x1 \
    -s 3x2x1x0 \
    -q 100x70x50x10 \
    -n 0 \
    -r 1 \
    -l 1 \
    -m CC[4] \
    -t SyN[0.1,3,0] \
    subject*.nii.gz
```

### Joint Label Fusion

```bash
# Multi-atlas segmentation
antsJointLabelFusion.sh \
    -d 3 \
    -t target_image.nii.gz \
    -o output_prefix \
    -g atlas1.nii.gz -l atlas1_labels.nii.gz \
    -g atlas2.nii.gz -l atlas2_labels.nii.gz \
    -g atlas3.nii.gz -l atlas3_labels.nii.gz \
    -p posterior_label_%02d.nii.gz
```

## ANTsPy (Python Interface)

```python
import ants

# Read image
img = ants.image_read('input.nii.gz')

# Brain extraction
brain = ants.abp_n4(img)

# Registration
fixed = ants.image_read('fixed.nii.gz')
moving = ants.image_read('moving.nii.gz')

# Quick registration
result = ants.registration(fixed, moving, type_of_transform='SyN')
warped = result['warpedmovout']

# Apply transforms
warped = ants.apply_transforms(
    fixed=fixed,
    moving=moving,
    transformlist=result['fwdtransforms']
)

# N4 bias correction
corrected = ants.n4_bias_field_correction(img)

# Segmentation
seg = ants.atropos(
    a=img,
    m='[0.2,1x1x1]',
    c='[5,0]',
    i='kmeans[3]',
    x=mask
)

# Cortical thickness
thickness = ants.kelly_kapowski(
    s=segmentation,
    g=gray_matter,
    w=white_matter,
    its=45,
    r=0.025,
    m=1.5
)
```

## Integration with Claude Code

When helping users with ANTs:

1. **Check Installation:**
   ```bash
   echo $ANTSPATH
   antsRegistration --version
   ```

2. **Memory Requirements:** ANTs can be memory-intensive for large images

3. **Parallelization:** Use `-j` flag for multi-threading where available

4. **File Formats:** ANTs works with NIfTI, NRRD, and other formats

5. **Common Issues:**
   - ITK version conflicts
   - Memory limitations
   - Long processing times
   - Transform ordering matters

## Best Practices

- Use N4 bias correction before registration
- Start with coarse-to-fine registration strategies
- Use histogram matching for multimodal registration
- Check registration quality visually
- Use appropriate interpolation (Linear for images, NN for labels)
- Keep transforms for reproducibility
- Use ANTsPy for scripting and automation

## Useful Utilities

```bash
# Image information
PrintHeader image.nii.gz

# Image math
ImageMath 3 output.nii.gz + image1.nii.gz image2.nii.gz

# Measure image similarity
MeasureImageSimilarity 3 2 fixed.nii.gz moving.nii.gz

# Convert image format
ConvertImage 3 input.nii.gz output.nrrd

# Resample image
ResampleImage 3 input.nii.gz output.nii.gz 2x2x2 0 0 6

# Create mask from labels
ThresholdImage 3 labels.nii.gz mask.nii.gz 1 255

# Smooth image
SmoothImage 3 input.nii.gz 2.0 output.nii.gz
```

## Troubleshooting

**Problem:** "command not found: antsRegistration"
**Solution:** Check ANTSPATH is set and in PATH

**Problem:** Registration produces poor results
**Solution:** Check image orientations, use -t a (affine only) first

**Problem:** Out of memory
**Solution:** Reduce image resolution, use --float 1, or process on cluster

**Problem:** Wrong transform application order
**Solution:** ANTs applies transforms in order listed - last transform first

## Resources

- ANTs Documentation: https://github.com/ANTsX/ANTs/wiki
- ANTsPy Documentation: https://antspyx.readthedocs.io/
- ANTs Tutorial: https://github.com/stnava/ANTsTutorial
- ANTsX Ecosystem: https://github.com/ANTsX

## Related Tools

- **ANTsPy:** Python interface
- **ANTsR:** R interface
- **ANTsRNet:** Deep learning tools for ANTs
- **C3D:** Complementary image processing tool

## Citation

```bibtex
@article{avants2011reproducible,
  title={A reproducible evaluation of ANTs similarity metric performance in brain image registration},
  author={Avants, Brian B. and Tustison, Nicholas and Song, Gang and others},
  journal={NeuroImage},
  volume={54},
  number={3},
  pages={2033--2044},
  year={2011},
  doi={10.1016/j.neuroimage.2010.09.025}
}
```
