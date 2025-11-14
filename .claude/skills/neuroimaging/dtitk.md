# DTI-TK - Diffusion Tensor Imaging ToolKit

## Overview

DTI-TK (Diffusion Tensor Imaging ToolKit) is a specialized software package for spatial normalization and atlas construction of diffusion tensor images. Unlike standard registration methods that treat DTI as scalar images, DTI-TK preserves the full tensor information during registration using diffeomorphic transformations optimized for tensor data. This enables accurate population studies, voxel-based analysis, and atlas creation while maintaining the directional information critical for white matter analysis.

**Website:** http://dti-tk.sourceforge.net/
**Platform:** Linux/macOS
**Language:** C++
**License:** Open Source

## Key Features

- Tensor-based spatial normalization
- Diffeomorphic registration preserving tensor orientation
- Population-specific atlas construction
- Inter-subject registration
- Intra-subject registration (longitudinal)
- Tensor reorientation during warping
- Scalar maps (FA, MD, etc.) generation
- Integration with other DTI tools
- Command-line interface
- Supports NIfTI format
- Affine and deformable registration

## Installation

### Download and Install

```bash
# Download from: http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Main.Download

# Linux
tar -xzf dtitk*.tar.gz
cd dtitk

# Add to PATH
export PATH=/path/to/dtitk/bin:${PATH}
export DTITK_ROOT=/path/to/dtitk

# Add to ~/.bashrc for persistence
echo 'export PATH=/path/to/dtitk/bin:${PATH}' >> ~/.bashrc
echo 'export DTITK_ROOT=/path/to/dtitk' >> ~/.bashrc

# Verify installation
dti_rigid_reg --help
TVtool -h
```

### macOS Installation

```bash
# Download macOS version
tar -xzf dtitk_macos*.tar.gz

# Same PATH setup as above
export PATH=/path/to/dtitk/bin:${PATH}

# May need to allow in Security & Privacy settings
```

## Data Preparation

### Input Requirements

```bash
# DTI-TK requires:
# 1. DTI tensor images (6 independent components)
# 2. Or separate files for each component

# Preprocessing steps:
# - Motion/eddy correction
# - Tensor fitting
# - Brain extraction
# - Convert to DTI-TK format
```

### Convert from FSL to DTI-TK

```bash
# FSL dtifit outputs:
# - dti_V1.nii.gz (first eigenvector)
# - dti_L1.nii.gz, dti_L2.nii.gz, dti_L3.nii.gz (eigenvalues)
# - dti_FA.nii.gz, dti_MD.nii.gz

# Reconstruct tensor from FSL outputs
fsl_to_dtitk \
  dti_L1.nii.gz dti_L2.nii.gz dti_L3.nii.gz \
  dti_V1.nii.gz dti_V2.nii.gz dti_V3.nii.gz \
  dti_tensor.nii.gz

# Or use DTI-TK's own fitting
dti_fit_tensor dwi.nii.gz bvecs bvals output_tensor.nii.gz
```

### Tensor File Format

```bash
# DTI-TK tensor format (NIfTI with 6 volumes):
# Volume 1: Dxx
# Volume 2: Dxy
# Volume 3: Dxz
# Volume 4: Dyy
# Volume 5: Dyz
# Volume 6: Dzz

# Check tensor file
TVtool -in tensor.nii.gz -tr

# Generate scalar maps from tensor
TVtool -in tensor.nii.gz -fa
# Creates tensor_fa.nii.gz

TVtool -in tensor.nii.gz -tr
# Creates tensor_tr.nii.gz (trace/MD)
```

## Single Subject Registration

### Rigid Registration

```bash
# Rigid alignment to template
dti_rigid_reg \
  template.nii.gz \
  subject_tensor.nii.gz \
  EDS 4 4 4 0.01

# Parameters:
#   template.nii.gz: Target template
#   subject_tensor.nii.gz: Moving tensor image
#   EDS: Euclidean Distance Squared similarity metric
#   4 4 4: Multi-resolution levels (x,y,z)
#   0.01: Tolerance

# Output: subject_tensor_aff.nii.gz
#         subject_tensor.aff (affine transform)
```

### Affine Registration

```bash
# Affine (12 DOF) registration
dti_affine_reg \
  template.nii.gz \
  subject_tensor_aff.nii.gz \
  EDS 4 4 4 0.01 1

# Last parameter (1) = use previous rigid as initialization

# Output: subject_tensor_aff_aff.nii.gz
#         subject_tensor_aff.aff
```

### Deformable Registration

```bash
# Non-linear diffeomorphic registration
dti_diffeomorphic_reg \
  template.nii.gz \
  subject_tensor_aff_aff.nii.gz \
  mask.nii.gz \
  1 6 0.002

# Parameters:
#   template.nii.gz: Template
#   subject: Affine-aligned subject
#   mask: Template brain mask
#   1: Use full tensor (vs 0 for FA only)
#   6: Number of iterations
#   0.002: Step size

# Output: subject_tensor_aff_aff_diffeo.nii.gz
#         subject_tensor_aff_aff_diffeo.df.nii.gz (deformation field)
```

### Complete Single-Subject Pipeline

```bash
#!/bin/bash
# Register subject to template

TEMPLATE="IXI_template.nii.gz"
SUBJECT="sub-01_tensor.nii.gz"
MASK="template_mask.nii.gz"

# 1. Rigid
echo "Rigid registration..."
dti_rigid_reg ${TEMPLATE} ${SUBJECT} EDS 4 4 4 0.01

# 2. Affine
echo "Affine registration..."
dti_affine_reg ${TEMPLATE} ${SUBJECT}_aff.nii.gz EDS 4 4 4 0.01 1

# 3. Deformable
echo "Deformable registration..."
dti_diffeomorphic_reg ${TEMPLATE} ${SUBJECT}_aff_aff.nii.gz ${MASK} 1 6 0.002

# Final output
FINAL="${SUBJECT}_aff_aff_diffeo.nii.gz"
echo "Registration complete: ${FINAL}"

# Generate FA from registered tensor
TVtool -in ${FINAL} -fa
```

## Population Atlas Construction

### Create Study-Specific Template

```bash
#!/bin/bash
# Build population template from multiple subjects

# Subject list
SUBJECTS=(
  sub-01_tensor.nii.gz
  sub-02_tensor.nii.gz
  sub-03_tensor.nii.gz
  sub-04_tensor.nii.gz
  sub-05_tensor.nii.gz
)

# 1. Bootstrap: rigidly align all to first subject
TEMPLATE=${SUBJECTS[0]}

for subj in "${SUBJECTS[@]}"; do
    dti_rigid_reg ${TEMPLATE} ${subj} EDS 4 4 4 0.01
done

# 2. Initial mean
TVMean -in ${SUBJECTS[@]/%/_aff.nii.gz} -out mean_initial.nii.gz

# 3. Iterative template construction
for iter in {1..5}; do
    echo "Iteration ${iter}..."

    TEMPLATE="mean_iteration${iter}.nii.gz"

    # Affine register all to current template
    for subj in "${SUBJECTS[@]}"; do
        dti_affine_reg ${TEMPLATE} ${subj}_aff.nii.gz EDS 4 4 4 0.01 1
    done

    # Deformable registration
    for subj in "${SUBJECTS[@]}"; do
        dti_diffeomorphic_reg \
          ${TEMPLATE} \
          ${subj}_aff_aff.nii.gz \
          template_mask.nii.gz \
          1 6 0.002
    done

    # Compute mean
    TVMean -in ${SUBJECTS[@]/%/_aff_aff_diffeo.nii.gz} \
      -out mean_iteration$((iter+1)).nii.gz

done

# Final template
cp mean_iteration6.nii.gz study_template_final.nii.gz
```

### Automated Template Building

```bash
# Use DTI-TK's automated script
dti_template_bootstrap \
  subjects_list.txt \
  EDS 1

# subjects_list.txt contains paths to all tensor files

# Then refine template
dti_template_refine \
  subjects_list.txt \
  template_mask.nii.gz \
  6 \
  study_template_final.nii.gz

# This automates the iterative process
```

## Apply Transformations

### Warp Scalar Maps

```bash
# After registration, apply transforms to FA/MD maps

# Generate FA from original tensor
TVtool -in subject_tensor.nii.gz -fa

# Apply same transform to FA
deformationSymTensor3DVolume \
  -in subject_tensor_fa.nii.gz \
  -trans subject_tensor_aff.aff \
  -target template.nii.gz \
  -out subject_fa_in_template_space.nii.gz

# For deformation field
deformationSymTensor3DVolume \
  -in subject_tensor_fa.nii.gz \
  -trans subject_tensor_aff_aff_diffeo.df.nii.gz \
  -target template.nii.gz \
  -out subject_fa_warped.nii.gz
```

### Warp Tensors

```bash
# Warp tensors with proper reorientation
deformationSymTensor3DVolume \
  -in subject_tensor.nii.gz \
  -trans subject_transform.df.nii.gz \
  -target template.nii.gz \
  -out subject_tensor_warped.nii.gz

# This preserves tensor orientation during warping
```

## Tract-Based Spatial Statistics (TBSS) Alternative

### DTI-TK TBSS Workflow

```bash
# DTI-TK can create skeleton for TBSS-like analysis

# 1. Register all subjects to template
for subj in sub-*_tensor.nii.gz; do
    ./register_to_template.sh ${subj}
done

# 2. Create mean FA in template space
fslmaths sub-01_fa_warped.nii.gz -add sub-02_fa_warped.nii.gz \
  -add sub-03_fa_warped.nii.gz [...]  \
  -div 20 mean_FA_template_space.nii.gz

# 3. Create skeleton
tbss_skeleton -i mean_FA_template_space.nii.gz \
  -o mean_FA_skeleton.nii.gz

# 4. Project subjects to skeleton
for subj_fa in *_fa_warped.nii.gz; do
    tbss_skeleton -i mean_FA_template_space.nii.gz \
      -p 0.2 mean_FA_skeleton.nii.gz \
      ${subj_fa} ${subj_fa/.nii.gz/_skeletonised.nii.gz}
done
```

## Voxel-Based Analysis

### Group Comparison

```bash
# After registering all subjects to common space

# Prepare 4D images for randomise
fslmerge -t controls_FA.nii.gz controls/*_fa_warped.nii.gz
fslmerge -t patients_FA.nii.gz patients/*_fa_warped.nii.gz

# Create design matrix
design_ttest2 design 10 10

# Run permutation testing
randomise -i all_FA.nii.gz \
  -o tbss_FA \
  -d design.mat \
  -t design.con \
  -m mean_FA_mask.nii.gz \
  -n 5000 \
  --T2

# View results
fsleyes template.nii.gz \
  tbss_FA_tfce_corrp_tstat1.nii.gz -cm red-yellow -dr 0.95 1
```

## Longitudinal Analysis

### Within-Subject Registration

```bash
# Register timepoint 2 to timepoint 1
dti_diffeomorphic_reg \
  sub-01_tp1_tensor.nii.gz \
  sub-01_tp2_tensor.nii.gz \
  subject_mask.nii.gz \
  1 6 0.002

# Measure changes
# Compare tensors, FA, etc. between timepoints

# Jacobian determinant (volume change)
deformationJacobianVolume \
  sub-01_tp2_tensor_diffeo.df.nii.gz \
  jacobian.nii.gz

# Values > 1: expansion
# Values < 1: contraction
```

## Utilities

### Tensor Math

```bash
# Compute FA, MD, etc.
TVtool -in tensor.nii.gz -fa
TVtool -in tensor.nii.gz -tr  # Trace (3*MD)
TVtool -in tensor.nii.gz -ad  # Axial diffusivity
TVtool -in tensor.nii.gz -rd  # Radial diffusivity

# Reorient tensor
TVtool -in tensor.nii.gz -reorient affine.aff -out tensor_reoriented.nii.gz

# Smooth tensor
TVtool -in tensor.nii.gz -smooth 2.0 -out tensor_smooth.nii.gz
```

### Quality Control

```bash
# Check tensor validity
TVtool -in tensor.nii.gz -check

# Visualize principal eigenvector
TVtool -in tensor.nii.gz -vec

# Compute tensor determinant (detect negative eigenvalues)
TVtool -in tensor.nii.gz -det
```

## Integration with Other Tools

### Export to MRtrix3

```bash
# Convert DTI-TK tensor to MRtrix format
# (requires custom script or manual conversion)

# Alternative: use MRtrix for preprocessing, DTI-TK for registration
```

### Use with FSL

```bash
# Register FA images using FSL
# Then apply to tensors using DTI-TK

# Or vice versa:
# Register tensors with DTI-TK
# Apply to other modalities with FSL
```

## Integration with Claude Code

When helping users with DTI-TK:

1. **Check Installation:**
   ```bash
   which dti_rigid_reg
   TVtool -h
   ```

2. **Common Issues:**
   - Tensor format incorrect (need 6 volumes)
   - Missing brain mask for deformable registration
   - Insufficient iterations for convergence
   - Memory errors with large datasets
   - Negative eigenvalues in tensors

3. **Best Practices:**
   - Always start with rigid → affine → deformable
   - Use study-specific templates when possible
   - Visual QC at each registration step
   - Check tensor validity after warping
   - Document all registration parameters
   - Save transformation files for reproducibility
   - Use consistent preprocessing across subjects

4. **Parameter Guidelines:**
   - Multi-resolution: 4 4 4 (standard)
   - Tolerance: 0.01 (rigid/affine)
   - Step size: 0.002 (deformable)
   - Iterations: 6 (minimum for deformable)

## Troubleshooting

**Problem:** Registration fails to converge
**Solution:** Increase iterations, adjust step size, check initial alignment quality

**Problem:** Tensors have negative eigenvalues after warping
**Solution:** Check original tensor validity, reduce deformation smoothness, use smaller step size

**Problem:** Results look over-smoothed
**Solution:** Reduce number of iterations, increase step size, use sparser grid

**Problem:** Memory errors
**Solution:** Downsample images, process in batches, increase system RAM

**Problem:** Misalignment after registration
**Solution:** Check rigid/affine initialization, verify similar contrast between images, use appropriate mask

## Resources

- Website: http://dti-tk.sourceforge.net/
- User Guide: http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.HomePage
- Forum: http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Main.Forum
- Publications: http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Main.Publications

## Citation

```bibtex
@article{zhang2006deformable,
  title={Deformable registration of diffusion tensor MR images with explicit orientation optimization},
  author={Zhang, Hui and Yushkevich, Paul A and Alexander, Daniel C and Gee, James C},
  journal={Medical image analysis},
  volume={10},
  number={5},
  pages={764--785},
  year={2006}
}

@article{zhang2007tensor,
  title={High-dimensional spatial normalization of diffusion tensor images improves the detection of white matter differences: an example study using amyotrophic lateral sclerosis},
  author={Zhang, Hui and Avants, Brian B and Yushkevich, Paul A and Woo, John H and Wang, Shaowu and McCluskey, Leo F and Elman, Lauren B and Melhem, Elias R and Gee, James C},
  journal={IEEE transactions on medical imaging},
  volume={26},
  number={11},
  pages={1585--1597},
  year={2007}
}
```

## Related Tools

- **ANTs:** General-purpose registration (can handle tensors)
- **FSL:** TBSS and standard registration
- **TORTOISE:** DTI preprocessing
- **MRtrix3:** FOD-based registration
- **Camino:** DTI processing and analysis
- **AFNI:** 3dDWItoDT for tensor estimation
