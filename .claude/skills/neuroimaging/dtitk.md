# DTI-TK - Diffusion Tensor Imaging ToolKit

## Overview

DTI-TK (Diffusion Tensor Imaging ToolKit) is a specialized spatial normalization and atlas construction toolkit for diffusion tensor images (DTI). Developed by Gary Hui Zhang at UCL, DTI-TK performs tensor-based registration that directly optimizes tensor reorientation and alignment, rather than treating DTI as scalar images. This tensor-aware approach preserves the directional information in diffusion data, making it superior to scalar-based methods for DTI normalization, atlas construction, and voxel-based analysis of diffusion parameters.

**Website:** http://dti-tk.sourceforge.net/
**Platform:** Linux/macOS (command-line tools)
**License:** Free for academic use
**Key Application:** DTI registration, population template creation, tract-based analysis

### Why Tensor-Based Registration?

Traditional scalar registration (using FA or MD maps) has limitations:
- **Loses directional information** - Tensors encode fiber orientation
- **Suboptimal alignment** - Fiber tracts may not align properly
- **Reduced sensitivity** - Misses subtle white matter changes

DTI-TK addresses these by:
- **Direct tensor matching** - Optimizes tensor similarity
- **Proper tensor reorientation** - Preserves fiber directions during warping
- **Better white matter alignment** - Aligns fiber bundles accurately

## Key Features

- **Tensor-based registration** - Directly registers full diffusion tensors
- **Preservation of Principal Direction (PPD)** - Maintains fiber orientation during warping
- **Population template construction** - Create study-specific DTI atlases
- **Affine and deformable registration** - Multi-stage pipeline
- **Diffeomorphic transforms** - Smooth, invertible deformations
- **Tensor reorientation** - Proper handling of directional data
- **TBSS-style analysis** - Voxel-based stats on aligned DTI
- **Quality control tools** - Assess registration accuracy
- **Fast processing** - Optimized C++ implementation
- **Well-validated** - Published and widely used
- **Command-line interface** - Scriptable and reproducible
- **Format support** - NIFTI input/output

## Installation

### Linux Installation

\`\`\`bash
# Download DTI-TK
cd ~/software
wget http://dti-tk.sourceforge.net/pmwiki/uploads/DTI-TK/dtitk_2.3.1_Linux_x86_64.tar.gz

# Extract
tar -xzf dtitk_2.3.1_Linux_x86_64.tar.gz
cd dtitk-2.3.1-Linux-x86_64

# Add to PATH
echo 'export PATH=$HOME/software/dtitk-2.3.1-Linux-x86_64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Add library path
echo 'export LD_LIBRARY_PATH=$HOME/software/dtitk-2.3.1-Linux-x86_64/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
\`\`\`

### macOS Installation

\`\`\`bash
# Download macOS version
cd ~/software
wget http://dti-tk.sourceforge.net/pmwiki/uploads/DTI-TK/dtitk_2.3.1_MacOSX_x86_64.tar.gz

# Extract and setup
tar -xzf dtitk_2.3.1_MacOSX_x86_64.tar.gz
cd dtitk-2.3.1-MacOSX_x86_64

# Add to PATH
echo 'export PATH=$HOME/software/dtitk-2.3.1-MacOSX_x86_64/bin:$PATH' >> ~/.bash_profile
source ~/.bash_profile

# Add library path
echo 'export DYLD_LIBRARY_PATH=$HOME/software/dtitk-2.3.1-MacOSX_x86_64/lib:$DYLD_LIBRARY_PATH' >> ~/.bash_profile
source ~/.bash_profile
\`\`\`

### Verify Installation

\`\`\`bash
# Check installation
dti_rigid_reg --version

# List available commands
ls ~/software/dtitk-*/bin/

# Should see:
# dti_rigid_reg, dti_affine_reg, dti_diffeomorphic_reg
# TVtool, dfRightComposeAffine, etc.
\`\`\`

## DTI Data Preparation

### Convert from FSL Format

DTI-TK uses NIFTI format with specific tensor encoding:

\`\`\`bash
# FSL dtifit output: dti_V1, dti_V2, dti_V3, dti_L1, dti_L2, dti_L3
# Convert to DTI-TK format (6 component: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)

# Use fsl_to_dtitk (if available) or manual conversion
# Assuming you have: dti_FA.nii.gz, dti_MD.nii.gz, dti_tensor.nii.gz

# DTI-TK expects tensor in specific order
# This is usually handled by the fitting tool
\`\`\`

### From DIPY/MRtrix3

\`\`\`bash
# If using DIPY or MRtrix3 for tensor fitting:

# MRtrix3:
dwi2tensor dwi.mif -fslgrad bvecs bvals tensor.mif
mrconvert tensor.mif tensor.nii.gz

# DIPY (Python):
# Use dipy.io.save_nifti to save tensors
# Ensure proper ordering for DTI-TK
\`\`\`

### Create Scalar Maps

\`\`\`bash
# Extract FA, MD, AD, RD from tensor
# DTI-TK provides TVtool

TVtool -in tensor.nii.gz -fa
# Output: tensor_fa.nii.gz

TVtool -in tensor.nii.gz -tr
# Output: tensor_tr.nii.gz (trace / mean diffusivity × 3)

TVtool -in tensor.nii.gz -ad
# Output: tensor_ad.nii.gz (axial diffusivity)

TVtool -in tensor.nii.gz -rd
# Output: tensor_rd.nii.gz (radial diffusivity)
\`\`\`

## Inter-Subject Registration

### Step 1: Rigid Registration

Align tensors using rigid transformation (6 DOF):

\`\`\`bash
# Rigid registration between two subjects
dti_rigid_reg \\
  template_tensor.nii.gz \\
  subject01_tensor.nii.gz \\
  EDS \\
  4 4 4 \\
  0.01

# Arguments:
# - template_tensor.nii.gz: target/fixed tensor image
# - subject01_tensor.nii.gz: moving tensor image
# - EDS: similarity metric (Euclidean Distance Squared of tensors)
# - 4 4 4: resampling resolution (mm)
# - 0.01: convergence threshold

# Output:
# - subject01_tensor_aff.nii.gz: registered tensor
# - subject01_tensor.aff: affine transformation
\`\`\`

### Step 2: Affine Registration

Add scaling and shearing (12 DOF):

\`\`\`bash
# Affine registration (starts from rigid result)
dti_affine_reg \\
  template_tensor.nii.gz \\
  subject01_tensor \\
  EDS \\
  4 4 4 \\
  0.01 \\
  1

# Last argument:
# 1 = start from existing affine (.aff file)
# 0 = start from scratch

# Output:
# - subject01_tensor_aff.nii.gz: affine-registered tensor
# - subject01_tensor.aff: updated affine transform
\`\`\`

### Step 3: Deformable Registration

High-dimensional diffeomorphic registration:

\`\`\`bash
# Deformable registration
dti_diffeomorphic_reg \\
  template_tensor.nii.gz \\
  subject01_tensor_aff \\
  subject01_template \\
  0.002

# Arguments:
# - template_tensor.nii.gz: target
# - subject01_tensor_aff: input (after affine)
# - subject01_template: output prefix
# - 0.002: convergence threshold

# Output:
# - subject01_template.nii.gz: deformably registered tensor
# - subject01_template_diffeo.df.nii.gz: deformation field

# This can take 30-60 minutes per subject
\`\`\`

### Complete Pipeline Script

\`\`\`bash
#!/bin/bash
# register_subject_to_template.sh

template=$1
subject=$2
output_prefix=$3

echo "Registering ${subject} to ${template}..."

# Step 1: Rigid
echo "  Rigid registration..."
dti_rigid_reg ${template} ${subject} EDS 4 4 4 0.01

# Step 2: Affine
echo "  Affine registration..."
dti_affine_reg ${template} ${subject} EDS 4 4 4 0.01 1

# Step 3: Deformable
echo "  Deformable registration..."
dti_diffeomorphic_reg ${template} ${subject}_aff ${output_prefix} 0.002

echo "Registration complete!"
echo "Output: ${output_prefix}.nii.gz"
\`\`\`

## Population Template Construction

### Create Study-Specific Atlas

Build DTI template from your subjects:

\`\`\`bash
# Create list of subject tensors
ls /data/subjects/sub-*_tensor.nii.gz > tensor_list.txt

# Initialize template bootstrap
# Use one subject as initial template, or create mean
dti_mean_template \\
  tensor_list.txt \\
  initial_template.nii.gz

# Or use a subject:
cp /data/subjects/sub-01_tensor.nii.gz initial_template.nii.gz
\`\`\`

### Iterative Template Refinement

\`\`\`bash
#!/bin/bash
# build_population_template.sh

template_list="tensor_list.txt"
n_iterations=6

# Read subject list
subjects=($(cat ${template_list}))
n_subjects=${#subjects[@]}

echo "Building template from ${n_subjects} subjects"

# Initialize with mean or first subject
echo "Creating initial template..."
dti_mean_template ${template_list} mean_template0.nii.gz

# Iterative refinement
for iter in $(seq 1 ${n_iterations}); do
    echo "Iteration ${iter}/${n_iterations}..."

    prev_iter=$((iter - 1))
    template="mean_template${prev_iter}.nii.gz"

    # Register all subjects to current template
    for i in $(seq 0 $((n_subjects - 1))); do
        subj=${subjects[$i]}
        subj_name=$(basename ${subj} _tensor.nii.gz)

        echo "  Registering ${subj_name}..."

        # Rigid + Affine
        dti_rigid_reg ${template} ${subj} EDS 4 4 4 0.01
        dti_affine_reg ${template} ${subj} EDS 4 4 4 0.01 1

        # Deformable
        dti_diffeomorphic_reg ${template} ${subj}_aff \\
          ${subj_name}_iter${iter} 0.002
    done

    # Compute new template (mean of warped subjects)
    ls *_iter${iter}.nii.gz > warped_list_iter${iter}.txt
    dti_mean_template warped_list_iter${iter}.txt \\
      mean_template${iter}.nii.gz

    echo "Iteration ${iter} complete"
done

echo "Final template: mean_template${n_iterations}.nii.gz"
\`\`\`

## Applying Transformations

### Transform Additional Images

Apply computed transformations to other modalities:

\`\`\`bash
# After registration, apply transform to FA map

# 1. Affine transform
dfRightComposeAffine \\
  -aff subject01_tensor.aff \\
  -in subject01_FA.nii.gz \\
  -out subject01_FA_aff.nii.gz

# 2. Deformable transform
deformationSymTensor3DVolume \\
  -in subject01_tensor_aff.nii.gz \\
  -trans subject01_template_diffeo.df.nii.gz \\
  -target template_tensor.nii.gz \\
  -out subject01_tensor_def.nii.gz

# For scalar images (FA, MD):
deformationScalarVolume \\
  -in subject01_FA_aff.nii.gz \\
  -trans subject01_template_diffeo.df.nii.gz \\
  -target template_FA.nii.gz \\
  -out subject01_FA_registered.nii.gz
\`\`\`

### Compose Transformations

\`\`\`bash
# Combine affine and deformable into single transform
dfComposition \\
  -df1 subject01_template_diffeo.df.nii.gz \\
  -aff subject01_tensor.aff \\
  -out subject01_combined.df.nii.gz

# Apply combined transform
deformationScalarVolume \\
  -in subject01_FA.nii.gz \\
  -trans subject01_combined.df.nii.gz \\
  -target template_FA.nii.gz \\
  -out subject01_FA_final.nii.gz
\`\`\`

## Voxel-Based Analysis

### Prepare Data for Statistics

\`\`\`bash
#!/bin/bash
# Normalize all subjects to template and extract FA

template="population_template.nii.gz"
output_dir="/data/normalized"

mkdir -p ${output_dir}

for subj in /data/subjects/sub-*_tensor.nii.gz; do
    subj_id=$(basename ${subj} _tensor.nii.gz)

    echo "Processing ${subj_id}..."

    # Register to template (rigid → affine → deformable)
    dti_rigid_reg ${template} ${subj} EDS 4 4 4 0.01
    dti_affine_reg ${template} ${subj} EDS 4 4 4 0.01 1
    dti_diffeomorphic_reg ${template} ${subj}_aff ${subj_id}_norm 0.002

    # Extract FA from registered tensor
    TVtool -in ${subj_id}_norm.nii.gz -fa
    mv ${subj_id}_norm_fa.nii.gz ${output_dir}/${subj_id}_FA_norm.nii.gz

    # Extract MD
    TVtool -in ${subj_id}_norm.nii.gz -tr
    # MD = trace / 3
    fslmaths ${subj_id}_norm_tr.nii.gz -div 3 ${output_dir}/${subj_id}_MD_norm.nii.gz

    echo "${subj_id} complete"
done

echo "All subjects normalized. Ready for voxel-based analysis."
\`\`\`

### Statistical Analysis with FSL

\`\`\`bash
# After normalization, use FSL randomise for statistics

# Create 4D volume of all subject FA maps
fslmerge -t all_FA_4D.nii.gz /data/normalized/*_FA_norm.nii.gz

# Create design matrix (FSL Glm)
# E.g., two-group comparison

# Run randomise
randomise \\
  -i all_FA_4D.nii.gz \\
  -o tbss \\
  -d design.mat \\
  -t design.con \\
  -m mean_FA_mask.nii.gz \\
  -n 5000 \\
  -T

# Results: tbss_tstat*.nii.gz, tbss_tfce_corrp_*.nii.gz
\`\`\`

## Quality Control

### Visual Inspection

\`\`\`bash
# Generate FA from registered tensors for visual QC
for tensor in *_registered.nii.gz; do
    TVtool -in ${tensor} -fa
done

# View in FSLeyes or ITK-SNAP
# Check alignment of fiber bundles
\`\`\`

### Quantitative Metrics

\`\`\`bash
# Compute overlap of white matter masks

# 1. Create WM mask from template FA
fslmaths template_FA.nii.gz -thr 0.2 -bin template_WM_mask.nii.gz

# 2. Create subject WM mask
fslmaths subject01_FA_registered.nii.gz -thr 0.2 -bin subject01_WM_mask.nii.gz

# 3. Compute Dice coefficient
fslmaths template_WM_mask.nii.gz -mul subject01_WM_mask.nii.gz overlap.nii.gz
overlap=$(fslstats overlap.nii.gz -V | awk '{print $1}')
template_vol=$(fslstats template_WM_mask.nii.gz -V | awk '{print $1}')
subject_vol=$(fslstats subject01_WM_mask.nii.gz -V | awk '{print $1}')

dice=$(echo "scale=4; 2 * ${overlap} / (${template_vol} + ${subject_vol})" | bc)
echo "Dice coefficient: ${dice}"
# Good registration: Dice > 0.8
\`\`\`

### Tensor Similarity

\`\`\`bash
# Measure tensor similarity in overlapping regions
# Use DTI-TK's tensor comparison tools

TVFromEigenSystem \\
  -basename1 subject01_registered \\
  -basename2 template \\
  -out tensor_diff.nii.gz

# Compute mean difference
fslstats tensor_diff.nii.gz -M
\`\`\`

## Integration with Claude Code

DTI-TK workflows integrate well with Claude-assisted pipelines:

### Automated Template Construction

\`\`\`markdown
**Prompt to Claude:**
"Create a DTI-TK pipeline to build a population template:
1. Start from 30 subjects with DTI tensors
2. Perform 6 iterations of template refinement
3. Track registration quality (Dice) at each iteration
4. Generate QC report with FA alignment images
5. Parallelize subject registration using GNU parallel
Include error handling and progress logging."
\`\`\`

### Tract-Based Analysis

\`\`\`markdown
**Prompt to Claude:**
"Set up DTI-TK + FSL pipeline for TBSS-style analysis:
1. Register all subjects to study template
2. Extract FA, MD, AD, RD maps
3. Create white matter skeleton
4. Project metrics onto skeleton
5. Run randomise for group comparison
6. Generate results summary with cluster tables
Provide complete bash scripts."
\`\`\`

### Quality Control Automation

\`\`\`markdown
**Prompt to Claude:**
"Generate DTI-TK QC script that:
1. Computes Dice for all registered subjects
2. Creates before/after registration mosaics
3. Plots FA value distributions
4. Identifies outliers (Dice < 0.75)
5. Generates HTML report
Include visualization code."
\`\`\`

## Integration with Other Tools

### MRtrix3

\`\`\`bash
# Use MRtrix3 for preprocessing, DTI-TK for registration

# 1. MRtrix3: Preprocessing and tensor fitting
dwi2tensor dwi_preprocessed.mif -fslgrad bvecs bvals -mask mask.mif tensor.mif
tensor2metric tensor.mif -fa FA.mif -adc MD.mif

# Convert to NIfTI for DTI-TK
mrconvert tensor.mif tensor.nii.gz
mrconvert FA.mif FA.nii.gz

# 2. DTI-TK: Registration
dti_rigid_reg template.nii.gz tensor.nii.gz EDS 4 4 4 0.01
# ... continue with affine and deformable

# 3. Back to MRtrix3: Tractography in template space
mrconvert tensor_registered.nii.gz tensor_registered.mif
tckgen tensor_registered.mif tracks.tck -algorithm iFOD2
\`\`\`

### FSL

\`\`\`bash
# FSL dtifit → DTI-TK registration → FSL statistics

# 1. FSL: Tensor fitting
dtifit -k dwi.nii.gz -o dti -m mask.nii.gz -r bvecs -b bvals

# 2. Convert FSL to DTI-TK format (manual tensor construction)
# DTI-TK expects: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz

# 3. DTI-TK: Registration
dti_rigid_reg template.nii.gz dti_tensor.nii.gz EDS 4 4 4 0.01

# 4. FSL: TBSS or randomise
# Use registered FA maps
\`\`\`

### DIPY

\`\`\`python
import numpy as np
import nibabel as nib
from dipy.io.image import save_nifti

# Fit tensor with DIPY
# ... (tensor fitting code)

# Save in format for DTI-TK
# Ensure proper tensor component ordering
tensor_data = np.stack([
    tensor_fit.quadratic_form[:,:,:,0,0],  # Dxx
    tensor_fit.quadratic_form[:,:,:,0,1],  # Dxy
    tensor_fit.quadratic_form[:,:,:,0,2],  # Dxz
    tensor_fit.quadratic_form[:,:,:,1,1],  # Dyy
    tensor_fit.quadratic_form[:,:,:,1,2],  # Dyz
    tensor_fit.quadratic_form[:,:,:,2,2],  # Dzz
], axis=-1)

save_nifti('tensor_for_dtitk.nii.gz', tensor_data, affine)
\`\`\`

## Troubleshooting

### Problem 1: Registration Diverges

**Symptoms:** Very poor alignment, extreme warping

**Solutions:**
\`\`\`bash
# Use smaller convergence threshold
dti_diffeomorphic_reg template.nii.gz subject_aff output 0.001  # Was 0.002

# Better initialization with affine
dti_affine_reg template.nii.gz subject EDS 2 2 2 0.001 1  # Finer resolution

# Check input data quality
TVtool -in subject.nii.gz -fa
# Ensure FA looks reasonable (0-1 range, clear WM structure)
\`\`\`

### Problem 2: Out of Memory

**Symptoms:** Process killed during deformable registration

**Solutions:**
\`\`\`bash
# Use coarser resolution during registration
dti_rigid_reg template.nii.gz subject.nii.gz EDS 6 6 6 0.01  # Was 4 4 4

# Downsample tensors first
# (Requires custom scripting or external tools)

# Run on machine with more RAM
# Deformable registration can use 8-16 GB
\`\`\`

### Problem 3: Tensor Reorientation Issues

**Symptoms:** Incorrect fiber directions after registration

**Solutions:**
\`\`\`bash
# Ensure using tensor-specific transforms
deformationSymTensor3DVolume -in tensor.nii.gz ...
# NOT deformationScalarVolume (which doesn't reorient)

# Verify tensor eigenvectors
TVtool -in registered_tensor.nii.gz -eig
# Check that V1 directions make anatomical sense
\`\`\`

### Problem 4: Template Construction Slow

**Symptoms:** Template building takes days

**Solutions:**
\`\`\`bash
# Reduce number of iterations
# 3-4 iterations often sufficient instead of 6

# Parallelize subject registrations
# Use GNU parallel for multi-subject processing

parallel -j 4 \\
  'dti_rigid_reg template.nii.gz {} EDS 4 4 4 0.01' \\
  ::: subjects/*.nii.gz

# Use cluster/HPC if available
\`\`\`

## Best Practices

### Data Preparation

1. **Quality DTI data** - Good SNR, minimal artifacts
2. **Proper tensor fitting** - Use robust methods (DIPY, FSL, MRtrix3)
3. **Brain extraction** - Remove skull before registration
4. **Check tensor quality** - No negative eigenvalues

### Registration Workflow

1. **Multi-stage approach** - Always rigid → affine → deformable
2. **Use study-specific template** - Better than atlas template
3. **Visual QC at each stage** - Don't proceed with bad registrations
4. **Consistent parameters** - Same settings for all subjects
5. **Document transformations** - Save all .aff and .df files

### Template Construction

1. **Homogeneous cohort** - Similar age, pathology for template
2. **Sufficient iterations** - 4-6 typically adequate
3. **Monitor convergence** - Check template stability across iterations
4. **Final QC** - Ensure good WM alignment in template

### Analysis

1. **Appropriate statistics** - Account for spatial smoothness
2. **Multiple comparison correction** - FWE or FDR mandatory
3. **Validate findings** - ROI analysis to confirm voxel-wise results
4. **Report methods clearly** - DTI-TK version, parameters, template

## Resources

### Official Documentation

- **DTI-TK Website:** http://dti-tk.sourceforge.net/
- **User Guide:** http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.HomePage
- **Download:** http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Main.Download

### Key Publications

- **DTI-TK Method:** Zhang et al. (2006) "Deformable registration of diffusion tensor MR images with explicit orientation optimization" Medical Image Analysis
- **Template Construction:** Zhang et al. (2007) "High-dimensional spatial normalization of diffusion tensor images improves the detection of white matter differences" Neuroimage

### Learning Resources

- **Tutorial:** http://dti-tk.sourceforge.net/pmwiki/pmwiki.php?n=Documentation.Tutorial
- **Example Scripts:** Included in software distribution

### Community Support

- **Email:** dti-tk-users mailing list
- **SourceForge:** http://sourceforge.net/projects/dti-tk/

## Citation

\`\`\`bibtex
@article{Zhang2006,
  title = {Deformable registration of diffusion tensor MR images with explicit orientation optimization},
  author = {Zhang, Hui and Yushkevich, Paul A and Alexander, Daniel C and Gee, James C},
  journal = {Medical Image Analysis},
  volume = {10},
  number = {5},
  pages = {764--785},
  year = {2006},
  doi = {10.1016/j.media.2006.06.004}
}
\`\`\`

## Related Tools

- **FSL TBSS** - Tract-based spatial statistics (alternative white matter analysis)
- **ANTs** - General registration (can register DTI but doesn't reorient tensors)
- **MRtrix3** - Comprehensive diffusion analysis suite
- **DIPY** - Python diffusion imaging toolkit
- **DSI Studio** - Diffusion visualization and analysis
- **Camino** - Diffusion MRI toolkit
- **TORTOISE** - DTI processing and quality control

---

**Skill Type:** Diffusion MRI Registration
**Difficulty Level:** Intermediate to Advanced
**Prerequisites:** Linux/macOS, DTI preprocessing experience, Understanding of diffusion tensors
**Typical Use Cases:** DTI normalization, population template creation, voxel-based DTI analysis, white matter studies
