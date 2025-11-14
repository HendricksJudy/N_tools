# TORTOISE - Diffusion MRI Preprocessing and Analysis

## Overview

TORTOISE (Tolerably Obsessive Registration and Tensor Optimization Indolent Software Ensemble) is a comprehensive software package from NIH for processing diffusion MRI data. It specializes in robust preprocessing including distortion correction (EPI, eddy current, motion), registration, and advanced tensor fitting for DTI and diffusional kurtosis imaging (DKI). TORTOISE is particularly powerful for handling complex distortions and is widely used for clinical and research applications requiring high-quality diffusion data.

**Website:** https://tortoise.nibib.nih.gov/
**Platform:** Windows/Linux
**Language:** C++/Python
**License:** Free for research use (NIH software)

## Key Features

- Comprehensive distortion correction (EPI, eddy, motion)
- DIFFPREP: Automated preprocessing pipeline
- DR-BUDDI: Advanced distortion correction using blip-up/down
- DIFF_CALC: DTI and DKI tensor estimation
- Susceptibility-induced distortion correction
- Eddy current and motion correction
- Gradient nonlinearity correction
- Registration to structural images
- Advanced outlier detection and correction
- Multi-shell DWI support
- Supports DTI, DKI, HARDI acquisitions
- Quality control metrics and reports
- Integration with other diffusion tools

## Installation

### Windows

```bash
# Download installer from: https://tortoise.nibib.nih.gov/
# Run TORTOISE_V*_windows.exe installer
# Follow installation wizard

# Add to PATH (optional)
# Control Panel → System → Environment Variables
# Add: C:\TORTOISE_V3.2.0\DIFFPREPV320
```

### Linux

```bash
# Download from website
wget https://tortoise.nibib.nih.gov/TORTOISE_V*.tar.gz

# Extract
tar -xzf TORTOISE_V3.2.0.tar.gz
cd TORTOISE_V3.2.0

# Set environment variables
export TORTOISEPATH=/path/to/TORTOISE_V3.2.0
export PATH=${TORTOISEPATH}/bin:${PATH}
export LD_LIBRARY_PATH=${TORTOISEPATH}/lib:${LD_LIBRARY_PATH}

# Add to ~/.bashrc for persistence
echo 'export TORTOISEPATH=/path/to/TORTOISE_V3.2.0' >> ~/.bashrc
echo 'export PATH=${TORTOISEPATH}/bin:${PATH}' >> ~/.bashrc

# Verify installation
which DIFFPREP
DIFFPREP --help
```

## Data Preparation

### Input Requirements

```bash
# TORTOISE expects:
# 1. Raw DWI data (NIfTI or DICOM)
# 2. Gradient table (bval/bvec files)
# 3. Optional: Blip-up/blip-down for DRBUDDI
# 4. Optional: Structural T1/T2 for registration

# Organize data
subject/
├── dwi.nii.gz           # Primary DWI acquisition
├── dwi.bval             # b-values
├── dwi.bvec             # Gradient directions
├── dwi_PA.nii.gz        # Reverse phase-encode (optional)
└── T2.nii.gz            # Structural image (optional)
```

### BIDS Data

```bash
# TORTOISE can work with BIDS format
bids_dataset/
└── sub-01/
    └── dwi/
        ├── sub-01_dwi.nii.gz
        ├── sub-01_dwi.bval
        ├── sub-01_dwi.bvec
        └── sub-01_dwi.json
```

## DIFFPREP - Automated Preprocessing

### Basic DIFFPREP Workflow

```bash
# Run DIFFPREP with default settings
DIFFPREP \
  --dwi dwi.nii.gz \
  --bvals dwi.bval \
  --bvecs dwi.bvec \
  --phase vertical \
  --output_prefix sub-01

# Parameters:
#   --dwi: Input DWI data
#   --bvals: b-value file
#   --bvecs: b-vector file (gradient directions)
#   --phase: Phase encoding direction (vertical/horizontal)
#   --output_prefix: Prefix for output files
```

### Complete DIFFPREP Command

```bash
# Full preprocessing with all corrections
DIFFPREP \
  --dwi dwi.nii.gz \
  --bvals dwi.bval \
  --bvecs dwi.bvec \
  --phase vertical \
  --structural T2.nii.gz \
  --do_QC 1 \
  --step all \
  --output_prefix sub-01_proc

# Options:
#   --structural: Structural image for registration
#   --do_QC: Generate quality control images (0/1)
#   --step: Which steps to run (all, or specific: import, denoising, etc.)
#   --output_prefix: Output file prefix
```

### DIFFPREP Steps

```bash
# DIFFPREP performs multiple steps:

# 1. Import and convert data
DIFFPREP --step import --dwi dwi.nii.gz --bvals dwi.bval --bvecs dwi.bvec

# 2. Denoising (optional)
DIFFPREP --step denoising --dwi imported_data.list

# 3. Gibbs ringing correction (optional)
DIFFPREP --step gibbs --dwi imported_data.list

# 4. Motion and eddy current correction
DIFFPREP --step motion_eddy --dwi imported_data.list

# 5. EPI distortion correction
DIFFPREP --step epi --dwi imported_data.list --phase vertical

# 6. Structural registration (optional)
DIFFPREP --step reg --dwi corrected_data.list --structural T2.nii.gz

# Or run all steps
DIFFPREP --step all --dwi dwi.nii.gz --bvals dwi.bval --bvecs dwi.bvec
```

## DR-BUDDI - Advanced Distortion Correction

### Blip-Up/Blip-Down Correction

```bash
# DR-BUDDI uses paired acquisitions with opposite phase encoding
# Most accurate distortion correction available

# Run DR-BUDDI
DRBUDDI \
  --up dwi_AP.nii.gz \
  --down dwi_PA.nii.gz \
  --bvals dwi.bval \
  --bvecs dwi.bvec \
  --structural T2.nii.gz \
  --output sub-01_drbuddi

# Parameters:
#   --up: Anterior-posterior (or LR) acquisition
#   --down: Posterior-anterior (or RL) reverse acquisition
#   --bvals: b-values
#   --bvecs: b-vectors
#   --structural: Structural image
#   --output: Output prefix
```

### DR-BUDDI Advanced Options

```bash
# Fine-tuned DR-BUDDI
DRBUDDI \
  --up dwi_AP.nii.gz \
  --down dwi_PA.nii.gz \
  --bvals dwi.bval \
  --bvecs dwi.bvec \
  --structural T2.nii.gz \
  --output sub-01_drbuddi \
  --learning_rate 0.001 \
  --niter 3 \
  --outlier_prob 0.05 \
  --s_reg 0.5

# Options:
#   --learning_rate: Optimization step size (default: 0.001)
#   --niter: Number of iterations (default: 3)
#   --outlier_prob: Outlier detection threshold
#   --s_reg: Structural regularization weight
```

## DTI/DKI Tensor Estimation

### DIFF_CALC - DTI Fitting

```bash
# Estimate diffusion tensor from preprocessed data
DIFFCALC \
  --dwi sub-01_proc_DMC.nii.gz \
  --bvals sub-01_proc.bval \
  --bvecs sub-01_proc.bvec \
  --output sub-01_DTI

# Outputs DTI metrics:
# - sub-01_DTI_FA.nii.gz (Fractional Anisotropy)
# - sub-01_DTI_MD.nii.gz (Mean Diffusivity)
# - sub-01_DTI_AD.nii.gz (Axial Diffusivity)
# - sub-01_DTI_RD.nii.gz (Radial Diffusivity)
# - sub-01_DTI_L1/L2/L3.nii.gz (Eigenvalues)
# - sub-01_DTI_V1/V2/V3.nii.gz (Eigenvectors)
# - sub-01_DTI_DT.nii.gz (Full tensor)
```

### DKI Fitting

```bash
# Diffusional Kurtosis Imaging (requires multi-shell data)
DIFFCALC \
  --dwi sub-01_proc_DMC.nii.gz \
  --bvals sub-01_proc.bval \
  --bvecs sub-01_proc.bvec \
  --model DKI \
  --output sub-01_DKI

# Outputs DKI metrics:
# - FA, MD, AD, RD (as above)
# - sub-01_DKI_MK.nii.gz (Mean Kurtosis)
# - sub-01_DKI_AK.nii.gz (Axial Kurtosis)
# - sub-01_DKI_RK.nii.gz (Radial Kurtosis)
# - sub-01_DKI_KFA.nii.gz (Kurtosis FA)
```

### MAPMRI (Advanced)

```bash
# Mean Apparent Propagator (MAP) MRI
DIFFCALC \
  --dwi sub-01_proc_DMC.nii.gz \
  --bvals sub-01_proc.bval \
  --bvecs sub-01_proc.bvec \
  --model MAPMRI \
  --output sub-01_MAPMRI
```

## Quality Control

### Visual QC

```bash
# TORTOISE generates QC images during DIFFPREP

# Check outputs
ls sub-01_proc_QC/
# - motion_plots.png
# - eddy_current_check.png
# - epi_correction_check.png
# - registration_check.png

# View with image viewer
eog sub-01_proc_QC/*.png
# or
fslview sub-01_proc_QC/*.nii.gz
```

### Motion and Eddy Assessment

```bash
# Check motion parameters
cat sub-01_proc_motion.txt

# Framewise displacement calculation
python << EOF
import numpy as np

# Load motion parameters (6 columns: 3 rotation, 3 translation)
motion = np.loadtxt('sub-01_proc_motion.txt')

# Calculate framewise displacement
# Convert rotations (radians) to mm at 50mm radius
rot_mm = motion[:, :3] * 50
trans = motion[:, 3:]

# FD = sum of absolute derivatives
fd = np.sum(np.abs(np.diff(rot_mm, axis=0)), axis=1) + \
     np.sum(np.abs(np.diff(trans, axis=0)), axis=1)

print(f'Mean FD: {np.mean(fd):.3f} mm')
print(f'Max FD: {np.max(fd):.3f} mm')
print(f'Volumes with FD > 2mm: {np.sum(fd > 2)}')
EOF
```

### Gradient Table Verification

```bash
# Check gradient orientations after correction
cat sub-01_proc.bvec

# Visualize gradient scheme
python << EOF
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

bvecs = np.loadtxt('sub-01_proc.bvec').T
bvals = np.loadtxt('sub-01_proc.bval')

# Plot only DWI directions (b>0)
dwi_idx = bvals > 50
dwi_bvecs = bvecs[dwi_idx]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dwi_bvecs[:, 0], dwi_bvecs[:, 1], dwi_bvecs[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gradient Directions')
plt.savefig('gradient_scheme.png')
EOF
```

## Registration to Structural

### DTI to T2 Registration

```bash
# Register DTI to structural T2
DIFFPREP_STRUCT_REG \
  --dwi sub-01_proc_DMC.nii.gz \
  --structural T2.nii.gz \
  --output sub-01_to_T2

# Outputs:
# - sub-01_to_T2_dwi.nii.gz (registered DWI)
# - sub-01_to_T2_transform.mat (transformation matrix)
# - sub-01_to_T2_FA.nii.gz (FA in structural space)
```

### Apply Transform to DTI Maps

```bash
# Apply registration to DTI metrics
TORTOISE_APPLY_TRANSFORM \
  --input sub-01_DTI_FA.nii.gz \
  --reference T2.nii.gz \
  --transform sub-01_to_T2_transform.mat \
  --output sub-01_FA_in_T2space.nii.gz

# Repeat for other metrics (MD, AD, RD, etc.)
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# Batch TORTOISE preprocessing

subjects=(sub-01 sub-02 sub-03 sub-04)

for subj in "${subjects[@]}"; do
    echo "Processing ${subj}..."

    # Input files
    dwi="${subj}/dwi.nii.gz"
    bval="${subj}/dwi.bval"
    bvec="${subj}/dwi.bvec"
    output="${subj}/${subj}_proc"

    # Run DIFFPREP
    DIFFPREP \
      --dwi ${dwi} \
      --bvals ${bval} \
      --bvecs ${bvec} \
      --phase vertical \
      --do_QC 1 \
      --step all \
      --output_prefix ${output}

    # DTI fitting
    DIFFCALC \
      --dwi ${output}_DMC.nii.gz \
      --bvals ${output}.bval \
      --bvecs ${output}.bvec \
      --output ${subj}/${subj}_DTI

    echo "${subj} complete"
done
```

### Parallel Processing

```bash
# GNU Parallel
parallel -j 4 \
  'DIFFPREP --dwi {}/dwi.nii.gz --bvals {}/dwi.bval --bvecs {}/dwi.bvec --output_prefix {}/{}_proc' \
  ::: sub-*/

# SLURM array job
#!/bin/bash
#SBATCH --array=1-50
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)

DIFFPREP \
  --dwi ${SUBJECT}/dwi.nii.gz \
  --bvals ${SUBJECT}/dwi.bval \
  --bvecs ${SUBJECT}/dwi.bvec \
  --output_prefix ${SUBJECT}/${SUBJECT}_proc
```

## Integration with Other Tools

### Export to MRtrix3

```bash
# Convert TORTOISE output for MRtrix3
mrconvert sub-01_proc_DMC.nii.gz sub-01_dwi.mif \
  -fslgrad sub-01_proc.bvec sub-01_proc.bval

# Or import TORTOISE gradients
dwi2tensor sub-01_dwi.mif -fslgrad sub-01_proc.bvec sub-01_proc.bval tensor.mif
```

### Export to DSI Studio

```bash
# DSI Studio can read TORTOISE outputs directly
# Create source file in DSI Studio:
# Load: sub-01_proc_DMC.nii.gz
# b-table: sub-01_proc.bval, sub-01_proc.bvec
```

### Export to FSL

```bash
# TORTOISE outputs are compatible with FSL
# Use directly with dtifit, bedpostx, etc.

dtifit \
  -k sub-01_proc_DMC.nii.gz \
  -o sub-01_dtifit \
  -m sub-01_proc_mask.nii.gz \
  -r sub-01_proc.bvec \
  -b sub-01_proc.bval
```

## Integration with Claude Code

When helping users with TORTOISE:

1. **Check Installation:**
   ```bash
   which DIFFPREP
   DIFFPREP --help
   echo $TORTOISEPATH
   ```

2. **Common Issues:**
   - Phase encoding direction incorrect
   - Gradient table format issues
   - Insufficient memory for processing
   - Missing structural image for registration
   - Blip-up/down acquisitions not matched

3. **Best Practices:**
   - Always run quality control
   - Use DR-BUDDI if blip-up/down available
   - Check motion parameters
   - Verify gradient orientations
   - Use structural registration when possible
   - Save intermediate files for debugging
   - Document processing parameters

4. **Parameter Recommendations:**
   - Denoising: Enable for SNR < 20
   - Outlier detection: 0.05 probability threshold
   - Motion correction: Use robust estimation
   - EPI correction: Requires accurate phase direction
   - Multi-shell: Use DKI or MAPMRI models

## Troubleshooting

**Problem:** DIFFPREP fails at motion correction
**Solution:** Check data quality, reduce outlier threshold, verify input format

**Problem:** EPI distortion correction insufficient
**Solution:** Use DR-BUDDI with blip-up/down, verify phase encoding direction, check structural alignment

**Problem:** DTI metrics look wrong
**Solution:** Verify gradient table orientations, check preprocessing quality, ensure b-values correct

**Problem:** Gradient directions flipped
**Solution:** Check bvec file sign conventions, compare with FSL/MRtrix outputs

**Problem:** Out of memory errors
**Solution:** Reduce number of volumes processed simultaneously, increase system memory, or downsample

## Resources

- Website: https://tortoise.nibib.nih.gov/
- User Guide: https://tortoise.nibib.nih.gov/tortoise-user-guide
- Forum: https://tortoise.nibib.nih.gov/forum
- YouTube: TORTOISE Tutorial Videos
- Publications: https://tortoise.nibib.nih.gov/publications

## Citation

```bibtex
@article{pierpaoli2010tortoise,
  title={TORTOISE: an integrated software package for processing of diffusion MRI data},
  author={Pierpaoli, Carlo and Walker, Lindsay and Irfanoglu, M Okan and Barnett, Alex and Basser, Peter and Chang, Lin-Ching and Koay, Cheng and Pajevic, Sinisa and Rohde, Gustavo and Sarlls, Joelle and others},
  journal={ISMRM},
  volume={18},
  pages={1597},
  year={2010}
}

@article{irfanoglu2012drbuddi,
  title={DR-BUDDI (Diffeomorphic Registration for Blip-Up blip-Down Diffusion Imaging) method for correcting echo planar imaging distortions},
  author={Irfanoglu, M Okan and Modi, Parita and Nayak, Amritha and Hutchinson, Elizabeth B and Sarlls, Joelle and Pierpaoli, Carlo},
  journal={NeuroImage},
  volume={106},
  pages={284--299},
  year={2015}
}
```

## Related Tools

- **FSL (EDDY):** Alternative eddy/motion correction
- **MRtrix3:** Comprehensive diffusion analysis
- **DSI Studio:** Tractography and reconstruction
- **DIPY:** Python-based diffusion toolkit
- **ANTs:** For structural registration
- **DTI-TK:** DTI-specific registration
