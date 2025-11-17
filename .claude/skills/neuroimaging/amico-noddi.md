# AMICO and NODDI - Microstructure Modeling

## Overview

AMICO (Accelerated Microstructure Imaging via Convex Optimization) and NODDI (Neurite Orientation Dispersion and Density Imaging) are complementary frameworks for estimating microstructural properties of brain tissue from diffusion MRI. NODDI provides a biophysical model distinguishing intracellular, extracellular, and CSF compartments, while AMICO accelerates the fitting process from hours to minutes using linear optimization. Together, they enable fast, robust estimation of neurite density and orientation dispersion for clinical and research applications.

**AMICO Website:** https://github.com/daducci/AMICO
**NODDI Website:** http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab
**Platform:** Python (AMICO), MATLAB (NODDI)
**Language:** Python/MATLAB
**License:** BSD (AMICO), Academic (NODDI)

## Key Features

### AMICO
- 1000x faster than traditional NODDI fitting
- Linear convex optimization
- Multiple model support (NODDI, NODDI-DTI, AxCaliber, SANDI)
- GPU acceleration
- Robust to noise
- Python-based, easy installation
- Multi-shell DWI support

### NODDI
- Biophysical tissue model
- Three compartments: intra/extracellular, CSF
- Neurite Density Index (NDI)
- Orientation Dispersion Index (ODI)
- Free water fraction
- Clinically validated
- Multi-shell acquisition protocol

## Installation

### AMICO (Python)

```bash
# Install via pip
pip install dmri-amico

# Or from GitHub
git clone https://github.com/daducci/AMICO.git
cd AMICO
pip install .

# Verify installation
python -c "import amico; print(amico.__version__)"

# Install dependencies
pip install numpy scipy dipy nibabel
```

### NODDI Toolbox (MATLAB)

```bash
# Download from: http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab

# Extract toolbox
unzip NODDI_toolbox_v1.05.zip

# Add to MATLAB path
matlab -nodisplay -r "addpath('/path/to/NODDI_toolbox'); savepath; exit"

# Verify in MATLAB
which CreateROI
which batch_fitting
```

## Data Requirements

### Multi-Shell Acquisition

```bash
# NODDI requires multi-shell DWI:
# - At least 2 non-zero b-shells
# - Recommended protocol:
#   - b=0 (5-10 volumes)
#   - b=1000 s/mm² (30+ directions)
#   - b=2000 s/mm² (60+ directions)

# Minimum acceptable:
#   - b=0, b=700, b=2000 s/mm²
#   - 20+ directions per shell

# Example bval file:
# 0 0 0 1000 1000 1000 ... 2000 2000 2000 ...
```

### Preprocessing

```bash
# Standard preprocessing required:
# 1. Denoising (optional but recommended)
# 2. Gibbs ringing removal (optional)
# 3. Motion and eddy current correction
# 4. Brain extraction

# Using MRtrix3
dwidenoise dwi.mif dwi_denoised.mif
mrdegibbs dwi_denoised.mif dwi_degibbs.mif
dwifslpreproc dwi_degibbs.mif dwi_preproc.mif -rpe_none -pe_dir AP
dwi2mask dwi_preproc.mif mask.mif

# Convert to NIfTI for AMICO/NODDI
mrconvert dwi_preproc.mif dwi.nii.gz -export_grad_fsl bvecs bvals
mrconvert mask.mif mask.nii.gz
```

## AMICO Usage

### Basic AMICO-NODDI

```python
#!/usr/bin/env python
"""
AMICO-NODDI fitting example
"""
import amico

# Setup AMICO
amico.core.setup()

# Load study
ae = amico.Evaluation(".", "sub-01")

# Load data
ae.load_data(
    dwi_filename="dwi.nii.gz",
    scheme_filename="protocol.scheme",
    mask_filename="mask.nii.gz",
    b0_thr=10
)

# Set model
ae.set_model("NODDI")

# Generate kernels (run once per protocol)
ae.generate_kernels(regenerate=True)

# Load kernels
ae.load_kernels()

# Fit model
ae.fit()

# Save results
ae.save_results()

print("NODDI fitting complete!")
```

### Create AMICO Scheme File

```python
#!/usr/bin/env python
"""
Convert FSL bvals/bvecs to AMICO scheme format
"""
import numpy as np

# Load bvals and bvecs
bvals = np.loadtxt('bvals')
bvecs = np.loadtxt('bvecs').T

# Create scheme file
# Format: [gx, gy, gz, |G|, Delta, delta, TE]
# For NODDI, use simplified format
scheme = np.zeros((len(bvals), 7))

# Gradient directions
scheme[:, 0:3] = bvecs

# G = sqrt(b / (2.68e8 * gamma^2 * delta^2 * (Delta - delta/3)))
# Simplified for standard protocol:
# Delta = 30ms, delta = 20ms, TE = 80ms
scheme[:, 3] = np.sqrt(bvals / 2000.0)  # Normalized |G|
scheme[:, 4] = 0.03  # Delta (s)
scheme[:, 5] = 0.02  # delta (s)
scheme[:, 6] = 0.08  # TE (s)

# Save
with open('protocol.scheme', 'w') as f:
    f.write(f"VERSION: BVECTOR\n")
    for row in scheme:
        f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} "
                f"{row[4]:.6f} {row[5]:.6f} {row[6]:.6f}\n")
```

### Complete AMICO Pipeline

```python
#!/usr/bin/env python
"""
Complete AMICO-NODDI pipeline
"""
import amico
import nibabel as nib
import numpy as np
import os

# Configuration
subject = "sub-01"
data_dir = "."
output_dir = "AMICO"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Setup
amico.core.setup()

# Initialize
ae = amico.Evaluation(data_dir, subject)

# Load data
print("Loading data...")
ae.load_data(
    dwi_filename="dwi.nii.gz",
    scheme_filename="protocol.scheme",
    mask_filename="mask.nii.gz",
    b0_thr=10
)

# Set model
print("Setting up NODDI model...")
ae.set_model("NODDI")

# Generate kernels (only needed once per protocol)
if not os.path.exists(f"kernels/NODDI"):
    print("Generating kernels...")
    ae.generate_kernels(regenerate=True)

# Load kernels
print("Loading kernels...")
ae.load_kernels()

# Fit model
print("Fitting NODDI model (this may take 5-30 minutes)...")
ae.fit()

# Save results
print("Saving results...")
ae.save_results()

# Outputs saved in AMICO/NODDI/
print("Complete! Results in:", f"{data_dir}/AMICO/NODDI/")
print("  - FIT_ICVF.nii.gz (Neurite Density Index)")
print("  - FIT_OD.nii.gz (Orientation Dispersion Index)")
print("  - FIT_ISOVF.nii.gz (CSF volume fraction)")
```

## NODDI Toolbox (MATLAB)

### Basic NODDI Fitting

```matlab
%% NODDI fitting using MATLAB toolbox

% Create protocol
protocol = FSL2Protocol('bvals', 'bvecs');

% Save protocol
SaveProtocol(protocol, 'NODDI_protocol.mat');

% Create ROI (whole brain)
noddi = MakeModel('WatsonSHStickTortIsoV_B0');

% Set up batch
batch_create_roi_single_voxel;

% Or create full brain ROI
brain_mask = niftiread('mask.nii.gz');
roi = find(brain_mask > 0);

CreateROI('dwi.nii.gz', brain_mask, 'NODDI_roi.mat');

% Fit NODDI model
batch_fitting('NODDI_protocol.mat', 'NODDI_roi.mat', 'NODDI_fit.mat', 4);

% Save results as NIfTI
SaveParamsAsNIfTI('NODDI_fit.mat', 'NODDI_roi.mat', brain_mask, 'NODDI_');

% Outputs:
% - NODDI_ficvf.nii.gz (Intracellular volume fraction = NDI)
% - NODDI_odi.nii.gz (Orientation dispersion index)
% - NODDI_fiso.nii.gz (Isotropic volume fraction = CSF)
```

### Custom Protocol

```matlab
%% Define custom acquisition protocol

% Initialize protocol structure
protocol.pulseseq = 'PGSE';  % Pulsed Gradient Spin Echo
protocol.schemetype = 'multishell';

% Define shells
% b = 0 (5 volumes)
protocol.b0_vols = 5;

% b = 1000 (30 directions)
protocol.grad_dirs_1000 = load('grad_dirs_30.txt');
protocol.bval_1000 = 1000;

% b = 2000 (60 directions)
protocol.grad_dirs_2000 = load('grad_dirs_60.txt');
protocol.bval_2000 = 2000;

% Timing parameters
protocol.delta = 20e-3;  % gradient duration (s)
protocol.smalldel = 20e-3;  % same as delta for PGSE
protocol.Delta = 30e-3;  % gradient separation (s)
protocol.TE = 80e-3;  % echo time (s)

% Save
save('custom_protocol.mat', 'protocol');
```

## Output Metrics

### NODDI Parameters

```python
# Main NODDI outputs (AMICO):

# 1. NDI (Neurite Density Index) = FIT_ICVF.nii.gz
#    - Intracellular volume fraction
#    - Range: 0-1
#    - Higher = more neurites
#    - Sensitive to axon/dendrite density

# 2. ODI (Orientation Dispersion Index) = FIT_OD.nii.gz
#    - Dispersion of neurite orientations
#    - Range: 0-1
#    - 0 = parallel fibers
#    - 1 = isotropic dispersion

# 3. ISOVF (Isotropic volume fraction) = FIT_ISOVF.nii.gz
#    - Free water fraction (CSF)
#    - Range: 0-1
#    - Useful for partial volume correction
```

### Visualize Results

```bash
# View NDI map
fsleyes T1.nii.gz AMICO/NODDI/FIT_ICVF.nii.gz -cm hot -dr 0 1

# View ODI map
fsleyes T1.nii.gz AMICO/NODDI/FIT_OD.nii.gz -cm cool -dr 0 1

# Create RGB image (direction-encoded)
# Red = NDI, Green = ODI, Blue = ISOVF
python << EOF
import nibabel as nib
import numpy as np

ndi = nib.load('AMICO/NODDI/FIT_ICVF.nii.gz')
odi = nib.load('AMICO/NODDI/FIT_OD.nii.gz')
isovf = nib.load('AMICO/NODDI/FIT_ISOVF.nii.gz')

# Create RGB
rgb = np.stack([
    ndi.get_fdata(),
    odi.get_fdata(),
    isovf.get_fdata()
], axis=-1)

# Normalize to 0-255
rgb = (rgb * 255).astype(np.uint8)

# Save
rgb_img = nib.Nifti1Image(rgb, ndi.affine)
nib.save(rgb_img, 'NODDI_RGB.nii.gz')
EOF
```

## ROI Analysis

### Extract Mean Values

```python
#!/usr/bin/env python
"""
Extract NODDI metrics from ROIs
"""
import nibabel as nib
import numpy as np

# Load NODDI maps
ndi = nib.load('AMICO/NODDI/FIT_ICVF.nii.gz').get_fdata()
odi = nib.load('AMICO/NODDI/FIT_OD.nii.gz').get_fdata()
isovf = nib.load('AMICO/NODDI/FIT_ISOVF.nii.gz').get_fdata()

# Load ROI atlas
atlas = nib.load('atlas.nii.gz').get_fdata()

# Extract metrics for each ROI
roi_ids = np.unique(atlas[atlas > 0])
results = []

for roi_id in roi_ids:
    mask = atlas == roi_id

    result = {
        'ROI': int(roi_id),
        'NDI_mean': np.mean(ndi[mask]),
        'NDI_std': np.std(ndi[mask]),
        'ODI_mean': np.mean(odi[mask]),
        'ODI_std': np.std(odi[mask]),
        'ISOVF_mean': np.mean(isovf[mask]),
        'ISOVF_std': np.std(isovf[mask]),
    }
    results.append(result)

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('NODDI_ROI_metrics.csv', index=False)
print(df)
```

## Batch Processing

### AMICO Batch Script

```python
#!/usr/bin/env python
"""
Batch AMICO-NODDI processing
"""
import amico
import os

# Setup
amico.core.setup()

# Subjects
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']

for subject in subjects:
    print(f"\nProcessing {subject}...")

    # Subject directory
    subject_dir = f"data/{subject}"

    # Initialize
    ae = amico.Evaluation(subject_dir, subject)

    # Load data
    try:
        ae.load_data(
            dwi_filename="dwi.nii.gz",
            scheme_filename="protocol.scheme",
            mask_filename="mask.nii.gz"
        )
    except Exception as e:
        print(f"  Error loading {subject}: {e}")
        continue

    # Set model
    ae.set_model("NODDI")

    # Generate kernels (only once)
    if subject == subjects[0]:
        ae.generate_kernels(regenerate=True)

    # Load kernels
    ae.load_kernels()

    # Fit
    ae.fit()

    # Save
    ae.save_results()

    print(f"  {subject} complete!")

print("\nAll subjects processed!")
```

### Parallel Processing

```bash
# Process subjects in parallel
#!/bin/bash

subjects=(sub-01 sub-02 sub-03 sub-04)

# GNU Parallel
parallel -j 4 'python fit_noddi.py {}' ::: "${subjects[@]}"

# Or SLURM array
#!/bin/bash
#SBATCH --array=1-50
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)

python << EOF
import amico
amico.core.setup()
ae = amico.Evaluation("data/${SUBJECT}", "${SUBJECT}")
ae.load_data("dwi.nii.gz", "protocol.scheme", "mask.nii.gz")
ae.set_model("NODDI")
ae.load_kernels()
ae.fit()
ae.save_results()
EOF
```

## Advanced Models

### NODDI-DTI

```python
# Combine NODDI with DTI metrics
ae.set_model("NODDI")
ae.fit()

# Also compute DTI
import dipy.reconst.dti as dti
from dipy.io.image import load_nifti

data, affine = load_nifti('dwi.nii.gz')
bvals = np.loadtxt('bvals')
bvecs = np.loadtxt('bvecs').T

from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs)

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data, mask=mask)

# Save DTI metrics alongside NODDI
nib.save(nib.Nifti1Image(tenfit.fa, affine), 'DTI_FA.nii.gz')
nib.save(nib.Nifti1Image(tenfit.md, affine), 'DTI_MD.nii.gz')
```

### AxCaliber

```python
# AxCaliber model for axon diameter estimation
ae.set_model("AxCaliber")
ae.generate_kernels()
ae.load_kernels()
ae.fit()
ae.save_results()

# Outputs axon diameter distributions
```

## Integration with Claude Code

When helping users with AMICO/NODDI:

1. **Check Installation:**
   ```bash
   python -c "import amico; print(amico.__version__)"
   # MATLAB: which batch_fitting
   ```

2. **Common Issues:**
   - Single-shell data (need multi-shell)
   - Wrong scheme file format
   - Insufficient b-value range
   - Missing brain mask
   - Memory errors with large datasets

3. **Best Practices:**
   - Use recommended acquisition protocol
   - Preprocess data thoroughly
   - Generate kernels once per protocol
   - Visual QC of all outputs
   - Check for outliers in NDI/ODI
   - Compare with FA for validation
   - Document acquisition parameters

4. **Quality Checks:**
   - NDI: 0.3-0.8 in white matter
   - ODI: 0.1-0.5 in white matter (higher in crossing)
   - ISOVF: near 0 in tissue, 1 in ventricles
   - Visual inspection for artifacts
   - Compare with anatomical images

## Troubleshooting

**Problem:** "Not enough shells" error
**Solution:** Ensure at least 2 non-zero b-values, check bval file

**Problem:** Negative or NaN values in output
**Solution:** Check preprocessing quality, verify scheme file correct, ensure adequate SNR

**Problem:** Very slow fitting
**Solution:** Use AMICO instead of MATLAB NODDI, enable GPU, reduce voxels

**Problem:** Unrealistic parameter values
**Solution:** Check acquisition parameters in scheme, verify timing parameters, validate data quality

**Problem:** Kernel generation fails
**Solution:** Check AMICO installation, verify protocol format, try regenerate=True

## Resources

- AMICO GitHub: https://github.com/daducci/AMICO
- NODDI Website: http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab
- NODDI Paper: Zhang et al. (2012) NeuroImage
- AMICO Paper: Daducci et al. (2015) NeuroImage
- Acquisition Protocol: http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDIprotocol

## Citation

```bibtex
@article{zhang2012noddi,
  title={NODDI: practical in vivo neurite orientation dispersion and density imaging of the human brain},
  author={Zhang, Hui and Schneider, Torben and Wheeler-Kingshott, Claudia A and Alexander, Daniel C},
  journal={Neuroimage},
  volume={61},
  number={4},
  pages={1000--1016},
  year={2012}
}

@article{daducci2015amico,
  title={Accelerated microstructure imaging via convex optimization (AMICO) from diffusion MRI data},
  author={Daducci, Alessandro and Canales-Rodr{\'\i}guez, Erick J and Zhang, Hui and Dyrby, Tim B and Alexander, Daniel C and Thiran, Jean-Philippe},
  journal={NeuroImage},
  volume={105},
  pages={32--44},
  year={2015}
}
```

## Related Tools

- **DIPY:** Python diffusion toolkit
- **MRtrix3:** Preprocessing and CSD
- **MDT:** Multi-compartment modeling
- **DMIPY:** Microstructure modeling framework
- **FSL:** BEDPOSTX multi-fiber model
- **Camino:** Diffusion toolkit with various models
