# SIRF - Synergistic Image Reconstruction Framework

## Overview

**SIRF** (Synergistic Image Reconstruction Framework) is a comprehensive Python/C++ framework for PET and MR image reconstruction, developed by the Collaborative Computational Project in Positron Emission Tomography and Magnetic Resonance imaging (CCP PET-MR). SIRF provides a unified, high-level interface to multiple reconstruction engines—**STIR** for PET and **Gadgetron** for MR—enabling researchers to perform synergistic multi-modal reconstruction, motion correction, and joint PET-MR optimization.

SIRF's key innovation is its ability to combine information from PET and MR within the reconstruction process itself, not just at the post-processing stage. This enables advanced techniques like MR-guided PET reconstruction, motion estimation from MR applied to PET, and anatomically-informed regularization.

**Key Features:**
- Unified Python interface to STIR (PET) and Gadgetron (MR)
- Synergistic PET-MR reconstruction
- Motion-corrected PET and MR reconstruction
- Multi-modal prior-based reconstruction
- Anatomically-guided regularization
- Iterative reconstruction algorithms (OSEM, gradient-based)
- Joint estimation of activity and motion parameters
- GPU acceleration support (via STIR and Gadgetron)
- Integration with NiftyReg and SPM for registration
- Educational framework for teaching reconstruction concepts
- Open-source and extensible architecture
- Docker containers for easy deployment

**Primary Use Cases:**
- PET-MR simultaneous imaging studies
- Motion-corrected PET reconstruction using MR navigators
- Anatomically-guided PET reconstruction
- Reconstruction algorithm development and validation
- Multi-modal imaging research
- Teaching medical image reconstruction
- Clinical PET-MR protocol optimization

**Official Documentation:** https://www.ccppetmr.ac.uk/sites/sirf

---

## Installation

### Prerequisites

**System Requirements:**
- Linux (Ubuntu 18.04+, CentOS 7+) or macOS
- 16GB+ RAM recommended
- Multi-core CPU
- Optional: NVIDIA GPU with CUDA for acceleration
- Python 3.6 or later

**Dependencies:**
- CMake 3.10+
- C++ compiler (GCC 7+, Clang)
- Python 3.6+
- STIR 4.0+ (PET reconstruction engine)
- Gadgetron 4.0+ (MR reconstruction engine)
- Optional: NiftyReg, SPM for registration
- Optional: CUDA for GPU acceleration

### Installation via Docker (Recommended)

```bash
# Pull SIRF Docker image (easiest method)
docker pull synerbi/sirf:latest

# Run SIRF container
docker run -it --rm \
  -v /path/to/data:/data \
  -p 8888:8888 \
  synerbi/sirf:latest

# Inside container, verify installation
python -c "import sirf.STIR; print(sirf.STIR.__version__)"
python -c "import sirf.Gadgetron; print(sirf.Gadgetron.__version__)"
```

### Installation from Source (Linux)

```bash
# Install system dependencies (Ubuntu)
sudo apt-get update
sudo apt-get install -y git cmake g++ libboost-all-dev \
  libhdf5-serial-dev python3-dev python3-pip \
  libfftw3-dev libarmadillo-dev libace-dev \
  libgtest-dev libplplot-dev swig

# Create installation directory
mkdir -p $HOME/devel
cd $HOME/devel

# Install STIR (PET reconstruction engine)
git clone https://github.com/UCL/STIR.git
cd STIR
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/devel/install \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install
cd ../..

# Install Gadgetron (MR reconstruction engine)
git clone https://github.com/gadgetron/gadgetron.git
cd gadgetron
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/devel/install \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install
cd ../..

# Install SIRF
git clone https://github.com/SyneRBI/SIRF.git
cd SIRF
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/devel/install \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)
make install

# Set environment variables
export PATH=$HOME/devel/install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/devel/install/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/devel/install/python:$PYTHONPATH

# Verify installation
python3 -c "import sirf.STIR as pet; print('SIRF PET loaded successfully')"
python3 -c "import sirf.Gadgetron as mr; print('SIRF MR loaded successfully')"
```

### Installation via Conda (Alternative)

```bash
# Create conda environment
conda create -n sirf python=3.9
conda activate sirf

# Install SIRF and dependencies via conda-forge
conda install -c conda-forge sirf stir gadgetron

# Verify
python -c "import sirf; print(sirf.__version__)"
```

---

## Basic PET Reconstruction with SIRF

### Load PET Sinogram Data

```python
import sirf.STIR as pet
from sirf.Utilities import show_2D_array
import matplotlib.pyplot as plt

# Set up message redirection
pet.MessageRedirector('info.txt', 'warn.txt')

# Load raw PET data (sinogram)
sino_file = '/data/pet/sino.hs'  # STIR format
acq_data = pet.AcquisitionData(sino_file)

print(f"Sinogram dimensions: {acq_data.dimensions()}")
print(f"Number of sinograms: {acq_data.get_num_sinograms()}")

# Display sinogram
sino_array = acq_data.as_array()
show_2D_array('Sinogram', sino_array[0, :, :])
plt.savefig('sinogram.png', dpi=300)
```

### Load Attenuation Image

```python
# Load attenuation map (μ-map)
attn_file = '/data/pet/attn_map.hv'
attn_image = pet.ImageData(attn_file)

print(f"Attenuation map dimensions: {attn_image.dimensions()}")

# Display central slice
attn_array = attn_image.as_array()
show_2D_array('Attenuation Map', attn_array[attn_array.shape[0]//2, :, :])
plt.savefig('attn_map.png', dpi=300)
```

### Basic OSEM Reconstruction

```python
import sirf.STIR as pet

# Load data
acq_data = pet.AcquisitionData('/data/pet/sino.hs')
attn_image = pet.ImageData('/data/pet/attn_map.hv')

# Create acquisition model (system matrix)
acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
acq_model.set_num_tangential_LORs(10)

# Create initial image estimate
init_image = acq_data.create_uniform_image(value=1.0)

# Set up OSEM reconstructor
recon = pet.OSMAPOSLReconstructor()
recon.set_num_subsets(12)
recon.set_num_subiterations(24)  # 2 full iterations

# Set acquisition model and data
recon.set_acquisition_model(acq_model)
recon.set_input(acq_data)
recon.set_up(init_image)

# Perform reconstruction
recon.process()

# Get reconstructed image
recon_image = recon.get_output()

# Display
recon_array = recon_image.as_array()
show_2D_array('Reconstructed PET', recon_array[recon_array.shape[0]//2, :, :])
plt.savefig('pet_recon.png', dpi=300)

# Save result
recon_image.write('/data/pet/recon_osem.hv')
```

### Apply Attenuation Correction

```python
import sirf.STIR as pet

# Load data
acq_data = pet.AcquisitionData('/data/pet/sino.hs')
attn_image = pet.ImageData('/data/pet/attn_map.hv')

# Create acquisition model with attenuation
acq_model = pet.AcquisitionModelUsingRayTracingMatrix()

# Create attenuation model
asm_attn = pet.AcquisitionSensitivityModel(attn_image, acq_model)
asm_attn.set_up(acq_data)

# Compute attenuation factors
ac_factors = acq_data.get_uniform_copy(1.0)
asm_attn.unnormalise(ac_factors)

# Apply to acquisition model
acq_model.set_acquisition_sensitivity(asm_attn)

# Set up OSEM with attenuation correction
init_image = acq_data.create_uniform_image(value=1.0)
recon = pet.OSMAPOSLReconstructor()
recon.set_num_subsets(12)
recon.set_num_subiterations(24)
recon.set_acquisition_model(acq_model)
recon.set_input(acq_data)
recon.set_up(init_image)

# Reconstruct
recon.process()
recon_image_ac = recon.get_output()

# Save
recon_image_ac.write('/data/pet/recon_osem_ac.hv')
print("Attenuation-corrected reconstruction complete")
```

---

## MR Reconstruction with SIRF

### Load MR Raw Data

```python
import sirf.Gadgetron as mr
from sirf.Utilities import show_2D_array

# Set up message redirection
mr.MessageRedirector('info_mr.txt', 'warn_mr.txt')

# Load raw k-space data (Gadgetron format)
kspace_file = '/data/mr/kspace.h5'
acq_data_mr = mr.AcquisitionData(kspace_file)

print(f"MR acquisition dimensions: {acq_data_mr.dimensions()}")
print(f"Number of readouts: {acq_data_mr.number()}")

# Get k-space data as array
kspace_array = acq_data_mr.as_array()
print(f"K-space shape: {kspace_array.shape}")
```

### Basic MR Reconstruction

```python
import sirf.Gadgetron as mr

# Load k-space data
acq_data_mr = mr.AcquisitionData('/data/mr/kspace.h5')

# Set up Cartesian MR reconstructor
recon_mr = mr.CartesianGRAPPAReconstructor()

# Preprocess acquisitions
processed_acq = recon_mr.process(acq_data_mr)

# Reconstruct image
recon_mr.reconstruct(processed_acq)
mr_image = recon_mr.get_output()

# Display
mr_array = mr_image.as_array()
show_2D_array('MR Image', abs(mr_array[0, :, :]))
plt.savefig('mr_recon.png', dpi=300)

# Save
mr_image.write('/data/mr/recon_grappa.h5')
```

---

## Motion Correction

### Estimate Motion from MR Navigators

```python
import sirf.Gadgetron as mr
import sirf.Reg as reg
import numpy as np

# Load dynamic MR data with navigators
acq_data_dynamic = mr.AcquisitionData('/data/mr/dynamic_with_nav.h5')

# Reconstruct navigator images (simplified)
recon = mr.FullySampledReconstructor()

# Extract individual time frames
n_frames = 10
mr_frames = []

for frame in range(n_frames):
    # Extract frame data (implementation depends on data structure)
    # frame_acq = extract_frame(acq_data_dynamic, frame)

    # Reconstruct frame
    # frame_img = recon.reconstruct(frame_acq)
    # mr_frames.append(frame_img)
    pass

# Register frames to estimate motion
ref_frame = 0  # Use first frame as reference

for i in range(1, n_frames):
    # Set up registration using NiftyReg
    # SIRF integrates with NiftyReg for registration

    # Estimate transformation
    # transformation = reg.NiftyAladinSym()
    # transformation.set_reference_image(mr_frames[ref_frame])
    # transformation.set_floating_image(mr_frames[i])
    # transformation.process()

    # Get motion parameters
    # motion_params = transformation.get_transformation_matrix()

    print(f"Frame {i}: Motion estimated")

print("Motion estimation complete")
```

### Apply Motion Correction to PET

```python
import sirf.STIR as pet
import sirf.Reg as reg
import numpy as np

# Load PET list-mode data
listmode_file = '/data/pet/listmode.l.hdr'
lm_data = pet.ListmodeData(listmode_file)

# Divide into time frames
frame_starts = [0, 300, 600, 900, 1200]  # seconds
frame_ends = [300, 600, 900, 1200, 1800]

# Motion transformations from MR (simplified)
motion_transforms = []  # List of transformation matrices

# Reconstruct with motion correction
acq_model = pet.AcquisitionModelUsingRayTracingMatrix()

for frame_idx in range(len(frame_starts)):
    # Extract frame from list-mode
    # (SIRF provides utilities for this)

    # Apply motion transformation to acquisition model
    if frame_idx > 0:
        # Transform acquisition model
        # motion_transform = motion_transforms[frame_idx]
        # acq_model.set_transformation(motion_transform)
        pass

    # Reconstruct frame with motion correction
    # (Reconstruction code as before)

    print(f"Frame {frame_idx} reconstructed with motion correction")

print("Motion-corrected PET reconstruction complete")
```

---

## Synergistic PET-MR Reconstruction

### MR-Guided PET Reconstruction

```python
import sirf.STIR as pet
import sirf.Reg as reg
import numpy as np

# Load PET data
acq_data_pet = pet.AcquisitionData('/data/pet/sino.hs')

# Load co-registered MR image
mr_image = pet.ImageData('/data/mr/t1w_in_pet_space.hv')

# Create anatomical prior from MR
# Use edge-preserving regularization guided by MR

# Compute MR gradients for edge information
mr_array = mr_image.as_array()
grad_x = np.gradient(mr_array, axis=2)
grad_y = np.gradient(mr_array, axis=1)
grad_z = np.gradient(mr_array, axis=0)
edge_map = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

# Normalize edge map
edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min())

# Use edge map in PET reconstruction prior
# (Requires custom implementation or SIRF plugins)

# Set up OSEM with anatomical prior
acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
init_image = acq_data_pet.create_uniform_image(value=1.0)

recon = pet.OSMAPOSLReconstructor()
recon.set_num_subsets(12)
recon.set_num_subiterations(24)
recon.set_acquisition_model(acq_model)
recon.set_input(acq_data_pet)

# Set up prior (example: quadratic prior)
prior = pet.QuadraticPrior()
prior.set_penalisation_factor(0.5)
prior.set_up(init_image)
recon.set_prior(prior)

recon.set_up(init_image)
recon.process()

recon_with_prior = recon.get_output()
recon_with_prior.write('/data/pet/recon_mr_guided.hv')
print("MR-guided PET reconstruction complete")
```

### Joint PET-MR Reconstruction

```python
import sirf.STIR as pet
import sirf.Gadgetron as mr

# Conceptual example of joint PET-MR reconstruction
# (Full implementation requires advanced SIRF features)

# Load PET and MR data
acq_data_pet = pet.AcquisitionData('/data/pet/sino.hs')
acq_data_mr = mr.AcquisitionData('/data/mr/kspace.h5')

# Initial estimates
pet_image_init = acq_data_pet.create_uniform_image(value=1.0)
mr_recon_init = mr.FullySampledReconstructor()
mr_image_init = mr_recon_init.reconstruct(acq_data_mr)

# Alternating optimization (simplified)
n_outer_iterations = 3

for outer_iter in range(n_outer_iterations):
    print(f"Joint reconstruction iteration {outer_iter + 1}")

    # 1. Reconstruct PET using current MR as prior
    acq_model_pet = pet.AcquisitionModelUsingRayTracingMatrix()
    recon_pet = pet.OSMAPOSLReconstructor()
    recon_pet.set_num_subsets(12)
    recon_pet.set_num_subiterations(12)
    recon_pet.set_acquisition_model(acq_model_pet)
    recon_pet.set_input(acq_data_pet)
    # Set MR-derived prior here
    recon_pet.set_up(pet_image_init)
    recon_pet.process()
    pet_image_updated = recon_pet.get_output()

    # 2. Reconstruct MR using current PET information
    # (If using PET for MR coil sensitivity estimation, etc.)
    recon_mr = mr.FullySampledReconstructor()
    mr_image_updated = recon_mr.reconstruct(acq_data_mr)

    # Update initial estimates
    pet_image_init = pet_image_updated
    mr_image_init = mr_image_updated

# Save final results
pet_image_updated.write('/data/results/pet_joint.hv')
mr_image_updated.write('/data/results/mr_joint.h5')
print("Joint PET-MR reconstruction complete")
```

---

## Registration within SIRF

### Register MR to PET Using NiftyReg

```python
import sirf.Reg as reg
import sirf.STIR as pet
import sirf.Gadgetron as mr

# Load PET and MR images
pet_image = pet.ImageData('/data/pet/pet_recon.hv')
mr_image = pet.ImageData('/data/mr/t1w.hv')  # Converted to STIR format

# Set up NiftyReg rigid registration
niftyreg = reg.NiftyAladinSym()
niftyreg.set_reference_image(pet_image)
niftyreg.set_floating_image(mr_image)

# Set registration parameters
niftyreg.set_parameter('SetPerformRigid', '1')
niftyreg.set_parameter('SetPerformAffine', '0')

# Perform registration
niftyreg.process()

# Get registered MR image
mr_registered = niftyreg.get_output()
mr_registered.write('/data/results/mr_in_pet_space.hv')

# Get transformation
transformation = niftyreg.get_transformation_matrix()
print(f"Transformation matrix:\n{transformation}")

# Save transformation
niftyreg.get_transformation().write('/data/results/mr_to_pet_transform')
print("Registration complete")
```

### Apply Transformation to Images

```python
import sirf.Reg as reg
import sirf.STIR as pet

# Load transformation
transform_file = '/data/results/mr_to_pet_transform.txt'
transformation = reg.AffineTransformation(transform_file)

# Load image to transform
mr_image = pet.ImageData('/data/mr/t1w.hv')

# Load reference image (defines output space)
pet_reference = pet.ImageData('/data/pet/pet_recon.hv')

# Create resampler
resampler = reg.NiftyResample()
resampler.set_reference_image(pet_reference)
resampler.set_floating_image(mr_image)
resampler.set_interpolation_type_to_linear()
resampler.add_transformation(transformation)

# Resample
resampler.process()
mr_resampled = resampler.get_output()

# Save
mr_resampled.write('/data/results/mr_resampled_to_pet.hv')
print("Image resampled to PET space")
```

---

## Advanced Features

### Gradient-Based Reconstruction

```python
import sirf.STIR as pet

# Load data
acq_data = pet.AcquisitionData('/data/pet/sino.hs')

# Create acquisition model
acq_model = pet.AcquisitionModelUsingRayTracingMatrix()

# Initial image
init_image = acq_data.create_uniform_image(value=1.0)

# Set up gradient-based reconstructor (e.g., FISTA)
# Note: SIRF provides framework, specific algorithms may require extensions

# Create objective function
obj_fun = pet.make_Poisson_loglikelihood(acq_data)
obj_fun.set_acquisition_model(acq_model)
obj_fun.set_up(init_image)

# Add prior
prior = pet.QuadraticPrior()
prior.set_penalisation_factor(0.1)
prior.set_up(init_image)
obj_fun.set_prior(prior)

# Create reconstructor
recon = pet.OSMAPOSLReconstructor()
recon.set_objective_function(obj_fun)
recon.set_num_subsets(1)  # Full gradient
recon.set_num_subiterations(100)
recon.set_up(init_image)

# Reconstruct
recon.process()
recon_gradient = recon.get_output()

recon_gradient.write('/data/results/recon_gradient.hv')
```

### Custom Priors and Regularization

```python
import sirf.STIR as pet
import numpy as np

# Load data and set up basic reconstruction
acq_data = pet.AcquisitionData('/data/pet/sino.hs')
acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
init_image = acq_data.create_uniform_image(value=1.0)

# Try different priors

# 1. Quadratic prior (smoothing)
prior_quad = pet.QuadraticPrior()
prior_quad.set_penalisation_factor(0.5)
prior_quad.set_up(init_image)

# 2. LogCosh prior (edge-preserving)
# prior_logcosh = pet.LogcoshPrior()
# prior_logcosh.set_penalisation_factor(0.3)
# prior_logcosh.set_scalar_parameter(0.1)
# prior_logcosh.set_up(init_image)

# 3. Relative Difference prior (anatomical)
# Requires anatomical image
prior_rd = pet.RelativeDifferencePrior()
prior_rd.set_penalisation_factor(0.5)
prior_rd.set_kappa(pet.ImageData('/data/mr/t1w_edges.hv'))  # Edge map
prior_rd.set_up(init_image)

# Reconstruct with chosen prior
recon = pet.OSMAPOSLReconstructor()
recon.set_num_subsets(12)
recon.set_num_subiterations(36)
recon.set_acquisition_model(acq_model)
recon.set_input(acq_data)
recon.set_prior(prior_rd)
recon.set_up(init_image)
recon.process()

recon_with_anatomical_prior = recon.get_output()
recon_with_anatomical_prior.write('/data/results/recon_anatomical_prior.hv')
```

---

## Integration with Neuroimaging Pipelines

### SIRF to BIDS Workflow

```python
import sirf.STIR as pet
import nibabel as nib
import numpy as np
import json
from pathlib import Path

def sirf_to_bids(sirf_image_path, bids_root, subject, session=None):
    """Convert SIRF PET reconstruction to BIDS format"""

    # Load SIRF image
    pet_image_sirf = pet.ImageData(sirf_image_path)
    pet_array = pet_image_sirf.as_array()

    # Get voxel sizes (from SIRF image)
    vox_sizes = pet_image_sirf.voxel_sizes()

    # Create affine matrix (simplified)
    affine = np.diag([vox_sizes['x'], vox_sizes['y'], vox_sizes['z'], 1.0])

    # Convert to NIfTI
    nii_img = nib.Nifti1Image(pet_array, affine)

    # Create BIDS directory structure
    if session:
        pet_dir = Path(bids_root) / f'sub-{subject}' / f'ses-{session}' / 'pet'
    else:
        pet_dir = Path(bids_root) / f'sub-{subject}' / 'pet'

    pet_dir.mkdir(parents=True, exist_ok=True)

    # Save NIfTI
    if session:
        nii_path = pet_dir / f'sub-{subject}_ses-{session}_pet.nii.gz'
    else:
        nii_path = pet_dir / f'sub-{subject}_pet.nii.gz'

    nib.save(nii_img, nii_path)

    # Create JSON sidecar
    metadata = {
        "Modality": "PT",
        "ReconstructionMethod": "OSEM",
        "ReconstructionSoftware": "SIRF-STIR",
        "NumberOfIterations": 4,
        "NumberOfSubsets": 12,
        "AttenuationCorrection": "MR-based"
    }

    json_path = nii_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"BIDS PET saved: {nii_path}")

# Usage
sirf_to_bids(
    sirf_image_path='/data/pet/recon_osem.hv',
    bids_root='/data/bids_dataset',
    subject='01',
    session='baseline'
)
```

### Integration with FreeSurfer

```bash
#!/bin/bash
# SIRF PET to FreeSurfer pipeline

SUBJECT=sub-01
SUBJECTS_DIR=/data/freesurfer

# 1. Convert SIRF output to NIfTI (using Python script above)
python3 << EOF
import sirf.STIR as pet
import nibabel as nib
import numpy as np

pet_img = pet.ImageData('/data/pet/${SUBJECT}/recon_osem.hv')
pet_array = pet_img.as_array()

vox = pet_img.voxel_sizes()
affine = np.diag([vox['x'], vox['y'], vox['z'], 1.0])

nii = nib.Nifti1Image(pet_array, affine)
nib.save(nii, '/data/pet/${SUBJECT}/pet_recon.nii.gz')
EOF

# 2. Register PET to FreeSurfer T1
mri_coreg \
  --mov /data/pet/${SUBJECT}/pet_recon.nii.gz \
  --ref $SUBJECTS_DIR/${SUBJECT}/mri/brain.mgz \
  --reg /data/pet/${SUBJECT}/pet_to_fs.lta

# 3. Sample PET to cortical surface
mri_vol2surf \
  --mov /data/pet/${SUBJECT}/pet_recon.nii.gz \
  --reg /data/pet/${SUBJECT}/pet_to_fs.lta \
  --hemi lh \
  --projfrac 0.5 \
  --o /data/pet/${SUBJECT}/lh.pet.mgh

mri_vol2surf \
  --mov /data/pet/${SUBJECT}/pet_recon.nii.gz \
  --reg /data/pet/${SUBJECT}/pet_to_fs.lta \
  --hemi rh \
  --projfrac 0.5 \
  --o /data/pet/${SUBJECT}/rh.pet.mgh

# 4. Visualize
freeview -f $SUBJECTS_DIR/${SUBJECT}/surf/lh.pial:overlay=/data/pet/${SUBJECT}/lh.pet.mgh
```

---

## Batch Processing

### Batch PET Reconstruction

```python
import sirf.STIR as pet
from pathlib import Path
import os

# List of subjects
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
data_root = Path('/data/pet')
output_root = Path('/data/pet_recon')

# Reconstruction parameters
recon_params = {
    'num_subsets': 12,
    'num_subiterations': 24,
}

for subject in subjects:
    print(f"\n{'='*60}")
    print(f"Processing {subject}")
    print(f"{'='*60}")

    # Define paths
    sino_file = data_root / subject / 'sino.hs'
    attn_file = data_root / subject / 'attn_map.hv'
    output_dir = output_root / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check files exist
    if not sino_file.exists() or not attn_file.exists():
        print(f"Missing files for {subject}, skipping...")
        continue

    try:
        # Load data
        acq_data = pet.AcquisitionData(str(sino_file))
        attn_image = pet.ImageData(str(attn_file))

        # Set up acquisition model with attenuation
        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        asm_attn = pet.AcquisitionSensitivityModel(attn_image, acq_model)
        asm_attn.set_up(acq_data)
        acq_model.set_acquisition_sensitivity(asm_attn)

        # Initial image
        init_image = acq_data.create_uniform_image(value=1.0)

        # Reconstruct
        recon = pet.OSMAPOSLReconstructor()
        recon.set_num_subsets(recon_params['num_subsets'])
        recon.set_num_subiterations(recon_params['num_subiterations'])
        recon.set_acquisition_model(acq_model)
        recon.set_input(acq_data)
        recon.set_up(init_image)
        recon.process()

        # Get and save result
        recon_image = recon.get_output()
        output_file = output_dir / 'pet_osem.hv'
        recon_image.write(str(output_file))

        print(f"{subject}: Success")

    except Exception as e:
        print(f"{subject}: Failed - {e}")

print("\nBatch processing complete!")
```

---

## Troubleshooting

### Installation Issues

```bash
# Check SIRF installation
python -c "import sirf; print(sirf.__version__)"

# Check STIR is accessible
python -c "import sirf.STIR as pet; print('STIR OK')"

# Check Gadgetron is accessible
python -c "import sirf.Gadgetron as mr; print('Gadgetron OK')"

# If import errors, check paths
export PYTHONPATH=$HOME/devel/install/python:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/devel/install/lib:$LD_LIBRARY_PATH

# Verify STIR installation separately
stir_exe=$(which OSMAPOSL)
if [ -z "$stir_exe" ]; then
    echo "STIR not in PATH"
else
    echo "STIR found: $stir_exe"
fi
```

### Reconstruction Errors

```python
# Enable verbose output
import sirf.STIR as pet
pet.MessageRedirector('info.txt', 'warn.txt', 'error.txt')

# Check data compatibility
acq_data = pet.AcquisitionData('/data/pet/sino.hs')
print(f"Dimensions: {acq_data.dimensions()}")

# Verify attenuation map matches PET geometry
attn_image = pet.ImageData('/data/pet/attn_map.hv')
init_image = acq_data.create_uniform_image()
print(f"PET image dimensions: {init_image.dimensions()}")
print(f"Attn map dimensions: {attn_image.dimensions()}")
# Should match

# Test acquisition model setup
acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
try:
    acq_model.set_up(acq_data, init_image)
    print("Acquisition model setup successful")
except Exception as e:
    print(f"Acquisition model error: {e}")
```

---

## Best Practices

### Reconstruction Strategy

1. **Start Simple:**
   - Begin with basic OSEM
   - Add corrections incrementally
   - Validate at each step

2. **Attenuation Correction:**
   - Essential for quantitative PET
   - Verify μ-map quality
   - Check registration to PET

3. **Regularization:**
   - Use priors for low-count data
   - Adjust penalization factor carefully
   - Anatomical priors for PET-MR

### Multi-Modal Workflows

1. **Registration:**
   - Use high-quality registration (NiftyReg, ANTs)
   - Verify alignment visually
   - Check multiple anatomical landmarks

2. **Motion Correction:**
   - Extract motion from MR navigators
   - Apply to PET list-mode data
   - Validate motion estimates

---

## Resources and Further Reading

### Official Documentation

- **SIRF Website:** https://www.ccppetmr.ac.uk/sites/sirf
- **GitHub:** https://github.com/SyneRBI/SIRF
- **Documentation:** https://github.com/SyneRBI/SIRF/wiki
- **Tutorials:** https://github.com/SyneRBI/SIRF-Exercises

### Related Tools

- **STIR:** PET reconstruction engine
- **Gadgetron:** MR reconstruction framework
- **NiftyPET:** GPU-accelerated PET
- **AMIDE:** PET/SPECT viewer
- **NiftyReg:** Image registration

### Citations

If you use SIRF, please cite:

```
Ovtchinnikov, E., et al. (2020).
SIRF: Synergistic Image Reconstruction Framework.
Computer Physics Communications, 249, 107087.
```

---

## Summary

**SIRF** excels at:
- Synergistic PET-MR reconstruction
- Motion-corrected imaging
- Multi-modal research
- Educational framework
- Algorithm development

**Best for:**
- PET-MR simultaneous imaging
- Motion correction research
- Anatomically-guided reconstruction
- Teaching reconstruction concepts
- Method development

**Limitations:**
- Complex installation (use Docker)
- Steeper learning curve
- Smaller community than standalone STIR
- Primarily research-focused

For clinical PET viewing, use **AMIDE**. For GPU-accelerated reconstruction, use **NiftyPET**. For general PET reconstruction, use **STIR** directly. SIRF is ideal for PET-MR synergistic reconstruction and advanced multi-modal research.
