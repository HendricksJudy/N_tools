# NiftyPET - GPU-Accelerated PET Reconstruction

## Overview

**NiftyPET** is a Python package for high-throughput PET image reconstruction and analysis, leveraging NVIDIA CUDA for GPU acceleration. Developed at University College London (UCL), NiftyPET provides fast, accurate image reconstruction from PET raw data with comprehensive support for attenuation correction, scatter correction, motion correction, and advanced features like time-of-flight (TOF) and point spread function (PSF) modeling.

NiftyPET was specifically designed for the Siemens Biograph mMR (PET-MR) scanner but supports generic PET data formats. Its Python API enables seamless integration into neuroimaging pipelines, and its GPU acceleration provides dramatic speedups over CPU-based reconstruction, making it ideal for large studies and real-time applications.

**Key Features:**
- GPU-accelerated PET reconstruction using NVIDIA CUDA
- List-mode and sinogram reconstruction
- Iterative reconstruction algorithms (OSEM, MLEM)
- Attenuation correction from CT or MR images
- Scatter and randoms correction
- Motion correction
- Time-of-flight (TOF) reconstruction
- Point spread function (PSF) modeling
- Siemens Biograph mMR native support
- Python API for automation
- Integration with NiftyReg for registration
- DICOM and NIFTI I/O
- Multi-GPU support for parallel processing

**Primary Use Cases:**
- High-throughput PET reconstruction for large studies
- PET-MR imaging with MR-based attenuation correction
- Dynamic PET reconstruction with motion correction
- Method development and validation
- Real-time or near-real-time PET reconstruction
- GPU computing research for PET

**Official Documentation:** https://niftypet.readthedocs.io/

---

## Installation

### Prerequisites

**Hardware Requirements:**
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- Recommended: Modern NVIDIA GPU (GTX 1060+, RTX series, Tesla, Quadro)
- 8GB+ GPU RAM for typical datasets
- 16GB+ system RAM

**Software Requirements:**
- NVIDIA CUDA Toolkit 9.0 or later (CUDA 10.x, 11.x recommended)
- Python 3.7 or later
- Linux recommended (Ubuntu 18.04+, CentOS 7+)
- Windows and macOS supported with limitations

### Linux Installation (Ubuntu/Debian)

```bash
# Install NVIDIA drivers and CUDA
sudo apt-get update
sudo apt-get install nvidia-driver-495  # Or latest version
sudo apt-get install nvidia-cuda-toolkit

# Verify CUDA installation
nvidia-smi
nvcc --version

# Create conda environment
conda create -n niftypet python=3.9
conda activate niftypet

# Install dependencies
conda install numpy scipy nibabel h5py matplotlib
pip install pydicom

# Install NiftyPET
pip install niftypet

# Or install from source for latest version
git clone https://github.com/NiftyPET/NiftyPET.git
cd NiftyPET
pip install -e .
```

### Install NiftyPET Components

```bash
# NiftyPET consists of multiple modules:
# 1. NIMPA - Image processing and analysis
# 2. NIPET - PET reconstruction engine

# Install NIMPA
pip install nimpa

# Install NIPET (PET reconstruction)
pip install nipet

# Verify installation
python -c "import niftypet; print(niftypet.__version__)"
python -c "import nimpa; print(nimpa.__version__)"
python -c "import nipet; print(nipet.__version__)"
```

### Configure GPU

```python
# Test GPU availability
import nipet
from nipet import mmraux

# Check available CUDA devices
print("CUDA devices:", nipet.get_cuda_devices())

# Test reconstruction capability
# This will compile CUDA kernels on first run
```

### Install NiftyReg (Optional, for Registration)

```bash
# NiftyReg for image registration
conda install -c conda-forge niftyreg

# Or build from source
git clone https://github.com/KCL-BMEIS/niftyreg.git
cd niftyreg
mkdir build && cd build
cmake ..
make
sudo make install
```

---

## Basic PET Reconstruction

### Load PET Raw Data

```python
import os
from pathlib import Path
import numpy as np
from nipet import mmraux, mmrchain
import nimpa

# Define paths
pet_data_dir = '/data/pet/sub-01'
output_dir = '/data/pet/sub-01/reconstruction'
os.makedirs(output_dir, exist_ok=True)

# NiftyPET uses a "datain" dictionary for input data
# For Siemens mMR data:
datain = {}
datain['lm'] = os.path.join(pet_data_dir, 'listmode.bf')  # List-mode data
datain['norm'] = os.path.join(pet_data_dir, 'norm.n.hdr')  # Normalization
datain['mu_h'] = os.path.join(pet_data_dir, 'mumap.v.hdr')  # Attenuation map

# Get scanner constants (mMR-specific)
mMRpars = mmraux.get_mmrparams()

# Display scanner parameters
print(f"Scanner: {mMRpars['ScannerName']}")
print(f"Radial bins: {mMRpars['Naw']}")
print(f"Angles: {mMRpars['Nang']}")
print(f"Sinogram dimensions: {mMRpars['nsinos']}")
```

### Basic OSEM Reconstruction

```python
from nipet import mmrchain

# Define reconstruction parameters
recon_params = {
    'iterations': 4,       # OSEM iterations
    'subsets': 14,         # OSEM subsets
    'attenuation': True,   # Enable attenuation correction
    'scatter': True,       # Enable scatter correction
    'randoms': True,       # Enable randoms correction
    'gpu_id': 0            # Use first GPU
}

# Perform OSEM reconstruction
recimg = mmrchain.osem(
    datain,
    mMRpars,
    outpath=output_dir,
    fcomment='_osem_4i14s',
    itr=recon_params['iterations'],
    subs=recon_params['subsets'],
    fwhm=0.0,              # No post-smoothing
    recmod=3,              # Reconstruction mode: 3 = full corrections
    store_img=True         # Save intermediate iterations
)

# recimg is a 3D numpy array (voxel values in Bq/mL)
print(f"Reconstructed image shape: {recimg.shape}")
print(f"Image range: {recimg.min():.2f} - {recimg.max():.2f} Bq/mL")
```

### Save Reconstructed Image

```python
import nibabel as nib

# Get image orientation info from scanner parameters
affine = np.diag([2.0, 2.0, 2.03125, 1.0])  # mMR voxel sizes
affine[:3, 3] = [-127.5, -127.5, -78.8]      # Origin

# Create NIfTI image
nii_img = nib.Nifti1Image(recimg, affine)

# Save as NIfTI
output_file = os.path.join(output_dir, 'pet_osem.nii.gz')
nib.save(nii_img, output_file)
print(f"Saved: {output_file}")
```

---

## Attenuation Correction

### MR-Based Attenuation Correction (PET-MR)

```python
from nipet import mmrchain
from nimpa import prc

# Load MR-based μ-map (created from Dixon MR sequences)
mu_file = '/data/pet/sub-01/mumap_mr.v'

# Add to datain dictionary
datain['mu_h'] = mu_file

# Reconstruct with MR-based attenuation
recimg_mumap = mmrchain.osem(
    datain,
    mMRpars,
    outpath=output_dir,
    fcomment='_mumap_mr',
    itr=4,
    subs=14,
    recmod=3  # Full corrections including attenuation
)

print("Reconstruction with MR-based attenuation complete")
```

### CT-Based Attenuation Correction

```python
from nimpa import prc

# Convert CT to μ-map at 511 keV
ct_file = '/data/pet/sub-01/ct.nii.gz'
ct_img = nib.load(ct_file)
ct_data = ct_img.get_fdata()

# CT to μ-map conversion (simplified bilinear)
# HU to linear attenuation coefficient at 511 keV
def ct_to_mumap(ct_hu):
    """
    Convert CT Hounsfield Units to μ-map
    Bilinear scaling for PET energy (511 keV)
    """
    mu = np.zeros_like(ct_hu, dtype=np.float32)

    # Air and soft tissue
    mask_soft = ct_hu < 0
    mu[mask_soft] = (ct_hu[mask_soft] + 1000) * 9.6e-5 / 1000

    # Bone
    mask_bone = ct_hu >= 0
    mu[mask_bone] = (ct_hu[mask_bone] * 5.464e-5 / 1000) + 9.6e-5

    return mu

mu_data = ct_to_mumap(ct_data)

# Save μ-map
mu_img = nib.Nifti1Image(mu_data, ct_img.affine, ct_img.header)
nib.save(mu_img, '/data/pet/sub-01/mumap_ct.nii.gz')

# Convert to NiftyPET format and use in reconstruction
# (Requires resampling to PET space and format conversion)
```

### Generate μ-map from Segmentation

```python
# Create synthetic μ-map from tissue segmentation
import numpy as np

# Load tissue segmentation (e.g., from FreeSurfer)
seg_img = nib.load('/data/pet/sub-01/aparc+aseg.nii.gz')
seg_data = seg_img.get_fdata()

# Assign linear attenuation coefficients (cm⁻¹ at 511 keV)
mu_values = {
    'air': 0.0,
    'soft_tissue': 0.096,
    'bone': 0.151,
    'water': 0.096
}

# Create μ-map
mu_data = np.zeros_like(seg_data, dtype=np.float32)

# Background (air)
mu_data[seg_data == 0] = mu_values['air']

# Soft tissue (brain, most labels)
mu_data[seg_data > 0] = mu_values['soft_tissue']

# Bone (skull)
skull_labels = [258]  # Example: skull label
for label in skull_labels:
    mu_data[seg_data == label] = mu_values['bone']

# Save
mu_img = nib.Nifti1Image(mu_data, seg_img.affine, seg_img.header)
nib.save(mu_img, '/data/pet/sub-01/mumap_seg.nii.gz')
```

---

## Scatter and Randoms Correction

### Enable Scatter Correction

```python
from nipet import mmrchain

# Reconstruction with scatter correction
# recmod parameter controls corrections:
# recmod = 0: No corrections
# recmod = 1: Attenuation only
# recmod = 2: Attenuation + randoms
# recmod = 3: Attenuation + randoms + scatter (full corrections)

recimg_scatter = mmrchain.osem(
    datain,
    mMRpars,
    outpath=output_dir,
    fcomment='_full_corrections',
    itr=4,
    subs=14,
    recmod=3,      # Full corrections
    store_img=True
)

# Scatter is estimated using single scatter simulation (SSS)
# Automatically computed during reconstruction
```

### Compare Reconstruction with/without Corrections

```python
import matplotlib.pyplot as plt

# Reconstruct without corrections
recimg_nac = mmrchain.osem(
    datain, mMRpars,
    itr=4, subs=14,
    recmod=0,  # No corrections
    fcomment='_nac'
)

# Reconstruct with full corrections
recimg_ac = mmrchain.osem(
    datain, mMRpars,
    itr=4, subs=14,
    recmod=3,  # Full corrections
    fcomment='_ac'
)

# Compare central slices
central_slice = recimg_nac.shape[2] // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(recimg_nac[:, :, central_slice], cmap='hot')
axes[0].set_title('No Corrections')
axes[0].axis('off')

axes[1].imshow(recimg_ac[:, :, central_slice], cmap='hot')
axes[1].set_title('Full Corrections (AC + SC + RC)')
axes[1].axis('off')

plt.savefig('correction_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison saved")
```

---

## Motion Correction

### Frame-by-Frame Motion Correction

```python
from nipet import mmrchain
from nimpa import prc
import numpy as np

# Dynamic PET with potential motion
# Reconstruct individual frames
n_frames = 6
frame_duration = 300  # seconds

reconstructed_frames = []

for frame_idx in range(n_frames):
    print(f"Reconstructing frame {frame_idx + 1}/{n_frames}")

    # Define time window for this frame
    t_start = frame_idx * frame_duration
    t_end = (frame_idx + 1) * frame_duration

    # Extract frame data (implementation depends on data format)
    # datain_frame = extract_frame(datain, t_start, t_end)

    # Reconstruct frame
    recimg_frame = mmrchain.osem(
        datain,  # Would be datain_frame with proper frame extraction
        mMRpars,
        itr=4,
        subs=14,
        recmod=3,
        fcomment=f'_frame{frame_idx:02d}'
    )

    reconstructed_frames.append(recimg_frame)

# Stack frames into 4D array
pet_4d = np.stack(reconstructed_frames, axis=-1)
print(f"Dynamic PET shape: {pet_4d.shape}")
```

### Register and Align Frames

```python
from nipet import nimpa
from scipy.ndimage import affine_transform

# Use first frame as reference
reference_frame = reconstructed_frames[0]
aligned_frames = [reference_frame]

for i in range(1, len(reconstructed_frames)):
    moving_frame = reconstructed_frames[i]

    # Register moving to reference using NiftyReg
    # (Simplified - actual implementation uses nimpa.prc.align)

    # Placeholder for registration
    # In practice, use nimpa.prc or external tools
    aligned_frame = moving_frame  # Would be registered version

    aligned_frames.append(aligned_frame)

# Average motion-corrected frames
mean_image = np.mean(aligned_frames, axis=0)

# Save
mean_nii = nib.Nifti1Image(mean_image, affine)
nib.save(mean_nii, os.path.join(output_dir, 'pet_motion_corrected.nii.gz'))
```

---

## Time-of-Flight (TOF) Reconstruction

### Enable TOF Reconstruction

```python
from nipet import mmrchain

# TOF reconstruction (if scanner supports it)
# Siemens mMR does not have TOF, but newer scanners do

# For TOF-capable scanners:
recimg_tof = mmrchain.osem(
    datain,
    mMRpars,
    itr=4,
    subs=14,
    recmod=3,
    fwhm=0.0,
    # TOF parameters (scanner-specific)
    # tof=True,  # Enable TOF
    # tof_bins=13,  # TOF bins
    # tof_width=500  # TOF resolution (ps)
)

# TOF improves SNR and convergence speed
# Particularly beneficial for large patients
```

---

## Point Spread Function (PSF) Modeling

### Reconstruct with PSF Correction

```python
from nipet import mmrchain

# PSF modeling improves resolution recovery
# Accounts for detector blurring

recimg_psf = mmrchain.osem(
    datain,
    mMRpars,
    itr=4,
    subs=14,
    recmod=3,
    fwhm=0.0,
    # PSF modeling (if supported by your NiftyPET version)
    # psf=True,
    # psf_fwhm=4.5  # Scanner PSF in mm
)

# PSF reconstruction produces sharper images
# Important for small structures
```

### Compare Standard vs. PSF Reconstruction

```python
import matplotlib.pyplot as plt

# Standard reconstruction
recimg_standard = mmrchain.osem(datain, mMRpars, itr=4, subs=14, recmod=3)

# PSF reconstruction (if available)
# recimg_psf = mmrchain.osem(datain, mMRpars, itr=4, subs=14, recmod=3, psf=True)

# For demonstration, use standard image
recimg_psf = recimg_standard  # Placeholder

# Compare
slice_idx = recimg_standard.shape[2] // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(recimg_standard[:, :, slice_idx], cmap='hot', vmin=0, vmax=np.percentile(recimg_standard, 99))
axes[0].set_title('Standard OSEM')
axes[0].axis('off')

axes[1].imshow(recimg_psf[:, :, slice_idx], cmap='hot', vmin=0, vmax=np.percentile(recimg_psf, 99))
axes[1].set_title('OSEM with PSF')
axes[1].axis('off')

plt.savefig('psf_comparison.png', dpi=300, bbox_inches='tight')
```

---

## GPU Optimization

### Multi-GPU Reconstruction

```python
import nipet
from nipet import mmrchain

# Check available GPUs
cuda_devices = nipet.get_cuda_devices()
print(f"Available CUDA devices: {cuda_devices}")

# Use specific GPU
gpu_id = 0
recimg = mmrchain.osem(
    datain,
    mMRpars,
    itr=4,
    subs=14,
    recmod=3,
    gpu_id=gpu_id
)

# For parallel processing of multiple subjects:
# Launch separate Python processes, each using different GPU
```

### Benchmark Reconstruction Speed

```python
import time

# Benchmark reconstruction time
iterations_list = [2, 4, 6, 8]
timing_results = {}

for itr in iterations_list:
    start_time = time.time()

    recimg = mmrchain.osem(
        datain,
        mMRpars,
        itr=itr,
        subs=14,
        recmod=3,
        gpu_id=0
    )

    elapsed = time.time() - start_time
    timing_results[itr] = elapsed
    print(f"{itr} iterations: {elapsed:.1f} seconds")

# Typical GPU speedup: 10-50x vs CPU
```

### Optimize Memory Usage

```python
# For large datasets, control memory usage

# Process frames sequentially instead of simultaneously
frames = range(10)

for frame_idx in frames:
    print(f"Frame {frame_idx}...")

    # Reconstruct frame
    recimg_frame = mmrchain.osem(
        datain,
        mMRpars,
        itr=4,
        subs=14,
        recmod=3
    )

    # Save immediately
    nii = nib.Nifti1Image(recimg_frame, affine)
    nib.save(nii, f'frame_{frame_idx:03d}.nii.gz')

    # Free memory
    del recimg_frame

print("All frames processed")
```

---

## Batch Processing and Automation

### Batch Reconstruct Multiple Subjects

```python
import os
from pathlib import Path
from nipet import mmrchain, mmraux
import nibabel as nib

# List of subjects
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
data_root = '/data/pet'
output_root = '/data/pet_reconstructed'

# Scanner parameters
mMRpars = mmraux.get_mmrparams()

# Reconstruction parameters
recon_config = {
    'itr': 4,
    'subs': 14,
    'recmod': 3,
    'fwhm': 0.0,
    'gpu_id': 0
}

for subject in subjects:
    print(f"\n{'='*60}")
    print(f"Processing {subject}")
    print(f"{'='*60}")

    # Define input paths
    subj_dir = os.path.join(data_root, subject)
    output_dir = os.path.join(output_root, subject)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare datain
    datain = {
        'lm': os.path.join(subj_dir, 'listmode.bf'),
        'norm': os.path.join(subj_dir, 'norm.n.hdr'),
        'mu_h': os.path.join(subj_dir, 'mumap.v.hdr')
    }

    # Check files exist
    if not all(os.path.exists(f) for f in datain.values()):
        print(f"Missing files for {subject}, skipping...")
        continue

    # Reconstruct
    try:
        recimg = mmrchain.osem(
            datain,
            mMRpars,
            outpath=output_dir,
            fcomment='_osem',
            **recon_config
        )

        # Save as NIfTI
        affine = np.diag([2.0, 2.0, 2.03125, 1.0])
        nii = nib.Nifti1Image(recimg, affine)
        nib.save(nii, os.path.join(output_dir, 'pet_osem.nii.gz'))

        print(f"{subject}: Success")

    except Exception as e:
        print(f"{subject}: Failed - {e}")

print("\nBatch processing complete!")
```

### Parallel Processing with Multiple GPUs

```python
from multiprocessing import Process
import os

def reconstruct_subject(subject, gpu_id):
    """Reconstruct single subject on specific GPU"""
    from nipet import mmrchain, mmraux
    import nibabel as nib
    import numpy as np

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"[GPU {gpu_id}] Processing {subject}")

    # Setup paths
    datain = {
        'lm': f'/data/pet/{subject}/listmode.bf',
        'norm': f'/data/pet/{subject}/norm.n.hdr',
        'mu_h': f'/data/pet/{subject}/mumap.v.hdr'
    }

    mMRpars = mmraux.get_mmrparams()

    # Reconstruct
    recimg = mmrchain.osem(datain, mMRpars, itr=4, subs=14, recmod=3)

    # Save
    affine = np.diag([2.0, 2.0, 2.03125, 1.0])
    nii = nib.Nifti1Image(recimg, affine)
    nib.save(nii, f'/data/pet_recon/{subject}/pet.nii.gz')

    print(f"[GPU {gpu_id}] {subject} complete")

# Distribute subjects across GPUs
subjects = [f'sub-{i:02d}' for i in range(1, 9)]
n_gpus = 2

processes = []
for i, subject in enumerate(subjects):
    gpu_id = i % n_gpus
    p = Process(target=reconstruct_subject, args=(subject, gpu_id))
    p.start()
    processes.append(p)

    # Limit concurrent processes
    if len(processes) >= n_gpus:
        for proc in processes:
            proc.join()
        processes = []

# Wait for remaining
for proc in processes:
    proc.join()

print("Parallel batch processing complete!")
```

---

## Integration with Neuroimaging Pipelines

### NiftyPET to FreeSurfer Workflow

```bash
#!/bin/bash
# Integrate NiftyPET reconstruction with FreeSurfer analysis

SUBJECT=sub-01
SUBJECTS_DIR=/data/freesurfer

# 1. Reconstruct PET with NiftyPET (Python)
python << EOF
from nipet import mmrchain, mmraux
import nibabel as nib
import numpy as np

datain = {
    'lm': '/data/pet/${SUBJECT}/listmode.bf',
    'norm': '/data/pet/${SUBJECT}/norm.n.hdr',
    'mu_h': '/data/pet/${SUBJECT}/mumap.v.hdr'
}

mMRpars = mmraux.get_mmrparams()
recimg = mmrchain.osem(datain, mMRpars, itr=4, subs=14, recmod=3)

affine = np.diag([2.0, 2.0, 2.03125, 1.0])
nii = nib.Nifti1Image(recimg, affine)
nib.save(nii, '/data/pet/${SUBJECT}/pet_recon.nii.gz')
EOF

# 2. Register PET to FreeSurfer T1
mri_coreg \
  --mov /data/pet/${SUBJECT}/pet_recon.nii.gz \
  --ref $SUBJECTS_DIR/${SUBJECT}/mri/brain.mgz \
  --reg /data/pet/${SUBJECT}/pet_to_t1.lta

# 3. Apply transformation
mri_vol2vol \
  --mov /data/pet/${SUBJECT}/pet_recon.nii.gz \
  --targ $SUBJECTS_DIR/${SUBJECT}/mri/brain.mgz \
  --lta /data/pet/${SUBJECT}/pet_to_t1.lta \
  --o /data/pet/${SUBJECT}/pet_in_t1space.nii.gz

# 4. Sample PET to cortical surface
mri_vol2surf \
  --mov /data/pet/${SUBJECT}/pet_in_t1space.nii.gz \
  --reg /data/pet/${SUBJECT}/pet_to_t1.lta \
  --hemi lh \
  --projfrac 0.5 \
  --o /data/pet/${SUBJECT}/lh.pet.mgh

# 5. Visualize on surface
freeview -f $SUBJECTS_DIR/${SUBJECT}/surf/lh.pial:overlay=/data/pet/${SUBJECT}/lh.pet.mgh
```

### Integration with BIDS and QSIPrep

```python
# Convert NiftyPET output to BIDS format
import os
import json
import shutil
from pathlib import Path

def niftypet_to_bids(recon_file, bids_root, subject, session=None):
    """Convert NiftyPET reconstruction to BIDS format"""

    # Create BIDS structure
    if session:
        pet_dir = Path(bids_root) / f'sub-{subject}' / f'ses-{session}' / 'pet'
    else:
        pet_dir = Path(bids_root) / f'sub-{subject}' / 'pet'

    pet_dir.mkdir(parents=True, exist_ok=True)

    # Copy reconstructed PET
    if session:
        bids_file = pet_dir / f'sub-{subject}_ses-{session}_pet.nii.gz'
    else:
        bids_file = pet_dir / f'sub-{subject}_pet.nii.gz'

    shutil.copy(recon_file, bids_file)

    # Create JSON sidecar
    metadata = {
        "Modality": "PT",
        "Manufacturer": "Siemens",
        "ManufacturersModelName": "Biograph mMR",
        "ReconstructionMethod": "OSEM",
        "ReconstructionSoftware": "NiftyPET",
        "NumberOfIterations": 4,
        "NumberOfSubsets": 14,
        "AttenuationCorrection": "MR-based",
        "ScatterCorrection": "single scatter simulation",
        "RandomsCorrection": "delayed window"
    }

    json_file = bids_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"BIDS PET saved: {bids_file}")

# Usage
niftypet_to_bids(
    recon_file='/data/pet/sub-01/pet_osem.nii.gz',
    bids_root='/data/bids_dataset',
    subject='01',
    session='baseline'
)
```

---

## Advanced Applications

### Dynamic PET Reconstruction

```python
from nipet import mmrchain, mmraux
import numpy as np
import nibabel as nib

# Dynamic PET frame definition
frame_times = [
    (0, 30), (30, 60), (60, 120), (120, 300),
    (300, 600), (600, 900), (900, 1800), (1800, 3600)
]  # (start, end) in seconds

mMRpars = mmraux.get_mmrparams()

dynamic_images = []

for frame_idx, (t_start, t_end) in enumerate(frame_times):
    print(f"Frame {frame_idx + 1}: {t_start}-{t_end} sec")

    # Prepare datain for this frame
    # (Frame extraction depends on data format)
    datain_frame = {
        'lm': '/data/pet/listmode.bf',
        'norm': '/data/pet/norm.n.hdr',
        'mu_h': '/data/pet/mumap.v.hdr',
        'tstart': t_start,
        'tend': t_end
    }

    # Reconstruct frame
    recimg = mmrchain.osem(
        datain_frame,
        mMRpars,
        itr=4,
        subs=14,
        recmod=3
    )

    dynamic_images.append(recimg)

# Stack into 4D volume
pet_4d = np.stack(dynamic_images, axis=-1)

# Save as 4D NIfTI
affine = np.diag([2.0, 2.0, 2.03125, 1.0])
nii_4d = nib.Nifti1Image(pet_4d, affine)
nib.save(nii_4d, '/data/pet/pet_dynamic_4d.nii.gz')

print(f"Dynamic PET shape: {pet_4d.shape}")
```

### Parametric Imaging

```python
# Create parametric images from dynamic PET
import numpy as np
from scipy.optimize import curve_fit

# Load dynamic PET
pet_4d_img = nib.load('/data/pet/pet_dynamic_4d.nii.gz')
pet_4d = pet_4d_img.get_fdata()

# Frame mid-times
frame_times = np.array([15, 45, 90, 210, 450, 750, 1350, 2700]) / 60  # minutes

# Simple Logan plot for each voxel
def logan_model(t, dv, intercept):
    return dv * t + intercept

# Initialize parametric image (distribution volume)
dv_map = np.zeros(pet_4d.shape[:3])

# Process each voxel (simplified - should use vectorization)
print("Computing parametric map...")
for i in range(0, pet_4d.shape[0], 10):  # Subsample for speed
    for j in range(0, pet_4d.shape[1], 10):
        for k in range(pet_4d.shape[2]):
            tac = pet_4d[i, j, k, :]

            if tac.max() > 0:
                try:
                    popt, _ = curve_fit(logan_model, frame_times, tac, p0=[1.0, 0.0])
                    dv_map[i, j, k] = popt[0]
                except:
                    pass

# Save parametric map
dv_nii = nib.Nifti1Image(dv_map, pet_4d_img.affine)
nib.save(dv_nii, '/data/pet/dv_map.nii.gz')
print("Parametric map created")
```

---

## Troubleshooting

### CUDA Errors

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify GPU compatibility
python -c "import torch; print(torch.cuda.is_available())"

# Test NiftyPET CUDA
python -c "from nipet import mmraux; print(mmraux.get_cuda_devices())"

# If CUDA not found:
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Reinstall NiftyPET
pip uninstall nipet nimpa
pip install --no-cache-dir nipet nimpa
```

### Out of Memory Errors

```python
# Reduce memory usage

# 1. Process frames sequentially (not in parallel)
# 2. Use smaller subset numbers
recimg = mmrchain.osem(datain, mMRpars, itr=4, subs=7, recmod=3)  # 7 instead of 14

# 3. Clear GPU cache
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 4. Use smaller GPU batch sizes
# (Internal NiftyPET parameter, check documentation)
```

### Reconstruction Artifacts

```python
# Check for common issues

# 1. Verify attenuation map quality
mu_img = nib.load('mumap.nii.gz')
mu_data = mu_img.get_fdata()
print(f"μ-map range: {mu_data.min():.4f} - {mu_data.max():.4f}")
# Should be ~0 for air, ~0.096 for soft tissue, ~0.15 for bone

# 2. Check for motion
# Visual inspection of dynamic frames

# 3. Increase iterations if image noisy
recimg = mmrchain.osem(datain, mMRpars, itr=8, subs=14, recmod=3)

# 4. Apply post-smoothing
recimg = mmrchain.osem(datain, mMRpars, itr=4, subs=14, recmod=3, fwhm=4.0)
# fwhm in mm
```

---

## Best Practices

### Reconstruction Parameters

1. **Iterations and Subsets:**
   - Start with 4 iterations, 14 subsets (standard)
   - Brain PET: 4-6 iterations sufficient
   - Whole-body: May need more iterations
   - Higher iterations → better convergence but more noise

2. **Corrections:**
   - Always use full corrections (recmod=3)
   - Attenuation correction is critical
   - Scatter and randoms improve quantification

3. **Post-Smoothing:**
   - fwhm=0: No smoothing (preserve resolution)
   - fwhm=4-6 mm: Typical for brain PET
   - Match smoothing to analysis needs

### GPU Utilization

1. **Batch Processing:**
   - Use multiple GPUs for parallel subjects
   - Monitor GPU memory with nvidia-smi
   - Process frames sequentially if memory limited

2. **Performance:**
   - GPU reconstruction is 10-50x faster than CPU
   - Keep GPU busy with continuous data flow
   - Minimize data transfers to/from GPU

---

## Resources and Further Reading

### Official Documentation

- **NiftyPET Docs:** https://niftypet.readthedocs.io/
- **GitHub:** https://github.com/NiftyPET/NiftyPET
- **NIMPA Docs:** https://github.com/NiftyPET/NIMPA
- **NIPET Docs:** https://github.com/NiftyPET/NIPET

### Related Tools

- **SIRF:** Synergistic reconstruction framework using STIR
- **STIR:** Tomographic image reconstruction library
- **AMIDE:** PET/SPECT viewer and analysis
- **FreeSurfer:** Anatomical segmentation for PET ROIs
- **NiftyReg:** Image registration

### Citations

If you use NiftyPET, please cite:

```
Markiewicz, P. J., et al. (2018).
NiftyPET: a High-throughput Software Platform for High Quantitative Accuracy
and Precision PET Imaging and Analysis.
Neuroinformatics, 16(1), 95-115.
```

---

## Summary

**NiftyPET** excels at:
- GPU-accelerated PET reconstruction
- High-throughput studies
- PET-MR imaging
- Dynamic PET
- Python-based automation

**Best for:**
- Large PET studies requiring fast reconstruction
- PET-MR research
- Dynamic PET and kinetic modeling
- GPU computing applications
- Integrated Python pipelines

**Limitations:**
- Requires NVIDIA GPU with CUDA
- Primarily designed for Siemens mMR
- Smaller community than STIR
- Linux preferred (Windows/macOS limited)

For clinical viewing, use **AMIDE**. For multi-modal PET-MR reconstruction, consider **SIRF**. For general-purpose PET reconstruction, use **STIR**. NiftyPET is ideal when GPU acceleration and Python integration are priorities.
