# STIR - Software for Tomographic Image Reconstruction

## Overview

**STIR** (Software for Tomographic Image Reconstruction) is a comprehensive, open-source C++ library for tomographic image reconstruction, with particular focus on PET (Positron Emission Tomography) and SPECT (Single Photon Emission Computed Tomography). Developed over more than two decades, STIR provides a wide range of reconstruction algorithms, correction methods, and utilities for sinogram processing, making it a foundational tool for PET reconstruction research and clinical applications.

STIR's flexibility and extensibility make it ideal for method development, supporting multiple scanner geometries, various reconstruction algorithms (FBP, OSEM, OSSPS), and comprehensive correction capabilities including attenuation, scatter, and randoms correction. Its modular architecture and parameter file-based configuration enable reproducible research and easy algorithm prototyping.

**Key Features:**
- Multiple reconstruction algorithms: FBP, MLEM, OSEM, OSSPS, and more
- Comprehensive corrections: attenuation, scatter, randoms, normalization
- Support for multiple scanner geometries (cylindrical PET, block detectors, SPECT)
- List-mode and sinogram reconstruction
- Forward and back-projection operators
- PSF (Point Spread Function) modeling for resolution recovery
- Parameterized reconstruction via parfiles
- Python bindings for scripting
- GPU acceleration support (experimental)
- Monte Carlo scatter simulation
- Time-of-flight (TOF) data support
- Utilities for data conversion and processing
- Extensible C++ framework for algorithm development

**Primary Use Cases:**
- PET and SPECT image reconstruction from raw data
- Reconstruction algorithm development and validation
- Scanner calibration and quality control
- Method comparison and benchmarking
- Research in PET physics and quantification
- Teaching tomographic reconstruction concepts
- Clinical PET reconstruction pipelines

**Official Documentation:** http://stir.sourceforge.net/

---

## Installation

### Prerequisites

**System Requirements:**
- Linux (Ubuntu 18.04+, CentOS 7+), macOS, or Windows
- C++ compiler (GCC 7+, Clang 9+, MSVC 2017+)
- CMake 3.10+
- 8GB+ RAM recommended
- Optional: CUDA for GPU acceleration
- Optional: Python 3.6+ for Python bindings

**Dependencies:**
- Boost libraries (1.58+)
- Optional: ITK (for additional I/O formats)
- Optional: ROOT or CERN ROOT (for list-mode)
- Optional: HDF5 (for Interfile variant)

### Installation from Source (Linux)

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y git cmake g++ libboost-all-dev \
  libinsighttoolkit4-dev libhdf5-dev

# Clone STIR repository
cd $HOME
git clone https://github.com/UCL/STIR.git
cd STIR

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/stir-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DSTIR_ENABLE_EXPERIMENTAL=ON \
  -DBUILD_TESTING=ON

# Build (use all cores)
make -j$(nproc)

# Run tests (optional)
ctest

# Install
make install

# Set environment variables
export PATH=$HOME/stir-install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/stir-install/lib:$LD_LIBRARY_PATH
```

### Install Python Bindings

```bash
# Within STIR build directory
cmake .. -DBUILD_SWIG_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)
make install

# Set Python path
export PYTHONPATH=$HOME/stir-install/python:$PYTHONPATH

# Verify Python bindings
python3 -c "import stir; print(stir.__version__)"
```

### Installation via Docker

```bash
# Pull STIR Docker image
docker pull ghcr.io/ucl/stir:latest

# Run STIR container
docker run -it --rm \
  -v /path/to/data:/data \
  ghcr.io/ucl/stir:latest

# Verify installation
OSMAPOSL --version
```

### Verify Installation

```bash
# Check STIR executables
which OSMAPOSL
which FBP2D
which list_projdata_info

# List available executables
ls $HOME/stir-install/bin/

# Test basic functionality
echo "STIR installation verified"
```

---

## Basic Reconstruction with Parameter Files

### Understanding STIR Parameter Files (Parfiles)

STIR uses text-based parameter files (.par) to configure reconstruction:

```text
; example_osem.par - Basic OSEM reconstruction
OSMAPOSLParameters :=

  ; Input data
  input file := sino.hs

  ; Output file prefix
  output filename prefix := recon_osem

  ; Reconstruction parameters
  number of subsets := 12
  number of subiterations := 24
  save estimates at subiteration intervals := 24

  ; Initial estimate
  initial estimate := init.hv

  ; Acquisition model
  projector pair type := Matrix
    Projector Pair Using Matrix Parameters :=
      Matrix type := Ray Tracing
      Ray tracing matrix parameters :=
        number of rays in tangential direction to det:= 10
      End Ray tracing matrix parameters :=
    End Projector Pair Using Matrix Parameters :=

  ; Subset scheme
  subset scheme := random

End :=
```

### Run OSEM Reconstruction

```bash
# Create parameter file (osem.par as shown above)

# Run reconstruction
OSMAPOSL osem.par

# This produces: recon_osem_24.hv (final iteration)

# Examine output
list_image_info recon_osem_24.hv

# Convert to analyze format (if needed)
conv_to_ecat7 recon_osem_24.hv recon_osem_24.img
```

### FBP Reconstruction

```bash
# Create FBP parameter file
cat > fbp.par << 'EOF'
FBP2DParameters :=
  input file := sino.hs
  output filename prefix := recon_fbp

  ; Ramp filter
  filter type := Ramp
  Alpha parameter := 0.5
  Cut-off frequency := 0.5

  ; Zoom factor
  zoom := 1.0

  ; Radial positions
  xy output image size (in pixels) := 128

End :=
EOF

# Run FBP reconstruction
FBP2D fbp.par

# Output: recon_fbp.hv
```

---

## Attenuation Correction

### Create Attenuation Sinogram

```bash
# Forward project attenuation map to create attenuation factors
cat > create_acf.par << 'EOF'
ForwardProjectorParameters :=
  type := Matrix
  Matrix type := Ray Tracing

  ; Input attenuation map (μ-map)
  input file := mumap.hv

  ; Output attenuation correction factors
  output filename prefix := acf

  ; Template sinogram (defines geometry)
  template := sino.hs

End :=
EOF

# Generate ACF sinogram
forward_project create_acf.par

# This creates: acf.hs (attenuation factors in sinogram space)
```

### OSEM with Attenuation Correction

```bash
# Parameter file with attenuation correction
cat > osem_ac.par << 'EOF'
OSMAPOSLParameters :=
  input file := sino.hs
  output filename prefix := recon_osem_ac

  number of subsets := 12
  number of subiterations := 24
  save estimates at subiteration intervals := 24

  initial estimate := init.hv

  ; Acquisition model with attenuation
  projector pair type := Matrix
    Projector Pair Using Matrix Parameters :=
      Matrix type := Ray Tracing
    End Projector Pair Using Matrix Parameters :=

  ; Attenuation correction
  attenuation image filename := mumap.hv

  subset scheme := random

End :=
EOF

# Run reconstruction with AC
OSMAPOSL osem_ac.par
```

---

## Scatter Correction

### Estimate Scatter Using Single Scatter Simulation

```bash
# Parameter file for scatter estimation
cat > scatter_est.par << 'EOF'
ScatterEstimationParameters :=
  ; Input emission sinogram
  input file := sino.hs

  ; Attenuation map
  attenuation image := mumap.hv

  ; Initial activity estimate (can be FBP reconstruction)
  activity image := recon_fbp.hv

  ; Output scatter estimate
  output filename prefix := scatter

  ; Number of scatter iterations
  number of scatter iterations := 3

  ; Scatter simulation parameters
  scatter simulation type := Single Scatter Simulation
    Single Scatter Simulation Parameters :=
      ; Detector efficiency
      detector efficiency := 0.9

      ; Scatter energy window
      energy window lower level := 400
      energy window upper level := 650

    End Single Scatter Simulation Parameters :=

End :=
EOF

# Estimate scatter
estimate_scatter scatter_est.par

# Output: scatter.hs (scatter sinogram estimate)
```

### OSEM with Scatter and Attenuation Correction

```bash
# Full corrections: attenuation + scatter + randoms
cat > osem_full.par << 'EOF'
OSMAPOSLParameters :=
  input file := sino.hs
  output filename prefix := recon_osem_full

  number of subsets := 12
  number of subiterations := 24
  save estimates at subiteration intervals := 24

  initial estimate := init.hv

  projector pair type := Matrix
    Projector Pair Using Matrix Parameters :=
      Matrix type := Ray Tracing
    End Projector Pair Using Matrix Parameters :=

  ; Attenuation correction
  attenuation image filename := mumap.hv

  ; Scatter correction
  scatter simulation type := Single Scatter Simulation
    Single Scatter Simulation Parameters :=
      scatter sinogram := scatter.hs
    End Single Scatter Simulation Parameters :=

  ; Randoms correction
  randoms := randoms.hs

  subset scheme := random

End :=
EOF

# Reconstruct with full corrections
OSMAPOSL osem_full.par
```

---

## List-Mode Reconstruction

### Convert List-Mode to Sinogram

```bash
# List-mode to sinogram conversion
lm_to_projdata \
  --input listmode.l.hdr \
  --output sino \
  --template template_sino.hs \
  --time-interval 0 300  # First 5 minutes

# This creates sino.hs from list-mode data
```

### Direct List-Mode Reconstruction

```bash
# OSEM from list-mode data directly
cat > osem_listmode.par << 'EOF'
OSMAPOSLParameters :=
  ; List-mode input
  input type := listmode
  input file := listmode.l.hdr

  ; Time frame
  time frame definition := time_frames.fdef

  output filename prefix := recon_lm

  number of subsets := 12
  number of subiterations := 24

  initial estimate := init.hv

  projector pair type := Matrix
    Projector Pair Using Matrix Parameters :=
      Matrix type := Ray Tracing
    End Projector Pair Using Matrix Parameters :=

  ; Corrections
  attenuation image filename := mumap.hv

End :=
EOF

# Reconstruct
OSMAPOSL osem_listmode.par
```

---

## Python Interface

### Basic Python Reconstruction

```python
import stir
import numpy as np

# Load sinogram
sino = stir.ProjData.read_from_file('sino.hs')
print(f"Sinogram dimensions: {sino.get_num_segments()}, "
      f"{sino.get_num_axial_poss(0)}, {sino.get_num_views()}, {sino.get_num_tangential_poss()}")

# Create initial image
target_image = sino.get_empty_image()
target_image.fill(1.0)

# Set up projector
proj_data_info = sino.get_proj_data_info()
projector = stir.ProjectorByBinUsingRayTracing()
projector.set_up(proj_data_info, target_image.get_exam_info())

# Create acquisition model
acq_model = stir.AcqModUsingMatrix()
acq_model.set_up(proj_data_info, target_image.get_exam_info())

# Set up OSEM reconstructor
recon = stir.OSMAPOSLReconstructor()
recon.set_num_subsets(12)
recon.set_num_subiterations(24)
recon.set_input_data(sino)
recon.set_up(target_image)

# Reconstruct
recon.set_current_estimate(target_image)
recon.reconstruct()

# Get result
result_image = recon.get_current_estimate()

# Save
output = stir.ImageData(result_image)
output.write('recon_python.hv')

print("Reconstruction complete")
```

### Python with Attenuation Correction

```python
import stir

# Load data
sino = stir.ProjData.read_from_file('sino.hs')
attn_image = stir.ImageData.read_from_file('mumap.hv')

# Create initial image
init_image = sino.get_empty_image()
init_image.fill(1.0)

# Set up acquisition model with attenuation
proj_data_info = sino.get_proj_data_info()
acq_model = stir.AcqModUsingMatrix()

# Create attenuation model
attn_model = stir.AttenModUsingImages(attn_image)

# Combine into acquisition model
acq_model.set_up(proj_data_info, init_image.get_exam_info())

# Set up OSEM with attenuation
recon = stir.OSMAPOSLReconstructor()
recon.set_num_subsets(12)
recon.set_num_subiterations(24)
recon.set_input_data(sino)
# recon.set_attenuation_image(attn_image)  # Method may vary by version
recon.set_up(init_image)

# Reconstruct
recon.set_current_estimate(init_image)
recon.reconstruct()

# Get and save result
result_image = recon.get_current_estimate()
result_image.write('recon_python_ac.hv')

print("Attenuation-corrected reconstruction complete")
```

### Batch Processing with Python

```python
import stir
import os
from pathlib import Path

# List of subjects
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
data_root = Path('/data/pet')
output_root = Path('/data/pet_recon')

# Reconstruction parameters
n_subsets = 12
n_subiterations = 24

for subject in subjects:
    print(f"\nProcessing {subject}")

    # Define paths
    sino_file = data_root / subject / 'sino.hs'
    attn_file = data_root / subject / 'mumap.hv'
    output_dir = output_root / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check files exist
    if not sino_file.exists() or not attn_file.exists():
        print(f"Missing files for {subject}, skipping...")
        continue

    try:
        # Load data
        sino = stir.ProjData.read_from_file(str(sino_file))
        attn_image = stir.ImageData.read_from_file(str(attn_file))

        # Create initial image
        init_image = sino.get_empty_image()
        init_image.fill(1.0)

        # Set up OSEM
        recon = stir.OSMAPOSLReconstructor()
        recon.set_num_subsets(n_subsets)
        recon.set_num_subiterations(n_subiterations)
        recon.set_input_data(sino)
        recon.set_up(init_image)

        # Reconstruct
        recon.set_current_estimate(init_image)
        recon.reconstruct()

        # Get result
        result_image = recon.get_current_estimate()

        # Save
        output_file = output_dir / 'recon_osem.hv'
        result_image.write(str(output_file))

        print(f"{subject}: Success")

    except Exception as e:
        print(f"{subject}: Failed - {e}")

print("\nBatch processing complete!")
```

---

## Sinogram Processing Utilities

### Display Sinogram Information

```bash
# Get sinogram metadata
list_projdata_info sino.hs

# Output includes:
# - Scanner type
# - Number of segments, views, tangential positions
# - Axial compression
# - Time frame information
```

### Extract Sinogram Subset

```bash
# Extract specific segment
cat > extract.par << 'EOF'
ExtractSegmentsParameters :=
  input file := sino.hs
  output filename prefix := sino_segment0

  ; Extract only segment 0
  segment numbers := {0}

End :=
EOF

extract_segments extract.par

# Output: sino_segment0.hs
```

### Sinogram Arithmetic

```bash
# Subtract scatter from emission sinogram
stir_subtract sino_corrected.hs sino.hs scatter.hs

# Add noise
# (Use Python or custom utilities)
```

---

## PSF Modeling for Resolution Recovery

### OSEM with PSF Correction

```bash
# Parameter file with PSF modeling
cat > osem_psf.par << 'EOF'
OSMAPOSLParameters :=
  input file := sino.hs
  output filename prefix := recon_osem_psf

  number of subsets := 12
  number of subiterations := 36
  save estimates at subiteration intervals := 36

  initial estimate := init.hv

  ; Projector with PSF modeling
  projector pair type := Matrix
    Projector Pair Using Matrix Parameters :=
      Matrix type := Ray Tracing
      Ray tracing matrix parameters :=
        number of rays in tangential direction to det := 10

        ; PSF parameters (Gaussian model)
        do symmetry 90degrees min phi := 1
        do symmetry 180degrees min phi := 1
        do symmetry swap segment := 1
        do symmetry swap side := 1
        do symmetry shift z := 1

        ; Resolution modeling
        ; (Specific PSF parameters depend on scanner)

      End Ray tracing matrix parameters :=
    End Projector Pair Using Matrix Parameters :=

  attenuation image filename := mumap.hv

End :=
EOF

# Reconstruct with PSF
OSMAPOSL osem_psf.par
```

---

## Advanced Features

### Reconstruction with Priors

```bash
# OSEM with quadratic prior
cat > osem_prior.par << 'EOF'
OSMAPOSLParameters :=
  input file := sino.hs
  output filename prefix := recon_osem_prior

  number of subsets := 12
  number of subiterations := 36

  initial estimate := init.hv

  projector pair type := Matrix
    Projector Pair Using Matrix Parameters :=
      Matrix type := Ray Tracing
    End Projector Pair Using Matrix Parameters :=

  ; Regularization with quadratic prior
  prior type := Quadratic
    Quadratic Prior Parameters :=
      penalisation factor := 0.5
      ; Neighbors in each direction
      only 2D := 0
    End Quadratic Prior Parameters :=

  attenuation image filename := mumap.hv

End :=
EOF

# Reconstruct
OSMAPOSL osem_prior.par
```

### Custom Algorithm Development

```cpp
// Example C++ code for custom reconstruction algorithm
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"

// Create custom objective function
stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData
  objective_function;

// Set projection data
objective_function.set_proj_data_sptr(proj_data_sptr);

// Set acquisition model (including corrections)
objective_function.set_projector_pair_sptr(projector_pair_sptr);

// Custom reconstruction loop
for (int iter = 0; iter < num_iterations; ++iter) {
  // Custom algorithm implementation
  // Access gradient, Hessian, etc.
}
```

---

## Integration with Neuroimaging Tools

### STIR to NIfTI Conversion

```bash
# Convert STIR Interfile to NIfTI
stir_image_to_nifti recon_osem_24.hv recon_osem_24.nii.gz

# Or use Python
python3 << EOF
import stir
import nibabel as nib
import numpy as np

# Load STIR image
img_stir = stir.ImageData.read_from_file('recon_osem_24.hv')
img_array = stir.stirextra.to_numpy(img_stir)

# Get voxel sizes
vox_sizes = img_stir.get_voxel_size()
affine = np.diag([vox_sizes.x(), vox_sizes.y(), vox_sizes.z(), 1.0])

# Create NIfTI
nii_img = nib.Nifti1Image(img_array, affine)
nib.save(nii_img, 'recon_osem_24.nii.gz')

print("Converted to NIfTI")
EOF
```

### STIR in BIDS Pipeline

```bash
#!/bin/bash
# STIR reconstruction in BIDS workflow

SUBJECT=sub-01
SESSION=baseline
BIDS_ROOT=/data/bids_dataset

# Input raw PET data
RAW_DATA=/data/raw_pet/${SUBJECT}/sino.hs

# Output directory
PET_DIR=${BIDS_ROOT}/sub-${SUBJECT}/ses-${SESSION}/pet
mkdir -p ${PET_DIR}

# Reconstruct with STIR
OSMAPOSL osem.par

# Convert to NIfTI
stir_image_to_nifti recon_osem_24.hv ${PET_DIR}/sub-${SUBJECT}_ses-${SESSION}_pet.nii.gz

# Create JSON sidecar
cat > ${PET_DIR}/sub-${SUBJECT}_ses-${SESSION}_pet.json << EOF
{
  "Modality": "PT",
  "Manufacturer": "Siemens",
  "ReconstructionMethod": "OSEM",
  "ReconstructionSoftware": "STIR",
  "NumberOfIterations": 2,
  "NumberOfSubsets": 12,
  "NumberOfSubiterations": 24,
  "AttenuationCorrection": "CT-based",
  "ScatterCorrection": "single scatter simulation",
  "RandomsCorrection": "delayed window"
}
EOF

echo "BIDS PET created"
```

---

## Quality Control and Validation

### Verify Reconstruction Quality

```python
import stir
import matplotlib.pyplot as plt
import numpy as np

# Load reconstructed image
img = stir.ImageData.read_from_file('recon_osem_24.hv')
img_array = stir.stirextra.to_numpy(img)

# Central slices
nz, ny, nx = img_array.shape
central_slice = img_array[nz//2, :, :]

# Display
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_array[nz//2, :, :], cmap='hot')
plt.title('Axial Slice')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_array[:, ny//2, :], cmap='hot')
plt.title('Coronal Slice')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_array[:, :, nx//2], cmap='hot')
plt.title('Sagittal Slice')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(img_array.flatten(), bins=100, range=(0, np.percentile(img_array, 99)))
plt.title('Intensity Histogram')
plt.xlabel('Voxel Value (Bq/mL)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('qc_reconstruction.png', dpi=300)
print("QC plot saved")
```

### Compare Reconstruction Algorithms

```bash
#!/bin/bash
# Compare FBP vs. OSEM

# FBP reconstruction
FBP2D fbp.par

# OSEM reconstruction
OSMAPOSL osem.par

# Display both (requires STIR visualization tools or external)
echo "Reconstructions complete - compare outputs"
```

---

## Troubleshooting

### Compilation Errors

```bash
# Check CMake configuration
cd build
cmake .. -LAH | less  # View all CMake variables

# Common issues:
# 1. Boost not found
export BOOST_ROOT=/usr/local/boost_1_70_0

# 2. ITK not found
cmake .. -DITK_DIR=/path/to/itk/lib/cmake/ITK-5.0

# 3. Missing dependencies
sudo apt-get install libboost-all-dev libinsighttoolkit4-dev

# Reconfigure and rebuild
cmake ..
make clean
make -j$(nproc)
```

### Runtime Errors

```bash
# Check parfile syntax
# STIR will report line number of errors

# Verify input files exist
list_image_info mumap.hv
list_projdata_info sino.hs

# Check file format compatibility
# Ensure sinogram and image geometries match

# Enable verbose output
OSMAPOSL osem.par --verbosity 2
```

### Memory Issues

```bash
# For large datasets, use iterative saving
# In parfile:
save estimates at subiteration intervals := 6

# This saves intermediate results and frees memory

# Or increase system swap
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Best Practices

### Reconstruction Workflow

1. **Start with FBP:**
   - Quick, analytic reconstruction
   - Good initial estimate for iterative methods
   - Helps identify data quality issues

2. **Progress to OSEM:**
   - Better SNR than FBP
   - Standard clinical choice
   - 2-4 iterations usually sufficient

3. **Add Corrections Incrementally:**
   - First: Attenuation correction
   - Second: Scatter correction
   - Third: Randoms correction
   - Verify at each step

4. **Use Priors for Low Counts:**
   - Quadratic prior for smoothing
   - Adjust penalization factor carefully
   - Avoid over-smoothing

### Parameter Selection

**OSEM:**
- Subsets: 12-21 (more = faster but noisier early iterations)
- Subiterations: 24-48 (2-4 full iterations)
- Subset scheme: Random (better convergence)

**FBP:**
- Ramp filter for standard resolution
- Hanning or Hamming for noise reduction
- Adjust cut-off frequency for smoothness

---

## Resources and Further Reading

### Official Documentation

- **STIR Homepage:** http://stir.sourceforge.net/
- **Documentation:** http://stir.sourceforge.net/documentation/
- **GitHub:** https://github.com/UCL/STIR
- **Mailing List:** stir-users@lists.sourceforge.net

### Tutorials

- **STIR User's Guide:** Comprehensive manual
- **Example Parfiles:** In STIR examples/ directory
- **Python Examples:** examples/python/ directory

### Related Tools

- **SIRF:** High-level Python interface to STIR
- **NiftyPET:** GPU-accelerated alternative
- **AMIDE:** PET viewer for STIR output
- **GATE:** Monte Carlo simulation (STIR-compatible)

### Citations

If you use STIR, please cite:

```
Thielemans, K., et al. (2012).
STIR: Software for Tomographic Image Reconstruction Release 2.
Physics in Medicine & Biology, 57(4), 867-883.
```

---

## Summary

**STIR** is the foundation library for PET/SPECT reconstruction research:

**Strengths:**
- Comprehensive reconstruction algorithms
- Flexible and extensible C++ framework
- Well-established (20+ years development)
- Support for multiple scanners
- Excellent documentation
- Active community and development
- Ideal for algorithm development

**Limitations:**
- Steeper learning curve than specialized tools
- Parameter file syntax can be verbose
- GPU acceleration experimental
- Slower than GPU-based tools (NiftyPET)

**Best For:**
- Reconstruction algorithm research and development
- Multi-algorithm comparison
- Scanner calibration and QC
- Teaching reconstruction concepts
- Reproducible research with parfiles
- Integration via SIRF for high-level interface

For clinical viewing, use **AMIDE**. For GPU acceleration, use **NiftyPET**. For Python-based workflows, use **SIRF** (which wraps STIR). STIR excels as the foundational reconstruction engine and algorithm development platform.
