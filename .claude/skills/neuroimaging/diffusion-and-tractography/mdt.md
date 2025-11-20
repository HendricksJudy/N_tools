# MDT (Microstructure Diffusion Toolbox)

## Overview

**MDT** (Microstructure Diffusion Toolbox) is a GPU/OpenCL-accelerated framework for fitting microstructure models to diffusion MRI data. Developed at the Maastricht Brain Imaging Centre, MDT implements numerous biophysical models with fast parallel processing capabilities on GPUs or multi-core CPUs. The toolbox emphasizes ease of use with sensible defaults while providing flexibility for advanced users to define custom models and cascades.

MDT's key strength is computational speed through GPU acceleration, making it practical to fit complex multi-compartment models to large datasets. It includes 20+ pre-implemented models (NODDI, CHARMED, Ball&Stick, Tensor, etc.), cascade fitting strategies for robust parameter estimation, and tools for protocol optimization and model comparison.

**Key Use Cases:**
- Fast GPU-accelerated microstructure modeling
- Large-scale diffusion studies requiring quick processing
- Clinical research with complex models (NODDI, CHARMED)
- Protocol optimization for acquisition design
- Custom model development and testing
- Multi-subject batch processing
- Uncertainty quantification via sampling

**Official Website:** https://mdt-toolbox.readthedocs.io/
**Documentation:** https://mdt-toolbox.readthedocs.io/en/latest/
**Source Code:** https://github.com/robbert-harms/MDT

---

## Key Features

- **GPU/OpenCL Acceleration:** 10-100x faster than CPU-only methods
- **20+ Pre-Implemented Models:** NODDI, CHARMED, Tensor, Ball&Stick, ActiveAx, CHARMED_r1-3, AxCaliber
- **Cascade Fitting:** Automatic initialization for complex models
- **Multi-Device Support:** GPU (NVIDIA, AMD) and CPU via OpenCL
- **Command-Line Interface:** Simple one-line fitting commands
- **Python API:** Full programmatic control
- **Protocol Optimization:** Design optimal acquisition schemes
- **Model Comparison:** Statistical model selection tools
- **Uncertainty Quantification:** MCMC sampling for parameter uncertainty
- **Automatic Masking:** Built-in brain extraction
- **Flexible Configuration:** YAML-based model and cascade definitions
- **Custom Models:** Define new compartments and models
- **Batch Processing:** Process multiple subjects efficiently
- **Cross-Platform:** Windows, Linux, macOS
- **Well-Documented:** Comprehensive tutorials and examples
- **Active Development:** Regular updates and community support

---

## Installation

### Prerequisites: OpenCL

MDT requires OpenCL drivers for GPU or CPU:

```bash
# NVIDIA GPU (install CUDA)
# Download from: https://developer.nvidia.com/cuda-downloads

# AMD GPU (install ROCm)
# Linux: https://rocmdocs.amd.com/

# Intel CPU/GPU (install Intel OpenCL)
# Download from: https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html

# Verify OpenCL installation
clinfo  # Should list available OpenCL devices
```

### Install MDT

```bash
# Install MDT via pip
pip install mdt

# Install with all optional dependencies
pip install mdt[all]

# Verify installation
mdt-info  # Shows OpenCL devices and MDT version
python -c "import mdt; print(mdt.__version__)"
```

### Using Conda

```bash
# Create conda environment
conda create -n mdt python=3.9
conda activate mdt

# Install OpenCL support
conda install -c conda-forge pyopencl

# Install MDT
pip install mdt

# Verify GPU/OpenCL
python -c "import pyopencl as cl; print(cl.get_platforms())"
```

### GPU Setup Verification

```python
import mdt

# List available compute devices
devices = mdt.get_cl_devices()
for device in devices:
    print(f"Device: {device.name}")
    print(f"  Type: {device.device_type}")
    print(f"  Memory: {device.global_mem_size / 1024**3:.2f} GB")
```

---

## Basic Usage

### Command-Line Fitting

```bash
# Fit NODDI model (simplest usage)
mdt-model-fit NODDI dwi.nii.gz dwi.prtcl brain_mask.nii.gz -o output/

# Protocol file (.prtcl) contains bvals, bvecs, and other parameters
# Can be created from separate bval/bvec files:
mdt-create-protocol dwi.bval dwi.bvec -o dwi.prtcl

# Fit with specific GPU
mdt-model-fit NODDI dwi.nii.gz dwi.prtcl brain_mask.nii.gz -o output/ --cl-device-ind 0

# Fit Ball&Stick model
mdt-model-fit BallStick_r1 dwi.nii.gz dwi.prtcl brain_mask.nii.gz -o output/
```

### Create Protocol File

```bash
# From bval and bvec files
mdt-create-protocol dwi.bval dwi.bvec -o dwi.prtcl

# With additional parameters
mdt-create-protocol dwi.bval dwi.bvec \
    --TE 100 \
    --delta 12.9 \
    --Delta 21.8 \
    -o dwi.prtcl
```

### View Results

```bash
# MDT saves parameter maps as NIFTI files
ls output/NODDI/
# Output files:
# - NODDI.nii.gz (4D volume with all parameters)
# - NDI.nii.gz (Neurite Density Index)
# - ODI.nii.gz (Orientation Dispersion Index)
# - FR.nii.gz (Free water fraction)
# etc.
```

---

## Python API

### Basic Model Fitting

```python
import mdt
import nibabel as nib

# Load data
dwi = mdt.load_nifti('dwi.nii.gz').get_fdata()
mask = mdt.load_nifti('brain_mask.nii.gz').get_fdata()

# Load protocol
protocol = mdt.load_protocol('dwi.prtcl')

# Fit NODDI model
noddi_results = mdt.fit_model(
    'NODDI',
    dwi,
    protocol,
    mask,
    output_folder='output/noddi'
)

# Access results
ndi = noddi_results['NDI']
odi = noddi_results['ODI']
fiso = noddi_results['w_csf.w']

print(f"Mean NDI: {ndi[mask > 0].mean():.3f}")
print(f"Mean ODI: {odi[mask > 0].mean():.3f}")
```

### Load and Save Results

```python
# Save individual parameter maps
import nibabel as nib

# Get affine from original image
img = nib.load('dwi.nii.gz')
affine = img.affine

# Save NDI map
ndi_img = nib.Nifti1Image(ndi, affine)
nib.save(ndi_img, 'ndi_map.nii.gz')

# Load previously fitted results
noddi_results = mdt.load_volume_maps('output/noddi')
ndi_loaded = noddi_results['NDI']
```

---

## Pre-Implemented Models

### Available Models

```python
# List all available models
models = mdt.get_models_list()
print("Available models:")
for model in models:
    print(f"  - {model}")

# Common models:
# - Tensor
# - BallStick_r1, BallStick_r2, BallStick_r3
# - NODDI
# - CHARMED_r1, CHARMED_r2, CHARMED_r3
# - ActiveAx
# - AxCaliber
# - NODDI_GM (gray matter)
```

### DTI (Tensor Model)

```python
# Fit Diffusion Tensor Imaging model
tensor_results = mdt.fit_model(
    'Tensor',
    dwi,
    protocol,
    mask
)

# Extract DTI metrics
fa = tensor_results['Tensor.FA']
md = tensor_results['Tensor.MD']
ad = tensor_results['Tensor.AD']
rd = tensor_results['Tensor.RD']

print(f"Mean FA: {fa[mask > 0].mean():.3f}")
```

### Ball&Stick Models

```python
# Ball&Stick with 1 stick (single fiber)
ballstick_r1 = mdt.fit_model(
    'BallStick_r1',
    dwi,
    protocol,
    mask
)

# Ball&Stick with 2 sticks (crossing fibers)
ballstick_r2 = mdt.fit_model(
    'BallStick_r2',
    dwi,
    protocol,
    mask
)

# Ball&Stick with 3 sticks (complex crossing)
ballstick_r3 = mdt.fit_model(
    'BallStick_r3',
    dwi,
    protocol,
    mask
)

# Get fiber fractions
f1 = ballstick_r2['w_stick0.w']
f2 = ballstick_r2['w_stick1.w']
```

### CHARMED

```python
# CHARMED: Composite Hindered and Restricted Model of Diffusion
# CHARMED_r1: 1 restricted compartment
charmed_r1 = mdt.fit_model(
    'CHARMED_r1',
    dwi,
    protocol,
    mask
)

# CHARMED_r2: 2 restricted compartments (crossing)
charmed_r2 = mdt.fit_model(
    'CHARMED_r2',
    dwi,
    protocol,
    mask
)

# Extract restricted fraction
restricted_fraction = charmed_r1['FR']
print(f"Mean restricted fraction: {restricted_fraction[mask > 0].mean():.3f}")
```

---

## Cascade Fitting

### Understand Cascades

MDT uses cascade fitting to initialize complex models:

```python
# View cascade for NODDI
cascade = mdt.get_model_list_per_model_name('NODDI')
print("NODDI cascade:")
for i, model in enumerate(cascade):
    print(f"  {i+1}. {model}")

# Typical NODDI cascade:
# 1. BallStick_r1 (simple initialization)
# 2. NODDI (final model)
```

### Custom Cascade

```python
# Define custom cascade
custom_cascade = ['Tensor', 'BallStick_r1', 'NODDI']

# Fit with custom cascade
noddi_custom = mdt.fit_model(
    'NODDI',
    dwi,
    protocol,
    mask,
    initialization_data={
        'inits': None,  # Start from scratch
        'fixes': None
    }
)
```

### Provide Initial Values

```python
# Use previous fit as initialization
ballstick_results = mdt.fit_model('BallStick_r1', dwi, protocol, mask)

# Initialize NODDI with BallStick results
noddi_initialized = mdt.fit_model(
    'NODDI',
    dwi,
    protocol,
    mask,
    initialization_data={
        'inits': ballstick_results,  # Use previous fit
        'fixes': {}
    }
)
```

---

## GPU Configuration

### Select GPU Device

```python
# List available devices
devices = mdt.get_cl_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device.name}")

# Use specific device
mdt.set_cl_device_ind(0)  # Use first GPU

# Or specify during fitting
noddi_results = mdt.fit_model(
    'NODDI',
    dwi,
    protocol,
    mask,
    cl_device_ind=0  # Use specific GPU
)
```

### Optimize GPU Performance

```python
# Configure GPU settings
import mdt.configuration as config

# Set number of workers (parallel voxels)
config.set_cl_device_ind(0)
config.set_use_local_reduction(True)

# Batch size for GPU processing
mdt.fit_model(
    'NODDI',
    dwi,
    protocol,
    mask,
    cl_device_ind=0,
    double_precision=False  # Use single precision for speed
)
```

### CPU Fallback

```python
# Use CPU if GPU not available
import pyopencl as cl

# Find CPU device
cpu_devices = [d for d in mdt.get_cl_devices() if 'CPU' in d.name.upper()]

if cpu_devices:
    mdt.set_cl_device_ind(cpu_devices[0].index)
    print("Using CPU for computation")
```

---

## Protocol Optimization

### Generate Protocol

```python
from mdt.protocols import write_protocol, generate_protocol

# Generate simple protocol
protocol_data = generate_protocol(
    bvalues=[0, 1000, 2000, 3000],  # b-values
    n_directions=[1, 30, 60, 60],   # Directions per shell
    delta=12.9,  # Gradient duration (ms)
    Delta=21.8,  # Gradient separation (ms)
    TE=100       # Echo time (ms)
)

# Save protocol
write_protocol(protocol_data, 'optimized_protocol.prtcl')
```

### Optimize for Model

```python
# Optimize protocol for specific model
from mdt.protocols import create_protocol_optimal_for_model

optimal_protocol = create_protocol_optimal_for_model(
    'NODDI',
    n_measurements=200,  # Total number of volumes
    min_bval=0,
    max_bval=3000,
    delta=12.9,
    Delta=21.8
)

write_protocol(optimal_protocol, 'noddi_optimal.prtcl')
```

---

## Batch Processing

### Process Multiple Subjects

```python
from pathlib import Path

def batch_fit_mdt(subjects_dir, output_dir, model='NODDI'):
    """Batch process multiple subjects with MDT."""

    subjects_dir = Path(subjects_dir)
    output_dir = Path(output_dir)

    # Find all subjects
    subjects = sorted(subjects_dir.glob('sub-*'))

    for subject_dir in subjects:
        subject_id = subject_dir.name
        print(f"\nProcessing {subject_id}...")

        # File paths
        dwi_file = subject_dir / 'dwi' / f'{subject_id}_dwi.nii.gz'
        mask_file = subject_dir / 'dwi' / f'{subject_id}_brain_mask.nii.gz'
        protocol_file = subject_dir / 'dwi' / f'{subject_id}_dwi.prtcl'

        # Check files exist
        if not all([f.exists() for f in [dwi_file, mask_file, protocol_file]]):
            print(f"  Skipping {subject_id}: missing files")
            continue

        # Load data
        dwi = mdt.load_nifti(str(dwi_file)).get_fdata()
        mask = mdt.load_nifti(str(mask_file)).get_fdata()
        protocol = mdt.load_protocol(str(protocol_file))

        # Fit model
        output_subdir = output_dir / subject_id / model
        output_subdir.mkdir(parents=True, exist_ok=True)

        try:
            results = mdt.fit_model(
                model,
                dwi,
                protocol,
                mask,
                output_folder=str(output_subdir)
            )
            print(f"  Success: {subject_id}")

        except Exception as e:
            print(f"  Error: {subject_id} - {e}")

# Run batch processing
batch_fit_mdt(
    subjects_dir='/data/subjects',
    output_dir='/data/derivatives/mdt',
    model='NODDI'
)
```

---

## Custom Model Definition

### Define Custom Compartment

```python
# Create custom compartment model
from mdt.models.compartments import CompartmentTemplate

custom_compartment = '''
model_name = 'MyCustomCompartment'

class MyCustomCompartment(CompartmentTemplate):
    """Custom diffusion compartment."""

    parameters = (
        'g',       # Gradient strength
        'b',       # b-value
        'theta',   # Polar angle
        'phi',     # Azimuthal angle
        'd',       # Diffusivity
    )

    cl_code = """
        double MyCustomCompartment(double g, double b, double theta, double phi, double d) {
            // Custom signal equation
            double signal = exp(-b * d);
            return signal;
        }
    """
'''

# Register custom compartment
mdt.add_cl_code(custom_compartment)
```

### Build Custom Model

```python
# Combine compartments into custom model
from mdt.models.composite import CompositeModelTemplate

custom_model = '''
model_name = 'MyCustomModel'

class MyCustomModel(CompositeModelTemplate):
    """Custom multi-compartment model."""

    compartments = {
        'Ball': 'Ball()',
        'Stick': 'Stick()',
    }

    parameters = ('w_ball', 'w_stick')

    dependencies = ['Ball', 'Stick']

    signal_equation = """
        w_ball * Ball + w_stick * Stick
    """

    constraints = """
        w_ball + w_stick == 1
    """
'''

# Fit custom model
custom_results = mdt.fit_model(
    'MyCustomModel',
    dwi,
    protocol,
    mask
)
```

---

## Model Comparison

### Compare Model Fit Quality

```python
# Fit multiple models
tensor_fit = mdt.fit_model('Tensor', dwi, protocol, mask)
ballstick_fit = mdt.fit_model('BallStick_r1', dwi, protocol, mask)
noddi_fit = mdt.fit_model('NODDI', dwi, protocol, mask)

# Calculate BIC (Bayesian Information Criterion) for each
from mdt.lib.model_selection import compute_BIC

tensor_bic = compute_BIC(
    'Tensor',
    dwi,
    protocol,
    mask,
    tensor_fit
)

ballstick_bic = compute_BIC(
    'BallStick_r1',
    dwi,
    protocol,
    mask,
    ballstick_fit
)

noddi_bic = compute_BIC(
    'NODDI',
    dwi,
    protocol,
    mask,
    noddi_fit
)

# Lower BIC is better
print(f"Tensor BIC: {tensor_bic.mean():.2f}")
print(f"BallStick BIC: {ballstick_bic.mean():.2f}")
print(f"NODDI BIC: {noddi_bic.mean():.2f}")
```

### Likelihood Ratio Test

```python
# Compare nested models
from scipy import stats

# Calculate log-likelihoods
ll_simple = -tensor_bic / 2  # Simplified
ll_complex = -noddi_bic / 2

# Likelihood ratio statistic
lr_stat = 2 * (ll_complex - ll_simple)

# Degrees of freedom difference
df_diff = len(noddi_fit) - len(tensor_fit)

# p-value
p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

print(f"Likelihood ratio test p-value: {p_value}")
```

---

## Uncertainty Quantification

### MCMC Sampling

```python
# Run MCMC for uncertainty estimation
from mdt import sample_model

# Sample NODDI parameters
samples = sample_model(
    'NODDI',
    dwi,
    protocol,
    mask,
    nmr_samples=1000,  # Number of MCMC samples
    burnin=100,        # Burn-in samples
    thinning=1         # Thinning factor
)

# Get parameter distributions
ndi_samples = samples['NDI']  # Shape: (n_voxels, n_samples)

# Calculate statistics
ndi_mean = ndi_samples.mean(axis=1)
ndi_std = ndi_samples.std(axis=1)
ndi_95ci_low = np.percentile(ndi_samples, 2.5, axis=1)
ndi_95ci_high = np.percentile(ndi_samples, 97.5, axis=1)

print(f"NDI mean: {ndi_mean.mean():.3f} ± {ndi_std.mean():.3f}")
```

---

## Integration with Claude Code

MDT integrates well with automated pipelines:

```python
# mdt_pipeline.py - Automated MDT processing

import mdt
import nibabel as nib
from pathlib import Path
import logging

class MDTPipeline:
    """Automated MDT microstructure analysis pipeline."""

    def __init__(self, model='NODDI', device_ind=0):
        self.model = model
        self.device_ind = device_ind
        mdt.set_cl_device_ind(device_ind)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_subject(self, dwi_file, protocol_file, mask_file, output_dir):
        """Process single subject with MDT."""

        self.logger.info(f"Processing: {dwi_file}")

        # Load data
        dwi = mdt.load_nifti(dwi_file).get_fdata()
        mask = mdt.load_nifti(mask_file).get_fdata()
        protocol = mdt.load_protocol(protocol_file)

        # Fit model
        self.logger.info(f"Fitting {self.model} model...")
        results = mdt.fit_model(
            self.model,
            dwi,
            protocol,
            mask,
            output_folder=output_dir,
            cl_device_ind=self.device_ind
        )

        self.logger.info(f"Results saved to: {output_dir}")
        return results

    def batch_process(self, subjects_list, output_base):
        """Batch process list of subjects."""

        for subject_info in subjects_list:
            subject_id = subject_info['id']
            output_dir = Path(output_base) / subject_id / self.model

            try:
                self.process_subject(
                    subject_info['dwi'],
                    subject_info['protocol'],
                    subject_info['mask'],
                    str(output_dir)
                )
            except Exception as e:
                self.logger.error(f"Failed {subject_id}: {e}")

# Usage
pipeline = MDTPipeline(model='NODDI', device_ind=0)

# Process single subject
results = pipeline.process_subject(
    dwi_file='/data/sub-01/dwi.nii.gz',
    protocol_file='/data/sub-01/dwi.prtcl',
    mask_file='/data/sub-01/mask.nii.gz',
    output_dir='/data/derivatives/mdt/sub-01/NODDI'
)
```

---

## Integration with Other Tools

### QSIPrep Integration

```python
# Process QSIPrep outputs with MDT
from pathlib import Path

def process_qsiprep_with_mdt(qsiprep_dir, subject, output_dir):
    """Process QSIPrep preprocessed data with MDT."""

    subj_dir = Path(qsiprep_dir) / subject / 'dwi'

    # Load QSIPrep outputs
    dwi_file = subj_dir / f'{subject}_space-T1w_desc-preproc_dwi.nii.gz'
    bval_file = subj_dir / f'{subject}_space-T1w_desc-preproc_dwi.bval'
    bvec_file = subj_dir / f'{subject}_space-T1w_desc-preproc_dwi.bvec'
    mask_file = subj_dir / f'{subject}_space-T1w_desc-brain_mask.nii.gz'

    # Create MDT protocol
    protocol_file = f'/tmp/{subject}_dwi.prtcl'
    mdt.create_protocol(str(bval_file), str(bvec_file), protocol_file)

    # Load for MDT
    dwi = mdt.load_nifti(str(dwi_file)).get_fdata()
    mask = mdt.load_nifti(str(mask_file)).get_fdata()
    protocol = mdt.load_protocol(protocol_file)

    # Fit NODDI
    results = mdt.fit_model('NODDI', dwi, protocol, mask, output_folder=output_dir)

    return results

# Process QSIPrep subject
results = process_qsiprep_with_mdt(
    '/data/derivatives/qsiprep',
    'sub-01',
    '/data/derivatives/mdt/sub-01'
)
```

### DIPY Integration

```python
# Use DIPY for preprocessing, MDT for modeling
from dipy.io.image import load_nifti
from dipy.core.gradients import gradient_table

# Load with DIPY
data, affine = load_nifti('dwi.nii.gz')
bvals, bvecs = read_bvals_bvecs('dwi.bval', 'dwi.bvec')
gtab = gradient_table(bvals, bvecs)

# Convert to MDT protocol
protocol = mdt.protocols.create_protocol_from_bvals_bvecs(
    gtab.bvals,
    gtab.bvecs
)

# Fit with MDT
results = mdt.fit_model('NODDI', data, protocol, mask)
```

---

## Troubleshooting

### Problem 1: OpenCL Not Found

**Symptoms:** "No OpenCL devices found" error

**Solution:**
```bash
# Install OpenCL drivers

# NVIDIA GPU
sudo apt-get install nvidia-opencl-dev

# AMD GPU (Linux)
sudo apt-get install mesa-opencl-icd

# Intel CPU
sudo apt-get install intel-opencl-icd

# Verify
clinfo
python -c "import pyopencl; print(pyopencl.get_platforms())"
```

### Problem 2: GPU Out of Memory

**Symptoms:** CUDA/OpenCL out of memory errors

**Solution:**
```python
# Process in smaller batches
# Or use CPU
cpu_devices = [d for d in mdt.get_cl_devices() if 'CPU' in d.name.upper()]
mdt.set_cl_device_ind(cpu_devices[0].index)

# Or reduce problem size
# Process ROI instead of whole brain
```

### Problem 3: Fitting Fails

**Symptoms:** Parameters at bounds or unrealistic values

**Solution:**
```python
# Use cascade fitting (default for complex models)
# Check data quality
# Ensure sufficient b-values for model

# Try simpler model first
ballstick = mdt.fit_model('BallStick_r1', dwi, protocol, mask)
# Then fit complex model
noddi = mdt.fit_model('NODDI', dwi, protocol, mask, inits=ballstick)
```

### Problem 4: Protocol File Issues

**Symptoms:** Error loading protocol

**Solution:**
```bash
# Recreate protocol file
mdt-create-protocol dwi.bval dwi.bvec -o dwi.prtcl

# Check protocol validity
mdt-view-protocol dwi.prtcl
```

---

## Best Practices

### 1. Hardware Selection

- **Use GPU:** 10-100x faster than CPU
- **Multiple GPUs:** Set device index appropriately
- **Memory:** Ensure sufficient GPU RAM
- **OpenCL version:** Keep drivers updated

### 2. Model Selection

- **Start simple:** DTI → Ball&Stick → NODDI
- **Match acquisition:** Ensure b-values suit model
- **Use cascades:** Let MDT initialize complex models
- **Validate:** Check parameter maps for artifacts

### 3. Data Quality

- **Preprocessing:** Use QSIPrep, TractoFlow, or DIPY
- **Brain mask:** Accurate masking essential
- **SNR:** Sufficient signal-to-noise ratio
- **Multi-shell:** Required for advanced models (NODDI, CHARMED)

### 4. Performance

- **GPU device:** Use fastest available GPU
- **Batch size:** Process multiple subjects
- **Single precision:** Use `double_precision=False` for speed
- **Parallel:** Use multiple GPUs if available

### 5. Reproducibility

- **Version:** Record MDT version
- **Model:** Save model name and cascade
- **Protocol:** Archive acquisition parameters
- **Seeds:** Set random seeds if using sampling

---

## Resources

### Official Documentation

- **Documentation:** https://mdt-toolbox.readthedocs.io/
- **GitHub:** https://github.com/robbert-harms/MDT
- **Issue Tracker:** https://github.com/robbert-harms/MDT/issues

### Publications

- **MDT Paper:** Harms et al. (2017) "Robust and fast nonlinear optimization of diffusion MRI microstructure models" *NeuroImage*
- **NODDI:** Zhang et al. (2012) "NODDI: Practical in vivo neurite orientation dispersion and density imaging"

### Community

- **GitHub Discussions:** For questions and issues
- **Email:** Contact developers via GitHub

---

## Citation

```bibtex
@article{harms2017robust,
  title={Robust and fast nonlinear optimization of diffusion MRI microstructure models},
  author={Harms, Robbert L and Fritz, Francisco J and Tobisch, Alexander and Goebel, Rainer and Roebroeck, Alard},
  journal={NeuroImage},
  volume={155},
  pages={82--96},
  year={2017},
  publisher={Elsevier},
  doi={10.1016/j.neuroimage.2017.04.064}
}
```

---

## Related Tools

- **DMIPY:** Python microstructure framework (see `dmipy.md`)
- **DIPY:** Foundation for diffusion processing (see `dipy.md`)
- **QSIPrep:** Preprocessing pipeline (see `qsiprep.md`)
- **MRtrix3:** Preprocessing and analysis (see `mrtrix3.md`)
- **NODDI MATLAB Toolbox:** Original NODDI implementation
- **Camino:** Java diffusion toolkit

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**MDT Version Covered:** 1.2.x
**Maintainer:** Claude Code Neuroimaging Skills
