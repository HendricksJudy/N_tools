# DMIPY (Diffusion Microstructure Imaging in Python)

## Overview

**DMIPY** (Diffusion Microstructure Imaging in Python) is a comprehensive open-source framework for estimating microstructural features from diffusion MRI data using multi-compartment models. Developed at the Athena Project Team (Inria Sophia Antipolis), DMIPY provides modular building blocks for creating custom tissue models, implementing state-of-the-art biophysical models, and performing robust parameter estimation with multiple optimization strategies.

DMIPY's modular design allows researchers to construct complex multi-compartment models from simple components (cylinders, spheres, tensors), implement established models (NODDI, CHARMED, AxCaliber), or develop novel approaches. The framework emphasizes reproducibility, supports multi-shell and multi-tissue imaging, and provides tools for model comparison and validation.

**Key Use Cases:**
- Quantitative tissue microstructure imaging
- White matter axon diameter and density estimation
- Neurite orientation dispersion quantification
- Multi-compartment model development and testing
- Biophysical parameter mapping
- Method development and validation
- Clinical biomarker extraction

**Official Website:** https://dmipy.readthedocs.io/
**Documentation:** https://dmipy.readthedocs.io/en/latest/
**Source Code:** https://github.com/AthenaEPI/dmipy

---

## Key Features

- **Modular Framework:** Build models from basic compartment components
- **Biophysical Models:** NODDI, CHARMED, AxCaliber, Ball-Stick, ActiveAx, SMTMC
- **Spherical Mean Technique (SMT):** Rotationally invariant microstructure imaging
- **Multi-Shell Support:** Leverage multiple b-values for tissue characterization
- **Multi-Tissue Modeling:** White matter, gray matter, CSF compartments
- **Parameter Estimation:** Multiple optimizers (L-BFGS, differential evolution, MIX)
- **Microstructure Fingerprinting:** Fast dictionary-based estimation
- **Model Comparison:** Statistical tools for model selection
- **DIPY Integration:** Seamless gradient table and data handling
- **Custom Model Building:** Combine compartments with distribution models
- **GPU Acceleration:** Numba JIT compilation for speed
- **Open Source:** Fully transparent implementations
- **Well-Documented:** Extensive tutorials and examples
- **Reproducible Science:** Version-controlled, citable methods
- **Active Development:** Regular updates and community support

---

## Installation

### Using Pip

```bash
# Install DMIPY
pip install dmipy

# Install with all dependencies
pip install dmipy[all]

# Verify installation
python -c "import dmipy; print(dmipy.__version__)"
```

### Using Conda

```bash
# Create conda environment
conda create -n dmipy python=3.9
conda activate dmipy

# Install dependencies
conda install -c conda-forge numpy scipy dipy nibabel matplotlib

# Install DMIPY
pip install dmipy
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/AthenaEPI/dmipy.git
cd dmipy

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Dependencies

```bash
# Core dependencies
pip install numpy scipy dipy nibabel

# Optional for acceleration
pip install numba

# For visualization
pip install matplotlib

# For optimization
pip install cvxpy
```

---

## Basic Concepts

### Compartment Models

DMIPY represents tissue with compartments:

```python
from dmipy.signal_models import cylinder_models, gaussian_models

# Stick: zero-radius cylinder (axon)
stick = cylinder_models.C1Stick()

# Ball: isotropic Gaussian (free water)
ball = gaussian_models.G1Ball()

# Zeppelin: anisotropic Gaussian (hindered diffusion)
zeppelin = gaussian_models.G2Zeppelin()

# Cylinder: finite-radius cylinder (axon with diameter)
cylinder = cylinder_models.C4CylinderGaussianPhaseApproximation()
```

### Multi-Compartment Models

Combine compartments to create tissue models:

```python
from dmipy.core.modeling_framework import MultiCompartmentModel

# Ball-Stick model (simple white matter)
ball_stick = MultiCompartmentModel(models=[ball, stick])

# Ball-Stick-Stick (crossing fibers)
ball_stick_stick = MultiCompartmentModel(
    models=[ball, stick, stick]
)
```

---

## Ball-Stick Model

### Basic Ball-Stick Fitting

```python
import numpy as np
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
from dipy.data import get_fnames, read_stanford_hardi
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib

# Load data
img, gtab = read_stanford_hardi()
data = img.get_fdata()

# Create acquisition scheme from DIPY gradient table
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

bvals = gtab.bvals
bvecs = gtab.bvecs
acq_scheme = acquisition_scheme_from_bvalues(bvals, bvecs)

# Create Ball-Stick model
stick = cylinder_models.C1Stick()
ball = gaussian_models.G1Ball()
ball_stick = MultiCompartmentModel(models=[ball, stick])

# Fit model to single voxel
voxel_data = data[50, 50, 35]
ball_stick_fit = ball_stick.fit(acq_scheme, voxel_data)

# Get fitted parameters
fitted_params = ball_stick_fit.fitted_parameters
print("Fitted parameters:", fitted_params)

# Get parameter names
print("Parameter names:", ball_stick.parameter_names)
```

### Batch Fitting

```python
# Fit to brain region
brain_mask = img.get_fdata()[..., 0] > 100  # Simple threshold mask
roi_data = data[brain_mask]

# Fit model to all voxels in ROI
ball_stick_fit_roi = ball_stick.fit(
    acq_scheme,
    roi_data,
    mask=np.ones(len(roi_data), dtype=bool)
)

# Extract parameter maps
parameter_vector = ball_stick_fit_roi.fitted_parameters

# Get specific parameter (e.g., stick fraction)
stick_fraction = ball_stick_fit_roi.fitted_parameters['partial_volume_0']
print(f"Mean stick fraction in ROI: {stick_fraction.mean():.3f}")
```

### Multi-Fiber Ball-Stick

```python
# Ball with 2 sticks for crossing fibers
stick1 = cylinder_models.C1Stick()
stick2 = cylinder_models.C1Stick()
ball = gaussian_models.G1Ball()

ball_stick_stick = MultiCompartmentModel(
    models=[ball, stick1, stick2]
)

# Fit crossing fiber model
voxel_crossing = data[45, 55, 35]  # Voxel with crossing
ball_stick_stick_fit = ball_stick_stick.fit(
    acq_scheme,
    voxel_crossing
)

# Get fiber orientations
mu1 = ball_stick_stick_fit.fitted_parameters['C1Stick_1_mu']
mu2 = ball_stick_stick_fit.fitted_parameters['C1Stick_2_mu']
print(f"Fiber 1 orientation: {mu1}")
print(f"Fiber 2 orientation: {mu2}")
```

---

## NODDI Model

### Standard NODDI

```python
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel

# NODDI compartments
stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()
ball = gaussian_models.G1Ball()

# Watson distribution for orientation dispersion
watson_bundle = SD1WatsonDistributed(models=[stick, zeppelin])

# NODDI model: Ball + Watson(Stick + Zeppelin)
noddi = MultiCompartmentModel(models=[ball, watson_bundle])

# Set tortuous parameter (lambda_par = lambda_perp for zeppelin)
noddi.set_tortuous_parameter(
    'G2Zeppelin_1_lambda_perp',
    'C1Stick_1_lambda_par',
    'partial_volume_0',
    'partial_volume_1'
)

# Fit NODDI model
noddi_fit = noddi.fit(acq_scheme, voxel_data)

# Extract NODDI parameters
noddi_params = noddi_fit.fitted_parameters

# Neurite density index (NDI)
ndi = noddi_params.get('partial_volume_1', 0)

# Orientation dispersion index (ODI)
odi = noddi_params.get('SD1Watson_1_odi', 0)

# Free water fraction
fiso = noddi_params.get('partial_volume_0', 0)

print(f"NDI: {ndi:.3f}, ODI: {odi:.3f}, Fiso: {fiso:.3f}")
```

### NODDI Parameter Maps

```python
# Fit NODDI to whole brain
brain_mask = nib.load('brain_mask.nii.gz').get_fdata().astype(bool)
brain_data = data[brain_mask]

# Fit NODDI
noddi_fit_brain = noddi.fit(
    acq_scheme,
    brain_data
)

# Create parameter maps
ndi_map = np.zeros(data.shape[:3])
odi_map = np.zeros(data.shape[:3])
fiso_map = np.zeros(data.shape[:3])

# Fill maps with fitted parameters
fitted_params = noddi_fit_brain.fitted_parameters

ndi_map[brain_mask] = fitted_params.get('partial_volume_1', 0)
odi_map[brain_mask] = fitted_params.get('SD1Watson_1_odi', 0)
fiso_map[brain_mask] = fitted_params.get('partial_volume_0', 0)

# Save parameter maps
ndi_img = nib.Nifti1Image(ndi_map, img.affine)
nib.save(ndi_img, 'ndi_map.nii.gz')

odi_img = nib.Nifti1Image(odi_map, img.affine)
nib.save(odi_img, 'odi_map.nii.gz')

print("NODDI parameter maps saved")
```

---

## Custom Model Building

### Create Multi-Compartment Model

```python
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed

# Define compartments
cylinder = cylinder_models.C4CylinderGaussianPhaseApproximation()
zeppelin = gaussian_models.G2Zeppelin()
ball = gaussian_models.G1Ball()

# Distribute cylinder with Watson
watson_cylinders = SD1WatsonDistributed(models=[cylinder])

# Create 3-compartment model
# Intra-axonal (cylinder) + Extra-axonal (zeppelin) + CSF (ball)
custom_model = MultiCompartmentModel(
    models=[watson_cylinders, zeppelin, ball]
)

# Set parameter links
custom_model.set_equal_parameter(
    'G2Zeppelin_1_lambda_par',
    'C4CylinderGaussianPhaseApproximation_1_lambda_par'
)

# Fit custom model
custom_fit = custom_model.fit(acq_scheme, voxel_data)
print("Custom model fitted successfully")
```

### Add Parameter Constraints

```python
# Set fixed parameters
custom_model.set_fixed_parameter(
    'G1Ball_1_lambda_iso',
    3e-9  # Fix CSF diffusivity to 3 μm²/ms
)

# Set parameter bounds
custom_model.set_parameter_optimization_bounds(
    'C4CylinderGaussianPhaseApproximation_1_diameter',
    [0.1e-6, 20e-6]  # Diameter between 0.1 and 20 μm
)

# Fit with constraints
constrained_fit = custom_model.fit(acq_scheme, voxel_data)
```

---

## Spherical Mean Technique (SMT)

### SMT-NODDI (Rotation Invariant)

```python
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel

# Create SMT version of compartments (spherical mean)
stick_smt = cylinder_models.C1Stick()
zeppelin_smt = gaussian_models.G2Zeppelin()
ball_smt = gaussian_models.G1Ball()

# SMT multi-compartment model
smt_model = MultiCompartmentSphericalMeanModel(
    models=[stick_smt, zeppelin_smt, ball_smt]
)

# Fit SMT model (rotation invariant)
smt_fit = smt_model.fit(acq_scheme, voxel_data)

# Get SMT parameters (no orientation needed)
smt_params = smt_fit.fitted_parameters
print("SMT parameters:", smt_params)

# SMT is faster and doesn't require orientation estimation
```

### SMT for Gray Matter

```python
# SMT for gray matter (no dominant orientation)
from dmipy.signal_models import sphere_models

# Sphere for cell bodies
sphere = sphere_models.S2SphereStejskalTannerApproximation()
ball = gaussian_models.G1Ball()

# SMT gray matter model
gm_smt_model = MultiCompartmentSphericalMeanModel(
    models=[sphere, ball]
)

# Fit to gray matter voxel
gm_voxel = data[30, 40, 35]
gm_smt_fit = gm_smt_model.fit(acq_scheme, gm_voxel)

# Get sphere fraction (cellular density)
sphere_fraction = gm_smt_fit.fitted_parameters.get('partial_volume_0', 0)
print(f"Cellular fraction: {sphere_fraction:.3f}")
```

---

## Multi-Shell Optimization

### Optimize for Multi-Shell Data

```python
# Load multi-shell data
bvals_multishell = np.array([0, 1000, 2000, 3000])  # Example
bvecs_multishell = np.random.randn(100, 3)  # Example
bvecs_multishell /= np.linalg.norm(bvecs_multishell, axis=1, keepdims=True)

acq_scheme_multishell = acquisition_scheme_from_bvalues(
    bvals_multishell,
    bvecs_multishell
)

# NODDI benefits from multi-shell
noddi_multishell_fit = noddi.fit(
    acq_scheme_multishell,
    voxel_data
)

# Multi-shell improves parameter estimation
print("Multi-shell NODDI parameters:")
print(noddi_multishell_fit.fitted_parameters)
```

### Shell-Specific Processing

```python
# Analyze contribution of each shell
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues

# Separate shells
shells = {}
for bval in [1000, 2000, 3000]:
    shell_mask = np.abs(bvals_multishell - bval) < 100
    shells[bval] = acq_scheme_multishell[shell_mask]

# Fit using different shell combinations
for shell_combo in [[1000], [1000, 2000], [1000, 2000, 3000]]:
    # Create combined scheme
    masks = [np.abs(bvals_multishell - b) < 100 for b in shell_combo]
    combined_mask = np.any(masks, axis=0)

    scheme = acq_scheme_multishell[combined_mask]
    fit = noddi.fit(scheme, voxel_data)

    print(f"Shells {shell_combo}: ODI = {fit.fitted_parameters.get('SD1Watson_1_odi', 0):.3f}")
```

---

## Advanced Models

### CHARMED

```python
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed

# CHARMED: Composite Hindered and Restricted Model of Diffusion
cylinder_charmed = cylinder_models.C4CylinderGaussianPhaseApproximation()
zeppelin_charmed = gaussian_models.G2Zeppelin()
ball_charmed = gaussian_models.G1Ball()

# Watson distributed cylinders (restricted)
watson_restricted = SD1WatsonDistributed(models=[cylinder_charmed])

# CHARMED model
charmed = MultiCompartmentModel(
    models=[watson_restricted, zeppelin_charmed, ball_charmed]
)

# Fit CHARMED
charmed_fit = charmed.fit(acq_scheme, voxel_data)

# Extract axon diameter
diameter = charmed_fit.fitted_parameters.get(
    'C4CylinderGaussianPhaseApproximation_1_diameter',
    0
)
print(f"Estimated axon diameter: {diameter*1e6:.2f} μm")
```

### AxCaliber (Axon Diameter Distribution)

```python
from dmipy.distributions.distribute_models import SD1GammaDistributed

# Gamma-distributed cylinder diameters
cylinder_axcaliber = cylinder_models.C4CylinderGaussianPhaseApproximation()
gamma_cylinders = SD1GammaDistributed(models=[cylinder_axcaliber])

# AxCaliber model
axcaliber = MultiCompartmentModel(
    models=[gamma_cylinders, zeppelin, ball]
)

# Fit AxCaliber (requires strong gradients)
axcaliber_fit = axcaliber.fit(acq_scheme, voxel_data)

# Get diameter distribution parameters
alpha = axcaliber_fit.fitted_parameters.get('SD1Gamma_1_alpha', 0)
beta = axcaliber_fit.fitted_parameters.get('SD1Gamma_1_beta', 0)

# Mean diameter
mean_diameter = alpha * beta
print(f"Mean axon diameter: {mean_diameter*1e6:.2f} μm")
```

### ActiveAx

```python
# ActiveAx: Axon diameter imaging
from dmipy.signal_models import cylinder_models

# ActiveAx cylinder
cylinder_activeax = cylinder_models.C4CylinderGaussianPhaseApproximation()
zeppelin_activeax = gaussian_models.G2Zeppelin()

# Simple 2-compartment ActiveAx
activeax = MultiCompartmentModel(
    models=[cylinder_activeax, zeppelin_activeax]
)

# Link parallel diffusivities
activeax.set_equal_parameter(
    'G2Zeppelin_1_lambda_par',
    'C4CylinderGaussianPhaseApproximation_1_lambda_par'
)

# Fit ActiveAx
activeax_fit = activeax.fit(acq_scheme, voxel_data)
print("ActiveAx fitted")
```

---

## Optimization Strategies

### Choose Optimizer

```python
# Different optimization methods
from dmipy.core.modeling_framework import MultiCompartmentModel

# Create model
model = MultiCompartmentModel(models=[ball, stick])

# Use L-BFGS-B (default, fast)
fit_lbfgs = model.fit(
    acq_scheme,
    voxel_data,
    solver='brute2fine'
)

# Use differential evolution (global, slower)
fit_de = model.fit(
    acq_scheme,
    voxel_data,
    solver='mix'  # Combines methods
)

# Use MIX (recommended for complex models)
fit_mix = noddi.fit(
    acq_scheme,
    voxel_data,
    solver='mix'
)
```

### Cascade Fitting

```python
# Fit simple model first for initialization
ball_stick_fit = ball_stick.fit(acq_scheme, voxel_data)

# Use simple fit to initialize complex model
initial_params = {}
initial_params['C1Stick_1_mu'] = ball_stick_fit.fitted_parameters['C1Stick_1_mu']

# Fit NODDI with initialization
noddi_fit_initialized = noddi.fit(
    acq_scheme,
    voxel_data,
    x0_vector=initial_params  # Initial guess
)
```

---

## Parameter Maps and Visualization

### Generate All Parameter Maps

```python
import matplotlib.pyplot as plt

# Fit model to masked brain
brain_mask = nib.load('brain_mask.nii.gz').get_fdata().astype(bool)
brain_data = data[brain_mask]

noddi_fit_brain = noddi.fit(acq_scheme, brain_data)

# Extract all parameters
param_names = noddi.parameter_names
param_maps = {}

for param_name in param_names:
    param_map = np.zeros(data.shape[:3])
    param_map[brain_mask] = noddi_fit_brain.fitted_parameters.get(param_name, 0)
    param_maps[param_name] = param_map

# Save all maps
for param_name, param_map in param_maps.items():
    img_out = nib.Nifti1Image(param_map, img.affine)
    nib.save(img_out, f'{param_name}_map.nii.gz')

print(f"Saved {len(param_maps)} parameter maps")
```

### Visualize Parameter Maps

```python
# Visualize NODDI parameters
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# NDI
axes[0].imshow(ndi_map[:, :, 35].T, cmap='hot', vmin=0, vmax=1, origin='lower')
axes[0].set_title('Neurite Density Index (NDI)')
axes[0].axis('off')

# ODI
axes[1].imshow(odi_map[:, :, 35].T, cmap='viridis', vmin=0, vmax=1, origin='lower')
axes[1].set_title('Orientation Dispersion Index (ODI)')
axes[1].axis('off')

# Free water fraction
axes[2].imshow(fiso_map[:, :, 35].T, cmap='Blues', vmin=0, vmax=1, origin='lower')
axes[2].set_title('Free Water Fraction')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('noddi_parameters.png', dpi=300)
plt.show()
```

---

## Model Comparison

### Compare Models Statistically

```python
# Fit multiple models
ball_stick_fit = ball_stick.fit(acq_scheme, voxel_data)
noddi_fit = noddi.fit(acq_scheme, voxel_data)

# Get model predictions
ball_stick_pred = ball_stick.simulate_signal(
    acq_scheme,
    ball_stick_fit.fitted_parameters
)

noddi_pred = noddi.simulate_signal(
    acq_scheme,
    noddi_fit.fitted_parameters
)

# Calculate residuals
residuals_ball_stick = voxel_data - ball_stick_pred
residuals_noddi = voxel_data - noddi_pred

# Compare fit quality
mse_ball_stick = np.mean(residuals_ball_stick**2)
mse_noddi = np.mean(residuals_noddi**2)

print(f"Ball-Stick MSE: {mse_ball_stick:.6f}")
print(f"NODDI MSE: {mse_noddi:.6f}")

# NODDI should fit better for complex tissue
```

### AIC/BIC Model Selection

```python
# Calculate AIC (Akaike Information Criterion)
def calculate_aic(model, fit, data, n_params):
    """Calculate AIC for model."""
    predicted = model.simulate_signal(acq_scheme, fit.fitted_parameters)
    residuals = data - predicted
    rss = np.sum(residuals**2)
    n = len(data)

    aic = n * np.log(rss / n) + 2 * n_params
    return aic

# Compare models
aic_ball_stick = calculate_aic(
    ball_stick,
    ball_stick_fit,
    voxel_data,
    n_params=len(ball_stick.parameter_names)
)

aic_noddi = calculate_aic(
    noddi,
    noddi_fit,
    voxel_data,
    n_params=len(noddi.parameter_names)
)

print(f"Ball-Stick AIC: {aic_ball_stick:.2f}")
print(f"NODDI AIC: {aic_noddi:.2f}")
print(f"Preferred model: {'NODDI' if aic_noddi < aic_ball_stick else 'Ball-Stick'}")
```

---

## Integration with Claude Code

DMIPY integrates seamlessly with automated pipelines:

```python
# dmipy_pipeline.py - Automated microstructure analysis

import numpy as np
import nibabel as nib
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel
from pathlib import Path

class DMIPYPipeline:
    """Automated DMIPY microstructure analysis."""

    def __init__(self, model_type='noddi'):
        self.model_type = model_type
        self.model = self._build_model()

    def _build_model(self):
        """Build specified model."""
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()
        ball = gaussian_models.G1Ball()

        if self.model_type == 'noddi':
            watson_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
            model = MultiCompartmentModel(models=[ball, watson_bundle])

            # Set tortuous parameter
            model.set_tortuous_parameter(
                'G2Zeppelin_1_lambda_perp',
                'C1Stick_1_lambda_par',
                'partial_volume_0',
                'partial_volume_1'
            )
            return model

        elif self.model_type == 'ball_stick':
            return MultiCompartmentModel(models=[ball, stick])

        else:
            raise ValueError(f"Unknown model: {self.model_type}")

    def process_subject(self, dwi_file, bval_file, bvec_file, mask_file, output_dir):
        """Process single subject."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print(f"Loading {dwi_file}...")
        img = nib.load(dwi_file)
        data = img.get_fdata()

        bvals = np.loadtxt(bval_file)
        bvecs = np.loadtxt(bvec_file).T

        mask = nib.load(mask_file).get_fdata().astype(bool)

        # Create acquisition scheme
        acq_scheme = acquisition_scheme_from_bvalues(bvals, bvecs)

        # Fit model
        print(f"Fitting {self.model_type} model...")
        masked_data = data[mask]
        fit = self.model.fit(acq_scheme, masked_data)

        # Extract parameter maps
        print("Generating parameter maps...")
        param_maps = {}

        for param_name in self.model.parameter_names:
            param_map = np.zeros(data.shape[:3])
            param_map[mask] = fit.fitted_parameters.get(param_name, 0)
            param_maps[param_name] = param_map

        # Save parameter maps
        for param_name, param_map in param_maps.items():
            output_file = output_dir / f'{param_name}.nii.gz'
            nib.save(nib.Nifti1Image(param_map, img.affine), output_file)
            print(f"  Saved: {output_file}")

        return param_maps

# Usage
pipeline = DMIPYPipeline(model_type='noddi')

# Process subject
param_maps = pipeline.process_subject(
    dwi_file='/data/sub-01/dwi/dwi.nii.gz',
    bval_file='/data/sub-01/dwi/dwi.bval',
    bvec_file='/data/sub-01/dwi/dwi.bvec',
    mask_file='/data/sub-01/dwi/brain_mask.nii.gz',
    output_dir='/data/derivatives/dmipy/sub-01'
)
```

**Batch Processing:**

```python
# batch_dmipy.py
from pathlib import Path

class BatchDMIPY:
    """Batch process multiple subjects."""

    def __init__(self, bids_dir, output_dir, model='noddi'):
        self.bids_dir = Path(bids_dir)
        self.output_dir = Path(output_dir)
        self.pipeline = DMIPYPipeline(model_type=model)

    def process_all_subjects(self):
        """Process all subjects in BIDS directory."""

        subjects = sorted(self.bids_dir.glob('sub-*'))

        for subject_dir in subjects:
            subject_id = subject_dir.name
            print(f"\nProcessing {subject_id}...")

            # Find DWI files
            dwi_file = subject_dir / 'dwi' / f'{subject_id}_dwi.nii.gz'
            bval_file = subject_dir / 'dwi' / f'{subject_id}_dwi.bval'
            bvec_file = subject_dir / 'dwi' / f'{subject_id}_dwi.bvec'
            mask_file = subject_dir / 'dwi' / f'{subject_id}_brain_mask.nii.gz'

            if not all([f.exists() for f in [dwi_file, bval_file, bvec_file, mask_file]]):
                print(f"  Skipping {subject_id}: missing files")
                continue

            # Process
            try:
                self.pipeline.process_subject(
                    dwi_file, bval_file, bvec_file, mask_file,
                    self.output_dir / subject_id
                )
            except Exception as e:
                print(f"  Error processing {subject_id}: {e}")

# Run batch processing
batch = BatchDMIPY(
    bids_dir='/data/raw',
    output_dir='/data/derivatives/dmipy',
    model='noddi'
)

batch.process_all_subjects()
```

---

## Integration with Other Tools

### DIPY Integration

```python
# Use DIPY for preprocessing, DMIPY for modeling
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti

# Load with DIPY
data, affine = load_nifti('dwi.nii.gz')
bvals = np.loadtxt('dwi.bval')
bvecs = np.loadtxt('dwi.bvec')

# DIPY gradient table
gtab = gradient_table(bvals, bvecs)

# Convert to DMIPY acquisition scheme
acq_scheme = acquisition_scheme_from_bvalues(gtab.bvals, gtab.bvecs)

# Fit DMIPY model
noddi_fit = noddi.fit(acq_scheme, data[50, 50, 35])
```

### QSIPrep Integration

```python
# Process QSIPrep preprocessed data
from pathlib import Path

def process_qsiprep_output(qsiprep_dir, subject):
    """Process QSIPrep preprocessed DWI."""

    subj_dir = Path(qsiprep_dir) / subject / 'dwi'

    # Load preprocessed DWI
    dwi_file = subj_dir / f'{subject}_space-T1w_desc-preproc_dwi.nii.gz'
    bval_file = subj_dir / f'{subject}_space-T1w_desc-preproc_dwi.bval'
    bvec_file = subj_dir / f'{subject}_space-T1w_desc-preproc_dwi.bvec'
    mask_file = subj_dir / f'{subject}_space-T1w_desc-brain_mask.nii.gz'

    # Load
    img = nib.load(dwi_file)
    data = img.get_fdata()
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file).T
    mask = nib.load(mask_file).get_fdata().astype(bool)

    # Create acquisition scheme
    acq_scheme = acquisition_scheme_from_bvalues(bvals, bvecs)

    # Fit NODDI
    masked_data = data[mask]
    noddi_fit = noddi.fit(acq_scheme, masked_data)

    return noddi_fit

# Process QSIPrep subject
fit = process_qsiprep_output('/data/derivatives/qsiprep', 'sub-01')
```

### MRtrix3 Integration

```python
# Load MRtrix3 preprocessed data
import subprocess

def load_mrtrix_dwi(mif_file):
    """Convert MRtrix MIF to NIFTI and load."""

    # Convert MIF to NIFTI
    subprocess.run([
        'mrconvert', mif_file, 'dwi_temp.nii.gz',
        '-export_grad_fsl', 'bvec_temp', 'bval_temp'
    ])

    # Load converted data
    img = nib.load('dwi_temp.nii.gz')
    data = img.get_fdata()
    bvals = np.loadtxt('bval_temp')
    bvecs = np.loadtxt('bvec_temp').T

    acq_scheme = acquisition_scheme_from_bvalues(bvals, bvecs)

    return data, acq_scheme, img.affine

# Use with DMIPY
data, acq_scheme, affine = load_mrtrix_dwi('dwi_preproc.mif')
```

---

## Troubleshooting

### Problem 1: Optimization Fails

**Symptoms:** Fitting returns unrealistic parameters

**Solution:**
```python
# Use more robust optimizer
fit = noddi.fit(acq_scheme, voxel_data, solver='mix')

# Or add parameter bounds
noddi.set_parameter_optimization_bounds(
    'SD1Watson_1_odi',
    [0, 1]  # ODI between 0 and 1
)

# Check data quality
signal_to_noise = np.mean(voxel_data[bvals < 100]) / np.std(voxel_data[bvals < 100])
print(f"SNR: {signal_to_noise:.1f}")  # Should be > 10
```

### Problem 2: Slow Fitting

**Symptoms:** Model fitting takes very long

**Solution:**
```python
# Use simpler model for initialization
ball_stick_fit = ball_stick.fit(acq_scheme, voxel_data)

# Use SMT (faster, no orientation)
smt_fit = smt_model.fit(acq_scheme, voxel_data)

# Reduce number of parameters
# Use fewer compartments

# Install numba for acceleration
# pip install numba
```

### Problem 3: Memory Issues

**Symptoms:** Out of memory with whole-brain fitting

**Solution:**
```python
# Process in chunks
chunk_size = 1000
n_voxels = np.sum(brain_mask)

for i in range(0, n_voxels, chunk_size):
    chunk_data = masked_data[i:i+chunk_size]
    chunk_fit = noddi.fit(acq_scheme, chunk_data)

    # Store results
    # ...
```

### Problem 4: Parameter Interpretation

**Symptoms:** Unclear what parameters mean

**Solution:**
```python
# Print parameter descriptions
for param in noddi.parameter_names:
    print(f"{param}: {noddi.parameter_descriptions.get(param, 'N/A')}")

# Check parameter ranges
print(noddi.parameter_ranges)

# Consult documentation for biophysical meaning
```

---

## Best Practices

### 1. Data Quality

- **SNR > 10:** Ensure sufficient signal-to-noise ratio
- **Multi-shell:** Use multiple b-values for advanced models (NODDI, CHARMED)
- **Preprocessing:** Motion correction, eddy current correction essential
- **Brain mask:** Use accurate brain extraction

### 2. Model Selection

- **Start simple:** Begin with Ball-Stick, progress to NODDI
- **Match data:** Ensure model complexity matches acquisition
- **Validate:** Check fit quality, compare models statistically
- **Biological plausibility:** Ensure parameters make sense

### 3. Parameter Estimation

- **Use MIX solver:** For complex models (NODDI, CHARMED)
- **Set bounds:** Constrain parameters to reasonable ranges
- **Initialize wisely:** Use cascade fitting for complex models
- **Check convergence:** Verify optimization succeeded

### 4. Computational Efficiency

- **Use SMT:** When orientation not needed (faster)
- **Batch processing:** Process multiple voxels together
- **Numba:** Install for JIT compilation speed-up
- **Simplify:** Use minimal compartments needed

### 5. Reproducibility

- **Version control:** Record DMIPY version
- **Save parameters:** Store all model parameters
- **Document acquisition:** Record b-values, directions
- **Seed random:** For consistency in optimization

---

## Resources

### Official Documentation

- **Documentation:** https://dmipy.readthedocs.io/
- **GitHub:** https://github.com/AthenaEPI/dmipy
- **Tutorials:** https://dmipy.readthedocs.io/en/latest/examples.html
- **API Reference:** https://dmipy.readthedocs.io/en/latest/api.html

### Publications

- **DMIPY Paper:** Fick et al. (2019) "DMIPY: An open-source framework for reproducible dMRI-based microstructure research" *Frontiers in Neuroinformatics*
- **NODDI:** Zhang et al. (2012) "NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain"
- **CHARMED:** Assaf & Basser (2005) "Composite hindered and restricted model of diffusion"

### Community Support

- **GitHub Issues:** https://github.com/AthenaEPI/dmipy/issues
- **Gitter Chat:** https://gitter.im/AthenaEPI/dmipy

---

## Citation

```bibtex
@article{fick2019dmipy,
  title={DMIPY: An open-source framework for reproducible dMRI-based microstructure research},
  author={Fick, Rutger HJ and Wassermann, Demian and Deriche, Rachid},
  journal={Frontiers in Neuroinformatics},
  volume={13},
  pages={64},
  year={2019},
  publisher={Frontiers},
  doi={10.3389/fninf.2019.00064}
}
```

---

## Related Tools

- **MDT:** GPU-accelerated microstructure toolkit (see `mdt.md`)
- **DIPY:** Foundation for diffusion processing (see `dipy.md`)
- **MRtrix3:** Preprocessing and response functions (see `mrtrix3.md`)
- **QSIPrep:** Preprocessing pipeline (see `qsiprep.md`)
- **DSI Studio:** Alternative diffusion analysis (see `dsistudio.md`)
- **NODDI MATLAB Toolbox:** Original NODDI implementation
- **Camino:** Java diffusion toolkit
- **Recobundles:** Bundle extraction (see `recobundles.md`)

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**DMIPY Version Covered:** 1.0.x
**Maintainer:** Claude Code Neuroimaging Skills
