# DIPY (Diffusion Imaging in Python)

## Overview

DIPY is a free and open-source Python library for computational neuroanatomy, focusing on diffusion MRI analysis. It provides state-of-the-art algorithms for diffusion MRI reconstruction, tractography, registration, and visualization, all accessible through a clean Python API.

**Website:** https://dipy.org/
**Platform:** Cross-platform (Linux/macOS/Windows)
**Language:** Python (with Cython optimizations)
**License:** BSD 3-Clause

## Key Features

- Multi-shell diffusion reconstruction (DTI, DKI, NODDI, MAPMRI)
- Deterministic and probabilistic tractography
- Advanced tractography methods (PFT, particle filtering)
- Bundle segmentation and recognition
- Image registration and resampling
- Denoising and artifact correction
- Machine learning for tissue microstructure
- Visualization tools (FURY)
- DICOM and NIfTI support

## Installation

### Using pip

```bash
# Basic installation
pip install dipy

# With all dependencies
pip install dipy[all]

# Development version
pip install git+https://github.com/dipy/dipy.git
```

### Using conda

```bash
conda install -c conda-forge dipy
```

### Verify Installation

```python
import dipy
print(dipy.__version__)
```

## Data Loading

```python
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

# Load DWI data
data, affine = load_nifti('dwi.nii.gz')
bvals, bvecs = read_bvals_bvecs('bvals', 'bvecs')

# Create gradient table
gtab = gradient_table(bvals, bvecs)

# Check data shape
print(f"Data shape: {data.shape}")
print(f"Number of b0 volumes: {np.sum(gtab.b0s_mask)}")
print(f"Number of gradients: {len(gtab.bvals)}")
```

## Preprocessing

### Denoising

```python
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate

# Estimate noise
sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=3)

# Denoise with local PCA
denoised_data = localpca(data, sigma=sigma, patch_radius=2)

# Save denoised data
save_nifti('dwi_denoised.nii.gz', denoised_data, affine)
```

### Gibbs Ringing Removal

```python
from dipy.denoise.gibbs import gibbs_removal

# Remove Gibbs ringing artifacts
data_unring = gibbs_removal(data, slice_axis=2)
```

### Brain Masking

```python
from dipy.segment.mask import median_otsu

# Create brain mask
masked_data, mask = median_otsu(
    data,
    vol_idx=range(10, 50),  # Use subset of volumes
    median_radius=4,
    numpass=4,
    autocrop=False,
    dilate=2
)

# Save mask
save_nifti('mask.nii.gz', mask.astype(np.uint8), affine)
```

## Diffusion Tensor Imaging (DTI)

```python
from dipy.reconst.dti import TensorModel, fractional_anisotropy, color_fa

# Fit DTI model
tensor_model = TensorModel(gtab)
tensor_fit = tensor_model.fit(data, mask=mask)

# Compute DTI metrics
FA = fractional_anisotropy(tensor_fit.evals)
MD = tensor_fit.md
AD = tensor_fit.ad
RD = tensor_fit.rd

# Directionally-encoded color FA
RGB = color_fa(FA, tensor_fit.evecs)

# Save metrics
save_nifti('fa.nii.gz', FA, affine)
save_nifti('md.nii.gz', MD, affine)
save_nifti('color_fa.nii.gz', RGB, affine)

# Get eigenvectors for tractography
evecs = tensor_fit.evecs
```

## Advanced Reconstruction

### Constrained Spherical Deconvolution (CSD)

```python
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                    auto_response_ssst)
from dipy.data import default_sphere

# Estimate response function
response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

# Fit CSD model
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
csd_fit = csd_model.fit(data, mask=mask)

# Get FOD
fod = csd_fit.odf(default_sphere)
```

### Multi-Shell Multi-Tissue CSD (MSMT-CSD)

```python
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response

# Estimate multi-tissue response
response_wm, response_gm, response_csf = multi_shell_fiber_response(
    gtab, data, mask=mask, fa_thr=0.7
)

# MSMT-CSD model
mcsd_model = MultiShellDeconvModel(gtab, response_wm, response_gm, response_csf)
mcsd_fit = mcsd_model.fit(data, mask=mask)

# Extract tissue compartments
wm_fod = mcsd_fit.shm_coeff
```

### Diffusion Kurtosis Imaging (DKI)

```python
from dipy.reconst.dki import DiffusionKurtosisModel

# Fit DKI model (requires multi-shell data)
dki_model = DiffusionKurtosisModel(gtab)
dki_fit = dki_model.fit(data, mask=mask)

# Get DKI metrics
kfa = dki_fit.kfa
mk = dki_fit.mk()
ak = dki_fit.ak()
rk = dki_fit.rk()

# Get DTI metrics from DKI
fa_dki = dki_fit.fa
md_dki = dki_fit.md
```

## Tractography

### Deterministic Tractography

```python
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.direction import peaks_from_model

# Get peaks for tracking
peaks = peaks_from_model(
    csd_model, data, default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    mask=mask
)

# Define stopping criterion
stopping_criterion = ThresholdStoppingCriterion(peaks.gfa, 0.25)

# Generate seeds (2 seeds per voxel)
seeds = utils.seeds_from_mask(mask, affine, density=2)

# Perform tracking
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines

streamline_generator = LocalTracking(
    peaks,
    stopping_criterion,
    seeds,
    affine,
    step_size=0.5,
    max_cross=1
)

streamlines = Streamlines(streamline_generator)
print(f"Generated {len(streamlines)} streamlines")
```

### Probabilistic Tractography

```python
from dipy.direction import ProbabilisticDirectionGetter

# Probabilistic direction getter
prob_dg = ProbabilisticDirectionGetter.from_shcoeff(
    csd_fit.shm_coeff,
    max_angle=30.0,
    sphere=default_sphere
)

# Probabilistic tracking
prob_streamline_generator = LocalTracking(
    prob_dg,
    stopping_criterion,
    seeds,
    affine,
    step_size=0.5,
    max_cross=1
)

prob_streamlines = Streamlines(prob_streamline_generator)
```

### Particle Filtering Tractography

```python
from dipy.tracking.local_tracking import ParticleFilteringTracking

# PFT tracking
pft_streamline_generator = ParticleFilteringTracking(
    prob_dg,
    stopping_criterion,
    seeds,
    affine,
    max_cross=1,
    step_size=0.5,
    maxlen=1000,
    pft_back_tracking_dist=2,
    pft_front_tracking_dist=1,
    particle_count=15
)

pft_streamlines = Streamlines(pft_streamline_generator)
```

### Save Tractography

```python
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram

# Create tractogram object
sft = StatefulTractogram(streamlines, 'dwi.nii.gz', Space.RASMM)

# Save as TRK or TCK
save_tractogram(sft, 'tractography.trk')
save_tractogram(sft, 'tractography.tck')
```

## Bundle Segmentation

```python
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points

# Load atlas bundles (example)
from dipy.data import fetch_bundle_atlas_hcp842
atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

# Load atlas streamlines
from dipy.io.streamline import load_tractogram
atlas_bundle_sft = load_tractogram(atlas_file, 'same')
atlas_bundle = atlas_bundle_sft.streamlines

# RecoBundles segmentation
rb = RecoBundles(streamlines, clust_thr=10)
recognized_bundle, labels = rb.recognize(
    model_bundle=atlas_bundle,
    model_clust_thr=5,
    reduction_thr=10,
    pruning_thr=8
)

print(f"Recognized {len(recognized_bundle)} streamlines")
```

## Registration

```python
from dipy.align.imaffine import (AffineMap, MutualInformationMetric,
                                  AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                    RigidTransform3D,
                                    AffineTransform3D)

# Load images
static, static_affine = load_nifti('template.nii.gz')
moving, moving_affine = load_nifti('subject.nii.gz')

# Setup registration
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

# Multi-resolution settings
level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]

# Rigid registration
affreg = AffineRegistration(
    metric=metric,
    level_iters=level_iters,
    sigmas=sigmas,
    factors=factors
)

transform = RigidTransform3D()
rigid = affreg.optimize(
    static, moving, transform, None,
    static_affine, moving_affine
)

# Apply transformation
transformed = rigid.transform(moving)
save_nifti('registered.nii.gz', transformed, static_affine)
```

## Visualization

```python
from dipy.viz import window, actor

# Create scene
scene = window.Scene()

# Add streamlines
streamlines_actor = actor.line(streamlines)
scene.add(streamlines_actor)

# Add FA as background
from dipy.viz import actor
fa_actor = actor.slicer(FA, affine)
scene.add(fa_actor)

# Show interactive window
window.show(scene, size=(800, 800))

# Save screenshot
window.record(scene, out_path='tractography.png', size=(800, 800))
```

## Integration with Claude Code

When helping users with DIPY:

1. **Check Installation:**
   ```python
   import dipy
   print(dipy.__version__)
   ```

2. **Data Format:** Ensure proper gradient table (bvals/bvecs)

3. **Memory Management:** Large datasets may need chunking

4. **Common Issues:**
   - Gradient table orientation mismatches
   - Missing or incorrect b-values
   - Memory errors with whole-brain tractography
   - Affine matrix inconsistencies

5. **Performance:** Use Cython-compiled functions for speed

## Best Practices

- Always denoise data before analysis
- Verify gradient table correctness
- Use appropriate reconstruction for your data (single vs multi-shell)
- Apply brain masking to reduce computation
- Check tractography results visually
- Use SIFT/SIFT2 for quantitative analysis
- Save intermediate results
- Document analysis parameters

## Troubleshooting

**Problem:** "No module named 'dipy'"
**Solution:** Install with `pip install dipy`

**Problem:** Slow tractography
**Solution:** Reduce number of seeds, use coarser step size, enable parallel processing

**Problem:** Poor bundle recognition
**Solution:** Adjust clustering thresholds, ensure proper registration

## Resources

- DIPY Documentation: https://dipy.org/documentation/
- Tutorials: https://dipy.org/tutorials/
- Gallery: https://dipy.org/gallery/
- GitHub: https://github.com/dipy/dipy
- Forum: https://github.com/dipy/dipy/discussions

## Related Tools

- **FURY:** 3D visualization library
- **CVXPY:** Optimization for DIPY
- **Nibabel:** For neuroimaging file I/O
- **MRtrix3:** Complementary diffusion toolkit

## Citation

```bibtex
@article{garyfallidis2014dipy,
  title={DIPY, a library for the analysis of diffusion MRI data},
  author={Garyfallidis, Eleftherios and Brett, Matthew and Amirbekian, Bagrat and others},
  journal={Frontiers in Neuroinformatics},
  volume={8},
  pages={8},
  year={2014},
  doi={10.3389/fninf.2014.00008}
}
```
