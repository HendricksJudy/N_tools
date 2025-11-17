# BrainSpace - Gradient Analysis and Manifold Learning

## Overview

**BrainSpace** is a comprehensive toolbox for identifying and analyzing gradients of brain organization using manifold learning techniques. Developed at the Montreal Neurological Institute (MNI), BrainSpace reveals smooth, continuous transitions in brain connectivity, microstructure, gene expression, and other features, moving beyond traditional discrete parcellation approaches. By applying dimensionality reduction to high-dimensional brain data, BrainSpace uncovers the low-dimensional organizational principles that govern brain structure and function.

BrainSpace implements multiple manifold learning algorithms including diffusion map embedding, Laplacian eigenmaps, and PCA, along with powerful alignment techniques (Procrustes analysis) for comparing gradients across individuals, modalities, and species. The toolbox provides comprehensive statistical testing via spatial permutation (spin tests) and integrates seamlessly with standard neuroimaging formats and surfaces.

**Key Features:**
- Gradient computation using manifold learning (diffusion maps, Laplacian eigenmaps, PCA)
- Multiple kernel functions for similarity matrices
- Gradient alignment across individuals using Procrustes analysis
- Cross-modal gradient alignment (e.g., structural vs. functional connectivity)
- Spatial permutation testing (spin tests) for gradient associations
- Null model generation preserving spatial autocorrelation
- Surface-based and volumetric gradient analysis
- Visualization on cortical surfaces (fsaverage, fslr)
- Integration with connectivity matrices, morphometry, gene expression
- Python and MATLAB implementations
- Support for FreeSurfer, GIFTI, CIFTI formats

**Primary Use Cases:**
- Connectivity gradient analysis from resting-state fMRI
- Multi-modal integration (structure-function-genetics)
- Individual differences in brain organization
- Hierarchical brain organization studies
- Cross-species comparative neuroanatomy
- Clinical alterations in gradient organization
- Developmental and aging trajectory analysis

**Official Documentation:** https://brainspace.readthedocs.io/

---

## Installation

### Python Installation

```bash
# Install via pip
pip install brainspace

# Or install from GitHub for latest version
pip install git+https://github.com/MICA-MNI/BrainSpace.git

# Install with visualization dependencies
pip install brainspace[plotting]

# Verify installation
python -c "import brainspace; print(brainspace.__version__)"
```

### MATLAB Installation

```matlab
% Download BrainSpace for MATLAB
% Visit: https://github.com/MICA-MNI/BrainSpace/releases

% Add to MATLAB path
addpath('/path/to/BrainSpace/matlab');
addpath(genpath('/path/to/BrainSpace/matlab'));

% Verify installation
help GradientMaps
```

### Install Dependencies

```bash
# Python dependencies
pip install numpy scipy scikit-learn nibabel vtk matplotlib

# Optional: for advanced visualization
pip install pyvista panel

# For surface processing
pip install trimesh
```

### Download Example Data

```python
from brainspace.datasets import load_conte69, load_gradient

# Load Conte69 surface template
surf_lh, surf_rh = load_conte69()

# Load example connectivity gradient
gradient_data = load_gradient('fc')

print("Example data loaded successfully")
```

---

## Basic Gradient Analysis

### Compute Gradients from Connectivity Matrix

```python
import numpy as np
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_conte69
import matplotlib.pyplot as plt

# Load example connectivity matrix (400x400 parcels)
# In practice, compute from resting-state fMRI
connectivity = np.corrcoef(np.random.randn(400, 100))  # Example: 400 regions, 100 timepoints

# Initialize GradientMaps with diffusion embedding
gm = GradientMaps(
    n_components=10,           # Number of gradients to compute
    approach='dm',             # Diffusion map embedding
    kernel='normalized_angle'  # Similarity kernel
)

# Fit to connectivity matrix
gm.fit(connectivity)

# Access gradients
gradients = gm.gradients_  # Shape: (400, 10)
lambdas = gm.lambdas_      # Eigenvalues

print(f"Computed {gradients.shape[1]} gradients")
print(f"First 5 eigenvalues: {lambdas[:5]}")

# Visualize gradient weights (scree plot)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), lambdas[:10], 'o-')
plt.xlabel('Gradient')
plt.ylabel('Eigenvalue')
plt.title('Gradient Eigenvalues')
plt.grid(True)
plt.savefig('gradient_eigenvalues.png', dpi=300)
```

### Visualize Gradients on Cortical Surface

```python
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels
import numpy as np

# Load surfaces
surf_lh, surf_rh = load_conte69()

# Load parcellation (e.g., Schaefer 400)
from brainspace.datasets import load_parcellation
labeling = load_parcellation('schaefer', scale=400, join=True)

# Map gradient values to surface vertices
# gradient1 has shape (400,) - one value per parcel
gradient1 = gm.gradients_[:, 0]

# Map parcels to surface vertices
gradient_lh = map_to_labels(
    gradient1[:200],  # Left hemisphere parcels
    labeling['lh'],
    mask=labeling['lh'] != 0,
    fill=np.nan
)

gradient_rh = map_to_labels(
    gradient1[200:],  # Right hemisphere parcels
    labeling['rh'],
    mask=labeling['rh'] != 0,
    fill=np.nan
)

# Plot on surface
plot_hemispheres(
    surf_lh, surf_rh,
    array_name=[gradient_lh, gradient_rh],
    size=(800, 200),
    cmap='viridis',
    color_bar=True,
    label_text=['Gradient 1'],
    zoom=1.25
)
```

---

## Manifold Learning Approaches

### Diffusion Map Embedding

```python
from brainspace.gradient import GradientMaps

# Diffusion map embedding (default, recommended)
gm_dm = GradientMaps(
    n_components=10,
    approach='dm',              # Diffusion map
    kernel='normalized_angle',  # Cosine similarity
    random_state=0
)

gm_dm.fit(connectivity)
gradients_dm = gm_dm.gradients_

# Diffusion maps reveal hierarchical organization
# Gradient 1: typically sensorimotor-to-transmodal axis
# Gradient 2: visual-to-other sensory-to-association
```

### Laplacian Eigenmaps

```python
# Laplacian eigenmap embedding
gm_le = GradientMaps(
    n_components=10,
    approach='le',  # Laplacian eigenmaps
    kernel='normalized_angle'
)

gm_le.fit(connectivity)
gradients_le = gm_le.gradients_

# Laplacian eigenmaps: spectral graph theory approach
# Similar to diffusion maps but different normalization
```

### Principal Component Analysis (PCA)

```python
# PCA approach (linear dimensionality reduction)
gm_pca = GradientMaps(
    n_components=10,
    approach='pca',  # Principal component analysis
    kernel=None      # No kernel for PCA
)

gm_pca.fit(connectivity)
gradients_pca = gm_pca.gradients_

# PCA: simple linear approach
# Less suited for non-linear manifolds
```

### Compare Approaches

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot first gradient from each approach
approaches = [
    ('Diffusion Map', gradients_dm[:, 0]),
    ('Laplacian Eigenmaps', gradients_le[:, 0]),
    ('PCA', gradients_pca[:, 0])
]

for ax, (name, grad) in zip(axes, approaches):
    ax.scatter(range(len(grad)), grad, alpha=0.6, s=10)
    ax.set_title(name)
    ax.set_xlabel('Region')
    ax.set_ylabel('Gradient Value')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('approach_comparison.png', dpi=300)
```

---

## Kernel Functions

### Normalized Angle Kernel (Cosine Similarity)

```python
# Normalized angle kernel (recommended for functional connectivity)
gm = GradientMaps(
    n_components=5,
    approach='dm',
    kernel='normalized_angle'  # Cosine similarity
)

gm.fit(connectivity)

# Good for correlation matrices
# Insensitive to scaling
```

### Gaussian Kernel

```python
# Gaussian (RBF) kernel
gm_gauss = GradientMaps(
    n_components=5,
    approach='dm',
    kernel='gaussian',
    gamma=1.0  # Kernel width parameter
)

gm_gauss.fit(connectivity)

# Suitable for distance matrices
# Gamma controls locality of relationships
```

### Pearson Correlation Kernel

```python
# Pearson correlation as kernel
gm_pearson = GradientMaps(
    n_components=5,
    approach='dm',
    kernel='pearson'
)

gm_pearson.fit(connectivity)

# Direct correlation as similarity
```

### Custom Kernel Function

```python
# Define custom kernel
def custom_kernel(X):
    """Custom similarity function"""
    from scipy.spatial.distance import pdist, squareform

    # Compute pairwise distances
    distances = squareform(pdist(X, metric='euclidean'))

    # Convert to similarity
    similarity = np.exp(-distances**2 / (2 * 1.0**2))

    return similarity

# Use custom kernel
gm_custom = GradientMaps(
    n_components=5,
    approach='dm',
    kernel=custom_kernel
)

gm_custom.fit(connectivity)
```

---

## Gradient Alignment Across Subjects

### Procrustes Alignment

```python
from brainspace.gradient import GradientMaps
import numpy as np

# Simulate multiple subjects' connectivity matrices
n_subjects = 20
n_regions = 400
connectivities = [np.corrcoef(np.random.randn(n_regions, 100))
                  for _ in range(n_subjects)]

# Compute gradients for each subject
gradients_list = []
for conn in connectivities:
    gm = GradientMaps(n_components=10, approach='dm', kernel='normalized_angle')
    gm.fit(conn)
    gradients_list.append(gm.gradients_)

# Align all subjects to first subject (reference)
from brainspace.gradient import ProcrustesAlignment

# Initialize aligner
pa = ProcrustesAlignment(n_iter=10)

# Align gradients
gradients_aligned = pa.fit_transform(gradients_list)

print(f"Aligned {len(gradients_aligned)} subjects")
print(f"Gradient shape per subject: {gradients_aligned[0].shape}")

# Now gradients are in common space for group analysis
```

### Compute Group-Average Gradient

```python
import numpy as np

# After alignment, compute mean gradient across subjects
mean_gradient = np.mean(gradients_aligned, axis=0)

# Standard deviation across subjects
std_gradient = np.std(gradients_aligned, axis=0)

print(f"Group mean gradient shape: {mean_gradient.shape}")

# Visualize group-average gradient
gradient1_mean = mean_gradient[:, 0]
```

### Individual Differences Analysis

```python
# Compute individual deviations from group mean
individual_deviations = []

for subj_grad in gradients_aligned:
    deviation = subj_grad - mean_gradient
    individual_deviations.append(deviation)

individual_deviations = np.array(individual_deviations)

# Correlate individual gradient with behavior
# Example: gradient 1 vs. cognitive score
cognitive_scores = np.random.randn(n_subjects)  # Example scores

gradient1_values = [grad[:, 0] for grad in gradients_aligned]

# For each region, correlate across subjects
from scipy.stats import pearsonr

correlations = []
for region in range(n_regions):
    region_gradients = [grad1[region] for grad1 in gradient1_values]
    r, p = pearsonr(region_gradients, cognitive_scores)
    correlations.append(r)

correlations = np.array(correlations)
print(f"Max correlation: {np.max(np.abs(correlations)):.3f}")
```

---

## Cross-Modal Gradient Alignment

### Align Structural and Functional Gradients

```python
from brainspace.gradient import GradientMaps, ProcrustesAlignment
import numpy as np

# Structural connectivity (e.g., from tractography)
sc_matrix = np.random.rand(400, 400)
sc_matrix = (sc_matrix + sc_matrix.T) / 2  # Symmetric

# Functional connectivity (e.g., from fMRI)
fc_matrix = np.corrcoef(np.random.randn(400, 100))

# Compute gradients for both modalities
gm_sc = GradientMaps(n_components=10, approach='dm', kernel='normalized_angle')
gm_sc.fit(sc_matrix)
gradients_sc = gm_sc.gradients_

gm_fc = GradientMaps(n_components=10, approach='dm', kernel='normalized_angle')
gm_fc.fit(fc_matrix)
gradients_fc = gm_fc.gradients_

# Align functional to structural gradients
pa = ProcrustesAlignment()
gradients_fc_aligned = pa.fit_transform([gradients_sc, gradients_fc])[1]

# Compare gradients
from scipy.stats import spearmanr

for i in range(3):
    r, p = spearmanr(gradients_sc[:, i], gradients_fc_aligned[:, i])
    print(f"Gradient {i+1}: r={r:.3f}, p={p:.3e}")

# High correlation = similar organization across modalities
```

### Multi-Modal Gradient Integration

```python
# Combine structural, functional, and morphometric gradients

# Morphometric similarity (cortical thickness correlation)
thickness = np.random.randn(400)  # Example thickness values
morph_similarity = np.corrcoef(thickness.reshape(-1, 1))  # Simplified

# Compute gradients for all modalities
modalities = {
    'structural': sc_matrix,
    'functional': fc_matrix,
    'morphometric': morph_similarity
}

multi_modal_gradients = {}

for name, matrix in modalities.items():
    gm = GradientMaps(n_components=5, approach='dm', kernel='normalized_angle')
    gm.fit(matrix)
    multi_modal_gradients[name] = gm.gradients_

# Align all to structural
reference = multi_modal_gradients['structural']
aligned = {'structural': reference}

pa = ProcrustesAlignment()
for name in ['functional', 'morphometric']:
    aligned[name] = pa.fit_transform([reference, multi_modal_gradients[name]])[1]

# Compute cross-modal similarity
for name in ['functional', 'morphometric']:
    r, p = spearmanr(aligned['structural'][:, 0], aligned[name][:, 0])
    print(f"Structural vs {name}: r={r:.3f}, p={p:.3e}")
```

---

## Statistical Testing with Spatial Permutations

### Spin Test for Spatial Null Model

```python
from brainspace.null_models import SpinPermutations
from brainspace.datasets import load_conte69
import numpy as np

# Load surfaces for spin test
surf_lh, surf_rh = load_conte69()

# Example: correlate gradient with cortical thickness
gradient1 = gm.gradients_[:, 0]  # 400 regions
cortical_thickness = np.random.randn(400)  # Example thickness

# Observed correlation
from scipy.stats import spearmanr
r_obs, _ = spearmanr(gradient1, cortical_thickness)

# Generate spatial null distribution using spin test
sp = SpinPermutations(
    n_rep=1000,              # Number of permutations
    random_state=0
)

# Fit to surface (left hemisphere example)
sp.fit(surf_lh)

# Randomize gradient values while preserving spatial structure
gradient1_lh = gradient1[:200]  # Left hemisphere
thickness_lh = cortical_thickness[:200]

# Generate null distribution
null_correlations = []
for perm in sp(gradient1_lh):
    r_null, _ = spearmanr(perm, thickness_lh)
    null_correlations.append(r_null)

null_correlations = np.array(null_correlations)

# Compute p-value
p_value = np.mean(np.abs(null_correlations) >= np.abs(r_obs))

print(f"Observed correlation: r={r_obs:.3f}")
print(f"Spin test p-value: p={p_value:.3f}")
```

### Spatial Autocorrelation-Preserving Permutations

```python
from brainspace.null_models import MoranRandomization

# Moran spectral randomization
# Preserves spatial autocorrelation structure

mr = MoranRandomization(
    n_rep=1000,
    procedure='singleton',  # or 'pair'
    random_state=0
)

# Fit to surface
mr.fit(surf_lh)

# Generate spatially autocorrelated nulls
null_correlations_moran = []
for perm in mr(gradient1_lh):
    r_null, _ = spearmanr(perm, thickness_lh)
    null_correlations_moran.append(r_null)

p_value_moran = np.mean(np.abs(null_correlations_moran) >= np.abs(r_obs))

print(f"Moran randomization p-value: p={p_value_moran:.3f}")
```

---

## Advanced Visualization

### 2D Gradient Scatter Plot

```python
import matplotlib.pyplot as plt
from brainspace.plotting import plot_hemispheres

# Plot first two gradients as 2D scatter
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    gm.gradients_[:, 0],
    gm.gradients_[:, 1],
    c=gm.gradients_[:, 0],  # Color by gradient 1
    cmap='viridis',
    alpha=0.6,
    s=30
)

ax.set_xlabel('Gradient 1 (Sensorimotor-Transmodal)', fontsize=12)
ax.set_ylabel('Gradient 2 (Visual-Somatomotor)', fontsize=12)
ax.set_title('2D Gradient Space', fontsize=14)
ax.grid(True, alpha=0.3)

plt.colorbar(scatter, ax=ax, label='Gradient 1')
plt.tight_layout()
plt.savefig('gradient_2d_scatter.png', dpi=300)
```

### Multiple Gradients on Surface

```python
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

# Map first 3 gradients to surface
n_gradients = 3
gradient_surfaces_lh = []
gradient_surfaces_rh = []

for i in range(n_gradients):
    grad_lh = map_to_labels(
        gm.gradients_[:200, i],
        labeling['lh'],
        mask=labeling['lh'] != 0,
        fill=np.nan
    )
    grad_rh = map_to_labels(
        gm.gradients_[200:, i],
        labeling['rh'],
        mask=labeling['rh'] != 0,
        fill=np.nan
    )
    gradient_surfaces_lh.append(grad_lh)
    gradient_surfaces_rh.append(grad_rh)

# Plot all gradients
plot_hemispheres(
    surf_lh, surf_rh,
    array_name=list(zip(gradient_surfaces_lh, gradient_surfaces_rh)),
    size=(1200, 800),
    cmap='viridis',
    color_bar=True,
    label_text=[f'Gradient {i+1}' for i in range(n_gradients)],
    zoom=1.3
)
```

---

## Integration with Neuroimaging Data

### From Resting-State fMRI to Gradients

```python
import nibabel as nib
import numpy as np
from nilearn import datasets, connectome

# Load preprocessed fMRI data (e.g., from fMRIPrep)
# Example: using nilearn dataset
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400)
parcels = atlas['maps']

from nilearn.input_data import NiftiLabelsMasker

masker = NiftiLabelsMasker(
    labels_img=parcels,
    standardize=True,
    memory='nilearn_cache'
)

# Load fMRI timeseries
fmri_file = '/data/sub-01/func/sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
# timeseries = masker.fit_transform(fmri_file)

# For example purposes, simulate
timeseries = np.random.randn(200, 400)  # 200 TRs, 400 regions

# Compute connectivity matrix
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]

# Compute gradients
from brainspace.gradient import GradientMaps

gm = GradientMaps(n_components=10, approach='dm', kernel='normalized_angle')
gm.fit(connectivity_matrix)

print(f"Computed gradients from fMRI connectivity")
print(f"Gradient shape: {gm.gradients_.shape}")
```

### Morphometric Gradient Analysis

```python
# Compute gradients from morphometric similarity

# Load FreeSurfer data (cortical thickness)
# Example: Schaefer 400 parcellation
thickness_file = '/data/sub-01/freesurfer/sub-01/surf/lh.thickness'

# For parcellated data
parcel_thickness = np.random.randn(400)  # Example

# Compute morphometric similarity matrix
# (regions with similar thickness profiles)
morph_similarity = np.corrcoef(parcel_thickness.reshape(-1, 1))

# More realistically, use multivariate features:
# Combine thickness, curvature, surface area, etc.

# Compute gradients
gm_morph = GradientMaps(n_components=5, approach='dm', kernel='normalized_angle')
gm_morph.fit(morph_similarity)

print("Morphometric gradients computed")
```

---

## Real-World Applications

### Sensorimotor-to-Transmodal Gradient

```python
# The principal gradient typically reflects sensorimotor-to-transmodal hierarchy

# Identify sensorimotor and transmodal regions
sensorimotor_regions = np.array([0, 1, 2, 3, 4])  # Example indices
transmodal_regions = np.array([395, 396, 397, 398, 399])  # Example indices

gradient1 = gm.gradients_[:, 0]

sensorimotor_values = gradient1[sensorimotor_regions]
transmodal_values = gradient1[transmodal_regions]

print(f"Sensorimotor gradient values: {sensorimotor_values.mean():.3f} ± {sensorimotor_values.std():.3f}")
print(f"Transmodal gradient values: {transmodal_values.mean():.3f} ± {transmodal_values.std():.3f}")

# Typical pattern: sensorimotor at one extreme, transmodal at other
```

### Clinical Gradient Alterations

```python
# Compare gradients between patient and control groups

# Patients
connectivity_patients = [np.corrcoef(np.random.randn(400, 100)) for _ in range(20)]

# Controls
connectivity_controls = [np.corrcoef(np.random.randn(400, 100)) for _ in range(20)]

# Compute and align gradients
def compute_aligned_gradients(connectivity_list):
    gradients = []
    for conn in connectivity_list:
        gm = GradientMaps(n_components=5, approach='dm', kernel='normalized_angle')
        gm.fit(conn)
        gradients.append(gm.gradients_)

    pa = ProcrustesAlignment()
    aligned = pa.fit_transform(gradients)
    return np.array(aligned)

patients_gradients = compute_aligned_gradients(connectivity_patients)
controls_gradients = compute_aligned_gradients(connectivity_controls)

# Compare gradient 1 between groups
from scipy.stats import ttest_ind

gradient_idx = 0
for region in range(400):
    patient_vals = patients_gradients[:, region, gradient_idx]
    control_vals = controls_gradients[:, region, gradient_idx]

    t, p = ttest_ind(patient_vals, control_vals)

    # Apply multiple comparison correction
    # (simplified - use FDR in practice)

# Identify regions with altered gradients
```

### Developmental Trajectories

```python
# Analyze gradient changes across development

# Age groups
ages = np.array([8, 10, 12, 14, 16, 18, 20, 25, 30])  # Years
n_subjects_per_age = 5

# Simulate gradients for different ages
all_gradients = []
all_ages = []

for age in ages:
    for _ in range(n_subjects_per_age):
        # Simulate age-related connectivity changes
        conn = np.corrcoef(np.random.randn(400, 100))
        gm = GradientMaps(n_components=5, approach='dm', kernel='normalized_angle')
        gm.fit(conn)
        all_gradients.append(gm.gradients_[:, 0])
        all_ages.append(age)

all_gradients = np.array(all_gradients)
all_ages = np.array(all_ages)

# For each region, correlate gradient with age
from scipy.stats import spearmanr

age_correlations = []
for region in range(400):
    r, p = spearmanr(all_ages, all_gradients[:, region])
    age_correlations.append(r)

age_correlations = np.array(age_correlations)

print(f"Regions showing age-related gradient change: {np.sum(np.abs(age_correlations) > 0.3)}")
```

---

## MATLAB Implementation

### Basic Gradient Analysis in MATLAB

```matlab
% Load connectivity matrix
load('connectivity_matrix.mat'); % 400x400

% Create GradientMaps object
gm = GradientMaps('n_components', 10, ...
                  'approach', 'dm', ...
                  'kernel', 'na'); % normalized angle

% Fit to data
gm = gm.fit(connectivity_matrix);

% Access gradients
gradients = gm.gradients; % 400x10
lambdas = gm.lambda;      % Eigenvalues

% Plot eigenvalues
figure;
plot(1:10, lambdas(1:10), 'o-', 'LineWidth', 2);
xlabel('Gradient');
ylabel('Eigenvalue');
title('Gradient Eigenvalues');
grid on;
```

### Procrustes Alignment in MATLAB

```matlab
% Multiple subjects
n_subjects = 20;

% Compute gradients for each subject
gradients_cell = cell(n_subjects, 1);

for i = 1:n_subjects
    load(sprintf('connectivity_sub%02d.mat', i));

    gm = GradientMaps('n_components', 10, 'approach', 'dm', 'kernel', 'na');
    gm = gm.fit(connectivity_matrix);

    gradients_cell{i} = gm.gradients;
end

% Procrustes alignment
pa = ProcrustesAlignment('n_iter', 10);
gradients_aligned = pa.fit_transform(gradients_cell);

% Compute group average
mean_gradient = mean(cat(3, gradients_aligned{:}), 3);

disp('Alignment complete');
```

---

## Troubleshooting

### Installation Issues

```bash
# If pip install fails
pip install --upgrade pip setuptools wheel
pip install brainspace --no-cache-dir

# For VTK issues (visualization)
conda install -c conda-forge vtk

# For plotting issues
pip install pyvista matplotlib

# Verify imports
python -c "from brainspace.gradient import GradientMaps; print('OK')"
```

### Gradient Computation Errors

```python
# Check connectivity matrix
import numpy as np

# Must be square
assert connectivity.shape[0] == connectivity.shape[1]

# Check for NaN/Inf
assert not np.any(np.isnan(connectivity))
assert not np.any(np.isinf(connectivity))

# For correlation matrices, diagonal should be 1
# np.fill_diagonal(connectivity, 1.0)

# Ensure proper range for correlation
# connectivity = np.clip(connectivity, -1, 1)
```

### Alignment Issues

```python
# Ensure all gradient arrays have same shape
shapes = [g.shape for g in gradients_list]
assert len(set(shapes)) == 1, f"Inconsistent shapes: {shapes}"

# Check for sign flips in gradients
# Gradients are arbitrary in sign - alignment handles this

# If alignment fails, reduce n_iter or check for degenerate data
pa = ProcrustesAlignment(n_iter=5)
```

---

## Best Practices

### Gradient Analysis Workflow

1. **Preprocessing:**
   - Ensure connectivity matrices are properly computed
   - Apply Fisher z-transform to correlations if needed
   - Handle negative correlations appropriately

2. **Kernel Selection:**
   - Normalized angle (cosine) for correlation matrices
   - Gaussian for distance matrices
   - Match kernel to data type

3. **Number of Components:**
   - Start with 10 gradients
   - Examine eigenvalue scree plot
   - Focus on first 2-3 for interpretation

4. **Alignment:**
   - Always align when comparing across subjects
   - Use Procrustes for group analysis
   - Check alignment quality

5. **Statistical Testing:**
   - Use spatial null models (spin tests)
   - Account for spatial autocorrelation
   - Apply multiple comparison correction

### Interpretation Guidelines

**Gradient 1 (Principal):**
- Typically: sensorimotor ↔ transmodal
- Reflects functional hierarchy
- Most variance explained

**Gradient 2:**
- Often: visual ↔ other sensory/association
- Sensory organization
- Second most variance

**Gradient 3+:**
- More subtle organizational axes
- Specific network differentiation
- Lower variance

---

## Resources and Further Reading

### Official Documentation

- **BrainSpace Docs:** https://brainspace.readthedocs.io/
- **GitHub:** https://github.com/MICA-MNI/BrainSpace
- **Tutorials:** https://brainspace.readthedocs.io/en/latest/pages/tutorials.html
- **API Reference:** https://brainspace.readthedocs.io/en/latest/pages/reference.html

### Key Publications

```
Vos de Wael, R., et al. (2020).
BrainSpace: a toolbox for the analysis of macroscale gradients in neuroimaging and connectomics datasets.
Communications Biology, 3(1), 103.
```

```
Margulies, D. S., et al. (2016).
Situating the default-mode network along a principal gradient of macroscale cortical organization.
Proceedings of the National Academy of Sciences, 113(44), 12574-12579.
```

### Related Tools

- **neuromaps:** Brain annotation maps for gradient contextualization
- **abagen:** Gene expression gradients
- **BrainStat:** Statistical testing for gradients
- **Connectome Workbench:** Visualization
- **FreeSurfer:** Surface generation

---

## Summary

**BrainSpace** enables cutting-edge gradient-based analysis of brain organization:

**Strengths:**
- Reveals continuous brain organization (vs. discrete parcels)
- Multiple manifold learning algorithms
- Robust alignment across subjects and modalities
- Spatial permutation testing
- Python and MATLAB support
- Excellent documentation and tutorials

**Best For:**
- Connectivity gradient analysis
- Multi-modal integration (structure-function-genetics)
- Hierarchical brain organization
- Individual differences and clinical studies
- Cross-species comparative work
- Systems neuroscience research

**Typical Workflow:**
1. Compute connectivity matrix from fMRI
2. Generate gradients with diffusion maps
3. Align across subjects with Procrustes
4. Test associations with spin tests
5. Visualize on cortical surfaces

BrainSpace has transformed our understanding of brain organization, revealing smooth hierarchical transitions from sensory to association cortex and enabling mechanistic insights into brain function, development, and disease.
