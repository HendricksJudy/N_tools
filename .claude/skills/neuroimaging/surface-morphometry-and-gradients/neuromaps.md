# neuromaps - Brain Annotation and Multi-Modal Atlas Integration

## Overview

**neuromaps** is a Python toolbox for accessing, transforming, and analyzing a comprehensive collection of brain annotation maps. Developed by the Network Neuroscience Lab, neuromaps provides standardized access to over 50 curated brain maps spanning receptor densities, neurotransmitter systems, gene expression, metabolic profiles, developmental trajectories, and more. By enabling researchers to compare custom brain maps to established reference annotations through spatial correlation and robust statistical testing, neuromaps facilitates multi-modal contextualization of neuroimaging findings.

neuromaps implements sophisticated spatial null models that preserve the spatial autocorrelation structure of brain data, ensuring valid statistical inference when correlating brain maps. The toolbox handles transformations between different surface templates (fsaverage, fslr) and volumetric spaces (MNI152), making it easy to integrate data from diverse sources.

**Key Features:**
- Access to 50+ curated brain annotation maps from multiple modalities
- Receptor density maps (5HT, DA, GABA, glutamate, opioid, cholinergic, etc.)
- Neurotransmitter and neuromodulator atlases
- Gene expression maps from Allen Brain Atlas
- Metabolic maps (glucose metabolism, CBF, CMRO2)
- Developmental and aging trajectories
- Standardized spatial transformations (fsaverage ↔ fslr ↔ MNI152)
- Surface-to-volume and volume-to-surface transformations
- Spatial null model generation (spin tests, variogram matching)
- Statistical testing with spatial autocorrelation correction
- Parcellation-based aggregation
- Visualization tools for brain maps
- Integration with multiple data sources (PET, histology, transcriptomics)

**Primary Use Cases:**
- Contextualize custom brain maps with reference annotations
- Test structure-function-genetics associations
- Identify neurotransmitter correlates of brain features
- Link imaging phenotypes to molecular systems
- Multi-modal biomarker discovery
- Validate findings across independent datasets
- Cross-species comparative neuroscience

**Official Documentation:** https://netneurolab.github.io/neuromaps/

---

## Installation

### Install neuromaps

```bash
# Install via pip
pip install neuromaps

# Or install from GitHub for latest version
pip install git+https://github.com/netneurolab/neuromaps.git

# Verify installation
python -c "import neuromaps; print(neuromaps.__version__)"
```

### Install Dependencies

```bash
# Core dependencies
pip install numpy scipy nibabel nilearn scikit-learn

# Visualization
pip install matplotlib seaborn

# Surface processing
pip install nibabel

# For advanced features
pip install netneurotools brainspace
```

### Download Example Data

```python
from neuromaps import datasets

# Fetch example annotation (serotonin 5-HT1a receptor density)
serotonin = datasets.fetch_annotation(source='hcps1200', desc='5ht1a')

print(f"Downloaded annotation: {serotonin}")
print(f"Data type: {type(serotonin)}")
```

---

## Browse Available Annotations

### List All Available Maps

```python
from neuromaps.datasets import available_annotations
import pandas as pd

# Get all available annotations
annotations = available_annotations()

# Convert to DataFrame for easier viewing
df = pd.DataFrame(annotations)

print(f"Total annotations available: {len(df)}")
print(df[['source', 'desc', 'space', 'den']].head(20))

# Filter by modality
receptor_maps = df[df['desc'].str.contains('receptor|5ht|da|gaba', case=False, na=False)]
print(f"\nReceptor maps: {len(receptor_maps)}")

# Filter by space
surface_maps = df[df['space'].str.contains('fsaverage|fslr')]
print(f"Surface maps: {len(surface_maps)}")
```

### Search for Specific Annotations

```python
from neuromaps.datasets import available_annotations

# Search for dopamine-related maps
dopamine_maps = available_annotations(desc='dopamine')

# Search for maps in MNI space
mni_maps = available_annotations(space='MNI152')

# Search by source
hcp_maps = available_annotations(source='hcps1200')

print(f"Dopamine maps: {len(dopamine_maps)}")
print(f"MNI space maps: {len(mni_maps)}")
print(f"HCP maps: {len(hcp_maps)}")
```

---

## Load Brain Annotation Maps

### Fetch Receptor Density Maps

```python
from neuromaps import datasets

# Serotonin 5-HT1a receptor
serotonin_5ht1a = datasets.fetch_annotation(
    source='savli2012',
    desc='5HT1a',
    space='MNI152',
    res='1mm'
)

# Dopamine D2 receptor
dopamine_d2 = datasets.fetch_annotation(
    source='sandiego2015',
    desc='D2',
    space='MNI152',
    res='2mm'
)

# GABA-A receptor
gaba_a = datasets.fetch_annotation(
    source='norgaard2021',
    desc='GABAa',
    space='MNI152',
    res='1mm'
)

print("Receptor maps loaded")
print(f"5-HT1a: {serotonin_5ht1a}")
```

### Fetch Gene Expression Maps

```python
# Allen Brain Atlas gene expression principal component 1
gene_pc1 = datasets.fetch_annotation(
    source='abagen',
    desc='genepc1',
    space='fsaverage',
    den='10k'
)

print(f"Gene PC1: {gene_pc1}")
```

### Fetch Metabolic Maps

```python
# Cerebral blood flow (CBF)
cbf = datasets.fetch_annotation(
    source='hcps1200',
    desc='cbf',
    space='fsLR',
    den='32k'
)

# Glucose metabolism (FDG-PET)
fdg = datasets.fetch_annotation(
    source='агора2009',
    desc='fdg',
    space='MNI152',
    res='2mm'
)

print("Metabolic maps loaded")
```

---

## Spatial Transformations

### Transform Between Surface Templates

```python
from neuromaps import transforms
import nibabel as nib

# Load map in fsaverage space
annotation_fsaverage = datasets.fetch_annotation(
    source='abagen',
    desc='genepc1',
    space='fsaverage',
    den='10k'
)

# Transform fsaverage to fsLR (HCP surface)
annotation_fslr = transforms.fsaverage_to_fslr(
    annotation_fsaverage,
    target_density='32k'
)

print("Transformed fsaverage → fsLR")
print(f"Output: {annotation_fslr}")
```

### Transform fsLR to fsaverage

```python
# Load map in fsLR (HCP) space
annotation_fslr = datasets.fetch_annotation(
    source='hcps1200',
    desc='myelin',
    space='fsLR',
    den='32k'
)

# Transform to fsaverage
annotation_fsaverage = transforms.fslr_to_fsaverage(
    annotation_fslr,
    target_density='10k'
)

print("Transformed fsLR → fsaverage")
```

### Surface to Volume Transformation

```python
from neuromaps import transforms

# Load surface annotation
surf_annotation = datasets.fetch_annotation(
    source='abagen',
    desc='genepc1',
    space='fsaverage',
    den='10k'
)

# Transform to MNI152 volume
vol_annotation = transforms.surf_to_mni(
    surf_annotation,
    method='linear',  # Interpolation method
    target='MNI152NLin2009cAsym'
)

print("Transformed surface → volume (MNI152)")
```

### Volume to Surface Transformation

```python
# Load volumetric annotation (MNI152)
vol_annotation = datasets.fetch_annotation(
    source='savli2012',
    desc='5HT1a',
    space='MNI152',
    res='2mm'
)

# Transform to fsaverage surface
surf_annotation = transforms.mni_to_surf(
    vol_annotation,
    target='fsaverage',
    density='10k'
)

print("Transformed volume → surface (fsaverage)")
```

---

## Statistical Testing with Spatial Null Models

### Spin Test (Rotation-Based Permutation)

```python
from neuromaps import nulls, stats
import numpy as np

# Load two brain maps to correlate
map1 = datasets.fetch_annotation(source='abagen', desc='genepc1', space='fsaverage', den='10k')
map2 = datasets.fetch_annotation(source='hcps1200', desc='myelin', space='fsaverage', den='10k')

# Ensure both maps are on same surface template
# (already are in this example)

# Compute observed correlation
from scipy.stats import spearmanr

# Load actual data values (map1 and map2 are tuples of left/right hemisphere files)
map1_lh = nib.load(map1[0]).darrays[0].data
map1_rh = nib.load(map1[1]).darrays[0].data
map1_data = np.concatenate([map1_lh, map1_rh])

map2_lh = nib.load(map2[0]).darrays[0].data
map2_rh = nib.load(map2[1]).darrays[0].data
map2_data = np.concatenate([map2_lh, map2_rh])

# Remove medial wall (zeros)
mask = (map1_data != 0) & (map2_data != 0)
map1_masked = map1_data[mask]
map2_masked = map2_data[mask]

r_observed, _ = spearmanr(map1_masked, map2_masked)

# Generate spin-based null distribution
nulls_spin = nulls.alexander_bloch(
    map1,
    atlas='fsaverage',
    density='10k',
    n_perm=1000,
    seed=1234
)

# Compute null correlations
null_correlations = []
for perm_map in nulls_spin:
    perm_data = np.concatenate([perm_map[0], perm_map[1]])
    perm_masked = perm_data[mask]
    r_null, _ = spearmanr(perm_masked, map2_masked)
    null_correlations.append(r_null)

null_correlations = np.array(null_correlations)

# Compute p-value
p_spin = np.mean(np.abs(null_correlations) >= np.abs(r_observed))

print(f"Observed correlation: r = {r_observed:.3f}")
print(f"Spin test p-value: p = {p_spin:.4f}")
```

### Variogram Matching (Spatial Autocorrelation Preserving)

```python
from neuromaps import nulls

# Generate variogram-matched null maps
# Preserves spatial autocorrelation structure more accurately than spin test

nulls_variogram = nulls.burt2020(
    map1,
    atlas='fsaverage',
    density='10k',
    n_perm=1000,
    seed=1234
)

# Compute null distribution
null_correlations_vgm = []
for perm_map in nulls_variogram:
    perm_data = np.concatenate([perm_map[0], perm_map[1]])
    perm_masked = perm_data[mask]
    r_null, _ = spearmanr(perm_masked, map2_masked)
    null_correlations_vgm.append(r_null)

p_variogram = np.mean(np.abs(null_correlations_vgm) >= np.abs(r_observed))

print(f"Variogram matching p-value: p = {p_variogram:.4f}")
```

### Compare Custom Map to Multiple Annotations

```python
from neuromaps import datasets, stats
import pandas as pd

# Your custom brain map (e.g., cortical thickness from patient group)
custom_map = np.random.randn(20484)  # fsaverage 10k vertices (both hemispheres)

# List of annotations to compare against
annotation_list = [
    ('savli2012', '5HT1a'),
    ('sandiego2015', 'D2'),
    ('norgaard2021', 'GABAa'),
    ('hcps1200', 'cbf'),
    ('abagen', 'genepc1')
]

results = []

for source, desc in annotation_list:
    # Fetch annotation
    try:
        annot = datasets.fetch_annotation(source=source, desc=desc, space='fsaverage', den='10k')

        # Load data
        annot_lh = nib.load(annot[0]).darrays[0].data
        annot_rh = nib.load(annot[1]).darrays[0].data
        annot_data = np.concatenate([annot_lh, annot_rh])

        # Compute correlation
        mask = (custom_map != 0) & (annot_data != 0)
        r, p = spearmanr(custom_map[mask], annot_data[mask])

        results.append({
            'source': source,
            'description': desc,
            'r': r,
            'p_uncorrected': p
        })

    except Exception as e:
        print(f"Could not fetch {source}/{desc}: {e}")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('r', key=abs, ascending=False)

print(results_df)

# Apply FDR correction
from statsmodels.stats.multitest import multipletests
_, p_fdr, _, _ = multipletests(results_df['p_uncorrected'], method='fdr_bh')
results_df['p_fdr'] = p_fdr

print("\nTop associations (FDR corrected):")
print(results_df[results_df['p_fdr'] < 0.05])
```

---

## Parcellation-Based Analysis

### Aggregate Maps to Parcels

```python
from neuromaps import transforms, datasets
from nilearn import datasets as nilearn_datasets

# Load parcellation (Schaefer 400)
parcellation = nilearn_datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)

# Load annotation in MNI space
annotation = datasets.fetch_annotation(
    source='savli2012',
    desc='5HT1a',
    space='MNI152',
    res='2mm'
)

# Load parcellation and annotation as nibabel images
import nibabel as nib
parc_img = nib.load(parcellation['maps'])
annot_img = nib.load(annotation)

# Aggregate annotation values within each parcel
from nilearn.input_data import NiftiLabelsMasker

masker = NiftiLabelsMasker(
    labels_img=parc_img,
    standardize=False,
    strategy='mean'
)

parcellated_values = masker.fit_transform(annot_img)

print(f"Parcellated annotation shape: {parcellated_values.shape}")
print(f"Values per parcel: {parcellated_values.flatten()}")
```

### Surface Parcellation

```python
from neuromaps import datasets
import nibabel as nib
import numpy as np

# Load surface annotation
annotation = datasets.fetch_annotation(
    source='abagen',
    desc='genepc1',
    space='fsaverage',
    den='10k'
)

# Load Desikan-Killiany parcellation for fsaverage
from nilearn import datasets as nilearn_datasets
parcellation = nilearn_datasets.fetch_atlas_surf_destrieux()

# Load parcellation labels
parc_lh = nib.load(parcellation['map_left']).darrays[0].data
parc_rh = nib.load(parcellation['map_right']).darrays[0].data

# Load annotation
annot_lh = nib.load(annotation[0]).darrays[0].data
annot_rh = nib.load(annotation[1]).darrays[0].data

# Aggregate to parcels (left hemisphere)
unique_labels = np.unique(parc_lh[parc_lh != 0])
parcellated_lh = []

for label in unique_labels:
    mask = parc_lh == label
    mean_value = np.mean(annot_lh[mask])
    parcellated_lh.append(mean_value)

parcellated_lh = np.array(parcellated_lh)

print(f"Parcellated to {len(parcellated_lh)} regions (LH)")
```

---

## Multi-Modal Correlation Analysis

### Structure-Function-Receptor Associations

```python
from neuromaps import datasets
from scipy.stats import spearmanr
import numpy as np
import nibabel as nib

# Load structural map (cortical thickness)
# In practice, from FreeSurfer or fMRIPrep
thickness = np.random.randn(20484)  # Example

# Load functional map (ALFF from resting-state)
alff = datasets.fetch_annotation(source='hcps1200', desc='alff', space='fsaverage', den='10k')
alff_lh = nib.load(alff[0]).darrays[0].data
alff_rh = nib.load(alff[1]).darrays[0].data
alff_data = np.concatenate([alff_lh, alff_rh])

# Load receptor map (5-HT1a)
receptor = datasets.fetch_annotation(source='beliveau2017', desc='5ht1a', space='fsaverage', den='10k')
receptor_lh = nib.load(receptor[0]).darrays[0].data
receptor_rh = nib.load(receptor[1]).darrays[0].data
receptor_data = np.concatenate([receptor_lh, receptor_rh])

# Compute correlations
mask = (thickness != 0) & (alff_data != 0) & (receptor_data != 0)

r_thickness_alff, p1 = spearmanr(thickness[mask], alff_data[mask])
r_thickness_receptor, p2 = spearmanr(thickness[mask], receptor_data[mask])
r_alff_receptor, p3 = spearmanr(alff_data[mask], receptor_data[mask])

print("Multi-modal correlations:")
print(f"  Thickness ↔ ALFF: r = {r_thickness_alff:.3f}, p = {p1:.3e}")
print(f"  Thickness ↔ Receptor: r = {r_thickness_receptor:.3f}, p = {p2:.3e}")
print(f"  ALFF ↔ Receptor: r = {r_alff_receptor:.3f}, p = {p3:.3e}")
```

### Receptor Density Profile Analysis

```python
# Compare multiple receptor systems

receptors = {
    '5-HT1a': datasets.fetch_annotation(source='beliveau2017', desc='5ht1a', space='fsaverage', den='10k'),
    '5-HT2a': datasets.fetch_annotation(source='beliveau2017', desc='5ht2a', space='fsaverage', den='10k'),
    'D1': datasets.fetch_annotation(source='kaller2017', desc='D1', space='fsaverage', den='10k'),
    'D2': datasets.fetch_annotation(source='smith2017', desc='D2', space='fsaverage', den='10k')
}

receptor_data = {}

for name, annot in receptors.items():
    try:
        lh = nib.load(annot[0]).darrays[0].data
        rh = nib.load(annot[1]).darrays[0].data
        receptor_data[name] = np.concatenate([lh, rh])
    except:
        print(f"Could not load {name}")

# Compute receptor correlation matrix
import pandas as pd

receptor_df = pd.DataFrame(receptor_data)
receptor_corr = receptor_df.corr(method='spearman')

print("Receptor correlation matrix:")
print(receptor_corr)

# Visualize
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(receptor_corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1)
plt.title('Receptor Density Correlations')
plt.tight_layout()
plt.savefig('receptor_correlations.png', dpi=300)
```

---

## Visualization

### Plot Brain Map on Surface

```python
from neuromaps import datasets
from nilearn import plotting, surface
import matplotlib.pyplot as plt

# Load annotation
annotation = datasets.fetch_annotation(
    source='abagen',
    desc='genepc1',
    space='fsaverage',
    den='10k'
)

# Load fsaverage surface
from nilearn.datasets import fetch_surf_fsaverage
fsaverage = fetch_surf_fsaverage(mesh='fsaverage5')

# Load annotation data
annot_lh = nib.load(annotation[0]).darrays[0].data

# Plot on surface
fig = plotting.plot_surf_stat_map(
    fsaverage['pial_left'],
    annot_lh,
    hemi='left',
    view='lateral',
    colorbar=True,
    cmap='viridis',
    title='Gene Expression PC1 (Left Hemisphere)',
    threshold=0.0
)

plt.savefig('gene_pc1_surface.png', dpi=300, bbox_inches='tight')
```

### Create Multi-Panel Visualization

```python
import matplotlib.pyplot as plt
from nilearn import plotting, datasets as nilearn_datasets

# Load multiple annotations
annotations = {
    '5-HT1a': datasets.fetch_annotation(source='beliveau2017', desc='5ht1a', space='fsaverage', den='10k'),
    'Myelin': datasets.fetch_annotation(source='hcps1200', desc='myelin', space='fsaverage', den='10k'),
    'Gene PC1': datasets.fetch_annotation(source='abagen', desc='genepc1', space='fsaverage', den='10k')
}

fsaverage = nilearn_datasets.fetch_surf_fsaverage(mesh='fsaverage5')

fig, axes = plt.subplots(len(annotations), 2, figsize=(12, 12),
                         subplot_kw={'projection': '3d'})

for idx, (name, annot) in enumerate(annotations.items()):
    annot_lh = nib.load(annot[0]).darrays[0].data
    annot_rh = nib.load(annot[1]).darrays[0].data

    # Plot left hemisphere
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        annot_lh,
        hemi='left',
        view='lateral',
        colorbar=False,
        cmap='viridis',
        axes=axes[idx, 0],
        title=f'{name} (LH)'
    )

    # Plot right hemisphere
    plotting.plot_surf_stat_map(
        fsaverage['pial_right'],
        annot_rh,
        hemi='right',
        view='lateral',
        colorbar=False,
        cmap='viridis',
        axes=axes[idx, 1],
        title=f'{name} (RH)'
    )

plt.tight_layout()
plt.savefig('multi_annotation_visualization.png', dpi=300, bbox_inches='tight')
```

---

## Real-World Applications

### Contextualize Cortical Thinning Pattern

```python
from neuromaps import datasets, nulls
from scipy.stats import spearmanr
import numpy as np

# Patient cortical thickness pattern (vs. controls)
# Negative values = thinning in patients
patient_thinning = np.random.randn(20484)  # Example

# Compare to receptor densities to identify neurotransmitter associations
receptors_to_test = [
    ('beliveau2017', '5ht1a', 'Serotonin 5-HT1a'),
    ('beliveau2017', '5ht2a', 'Serotonin 5-HT2a'),
    ('kaller2017', 'D1', 'Dopamine D1'),
    ('norgaard2021', 'GABAa', 'GABA-A')
]

results = []

for source, desc, name in receptors_to_test:
    try:
        # Fetch receptor map
        receptor = datasets.fetch_annotation(source=source, desc=desc, space='fsaverage', den='10k')

        # Load data
        receptor_lh = nib.load(receptor[0]).darrays[0].data
        receptor_rh = nib.load(receptor[1]).darrays[0].data
        receptor_data = np.concatenate([receptor_lh, receptor_rh])

        # Correlate with thinning pattern
        mask = (patient_thinning != 0) & (receptor_data != 0)
        r_obs, _ = spearmanr(patient_thinning[mask], receptor_data[mask])

        # Spin test for significance
        null_maps = nulls.alexander_bloch(receptor, atlas='fsaverage', density='10k', n_perm=1000, seed=123)

        null_rs = []
        for null_map in null_maps:
            null_data = np.concatenate([null_map[0], null_map[1]])
            r_null, _ = spearmanr(patient_thinning[mask], null_data[mask])
            null_rs.append(r_null)

        p_spin = np.mean(np.abs(null_rs) >= np.abs(r_obs))

        results.append({
            'receptor': name,
            'r': r_obs,
            'p_spin': p_spin
        })

        print(f"{name}: r = {r_obs:.3f}, p = {p_spin:.3f}")

    except Exception as e:
        print(f"Could not process {name}: {e}")

# Interpretation: Negative correlation = greater receptor density associated with more thinning
```

### Link fMRI Activation to Neurotransmitter Systems

```python
# Task fMRI activation map
task_activation = np.random.randn(20484)  # Example: z-scores from group analysis

# Test association with dopamine system
dopamine_maps = [
    datasets.fetch_annotation(source='kaller2017', desc='D1', space='fsaverage', den='10k'),
    datasets.fetch_annotation(source='smith2017', desc='D2', space='fsaverage', den='10k')
]

for idx, da_map in enumerate(['D1', 'D2']):
    da_annot = dopamine_maps[idx]

    da_lh = nib.load(da_annot[0]).darrays[0].data
    da_rh = nib.load(da_annot[1]).darrays[0].data
    da_data = np.concatenate([da_lh, da_rh])

    mask = (task_activation != 0) & (da_data != 0)
    r, p = spearmanr(task_activation[mask], da_data[mask])

    print(f"Task activation ↔ {da_map}: r = {r:.3f}, p = {p:.3e}")

# Interpretation: Activation pattern overlaps with dopamine distribution
```

---

## Advanced Features

### Cross-Species Comparison

```python
# Compare human and macaque brain organization

# Human receptor map
human_receptor = datasets.fetch_annotation(
    source='beliveau2017',
    desc='5ht1a',
    space='fsaverage',
    den='10k'
)

# Macaque data would need to be in compatible format
# Then transform and compare

# neuromaps supports cross-species transformations
# via surface registration
```

### Developmental Trajectories

```python
# Analyze age-related changes in receptor density

# Example: compare receptor maps to cortical thinning with age
age_thinning = np.random.randn(20484)  # Example: correlation with age

receptor = datasets.fetch_annotation(source='beliveau2017', desc='5ht1a', space='fsaverage', den='10k')
receptor_lh = nib.load(receptor[0]).darrays[0].data
receptor_rh = nib.load(receptor[1]).darrays[0].data
receptor_data = np.concatenate([receptor_lh, receptor_rh])

mask = (age_thinning != 0) & (receptor_data != 0)
r, p = spearmanr(age_thinning[mask], receptor_data[mask])

print(f"Age-related thinning ↔ 5-HT1a: r = {r:.3f}, p = {p:.3e}")

# Positive correlation = regions with high receptor density thin more with age
```

---

## Troubleshooting

### Data Download Issues

```bash
# If fetch_annotation fails
# Check internet connection
# Clear cache and retry

# neuromaps cache location
echo $HOME/.cache/neuromaps

# Clear cache
rm -rf $HOME/.cache/neuromaps

# Retry download
```

### Transformation Errors

```python
# If spatial transformation fails, check:

# 1. Input format
print(type(annotation))  # Should be tuple of file paths for surface

# 2. Hemisphere structure
# Surface annotations are (left, right) tuples

# 3. Density/resolution compatibility
# Ensure target density is valid for template
```

### Missing Annotations

```python
# If annotation not found, check spelling
from neuromaps.datasets import available_annotations

# List all with 'serotonin' in description
avail = available_annotations()
serotonin_maps = [a for a in avail if 'serotonin' in str(a).lower() or '5ht' in str(a).lower()]

print(serotonin_maps)
```

---

## Best Practices

### Statistical Testing

1. **Always use spatial null models:**
   - Brain data is spatially autocorrelated
   - Standard permutation tests inflate Type I error
   - Use spin tests or variogram matching

2. **Choose appropriate null model:**
   - Spin test: Good for surface data, fast
   - Variogram matching: Better preserves autocorrelation structure

3. **Multiple comparison correction:**
   - Apply FDR or Bonferroni when testing multiple annotations
   - Report both uncorrected and corrected p-values

### Data Integration

1. **Match spatial templates:**
   - Transform all maps to common space
   - Use appropriate transformations (surface/volume)
   - Verify alignment visually

2. **Handle missing data:**
   - Exclude medial wall (zeros) from correlations
   - Use consistent masking across comparisons

3. **Parcellation considerations:**
   - Choose parcellation granularity appropriately
   - Aggregate with meaningful strategy (mean, median, mode)

---

## Resources and Further Reading

### Official Documentation

- **neuromaps Docs:** https://netneurolab.github.io/neuromaps/
- **GitHub:** https://github.com/netneurolab/neuromaps
- **Tutorials:** https://netneurolab.github.io/neuromaps/user_guide/index.html
- **API Reference:** https://netneurolab.github.io/neuromaps/api.html

### Key Publications

```
Markello, R. D., et al. (2022).
neuromaps: structural and functional interpretation of brain maps.
Nature Methods, 19(11), 1472-1479.
```

### Related Tools

- **BrainSpace:** Gradient analysis with neuromaps annotations
- **abagen:** Gene expression for neuromaps integration
- **BrainStat:** Statistical testing framework
- **nilearn:** Neuroimaging visualization
- **netneurotools:** Network neuroscience utilities

---

## Summary

**neuromaps** enables comprehensive multi-modal brain map analysis:

**Strengths:**
- 50+ curated annotations across modalities
- Standardized spatial transformations
- Robust spatial null models
- Easy-to-use Python API
- Regular updates with new maps
- Excellent documentation

**Best For:**
- Contextualizing neuroimaging findings
- Multi-modal integration
- Receptor-imaging associations
- Gene-imaging links
- Cross-modal validation
- Hypothesis generation

**Typical Workflow:**
1. Compute custom brain map (e.g., patient vs. control)
2. Transform to standard space
3. Fetch relevant neuromaps annotations
4. Correlate with spatial null models
5. Interpret findings in multi-modal context

neuromaps has revolutionized how we contextualize neuroimaging findings by providing standardized access to diverse brain annotations and robust statistical frameworks for multi-modal integration.

## Citation

```bibtex
@article{markello2022neuromaps,
  title={neuromaps: structural and functional brain annotations for neuroimaging},
  author={Markello, Ross D. and Hansen, Julie Y. and Liu, Zhen and others},
  journal={Nature Methods},
  volume={19},
  pages={1711--1717},
  year={2022},
  doi={10.1038/s41592-022-01625-5}
}
```
