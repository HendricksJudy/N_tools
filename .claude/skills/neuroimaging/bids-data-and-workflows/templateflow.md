# TemplateFlow - Brain Template and Atlas Management

## Overview

**TemplateFlow** is a framework for managing, distributing, and using neuroimaging templates and atlases in a standardized, version-controlled manner. Developed by the NiPreps (NeuroImaging PREProcessing toolS) community, TemplateFlow provides a centralized Archive of Research-grade Templates (ART) with consistent metadata and BIDS-compatible naming conventions, ensuring spatial normalization and atlas-based analyses use validated, reproducible references.

TemplateFlow eliminates ambiguity about which template variant or version is used in analyses by providing programmatic access to curated brain templates (MNI152, MNI305, fsaverage, OASIS, etc.) with complete provenance. It supports multiple coordinate systems, resolutions, modalities, and cohorts (adult, pediatric, aging), enabling researchers to select appropriate templates for their populations while maintaining reproducibility through version control.

**Key Features:**
- Centralized repository of standardized brain templates
- Programmatic access via Python API
- Automatic template download and local caching
- Template metadata (resolution, modality, cohort, symmetry)
- Multiple coordinate systems (MNI, fsaverage, fsnative, fsLR)
- Volumetric and surface templates
- Parcellations and atlas integration (Schaefer, DKT, Destrieux)
- Template versioning and provenance tracking
- BIDS-compatible naming conventions
- Symmetric and asymmetric template variants
- Pediatric, adult, and aging templates
- Integration with fMRIPrep, Nilearn, ANTs, and other tools
- Custom template addition support

**Primary Use Cases:**
- Spatial normalization to standard space
- Atlas-based anatomical labeling
- Template selection for specific populations
- Reproducible group-level analyses
- Surface-based analysis with fsaverage
- Multi-modal template-based registration
- Quality control with standardized references
- Method development with validated templates

**Official Documentation:** https://www.templateflow.org/

---

## Installation

### Install TemplateFlow

```bash
# Install via pip (recommended)
pip install templateflow

# Or install from GitHub for latest version
pip install git+https://github.com/templateflow/python-client.git

# Verify installation
python -c "import templateflow; print(templateflow.__version__)"
```

### Configure Template Cache

```bash
# By default, templates stored in ~/.cache/templateflow
# Set custom cache location
export TEMPLATEFLOW_HOME=/data/templates

# Or set in Python
import templateflow
templateflow.conf.TF_HOME = '/data/templates'

# Check cache location
python -c "import templateflow.conf; print(templateflow.conf.TF_HOME)"
```

---

## Template Basics

### What are Brain Templates?

Brain templates are standard reference spaces used for:
- **Spatial normalization**: Align individual brains to common space
- **Group analysis**: Compare activation across subjects
- **Anatomical reference**: Standard coordinates for brain regions
- **Atlas labeling**: Map parcellations to individual brains

### Coordinate Systems

```python
import templateflow.api as tf

# MNI152 (volumetric, most common for fMRI)
# - MNI152NLin2009cAsym: fMRIPrep default
# - MNI152NLin6Asym: Older variant
# - MNI152NLin2009aSym: Symmetric version

# fsaverage (surface, FreeSurfer standard)
# - fsaverage: Full resolution
# - fsaverage5, fsaverage6: Lower resolution

# fsLR (surface, HCP standard)
# - fsLR 32k, fsLR 59k: Different resolutions

# Other coordinate systems
# - MNI305: Original MNI space
# - OASIS30ANTs: Aging population
```

---

## Accessing Templates

### List Available Templates

```python
import templateflow.api as tf

# Get all available template identifiers
templates = tf.templates()
print(f"Available templates: {len(templates)}")
print(templates[:10])  # First 10

# Common templates:
# MNI152NLin2009cAsym, MNI152NLin6Asym, MNI152NLin2009aSym,
# fsaverage, fsLR, OASIS30ANTs, NKI, MNI152Lin, MNIPediatricAsym

# Check if specific template exists
if 'MNI152NLin2009cAsym' in templates:
    print("fMRIPrep default template available")
```

### Query Template Metadata

```python
import templateflow.api as tf

# Get metadata for template
metadata = tf.get(
    'MNI152NLin2009cAsym',
    return_type='metadata'
)

print(f"Name: {metadata['Name']}")
print(f"Authors: {metadata['Authors']}")
print(f"License: {metadata['License']}")
print(f"RRID: {metadata['RRID']}")

# List available resources for template
resources = tf.get(
    'MNI152NLin2009cAsym',
    desc=None,  # All descriptions
    resolution=None,  # All resolutions
    return_type='filename'
)
print(f"\nAvailable resources: {len(resources)}")
```

### Download and Access Template

```python
import templateflow.api as tf
from nilearn import plotting

# Get T1w template (automatically downloads if not cached)
t1w_template = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,  # 1mm isotropic
    desc='brain',  # Brain-extracted
    suffix='T1w'
)

print(f"Template path: {t1w_template}")
# /home/user/.cache/templateflow/tpl-MNI152NLin2009cAsym/
#   tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz

# Visualize template
plotting.plot_anat(t1w_template, title='MNI152NLin2009cAsym T1w')
plotting.show()
```

---

## Common Templates

### MNI152NLin2009cAsym (fMRIPrep Default)

```python
import templateflow.api as tf

# T1w template (brain-extracted, 1mm)
t1w_brain = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    desc='brain',
    suffix='T1w'
)

# T1w with skull (for visualization)
t1w_head = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    suffix='T1w'
)

# Brain mask
brain_mask = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    desc='brain',
    suffix='mask'
)

# T2w template
t2w = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    suffix='T2w'
)

# Probabilistic tissue maps
gm_prob = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    label='GM',
    suffix='probseg'
)

wm_prob = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    label='WM',
    suffix='probseg'
)

csf_prob = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    label='CSF',
    suffix='probseg'
)
```

### MNI152NLin2009aSym (Symmetric Variant)

```python
# Symmetric template (left-right mirrored)
# Useful for bilateral analyses
t1w_sym = tf.get(
    'MNI152NLin2009aSym',
    resolution=1,
    desc='brain',
    suffix='T1w'
)

# Symmetric templates have:
# - Left and right hemispheres are mirror images
# - Useful for laterality analyses
# - Some atlases require symmetric templates
```

### OASIS30ANTs (Aging Population)

```python
# Template from older adult population
# Better for aging studies than young adult MNI
oasis_t1w = tf.get(
    'OASIS30ANTs',
    resolution=1,
    desc='brain',
    suffix='T1w'
)

# OASIS template metadata
# - Age range: 65-90+ years
# - N=30 subjects
# - Alzheimer's Disease Neuroimaging Initiative
```

### MNIPediatricAsym (Pediatric Templates)

```python
# Pediatric templates for different age cohorts
# Available cohorts: 1, 2 (cohort 1 = younger, cohort 2 = older)

# 4.5-8.5 years (cohort 1)
pediatric_young = tf.get(
    'MNIPediatricAsym',
    cohort=1,
    resolution=1,
    suffix='T1w'
)

# 7-11 years (cohort 2)
pediatric_older = tf.get(
    'MNIPediatricAsym',
    cohort=2,
    resolution=1,
    suffix='T1w'
)

# Use age-appropriate template for pediatric studies
```

---

## Surface Templates

### fsaverage (FreeSurfer Standard)

```python
import templateflow.api as tf

# Full resolution fsaverage
fsavg_pial_lh = tf.get(
    'fsaverage',
    hemi='L',
    density='164k',
    suffix='pial'
)

fsavg_pial_rh = tf.get(
    'fsaverage',
    hemi='R',
    density='164k',
    suffix='pial'
)

# Inflated surface for visualization
fsavg_inflated_lh = tf.get(
    'fsaverage',
    hemi='L',
    density='164k',
    suffix='inflated'
)

# White matter surface
fsavg_white_lh = tf.get(
    'fsaverage',
    hemi='L',
    density='164k',
    suffix='white'
)

# Sphere for registration
fsavg_sphere_lh = tf.get(
    'fsaverage',
    hemi='L',
    density='164k',
    suffix='sphere'
)
```

### fsaverage Lower Resolutions

```python
# fsaverage5 (~10k vertices, fast computations)
fsavg5_pial = tf.get(
    'fsaverage',
    hemi='L',
    density='10k',
    desc='std',
    suffix='pial'
)

# fsaverage6 (~40k vertices, medium resolution)
fsavg6_pial = tf.get(
    'fsaverage',
    hemi='L',
    density='41k',
    desc='std',
    suffix='pial'
)

# Resolution selection:
# - fsaverage5: Fast, lower detail
# - fsaverage6: Balanced
# - fsaverage: High detail, slower
```

### fsLR (HCP Surface Space)

```python
# HCP fsLR space (32k vertices standard)
fslr_pial_lh = tf.get(
    'fsLR',
    hemi='L',
    density='32k',
    suffix='pial'
)

# fsLR 59k (higher resolution)
fslr_59k_pial = tf.get(
    'fsLR',
    hemi='L',
    density='59k',
    suffix='pial'
)
```

---

## Parcellations and Atlases

### Schaefer Parcellation

```python
import templateflow.api as tf

# Schaefer parcellation (popular for connectivity)
# Available: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 parcels
# Networks: 7 (Yeo) or 17 (detailed)

# 400 parcels, 7 networks, MNI space
schaefer_400 = tf.get(
    'MNI152NLin2009cAsym',
    atlas='Schaefer2018',
    desc='400Parcels7Networks',
    resolution=1,
    suffix='dseg'
)

# 200 parcels, 17 networks
schaefer_200 = tf.get(
    'MNI152NLin2009cAsym',
    atlas='Schaefer2018',
    desc='200Parcels17Networks',
    resolution=1,
    suffix='dseg'
)

# Surface version (fsaverage)
schaefer_surf_lh = tf.get(
    'fsaverage',
    atlas='Schaefer2018',
    desc='400Parcels7Networks',
    hemi='L',
    density='164k',
    suffix='dseg'
)
```

### DKT Atlas (Desikan-Killiany-Tourville)

```python
# DKT cortical parcellation (FreeSurfer-based)
dkt_volumetric = tf.get(
    'MNI152NLin2009cAsym',
    atlas='DKT',
    resolution=1,
    suffix='dseg'
)

# DKT on fsaverage surface
dkt_surf_lh = tf.get(
    'fsaverage',
    atlas='DKT',
    hemi='L',
    density='164k',
    suffix='dseg'
)

# DKT includes 31 cortical regions per hemisphere
# Plus subcortical structures
```

### Harvard-Oxford Atlas

```python
# Harvard-Oxford cortical and subcortical atlases
ho_cortical = tf.get(
    'MNI152NLin2009cAsym',
    atlas='HarvardOxford',
    desc='cort',  # Cortical
    resolution=1,
    suffix='probseg'
)

ho_subcortical = tf.get(
    'MNI152NLin2009cAsym',
    atlas='HarvardOxford',
    desc='sub',  # Subcortical
    resolution=1,
    suffix='probseg'
)

# Harvard-Oxford is probabilistic (not discrete labels)
# Values represent probability of being in each region
```

---

## Template Selection Guidelines

### Choose Template for Population

```python
def select_template(age, study_type='structural'):
    """Select appropriate template based on age and study"""

    if age < 4:
        print("WARNING: Limited templates for very young children")
        return 'MNIPediatricAsym', {'cohort': 1}

    elif age < 9:
        # Young children
        return 'MNIPediatricAsym', {'cohort': 1}

    elif age < 12:
        # Older children
        return 'MNIPediatricAsym', {'cohort': 2}

    elif age < 65:
        # Adults
        if study_type == 'functional':
            return 'MNI152NLin2009cAsym', {}
        else:
            return 'MNI152NLin6Asym', {}

    else:
        # Older adults
        return 'OASIS30ANTs', {}

# Example usage
template, params = select_template(age=8)
print(f"Recommended template: {template}")

t1w = tf.get(template, resolution=1, suffix='T1w', **params)
```

### Resolution Considerations

```python
# Resolution selection depends on:
# - Data resolution (match or slightly higher)
# - Computational resources
# - Analysis precision needed

# For 3mm functional data, use 2mm template
functional_template = tf.get(
    'MNI152NLin2009cAsym',
    resolution=2,  # 2mm isotropic
    desc='brain',
    suffix='T1w'
)

# For structural analysis, use 1mm
structural_template = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,  # 1mm isotropic
    desc='brain',
    suffix='T1w'
)
```

### Symmetric vs Asymmetric

```python
# Use symmetric templates for:
# - Left-right comparisons
# - Laterality analyses
# - Some specific atlases (require symmetric)

symmetric = tf.get(
    'MNI152NLin2009aSym',  # Note 'aSym' = asymmetric
    resolution=1,
    suffix='T1w'
)

# Use asymmetric templates for:
# - Most standard analyses
# - fMRIPrep default
# - Better anatomical accuracy

asymmetric = tf.get(
    'MNI152NLin2009cAsym',  # 'cAsym' = asymmetric
    resolution=1,
    suffix='T1w'
)
```

---

## Integration with Analysis Pipelines

### Use with fMRIPrep

```python
# fMRIPrep uses TemplateFlow by default
# Specify template via command line

# fmriprep /data/bids /output participant \
#   --output-spaces MNI152NLin2009cAsym:res-2 \
#                   fsaverage:den-10k \
#                   T1w

# fMRIPrep automatically fetches templates from TemplateFlow
# No manual download needed

# Check which templates fMRIPrep will use
from templateflow.api import get
template = get(
    'MNI152NLin2009cAsym',
    resolution=2,
    desc='brain',
    suffix='T1w'
)
print(f"fMRIPrep will use: {template}")
```

### Use with Nilearn

```python
from nilearn import datasets, plotting
import templateflow.api as tf

# Load template via TemplateFlow
template = tf.get(
    'MNI152NLin2009cAsym',
    resolution=2,
    desc='brain',
    suffix='T1w'
)

# Use in Nilearn plotting
from nilearn.image import load_img
stat_map = '/path/to/statmap.nii.gz'

plotting.plot_stat_map(
    stat_map,
    bg_img=str(template),
    threshold=2.5,
    title='Statistical Map on TemplateFlow MNI'
)
plotting.show()

# Load Schaefer atlas
schaefer = tf.get(
    'MNI152NLin2009cAsym',
    atlas='Schaefer2018',
    desc='400Parcels7Networks',
    resolution=1,
    suffix='dseg'
)

# Use for parcellation
from nilearn.maskers import NiftiLabelsMasker
masker = NiftiLabelsMasker(
    labels_img=str(schaefer),
    standardize=True
)
```

### Use with ANTsPy

```python
import ants
import templateflow.api as tf

# Get template for registration
template_path = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    desc='brain',
    suffix='T1w'
)

template_img = ants.image_read(str(template_path))

# Register individual T1 to template
moving_img = ants.image_read('/data/sub-01_T1w.nii.gz')

registration = ants.registration(
    fixed=template_img,
    moving=moving_img,
    type_of_transform='SyN'
)

# Apply transform
warped_img = registration['warpedmovout']
ants.image_write(warped_img, '/output/sub-01_MNI.nii.gz')
```

---

## Working with Surface Data

### Load Surface for Visualization

```python
import templateflow.api as tf
from nilearn import surface, plotting

# Get fsaverage surface
surf_lh = tf.get(
    'fsaverage',
    hemi='L',
    density='10k',  # fsaverage5 for speed
    desc='std',
    suffix='pial'
)

# Load surface data (e.g., from FreeSurfer)
surf_data = '/path/to/lh.thickness'

# Plot on surface
plotting.plot_surf_stat_map(
    surf_lh,
    surf_data,
    hemi='left',
    title='Cortical Thickness on fsaverage5',
    colorbar=True
)
plotting.show()
```

### Project Volume to Surface

```python
from nilearn import surface
import templateflow.api as tf

# Get fsaverage surfaces
surf_lh = tf.get('fsaverage', hemi='L', density='10k', suffix='pial')
surf_rh = tf.get('fsaverage', hemi='R', density='10k', suffix='pial')

# Volume data in MNI space
vol_data = '/path/to/statmap_MNI.nii.gz'

# Project to surface
texture_lh = surface.vol_to_surf(vol_data, surf_lh)
texture_rh = surface.vol_to_surf(vol_data, surf_rh)

# Plot both hemispheres
from nilearn import plotting
plotting.plot_surf_stat_map(surf_lh, texture_lh, hemi='left')
plotting.plot_surf_stat_map(surf_rh, texture_rh, hemi='right')
```

---

## Custom Templates

### Add Custom Template

```python
# To add custom template to TemplateFlow:
# 1. Follow BIDS naming conventions
# 2. Create template_description.json
# 3. Add to local TemplateFlow repository

# Example template_description.json
import json

template_metadata = {
    "Name": "My Custom Template",
    "Authors": ["Author One", "Author Two"],
    "License": "CC0",
    "RRID": "SCR_XXXXXX",
    "ReferencesAndLinks": ["https://doi.org/XX.XXXX/XXXXX"],
    "Cohort": "Adult",
    "Species": "Homo sapiens"
}

# Save metadata
with open('template_description.json', 'w') as f:
    json.dump(template_metadata, f, indent=2)

# Organize files following BIDS:
# tpl-MyTemplate/
#   template_description.json
#   tpl-MyTemplate_res-01_T1w.nii.gz
#   tpl-MyTemplate_res-01_desc-brain_mask.nii.gz
#   ...

# Add to TemplateFlow home
# cp -r tpl-MyTemplate $TEMPLATEFLOW_HOME/
```

---

## Reproducibility Practices

### Document Template Usage

```python
import templateflow.api as tf

# Get template and record provenance
template = tf.get(
    'MNI152NLin2009cAsym',
    resolution=1,
    desc='brain',
    suffix='T1w'
)

# Get template version
import templateflow
tf_version = templateflow.__version__
template_version = tf.get(
    'MNI152NLin2009cAsym',
    return_type='metadata'
).get('version', 'unknown')

# Document in methods
methods_text = f"""
Spatial normalization was performed to the MNI152NLin2009cAsym template
(version {template_version}) obtained from TemplateFlow (version {tf_version};
https://doi.org/10.1101/2021.02.10.430678).
"""

print(methods_text)
```

### Version Pinning

```bash
# Pin TemplateFlow version in requirements.txt
echo "templateflow==0.8.1" >> requirements.txt

# Or specify minimum version
echo "templateflow>=0.8.0" >> requirements.txt

# Record template versions used
python -c "
import templateflow.api as tf
import json

templates_used = {
    'MNI152NLin2009cAsym': tf.get('MNI152NLin2009cAsym', return_type='metadata').get('version'),
    'fsaverage': tf.get('fsaverage', return_type='metadata').get('version'),
}

with open('template_versions.json', 'w') as f:
    json.dump(templates_used, f, indent=2)
"
```

---

## Cache Management

### Check Cache Status

```python
import templateflow.conf as tfconf
from pathlib import Path

# Cache location
cache_dir = Path(tfconf.TF_HOME)
print(f"TemplateFlow cache: {cache_dir}")

# List cached templates
cached_templates = [d.name for d in cache_dir.glob('tpl-*') if d.is_dir()]
print(f"Cached templates ({len(cached_templates)}): {cached_templates}")

# Cache size
total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
print(f"Total cache size: {total_size / 1e9:.2f} GB")
```

### Clear and Rebuild Cache

```bash
# Clear specific template
rm -rf ~/.cache/templateflow/tpl-MNI152NLin2009cAsym

# Clear all cache (will re-download as needed)
rm -rf ~/.cache/templateflow/*

# Update templates to latest versions
python -c "
import templateflow.api as tf
# Re-fetch template (gets latest version)
tf.get('MNI152NLin2009cAsym', resolution=1, suffix='T1w')
"
```

---

## Troubleshooting

### Download Failures

```python
import templateflow.api as tf

# If download fails, try manual fetch
try:
    template = tf.get('MNI152NLin2009cAsym', resolution=1, suffix='T1w')
except Exception as e:
    print(f"Download failed: {e}")
    print("Try: pip install --upgrade templateflow")
    print("Or check internet connection")

# Set custom download location
import templateflow.conf as tfconf
tfconf.TF_HOME = '/scratch/templates'  # Use fast local storage
```

### Missing Templates

```bash
# Update TemplateFlow to get latest templates
pip install --upgrade templateflow

# Manually clone template repository
git clone https://github.com/templateflow/templateflow.git
export TEMPLATEFLOW_HOME=/path/to/templateflow
```

---

## Related Tools and Integration

**Preprocessing:**
- **fMRIPrep** (Batch 5): Uses TemplateFlow by default
- **QSIPrep** (Batch 6): Template-based diffusion processing
- **ANTs** (Batch 1): Registration to templates

**Analysis:**
- **Nilearn** (Batch 2): Template loading and visualization
- **Pydra** (Batch 28): Template-based workflows
- **All analysis tools**: Benefit from standardized templates

**Containers:**
- **NeuroDocker** (Batch 28): Include templates in containers
- **Boutiques** (Batch 28): Specify templates in descriptors

---

## References

- Ciric, R., et al. (2022). TemplateFlow: FAIR-sharing of multi-scale, multi-species brain models. *Nature Methods*, 19, 1568–1571.
- Fonov, V., et al. (2011). Unbiased average age-appropriate atlases for pediatric studies. *NeuroImage*, 54(1), 313-327.
- Avants, B. B., et al. (2011). The optimal template effect in hippocampus studies of diseased populations. *NeuroImage*, 49(3), 2457-2466.
- Grabner, G., et al. (2006). Symmetric atlasing and model based segmentation: an application to the hippocampus in older adults. *MICCAI*, 9(Pt 2), 58-66.

**Official Website:** https://www.templateflow.org/
**GitHub Repository:** https://github.com/templateflow/templateflow
**Python Client:** https://github.com/templateflow/python-client
**Paper:** https://doi.org/10.1038/s41592-022-01681-2
**Archive:** https://osf.io/ue5gx/

## Citation

```bibtex
@article{ciric2022templateflow,
  title={TemplateFlow: A community archive of imaging templates and atlases},
  author={Ciric, Rastko and Rosen, Andrew and Moore, Theodoros M. and others},
  journal={Nature Methods},
  volume={19},
  pages={1568--1571},
  year={2022},
  doi={10.1038/s41592-022-01545-3}
}
```
