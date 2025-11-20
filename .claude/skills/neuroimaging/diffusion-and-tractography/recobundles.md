# Recobundles

## Overview

**Recobundles** is an atlas-based fiber bundle recognition method integrated into DIPY for automatically identifying and extracting white matter bundles from whole-brain tractography. Using streamline-based registration (SLR) and clustering techniques, Recobundles enables reproducible bundle extraction without manual ROI placement, making it valuable for standardized tractography analysis across subjects and studies.

The method works by registering a subject's whole-brain tractogram to a bundle atlas, identifying streamlines that match known anatomical bundles, and extracting them with quality control metrics. This automated approach reduces inter-rater variability and enables large-scale tractography studies with consistent bundle definitions.

**Key Use Cases:**
- Automated white matter bundle extraction
- Reproducible tract-specific analysis across subjects
- Large-scale tractography studies
- Clinical tract comparison (patients vs. controls)
- Atlas-based bundle standardization
- Tract-specific microstructure analysis

**Website:** https://dipy.org/documentation/latest/examples_built/bundle_extraction/
**DIPY Documentation:** https://dipy.org/documentation/latest/
**Source Code:** https://github.com/dipy/dipy (integrated)

---

## Key Features

- **Atlas-Based Recognition:** Automatic bundle identification using template bundles
- **Streamline Registration (SLR):** Align subject tractography to atlas
- **No Manual ROIs:** Eliminates need for manual region placement
- **Multiple Atlases:** Support for various bundle atlases (RecobundlesX, etc.)
- **Quality Control:** Automated metrics for bundle quality
- **Reproducible:** Consistent bundle extraction across subjects
- **DIPY Integration:** Seamless use with DIPY tractography
- **Flexible Thresholds:** Adjustable similarity criteria
- **Batch Processing:** Process multiple subjects efficiently
- **Bundle Refinement:** Iterative improvement of extracted bundles
- **Cross-Subject Comparison:** Enable standardized analyses
- **Open Source:** Part of DIPY ecosystem

---

## Installation

Recobundles is part of DIPY:

```bash
# Install DIPY
pip install dipy

# Or with conda
conda install -c conda-forge dipy

# Verify
python -c "import dipy; print(dipy.__version__)"
```

---

## Basic Bundle Extraction

### Load Tractography

```python
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import Streamlines
import numpy as np

# Load whole-brain tractogram
tractogram = load_tractogram(
    'whole_brain_tractography.trk',
    'same',
    bbox_valid_check=False
)

streamlines = tractogram.streamlines
print(f"Loaded {len(streamlines)} streamlines")
```

### Load Bundle Atlas

```python
from dipy.data import fetch_bundle_atlas_hcp842

# Fetch HCP bundle atlas
atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

# Load atlas bundle (e.g., arcuate fasciculus left)
atlas_bundle = load_tractogram(
    f'{atlas_folder}/Atlas_80_Bundles/whole_brain/AF_L.trk',
    'same'
).streamlines

print(f"Atlas bundle: {len(atlas_bundle)} streamlines")
```

### Extract Bundle

```python
from dipy.segment.bundles import RecoBundles

# Create RecoBundles object
rb = RecoBundles(streamlines, cluster_map=None, clust_thr=10)

# Recognize bundle
recognized_bundle, labels = rb.recognize(
    model_bundle=atlas_bundle,
    model_clust_thr=5,
    reduction_thr=10,
    reduction_distance='mdf',
    slr=True,
    slr_metric='symmetric',
    pruning_thr=5
)

print(f"Extracted {len(recognized_bundle)} streamlines")
```

---

## Streamline-Based Registration (SLR)

### Basic SLR

```python
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points

# Resample streamlines to same number of points
static = set_number_of_points(atlas_bundle, nb_points=20)
moving = set_number_of_points(Streamlines(streamlines[:1000]), nb_points=20)

# Perform SLR
srr = StreamlineLinearRegistration()
srm = srr.optimize(static=static, moving=moving)

# Apply transform
moved_streamlines = srm.transform(moving)
```

### SLR for Bundle Recognition

```python
# SLR is automatically used in RecoBundles with slr=True
rb = RecoBundles(streamlines, cluster_map=None, clust_thr=10)

recognized, labels = rb.recognize(
    model_bundle=atlas_bundle,
    slr=True,  # Enable SLR
    slr_metric='symmetric'  # Or 'asymmetric'
)
```

---

## Multiple Bundle Extraction

### Extract All Major Bundles

```python
import os

# Define bundles to extract
bundle_names = [
    'AF_L', 'AF_R',  # Arcuate fasciculus
    'CST_L', 'CST_R',  # Corticospinal tract
    'CC',  # Corpus callosum
    'UF_L', 'UF_R',  # Uncinate fasciculus
    'ILF_L', 'ILF_R',  # Inferior longitudinal fasciculus
]

extracted_bundles = {}

for bundle_name in bundle_names:
    print(f"\nExtracting {bundle_name}...")

    # Load atlas bundle
    atlas_path = f'{atlas_folder}/Atlas_80_Bundles/whole_brain/{bundle_name}.trk'

    if not os.path.exists(atlas_path):
        print(f"  Atlas not found: {bundle_name}")
        continue

    atlas_bundle = load_tractogram(atlas_path, 'same').streamlines

    # Extract
    rb = RecoBundles(streamlines, cluster_map=None, clust_thr=10)
    recognized, labels = rb.recognize(
        model_bundle=atlas_bundle,
        model_clust_thr=5,
        reduction_thr=10,
        slr=True
    )

    extracted_bundles[bundle_name] = recognized
    print(f"  Extracted: {len(recognized)} streamlines")
```

### Save Extracted Bundles

```python
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space

for bundle_name, bundle_streamlines in extracted_bundles.items():
    if len(bundle_streamlines) == 0:
        continue

    # Create StatefulTractogram
    sft = StatefulTractogram(
        bundle_streamlines,
        tractogram.affine,
        Space.RASMM
    )

    # Save
    output_file = f'bundles/{bundle_name}_extracted.trk'
    os.makedirs('bundles', exist_ok=True)
    save_tractogram(sft, output_file)

    print(f"Saved: {output_file}")
```

---

## Bundle Refinement

### Refine Extracted Bundle

```python
# Use refine method for iterative improvement
rb = RecoBundles(streamlines, cluster_map=None, clust_thr=10)

# Initial recognition
recognized, labels = rb.recognize(
    model_bundle=atlas_bundle,
    slr=True
)

# Refine (use recognized bundle as new model)
refined_bundle = rb.refine(
    model_bundle=atlas_bundle,
    pruning_thr=6,
    reduction_thr=10
)

print(f"Initial: {len(recognized)}, Refined: {len(refined_bundle)}")
```

---

## Quality Control

### Calculate Bundle Metrics

```python
from dipy.segment.metric import mdf

# Mean distance flip (MDF) between bundles
distance = mdf(recognized_bundle, atlas_bundle)
print(f"MDF distance: {distance:.3f}")

# Coverage (percentage of atlas covered)
coverage = len(recognized_bundle) / len(atlas_bundle) * 100
print(f"Coverage: {coverage:.1f}%")
```

### Visualize Extracted Bundle

```python
from dipy.viz import window, actor

# Create scene
scene = window.Scene()

# Add extracted bundle
bundle_actor = actor.line(recognized_bundle, colors=(1, 0, 0))
scene.add(bundle_actor)

# Add atlas for comparison
atlas_actor = actor.line(atlas_bundle, colors=(0, 1, 0))
scene.add(atlas_actor)

# Show
window.show(scene)

# Or save screenshot
window.record(scene, out_path='bundle_comparison.png', size=(800, 800))
```

---

## Tract-Specific Analysis

### Calculate FA Along Bundle

```python
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
import nibabel as nib

# Load FA map
fa_img = nib.load('fa.nii.gz')
fa_data = fa_img.get_fdata()
affine = fa_img.affine

# Sample FA along streamlines
def sample_fa_along_streamlines(streamlines, fa_data, affine):
    """Sample FA values along each streamline."""
    from dipy.tracking.utils import values_from_volume

    fa_values = values_from_volume(
        fa_data,
        streamlines,
        affine=affine
    )

    return fa_values

# Get FA along bundle
fa_along_bundle = sample_fa_along_streamlines(
    recognized_bundle,
    fa_data,
    affine
)

# Calculate mean FA per streamline
mean_fa_per_streamline = [np.mean(fa) for fa in fa_along_bundle]

# Overall bundle FA
bundle_mean_fa = np.mean(mean_fa_per_streamline)
bundle_std_fa = np.std(mean_fa_per_streamline)

print(f"Bundle FA: {bundle_mean_fa:.3f} ± {bundle_std_fa:.3f}")
```

### Along-Tract Analysis

```python
from dipy.stats.analysis import afq_profile

# Calculate FA profile along tract
fa_profile = afq_profile(
    fa_data,
    recognized_bundle,
    affine,
    n_points=100  # Sample 100 points along tract
)

# Visualize profile
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(fa_profile, linewidth=2)
plt.xlabel('Position along tract')
plt.ylabel('FA')
plt.title('FA profile along arcuate fasciculus')
plt.grid(True)
plt.savefig('fa_profile.png', dpi=300)
```

---

## Batch Processing

### Process Multiple Subjects

```python
from pathlib import Path

def extract_bundles_for_subject(subject_dir, atlas_folder, output_dir, bundle_names):
    """Extract bundles for single subject."""

    subject_id = subject_dir.name
    print(f"\n### Processing {subject_id} ###")

    # Load tractography
    tract_file = subject_dir / 'tractography.trk'
    if not tract_file.exists():
        print(f"  Skipping: tractography not found")
        return

    tractogram = load_tractogram(str(tract_file), 'same')
    streamlines = tractogram.streamlines

    # Output directory
    subj_output = Path(output_dir) / subject_id
    subj_output.mkdir(parents=True, exist_ok=True)

    # Extract each bundle
    for bundle_name in bundle_names:
        # Load atlas
        atlas_path = f'{atlas_folder}/Atlas_80_Bundles/whole_brain/{bundle_name}.trk'
        if not os.path.exists(atlas_path):
            continue

        atlas_bundle = load_tractogram(atlas_path, 'same').streamlines

        # Extract
        rb = RecoBundles(streamlines, cluster_map=None, clust_thr=10)
        recognized, _ = rb.recognize(
            model_bundle=atlas_bundle,
            slr=True,
            model_clust_thr=5
        )

        # Save
        if len(recognized) > 0:
            sft = StatefulTractogram(recognized, tractogram.affine, Space.RASMM)
            output_file = subj_output / f'{bundle_name}.trk'
            save_tractogram(sft, str(output_file))

            print(f"  {bundle_name}: {len(recognized)} streamlines")

# Batch process
subjects_dir = Path('/data/tractography')
subjects = sorted(subjects_dir.glob('sub-*'))

bundle_names = ['AF_L', 'AF_R', 'CST_L', 'CST_R']

for subject_dir in subjects:
    extract_bundles_for_subject(
        subject_dir,
        atlas_folder,
        '/data/extracted_bundles',
        bundle_names
    )
```

---

## Integration with Claude Code

```python
# recobundles_pipeline.py - Automated bundle extraction pipeline

from pathlib import Path
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.bundles import RecoBundles
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import logging

class RecobundlesPipeline:
    """Automated Recobundles bundle extraction."""

    def __init__(self, atlas_folder):
        self.atlas_folder = Path(atlas_folder)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_subject_bundles(self, tractography_file, output_dir, bundles):
        """Extract specified bundles for subject."""

        self.logger.info(f"Loading: {tractography_file}")

        # Load tractogram
        tractogram = load_tractogram(str(tractography_file), 'same')
        streamlines = tractogram.streamlines

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for bundle_name in bundles:
            self.logger.info(f"Extracting: {bundle_name}")

            # Load atlas bundle
            atlas_file = self.atlas_folder / 'whole_brain' / f'{bundle_name}.trk'

            if not atlas_file.exists():
                self.logger.warning(f"Atlas not found: {bundle_name}")
                continue

            atlas_bundle = load_tractogram(str(atlas_file), 'same').streamlines

            # Extract
            rb = RecoBundles(streamlines, cluster_map=None, clust_thr=10)
            recognized, _ = rb.recognize(
                model_bundle=atlas_bundle,
                slr=True
            )

            # Save
            if len(recognized) > 0:
                sft = StatefulTractogram(recognized, tractogram.affine, Space.RASMM)
                output_file = output_dir / f'{bundle_name}.trk'
                save_tractogram(sft, str(output_file))

                results[bundle_name] = len(recognized)
                self.logger.info(f"  Saved {len(recognized)} streamlines")

        return results

# Usage
pipeline = RecobundlesPipeline(atlas_folder='/path/to/atlas')

results = pipeline.extract_subject_bundles(
    tractography_file='/data/sub-01/whole_brain.trk',
    output_dir='/data/bundles/sub-01',
    bundles=['AF_L', 'AF_R', 'CST_L', 'CST_R']
)
```

---

## Integration with Other Tools

### Integration with TractoFlow

```python
# Use TractoFlow tractography outputs
from pathlib import Path

def process_tractoflow_output(tractoflow_dir, subject):
    """Extract bundles from TractoFlow outputs."""

    subj_dir = Path(tractoflow_dir) / subject / 'Tracking'

    # TractoFlow output tractogram
    tract_file = subj_dir / f'{subject}__tracking.trk'

    tractogram = load_tractogram(str(tract_file), 'same')
    streamlines = tractogram.streamlines

    # Extract bundles using Recobundles
    # ...

# Process
process_tractoflow_output('/data/tractoflow_output', 'sub-01')
```

### Integration with MRtrix3

```python
# Convert TCK to TRK for Recobundles
import subprocess

def convert_tck_to_trk(tck_file, reference_file, trk_file):
    """Convert MRtrix TCK to TRK format."""

    cmd = [
        'tckconvert',
        tck_file,
        trk_file,
        '-force',
        '-scanner2voxel', reference_file
    ]

    subprocess.run(cmd, check=True)

# Use
convert_tck_to_trk('tracks.tck', 'fa.nii.gz', 'tracks.trk')

# Then use with Recobundles
tractogram = load_tractogram('tracks.trk', 'same')
```

---

## Troubleshooting

### Problem 1: Low Bundle Extraction

**Symptoms:** Very few streamlines extracted

**Solution:**
```python
# Reduce threshold
rb = RecoBundles(streamlines, cluster_map=None, clust_thr=10)
recognized, _ = rb.recognize(
    model_bundle=atlas_bundle,
    reduction_thr=15,  # Increase (less strict)
    pruning_thr=8      # Increase
)
```

### Problem 2: Too Many False Positives

**Symptoms:** Extracted bundle contains wrong fibers

**Solution:**
```python
# Increase strictness
recognized, _ = rb.recognize(
    model_bundle=atlas_bundle,
    reduction_thr=8,   # Decrease (more strict)
    pruning_thr=4,     # Decrease
    slr=True           # Enable SLR
)
```

### Problem 3: SLR Fails

**Symptoms:** Error during streamline registration

**Solution:**
```python
# Disable SLR or use different metric
recognized, _ = rb.recognize(
    model_bundle=atlas_bundle,
    slr=False  # Disable SLR
)

# Or try different metric
recognized, _ = rb.recognize(
    model_bundle=atlas_bundle,
    slr=True,
    slr_metric='asymmetric'  # Instead of 'symmetric'
)
```

---

## Best Practices

### 1. Atlas Selection

- Use atlas matching your population (age, pathology)
- Verify atlas quality before use
- Consider creating custom atlas for specific studies

### 2. Quality Control

- Visual inspection of extracted bundles
- Check bundle metrics (size, shape)
- Compare across subjects for consistency
- Validate against anatomical knowledge

### 3. Parameter Tuning

- Start with default parameters
- Adjust thresholds based on results
- Document parameter choices
- Use same parameters across subjects

### 4. Reproducibility

- Version DIPY and Recobundles
- Save exact parameters used
- Archive atlas bundles
- Document any manual refinements

---

## Resources

### Documentation

- **DIPY Recobundles:** https://dipy.org/documentation/latest/examples_built/bundle_extraction/
- **DIPY:** https://dipy.org/
- **GitHub:** https://github.com/dipy/dipy

### Publications

- **RecoBundles:** Garyfallidis et al. (2018) "Recognition of white matter bundles using local and global streamline-based registration and clustering" *NeuroImage*

---

## Citation

```bibtex
@article{garyfallidis2018recognition,
  title={Recognition of white matter bundles using local and global streamline-based registration and clustering},
  author={Garyfallidis, Eleftherios and Côté, Marc-Alexandre and Rheault, François and Sidhu, Jasmeen and Hau, Janice and Petit, Laurent and Fortin, David and Cunanne, Stephen and Descoteaux, Maxime},
  journal={NeuroImage},
  volume={170},
  pages={283--295},
  year={2018},
  publisher={Elsevier}
}
```

---

## Related Tools

- **DIPY:** Foundation for diffusion processing (see `dipy.md`)
- **TractSeg:** Deep learning bundle segmentation (see `tractseg.md`)
- **TractoFlow:** Automated tractography pipeline (see `tractoflow.md`)
- **MRtrix3:** Tractography generation (see `mrtrix3.md`)
- **Surfice:** Bundle visualization (see `surfice.md`)

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**DIPY Version Covered:** 1.7.x+
**Maintainer:** Claude Code Neuroimaging Skills
