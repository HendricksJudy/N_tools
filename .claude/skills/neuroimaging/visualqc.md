# VisualQC

## Overview

VisualQC is a Python tool for streamlined, efficient manual visual quality control of neuroimaging data. Developed by Pradeep Reddy Raamana, VisualQC provides interactive viewers for various neuroimaging modalities and processing stages with keyboard shortcuts, rating systems, and note-taking capabilities. It fills the critical gap between automated QC tools (like MRIQC) and the need for expert visual inspection, particularly for detecting subtle artifacts that automated methods might miss.

**Website:** https://raamana.github.io/visualqc/
**GitHub:** https://github.com/raamana/visualqc
**Platform:** Python
**Language:** Python
**License:** Apache License 2.0

## Key Features

- Interactive visual inspection interfaces
- T1w structural QC (raw and processed)
- FreeSurfer segmentation and parcellation QC
- fMRI preprocessing QC
- Functional connectivity QC
- Diffusion MRI and tractography QC
- Registration quality assessment
- Keyboard shortcuts for rapid review
- Multi-view displays (axial, sagittal, coronal)
- Rating scales (pass/fail/maybe/outlier)
- Note-taking and annotation
- Batch processing mode
- Resume interrupted sessions
- Export quality ratings
- Outlier flagging

## Installation

### Install via Pip

```bash
# Install latest version
pip install visualqc

# Install from GitHub (development version)
pip install git+https://github.com/raamana/visualqc.git

# Verify installation
visualqc --version
vqcT1 --help
```

### Dependencies

```bash
# VisualQC requires:
pip install numpy matplotlib nibabel
pip install PyQt5  # or PySide2

# Optional dependencies
pip install pandas seaborn
```

## T1w Quality Control

### Raw T1w Inspection

```bash
# Basic T1w QC
vqcT1 \
  --id_list subject_list.txt \
  --in_dir /data/raw_t1w \
  --out_dir /qc/t1w_ratings

# With specific subjects
vqcT1 \
  --id_list sub-01 sub-02 sub-03 \
  --in_dir /data/t1w \
  --out_dir /qc/ratings \
  --suffix _T1w.nii.gz
```

### Keyboard Shortcuts

```text
# Navigation
Right Arrow / n : Next subject
Left Arrow / p  : Previous subject
Home           : First subject
End            : Last subject

# Rating
q : Quit/Exit (saves ratings)
1 : Rate as "Good"
2 : Rate as "Questionable"
3 : Rate as "Bad"
0 : Rate as "Outlier"
u : Undo rating (mark as unrated)

# Visualization
+/- : Zoom in/out
r   : Reset view
c   : Toggle crosshair
```

### Review Session

```python
# Launch interactive T1w review
from visualqc import T1_mri

# Define subjects
subjects = ['sub-01', 'sub-02', 'sub-03']
in_dir = '/data/t1w/'
out_dir = '/qc/ratings/'

# Start review session
T1_mri.rate_t1_mri(
    id_list=subjects,
    in_dir=in_dir,
    out_dir=out_dir,
    views=[0, 1, 2],  # Axial, sagittal, coronal
    num_slices=12,
    num_rows=3
)
```

## FreeSurfer QC

### Segmentation Quality

```bash
# FreeSurfer segmentation QC
vqcFreeSurfer \
  --id_list subject_list.txt \
  --fs_dir /data/freesurfer \
  --out_dir /qc/freesurfer_ratings

# Review specific segmentations
vqcFreeSurfer \
  --id_list sub-01 sub-02 \
  --fs_dir /data/freesurfer \
  --out_dir /qc/fs_ratings \
  --views 0 1 2  # All three views
```

### Pial Surface Accuracy

```python
# Check pial surface accuracy
from visualqc import freesurfer

subjects = ['sub-01', 'sub-02']
fs_dir = '/data/freesurfer'
out_dir = '/qc/pial_ratings'

# Review pial surfaces
freesurfer.rate_freesurfer(
    id_list=subjects,
    fs_dir=fs_dir,
    out_dir=out_dir,
    type='pial_surface'  # Focus on pial surfaces
)
```

### White Matter Surface

```bash
# White matter surface QC
vqcFreeSurfer \
  --id_list subjects.txt \
  --fs_dir /fs \
  --out_dir /qc/wm_surface \
  --type wm_surface
```

### Subcortical Segmentation

```bash
# Subcortical structure segmentation QC
vqcFreeSurfer \
  --id_list subjects.txt \
  --fs_dir /fs \
  --out_dir /qc/subcortical \
  --type subcortical
```

## Functional MRI QC

### fMRI Preprocessing Review

```bash
# Review preprocessed fMRI
vqcFunc \
  --id_list subject_list.txt \
  --in_dir /data/func_preprocessed \
  --out_dir /qc/func_ratings \
  --task rest \
  --suffix _bold.nii.gz

# With motion parameters
vqcFunc \
  --id_list subjects.txt \
  --in_dir /preprocessed/func \
  --out_dir /qc/func \
  --confounds_dir /preprocessed/confounds \
  --fd_threshold 0.5
```

### Carpet Plot Review

```python
# Review BOLD timeseries with carpet plots
from visualqc import functional_mri

subjects = ['sub-01', 'sub-02']
func_dir = '/data/func/'
out_dir = '/qc/func_ratings/'

functional_mri.rate_func_mri(
    id_list=subjects,
    in_dir=func_dir,
    out_dir=out_dir,
    carpet_plot=True,  # Show carpet plot
    show_motion=True   # Display motion traces
)
```

### Temporal Artifacts

```bash
# Focus on temporal quality
vqcFunc \
  --id_list subjects.txt \
  --in_dir /func \
  --out_dir /qc/temporal \
  --carpet_plot \
  --show_spikes  # Highlight motion spikes
```

## Registration QC

### Anatomical-to-Template Alignment

```bash
# T1w to MNI registration QC
vqcReg \
  --id_list subject_list.txt \
  --in_dir /data/t1w_normalized \
  --template /templates/MNI152_T1_1mm.nii.gz \
  --out_dir /qc/registration

# Multiple subjects
vqcReg \
  --id_list sub-01 sub-02 sub-03 \
  --in_dir /normalized \
  --template MNI152 \
  --out_dir /qc/reg_ratings
```

### Functional-to-Anatomical

```bash
# BOLD to T1w registration QC
vqcReg \
  --id_list subjects.txt \
  --in_dir /func_to_anat \
  --reference_dir /anat \
  --out_dir /qc/func_to_anat \
  --type func_to_anat
```

### Overlay Visualization

```python
# Registration QC with custom overlays
from visualqc import alignment

subjects = ['sub-01', 'sub-02']
moving_dir = '/data/t1w_normalized/'
template = '/templates/MNI152_T1_1mm.nii.gz'
out_dir = '/qc/alignment/'

alignment.rate_alignment(
    id_list=subjects,
    in_dir=moving_dir,
    template=template,
    out_dir=out_dir,
    alpha_mixin=0.7,  # Overlay transparency
    contour_overlay=True  # Show contours
)
```

## Diffusion MRI QC

### DWI Quality Inspection

```bash
# Diffusion MRI QC
vqcDWI \
  --id_list subject_list.txt \
  --in_dir /data/dwi \
  --out_dir /qc/dwi_ratings \
  --bval_dir /data/bvals \
  --bvec_dir /data/bvecs
```

### Eddy Current Artifacts

```bash
# Check for eddy current distortions
vqcDWI \
  --id_list subjects.txt \
  --in_dir /dwi \
  --out_dir /qc/eddy \
  --focus eddy_artifacts
```

### Tractography QC

```bash
# Tractography plausibility
vqcTractography \
  --id_list subjects.txt \
  --tract_dir /tractography \
  --out_dir /qc/tracts \
  --anat_dir /anat  # Overlay on anatomy
```

## Rating Systems

### Rating Scales

```python
# Default rating scale:
# 1 = Good/Pass
# 2 = Questionable/Maybe
# 3 = Bad/Fail
# 0 = Outlier/Exclude

# Custom rating scale
custom_scale = {
    1: 'Excellent',
    2: 'Good',
    3: 'Acceptable',
    4: 'Poor',
    5: 'Unusable'
}
```

### Export Ratings

```bash
# Ratings saved automatically to:
# <out_dir>/ratings_<timestamp>.csv

# Example: /qc/t1w_ratings/ratings_20231117_143022.csv
```

```python
# Load and analyze ratings
import pandas as pd

ratings = pd.read_csv('ratings_20231117_143022.csv')

# Count by quality
print(ratings['rating'].value_counts())

# Failed subjects
failed = ratings[ratings['rating'] >= 3]
print(f"Failed: {len(failed)} subjects")
print(failed[['subject_id', 'rating', 'notes']])

# Export exclusion list
failed['subject_id'].to_csv('exclude_subjects.txt', index=False, header=False)
```

### Notes and Annotations

```python
# During review, press 'n' to add notes
# Notes saved with ratings

# Load notes
ratings = pd.read_csv('ratings.csv')
notes_df = ratings[ratings['notes'].notna()]

# Common issues
for idx, row in notes_df.iterrows():
    print(f"{row['subject_id']}: {row['notes']}")
```

## Batch Processing

### Organize Review Sessions

```python
# Batch review workflow
import visualqc
import pandas as pd

# Load subject list
subjects_df = pd.read_csv('all_subjects.csv')
subjects = subjects_df['subject_id'].tolist()

# Split into batches (for multiple raters)
batch_size = 50
batches = [subjects[i:i+batch_size] for i in range(0, len(subjects), batch_size)]

# Review batch 1
batch_1 = batches[0]

# Save batch list
with open('batch_1_subjects.txt', 'w') as f:
    f.write('\n'.join(batch_1))

# Review with VisualQC
# vqcT1 --id_list batch_1_subjects.txt --in_dir /data --out_dir /qc/batch1
```

### Resume Interrupted Session

```bash
# VisualQC automatically saves progress
# To resume, simply rerun with same output directory

vqcT1 \
  --id_list subjects.txt \
  --in_dir /data/t1w \
  --out_dir /qc/ratings  # Same output directory

# VisualQC will skip already-rated subjects
# Or prompt to re-review if desired
```

### Track Progress

```python
# Monitor QC progress
import pandas as pd
import glob

# Find all rating files
rating_files = glob.glob('/qc/*/ratings_*.csv')

# Load and combine
all_ratings = pd.concat([pd.read_csv(f) for f in rating_files])

# Progress summary
total_subjects = len(all_subjects)
rated_subjects = all_ratings['subject_id'].nunique()

print(f"Progress: {rated_subjects}/{total_subjects} ({rated_subjects/total_subjects*100:.1f}%)")

# Subjects remaining
all_subjects_set = set(all_subjects)
rated_subjects_set = set(all_ratings['subject_id'])
remaining = all_subjects_set - rated_subjects_set

print(f"Remaining: {len(remaining)} subjects")
with open('remaining_subjects.txt', 'w') as f:
    f.write('\n'.join(remaining))
```

## Multi-Rater Reliability

### Inter-Rater Agreement

```python
# Compare ratings from multiple raters
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load ratings from two raters
rater1 = pd.read_csv('qc/rater1/ratings.csv')
rater2 = pd.read_csv('qc/rater2/ratings.csv')

# Merge on subject_id
merged = rater1.merge(rater2, on='subject_id', suffixes=('_r1', '_r2'))

# Cohen's kappa
kappa = cohen_kappa_score(merged['rating_r1'], merged['rating_r2'])
print(f"Inter-rater agreement (Cohen's kappa): {kappa:.3f}")

# Percent agreement
agreement = (merged['rating_r1'] == merged['rating_r2']).mean()
print(f"Percent agreement: {agreement*100:.1f}%")

# Disagreements
disagreements = merged[merged['rating_r1'] != merged['rating_r2']]
print(f"Disagreements: {len(disagreements)} subjects")
print(disagreements[['subject_id', 'rating_r1', 'rating_r2']])
```

### Consensus Ratings

```python
# Resolve disagreements with third rater or consensus meeting

# For subjects with disagreement, get consensus
consensus_ratings = []

for idx, row in disagreements.iterrows():
    subject = row['subject_id']
    r1_rating = row['rating_r1']
    r2_rating = row['rating_r2']

    print(f"\nSubject: {subject}")
    print(f"Rater 1: {r1_rating}, Rater 2: {r2_rating}")

    # Manual consensus or third rater
    consensus = input("Consensus rating (1-3): ")
    consensus_ratings.append({
        'subject_id': subject,
        'rating': int(consensus)
    })

# Combine with agreements
agreements = merged[merged['rating_r1'] == merged['rating_r2']]
final_ratings = pd.concat([
    agreements[['subject_id', 'rating_r1']].rename(columns={'rating_r1': 'rating'}),
    pd.DataFrame(consensus_ratings)
])

# Save final ratings
final_ratings.to_csv('final_consensus_ratings.csv', index=False)
```

## Integration with MRIQC

### Combine Automated and Manual QC

```python
# Combine MRIQC automated metrics with VisualQC manual ratings

# Load MRIQC metrics
mriqc = pd.read_csv('mriqc/group_T1w.tsv', sep='\t')
mriqc['subject_id'] = mriqc['bids_name'].str.extract(r'(sub-[^_]+)')

# Load VisualQC ratings
visualqc_ratings = pd.read_csv('visualqc/ratings.csv')

# Merge
qc_combined = mriqc.merge(visualqc_ratings, on='subject_id')

# Analyze relationship
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(['snr_total', 'cnr', 'fber']):
    sns.boxplot(data=qc_combined, x='rating', y=metric, ax=axes[i])
    axes[i].set_xlabel('Visual Rating (1=Good, 3=Bad)')
    axes[i].set_ylabel(metric)
    axes[i].set_title(f'{metric} by Visual Rating')

plt.tight_layout()
plt.savefig('mriqc_vs_visualqc.png')

# Identify discrepancies
# High automated metrics but bad visual rating
high_snr_bad_visual = qc_combined[
    (qc_combined['snr_total'] > qc_combined['snr_total'].median()) &
    (qc_combined['rating'] == 3)
]

print(f"Good automated QC but bad visual: {len(high_snr_bad_visual)}")
```

## FreeSurfer Workflow Integration

### Complete FreeSurfer QC Pipeline

```bash
# 1. Run FreeSurfer recon-all
for sub in sub-01 sub-02 sub-03; do
    recon-all -s ${sub} -i /data/${sub}/anat/${sub}_T1w.nii.gz -all
done

# 2. Visual QC of segmentations
vqcFreeSurfer \
  --id_list sub-01 sub-02 sub-03 \
  --fs_dir $SUBJECTS_DIR \
  --out_dir /qc/freesurfer

# 3. Review ratings and identify failures
python check_fs_qc.py

# 4. Reprocess failed subjects
# Edit control points, rerun recon-all
```

```python
# check_fs_qc.py
import pandas as pd

ratings = pd.read_csv('/qc/freesurfer/ratings.csv')

failed = ratings[ratings['rating'] >= 3]

print("Failed FreeSurfer segmentations:")
for idx, row in failed.iterrows():
    print(f"  {row['subject_id']}: {row['notes']}")

# Save for manual editing
failed['subject_id'].to_csv('fs_failures.txt', index=False, header=False)
```

## Custom Visualization Settings

### Adjust Display Parameters

```python
# Custom visualization parameters
from visualqc import T1_mri

# More slices for detailed review
T1_mri.rate_t1_mri(
    id_list=subjects,
    in_dir='/data/t1w',
    out_dir='/qc/detailed',
    num_slices=24,  # More slices (default: 12)
    num_rows=4,     # More rows
    alpha_mixin=0.8  # Transparency for overlays
)
```

### Color Maps

```python
# Use different color maps
T1_mri.rate_t1_mri(
    id_list=subjects,
    in_dir='/data/t1w',
    out_dir='/qc/ratings',
    cmap='gray'  # Options: 'gray', 'hot', 'viridis', etc.
)
```

## Troubleshooting

**Problem:** Qt/GUI errors
**Solution:** Install PyQt5 or PySide2, check DISPLAY variable for remote sessions

**Problem:** Images not loading
**Solution:** Verify file paths, check nibabel compatibility, ensure NIfTI format

**Problem:** Slow performance
**Solution:** Reduce `num_slices`, use local storage (not network), downsample large images

**Problem:** Can't save ratings
**Solution:** Check write permissions on output directory, ensure disk space available

**Problem:** Resume not working
**Solution:** Use exact same output directory, check for corrupted rating files

## Best Practices

1. **Training Raters:**
   - Create training set with known quality
   - Establish rating criteria
   - Practice sessions before real QC
   - Regular calibration

2. **Systematic Review:**
   - Review in batches
   - Take breaks to avoid fatigue
   - Randomize subject order
   - Blind to group assignments

3. **Documentation:**
   - Use notes field liberally
   - Document common issues
   - Keep QC protocol
   - Track rating criteria

4. **Quality Assurance:**
   - Multiple raters for subset
   - Inter-rater reliability checks
   - Consensus meetings
   - Combine with automated QC

5. **Workflow Integration:**
   - QC before preprocessing
   - Re-QC after processing
   - Document all exclusions
   - Version control ratings

## Resources

- **Documentation:** https://raamana.github.io/visualqc/
- **GitHub:** https://github.com/raamana/visualqc
- **Paper:** Raamana & Strother (2018). Journal of Open Source Software
- **Tutorial:** https://raamana.github.io/visualqc/tutorial.html
- **Issues:** https://github.com/raamana/visualqc/issues

## Citation

```bibtex
@article{raamana2018visualqc,
  title={VisualQC: Manual quality control for neuroimaging pipelines},
  author={Raamana, Pradeep Reddy and Strother, Stephen C},
  journal={Journal of Open Source Software},
  volume={3},
  number={28},
  pages={1004},
  year={2018}
}
```

## Related Tools

- **MRIQC:** Automated quality metrics
- **FreeSurfer:** Anatomical preprocessing
- **fMRIPrep:** Functional preprocessing
- **QSIPrep:** Diffusion preprocessing
- **ENIGMA QC:** Standardized QC protocols
- **Mindcontrol:** Web-based QC platform
