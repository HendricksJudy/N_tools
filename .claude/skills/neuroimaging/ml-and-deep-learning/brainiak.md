# BrainIAK (Brain Imaging Analysis Kit)

## Overview

BrainIAK (Brain Imaging Analysis Kit) is a high-performance Python package for advanced fMRI analysis developed by Intel Labs and Princeton Neuroscience Institute. It provides computationally efficient implementations of cutting-edge algorithms for multivariate pattern analysis, real-time fMRI, functional connectivity, and machine learning on neuroimaging data. BrainIAK emphasizes scalability with MPI parallelization and is optimized for high-performance computing (HPC) environments.

**Website:** https://brainiak.org/
**Platform:** Python (Linux/macOS)
**Language:** Python (with Cython optimization)
**License:** Apache 2.0

## Key Features

- Searchlight analysis with MPI parallelization
- Shared response modeling (SRM) for hyperalignment
- Full correlation matrix analysis (FCMA)
- Inter-subject correlation (ISC) and connectivity (ISFC)
- Representational similarity analysis (RSA)
- Real-time fMRI analysis and neurofeedback
- Event segmentation with Hidden Markov Models
- Template-based rotation (TBR) for small samples
- Topographic factor analysis (TFA)
- HPC-optimized with MPI and Cython
- Integration with scikit-learn
- GPU support for select algorithms

## Installation

### Using pip

```bash
# Basic installation
pip install brainiak

# With MPI support (recommended for HPC)
pip install brainiak[mpi]

# Development installation
git clone https://github.com/brainiak/brainiak
cd brainiak
pip install -e .
```

### Using conda

```bash
# Create environment
conda create -n brainiak python=3.9
conda activate brainiak

# Install BrainIAK
conda install -c conda-forge brainiak

# Install MPI dependencies
conda install -c conda-forge mpi4py openmpi
```

### MPI Setup for HPC

```bash
# Install OpenMPI (Ubuntu/Debian)
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# Install mpi4py
pip install mpi4py

# Test MPI installation
mpirun -n 4 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.rank}')"
```

### Verify Installation

```python
import brainiak
print(f"BrainIAK version: {brainiak.__version__}")

# Check available modules
import brainiak.searchlight
import brainiak.funcalign.srm
import brainiak.isc
print("BrainIAK modules imported successfully")
```

## Searchlight Analysis

### Basic Searchlight

```python
import numpy as np
from brainiak.searchlight.searchlight import Searchlight
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import nibabel as nib

# Load fMRI data
img = nib.load('sub-01_task-func_bold.nii.gz')
data = img.get_fdata()  # (x, y, z, time)

# Load brain mask
mask_img = nib.load('sub-01_brain_mask.nii.gz')
mask = mask_img.get_fdata().astype(bool)

# Load labels (experimental conditions)
labels = np.load('labels.npy')  # (time,)

# Define searchlight function
def searchlight_fn(X, y, sl_mask):
    """
    X: voxel x time
    y: labels
    sl_mask: boolean mask of searchlight sphere
    """
    # Classifier
    clf = SVC(kernel='linear')

    # Cross-validation
    scores = cross_val_score(clf, X.T, y, cv=3)

    return scores.mean()

# Create searchlight
sl = Searchlight(sl_rad=3, max_blk_edge=5)

# Run searchlight (single-core)
result = sl.run_searchlight(data, mask, searchlight_fn, labels)

print(f"Searchlight result shape: {result.shape}")

# Save results
result_img = nib.Nifti1Image(result, img.affine)
nib.save(result_img, 'searchlight_accuracy.nii.gz')
```

### MPI Parallelized Searchlight

```python
from brainiak.searchlight.searchlight import Searchlight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import nibabel as nib

# Load data
img = nib.load('sub-01_bold.nii.gz')
data = img.get_fdata()
mask = nib.load('sub-01_mask.nii.gz').get_fdata().astype(bool)
labels = np.load('labels.npy')

# Searchlight function
def decode_fn(X, y, sl_mask):
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X.T, y, cv=5)
    return scores.mean()

# Run with MPI
# Execute: mpirun -n 8 python searchlight_mpi.py
sl = Searchlight(sl_rad=3, max_blk_edge=4)
result = sl.run_searchlight(data, mask, decode_fn, labels)

# Save on rank 0
from mpi4py import MPI
comm = MPI.COMM_WORLD

if comm.rank == 0:
    result_img = nib.Nifti1Image(result, img.affine)
    nib.save(result_img, 'searchlight_mpi_result.nii.gz')
    print(f"Searchlight complete across {comm.size} processes")
```

### Custom Searchlight Metric

```python
import numpy as np
from scipy.stats import pearsonr
from brainiak.searchlight.searchlight import Searchlight

# Representational similarity in searchlight
def rsa_searchlight(X, model_rdm, sl_mask):
    """
    Compute RSA correlation with model RDM
    X: voxels x conditions
    model_rdm: model representational dissimilarity matrix
    """
    from scipy.spatial.distance import pdist, squareform

    # Compute neural RDM
    neural_rdm = pdist(X.T, metric='correlation')

    # Correlation with model RDM
    model_rdm_vec = squareform(model_rdm, checks=False)
    r, _ = pearsonr(neural_rdm, model_rdm_vec)

    return r

# Run RSA searchlight
model_rdm = np.load('model_rdm.npy')

sl = Searchlight(sl_rad=4)
rsa_result = sl.run_searchlight(
    data, mask, rsa_searchlight, model_rdm
)

print(f"RSA searchlight complete")
```

## Shared Response Modeling (SRM)

### Basic SRM for Hyperalignment

```python
from brainiak.funcalign.srm import SRM
import numpy as np

# Load data from multiple subjects
# Each subject: voxels x timepoints
subject_data = []
for subj_id in ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']:
    data = np.load(f'{subj_id}_roi_data.npy')
    subject_data.append(data)

print(f"Subjects: {len(subject_data)}")
print(f"Example shape: {subject_data[0].shape}")

# Fit SRM
n_features = 50  # Shared features

srm = SRM(n_iter=10, features=n_features)
srm.fit(subject_data)

# Get shared response
shared_response = srm.transform(subject_data)

print(f"Shared response shape per subject: {shared_response[0].shape}")
print(f"Shared response (features x time)")

# Get transformation matrices (voxels to shared space)
w_matrices = srm.w_
print(f"Number of transformation matrices: {len(w_matrices)}")
print(f"Example W shape (voxels x features): {w_matrices[0].shape}")
```

### Apply SRM to New Subject

```python
# Transform new subject to shared space
new_subject_data = np.load('sub-06_roi_data.npy')

# Project to shared space
new_shared = srm.transform([new_subject_data])[0]

print(f"New subject in shared space: {new_shared.shape}")

# Predict new subject data from shared response
# (For testing reconstruction quality)
predicted_data = srm.inverse_transform([new_shared])[0]

# Compute reconstruction accuracy
from scipy.stats import pearsonr
r_values = []
for v in range(new_subject_data.shape[0]):
    r, _ = pearsonr(new_subject_data[v, :], predicted_data[v, :])
    r_values.append(r)

mean_r = np.mean(r_values)
print(f"Mean reconstruction correlation: {mean_r:.3f}")
```

### Deterministic SRM

```python
from brainiak.funcalign.srm import DetSRM

# Deterministic SRM (faster, deterministic results)
det_srm = DetSRM(n_iter=10, features=50)
det_srm.fit(subject_data)

shared_det = det_srm.transform(subject_data)

print("Deterministic SRM complete")
print(f"Shared response: {shared_det[0].shape}")
```

## Inter-Subject Correlation (ISC)

### Basic ISC

```python
from brainiak.isc import isc
import numpy as np

# Load data: subjects x voxels x time
data = np.array([
    np.load('sub-01_data.npy'),
    np.load('sub-02_data.npy'),
    np.load('sub-03_data.npy'),
    np.load('sub-04_data.npy'),
    np.load('sub-05_data.npy')
])

print(f"Data shape: {data.shape}")  # (subjects, voxels, time)

# Compute ISC
isc_values = isc(data, pairwise=False, summary_statistic='median')

print(f"ISC shape: {isc_values.shape}")  # (voxels,)
print(f"Mean ISC: {np.mean(isc_values):.3f}")

# Save ISC map
import nibabel as nib
mask = nib.load('mask.nii.gz')
isc_vol = np.zeros(mask.shape)
isc_vol[mask.get_fdata().astype(bool)] = isc_values

isc_img = nib.Nifti1Image(isc_vol, mask.affine)
nib.save(isc_img, 'isc_map.nii.gz')
```

### ISC with Statistical Testing

```python
from brainiak.isc import isc, bootstrap_isc
import numpy as np

# Compute ISC
isc_values = isc(data, pairwise=False)

# Bootstrap for significance
observed, ci, p = bootstrap_isc(
    data,
    pairwise=False,
    summary_statistic='median',
    n_bootstraps=1000,
    ci_percentile=95
)

# Threshold by significance
significant_voxels = p < 0.05
print(f"Significant voxels: {significant_voxels.sum()}")

# FDR correction
from scipy.stats import false_discovery_control
p_fdr = false_discovery_control(p)
significant_fdr = p_fdr < 0.05

print(f"Significant voxels (FDR): {significant_fdr.sum()}")
```

### Inter-Subject Functional Connectivity (ISFC)

```python
from brainiak.isc import isfc
import numpy as np

# Compute ISFC
isfc_matrix = isfc(data, pairwise=False, summary_statistic='mean')

print(f"ISFC matrix shape: {isfc_matrix.shape}")  # (voxels, voxels)

# Visualize ISFC
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(isfc_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
plt.colorbar(label='ISFC')
plt.title('Inter-Subject Functional Connectivity')
plt.xlabel('Voxel')
plt.ylabel('Voxel')
plt.savefig('isfc_matrix.png', dpi=300)
```

### Time-Segment ISC

```python
from brainiak.isc import isc
import numpy as np

# Compute ISC in time windows
window_length = 20  # TRs
n_timepoints = data.shape[2]
n_windows = n_timepoints // window_length

isc_timecourse = []

for w in range(n_windows):
    start = w * window_length
    end = start + window_length

    # ISC for this window
    window_data = data[:, :, start:end]
    isc_w = isc(window_data, pairwise=False, summary_statistic='median')

    isc_timecourse.append(isc_w.mean())

# Plot ISC over time
plt.figure(figsize=(12, 4))
plt.plot(isc_timecourse, linewidth=2)
plt.xlabel('Time Window')
plt.ylabel('Mean ISC')
plt.title('ISC Time Course')
plt.grid(True, alpha=0.3)
plt.savefig('isc_timecourse.png', dpi=300)
```

## Real-Time fMRI

### Incremental GLM

```python
from brainiak.funcalign.rsrm import RSRM
import numpy as np

# Simulate real-time fMRI acquisition
n_voxels = 1000
n_timepoints_total = 200

# Design matrix (stimulus onsets)
design = np.zeros((n_timepoints_total, 2))
design[10:20, 0] = 1  # Condition 1
design[30:40, 1] = 1  # Condition 2

# Incremental GLM
betas = np.zeros((n_voxels, 2))
residuals = []

for t in range(n_timepoints_total):
    # New timepoint arrives
    new_data = np.random.randn(n_voxels)  # Simulated

    # Update GLM incrementally
    X = design[:t+1, :]
    y = np.concatenate([residuals, [new_data]])

    # Compute beta (online update possible with Kalman filter)
    if t > 5:  # Need minimum timepoints
        beta_t = np.linalg.lstsq(X, y, rcond=None)[0]
        betas = beta_t.T

    residuals.append(new_data)

    # Real-time analysis
    if t % 20 == 0:
        print(f"Timepoint {t}: Updated GLM")

print("Real-time GLM complete")
```

### Real-Time Neurofeedback Classifier

```python
from sklearn.linear_model import SGDClassifier
import numpy as np

# Initialize online classifier
clf = SGDClassifier(loss='log_loss', random_state=42)

# Training phase
X_train = np.random.randn(100, 1000)  # 100 TRs, 1000 voxels
y_train = np.random.randint(0, 2, 100)  # Binary conditions

# Train in mini-batches
clf.partial_fit(X_train, y_train, classes=[0, 1])

# Real-time neurofeedback loop
for t in range(50):
    # New TR acquired
    new_tr = np.random.randn(1, 1000)

    # Predict condition
    prediction = clf.predict(new_tr)[0]
    probability = clf.predict_proba(new_tr)[0, 1]

    # Provide feedback to participant
    if probability > 0.7:
        feedback_signal = "High"
    elif probability < 0.3:
        feedback_signal = "Low"
    else:
        feedback_signal = "Neutral"

    print(f"TR {t}: Prediction={prediction}, Prob={probability:.2f}, Feedback={feedback_signal}")

    # Optional: update classifier with labeled data
    # true_label = get_true_label()  # From behavioral response
    # clf.partial_fit(new_tr, [true_label])
```

## Event Segmentation

### Hidden Markov Model for Events

```python
from brainiak.eventseg.event import EventSegment
import numpy as np

# Load naturalistic fMRI data
data = np.load('movie_fmri_data.npy')  # (voxels, time)

# Event segmentation
n_events = 10  # Number of events to detect

seg = EventSegment(n_events=n_events)
seg.fit(data.T)  # Fit expects (time, voxels)

# Get event boundaries
event_patterns = seg.event_pat_  # (time, events)
event_labels = np.argmax(event_patterns, axis=1)

# Plot event boundaries
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Event probabilities
axes[0].imshow(event_patterns.T, aspect='auto', cmap='viridis')
axes[0].set_ylabel('Event')
axes[0].set_title('Event Probabilities Over Time')

# Event labels
axes[1].plot(event_labels, linewidth=2)
axes[1].set_xlabel('Time (TR)')
axes[1].set_ylabel('Event Label')
axes[1].set_title('Most Likely Event')

plt.tight_layout()
plt.savefig('event_segmentation.png', dpi=300)

# Detect boundaries
boundaries = np.where(np.diff(event_labels) != 0)[0] + 1
print(f"Event boundaries at TRs: {boundaries}")
```

### Cross-Subject Event Alignment

```python
from brainiak.eventseg.event import EventSegment
import numpy as np

# Multiple subjects watching same stimulus
subject_data = [
    np.load(f'sub-0{i}_movie_data.npy') for i in range(1, 6)
]

n_events = 8

# Fit on group average
group_data = np.mean(subject_data, axis=0)
seg = EventSegment(n_events=n_events)
seg.fit(group_data.T)

# Apply to each subject
subject_events = []
for data in subject_data:
    event_pat = seg.predict(data.T)
    subject_events.append(np.argmax(event_pat, axis=1))

# Compute event boundary consistency
boundaries_all = []
for events in subject_events:
    bounds = np.where(np.diff(events) != 0)[0] + 1
    boundaries_all.append(bounds)

print(f"Subjects: {len(boundaries_all)}")
for i, bounds in enumerate(boundaries_all):
    print(f"  Subject {i+1}: {len(bounds)} boundaries")
```

## Representational Similarity Analysis (RSA)

### Basic RSA

```python
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import numpy as np

# Load pattern data (conditions x voxels)
neural_data = np.load('neural_patterns.npy')  # (conditions, voxels)

# Compute neural RDM
neural_rdm = pdist(neural_data, metric='correlation')
neural_rdm_matrix = squareform(neural_rdm)

# Load model RDM
model_rdm = np.load('model_rdm.npy')  # (conditions, conditions)

# Compare neural and model RDM
neural_rdm_vec = neural_rdm_matrix[np.triu_indices(len(neural_data), k=1)]
model_rdm_vec = model_rdm[np.triu_indices(len(model_rdm), k=1)]

r, p = pearsonr(neural_rdm_vec, model_rdm_vec)

print(f"RSA: r = {r:.3f}, p = {p:.4f}")

# Visualize RDMs
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(neural_rdm_matrix, cmap='viridis')
axes[0].set_title('Neural RDM')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(model_rdm, cmap='viridis')
axes[1].set_title('Model RDM')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('rsa_comparison.png', dpi=300)
```

### Searchlight RSA

```python
from brainiak.searchlight.searchlight import Searchlight
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import numpy as np
import nibabel as nib

# Load model RDM
model_rdm_vec = pdist(np.load('model_rdm.npy'))

# Searchlight RSA function
def rsa_fn(X, model_rdm, sl_mask):
    """
    X: voxels x conditions
    """
    # Compute neural RDM
    neural_rdm = pdist(X.T, metric='correlation')

    # Correlation with model
    r, _ = pearsonr(neural_rdm, model_rdm)

    return r

# Load fMRI data (x, y, z, conditions)
data = nib.load('patterns_all.nii.gz').get_fdata()
mask = nib.load('mask.nii.gz').get_fdata().astype(bool)

# Run searchlight RSA
sl = Searchlight(sl_rad=3)
rsa_result = sl.run_searchlight(data, mask, rsa_fn, model_rdm_vec)

# Save
result_img = nib.Nifti1Image(rsa_result, affine=nib.load('mask.nii.gz').affine)
nib.save(result_img, 'searchlight_rsa.nii.gz')
```

## Full Correlation Matrix Analysis (FCMA)

### FCMA for Connectivity-Based Classification

```python
from brainiak.fcma.classifier import Classifier
from brainiak.fcma.preprocessing import prepare_fcma_data
import numpy as np

# Load data
data_list = []
labels_list = []

for subj in range(1, 6):
    data = np.load(f'sub-0{subj}_data.npy')  # (voxels, time)
    labels = np.load(f'sub-0{subj}_labels.npy')  # (time,)

    data_list.append(data)
    labels_list.append(labels)

# Prepare for FCMA
processed_data, labels_all = prepare_fcma_data(data_list, labels_list)

# Create FCMA classifier
clf = Classifier(svm_type='linear')

# Fit classifier
clf.fit(processed_data, labels_all)

# Get important connections
important_connections = clf.decision_function(processed_data)

print(f"FCMA classification complete")
print(f"Important connections shape: {important_connections.shape}")
```

## Integration with fMRIPrep

### Load fMRIPrep Outputs

```python
import numpy as np
import nibabel as nib
from nilearn import image, masking

# Load fMRIPrep preprocessed data
bold_file = 'sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
bold_img = nib.load(bold_file)

# Load brain mask
mask_file = 'sub-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
mask_img = nib.load(mask_file)

# Extract time series from mask
masked_data = masking.apply_mask(bold_img, mask_img)

print(f"Masked data shape: {masked_data.shape}")  # (time, voxels)

# Prepare for BrainIAK (voxels x time)
brainiak_data = masked_data.T

# Extract ROI data for SRM
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker

# Load atlas
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)

# Extract ROI time series
roi_data = masker.fit_transform(bold_img).T  # (ROIs, time)

print(f"ROI data for SRM: {roi_data.shape}")
```

### BrainIAK + fMRIPrep Pipeline

```python
import numpy as np
import nibabel as nib
from nilearn import image, masking
from brainiak.funcalign.srm import SRM

# Process multiple fMRIPrep subjects
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
roi_data_all = []

for subj in subjects:
    # Load preprocessed data
    bold_file = f'{subj}_task-movie_desc-preproc_bold.nii.gz'
    mask_file = f'{subj}_desc-brain_mask.nii.gz'

    # Extract ROI time series
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)
    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)

    roi_ts = masker.fit_transform(bold_file).T
    roi_data_all.append(roi_ts)

# Apply SRM
srm = SRM(n_iter=10, features=50)
srm.fit(roi_data_all)

shared_responses = srm.transform(roi_data_all)

print(f"Shared response per subject: {shared_responses[0].shape}")
```

## HPC Job Submission

### SLURM Script for Searchlight

```bash
#!/bin/bash
#SBATCH --job-name=brainiak_searchlight
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --output=searchlight_%j.out

module load python/3.9
module load openmpi/4.1

source activate brainiak

# Run MPI searchlight
mpirun -n 40 python searchlight_mpi.py

echo "Searchlight complete"
```

### PBS Script for SRM

```bash
#!/bin/bash
#PBS -N brainiak_srm
#PBS -l nodes=1:ppn=16
#PBS -l walltime=2:00:00
#PBS -l mem=32gb

cd $PBS_O_WORKDIR

module load anaconda3
source activate brainiak

python srm_analysis.py

echo "SRM complete"
```

## Integration with Claude Code

When helping users with BrainIAK:

1. **Environment Check:**
   ```python
   import brainiak
   print(f"BrainIAK version: {brainiak.__version__}")
   ```

2. **Data Format:** BrainIAK typically expects voxels × time (transpose of NiBabel)

3. **MPI Setup:** For HPC, ensure MPI is properly configured

4. **Common Issues:**
   - MPI import errors (install mpi4py)
   - Memory errors with searchlight (reduce radius or use MPI)
   - Dimension mismatches (check voxels × time vs time × voxels)
   - Installation issues on macOS (use Linux or conda)

5. **Performance:** Use MPI for searchlight, consider data chunking for large datasets

## Best Practices

- Use MPI parallelization for searchlight analysis
- Standardize/zscore data before SRM
- Apply appropriate multiple comparison correction
- Use appropriate number of SRM features (typically 50-100)
- Save intermediate results for long computations
- Monitor memory usage with large datasets
- Use ROI-based analysis when possible to reduce dimensionality
- Validate hyperalignment with left-out data
- Document all preprocessing steps
- Use consistent parcellations across subjects for SRM

## Troubleshooting

**Problem:** "MPI import error"
**Solution:** Install mpi4py: `pip install mpi4py`, ensure OpenMPI is installed

**Problem:** "Searchlight very slow"
**Solution:** Use MPI parallelization, reduce searchlight radius, use smaller mask

**Problem:** "SRM fit fails"
**Solution:** Check data dimensions (voxels × time), ensure consistent timepoints across subjects, reduce number of features

**Problem:** "Memory error"
**Solution:** Process data in chunks, use ROI-based analysis, reduce spatial resolution

## Resources

- BrainIAK Documentation: https://brainiak.org/docs/
- Tutorials: https://brainiak.org/tutorials/
- GitHub: https://github.com/brainiak/brainiak
- Paper: Kumar et al. (2020) "BrainIAK: The Brain Imaging Analysis Kit"
- Course Materials: https://brainiak.org/events/

## Related Tools

- **fMRIPrep:** Preprocessing pipeline (see `fmriprep.md`)
- **Nilearn:** Machine learning for neuroimaging (see `nilearn.md`)
- **PyMVPA:** Alternative MVPA package
- **scikit-learn:** Machine learning library
- **NiBabel:** Neuroimaging file I/O (see `nibabel.md`)
- **MNE-Python:** For MEG/EEG analysis (see `mne-python.md`)

## Citation

```bibtex
@article{kumar2020brainiak,
  title={BrainIAK: The Brain Imaging Analysis Kit},
  author={Kumar, Maya and Ellis, C. and Millman, K. and others},
  journal={Frontiers in Neuroinformatics},
  volume={14},
  pages={62},
  year={2020},
  doi={10.3389/fninf.2020.00062}
}
```
