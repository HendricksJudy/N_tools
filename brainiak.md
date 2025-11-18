# BrainIAK: Brain Imaging Analysis Kit

## Overview

BrainIAK (Brain Imaging Analysis Kit) is a Python package providing advanced machine learning and statistical methods specifically designed for neuroimaging data. While general-purpose ML libraries like scikit-learn are powerful, they don't account for the unique properties of brain imaging data: extreme high dimensionality (hundreds of thousands of voxels), strong spatial and temporal autocorrelation, inter-subject variability, and computational demands of whole-brain analysis. BrainIAK implements state-of-the-art algorithms optimized for these challenges, including real-time fMRI analysis, inter-subject correlation for naturalistic stimuli, shared response modeling for functional alignment, searchlight MVPA, and event segmentation.

BrainIAK emerged from the Intel-Princeton collaboration and leverages high-performance computing techniques, including MPI parallelization and optimized linear algebra, to make sophisticated analyses tractable on modern HPC systems. The toolkit is particularly valuable for studies using naturalistic stimuli (movies, narratives, music), real-time adaptive paradigms, multivariate pattern analysis across subjects, and large-scale neuroimaging datasets requiring efficient computation.

**Key Features:**
- Inter-subject correlation (ISC) for naturalistic neuroimaging
- Inter-subject functional correlation (ISFC) for network synchronization
- Shared Response Model (SRM) for functional alignment across subjects
- Searchlight multivariate pattern analysis (MVPA) with efficient parallelization
- Event segmentation using Hidden Markov Models
- Full Correlation Matrix Analysis (FCMA) for whole-brain connectivity
- Topographic Factor Analysis (TFA) for spatiotemporal decomposition
- Real-time fMRI infrastructure for closed-loop experiments
- MPI parallelization for HPC environments
- Optimized for memory efficiency and computational speed

**Primary Use Cases:**
- Analyzing naturalistic stimuli experiments (movies, stories, music)
- Real-time fMRI neurofeedback and adaptive paradigms
- Cross-subject functional alignment and decoding
- Searchlight MVPA for representational similarity
- Event detection in continuous brain activity
- High-performance neuroimaging on compute clusters
- Cognitive neuroscience with complex experimental designs

**Citation:**
```
Kumar, M., Ellis, C. T., Lu, Q., Zhang, H., Capotă, M., Willke, T. L., ... &
Norman, K. A. (2020). BrainIAK: The Brain Imaging Analysis Kit. Aperture
Neuro, 1, 1-14.
```

## Installation

### Conda Installation (Recommended)

```bash
# Create environment with BrainIAK
conda create -n brainiak python=3.8
conda activate brainiak

# Install BrainIAK
conda install -c brainiak -c defaults -c conda-forge brainiak

# Verify installation
python -c "import brainiak; print(brainiak.__version__)"

# Expected output: 0.11 or later
```

### Pip Installation

```bash
# Install via pip
pip install brainiak

# For MPI support (HPC usage)
# First install MPI library (e.g., OpenMPI)
sudo apt-get install libopenmpi-dev openmpi-bin  # Ubuntu/Debian

# Install mpi4py
pip install mpi4py

# Test MPI installation
mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

### Docker Installation

```bash
# Pull BrainIAK Docker image
docker pull brainiak/brainiak

# Run BrainIAK in container
docker run -it --rm \
  -v $(pwd):/work \
  brainiak/brainiak \
  python /work/my_analysis.py
```

### Dependencies

```bash
# Core dependencies (automatically installed)
# - NumPy, SciPy (numerical computing)
# - scikit-learn (ML utilities)
# - Nilearn (neuroimaging utilities)
# - nibabel (neuroimaging I/O)
# - mpi4py (parallel processing, optional)

# Verify all dependencies
python << EOF
import numpy
import scipy
import sklearn
import nilearn
import nibabel
import brainiak
print("All dependencies installed successfully")
EOF
```

## Inter-Subject Correlation (ISC)

**Example 1: Basic ISC Analysis**

```python
from brainiak.isc import isc
import numpy as np

# Load fMRI data from multiple subjects watching the same movie
# Data shape: (n_subjects, n_timepoints, n_voxels)
# Example: (20 subjects, 300 timepoints, 50000 voxels)

subjects = []
for i in range(1, 21):
    data = np.load(f'sub-{i:02d}_movie_bold.npy')  # (300, 50000)
    subjects.append(data)

data = np.array(subjects)  # (20, 300, 50000)
print(f"Data shape: {data.shape}")

# Compute ISC (correlation of each subject with mean of others)
isc_values = isc(data, pairwise=False)

# isc_values shape: (n_voxels,) - ISC for each voxel
print(f"ISC shape: {isc_values.shape}")
print(f"Mean ISC: {np.nanmean(isc_values):.3f}")

# High ISC indicates brain regions that respond similarly across subjects
# Useful for naturalistic stimuli (movies, stories, music)
```

**Example 2: Statistical Testing for ISC**

```python
from brainiak.isc import isc, permutation_isc
import numpy as np

# Compute observed ISC
data = np.random.randn(20, 300, 1000)  # (subjects, TRs, voxels)
observed_isc = isc(data)

# Permutation test (shuffle subject labels)
n_permutations = 1000
null_distribution, p_values = permutation_isc(
    data,
    pairwise=False,
    n_permutations=n_permutations
)

# FDR correction
from scipy.stats import false_discovery_control
p_fdr = false_discovery_control(p_values, method='bh')

# Identify significant voxels
significant_voxels = p_fdr < 0.05
print(f"Significant voxels: {significant_voxels.sum()} / {len(p_values)}")

# Visualize ISC map
import nibabel as nib
from nilearn import plotting

# Reshape to brain volume (assumes you have a mask)
# isc_brain = np.zeros(brain_shape)
# isc_brain[mask] = observed_isc
# isc_img = nib.Nifti1Image(isc_brain, affine)
# plotting.plot_stat_map(isc_img, threshold=0.1, title='ISC Map')
```

**Example 3: ROI-based ISC**

```python
from brainiak.isc import isc
import numpy as np
from nilearn import datasets, input_data

# Load atlas (e.g., Schaefer 400 parcellation)
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400)
masker = input_data.NiftiLabelsMasker(labels_img=atlas.maps)

# Extract ROI timeseries for all subjects
roi_timeseries_all = []
for i in range(1, 21):
    func_img = nib.load(f'sub-{i:02d}_movie_bold.nii.gz')
    roi_ts = masker.fit_transform(func_img)  # (timepoints, 400)
    roi_timeseries_all.append(roi_ts)

data = np.array(roi_timeseries_all)  # (20, timepoints, 400)

# Compute ROI-level ISC
roi_isc = isc(data, pairwise=False)

# Identify ROIs with high synchronization
high_isc_rois = np.where(roi_isc > np.percentile(roi_isc, 90))[0]
print(f"High ISC ROIs: {high_isc_rois}")

# Often: visual cortex, auditory cortex, language areas for movies
```

## Inter-Subject Functional Correlation (ISFC)

**Example 4: Computing ISFC**

```python
from brainiak.isfc import isfc
import numpy as np

# Load ROI timeseries (shape: subjects x timepoints x ROIs)
data = np.random.randn(15, 250, 100)  # Example data

# Compute ISFC (connectivity synchronized across subjects)
isfc_matrix = isfc(data, pairwise=False)

# isfc_matrix shape: (n_rois, n_rois) = (100, 100)
# isfc_matrix[i, j] = ISC of connectivity between ROI i and ROI j

print(f"ISFC shape: {isfc_matrix.shape}")

# Visualize ISFC matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(isfc_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='ISFC')
plt.title('Inter-Subject Functional Correlation')
plt.xlabel('ROI')
plt.ylabel('ROI')
plt.savefig('isfc_matrix.png', dpi=300)

# High ISFC indicates network connections that are synchronized during stimulus
```

**Example 5: Statistical Testing for ISFC**

```python
from brainiak.isfc import isfc, permutation_isfc
import numpy as np

data = np.random.randn(15, 250, 50)

# Observed ISFC
observed_isfc = isfc(data)

# Permutation test
null_isfc, p_values = permutation_isfc(
    data,
    pairwise=False,
    n_permutations=500,
    random_state=42
)

# Threshold at p < 0.01
significant_edges = p_values < 0.01

print(f"Significant edges: {significant_edges.sum()} / {p_values.size}")

# Network-level analysis: which networks show high ISFC?
```

## Shared Response Model (SRM)

**Example 6: Basic SRM for Functional Alignment**

```python
from brainiak.funcalign.srm import SRM
import numpy as np

# Load data from multiple subjects (different voxels, same stimuli)
# Shape: list of (n_timepoints, n_voxels_subject_i)
subjects_data = []
for i in range(1, 16):
    data = np.load(f'sub-{i:02d}_bold.npy')  # (300, ~10000 voxels)
    subjects_data.append(data.T)  # SRM expects (voxels, timepoints)

# Initialize SRM
srm = SRM(n_iter=10, features=50)  # 50 shared features

# Fit SRM to find shared space
srm.fit(subjects_data)

# Transform individual data to shared space
shared_data = srm.transform(subjects_data)

# shared_data[i] shape: (50, 300) - shared representation for subject i
print(f"Shared space dimensions: {shared_data[0].shape[0]}")

# Benefit: Enables cross-subject decoding
# Can train classifier on subjects 1-14, test on subject 15
```

**Example 7: Cross-Subject Decoding with SRM**

```python
from brainiak.funcalign.srm import SRM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Assume you have labeled data (e.g., different movie scenes)
# labels: (n_timepoints,) indicating scene category

# Load data and labels
subjects_data = []  # List of (voxels, timepoints) for each subject
labels = np.load('scene_labels.npy')  # (300,)

for i in range(1, 16):
    data = np.load(f'sub-{i:02d}_bold.npy').T
    subjects_data.append(data)

# Fit SRM
srm = SRM(n_iter=10, features=50)
srm.fit(subjects_data)
shared_data = srm.transform(subjects_data)

# Leave-one-subject-out cross-validation
for test_subj in range(15):
    # Training data: all except test subject
    X_train = np.hstack([shared_data[i] for i in range(15) if i != test_subj]).T
    y_train = np.tile(labels, 14)  # Repeat labels for training subjects

    # Test data
    X_test = shared_data[test_subj].T
    y_test = labels

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict on test subject
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Subject {test_subj+1} accuracy: {accuracy:.3f}")

# SRM enables generalization across subjects
```

**Example 8: Probabilistic SRM**

```python
from brainiak.funcalign.rsrm import RSRM  # Robust SRM
import numpy as np

# RSRM is robust to noise and outliers
subjects_data = []
for i in range(1, 11):
    data = np.random.randn(5000, 200)  # (voxels, timepoints)
    subjects_data.append(data)

# Fit RSRM
rsrm = RSRM(n_iter=20, features=30)
rsrm.fit(subjects_data)

# Transform to shared space
shared = rsrm.transform(subjects_data)

print(f"Shared representations: {len(shared)} subjects")
print(f"Shared space shape: {shared[0].shape}")
```

## Searchlight MVPA

**Example 9: Searchlight Classification**

```python
from brainiak.searchlight.searchlight import Searchlight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import nibabel as nib

# Load fMRI data and labels
func_img = nib.load('sub-001_task_bold.nii.gz')
data = func_img.get_fdata()  # (x, y, z, time)
labels = np.load('trial_labels.npy')  # (time,) - stimulus categories

# Reshape to (time, x, y, z)
data = np.transpose(data, (3, 0, 1, 2))

# Create brain mask
mask = nib.load('sub-001_mask.nii.gz').get_fdata().astype(bool)

# Define searchlight function
def searchlight_fn(data, mask, labels):
    """Classification accuracy within searchlight sphere."""
    # data shape: (time, n_voxels_in_sphere)
    clf = LogisticRegression(max_iter=500)
    scores = cross_val_score(clf, data, labels, cv=3)
    return scores.mean()

# Initialize searchlight
sl = Searchlight(
    sl_rad=3,  # 3-voxel radius sphere
    max_blk_edge=10
)

# Run searchlight (returns accuracy at each voxel)
sl_result = sl.run_searchlight(
    data,
    mask,
    searchlight_fn,
    labels=labels
)

# sl_result shape: same as brain (x, y, z)
# High values indicate decodable information

# Save result
result_img = nib.Nifti1Image(sl_result, func_img.affine)
nib.save(result_img, 'searchlight_accuracy.nii.gz')

# Visualize
from nilearn import plotting
plotting.plot_stat_map(
    result_img,
    threshold=0.6,  # Above chance (0.5 for binary)
    title='Searchlight Decoding Accuracy'
)
```

**Example 10: Parallel Searchlight with MPI**

```python
# Save as searchlight_parallel.py
from brainiak.searchlight.searchlight import Searchlight
from sklearn.svm import SVC
import numpy as np
import nibabel as nib
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # Load data on master process
    func_img = nib.load('sub-001_bold.nii.gz')
    data = np.transpose(func_img.get_fdata(), (3, 0, 1, 2))
    mask = nib.load('sub-001_mask.nii.gz').get_fdata().astype(bool)
    labels = np.load('labels.npy')
else:
    data = None
    mask = None
    labels = None

# Broadcast data to all processes
data = comm.bcast(data, root=0)
mask = comm.bcast(mask, root=0)
labels = comm.bcast(labels, root=0)

# Define searchlight function
def sl_svm(data, mask, labels):
    clf = SVC(kernel='linear')
    scores = cross_val_score(clf, data, labels, cv=5)
    return scores.mean()

# Run parallel searchlight
sl = Searchlight(sl_rad=3, max_blk_edge=10)
sl_result = sl.run_searchlight(data, mask, sl_svm, labels=labels)

if rank == 0:
    # Save result
    result_img = nib.Nifti1Image(sl_result, func_img.affine)
    nib.save(result_img, 'searchlight_svm.nii.gz')
    print("Searchlight complete!")

# Run with: mpirun -n 8 python searchlight_parallel.py
```

## Event Segmentation

**Example 11: Detecting Event Boundaries**

```python
from brainiak.eventseg.event import EventSegment
import numpy as np

# Load continuous brain activity during movie/narrative
# Shape: (n_timepoints, n_voxels)
data = np.load('movie_bold.npy')  # (500, 10000)

# Initialize event segmentation with HMM
n_events = 10  # Number of events to detect
ev = EventSegment(n_events=n_events)

# Fit model
ev.fit(data.T)  # Expects (voxels, timepoints)

# Get event boundaries
event_pattern = ev.segments_[0]  # Probability of being in each event
boundaries = np.diff(np.argmax(event_pattern, axis=0))

# Find boundary timepoints
boundary_times = np.where(boundaries != 0)[0]
print(f"Event boundaries at timepoints: {boundary_times}")

# Visualize event segmentation
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.imshow(event_pattern, aspect='auto', cmap='viridis')
plt.xlabel('Time (TRs)')
plt.ylabel('Event')
plt.title('Event Segmentation')
plt.colorbar(label='Probability')
plt.savefig('event_segmentation.png', dpi=300)

# High probability regions show distinct cognitive states/events
```

**Example 12: Cross-Subject Event Segmentation**

```python
from brainiak.eventseg.event import EventSegment
import numpy as np

# Load data from multiple subjects
subjects_data = []
for i in range(1, 11):
    data = np.load(f'sub-{i:02d}_narrative_bold.npy')  # (time, voxels)
    subjects_data.append(data.T)  # (voxels, time)

# Average event segmentation across subjects
ev = EventSegment(n_events=8)

for data in subjects_data:
    ev.fit(data)

# Compare event boundaries across subjects
# Check if subjects segment the narrative similarly
```

## Full Correlation Matrix Analysis (FCMA)

**Example 13: FCMA Classification**

```python
from brainiak.fcma.classifier import Classifier
from brainiak.fcma.preprocessing import prepare_fcma_data
import numpy as np

# FCMA uses correlation patterns for classification
# More powerful than activation patterns for some tasks

# Prepare data
# data: (subjects, TRs, voxels)
# labels: (subjects, TRs) - condition labels

# This is computationally intensive
# Typically run on HPC with MPI

# Simplified example
n_subjects = 10
n_trs = 200
n_voxels = 5000

data = np.random.randn(n_subjects, n_trs, n_voxels)
labels = np.random.randint(0, 2, (n_subjects, n_trs))

# FCMA computes voxel-to-voxel correlations for classification
# Identifies which connectivity patterns predict conditions

# Full implementation requires MPI
# See BrainIAK documentation for complete examples
```

## Real-Time fMRI

**Example 14: Real-Time Preprocessing and Classification**

```python
from brainiak.rt import RealTimeFMRI
import numpy as np

# Real-time fMRI for neurofeedback or adaptive experiments

class RealTimeExperiment:
    def __init__(self):
        self.rt = RealTimeFMRI()
        self.classifier = None  # Trained classifier
        self.buffer = []

    def process_new_volume(self, volume):
        """Process incoming fMRI volume in real-time."""
        # 1. Preprocessing
        preprocessed = self.rt.preprocess(volume)

        # 2. Feature extraction
        features = self.extract_features(preprocessed)

        # 3. Classification (if model trained)
        if self.classifier is not None:
            prediction = self.classifier.predict([features])
            return prediction

        # 4. Store for training
        self.buffer.append(features)

    def extract_features(self, volume):
        """Extract ROI mean values."""
        # Simplified: mean of predefined ROIs
        roi_means = []
        for roi_mask in self.roi_masks:
            mean_val = volume[roi_mask].mean()
            roi_means.append(mean_val)
        return np.array(roi_means)

    def train_classifier(self, labels):
        """Train classifier on buffered data."""
        from sklearn.svm import SVC
        X = np.array(self.buffer)
        y = labels
        self.classifier = SVC(kernel='linear')
        self.classifier.fit(X, y)

# Usage in real-time loop
exp = RealTimeExperiment()

# During localizer run, collect training data
# for volume, label in localizer_data:
#     exp.process_new_volume(volume)
# exp.train_classifier(localizer_labels)

# During neurofeedback run, classify in real-time
# for volume in neurofeedback_data:
#     prediction = exp.process_new_volume(volume)
#     # Show feedback based on prediction
```

## Topographic Factor Analysis (TFA)

**Example 15: Spatial-Temporal Decomposition**

```python
from brainiak.factoranalysis.tfa import TFA
import numpy as np

# TFA decomposes fMRI data into spatial and temporal components
# Similar to ICA but with different assumptions

# Load data (timepoints, voxels)
data = np.load('task_bold.npy')  # (300, 50000)

# Initialize TFA
tfa = TFA(
    K=10,  # Number of factors
    max_iter=100,
    verbose=True
)

# Fit TFA
tfa.fit(data)

# Extract components
spatial_maps = tfa.get_spatial_maps()  # (10, 50000)
temporal_factors = tfa.get_temporal_factors()  # (10, 300)

# spatial_maps[i]: spatial distribution of factor i
# temporal_factors[i]: temporal profile of factor i

# Visualize factors
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for i in range(min(5, 10)):
    plt.subplot(5, 1, i+1)
    plt.plot(temporal_factors[i, :])
    plt.ylabel(f'Factor {i+1}')
plt.xlabel('Time (TRs)')
plt.suptitle('TFA Temporal Factors')
plt.tight_layout()
plt.savefig('tfa_temporal.png', dpi=300)
```

## HPC and Parallel Processing

**Example 16: MPI-based Searchlight**

```python
# Distributed searchlight across compute nodes
# Save as distributed_searchlight.py

from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.svm import LinearSVC
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each process handles subset of searchlights
# BrainIAK automatically distributes work

def svm_classify(data, mask, labels):
    clf = LinearSVC()
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, data, labels, cv=3)
    return scores.mean()

# Load data (rank 0 broadcasts to others)
if rank == 0:
    data = np.load('bold_data.npy')
    mask = np.load('brain_mask.npy')
    labels = np.load('labels.npy')
else:
    data, mask, labels = None, None, None

data = comm.bcast(data, root=0)
mask = comm.bcast(mask, root=0)
labels = comm.bcast(labels, root=0)

# Run distributed searchlight
sl = Searchlight(sl_rad=3, max_blk_edge=8)
result = sl.run_searchlight(data, mask, svm_classify, labels=labels)

if rank == 0:
    print("Searchlight complete!")
    np.save('searchlight_result.npy', result)

# Run with: mpirun -n 16 python distributed_searchlight.py
# Automatically scales across nodes
```

## Integration with Other Tools

**Example 17: BrainIAK with Nilearn**

```python
from brainiak.isc import isc
from nilearn import datasets, input_data, plotting
import numpy as np
import nibabel as nib

# Load atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
masker = input_data.NiftiLabelsMasker(atlas.maps)

# Extract timeseries for multiple subjects
subjects_ts = []
for i in range(1, 11):
    img = nib.load(f'sub-{i:02d}_bold.nii.gz')
    ts = masker.fit_transform(img)  # (time, ROIs)
    subjects_ts.append(ts)

data = np.array(subjects_ts)  # (subjects, time, ROIs)

# Compute ISC
isc_rois = isc(data)

# Visualize on brain
from nilearn import plotting
# Create image with ISC values in each ROI
# plotting.plot_roi(atlas.maps, isc_rois, title='ISC per ROI')

# Combination of BrainIAK algorithms with Nilearn visualization
```

## Troubleshooting

**Memory Management:**
```python
# For large datasets:

# 1. Use memory mapping
data = np.load('large_data.npy', mmap_mode='r')

# 2. Process in chunks
chunk_size = 1000  # voxels per chunk
n_voxels = data.shape[-1]
results = []
for i in range(0, n_voxels, chunk_size):
    chunk = data[..., i:i+chunk_size]
    result = process_chunk(chunk)
    results.append(result)

# 3. Use lower precision
data = data.astype(np.float32)  # Instead of float64
```

**MPI Issues:**
```bash
# If MPI fails:

# Check MPI installation
mpirun --version

# Test MPI
mpirun -n 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}')"

# Set MPI threading
export OMP_NUM_THREADS=1  # Important for MPI

# For SLURM clusters
srun --mpi=pmi2 python script.py
```

**Numerical Stability:**
```python
# If getting NaN or Inf:

# 1. Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 2. Check for missing values
data = np.nan_to_num(data, nan=0.0)

# 3. Regularization in SRM
srm = SRM(n_iter=10, features=50, gamma=1.0)  # Add regularization
```

## Best Practices

**Data Quality:**
- Remove high-motion volumes before ISC/ISFC
- Use fMRIPrep for preprocessing
- Verify temporal alignment across subjects
- Check for scanner artifacts

**Cross-Validation:**
- Always use proper cross-validation
- Leave-one-subject-out for cross-subject decoding
- Avoid double-dipping (same data for selection and testing)

**Computational Efficiency:**
- Use MPI for searchlight and large analyses
- Precompute and save intermediate results
- Use appropriate data types (float32 vs float64)
- Leverage HPC job arrays for parameter sweeps

**Result Interpretation:**
- ISC: stimulus-driven vs. spontaneous activity
- SRM: validate with known structure (e.g., visual cortex)
- Searchlight: correct for multiple comparisons
- Event segmentation: compare to external annotations

## Integration with Analysis Ecosystem

**Nilearn:**
- Load data with nilearn.image
- Visualization with nilearn.plotting
- Atlases from nilearn.datasets

**fMRIPrep:**
- Preprocessed data ready for BrainIAK
- Standard space for cross-subject analysis

**MNE-Python:**
- ISC on MEG/EEG data
- Event segmentation on continuous recordings

**Scikit-learn:**
- Classifiers for searchlight and decoding
- Cross-validation utilities
- Preprocessing transformers

## References

**BrainIAK:**
- Kumar et al. (2020). BrainIAK: The Brain Imaging Analysis Kit. *Aperture Neuro*, 1, 1-14.

**ISC/ISFC:**
- Hasson et al. (2004). Intersubject synchronization of cortical activity during natural vision. *Science*, 303(5664), 1634-1640.
- Simony et al. (2016). Dynamic reconfiguration of the default mode network during narrative comprehension. *Nature Communications*, 7, 12141.

**SRM:**
- Chen et al. (2015). A reduced-dimension fMRI shared response model. *NIPS*, 460-468.

**Event Segmentation:**
- Baldassano et al. (2017). Discovering event structure in continuous narrative perception and memory. *Neuron*, 95(3), 709-721.

**Searchlight:**
- Kriegeskorte et al. (2006). Information-based functional brain mapping. *PNAS*, 103(10), 3863-3868.

**Online Resources:**
- BrainIAK Documentation: https://brainiak.org/
- BrainIAK GitHub: https://github.com/brainiak/brainiak
- BrainIAK Tutorials: https://brainiak.org/tutorials/
- Paper Collection: https://brainiak.org/papers/
