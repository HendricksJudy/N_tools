# PyMVPA - Multivariate Pattern Analysis in Python

## Overview

**PyMVPA** (Python for Multivariate Pattern Analysis) is a comprehensive Python framework for multivariate analysis of neuroimaging data, enabling researchers to decode cognitive states, predict behavioral outcomes, and understand distributed neural representations. Originally developed by Michael Hanke and Yaroslav Halchenko, PyMVPA provides a unified interface for classification, regression, feature selection, and advanced techniques like searchlight analysis and hyperalignment, all tailored to the unique challenges of high-dimensional neuroimaging data.

PyMVPA bridges the gap between general-purpose machine learning libraries and neuroimaging-specific needs, providing seamless integration with NiBabel for data I/O, specialized cross-validation strategies that respect temporal and run structure, and powerful tools for whole-brain searchlight decoding. Its flexible dataset structure preserves critical metadata (labels, chunks, voxel coordinates) throughout complex analysis pipelines.

**Key Features:**
- Classification and regression with scikit-learn integration
- Whole-brain and ROI-based searchlight analysis
- Hyperalignment for functional alignment across subjects
- Cross-validation strategies for neuroimaging (leave-one-run-out, k-fold)
- Feature selection and dimensionality reduction methods
- Representational similarity analysis (RSA)
- Time-series and event-related analysis
- Support for volumetric (NIfTI), surface (GIFTI), and sensor-space data
- Permutation testing and statistical inference
- Sensitivity analysis and weight mapping
- Integration with preprocessing pipelines
- Parallel processing support

**Primary Use Cases:**
- Cognitive state decoding from fMRI data
- Visual category classification and object decoding
- Attention and memory state prediction
- Clinical outcome prediction and biomarker discovery
- Hyperalignment for improved cross-subject decoding
- Representational similarity analysis
- Time-resolved decoding and temporal generalization
- Brain-computer interface development

**Official Documentation:** http://www.pymvpa.org/

---

## Installation

### Basic Installation

```bash
# Install via pip (recommended)
pip install pymvpa2

# Install with optional dependencies for neuroimaging
pip install pymvpa2[nibabel,nipy]

# Install with HDF5 support for large datasets
pip install pymvpa2[hdf5]

# Install from GitHub for latest development version
pip install git+https://github.com/PyMVPA/PyMVPA.git

# Verify installation
python -c "import mvpa2; print(mvpa2.__version__)"
```

### Dependencies

```bash
# Core dependencies (automatically installed)
pip install numpy scipy

# Neuroimaging I/O
pip install nibabel

# Machine learning
pip install scikit-learn

# Visualization
pip install matplotlib

# Optional: parallel processing
pip install joblib

# Optional: statistical analysis
pip install statsmodels
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/PyMVPA/PyMVPA.git
cd PyMVPA

# Install in development mode
pip install -e .

# Run tests to verify
python -m pytest mvpa2/tests/
```

---

## Data Loading and Dataset Structure

### Load fMRI Data from NIfTI

```python
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets import vstack
import numpy as np

# Load single-subject fMRI data with event labels
# Assumes preprocessed 4D NIfTI: time x x x y x z
dataset = fmri_dataset(
    samples='sub-01_task-faces_bold.nii.gz',  # 4D fMRI data
    targets='conditions.txt',  # Labels for each volume
    chunks='runs.txt',  # Run/session information
    mask='sub-01_brain_mask.nii.gz'  # Brain mask
)

print(dataset.shape)  # (n_volumes, n_voxels)
print(dataset.sa.targets)  # Sample attributes: labels
print(dataset.sa.chunks)  # Sample attributes: runs
print(dataset.fa.voxel_indices)  # Feature attributes: voxel coordinates
```

### Create Dataset from NumPy Array

```python
from mvpa2.datasets.base import Dataset

# Create dataset from array
data = np.random.rand(100, 1000)  # 100 samples, 1000 features
targets = np.repeat(['face', 'house', 'object'], [30, 40, 30])
chunks = np.repeat([1, 2, 3, 4], 25)

ds = Dataset(
    samples=data,
    sa={'targets': targets, 'chunks': chunks}
)

# Add feature attributes (e.g., voxel coordinates)
ds.fa['voxel_indices'] = np.random.randint(0, 50, (1000, 3))
```

### Load Multiple Runs and Stack

```python
# Load multiple runs from BIDS dataset
datasets = []
for run in range(1, 5):
    ds = fmri_dataset(
        samples=f'sub-01_task-faces_run-{run}_bold.nii.gz',
        targets=f'run-{run}_conditions.txt',
        chunks=run,
        mask='sub-01_brain_mask.nii.gz'
    )
    datasets.append(ds)

# Stack datasets vertically (concatenate along samples)
full_dataset = vstack(datasets, a='all')
print(f"Combined: {full_dataset.shape}")
```

---

## Data Preprocessing

### Detrending and Normalization

```python
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.zscore import zscore

# Polynomial detrending (remove linear/quadratic trends)
# Detrend separately for each run
poly_detrend(dataset, polyord=1, chunks_attr='chunks')

# Z-score normalization (mean=0, std=1)
# Normalize separately per run to avoid information leakage
zscore(dataset, chunks_attr='chunks')

# Check normalization
print(f"Mean: {dataset.samples.mean(axis=0).mean():.4f}")
print(f"Std: {dataset.samples.std(axis=0).mean():.4f}")
```

### Feature Selection by Mask

```python
from mvpa2.datasets.mri import map2nifti

# Select features from specific ROI
roi_mask = 'roi_fusiform_face_area.nii.gz'
roi_dataset = fmri_dataset(
    samples='sub-01_task-faces_bold.nii.gz',
    targets='conditions.txt',
    chunks='runs.txt',
    mask=roi_mask
)

print(f"ROI voxels: {roi_dataset.nfeatures}")
```

### Temporal Averaging for Event-Related Designs

```python
from mvpa2.mappers.fx import mean_group_sample

# Average volumes within events (e.g., trials)
# Assumes event_id labels each trial
event_avg = mean_group_sample(['event_id'])(dataset)
print(f"Original: {dataset.nsamples}, Averaged: {event_avg.nsamples}")
```

---

## Basic Classification

### Linear SVM Classification with Cross-Validation

```python
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.measures.base import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner

# Create linear SVM classifier (C=1)
clf = LinearCSVMC(C=1.0)

# Create N-fold cross-validation (leave-one-run-out)
cv = CrossValidation(
    clf,
    NFoldPartitioner(attr='chunks'),  # Split by runs
    enable_ca=['stats']  # Enable confusion matrix
)

# Run cross-validation
results = cv(dataset)
accuracy = np.mean(results)

print(f"Cross-validated accuracy: {accuracy:.3f}")
print("Confusion matrix:")
print(cv.ca.stats.matrix)
```

### Multi-Class Classification

```python
from sklearn.metrics import classification_report

# For multi-class (e.g., face, house, object, scrambled)
clf = LinearCSVMC()
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
results = cv(dataset)

# Get predictions for each fold
predictions = cv.ca.predictions
true_labels = dataset.sa.targets

# Classification report
print(classification_report(true_labels, predictions))
```

### Leave-One-Subject-Out Cross-Validation

```python
# For multi-subject datasets with subject attribute
cv_loso = CrossValidation(
    clf,
    NFoldPartitioner(attr='subject')  # Leave one subject out
)

results_loso = cv_loso(multi_subject_dataset)
print(f"LOSO accuracy: {np.mean(results_loso):.3f}")
```

---

## Searchlight Analysis

### Sphere-Based Searchlight

```python
from mvpa2.measures.searchlight import sphere_searchlight
from mvpa2.base.learner import ChainLearner
from mvpa2.mappers.fx import mean_sample

# Create searchlight with 3-voxel radius
sl = sphere_searchlight(
    CrossValidation(clf, NFoldPartitioner(attr='chunks')),
    radius=3,  # Voxels
    space='voxel_indices',  # Use voxel coordinates
    postproc=mean_sample()  # Average across folds
)

# Run searchlight (can take time)
sl_map = sl(dataset)

# sl_map.samples contains accuracy for each voxel
print(f"Searchlight map shape: {sl_map.shape}")
print(f"Mean accuracy: {sl_map.samples.mean():.3f}")

# Save to NIfTI
from mvpa2.datasets.mri import map2nifti
nii = map2nifti(sl_map, imghdr=dataset.a.imghdr)
nii.to_filename('searchlight_accuracy_map.nii.gz')
```

### Statistical Significance of Searchlight

```python
from scipy.stats import binom_test

# Test each voxel against chance (e.g., 0.5 for binary)
n_folds = len(np.unique(dataset.sa.chunks))
chance_level = 1.0 / len(np.unique(dataset.sa.targets))

# Convert accuracy to proportion correct
p_values = np.zeros(sl_map.nfeatures)
for i, acc in enumerate(sl_map.samples[0]):
    # Binomial test: n_folds trials, acc success rate
    n_correct = int(acc * n_folds)
    p_values[i] = binom_test(n_correct, n_folds, chance_level, alternative='greater')

# FDR correction
from statsmodels.stats.multitest import fdrcorrection
reject, p_corrected = fdrcorrection(p_values, alpha=0.05)

# Create thresholded map
sl_map_thresh = sl_map.copy()
sl_map_thresh.samples[0, ~reject] = 0

nii_thresh = map2nifti(sl_map_thresh, imghdr=dataset.a.imghdr)
nii_thresh.to_filename('searchlight_fdr_thresholded.nii.gz')
```

### Parallel Searchlight with Joblib

```python
from joblib import Parallel, delayed

# Enable parallel processing
sl_parallel = sphere_searchlight(
    CrossValidation(clf, NFoldPartitioner(attr='chunks')),
    radius=3,
    space='voxel_indices',
    postproc=mean_sample(),
    nproc=8  # Use 8 cores
)

sl_map_parallel = sl_parallel(dataset)
```

---

## Feature Selection

### ANOVA-Based Feature Selection

```python
from mvpa2.featsel.base import SensitivityBasedFeatureSelection
from mvpa2.clfs.stats import MCNullDist
from mvpa2.featsel.helpers import FixedNElementTailSelector

# Select top 500 features by ANOVA F-score
from mvpa2.measures.anova import OneWayAnova
anova = OneWayAnova()
anova_scores = anova(dataset)

# Sort and select top features
sorted_idx = np.argsort(anova_scores.samples[0])[::-1]
top_500_idx = sorted_idx[:500]

# Create reduced dataset
dataset_reduced = dataset[:, top_500_idx]
print(f"Reduced to {dataset_reduced.nfeatures} features")

# Run classification on reduced dataset
cv_reduced = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
acc_reduced = np.mean(cv_reduced(dataset_reduced))
print(f"Accuracy with top 500 features: {acc_reduced:.3f}")
```

### Recursive Feature Elimination (RFE)

```python
from mvpa2.featsel.rfe import RFE

# RFE with linear SVM
rfe = RFE(
    sensitivity_analyzer=clf.get_sensitivity_analyzer(),
    transfer_error=CrossValidation(clf, NFoldPartitioner(attr='chunks')),
    feature_selector=FixedNElementTailSelector(100, tail='upper'),
    update_sensitivity=True
)

# Run RFE
rfe.train(dataset)
selected_features = rfe.ca.history[-1]
print(f"Selected {len(selected_features)} features")
```

### ROI-Based Feature Extraction

```python
# Load multiple ROI masks
rois = {
    'V1': 'roi_v1.nii.gz',
    'FFA': 'roi_ffa.nii.gz',
    'PPA': 'roi_ppa.nii.gz'
}

roi_accuracies = {}
for roi_name, roi_mask in rois.items():
    roi_ds = fmri_dataset(
        samples='sub-01_task-faces_bold.nii.gz',
        targets='conditions.txt',
        chunks='runs.txt',
        mask=roi_mask
    )
    poly_detrend(roi_ds, polyord=1, chunks_attr='chunks')
    zscore(roi_ds, chunks_attr='chunks')

    cv_roi = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
    acc_roi = np.mean(cv_roi(roi_ds))
    roi_accuracies[roi_name] = acc_roi
    print(f"{roi_name}: {acc_roi:.3f}")
```

---

## Hyperalignment

### Functional Hyperalignment Across Subjects

```python
from mvpa2.algorithms.hyperalignment import Hyperalignment

# Prepare datasets from multiple subjects
# Each dataset should have same task structure
subjects = []
for sub in ['01', '02', '03', '04', '05']:
    ds = fmri_dataset(
        samples=f'sub-{sub}_task-movie_bold.nii.gz',
        mask=f'sub-{sub}_brain_mask.nii.gz'
    )
    poly_detrend(ds, polyord=1)
    zscore(ds)
    subjects.append(ds)

# Perform hyperalignment
hyper = Hyperalignment()
hyper_subjects = hyper(subjects)

# hyper_subjects contains aligned datasets
# Verify alignment improved consistency
from mvpa2.misc.stats import compute_isc
isc_before = compute_isc(subjects)
isc_after = compute_isc(hyper_subjects)

print(f"ISC before hyperalignment: {isc_before.mean():.3f}")
print(f"ISC after hyperalignment: {isc_after.mean():.3f}")
```

### Cross-Subject Decoding with Hyperalignment

```python
# Train on N-1 subjects, test on left-out subject
# Using hyperaligned data improves generalization

# Combine training subjects
train_subjects = hyper_subjects[:-1]
test_subject = hyper_subjects[-1]

train_data = vstack(train_subjects, a='all')
test_data = test_subject

# Train classifier on combined training data
clf = LinearCSVMC()
clf.train(train_data)

# Test on held-out subject
predictions = clf.predict(test_data.samples)
accuracy = np.mean(predictions == test_data.sa.targets)
print(f"Cross-subject decoding accuracy: {accuracy:.3f}")
```

---

## Advanced Classifiers and Regression

### Regularized Logistic Regression

```python
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from sklearn.linear_model import LogisticRegression

# Wrap scikit-learn classifier
lr = SKLLearnerAdapter(
    LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        multi_class='multinomial'
    )
)

# Cross-validation with logistic regression
cv_lr = CrossValidation(lr, NFoldPartitioner(attr='chunks'))
acc_lr = np.mean(cv_lr(dataset))
print(f"Logistic regression accuracy: {acc_lr:.3f}")
```

### Support Vector Regression (SVR)

```python
from mvpa2.clfs.svm import LinearCSVMC
from sklearn.svm import SVR
from scipy.stats import pearsonr

# Predict continuous variable (e.g., age, reaction time)
# Assuming dataset.sa.age contains continuous values

svr = SKLLearnerAdapter(SVR(kernel='linear', C=1.0))

# Cross-validation for regression
predictions_all = []
true_all = []

for train_idx, test_idx in NFoldPartitioner(attr='chunks').generate(dataset):
    train_ds = dataset[train_idx]
    test_ds = dataset[test_idx]

    svr.train(train_ds)
    predictions = svr.predict(test_ds.samples)

    predictions_all.extend(predictions)
    true_all.extend(test_ds.sa.age)

# Correlation between predicted and true values
r, p = pearsonr(predictions_all, true_all)
print(f"Prediction correlation: r={r:.3f}, p={p:.4f}")

# Mean absolute error
mae = np.mean(np.abs(np.array(predictions_all) - np.array(true_all)))
print(f"Mean absolute error: {mae:.3f}")
```

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

# Random forest with 100 trees
rf = SKLLearnerAdapter(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
)

cv_rf = CrossValidation(rf, NFoldPartitioner(attr='chunks'))
acc_rf = np.mean(cv_rf(dataset))
print(f"Random forest accuracy: {acc_rf:.3f}")
```

---

## Representational Similarity Analysis (RSA)

### Compute Neural Representational Dissimilarity Matrix

```python
from mvpa2.measures.rsa import PDistConsistency
from scipy.spatial.distance import pdist, squareform

# Create RDM for each run separately
runs = np.unique(dataset.sa.chunks)
rdms = []

for run in runs:
    run_data = dataset[dataset.sa.chunks == run]

    # Compute pairwise distances (correlation distance)
    distances = pdist(run_data.samples, metric='correlation')
    rdm = squareform(distances)
    rdms.append(rdm)

# Average RDMs across runs
rdm_avg = np.mean(rdms, axis=0)

# Visualize RDM
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(rdm_avg, cmap='viridis')
plt.colorbar(label='Dissimilarity')
plt.title('Neural Representational Dissimilarity Matrix')
plt.xlabel('Conditions')
plt.ylabel('Conditions')
plt.savefig('neural_rdm.png', dpi=150)
plt.close()
```

### Compare Neural RDM with Model RDM

```python
from scipy.stats import spearmanr

# Create model RDM (e.g., based on visual similarity)
# model_rdm should be same shape as neural RDM
model_rdm = np.random.rand(rdm_avg.shape[0], rdm_avg.shape[1])
model_rdm = (model_rdm + model_rdm.T) / 2  # Make symmetric

# Compare neural and model RDMs
# Correlate upper triangular parts
triu_idx = np.triu_indices(rdm_avg.shape[0], k=1)
neural_upper = rdm_avg[triu_idx]
model_upper = model_rdm[triu_idx]

rho, p = spearmanr(neural_upper, model_upper)
print(f"Neural-model RDM correlation: rho={rho:.3f}, p={p:.4f}")
```

### Searchlight RSA

```python
from mvpa2.measures.rsa import PDistTargetSimilarity

# Create searchlight RSA
# Compare local neural RDM to model RDM at each searchlight
target_sim = PDistTargetSimilarity(
    dataset,
    comparison_metric='spearman'
)

sl_rsa = sphere_searchlight(
    target_sim,
    radius=3,
    space='voxel_indices'
)

rsa_map = sl_rsa(dataset)
nii_rsa = map2nifti(rsa_map, imghdr=dataset.a.imghdr)
nii_rsa.to_filename('searchlight_rsa_map.nii.gz')
```

---

## Time-Series and Event-Related Analysis

### Time-Resolved Decoding

```python
# Classify each time point in event
# Assumes dataset has time_coords attribute

time_points = np.unique(dataset.sa.time_coords)
accuracies_time = []

for t in time_points:
    ds_t = dataset[dataset.sa.time_coords == t]

    cv_t = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
    acc_t = np.mean(cv_t(ds_t))
    accuracies_time.append(acc_t)

# Plot decoding over time
plt.figure(figsize=(10, 4))
plt.plot(time_points, accuracies_time, marker='o')
plt.axhline(y=chance_level, color='r', linestyle='--', label='Chance')
plt.xlabel('Time (s)')
plt.ylabel('Classification Accuracy')
plt.title('Time-Resolved Decoding')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('time_resolved_decoding.png', dpi=150)
plt.close()
```

### Temporal Generalization Matrix

```python
# Train at time t1, test at time t2
# Reveals when similar representations are active

n_time = len(time_points)
gen_matrix = np.zeros((n_time, n_time))

for i, train_time in enumerate(time_points):
    for j, test_time in enumerate(time_points):
        train_ds = dataset[dataset.sa.time_coords == train_time]
        test_ds = dataset[dataset.sa.time_coords == test_time]

        # Average across chunks for simplicity
        clf_temp = LinearCSVMC()
        clf_temp.train(train_ds)
        predictions = clf_temp.predict(test_ds.samples)
        gen_matrix[i, j] = np.mean(predictions == test_ds.sa.targets)

# Plot temporal generalization
plt.figure(figsize=(8, 7))
plt.imshow(gen_matrix, origin='lower', cmap='RdYlBu_r',
           extent=[time_points[0], time_points[-1], time_points[0], time_points[-1]])
plt.colorbar(label='Accuracy')
plt.xlabel('Test Time (s)')
plt.ylabel('Train Time (s)')
plt.title('Temporal Generalization Matrix')
plt.savefig('temporal_generalization.png', dpi=150)
plt.close()
```

---

## Statistical Inference and Permutation Testing

### Permutation Testing for Classification

```python
from mvpa2.clfs.stats import MCNullDist
from mvpa2.generators.permutation import AttributePermutator

# Create permutation distribution
# Permute labels and recompute accuracy
n_permutations = 1000

null_dist = MCNullDist(
    permutator=AttributePermutator('targets', count=n_permutations),
    tail='right'  # One-tailed test
)

# Add null distribution to cross-validation
cv_perm = CrossValidation(
    clf,
    NFoldPartitioner(attr='chunks'),
    null_dist=null_dist,
    enable_ca=['stats']
)

results_perm = cv_perm(dataset)
p_value = cv_perm.ca.null_prob

print(f"True accuracy: {np.mean(results_perm):.3f}")
print(f"Permutation p-value: {p_value:.4f}")
```

### Cluster-Based Correction for Searchlight

```python
from scipy.ndimage import label

# Threshold searchlight map at p < 0.001
threshold = 0.001
sig_voxels = p_values < threshold

# Find clusters of significant voxels
labeled_array, num_clusters = label(sig_voxels.reshape(dataset.a.voldim))

# Compute cluster sizes
cluster_sizes = []
for i in range(1, num_clusters + 1):
    cluster_sizes.append(np.sum(labeled_array == i))

print(f"Found {num_clusters} clusters")
print(f"Cluster sizes: {cluster_sizes}")

# Apply cluster-size threshold (e.g., min 10 voxels)
min_cluster_size = 10
for i in range(1, num_clusters + 1):
    if cluster_sizes[i-1] < min_cluster_size:
        labeled_array[labeled_array == i] = 0

# Create corrected map
sig_voxels_corrected = labeled_array > 0
```

---

## Sensitivity Analysis and Weight Mapping

### Extract Classifier Weights

```python
# Get voxel weights from linear classifier
clf_trained = LinearCSVMC()
clf_trained.train(dataset)

# Get sensitivity (weights)
sensitivities = clf_trained.get_sensitivity_analyzer()
weights = sensitivities(dataset)

print(f"Weight shape: {weights.shape}")  # (1, n_voxels)

# Map weights back to brain
weight_map = weights.samples[0]
weights_ds = Dataset(weight_map[np.newaxis, :], fa=dataset.fa)
nii_weights = map2nifti(weights_ds, imghdr=dataset.a.imghdr)
nii_weights.to_filename('classifier_weights.nii.gz')
```

### Threshold Weights by Magnitude

```python
# Keep only strongest positive and negative weights
percentile_threshold = 95
threshold_val = np.percentile(np.abs(weight_map), percentile_threshold)

weight_map_thresh = weight_map.copy()
weight_map_thresh[np.abs(weight_map_thresh) < threshold_val] = 0

# Save thresholded weights
weights_thresh_ds = Dataset(weight_map_thresh[np.newaxis, :], fa=dataset.fa)
nii_weights_thresh = map2nifti(weights_thresh_ds, imghdr=dataset.a.imghdr)
nii_weights_thresh.to_filename('classifier_weights_thresholded.nii.gz')
```

---

## Integration with BIDS and Preprocessing

### Load BIDS Dataset with PyBIDS

```python
from bids import BIDSLayout
import os

# Initialize BIDS layout
bids_root = '/data/bids_dataset'
layout = BIDSLayout(bids_root, derivatives=True)

# Get preprocessed fMRI files for specific task
subjects = layout.get_subjects()
task = 'faces'

for sub in subjects[:3]:  # First 3 subjects
    # Get preprocessed BOLD files
    bold_files = layout.get(
        subject=sub,
        task=task,
        suffix='bold',
        extension='nii.gz',
        space='MNI152NLin2009cAsym',
        return_type='file'
    )

    # Get brain mask
    mask_file = layout.get(
        subject=sub,
        suffix='mask',
        space='MNI152NLin2009cAsym',
        return_type='file'
    )[0]

    # Get events file
    events_file = layout.get(
        subject=sub,
        task=task,
        suffix='events',
        extension='tsv',
        return_type='file'
    )[0]

    print(f"Subject {sub}: {len(bold_files)} runs")
```

### Process Events File for Labels

```python
import pandas as pd

# Load events from BIDS
events_df = pd.read_csv(events_file, sep='\t')

# Extract trial types (labels)
trial_types = events_df['trial_type'].values

# Extract onsets for event-related averaging
onsets = events_df['onset'].values
durations = events_df['duration'].values

# Map to volume indices (assuming TR = 2s)
TR = 2.0
volume_indices = (onsets / TR).astype(int)

# Create targets array matching volumes
# (simplified - real implementation needs proper event modeling)
targets = np.array(['baseline'] * n_volumes)
for i, vol_idx in enumerate(volume_indices):
    targets[vol_idx] = trial_types[i]
```

### Batch Processing Multiple Subjects

```python
# Collect datasets from all subjects
all_datasets = []

for sub in subjects:
    # Load data (simplified)
    ds = fmri_dataset(
        samples=f'sub-{sub}_task-faces_bold.nii.gz',
        targets=f'sub-{sub}_targets.txt',
        chunks=f'sub-{sub}_chunks.txt',
        mask=f'sub-{sub}_mask.nii.gz'
    )

    # Preprocess
    poly_detrend(ds, polyord=1, chunks_attr='chunks')
    zscore(ds, chunks_attr='chunks')

    # Add subject attribute
    ds.sa['subject'] = [sub] * ds.nsamples

    all_datasets.append(ds)

# Combine for group analysis
group_dataset = vstack(all_datasets, a='all')
print(f"Group dataset: {group_dataset.shape}")
```

---

## Visualization

### Plot Classification Accuracy by ROI

```python
import seaborn as sns

# From earlier ROI analysis
roi_names = list(roi_accuracies.keys())
accuracies = list(roi_accuracies.values())

plt.figure(figsize=(10, 6))
sns.barplot(x=roi_names, y=accuracies)
plt.axhline(y=chance_level, color='r', linestyle='--', label='Chance')
plt.xlabel('ROI')
plt.ylabel('Classification Accuracy')
plt.title('Decoding Accuracy by ROI')
plt.legend()
plt.ylim([0, 1])
plt.savefig('roi_accuracies.png', dpi=150)
plt.close()
```

### Visualize Searchlight Results with Nilearn

```python
from nilearn import plotting

# Overlay searchlight map on anatomical
plotting.plot_stat_map(
    'searchlight_accuracy_map.nii.gz',
    bg_img='MNI152_T1_1mm.nii.gz',
    threshold=0.6,  # Show only above 60% accuracy
    display_mode='z',
    cut_coords=[-10, 0, 10, 20, 30],
    title='Searchlight Decoding Accuracy',
    colorbar=True
)
plt.savefig('searchlight_overlay.png', dpi=150)
plt.close()
```

---

## Best Practices and Tips

### Avoid Data Leakage

```python
# WRONG: Normalize before splitting into train/test
# This leaks test set statistics into training
zscore(dataset)  # Don't do this
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))

# CORRECT: Normalize within each CV fold
# Use chunks_attr to normalize separately per run
poly_detrend(dataset, polyord=1, chunks_attr='chunks')
zscore(dataset, chunks_attr='chunks')
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
```

### Cross-Validation Strategy Selection

```python
# For fMRI: respect run/session structure
# Use leave-one-run-out to avoid temporal autocorrelation
cv_run = CrossValidation(clf, NFoldPartitioner(attr='chunks'))

# For multi-subject: leave-one-subject-out
cv_subject = CrossValidation(clf, NFoldPartitioner(attr='subject'))

# For time-series: use temporal splits
# Don't use random k-fold with time-series data
```

### Feature Scaling for Different Classifiers

```python
# Linear SVM: benefits from normalization (already done with zscore)
# Decision trees/Random forest: don't require normalization
# RBF kernel SVM: definitely needs normalization

# Ensure normalization is done within CV folds
# PyMVPA handles this with chunks_attr parameter
```

---

## Troubleshooting

### Memory Issues with Large Datasets

```python
# Use HDF5 for large datasets
from mvpa2.base.hdf5 import h5save, h5load

# Save processed dataset
h5save('dataset_preprocessed.hdf5', dataset)

# Load when needed
dataset_loaded = h5load('dataset_preprocessed.hdf5')

# For searchlight: process in chunks or use nproc
sl = sphere_searchlight(
    cv,
    radius=3,
    nproc=4,  # Parallel processing
    space='voxel_indices'
)
```

### Handling Imbalanced Classes

```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
classes = np.unique(dataset.sa.targets)
class_weights = compute_class_weight('balanced', classes=classes, y=dataset.sa.targets)
weight_dict = dict(zip(classes, class_weights))

# Use weighted SVM
from sklearn.svm import SVC
clf_weighted = SKLLearnerAdapter(
    SVC(kernel='linear', class_weight=weight_dict)
)

cv_weighted = CrossValidation(clf_weighted, NFoldPartitioner(attr='chunks'))
acc_weighted = np.mean(cv_weighted(dataset))
```

### Debugging Low Classification Performance

```python
# Check class balance
print("Class distribution:")
print(pd.Series(dataset.sa.targets).value_counts())

# Check chance level
n_classes = len(np.unique(dataset.sa.targets))
chance = 1.0 / n_classes
print(f"Chance level: {chance:.3f}")

# Try simpler ROI-based analysis first
# Check data quality (motion, SNR)
# Verify preprocessing (detrending, normalization)
# Try different classifiers
# Examine confusion matrix for systematic errors
```

---

## Related Tools and Integration

**Preprocessing:**
- **fMRIPrep** (Batch 5): Preprocessing pipeline for PyMVPA input
- **Nilearn** (Batch 2): Alternative Python ML framework
- **SPM/FSL** (Batch 1): Classical preprocessing

**Advanced Analysis:**
- **BrainIAK** (Batch 26): Complementary MVPA methods (SRM, FCMA)
- **Nilearn.decoding**: Simpler decoding pipelines
- **PRoNTo** (Batch 27): MATLAB alternative with GUI

**Visualization:**
- **Nilearn.plotting**: Display statistical maps
- **FSLeyes** (Batch 7): Interactive visualization

---

## References

- Hanke, M., et al. (2009). PyMVPA: A Python toolbox for multivariate pattern analysis of fMRI data. *Neuroinformatics*, 7(1), 37-53.
- Kriegeskorte, N., et al. (2006). Information-based functional brain mapping. *PNAS*, 103(10), 3863-3868.
- Haxby, J. V., et al. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science*, 293(5539), 2425-2430.
- Poldrack, R. A., et al. (2020). Scanning the horizon: towards transparent and reproducible neuroimaging research. *Nature Reviews Neuroscience*, 21(3), 1-15.

**Official Documentation:** http://www.pymvpa.org/
**GitHub Repository:** https://github.com/PyMVPA/PyMVPA
**Tutorial:** http://www.pymvpa.org/tutorial.html
**Paper:** https://doi.org/10.1007/s12021-008-9041-y
