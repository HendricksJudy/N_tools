# MNE-Python

## Overview

MNE-Python is a comprehensive open-source Python package for exploring, visualizing, and analyzing MEG, EEG, and intracranial electrophysiological data. It provides state-of-the-art algorithms for preprocessing, source estimation, time-frequency analysis, connectivity, and statistics, with excellent visualization capabilities and integration with the scientific Python ecosystem.

**Website:** https://mne.tools/
**Platform:** Cross-platform (Linux/macOS/Windows)
**Language:** Python
**License:** BSD 3-Clause

## Key Features

- MEG, EEG, sEEG, ECoG, and NIRS support
- Comprehensive preprocessing pipeline
- Source localization (dSPM, sLORETA, LCMV, DICS)
- Time-frequency analysis
- Connectivity and network analysis
- Non-parametric cluster-based statistics
- Machine learning integration (scikit-learn)
- Interactive 3D visualization
- BIDS compatibility
- Integration with FieldTrip and EEGLAB

## Installation

```bash
# Using pip
pip install mne

# With all dependencies
pip install mne[full]

# Using conda (recommended)
conda install -c conda-forge mne

# Development version
pip install git+https://github.com/mne-tools/mne-python.git
```

### Additional Dependencies

```bash
# For 3D visualization
pip install pyvista pyvistaqt vtk

# For source analysis
pip install nibabel nilearn

# For parallel processing
pip install joblib

# Check installation
python -c "import mne; mne.sys_info()"
```

## Loading Data

```python
import mne
import numpy as np
import matplotlib.pyplot as plt

# Load example data
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                        'sample_audvis_raw.fif')

# Load raw data
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
print(raw.info)

# Load from other formats
raw_bdf = mne.io.read_raw_bdf('data.bdf', preload=True)  # BioSemi
raw_edf = mne.io.read_raw_edf('data.edf', preload=True)  # European Data Format
raw_cnt = mne.io.read_raw_cnt('data.cnt', preload=True)  # Neuroscan
raw_eeglab = mne.io.read_raw_eeglab('data.set', preload=True)  # EEGLAB

# Load from NumPy array
info = mne.create_info(ch_names=['Ch1', 'Ch2'], sfreq=250, ch_types='eeg')
raw_array = mne.io.RawArray(data, info)
```

## Preprocessing

### Filtering

```python
# High-pass filter (remove slow drifts)
raw_highpass = raw.copy().filter(l_freq=0.5, h_freq=None)

# Low-pass filter (anti-aliasing)
raw_lowpass = raw.copy().filter(l_freq=None, h_freq=40)

# Band-pass filter
raw_bandpass = raw.copy().filter(l_freq=1, h_freq=40)

# Notch filter (remove line noise)
raw_notch = raw.copy().notch_filter(freqs=[50, 100, 150])

# Or use built-in line noise removal
raw_clean = raw.copy().filter(l_freq=1, h_freq=40)
raw_clean.notch_filter([50, 100])
```

### Channel Operations

```python
# Set montage (electrode positions)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Re-reference to average
raw_avg_ref = raw.copy().set_eeg_reference('average')

# Re-reference to specific channels
raw_mastoid = raw.copy().set_eeg_reference(['M1', 'M2'])

# Interpolate bad channels
raw.info['bads'] = ['EEG 001', 'EEG 002']
raw_interp = raw.copy().interpolate_bads()

# Drop channels
raw.drop_channels(['EEG 064'])

# Pick channel types
raw_meg = raw.copy().pick_types(meg=True, eeg=False)
raw_eeg = raw.copy().pick_types(meg=False, eeg=True)
```

### Artifact Detection and Removal

```python
# Detect and remove bad channels
from mne.preprocessing import find_bad_channels_maxwell
raw_check = raw.copy()
noisy_chs, flat_chs = find_bad_channels_maxwell(raw_check)
raw.info['bads'] = noisy_chs + flat_chs

# Annotate bad segments
annotations = mne.Annotations(onset=[1, 5], duration=[0.5, 0.3],
                               description=['bad_movement', 'bad_blink'])
raw.set_annotations(annotations)

# ICA for artifact removal
from mne.preprocessing import ICA

ica = ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw, picks='eeg', reject={'eeg': 100e-6})

# Find EOG artifacts
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices

# Find ECG artifacts
ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
ica.exclude.extend(ecg_indices)

# Apply ICA
raw_clean = ica.apply(raw.copy())

# Visualize components
ica.plot_components()
ica.plot_sources(raw)
```

### Epoching

```python
# Find events
events = mne.find_events(raw, stim_channel='STI 014')

# Create epochs
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3}
tmin, tmax = -0.2, 0.5

epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), reject={'eeg': 100e-6},
                    preload=True)

# Equalize event counts
epochs.equalize_event_counts(['auditory/left', 'auditory/right'])

# Drop bad epochs
epochs.drop_bad(reject={'eeg': 100e-6, 'eog': 150e-6})

# Save epochs
epochs.save('epochs-epo.fif', overwrite=True)
```

## Event-Related Potentials/Fields

```python
# Compute evoked response (average across trials)
evoked = epochs.average()

# Plot ERP/ERF
evoked.plot(spatial_colors=True, gfp=True)

# Topographic maps
evoked.plot_topomap(times=[0, 0.1, 0.2, 0.3], ch_type='eeg')

# Joint plot (timecourse + topomap)
evoked.plot_joint()

# Compare conditions
evoked_left = epochs['auditory/left'].average()
evoked_right = epochs['auditory/right'].average()

# Contrast
evoked_diff = mne.combine_evoked([evoked_left, evoked_right], weights=[1, -1])
evoked_diff.plot_joint(title='Left - Right')

# Plot multiple conditions
mne.viz.plot_compare_evokeds({'Left': evoked_left, 'Right': evoked_right})
```

## Time-Frequency Analysis

### Morlet Wavelets

```python
# Define frequencies
freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs / 2.  # Different number of cycles per frequency

# Compute TFR
power = mne.time_frequency.tfr_morlet(
    epochs, freqs=freqs, n_cycles=n_cycles,
    use_fft=True, return_itc=False, decim=3, n_jobs=-1
)

# Plot
power.plot([0], baseline=(-0.5, 0), mode='logratio',
           title='Channel EEG 001', vmin=-1, vmax=1)

# Compute inter-trial coherence
power, itc = mne.time_frequency.tfr_morlet(
    epochs, freqs=freqs, n_cycles=n_cycles,
    use_fft=True, return_itc=True, decim=3, n_jobs=-1
)

itc.plot([0], baseline=None, title='ITC')
```

### Multitaper Method

```python
# Multitaper TFR
power_mt = mne.time_frequency.tfr_multitaper(
    epochs, freqs=freqs, n_cycles=n_cycles,
    time_bandwidth=2.0, return_itc=False, n_jobs=-1
)

power_mt.plot([0], baseline=(-0.5, 0), mode='logratio',
              title='Multitaper TFR')
```

### Power Spectral Density

```python
# Compute PSD
psd = epochs.compute_psd(method='welch', fmin=1, fmax=100,
                          n_fft=2048, n_overlap=512)

# Plot
psd.plot(picks='eeg', average=True)

# Or for raw data
raw.compute_psd(fmin=1, fmax=100).plot(picks='eeg', average=True)
```

## Source Estimation

### Setup Forward Model

```python
# Load FreeSurfer surfaces
subjects_dir = sample_data_folder / 'subjects'
subject = 'sample'

# Load BEM model
bem = mne.read_bem_surfaces(
    sample_data_folder / 'subjects' / subject / 'bem' /
    f'{subject}-5120-bem-sol.fif'
)

# Or create BEM
model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=[0.3],
                           subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(model)

# Create source space
src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir)

# Compute forward solution
fwd = mne.make_forward_solution(
    evoked.info, trans='sample-trans.fif', src=src,
    bem=bem_sol, meg=True, eeg=True, mindist=5.0, n_jobs=-1
)
```

### Minimum Norm Estimate (MNE)

```python
# Compute noise covariance
noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank=None
)

# Compute inverse operator
inverse_operator = mne.minimum_norm.make_inverse_operator(
    evoked.info, fwd, noise_cov, loose=0.2, depth=0.8
)

# Apply inverse solution
method = "dSPM"  # or "MNE", "sLORETA"
snr = 3.
lambda2 = 1. / snr ** 2

stc = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator, lambda2, method=method, pick_ori=None
)

# Visualize on brain
brain = stc.plot(subjects_dir=subjects_dir, subject=subject,
                 hemi='both', time_viewer=True,
                 views=['lateral', 'medial'])
```

### Beamformer (LCMV/DICS)

```python
# LCMV beamformer
from mne.beamformer import make_lcmv, apply_lcmv

# Compute data covariance
data_cov = mne.compute_covariance(epochs, tmin=0, tmax=None,
                                   method='empirical')

# Make LCMV beamformer
filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='unit-noise-gain')

# Apply beamformer
stc_lcmv = apply_lcmv(evoked, filters, max_ori_out='signed')

# DICS beamformer (frequency domain)
from mne.beamformer import make_dics, apply_dics_csd

# Compute cross-spectral density
csd = mne.time_frequency.csd_morlet(epochs, frequencies=[10],
                                     tmin=0, tmax=0.5, decim=20)

# Make DICS beamformer
filters = make_dics(evoked.info, fwd, csd, noise_csd=noise_cov,
                    pick_ori='max-power', weight_norm='unit-noise-gain')

# Apply beamformer
stc_dics, freqs = apply_dics_csd(csd, filters)
```

## Connectivity Analysis

```python
from mne.connectivity import spectral_connectivity_epochs, phase_slope_index

# Compute connectivity
con = spectral_connectivity_epochs(
    epochs, method=['coh', 'plv', 'pli', 'wpli'],
    mode='multitaper', sfreq=epochs.info['sfreq'],
    fmin=8, fmax=13, faverage=True, mt_adaptive=False, n_jobs=-1
)

# Extract coherence
coherence = con[0].get_data()

# Phase slope index
psi = phase_slope_index(epochs, mode='multitaper', sfreq=epochs.info['sfreq'],
                        fmin=8, fmax=13, n_jobs=-1)

# Source-space connectivity
from mne.minimum_norm import apply_inverse_epochs

snr = 3.0
lambda2 = 1.0 / snr ** 2
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method='dSPM',
                            pick_ori='normal', return_generator=True)

# Compute connectivity in source space
src_con = spectral_connectivity_epochs(
    stcs, method='coh', mode='multitaper', sfreq=epochs.info['sfreq'],
    fmin=8, fmax=13, n_jobs=-1
)
```

## Statistics

### Cluster-Based Permutation Test

```python
from mne.stats import permutation_cluster_test

# Prepare data
condition1 = epochs['auditory/left'].get_data()
condition2 = epochs['auditory/right'].get_data()

# Cluster-based permutation test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    [condition1, condition2], n_permutations=1000,
    threshold=None, tail=0, n_jobs=-1, out_type='mask'
)

# Plot significant clusters
times = epochs.times
fig, ax = plt.subplots()
ax.plot(times, T_obs.mean(axis=0), 'k-', label='T-statistic')
for i_c, c in enumerate(clusters):
    if cluster_p_values[i_c] < 0.05:
        h = ax.axvspan(times[c.start], times[c.stop-1],
                       color='r', alpha=0.3)
ax.legend()
ax.set(xlabel='Time (s)', ylabel='T-value')
```

### Spatio-Temporal Cluster Test

```python
from mne.stats import spatio_temporal_cluster_test

# Get connectivity (spatial adjacency)
connectivity, ch_names = mne.channels.find_ch_adjacency(evoked.info, ch_type='eeg')

# Cluster test
T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
    [condition1, condition2], n_permutations=1000,
    threshold=None, tail=0, n_jobs=-1, adjacency=connectivity
)

# Visualize
evoked_diff = mne.combine_evoked([evoked_left, evoked_right], weights=[1, -1])

# Mark significant sensors
significant_points = np.where(cluster_p_values < 0.05)[0]
print(f'Significant clusters: {len(significant_points)}')
```

## Machine Learning and Decoding

```python
from mne.decoding import (Scaler, Vectorizer, get_coef,
                          LinearModel, CSP, SlidingEstimator,
                          cross_val_multiscore)
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Prepare data
X = epochs.get_data()
y = epochs.events[:, 2]

# Classification pipeline
clf = make_pipeline(
    Scaler(epochs.info),
    Vectorizer(),
    LinearModel(LogisticRegression(solver='liblinear'))
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_multiscore(clf, X, y, cv=cv, n_jobs=-1)

print(f'Accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')

# Temporal decoding
time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)
scores_time = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=-1)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores_time.mean(0), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set(xlabel='Time (s)', ylabel='AUC')
ax.legend()
```

### Common Spatial Patterns (CSP)

```python
# CSP for motor imagery
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Create pipeline with CSP
clf_csp = make_pipeline(csp, LinearModel(LogisticRegression()))

# Cross-validation
scores_csp = cross_val_multiscore(clf_csp, X, y, cv=cv, n_jobs=-1)
print(f'CSP Accuracy: {np.mean(scores_csp):.3f}')

# Plot CSP patterns
csp.fit(X, y)
csp.plot_patterns(epochs.info)
```

## Visualization

```python
# Interactive data browser
raw.plot(duration=5, n_channels=30, scalings='auto')

# Interactive topomap
evoked.plot_topomap(times='auto', ch_type='eeg')

# 3D field maps
evoked.plot_field(surf_maps=['all'])

# Sensor positions
raw.plot_sensors(kind='3d', ch_type='eeg', show_names=True)

# Interactive brain visualization
stc.plot(subjects_dir=subjects_dir, subject=subject,
         surface='inflated', hemi='both', time_viewer=True)
```

## Integration with Claude Code

When helping users with MNE-Python:

1. **Check Installation:**
   ```python
   import mne
   mne.sys_info()
   ```

2. **Common Issues:**
   - 3D visualization requires pyvista/mayavi
   - FreeSurfer subjects_dir path
   - Forward model alignment errors
   - Memory issues with large datasets

3. **Best Practices:**
   - Always set montage for EEG
   - Use copy() to preserve original data
   - Preload data when doing heavy processing
   - Use n_jobs=-1 for parallel processing
   - Visualize at each processing step

4. **Performance:**
   - Use decimation to reduce data size
   - Enable CUDA for faster computations
   - Use memory mapping for very large files
   - Parallelize with n_jobs parameter

## Troubleshooting

**Problem:** "No module named 'pyvista'"
**Solution:** Install visualization dependencies: `pip install mne[full]`

**Problem:** FreeSurfer subject directory not found
**Solution:** Set MNE_DATA environment variable or pass subjects_dir explicitly

**Problem:** Memory errors
**Solution:** Use decimation, process in chunks, or don't preload data

**Problem:** Forward model misalignment
**Solution:** Use `mne.gui.coregistration()` to align sensors and head model

## Resources

- Documentation: https://mne.tools/stable/documentation.html
- Tutorials: https://mne.tools/stable/auto_tutorials/index.html
- Examples: https://mne.tools/stable/auto_examples/index.html
- Forum: https://mne.discourse.group/
- GitHub: https://github.com/mne-tools/mne-python
- Paper: https://doi.org/10.3389/fnins.2013.00267

## Related Tools

- **FieldTrip:** MATLAB alternative
- **EEGLAB:** MATLAB alternative
- **Brainstorm:** GUI-based alternative
- **Wonambi:** Python sleep analysis
- **autoreject:** Automated epoch rejection

## Citation

```bibtex
@article{gramfort2013mne,
  title={MEG and EEG data analysis with MNE-Python},
  author={Gramfort, Alexandre and Luessi, Martin and Larson, Eric and others},
  journal={Frontiers in Neuroscience},
  volume={7},
  pages={267},
  year={2013},
  doi={10.3389/fnins.2013.00267}
}
```
