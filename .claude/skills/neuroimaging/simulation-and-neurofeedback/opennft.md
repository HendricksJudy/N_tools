# OpenNFT: Real-Time fMRI Neurofeedback Platform

## Overview

OpenNFT (Open NeuroFeedback Training) is an open-source platform for conducting real-time fMRI neurofeedback experiments. It enables closed-loop paradigms where participants receive immediate feedback based on their ongoing brain activity, facilitating self-regulation training for therapeutic and research applications.

**Key Features:**
- **Real-Time Processing**: Low-latency fMRI preprocessing and analysis
- **Multiple Feedback Methods**: ROI-based, connectivity-based, multivariate pattern analysis (MVPA)
- **Flexible Feedback Presentation**: Visual, auditory, haptic interfaces
- **Scanner Support**: Compatible with Siemens, GE, Philips scanners
- **Offline Simulation**: Test protocols without scanner access
- **Python + MATLAB**: Dual implementation for flexibility
- **Quality Monitoring**: Real-time QC of motion and signal

**Website:** https://opennft.org/

**Citation:** Koush, Y., et al. (2017). OpenNFT: An open-source Python/Matlab framework for real-time fMRI neurofeedback training based on activity, connectivity and multivariate pattern analysis. *NeuroImage*, 156, 489-503.

## Installation

### System Requirements

- **Computer**: Dedicated workstation near scanner
- **OS**: Linux (Ubuntu 18.04+), Windows 10+, macOS
- **RAM**: 16GB minimum, 32GB recommended
- **Python**: 3.7+ or MATLAB R2017a+
- **Network**: Fast connection to scanner (for DICOM transfer)

### Python Installation

```bash
# Clone repository
git clone https://github.com/OpenNFT/OpenNFT.git
cd OpenNFT

# Create conda environment
conda create -n opennft python=3.8
conda activate opennft

# Install dependencies
pip install -r requirements.txt

# Install OpenNFT
pip install -e .

# Download SPM standalone (for preprocessing)
python opennft/download_spm.py

# Test installation
python tests/test_installation.py
```

### MATLAB Installation

```bash
# Clone repository
git clone https://github.com/OpenNFT/OpenNFT.git

# Open MATLAB
matlab

# In MATLAB, navigate to OpenNFT directory and run:
>> setupOpenNFT

# Add SPM12 to path
>> addpath('/path/to/spm12')

# Test installation
>> testOpenNFT
```

### Scanner-Side Configuration

```bash
# Configure DICOM receiver on experiment computer

# Option 1: Built-in DICOM listener
python opennft/utils/start_dicom_receiver.py --port 4006

# Option 2: Use Orthanc server
docker run -p 4242:4242 -p 8042:8042 jodogne/orthanc

# Configure scanner to send DICOM to experiment computer IP
# Scanner settings: Export to network node
#   - IP: <experiment_computer_IP>
#   - Port: 4006 (or 4242 for Orthanc)
#   - AE Title: OPENNFT
```

## Setting Up a Neurofeedback Experiment

### Experimental Design Considerations

**Key Design Elements:**
1. **Baseline/Localizer**: Identify target regions (10-15 min)
2. **Neurofeedback Runs**: Training with feedback (4-6 runs × 8-10 min)
3. **Transfer Run**: Test regulation without feedback (8-10 min)
4. **Control Condition**: Sham feedback or alternative target

**Typical Timeline:**
```
Session 1: Structural MRI + Localizer + 2 NF runs
Session 2-4: 4-6 NF runs per session
Session 5: Transfer run + follow-up structural
```

### Creating a Study Protocol

```python
# opennft/configs/emotion_regulation_config.py

import os

# Study parameters
STUDY_NAME = 'amygdala_downregulation'
SUBJECT_PREFIX = 'sub'
N_SESSIONS = 4
N_RUNS = 4

# Scanner parameters
TR = 2.0  # Repetition time (seconds)
N_VOLUMES = 240  # Volumes per run
SLICE_ORDER = list(range(1, 41))  # Interleaved
MULTIBAND_FACTOR = 2

# ROI definition
ROI_NAME = 'bilateral_amygdala'
ROI_MASK = '/path/to/amygdala_mask.nii'

# Preprocessing parameters
REALIGNMENT = True
SMOOTHING_FWHM = 6  # mm
DETRENDING = 'linear'
TEMPORAL_FILTERING = [0.01, 0.1]  # Hz

# Feedback parameters
FEEDBACK_TYPE = 'percent_signal_change'
BASELINE_VOLUMES = 20  # Initial volumes for baseline
DISPLAY_TYPE = 'thermometer'
UPDATE_RATE = 'TR'  # Update every TR

# Paths
DATA_DIR = '/data/neurofeedback'
OUTPUT_DIR = '/results/neurofeedback'
DICOM_DIR = '/incoming/dicom'
```

### Defining Regions of Interest

```python
# Method 1: Use anatomical atlas
from nilearn import datasets, image

# Load Harvard-Oxford atlas
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
atlas_img = image.load_img(atlas.maps)

# Extract amygdala (bilateral)
amygdala_mask = image.math_img('img == 10', img=atlas_img)
amygdala_mask.to_filename('amygdala_roi.nii.gz')

# Method 2: Use functional localizer
from nilearn.glm.first_level import FirstLevelModel

# Run localizer task (e.g., emotional faces)
model = FirstLevelModel(t_r=2.0, smoothing_fwhm=6)
model.fit(localizer_files, events=localizer_events)

# Compute contrast
z_map = model.compute_contrast('faces > baseline', output_type='z_score')

# Threshold and create ROI
from nilearn.image import threshold_img, math_img
roi_mask = threshold_img(z_map, threshold=3.1)  # p < 0.001
roi_mask.to_filename('amygdala_functional_roi.nii.gz')

# Method 3: Manual drawing
# Use FSLeyes, MRIcroGL, or 3D Slicer to draw ROI manually
```

## Real-Time Preprocessing

### Motion Correction

```python
# Real-time motion correction using SPM realignment

from opennft.preprocessing import RealTimeRealignment

# Initialize realigner
realigner = RealTimeRealignment(
    reference_volume='/path/to/mean_epi.nii',
    quality=0.9,
    separation=4,
    smoothing=5
)

# Process incoming volume
def process_new_volume(volume_path):
    # Realign to reference
    realigned, params = realigner.realign(volume_path)

    # Check motion parameters
    translation = params[:3]  # x, y, z
    rotation = params[3:]  # pitch, roll, yaw

    if max(abs(translation)) > 3.0:  # 3mm threshold
        print(f"WARNING: Excessive motion detected: {translation}")

    return realigned, params
```

### Temporal Filtering

```python
# Real-time high-pass filtering

from opennft.preprocessing import RealTimeFilter
import numpy as np

# Initialize filter
hp_filter = RealTimeFilter(
    tr=2.0,
    cutoff_freq=0.01,  # Hz
    filter_type='butterworth',
    order=2
)

# Process timeseries
timeseries = []  # Accumulating volumes

def apply_filtering(new_volume_data, roi_mask):
    # Extract ROI signal
    roi_signal = new_volume_data[roi_mask > 0].mean()
    timeseries.append(roi_signal)

    # Apply causal filtering (only uses past data)
    filtered_signal = hp_filter.filter_online(timeseries)

    return filtered_signal[-1]  # Return latest filtered value
```

### Baseline Estimation

```python
# Establish baseline during initial volumes

class BaselineEstimator:
    def __init__(self, n_baseline_volumes=20):
        self.n_baseline = n_baseline_volumes
        self.baseline_values = []
        self.baseline_mean = None
        self.baseline_std = None

    def add_volume(self, signal_value):
        if len(self.baseline_values) < self.n_baseline:
            self.baseline_values.append(signal_value)

            if len(self.baseline_values) == self.n_baseline:
                self.baseline_mean = np.mean(self.baseline_values)
                self.baseline_std = np.std(self.baseline_values)
                print(f"Baseline established: {self.baseline_mean:.3f} ± {self.baseline_std:.3f}")

    def compute_percent_change(self, signal_value):
        if self.baseline_mean is None:
            return 0.0
        return 100 * (signal_value - self.baseline_mean) / self.baseline_mean
```

## Feedback Computation Methods

### ROI-Based Feedback

```python
# Simple ROI mean activation

from opennft.feedback import ROIFeedback
import nibabel as nib

class AmygdalaFeedback(ROIFeedback):
    def __init__(self, roi_mask_path):
        self.roi_mask = nib.load(roi_mask_path).get_fdata() > 0

    def compute_feedback(self, volume_data, baseline_mean):
        # Extract ROI mean
        roi_mean = volume_data[self.roi_mask].mean()

        # Compute percent signal change
        psc = 100 * (roi_mean - baseline_mean) / baseline_mean

        # Return feedback signal (negative = successful downregulation)
        return -psc  # Flip sign so higher = better performance

# Usage
feedback_computer = AmygdalaFeedback('amygdala_roi.nii.gz')

def process_volume(volume_path, baseline_mean):
    volume_data = nib.load(volume_path).get_fdata()
    feedback_value = feedback_computer.compute_feedback(volume_data, baseline_mean)
    return feedback_value
```

### Multivariate Pattern Analysis (MVPA)

```python
# Classifier-based feedback (e.g., emotion decoding)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

class MVPAFeedback:
    def __init__(self, mask_path):
        self.mask = nib.load(mask_path).get_fdata() > 0
        self.classifier = None
        self.scaler = StandardScaler()

    def train_classifier(self, training_data, training_labels):
        """
        Train classifier on localizer data
        training_data: list of 3D volumes
        training_labels: corresponding class labels (e.g., 0=neutral, 1=happy)
        """
        # Extract features
        n_samples = len(training_data)
        n_features = self.mask.sum()
        X = np.zeros((n_samples, n_features))

        for i, volume in enumerate(training_data):
            X[i, :] = volume[self.mask]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train SVM
        self.classifier = SVC(kernel='linear', probability=True)
        self.classifier.fit(X_scaled, training_labels)

        print(f"Classifier trained: accuracy = {self.classifier.score(X_scaled, training_labels):.3f}")

    def compute_feedback(self, volume_data):
        """
        Compute probability of target class
        """
        # Extract features
        features = volume_data[self.mask].reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # Get probability
        proba = self.classifier.predict_proba(features_scaled)[0]

        # Return probability of class 1 (target emotion)
        return proba[1]
```

### Connectivity-Based Feedback

```python
# Functional connectivity neurofeedback

from opennft.feedback import ConnectivityFeedback
from nilearn.connectome import ConnectivityMeasure

class AmygdalaPFCConnectivity(ConnectivityFeedback):
    def __init__(self, amygdala_mask, pfc_mask):
        self.amygdala_mask = nib.load(amygdala_mask).get_fdata() > 0
        self.pfc_mask = nib.load(pfc_mask).get_fdata() > 0
        self.amygdala_ts = []
        self.pfc_ts = []
        self.window_size = 20  # Volumes for sliding window

    def update(self, volume_data):
        # Extract regional timeseries
        amyg_signal = volume_data[self.amygdala_mask].mean()
        pfc_signal = volume_data[self.pfc_mask].mean()

        self.amygdala_ts.append(amyg_signal)
        self.pfc_ts.append(pfc_signal)

        # Compute correlation (sliding window)
        if len(self.amygdala_ts) >= self.window_size:
            window_amyg = self.amygdala_ts[-self.window_size:]
            window_pfc = self.pfc_ts[-self.window_size:]

            correlation = np.corrcoef(window_amyg, window_pfc)[0, 1]
            return correlation
        else:
            return 0.0
```

## Feedback Presentation

### Visual Thermometer Feedback

```python
# PsychoPy-based visual feedback

from psychopy import visual, core, event
import numpy as np

class ThermometerFeedback:
    def __init__(self, screen_resolution=(1920, 1080), fullscreen=True):
        self.win = visual.Window(
            size=screen_resolution,
            fullscr=fullscreen,
            color='black',
            units='norm'
        )

        # Create thermometer components
        self.background = visual.Rect(
            self.win,
            width=0.2,
            height=1.5,
            fillColor='gray',
            pos=(0, 0)
        )

        self.fill = visual.Rect(
            self.win,
            width=0.18,
            height=0.0,  # Will update
            fillColor='red',
            pos=(0, -0.75)  # Bottom of thermometer
        )

        self.target_line = visual.Line(
            self.win,
            start=(-0.15, 0),
            end=(0.15, 0),
            lineColor='green',
            lineWidth=3
        )

    def update(self, feedback_value, target_value=0.5):
        """
        Update thermometer display
        feedback_value: 0-1 scale
        """
        # Scale fill height
        fill_height = feedback_value * 1.5
        self.fill.height = fill_height
        self.fill.pos = (0, -0.75 + fill_height / 2)

        # Update target line position
        target_y = -0.75 + target_value * 1.5
        self.target_line.start = (-0.15, target_y)
        self.target_line.end = (0.15, target_y)

        # Draw
        self.background.draw()
        self.fill.draw()
        self.target_line.draw()
        self.win.flip()

    def close(self):
        self.win.close()

# Usage in neurofeedback loop
display = ThermometerFeedback()

for volume_idx in range(n_volumes):
    # Compute feedback
    feedback = compute_feedback(volume_idx)

    # Normalize to 0-1
    feedback_norm = (feedback + 2) / 4  # Assuming range [-2, 2]
    feedback_norm = np.clip(feedback_norm, 0, 1)

    # Update display
    display.update(feedback_norm, target_value=0.3)

    # Check for quit
    if 'escape' in event.getKeys():
        break

display.close()
```

### Intermittent Feedback Protocol

```python
# Present feedback only during specific blocks

class IntermittentFeedbackSchedule:
    def __init__(self, tr, block_duration=30, rest_duration=20):
        self.tr = tr
        self.block_duration = block_duration
        self.rest_duration = rest_duration
        self.cycle_duration = block_duration + rest_duration

    def is_feedback_active(self, volume_idx):
        """
        Determine if feedback should be shown
        """
        time_in_cycle = (volume_idx * self.tr) % self.cycle_duration
        return time_in_cycle < self.block_duration

# Usage
schedule = IntermittentFeedbackSchedule(tr=2.0)

for vol_idx in range(n_volumes):
    feedback_value = compute_feedback(vol_idx)

    if schedule.is_feedback_active(vol_idx):
        display.update(feedback_value)
    else:
        display.show_rest_screen()
```

## Quality Control During Acquisition

### Real-Time Motion Monitoring

```python
# Monitor and alert for excessive motion

class MotionMonitor:
    def __init__(self, translation_threshold=3.0, rotation_threshold=3.0):
        self.trans_thresh = translation_threshold  # mm
        self.rot_thresh = rotation_threshold  # degrees
        self.motion_history = []

    def check_motion(self, motion_params):
        """
        motion_params: [tx, ty, tz, rx, ry, rz]
        """
        translation = motion_params[:3]
        rotation = np.degrees(motion_params[3:])  # Convert to degrees

        max_trans = np.max(np.abs(translation))
        max_rot = np.max(np.abs(rotation))

        self.motion_history.append({
            'translation': max_trans,
            'rotation': max_rot
        })

        # Alert if exceeds threshold
        if max_trans > self.trans_thresh:
            print(f"⚠️  MOTION ALERT: Translation = {max_trans:.2f} mm")
            return 'high_motion'
        elif max_rot > self.rot_thresh:
            print(f"⚠️  MOTION ALERT: Rotation = {max_rot:.2f}°")
            return 'high_motion'
        else:
            return 'ok'

    def get_summary(self):
        """
        Return motion summary statistics
        """
        translations = [h['translation'] for h in self.motion_history]
        rotations = [h['rotation'] for h in self.motion_history]

        return {
            'mean_translation': np.mean(translations),
            'max_translation': np.max(translations),
            'mean_rotation': np.mean(rotations),
            'max_rotation': np.max(rotations),
            'n_high_motion_volumes': sum(1 for t in translations if t > self.trans_thresh)
        }
```

### Signal Quality Assessment

```python
# Monitor tSNR and signal drift

class SignalQualityMonitor:
    def __init__(self, roi_mask):
        self.roi_mask = roi_mask
        self.signals = []

    def update(self, volume_data):
        roi_mean = volume_data[self.roi_mask].mean()
        self.signals.append(roi_mean)

    def compute_tsnr(self):
        """
        Temporal SNR = mean / std over time
        """
        if len(self.signals) < 10:
            return None

        mean_signal = np.mean(self.signals)
        std_signal = np.std(self.signals)

        tsnr = mean_signal / std_signal if std_signal > 0 else 0
        return tsnr

    def detect_drift(self):
        """
        Check for linear drift in signal
        """
        if len(self.signals) < 20:
            return None

        x = np.arange(len(self.signals))
        slope, intercept = np.polyfit(x, self.signals, 1)

        # Drift as percentage per 100 volumes
        drift_pct = 100 * slope * 100 / np.mean(self.signals)

        return drift_pct
```

## Offline Analysis

### Analyzing Neurofeedback Success

```python
# Post-hoc analysis of regulation success

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def analyze_nf_session(session_dir):
    """
    Analyze single neurofeedback session
    """
    # Load feedback log
    feedback_log = pd.read_csv(f'{session_dir}/feedback_log.csv')

    # Separate regulation vs. rest blocks
    regulate_blocks = feedback_log[feedback_log['condition'] == 'regulate']
    rest_blocks = feedback_log[feedback_log['condition'] == 'rest']

    # Compute mean feedback during regulation
    mean_regulate = regulate_blocks['feedback_value'].mean()
    mean_rest = rest_blocks['feedback_value'].mean()

    # Statistical test
    t_stat, p_value = stats.ttest_ind(
        regulate_blocks['feedback_value'],
        rest_blocks['feedback_value']
    )

    print(f"Regulate: {mean_regulate:.3f}")
    print(f"Rest: {mean_rest:.3f}")
    print(f"t({t_stat:.2f}), p={p_value:.4f}")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(feedback_log['volume'], feedback_log['feedback_value'], label='Feedback')
    plt.axhline(mean_rest, color='gray', linestyle='--', label='Rest mean')
    plt.xlabel('Volume')
    plt.ylabel('Feedback Signal')
    plt.legend()
    plt.title(f'Neurofeedback Session: p={p_value:.4f}')
    plt.savefig(f'{session_dir}/feedback_timecourse.png', dpi=300)

    return {'mean_regulate': mean_regulate, 'mean_rest': mean_rest, 'p_value': p_value}
```

### Learning Curves Across Sessions

```python
# Analyze improvement over multiple sessions

def analyze_learning_curve(subject_dir):
    """
    Track neurofeedback performance across sessions
    """
    sessions = sorted(glob.glob(f'{subject_dir}/session*/'))

    performance = []

    for session in sessions:
        result = analyze_nf_session(session)
        performance.append(result)

    # Plot learning curve
    session_nums = range(1, len(sessions) + 1)
    regulate_means = [p['mean_regulate'] for p in performance]

    plt.figure(figsize=(8, 6))
    plt.plot(session_nums, regulate_means, marker='o', linewidth=2)
    plt.xlabel('Session Number')
    plt.ylabel('Mean Regulation Performance')
    plt.title('Neurofeedback Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{subject_dir}/learning_curve.png', dpi=300)

    # Fit linear trend
    slope, intercept = np.polyfit(session_nums, regulate_means, 1)
    print(f"Learning rate: {slope:.4f} per session")

    return performance
```

### Transfer Effects

```python
# Assess regulation ability without feedback

def analyze_transfer_run(transfer_dir, baseline_mean):
    """
    Analyze transfer run (no feedback)
    """
    # Load timeseries data
    timeseries = pd.read_csv(f'{transfer_dir}/timeseries.csv')

    # Separate instruction blocks
    regulate_vols = timeseries[timeseries['instruction'] == 'regulate']
    rest_vols = timeseries[timeseries['instruction'] == 'rest']

    # Compute percent signal change
    regulate_psc = 100 * (regulate_vols['signal'].mean() - baseline_mean) / baseline_mean
    rest_psc = 100 * (rest_vols['signal'].mean() - baseline_mean) / baseline_mean

    # Test for significant difference
    t_stat, p_value = stats.ttest_ind(regulate_vols['signal'], rest_vols['signal'])

    print(f"Transfer Performance:")
    print(f"  Regulate: {regulate_psc:.2f}% signal change")
    print(f"  Rest: {rest_psc:.2f}% signal change")
    print(f"  t={t_stat:.2f}, p={p_value:.4f}")

    return {
        'regulate_psc': regulate_psc,
        'rest_psc': rest_psc,
        'transfer_success': p_value < 0.05
    }
```

## Advanced Features

### Dynamic Connectivity Feedback

```python
# Real-time dynamic connectivity

from nilearn.connectome import ConnectivityMeasure

class DynamicConnectivityNF:
    def __init__(self, roi_masks, window_size=30):
        self.roi_masks = [nib.load(m).get_fdata() > 0 for m in roi_masks]
        self.window_size = window_size
        self.timeseries = {i: [] for i in range(len(roi_masks))}
        self.conn_measure = ConnectivityMeasure(kind='correlation')

    def update(self, volume_data):
        # Extract signals from all ROIs
        for i, mask in enumerate(self.roi_masks):
            signal = volume_data[mask].mean()
            self.timeseries[i].append(signal)

        # Compute connectivity matrix (sliding window)
        if len(self.timeseries[0]) >= self.window_size:
            # Get recent data
            ts_matrix = np.array([
                self.timeseries[i][-self.window_size:]
                for i in range(len(self.roi_masks))
            ]).T

            # Compute connectivity
            conn_matrix = self.conn_measure.fit_transform([ts_matrix])[0]

            return conn_matrix
        else:
            return None
```

### Adaptive Baseline

```python
# Update baseline adaptively during rest blocks

class AdaptiveBaseline:
    def __init__(self, initial_baseline):
        self.baseline = initial_baseline
        self.rest_values = []

    def update(self, signal_value, is_rest_block):
        if is_rest_block:
            self.rest_values.append(signal_value)

            # Update baseline every 10 rest volumes
            if len(self.rest_values) >= 10:
                self.baseline = np.mean(self.rest_values[-10:])
                print(f"Baseline updated: {self.baseline:.3f}")

    def get_baseline(self):
        return self.baseline
```

## Troubleshooting

### DICOM Transfer Issues

```bash
# Test DICOM connectivity

# 1. Check network connectivity
ping <scanner_IP>

# 2. Test DICOM receiver
python opennft/utils/test_dicom_receiver.py --port 4006

# 3. Send test DICOM
dcmsend <experiment_computer_IP> 4006 test_image.dcm

# 4. Check firewall settings
sudo ufw allow 4006/tcp
```

### Timing Delays

**Problem:** Feedback lags behind acquisition

```python
# Profile processing time

import time

def profile_processing():
    times = {
        'dicom_load': [],
        'preprocessing': [],
        'feedback_computation': [],
        'display_update': []
    }

    # During processing
    t0 = time.time()
    volume = load_dicom(dicom_file)
    times['dicom_load'].append(time.time() - t0)

    t0 = time.time()
    preprocessed = preprocess(volume)
    times['preprocessing'].append(time.time() - t0)

    # ... etc

    # Report
    print("Processing time breakdown:")
    for step, time_list in times.items():
        print(f"  {step}: {np.mean(time_list):.3f}s")

# Optimize slow steps:
# - Use SPM standalone (faster than MATLAB)
# - Reduce smoothing kernel
# - Simplify ROI masks
# - Pre-load reference volumes
```

### Artifact Handling

```python
# Detect and handle artifacts

def detect_artifact(volume_data, previous_data, threshold=5.0):
    """
    Detect sudden signal changes (potential artifact)
    """
    if previous_data is None:
        return False

    # Compute volume-to-volume difference
    diff = np.abs(volume_data - previous_data)
    mean_diff = diff.mean()
    std_diff = diff.std()

    # Z-score
    z_score = mean_diff / std_diff if std_diff > 0 else 0

    if z_score > threshold:
        print(f"⚠️  Artifact detected (z={z_score:.2f})")
        return True

    return False

# In processing loop
previous_volume = None

for volume in volumes:
    if detect_artifact(volume, previous_volume):
        # Skip this volume or use interpolation
        feedback_value = previous_feedback_value
    else:
        feedback_value = compute_feedback(volume)

    previous_volume = volume
    previous_feedback_value = feedback_value
```

## Best Practices

### Participant Instruction

**Before Scanning:**
- Explain neurofeedback concept clearly
- Practice with mock feedback (outside scanner)
- Emphasize strategies are individual
- Set realistic expectations (gradual learning)

**During Scanning:**
- Provide clear visual instructions
- Use familiar feedback displays
- Give encouragement between runs
- Monitor comfort and motion

**Regulation Strategies:**
- Imagery (e.g., calming scenes for amygdala down-regulation)
- Cognitive reappraisal
- Attention redirection
- Emotion labeling

### Control Conditions

**Essential Controls:**
1. **Sham Feedback**: Yoked feedback from another participant
2. **Alternative Target**: Control region neurofeedback
3. **No Feedback**: Instruction-only runs
4. **Baseline Comparison**: Rest vs. regulate

### Reporting Standards

**Methods should include:**
- OpenNFT version
- Processing delay (TR lag)
- ROI definition method
- Feedback computation algorithm
- Display parameters
- Quality control criteria
- Number of volumes excluded

## Integration

- **Real-time data flow:** Use LabStreamingLayer (LSL) to ingest scanner data and route to OpenNFT with minimal latency.
- **Hardware sanity checks:** Verify projector/response devices and low-latency network paths before participant runs.
- **Dry runs:** Replay recorded BOLD data to test ROI selection, feedback timing, and crash recovery pre-scan.

## Resources

- Documentation: https://opennft.readthedocs.io/
- Source: https://github.com/OpenNFT/OpenNFT
- Community: https://neurostars.org/ (tag `opennft`)
- Google Group: https://groups.google.com/g/opennft
- Example datasets: https://opennft.readthedocs.io/en/latest/datasets.html

## Related Tools

- **BrainIAK real-time fMRI:** Alternative real-time modules
- **SIMNIBS:** Field modeling for stimulation planning
- **TVB:** Network simulations that can integrate with feedback paradigms
- **LabStreamingLayer:** Low-latency data bus for acquisition and presentation

## References

- **OpenNFT Main Paper**: Koush, Y., et al. (2017). OpenNFT: An open-source Python/Matlab framework for real-time fMRI neurofeedback training. *NeuroImage*, 156, 489-503.
- **Review**: Sulzer, J., et al. (2013). Real-time fMRI neurofeedback: Progress and challenges. *NeuroImage*, 76, 386-399.
- **Clinical Applications**: Thibault, R. T., et al. (2016). Neurofeedback with fMRI: A critical systematic review. *NeuroImage*, 172, 786-807.
- **Best Practices**: Ros, T., et al. (2020). Consensus on the reporting and experimental design of clinical and cognitive-behavioral neurofeedback studies. *Brain*, 143(12), 3461-3480.
- **Website**: https://opennft.org/
- **Documentation**: https://opennft.readthedocs.io/
- **Forum**: https://groups.google.com/g/opennft
