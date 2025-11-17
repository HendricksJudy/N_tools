# Brainstorm

## Overview

Brainstorm is a collaborative, open-source MATLAB application dedicated to MEG/EEG data analysis, visualization, and source imaging. It features an intuitive GUI, comprehensive analysis pipelines, and advanced source reconstruction methods, making complex analyses accessible without extensive programming knowledge.

**Website:** https://neuroimage.usc.edu/brainstorm/
**Platform:** MATLAB (Windows/macOS/Linux) or standalone
**Language:** MATLAB
**License:** GNU GPL v3

## Key Features

- User-friendly GUI for MEG/EEG analysis
- Comprehensive preprocessing pipeline
- Advanced source imaging (MNE, dSPM, sLORETA, LCMV)
- Time-frequency analysis
- Connectivity analysis (coherence, Granger, PLV)
- Anatomical MRI processing
- Integration with FreeSurfer
- Group analysis and statistics
- Interactive 3D visualization
- Plugin system for extensions
- Standalone version (no MATLAB license required)

## Installation

### MATLAB Version

```matlab
% Download from: https://neuroimage.usc.edu/brainstorm/Download
% Extract to desired location

% Start Brainstorm
brainstorm

% Or add to path and start
addpath('/path/to/brainstorm3');
brainstorm
```

### Standalone Version

```bash
# Download standalone installer for your OS
# No MATLAB license required
# Includes MATLAB Runtime

# Extract and run
./brainstorm3/bin/brainstorm.sh  # Linux/macOS
brainstorm3\bin\brainstorm.bat   # Windows
```

### Database Location

```matlab
% Set database location on first start
% Recommended: Create dedicated folder
% Example: /home/user/brainstorm_db
```

## Getting Started

### Creating a Protocol

```matlab
% Start Brainstorm
brainstorm

% From GUI: Create new protocol
% File > New protocol
% Name: MyStudy
% Set anatomy defaults (use MNI template or FreeSurfer)

% Or programmatically
gui_brainstorm('CreateProtocol', 'MyStudy', 0, 0);
% Parameters: UseDefaultAnat, UseDefaultChannel
```

### Importing Data

```matlab
% Import anatomy (MRI)
% Right-click on subject > Import MRI
% Supported: NIfTI, Analyze, MINC, CTF, etc.

% Import recordings
% Right-click on condition > Import MEG/EEG
% Supported: CTF, Neuromag, BrainVision, EEGLAB, FieldTrip, etc.

% Programmatic import
SubjectName = 'Subject01';
ConditionName = 'Rest';

% Import CTF data
DataFile = '/data/subject01.ds';
bst_process('CallProcess', 'process_import_data_raw', [], [], ...
    'subjectname', SubjectName, ...
    'datafile', {DataFile, 'CTF'}, ...
    'channelalign', 1);
```

## Preprocessing

### Artifact Detection and Removal

```matlab
% Detect artifacts
% Process > Artifacts > Detect heartbeats (ECG)
% Process > Artifacts > Detect eye blinks (EOG)

% Or use scripting
sFiles = bst_process('CallProcess', 'process_evt_detect_ecg', sFiles, []);
sFiles = bst_process('CallProcess', 'process_evt_detect_eog', sFiles, []);

% SSP (Signal-Space Projection) for artifact removal
sFiles = bst_process('CallProcess', 'process_ssp_ecg', sFiles, []);
sFiles = bst_process('CallProcess', 'process_ssp_eog', sFiles, []);

% Or ICA
sFiles = bst_process('CallProcess', 'process_ica', sFiles, [], ...
    'timewindow', [], ...
    'eventname', '', ...
    'saveerp', 0, ...
    'icasort', 1, ...
    'usessp', 1);
```

### Filtering

```matlab
% Band-pass filter
sFiles = bst_process('CallProcess', 'process_bandpass', sFiles, [], ...
    'highpass', 1, ...
    'lowpass', 40, ...
    'attenuation', 'strict', ...
    'mirror', 0, ...
    'useold', 0);

% Notch filter (remove line noise)
sFiles = bst_process('CallProcess', 'process_notch', sFiles, [], ...
    'freqlist', [50, 100, 150], ...
    'sensortypes', 'MEG, EEG', ...
    'read_all', 0);
```

### Epoching

```matlab
% Import epochs around events
sFiles = bst_process('CallProcess', 'process_import_data_event', sFiles, [], ...
    'subjectname', SubjectName, ...
    'condition', '', ...
    'eventname', 'stimulus', ...
    'timewindow', [], ...
    'epochtime', [-0.2, 0.5], ...
    'createcond', 1, ...
    'ignoreshort', 1, ...
    'usectfcomp', 1, ...
    'usessp', 1, ...
    'freq', [], ...
    'baseline', [-0.2, 0]);
```

## Anatomy Processing

### MRI Segmentation

```matlab
% Generate head surfaces from MRI
% Anatomy > Generate surfaces

% Or use process
sFiles = bst_process('CallProcess', 'process_generate_bem', sFiles, [], ...
    'subjectname', SubjectName, ...
    'nscalp', 1922, ...
    'nouter', 1922, ...
    'ninner', 1922, ...
    'thickness', 4);
```

### FreeSurfer Integration

```matlab
% Import FreeSurfer surfaces
% File > Import anatomy folder
% Select FreeSurfer subject directory

% Or programmatic
bst_process('CallProcess', 'process_import_anatomy', [], [], ...
    'subjectname', SubjectName, ...
    'mrifile', {'/path/to/freesurfer/subjects/subject01', 'FreeSurfer'}, ...
    'nvertices', 15000);
```

### Co-registration

```matlab
% Align sensors with head surface
% Channel file > MRI Registration
% Use fiducials (nasion, LPA, RPA)

% Or automatic alignment
sFiles = bst_process('CallProcess', 'process_headpoints_refine', sFiles, []);
```

## Source Estimation

### Forward Model

```matlab
% Compute forward model (lead field)
% Process > Sources > Compute head model

sFiles = bst_process('CallProcess', 'process_headmodel', sFiles, [], ...
    'sourcespace', 1, ...  % Cortex surface
    'volumegrid', [], ...
    'meg', 3, ...  % Overlapping spheres
    'eeg', 3, ...  % BEM
    'ecog', 2, ...
    'seeg', 2, ...
    'openmeeg', struct(...
         'BemSelect', [1, 1, 1], ...
         'BemCond', [1, 0.0125, 1], ...
         'BemNames', {{'Scalp', 'Skull', 'Brain'}}, ...
         'BemFiles', {{}}));
```

### Noise Covariance

```matlab
% Compute noise covariance from baseline
sFiles = bst_process('CallProcess', 'process_noisecov', sFiles, [], ...
    'baseline', [-0.2, 0], ...
    'datatimewindow', [], ...
    'sensortypes', 'MEG, EEG', ...
    'target', 1, ...  % Noise covariance
    'dcoffset', 1, ...
    'identity', 0, ...
    'copycond', 0, ...
    'copysubj', 0, ...
    'copymatch', 0, ...
    'replacefile', 1);
```

### Source Reconstruction

```matlab
% Minimum norm estimate (MNE)
sFiles = bst_process('CallProcess', 'process_inverse_2018', sFiles, [], ...
    'output', 1, ...  % Kernel only
    'inverse', struct(...
         'Comment', 'MN: MEG', ...
         'InverseMethod', 'minnorm', ...
         'InverseMeasure', 'amplitude', ...
         'SourceOrient', {{'fixed'}}, ...
         'Loose', 0.2, ...
         'UseDepth', 1, ...
         'WeightExp', 0.5, ...
         'WeightLimit', 10, ...
         'NoiseMethod', 'reg', ...
         'NoiseReg', 0.1, ...
         'SnrMethod', 'fixed', ...
         'SnrRms', 1e-06, ...
         'SnrFixed', 3, ...
         'ComputeKernel', 1, ...
         'DataTypes', {{'MEG'}}));

% Beamformer (LCMV)
sFiles = bst_process('CallProcess', 'process_inverse_2018', sFiles, [], ...
    'output', 1, ...
    'inverse', struct(...
         'InverseMethod', 'lcmv', ...
         'DataTypes', {{'MEG'}}));

% dSPM (noise-normalized)
sFiles = bst_process('CallProcess', 'process_inverse_2018', sFiles, [], ...
    'output', 1, ...
    'inverse', struct(...
         'InverseMethod', 'minnorm', ...
         'InverseMeasure', 'dspm2018'));
```

## Time-Frequency Analysis

### Morlet Wavelets

```matlab
% Compute time-frequency decomposition
sFiles = bst_process('CallProcess', 'process_timefreq', sFiles, [], ...
    'sensortypes', 'MEG', ...
    'edit', struct(...
         'Comment', 'Power', ...
         'TimeBands', [], ...
         'Freqs', [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80], ...
         'MorletFc', 1, ...
         'MorletFwhmTc', 3, ...
         'ClusterFuncTime', 'none', ...
         'Measure', 'power', ...
         'Output', 'all', ...
         'SaveKernel', 0), ...
    'normalize', 'relative');  % Baseline normalization
```

### Hilbert Transform

```matlab
% Hilbert transform for specific frequency band
sFiles = bst_process('CallProcess', 'process_hilbert', sFiles, [], ...
    'sensortypes', 'MEG', ...
    'freqbands', {'alpha', '8, 12', 'mean'}, ...
    'ispower', 1, ...
    'mirror', 0);
```

## Connectivity Analysis

### Coherence

```matlab
% Compute connectivity (coherence)
sFiles = bst_process('CallProcess', 'process_corr1n', sFiles, [], ...
    'timewindow', [0, 0.5], ...
    'scouts', {}, ...
    'scoutfunc', 1, ...  % Mean
    'scouttime', 1, ...  % Before
    'method', 'corr', ...  % Coherence
    'outputmode', 1);  % Save individual results
```

### Granger Causality

```matlab
% Granger causality
sFiles = bst_process('CallProcess', 'process_granger', sFiles, [], ...
    'timewindow', [0, 0.5], ...
    'scouts', {}, ...
    'scoutfunc', 1, ...
    'scouttime', 1, ...
    'freqbands', {'alpha', '8, 12', 'mean'}, ...
    'outputmode', 1);
```

## Group Analysis

### Average Across Subjects

```matlab
% Average sources across subjects
sFiles = bst_process('CallProcess', 'process_average', sFiles, [], ...
    'avgtype', 5, ...  % By subject name
    'avg_func', 1, ...  % Arithmetic average
    'weighted', 0, ...
    'keepevents', 0);
```

### Statistics

```matlab
% Parametric t-test
sFiles = bst_process('CallProcess', 'process_test_parametric2', sFilesA, sFilesB, ...
    'timewindow', [], ...
    'scoutsel', {}, ...
    'scoutfunc', 1, ...
    'isnorm', 0, ...
    'avgtime', 0, ...
    'avgrow', 0, ...
    'Comment', '', ...
    'test_type', 'ttest_paired', ...  % Paired t-test
    'tail', 'two');  % Two-tailed

% Cluster-based permutation test
sFiles = bst_process('CallProcess', 'process_test_permutation2', sFilesA, sFilesB, ...
    'timewindow', [], ...
    'scoutsel', {}, ...
    'scoutfunc', 1, ...
    'isnorm', 0, ...
    'avgtime', 0, ...
    'avgrow', 0, ...
    'Comment', '', ...
    'test_type', 'ttest_paired', ...
    'randomizations', 1000, ...
    'tail', 'two');
```

## Scripting Complete Pipeline

```matlab
%% Complete Brainstorm MEG Analysis Pipeline

% Start Brainstorm
if ~brainstorm('status')
    brainstorm nogui
end

% Create protocol
ProtocolName = 'MEG_Study';
gui_brainstorm('CreateProtocol', ProtocolName, 0, 0);

% Import subject anatomy
SubjectName = 'Subject01';
MriFile = '/data/subject01/anat/T1.nii';
bst_process('CallProcess', 'process_import_anatomy', [], [], ...
    'subjectname', SubjectName, ...
    'mrifile', {MriFile, 'Nifti1'}, ...
    'nvertices', 15000);

% Import MEG data
DataFile = '/data/subject01/meg/run01.fif';
sFiles = bst_process('CallProcess', 'process_import_data_raw', [], [], ...
    'subjectname', SubjectName, ...
    'datafile', {DataFile, 'FIF'}, ...
    'channelalign', 1);

% Preprocessing
sFiles = bst_process('CallProcess', 'process_bandpass', sFiles, [], ...
    'highpass', 1, 'lowpass', 40);
sFiles = bst_process('CallProcess', 'process_notch', sFiles, [], ...
    'freqlist', [50, 100, 150]);

% Detect and remove artifacts
sFiles = bst_process('CallProcess', 'process_evt_detect_ecg', sFiles, []);
sFiles = bst_process('CallProcess', 'process_ssp_ecg', sFiles, []);

% Import epochs
sFiles = bst_process('CallProcess', 'process_import_data_event', sFiles, [], ...
    'subjectname', SubjectName, ...
    'condition', 'Task', ...
    'eventname', 'stimulus', ...
    'epochtime', [-0.2, 0.5], ...
    'baseline', [-0.2, 0]);

% Compute head model
sFiles = bst_process('CallProcess', 'process_headmodel', sFiles, []);

% Compute noise covariance
sFiles = bst_process('CallProcess', 'process_noisecov', sFiles, [], ...
    'baseline', [-0.2, 0]);

% Source reconstruction (dSPM)
sFiles = bst_process('CallProcess', 'process_inverse_2018', sFiles, [], ...
    'inverse', struct('InverseMethod', 'minnorm', ...
                      'InverseMeasure', 'dspm2018'));

% Average
sFiles = bst_process('CallProcess', 'process_average', sFiles, [], ...
    'avgtype', 1, ...  % Everything
    'avg_func', 1);

disp('Analysis complete!');
```

## Visualization

```matlab
% View sensor data
view_timeseries(DataFile);

% View topography
view_topography(DataFile, 'EEG', 'FreqBands');

% View sources on cortex
hFig = view_surface_data([], SourceFile);

% Generate figures for publication
bst_report('Snapshot', hFig, SourceFile, 'Source: dSPM', [200,200,600,400]);
```

## Integration with Claude Code

When helping users with Brainstorm:

1. **Check Installation:**
   ```matlab
   brainstorm info
   ```

2. **Common Issues:**
   - Database location not set
   - Memory errors with large datasets
   - Co-registration failures
   - Anatomy processing errors

3. **Best Practices:**
   - Use scripting for reproducibility
   - Save processing pipelines
   - Check co-registration visually
   - Validate source reconstruction
   - Use reports for QC
   - Keep database organized

4. **Performance:**
   - Use downsampling for large files
   - Process in batches
   - Enable parallel processing
   - Use appropriate source space resolution

## Troubleshooting

**Problem:** "Database not found"
**Solution:** Set database location in Brainstorm preferences

**Problem:** Out of memory
**Solution:** Downsample data, reduce source space vertices, process fewer files

**Problem:** Co-registration poor
**Solution:** Check fiducial positions, use head points, manual adjustment

**Problem:** No sources computed
**Solution:** Verify head model, check noise covariance, validate channel locations

## Resources

- Website: https://neuroimage.usc.edu/brainstorm/
- Tutorials: https://neuroimage.usc.edu/brainstorm/Tutorials
- Forum: https://neuroimage.usc.edu/forums/
- YouTube: Brainstorm MEG/EEG Tutorial Series
- Download: https://neuroimage.usc.edu/brainstorm/Download

## Citation

```bibtex
@article{tadel2011brainstorm,
  title={Brainstorm: a user-friendly application for MEG/EEG analysis},
  author={Tadel, Fran{\c{c}}ois and Baillet, Sylvain and Mosher, John C and others},
  journal={Computational intelligence and neuroscience},
  volume={2011},
  year={2011},
  publisher={Hindawi}
}
```

## Related Tools

- **FieldTrip:** Alternative MATLAB toolbox
- **MNE-Python:** Python alternative
- **EEGLAB:** EEG-focused toolbox
- **SPM:** M/EEG module
- **FreeSurfer:** Anatomy processing
