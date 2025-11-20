# PhysIO: SPM Toolbox for Physiological Noise Correction

## Overview

**PhysIO** (Physiological Noise Modeling Toolbox) is a comprehensive MATLAB/SPM toolbox for model-based physiological noise correction in fMRI, developed as part of the TAPAS (Translational Algorithms for Psychiatry-Advancing Science) software suite at ETH Zurich. PhysIO creates nuisance regressors from cardiac and respiratory recordings using the RETROICOR (RETROspective Image-based CORrection) method and variants, handles multiple physiological recording formats, provides extensive quality control, and integrates seamlessly with SPM for statistical analysis.

### Key Features

- **RETROICOR Implementation**: Cardiac and respiratory phase-based noise correction
- **Multiple Recording Formats**: Siemens, Philips, GE, and custom physiological logs
- **Automatic Peak Detection**: Cardiac R-peaks and respiratory cycles
- **Manual Correction GUI**: Interactive peak editing and validation
- **RVT Regressors**: Respiratory Volume per Time modeling
- **HRV Regressors**: Heart Rate Variability correction
- **Slice-Timing Integration**: Slice-specific physiological regressors
- **SPM Integration**: Direct incorporation into first-level GLM analysis
- **Comprehensive QC**: Detailed quality control plots and diagnostics
- **Batch Processing**: SPM batch system compatibility
- **Multi-Modal Support**: Standard and simultaneous multi-slice (SMS) acquisition

### Scientific Foundation

RETROICOR corrects for physiological noise by:

1. **Detecting cardiac and respiratory cycles** from physiological recordings
2. **Computing phase information** for each TR
3. **Creating Fourier expansion regressors** based on cardiac/respiratory phase
4. **Modeling systematic BOLD fluctuations** time-locked to physiological cycles
5. **Regressing out nuisance variance** in GLM analysis

This model-based approach is highly effective when physiological recordings are available and provides more targeted correction than data-driven methods alone.

### Primary Use Cases

1. **Task fMRI**: Maximum sensitivity for subtle activation
2. **High-field imaging**: Enhanced artifact correction at 7T+
3. **Precision fMRI**: Layer-specific and high-resolution studies
4. **Clinical fMRI**: Robust correction with external monitoring
5. **Pharmacological fMRI**: Detect subtle drug-induced BOLD changes
6. **Multi-site studies**: Standardized physiological correction protocols

---

## Installation

### Download TAPAS Toolbox

```matlab
% Method 1: Download from website
% Visit: https://www.tnu.ethz.ch/en/software/tapas
% Download latest TAPAS release (includes PhysIO)

% Method 2: Clone from GitHub
% In terminal:
% git clone https://github.com/translationalneuromodeling/tapas.git

% Add to MATLAB path
tapas_path = '/path/to/tapas';
addpath(genpath(fullfile(tapas_path, 'PhysIO')));

% Verify installation
which tapas_physio_main_create_regressors

% Check version
tapas_physio_version
```

### SPM Integration

```matlab
% PhysIO requires SPM12
% Download SPM12 from: https://www.fil.ion.ucl.ac.uk/spm/

% Add SPM to path
spm_path = '/path/to/spm12';
addpath(spm_path);

% Initialize SPM
spm('defaults', 'fmri');
spm_jobman('initcfg');

% Verify PhysIO is available
tapas_physio_check_installation
```

### Test Installation

```matlab
% Run example script
cd(fullfile(tapas_path, 'PhysIO', 'examples'));
tapas_physio_example_main

% This should:
% 1. Load example physiological data
% 2. Detect peaks
% 3. Create RETROICOR regressors
% 4. Generate QC plots
% 5. Display summary
```

---

## Physiological Recording Formats

### Siemens DICOM Physiological Logs

```matlab
% Siemens scanners save physiological data in DICOM files

% Setup for Siemens data
physio = tapas_physio_new();

% Specify log files
physio.log_files.vendor = 'Siemens';
physio.log_files.cardiac = 'sub-01_cardiac.log';  % DICOM physiological log
physio.log_files.respiration = 'sub-01_resp.log';  % Same file typically

% Siemens-specific settings
physio.log_files.relative_start_acquisition = 0;  % Start of acquisition in log
physio.log_files.align_scan = 'last';  % Alignment method

disp('Configured for Siemens physiological data');
```

### Philips SCANPHYSLOG Files

```matlab
% Philips uses SCANPHYSLOG.log format

physio = tapas_physio_new();

% Specify Philips files
physio.log_files.vendor = 'Philips';
physio.log_files.cardiac = 'SCANPHYSLOG.log';
physio.log_files.respiration = 'SCANPHYSLOG.log';  % Same file
physio.log_files.scan_timing = 'SCANPHYSLOG.log';

% Philips-specific settings
physio.log_files.relative_start_acquisition = 0;

disp('Configured for Philips physiological data');
```

### GE Physiological Data

```matlab
% GE format setup

physio = tapas_physio_new();

physio.log_files.vendor = 'GE';
physio.log_files.cardiac = 'CardiacLog.txt';
physio.log_files.respiration = 'RespLog.txt';
physio.log_files.scan_timing = 'ScanTiming.txt';

disp('Configured for GE physiological data');
```

### Custom Text Format

```matlab
% For custom physiological recording systems

physio = tapas_physio_new();

physio.log_files.vendor = 'Custom';

% Cardiac: timestamps of R-peaks (seconds)
% Format: One timestamp per line
physio.log_files.cardiac = 'cardiac_peaks.txt';

% Respiration: continuous recording (samples)
% Format: One sample per line
physio.log_files.respiration = 'respiration_trace.txt';

% Sampling rate (Hz)
physio.log_files.sampling_interval = 1/500;  % 500 Hz

disp('Configured for custom physiological data');
```

---

## Basic RETROICOR Workflow

### Configure Scan Timing

```matlab
% Essential: Match physiological logs to fMRI acquisition

physio = tapas_physio_new();

% Scan timing parameters
physio.scan_timing.sqpar.Nslices = 36;  % Number of slices
physio.scan_timing.sqpar.NslicesPerBeat = [];  % For cardiac triggering (usually empty)
physio.scan_timing.sqpar.TR = 2.0;  % Repetition time (seconds)
physio.scan_timing.sqpar.Ndummies = 5;  % Dummy scans
physio.scan_timing.sqpar.Nscans = 200;  % Total volumes (including dummies)
physio.scan_timing.sqpar.onset_slice = 18;  % Reference slice for timing

% Slice timing (acquisition order)
% Example: Interleaved ascending
slice_order = [1:2:36, 2:2:36];  % Odd slices first, then even
physio.scan_timing.sqpar.time_slice_to_slice = physio.scan_timing.sqpar.TR / physio.scan_timing.sqpar.Nslices;

% Synchronization
physio.scan_timing.sync.method = 'scan_timing_log';  % Or 'nominal', 'gradient_log'

disp('Scan timing configured');
```

### Run RETROICOR

```matlab
% Complete RETROICOR workflow

% 1. Load physiological data and create regressors
physio = tapas_physio_main_create_regressors(physio);

% This performs:
% - Read physiological log files
% - Detect cardiac R-peaks
% - Detect respiratory cycles
% - Synchronize with scan acquisition
% - Create RETROICOR regressors (Fourier expansion)
% - Generate quality control plots
% - Save regressors for SPM

% Outputs saved to physio.save_dir:
% - multiple_regressors.txt (for SPM)
% - physio.mat (complete physio structure)
% - QC plots (cardiac, respiratory, regressors)

disp('RETROICOR regressors created successfully');
```

### Load and Inspect Regressors

```matlab
% Load created regressors
regressors = load('multiple_regressors.txt');

[n_scans, n_regressors] = size(regressors);

fprintf('Regressors: %d scans x %d regressors\n', n_scans, n_regressors);

% Plot regressors
figure('Name', 'RETROICOR Regressors');
for i = 1:n_regressors
    subplot(n_regressors, 1, i);
    plot(regressors(:, i));
    ylabel(sprintf('R%d', i));
    if i == n_regressors
        xlabel('Scan number');
    end
end

% Regressors typically include:
% - Cardiac phase (sine/cosine, order 1-3)
% - Respiratory phase (sine/cosine, order 1-4)
% - Interaction terms
```

---

## RETROICOR Model Configuration

### Cardiac Phase Regressors

```matlab
% Configure cardiac phase modeling

physio.model.retroicor.yes = true;

% Cardiac Fourier expansion order (typically 3)
physio.model.retroicor.order.c = 3;

% This creates 2*order = 6 cardiac regressors:
% sin(1*theta_c), cos(1*theta_c),
% sin(2*theta_c), cos(2*theta_c),
% sin(3*theta_c), cos(3*theta_c)
% where theta_c is cardiac phase (0 to 2*pi)

disp('Cardiac RETROICOR order: 3 (6 regressors)');
```

### Respiratory Phase Regressors

```matlab
% Configure respiratory phase modeling

% Respiratory Fourier expansion order (typically 4)
physio.model.retroicor.order.r = 4;

% This creates 2*order = 8 respiratory regressors:
% sin(1*theta_r), cos(1*theta_r),
% sin(2*theta_r), cos(2*theta_r),
% sin(3*theta_r), cos(3*theta_r),
% sin(4*theta_r), cos(4*theta_r)
% where theta_r is respiratory phase

disp('Respiratory RETROICOR order: 4 (8 regressors)');
```

### Interaction Terms

```matlab
% Cardiac-respiratory interaction

% Interaction order
physio.model.retroicor.order.cr = 1;  % Typically 1

% This creates 4 interaction regressors:
% sin(theta_c)*sin(theta_r)
% sin(theta_c)*cos(theta_r)
% cos(theta_c)*sin(theta_r)
% cos(theta_c)*cos(theta_r)

% Total RETROICOR regressors = 6 (cardiac) + 8 (resp) + 4 (interaction) = 18

disp('Cardiac-respiratory interaction included');
```

### Optimize Model Order

```matlab
% Test different model orders

orders_to_test = [
    3, 4, 1;  % Default
    2, 3, 1;  % Lower order
    4, 5, 1;  % Higher order
];

for i = 1:size(orders_to_test, 1)
    physio_test = physio;
    physio_test.model.retroicor.order.c = orders_to_test(i, 1);
    physio_test.model.retroicor.order.r = orders_to_test(i, 2);
    physio_test.model.retroicor.order.cr = orders_to_test(i, 3);

    physio_test.save_dir = sprintf('physio_order_c%d_r%d_cr%d', ...
        orders_to_test(i, 1), orders_to_test(i, 2), orders_to_test(i, 3));

    physio_test = tapas_physio_main_create_regressors(physio_test);

    fprintf('Order c=%d, r=%d, cr=%d: %d total regressors\n', ...
        orders_to_test(i, 1), orders_to_test(i, 2), orders_to_test(i, 3), ...
        size(physio_test.model.R, 2));
end
```

---

## Additional Physiological Regressors

### Respiratory Volume per Time (RVT)

```matlab
% RVT models BOLD changes related to respiratory volume changes

physio.model.rvt.yes = true;

% RVT delay (seconds) - typical range 0-10s
physio.model.rvt.delays = 0;  % Can be scalar or vector [0, 5, 10]

% Create shifted RVT regressors
if isvector(physio.model.rvt.delays)
    fprintf('RVT regressors: %d (delays: %s)\n', ...
        length(physio.model.rvt.delays), mat2str(physio.model.rvt.delays));
end

% RVT can significantly improve variance explained beyond RETROICOR
```

### Heart Rate Variability (HRV)

```matlab
% HRV models BOLD changes related to heart rate fluctuations

physio.model.hrv.yes = true;

% HRV delay (seconds)
physio.model.hrv.delays = 0;  % Or multiple delays: [0, 5]

disp('HRV regressors enabled');

% HRV is particularly useful for:
% - Resting-state fMRI
% - Studies with autonomic modulation
% - High-field imaging
```

### Motion Parameters Integration

```matlab
% Include motion parameters as additional regressors

physio.model.movement.yes = true;

% Specify motion parameter file (from realignment)
physio.model.movement.file_realignment_parameters = 'rp_sub-01_bold.txt';

% Motion censoring (optional)
physio.model.movement.censoring_threshold = 0.5;  % mm
physio.model.movement.censoring_method = 'FD';  % Framewise displacement

disp('Motion parameters will be included in regressor file');
```

---

## Peak Detection and Quality Control

### Automatic Peak Detection

```matlab
% Configure automatic cardiac peak detection

physio.preproc.cardiac.modality = 'ECG';  % Or 'PPU' for pulse oximetry
physio.preproc.cardiac.initial_cpulse_select.method = 'auto_matched';

% Peak detection parameters
physio.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 120;
physio.preproc.cardiac.initial_cpulse_select.min = 0.4;  % Minimum interval (s)

% Respiratory peak detection
physio.preproc.respiratory.filter.passband = [0.01, 2];  % Hz

disp('Automatic peak detection configured');
```

### Manual Peak Correction GUI

```matlab
% Launch interactive peak correction GUI

% After initial automatic detection
physio = tapas_physio_main_create_regressors(physio);

% If peaks need manual correction:
% 1. In QC plots, identify incorrect peaks
% 2. Launch GUI for manual editing

% Edit cardiac peaks
physio = tapas_physio_cardiac_peak_editor(physio);

% Edit respiratory peaks
physio = tapas_physio_respiratory_peak_editor(physio);

% Re-create regressors with corrected peaks
physio = tapas_physio_main_create_regressors(physio);

disp('Manual peak correction completed');
```

### Quality Control Plots

```matlab
% PhysIO automatically generates comprehensive QC plots

% 1. Cardiac trace with detected peaks
% 2. Respiratory trace with detected peaks
% 3. Slice-by-slice timing diagram
% 4. Histogram of cardiac/respiratory intervals
% 5. Created regressors timecourse
% 6. Power spectra of regressors

% Access QC figure handles
fig_handles = physio.fig_handles;

% Save QC figures
for i = 1:length(fig_handles)
    if ishandle(fig_handles(i))
        saveas(fig_handles(i), sprintf('QC_figure_%d.png', i));
    end
end

disp('QC plots saved');
```

---

## SPM Integration

### Add Regressors to SPM GLM

```matlab
% First-level fMRI analysis with PhysIO regressors

% Define SPM model
matlabbatch{1}.spm.stats.fmri_spec.dir = {'./GLM'};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'scans';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2.0;

% Functional data
matlabbatch{1}.spm.stats.fmri_spec.sess.scans = cellstr(spm_select('FPList', ...
    './func', '^sub-01.*\.nii$'));

% Conditions (task design)
matlabbatch{1}.spm.stats.fmri_spec.sess.cond.name = 'TaskCondition';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond.onset = [10 30 50 70 90]';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond.duration = 5;

% Add PhysIO regressors
matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {
    './physio_output/multiple_regressors.txt'
};

% Estimate model
matlabbatch{2}.spm.stats.fmri_est.spmmat = {'./GLM/SPM.mat'};

% Run batch
spm_jobman('run', matlabbatch);

disp('SPM GLM with PhysIO regressors estimated');
```

### Compare With and Without Physiological Correction

```matlab
% Run two models for comparison

% Model 1: Without physio correction
matlabbatch_nophysio = matlabbatch;
matlabbatch_nophysio{1}.spm.stats.fmri_spec.dir = {'./GLM_no_physio'};
matlabbatch_nophysio{1}.spm.stats.fmri_spec.sess.multi_reg = {''};  % No regressors

% Model 2: With physio correction
matlabbatch_physio = matlabbatch;
matlabbatch_physio{1}.spm.stats.fmri_spec.dir = {'./GLM_with_physio'};
matlabbatch_physio{1}.spm.stats.fmri_spec.sess.multi_reg = {
    './physio_output/multiple_regressors.txt'
};

% Run both models
spm_jobman('run', matlabbatch_nophysio);
spm_jobman('run', matlabbatch_physio);

% Compare residual variance
ResMS_nophysio = spm_vol('./GLM_no_physio/ResMS.nii');
ResMS_physio = spm_vol('./GLM_with_physio/ResMS.nii');

res_nophysio = spm_read_vols(ResMS_nophysio);
res_physio = spm_read_vols(ResMS_physio);

% Calculate variance reduction
variance_reduction = (res_nophysio - res_physio) ./ res_nophysio * 100;

fprintf('Mean variance reduction: %.2f%%\n', nanmean(variance_reduction(:)));
```

---

## Slice-Timing Considerations

### Slice-Specific Regressors

```matlab
% For optimal correction, create slice-specific regressors

% PhysIO automatically handles slice timing
% Ensure scan_timing.sqpar is correctly specified

% Slice timing already configured in scan_timing.sqpar
physio.scan_timing.sqpar.Nslices = 36;
physio.scan_timing.sqpar.TR = 2.0;
physio.scan_timing.sqpar.onset_slice = 18;  % Reference slice

% Slice acquisition order
slice_order = [1:2:36, 2:2:36];  % Interleaved
physio.scan_timing.sqpar.time_slice_to_slice = ...
    physio.scan_timing.sqpar.TR / physio.scan_timing.sqpar.Nslices;

% PhysIO creates regressors matched to each slice's acquisition time
disp('Slice-specific physiological regressors enabled');
```

### Simultaneous Multi-Slice (SMS) Support

```matlab
% For multiband/simultaneous multi-slice acquisition

% Specify SMS factor
physio.scan_timing.sqpar.Nslices = 72;  % Total slices
physio.scan_timing.sqpar.TR = 1.0;  % Short TR
physio.scan_timing.sqpar.multiband_factor = 6;  % SMS factor

% Effective slices per TR
effective_slices = physio.scan_timing.sqpar.Nslices / ...
    physio.scan_timing.sqpar.multiband_factor;

fprintf('SMS acquisition: %d slices, MB factor %d\n', ...
    physio.scan_timing.sqpar.Nslices, ...
    physio.scan_timing.sqpar.multiband_factor);

% PhysIO handles SMS timing automatically
```

---

## Batch Processing

### SPM Batch System Integration

```matlab
% PhysIO can be integrated into SPM batch system

% Create PhysIO batch job
matlabbatch{1}.spm.tools.physio.save_dir = {'./physio_output'};
matlabbatch{1}.spm.tools.physio.log_files.vendor = 'Siemens';
matlabbatch{1}.spm.tools.physio.log_files.cardiac = {'cardiac.log'};
matlabbatch{1}.spm.tools.physio.log_files.respiration = {'resp.log'};

% Scan timing
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nslices = 36;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.TR = 2.0;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nscans = 200;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Ndummies = 5;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.onset_slice = 18;

% Model
matlabbatch{1}.spm.tools.physio.model.retroicor.yes = true;
matlabbatch{1}.spm.tools.physio.model.retroicor.order.c = 3;
matlabbatch{1}.spm.tools.physio.model.retroicor.order.r = 4;
matlabbatch{1}.spm.tools.physio.model.retroicor.order.cr = 1;

% Run batch
spm_jobman('run', matlabbatch);
```

### Process Multiple Subjects

```matlab
% Loop through subjects

subjects = {'01', '02', '03', '04', '05'};

for s = 1:length(subjects)
    sub_id = subjects{s};

    fprintf('Processing sub-%s...\n', sub_id);

    % Initialize physio structure
    physio = tapas_physio_new();

    % Subject-specific files
    physio.save_dir = sprintf('./derivatives/physio/sub-%s', sub_id);
    physio.log_files.vendor = 'Siemens';
    physio.log_files.cardiac = sprintf('./sub-%s/func/cardiac.log', sub_id);
    physio.log_files.respiration = sprintf('./sub-%s/func/resp.log', sub_id);

    % Scan timing (same for all subjects)
    physio.scan_timing.sqpar.Nslices = 36;
    physio.scan_timing.sqpar.TR = 2.0;
    physio.scan_timing.sqpar.Nscans = 200;
    physio.scan_timing.sqpar.Ndummies = 5;
    physio.scan_timing.sqpar.onset_slice = 18;

    % Model
    physio.model.retroicor.yes = true;
    physio.model.retroicor.order.c = 3;
    physio.model.retroicor.order.r = 4;
    physio.model.retroicor.order.cr = 1;
    physio.model.rvt.yes = true;
    physio.model.hrv.yes = true;

    % Create regressors
    try
        physio = tapas_physio_main_create_regressors(physio);
        fprintf('  ✓ sub-%s completed\n', sub_id);
    catch ME
        fprintf('  ✗ sub-%s failed: %s\n', sub_id, ME.message);
    end
end

fprintf('Batch processing complete\n');
```

---

## Advanced Options and Best Practices

### Optimize for Different Scenarios

```matlab
% Resting-state fMRI
physio_rest = physio;
physio_rest.model.retroicor.order.c = 3;  % Standard
physio_rest.model.retroicor.order.r = 4;
physio_rest.model.rvt.yes = true;
physio_rest.model.hrv.yes = true;  % Important for rest

% Task fMRI (high sensitivity required)
physio_task = physio;
physio_task.model.retroicor.order.c = 4;  % Higher order
physio_task.model.retroicor.order.r = 5;
physio_task.model.rvt.yes = true;
physio_task.model.hrv.yes = false;

% High-field fMRI (7T+)
physio_highfield = physio;
physio_highfield.model.retroicor.order.c = 4;
physio_highfield.model.retroicor.order.r = 5;
physio_highfield.model.rvt.yes = true;
physio_highfield.model.hrv.yes = true;
physio_highfield.model.movement.yes = true;  % Include motion

disp('Scenario-specific configurations prepared');
```

### Handling Missing Data

```matlab
% If cardiac or respiratory data is missing

% Use only available modality
physio_cardiac_only = physio;
physio_cardiac_only.log_files.respiration = '';  % No respiratory data
physio_cardiac_only.model.retroicor.order.r = 0;  % Disable resp RETROICOR
physio_cardiac_only.model.rvt.yes = false;

% Or use only respiratory
physio_resp_only = physio;
physio_resp_only.log_files.cardiac = '';
physio_resp_only.model.retroicor.order.c = 0;
physio_resp_only.model.hrv.yes = false;

disp('Configured for partial physiological data');
```

---

## Troubleshooting

### Common Issues

```matlab
% Issue: Peak detection failures

% Check physiological signal quality
physio_check = physio;
physio_check.verbose.level = 2;  % Increase verbosity
physio_check.verbose.fig_output_file = 'debug_plots.ps';

% Adjust peak detection thresholds
physio_check.preproc.cardiac.initial_cpulse_select.min = 0.3;  % Lower threshold
physio_check.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 150;  % Increase max

% If still fails, use manual correction GUI
tapas_physio_cardiac_peak_editor(physio_check);
```

### Validation Checks

```matlab
% Validate PhysIO outputs

function validate_physio_output(physio)
    % Check regressors
    if isempty(physio.model.R)
        warning('No regressors created!');
        return;
    end

    fprintf('Validation checks:\n');
    fprintf('  Regressors: %d x %d\n', size(physio.model.R));
    fprintf('  Expected scans: %d\n', physio.scan_timing.sqpar.Nscans);

    % Check regressor variance
    regressor_std = std(physio.model.R);
    if any(regressor_std < 1e-10)
        warning('Some regressors have zero variance!');
    end

    % Check for NaN/Inf
    if any(isnan(physio.model.R(:))) || any(isinf(physio.model.R(:)))
        error('Regressors contain NaN or Inf values!');
    end

    fprintf('  ✓ Validation passed\n');
end

% Run validation
validate_physio_output(physio);
```

---

## Comparison and When to Use PhysIO

### PhysIO vs Data-Driven Methods

```matlab
% PhysIO (model-based)
% Advantages:
% - Highly effective when recordings available
% - Targeted correction of known noise sources
% - Less risk of removing neural signals
% - Interpretable (cardiac, respiratory phases)

% Disadvantages:
% - Requires physiological recordings
% - Need good signal quality
% - Vendor-specific formats

% tedana (multi-echo ICA)
% - No recordings needed
% - Automatic, data-driven
% - Requires multi-echo acquisition

% RapidTide (sLFO detection)
% - No recordings needed
% - Data-driven from BOLD
% - Detects blood flow delays

% Best practice: Combine approaches when possible
% Example: PhysIO (cardiac/resp) + motion confounds + tedana
```

### Recommended Workflow

```matlab
% 1. Preprocessing (fMRIPrep, SPM realignment)
% 2. PhysIO regressor creation
% 3. SPM first-level with PhysIO regressors
% 4. Validate improvement via residual variance

fprintf('Recommended PhysIO workflow:\n');
fprintf('1. Quality preprocessing (motion correction, distortion)\n');
fprintf('2. Run PhysIO to create physiological regressors\n');
fprintf('3. Visual QC of peak detection\n');
fprintf('4. Manual correction if needed\n');
fprintf('5. Incorporate regressors in GLM\n');
fprintf('6. Compare with/without correction\n');
```

---

## References

### Key Publications

1. Glover, G. H., et al. (2000). "Image-based method for retrospective correction of physiological motion effects in fMRI: RETROICOR." *Magnetic Resonance in Medicine*, 44(1), 162-167.

2. Kasper, L., et al. (2017). "The PhysIO Toolbox for modeling physiological noise in fMRI data." *Journal of Neuroscience Methods*, 276, 56-72.

3. Harvey, A. K., et al. (2008). "Brainstem functional magnetic resonance imaging: Disentangling signal from physiological noise." *Journal of Magnetic Resonance Imaging*, 28(6), 1337-1344.

### Documentation and Resources

- **TAPAS Website**: https://www.tnu.ethz.ch/en/software/tapas
- **GitHub**: https://github.com/translationalneuromodeling/tapas
- **Documentation**: https://www.tnu.ethz.ch/en/software/tapas/documentations/physio-toolbox
- **Tutorials**: Included in TAPAS/PhysIO/examples/
- **Support**: https://github.com/translationalneuromodeling/tapas/issues

### Related Tools

- **tedana**: Multi-echo denoising (data-driven)
- **RapidTide**: Systemic oscillation detection (data-driven)
- **SPM**: Statistical analysis platform
- **fMRIPrep**: Comprehensive preprocessing
- **CONN**: Functional connectivity toolbox

---

## See Also

- **spm.md**: SPM statistical analysis
- **tedana.md**: Multi-echo fMRI denoising
- **rapidtide.md**: Physiological noise detection
- **fmriprep.md**: fMRI preprocessing pipeline
- **conn.md**: Functional connectivity analysis

## Citation

```bibtex
@article{kasper2017physio,
  title={Building physiological noise models for fMRI using TAPAS PhysIO},
  author={Kasper, Lars and Bollmann, Steffen and Diaconescu, Andreea and others},
  journal={Journal of Neuroscience Methods},
  volume={276},
  pages={56--72},
  year={2017},
  doi={10.1016/j.jneumeth.2016.10.019}
}
```
