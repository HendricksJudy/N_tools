# LIMO EEG (LInear MOdeling of EEG/MEG)

## Overview

LIMO EEG is a MATLAB toolbox for hierarchical linear modeling of electrophysiological data (EEG/MEG). Developed by Cyril Pernet and colleagues, LIMO implements advanced statistical methods specifically designed for the massive univariate testing required by EEG/MEG data's multi-dimensional nature (electrodes × time × frequency). It supports single-trial analysis, complex experimental designs, and proper multiple comparison correction while integrating seamlessly with EEGLAB for preprocessing and data management.

**Website:** https://github.com/LIMO-EEG-Toolbox/limo_tools
**GitHub:** https://github.com/LIMO-EEG-Toolbox/limo_tools
**Platform:** MATLAB (EEGLAB plugin)
**Language:** MATLAB
**License:** MIT License

## Key Features

- Hierarchical linear models for EEG/MEG data
- Single-trial analysis (avoids trial averaging artifacts)
- Massive univariate testing (electrode × time × frequency)
- Event-related potential (ERP) analysis
- Time-frequency decomposition (ERSP, ITC)
- Cluster-based permutation testing
- Threshold-Free Cluster Enhancement (TFCE) for EEG
- Multiple comparison correction (FWE, FDR, cluster)
- Integration with EEGLAB preprocessing
- Repeated-measures and factorial designs
- Covariate effects (continuous and categorical)
- Group-level random-effects inference
- Robust regression methods
- Support for source-space analysis

## Installation

### Requirements

```matlab
% MATLAB R2016a or later
% EEGLAB 14.0 or later
% Statistics and Machine Learning Toolbox
% Signal Processing Toolbox (recommended)
```

### Install as EEGLAB Plugin

```bash
# Method 1: Via EEGLAB plugin manager
# Launch EEGLAB
# File → Manage EEGLAB extensions
# Search for "LIMO EEG"
# Install

# Method 2: Manual installation
git clone https://github.com/LIMO-EEG-Toolbox/limo_tools.git
```

```matlab
% Add LIMO to MATLAB path
addpath('/path/to/limo_tools');

% Start EEGLAB
eeglab;

% Verify LIMO installation
% EEGLAB → Tools → LIMO EEG
% Should appear in menu
```

## Data Preparation

### Load EEGLAB Dataset

```matlab
% Start EEGLAB
eeglab;

% Load preprocessed EEG data
EEG = pop_loadset('filename', 'sub-01_preprocessed.set', 'filepath', '/data/eeg/');

% Verify epoching
fprintf('Number of epochs: %d\n', EEG.trials);
fprintf('Number of channels: %d\n', EEG.nbchan);
fprintf('Time points per epoch: %d\n', EEG.pnts);
fprintf('Sampling rate: %.1f Hz\n', EEG.srate);

% Check event structure
fprintf('Event types: ');
unique_events = unique({EEG.event.type});
disp(unique_events);
```

### Organize Single-Trial Data

```matlab
% LIMO uses single-trial data (not averaged ERPs)
% Each trial treated as separate observation

% Extract trial information
n_trials = EEG.trials;
trial_indices = 1:n_trials;

% Extract condition labels from events
conditions = zeros(n_trials, 1);
for t = 1:n_trials
    event_idx = find([EEG.event.epoch] == t, 1);
    if strcmp(EEG.event(event_idx).type, 'standard')
        conditions(t) = 1;
    elseif strcmp(EEG.event(event_idx).type, 'deviant')
        conditions(t) = 2;
    end
end

fprintf('Standard trials: %d\n', sum(conditions == 1));
fprintf('Deviant trials: %d\n', sum(conditions == 2));
```

### Create Design Matrix

```matlab
% Design matrix for LIMO
% Rows = trials
% Columns = predictors

% Simple design: condition (categorical)
X = zeros(n_trials, 2);  % Two conditions
X(conditions == 1, 1) = 1;  % Standard
X(conditions == 2, 2) = 1;  % Deviant

% Add continuous covariate (e.g., reaction time)
reaction_times = zeros(n_trials, 1);
for t = 1:n_trials
    event_idx = find([EEG.event.epoch] == t, 1);
    if isfield(EEG.event, 'rt')
        reaction_times(t) = EEG.event(event_idx).rt;
    end
end

% Z-score continuous predictors
reaction_times_z = zscore(reaction_times);

% Extended design matrix
X_extended = [X, reaction_times_z];

fprintf('Design matrix: %d trials × %d predictors\n', size(X_extended));
```

## First-Level Analysis

### Single-Subject ERP Analysis

```matlab
% Launch LIMO from EEGLAB
% EEGLAB → Tools → LIMO EEG → LIMO EEG

% GUI Steps:
% 1. Select "1st level analysis"
% 2. Load EEGLAB dataset
% 3. Specify design matrix
% 4. Choose analysis type: "Time" for ERP
% 5. Select output directory

% Or via script:
LIMO = struct();
LIMO.dir = '/results/limo/sub-01/';
LIMO.data.data = EEG.data;  % Channels × Time × Trials
LIMO.data.chanlocs = EEG.chanlocs;
LIMO.data.sampling_rate = EEG.srate;
LIMO.data.start = EEG.xmin * 1000;  % Convert to ms
LIMO.data.end = EEG.xmax * 1000;

% Design specification
LIMO.design.X = X;
LIMO.design.name = {'Standard', 'Deviant'};
LIMO.design.method = 'OLS';  % Ordinary least squares

% Run first-level analysis
limo_eeg(4);  % Main LIMO function
```

### Time-Frequency Analysis

```matlab
% Analyze time-frequency decomposition (ERSP)

% In EEGLAB, compute ERSP first
% EEGLAB → Tools → Time-frequency → Channel spectrogram

% Or via script:
[ersp, itc, powbase, times, freqs] = newtimef(EEG.data, ...
    EEG.pnts, [EEG.xmin EEG.xmax]*1000, EEG.srate, [3 0.5], ...
    'freqs', [4 40], 'nfreqs', 20, 'baseline', [-200 0], ...
    'plotitc', 'off', 'plotersp', 'off');

% LIMO on time-frequency data
LIMO.data.data = ersp;  % Channels × Freqs × Time × Trials
LIMO.data.tf_times = times;
LIMO.data.tf_freqs = freqs;

% Specify as time-frequency analysis
LIMO.design.type = 'Time-Frequency';

% Run LIMO
limo_eeg(4);
```

### Robust Regression

```matlab
% Use robust regression to handle outliers
% Iteratively reweighted least squares (IRLS)

% Set method to robust
LIMO.design.method = 'WLS';  % Weighted least squares

% Run with robust estimation
limo_eeg(4);

% Results less sensitive to trial outliers
```

## Group-Level Analysis

### One-Sample T-Test

```matlab
% Test if ERP component differs from zero across subjects
% Requires first-level analyses for all subjects

% Specify subjects
n_subjects = 20;
subject_dirs = cell(n_subjects, 1);
for s = 1:n_subjects
    subject_dirs{s} = sprintf('/results/limo/sub-%02d/', s);
end

% Setup group analysis
% LIMO → 2nd level analysis → One-sample t-test

% Load Betas from first-level
Betas = cell(n_subjects, 1);
for s = 1:n_subjects
    beta_file = fullfile(subject_dirs{s}, 'Betas.mat');
    Betas{s} = load(beta_file);
end

% Run group-level test
% Tests each electrode × time point
% Output: T-statistics, p-values
```

### Two-Sample Comparison

```matlab
% Compare two groups (e.g., patients vs. controls)

% Group 1: Controls
controls = {'/results/limo/ctrl-01/', '/results/limo/ctrl-02/', ...};

% Group 2: Patients
patients = {'/results/limo/pat-01/', '/results/limo/pat-02/', ...};

% LIMO GUI:
% 2nd level → Two-sample t-test
% Select control directories
% Select patient directories
% Choose contrast (typically first condition or averaged beta)

% Run analysis
% Computes independent samples t-test at each electrode × time
```

### Repeated-Measures ANOVA

```matlab
% Within-subjects factors (e.g., multiple conditions per subject)

% Load single-subject contrasts
% E.g., Condition1 - Baseline, Condition2 - Baseline

condition1_files = cell(n_subjects, 1);
condition2_files = cell(n_subjects, 1);

for s = 1:n_subjects
    condition1_files{s} = sprintf('/results/limo/sub-%02d/con_1.mat', s);
    condition2_files{s} = sprintf('/results/limo/sub-%02d/con_2.mat', s);
end

% LIMO GUI:
% 2nd level → Repeated-measures ANOVA
% Factor levels: 2 (two conditions)
% Specify files for each level

% Output: F-statistics for condition effect
```

## Statistical Inference

### Cluster-Based Permutation Testing

```matlab
% Cluster-based inference to control FWER
% Groups adjacent significant electrodes/timepoints into clusters

% After running 2nd-level analysis
% LIMO → Results → Cluster correction

% Parameters
cluster_p = 0.05;  % Cluster-forming threshold
n_permutations = 1000;

% Run cluster test
% Identifies spatiotemporal clusters
% Tests cluster mass against null distribution

% Output:
% - Cluster map
% - Cluster p-values
% - Significant clusters highlighted
```

### Threshold-Free Cluster Enhancement

```matlab
% TFCE: Cluster-like inference without arbitrary threshold

% After 2nd-level analysis
% LIMO → Results → TFCE

% TFCE parameters
tfce_h = 2;  % Height exponent
tfce_e = 0.5;  % Extent exponent (spatial connectivity for EEG)

% Run TFCE
% Enhances signals based on local spatial/temporal extent
% More sensitive than traditional clustering

% Output: TFCE-corrected p-values
```

### FDR Correction

```matlab
% False discovery rate control
% Less conservative than FWER

% After analysis
% LIMO → Results → FDR

% Benjamini-Hochberg procedure
% Controls proportion of false discoveries

% Useful for exploratory analyses
% More power than cluster methods
```

## Advanced Models

### Regression with Continuous Predictors

```matlab
% Model ERP amplitude as function of continuous variable
% E.g., age, behavioral performance, symptom severity

% First level: Include covariate in design
age = [25, 30, 28, ...];  % Age for each trial/subject
age_z = zscore(age);

LIMO.design.X = [ones(length(age), 1), age_z];  % Intercept + Age
LIMO.design.name = {'Intercept', 'Age'};

% Run first-level
limo_eeg(4);

% Second level: Test age effect across subjects
% Each subject's age regression slope tested against zero
```

### Factorial Designs

```matlab
% 2×2 factorial: Group (Patient, Control) × Condition (A, B)

% First level: Separate models for each condition
% sub-01, Condition A: LIMO analysis
% sub-01, Condition B: LIMO analysis

% Second level: Factorial ANOVA
% Factors: Group (2 levels) × Condition (2 levels)

% LIMO GUI:
% 2nd level → Factorial ANOVA
% Between-subjects factor: Group (Patient, Control)
% Within-subjects factor: Condition (A, B)

% Tests:
% - Main effect of Group
% - Main effect of Condition
% - Group × Condition interaction
```

### Multivariate Analysis

```matlab
% Test multiple conditions simultaneously
% Multivariate ANOVA (MANOVA)

% Load beta weights for multiple conditions
% Test if conditions differ using multivariate statistics

% LIMO supports multivariate extensions
% Useful for hypothesis tests involving multiple contrasts
```

## Visualization

### Topographic Maps

```matlab
% Plot ERP topography at specific time point

% Load group results
load('/results/limo/group/one_sample_ttest_parameter_1.mat');

% Select time point (e.g., 300 ms for P300)
time_ms = 300;
time_idx = find(abs(LIMO.data.timevect - time_ms) == min(abs(LIMO.data.timevect - time_ms)));

% Extract T-statistics at this time
topo_data = squeeze(one_sample(:, time_idx));

% Plot topography
figure;
topoplot(topo_data, EEG.chanlocs, 'maplimits', [-5 5], 'electrodes', 'on');
colorbar;
title(sprintf('T-statistics at %d ms', time_ms));
```

### Time-Series Plots

```matlab
% Plot ERP time course at selected electrodes

electrodes = {'Cz', 'Pz', 'Oz'};
figure;

for e = 1:length(electrodes)
    subplot(length(electrodes), 1, e);

    % Find electrode index
    elec_idx = find(strcmp({EEG.chanlocs.labels}, electrodes{e}));

    % Plot T-statistics over time
    plot(LIMO.data.timevect, squeeze(one_sample(elec_idx, :)), 'LineWidth', 2);
    hold on;

    % Add significance line
    plot(LIMO.data.timevect, ones(size(LIMO.data.timevect)) * 3.0, 'r--');
    plot(LIMO.data.timevect, ones(size(LIMO.data.timevect)) * -3.0, 'r--');

    xlabel('Time (ms)');
    ylabel('T-statistic');
    title(electrodes{e});
    grid on;
end
```

### Time-Frequency Plots

```matlab
% Visualize time-frequency results

% Load time-frequency results
load('/results/limo/group/tfce_one_sample_ttest.mat');

% Select electrode
elec_idx = find(strcmp({EEG.chanlocs.labels}, 'Cz'));

% Extract TF data
tf_data = squeeze(tfce_score(elec_idx, :, :));  % Freqs × Time

% Plot
figure;
imagesc(LIMO.data.tf_times, LIMO.data.tf_freqs, tf_data);
axis xy;
colorbar;
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title('TFCE Scores at Cz');
colormap(jet);
```

### Cluster Visualization

```matlab
% Highlight significant clusters on ERP plot

% Load cluster results
load('/results/limo/group/cluster_one_sample.mat');

% Plot ERP with significant clusters shaded
figure;
elec_idx = find(strcmp({EEG.chanlocs.labels}, 'Fz'));

plot(LIMO.data.timevect, squeeze(one_sample(elec_idx, :)), 'b-', 'LineWidth', 2);
hold on;

% Shade significant time windows
for c = 1:length(clusters)
    if clusters{c}.p < 0.05
        sig_times = clusters{c}.times;
        y_limits = ylim;
        fill([sig_times, fliplr(sig_times)], [y_limits(1)*ones(1,2), y_limits(2)*ones(1,2)], ...
             'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
end

xlabel('Time (ms)');
ylabel('T-statistic');
title('Fz: Significant Clusters Highlighted');
legend('T-statistic', 'Significant cluster');
```

## Batch Processing

### Automated First-Level Pipeline

```matlab
% Process multiple subjects automatically
clear; clc;

% Configuration
data_dir = '/data/eeg/preprocessed/';
output_dir = '/results/limo/';
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'};

% Design matrix template
% Assume all subjects have same design

for s = 1:length(subjects)
    fprintf('Processing %s...\n', subjects{s});

    % Load EEGLAB dataset
    EEG = pop_loadset('filename', [subjects{s}, '_clean.set'], 'filepath', data_dir);

    % Extract design information
    n_trials = EEG.trials;
    conditions = zeros(n_trials, 1);

    for t = 1:n_trials
        event_idx = find([EEG.event.epoch] == t, 1);
        if strcmp(EEG.event(event_idx).type, 'standard')
            conditions(t) = 1;
        else
            conditions(t) = 2;
        end
    end

    % Create design matrix
    X = zeros(n_trials, 2);
    X(conditions == 1, 1) = 1;
    X(conditions == 2, 2) = 1;

    % Setup LIMO
    LIMO = struct();
    LIMO.dir = fullfile(output_dir, subjects{s});
    LIMO.data.data = EEG.data;
    LIMO.data.chanlocs = EEG.chanlocs;
    LIMO.data.sampling_rate = EEG.srate;
    LIMO.data.start = EEG.xmin * 1000;
    LIMO.data.end = EEG.xmax * 1000;
    LIMO.design.X = X;
    LIMO.design.name = {'Standard', 'Deviant'};
    LIMO.design.method = 'OLS';

    % Run LIMO
    cd(LIMO.dir);
    limo_eeg(4);

    fprintf('Completed %s\n\n', subjects{s});
end

fprintf('All subjects processed!\n');
```

### Group-Level Batch Script

```matlab
% Run group analysis for multiple contrasts

contrasts = {'standard', 'deviant', 'deviant-standard'};

for c = 1:length(contrasts)
    fprintf('Group analysis: %s\n', contrasts{c});

    % Collect subject files
    subject_files = cell(length(subjects), 1);
    for s = 1:length(subjects)
        subject_files{s} = fullfile(output_dir, subjects{s}, sprintf('con_%d.mat', c));
    end

    % Run one-sample t-test
    output_group = fullfile(output_dir, 'group', contrasts{c});
    if ~exist(output_group, 'dir')
        mkdir(output_group);
    end

    % LIMO group analysis
    cd(output_group);
    limo_random_select('one sample t-test', subject_files{:});

    % Run TFCE correction
    limo_tfce_handling();

    fprintf('Completed: %s\n\n', contrasts{c});
end
```

## Integration with EEGLAB

### Preprocessing in EEGLAB

```matlab
% Typical EEGLAB preprocessing before LIMO

% 1. Load raw data
EEG = pop_loadset('filename', 'raw_eeg.set');

% 2. Filter
EEG = pop_eegfiltnew(EEG, 'locutoff', 0.1, 'hicutoff', 40);

% 3. Re-reference to average
EEG = pop_reref(EEG, []);

% 4. Epoch data
EEG = pop_epoch(EEG, {'standard', 'deviant'}, [-0.2 0.8]);

% 5. Baseline correction
EEG = pop_rmbase(EEG, [-200 0]);

% 6. Artifact rejection (ICA or threshold)
EEG = pop_eegthresh(EEG, 1, 1:EEG.nbchan, -100, 100, -0.2, 0.8, 0, 1);

% 7. Save preprocessed data
EEG = pop_saveset(EEG, 'filename', 'preprocessed.set');

% Now ready for LIMO analysis
```

### Export LIMO Results to EEGLAB

```matlab
% Import LIMO statistical maps back to EEGLAB for visualization

% Load LIMO results
load('/results/limo/group/one_sample_ttest_parameter_1.mat');

% Create EEGLAB-compatible structure
EEG_results = EEG;  % Copy structure
EEG_results.data = one_sample;  % Replace with T-statistics
EEG_results.trials = 1;  % Single "trial" (the group map)
EEG_results.pnts = size(one_sample, 2);  % Time points

% Plot in EEGLAB
pop_eegplot(EEG_results, 1, 1, 1);
```

## Troubleshooting

**Problem:** "Dimensions mismatch" error
**Solution:** Verify design matrix rows match number of trials; check data dimensions (channels × time × trials)

**Problem:** LIMO runs very slowly
**Solution:** Reduce data resolution (downsample time), use robust regression only when needed, enable parallel computing

**Problem:** No significant results after correction
**Solution:** Check data quality, verify preprocessing, try different correction methods (cluster vs. TFCE vs. FDR)

**Problem:** Memory errors with large datasets
**Solution:** Process electrodes in batches, reduce time resolution, use single precision, increase MATLAB memory limit

**Problem:** TFCE values seem incorrect
**Solution:** Verify spatial connectivity parameters, check electrode locations, ensure proper normalization

## Best Practices

1. **Preprocessing:**
   - Remove bad channels and trials before LIMO
   - Use consistent preprocessing across subjects
   - Baseline correct appropriately
   - Document all preprocessing steps

2. **Design Matrix:**
   - Z-score continuous predictors
   - Check for multicollinearity
   - Use orthogonal contrasts when possible
   - Save design matrix for reproducibility

3. **Model Selection:**
   - Use robust regression if outliers expected
   - Include relevant covariates (e.g., reaction time)
   - Test model assumptions when possible

4. **Multiple Comparisons:**
   - Use TFCE for most analyses (good sensitivity/specificity)
   - Consider cluster-based for spatially extended effects
   - Report both corrected and uncorrected results
   - Justify correction method choice

5. **Reporting:**
   - Report LIMO version and EEGLAB version
   - Report correction method and parameters
   - Provide electrode locations and time windows
   - Include topographic maps and time series
   - Share design matrices and contrasts

## Resources

- **GitHub:** https://github.com/LIMO-EEG-Toolbox/limo_tools
- **Documentation:** https://github.com/LIMO-EEG-Toolbox/limo_tools/wiki
- **Tutorial:** https://github.com/LIMO-EEG-Toolbox/limo_tools/blob/master/limo_eeg_tutorial.pdf
- **EEGLAB:** https://sccn.ucsd.edu/eeglab/
- **Papers:** Pernet et al. (2011, 2015)

## Citation

```bibtex
@article{pernet2011limo,
  title={LIMO EEG: a toolbox for hierarchical LInear MOdeling of ElectroEncephaloGraphic data},
  author={Pernet, Cyril R and Chauveau, Nicolas and Gaspar, Carl and Rousselet, Guillaume A},
  journal={Computational intelligence and neuroscience},
  volume={2011},
  pages={3},
  year={2011},
  publisher={Hindawi}
}

@article{pernet2015robust,
  title={Robust correlation analyses: false positive and power validation using a new open source MATLAB toolbox},
  author={Pernet, Cyril R and Wilcox, Rand and Rousselet, Guillaume A},
  journal={Frontiers in psychology},
  volume={3},
  pages={606},
  year={2015},
  publisher={Frontiers}
}
```

## Related Tools

- **EEGLAB:** EEG preprocessing and visualization
- **FieldTrip:** Alternative EEG/MEG analysis
- **Brainstorm:** Source reconstruction
- **MNE-Python:** Python EEG/MEG analysis
- **SPM:** Statistical parametric mapping (adapted for MEG)
- **ERPLAB:** ERP analysis toolbox for EEGLAB
