# GIFT - Group ICA/IVA of fMRI Toolbox

## Overview

GIFT (Group ICA/IVA of fMRI Toolbox) is a comprehensive MATLAB toolbox for performing independent component analysis (ICA) and independent vector analysis (IVA) on functional MRI data. Developed at the Mind Research Network, GIFT specializes in identifying spatially independent brain networks and their temporal dynamics, making it essential for resting-state network analysis, artifact detection, and multi-subject functional connectivity studies.

**Website:** https://trendscenter.org/software/gift/
**Platform:** MATLAB (Windows/macOS/Linux) + Standalone
**Language:** MATLAB
**License:** GPL-3.0

## Key Features

- Group ICA for multi-subject analysis
- Independent Vector Analysis (IVA)
- Multiple ICA algorithms (Infomax, FastICA, Erica, etc.)
- Temporal ICA and spatial ICA
- Back-reconstruction for individual subjects
- Dual regression analysis
- Dynamic functional network connectivity (dFNC)
- Frequency domain analysis (spectral dFNC)
- Temporal sorting of components
- Component classification (RSN vs artifact)
- Batch processing and scripting
- Integration with SPM
- GUI and command-line interfaces

## Installation

### MATLAB Version

```matlab
% Download GIFT from: https://trendscenter.org/software/gift/

% Extract to desired location
% Add to MATLAB path
addpath(genpath('/path/to/GroupICATv4.0c'));

% Verify installation
which groupica

% Launch GUI
gift

% Check dependencies
% Requires Statistics Toolbox
% Optional: Signal Processing Toolbox, Parallel Computing Toolbox
```

### Standalone Version

```bash
# Download standalone installer
# Includes MATLAB Runtime (no MATLAB license needed)

# Linux
chmod +x GIFT_Linux_Installer.install
./GIFT_Linux_Installer.install

# Follow installation prompts
# Launch
./gift

# Windows: Run .exe installer
# macOS: Open .app bundle
```

## Basic Group ICA Workflow

### 1. Setup Analysis

```matlab
%% Launch GIFT Setup GUI
gift

% Or programmatic setup
clear param;

% Analysis name
param.prefix = 'groupica';
param.outputDir = '/results/gift_analysis';

% Input files (preprocessed fMRI data)
param.files = {
    '/data/sub-01/func/filtered_func_data.nii'
    '/data/sub-02/func/filtered_func_data.nii'
    '/data/sub-03/func/filtered_func_data.nii'
    % ... add all subjects
};

% Data reduction parameters
param.numReductionSteps = 2;  % Two-step PCA
param.numOfPC1 = 30;  % First reduction (per subject)
param.numOfPC2 = 20;  % Second reduction (group level)

% ICA parameters
param.numComp = 20;  % Number of components
param.algoType = 1;  % 1=Infomax, 2=FastICA, 3=Erica, 4=Simbec, etc.

% Save parameters
save('/results/gift_analysis/gift_params.mat', 'param');
```

### 2. Run Analysis

```matlab
%% Run Complete GIFT Analysis

% Method 1: Using batch file
icatb_batch_file_run('/results/gift_analysis/gift_batch.m');

% Method 2: Step-by-step
% Load parameter file
load('/results/gift_analysis/gift_params.mat');

% Data reduction (PCA)
param = icatb_dataReduction(param);

% ICA decomposition
param = icatb_runica(param);

% Back-reconstruction
param = icatb_backReconstruct(param);

% Calibrate components
param = icatb_calculateICA(param);

% Component scaling
icatb_scaleICA(param);

fprintf('Analysis complete!\n');
```

### 3. View Results

```matlab
%% Display Components

% Launch results viewer
icatb_displayGUI;

% Select analysis directory
% View:
% - Component spatial maps
% - Time courses
% - Spectra
% - Subject-specific maps
```

## Detailed Workflow Steps

### Data Preprocessing

```matlab
%% Preprocessing recommendations before GIFT

% Using SPM:
% 1. Realignment
% 2. Slice timing correction
% 3. Coregistration (optional)
% 4. Normalization to MNI
% 5. Spatial smoothing (4-6mm)
% 6. Temporal filtering (optional high-pass)
% 7. Motion regression (optional)

% Note: GIFT works on preprocessed 4D NIfTI files
% Organize as: SubjectID/func/filtered_func_data.nii
```

### Setup Parameters

```matlab
%% Detailed parameter configuration

param = struct();

% Basic settings
param.prefix = 'groupica_20comp';
param.outputDir = '/results/gift';
param.which_analysis = 1;  % 1=ICA, 2=IVA, 3=GIG-ICA

% Input data
param.dataType = 'real';  % 'real' or 'complex'
param.dataSelectionMethod = 4;  % 4=Select files manually

% File selection
param.files = icatb_selectEntry('typeEntity', 'file', ...
                                'typeSelection', 'multiple', ...
                                'filter', '*.nii');

% Mask
param.maskFile = fullfile(spm('dir'), 'tpm', 'mask_ICV.nii');

% Data reduction
param.numReductionSteps = 2;
param.numOfPC1 = 30;  % Components retained per subject
param.numOfPC2 = 20;  % Group-level components

% PCA options
param.pcaType = 1;  % 1=standard, 2=expectation maximization, 3=SVD

% ICA algorithm
param.algorithm = 1;  % 1=Infomax, 2=FastICA, 3=Erica, 14=Semi-blind Infomax
param.algoType = 1;

% ICA options
param.numComp = 20;  % Number of components to extract
param.numEstimation = 1;  % ICASSO: number of runs (1 or more for stability)

% Back-reconstruction method
param.backReconType = 4;  % 4=GICA3 (best for group)

% Calibration
param.scaleType = 2;  % 0=no scaling, 1=percent signal change, 2=Z-scores

% Parallel processing
param.parallel_info.mode = 'serial';  % 'serial' or 'parallel'
param.parallel_info.num_workers = 4;

% Save
icatb_save(fullfile(param.outputDir, 'gift_params.mat'), 'param');
```

### Running ICA

```matlab
%% Execute ICA Analysis

% Initialize
param = icatb_setup_analysis(param);

% Step 1: Data reduction (PCA)
fprintf('Step 1: Data reduction...\n');
param = icatb_dataReduction(param);

% Step 2: ICA
fprintf('Step 2: Running ICA...\n');
param = icatb_runica(param);

% Step 3: Back-reconstruction
fprintf('Step 3: Back-reconstruction...\n');
param = icatb_backReconstruct(param);

% Step 4: Calibrate
fprintf('Step 4: Calibrating components...\n');
param = icatb_calculateICA(param);

% Step 5: Scale
fprintf('Step 5: Scaling components...\n');
icatb_scaleICA(param);

disp('Group ICA analysis complete!');
```

## Component Analysis

### Viewing Components

```matlab
%% Display and explore components

% Launch component viewer
icatb_displayGUI;

% Or load programmatically
param_file = '/results/gift/groupica_20comp_ica_parameter_info.mat';
load(param_file);

% Display specific component
comp_number = 5;
icatb_displayComp(param_file, comp_number);

% Overlay on structural
icatb_overlayComp(param_file, comp_number, 'structFile', 'ch2.nii');
```

### Component Statistics

```matlab
%% Calculate component statistics

% Load subject component data
param_file = '/results/gift/groupica_20comp_ica_parameter_info.mat';
load(param_file);

% Load spatial maps
numComp = 20;
numSubjects = 10;

spatial_maps = zeros(64, 64, 30, numComp, numSubjects);  % Adjust dimensions
time_courses = zeros(150, numComp, numSubjects);  % timepoints × components × subjects

for comp = 1:numComp
    for subj = 1:numSubjects
        % Load subject-specific component
        filename = sprintf('sub%02d_component_ica_s1__component_%d.nii', subj, comp);
        spatial_maps(:,:,:,comp,subj) = spm_read_vols(spm_vol(fullfile(sesInfo.outputDir, filename)));

        % Load time course
        tc_file = sprintf('sub%02d_timecourses_ica_s1_.nii', subj);
        tc_data = load(fullfile(sesInfo.outputDir, tc_file));
        time_courses(:,comp,subj) = tc_data(:,comp);
    end
end

% Calculate spatial similarity across subjects
for comp = 1:numComp
    spatial_corr = corr(reshape(spatial_maps(:,:,:,comp,:), [], numSubjects));
    fprintf('Component %d: Mean inter-subject correlation = %.3f\n', ...
            comp, mean(spatial_corr(triu(true(size(spatial_corr)), 1))));
end
```

### Component Classification

```matlab
%% Classify components (RSN vs artifact)

% Manual classification
% View each component and label as:
% - RSN (resting-state network)
% - Artifact (motion, physiological noise)

% Automated classification using templates
template_file = '/templates/RSN_templates.nii';  % Standard RSN templates

% Spatial correlation with templates
[class_labels, max_corr] = icatb_component_viewer(param_file, ...
                                                    'template', template_file);

% Common RSN components:
% - Default Mode Network (DMN)
% - Salience Network (SN)
% - Central Executive Network (CEN)
% - Visual Network
% - Motor Network
% - Auditory Network

% Common artifacts:
% - Motion-related
% - Physiological noise (cardiac, respiratory)
% - Susceptibility artifacts
% - Scanner drift
```

## Dynamic Functional Network Connectivity (dFNC)

### Sliding Window dFNC

```matlab
%% Calculate dynamic functional connectivity

% Setup dFNC parameters
param_file = '/results/gift/groupica_20comp_ica_parameter_info.mat';
load(param_file);

% Window parameters
window_size = 30;  % TRs (e.g., 30 TRs * 2s = 60s window)
window_step = 1;  % Sliding step (TRs)

% Select components of interest (exclude artifacts)
comp_network_names = {'DMN_1', 'DMN_2', 'SN', 'CEN_L', 'CEN_R', ...
                      'Visual', 'Motor', 'Auditory'};
comp_indices = [1, 3, 5, 7, 9, 12, 15, 18];  % Component numbers

% Calculate windowed FC
[dFNC, window_info] = icatb_dfnc(param_file, ...
                                  'comp_number', comp_indices, ...
                                  'wsize', window_size, ...
                                  'window_alpha', 3, ...  % Tapered window
                                  'num_repetitions', 1);

% dFNC dimensions: [num_windows × num_connections × num_subjects]

% Visualize
figure;
imagesc(squeeze(mean(dFNC, 3)));  % Average across subjects
title('Dynamic Functional Network Connectivity');
xlabel('Connection'); ylabel('Window');
colorbar;
```

### dFNC State Analysis

```matlab
%% Identify recurring connectivity states

% Cluster dFNC windows using k-means
num_states = 5;

% Reshape dFNC for clustering
[num_windows, num_connections, num_subjects] = size(dFNC);
dFNC_reshaped = reshape(dFNC, num_windows * num_subjects, num_connections);

% K-means clustering
[state_assignments, centroids] = kmeans(dFNC_reshaped, num_states, ...
                                         'Replicates', 100, ...
                                         'Distance', 'correlation');

% Visualize states
figure;
for k = 1:num_states
    subplot(2, 3, k);
    imagesc(squareform(centroids(k,:)));
    title(sprintf('State %d', k));
    axis square;
end

% Calculate state metrics
% - Dwell time: how long subjects stay in each state
% - Transition frequency: how often subjects switch states
% - Fractional windows: proportion of time in each state

state_metrics = icatb_dfnc_stats(state_assignments, num_subjects);
```

## Advanced Features

### Independent Vector Analysis (IVA)

```matlab
%% IVA: Jointly estimate components across subjects

param.which_analysis = 2;  % IVA instead of ICA

% IVA algorithm
param.IVA_algorithm = 'iva-gl';  % 'iva-gl' or 'iva-l'

% Run IVA
param = icatb_setup_analysis(param);
param = icatb_dataReduction(param);
param = icatb_iva(param);  % IVA instead of runica
param = icatb_calculateICA(param);
```

### ICASSO for Stability

```matlab
%% Run multiple ICA realizations to assess stability

param.numEstimation = 20;  % Run ICA 20 times

% ICASSO will:
% - Run ICA multiple times with different initializations
% - Cluster components across runs
% - Provide stability index (Iq)

% After analysis, view ICASSO results
icatb_icassoShow(param_file);

% Components with high Iq (>0.8) are reliable
```

### Mancovan (Multivariate Analysis of Covariance)

```matlab
%% Test for group differences in components

% Setup design matrix
% Example: Test age effect on component spatial maps

% Prepare data
features = 'spatial maps';  % or 'timecourses spectra', 'fnc correlations'

% Covariates
age = [25, 30, 28, 35, 40, 32, ...];  % Age for each subject
group = [1, 1, 1, 2, 2, 2, ...];  % 1=Controls, 2=Patients

% Run Mancovan
icatb_run_mancovan(param_file, ...
                   'features', features, ...
                   'comp', 1:20, ...  % All components
                   'covariates', [age', group'], ...
                   'covariate_names', {'Age', 'Group'});

% Results show which components differ by group/covariate
```

## Batch Processing

```matlab
%% Complete batch script for multiple datasets

% Define multiple analyses
studies = {'study1', 'study2', 'study3'};

for s = 1:length(studies)
    study = studies{s};

    % Setup parameters
    param = struct();
    param.prefix = sprintf('groupica_%s', study);
    param.outputDir = sprintf('/results/%s/gift', study);

    % Collect subject files
    data_dir = sprintf('/data/%s', study);
    subjects = dir(fullfile(data_dir, 'sub-*'));

    param.files = cell(length(subjects), 1);
    for i = 1:length(subjects)
        param.files{i} = fullfile(data_dir, subjects(i).name, ...
                                  'func', 'filtered_func_data.nii');
    end

    % Set parameters
    param.numReductionSteps = 2;
    param.numOfPC1 = 30;
    param.numOfPC2 = 20;
    param.numComp = 20;
    param.algoType = 1;

    % Run analysis
    fprintf('Processing %s...\n', study);
    param = icatb_setup_analysis(param);
    param = icatb_dataReduction(param);
    param = icatb_runica(param);
    param = icatb_backReconstruct(param);
    param = icatb_calculateICA(param);
    icatb_scaleICA(param);

    fprintf('%s complete!\n\n', study);
end
```

## Integration with Claude Code

When helping users with GIFT:

1. **Check Installation:**
   ```matlab
   which gift
   which icatb_setup_analysis
   gift  % Should launch GUI
   ```

2. **Common Issues:**
   - Memory errors with large datasets (reduce PCA components)
   - ICA not converging (try different algorithm)
   - Components look noisy (increase number of subjects, check preprocessing)
   - ICASSO stability low (increase ICA runs, check data quality)
   - Mismatched dimensions (verify all images same size)

3. **Best Practices:**
   - Use 20-30 subjects minimum for stable group ICA
   - Number of components: typically 20-100 depending on data
   - Always visually inspect components
   - Use ICASSO for reliability assessment
   - Exclude motion-corrupted timepoints before ICA
   - Consider high-pass filtering (>0.01 Hz)
   - Scale components to Z-scores for group comparisons
   - Document component labeling decisions

4. **Choosing Parameters:**
   - **Number of components:** Data-driven (MDL/AIC) or fixed (20-50)
   - **PCA reduction:** Retain ~80-90% variance
   - **ICA algorithm:** Infomax (default), FastICA (faster), Erica (complex data)
   - **Window size (dFNC):** 30-60 seconds typical
   - **Number of states:** 3-7 common for dFNC

## Troubleshooting

**Problem:** Out of memory during data reduction
**Solution:** Reduce `numOfPC1`, process fewer subjects, or use expectation-maximization PCA

**Problem:** ICA produces noisy components
**Solution:** Increase number of subjects, improve preprocessing, try different ICA algorithm

**Problem:** ICASSO shows low stability (Iq < 0.7)
**Solution:** Increase `numEstimation`, reduce number of components, check data quality

**Problem:** Components don't match known RSNs
**Solution:** Verify preprocessing (especially normalization), check if using correct mask, increase number of components

**Problem:** dFNC windows show no structure
**Solution:** Check window size (may be too small/large), exclude artifacts, verify component selection

## Resources

- Website: https://trendscenter.org/software/gift/
- Manual: https://trendscenter.org/software/gift/docs/
- Forum: https://groups.google.com/forum/#!forum/gift-users
- Tutorials: https://trendscenter.org/software/gift/tutorials/
- Publications: https://trendscenter.org/software/gift/publications/

## Citation

```bibtex
@article{calhoun2001gift,
  title={A method for making group inferences from functional MRI data using independent component analysis},
  author={Calhoun, Vince D and Adali, Tulay and Pearlson, Godfrey D and Pekar, James J},
  journal={Human brain mapping},
  volume={14},
  number={3},
  pages={140--151},
  year={2001}
}

@article{allen2014tracking,
  title={Tracking whole-brain connectivity dynamics in the resting state},
  author={Allen, Elena A and Damaraju, Eswar and Plis, Sergey M and Erhardt, Erik B and Eichele, Tom and Calhoun, Vince D},
  journal={Cerebral cortex},
  volume={24},
  number={3},
  pages={663--676},
  year={2014}
}
```

## Related Tools

- **FSL MELODIC:** FSL's ICA tool (single/multi-subject)
- **DPABI:** Includes ICA modules
- **CONN:** Functional connectivity with ICA preprocessing
- **Nilearn:** Python ICA via scikit-learn
- **FastICA:** Standalone ICA algorithm
- **SPM:** Preprocessing and integration
