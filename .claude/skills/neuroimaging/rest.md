# REST - Resting-State fMRI Data Analysis Toolkit

## Overview

REST (REsting-State fMRI data analysis Toolkit) is a lightweight MATLAB toolbox designed for analyzing resting-state fMRI data. It provides essential functions for calculating functional connectivity metrics including ReHo, ALFF/fALFF, functional connectivity, and performing basic statistical analyses. REST serves as the foundation for DPABI and offers a simpler, more accessible interface for researchers new to rs-fMRI analysis.

**Website:** http://restfmri.net/forum/REST
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** GNU GPL

## Key Features

- Regional Homogeneity (ReHo) calculation
- ALFF/fALFF computation
- Functional connectivity (correlation) analysis
- Frequency band filtering
- Detrending and nuisance regression
- Image calculator and math operations
- Statistical maps generation (t-test, correlation)
- ROI signal extraction
- Slice viewer for quality control
- Integration with SPM for preprocessing
- Lightweight and easy to use

## Installation

```matlab
% Download REST from: http://restfmri.net/forum/

% Extract to desired location
% Add to MATLAB path
addpath(genpath('/path/to/REST'));

% Verify installation
rest

% Or use specific functions
which rest_alff
which rest_reho
```

## Data Preparation

### Expected Data Structure

```
data/
├── sub-01/
│   ├── FunImg/
│   │   └── sub-01_filtered.nii  % Preprocessed 4D fMRI
│   └── T1Img/
│       └── sub-01_T1w.nii
├── sub-02/
│   ├── FunImg/
│   │   └── sub-02_filtered.nii
│   └── T1Img/
│       └── sub-02_T1w.nii
...
```

### Preprocessing (Using SPM)

```matlab
%% Basic preprocessing before REST analysis

% 1. Slice timing correction
% 2. Realignment
% 3. Normalization to MNI space
% 4. Spatial smoothing (optional, usually 4-6mm)

% After SPM preprocessing, organize data for REST
% Place preprocessed 4D NIfTI in SubjectID/FunImg/
```

## Regional Homogeneity (ReHo)

### Calculate ReHo

```matlab
%% REST ReHo Calculation

clear; clc;

% Setup
data_dir = '/data';
subjects = {'sub-01', 'sub-02', 'sub-03'};
output_dir = '/results/ReHo';

% Mask file (optional, use MNI brain mask)
mask_file = fullfile(spm('dir'), 'toolbox', 'FieldMap', 'brainmask.nii');
if ~exist(mask_file, 'file')
    mask_file = '';  % Will use data-derived mask
end

% Calculate ReHo for each subject
for i = 1:length(subjects)
    subj = subjects{i};

    % Input functional data
    func_file = fullfile(data_dir, subj, 'FunImg', sprintf('%s_filtered.nii', subj));

    % Output file
    output_file = fullfile(output_dir, sprintf('ReHo_%s.nii', subj));

    % Calculate ReHo
    % Cluster size: 7, 19, or 27 voxels
    cluster_size = 27;

    rest_ReHo(func_file, cluster_size, output_file, mask_file);

    fprintf('Completed ReHo for %s\n', subj);
end

% Z-standardization
for i = 1:length(subjects)
    subj = subjects{i};

    reho_file = fullfile(output_dir, sprintf('ReHo_%s.nii', subj));
    z_file = fullfile(output_dir, sprintf('zReHo_%s.nii', subj));

    % Standardize within brain mask
    rest_Standardize(reho_file, mask_file, z_file);
end
```

### ReHo with Custom Parameters

```matlab
%% Advanced ReHo calculation

% Parameters
data_file = '/data/sub-01/FunImg/sub-01_filtered.nii';
output_prefix = '/results/ReHo/sub-01';

% Different cluster sizes
cluster_sizes = [7, 19, 27];  % Surface, edge, cube neighbors

for cluster_size = cluster_sizes
    output_file = sprintf('%s_KCC%d.nii', output_prefix, cluster_size);

    % Calculate
    [KCC_Brain, Header] = rest_KCC(data_file, cluster_size);

    % Save
    rest_WriteNiftiImage(KCC_Brain, Header, output_file);

    fprintf('ReHo (cluster=%d) saved: %s\n', cluster_size, output_file);
end
```

## ALFF and fALFF

### Calculate ALFF/fALFF

```matlab
%% ALFF and fALFF Calculation

clear; clc;

% Setup
data_dir = '/data';
subjects = {'sub-01', 'sub-02', 'sub-03'};
output_dir = '/results/ALFF';

% Parameters
TR = 2.0;  % Repetition time in seconds
low_freq = 0.01;  % Low frequency cutoff (Hz)
high_freq = 0.1;  % High frequency cutoff (Hz)

% Brain mask
mask_file = '';  % Empty = auto-detect

% Calculate for each subject
for i = 1:length(subjects)
    subj = subjects{i};

    % Input
    func_file = fullfile(data_dir, subj, 'FunImg', sprintf('%s_filtered.nii', subj));

    % Outputs
    alff_file = fullfile(output_dir, sprintf('ALFF_%s.nii', subj));
    falff_file = fullfile(output_dir, sprintf('fALFF_%s.nii', subj));

    % Calculate ALFF and fALFF
    [ALFF_Brain, fALFF_Brain, Header] = rest_alff(func_file, TR, ...
                                                   low_freq, high_freq, ...
                                                   mask_file, '', '');

    % Save ALFF
    rest_WriteNiftiImage(ALFF_Brain, Header, alff_file);

    % Save fALFF
    rest_WriteNiftiImage(fALFF_Brain, Header, falff_file);

    fprintf('ALFF/fALFF completed for %s\n', subj);
end

% Z-standardize
for i = 1:length(subjects)
    subj = subjects{i};

    % ALFF
    alff_file = fullfile(output_dir, sprintf('ALFF_%s.nii', subj));
    z_alff_file = fullfile(output_dir, sprintf('zALFF_%s.nii', subj));
    rest_Standardize(alff_file, mask_file, z_alff_file);

    % fALFF
    falff_file = fullfile(output_dir, sprintf('fALFF_%s.nii', subj));
    z_falff_file = fullfile(output_dir, sprintf('zfALFF_%s.nii', subj));
    rest_Standardize(falff_file, mask_file, z_falff_file);
end
```

### Multiple Frequency Bands

```matlab
%% Calculate ALFF in different frequency bands

func_file = '/data/sub-01/FunImg/sub-01_filtered.nii';
TR = 2.0;

% Define frequency bands
bands = struct();
bands.slow_5 = [0.01, 0.027];  % Slow-5
bands.slow_4 = [0.027, 0.073]; % Slow-4
bands.slow_3 = [0.073, 0.167]; % Slow-3
bands.slow_2 = [0.167, 0.25];  % Slow-2

band_names = fieldnames(bands);

for i = 1:length(band_names)
    band = band_names{i};
    freq_range = bands.(band);

    % Calculate ALFF for this band
    [ALFF_Brain, ~, Header] = rest_alff(func_file, TR, ...
                                        freq_range(1), freq_range(2));

    % Save
    output_file = sprintf('/results/ALFF/ALFF_%s.nii', band);
    rest_WriteNiftiImage(ALFF_Brain, Header, output_file);

    fprintf('ALFF for %s band [%.3f-%.3f Hz]: Done\n', ...
            band, freq_range(1), freq_range(2));
end
```

## Functional Connectivity

### Seed-Based Correlation

```matlab
%% Seed-based functional connectivity

clear; clc;

% Seed definition (PCC for default mode network)
seed_center = [0, -53, 26];  % MNI coordinates
seed_radius = 6;  % mm

% Create seed ROI
ref_file = '/data/sub-01/FunImg/sub-01_filtered.nii';  % For header info
seed_file = '/seeds/PCC_6mm.nii';

rest_Sphere(ref_file, seed_center, seed_radius, seed_file);

% Extract seed time series and calculate FC
subjects = {'sub-01', 'sub-02', 'sub-03'};
output_dir = '/results/FC_PCC';

for i = 1:length(subjects)
    subj = subjects{i};

    % Functional data
    func_file = fullfile('/data', subj, 'FunImg', sprintf('%s_filtered.nii', subj));

    % Extract seed time series
    [seed_timeseries, ~] = rest_ReadBrainData(func_file, seed_file);
    seed_timeseries = mean(seed_timeseries, 1)';  % Average across voxels

    % Load whole-brain data
    [brain_data, Header] = rest_ReadBrainData(func_file);

    % Calculate correlation
    FC_Brain = zeros(size(brain_data, 1), 1);
    for voxel = 1:size(brain_data, 1)
        if sum(brain_data(voxel, :)) == 0
            FC_Brain(voxel) = 0;
        else
            r = corrcoef(seed_timeseries, brain_data(voxel, :)');
            FC_Brain(voxel) = r(1, 2);
        end
    end

    % Fisher z-transform
    FC_Brain_z = 0.5 * log((1 + FC_Brain) ./ (1 - FC_Brain));
    FC_Brain_z(isinf(FC_Brain_z)) = 0;
    FC_Brain_z(isnan(FC_Brain_z)) = 0;

    % Save
    output_file = fullfile(output_dir, sprintf('FC_PCC_%s.nii', subj));
    rest_WriteNiftiImage(FC_Brain_z, Header, output_file);

    fprintf('FC completed for %s\n', subj);
end
```

### ROI-to-ROI Connectivity

```matlab
%% ROI-to-ROI correlation matrix

% Define ROIs (e.g., using AAL atlas)
atlas_file = '/atlases/AAL3_61x73x61.nii';
[atlas, atlas_header] = rest_ReadNiftiImage(atlas_file);
num_rois = max(atlas(:));

% Load functional data
func_file = '/data/sub-01/FunImg/sub-01_filtered.nii';
[func_data, func_header] = rest_ReadBrainData(func_file);

% Extract ROI time series
roi_timeseries = zeros(size(func_data, 2), num_rois);

for roi = 1:num_rois
    roi_mask = (atlas == roi);
    roi_voxels = func_data(roi_mask(:), :);

    if ~isempty(roi_voxels)
        roi_timeseries(:, roi) = mean(roi_voxels, 1)';
    end
end

% Calculate correlation matrix
fc_matrix = corr(roi_timeseries);

% Fisher z-transform
fc_matrix_z = 0.5 * log((1 + fc_matrix) ./ (1 - fc_matrix));
fc_matrix_z(isinf(fc_matrix_z)) = 0;
fc_matrix_z(isnan(fc_matrix_z)) = 0;

% Visualize
figure;
imagesc(fc_matrix_z);
colorbar;
title('ROI-to-ROI Connectivity');
xlabel('ROI'); ylabel('ROI');

% Save
save('/results/ROI_FC_sub01.mat', 'fc_matrix', 'fc_matrix_z');
```

## Temporal Filtering

```matlab
%% Band-pass filtering

% Input data
func_file = '/data/sub-01/FunImg/sub-01_preprocessed.nii';
output_file = '/data/sub-01/FunImg/sub-01_filtered.nii';

% Parameters
TR = 2.0;  % seconds
low_freq = 0.01;  % Hz
high_freq = 0.1;  % Hz

% Load data
[data, header] = rest_ReadNiftiImage(func_file);

% Reshape to 2D (voxels × time)
dims = size(data);
data_2d = reshape(data, [], dims(4));

% Filter each voxel
data_filtered = rest_IdealFilter(data_2d', TR, [low_freq, high_freq])';

% Reshape back
data_filtered_4d = reshape(data_filtered, dims);

% Save
rest_WriteNiftiImage(data_filtered_4d, header, output_file);

fprintf('Filtering complete: [%.3f - %.3f Hz]\n', low_freq, high_freq);
```

## Detrending

```matlab
%% Linear and polynomial detrending

func_file = '/data/sub-01/FunImg/sub-01_preprocessed.nii';
output_file = '/data/sub-01/FunImg/sub-01_detrended.nii';

% Load data
[data, header] = rest_ReadNiftiImage(func_file);
dims = size(data);

% Reshape
data_2d = reshape(data, [], dims(4));

% Detrend (0=constant, 1=linear, 2=quadratic)
detrend_type = 1;  % Linear
data_detrended = rest_Detrend(data_2d', detrend_type)';

% Reshape and save
data_detrended_4d = reshape(data_detrended, dims);
rest_WriteNiftiImage(data_detrended_4d, header, output_file);
```

## Statistical Analysis

### Group-Level T-Test

```matlab
%% One-sample t-test

% Collect z-scored ALFF maps
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'};
n_subj = length(subjects);

% Load first image for dimensions
first_img = fullfile('/results/ALFF', sprintf('zALFF_%s.nii', subjects{1}));
[sample_data, header] = rest_ReadNiftiImage(first_img);
dims = size(sample_data);

% Load all subjects
all_data = zeros([dims, n_subj]);
for i = 1:n_subj
    img_file = fullfile('/results/ALFF', sprintf('zALFF_%s.nii', subjects{i}));
    all_data(:,:,:,i) = rest_ReadNiftiImage(img_file);
end

% Calculate t-statistics
mean_data = mean(all_data, 4);
std_data = std(all_data, [], 4);
t_map = mean_data ./ (std_data / sqrt(n_subj));

% Calculate p-values
df = n_subj - 1;
p_map = 2 * (1 - tcdf(abs(t_map), df));

% Apply threshold (p < 0.001 uncorrected)
thresh = 0.001;
t_map_thresh = t_map;
t_map_thresh(p_map > thresh) = 0;

% Save results
rest_WriteNiftiImage(t_map, header, '/results/stats/tstat.nii');
rest_WriteNiftiImage(p_map, header, '/results/stats/pvalue.nii');
rest_WriteNiftiImage(t_map_thresh, header, '/results/stats/tstat_thresh.nii');
```

### Two-Sample T-Test

```matlab
%% Compare two groups

% Group 1 (controls)
group1_subjects = {'sub-01', 'sub-02', 'sub-03'};
% Group 2 (patients)
group2_subjects = {'sub-04', 'sub-05', 'sub-06'};

n1 = length(group1_subjects);
n2 = length(group2_subjects);

% Load data
data_dir = '/results/ReHo';
first_img = fullfile(data_dir, sprintf('zReHo_%s.nii', group1_subjects{1}));
[sample_data, header] = rest_ReadNiftiImage(first_img);
dims = size(sample_data);

% Load group 1
group1_data = zeros([dims, n1]);
for i = 1:n1
    img_file = fullfile(data_dir, sprintf('zReHo_%s.nii', group1_subjects{i}));
    group1_data(:,:,:,i) = rest_ReadNiftiImage(img_file);
end

% Load group 2
group2_data = zeros([dims, n2]);
for i = 1:n2
    img_file = fullfile(data_dir, sprintf('zReHo_%s.nii', group2_subjects{i}));
    group2_data(:,:,:,i) = rest_ReadNiftiImage(img_file);
end

% Two-sample t-test at each voxel
mean1 = mean(group1_data, 4);
mean2 = mean(group2_data, 4);
var1 = var(group1_data, [], 4);
var2 = var(group2_data, [], 4);

% Pooled variance
sp = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2));

% T-statistic
t_map = (mean1 - mean2) ./ (sp * sqrt(1/n1 + 1/n2));

% P-values
df = n1 + n2 - 2;
p_map = 2 * (1 - tcdf(abs(t_map), df));

% Save
rest_WriteNiftiImage(t_map, header, '/results/stats/twosamples_tstat.nii');
rest_WriteNiftiImage(p_map, header, '/results/stats/twosamples_pvalue.nii');
```

## Utilities

### Image Calculator

```matlab
%% Perform mathematical operations on images

% Add two images
img1 = '/results/ALFF/ALFF_sub01.nii';
img2 = '/results/ALFF/ALFF_sub02.nii';
output = '/results/ALFF/ALFF_mean.nii';

rest_ImgCalculate(img1, img2, 'add', output, 0.5, 0.5);  % (img1 + img2) / 2

% Subtract
rest_ImgCalculate(img1, img2, 'subtract', output, 1, 1);  % img1 - img2

% Multiply
rest_ImgCalculate(img1, img2, 'multiply', output, 1, 1);

% Threshold
rest_ImgCalculate(img1, '', 'threshold', output, 0.5, Inf);  % Values > 0.5
```

### ROI Signal Extraction

```matlab
%% Extract time series from ROI

func_file = '/data/sub-01/FunImg/sub-01_filtered.nii';
roi_file = '/rois/hippocampus_left.nii';

% Extract
[roi_timeseries, roi_header] = rest_ReadBrainData(func_file, roi_file);

% Average across voxels
mean_timeseries = mean(roi_timeseries, 1)';

% Plot
figure;
plot(mean_timeseries);
title('ROI Time Series');
xlabel('Time Point'); ylabel('Signal');

% Save
output_file = '/results/timeseries/hippocampus_left_sub01.txt';
dlmwrite(output_file, mean_timeseries);
```

## Integration with Claude Code

When helping users with REST:

1. **Check Installation:**
   ```matlab
   which rest
   which rest_alff
   which rest_reho
   ```

2. **Common Issues:**
   - Data not in expected structure (SubjectID/FunImg/)
   - Missing brain mask
   - Incorrect TR specification
   - Memory errors with large datasets
   - NaN values in output maps

3. **Best Practices:**
   - Preprocess data with SPM before REST analysis
   - Use appropriate frequency band (0.01-0.1 Hz for rs-fMRI)
   - Z-standardize results for group analysis
   - Fisher z-transform correlations
   - Use brain mask to limit calculations
   - Check for motion artifacts before analysis
   - Validate results with visualization

4. **Output Files:**
   - ReHo maps: Regional homogeneity values
   - ALFF/fALFF maps: Amplitude measures
   - FC maps: Functional connectivity (z-scored correlations)
   - z* prefix: Z-standardized maps

## Troubleshooting

**Problem:** "Index exceeds matrix dimensions"
**Solution:** Check data dimensions, ensure 4D NIfTI format, verify file paths

**Problem:** NaN or Inf values in output
**Solution:** Check for zero variance voxels, apply brain mask, verify preprocessing quality

**Problem:** Out of memory
**Solution:** Process data in chunks, reduce spatial resolution, or use subset of time points

**Problem:** ReHo values all near 1
**Solution:** Ensure data is properly preprocessed (filtered, detrended), check cluster size

**Problem:** ALFF shows edge effects
**Solution:** Apply appropriate spatial smoothing, use brain mask, check normalization

## Resources

- Website: http://restfmri.net/forum/REST
- Forum: http://restfmri.net/forum/
- Documentation: Included with download
- YouTube: REST Tutorial Videos
- Paper: Song et al. (2011) REST: A Toolkit for Resting-State fMRI Data Processing

## Citation

```bibtex
@article{song2011rest,
  title={REST: a toolkit for resting-state functional magnetic resonance imaging data processing},
  author={Song, Xiao-Wei and Dong, Zhang-Ye and Long, Xiang-Yu and Li, Su-Fang and Zuo, Xi-Nian and Zhu, Chao-Zhe and He, Yong and Yan, Chao-Gan and Zang, Yu-Feng},
  journal={PLoS One},
  volume={6},
  number={9},
  pages={e25031},
  year={2011}
}
```

## Related Tools

- **DPABI:** Successor to REST with more features
- **DPARSF:** Data Processing Assistant for Resting-State fMRI
- **SPM:** For preprocessing
- **CONN:** Alternative connectivity toolbox
- **GRETNA:** Graph theory network analysis
