# DPABI - Data Processing & Analysis for Brain Imaging

## Overview

DPABI (Data Processing & Analysis for Brain Imaging, formerly DPARSF) is a comprehensive MATLAB-based toolbox for resting-state fMRI data analysis. It provides an intuitive GUI and batch processing capabilities for preprocessing, quality control, and various functional connectivity measures including ALFF/fALFF, ReHo, degree centrality, functional connectivity, and graph theory analysis. DPABI integrates seamlessly with SPM and includes specialized modules for clinical and developmental neuroscience.

**Website:** http://rfmri.org/dpabi
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB (requires SPM12)
**License:** GNU GPL v3

## Key Features

- Comprehensive rs-fMRI preprocessing pipeline
- Quality control with head motion assessment
- ALFF/fALFF (amplitude of low-frequency fluctuations)
- ReHo (regional homogeneity)
- Degree centrality and voxel-mirrored homotopic connectivity (VMHC)
- Seed-based functional connectivity
- Independent component analysis (ICA)
- Dynamic functional connectivity
- Graph theory analysis
- Scrubbing and framewise displacement calculation
- Integration with SPM12 and FSL
- BIDS data support
- Docker container available
- Parallel processing support

## Installation

### MATLAB Version

```matlab
% Download DPABI from: http://rfmri.org/dpabi

% Extract to desired location
% Add to MATLAB path
addpath(genpath('/path/to/DPABI'));

% Verify installation
dpabi

% Check dependencies
which spm  % SPM12 should be installed

% Optional: Add REST to path (included with DPABI)
addpath('/path/to/DPABI/Utilities/REST');
```

### Docker Version

```bash
# Pull DPABI Docker image
docker pull cgyan/dpabi

# Run DPABI in Docker
docker run -ti --rm -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/data:/data \
  cgyan/dpabi
```

### System Requirements

```matlab
% Check MATLAB version (R2014a or later recommended)
version

% Required toolboxes:
% - Image Processing Toolbox
% - Statistics and Machine Learning Toolbox
% - Signal Processing Toolbox (optional)
```

## Basic Preprocessing Pipeline

### GUI-Based Preprocessing

```matlab
%% Launch DPABI GUI
dpabi

% In GUI:
% 1. Click "DPARSFA" for standard preprocessing
% 2. Set Working Directory
% 3. Load participant list
% 4. Configure preprocessing steps:
%    - DICOM to NIfTI conversion
%    - Remove first timepoints
%    - Slice timing correction
%    - Realignment
%    - Normalize (via DARTEL or T1 unified segmentation)
%    - Smooth
%    - Detrend
%    - Filter (0.01-0.1 Hz)
% 5. Configure functional connectivity measures
% 6. Run
```

### Script-Based Preprocessing

```matlab
%% DPABI Preprocessing Script

clear; clc;

% Setup configuration
AutoDataProcessParameter.DataProcessDir = '/data/preprocessing';
AutoDataProcessParameter.SubjectNum = 20;
AutoDataProcessParameter.SubjectID = cellstr(num2str((1:20)', 'sub-%02d'));

% Functional data directory
AutoDataProcessParameter.FunctionalSessionNumber = 1;
AutoDataProcessParameter.FunctionalSessionDirs = {'FunRaw'};

% Starting directory name (DICOM or NIfTI)
AutoDataProcessParameter.StartingDirName = 'FunRaw';

% Preprocessing steps
AutoDataProcessParameter.RemoveFirstTimePoints = 5;
AutoDataProcessParameter.IsSliceTiming = 1;
AutoDataProcessParameter.SliceTiming.SliceNumber = 33;
AutoDataProcessParameter.SliceTiming.TR = 2.0;
AutoDataProcessParameter.SliceTiming.TA = 2.0 - 2.0/33;
AutoDataProcessParameter.SliceTiming.SliceOrder = [1:2:33, 2:2:33];  % Interleaved
AutoDataProcessParameter.SliceTiming.ReferenceSlice = 1;

% Realignment
AutoDataProcessParameter.IsRealign = 1;

% Normalize
AutoDataProcessParameter.IsNormalize = 1;
AutoDataProcessParameter.Normalize.Timing = 'OnFunctionalData';  % Or 'OnResults'
AutoDataProcessParameter.Normalize.BoundingBox = [-90 -126 -72; 90 90 108];
AutoDataProcessParameter.Normalize.VoxSize = [3 3 3];

% Smooth
AutoDataProcessParameter.IsSmooth = 1;
AutoDataProcessParameter.Smooth.Timing = 'OnFunctionalData';
AutoDataProcessParameter.Smooth.FWHM = [6 6 6];

% Detrend
AutoDataProcessParameter.IsDetrend = 1;
AutoDataProcessParameter.Detrend.Timing = 'AfterNormalize';

% Filter
AutoDataProcessParameter.IsFilter = 1;
AutoDataProcessParameter.Filter.Timing = 'AfterNormalize';
AutoDataProcessParameter.Filter.ALowPass_HighCutoff = 0.1;
AutoDataProcessParameter.Filter.AHighPass_LowCutoff = 0.01;

% Covariates removal
AutoDataProcessParameter.IsCovremove = 1;
AutoDataProcessParameter.Covremove.Timing = 'AfterNormalize';
AutoDataProcessParameter.Covremove.HeadMotion = 1;  % Friston 24-parameter model
AutoDataProcessParameter.Covremove.WholeBrain = 1;  % Global signal
AutoDataProcessParameter.Covremove.CSF = 1;
AutoDataProcessParameter.Covremove.WhiteMatter = 1;

% Scrubbing
AutoDataProcessParameter.IsScrubbing = 1;
AutoDataProcessParameter.Scrubbing.Timing = 'AfterPreprocessing';
AutoDataProcessParameter.Scrubbing.FD_Jenkinson_Threshold = 0.5;  % FD threshold

% Calculate metrics
AutoDataProcessParameter.IsCalALFF = 1;  % ALFF and fALFF
AutoDataProcessParameter.CalALFF.AHighPass_LowCutoff = 0.01;
AutoDataProcessParameter.CalALFF.ALowPass_HighCutoff = 0.1;

AutoDataProcessParameter.IsCalReHo = 1;  % ReHo
AutoDataProcessParameter.CalReHo.ClusterNVoxel = 27;  % 27 neighbors

AutoDataProcessParameter.IsCalDegreeCentrality = 1;  % Degree centrality
AutoDataProcessParameter.CalDegreeCentrality.rThreshold = 0.25;

% Run preprocessing
DPARSFA_run(AutoDataProcessParameter);
```

## Quality Control

### Head Motion Assessment

```matlab
%% Calculate and visualize head motion

% Directory with realignment parameters
data_dir = '/data/preprocessing';
subjects = cellstr(num2str((1:20)', 'sub-%02d'));

% Calculate framewise displacement
FD_threshold = 0.5;  % mm
[FD_Power, MeanFD_Power] = y_FD_Power(data_dir, subjects, FD_threshold);

% Jenkinson FD
[FD_Jenkinson, MeanFD_Jenkinson] = y_FD_Jenkinson(data_dir, subjects, FD_threshold);

% Plot head motion
figure;
subplot(2,1,1);
plot(MeanFD_Power);
title('Mean FD (Power)');
xlabel('Subject'); ylabel('Mean FD (mm)');
hold on; plot([1 20], [0.5 0.5], 'r--'); hold off;

subplot(2,1,2);
plot(MeanFD_Jenkinson);
title('Mean FD (Jenkinson)');
xlabel('Subject'); ylabel('Mean FD (mm)');
hold on; plot([1 20], [0.5 0.5], 'r--'); hold off;

% Identify high-motion subjects
high_motion = find(MeanFD_Power > 0.5);
fprintf('High-motion subjects (>0.5mm): %s\n', num2str(high_motion'));

% Save QC report
save(fullfile(data_dir, 'QC_HeadMotion.mat'), 'FD_Power', 'FD_Jenkinson', ...
     'MeanFD_Power', 'MeanFD_Jenkinson', 'high_motion');
```

### Temporal Signal-to-Noise Ratio

```matlab
%% Calculate tSNR

% Directory with preprocessed data
func_dir = '/data/preprocessing/FunImgNormalized';
subjects = cellstr(num2str((1:20)', 'sub-%02d'));

for i = 1:length(subjects)
    subj = subjects{i};

    % Load functional data
    func_file = fullfile(func_dir, subj, 'Filtered_4DVolume.nii');
    [Data, Header] = y_Read(func_file);

    % Calculate tSNR
    mean_signal = mean(Data, 4);
    std_signal = std(Data, [], 4);
    tSNR = mean_signal ./ std_signal;
    tSNR(isnan(tSNR)) = 0;
    tSNR(isinf(tSNR)) = 0;

    % Save tSNR map
    output_file = fullfile(func_dir, subj, 'tSNR.nii');
    y_Write(tSNR, Header, output_file);

    % Calculate mean tSNR in brain mask
    mean_tSNR = mean(tSNR(tSNR > 0));
    fprintf('%s: Mean tSNR = %.2f\n', subj, mean_tSNR);
end
```

## Functional Connectivity Measures

### ALFF and fALFF

```matlab
%% Calculate ALFF/fALFF

% Setup
data_dir = '/data/preprocessing/FunImgNormalized';
result_dir = '/data/results/ALFF';
subjects = cellstr(num2str((1:20)', 'sub-%02d'));

% Band for ALFF
low_freq = 0.01;
high_freq = 0.1;
TR = 2.0;

% Calculate ALFF and fALFF
y_alff_falff(data_dir, ...
             result_dir, ...
             low_freq, ...
             high_freq, ...
             TR, ...
             subjects);

% Results saved in:
% - mALFF_*.nii (ALFF maps)
% - mfALFF_*.nii (fALFF maps)

% Z-standardize for group analysis
y_Standardize(fullfile(result_dir, 'mALFF_001.nii'), ...
              fullfile(result_dir, 'zmALFF_001.nii'), ...
              'SPM_Mask', fullfile(spm('dir'), 'tpm', 'mask_ICV.nii'));
```

### Regional Homogeneity (ReHo)

```matlab
%% Calculate ReHo

% Setup
data_dir = '/data/preprocessing/FunImgNormalized';
result_dir = '/data/results/ReHo';
subjects = cellstr(num2str((1:20)', 'sub-%02d'));

% Cluster size (7, 19, or 27 neighbors)
cluster_size = 27;

% Mask file
mask_file = fullfile(spm('dir'), 'tpm', 'mask_ICV.nii');

% Calculate ReHo
for i = 1:length(subjects)
    subj = subjects{i};

    input_file = fullfile(data_dir, subj, 'Filtered_4DVolume.nii');
    output_file = fullfile(result_dir, sprintf('ReHo_%s.nii', subj));

    % Calculate
    y_reho(input_file, cluster_size, output_file, mask_file);

    % Z-standardize
    z_output = fullfile(result_dir, sprintf('zReHo_%s.nii', subj));
    y_Standardize(output_file, z_output, 'SPM_Mask', mask_file);
end
```

### Degree Centrality

```matlab
%% Calculate Degree Centrality

% Setup
data_dir = '/data/preprocessing/FunImgNormalized';
result_dir = '/data/results/DegreeCentrality';
subjects = cellstr(num2str((1:20)', 'sub-%02d'));

% Correlation threshold
r_threshold = 0.25;

% Mask
mask_file = fullfile(spm('dir'), 'tpm', 'mask_ICV.nii');

% Calculate degree centrality
for i = 1:length(subjects)
    subj = subjects{i};

    input_file = fullfile(data_dir, subj, 'Filtered_4DVolume.nii');

    % Degree centrality
    output_dc = fullfile(result_dir, sprintf('DegreeCentrality_PositiveWeightedSumBrain_%s.nii', subj));

    y_DegreeCentrality(input_file, ...
                       r_threshold, ...
                       output_dc, ...
                       mask_file);

    fprintf('Completed %s\n', subj);
end
```

### Seed-Based Functional Connectivity

```matlab
%% Seed-based FC analysis

% Define seed ROI (e.g., PCC for default mode network)
seed_center = [0, -52, 18];  % MNI coordinates
seed_radius = 6;  % mm

% Create seed mask
dim = [61, 73, 61];
voxel_size = [3, 3, 3];
seed_mask = y_Sphere(seed_center, seed_radius, dim, voxel_size);

% Save seed mask
seed_file = '/data/seeds/PCC_6mm.nii';
y_Write(seed_mask, Header, seed_file);

% Calculate FC for all subjects
data_dir = '/data/preprocessing/FunImgNormalized';
result_dir = '/data/results/FC_PCC';
subjects = cellstr(num2str((1:20)', 'sub-%02d'));

for i = 1:length(subjects)
    subj = subjects{i};

    input_file = fullfile(data_dir, subj, 'Filtered_4DVolume.nii');
    output_file = fullfile(result_dir, sprintf('FC_PCC_%s.nii', subj));

    % Calculate seed-based FC
    y_SCA(input_file, seed_file, output_file);

    % Fisher z-transform
    z_output = fullfile(result_dir, sprintf('zFC_PCC_%s.nii', subj));
    y_FisherZ(output_file, z_output);
end
```

## ROI-Based Analysis

```matlab
%% ROI-to-ROI connectivity matrix

% Define ROI atlas (e.g., AAL)
atlas_file = fullfile(spm('dir'), 'toolbox', 'DPABI', 'Templates', 'AAL3_61x73x61.nii');

% Load atlas
[atlas_data, atlas_header] = y_Read(atlas_file);
num_rois = max(atlas_data(:));

% Process subjects
data_dir = '/data/preprocessing/FunImgNormalized';
subjects = cellstr(num2str((1:20)', 'sub-%02d'));

% Initialize connectivity matrix
all_fc_matrices = zeros(num_rois, num_rois, length(subjects));

for i = 1:length(subjects)
    subj = subjects{i};

    % Load functional data
    func_file = fullfile(data_dir, subj, 'Filtered_4DVolume.nii');
    [func_data, func_header] = y_Read(func_file);

    % Extract ROI time series
    roi_timeseries = zeros(size(func_data, 4), num_rois);
    for roi = 1:num_rois
        roi_mask = (atlas_data == roi);
        roi_voxels = func_data(repmat(roi_mask, [1 1 1 size(func_data, 4)]));
        roi_voxels = reshape(roi_voxels, [], size(func_data, 4));
        roi_timeseries(:, roi) = mean(roi_voxels, 1)';
    end

    % Calculate correlation matrix
    fc_matrix = corr(roi_timeseries);

    % Fisher z-transform
    fc_matrix = 0.5 * log((1 + fc_matrix) ./ (1 - fc_matrix));
    fc_matrix(isinf(fc_matrix)) = 0;
    fc_matrix(isnan(fc_matrix)) = 0;

    all_fc_matrices(:, :, i) = fc_matrix;
end

% Save results
save('/data/results/ROI_FC_matrices.mat', 'all_fc_matrices');

% Visualize average connectivity
mean_fc = mean(all_fc_matrices, 3);
figure;
imagesc(mean_fc);
colorbar;
title('Average ROI-to-ROI Connectivity');
xlabel('ROI'); ylabel('ROI');
```

## Statistical Analysis

### One-Sample T-Test

```matlab
%% Group-level analysis (one-sample t-test)

% Collect ALFF maps
data_dir = '/data/results/ALFF';
subjects = cellstr(num2str((1:20)', 'sub-%02d'));

alff_files = cell(length(subjects), 1);
for i = 1:length(subjects)
    alff_files{i} = fullfile(data_dir, sprintf('zmALFF_%s.nii', subjects{i}));
end

% Setup SPM batch
matlabbatch{1}.spm.stats.factorial_design.dir = {'/data/stats/ALFF_onesample'};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = alff_files;

% Covariates (e.g., mean FD)
load('/data/preprocessing/QC_HeadMotion.mat', 'MeanFD_Power');
matlabbatch{1}.spm.stats.factorial_design.cov.c = MeanFD_Power;
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'MeanFD';
matlabbatch{1}.spm.stats.factorial_design.cov.iCFI = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.iCC = 1;

% Masking
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;

% Estimate model
matlabbatch{2}.spm.stats.fmri_est.spmmat = {'/data/stats/ALFF_onesample/SPM.mat'};

% Contrast
matlabbatch{3}.spm.stats.con.spmmat = {'/data/stats/ALFF_onesample/SPM.mat'};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Positive';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;

% Results
matlabbatch{4}.spm.stats.results.spmmat = {'/data/stats/ALFF_onesample/SPM.mat'};
matlabbatch{4}.spm.stats.results.conspec.contrasts = 1;
matlabbatch{4}.spm.stats.results.conspec.threshdesc = 'FWE';
matlabbatch{4}.spm.stats.results.conspec.thresh = 0.05;
matlabbatch{4}.spm.stats.results.conspec.extent = 0;

% Run
spm_jobman('run', matlabbatch);
```

### Two-Sample T-Test

```matlab
%% Group comparison (patients vs controls)

% Controls
controls = cellstr(num2str((1:10)', 'sub-%02d'));
controls_files = cell(length(controls), 1);
for i = 1:length(controls)
    controls_files{i} = fullfile('/data/results/ReHo', sprintf('zReHo_%s.nii', controls{i}));
end

% Patients
patients = cellstr(num2str((11:20)', 'sub-%02d'));
patients_files = cell(length(patients), 1);
for i = 1:length(patients)
    patients_files{i} = fullfile('/data/results/ReHo', sprintf('zReHo_%s.nii', patients{i}));
end

% Setup two-sample t-test
matlabbatch{1}.spm.stats.factorial_design.dir = {'/data/stats/ReHo_twosamples'};
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = controls_files;
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = patients_files;
matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0;
matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;
matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0;

% Run analysis (similar to above)
% ... estimation, contrasts, results ...
```

## Integration with Claude Code

When helping users with DPABI:

1. **Check Installation:**
   ```matlab
   which dpabi
   which spm
   dpabi  % Should launch GUI
   ```

2. **Common Issues:**
   - SPM12 not in MATLAB path
   - Missing Image Processing Toolbox
   - Incorrect file structure (DPABI expects specific naming)
   - Memory errors with large datasets
   - Parallel processing configuration

3. **Best Practices:**
   - Always perform QC before statistical analysis
   - Use scrubbing for high-motion subjects (FD > 0.5mm)
   - Regress out nuisance covariates (WM, CSF, motion)
   - Band-pass filter 0.01-0.1 Hz for rs-fMRI
   - Fisher z-transform correlation values
   - Include mean FD as covariate in group analysis
   - Save batch configuration for reproducibility

4. **Output Files:**
   - `Filtered_4DVolume.nii`: Preprocessed 4D fMRI
   - `rp_*.txt`: Realignment parameters
   - `FD_*.txt`: Framewise displacement
   - `zmALFF_*.nii`: Z-standardized ALFF maps
   - `zReHo_*.nii`: Z-standardized ReHo maps
   - `zFC_*.nii`: Z-transformed FC maps

## Troubleshooting

**Problem:** "Cannot find SPM"
**Solution:** Add SPM12 to MATLAB path: `addpath('/path/to/spm12')`

**Problem:** Out of memory during preprocessing
**Solution:** Process subjects sequentially instead of parallel, reduce number of workers

**Problem:** High motion in subjects
**Solution:** Use scrubbing (remove high-motion volumes), increase FD threshold, or exclude subjects

**Problem:** Negative or zero ReHo values
**Solution:** Check for proper detrending and filtering, ensure sufficient time points

**Problem:** ALFF maps show artifacts
**Solution:** Verify brain mask, check normalization quality, ensure proper filtering

## Resources

- Website: http://rfmri.org/dpabi
- Manual: http://rfmri.org/content/dpabi-manual
- Forum: http://rfmri.org/forum
- GitHub: https://github.com/Chaogan-Yan/DPABI
- Docker Hub: https://hub.docker.com/r/cgyan/dpabi
- YouTube: DPABI Tutorial Videos

## Citation

```bibtex
@article{yan2016dpabi,
  title={DPABI: data processing \& analysis for (resting-state) brain imaging},
  author={Yan, Chao-Gan and Wang, Xin-Di and Zuo, Xi-Nian and Zang, Yu-Feng},
  journal={Neuroinformatics},
  volume={14},
  number={3},
  pages={339--351},
  year={2016},
  publisher={Springer}
}
```

## Related Tools

- **SPM12:** Required for preprocessing
- **REST:** Resting-state fMRI toolkit (included with DPABI)
- **CONN:** Alternative connectivity toolbox
- **DPARSF:** Predecessor to DPABI
- **GRETNA:** Graph theory network analysis
