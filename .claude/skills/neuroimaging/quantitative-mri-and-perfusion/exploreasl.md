# ExploreASL: Automated Multi-Center ASL Processing Pipeline

## Overview

**ExploreASL** is a comprehensive, open-source MATLAB pipeline for processing and analyzing multi-center Arterial Spin Labeling (ASL) datasets. Developed by the ExploreASL consortium, it provides fully automated processing from raw ASL images to quantified cerebral blood flow (CBF) maps, with extensive quality control, multi-site harmonization support, and integration with structural MRI. ExploreASL follows ASL White Paper recommendations and supports all major ASL acquisition schemes.

### Key Features

- **End-to-End Automation**: From raw ASL to quantified CBF
- **Multi-Sequence Support**: pCASL, CASL, PASL, multi-PLD
- **BIDS Compatible**: Native support for BIDS ASL datasets
- **Quality Control**: Automated QC with visual and quantitative metrics
- **CBF Quantification**: Following ASL consensus paper recommendations
- **Partial Volume Correction**: Accurate gray matter and white matter CBF
- **Multi-Center**: Harmonization tools for consortium studies
- **Integration**: SPM12, CAT12, FSL compatibility
- **Vascular Territories**: Automated vascular territory labeling
- **Longitudinal**: Support for repeated measurements
- **Comprehensive Reports**: HTML reports with figures and statistics

### Scientific Foundation

Arterial Spin Labeling (ASL) uses magnetically labeled arterial blood water as an endogenous tracer to measure cerebral blood flow (CBF):

- **Labeling**: Arterial blood is magnetically tagged upstream
- **Post-Labeling Delay (PLD)**: Wait for labeled blood to reach tissue
- **Readout**: Acquire labeled and control images
- **Subtraction**: Control - Label = Perfusion-weighted signal
- **Quantification**: Convert to absolute CBF (mL/100g/min)

ExploreASL implements validated kinetic models, handles multiple ASL variants, and provides robust processing for clinical and research applications.

### Primary Use Cases

1. **Multi-Center Studies**: ENIGMA, EPAD, ADNI consortia
2. **Cerebrovascular Disease**: Stroke, stenosis, moyamoya
3. **Neurodegenerative Disease**: Alzheimer's, FTD, Parkinson's
4. **Brain Tumors**: Perfusion characterization and grading
5. **Aging Studies**: Longitudinal perfusion changes
6. **Clinical Trials**: Treatment response monitoring

---

## Installation and Setup

### Prerequisites

```matlab
% ExploreASL requires:
% - MATLAB R2017b or later
% - SPM12
% - Sufficient disk space (~2-5 GB per subject)

% Check MATLAB version
version

% Expected: R2017b or later
```

### Download ExploreASL

```matlab
% Method 1: Download from GitHub
% Visit: https://github.com/ExploreASL/ExploreASL
% Download latest release ZIP

% Method 2: Clone with git
% In terminal:
% git clone https://github.com/ExploreASL/ExploreASL.git

% Extract to desired location
ExploreASL_dir = '/path/to/ExploreASL';
```

### Install SPM12

```matlab
% Download SPM12 from: https://www.fil.ion.ucl.ac.uk/spm/

% Add SPM12 to MATLAB path
spm_dir = '/path/to/spm12';
addpath(spm_dir);

% Verify SPM12
spm('defaults', 'fmri');

fprintf('SPM12 version: %s\n', spm('version'));
```

### Initialize ExploreASL

```matlab
% Add ExploreASL to path
addpath(genpath(ExploreASL_dir));

% Initialize ExploreASL
ExploreASL_Initialize

% This will:
% - Set up paths
% - Check dependencies
% - Display version information

fprintf('ExploreASL initialized successfully\n');
```

---

## Data Preparation

### BIDS Format for ASL

```matlab
% BIDS structure for ASL data
% dataset/
%   sub-01/
%     anat/
%       sub-01_T1w.nii.gz
%       sub-01_T1w.json
%     perf/
%       sub-01_asl.nii.gz
%       sub-01_asl.json
%       sub-01_m0scan.nii.gz
%       sub-01_m0scan.json
%       sub-01_aslcontext.tsv

% Example asl.json parameters
asl_params = struct();
asl_params.MagneticFieldStrength = 3.0;
asl_params.ArterialSpinLabelingType = 'PCASL';
asl_params.PostLabelingDelay = 1.8;  % seconds
asl_params.LabelingDuration = 1.8;  % seconds
asl_params.BackgroundSuppression = true;
asl_params.M0Type = 'Separate';
asl_params.RepetitionTime = 4.0;

% Write JSON
jsonStr = jsonencode(asl_params);
fid = fopen('sub-01_asl.json', 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('ASL parameters configured\n');
```

### Import to ExploreASL

```matlab
% Import BIDS dataset to ExploreASL format
bids_dir = '/path/to/bids_dataset';
analysis_dir = '/path/to/ExploreASL_analysis';

% Run BIDS import
ExploreASL_ImportBIDS(bids_dir, analysis_dir);

% This creates ExploreASL directory structure:
% analysis/
%   DatasetRoot.mat
%   DatasetRoot.json
%   rawdata/
%   derivatives/
%     ExploreASL/
%       lock/
%       Population/
%       Sub-01/

fprintf('BIDS dataset imported\n');
```

### Configure Study Parameters

```matlab
% Create dataPar structure for processing
dataPar = struct();

% Dataset information
dataPar.D.ROOT = analysis_dir;
dataPar.D.MyStudy = 'ASL_Study';

% Sequence parameters (if not in JSON)
dataPar.Q.LabelingType = 'PASL';  % or 'PCASL', 'CASL'
dataPar.Q.PLD = 1800;  % ms
dataPar.Q.LabelingDuration = 1800;  % ms

% Processing options
dataPar.Q.BackgroundSuppressionNumberPulses = 2;
dataPar.Q.readoutDim = '2D';  % or '3D'

% Quality control
dataPar.Quality = 1;  % Enable QC

% Save configuration
save(fullfile(analysis_dir, 'dataPar.mat'), 'dataPar');

fprintf('Study parameters configured\n');
```

---

## Basic Processing

### Run ExploreASL Pipeline

```matlab
% Full ExploreASL pipeline has 3 modules:
% 1. Structural Module
% 2. ASL Module
% 3. Population Module

% Run complete pipeline
[x] = ExploreASL(analysis_dir);

% This performs:
% - Structural preprocessing (T1w)
% - ASL preprocessing and quantification
% - Population-level analysis and QC
% - Generation of reports

fprintf('ExploreASL pipeline completed\n');
```

### Run Individual Modules

```matlab
% Run structural module only
[x] = ExploreASL(analysis_dir, [1 0 0]);

% Run ASL module only (requires structural completed)
[x] = ExploreASL(analysis_dir, [0 1 0]);

% Run population module only (requires individual processing)
[x] = ExploreASL(analysis_dir, [0 0 1]);

fprintf('Individual modules can be run separately\n');
```

### Process Single Subject

```matlab
% Process specific subject
subject_id = 'sub-01';

% Set up minimal processing
[x] = ExploreASL_Initialize(analysis_dir);

% Process subject
iSubject = 1;  % Subject index
x = xASL_module_Structural(x, iSubject);
x = xASL_module_ASL(x, iSubject);

fprintf('Subject %s processed\n', subject_id);
```

---

## ASL Sequences

### pCASL (Pseudo-Continuous ASL)

```matlab
% Configure for pCASL
dataPar.Q.LabelingType = 'PCASL';
dataPar.Q.PLD = 1800;  % ms (typical: 1500-2000)
dataPar.Q.LabelingDuration = 1800;  % ms (typical: 1500-2000)
dataPar.Q.SliceReadoutTime = 40;  % ms (2D readout)

% Background suppression (recommended for pCASL)
dataPar.Q.BackgroundSuppressionNumberPulses = 2;
dataPar.Q.BackgroundSuppressionPulseTime = [1680 2830];  % ms

fprintf('pCASL configuration:\n');
fprintf('  PLD: %d ms\n', dataPar.Q.PLD);
fprintf('  Labeling duration: %d ms\n', dataPar.Q.LabelingDuration);
```

### PASL (Pulsed ASL)

```matlab
% Configure for PASL
dataPar.Q.LabelingType = 'PASL';
dataPar.Q.PLD = 800;  % ms (typical: 600-1200, shorter than pCASL)
dataPar.Q.LabelingDuration = 700;  % ms (QUIPSS II)

% PASL-specific parameters
dataPar.Q.PASL_Type = 'QUIPSS-II';  % or 'EPISTAR', 'PICORE'

fprintf('PASL configuration:\n');
fprintf('  PLD: %d ms\n', dataPar.Q.PLD);
fprintf('  Bolus duration: %d ms\n', dataPar.Q.LabelingDuration);
```

### Multi-PLD ASL

```matlab
% Configure for multi-PLD acquisition
dataPar.Q.LabelingType = 'PCASL';
dataPar.Q.Initial_PLD = 250;  % ms
dataPar.Q.PLD = [250 500 1000 1500 2000 2500];  % Multiple PLDs (ms)
dataPar.Q.LabelingDuration = 1800;  % ms

% Multi-PLD analysis options
dataPar.Q.MultiPLD = 1;  % Enable multi-PLD analysis
dataPar.Q.ATT_estimation = 1;  % Estimate arterial transit time

fprintf('Multi-PLD configuration:\n');
fprintf('  PLDs: %s ms\n', mat2str(dataPar.Q.PLD));
fprintf('  Arterial transit time estimation enabled\n');
```

---

## Quality Control

### Automated QC Metrics

```matlab
% Load QC results after processing
qc_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                   'Population', 'Stats', 'QC_collection_ASL.json');

if exist(qc_file, 'file')
    qc_data = spm_jsonread(qc_file);

    % Display QC metrics for first subject
    fprintf('QC Metrics for %s:\n', qc_data.subjects{1});
    fprintf('  Spatial CoV: %.3f\n', qc_data.SpatialCoV_WholeImage(1));
    fprintf('  CBF average: %.1f mL/100g/min\n', qc_data.mean_CBF_GM(1));
    fprintf('  SNR: %.1f\n', qc_data.SNR_GM(1));
    fprintf('  Motion: %.2f mm\n', qc_data.MotionSD_mm(1));
end
```

### Visual QC Reports

```matlab
% ExploreASL generates HTML reports automatically
% Reports are saved in:
report_dir = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                     'Population', 'Report');

% View main QC report
qc_report = fullfile(report_dir, 'QC_collection_ASL.html');

if exist(qc_report, 'file')
    % Open in browser
    web(qc_report, '-browser');

    fprintf('QC report opened in browser\n');
else
    fprintf('QC report not found. Run population module first.\n');
end
```

### Identify Poor Quality Scans

```matlab
% Load QC data
qc_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                   'Population', 'Stats', 'QC_collection_ASL.json');
qc_data = spm_jsonread(qc_file);

% Define QC thresholds
threshold_spatialCoV = 0.4;  % Spatial coefficient of variation
threshold_motion = 1.0;  % mm
threshold_CBF_min = 20;  % mL/100g/min (lower bound)
threshold_CBF_max = 80;  % mL/100g/min (upper bound)

% Identify problematic subjects
n_subjects = length(qc_data.subjects);
excluded = [];

for iSub = 1:n_subjects
    exclude_reasons = {};

    % Check spatial CoV
    if qc_data.SpatialCoV_WholeImage(iSub) > threshold_spatialCoV
        exclude_reasons{end+1} = 'High spatial CoV';
    end

    % Check motion
    if qc_data.MotionSD_mm(iSub) > threshold_motion
        exclude_reasons{end+1} = 'Excessive motion';
    end

    % Check CBF range
    mean_cbf = qc_data.mean_CBF_GM(iSub);
    if mean_cbf < threshold_CBF_min || mean_cbf > threshold_CBF_max
        exclude_reasons{end+1} = 'CBF out of range';
    end

    if ~isempty(exclude_reasons)
        fprintf('⚠ %s: %s\n', qc_data.subjects{iSub}, strjoin(exclude_reasons, ', '));
        excluded = [excluded; iSub];
    end
end

fprintf('\n%d/%d subjects flagged for exclusion\n', length(excluded), n_subjects);
```

---

## CBF Quantification

### Quantification Model

```matlab
% ExploreASL uses general kinetic model:
% CBF = λ · ΔM / (2 · α · M0 · T1blood · (exp(-w/T1blood) - exp(-(τ+w)/T1blood)))
%
% Where:
% - λ: blood-brain partition coefficient (default: 0.9 g/mL)
% - ΔM: perfusion-weighted signal
% - α: labeling efficiency
% - M0: equilibrium magnetization
% - T1blood: T1 of arterial blood (default: 1650 ms at 3T)
% - w: post-labeling delay (PLD)
% - τ: labeling duration

% View quantification parameters
fprintf('CBF Quantification Parameters:\n');
fprintf('  Blood-brain partition coefficient: %.2f g/mL\n', 0.9);
fprintf('  Labeling efficiency (pCASL): %.2f\n', 0.85);
fprintf('  Labeling efficiency (PASL): %.2f\n', 0.98);
fprintf('  T1 blood @ 3T: %d ms\n', 1650);
fprintf('  T1 blood @ 1.5T: %d ms\n', 1350);
```

### Load CBF Maps

```matlab
% Load quantified CBF map
subject_id = 'sub-01';
cbf_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                    subject_id, 'ASL_1', 'qCBF.nii');

% Read CBF image
cbf_img = xASL_io_Nifti2Im(cbf_file);

% Display CBF statistics
fprintf('CBF Statistics for %s:\n', subject_id);
fprintf('  Mean CBF: %.1f mL/100g/min\n', mean(cbf_img(cbf_img>0)));
fprintf('  Median CBF: %.1f mL/100g/min\n', median(cbf_img(cbf_img>0)));
fprintf('  Range: %.1f - %.1f mL/100g/min\n', ...
        min(cbf_img(cbf_img>0)), max(cbf_img(cbf_img>0)));

% Visualize CBF map (middle slice)
figure('Name', 'CBF Map');
slice_idx = round(size(cbf_img, 3) / 2);
imagesc(cbf_img(:, :, slice_idx), [0 80]);
colorbar;
title(sprintf('%s - CBF (mL/100g/min)', subject_id));
axis equal tight;
```

### Partial Volume Correction

```matlab
% ExploreASL performs PVC automatically
% Load PVC-corrected CBF maps

% Gray matter CBF (PVC-corrected)
cbf_gm_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                       subject_id, 'ASL_1', 'qCBF_PVC2_GM.nii');

% White matter CBF (PVC-corrected)
cbf_wm_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                       subject_id, 'ASL_1', 'qCBF_PVC2_WM.nii');

% Read images
if exist(cbf_gm_file, 'file')
    cbf_gm = xASL_io_Nifti2Im(cbf_gm_file);
    cbf_wm = xASL_io_Nifti2Im(cbf_wm_file);

    fprintf('Partial Volume Corrected CBF:\n');
    fprintf('  Gray matter CBF: %.1f mL/100g/min\n', mean(cbf_gm(cbf_gm>0)));
    fprintf('  White matter CBF: %.1f mL/100g/min\n', mean(cbf_wm(cbf_wm>0)));
    fprintf('  GM/WM ratio: %.2f\n', mean(cbf_gm(cbf_gm>0)) / mean(cbf_wm(cbf_wm>0)));
end
```

---

## Spatial Processing

### Registration to T1w

```matlab
% ExploreASL automatically registers ASL to T1w
% Check registration quality

% Load registration transformation
reg_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                   subject_id, 'ASL_1', 'ASL_to_T1.mat');

if exist(reg_file, 'file')
    % Load transformation
    reg_mat = load(reg_file);

    fprintf('ASL to T1w registration:\n');
    fprintf('  Transformation: %s\n', reg_file);
    fprintf('  Registration successful\n');

    % Visualize registration quality
    % Load registered mean ASL
    mean_asl_reg = xASL_io_Nifti2Im(fullfile(analysis_dir, 'derivatives', ...
                                     'ExploreASL', subject_id, 'ASL_1', ...
                                     'mean_control_reg.nii'));

    % Load T1w
    t1w = xASL_io_Nifti2Im(fullfile(analysis_dir, 'derivatives', ...
                           'ExploreASL', subject_id, 'T1.nii'));

    % Display overlay
    figure('Name', 'Registration QC');
    slice_idx = round(size(t1w, 3) / 2);
    imagesc(t1w(:, :, slice_idx), 'AlphaData', 0.7);
    hold on;
    contour(mean_asl_reg(:, :, slice_idx), 'r');
    title('ASL (red contour) registered to T1w');
    axis equal tight;
end
```

### Normalization to MNI Space

```matlab
% Load CBF in MNI space
cbf_mni_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                       subject_id, 'ASL_1', 'qCBF_MNI.nii');

if exist(cbf_mni_file, 'file')
    cbf_mni = xASL_io_Nifti2Im(cbf_mni_file);

    fprintf('CBF normalized to MNI152 space\n');
    fprintf('  MNI CBF dimensions: %s\n', mat2str(size(cbf_mni)));

    % Compare with template
    mni_template = fullfile(spm_dir, 'canonical', 'avg152T1.nii');

    if exist(mni_template, 'file')
        fprintf('  MNI152 template available for comparison\n');
    end
end
```

### ROI-Based CBF Extraction

```matlab
% Extract CBF from anatomical ROIs using atlas

% Load atlas (e.g., Hammers atlas provided by ExploreASL)
atlas_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                     'Population', 'Atlases', 'Hammers.nii');

% Load subject CBF in MNI space
cbf_mni = xASL_io_Nifti2Im(cbf_mni_file);

% Load atlas
atlas = xASL_io_Nifti2Im(atlas_file);

% Extract mean CBF per ROI
roi_labels = unique(atlas(atlas>0));
n_rois = length(roi_labels);

roi_cbf = zeros(n_rois, 1);
roi_names = cell(n_rois, 1);

for i = 1:n_rois
    roi_mask = atlas == roi_labels(i);
    roi_cbf(i) = mean(cbf_mni(roi_mask & cbf_mni>0));
    roi_names{i} = sprintf('ROI_%d', roi_labels(i));
end

% Display results
fprintf('Regional CBF Values:\n');
for i = 1:min(10, n_rois)  % Show first 10 ROIs
    fprintf('  %s: %.1f mL/100g/min\n', roi_names{i}, roi_cbf(i));
end
```

---

## Vascular Territories

### Vascular Territory Mapping

```matlab
% Load vascular territory maps
% ExploreASL includes vascular territory atlases

% Major cerebral arteries:
% - Left/Right Anterior Cerebral Artery (ACA)
% - Left/Right Middle Cerebral Artery (MCA)
% - Left/Right Posterior Cerebral Artery (PCA)

vascular_atlas_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                               'Population', 'Atlases', 'VascularTerritories.nii');

if exist(vascular_atlas_file, 'file')
    vascular_atlas = xASL_io_Nifti2Im(vascular_atlas_file);

    % Extract CBF by vascular territory
    territories = {'L_ACA', 'R_ACA', 'L_MCA', 'R_MCA', 'L_PCA', 'R_PCA'};
    territory_labels = [1, 2, 3, 4, 5, 6];

    fprintf('CBF by Vascular Territory:\n');
    for i = 1:length(territories)
        mask = vascular_atlas == territory_labels(i);
        cbf_territory = mean(cbf_mni(mask & cbf_mni>0));
        fprintf('  %s: %.1f mL/100g/min\n', territories{i}, cbf_territory);
    end
end
```

### Detect Vascular Pathology

```matlab
% Compare hemispheric CBF to detect asymmetry

% Load left/right hemisphere masks
lh_mask_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                       'Population', 'Atlases', 'LeftHemisphere.nii');
rh_mask_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                       'Population', 'Atlases', 'RightHemisphere.nii');

if exist(lh_mask_file, 'file') && exist(rh_mask_file, 'file')
    lh_mask = xASL_io_Nifti2Im(lh_mask_file);
    rh_mask = xASL_io_Nifti2Im(rh_mask_file);

    % Calculate hemispheric CBF
    cbf_lh = mean(cbf_mni(lh_mask>0 & cbf_mni>0));
    cbf_rh = mean(cbf_mni(rh_mask>0 & cbf_mni>0));

    % Calculate asymmetry index
    asymmetry_index = abs(cbf_lh - cbf_rh) / ((cbf_lh + cbf_rh) / 2) * 100;

    fprintf('Hemispheric CBF:\n');
    fprintf('  Left hemisphere: %.1f mL/100g/min\n', cbf_lh);
    fprintf('  Right hemisphere: %.1f mL/100g/min\n', cbf_rh);
    fprintf('  Asymmetry index: %.1f%%\n', asymmetry_index);

    if asymmetry_index > 10
        fprintf('  ⚠ Significant hemispheric asymmetry detected\n');
    else
        fprintf('  ✓ Normal hemispheric symmetry\n');
    end
end
```

---

## Advanced Analysis

### Multi-PLD Analysis

```matlab
% Analyze multi-PLD data for arterial transit time (ATT)

% Check if multi-PLD data was processed
att_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                   subject_id, 'ASL_1', 'ATT.nii');

if exist(att_file, 'file')
    % Load ATT map
    att_map = xASL_io_Nifti2Im(att_file);

    fprintf('Arterial Transit Time Analysis:\n');
    fprintf('  Mean ATT (GM): %.0f ms\n', mean(att_map(att_map>0)) * 1000);
    fprintf('  Median ATT (GM): %.0f ms\n', median(att_map(att_map>0)) * 1000);
    fprintf('  Range: %.0f - %.0f ms\n', ...
            min(att_map(att_map>0))*1000, max(att_map(att_map>0))*1000);

    % Visualize ATT map
    figure('Name', 'Arterial Transit Time');
    slice_idx = round(size(att_map, 3) / 2);
    imagesc(att_map(:, :, slice_idx) * 1000, [500 2500]);  % Convert to ms
    colorbar;
    title('ATT (ms)');
    axis equal tight;

    % Identify regions with delayed ATT (potential stenosis)
    delayed_threshold = 1.8;  % seconds
    delayed_mask = att_map > delayed_threshold;

    if sum(delayed_mask(:)) > 0
        fprintf('  ⚠ Regions with delayed ATT (>%.1f s) detected\n', delayed_threshold);
    end
else
    fprintf('Multi-PLD data not available for ATT analysis\n');
end
```

### Longitudinal Processing

```matlab
% Process longitudinal ASL data

% Subject with multiple timepoints
subject_id = 'sub-01';
sessions = {'ses-baseline', 'ses-6month', 'ses-12month'};

cbf_longitudinal = zeros(length(sessions), 1);

for i = 1:length(sessions)
    cbf_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                       [subject_id '_' sessions{i}], 'ASL_1', 'qCBF.nii');

    if exist(cbf_file, 'file')
        cbf_img = xASL_io_Nifti2Im(cbf_file);
        cbf_longitudinal(i) = mean(cbf_img(cbf_img>0));

        fprintf('%s - %s: %.1f mL/100g/min\n', subject_id, sessions{i}, ...
                cbf_longitudinal(i));
    end
end

% Calculate longitudinal change
if all(cbf_longitudinal > 0)
    cbf_change = (cbf_longitudinal(end) - cbf_longitudinal(1)) / cbf_longitudinal(1) * 100;

    fprintf('\nLongitudinal CBF change: %.1f%%\n', cbf_change);

    % Plot trajectory
    figure('Name', 'Longitudinal CBF');
    plot(1:length(sessions), cbf_longitudinal, 'o-', 'LineWidth', 2);
    xlabel('Timepoint');
    ylabel('Mean CBF (mL/100g/min)');
    title(sprintf('%s - Longitudinal CBF', subject_id));
    xticks(1:length(sessions));
    xticklabels(sessions);
    grid on;
end
```

---

## Population Analysis

### Group Statistics

```matlab
% Perform population-level analysis

% Load population statistics
pop_stats_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                         'Population', 'Stats', 'CBF_statistics.mat');

if exist(pop_stats_file, 'file')
    load(pop_stats_file);

    fprintf('Population Statistics:\n');
    fprintf('  N subjects: %d\n', length(subjects));
    fprintf('  Mean CBF (all subjects): %.1f ± %.1f mL/100g/min\n', ...
            mean(cbf_all), std(cbf_all));
    fprintf('  Age range: %.0f - %.0f years\n', min(age_all), max(age_all));

    % Age-CBF correlation
    [r, p] = corrcoef(age_all, cbf_all);
    fprintf('  Age-CBF correlation: r=%.3f, p=%.4f\n', r(1,2), p(1,2));

    % Plot age vs CBF
    figure('Name', 'Age-CBF Relationship');
    scatter(age_all, cbf_all, 50, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('Age (years)');
    ylabel('Mean CBF (mL/100g/min)');
    title('Age-Related CBF Changes');
    lsline;  % Add regression line
    grid on;
end
```

### Group Comparison

```matlab
% Compare CBF between clinical groups

% Load group labels
group_file = fullfile(analysis_dir, 'participants.tsv');

if exist(group_file, 'file')
    % Read participants file
    participants = tdfread(group_file, '\t');

    % Extract groups
    controls = strcmp(participants.diagnosis, 'control');
    patients = strcmp(participants.diagnosis, 'patient');

    % Compare CBF
    cbf_controls = cbf_all(controls);
    cbf_patients = cbf_all(patients);

    fprintf('Group Comparison:\n');
    fprintf('  Controls (n=%d): %.1f ± %.1f mL/100g/min\n', ...
            sum(controls), mean(cbf_controls), std(cbf_controls));
    fprintf('  Patients (n=%d): %.1f ± %.1f mL/100g/min\n', ...
            sum(patients), mean(cbf_patients), std(cbf_patients));

    % Statistical test
    [h, p, ci, stats] = ttest2(cbf_controls, cbf_patients);

    fprintf('  t-test: t(%.0f)=%.2f, p=%.4f\n', stats.df, stats.tstat, p);

    if p < 0.05
        diff_pct = (mean(cbf_patients) - mean(cbf_controls)) / mean(cbf_controls) * 100;
        fprintf('  ✓ Significant difference (%.1f%% change)\n', diff_pct);
    else
        fprintf('  No significant difference\n');
    end

    % Visualize
    figure('Name', 'Group Comparison');
    boxplot([cbf_controls; cbf_patients], [ones(length(cbf_controls),1); ...
            2*ones(length(cbf_patients),1)], 'Labels', {'Controls', 'Patients'});
    ylabel('Mean CBF (mL/100g/min)');
    title('CBF by Diagnostic Group');
    grid on;
end
```

---

## Batch Processing

### Process Multiple Subjects

```matlab
% Batch process entire dataset

% Define study directory
study_dir = '/path/to/analysis';

% Get list of subjects
subject_dirs = dir(fullfile(study_dir, 'rawdata', 'sub-*'));
n_subjects = length(subject_dirs);

fprintf('Processing %d subjects...\n', n_subjects);

% Process all subjects
for iSub = 1:n_subjects
    subject_id = subject_dirs(iSub).name;

    fprintf('Processing %s (%d/%d)...\n', subject_id, iSub, n_subjects);

    try
        % Run ExploreASL for this subject
        [x] = ExploreASL(study_dir, [], subject_id);
        fprintf('  ✓ %s completed\n', subject_id);

    catch ME
        fprintf('  ✗ %s failed: %s\n', subject_id, ME.message);
        continue;
    end
end

fprintf('Batch processing complete\n');
```

### Parallel Processing

```matlab
% Use MATLAB parallel computing for faster processing

% Check parallel pool
if isempty(gcp('nocreate'))
    % Start parallel pool
    parpool('local', 4);  % 4 workers
end

% Get subjects
subject_list = {subject_dirs.name};

% Process in parallel
parfor iSub = 1:length(subject_list)
    subject_id = subject_list{iSub};

    fprintf('Worker processing %s...\n', subject_id);

    try
        % Initialize ExploreASL in worker
        ExploreASL_Initialize;

        % Process subject
        [x] = ExploreASL(study_dir, [], subject_id);

        fprintf('  ✓ %s completed\n', subject_id);

    catch ME
        fprintf('  ✗ %s failed: %s\n', subject_id, ME.message);
    end
end

fprintf('Parallel processing complete\n');
```

---

## Troubleshooting

### Common Issues

```matlab
% Issue 1: Low CBF values
% Check:
% - M0 calibration image quality
% - Labeling efficiency parameter
% - PLD appropriate for population

% Issue 2: Poor registration
% Check:
% - T1w image quality
% - ASL image SNR
% - Registration parameters

% Issue 3: High motion
% Check QC metrics:
qc_data = spm_jsonread(qc_file);
high_motion = find(qc_data.MotionSD_mm > 1.0);

fprintf('Subjects with high motion (>1mm):\n');
for i = 1:length(high_motion)
    fprintf('  %s: %.2f mm\n', qc_data.subjects{high_motion(i)}, ...
            qc_data.MotionSD_mm(high_motion(i)));
end
```

### Debug Processing

```matlab
% Enable verbose logging
dataPar.settings.VERBOSE = true;

% Save intermediate files
dataPar.settings.DELETETEMP = false;

% Check processing log
log_file = fullfile(analysis_dir, 'derivatives', 'ExploreASL', ...
                   subject_id, 'ProcessingLog.txt');

if exist(log_file, 'file')
    % Read log
    fid = fopen(log_file, 'r');
    log_content = fread(fid, '*char')';
    fclose(fid);

    fprintf('Processing log:\n%s\n', log_content);
end
```

---

## Best Practices

### Recommended Workflow

```matlab
fprintf('ExploreASL Best Practices:\n');
fprintf('1. Use BIDS format for data organization\n');
fprintf('2. Include M0 calibration images\n');
fprintf('3. Document acquisition parameters in JSON\n');
fprintf('4. Run full QC before analysis\n');
fprintf('5. Use PVC for cortical CBF\n');
fprintf('6. Check registration quality visually\n');
fprintf('7. Report ExploreASL version\n');
fprintf('8. Use multi-center tools for consortia\n');
```

---

## References

### Key Publications

1. Mutsaerts, H. J., et al. (2020). "ExploreASL: An image processing pipeline for multi-center ASL perfusion MRI studies." *NeuroImage*, 219, 117031.

2. Alsop, D. C., et al. (2015). "Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications: A consensus of the ISMRM perfusion study group and the European consortium for ASL in dementia." *Magnetic Resonance in Medicine*, 73(1), 102-116.

## Citation

```bibtex
@article{mutsaerts2020multi,
  title={ExploreASL: An image processing pipeline for multi-center ASL perfusion MRI studies},
  author={Mutsaerts, Henk J. M. M. and Petr, Jan and Vaclavu, Lenka and others},
  journal={NeuroImage},
  volume={219},
  pages={117031},
  year={2020},
  doi={10.1016/j.neuroimage.2020.117031}
}
```

### Documentation and Resources

- **Website**: https://sites.google.com/view/exploreasl
- **GitHub**: https://github.com/ExploreASL/ExploreASL
- **Documentation**: https://exploreasl.github.io/Documentation/
- **Forum**: https://groups.google.com/g/exploreasl
- **Example Data**: Available on website

### Related Tools

- **BASIL**: FSL Bayesian ASL analysis
- **ASLPrep**: BIDS-compliant ASL preprocessing
- **SPM12**: Statistical analysis
- **FSL**: Alternative ASL tools
- **ASLtbx**: Alternative MATLAB toolbox

---

## See Also

- **basil.md**: FSL Bayesian ASL analysis
- **aslprep.md**: BIDS ASL preprocessing
- **spm.md**: Statistical parametric mapping
- **fsl.md**: FSL analysis tools
- **fmriprep.md**: fMRI preprocessing (structural integration)
