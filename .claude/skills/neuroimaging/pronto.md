# PRoNTo - Pattern Recognition for Neuroimaging Toolbox

## Overview

**PRoNTo** (Pattern Recognition for Neuroimaging Toolbox) is a comprehensive MATLAB toolbox for machine learning analysis of neuroimaging data, with full integration into the SPM ecosystem. Developed by the Machine Learning and Neuroimaging Laboratory (MLNL) at University College London, PRoNTo provides both a user-friendly graphical interface and powerful batch scripting capabilities for classification, regression, and feature extraction, with particular emphasis on clinical applications and biomarker discovery.

PRoNTo implements state-of-the-art pattern recognition methods including Support Vector Machines (SVM), Gaussian Process Models, and multi-kernel learning for integrating multi-modal data. The toolbox excels at diagnostic classification, prognostic prediction, and brain age estimation, providing interpretable weight maps and robust cross-validation strategies essential for clinical neuroimaging research.

**Key Features:**
- Support Vector Machines (classification and regression)
- Gaussian Process Models for probabilistic prediction
- Multi-kernel learning for multi-modal data integration
- Feature selection and dimensionality reduction
- Voxel-wise weight mapping for interpretation
- Nested cross-validation for unbiased performance estimation
- Permutation testing for statistical significance
- Brain age prediction and brain-PAD estimation
- ROI-based and whole-brain feature extraction
- Integration with SPM preprocessing
- User-friendly GUI for non-programmers
- MATLAB batch scripting for reproducibility
- Clinical diagnostic and prognostic modeling
- Multi-site and multi-center validation support

**Primary Use Cases:**
- Clinical diagnostic classification (disease vs. healthy)
- Prognostic prediction (treatment response, disease progression)
- Biomarker discovery for neurological and psychiatric disorders
- Brain age prediction and biological age estimation
- Multi-modal data integration (T1w + fMRI + DTI)
- Individual outcome prediction
- Disease subtype identification
- Treatment stratification and personalized medicine

**Official Documentation:** http://www.mlnl.cs.ucl.ac.uk/pronto/
**GitHub Repository:** https://github.com/pronto-toolbox/pronto

---

## Installation

### Download and Install

```matlab
% Download PRoNTo from official website
% http://www.mlnl.cs.ucl.ac.uk/pronto/download.html

% Extract to MATLAB toolbox directory
% e.g., /Users/username/Documents/MATLAB/PRoNTo

% Add PRoNTo to MATLAB path
addpath('/Users/username/Documents/MATLAB/PRoNTo')

% Add SPM12 to path (required)
addpath('/Users/username/Documents/MATLAB/spm12')

% Save path permanently
savepath

% Verify installation
which prt_init
% Should display PRoNTo path
```

### Launch PRoNTo GUI

```matlab
% Initialize PRoNTo
prt_init

% Launch main GUI
prt

% The PRoNTo GUI should open with options for:
% - Data & Design: Load and organize data
% - Prepare Feature Set: Define features for ML
% - Specify Model: Configure classifiers/regressors
% - Run Model: Execute analysis
% - Review Results: View performance metrics
% - Display Weights: Visualize discriminative patterns
```

### Check Dependencies

```matlab
% PRoNTo requires:
% - MATLAB R2014a or later
% - SPM12
% - Statistics and Machine Learning Toolbox (recommended)

% Verify Statistics Toolbox
ver('stats')

% If not available, some features may be limited
```

---

## Data Preparation

### Organize Data for PRoNTo

```matlab
% PRoNTo expects SPM-compatible NIfTI files
% Typical directory structure:
%
% study/
%   ├── group1/
%   │   ├── sub001_T1w.nii
%   │   ├── sub002_T1w.nii
%   │   └── ...
%   ├── group2/
%   │   ├── sub010_T1w.nii
%   │   └── ...
%   └── pronto_project.mat

% Navigate to project directory
cd('/data/pronto_study')
```

### Load Data via GUI

```matlab
% 1. Launch PRoNTo GUI
prt

% 2. Click "Data & Design"
% 3. Click "Add Groups"
%    - Group 1: Controls
%    - Group 2: Patients
% 4. Click "Add Scans" for each group
%    - Select NIfTI files using file browser
% 5. Verify number of scans per group
% 6. Click "Review" to check data organization
% 7. Save as 'PRT.mat'
```

### Load Data via Batch Script

```matlab
% Create PRoNTo project programmatically
clear all

% Initialize PRoNTo structure
PRT = struct();

% Define groups
PRT.group(1).name = 'Controls';
PRT.group(2).name = 'Patients';

% Add scans to Group 1 (Controls)
controls_dir = '/data/controls/';
control_files = spm_select('FPList', controls_dir, '^.*\.nii$');
PRT.group(1).scans = cellstr(control_files);

% Add scans to Group 2 (Patients)
patients_dir = '/data/patients/';
patient_files = spm_select('FPList', patients_dir, '^.*\.nii$');
PRT.group(2).scans = cellstr(patient_files);

% Save PRoNTo structure
save('PRT.mat', 'PRT');

fprintf('Loaded %d controls, %d patients\n', ...
    size(PRT.group(1).scans, 1), ...
    size(PRT.group(2).scans, 1));
```

---

## Feature Set Preparation

### Whole-Brain Feature Set (GUI)

```matlab
% In PRoNTo GUI:
% 1. Click "Prepare Feature Set"
% 2. Select "Build feature set"
% 3. Choose "Image modality": Structural MRI
% 4. Masking:
%    - "Use default mask" (SPM brain mask)
%    - OR "Use custom mask" (ROI mask)
% 5. Feature selection: "Use all voxels in mask"
% 6. Click "Build"
% 7. Review feature set:
%    - Number of features (voxels)
%    - Memory requirements
% 8. Save feature set
```

### ROI-Based Feature Set

```matlab
% Use AAL atlas for ROI-based features
% 1. In "Prepare Feature Set"
% 2. Select "Region-based features"
% 3. Load atlas: AAL2.nii
% 4. Select regions of interest:
%    - Hippocampus (L+R)
%    - Amygdala (L+R)
%    - Prefrontal cortex regions
% 5. Feature extraction method:
%    - "Mean" (average voxel value)
%    - "Eigenvariate" (1st principal component)
% 6. Build feature set

% This creates much smaller feature set
% Better for small sample sizes
```

### Feature Set via Batch Script

```matlab
% Load PRoNTo project
load('PRT.mat')

% Define feature set
fs_name = 'Whole_brain_T1w';

% Specify brain mask
mask_file = '/path/to/spm12/tpm/mask_ICV.nii';

% Build feature set structure
fs.name = fs_name;
fs.modality = 'sMRI';  % Structural MRI
fs.img_files = [PRT.group(1).scans; PRT.group(2).scans];

% Create mask
V = spm_vol(mask_file);
mask_data = spm_read_vols(V);
fs.mask_idx = find(mask_data > 0.5);
fs.n_features = length(fs.mask_idx);

% Add to PRT structure
PRT.fs{1} = fs;

% Save
save('PRT.mat', 'PRT');

fprintf('Feature set: %d features\n', fs.n_features);
```

---

## Classification with SVM

### Binary Classification (GUI)

```matlab
% In PRoNTo GUI:
% 1. Click "Specify Model"
% 2. Model type: "Classification"
% 3. Machine: "Support Vector Machine (SVM)"
% 4. Kernel: "Linear" (for interpretability)
% 5. Groups to classify:
%    - Group 1: Controls (label = -1)
%    - Group 2: Patients (label = +1)
% 6. Feature set: Select prepared feature set
% 7. Cross-validation:
%    - Type: "Leave-one-out" or "K-fold" (k=5 or 10)
% 8. Hyperparameters:
%    - C (regularization): 1 (default)
%    - For RBF kernel, also set gamma
% 9. Click "Save & Run Model"
```

### Linear SVM Batch Script

```matlab
% Load PRT structure
load('PRT.mat')

% Create model structure
model = struct();
model.model_name = 'SVM_Linear_AD_vs_HC';
model.machine.function = 'prt_machine_svm_bin';
model.machine.args.kernel.function = 'prt_kernel_linear';

% Regularization parameter C
model.machine.args.C = 1.0;

% Define labels
% Controls = -1, Patients = +1
n_controls = length(PRT.group(1).scans);
n_patients = length(PRT.group(2).scans);
model.labels = [ones(n_controls,1)*(-1); ones(n_patients,1)];

% Feature set
model.fs_idx = 1;  % Use first feature set

% Cross-validation
model.cv.type = 'lkout';  % Leave-k-out
model.cv.k = 1;  % Leave-one-out

% Add model to PRT
PRT.model{1} = model;
save('PRT.mat', 'PRT');

% Run model
prt_run_model(PRT, 1);  % Run model 1

% Results saved in PRT.model{1}.output
```

### View Classification Results

```matlab
% Load results
load('PRT.mat')
results = PRT.model{1}.output;

% Overall accuracy
accuracy = results.accuracy * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);

% Confusion matrix
fprintf('Confusion Matrix:\n');
disp(results.confusion_matrix);

% Sensitivity and Specificity
sensitivity = results.sensitivity * 100;
specificity = results.specificity * 100;
fprintf('Sensitivity: %.2f%%\n', sensitivity);
fprintf('Specificity: %.2f%%\n', specificity);

% Balanced accuracy
balanced_acc = (sensitivity + specificity) / 2;
fprintf('Balanced Accuracy: %.2f%%\n', balanced_acc);

% AUC (Area Under ROC Curve)
if isfield(results, 'auc')
    fprintf('AUC: %.3f\n', results.auc);
end

% Predictions per subject
predictions = results.predictions;
true_labels = model.labels;

% Subject-wise accuracy
correct = (predictions == true_labels);
fprintf('Correctly classified: %d / %d\n', sum(correct), length(correct));
```

---

## Regression and Brain Age Prediction

### Support Vector Regression (SVR)

```matlab
% Create regression model for brain age prediction
% Assumes you have age values for each subject

% Load age data
ages = load('subject_ages.txt');  % Chronological ages

% Create regression model
reg_model = struct();
reg_model.model_name = 'SVR_BrainAge';
reg_model.machine.function = 'prt_machine_svm_reg';
reg_model.machine.args.kernel.function = 'prt_kernel_linear';
reg_model.machine.args.C = 1.0;
reg_model.machine.args.epsilon = 0.1;  % SVR epsilon

% Regression targets (ages)
reg_model.targets = ages;

% Feature set
reg_model.fs_idx = 1;

% Cross-validation
reg_model.cv.type = 'lkout';
reg_model.cv.k = 1;

% Add to PRT and run
PRT.model{2} = reg_model;
save('PRT.mat', 'PRT');

prt_run_model(PRT, 2);
```

### Evaluate Regression Performance

```matlab
% Load regression results
load('PRT.mat')
reg_results = PRT.model{2}.output;

% Predicted ages
predicted_ages = reg_results.predictions;
true_ages = ages;

% Correlation
[r, p] = corr(predicted_ages, true_ages);
fprintf('Correlation: r = %.3f, p = %.4f\n', r, p);

% Mean Absolute Error (MAE)
mae = mean(abs(predicted_ages - true_ages));
fprintf('MAE: %.2f years\n', mae);

% Root Mean Square Error (RMSE)
rmse = sqrt(mean((predicted_ages - true_ages).^2));
fprintf('RMSE: %.2f years\n', rmse);

% Plot predicted vs. true ages
figure;
scatter(true_ages, predicted_ages, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(true_ages), max(true_ages)], [min(true_ages), max(true_ages)], ...
    'r--', 'LineWidth', 2);
xlabel('Chronological Age (years)');
ylabel('Predicted Age (years)');
title(sprintf('Brain Age Prediction (r=%.3f, MAE=%.2f)', r, mae));
grid on;
axis square;
saveas(gcf, 'brain_age_prediction.png');
close;
```

### Brain-PAD (Predicted Age Difference)

```matlab
% Compute brain-PAD (biomarker of accelerated/delayed aging)
brain_pad = predicted_ages - true_ages;

% Positive brain-PAD = accelerated aging
% Negative brain-PAD = younger-appearing brain

fprintf('Brain-PAD statistics:\n');
fprintf('Mean: %.2f years\n', mean(brain_pad));
fprintf('Std: %.2f years\n', std(brain_pad));

% Compare brain-PAD between groups
% Assuming first half is controls, second half is patients
n_controls = floor(length(brain_pad)/2);
brain_pad_controls = brain_pad(1:n_controls);
brain_pad_patients = brain_pad(n_controls+1:end);

[h, p, ci, stats] = ttest2(brain_pad_controls, brain_pad_patients);
fprintf('\nBrain-PAD group comparison:\n');
fprintf('Controls: %.2f ± %.2f years\n', ...
    mean(brain_pad_controls), std(brain_pad_controls));
fprintf('Patients: %.2f ± %.2f years\n', ...
    mean(brain_pad_patients), std(brain_pad_patients));
fprintf('t(%d) = %.2f, p = %.4f\n', stats.df, stats.tstat, p);

% Plot brain-PAD distributions
figure;
boxplot([brain_pad_controls; brain_pad_patients], ...
    [ones(n_controls,1); ones(length(brain_pad_patients),1)*2], ...
    'Labels', {'Controls', 'Patients'});
ylabel('Brain-PAD (years)');
title('Brain Age Gap by Group');
grid on;
saveas(gcf, 'brain_pad_comparison.png');
close;
```

---

## Weight Mapping and Interpretation

### Display Weight Maps (GUI)

```matlab
% In PRoNTo GUI after running model:
% 1. Click "Display Weights"
% 2. Select model to display
% 3. Choose display options:
%    - Threshold: Statistical threshold (e.g., 95th percentile)
%    - Overlay: Anatomical template (e.g., avg152T1.nii)
%    - Colormap: Hot (positive) and Cool (negative)
% 4. View weight maps in SPM graphics window
% 5. Save images for publication
```

### Extract and Threshold Weights

```matlab
% Load model results
load('PRT.mat')
model_idx = 1;
weights = PRT.model{model_idx}.output.weights;

% Weight map dimensions
fs = PRT.fs{1};
n_features = fs.n_features;

fprintf('Weight vector size: %d\n', length(weights));

% Map weights back to brain volume
% Get original mask
V_mask = spm_vol(fs.mask_file);
mask_data = spm_read_vols(V_mask);

% Create weight volume
weight_vol = zeros(size(mask_data));
weight_vol(fs.mask_idx) = weights;

% Threshold by absolute magnitude (top 5%)
threshold = prctile(abs(weights), 95);
weight_vol_thresh = weight_vol;
weight_vol_thresh(abs(weight_vol_thresh) < threshold) = 0;

% Save as NIfTI
V_out = V_mask;
V_out.fname = 'classifier_weights_thresholded.nii';
V_out.dt = [16 0];  % Float32
spm_write_vol(V_out, weight_vol_thresh);

fprintf('Saved thresholded weight map\n');
fprintf('%.2f%% of voxels survive threshold\n', ...
    100 * sum(weight_vol_thresh(:) ~= 0) / sum(mask_data(:) > 0));
```

### Anatomical Interpretation with AAL

```matlab
% Load AAL atlas
aal_file = '/path/to/AAL2.nii';
V_aal = spm_vol(aal_file);
aal_data = spm_read_vols(V_aal);
aal_labels = readtable('AAL2_labels.csv');  % Region names

% Reslice weight map to AAL space if needed
% (assuming already in same space)

% Find top discriminative regions
% Average absolute weight in each AAL region
n_regions = max(aal_data(:));
region_weights = zeros(n_regions, 1);

for r = 1:n_regions
    roi_mask = (aal_data == r);
    region_weights(r) = mean(abs(weight_vol(roi_mask)));
end

% Sort by discriminative power
[sorted_weights, sorted_idx] = sort(region_weights, 'descend');

% Top 10 regions
fprintf('Top 10 discriminative regions:\n');
for i = 1:10
    region_id = sorted_idx(i);
    region_name = aal_labels.Name{region_id};
    fprintf('%d. %s: %.4f\n', i, region_name, sorted_weights(i));
end
```

---

## Multi-Kernel Learning for Multi-Modal Data

### Combine Structural and Functional MRI

```matlab
% Prepare two feature sets:
% 1. Structural MRI (VBM maps)
% 2. Functional connectivity (FC matrices)

% Feature set 1: Gray matter density
fs1 = struct();
fs1.name = 'GM_density';
fs1.modality = 'sMRI';
fs1.img_files = cellstr(spm_select('FPList', '/data/vbm/', '^smwc1.*\.nii$'));
% (load mask and extract features as before)

% Feature set 2: Functional connectivity
fs2 = struct();
fs2.name = 'FC_matrix';
fs2.modality = 'fMRI';
% Load FC matrices (e.g., 90x90 AAL connectivity)
fc_files = dir('/data/fc/*_fc.mat');
fc_features = [];
for i = 1:length(fc_files)
    load(fullfile(fc_files(i).folder, fc_files(i).name), 'fc_matrix');
    % Extract upper triangle
    triu_idx = triu(true(size(fc_matrix)), 1);
    fc_features(i, :) = fc_matrix(triu_idx);
end
fs2.features = fc_features;
fs2.n_features = size(fc_features, 2);

% Add both feature sets to PRT
PRT.fs{1} = fs1;
PRT.fs{2} = fs2;
save('PRT.mat', 'PRT');
```

### Multi-Kernel SVM

```matlab
% Create multi-kernel model
mk_model = struct();
mk_model.model_name = 'MultiKernel_sMRI_fMRI';
mk_model.machine.function = 'prt_machine_svm_mkl';

% Define kernels for each modality
mk_model.machine.args.kernel{1}.function = 'prt_kernel_linear';
mk_model.machine.args.kernel{1}.fs_idx = 1;  % sMRI

mk_model.machine.args.kernel{2}.function = 'prt_kernel_linear';
mk_model.machine.args.kernel{2}.fs_idx = 2;  % fMRI

% Kernel weights (equal weighting initially)
mk_model.machine.args.kernel_weights = [0.5, 0.5];

% Regularization
mk_model.machine.args.C = 1.0;

% Labels
mk_model.labels = [ones(n_controls,1)*(-1); ones(n_patients,1)];

% Cross-validation
mk_model.cv.type = 'lkout';
mk_model.cv.k = 1;

% Run model
PRT.model{3} = mk_model;
save('PRT.mat', 'PRT');

prt_run_model(PRT, 3);

% Check learned kernel weights
learned_weights = PRT.model{3}.output.kernel_weights;
fprintf('Learned kernel weights:\n');
fprintf('sMRI: %.3f\n', learned_weights(1));
fprintf('fMRI: %.3f\n', learned_weights(2));
```

---

## Nested Cross-Validation

### Avoid Optimistic Bias

```matlab
% Nested CV: outer loop for performance estimation
%            inner loop for hyperparameter tuning

% Outer loop: 10-fold CV
outer_folds = 10;
outer_cv = struct();
outer_cv.type = 'kfold';
outer_cv.k = outer_folds;

% Inner loop: 5-fold CV for C selection
inner_folds = 5;
C_values = [0.01, 0.1, 1, 10, 100];

% Nested CV model
nested_model = struct();
nested_model.model_name = 'SVM_NestedCV';
nested_model.machine.function = 'prt_machine_svm_bin';
nested_model.machine.args.kernel.function = 'prt_kernel_linear';

% Enable hyperparameter optimization
nested_model.machine.args.optimize_hyperparams = true;
nested_model.machine.args.hyperparam_grid.C = C_values;
nested_model.machine.args.inner_cv.type = 'kfold';
nested_model.machine.args.inner_cv.k = inner_folds;

% Outer CV
nested_model.cv = outer_cv;
nested_model.labels = [ones(n_controls,1)*(-1); ones(n_patients,1)];
nested_model.fs_idx = 1;

% Run nested CV
PRT.model{4} = nested_model;
save('PRT.mat', 'PRT');

prt_run_model(PRT, 4);

% Unbiased performance estimate
unbiased_accuracy = PRT.model{4}.output.accuracy;
fprintf('Nested CV accuracy: %.2f%%\n', unbiased_accuracy * 100);

% Optimal C values per fold
optimal_C_per_fold = PRT.model{4}.output.optimal_hyperparams;
fprintf('Mean optimal C: %.3f\n', mean(optimal_C_per_fold));
```

---

## Permutation Testing

### Statistical Significance of Classification

```matlab
% Test if accuracy is significantly better than chance
% Permute labels and recompute accuracy

n_permutations = 1000;
null_accuracies = zeros(n_permutations, 1);

% Original model
original_model = PRT.model{1};
original_accuracy = original_model.output.accuracy;

% Permutation loop
fprintf('Running permutation test (%d permutations)...\n', n_permutations);
for perm = 1:n_permutations
    if mod(perm, 100) == 0
        fprintf('Permutation %d/%d\n', perm, n_permutations);
    end

    % Shuffle labels
    perm_model = original_model;
    perm_model.labels = perm_model.labels(randperm(length(perm_model.labels)));

    % Run model with permuted labels
    perm_model.model_name = sprintf('Perm_%04d', perm);
    PRT_perm = PRT;
    PRT_perm.model{1} = perm_model;

    % Suppress output
    prt_run_model(PRT_perm, 1, 'verbose', false);

    % Store null accuracy
    null_accuracies(perm) = PRT_perm.model{1}.output.accuracy;
end

% Compute p-value
p_value = sum(null_accuracies >= original_accuracy) / n_permutations;

fprintf('\nPermutation test results:\n');
fprintf('True accuracy: %.2f%%\n', original_accuracy * 100);
fprintf('Mean null accuracy: %.2f%%\n', mean(null_accuracies) * 100);
fprintf('P-value: %.4f\n', p_value);

% Plot null distribution
figure;
histogram(null_accuracies * 100, 30, 'FaceColor', [0.7 0.7 0.7]);
hold on;
xline(original_accuracy * 100, 'r-', 'LineWidth', 2, ...
    'Label', sprintf('True = %.1f%%', original_accuracy*100));
xlabel('Classification Accuracy (%)');
ylabel('Frequency');
title(sprintf('Permutation Test (p = %.4f)', p_value));
grid on;
saveas(gcf, 'permutation_test.png');
close;
```

---

## Gaussian Process Models

### Probabilistic Prediction with Uncertainty

```matlab
% Gaussian Process Classification
gp_model = struct();
gp_model.model_name = 'GP_Classification';
gp_model.machine.function = 'prt_machine_gpml_bin';

% GP kernel (squared exponential)
gp_model.machine.args.kernel.function = 'prt_kernel_se';
gp_model.machine.args.kernel.args.lengthscale = 1.0;
gp_model.machine.args.kernel.args.sigma = 1.0;

% Likelihood (for classification)
gp_model.machine.args.likelihood = 'logistic';

% Labels and features
gp_model.labels = [ones(n_controls,1)*(-1); ones(n_patients,1)];
gp_model.fs_idx = 1;

% Cross-validation
gp_model.cv.type = 'lkout';
gp_model.cv.k = 1;

% Run GP model
PRT.model{5} = gp_model;
save('PRT.mat', 'PRT');

prt_run_model(PRT, 5);

% GP provides probabilistic predictions
predictions = PRT.model{5}.output.predictions;  % Class labels
probabilities = PRT.model{5}.output.probabilities;  % P(class=+1)
uncertainty = PRT.model{5}.output.uncertainty;  % Prediction variance

% High uncertainty indicates low confidence
fprintf('GP Classification accuracy: %.2f%%\n', ...
    PRT.model{5}.output.accuracy * 100);
fprintf('Mean prediction uncertainty: %.3f\n', mean(uncertainty));
```

### GP Regression with Confidence Intervals

```matlab
% GP Regression for brain age
gp_reg_model = struct();
gp_reg_model.model_name = 'GP_BrainAge';
gp_reg_model.machine.function = 'prt_machine_gpml_reg';
gp_reg_model.machine.args.kernel.function = 'prt_kernel_se';
gp_reg_model.targets = ages;
gp_reg_model.fs_idx = 1;
gp_reg_model.cv.type = 'lkout';
gp_reg_model.cv.k = 1;

PRT.model{6} = gp_reg_model;
prt_run_model(PRT, 6);

% Extract predictions and confidence intervals
pred_mean = PRT.model{6}.output.predictions;
pred_std = sqrt(PRT.model{6}.output.variance);

% 95% confidence intervals
ci_lower = pred_mean - 1.96 * pred_std;
ci_upper = pred_mean + 1.96 * pred_std;

% Plot with confidence intervals
figure;
errorbar(true_ages, pred_mean, 1.96*pred_std, 'o', ...
    'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
hold on;
plot([min(true_ages), max(true_ages)], [min(true_ages), max(true_ages)], ...
    'r--', 'LineWidth', 2);
xlabel('Chronological Age (years)');
ylabel('Predicted Age (years) ± 95% CI');
title('GP Brain Age Prediction with Uncertainty');
grid on;
axis square;
saveas(gcf, 'gp_brain_age_ci.png');
close;
```

---

## Batch Processing and Automation

### Batch Script for Multiple Models

```matlab
% Automated analysis of multiple configurations
clear all
close all

% Configuration
study_dir = '/data/ad_study/';
output_dir = fullfile(study_dir, 'pronto_results');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Load data
load(fullfile(study_dir, 'PRT.mat'));

% Define multiple models to test
models_config = {
    struct('name', 'SVM_Linear_C01', 'kernel', 'linear', 'C', 0.1), ...
    struct('name', 'SVM_Linear_C1', 'kernel', 'linear', 'C', 1.0), ...
    struct('name', 'SVM_Linear_C10', 'kernel', 'linear', 'C', 10), ...
    struct('name', 'SVM_RBF_C1', 'kernel', 'rbf', 'C', 1.0, 'gamma', 0.01)
};

% Run all models
results_table = table();
for m = 1:length(models_config)
    cfg = models_config{m};

    fprintf('\n=== Running %s ===\n', cfg.name);

    % Create model
    model = struct();
    model.model_name = cfg.name;
    model.machine.function = 'prt_machine_svm_bin';

    % Kernel
    if strcmp(cfg.kernel, 'linear')
        model.machine.args.kernel.function = 'prt_kernel_linear';
    elseif strcmp(cfg.kernel, 'rbf')
        model.machine.args.kernel.function = 'prt_kernel_rbf';
        model.machine.args.kernel.args.gamma = cfg.gamma;
    end

    % Hyperparameters
    model.machine.args.C = cfg.C;

    % Labels and CV
    model.labels = PRT.labels;
    model.fs_idx = 1;
    model.cv.type = 'lkout';
    model.cv.k = 1;

    % Run
    PRT.model{m} = model;
    prt_run_model(PRT, m);

    % Store results
    results_table.Model{m} = cfg.name;
    results_table.Accuracy(m) = PRT.model{m}.output.accuracy;
    results_table.Sensitivity(m) = PRT.model{m}.output.sensitivity;
    results_table.Specificity(m) = PRT.model{m}.output.specificity;
    results_table.AUC(m) = PRT.model{m}.output.auc;
end

% Save results
writetable(results_table, fullfile(output_dir, 'model_comparison.csv'));
save(fullfile(output_dir, 'PRT_all_models.mat'), 'PRT');

% Display summary
fprintf('\n=== Results Summary ===\n');
disp(results_table);

% Find best model
[best_acc, best_idx] = max(results_table.Accuracy);
fprintf('\nBest model: %s (Accuracy: %.2f%%)\n', ...
    results_table.Model{best_idx}, best_acc*100);
```

---

## Multi-Site and External Validation

### Train on Site 1, Test on Site 2

```matlab
% Load data from two sites
site1_controls = cellstr(spm_select('FPList', '/data/site1/controls/', '\.nii$'));
site1_patients = cellstr(spm_select('FPList', '/data/site1/patients/', '\.nii$'));

site2_controls = cellstr(spm_select('FPList', '/data/site2/controls/', '\.nii$'));
site2_patients = cellstr(spm_select('FPList', '/data/site2/patients/', '\.nii$'));

% Build feature sets for both sites
% (assuming same preprocessing and mask)
fs_site1 = build_feature_set([site1_controls; site1_patients], mask_file);
fs_site2 = build_feature_set([site2_controls; site2_patients], mask_file);

% Labels
labels_site1 = [ones(length(site1_controls),1)*(-1); ...
                ones(length(site1_patients),1)];
labels_site2 = [ones(length(site2_controls),1)*(-1); ...
                ones(length(site2_patients),1)];

% Train on Site 1
model_site1 = struct();
model_site1.model_name = 'Train_Site1_Test_Site2';
model_site1.machine.function = 'prt_machine_svm_bin';
model_site1.machine.args.kernel.function = 'prt_kernel_linear';
model_site1.machine.args.C = 1.0;
model_site1.labels = labels_site1;

% Train (no cross-validation, just training)
PRT_train = struct();
PRT_train.fs{1} = fs_site1;
PRT_train.model{1} = model_site1;

% Manual training
svm_trained = prt_machine_svm_bin(fs_site1.features, labels_site1, model_site1.machine.args);

% Test on Site 2
predictions_site2 = prt_predict_svm(svm_trained, fs_site2.features);
accuracy_site2 = mean(predictions_site2 == labels_site2);

fprintf('Site 1 -> Site 2 generalization: %.2f%%\n', accuracy_site2*100);

% Compare to within-site performance
% (run LOOCV on Site 1)
model_site1_cv = model_site1;
model_site1_cv.cv.type = 'lkout';
model_site1_cv.cv.k = 1;
PRT_train.model{1} = model_site1_cv;
prt_run_model(PRT_train, 1);
accuracy_site1 = PRT_train.model{1}.output.accuracy;

fprintf('Within Site 1 accuracy: %.2f%%\n', accuracy_site1*100);
fprintf('Generalization gap: %.2f%%\n', (accuracy_site1 - accuracy_site2)*100);
```

---

## Visualization and Reporting

### Create ROC Curve

```matlab
% For binary classification, plot ROC curve
load('PRT.mat')
model = PRT.model{1};

% Extract decision values and labels
decision_values = model.output.decision_values;
true_labels = model.labels;

% Compute ROC
[X, Y, T, AUC] = perfcurve(true_labels, decision_values, 1);

% Plot ROC
figure;
plot(X, Y, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.3f)', AUC));
grid on;
axis square;
legend('Classifier', 'Chance', 'Location', 'southeast');
saveas(gcf, 'roc_curve.png');
close;

fprintf('AUC: %.3f\n', AUC);
```

### Generate Classification Report

```matlab
% Comprehensive report
fprintf('\n========================================\n');
fprintf('PRoNTo Classification Report\n');
fprintf('========================================\n\n');

fprintf('Model: %s\n', model.model_name);
fprintf('Date: %s\n\n', datestr(now));

fprintf('Data:\n');
fprintf('  Total subjects: %d\n', length(model.labels));
fprintf('  Controls: %d\n', sum(model.labels == -1));
fprintf('  Patients: %d\n\n', sum(model.labels == 1));

fprintf('Performance:\n');
fprintf('  Accuracy: %.2f%%\n', model.output.accuracy * 100);
fprintf('  Sensitivity: %.2f%%\n', model.output.sensitivity * 100);
fprintf('  Specificity: %.2f%%\n', model.output.specificity * 100);
fprintf('  AUC: %.3f\n\n', model.output.auc);

fprintf('Confusion Matrix:\n');
cm = model.output.confusion_matrix;
fprintf('               Predicted\n');
fprintf('               Ctrl  Pat\n');
fprintf('  Actual Ctrl   %3d   %3d\n', cm(1,1), cm(1,2));
fprintf('         Pat    %3d   %3d\n\n', cm(2,1), cm(2,2));

fprintf('========================================\n');

% Save to file
fid = fopen('classification_report.txt', 'w');
% (write same content to file)
fclose(fid);
```

---

## Best Practices and Tips

### Sample Size Considerations

```matlab
% PRoNTo works with small samples but consider:
% - Minimum ~20 subjects per class for stable results
% - More features (voxels) than samples requires regularization
% - Use nested CV to avoid overfitting
% - Consider ROI-based features for very small samples

% Check sample/feature ratio
n_samples = length(model.labels);
n_features = PRT.fs{1}.n_features;
ratio = n_samples / n_features;

fprintf('Sample/feature ratio: %.4f\n', ratio);
if ratio < 1
    warning('More features than samples! High risk of overfitting.');
    fprintf('Recommendations:\n');
    fprintf('  - Use strong regularization (small C)\n');
    fprintf('  - Reduce features via feature selection\n');
    fprintf('  - Use ROI-based features\n');
end
```

### Avoiding Common Pitfalls

```matlab
% 1. Data leakage - don't normalize across all data
% PRoNTo handles this internally in CV

% 2. Unbalanced classes - use balanced accuracy
% PRoNTo computes this automatically

% 3. Multiple testing - correct for multiple models tested
% Use Bonferroni correction if testing many configurations
n_models = 10;
alpha = 0.05;
corrected_alpha = alpha / n_models;
fprintf('Corrected significance level: %.4f\n', corrected_alpha);

% 4. Site effects - include site as covariate
% Or use harmonization (e.g., ComBat)

% 5. Overfitting - always use nested CV for hyperparameter tuning
```

---

## Troubleshooting

### Memory Issues

```matlab
% For large datasets, reduce memory usage:

% 1. Use ROI-based features instead of whole-brain
% 2. Downsample images (if appropriate)
% 3. Use linear kernel (more efficient than RBF)
% 4. Process subjects in batches

% Check memory usage
feature_size_mb = (n_features * n_samples * 8) / (1024^2);
fprintf('Feature matrix size: %.2f MB\n', feature_size_mb);
```

### Model Not Converging

```matlab
% If SVM optimization fails:
% 1. Check feature scaling (PRoNTo normalizes internally)
% 2. Try different C values
% 3. Use linear kernel instead of RBF
% 4. Check for NaN/Inf values in features

% Verify feature matrix
fs_features = PRT.fs{1}.features;
if any(isnan(fs_features(:)))
    warning('NaN values detected in features!');
end
if any(isinf(fs_features(:)))
    warning('Inf values detected in features!');
end
```

---

## Related Tools and Integration

**Preprocessing:**
- **SPM12** (Batch 1): VBM preprocessing for PRoNTo
- **CAT12** (Batch 8): Advanced VBM for structural features
- **fMRIPrep** (Batch 5): Functional preprocessing

**Alternative ML Tools:**
- **PyMVPA** (Batch 27): Python alternative with searchlight
- **Nilearn** (Batch 2): Python ML framework
- **BrainIAK** (Batch 26): Advanced MVPA methods

**Clinical Applications:**
- **Clinica** (Batch 35): Clinical neuroimaging pipeline

---

## References

- Schrouff, J., et al. (2013). PRoNTo: pattern recognition for neuroimaging toolbox. *Neuroinformatics*, 11(3), 319-337.
- Mourão-Miranda, J., et al. (2005). Classifying brain states and determining the discriminating activation patterns: Support Vector Machine on functional MRI data. *NeuroImage*, 28(4), 980-995.
- Schnack, H. G., & Kahn, R. S. (2016). Detecting neuroimaging biomarkers for psychiatric disorders: sample size matters. *Frontiers in Psychiatry*, 7, 50.
- Cole, J. H., & Franke, K. (2017). Predicting age using neuroimaging: innovative brain ageing biomarkers. *Trends in Neurosciences*, 40(12), 681-690.

**Official Website:** http://www.mlnl.cs.ucl.ac.uk/pronto/
**GitHub:** https://github.com/pronto-toolbox/pronto
**Manual:** http://www.mlnl.cs.ucl.ac.uk/pronto/prtdocs.html
**Paper:** https://doi.org/10.1007/s12021-013-9178-1
