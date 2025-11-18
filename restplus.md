# RESTplus - Enhanced Resting-State fMRI Analysis Toolkit

## Overview

RESTplus is an enhanced version of the Resting-State fMRI Data Analysis Toolkit (REST/DPABI), providing comprehensive tools for analyzing resting-state functional connectivity and spontaneous brain activity. RESTplus extends the original REST toolbox with additional metrics including dynamic functional connectivity, enhanced amplitude of low-frequency fluctuation (ALFF/fALFF), regional homogeneity (ReHo), degree centrality (DC), and voxel-mirrored homotopic connectivity (VMHC). It features both a user-friendly graphical interface and batch processing capabilities, making it accessible for interactive analysis and high-throughput studies.

RESTplus integrates preprocessing (via DPARSF), quality control, multiple resting-state metrics computation, and statistical analysis into a unified platform. It supports various preprocessing strategies including global signal regression options, CompCor, and ICA-based denoising. The toolkit is particularly valuable for clinical research, where standardized resting-state biomarkers are needed for characterizing brain disorders, tracking disease progression, and predicting treatment outcomes.

**Official Website:** http://restfmri.net/forum/restplus
**Repository:** Part of DPABI toolbox (http://rfmri.org/dpabi)
**Documentation:** http://restfmri.net/forum/

### Key Features

- **Enhanced ALFF/fALFF:** Improved amplitude metrics with multiple frequency bands
- **Dynamic Functional Connectivity:** Sliding window analysis of temporal FC variability
- **ReHo Extensions:** Regional homogeneity with flexible neighborhood sizes
- **Degree Centrality:** Weighted/unweighted graph metrics for hub identification
- **VMHC:** Voxel-mirrored homotopic connectivity for interhemispheric analysis
- **Preprocessing Integration:** Seamless connection with DPARSF pipeline
- **Global Signal Options:** Multiple strategies (GSR, non-GSR, aCompCor)
- **Seed-Based Analysis:** ROI-to-whole-brain functional connectivity
- **Quality Control:** Automated QC metrics and visualization
- **Batch Processing:** Script-based analysis for large datasets
- **GUI and CLI:** Both interactive and programmatic interfaces
- **Statistical Analysis:** Integration with SPM for group-level statistics

### Applications

- Resting-state biomarker discovery for brain disorders
- Clinical neuroimaging (Alzheimer's, schizophrenia, autism, depression)
- Dynamic functional connectivity analysis
- Brain network hub identification
- Interhemispheric connectivity studies
- Lifespan development research
- Treatment response prediction

### Citation

```bibtex
@article{YanZang2016DPARSF,
  title={DPARSF: A MATLAB Toolbox for "Pipeline" Data Analysis of
         Resting-State fMRI},
  author={Yan, Chao-Gan and Zang, Yu-Feng},
  journal={Frontiers in Systems Neuroscience},
  volume={4},
  pages={13},
  year={2010}
}

@article{Zang2007ReHo,
  title={Regional homogeneity approach to fMRI data analysis},
  author={Zang, Yufeng and Jiang, Tianzi and Lu, Yingli and He, Yong and Tian, Lixia},
  journal={NeuroImage},
  volume={22},
  pages={394--400},
  year={2004}
}

@article{Zuo2010ALFF,
  title={The oscillating brain: complex and reliable},
  author={Zuo, Xi-Nian and Di Martino, Adriana and Kelly, Clare and others},
  journal={NeuroImage},
  volume={49},
  pages={1432--1445},
  year={2010}
}
```

---

## Installation

### MATLAB Version Installation

RESTplus requires MATLAB (R2014a or later) and SPM12:

```matlab
% 1. Download RESTplus/DPABI from http://rfmri.org/dpabi
% 2. Extract to desired location
% 3. Add to MATLAB path

addpath('/path/to/DPABI_V6.1_220101');
addpath('/path/to/DPABI_V6.1_220101/DPARSF');
addpath(genpath('/path/to/spm12'));

% Save path for future sessions
savepath;

% Launch RESTplus GUI
DPABI;
```

### Standalone Version (No MATLAB Required)

```bash
# Download standalone version for your OS
# Windows: DPABI_StandAlone_V6.1_220101_Win64.exe
# Linux: DPABI_StandAlone_V6.1_220101_Linux.run
# macOS: DPABI_StandAlone_V6.1_220101_Mac.app

# Run installer (will install MATLAB Runtime automatically)
./DPABI_StandAlone_V6.1_220101_Linux.run

# Launch standalone version
./DPABI_StandAlone/run_DPABI.sh
```

### Dependencies

- **SPM12:** http://www.fil.ion.ucl.ac.uk/spm/software/spm12/
- **MATLAB Runtime:** Required for standalone version (auto-installed)
- **Sufficient RAM:** 8GB minimum, 16GB+ recommended for large datasets

### Testing Installation

```matlab
% Launch DPABI GUI
DPABI;

% If GUI opens successfully, installation is complete
% Check version
which DPABI
% Output: /path/to/DPABI_V6.1_220101/DPABI.m
```

---

## Preprocessing Integration

RESTplus integrates with DPARSF for preprocessing resting-state fMRI data.

### DPARSF Preprocessing Pipeline

```matlab
% Launch DPARSF for preprocessing
DPARSF;

% Configure preprocessing steps:
% 1. DICOM to NIfTI conversion
% 2. Remove first timepoints (dummy scans)
% 3. Slice timing correction
% 4. Realignment (motion correction)
% 5. Coregistration to anatomical
% 6. Segmentation
% 7. Normalization to MNI space
% 8. Smoothing
% 9. Nuisance regression
% 10. Filtering (0.01-0.08 Hz default)
```

### Nuisance Regression Strategies

```matlab
% Configure nuisance regression in DPARSF
% Options:
% - 6 motion parameters
% - 24 motion parameters (Friston-24)
% - Global signal regression (GSR)
% - White matter signal
% - CSF signal
% - CompCor (aCompCor/tCompCor)
% - Scrubbing (motion censoring)

% Example: Configure Friston-24 + WM + CSF (no GSR)
preprocessing_config = struct();
preprocessing_config.Covremove.PolynomialTrend = 1;  % linear detrend
preprocessing_config.Covremove.HeadMotion = 2;  % Friston-24
preprocessing_config.Covremove.WM = 1;  % white matter signal
preprocessing_config.Covremove.CSF = 1;  % CSF signal
preprocessing_config.Covremove.Global = 0;  % no global signal
```

### Global Signal Regression Options

```matlab
% Option 1: With GSR (controversial but reduces distance-dependent artifacts)
config_GSR = struct();
config_GSR.Covremove.Global = 1;

% Option 2: Without GSR (preserves absolute FC values)
config_noGSR = struct();
config_noGSR.Covremove.Global = 0;

% Option 3: aCompCor (GSR alternative using ventricles/WM)
config_CompCor = struct();
config_CompCor.Covremove.CompCor = 1;
config_CompCor.Covremove.CompCor_num = 5;  % number of components
```

---

## ALFF/fALFF Analysis

Amplitude of Low-Frequency Fluctuation (ALFF) measures spontaneous neural activity.

### Basic ALFF Computation

```matlab
% Launch DPABI
DPABI;

% Navigate to: Analysis > Standardized Processing > ALFF/fALFF

% Configure ALFF parameters
ALFF_config = struct();
ALFF_config.DataDir = '/path/to/preprocessed/FunImgARWSDCF';  % filtered data
ALFF_config.OutputDir = '/path/to/output/ALFF';
ALFF_config.TR = 2.0;  % repetition time
ALFF_config.Band = [0.01, 0.08];  % frequency band (Hz)
ALFF_config.MaskFile = '';  % empty = auto brain mask

% Compute ALFF
y_alff_falff(ALFF_config.DataDir, ALFF_config.OutputDir, ...
             ALFF_config.TR, ALFF_config.Band, ALFF_config.MaskFile);
```

### Fractional ALFF (fALFF)

Ratio of low-frequency power to total power (more specific than ALFF):

```matlab
% fALFF computed automatically with ALFF
% Output files:
% - mALFF_*.nii (ALFF map)
% - mfALFF_*.nii (fALFF map, recommended for group analysis)

% Load and visualize fALFF
falff_img = 'mfALFF_sub-01.nii';
Header = y_Read(falff_img);
y_Call_bet(falff_img);  % view in BET viewer
```

### Multiple Frequency Bands

```matlab
% Analyze different frequency bands
bands = {[0.01, 0.027], 'slow5';   % slow-5 band
         [0.027, 0.073], 'slow4';   % slow-4 band
         [0.01, 0.08], 'typical'};  % typical band

for i = 1:size(bands, 1)
    output_dir = sprintf('/path/to/output/ALFF_%s', bands{i,2});
    y_alff_falff(data_dir, output_dir, 2.0, bands{i,1}, '');
end
```

### Z-Score Standardization

```matlab
% Standardize ALFF maps for group comparison
input_files = dir('/path/to/output/ALFF/mfALFF_*.nii');
output_dir = '/path/to/output/ALFF_Z';

for i = 1:length(input_files)
    input_path = fullfile(input_files(i).folder, input_files(i).name);
    output_path = fullfile(output_dir, ['z', input_files(i).name]);

    % Z-score normalization (subtract mean, divide by std)
    y_Standardize_fALFF(input_path, output_path);
end
```

---

## Regional Homogeneity (ReHo)

ReHo measures local synchronization of BOLD fluctuations using Kendall's coefficient.

### Basic ReHo Computation

```matlab
% Launch DPABI > Analysis > ReHo

% Configure ReHo parameters
ReHo_config = struct();
ReHo_config.DataDir = '/path/to/preprocessed/FunImgARWSDCF';
ReHo_config.OutputDir = '/path/to/output/ReHo';
ReHo_config.ClusterSize = 27;  % 27-voxel neighborhood (3x3x3 cube)
ReHo_config.MaskFile = '';  % auto brain mask

% Compute ReHo
y_ReHo(ReHo_config.DataDir, ReHo_config.OutputDir, ...
       ReHo_config.ClusterSize, ReHo_config.MaskFile);
```

### Neighborhood Size Comparison

```matlab
% Test different neighborhood sizes
neighborhoods = [7, 19, 27];  % 7=face, 19=edge, 27=vertex

for cluster_size = neighborhoods
    output_dir = sprintf('/path/to/output/ReHo_%d', cluster_size);
    y_ReHo(data_dir, output_dir, cluster_size, '');
end

% Compare results
% 7-voxel: most localized
% 27-voxel: standard, balances locality and reliability
```

### Z-Score Standardization

```matlab
% Standardize ReHo for group analysis
reho_files = dir('/path/to/output/ReHo/ReHo_*.nii');

for i = 1:length(reho_files)
    input_path = fullfile(reho_files(i).folder, reho_files(i).name);
    output_path = fullfile('/path/to/output/ReHo_Z', ['z', reho_files(i).name]);

    % Z-score within brain mask
    y_Standardize_ReHo(input_path, output_path);
end
```

---

## Degree Centrality

Degree centrality identifies brain network hubs based on whole-brain connectivity.

### Weighted Degree Centrality

```matlab
% Launch DPABI > Analysis > Degree Centrality

% Configure DC parameters
DC_config = struct();
DC_config.DataDir = '/path/to/preprocessed/FunImgARWSDCF';
DC_config.OutputDir = '/path/to/output/DC';
DC_config.rThreshold = 0.25;  % correlation threshold
DC_config.MaskFile = '';  % brain mask

% Compute weighted DC (sum of correlation weights)
y_DegreeCentrality_Bilateral(DC_config.DataDir, DC_config.OutputDir, ...
                              DC_config.rThreshold, 'weighted', DC_config.MaskFile);
```

### Binary Degree Centrality

```matlab
% Binary DC: count of connections above threshold
y_DegreeCentrality_Bilateral(data_dir, output_dir, 0.25, 'binary', '');

% Output: mDegreeCentrality_*.nii (DC map)
```

### Correlation Threshold Selection

```matlab
% Test multiple thresholds
thresholds = [0.2, 0.25, 0.3];

for r_thresh = thresholds
    output_dir = sprintf('/path/to/output/DC_r%.2f', r_thresh);
    y_DegreeCentrality_Bilateral(data_dir, output_dir, r_thresh, 'weighted', '');
end

% Lower threshold: more connections, less specific
% Higher threshold: fewer strong connections, more specific hubs
```

### Hub Identification

```matlab
% Identify top hub regions
dc_img = 'mDegreeCentrality_PositiveBinarizedSumBrain_sub-01.nii';
[Data, Header] = y_Read(dc_img);

% Find top 5% voxels
threshold_95 = prctile(Data(:), 95);
hub_mask = Data > threshold_95;

% Visualize hubs
y_Write(hub_mask, Header, 'Hub_Mask.nii');
```

---

## Voxel-Mirrored Homotopic Connectivity (VMHC)

VMHC measures functional connectivity between symmetric brain regions across hemispheres.

### VMHC Computation

```matlab
% VMHC requires symmetric template and preprocessing

% Launch DPABI > Utilities > VMHC

% Configure VMHC
VMHC_config = struct();
VMHC_config.DataDir = '/path/to/preprocessed/FunImgARWSDCF_symmetric';
VMHC_config.OutputDir = '/path/to/output/VMHC';
VMHC_config.MaskFile = '';  % symmetric brain mask

% Compute VMHC (correlation between mirrored voxels)
y_VMHC(VMHC_config.DataDir, VMHC_config.OutputDir, VMHC_config.MaskFile);
```

### Symmetric Template Preprocessing

```matlab
% Prepare data in symmetric template space
% Use DPARSF with symmetric template option:
% Template: ch2bet_symmetric.nii (included in DPABI)

% In DPARSF GUI:
% Normalize > Templates > Starting with 'ch2bet_symmetric.nii'
% This ensures left-right anatomical symmetry
```

### Fisher Z-Transformation

```matlab
% Transform VMHC correlation to z-scores
vmhc_files = dir('/path/to/output/VMHC/VMHC_*.nii');

for i = 1:length(vmhc_files)
    input_path = fullfile(vmhc_files(i).folder, vmhc_files(i).name);
    [Data, Header] = y_Read(input_path);

    % Fisher's r-to-z transformation
    Data_z = 0.5 * log((1 + Data) ./ (1 - Data));

    output_path = fullfile('/path/to/output/VMHC_Z', ['z', vmhc_files(i).name]);
    y_Write(Data_z, Header, output_path);
end
```

---

## Dynamic Functional Connectivity

Analyze time-varying functional connectivity with sliding window approach.

### Sliding Window Configuration

```matlab
% Launch DPABI > Analysis > Dynamic FC

% Configure sliding window parameters
DynFC_config = struct();
DynFC_config.DataDir = '/path/to/preprocessed/FunImgARWSDCF';
DynFC_config.OutputDir = '/path/to/output/DynamicFC';
DynFC_config.WindowLength = 50;  % 50 TRs (100s at TR=2s)
DynFC_config.WindowStep = 1;  % 1 TR step (2s)
DynFC_config.WindowType = 'hamming';  % hamming or rectangular
DynFC_config.ROIFile = 'PowerROIs.nii';  % seed ROIs

% Compute dynamic FC
y_DynamicFC(DynFC_config);
```

### Window Length Selection

```matlab
% Test different window lengths
window_lengths = [30, 50, 100];  % TRs

for win_len = window_lengths
    config = DynFC_config;
    config.WindowLength = win_len;
    config.OutputDir = sprintf('/path/to/output/DynamicFC_win%d', win_len);
    y_DynamicFC(config);
end

% Shorter window: better temporal resolution, less stable
% Longer window: more stable, poorer temporal resolution
```

### Temporal Variability Measures

```matlab
% Compute standard deviation of FC across time
dynfc_timeseries = load('DynamicFC_timeseries.mat');
fc_windows = dynfc_timeseries.FC;  % [n_windows x n_connections]

% Variability (SD across windows)
fc_variability = std(fc_windows, [], 1);

% Coefficient of variation
fc_mean = mean(fc_windows, 1);
fc_cv = fc_variability ./ fc_mean;
```

### State-Based Analysis

```matlab
% Cluster FC patterns into discrete states (k-means)
n_states = 4;
[state_labels, centroids] = kmeans(fc_windows, n_states, 'Replicates', 100);

% Compute state metrics
for s = 1:n_states
    state_duration(s) = sum(state_labels == s);
    state_frequency(s) = state_duration(s) / length(state_labels);
end

fprintf('State frequencies: %s\n', mat2str(state_frequency, 3));
```

---

## Seed-Based Functional Connectivity

Compute whole-brain connectivity from seed regions of interest.

### Single Seed Analysis

```matlab
% Define seed ROI (e.g., PCC sphere)
seed_center = [-5, -49, 40];  % MNI coordinates
seed_radius = 6;  % mm

% Create seed mask
seed_mask = y_Sphere(seed_center, seed_radius, Header);
y_Write(seed_mask, Header, 'PCC_seed.nii');

% Compute seed-based FC
FC_map = y_SeedBasedFC('/path/to/preprocessed/sub-01.nii', 'PCC_seed.nii');
y_Write(FC_map, Header, 'FC_PCC_sub-01.nii');
```

### Multiple Seeds

```matlab
% Load atlas with multiple ROIs
roi_atlas = y_Read('PowerROIs_264.nii');
n_rois = max(roi_atlas(:));

% Compute FC for all ROIs
for roi_id = 1:n_rois
    roi_mask = (roi_atlas == roi_id);
    fc_map = y_SeedBasedFC(data_file, roi_mask);

    output_file = sprintf('FC_ROI%03d.nii', roi_id);
    y_Write(fc_map, Header, output_file);
end
```

### Fisher Z-Transformation

```matlab
% Transform correlation maps to z-scores
fc_img = 'FC_PCC_sub-01.nii';
[FC_r, Header] = y_Read(fc_img);

% Fisher's r-to-z
FC_z = 0.5 * log((1 + FC_r) ./ (1 - FC_r));
FC_z(isinf(FC_z)) = 0;  % handle perfect correlation (seed itself)

y_Write(FC_z, Header, 'FC_PCC_sub-01_Z.nii');
```

---

## Batch Processing

Process multiple subjects efficiently with scripting.

### Batch ALFF/fALFF

```matlab
% Configure batch processing
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'};
base_dir = '/path/to/data';

for i = 1:length(subjects)
    sub_id = subjects{i};
    data_dir = fullfile(base_dir, sub_id, 'FunImgARWSDCF');
    output_dir = fullfile(base_dir, 'Results', 'ALFF', sub_id);

    % Compute ALFF/fALFF
    y_alff_falff(data_dir, output_dir, 2.0, [0.01, 0.08], '');

    fprintf('Completed: %s\n', sub_id);
end
```

### Batch Multiple Metrics

```matlab
% Compute ALFF, ReHo, and DC for all subjects
metrics = {'ALFF', 'ReHo', 'DC'};
subjects = dir('/path/to/data/sub-*');

parfor i = 1:length(subjects)  % parallel processing
    sub_id = subjects(i).name;
    data_dir = fullfile(subjects(i).folder, sub_id, 'FunImgARWSDCF');

    % ALFF
    y_alff_falff(data_dir, fullfile('Results', 'ALFF', sub_id), 2.0, [0.01, 0.08], '');

    % ReHo
    y_ReHo(data_dir, fullfile('Results', 'ReHo', sub_id), 27, '');

    % Degree Centrality
    y_DegreeCentrality_Bilateral(data_dir, fullfile('Results', 'DC', sub_id), 0.25, 'weighted', '');
end
```

---

## Statistical Analysis

### Group-Level Comparison with SPM

```matlab
% Prepare data for SPM group analysis
% 1. Collect all subject maps
group1_files = dir('/path/to/Results/ALFF/group1/zmfALFF_*.nii');
group2_files = dir('/path/to/Results/ALFF/group2/zmfALFF_*.nii');

% 2. Launch SPM
spm fmri;

% 3. Configure two-sample t-test in SPM GUI:
% Specify 2nd-level > Two-sample t-test
% Group 1 scans: select group1 zmfALFF files
% Group 2 scans: select group2 zmfALFF files

% 4. Estimate model and view results
% Results > Select SPM.mat > Define contrasts > View results
```

### Multiple Comparison Correction

```matlab
% Load SPM results
spm_results_file = 'spm_results.mat';

% FWE correction (family-wise error)
% In SPM Results GUI:
% - Height threshold: FWE, p < 0.05
% - Extent threshold: 0 voxels

% FDR correction (false discovery rate)
% - Height threshold: FDR, p < 0.05
% - Extent threshold: 0 voxels

% Cluster-level correction
% - Height threshold: uncorrected p < 0.001
% - Extent threshold: cluster p(FWE-corr) < 0.05
```

---

## Troubleshooting

### Memory Issues with Large Datasets

**Problem:** Out of memory errors during degree centrality computation

**Solution:** Process in smaller chunks or reduce resolution
```matlab
% Use lower resolution mask
mask_file = 'BrainMask_05_91x109x91.nii';  % 2mm resolution

% Or process hemispheres separately
y_DegreeCentrality_Bilateral(data_dir, output_dir, 0.25, 'weighted', mask_file);
```

### Parameter Selection

**Problem:** Uncertain which parameters to use

**Solution:** Follow published recommendations
- ALFF band: 0.01-0.08 Hz (typical)
- ReHo neighborhood: 27 voxels (standard)
- DC threshold: 0.20-0.30 (test multiple)
- Sliding window: 30-60 TRs (balance stability/resolution)

### Result Interpretation

**Problem:** Understanding what each metric means

**Solution:**
- **ALFF:** Regional activity intensity (higher = more active)
- **fALFF:** Specificity of low-frequency activity (more specific than ALFF)
- **ReHo:** Local functional homogeneity (higher = more synchronized neighbors)
- **DC:** Network hub strength (higher = more connections to rest of brain)
- **VMHC:** Interhemispheric coordination (higher = more bilateral symmetry)

### Common Errors

**Error:** "Unable to find brain mask"

**Solution:** Specify mask explicitly or check preprocessing output
```matlab
mask_file = '/path/to/spm12/tpm/mask_ICV.nii';
```

---

## Best Practices

### Metric Selection

- **ALFF/fALFF:** Spontaneous activity amplitude (clinical biomarkers)
- **ReHo:** Local connectivity (early marker in neurodegenerative disease)
- **DC:** Network hubs (connectivity disruption in disorders)
- **VMHC:** Interhemispheric connectivity (asymmetry in lateralized disorders)
- **Dynamic FC:** Temporal variability (cognitive flexibility, aging)

### Preprocessing Choices

- **GSR Decision:** Consider research question (absolute vs. relative FC)
- **Scrubbing:** Essential for high-motion datasets (pediatric, clinical)
- **Filtering:** 0.01-0.08 Hz standard, adjust for specific hypotheses
- **Smoothing:** 4-6mm FWHM typical (balance sensitivity and specificity)

### Statistical Considerations

- **Sample Size:** N≥30 per group for adequate power
- **Covariates:** Control for age, sex, motion (FD), brain volume
- **Multiple Comparisons:** Always correct (FWE or FDR)
- **Effect Sizes:** Report Cohen's d, not just p-values

### Quality Control

- Check motion (exclude FD > 0.5mm or >20% censored volumes)
- Inspect preprocessing outputs visually
- Verify spatial normalization quality
- Check for outliers in metric distributions

---

## References

1. **RESTplus/DPABI:**
   - Yan & Zang (2010). DPARSF: A MATLAB toolbox for pipeline data analysis of resting-state fMRI. *Front Syst Neurosci*, 4:13.
   - Yan et al. (2016). DPABI: Data Processing & Analysis for Brain Imaging. *Neuroinformatics*, 14:339-351.

2. **ALFF/fALFF:**
   - Zang et al. (2007). Altered baseline brain activity in children with ADHD. *Brain Dev*, 29:83-91.
   - Zou et al. (2008). An improved approach to detection of amplitude of low-frequency fluctuation (ALFF). *J Neurosci Methods*, 172:137-141.

3. **ReHo:**
   - Zang et al. (2004). Regional homogeneity approach to fMRI data analysis. *NeuroImage*, 22:394-400.

4. **Degree Centrality:**
   - Buckner et al. (2009). Cortical hubs revealed by intrinsic functional connectivity. *J Neurosci*, 29:1860-1873.

5. **VMHC:**
   - Zuo et al. (2010). Growing together and growing apart: Regional and sex differences in lifespan developmental trajectories of functional homotopy. *J Neurosci*, 30:15034-15043.

**Official Resources:**
- Forum: http://restfmri.net/forum/
- Manual: http://rfmri.org/content/dpabi-manual
- Tutorials: http://restfmri.net/forum/tutorials
