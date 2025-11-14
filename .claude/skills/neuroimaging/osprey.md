# Osprey

## Overview

Osprey is an all-in-one software suite for state-of-the-art processing and quantitative analysis of in-vivo magnetic resonance spectroscopy (MRS) data. It provides a comprehensive pipeline from raw data loading through preprocessing, modeling, and quantification, with support for multiple MRS sequences and a user-friendly graphical interface.

**Website:** https://github.com/schorschinho/osprey
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** BSD 3-Clause License

## Key Features

- End-to-end MRS processing pipeline
- Support for multiple vendors (Siemens, Philips, GE)
- Wide range of sequence types (PRESS, MEGA-PRESS, HERMES, etc.)
- Automated quality control and data visualization
- Multiple quantification methods (water reference, tissue correction)
- LCModel-style linear combination modeling
- Statistical analysis tools
- Publication-ready figures
- Batch processing capabilities
- Integration with BIDS MRS format
- Real-time quality assessment during processing

## Installation

### Requirements

```matlab
% MATLAB R2017a or later
% No additional toolboxes required
% Optional: Optimization Toolbox for improved fitting
```

### Download and Setup

```bash
# Clone from GitHub
git clone https://github.com/schorschinho/osprey.git
cd osprey

# Or download latest release
# https://github.com/schorschinho/osprey/releases
```

```matlab
% Add Osprey to MATLAB path
addpath(genpath('/path/to/osprey'));
savepath;

% Verify installation
which osp_onboard
% Should return: /path/to/osprey/GUI/osp_onboard.m

% Launch Osprey
Osprey
```

### Test Installation

```matlab
% Run example dataset
cd /path/to/osprey/exampledata

% Load job file
load('jobFile_sdat.mat');

% Process example data
MRSCont = OspreyJob(jobFile);
MRSCont = OspreyLoad(MRSCont);
MRSCont = OspreyProcess(MRSCont);
MRSCont = OspreyFit(MRSCont);
MRSCont = OspreyQuantify(MRSCont);
```

## Data Organization

### BIDS MRS Format (Recommended)

```bash
# BIDS-compliant directory structure
study/
├── dataset_description.json
├── participants.tsv
├── sub-01/
│   └── ses-01/
│       └── mrs/
│           ├── sub-01_ses-01_svs.nii.gz
│           └── sub-01_ses-01_svs.json
├── sub-02/
│   └── ses-01/
│       └── mrs/
│           ├── sub-02_ses-01_svs.nii.gz
│           └── sub-02_ses-01_svs.json
└── derivatives/
```

### Vendor-Specific Formats

```bash
# Osprey supports:
# Siemens: .rda, .dat (TWIX)
# Philips: .sdat/.spar, .data/.list
# GE: .7 (P-files)
# Bruker: ser/fid files
# NIfTI-MRS: .nii/.nii.gz with JSON sidecar
```

## Basic Workflow (GUI)

### Step 1: Launch Osprey

```matlab
% Start Osprey GUI
Osprey

% The GUI provides tabs for:
% - Load: Data loading and QC
% - Process: Preprocessing steps
% - Fit: Spectral modeling
% - Quantify: Metabolite quantification
% - Overview: Results visualization
```

### Step 2: Create Job File

```matlab
% Click "New Job" button
% Specify:
% 1. Data files (metabolite spectra)
% 2. Reference files (water or short-TE)
% 3. Structural images (for tissue segmentation)
% 4. Sequence type (PRESS, MEGA-PRESS, etc.)
% 5. Output directory
% 6. Analysis options
```

### Step 3: Load Data

```matlab
% Click "Load" button
% Osprey will:
% - Load raw data
% - Parse headers
% - Combine coil channels
% - Average repetitions
% - Create quality control figures
```

### Step 4: Process Data

```matlab
% Click "Process" button
% Preprocessing includes:
% - Frequency drift correction
% - Phase correction
% - Eddy current correction
% - Water removal (for metabolite spectra)
% - Alignment across averages
% - Outlier removal
```

### Step 5: Fit Spectra

```matlab
% Click "Fit" button
% Spectral modeling:
% - Linear combination of basis functions
% - Macromolecule baseline modeling
% - Automatic baseline and lineshape optimization
% - Quality metrics (SNR, linewidth, fit residual)
```

### Step 6: Quantify Metabolites

```matlab
% Click "Quantify" button
% Quantification methods:
% - Water scaling
% - Tissue correction
% - Absolute quantification (institutional units)
% - Concentration estimates
```

## Command-Line Usage

### Create Job File

```matlab
% Initialize job file
jobFile = struct();

% Specify data files
jobFile.files = {
    '/data/sub-01/mrs/sub-01_svs.sdat'
    '/data/sub-02/mrs/sub-02_svs.sdat'
    '/data/sub-03/mrs/sub-03_svs.sdat'
};

% Reference files (water unsuppressed)
jobFile.files_ref = {
    '/data/sub-01/mrs/sub-01_water.sdat'
    '/data/sub-02/mrs/sub-02_water.sdat'
    '/data/sub-03/mrs/sub-03_water.sdat'
};

% Structural images for segmentation
jobFile.files_nii = {
    '/data/sub-01/anat/sub-01_T1w.nii'
    '/data/sub-02/anat/sub-02_T1w.nii'
    '/data/sub-03/anat/sub-03_T1w.nii'
};

% Sequence parameters
jobFile.seqType = 'PRESS';  % or 'MEGA', 'HERMES', 'HERCULES'

% Output directory
jobFile.outputFolder = '/output/osprey_results';

% Save job file
save('/data/jobFile.mat', 'jobFile');
```

### Run Complete Pipeline

```matlab
% Load job file
load('/data/jobFile.mat');

% Run full pipeline
MRSCont = OspreyJob(jobFile);
MRSCont = OspreyLoad(MRSCont);
MRSCont = OspreyProcess(MRSCont);
MRSCont = OspreyFit(MRSCont);
MRSCont = OspreyQuantify(MRSCont);

% Generate output
MRSCont = OspreyOverview(MRSCont);

% Save results
save(fullfile(jobFile.outputFolder, 'MRSCont.mat'), 'MRSCont');
```

### Load Data Only

```matlab
% Just load and inspect data
jobFile = load('/data/jobFile.mat');
MRSCont = OspreyJob(jobFile.jobFile);
MRSCont = OspreyLoad(MRSCont);

% Visualize loaded spectrum
osp_plotLoad(MRSCont, 1);  % Subject 1
```

### Process Data

```matlab
% Continue from loaded data
MRSCont = OspreyProcess(MRSCont);

% View processing results
osp_plotProcess(MRSCont, 1);

% Check quality metrics
fprintf('SNR: %.1f\n', MRSCont.QM.SNR.metab(1));
fprintf('Linewidth: %.1f Hz\n', MRSCont.QM.FWHM.metab(1));
fprintf('Frequency shift: %.2f Hz\n', MRSCont.QM.freqShift.metab(1));
```

### Fit Spectra

```matlab
% Perform spectral fitting
MRSCont = OspreyFit(MRSCont);

% View fit results
osp_plotFit(MRSCont, 1);

% Extract fit quality metrics
fitParams = MRSCont.fit.results.off.fitParams{1};
fprintf('Residual: %.4f\n', fitParams.residual);
fprintf('Baseline FWHM: %.1f Hz\n', fitParams.FWHM);
```

## Advanced Features

### MEGA-PRESS Analysis

```matlab
% Setup for edited spectroscopy (e.g., GABA)
jobFile.seqType = 'MEGA';

% Specify edit-ON and edit-OFF conditions
% Osprey automatically handles:
% - Difference spectrum calculation
% - Edit-ON/OFF alignment
% - Subtraction artifacts

% Run pipeline
MRSCont = OspreyJob(jobFile);
MRSCont = OspreyLoad(MRSCont);
MRSCont = OspreyProcess(MRSCont);
MRSCont = OspreyFit(MRSCont);

% Access edited metabolites
GABA = MRSCont.quantify.amplMets.diff(1, strcmp(MRSCont.quantify.metabs, 'GABA'));
fprintf('GABA+: %.2f i.u.\n', GABA);
```

### HERMES/HERCULES Multi-Editing

```matlab
% Multiplexed editing sequences
jobFile.seqType = 'HERMES';  % or 'HERCULES'

% Process all sub-spectra
MRSCont = OspreyJob(jobFile);
MRSCont = OspreyLoad(MRSCont);
MRSCont = OspreyProcess(MRSCont);
MRSCont = OspreyFit(MRSCont);

% Extract multiple edited metabolites
% HERMES provides: GABA, GSH
% HERCULES provides: GABA, GSH, Lac, PE

metabs = {'GABA', 'GSH'};
for m = 1:length(metabs)
    idx = strcmp(MRSCont.quantify.metabs, metabs{m});
    conc = MRSCont.quantify.amplMets.diff1(1, idx);
    fprintf('%s: %.2f i.u.\n', metabs{m}, conc);
end
```

### Tissue Segmentation and Correction

```matlab
% When structural images are provided
% Osprey automatically:
% 1. Co-registers T1 to MRS voxel
% 2. Segments into GM, WM, CSF
% 3. Calculates tissue fractions
% 4. Applies tissue correction to concentrations

% Access tissue fractions
tissue = MRSCont.seg.tissue;
fprintf('GM: %.1f%%\n', tissue.fGM(1) * 100);
fprintf('WM: %.1f%%\n', tissue.fWM(1) * 100);
fprintf('CSF: %.1f%%\n', tissue.fCSF(1) * 100);

% Tissue-corrected concentrations
NAA_raw = MRSCont.quantify.amplMets.off(1, strcmp(MRSCont.quantify.metabs, 'NAA'));
NAA_corr = MRSCont.quantify.amplMets.tCr_off(1, strcmp(MRSCont.quantify.metabs, 'NAA'));
fprintf('NAA (raw): %.2f\n', NAA_raw);
fprintf('NAA (corrected): %.2f\n', NAA_corr);
```

### Custom Basis Set

```matlab
% Use custom basis set for fitting
jobFile.fit.basisSetFile = '/path/to/custom_basis.mat';

% Basis set should contain:
% - Metabolite spectra
% - Same spectral parameters (TE, field strength, etc.)

% Create custom basis using FID-A
% (See FID-A skill for basis set generation)
```

### Quality Control Filtering

```matlab
% Set QC thresholds
jobFile.QC.SNR = 10;           % Minimum SNR
jobFile.QC.FWHM = 20;          % Maximum linewidth (Hz)
jobFile.QC.freqShift = 10;     % Maximum frequency shift (Hz)

% Run pipeline
MRSCont = OspreyJob(jobFile);
MRSCont = OspreyLoad(MRSCont);
MRSCont = OspreyProcess(MRSCont);

% Check which datasets pass QC
pass_QC = MRSCont.QM.SNR.metab > jobFile.QC.SNR & ...
          MRSCont.QM.FWHM.metab < jobFile.QC.FWHM;

fprintf('%d/%d datasets pass QC\n', sum(pass_QC), length(pass_QC));
```

## Batch Processing

### Process Multiple Subjects

```matlab
% Setup batch job
clear jobFile;

% Get list of all subject files
subjects = dir('/data/sub-*/mrs/*svs.sdat');
n_subjects = length(subjects);

% Create file lists
for s = 1:n_subjects
    jobFile.files{s} = fullfile(subjects(s).folder, subjects(s).name);

    % Corresponding water reference
    water_file = strrep(subjects(s).name, 'svs', 'water');
    jobFile.files_ref{s} = fullfile(subjects(s).folder, water_file);

    % Corresponding T1
    subj_id = regexp(subjects(s).name, 'sub-\d+', 'match', 'once');
    jobFile.files_nii{s} = sprintf('/data/%s/anat/%s_T1w.nii', subj_id, subj_id);
end

% Set common parameters
jobFile.seqType = 'PRESS';
jobFile.outputFolder = '/output/batch_results';

% Run batch processing
MRSCont = OspreyJob(jobFile);
MRSCont = OspreyLoad(MRSCont);
MRSCont = OspreyProcess(MRSCont);
MRSCont = OspreyFit(MRSCont);
MRSCont = OspreyQuantify(MRSCont);
MRSCont = OspreyOverview(MRSCont);

% Save
save(fullfile(jobFile.outputFolder, 'MRSCont_all.mat'), 'MRSCont', '-v7.3');
```

### Export Results to CSV

```matlab
% Extract quantification results
metabs = MRSCont.quantify.metabs;
n_subj = length(MRSCont.files);

% Create results table
results_table = array2table(zeros(n_subj, length(metabs)));
results_table.Properties.VariableNames = metabs;

% Fill with concentrations
for s = 1:n_subj
    results_table(s, :) = array2table(MRSCont.quantify.amplMets.tCr_off(s, :));
end

% Add subject IDs
subject_ids = cell(n_subj, 1);
for s = 1:n_subj
    [~, name, ~] = fileparts(MRSCont.files{s});
    subject_ids{s} = name;
end
results_table = [table(subject_ids, 'VariableNames', {'SubjectID'}), results_table];

% Write to CSV
writetable(results_table, fullfile(jobFile.outputFolder, 'metabolite_concentrations.csv'));
```

## Statistical Analysis

### Group Comparison

```matlab
% Compare two groups
group1_idx = 1:10;    % Control subjects
group2_idx = 11:20;   % Patient subjects

% Extract NAA concentrations
NAA_idx = strcmp(MRSCont.quantify.metabs, 'NAA');
NAA_ctrl = MRSCont.quantify.amplMets.tCr_off(group1_idx, NAA_idx);
NAA_pat = MRSCont.quantify.amplMets.tCr_off(group2_idx, NAA_idx);

% Statistical test
[h, p, ci, stats] = ttest2(NAA_ctrl, NAA_pat);

fprintf('NAA Comparison:\n');
fprintf('Controls: %.2f ± %.2f\n', mean(NAA_ctrl), std(NAA_ctrl));
fprintf('Patients: %.2f ± %.2f\n', mean(NAA_pat), std(NAA_pat));
fprintf('t(%d) = %.2f, p = %.4f\n', stats.df, stats.tstat, p);
```

### Correlation Analysis

```matlab
% Correlate metabolite with clinical variable
clinical_scores = [25, 30, 28, 35, 32, 27, 29, 31, 26, 33, ...
                   40, 38, 42, 37, 45, 39, 41, 36, 43, 44];

Glx_idx = strcmp(MRSCont.quantify.metabs, 'Glx');
Glx_all = MRSCont.quantify.amplMets.tCr_off(:, Glx_idx);

% Pearson correlation
[r, p] = corr(Glx_all, clinical_scores');

fprintf('Glx vs Clinical Score:\n');
fprintf('r = %.3f, p = %.4f\n', r, p);

% Plot
figure;
scatter(Glx_all, clinical_scores, 'filled');
lsline;
xlabel('Glx (tCr)');
ylabel('Clinical Score');
title(sprintf('r = %.3f, p = %.4f', r, p));
```

## Visualization

### Plot Spectrum

```matlab
% Plot processed spectrum
figure;
osp_plotLoad(MRSCont, 1, 'metab');

% Plot fit
figure;
osp_plotFit(MRSCont, 1, 'off');

% Plot individual metabolite contributions
figure;
osp_plotFit(MRSCont, 1, 'off', 'NAA');
```

### Create Overview Figure

```matlab
% Generate comprehensive overview
MRSCont = OspreyOverview(MRSCont);

% Overview includes:
% - Voxel placement on anatomy
% - Processed spectra
% - Fit results
% - Metabolite concentrations
% - Quality metrics
% - Tissue fractions

% Figures saved to output directory
```

### Custom Visualization

```matlab
% Extract spectrum for custom plotting
spec = MRSCont.processed.metab{1};
ppm = MRSCont.processed.ppm;

figure;
plot(ppm, real(spec));
set(gca, 'XDir', 'reverse');
xlim([0.5 4.5]);
xlabel('Chemical Shift (ppm)');
ylabel('Signal Intensity');
title('Processed MRS Spectrum');

% Add metabolite labels
hold on;
text(2.0, max(real(spec))*0.9, 'NAA', 'FontSize', 12);
text(3.0, max(real(spec))*0.8, 'Cr', 'FontSize', 12);
text(3.2, max(real(spec))*0.7, 'Cho', 'FontSize', 12);
```

## Integration with Other Tools

### Export to LCModel Format

```matlab
% Convert to LCModel .RAW format
osp_saveLCM(MRSCont, 1, '/output/subject1.RAW');

% Create LCModel control file
osp_writeLCMControlFile(MRSCont, 1, '/output/subject1.control');
```

### Import from MRSCloud

```matlab
% Osprey can read MRSCloud processed data
jobFile.dataType = 'MRSCloud';
jobFile.files = {'/data/mrscloud_output.mat'};

MRSCont = OspreyJob(jobFile);
MRSCont = OspreyLoad(MRSCont);
```

### Integration with FSL/SPM

```matlab
% After tissue segmentation, use masks with FSL/SPM
voxel_mask = MRSCont.seg.img_mask{1};

% Save as NIfTI for overlay
niftiwrite(voxel_mask, '/output/voxel_mask.nii');

% Use with FSL
system('fslview /data/T1.nii.gz /output/voxel_mask.nii');
```

## Integration with Claude Code

When helping users with Osprey:

1. **Check Installation:**
   ```matlab
   which Osprey
   which osp_onboard
   ```

2. **Verify Data Format:**
   ```matlab
   % Load single file to check
   MRSCont = OspreyJob(jobFile);
   MRSCont = OspreyLoad(MRSCont);
   % Check for errors in loading
   ```

3. **Common Workflows:**
   - Single-voxel PRESS: Standard brain metabolites
   - MEGA-PRESS: GABA quantification
   - HERMES: Multiple edited metabolites
   - Multi-voxel: MRSI data

4. **Quality Checks:**
   - SNR > 10 (preferably > 20)
   - Linewidth < 15 Hz
   - Minimal frequency drift
   - Good water suppression

## Troubleshooting

**Problem:** Osprey GUI doesn't launch
**Solution:** Check MATLAB version (need R2017a+), verify path setup, try `addpath(genpath('/path/to/osprey'))`

**Problem:** "Basis set not found" error
**Solution:** Ensure correct sequence type specified, check basis set path, download appropriate basis from Osprey repository

**Problem:** Poor spectral quality (broad peaks)
**Solution:** Check shimming quality, inspect for motion artifacts, verify TE/sequence parameters

**Problem:** Tissue segmentation fails
**Solution:** Verify T1 image quality, check co-registration, ensure proper NIfTI format

**Problem:** Fit residuals are high
**Solution:** Check for artifacts, adjust baseline model, inspect for lipid contamination, verify basis set matches acquisition

## Best Practices

1. **Data Acquisition:**
   - Acquire water reference for quantification
   - Use consistent voxel placement
   - Optimize shimming (target FWHM < 10 Hz)
   - Collect structural image for segmentation

2. **Processing:**
   - Always inspect raw data before processing
   - Check frequency/phase correction quality
   - Monitor averaging across repetitions
   - Review outlier detection results

3. **Quantification:**
   - Use tissue correction when available
   - Report reference method (water, Cr, etc.)
   - Include quality metrics in results
   - Consider relaxation corrections

4. **Reporting:**
   - Report Osprey version
   - Describe sequence parameters (TE, TR, voxel size)
   - Include QC metrics (SNR, linewidth)
   - Report quantification method
   - Show representative spectra

## Resources

- **GitHub:** https://github.com/schorschinho/osprey
- **Documentation:** https://schorschinho.github.io/osprey/
- **Forum:** https://forum.mrshub.org/c/mrs-software/osprey
- **Tutorials:** https://github.com/schorschinho/osprey/tree/develop/tutorials
- **Basis Sets:** https://github.com/schorschinho/osprey/tree/develop/fit/basissets

## Citation

```bibtex
@article{oeltzschner2020osprey,
  title={Osprey: open-source processing, reconstruction \& estimation of magnetic resonance spectroscopy data},
  author={Oeltzschner, Georg and Zöllner, Helge J and Hui, Steve CN and Mikkelsen, Mark and Saleh, Muhammad G and Tapper, Sofie and Edden, Richard AE},
  journal={Journal of Neuroscience Methods},
  volume={343},
  pages={108827},
  year={2020},
  publisher={Elsevier}
}
```

## Related Tools

- **FID-A:** MATLAB toolkit for MRS simulation and processing
- **TARQUIN:** Automatic MRS quantification
- **jMRUI:** Java-based MRS analysis
- **LCModel:** Commercial gold-standard quantification
- **Gannet:** GABA-specific analysis toolbox
- **INSPECTOR:** Web-based MRS quality control
