# CONN Toolbox

## Overview

CONN is a comprehensive MATLAB/SPM-based cross-platform software for functional connectivity analysis in fMRI data. It provides a user-friendly GUI and scripting capabilities for seed-based connectivity (SBC), region-of-interest to region-of-interest (ROI-to-ROI), and graph theory analyses, with integrated preprocessing, denoising, first and second-level analyses, and visualization tools.

**Website:** https://www.conn-toolbox.org/
**Platform:** MATLAB (Windows/macOS/Linux) or standalone
**Language:** MATLAB (requires SPM12)
**License:** Free for academic use

## Key Features

- Integrated preprocessing pipeline (SPM-based)
- Seed-based connectivity (SBC) analysis
- ROI-to-ROI connectivity matrices
- Graph theory and network analysis
- Dynamic functional connectivity
- ICA-based network identification
- Advanced denoising (aCompCor, ICA-AROMA)
- Group-level statistics with cluster correction
- Interactive 3D visualization
- Integration with standard atlases
- Batch processing and scripting
- Standalone version available (no MATLAB required)

## Installation

### MATLAB Version

```matlab
% Download CONN from: https://www.conn-toolbox.org/
% Extract to desired location

% Add to MATLAB path
addpath('/path/to/conn');
addpath('/path/to/spm12');

% Launch CONN
conn

% Or initialize without GUI
conn('load', project_file);
```

### Standalone Version

```bash
# Download standalone installer for your OS
# No MATLAB license required
# Includes MATLAB Runtime

# Launch on Linux/macOS
./conn

# Launch on Windows
conn.exe
```

## Project Setup

### Creating a New Project

```matlab
% Launch CONN GUI
conn

% Or create project programmatically
clear BATCH;
BATCH.filename = fullfile(pwd, 'myproject.mat');

% Setup subjects
BATCH.Setup.nsubjects = 10;
BATCH.Setup.RT = 2.0;  % TR in seconds

% Functional data
for nsub = 1:10
    BATCH.Setup.functionals{nsub}{1} = sprintf('/data/sub-%02d/func/sub-%02d_task-rest_bold.nii', nsub, nsub);
end

% Structural data
for nsub = 1:10
    BATCH.Setup.structurals{nsub} = sprintf('/data/sub-%02d/anat/sub-%02d_T1w.nii', nsub, nsub);
end

% ROIs (optional - can use built-in atlases)
BATCH.Setup.rois.names = {'PCC', 'mPFC'};
BATCH.Setup.rois.files{1} = '/atlases/PCC.nii';
BATCH.Setup.rois.files{2} = '/atlases/mPFC.nii';

% Conditions
BATCH.Setup.conditions.names = {'rest'};
for nsub = 1:10
    BATCH.Setup.conditions.onsets{1}{nsub}{1} = 0;
    BATCH.Setup.conditions.durations{1}{nsub}{1} = inf;
end

% Run setup
conn_batch(BATCH);
```

## Preprocessing

### Default Preprocessing Pipeline

```matlab
% CONN includes integrated preprocessing
% Default pipeline includes:
% - Realignment & unwarp
% - Slice-timing correction
% - Outlier detection (ART)
% - Segmentation & normalization
% - Smoothing

BATCH.Setup.preprocessing.steps = {
    'functional_realign&unwarp',
    'functional_center',
    'functional_art',
    'functional_segment&normalize',
    'functional_smooth'
};

BATCH.Setup.preprocessing.fwhm = 6;  % Smoothing kernel

conn_batch(BATCH);
```

### Using Preprocessed Data

```matlab
% If you have data preprocessed with fMRIPrep/SPM
BATCH.Setup.preprocessing = 'none';

% Specify preprocessed files
BATCH.Setup.functionals{1}{1} = '/derivatives/sub-01_space-MNI_desc-preproc_bold.nii';

% Provide confounds
BATCH.Setup.confounds.names = {'realignment', 'scrubbing', 'csf', 'white_matter'};
BATCH.Setup.confounds.files{1} = '/derivatives/sub-01_desc-confounds_timeseries.tsv';
```

## Denoising

### Default Denoising Strategy

```matlab
% Setup denoising
clear BATCH;
BATCH.filename = 'myproject.mat';

% Denoising options
BATCH.Denoising.filter = [0.008, 0.09];  % Band-pass filter (Hz)
BATCH.Denoising.detrending = 1;  % Linear detrending
BATCH.Denoising.despiking = 2;  % Before denoising

% Confound regression
BATCH.Denoising.confounds = {
    'White Matter',
    'CSF',
    'realignment',
    'scrubbing',
    'Effect of rest'
};

% CompCor
BATCH.Denoising.confounds.dimensions = {5, 5, 6, 1, 1};  % PCA dimensions

% Apply denoising
conn_batch(BATCH);
```

### Advanced Denoising (aCompCor)

```matlab
% Anatomical CompCor (aCompCor)
BATCH.Denoising.confounds = {
    'White Matter',  % aCompCor from WM
    'CSF',          % aCompCor from CSF
    'realignment',
    'scrubbing'
};

% Number of PCA components
BATCH.Denoising.confounds.dimensions = {5, 5, 6, 1};

% Optional: Physiological noise modeling
BATCH.Denoising.confounds = [BATCH.Denoising.confounds, 'RETROICOR'];
```

## First-Level Analysis

### Seed-Based Connectivity (SBC)

```matlab
% Define analysis
clear BATCH;
BATCH.filename = 'myproject.mat';

% Seeds
BATCH.Analysis.sources = {'PCC', 'mPFC', 'networks.DefaultMode'};

% Measure
BATCH.Analysis.measure = 1;  % 1=bivariate correlation, 2=semipartial, etc.

% Conditions
BATCH.Analysis.conditions = {'rest'};

% Run first-level
conn_batch(BATCH);

% View results
conn('gui_results');
```

### ROI-to-ROI Analysis

```matlab
% ROI-to-ROI connectivity matrix
BATCH.Analysis.type = 3;  % ROI-to-ROI

% Select ROIs
BATCH.Analysis.sources = conn('get', 'rois');  % All ROIs
% Or specify: {'PCC', 'mPFC', 'V1', 'M1'}

% Measure
BATCH.Analysis.measure = 1;  % Bivariate correlation

% Run analysis
conn_batch(BATCH);

% Export connectivity matrix
[Z, names] = conn_get_results('matrix');
% Z contains Fisher-transformed correlations
```

### Network Analysis (Graph Theory)

```matlab
% Setup graph analysis
BATCH.Analysis.type = 4;  % Graph theory

% Cost threshold (connection density)
BATCH.Analysis.graph_cost = 0.15;  % Keep 15% strongest connections

% Compute metrics
BATCH.Analysis.graph_measure = {
    'global efficiency',
    'local efficiency',
    'clustering coefficient',
    'degree',
    'betweenness centrality'
};

conn_batch(BATCH);
```

## Second-Level Analysis

### Group Statistics

```matlab
% Setup second-level analysis
clear BATCH;
BATCH.filename = 'myproject.mat';

% Define contrasts
BATCH.Results.between_subjects.effect_names = {'AllSubjects'};
BATCH.Results.between_subjects.contrast = [1];  % One-sample t-test

% Or group comparison
% BATCH.Results.between_subjects.effect_names = {'Group'};
% BATCH.Results.between_subjects.contrast = [1 -1];  % Patients vs Controls

% Thresholding
BATCH.Results.p_uncorrected = 0.001;
BATCH.Results.p_corrected = 0.05;
BATCH.Results.cluster_correction = 1;  % FWE cluster-level correction

% Run second-level
conn_batch(BATCH);
```

### Between-Conditions Analysis

```matlab
% Compare connectivity between conditions
BATCH.Results.between_conditions.effect_names = {'task > rest'};
BATCH.Results.between_conditions.contrast = [1 -1];

% Setup sources
BATCH.Results.seedbased.sources = {'PCC'};

% Run analysis
conn_batch(BATCH);
```

## Dynamic Functional Connectivity

### Sliding Window Analysis

```matlab
% Dynamic FC setup
BATCH.Analysis.type = 1;  % Seed-based
BATCH.Analysis.measure = 1;

% Window settings
BATCH.Analysis.modulation.nwindows = 20;  % Number of windows
BATCH.Analysis.modulation.window_length = 30;  % Seconds
BATCH.Analysis.modulation.window_overlap = 0.5;  % 50% overlap

% Run dynamic FC
conn_batch(BATCH);

% Analyze variability
conn_batch('Results.within_condition_variability', 1);
```

## Visualization

### Interactive Results Viewer

```matlab
% Launch results GUI
conn display

% Or programmatic visualization
conn('gui_results');

% Plot seed-based connectivity map
conn_mesh_display('', 'PCC', 'rest', 'AllSubjects');

% Plot connectome (ROI-to-ROI)
conn_mesh_display_connectome('', 'rest', 'AllSubjects');

% Export images
conn_print('connectivity_map.png');
```

### Connectivity Matrices

```matlab
% Plot connectivity matrix
conn_matrix_display('', 'rest', 'AllSubjects');

% Customize
conn_matrix_display('', 'rest', 'AllSubjects', ...
    'threshold', 0.3, ...
    'colormap', 'jet', ...
    'sort', 'networks');
```

### Glass Brain Visualization

```matlab
% 3D glass brain
conn_mesh_display('');

% Add ROIs
conn_mesh_display_addroi('PCC');
conn_mesh_display_addroi('mPFC');

% Add connections
conn_mesh_display_addconnection('PCC', 'mPFC', 0.5);
```

## Scripting Complete Analysis

```matlab
%% Complete CONN workflow script

% 1. Setup project
clear BATCH;
BATCH.filename = 'connectivity_study.mat';
BATCH.Setup.nsubjects = 20;
BATCH.Setup.RT = 2.0;

% Add data
for n = 1:20
    BATCH.Setup.functionals{n}{1} = sprintf('/data/sub-%02d_bold.nii', n);
    BATCH.Setup.structurals{n} = sprintf('/data/sub-%02d_T1w.nii', n);
end

% Conditions
BATCH.Setup.conditions.names = {'rest'};
for n = 1:20
    BATCH.Setup.conditions.onsets{1}{n}{1} = 0;
    BATCH.Setup.conditions.durations{1}{n}{1} = inf;
end

% Run setup
conn_batch(BATCH);

% 2. Preprocessing
BATCH.Setup.preprocessing.steps = 'default';
BATCH.Setup.preprocessing.fwhm = 6;
conn_batch(BATCH);

% 3. Denoising
BATCH.Denoising.filter = [0.008, 0.09];
BATCH.Denoising.confounds = {'White Matter', 'CSF', 'realignment', 'scrubbing'};
BATCH.Denoising.confounds.dimensions = {5, 5, 6, 1};
conn_batch(BATCH);

% 4. First-level analysis
BATCH.Analysis.type = 1;  % Seed-based
BATCH.Analysis.sources = {'networks.DefaultMode.PCC'};
BATCH.Analysis.measure = 1;
conn_batch(BATCH);

% 5. Second-level statistics
BATCH.Results.between_subjects.effect_names = {'AllSubjects'};
BATCH.Results.between_subjects.contrast = [1];
BATCH.Results.p_uncorrected = 0.001;
BATCH.Results.cluster_correction = 1;
conn_batch(BATCH);

% 6. View results
conn display
```

## Working with Atlases

### Built-in Atlases

```matlab
% CONN includes several atlases:
% - networks (ICN templates)
% - atlas (anatomical atlas)
% - Harvard-Oxford
% - AAL
% - Schaefer 2018

% Use ICN network ROIs
BATCH.Setup.rois.names = {
    'networks.DefaultMode',
    'networks.SensoriMotor',
    'networks.Visual',
    'networks.SalienceVentAttn',
    'networks.DorsalAttn',
    'networks.FrontoParietal'
};

% Access all network ROIs individually
BATCH.Analysis.sources = conn('get', 'rois.networks.DefaultMode.*');
```

### Custom ROIs

```matlab
% Add custom ROI
BATCH.Setup.rois.names{end+1} = 'MyROI';
BATCH.Setup.rois.files{end+1} = '/path/to/roi_mask.nii';

% Or create sphere ROI
BATCH.Setup.rois.names{end+1} = 'PCC_sphere';
BATCH.Setup.rois.dimensions{end+1} = 1;  % Sphere
BATCH.Setup.rois.files{end+1} = [0 -52 26; 6];  % [x y z; radius_mm]
```

## Quality Control

### Preprocessing QC

```matlab
% View QC plots
conn('gui_qa');

% Export QC measures
qa_data = conn_qaplots('subjects', 1:20, 'display', 'on');

% Check motion
motion_data = conn_get_confound('realignment', 1:20);

% Identify high-motion subjects
fd = conn_get_confound('scrubbing.FD', 1:20);
high_motion = find(mean(fd > 0.5, 2) > 0.2);  % >20% volumes with FD>0.5mm
fprintf('High-motion subjects: %s\n', mat2str(high_motion));
```

### Connectivity QC

```matlab
% Check for outlier subjects
[h, p, stats] = conn_qaplots('connectome');

% Distribution of connectivity values
conn_qaplots('histogram');
```

## Exporting Results

```matlab
% Export connectivity matrices to CSV
conn_batch('Results.savedirectory', '/output/matrices');
conn_batch('Results.saveas', 'csv');

% Export NIfTI maps
conn_batch('Results.saveas', 'nifti');

% Export to workspace
results = conn_module('results');
roi_names = results.names;
connectivity_matrix = results.Z;

% Save for external analysis
save('connectivity_results.mat', 'roi_names', 'connectivity_matrix');
```

## Integration with Claude Code

When helping users with CONN:

1. **Check Installation:**
   ```matlab
   which conn
   conn version
   which spm  % CONN requires SPM12
   ```

2. **Common Issues:**
   - SPM12 not in MATLAB path
   - Incorrect file paths in batch
   - Memory issues with large datasets
   - Missing confound files

3. **Best Practices:**
   - Always run QC before analysis
   - Use aCompCor for denoising
   - Apply appropriate band-pass filtering
   - Check for high-motion subjects
   - Use cluster correction for multiple comparisons
   - Save batch scripts for reproducibility

4. **Performance Tips:**
   - Use parallel processing (`conn_parallel`)
   - Process in batches for large studies
   - Reduce spatial resolution if needed
   - Use subset of ROIs for initial analyses

## Troubleshooting

**Problem:** "SPM not found"
**Solution:** Ensure SPM12 is installed and in MATLAB path: `addpath('/path/to/spm12')`

**Problem:** Out of memory errors
**Solution:** Process fewer subjects simultaneously, reduce smoothing kernel, or increase MATLAB memory

**Problem:** High motion artifacts
**Solution:** Use scrubbing, increase FD threshold, or exclude high-motion subjects

**Problem:** No significant connectivity
**Solution:** Check denoising strategy, verify ROI locations, adjust thresholds

## Resources

- Website: https://www.conn-toolbox.org/
- Manual: https://web.conn-toolbox.org/resources/manual
- Tutorials: https://www.conn-toolbox.org/resources/tutorials
- Forum: https://www.nitrc.org/forum/?group_id=279
- Publications: https://www.conn-toolbox.org/resources/publications
- YouTube: CONN Tutorial videos

## Citation

```bibtex
@article{whitfield2012conn,
  title={Conn: a functional connectivity toolbox for correlated and anticorrelated brain networks},
  author={Whitfield-Gabrieli, Susan and Nieto-Castanon, Alfonso},
  journal={Brain connectivity},
  volume={2},
  number={3},
  pages={125--141},
  year={2012},
  publisher={Mary Ann Liebert, Inc.}
}
```

## Related Tools

- **SPM12:** Required for CONN preprocessing
- **DPABI:** Alternative MATLAB connectivity toolbox
- **REST:** Resting-state fMRI analysis
- **Nilearn:** Python alternative for connectivity
- **Brain Connectivity Toolbox:** Graph theory analysis
