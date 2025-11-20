# EEGLAB

## Overview

EEGLAB is a comprehensive MATLAB toolbox for processing continuous and event-related EEG, MEG, and other electrophysiological data. It provides an interactive graphical interface combined with powerful command-line scripting capabilities, making it one of the most widely used tools for EEG analysis.

**Website:** https://sccn.ucsd.edu/eeglab/
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** BSD

## Key Features

- Interactive GUI and command-line interface
- Continuous and epoched data processing
- Independent Component Analysis (ICA/AMICA)
- Time-frequency analysis (ERSPs, ITCs)
- Source localization with plugins
- Event-related potentials (ERPs)
- Connectivity analysis
- Extensive plugin ecosystem
- BIDS-EEG support
- Integration with FieldTrip and MNE

## Installation

### Basic Installation

```matlab
% Download EEGLAB from: https://sccn.ucsd.edu/eeglab/download.php
% Extract to desired location

% Add to MATLAB path
addpath('/path/to/eeglab');
eeglab  % Launch EEGLAB GUI

% Or add permanently
pathtool  % Use MATLAB's path tool
```

### Installing Plugins

```matlab
% From EEGLAB GUI: File > Manage EEGLAB extensions
% Or from command line:
plugin_askinstall('clean_rawdata');  % Clean data plugin
plugin_askinstall('Fieldtrip-lite'); % FieldTrip integration
plugin_askinstall('ICLabel');        % IC classification
plugin_askinstall('dipfit');         % Source localization
plugin_askinstall('SIFT');           % Connectivity
```

## Loading Data

### Import from Different Formats

```matlab
% Launch EEGLAB
eeglab;

% Import raw data file
EEG = pop_biosig('data.bdf');  % BioSemi
EEG = pop_loadbv('path/', 'file.vhdr');  % BrainVision
EEG = pop_loadcnt('data.cnt');  % Neuroscan
EEG = pop_fileio('data.eeg');   % General file IO

% Import from MATLAB matrix
EEG = pop_importdata('dataformat', 'matlab', ...
                     'data', data_matrix, ...
                     'srate', 250, ...
                     'nbchan', 64);

% Set channel locations
EEG = pop_chanedit(EEG, 'lookup', 'standard-10-5-cap385.elp');

% Update EEGLAB dataset
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw;
```

## Preprocessing Pipeline

### Complete Preprocessing Script

```matlab
%% Initialize
clear; close all; clc;
eeglab nogui;  % Run without GUI

%% Load data
EEG = pop_biosig('subject01.bdf');
EEG.setname = 'subject01_raw';

%% Set channel locations
EEG = pop_chanedit(EEG, 'lookup', ...
    [eeglabpath '/plugins/dipfit/standard_BEM/elec/standard_1005.elc']);

%% Filter
% High-pass filter (remove slow drifts)
EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5, 'plotfreqz', 0);

% Low-pass filter (remove high-frequency noise)
EEG = pop_eegfiltnew(EEG, 'hicutoff', 40, 'plotfreqz', 0);

% Notch filter (remove line noise)
EEG = pop_cleanline(EEG, 'linefreqs', [50 100], ...  % 50 Hz (or 60 Hz in US)
                         'bandwidth', 2);

%% Re-reference
% Average reference
EEG = pop_reref(EEG, []);

% Or specific reference
EEG = pop_reref(EEG, [32]);  % Channel 32

% Or mastoid reference
EEG = pop_reref(EEG, [9 21], 'keepref', 'on');  % TP9, TP10

%% Epoch data
% Extract epochs around events
EEG = pop_epoch(EEG, {'stim'}, [-1 2], ...  % -1 to 2 seconds around 'stim'
                'newname', 'subject01_epochs', ...
                'epochinfo', 'yes');

% Baseline correction
EEG = pop_rmbase(EEG, [-200 0]);  % Use -200 to 0 ms as baseline

%% Artifact rejection
% Automatic rejection by amplitude
EEG = pop_eegthresh(EEG, 1, 1:EEG.nbchan, -100, 100, ...
                    EEG.xmin, EEG.xmax, 0, 1);

% Automatic rejection by probability
EEG = pop_jointprob(EEG, 1, 1:EEG.nbchan, 5, 5, 0, 1);

% Manual inspection
pop_eegplot(EEG, 1, 1, 1);  % Visual inspection

%% ICA
% Run ICA (AMICA is recommended, but runica is default)
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);

% Or use AMICA (better, but slower)
EEG = pop_runamica(EEG, 'num_chans', EEG.nbchan);

%% IC classification
% Label components automatically
EEG = pop_iclabel(EEG, 'default');

% Remove artifactual components
% Keep only brain components (>80% brain, <20% artifact)
EEG = pop_icflag(EEG, [NaN NaN; 0.9 1; 0.9 1; NaN NaN; NaN NaN; NaN NaN; NaN NaN]);
EEG = pop_subcomp(EEG, [], 0);

%% Save processed data
EEG = pop_saveset(EEG, 'filename', 'subject01_processed.set', ...
                       'filepath', '/output/');
```

## Independent Component Analysis (ICA)

### Running ICA

```matlab
% Standard ICA (Infomax)
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);

% Fast ICA
EEG = pop_runica(EEG, 'icatype', 'fastica');

% AMICA (adaptive mixture ICA - recommended)
EEG = pop_runamica(EEG, 'num_chans', EEG.nbchan, ...
                        'num_models', 1, ...
                        'max_iter', 2000);

% Plot components
pop_selectcomps(EEG, 1:35);  % Plot first 35 components

% Plot component properties
pop_prop(EEG, 1, 1, NaN, {'freqrange', [1 50]});  % Component 1
```

### IC Classification with ICLabel

```matlab
% Run ICLabel
EEG = pop_iclabel(EEG, 'default');

% View classifications
EEG.etc.ic_classification.ICLabel.classifications
% Columns: Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other

% Automatic removal
% Remove components with <20% brain probability
brain_prob = EEG.etc.ic_classification.ICLabel.classifications(:,1);
bad_comps = find(brain_prob < 0.2);

EEG = pop_subcomp(EEG, bad_comps, 0);

% Or use thresholds
EEG = pop_icflag(EEG, [NaN NaN;  % Brain (any)
                        0.9 1;    % Muscle (>90%)
                        0.9 1;    % Eye (>90%)
                        0.9 1;    % Heart (>90%)
                        0.9 1;    % Line Noise (>90%)
                        0.9 1;    % Channel Noise (>90%)
                        NaN NaN]); % Other (any)
```

## Event-Related Potentials (ERPs)

```matlab
% Average across trials
ERP = pop_averager(ALLEEG, 'Criterion', 'all');

% Plot ERP
figure; pop_plotdata(ERP, 1, 1:64, 'Channels');

% Topographic map at specific latency
figure; pop_topoplot(EEG, 1, 300);  % At 300 ms

% ERP image (sorted trials)
figure; pop_erpimage(EEG, 1, 10, {{}}, 'channel', 10, ...
                     'erpimageopt', {'erp', 'cbar'});

% Compare conditions
ERP_stim1 = pop_averager(ALLEEG, 'Criterion', 'type==1');
ERP_stim2 = pop_averager(ALLEEG, 'Criterion', 'type==2');

% Difference wave
ERP_diff = ERP_stim1;
ERP_diff.data = ERP_stim1.data - ERP_stim2.data;

% Plot comparison
pop_plottopo(ERP_diff, 1:64, 'ERP', 0, 'ydir', 1);
```

## Time-Frequency Analysis

### Event-Related Spectral Perturbation (ERSP)

```matlab
% Compute ERSP for one channel
figure; pop_newtimef(EEG, 1, 10, [-1000 2000], [3 0.5], ...
                     'baseline', [-500 -200], ...
                     'freqs', [3 50], ...
                     'plotphase', 'off', ...
                     'padratio', 4);

% ERSP for all channels
[STUDY ALLEEG] = std_precomp(STUDY, ALLEEG, {}, ...
                             'interp', 'on', ...
                             'recompute', 'on', ...
                             'ersp', 'on', ...
                             'erspparams', {'cycles', [3 0.5], ...
                                           'freqs', [3 50]});

% Plot results
STUDY = std_erspplot(STUDY, ALLEEG, 'channels', {'Cz'});
```

### Inter-Trial Coherence (ITC)

```matlab
% Compute ITC
figure; pop_newtimef(EEG, 1, 10, [-1000 2000], [3 0.5], ...
                     'baseline', NaN, ...  % No baseline for ITC
                     'freqs', [3 50], ...
                     'plotitc', 'on', ...
                     'plotersp', 'off');
```

## Source Localization

### DIPFIT Plugin

```matlab
% Setup DIPFIT
EEG = pop_dipfit_settings(EEG, ...
    'hdmfile', [eeglabpath '/plugins/dipfit/standard_BEM/standard_vol.mat'], ...
    'coordformat', 'MNI', ...
    'mrifile', [eeglabpath '/plugins/dipfit/standard_BEM/standard_mri.mat'], ...
    'chanfile', [eeglabpath '/plugins/dipfit/standard_BEM/elec/standard_1005.elc'], ...
    'coord_transform', [0 0 0 0 0 0 1 1 1], ...
    'chansel', 1:EEG.nbchan);

% Fit dipoles to components
EEG = pop_multifit(EEG, 1:35, 'threshold', 100, 'dipoles', 2);

% Plot dipoles
pop_dipplot(EEG, 1:35, 'mri', [eeglabpath '/plugins/dipfit/standard_BEM/standard_mri.mat']);
```

## STUDY Design (Group Analysis)

```matlab
% Create STUDY
[STUDY ALLEEG] = std_editset([], [], 'name', 'MyStudy', ...
                             'commands', {{'index', 1, 'subject', 'sub01', 'condition', 'control'}, ...
                                         {'index', 2, 'subject', 'sub02', 'condition', 'patient'}}, ...
                             'updatedat', 'on');

% Precompute measures
[STUDY ALLEEG] = std_precomp(STUDY, ALLEEG, {}, ...
                             'savetrials', 'on', ...
                             'interp', 'on', ...
                             'recompute', 'on', ...
                             'erp', 'on', ...
                             'spec', 'on', ...
                             'ersp', 'on');

% Statistics
[STUDY stats] = std_erpplot(STUDY, ALLEEG, ...
                            'channels', {'Cz'}, ...
                            'design', 1);
```

## Scripting Best Practices

```matlab
% Batch processing multiple subjects
subjects = {'sub01', 'sub02', 'sub03'};
eeglab nogui;

for s = 1:length(subjects)
    fprintf('Processing %s...\n', subjects{s});

    % Load
    EEG = pop_biosig([subjects{s} '.bdf']);

    % Preprocess
    EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5);
    EEG = pop_eegfiltnew(EEG, 'hicutoff', 40);
    EEG = pop_reref(EEG, []);

    % Epoch
    EEG = pop_epoch(EEG, {'stim'}, [-1 2]);
    EEG = pop_rmbase(EEG, [-200 0]);

    % ICA
    EEG = pop_runica(EEG, 'extended', 1);
    EEG = pop_iclabel(EEG, 'default');

    % Save
    EEG = pop_saveset(EEG, 'filename', [subjects{s} '_processed.set']);

    fprintf('Done with %s\n', subjects{s});
end
```

## Advanced Features

### Connectivity Analysis (SIFT Plugin)

```matlab
% Install SIFT
plugin_askinstall('SIFT');

% Compute connectivity
EEG = pop_pre_prepData(EEG, 'SignalType', {'Channels'}, ...
                            'EpochsToAnalyze', [], ...
                            'TrialsToAnalyze', []);

% Fit model
EEG = pop_est_fitMVAR(EEG, 'ModelOrder', 15);

% Compute connectivity measures
EEG = pop_est_mvarConnectivity(EEG, 'connmethods', {'dDTF08', 'ffDTF', 'pCoh'});

% Visualize
pop_vis_TimeFreqGrid(EEG);
```

### Clean Data (ASR - Artifact Subspace Reconstruction)

```matlab
% Install clean_rawdata plugin
plugin_askinstall('clean_rawdata');

% Clean continuous data
EEG = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion', 5, ...
    'ChannelCriterion', 0.8, ...
    'LineNoiseCriterion', 4, ...
    'Highpass', 'off', ...
    'BurstCriterion', 20, ...
    'WindowCriterion', 0.25, ...
    'BurstRejection', 'on', ...
    'Distance', 'Euclidian');
```

## Integration with Claude Code

When helping users with EEGLAB:

1. **Check Installation:**
   ```matlab
   which eeglab
   eeglab version
   ```

2. **Common Issues:**
   - MATLAB path not set
   - Memory errors with large datasets
   - Plugin compatibility
   - Channel location issues

3. **Best Practices:**
   - Always set channel locations
   - Save at each major processing step
   - Use pop_ functions for GUI equivalents
   - Document parameters in scripts
   - Visual inspection of data quality

4. **Performance:**
   - Use 'nogui' mode for batch processing
   - Clear unused datasets from memory
   - Process in chunks for large files

## Troubleshooting

**Problem:** "Undefined function 'eeglab'"
**Solution:** Add EEGLAB to MATLAB path with `addpath`

**Problem:** Out of memory
**Solution:** Process fewer channels, use `pop_select`, or increase Java heap

**Problem:** ICA produces poor results
**Solution:** Clean data first, use sufficient data (>30*channels^2 samples)

**Problem:** Channel locations not found
**Solution:** Use `pop_chanedit` with standard template file

## Resources

- EEGLAB Wiki: https://eeglab.org/
- Tutorial: https://eeglab.org/tutorials/
- Mailing List: https://eeglab.org/others/EEGLAB_mailing_lists.html
- YouTube: https://www.youtube.com/c/EEGLAB
- Workshops: https://eeglab.org/workshops/

## Related Tools

- **FieldTrip:** Alternative MATLAB toolbox
- **MNE-Python:** Python alternative
- **Brainstorm:** GUI-based alternative
- **ICLabel:** IC classification
- **ERPLAB:** ERP-focused extension

## Citation

```bibtex
@article{delorme2004eeglab,
  title={EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis},
  author={Delorme, Arnaud and Makeig, Scott},
  journal={Journal of Neuroscience Methods},
  volume={134},
  number={1},
  pages={9--21},
  year={2004},
  doi={10.1016/j.jneumeth.2003.10.009}
}
```
