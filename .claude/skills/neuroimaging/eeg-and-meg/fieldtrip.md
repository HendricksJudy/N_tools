# FieldTrip

## Overview

FieldTrip is an advanced MATLAB toolbox for MEG, EEG, and intracranial electrophysiological data analysis. It emphasizes a modular, script-based approach with powerful functions for time-frequency analysis, source reconstruction, connectivity analysis, and statistics. FieldTrip is particularly strong in MEG analysis and offers state-of-the-art methods for complex analyses.

**Website:** https://www.fieldtriptoolbox.org/
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** GNU GPL

## Key Features

- Modular, script-based analysis workflow
- Comprehensive MEG and EEG support
- Advanced time-frequency analysis (wavelets, multitapers)
- Source reconstruction (beamforming, minimum norm)
- Connectivity analysis (coherence, Granger causality, PLV)
- Non-parametric cluster-based permutation statistics
- Real-time data processing
- Integration with MNE-Python and EEGLAB
- Support for intracranial recordings

## Installation

```matlab
% Download FieldTrip from: https://www.fieldtriptoolbox.org/download/
% Extract to desired location

% Add to MATLAB path
addpath('/path/to/fieldtrip');
ft_defaults  % Initialize FieldTrip

% Check installation
ft_version

% For permanent installation, add to startup.m:
% addpath('/path/to/fieldtrip');
% ft_defaults;
```

## Data Structures

FieldTrip uses specific data structures:

```matlab
% Raw data structure
data =
    fsample: 1000              % Sampling frequency
      trial: {1×N cell}        % Trial data
       time: {1×N cell}        % Time axes
      label: {M×1 cell}        % Channel labels

% Timelock structure (averaged/epoched)
timelock =
       avg: [M×T double]       % Averaged data
      time: [1×T double]       % Time axis
     label: {M×1 cell}        % Channel labels
       dof: [M×T double]       % Degrees of freedom
       var: [M×T double]       % Variance

% Freq structure (time-frequency)
freq =
      freq: [1×F double]       % Frequencies
      time: [1×T double]       % Time points
powspctrm: [M×F×T double]     % Power spectrum
     label: {M×1 cell}        % Channel labels
```

## Loading and Preprocessing

### Reading Data

```matlab
% Define trial structure
cfg = [];
cfg.dataset = 'subject01.ds';  % MEG data
cfg.trialdef.eventtype = 'STIM';
cfg.trialdef.eventvalue = 1;
cfg.trialdef.prestim = 1;      % 1 second before
cfg.trialdef.poststim = 2;     % 2 seconds after

% Define trials
cfg = ft_definetrial(cfg);

% Preprocess
cfg.channel = 'MEG';
cfg.demean = 'yes';
cfg.baselinewindow = [-0.5 0];
data = ft_preprocessing(cfg);
```

### Filtering

```matlab
% Low-pass filter
cfg = [];
cfg.lpfilter = 'yes';
cfg.lpfreq = 40;
data_lp = ft_preprocessing(cfg, data);

% High-pass filter
cfg = [];
cfg.hpfilter = 'yes';
cfg.hpfreq = 0.5;
data_hp = ft_preprocessing(cfg, data);

% Band-pass filter
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [8 12];  % Alpha band
data_alpha = ft_preprocessing(cfg, data);

% Notch filter (line noise)
cfg = [];
cfg.dftfilter = 'yes';
cfg.dftfreq = [50 100 150];  % 50 Hz and harmonics
data_clean = ft_preprocessing(cfg, data);
```

### Re-referencing (EEG)

```matlab
% Average reference
cfg = [];
cfg.reref = 'yes';
cfg.refchannel = 'all';
data_reref = ft_preprocessing(cfg, data);

% Specific reference
cfg = [];
cfg.reref = 'yes';
cfg.refchannel = {'M1', 'M2'};  % Mastoids
data_reref = ft_preprocessing(cfg, data);
```

### Artifact Detection and Removal

```matlab
% Visual inspection
cfg = [];
cfg.method = 'summary';
data_clean = ft_rejectvisual(cfg, data);

% Automatic artifact detection
cfg = [];
cfg.method = 'zvalue';
cfg.channel = 'MEG';
cfg.cutoff = 4;  % Z-score threshold
[data_clean, artifacts] = ft_rejectartifact(cfg, data);

% ICA for artifact removal
cfg = [];
cfg.method = 'runica';
cfg.channel = 'MEG';
comp = ft_componentanalysis(cfg, data);

% Visualize components
cfg = [];
cfg.component = [1:20];
cfg.layout = 'CTF275.lay';
ft_topoplotIC(cfg, comp);

% Remove artifactual components
cfg = [];
cfg.component = [1 3 7];  % Components to remove
data_clean = ft_rejectcomponent(cfg, comp, data);
```

## Time-Frequency Analysis

### Wavelet Analysis

```matlab
% Time-frequency analysis with wavelets
cfg = [];
cfg.method = 'wavelet';
cfg.width = 7;  % Wavelet width
cfg.output = 'pow';
cfg.foi = 2:2:40;  % Frequencies of interest (2-40 Hz)
cfg.toi = -0.5:0.05:2;  % Time points of interest

freq = ft_freqanalysis(cfg, data);

% Plot TFR
cfg = [];
cfg.baseline = [-0.5 -0.1];
cfg.baselinetype = 'relchange';
cfg.channel = 'MEG0113';
cfg.layout = 'CTF275.lay';

ft_singleplotTFR(cfg, freq);
```

### Multitaper Method

```matlab
% Multitaper time-frequency analysis
cfg = [];
cfg.method = 'mtmconvol';
cfg.taper = 'hanning';
cfg.foi = 2:2:40;
cfg.t_ftimwin = ones(length(cfg.foi), 1) .* 0.5;  % Time window
cfg.toi = -0.5:0.05:2;

freq_mtm = ft_freqanalysis(cfg, data);

% For better frequency resolution, use multitapers
cfg = [];
cfg.method = 'mtmconvol';
cfg.taper = 'dpss';
cfg.tapsmofrq = 4;  % Frequency smoothing
cfg.foi = 5:1:40;
cfg.t_ftimwin = 0.5 * ones(length(cfg.foi), 1);
cfg.toi = -0.5:0.05:2;

freq_dpss = ft_freqanalysis(cfg, data);
```

### Power Spectral Density

```matlab
% Compute power spectrum
cfg = [];
cfg.method = 'mtmfft';
cfg.taper = 'hanning';
cfg.foi = 1:0.5:100;
cfg.tapsmofrq = 2;

freq_psd = ft_freqanalysis(cfg, data);

% Plot power spectrum
cfg = [];
cfg.channel = 'MEG0113';
cfg.xlim = [1 50];

ft_singleplotER(cfg, freq_psd);
```

## Event-Related Fields/Potentials

```matlab
% Compute ERF/ERP
cfg = [];
cfg.channel = 'MEG';
cfg.trials = 'all';
cfg.covariance = 'yes';
cfg.covariancewindow = [-0.5 0];
cfg.keeptrials = 'no';

timelock = ft_timelockanalysis(cfg, data);

% Plot ERF
cfg = [];
cfg.layout = 'CTF275.lay';
cfg.showlabels = 'yes';

figure;
ft_multiplotER(cfg, timelock);

% Topography at specific time
cfg = [];
cfg.xlim = [0.3 0.5];  % 300-500 ms
cfg.layout = 'CTF275.lay';
cfg.comment = 'xlim';

figure;
ft_topoplotER(cfg, timelock);
```

## Source Analysis

### Prepare Head Model

```matlab
% Load MRI
mri = ft_read_mri('subject01.nii');

% Segment MRI
cfg = [];
cfg.output = {'brain', 'skull', 'scalp'};
segmented = ft_volumesegment(cfg, mri);

% Create head model (BEM)
cfg = [];
cfg.method = 'singleshell';  % Or 'dipoli', 'bemcp'
headmodel = ft_prepare_headmodel(cfg, segmented);

% Alternatively, use template
headmodel = ft_read_headmodel('standard_bem.mat');
```

### Prepare Source Model

```matlab
% Create grid
cfg = [];
cfg.grid.resolution = 10;  % 10 mm grid
cfg.headmodel = headmodel;
cfg.grad = data.grad;

sourcemodel = ft_prepare_sourcemodel(cfg);

% Or use template
template = load('standard_sourcemodel3d10mm.mat');
cfg = [];
cfg.grid = template.sourcemodel;
cfg.headmodel = headmodel;
cfg.grad = data.grad;
sourcemodel = ft_prepare_sourcemodel(cfg);
```

### Beamformer Source Reconstruction

```matlab
% Compute covariance
cfg = [];
cfg.covariance = 'yes';
cfg.covariancewindow = [-0.5 2];

timelock = ft_timelockanalysis(cfg, data);

% LCMV beamformer
cfg = [];
cfg.method = 'lcmv';
cfg.grid = sourcemodel;
cfg.headmodel = headmodel;
cfg.lcmv.keepfilter = 'yes';
cfg.lcmv.fixedori = 'yes';

source = ft_sourceanalysis(cfg, timelock);

% Compute source power in time window
cfg = [];
cfg.method = 'lcmv';
cfg.grid = sourcemodel;
cfg.grid.filter = source.avg.filter;  % Use precomputed filters
cfg.headmodel = headmodel;
cfg.lcmv.projectmom = 'yes';
cfg.lcmv.fixedori = 'yes';

cfg.latency = [0.3 0.5];  % Time window
source_post = ft_sourceanalysis(cfg, timelock);

cfg.latency = [-0.5 0];  % Baseline
source_pre = ft_sourceanalysis(cfg, timelock);

% Contrast
cfg = [];
cfg.parameter = 'avg.pow';
cfg.operation = '(x1-x2)/(x1+x2)';

source_diff = ft_math(cfg, source_post, source_pre);

% Interpolate and plot
cfg = [];
cfg.parameter = 'pow';
source_int = ft_sourceinterpolate(cfg, source_diff, mri);

cfg = [];
cfg.method = 'slice';
cfg.funparameter = 'pow';

ft_sourceplot(cfg, source_int);
```

### Minimum Norm Estimate

```matlab
% Compute leadfield
cfg = [];
cfg.grid = sourcemodel;
cfg.headmodel = headmodel;
cfg.channel = {'MEG'};

leadfield = ft_prepare_leadfield(cfg, data);

% MNE source estimation
cfg = [];
cfg.method = 'mne';
cfg.grid = leadfield;
cfg.headmodel = headmodel;
cfg.mne.lambda = 3;  % Regularization parameter
cfg.mne.keepfilter = 'yes';

source_mne = ft_sourceanalysis(cfg, timelock);
```

## Connectivity Analysis

### Coherence

```matlab
% Frequency analysis with cross-spectral density
cfg = [];
cfg.method = 'mtmfft';
cfg.output = 'powandcsd';
cfg.taper = 'dpss';
cfg.tapsmofrq = 4;
cfg.foi = 5:2:30;

freq = ft_freqanalysis(cfg, data);

% Compute connectivity
cfg = [];
cfg.method = 'coh';  % Coherence
cfg.channelcmb = {'MEG0113', 'MEG0112'};  % Channel pair

coherence = ft_connectivityanalysis(cfg, freq);

% Plot
cfg = [];
cfg.parameter = 'cohspctrm';
cfg.xlim = [5 30];

ft_singleplotER(cfg, coherence);
```

### Granger Causality

```matlab
% Prepare data for connectivity
cfg = [];
cfg.method = 'mtmfft';
cfg.output = 'fourier';
cfg.taper = 'dpss';
cfg.tapsmofrq = 2;
cfg.foi = 5:1:30;

freq_fourier = ft_freqanalysis(cfg, data);

% Non-parametric Granger causality
cfg = [];
cfg.method = 'granger';

granger = ft_connectivityanalysis(cfg, freq_fourier);

% Plot
cfg = [];
cfg.parameter = 'grangerspctrm';
cfg.zlim = [0 1];

ft_connectivityplot(cfg, granger);
```

### Phase Locking Value (PLV)

```matlab
% Compute PLV
cfg = [];
cfg.method = 'plv';
cfg.channel = {'MEG'};

plv = ft_connectivityanalysis(cfg, freq_fourier);

% Network visualization
cfg = [];
cfg.parameter = 'plvspctrm';
cfg.layout = 'CTF275.lay';

ft_topoplotER(cfg, plv);
```

## Statistics

### Cluster-Based Permutation Test

```matlab
% Prepare data for two conditions
cfg = [];
cfg.trials = find(data.trialinfo == 1);  % Condition 1
timelock_cond1 = ft_timelockanalysis(cfg, data);

cfg.trials = find(data.trialinfo == 2);  % Condition 2
timelock_cond2 = ft_timelockanalysis(cfg, data);

% Cluster-based permutation test
cfg = [];
cfg.channel = 'MEG';
cfg.latency = [0 1];
cfg.method = 'montecarlo';
cfg.statistic = 'depsamplesT';  % Paired t-test
cfg.correctm = 'cluster';
cfg.clusteralpha = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan = 2;
cfg.tail = 0;
cfg.clustertail = 0;
cfg.alpha = 0.025;
cfg.numrandomization = 1000;

% Design matrix
nsubj = 10;
design = zeros(2, 2*nsubj);
design(1,:) = [ones(1,nsubj) 2*ones(1,nsubj)];
design(2,:) = [1:nsubj 1:nsubj];

cfg.design = design;
cfg.uvar = 1;  % Unit variable
cfg.ivar = 2;  % Independent variable

% Compute statistics
[stat] = ft_timelockstatistics(cfg, timelock_cond1, timelock_cond2);

% Plot results
cfg = [];
cfg.alpha = 0.025;
cfg.parameter = 'stat';
cfg.zlim = [-4 4];
cfg.layout = 'CTF275.lay';

ft_clusterplot(cfg, stat);
```

## Real-Time Processing

```matlab
% Setup real-time buffer
cfg = [];
cfg.target.datafile = 'buffer://localhost:1972';
cfg.target.dataformat = 'buffer';

% Read from buffer
hdr = ft_read_header(cfg.target.datafile);
data = ft_read_data(cfg.target.datafile, 'header', hdr);

% Process in real-time loop
while true
    % Read new data
    dat = ft_read_data(cfg.target.datafile, 'header', hdr, ...
                       'begsample', begsample, 'endsample', endsample);

    % Process
    % ... your analysis here ...

    % Update sample indices
    begsample = endsample + 1;
    endsample = begsample + blocksize - 1;

    pause(0.1);  % Small delay
end
```

## Integration with Claude Code

When helping users with FieldTrip:

1. **Check Installation:**
   ```matlab
   which ft_defaults
   ft_version
   ```

2. **Common Issues:**
   - Path not set (run ft_defaults)
   - Data structure incompatibility
   - Memory limitations with large datasets
   - Configuration parameter errors

3. **Best Practices:**
   - Always use cfg structures
   - Clear cfg between analyses
   - Save intermediate results
   - Use ft_databrowser for inspection
   - Check data structures with ft_datatype

4. **Performance:**
   - Use cfg.channel to select channels
   - Use cfg.trials to select trials
   - Process data in chunks
   - Parallelize with parfor where possible

## Troubleshooting

**Problem:** "Undefined function 'ft_defaults'"
**Solution:** Add FieldTrip to path: `addpath('/path/to/fieldtrip'); ft_defaults`

**Problem:** Memory errors
**Solution:** Process fewer channels/trials, use cfg.toilim to limit time

**Problem:** Incompatible data structures
**Solution:** Use ft_checkdata or ft_datatype to check/convert

**Problem:** Source reconstruction fails
**Solution:** Check headmodel, sourcemodel, and sensor alignment with ft_sourceplot

## Resources

- Website: https://www.fieldtriptoolbox.org/
- Tutorial: https://www.fieldtriptoolbox.org/tutorial/
- FAQ: https://www.fieldtriptoolbox.org/faq/
- Mailing List: https://www.fieldtriptoolbox.org/discussion_list/
- GitHub: https://github.com/fieldtrip/fieldtrip
- Workshops: https://www.fieldtriptoolbox.org/workshops/

## Related Tools

- **EEGLAB:** Alternative for EEG analysis
- **MNE-Python:** Python-based alternative
- **Brainstorm:** GUI-focused alternative
- **SPM:** Integration for M/EEG analysis

## Citation

```bibtex
@article{oostenveld2011fieldtrip,
  title={FieldTrip: Open source software for advanced analysis of MEG, EEG, and invasive electrophysiological data},
  author={Oostenveld, Robert and Fries, Pascal and Maris, Eric and Schoffelen, Jan-Mathijs},
  journal={Computational Intelligence and Neuroscience},
  volume={2011},
  pages={1--9},
  year={2011},
  doi={10.1155/2011/156869}
}
```
