# FID-A

## Overview

FID-A (Free Induction Decay Analysis) is a comprehensive MATLAB toolkit for simulation and processing of Magnetic Resonance Spectroscopy (MRS) data. It provides a modular framework for reading, manipulating, processing, simulating, and plotting MRS data from multiple vendors and sequence types. FID-A is particularly powerful for creating custom basis sets and developing new MRS processing methods.

**Website:** https://github.com/CIC-methods/FID-A
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** MIT License

## Key Features

- Multi-vendor data support (Siemens, Philips, GE, Bruker)
- Comprehensive preprocessing functions
- Density matrix MRS simulation
- Basis set generation for LCModel and Osprey
- Spectral registration and alignment
- Frequency and phase correction
- Eddy current correction
- Water removal and filtering
- Coil combination and averaging
- Publication-quality plotting
- Support for edited spectroscopy (MEGA, HERMES)
- Modular architecture for custom workflows

## Installation

### Requirements

```matlab
% MATLAB R2012b or later
% No additional toolboxes required
% Optional: Parallel Computing Toolbox for faster simulations
```

### Download and Setup

```bash
# Clone from GitHub
git clone https://github.com/CIC-methods/FID-A.git
cd FID-A

# Or download latest release
# https://github.com/CIC-methods/FID-A/releases
```

```matlab
% Add FID-A to MATLAB path
addpath(genpath('/path/to/FID-A'));
savepath;

% Verify installation
which io_loadspec_twix
% Should return path to FID-A function

% Test with example data
cd /path/to/FID-A/exampleData
```

## Data Structure

### FID-A Data Format

All FID-A functions use a standardized structure:

```matlab
% in.fids:      Raw time-domain data [coils x averages x subspecs]
% in.specs:     Frequency-domain data [coils x averages x subspecs]
% in.sz:        Size of data matrix
% in.spectralwidth: Spectral width (Hz)
% in.dwelltime: Dwell time (s)
% in.txfrq:     Transmitter frequency (MHz)
% in.te:        Echo time (ms)
% in.tr:        Repetition time (ms)
% in.Bo:        Field strength (T)
% in.ppm:       Chemical shift axis (ppm)
% in.t:         Time axis (s)
% in.centerFreq: Center frequency (ppm)
% in.averages:  Number of averages
% in.rawAverages: Number of raw averages
% in.subspecs:  Number of subspectra
% in.dims:      Data dimensions structure
```

## Loading Data

### Siemens Data

```matlab
% Load TWIX (.dat) file
in = io_loadspec_twix('/data/subject01.dat');

% Load RDA (.rda) file
in = io_loadspec_rda('/data/subject01.rda');

% Load DICOM
in = io_loadspec_dicom('/data/subject01.dcm');
```

### Philips Data

```matlab
% Load SDAT/SPAR pair
in = io_loadspec_sdat('/data/subject01.sdat');

% Load DATA/LIST pair
in = io_loadspec_data('/data/subject01.data');
```

### GE Data

```matlab
% Load P-file
in = io_loadspec_GE('/data/P12345.7');
```

### NIfTI-MRS

```matlab
% Load NIfTI-MRS format
in = io_loadspec_nii('/data/subject01_svs.nii.gz');
```

## Basic Processing Workflow

### Single-Voxel PRESS

```matlab
% Load data
in_raw = io_loadspec_twix('/data/subject01.dat');

% Remove bad averages
in_clean = op_rmbadaverages(in_raw, 3, 'f');  % 3 SD threshold

% Average across repetitions
in_avg = op_averaging(in_clean);

% Frequency and phase correction
[in_corrected, fs, phs] = op_robustSpecReg(in_avg, 'f');

% Combine coils
in_combined = op_combinecoils(in_corrected, 'h');  % WSVD method

% Left shift to remove pre-acquisition points
in_shifted = op_leftshift(in_combined, in_combined.pointsToLeftshift);

% Apply zero-order phase correction
in_phased = op_addphase(in_shifted, 0);  % Adjust as needed

% Frequency align to expected frequency
in_aligned = op_freqrange(in_phased, 0.2, 4.2);  % Focus on 0.2-4.2 ppm

% Apply line broadening (optional)
in_filtered = op_filter(in_aligned, 3);  % 3 Hz exponential filter

% Plot result
figure;
plot(in_filtered.ppm, real(in_filtered.specs));
set(gca, 'XDir', 'reverse');
xlim([0.2 4.2]);
xlabel('Chemical Shift (ppm)');
ylabel('Signal');
title('Processed Spectrum');
```

### Water Removal

```matlab
% For metabolite spectra with residual water

% Method 1: HSVD water removal
in_noh2o = op_removeWater(in_combined, [4.4 5.0]);  % Remove 4.4-5.0 ppm

% Method 2: Aggressive water removal
in_noh2o = op_removeWater2(in_combined, [4.0 5.5], 'y', 32);

% Method 3: Hankel-Lanczos SVD (HLSVD)
in_noh2o = op_hsvd(in_combined, 4.68, 100);  % Remove at 4.68 ppm, 100 components
```

## MEGA-PRESS Analysis

### Process Edited Spectrum

```matlab
% Load MEGA-PRESS data (edit-ON and edit-OFF interleaved)
in_raw = io_loadspec_twix('/data/MEGA_PRESS.dat');

% Separate edit-ON and edit-OFF
in_A = op_takesubspec(in_raw, 1);  % Edit-OFF
in_B = op_takesubspec(in_raw, 2);  % Edit-ON

% Remove bad averages from each
in_A = op_rmbadaverages(in_A, 3, 'f');
in_B = op_rmbadaverages(in_B, 3, 'f');

% Frequency/phase correction for each
[in_A_corr, ~, ~] = op_robustSpecReg(in_A, 'f');
[in_B_corr, ~, ~] = op_robustSpecReg(in_B, 'f');

% Align edit-ON to edit-OFF
[in_B_aligned, ~] = op_alignAverages(in_B_corr, 2.5, 'n');

% Average
in_A_avg = op_averaging(in_A_corr);
in_B_avg = op_averaging(in_B_aligned);

% Combine coils
in_A_comb = op_combinecoils(in_A_avg, 'h');
in_B_comb = op_combinecoils(in_B_avg, 'h');

% Calculate difference spectrum (for GABA)
in_diff = op_subtractScans(in_B_comb, in_A_comb);

% Plot all three
figure;
subplot(3,1,1);
plot(in_A_comb.ppm, real(in_A_comb.specs));
set(gca, 'XDir', 'reverse');
xlim([0.5 4.5]);
title('Edit-OFF');

subplot(3,1,2);
plot(in_B_comb.ppm, real(in_B_comb.specs));
set(gca, 'XDir', 'reverse');
xlim([0.5 4.5]);
title('Edit-ON');

subplot(3,1,3);
plot(in_diff.ppm, real(in_diff.specs));
set(gca, 'XDir', 'reverse');
xlim([2.5 3.5]);
title('Difference (GABA)');
xlabel('Chemical Shift (ppm)');
```

## Simulation

### Simulate Single Metabolite

```matlab
% Simulate NAA spectrum
% Parameters
sys.H = [2.01 2.49 2.67];  % Chemical shifts (ppm)
sys.J = [2.01 2.49 16.3];  % J-couplings (Hz)
sys.shifts = [2.01 2.49 2.67];

% Acquisition parameters
Bfield = 3;      % Field strength (T)
linewidth = 5;   % Linewidth (Hz)
npts = 2048;     % Number of points
sw = 2000;       % Spectral width (Hz)
Bo = Bfield;
centerFreq = 2.3; % Center frequency (ppm)
TE = 30;         % Echo time (ms)

% Simulate
out = sim_press(n, sys, TE);

% Convert to FID-A format
out.Bo = Bo;
out.spectralwidth = sw;
out.centerFreq = centerFreq;
out = op_addNoise(out, 100);  % Add noise (SNR = 100)

% Plot
figure;
plot(out.ppm, real(out.specs));
set(gca, 'XDir', 'reverse');
title('Simulated NAA');
xlabel('Chemical Shift (ppm)');
```

### Create Basis Set for Quantification

```matlab
% Simulate multiple metabolites for basis set
metabolites = {'NAA', 'Cr', 'Cho', 'Glu', 'mI', 'GABA', 'Glx', 'Lac'};
TE = 30;        % ms
Bfield = 3;     % Tesla
linewidth = 3;  % Hz

% Initialize basis set structure
basis = struct();

% Simulate each metabolite
for m = 1:length(metabolites)
    fprintf('Simulating %s...\n', metabolites{m});

    % Load spin system (pre-defined in FID-A)
    load(sprintf('spinSystems/%s.mat', metabolites{m}));

    % Simulate PRESS
    out = sim_press(2048, sys, TE);

    % Add linewidth
    out = op_addLinewidth(out, linewidth);

    % Store in basis
    basis.(metabolites{m}) = out;
end

% Save basis set
save(sprintf('basis_PRESS_TE%d_3T.mat', TE), 'basis');

% Export to LCModel format
io_writelcm_basis(basis, sprintf('basis_PRESS_TE%d_3T.BASIS', TE));
```

### Simulate Entire Acquisition

```matlab
% Simulate realistic dataset with multiple averages

% Define metabolite concentrations (mM)
conc = struct();
conc.NAA = 12.5;
conc.Cr = 8.0;
conc.Cho = 3.0;
conc.Glu = 12.0;
conc.mI = 7.5;

metabolites = fieldnames(conc);
n_averages = 64;

% Initialize
sim_sum = [];

% Simulate each metabolite
for m = 1:length(metabolites)
    load(sprintf('spinSystems/%s.mat', metabolites{m}));

    % Simulate with concentration scaling
    out = sim_press(2048, sys, 30);
    out = op_ampScale(out, conc.(metabolites{m}));

    % Add to sum
    if isempty(sim_sum)
        sim_sum = out;
    else
        sim_sum = op_addScans(sim_sum, out);
    end
end

% Add macromolecule baseline
MM = sim_make_MM(sim_sum, 3);  % 3T field strength
sim_sum = op_addScans(sim_sum, MM);

% Add noise
sim_sum = op_addNoise(sim_sum, 50);  % SNR = 50

% Add frequency/phase variations across averages
sim_multi = op_freqPhaseShiftReps(sim_sum, n_averages, 5, 10);  % 5 Hz, 10 deg

% Plot
figure;
subplot(2,1,1);
plot(sim_multi.ppm, real(mean(sim_multi.specs, 2)));
set(gca, 'XDir', 'reverse');
title('Simulated Spectrum (averaged)');

subplot(2,1,2);
imagesc(sim_multi.ppm, 1:n_averages, real(squeeze(sim_multi.specs)'));
set(gca, 'XDir', 'reverse');
title('Individual Averages');
xlabel('Chemical Shift (ppm)');
ylabel('Average #');
```

## Advanced Processing

### Spectral Alignment Across Subjects

```matlab
% Align multiple subjects to reference

% Load reference
ref = io_loadspec_twix('/data/sub-01.dat');
ref = op_averaging(ref);
ref = op_combinecoils(ref, 'h');

% Initialize array
n_subjects = 10;
all_specs = cell(n_subjects, 1);
all_specs{1} = ref;

% Load and align remaining subjects
for s = 2:n_subjects
    in = io_loadspec_twix(sprintf('/data/sub-%02d.dat', s));
    in = op_averaging(in);
    in = op_combinecoils(in, 'h');

    % Align to reference
    [in_aligned, ~] = op_alignScans(in, ref, 2.0, 3.0);  % Align using 2-3 ppm

    all_specs{s} = in_aligned;
end

% Calculate group average
group_avg = all_specs{1};
for s = 2:n_subjects
    group_avg = op_addScans(group_avg, all_specs{s});
end
group_avg = op_ampScale(group_avg, 1/n_subjects);

% Plot
figure;
plot(group_avg.ppm, real(group_avg.specs));
set(gca, 'XDir', 'reverse');
title('Group Average Spectrum');
```

### Eddy Current Correction

```matlab
% Correct for eddy currents using water reference

% Load metabolite data
in_metab = io_loadspec_twix('/data/subject01_metab.dat');

% Load water reference
in_water = io_loadspec_twix('/data/subject01_water.dat');

% Extract phase from water
water_avg = op_averaging(in_water);
water_phase = angle(water_avg.fids);

% Apply phase correction to metabolite data
in_corrected = op_ecc(in_metab, in_water);

% Compare before/after
figure;
subplot(2,1,1);
plot(in_metab.ppm, real(op_averaging(in_metab).specs));
set(gca, 'XDir', 'reverse');
title('Before ECC');

subplot(2,1,2);
plot(in_corrected.ppm, real(op_averaging(in_corrected).specs));
set(gca, 'XDir', 'reverse');
title('After ECC');
xlabel('Chemical Shift (ppm)');
```

### Custom Filtering

```matlab
% Apply various filters

% Exponential line broadening (apodization)
in_lb = op_filter(in, 5);  % 5 Hz line broadening

% Gaussian filter
in_gauss = op_gaussianFilter(in, 200);  % 200 Hz bandwidth

% Moving average filter
in_smooth = op_movingAverage(in, 5);  % 5-point moving average

% Tukey window
in_tukey = op_tukeyWindow(in, 0.5);  % 50% taper

% Compare filters
figure;
hold on;
plot(in.ppm, real(in.specs), 'k-', 'DisplayName', 'Original');
plot(in_lb.ppm, real(in_lb.specs), 'r-', 'DisplayName', '5 Hz LB');
plot(in_gauss.ppm, real(in_gauss.specs), 'b-', 'DisplayName', 'Gaussian');
set(gca, 'XDir', 'reverse');
xlim([1.8 2.2]);
legend;
title('Filter Comparison (NAA region)');
```

## Data Export

### Export to LCModel

```matlab
% Save in LCModel .RAW format
io_writelcm(in_processed, '/output/subject01.RAW');

% Create LCModel control file
fid = fopen('/output/subject01.control', 'w');
fprintf(fid, '$LCMODL\n');
fprintf(fid, 'FILBAS=''/path/to/basis.BASIS''\n');
fprintf(fid, 'FILRAW=''/output/subject01.RAW''\n');
fprintf(fid, 'FILPS=''/output/subject01.PS''\n');
fprintf(fid, 'ECHOT=%.1f\n', in_processed.te);
fprintf(fid, 'HZPPPM=%.6f\n', in_processed.txfrq);
fprintf(fid, 'NUNFIL=%d\n', in_processed.sz(1));
fprintf(fid, 'DELTAT=%.8f\n', in_processed.dwelltime);
fprintf(fid, '$END\n');
fclose(fid);
```

### Export to Text Files

```matlab
% Export spectrum to CSV
data_export = [in.ppm', real(in.specs), imag(in.specs)];
csvwrite('/output/spectrum.csv', data_export);

% Or with headers
fid = fopen('/output/spectrum.txt', 'w');
fprintf(fid, 'PPM,Real,Imaginary\n');
for i = 1:length(in.ppm)
    fprintf(fid, '%.4f,%.6f,%.6f\n', in.ppm(i), real(in.specs(i)), imag(in.specs(i)));
end
fclose(fid);
```

### Export Figures

```matlab
% Create publication-quality figure
figure('Position', [100 100 800 400]);
plot(in.ppm, real(in.specs), 'k-', 'LineWidth', 1.5);
set(gca, 'XDir', 'reverse');
xlim([0.2 4.2]);
xlabel('Chemical Shift (ppm)', 'FontSize', 14);
ylabel('Signal Intensity', 'FontSize', 14);
title('MRS Spectrum', 'FontSize', 16);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
box off;

% Save as high-resolution image
print('/output/spectrum.png', '-dpng', '-r300');
saveas(gcf, '/output/spectrum.fig');
```

## Batch Processing

```matlab
% Process multiple subjects
subjects = dir('/data/sub-*.dat');
n_subjects = length(subjects);

% Initialize results
results = struct();

for s = 1:n_subjects
    fprintf('Processing %s (%d/%d)...\n', subjects(s).name, s, n_subjects);

    % Load
    in = io_loadspec_twix(fullfile(subjects(s).folder, subjects(s).name));

    % Process
    in = op_rmbadaverages(in, 3, 'f');
    [in, ~, ~] = op_robustSpecReg(in, 'f');
    in = op_averaging(in);
    in = op_combinecoils(in, 'h');
    in = op_leftshift(in, in.pointsToLeftshift);

    % Store
    results(s).name = subjects(s).name;
    results(s).spectrum = in;

    % Calculate QC metrics
    results(s).SNR = op_getSNR(in, 1.8, 2.2, -2, 0);  % NAA region
    results(s).linewidth = op_getLW(in, 1.8, 2.2);    % From NAA

    % Export
    io_writelcm(in, sprintf('/output/sub-%02d.RAW', s));
end

% Save all results
save('/output/batch_results.mat', 'results');

% Summary statistics
SNRs = [results.SNR];
LWs = [results.linewidth];

fprintf('\nBatch Summary:\n');
fprintf('Mean SNR: %.1f ± %.1f\n', mean(SNRs), std(SNRs));
fprintf('Mean Linewidth: %.2f ± %.2f Hz\n', mean(LWs), std(LWs));
```

## Integration with Claude Code

When helping users with FID-A:

1. **Check Data Structure:**
   ```matlab
   % Verify FID-A structure
   fields(in)
   disp(in.dims)
   ```

2. **Common Processing Pipeline:**
   - Load → Remove bad averages → Align → Average → Combine coils → Process

3. **Simulation Workflow:**
   - Define spin system → Simulate sequence → Add noise → Export basis

4. **Quality Checks:**
   - Inspect raw data before processing
   - Check SNR and linewidth
   - Verify frequency stability

## Troubleshooting

**Problem:** "Undefined function or variable" errors
**Solution:** Ensure full FID-A path added with `addpath(genpath('/path/to/FID-A'))`

**Problem:** Data dimensions don't match
**Solution:** Check subspectra, coils, averages using `in.dims` structure

**Problem:** Poor frequency correction
**Solution:** Inspect individual averages, increase threshold for bad average removal

**Problem:** Simulation takes very long
**Solution:** Reduce number of points, use Parallel Computing Toolbox, simplify spin systems

**Problem:** Memory errors with large datasets
**Solution:** Process coils/averages separately, clear variables between subjects

## Best Practices

1. **Always inspect raw data first**
2. **Remove bad averages before alignment**
3. **Save intermediate processing steps**
4. **Document processing parameters**
5. **Validate simulations against real data**
6. **Use consistent basis sets for quantification**
7. **Check dimensions after each operation**

## Resources

- **GitHub:** https://github.com/CIC-methods/FID-A
- **Documentation:** https://github.com/CIC-methods/FID-A/wiki
- **Publication:** https://doi.org/10.1002/mrm.26091
- **Forum:** https://forum.mrshub.org/c/mrs-software/fid-a
- **Examples:** https://github.com/CIC-methods/FID-A/tree/master/exampleData

## Citation

```bibtex
@article{simpson2017fida,
  title={Advanced processing and simulation of MRS data using the FID appliance (FID-A)—An open source, MATLAB-based toolkit},
  author={Simpson, Robin and Devenyi, Gabriel A and Jezzard, Peter and Hennessy, Timothy J and Near, Jamie},
  journal={Magnetic Resonance in Medicine},
  volume={77},
  number={1},
  pages={23--33},
  year={2017},
  publisher={Wiley Online Library}
}
```

## Related Tools

- **Osprey:** Comprehensive MRS processing suite (uses FID-A functions)
- **Gannet:** GABA analysis toolkit
- **TARQUIN:** Automatic MRS quantification
- **LCModel:** Commercial gold-standard quantification
- **jMRUI:** Time-domain MRS analysis
- **INSPECTOR:** Web-based MRS quality control
