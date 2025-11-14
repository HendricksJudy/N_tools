# qMRLab - Quantitative MRI Analysis Library

## Overview

qMRLab is an open-source, comprehensive software package for quantitative MRI (qMRI) data simulation, analysis, and visualization. Developed as a collaborative effort by the NeuroPoly lab at Polytechnique Montreal and the qMRLab community, it provides implementations of numerous qMRI techniques with a focus on reproducibility, education, and method comparison. qMRLab supports both MATLAB and Octave (free alternative), with Python bindings available, and offers both GUI and command-line interfaces.

**Website:** https://qmrlab.org/
**Platform:** MATLAB/Octave/Python (Windows/macOS/Linux)
**License:** MIT
**Key Application:** Multi-technique quantitative MRI, protocol optimization, reproducible research

### What is Quantitative MRI?

Unlike conventional MRI that produces qualitative contrast, quantitative MRI measures physical tissue properties:

- **T1 relaxation** - Longitudinal relaxation time
- **T2 relaxation** - Transverse relaxation time
- **T2* relaxation** - Effective transverse relaxation (includes field inhomogeneity)
- **Magnetization transfer (MT)** - Macromolecular content
- **Diffusion** - Water diffusion properties (DTI, DKI, NODDI)
- **B0/B1 mapping** - Field inhomogeneity characterization

These quantitative measurements enable cross-scanner comparison, longitudinal monitoring, and tissue characterization with physical units.

## Key Features

- **20+ qMRI methods** - T1, T2, MT, diffusion, relaxometry, and more
- **Multiple implementations per method** - Compare different approaches
- **Interactive GUI** - User-friendly interface for exploration
- **Command-line scripting** - Batch processing and reproducible pipelines
- **MATLAB and Octave support** - Free alternative to MATLAB
- **Python wrapper** - Integration with Python workflows
- **Docker containers** - Reproducible computational environment
- **Protocol optimization** - Cramér-Rao Lower Bound (CRLB) analysis
- **Monte Carlo simulations** - Assess parameter estimation accuracy
- **BIDS compatibility** - Support for standardized data format
- **Jupyter notebooks** - Interactive tutorials and documentation
- **Open-source community** - Active development and contributions
- **Extensive documentation** - Method descriptions, tutorials, examples
- **Publication-ready figures** - High-quality visualization tools
- **Cross-platform** - Windows, macOS, Linux support

## Installation

### MATLAB Installation

**Prerequisites:** MATLAB R2016b or newer

```matlab
% Method 1: Clone from GitHub (recommended for latest version)
% In terminal:
cd ~/software
git clone https://github.com/qMRLab/qMRLab.git
cd qMRLab

% In MATLAB:
cd ~/software/qMRLab
startup  % Adds qMRLab to path and checks dependencies
```

### Octave Installation (Free Alternative)

```bash
# Install Octave first
# Ubuntu/Debian:
sudo apt-get install octave octave-signal octave-statistics octave-optim

# macOS (using Homebrew):
brew install octave

# Download qMRLab
cd ~/software
git clone https://github.com/qMRLab/qMRLab.git
cd qMRLab

# Start Octave and run startup
octave
>> startup
```

### Python Installation

```bash
# Install qMRLab Python wrapper
pip install qmrlab

# Or install from source
git clone https://github.com/qMRLab/qMRLab.git
cd qMRLab/python
pip install -e .
```

### Docker Installation (Recommended for Reproducibility)

```bash
# Pull qMRLab Docker image
docker pull qmrlab/octave:latest

# Run qMRLab in container
docker run -it --rm \
  -v $(pwd):/data \
  qmrlab/octave:latest \
  octave

# Or use Jupyter notebook interface
docker run -it --rm \
  -p 8888:8888 \
  -v $(pwd):/data \
  qmrlab/jupyter:latest
```

### Verify Installation

```matlab
% Start qMRLab GUI
qMRLab

% Check available methods
qMRLab_methods = methods('qMRLab');

% Test a simple method
help vfa_t1  % Variable flip angle T1 mapping
```

## Supported qMRI Techniques

qMRLab implements numerous methods organized by category:

### T1 Mapping

```matlab
% Variable Flip Angle (VFA)
vfa_t1

% Inversion Recovery (IR)
inversion_recovery

% MP2RAGE
mp2rage

% DESPOT1
despot1

% Magnetization Prepared 2 Rapid Gradient Echo
% (covered by mp2rage)
```

### T2 Mapping

```matlab
% Multi-echo spin echo
mese

% Multi-component T2 analysis
t2_mwf  % Myelin water fraction
```

### Magnetization Transfer

```matlab
% MT Ratio
mt_ratio

% MT Saturation
mt_sat

% Quantitative MT (qMT)
qmt_spgr  % Using SPGR
qmt_sirfse  % Using IR-FSE
```

### Diffusion MRI

```matlab
% Diffusion Tensor Imaging
dti

% Diffusion Kurtosis Imaging
dki

% NODDI (Neurite Orientation Dispersion and Density Imaging)
noddi

% CHARMED
charmed
```

### B1 Mapping

```matlab
% Double angle method
b1_dam

% Actual flip angle imaging
b1_afi
```

## Basic Usage

### GUI Workflow

Launch the GUI and process data interactively:

```matlab
% Start qMRLab
qMRLab

% GUI workflow:
% 1. Select method from dropdown (e.g., "vfa_t1")
% 2. Load data files using "Data" panel
% 3. Adjust fitting options in "Options" panel
% 4. Click "Fit Data" button
% 5. View results and export maps
```

### Command-Line Processing

```matlab
% Example: Variable Flip Angle T1 mapping

% 1. Create model object
Model = vfa_t1;

% 2. Set protocol parameters
Model.Prot.VFAData.Mat = [3; 20];  % Flip angles in degrees
Model.Prot.VFAData.Format = {'FlipAngle'};
Model.Prot.TimingTable.Mat = [15; 15];  % TR in ms

% 3. Load data
data = struct();
data.VFAData = load_nii_data('vfa_3deg.nii.gz');
data.VFAData(:,:,:,2) = load_nii_data('vfa_20deg.nii.gz');

% 4. Fit the model
FitResults = FitData(data, Model, 0);  % 0 = no GUI

% 5. Access results
T1map = FitResults.T1;  % T1 in seconds
M0map = FitResults.M0;  % Proton density

% 6. Save results
save_nii_data(T1map, 'T1map.nii.gz', 'vfa_3deg.nii.gz');
```

## T1 Mapping

### Variable Flip Angle (VFA) Method

Fast T1 mapping using gradient echo with multiple flip angles:

```matlab
% Load VFA T1 model
Model = vfa_t1;

% View default options
Model.options

% Set acquisition parameters
Model.Prot.VFAData.Mat = [2; 5; 10; 15; 20; 25];  % Flip angles (degrees)
Model.Prot.TimingTable.Mat = [20; 20; 20; 20; 20; 20];  % TR (ms)

% Load multi-flip angle data
data = struct();
for i = 1:6
    filename = sprintf('vfa_%02ddeg.nii.gz', Model.Prot.VFAData.Mat(i));
    data.VFAData(:,:,:,i) = load_nii_data(filename);
end

% Optional: Load B1 map for correction
data.B1map = load_nii_data('B1map.nii.gz');

% Fit model
FitResults = FitData(data, Model, 0);

% Results
T1_ms = FitResults.T1 * 1000;  % Convert to ms
M0 = FitResults.M0;

% Display typical brain values
% Gray matter: ~1500 ms
% White matter: ~900 ms
% CSF: ~4000 ms

% Save
save_nii_data(T1_ms, 'T1map_VFA.nii.gz', 'vfa_02deg.nii.gz');
```

### Inversion Recovery Method

Gold standard T1 mapping:

```matlab
% Load inversion recovery model
Model = inversion_recovery;

% Set inversion times
Model.Prot.IRData.Mat = [50; 200; 500; 1000; 2000; 4000];  % TI in ms

% Load data
data = struct();
for i = 1:length(Model.Prot.IRData.Mat)
    filename = sprintf('IR_TI%04d.nii.gz', Model.Prot.IRData.Mat(i));
    data.IRData(:,:,:,i) = load_nii_data(filename);
end

% Fit with different algorithms
Model.options.Algorithm = 'Non-linear fit (3-parameter)';
% Options: 'Non-linear fit (3-parameter)', 'Non-linear fit (2-parameter)'

FitResults = FitData(data, Model, 0);
T1_IR = FitResults.T1 * 1000;  % T1 in ms

save_nii_data(T1_IR, 'T1map_IR.nii.gz', 'IR_TI0050.nii.gz');
```

### MP2RAGE Method

Efficient T1 mapping at high field:

```matlab
% Load MP2RAGE model
Model = mp2rage;

% Set sequence parameters (example for 7T)
Model.Prot.TimingTable.Mat = [
    7.7,   % TR (ms)
    3.5    % TE (ms)
];
Model.Prot.MP2RAGEData.Mat = [
    800,   % TI1 (ms)
    2700,  % TI2 (ms)
    4,     % FA1 (degrees)
    5      % FA2 (degrees)
];

% Load data
data.MP2RAGEimg = load_nii_data('mp2rage_uni.nii.gz');  % Unified image

% Fit
FitResults = FitData(data, Model, 0);
T1_MP2RAGE = FitResults.T1 * 1000;

save_nii_data(T1_MP2RAGE, 'T1map_MP2RAGE.nii.gz', 'mp2rage_uni.nii.gz');
```

## T2 Mapping

### Multi-Echo Spin Echo (MESE)

```matlab
% Load MESE model
Model = mese;

% Set echo times
Model.Prot.MESEData.Mat = [10; 20; 30; 40; 50; 60; 70; 80];  % TE in ms

% Load data
data = struct();
for i = 1:length(Model.Prot.MESEData.Mat)
    filename = sprintf('MESE_echo%d.nii.gz', i);
    data.MESEData(:,:,:,i) = load_nii_data(filename);
end

% Fit model
FitResults = FitData(data, Model, 0);
T2_map = FitResults.T2 * 1000;  % T2 in ms

% Typical brain values at 3T:
% Gray matter: ~80-100 ms
% White matter: ~70-80 ms
% CSF: ~2000 ms

save_nii_data(T2_map, 'T2map.nii.gz', 'MESE_echo1.nii.gz');
```

### Myelin Water Fraction (MWF)

Multi-component T2 analysis for myelin:

```matlab
% Load MWF model
Model = mwf;

% Set protocol (32 echoes typical)
Model.Prot.MWData.Mat = (10:10:320)';  % TE: 10-320 ms in 10 ms steps

% Load multi-echo data
data.MWData = load_nii_data('multi_echo_T2.nii.gz');  % 4D volume

% Set fitting options
Model.options.MinReflex = 10;   % Minimum T2 for myelin (ms)
Model.options.MaxReflex = 40;   % Maximum T2 for myelin (ms)
Model.options.MinIEW = 40;      % Minimum T2 for intra/extracellular water
Model.options.MaxIEW = 200;     % Maximum T2 for IE water

% Fit (can be slow)
FitResults = FitData(data, Model, 0);

% Results
MWF = FitResults.MWF;  % Myelin water fraction (0-1)

% Typical values:
% White matter: 0.1-0.2
% Gray matter: 0.02-0.05

save_nii_data(MWF, 'MWF_map.nii.gz', 'multi_echo_T2.nii.gz');
```

## Magnetization Transfer

### MT Ratio (MTR)

Simple but useful MT measure:

```matlab
% Load MTR model
Model = mt_ratio;

% Load data
data.MToff = load_nii_data('MT_off.nii.gz');  % Without MT pulse
data.MTon = load_nii_data('MT_on.nii.gz');     % With MT pulse

% Compute MTR
FitResults = FitData(data, Model, 0);
MTR = FitResults.MTR;  % MT ratio (0-100%)

% Typical brain values:
% White matter: 40-50%
% Gray matter: 30-40%
% Lesions: reduced MTR

save_nii_data(MTR, 'MTR_map.nii.gz', 'MT_off.nii.gz');
```

### MT Saturation (MTsat)

Corrected MT measure accounting for T1:

```matlab
% Load MTsat model
Model = mt_sat;

% Set protocol
Model.Prot.MTData.Mat = [
    6,   % PDw flip angle
    21,  % T1w flip angle
    6    % MTw flip angle
];
Model.Prot.TimingTable.Mat = [
    24,  % PDw TR (ms)
    19,  % T1w TR (ms)
    24   % MTw TR (ms)
];

% Load data
data.PDw = load_nii_data('PDw.nii.gz');
data.T1w = load_nii_data('T1w.nii.gz');
data.MTw = load_nii_data('MTw.nii.gz');

% Optional: B1 map
data.B1map = load_nii_data('B1map.nii.gz');

% Fit
FitResults = FitData(data, Model, 0);

MTsat = FitResults.MTsat;  % MT saturation (p.u.)
T1 = FitResults.T1;        % T1 (s)
PD = FitResults.PD;        % Proton density

save_nii_data(MTsat, 'MTsat_map.nii.gz', 'MTw.nii.gz');
```

### Quantitative MT (qMT)

Full two-pool model fitting:

```matlab
% Load qMT model
Model = qmt_spgr;

% Set protocol: MT-weighted SPGR with various offsets and powers
Model.Prot.MTData.Mat = [
    % Columns: Offset(Hz), Angle(deg), Pulse Duration(ms), TR(ms)
    1000, 400, 12, 30;
    2000, 400, 12, 30;
    5000, 400, 12, 30;
    10000, 400, 12, 30;
    20000, 400, 12, 30;
    % ... more offset frequencies
];

% Load data (each row is one MT-weighted image)
data.MTData = zeros(128, 128, 60, size(Model.Prot.MTData.Mat, 1));
for i = 1:size(Model.Prot.MTData.Mat, 1)
    filename = sprintf('qMT_offset%d.nii.gz', i);
    data.MTData(:,:,:,i) = load_nii_data(filename);
end

% Optional: Load T1 map from separate acquisition
data.T1 = load_nii_data('T1map.nii.gz') / 1000;  % Convert to seconds

% Fit qMT model (slow - consider ROI or parallel processing)
FitResults = FitData(data, Model, 0);

% Results: two-pool model parameters
F = FitResults.F;          % Bound pool fraction (0-0.3)
kf = FitResults.kf;        % Forward exchange rate (s^-1)
R1f = FitResults.R1f;      % Free pool R1 (s^-1)
T2f = FitResults.T2f;      % Free pool T2 (ms)
T2r = FitResults.T2r;      % Restricted pool T2 (μs)

save_nii_data(F, 'qMT_BPF.nii.gz', 'qMT_offset1.nii.gz');
```

## Diffusion MRI

### DTI (Diffusion Tensor Imaging)

```matlab
% Load DTI model
Model = dti;

% Load data (assumes FSL format)
data.DWI = load_nii_data('dwi.nii.gz');        % 4D DWI data
data.bvecs = 'dwi.bvec';                       % b-vectors
data.bvals = 'dwi.bval';                       % b-values

% Or set protocol manually
Model.Prot.DWI.Mat = [
    % bx, by, bz for each direction
    0, 0, 0;          % b=0
    1, 0, 0;          % direction 1
    0, 1, 0;          % direction 2
    % ... more directions
];
Model.Prot.DWI.bval = [0; 1000; 1000; ...];  % b-values

% Fit
FitResults = FitData(data, Model, 0);

% DTI metrics
FA = FitResults.FA;    % Fractional anisotropy (0-1)
MD = FitResults.MD;    % Mean diffusivity (mm²/s)
AD = FitResults.AD;    % Axial diffusivity
RD = FitResults.RD;    % Radial diffusivity

% Typical white matter values:
% FA: 0.4-0.7
% MD: 0.7-0.9 × 10^-3 mm²/s

save_nii_data(FA, 'DTI_FA.nii.gz', 'dwi.nii.gz');
save_nii_data(MD, 'DTI_MD.nii.gz', 'dwi.nii.gz');
```

### NODDI

Neurite Orientation Dispersion and Density Imaging:

```matlab
% Load NODDI model
Model = noddi;

% NODDI requires multi-shell data
% Typical: b=300, 700, 1000, 2000 s/mm²

% Load data
data.DWI = load_nii_data('noddi_dwi.nii.gz');
data.bvecs = 'noddi.bvec';
data.bvals = 'noddi.bval';

% Optional: mask to speed up
data.Mask = load_nii_data('brain_mask.nii.gz');

% Set options
Model.options.model = 'WatsonSHStickTortIsoV_B0';

% Fit (very slow - use parallel processing)
FitResults = FitData(data, Model, 1);  % 1 = use parallel toolbox

% NODDI metrics
ICVF = FitResults.ficvf;   % Intracellular volume fraction
ISOVF = FitResults.fiso;   % Isotropic volume fraction (free water)
OD = FitResults.odi;       % Orientation dispersion index

% Typical white matter:
% ICVF: 0.6-0.8
% OD: 0.1-0.4

save_nii_data(ICVF, 'NODDI_ICVF.nii.gz', 'noddi_dwi.nii.gz');
save_nii_data(OD, 'NODDI_OD.nii.gz', 'noddi_dwi.nii.gz');
```

## Protocol Optimization

### Cramér-Rao Lower Bound (CRLB) Analysis

Optimize acquisition protocols:

```matlab
% Example: Optimize VFA protocol for T1 mapping

Model = vfa_t1;

% Current protocol
Model.Prot.VFAData.Mat = [3; 20];  % 2 flip angles
Model.Prot.TimingTable.Mat = [15; 15];  % TR = 15 ms

% Set tissue parameters for simulation
Model.options.T1 = 900e-3;  % WM T1 = 900 ms

% Compute CRLB
Model = Model.Sim_Optimize_Protocol(Model);

% View results
fprintf('T1 CRLB: %.2f%%\n', Model.Sim.CRLB.T1);

% Test different protocols
protocols = {
    [3, 20],
    [5, 15],
    [3, 10, 20],
    [3, 10, 15, 20]
};

for i = 1:length(protocols)
    Model.Prot.VFAData.Mat = protocols{i}(:);
    Model.Prot.TimingTable.Mat = 15 * ones(size(protocols{i}(:)));
    Model = Model.Sim_Optimize_Protocol(Model);
    fprintf('Protocol %d: CRLB = %.2f%%\n', i, Model.Sim.CRLB.T1);
end

% Result: More flip angles improve precision but increase scan time
```

### Monte Carlo Simulation

Assess parameter estimation accuracy:

```matlab
% Simulate VFA T1 mapping with noise

Model = vfa_t1;
Model.Prot.VFAData.Mat = [3; 10; 20];
Model.Prot.TimingTable.Mat = [15; 15; 15];

% True tissue parameters
True_T1 = 0.9;  % 900 ms
True_M0 = 1000;

% Generate signal
signal = Model.equation(Model.Prot, [True_T1, True_M0]);

% Add noise and repeat fits
n_iterations = 100;
T1_estimates = zeros(n_iterations, 1);
SNR = 50;

for i = 1:n_iterations
    % Add Rician noise
    noise_level = mean(signal) / SNR;
    noisy_signal = abs(signal + noise_level * randn(size(signal)) + ...
                       1i * noise_level * randn(size(signal)));

    % Fit
    FitResults = Model.fit(noisy_signal);
    T1_estimates(i) = FitResults.T1;
end

% Analyze results
mean_T1 = mean(T1_estimates);
std_T1 = std(T1_estimates);
bias = mean_T1 - True_T1;

fprintf('True T1: %.3f s\n', True_T1);
fprintf('Mean estimated T1: %.3f ± %.3f s\n', mean_T1, std_T1);
fprintf('Bias: %.3f s (%.1f%%)\n', bias, 100*bias/True_T1);
fprintf('Coefficient of variation: %.1f%%\n', 100*std_T1/mean_T1);

% Plot distribution
figure;
histogram(T1_estimates * 1000, 20);
xlabel('T1 (ms)');
ylabel('Count');
title('T1 estimation distribution');
hold on;
xline(True_T1 * 1000, 'r--', 'LineWidth', 2);
legend('Estimates', 'True value');
```

## Batch Processing

Process multiple subjects:

```matlab
% Batch T1 mapping for multiple subjects

subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'};
bids_dir = '/path/to/bids/dataset';

% Initialize model
Model = vfa_t1;
Model.Prot.VFAData.Mat = [3; 20];
Model.Prot.TimingTable.Mat = [15; 15];

% Loop over subjects
for i = 1:length(subjects)
    subj = subjects{i};
    fprintf('Processing %s...\n', subj);

    % Load data
    data = struct();
    data.VFAData(:,:,:,1) = load_nii_data(fullfile(bids_dir, subj, 'anat', ...
        sprintf('%s_acq-vfa3_T1w.nii.gz', subj)));
    data.VFAData(:,:,:,2) = load_nii_data(fullfile(bids_dir, subj, 'anat', ...
        sprintf('%s_acq-vfa20_T1w.nii.gz', subj)));

    % Fit
    try
        FitResults = FitData(data, Model, 0);

        % Save
        outfile = fullfile(bids_dir, subj, 'anat', sprintf('%s_T1map.nii.gz', subj));
        save_nii_data(FitResults.T1 * 1000, outfile, ...
            fullfile(bids_dir, subj, 'anat', sprintf('%s_acq-vfa3_T1w.nii.gz', subj)));

        fprintf('  Success! Saved to %s\n', outfile);
    catch ME
        fprintf('  Error: %s\n', ME.message);
    end
end
```

## Integration with Claude Code

qMRLab integrates well with Claude-assisted workflows:

### Protocol Design Assistant

```markdown
**Prompt to Claude:**
"Design an optimal VFA T1 mapping protocol for 3T. I want to:
1. Map T1 in brain white matter (T1 ~ 900 ms)
2. Keep total scan time under 5 minutes
3. Achieve <5% coefficient of variation
4. Use SPGR sequence with TR = 15 ms

Use qMRLab's CRLB analysis to find the best flip angles."

Claude can generate complete simulation scripts and analyze trade-offs.
```

### Method Comparison Studies

```markdown
**Prompt to Claude:**
"Compare three T1 mapping methods using qMRLab:
- Variable Flip Angle (VFA)
- Inversion Recovery (IR)
- MP2RAGE

For each method, report:
- Scan time
- T1 accuracy (bias)
- Precision (CV%)
- Sensitivity to B1 errors

Use Monte Carlo simulation with SNR=50."
```

### Batch Processing Helper

```markdown
**Prompt to Claude:**
"Create a qMRLab batch processing script that:
1. Processes 50 subjects in BIDS format
2. Computes MTR and MTsat for each
3. Includes error handling and logging
4. Generates QC report with parameter distributions
5. Uses parallel processing for speed"
```

## Integration with Other Tools

### BIDS Compatibility

```bash
# qMRLab supports BIDS format
# Example BIDS structure for qMRI:

dataset/
  sub-01/
    anat/
      sub-01_acq-vfa3deg_T1w.nii.gz
      sub-01_acq-vfa3deg_T1w.json
      sub-01_acq-vfa20deg_T1w.nii.gz
      sub-01_acq-vfa20deg_T1w.json
      sub-01_T1map.nii.gz
```

### FSL Integration

```matlab
% Use FSL-preprocessed diffusion data

% After eddy correction and topup:
data.DWI = load_nii_data('eddy_corrected.nii.gz');
data.bvecs = 'eddy_corrected.eddy_rotated_bvecs';
data.bvals = 'bvals';
data.Mask = load_nii_data('brain_mask.nii.gz');

Model = dti;
FitResults = FitData(data, Model, 0);
```

### ANTs Registration

```bash
# Register qMRI maps to template

antsRegistrationSyN.sh \
  -d 3 \
  -f MNI152_T1_1mm.nii.gz \
  -m T1map.nii.gz \
  -o T1map_to_MNI_

# Apply transform to other maps
antsApplyTransforms \
  -d 3 \
  -i MTsat_map.nii.gz \
  -r MNI152_T1_1mm.nii.gz \
  -t T1map_to_MNI_1Warp.nii.gz \
  -t T1map_to_MNI_0GenericAffine.mat \
  -o MTsat_map_MNI.nii.gz
```

### Python Workflow

```python
# Use qMRLab from Python

import qmrlab
import nibabel as nib
import numpy as np

# Load model
model = qmrlab.vfa_t1()

# Set protocol
model.Prot.VFAData.Mat = np.array([3, 20])
model.Prot.TimingTable.Mat = np.array([15, 15])

# Load data
img1 = nib.load('vfa_3deg.nii.gz')
img2 = nib.load('vfa_20deg.nii.gz')

data = {
    'VFAData': np.stack([img1.get_fdata(), img2.get_fdata()], axis=-1)
}

# Fit
results = model.fit(data)

# Save
T1_img = nib.Nifti1Image(results['T1'], img1.affine)
nib.save(T1_img, 'T1map.nii.gz')
```

## Troubleshooting

### Problem 1: Octave Compatibility Issues

**Symptoms:** Functions not working in Octave

**Solutions:**
```bash
# Install required Octave packages
pkg install -forge struct
pkg install -forge io
pkg install -forge statistics
pkg install -forge optim

# Load packages
pkg load struct io statistics optim
```

### Problem 2: Fitting Failures

**Symptoms:** "Fitting failed" or NaN in results

**Solutions:**
```matlab
% Check data quality
data_min = min(data.VFAData(:));
data_max = max(data.VFAData(:));
fprintf('Data range: %.1f to %.1f\n', data_min, data_max);

% Use mask to exclude background
data.Mask = data.VFAData(:,:,:,1) > 100;

% Try different starting values
Model.options.StartPoint = [1.0, 1000];  % [T1(s), M0]

% Use different fitting algorithm
Model.options.Algorithm = 'Trust-Region';
```

### Problem 3: Slow Processing

**Symptoms:** Fitting takes too long

**Solutions:**
```matlab
% Use brain mask
data.Mask = load_nii_data('brain_mask.nii.gz');

% Enable parallel processing
Model.options.UseParallel = true;

% Reduce iterations for complex models
Model.options.MaxIter = 100;  % Default: 500

% Process ROI first to test
roi_data = data;
roi_data.MTData = data.MTData(50:80, 50:80, 30, :);
FitResults_roi = FitData(roi_data, Model, 0);
```

### Problem 4: Memory Errors

**Symptoms:** Out of memory during large dataset processing

**Solutions:**
```matlab
% Process slice-by-slice
n_slices = size(data.VFAData, 3);
T1map = zeros(size(data.VFAData(:,:,:,1)));

for slice = 1:n_slices
    fprintf('Slice %d/%d\n', slice, n_slices);

    slice_data = struct();
    slice_data.VFAData = squeeze(data.VFAData(:,:,slice,:));

    FitResults = FitData(slice_data, Model, 0);
    T1map(:,:,slice) = FitResults.T1;
end
```

### Problem 5: Protocol Loading Errors

**Symptoms:** Cannot load protocol file

**Solutions:**
```matlab
% Check protocol format
% qMRLab expects specific structure

% Manually create protocol
Model = vfa_t1;
Model.Prot.VFAData.Mat = [3; 20];  % Must be column vector
Model.Prot.TimingTable.Mat = [15; 15];  % Must match size

% Verify protocol
Model.Prot

% Save protocol for reuse
save('vfa_protocol.mat', 'Model');
```

### Problem 6: Unrealistic Parameter Values

**Symptoms:** T1 > 10 seconds or negative values

**Solutions:**
```matlab
% Add constraints
Model.options.LowerBound = [0.1, 0];      % Minimum T1=100ms, M0=0
Model.options.UpperBound = [5.0, 10000];  % Maximum T1=5s

% Check SNR
signal = mean(data.VFAData(mask));
noise = std(data.VFAData(~mask));
SNR = signal / noise;
fprintf('SNR: %.1f\n', SNR);
% If SNR < 20, results may be unreliable

% Filter results
T1map_filtered = T1map;
T1map_filtered(T1map < 0.1 | T1map > 5) = NaN;
```

## Best Practices

### Data Acquisition

1. **Follow recommended protocols** - Check qMRLab documentation for validated sequences
2. **Acquire sufficient contrasts** - More data points improve precision
3. **Maintain high SNR** - Target SNR > 50 for quantitative accuracy
4. **Use B1 correction** - Essential at high field (≥3T)
5. **Acquire calibration data** - B0/B1 maps, phantoms for validation

### Processing Workflow

1. **Start with GUI** - Understand method before scripting
2. **Test on ROI first** - Verify parameters before whole-brain fitting
3. **Use brain masks** - Speed up processing, avoid edge artifacts
4. **Check intermediate results** - Visualize signal vs. model fit
5. **Save processing parameters** - Document for reproducibility

### Method Selection

1. **Consider scan time** - IR methods: slow but accurate; VFA: fast but needs B1 correction
2. **Match to research question** - MTR for sensitivity; qMT for specificity
3. **Validate with phantoms** - Especially for cross-site studies
4. **Compare multiple methods** - Use qMRLab's multiple implementations
5. **Read method papers** - Understand assumptions and limitations

### Quality Control

1. **Visual inspection mandatory** - Check for artifacts, motion
2. **Compare to literature** - Ensure values in expected range
3. **Plot parameter distributions** - Identify outliers
4. **Check model fit quality** - Look at residuals
5. **Validate with known phantom** - Regular QC for longitudinal studies

### Reproducibility

1. **Use Docker containers** - Freeze computational environment
2. **Document all parameters** - Save Model object with results
3. **Version control scripts** - Track processing changes
4. **Share code and data** - Enable replication
5. **Report qMRLab version** - Important for reproducibility

## Resources

### Official Documentation

- **qMRLab Website:** https://qmrlab.org/
- **GitHub Repository:** https://github.com/qMRLab/qMRLab
- **Documentation:** https://qmrlab.readthedocs.io/
- **Interactive Tutorials:** https://qmrlab.org/tutorials (Jupyter notebooks)
- **Method Descriptions:** https://qmrlab.org/methods

### Key Publications

- **qMRLab Paper:** Karakuzu et al. (2020) "qMRLab: Quantitative MRI analysis, under one umbrella" JOSS
- **Review Article:** Cercignani et al. (2018) "Brain microstructure by multi-modal MRI" Neuroimage
- **Best Practices:** Keenan et al. (2019) "Quantitative magnetic resonance imaging phantoms" WIREs

### Learning Resources

- **Video Tutorials:** https://www.youtube.com/c/qMRLab
- **Example Datasets:** https://osf.io/tmdfu/ (qMRLab OSF repository)
- **ISMRM Lectures:** https://www.ismrm.org/workshops/ (quantitative MRI workshops)
- **qMRI-MOOC:** Online course on quantitative MRI

### Community Support

- **GitHub Discussions:** https://github.com/qMRLab/qMRLab/discussions
- **Gitter Chat:** https://gitter.im/qMRLab/qMRLab
- **Issue Tracker:** https://github.com/qMRLab/qMRLab/issues
- **Forum:** https://forum.qmrlab.org/

## Citation

```bibtex
@article{Karakuzu2020,
  title = {qMRLab: Quantitative MRI analysis, under one umbrella},
  author = {Karakuzu, Agah and Boudreau, Mathieu and Duval, Tanguy and
            Boshkovski, Tommy and Leppert, Ilana R and Cabana, Jean-François and
            Gagnon, Ian and Beliveau, Philippe and Pike, G Bruce and
            Cohen-Adad, Julien and Stikov, Nikola},
  journal = {Journal of Open Source Software},
  volume = {5},
  number = {53},
  pages = {2343},
  year = {2020},
  doi = {10.21105/joss.02343}
}
```

## Related Tools

- **hMRI Toolbox** - SPM-based multi-parametric mapping (R1, R2*, MT, PD)
- **QUIT** - C++ quantitative imaging tools for high performance
- **FSL** - FMRIB Software Library with diffusion tools
- **MRtrix3** - Advanced diffusion MRI processing
- **DIPY** - Diffusion imaging in Python
- **PyQMRI** - Python-based quantitative MRI framework
- **MP2RAGE** - T1 mapping tools for high field
- **NODDI Toolbox** - Official NODDI MATLAB implementation

---

**Skill Type:** Quantitative MRI Analysis
**Difficulty Level:** Intermediate to Advanced
**Prerequisites:** MATLAB or Octave (or Python), Basic MRI physics, Understanding of model fitting
**Typical Use Cases:** T1/T2 mapping, magnetization transfer, diffusion analysis, protocol optimization, method development, reproducible research
