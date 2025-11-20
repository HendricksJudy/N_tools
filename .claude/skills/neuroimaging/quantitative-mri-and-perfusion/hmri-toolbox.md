# hMRI Toolbox - Quantitative Multi-Parametric Mapping

## Overview

The hMRI Toolbox is a comprehensive MATLAB/SPM-based framework for quantitative multi-parametric mapping (MPM) of brain tissue properties. Developed at the Wellcome Centre for Human Neuroimaging (UCL), it provides standardized processing pipelines for calculating quantitative MRI parameter maps including R1 (longitudinal relaxation rate), R2* (effective transverse relaxation rate), MT (magnetization transfer saturation), and PD (proton density). The toolbox emphasizes reproducibility, standardization, and clinical translation of quantitative MRI measurements.

**Website:** https://hmri.info/
**Platform:** MATLAB/SPM (Windows/macOS/Linux)
**License:** GPL v2
**Key Application:** Quantitative tissue characterization, voxel-based quantification (VBQ)

### Physics Background

Quantitative MRI measures physical tissue properties that relate directly to brain microstructure:

- **R1 (1/T1):** Longitudinal relaxation rate, sensitive to myelination and iron content
- **R2* (1/T2*):** Effective transverse relaxation rate, sensitive to iron and myelin
- **MT:** Magnetization transfer, reflects macromolecular content and myelin
- **PD:** Proton density, relates to water content and cellularity

Unlike conventional MRI contrast, these quantitative maps have physical units and can be compared across scanners, sites, and time points, making them valuable for clinical biomarker development.

## Key Features

- **Multi-parametric mapping (MPM) protocol** - Standardized acquisition for multiple parameters
- **Quantitative parameter maps** - R1, R2*, MT, PD with physical units
- **B1 field correction** - RF transmit field inhomogeneity correction
- **Imperfect spoiling correction** - Accurate relaxation rate estimation
- **Unified segmentation integration** - Tissue classification with quantitative priors
- **Voxel-based quantification (VBQ)** - Statistical analysis of parameter maps
- **Quality control tools** - Automated QC metrics and visual inspection
- **Multi-contrast segmentation** - Improved tissue classification using all contrasts
- **Population templates** - Age-specific reference templates
- **BIDS compatibility** - Support for Brain Imaging Data Structure
- **Phantom validation tools** - Standardized QC with calibration phantoms
- **Batch processing** - Automated pipeline for multiple subjects
- **Clinical protocols** - Optimized sequences for 1.5T, 3T, 7T
- **ROI analysis tools** - Extract quantitative values from regions
- **Longitudinal processing** - Track parameter changes over time

## Installation

### Prerequisites

Install SPM12 first (hMRI Toolbox is built on SPM):

```bash
# Download SPM12
cd ~/software
wget https://www.fil.ion.ucl.ac.uk/spm/download/restricted/eldorado/spm12.zip
unzip spm12.zip
```

### Install hMRI Toolbox

**Method 1: Download from GitHub**

```bash
# Clone the repository
cd ~/software
git clone https://github.com/hMRI-group/hMRI-toolbox.git

# Add to MATLAB path
# In MATLAB:
addpath(genpath('~/software/hMRI-toolbox'))
addpath('~/software/spm12')
savepath
```

**Method 2: Install via SPM Extension Manager**

```matlab
% In MATLAB with SPM12
spm
% Click on "Extensions" -> "hMRI Toolbox"
% Follow installation prompts
```

**Method 3: Manual Download**

1. Visit https://hmri.info/
2. Download latest release
3. Extract to MATLAB toolbox directory
4. Add to MATLAB path

### Verify Installation

```matlab
% Start MATLAB and check
spm
% Look for "hMRI Tools" in the SPM menu
% Or test directly:
hmri_get_defaults
```

## Data Acquisition Requirements

### Multi-Parametric Mapping (MPM) Protocol

hMRI Toolbox requires specific multi-echo 3D FLASH acquisitions:

**Three Contrasts Required:**
1. **PDw** - Proton density-weighted (low flip angle, long TR)
2. **T1w** - T1-weighted (high flip angle, short TR)
3. **MTw** - Magnetization transfer-weighted (MT pulse + PDw parameters)

**Additional Acquisitions (Recommended):**
- **B1 map** - RF transmit field map (for correction)
- **B0 map** - Static field map (for distortion correction)

### Example Protocol Parameters (3T)

```matlab
% PDw acquisition
TR = 24 ms
TE = [2.46, 4.92, 7.38, 9.84, 12.30, 14.76, 17.22, 19.68] ms  % 8 echoes
FA = 6 degrees
Resolution = 1 mm isotropic

% T1w acquisition
TR = 19 ms
TE = [2.46, 4.92, 7.38, 9.84, 12.30, 14.76, 17.22, 19.68] ms
FA = 21 degrees
Resolution = 1 mm isotropic

% MTw acquisition
TR = 24 ms
TE = [2.46, 4.92, 7.38, 9.84, 12.30, 14.76, 17.22, 19.68] ms
FA = 6 degrees
MT pulse offset = 2 kHz
Resolution = 1 mm isotropic

% Total scan time: ~20-25 minutes
```

## Basic Processing Pipeline

### Step 1: Configure hMRI Defaults

```matlab
% Create hmri_local_defaults.m in your project directory
function hmri_local_defaults
global hmri_def

% Set your site/scanner specific defaults
hmri_def.centre = 'MyInstitution';
hmri_def.customised = struct();
end

% Load defaults
hmri_get_defaults
```

### Step 2: DICOM Import

```matlab
% Import DICOM to NIfTI
matlabbatch{1}.spm.tools.hmri.hmri_config.hmri_setdef = struct([]);
matlabbatch{2}.spm.tools.hmri.dicom.data = {
    '/path/to/dicom/PDw'
    '/path/to/dicom/T1w'
    '/path/to/dicom/MTw'
    '/path/to/dicom/B1map'
    '/path/to/dicom/B0map'
};
matlabbatch{2}.spm.tools.hmri.dicom.root = 'flat';
matlabbatch{2}.spm.tools.hmri.dicom.outdir = {'/path/to/nifti'};

spm_jobman('run', matlabbatch);
```

### Step 3: Create Maps - Full Pipeline

```matlab
% Complete hMRI processing pipeline
clear matlabbatch;

% 1. Configure defaults
matlabbatch{1}.spm.tools.hmri.hmri_config.hmri_setdef.customised = struct([]);

% 2. Specify input data
matlabbatch{2}.spm.tools.hmri.create_mpm.subj.output.indir = 'yes';
matlabbatch{2}.spm.tools.hmri.create_mpm.subj.sensitivity.RF_once = {''};

% PDw images (all echoes)
matlabbatch{2}.spm.tools.hmri.create_mpm.subj.raw_mpm.PDw = {
    '/path/to/PDw_echo1.nii'
    '/path/to/PDw_echo2.nii'
    '/path/to/PDw_echo3.nii'
    % ... all echoes
};

% T1w images (all echoes)
matlabbatch{2}.spm.tools.hmri.create_mpm.subj.raw_mpm.T1w = {
    '/path/to/T1w_echo1.nii'
    '/path/to/T1w_echo2.nii'
    '/path/to/T1w_echo3.nii'
    % ... all echoes
};

% MTw images (all echoes)
matlabbatch{2}.spm.tools.hmri.create_mpm.subj.raw_mpm.MTw = {
    '/path/to/MTw_echo1.nii'
    '/path/to/MTw_echo2.nii'
    '/path/to/MTw_echo3.nii'
    % ... all echoes
};

% B1 map (optional but recommended)
matlabbatch{2}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI = {
    '/path/to/B1map.nii'
};

% 3. Processing options
matlabbatch{2}.spm.tools.hmri.create_mpm.subj.popup = false;

% Run the batch
spm_jobman('run', matlabbatch);
```

### Expected Outputs

After processing, you'll find in the output directory:

```bash
# Quantitative parameter maps
R1_<subject>.nii           # Longitudinal relaxation rate (s⁻¹)
R2s_<subject>.nii          # Effective transverse relaxation rate (s⁻¹)
MT_<subject>.nii           # Magnetization transfer saturation (%)
PD_<subject>.nii           # Proton density (%)
A_<subject>.nii            # Transmit field amplitude (%)

# Quality metrics
<subject>_MPMcalc_QA.json  # Quality assurance metrics
```

## Quantitative Parameter Maps

### Understanding the Maps

**R1 Map (Longitudinal Relaxation Rate):**

```matlab
% Typical brain values at 3T
% Gray matter: 1.5-1.8 s⁻¹
% White matter: 1.0-1.2 s⁻¹
% CSF: 0.2-0.3 s⁻¹

% Load and inspect R1 map
V = spm_vol('R1_subject01.nii');
R1 = spm_read_vols(V);

% Display
figure; imagesc(R1(:,:,90));
colorbar; title('R1 map (s^{-1})');
caxis([0 2]);
```

**R2* Map (Effective Transverse Relaxation Rate):**

```matlab
% Typical brain values at 3T
% Gray matter: 15-20 s⁻¹
% White matter: 20-30 s⁻¹
% Iron-rich regions: 40-100 s⁻¹

% Load R2* map
V = spm_vol('R2s_subject01.nii');
R2s = spm_read_vols(V);

% Display
figure; imagesc(R2s(:,:,90));
colorbar; title('R2* map (s^{-1})');
caxis([0 50]);
```

**MT Map (Magnetization Transfer):**

```matlab
% Typical brain values
% Gray matter: 1.0-1.5 p.u.
% White matter: 2.0-3.0 p.u.
% Highly myelinated: 3.0-4.0 p.u.

% Load MT map
V = spm_vol('MT_subject01.nii');
MT = spm_read_vols(V);

% Display
figure; imagesc(MT(:,:,90));
colorbar; title('MT saturation (p.u.)');
caxis([0 4]);
```

**PD Map (Proton Density):**

```matlab
% Relative proton density (water content)
% Normalized to 100% in pure water

% Load PD map
V = spm_vol('PD_subject01.nii');
PD = spm_read_vols(V);

% Display
figure; imagesc(PD(:,:,90));
colorbar; title('Proton Density (%)');
caxis([0 100]);
```

## Advanced Features

### B1 Field Correction

B1+ (transmit field) inhomogeneities affect flip angles and must be corrected:

```matlab
% Option 1: Provide B1 map
matlabbatch{1}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI = {
    '/path/to/B1map.nii'
};

% Option 2: No B1 correction (not recommended)
matlabbatch{1}.spm.tools.hmri.create_mpm.subj.b1_type.no_B1_correction = true;

% Option 3: Use pre-calculated B1 map
matlabbatch{1}.spm.tools.hmri.create_mpm.subj.b1_type.pre_processed_B1 = {
    '/path/to/preprocessed_B1.nii'
};
```

### Imperfect Spoiling Correction

Accounts for residual transverse magnetization:

```matlab
% Enabled by default for improved R1 accuracy
% Based on Preibisch & Deichmann method

% In hmri_local_defaults.m:
hmri_def.imperfectSpoilCorr = 1;  % 1 = enabled, 0 = disabled
hmri_def.T2scorr = 1;  % R2* correction
```

### Multi-Contrast Segmentation

Use all quantitative maps for improved tissue segmentation:

```matlab
% Unified segmentation with quantitative priors
matlabbatch{1}.spm.tools.hmri.autoreor.reference = {
    'MT_subject01.nii'  % Use MT as reference
};

matlabbatch{2}.spm.spatial.preproc.channel(1).vols = {'R1_subject01.nii'};
matlabbatch{2}.spm.spatial.preproc.channel(2).vols = {'R2s_subject01.nii'};
matlabbatch{2}.spm.spatial.preproc.channel(3).vols = {'MT_subject01.nii'};
matlabbatch{2}.spm.spatial.preproc.channel(4).vols = {'PD_subject01.nii'};

% Tissue probability maps
matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {
    '/spm12/tpm/TPM.nii,1'  % Gray matter
};
matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {
    '/spm12/tpm/TPM.nii,2'  % White matter
};
matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {
    '/spm12/tpm/TPM.nii,3'  % CSF
};

spm_jobman('run', matlabbatch);
```

### Population Templates

Use age-specific templates for better normalization:

```matlab
% Available templates
% - IXI dataset based templates
% - Age ranges: 20-30, 30-40, 40-50, 50-60, 60-70, 70-80 years

% Specify template in batch
template_dir = fullfile(spm('dir'), 'toolbox', 'hMRI', 'templates');
age_template = 'R1_template_40-50yrs.nii';

matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.template = {
    fullfile(template_dir, age_template)
};
```

## Voxel-Based Quantification (VBQ)

VBQ is the statistical analysis of quantitative parameter maps across subjects:

### Preprocessing for VBQ

```matlab
% 1. Auto-reorient to MNI space
matlabbatch{1}.spm.tools.hmri.autoreor.reference = {'MT_subject01.nii'};
matlabbatch{1}.spm.tools.hmri.autoreor.template = {
    fullfile(spm('dir'), 'canonical', 'avg152T1.nii')
};
matlabbatch{1}.spm.tools.hmri.autoreor.other = {
    'R1_subject01.nii'
    'R2s_subject01.nii'
    'PD_subject01.nii'
};

% 2. Segment using unified segmentation
matlabbatch{2}.spm.spatial.preproc.channel.vols = {'rMT_subject01.nii'};

% 3. Normalize using DARTEL or standard normalization
matlabbatch{3}.spm.spatial.normalise.write.subj.def = {
    'y_rMT_subject01.nii'
};
matlabbatch{3}.spm.spatial.normalise.write.subj.resample = {
    'R1_subject01.nii'
    'R2s_subject01.nii'
    'MT_subject01.nii'
    'PD_subject01.nii'
};
matlabbatch{3}.spm.spatial.normalise.write.woptions.vox = [1 1 1];

% 4. Smooth
matlabbatch{4}.spm.spatial.smooth.data = {
    'wR1_subject01.nii'
    'wR2s_subject01.nii'
    'wMT_subject01.nii'
    'wPD_subject01.nii'
};
matlabbatch{4}.spm.spatial.smooth.fwhm = [6 6 6];

spm_jobman('run', matlabbatch);
```

### Statistical Analysis

```matlab
% Group comparison using SPM
% Example: Compare R1 between patients and controls

% Set up design matrix
matlabbatch{1}.spm.stats.factorial_design.dir = {'/path/to/stats'};
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = {
    % Control subjects (smoothed, normalized R1 maps)
    'swR1_control01.nii,1'
    'swR1_control02.nii,1'
    % ... more controls
};
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = {
    % Patient group
    'swR1_patient01.nii,1'
    'swR1_patient02.nii,1'
    % ... more patients
};

% Add covariates (age, sex, TIV)
matlabbatch{1}.spm.stats.factorial_design.cov.c = [
    55, 62, 48, ...  % ages
];
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'Age';

% Estimate model
matlabbatch{2}.spm.stats.fmri_est.spmmat = {
    '/path/to/stats/SPM.mat'
};

% Set up contrasts
matlabbatch{3}.spm.stats.con.spmmat = {'/path/to/stats/SPM.mat'};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Patients > Controls';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'Controls > Patients';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1];

% Run inference
matlabbatch{4}.spm.stats.results.spmmat = {'/path/to/stats/SPM.mat'};
matlabbatch{4}.spm.stats.results.conspec.contrasts = 1;
matlabbatch{4}.spm.stats.results.conspec.threshdesc = 'FWE';
matlabbatch{4}.spm.stats.results.conspec.thresh = 0.05;
matlabbatch{4}.spm.stats.results.conspec.extent = 0;

spm_jobman('run', matlabbatch);
```

## ROI Analysis

Extract quantitative values from anatomical regions:

```matlab
% Load parameter map and ROI mask
V_R1 = spm_vol('R1_subject01.nii');
R1_data = spm_read_vols(V_R1);

V_roi = spm_vol('hippocampus_mask.nii');
roi_mask = spm_read_vols(V_roi);

% Extract mean R1 in ROI
R1_roi = R1_data(roi_mask > 0);
mean_R1 = mean(R1_roi);
std_R1 = std(R1_roi);
median_R1 = median(R1_roi);

fprintf('Hippocampus R1: %.3f ± %.3f s^-1\n', mean_R1, std_R1);

% Extract values from multiple ROIs
atlas = spm_read_vols(spm_vol('AAL_atlas.nii'));
n_rois = max(atlas(:));

roi_stats = struct();
for roi = 1:n_rois
    mask = (atlas == roi);
    roi_stats(roi).mean_R1 = mean(R1_data(mask));
    roi_stats(roi).std_R1 = std(R1_data(mask));
    roi_stats(roi).mean_MT = mean(MT_data(mask));
end

% Save to CSV
writetable(struct2table(roi_stats), 'roi_quantitative_values.csv');
```

## Clinical Applications

### Multiple Sclerosis

```matlab
% MS lesions show altered R1, R2*, MT
% Useful for:
% - Lesion characterization
% - Normal-appearing tissue changes
% - Treatment monitoring

% Detect lesions based on MT threshold
MT_data = spm_read_vols(spm_vol('MT_patient_ms.nii'));
WM_mask = spm_read_vols(spm_vol('c2_patient_ms.nii')) > 0.9;

% Normal WM MT typically 2.0-3.0 p.u.
lesion_mask = (MT_data < 1.5) & WM_mask;

% Quantify lesion load
lesion_volume_mm3 = sum(lesion_mask(:)) * prod(V.mat(1:3));
fprintf('Lesion volume: %.1f mL\n', lesion_volume_mm3/1000);
```

### Aging Studies

```matlab
% R1 and MT decrease with age in WM
% R2* increases with age (iron accumulation)

% Correlate with age
ages = [25, 30, 35, 42, 48, 55, 61, 68, 75];
R1_values = [1.15, 1.14, 1.13, 1.10, 1.08, 1.05, 1.02, 0.99, 0.96];

[r, p] = corr(ages', R1_values');
fprintf('R1-age correlation: r=%.3f, p=%.4f\n', r, p);

% Plot
figure;
scatter(ages, R1_values, 'filled');
xlabel('Age (years)');
ylabel('White Matter R1 (s^{-1})');
title('R1 decreases with age');
lsline;
```

### Neurodegenerative Diseases

```matlab
% Alzheimer's disease: hippocampal R1/MT changes
% Parkinson's disease: substantia nigra R2* changes
% Huntington's disease: striatal changes

% Compare patient to normative database
patient_R1_hippocampus = 1.45;
control_mean = 1.62;
control_std = 0.08;

z_score = (patient_R1_hippocampus - control_mean) / control_std;
fprintf('Patient z-score: %.2f (p < 0.001)\n', z_score);
```

## Integration with Claude Code

hMRI Toolbox integrates naturally into Claude-assisted neuroimaging workflows:

### Automated Pipeline Generation

```markdown
**Prompt to Claude:**
"Generate an hMRI processing pipeline for 20 subjects with PDw, T1w, MTw images
in BIDS format. Include B1 correction, VBQ preprocessing, and group analysis
comparing young vs. elderly on R1 and MT."

Claude can generate complete MATLAB batch scripts with proper file handling,
error checking, and parallel processing.
```

### Quality Control Automation

```markdown
**Prompt to Claude:**
"Create a quality control script for hMRI outputs that:
1. Checks parameter value ranges (R1: 0-3, R2*: 0-100, MT: 0-5)
2. Generates thumbnail images for visual inspection
3. Computes SNR and CNR metrics
4. Flags outliers based on mean ROI values
5. Generates an HTML report"
```

### Statistical Analysis Helper

```markdown
**Prompt to Claude:**
"Set up VBQ analysis in SPM for comparing R1 maps between:
- Group 1: 25 controls (mean age 35)
- Group 2: 25 patients (mean age 38)
Include age and sex as covariates, use FWE correction p<0.05,
and generate results tables for significant clusters."
```

## Integration with Other Tools

### SPM12

hMRI Toolbox is built on SPM12:

```matlab
% Use SPM's unified segmentation
% Use SPM's normalization
% Use SPM's statistical analysis
% Access all SPM utilities

% Example: Use SPM's Check Reg
spm_check_registration('R1_subject01.nii', 'MT_subject01.nii', ...
                       'c1MT_subject01.nii', 'c2MT_subject01.nii');
```

### CAT12

Combine with CAT12 for surface-based VBQ:

```matlab
% 1. Create parameter maps with hMRI
% 2. Project to surface with CAT12

% Project R1 to surface
matlabbatch{1}.spm.tools.cat.stools.vol2surf.data_vol = {
    'wR1_subject01.nii'
};
matlabbatch{1}.spm.tools.cat.stools.vol2surf.data_mesh_lh = {
    'lh.central.subject01.gii'
};
matlabbatch{1}.spm.tools.cat.stools.vol2surf.sample = {'maxabs'};

% 3. Surface-based statistics in CAT12
```

### FreeSurfer

Use FreeSurfer surfaces with hMRI maps:

```bash
# Project hMRI maps to FreeSurfer surface
mri_vol2surf \
  --src R1_subject01.nii \
  --out lh.R1.mgh \
  --regheader subject01 \
  --hemi lh \
  --projfrac 0.5

# Surface-based smoothing
mri_surf2surf \
  --hemi lh \
  --s subject01 \
  --sval lh.R1.mgh \
  --tval lh.R1.sm5.mgh \
  --fwhm 5
```

### BIDS Compatibility

```bash
# Organize hMRI data in BIDS format
sub-01/
  anat/
    sub-01_acq-PDw_echo-1_part-mag_MESE.nii.gz
    sub-01_acq-PDw_echo-2_part-mag_MESE.nii.gz
    # ... more echoes
    sub-01_acq-T1w_echo-1_part-mag_MESE.nii.gz
    sub-01_acq-MTw_echo-1_part-mag_MESE.nii.gz
    sub-01_TB1map.nii.gz

# Use hMRI BIDS processing script
hmri_create_bids_maps('bids_directory', '/path/to/bids', ...
                      'subject', 'sub-01');
```

## Troubleshooting

### Problem 1: Maps Show Extreme Values

**Symptoms:** R1 > 5 s⁻¹ or negative values

**Solutions:**
```matlab
% Check input data quality
% Verify echo times are correct
% Ensure proper DICOM conversion
% Check for motion artifacts

% Inspect metadata
V = spm_vol('PDw_echo1.nii');
json_file = strrep(V.fname, '.nii', '.json');
metadata = jsondecode(fileread(json_file));
disp(metadata.EchoTime);  % Should match protocol
```

### Problem 2: B1 Map Not Applied

**Symptoms:** Streaky artifacts, especially at high field

**Solutions:**
```matlab
% Verify B1 map is specified
% Check B1 map orientation matches structural

spm_check_registration('MT_subject01.nii', 'B1map_subject01.nii');

% Recompute with explicit B1 map
matlabbatch{1}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI = {
    '/full/path/to/B1map_subject01.nii'
};
```

### Problem 3: Segmentation Failures

**Symptoms:** Poor tissue classification

**Solutions:**
```matlab
% Use bias-corrected MT image for segmentation
% Check image orientation

% Manual reorientation if needed
spm_image('Display', 'MT_subject01.nii');
% Use reorient tool to align to MNI

% Or use auto-reorient
matlabbatch{1}.spm.tools.hmri.autoreor.reference = {'MT_subject01.nii'};
```

### Problem 4: Out of Memory Errors

**Symptoms:** MATLAB runs out of memory during processing

**Solutions:**
```matlab
% Reduce number of simultaneous subjects
% Process one contrast at a time
% Increase MATLAB Java heap size

% In MATLAB preferences or java.opts:
-Xmx8g  % Allocate 8GB heap

% Or use implicit masking to reduce memory
hmri_def.qMRI_maps_implicitMasking = true;
```

### Problem 5: Different Values Across Scanners

**Symptoms:** Same tissue shows different R1/MT values

**Solutions:**
```matlab
% This is expected without full standardization
% Solutions:
% 1. Use B1 correction on all scanners
% 2. Use identical protocols (TR, TE, FA)
% 3. Validate with phantom
% 4. Consider ComBat harmonization post-processing

% Phantom-based calibration
phantom_R1_measured = 1.45;
phantom_R1_expected = 1.50;
correction_factor = phantom_R1_expected / phantom_R1_measured;

R1_corrected = R1_data * correction_factor;
```

### Problem 6: Quality Control Metrics Not Generated

**Symptoms:** Missing QA JSON files

**Solutions:**
```matlab
% Enable QA output explicitly
hmri_def.qMRI_maps.QA.enable = true;

% Rerun map creation
% Or generate QA separately
hmri_quality_check('R1_subject01.nii', 'MT_subject01.nii');
```

## Best Practices

### Acquisition Protocol

1. **Use manufacturer-optimized protocols** - Check hmri.info for validated sequences
2. **Always acquire B1 maps** - Essential for accurate quantification
3. **Match resolution across contrasts** - Typically 1mm isotropic
4. **Use 8 echoes minimum** - Better R2* estimation
5. **Maintain protocol consistency** - Especially for longitudinal studies

### Processing Workflow

1. **Visual QC of raw data** - Check for motion, artifacts before processing
2. **Use auto-reorient first** - Ensures proper alignment
3. **Enable B1 and spoiling corrections** - Improved accuracy
4. **Save intermediate outputs** - Easier debugging
5. **Document processing parameters** - Critical for reproducibility

### Statistical Analysis

1. **Always control for age** - Strong age effects on all parameters
2. **Use appropriate smoothing** - 6-8mm FWHM typical for VBQ
3. **Check for normality** - Quantitative maps may need transformation
4. **Multiple comparison correction** - FWE or FDR mandatory
5. **Report parameter ranges** - Helps identify outliers

### Quality Control

1. **Visual inspection mandatory** - Automated QC not sufficient
2. **Check parameter distributions** - Identify outliers
3. **Validate against literature** - Ensure values in expected range
4. **Use phantom data** - Regular QC for multi-site studies
5. **Monitor longitudinal stability** - Track scanner drift

### Multi-Site Studies

1. **Harmonize protocols exactly** - Same TR, TE, FA, resolution
2. **Acquire traveling phantom data** - Quantify cross-site variance
3. **Perform site-specific QC** - Different scanners may have different issues
4. **Consider harmonization methods** - ComBat, normative modeling
5. **Include site as covariate** - Account for residual differences

## Resources

### Official Documentation

- **hMRI Website:** https://hmri.info/
- **GitHub Repository:** https://github.com/hMRI-group/hMRI-toolbox
- **User Manual:** https://hmri.info/documentation
- **Protocol Database:** https://hmri.info/protocols
- **Tutorial Videos:** https://hmri.info/tutorials

### Key Publications

- **MPM Method:** Weiskopf et al. (2013) "Quantitative multi-parameter mapping" Front Neurosci
- **hMRI Toolbox:** Tabelow et al. (2019) "hMRI – A toolbox for quantitative MRI" Neuroimage
- **Clinical Applications:** Draganski et al. (2011) "Regional specificity of MRI contrast parameter changes" Neuroimage

### Learning Resources

- **SPM Course Materials:** https://www.fil.ion.ucl.ac.uk/spm/course/
- **Quantitative MRI Tutorial:** https://qmrlab.org/tutorials
- **ISMRM Educational Materials:** https://www.ismrm.org/

### Community Support

- **SPM Mailing List:** https://www.fil.ion.ucl.ac.uk/spm/support/
- **GitHub Issues:** https://github.com/hMRI-group/hMRI-toolbox/issues
- **Neurostars Forum:** https://neurostars.org/ (tag: quantitative-mri)

## Citation

```bibtex
@article{Tabelow2019,
  title = {hMRI – A toolbox for quantitative MRI in neuroscience and clinical research},
  author = {Tabelow, Karsten and Balteau, Evelyne and Ashburner, John and
            Callaghan, Martina F and Draganski, Bogdan and Helms, Gunther and
            Kherif, Ferath and Leutritz, Tobias and Lutti, Antoine and
            Phillips, Christophe and Reimer, Enrico and Ruthotto, Lars and
            Seif, Maryam and Weiskopf, Nikolaus and Ziegler, Gabriel and
            Mohammadi, Siawoosh},
  journal = {NeuroImage},
  volume = {194},
  pages = {191--210},
  year = {2019},
  doi = {10.1016/j.neuroimage.2019.01.029}
}

@article{Weiskopf2013,
  title = {Quantitative multi-parameter mapping of R1, PD*, MT, and R2* at 3T:
           A multi-center validation},
  author = {Weiskopf, Nikolaus and Suckling, John and Williams, Guy and
            Correia, Marta M and Inkster, Becky and Tait, Roger and
            Ooi, Cinly and Bullmore, Edward T and Lutti, Antoine},
  journal = {Frontiers in Neuroscience},
  volume = {7},
  pages = {95},
  year = {2013},
  doi = {10.3389/fnins.2013.00095}
}
```

## Related Tools

- **SPM12** - Statistical Parametric Mapping platform (required dependency)
- **CAT12** - Computational Anatomy Toolbox for surface-based VBQ
- **qMRLab** - Alternative quantitative MRI library with more methods
- **FreeSurfer** - Surface-based analysis of quantitative maps
- **ANTs** - Advanced normalization for VBQ preprocessing
- **QUIT** - Quantitative imaging tools (C++ alternative)
- **MP2RAGE** - Alternative T1 mapping method
- **SyMRI** - Commercial quantitative MRI solution

---

**Skill Type:** Quantitative MRI Analysis
**Difficulty Level:** Advanced
**Prerequisites:** MATLAB, SPM12, Basic MRI physics
**Typical Use Cases:** Tissue characterization, VBQ studies, clinical biomarkers, aging research, multi-site studies
