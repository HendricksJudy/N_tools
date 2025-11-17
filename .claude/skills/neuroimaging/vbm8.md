# VBM8 - Voxel-Based Morphometry Toolbox (Legacy)

## Overview

VBM8 (Voxel-Based Morphometry 8) is a MATLAB/SPM-based toolbox for automated analysis of structural brain MRI. Developed by Christian Gaser at the University of Jena, VBM8 was widely used for detecting regional differences in brain tissue composition. While VBM8 has been superseded by CAT12 (offering improved methods and additional features), it remains relevant for reproducing older studies and understanding the evolution of morphometric analysis techniques.

**Website:** http://dbm.neuro.uni-jena.de/vbm/
**Platform:** MATLAB/SPM (Windows/macOS/Linux)
**License:** GPL
**Status:** Legacy (superseded by CAT12)
**Key Application:** Voxel-based morphometry, tissue segmentation, DARTEL registration

### What is Voxel-Based Morphometry?

VBM is a neuroimaging analysis technique that investigates focal differences in brain anatomy by statistically comparing tissue composition (gray matter, white matter, CSF) across groups or relating it to behavioral/clinical variables. VBM involves:

1. **Segmentation:** Classify voxels into tissue types
2. **Normalization:** Warp brains to common template space
3. **Modulation:** Preserve regional volumes after warping
4. **Smoothing:** Apply Gaussian kernel for statistical analysis
5. **Statistical testing:** Identify regional group differences

## Key Features

- **Adaptive Maximum A Posteriori (AMAP) segmentation** - Improved tissue classification
- **DARTEL registration** - High-dimensional nonlinear warping
- **Automated pipeline** - Preprocessing with minimal user interaction
- **Bias field correction** - Correction for intensity inhomogeneity
- **Partial volume estimation** - Account for mixed tissue types
- **Quality control tools** - Assess segmentation and registration quality
- **Total intracranial volume (TIV) calculation** - For normalization in statistics
- **SPM integration** - Seamless workflow with SPM statistical analysis
- **Batch processing** - Process multiple subjects efficiently
- **Customizable templates** - Create study-specific templates
- **Longitudinal processing** - Within-subject analysis over time
- **Well-documented** - Extensive user manual and publications

## Installation

### Prerequisites

VBM8 requires SPM8 or SPM12:

```bash
# Download SPM12 (recommended for modern MATLAB versions)
cd ~/software
wget https://www.fil.ion.ucl.ac.uk/spm/download/restricted/eldorado/spm12.zip
unzip spm12.zip
```

### Install VBM8

**Download VBM8:**

```bash
# Visit website and download
# http://dbm.neuro.uni-jena.de/vbm/download/

cd ~/software
# Extract downloaded archive
unzip vbm8_r445.zip -d vbm8

# Or clone if available via repository
```

**Add to MATLAB Path:**

```matlab
% In MATLAB
addpath('~/software/spm12');
addpath('~/software/vbm8');
savepath;

% Start SPM to verify
spm fmri  % or spm pet
```

### Verify Installation

```matlab
% Check VBM8 menu appears in SPM
spm
% Look for "VBM8" in the Toolbox menu

% Or check directly
which vbm8_defaults
% Should return path to vbm8_defaults.m
```

## Basic VBM8 Pipeline

### Step 1: Organize Data

```bash
# Recommended directory structure
project/
  raw/
    sub-01_T1w.nii
    sub-02_T1w.nii
    ...
  preprocessing/
  stats/
```

### Step 2: Simple Preprocessing (All-in-One)

VBM8 offers an automated pipeline:

```matlab
% Initialize batch
clear matlabbatch;

% List of subjects
subjects = {
    '/project/raw/sub-01_T1w.nii'
    '/project/raw/sub-02_T1w.nii'
    '/project/raw/sub-03_T1w.nii'
};

% VBM8 preprocessing
matlabbatch{1}.spm.tools.vbm8.estwrite.data = subjects;

% Estimation options
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.tpm = {
    fullfile(spm('dir'), 'toolbox', 'Seg', 'TPM.nii')
};
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.ngaus = [2 2 2 4];
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.biasreg = 0.0001;
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.biasfwhm = 60;
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.affreg = 'mni';
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.warpreg = 4;
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.samp = 3;

% Writing options - what to output
matlabbatch{1}.spm.tools.vbm8.estwrite.output.GM.native = 1;    % Native space GM
matlabbatch{1}.spm.tools.vbm8.estwrite.output.GM.warped = 0;    % Skip warped
matlabbatch{1}.spm.tools.vbm8.estwrite.output.GM.modulated = 1; % Modulated GM
matlabbatch{1}.spm.tools.vbm8.estwrite.output.GM.dartel = 1;    % DARTEL exports

matlabbatch{1}.spm.tools.vbm8.estwrite.output.WM.native = 1;
matlabbatch{1}.spm.tools.vbm8.estwrite.output.WM.warped = 0;
matlabbatch{1}.spm.tools.vbm8.estwrite.output.WM.modulated = 1;
matlabbatch{1}.spm.tools.vbm8.estwrite.output.WM.dartel = 1;

matlabbatch{1}.spm.tools.vbm8.estwrite.output.bias.native = 1;  % Bias-corrected image
matlabbatch{1}.spm.tools.vbm8.estwrite.output.label.native = 1; % Tissue labels

% Run preprocessing
spm_jobman('run', matlabbatch);
```

### Expected Outputs

After preprocessing, VBM8 generates:

```bash
# For each subject:
p1sub-01_T1w.nii      # Native space gray matter probability
p2sub-01_T1w.nii      # Native space white matter probability
p3sub-01_T1w.nii      # Native space CSF probability

rp1sub-01_T1w.nii     # DARTEL import: rigidly aligned GM
rp2sub-01_T1w.nii     # DARTEL import: rigidly aligned WM

mwp1sub-01_T1w.nii    # Modulated warped GM (for VBM analysis)
mwp2sub-01_T1w.nii    # Modulated warped WM

msub-01_T1w.nii       # Bias-corrected structural image
sub-01_T1w_seg8.mat   # Segmentation parameters
```

## DARTEL Registration Workflow

DARTEL (Diffeomorphic Anatomical Registration Through Exponentiated Lie Algebra) provides high-dimensional nonlinear registration:

### Step 1: Create DARTEL Template

```matlab
% Collect all DARTEL-imported tissue maps
% (rp1* and rp2* files from VBM8 preprocessing)

clear matlabbatch;

% Find all GM and WM DARTEL images
gm_files = spm_select('FPList', '/project/preprocessing', '^rp1.*\.nii$');
wm_files = spm_select('FPList', '/project/preprocessing', '^rp2.*\.nii$');

% DARTEL: Create Template
matlabbatch{1}.spm.tools.dartel.warp.images{1} = cellstr(gm_files);
matlabbatch{1}.spm.tools.dartel.warp.images{2} = cellstr(wm_files);

% Settings
matlabbatch{1}.spm.tools.dartel.warp.settings.template = 'Template';
matlabbatch{1}.spm.tools.dartel.warp.settings.rform = 0;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).its = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).rparam = [4 2 1e-06];
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).K = 0;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).slam = 16;

% Additional iterations (6 total outer iterations)
% ... configure remaining parameter sets

matlabbatch{1}.spm.tools.dartel.warp.settings.optim.lmreg = 0.01;
matlabbatch{1}.spm.tools.dartel.warp.settings.optim.cyc = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.optim.its = 3;

% Run (this takes time - hours for large cohorts)
spm_jobman('run', matlabbatch);
```

### Step 2: Normalize to MNI Space

```matlab
% Apply DARTEL flow fields to normalize images to MNI space

clear matlabbatch;

% Flow fields from DARTEL (u_* files)
flowfields = spm_select('FPList', '/project/preprocessing', '^u_.*\.nii$');

% Images to normalize (modulated GM segments)
images = spm_select('FPList', '/project/preprocessing', '^p1.*\.nii$');

% DARTEL: Normalize to MNI
matlabbatch{1}.spm.tools.dartel.mni_norm.template = {
    '/project/preprocessing/Template_6.nii'  % Final DARTEL template
};
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs(1).flowfield = {flowfields(1,:)};
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs(1).images = {images(1,:)};
% ... repeat for all subjects

% Voxel size and bounding box
matlabbatch{1}.spm.tools.dartel.mni_norm.vox = [1 1 1];  % 1mm isotropic
matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [NaN NaN NaN; NaN NaN NaN];  % Auto

% Preserve concentration (modulation for VBM)
matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 1;  % 1 = yes, 0 = no

% FWHM smoothing
matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [8 8 8];  % 8mm smoothing

spm_jobman('run', matlabbatch);
```

## Tissue Segmentation Details

### AMAP Segmentation

VBM8 uses Adaptive Maximum A Posteriori segmentation:

```matlab
% Segmentation with custom settings

clear matlabbatch;

matlabbatch{1}.spm.tools.vbm8.estwrite.data = {'sub-01_T1w.nii'};

% Number of Gaussians per tissue class
% GM: 2, WM: 2, CSF: 2, non-brain: 4
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.ngaus = [2 2 2 4];

% Bias regularization (smaller = more regularization)
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.biasreg = 0.0001;  % 0.001 for noisy data

% Bias FWHM (mm) - cutoff for bias correction
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.biasfwhm = 60;  % 60mm typical

% Affine regularization
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.affreg = 'mni';  % 'mni' or 'eastern'

% Warping regularization (higher = smoother)
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.warpreg = 4;

% Sampling distance (mm)
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.samp = 3;  % 3mm typical

spm_jobman('run', matlabbatch);
```

### Understanding Modulation

```matlab
% Modulation preserves total amount of tissue after spatial normalization

% Example: Small brain warped to larger template
% Without modulation: Voxel intensities unchanged → overestimate tissue density
% With modulation: Intensities scaled by Jacobian determinant → correct volume

% For VBM analysis, USE MODULATED images:
% mwp1*.nii (modulated, warped, probability map for GM)
% mwp2*.nii (modulated, warped, probability map for WM)

% Modulated GM represents "volume" of gray matter
% Non-modulated represents "concentration" or "density"
```

## Quality Control

### Visual Inspection

```matlab
% Check segmentation quality

% Display original and segmented images
spm_check_registration('sub-01_T1w.nii', 'p1sub-01_T1w.nii', ...
                       'p2sub-01_T1w.nii', 'p3sub-01_T1w.nii');

% Or use VBM8's check data quality tool
% SPM → Toolbox → VBM8 → Check Data Quality
```

### Automated QC Metrics

```matlab
% Extract total tissue volumes for outlier detection

subjects = {
    'sub-01_T1w.nii'
    'sub-02_T1w.nii'
    % ... more subjects
};

n_subjects = length(subjects);
volumes = zeros(n_subjects, 3);  % GM, WM, CSF

for i = 1:n_subjects
    % Load segmentation results
    [pth, nam, ext] = fileparts(subjects{i});

    % Load tissue probability maps
    Vgm = spm_vol(fullfile(pth, ['p1' nam ext]));
    Vwm = spm_vol(fullfile(pth, ['p2' nam ext]));
    Vcsf = spm_vol(fullfile(pth, ['p3' nam ext]));

    gm = spm_read_vols(Vgm);
    wm = spm_read_vols(Vwm);
    csf = spm_read_vols(Vcsf);

    % Calculate volumes (sum of probabilities × voxel volume)
    voxel_vol = abs(det(Vgm.mat));  % mm³
    volumes(i,1) = sum(gm(:)) * voxel_vol / 1000;   % mL
    volumes(i,2) = sum(wm(:)) * voxel_vol / 1000;
    volumes(i,3) = sum(csf(:)) * voxel_vol / 1000;

    fprintf('Subject %d: GM=%.0f mL, WM=%.0f mL, CSF=%.0f mL, TIV=%.0f mL\n', ...
            i, volumes(i,1), volumes(i,2), volumes(i,3), sum(volumes(i,:)));
end

% Detect outliers
TIV = sum(volumes, 2);
mean_TIV = mean(TIV);
std_TIV = std(TIV);

fprintf('\nTIV: %.0f ± %.0f mL\n', mean_TIV, std_TIV);
outliers = find(abs(TIV - mean_TIV) > 2*std_TIV);
if ~isempty(outliers)
    fprintf('Potential outliers: ');
    fprintf('%d ', outliers);
    fprintf('\n');
end

% Plot distributions
figure;
subplot(2,2,1); histogram(volumes(:,1)); xlabel('GM volume (mL)'); title('Gray Matter');
subplot(2,2,2); histogram(volumes(:,2)); xlabel('WM volume (mL)'); title('White Matter');
subplot(2,2,3); histogram(volumes(:,3)); xlabel('CSF volume (mL)'); title('CSF');
subplot(2,2,4); histogram(TIV); xlabel('TIV (mL)'); title('Total Intracranial Volume');
```

### Check Registration Quality

```matlab
% Overlay normalized images on template

template = fullfile(spm('dir'), 'canonical', 'avg152T1.nii');
normalized_gm = 'smwp1sub-01_T1w.nii';

spm_check_registration(template, normalized_gm);

% Check multiple subjects
normalized_images = spm_select('FPList', '/project/preprocessing', '^smwp1.*\.nii$');
spm_check_registration(template, normalized_images(1:5,:));  % First 5
```

## Statistical Analysis with SPM

### Group Comparison (Two-Sample t-Test)

```matlab
% Compare gray matter between patients and controls

clear matlabbatch;

% Smoothed, modulated, warped GM images
controls = spm_select('FPList', '/project/preprocessing', '^smwp1ctrl.*\.nii$');
patients = spm_select('FPList', '/project/preprocessing', '^smwp1pat.*\.nii$');

% Factorial design specification
matlabbatch{1}.spm.stats.factorial_design.dir = {'/project/stats/gm_group_comparison'};
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = cellstr(controls);
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = cellstr(patients);
matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0;      % Independent samples
matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;  % Unequal variance
matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0;     % No grand mean scaling
matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0;    % No interactions

% Global calculation (for proportional scaling if needed)
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;       % Implicit mask
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;  % No normalization

% Estimate model
matlabbatch{2}.spm.stats.fmri_est.spmmat = {'/project/stats/gm_group_comparison/SPM.mat'};
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% Define contrasts
matlabbatch{3}.spm.stats.con.spmmat = {'/project/stats/gm_group_comparison/SPM.mat'};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Controls > Patients';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'Patients > Controls';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

% Run analysis
spm_jobman('run', matlabbatch);
```

### Adding Covariates (Age, TIV)

```matlab
% Include age and TIV as covariates

% Load covariate data
ages = [45, 52, 38, 41, 55, 48, ...];  % Subject ages
TIV_values = [1450, 1520, 1380, ...];  % Subject TIV in mL

% Add to design
matlabbatch{1}.spm.stats.factorial_design.cov(1).c = ages;
matlabbatch{1}.spm.stats.factorial_design.cov(1).cname = 'Age';
matlabbatch{1}.spm.stats.factorial_design.cov(1).iCFI = 1;  % Interactions: none
matlabbatch{1}.spm.stats.factorial_design.cov(1).iCC = 1;   % Centering: overall mean

matlabbatch{1}.spm.stats.factorial_design.cov(2).c = TIV_values;
matlabbatch{1}.spm.stats.factorial_design.cov(2).cname = 'TIV';
matlabbatch{1}.spm.stats.factorial_design.cov(2).iCFI = 1;
matlabbatch{1}.spm.stats.factorial_design.cov(2).iCC = 1;

% Contrast weights need to account for covariates
% [Group1 Group2 Age TIV]
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1 0 0];
```

### View Results

```matlab
% Results interface
matlabbatch{1}.spm.stats.results.spmmat = {'/project/stats/gm_group_comparison/SPM.mat'};
matlabbatch{1}.spm.stats.results.conspec.titlestr = '';
matlabbatch{1}.spm.stats.results.conspec.contrasts = 1;  % Contrast number
matlabbatch{1}.spm.stats.results.conspec.threshdesc = 'FWE';  % Family-wise error
matlabbatch{1}.spm.stats.results.conspec.thresh = 0.05;
matlabbatch{1}.spm.stats.results.conspec.extent = 100;  % Cluster extent (voxels)
matlabbatch{1}.spm.stats.results.conspec.mask = struct('contrasts', {}, 'thresh', {}, 'mtype', {});
matlabbatch{1}.spm.stats.results.units = 1;  % 1=mm, 2=voxels
matlabbatch{1}.spm.stats.results.print = false;

spm_jobman('run', matlabbatch);
```

## VBM8 vs CAT12

### Key Differences

```matlab
% VBM8 (Legacy):
% - AMAP segmentation
% - DARTEL registration
% - Standard preprocessing pipeline
% - Last update: ~2013

% CAT12 (Current):
% - Improved segmentation with projection-based thickness
% - SHOOT registration (successor to DARTEL)
% - Surface-based morphometry
% - Longitudinal pipeline
% - Better handling of pathology
% - Regular updates

% When to use VBM8:
% 1. Reproducing older studies
% 2. Comparing with previous VBM8 results
% 3. Understanding method development
```

### Migration from VBM8 to CAT12

```matlab
% CAT12 installation (recommended upgrade)
% Download from: http://www.neuro.uni-jena.de/cat/

% CAT12 usage is similar but offers more options
% Basic preprocessing:
matlabbatch{1}.spm.tools.cat.estwrite.data = subjects;
matlabbatch{1}.spm.tools.cat.estwrite.nproc = 0;  % Use all CPUs
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 1;  % Modulated GM
matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 1; % Surface reconstruction

% CAT12 provides:
% - Better quality control reports
% - Surface-based analysis
% - More robust to motion and artifacts
```

## Integration with Claude Code

VBM8 integrates well with Claude-assisted workflows:

### Pipeline Generation

```markdown
**Prompt to Claude:**
"Generate a complete VBM8 preprocessing pipeline for 50 subjects:
1. Segment all T1w images
2. Create DARTEL template
3. Normalize to MNI with 8mm smoothing
4. Extract QC metrics (TIV, tissue volumes)
5. Flag outliers > 2 SD from mean
Include error handling and progress logging."
```

### Statistical Analysis Setup

```markdown
**Prompt to Claude:**
"Create SPM batch script for VBM8 results:
- Compare 25 AD patients vs 25 controls
- Control for age, sex, and TIV
- Test GM and WM separately
- Use FWE correction p<0.05
- Generate results tables for significant clusters"
```

### Batch Quality Control

```markdown
**Prompt to Claude:**
"Create QC script for VBM8 outputs that:
1. Generates thumbnail mosaics for all segmentations
2. Plots tissue volume distributions
3. Checks for failed segmentations
4. Creates HTML report with pass/fail status
5. Flags subjects needing manual review"
```

## Integration with Other Tools

### SPM12

VBM8 is built on SPM:

```matlab
% Access all SPM functions
% Coregister additional modalities
% Perform ROI analysis
% Advanced statistical models

% Example: Extract ROI values
marsbar('on');  % MarsBaR toolbox
roi = maroi('load_cell', 'hippocampus_left.mat');
Y = getdata(roi, 'smwp1sub-01_T1w.nii');
mean_gm = mean(Y);
```

### CAT12

Combine VBM8 preprocessing with CAT12 tools:

```matlab
% Use VBM8 segmentations with CAT12 utilities
% CAT12 offers better visualization and QC
```

### FreeSurfer

Compare volume-based (VBM8) with surface-based (FreeSurfer):

```bash
# FreeSurfer for surface-based morphometry
recon-all -i sub-01_T1w.nii -s sub-01 -all

# Compare:
# - VBM8: Voxel-wise tissue density/volume
# - FreeSurfer: Cortical thickness, surface area
```

### ANTs

Use ANTs for improved registration:

```bash
# Alternative to DARTEL: ANTs registration
antsMultivariateTemplateConstruction2.sh \
  -d 3 \
  -o cohort_template \
  -i 4 \
  -g 0.2 \
  -j 4 \
  -c 2 \
  -k 1 \
  -w 1 \
  -f 8x4x2x1 \
  -s 3x2x1x0 \
  -q 100x70x50x10 \
  -n 1 \
  -r 1 \
  -l 1 \
  -m CC[2] \
  -t SyN[0.1,3,0] \
  *.nii.gz
```

## Troubleshooting

### Problem 1: Segmentation Failures

**Symptoms:** Poor tissue classification, artifacts

**Solutions:**
```matlab
% Adjust bias regularization
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.biasreg = 0.001;  % More regularization

% Increase bias FWHM
matlabbatch{1}.spm.tools.vbm8.estwrite.opts.biasfwhm = 120;  % Larger cutoff

% Check image orientation
spm_check_registration('sub-01_T1w.nii');
% If not aligned to MNI, use Display → Reorient

% Try SPM's unified segmentation
% For difficult cases, preprocess with SPM12's segment
```

### Problem 2: DARTEL Fails to Converge

**Symptoms:** Error during template creation

**Solutions:**
```matlab
% Reduce regularization
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).slam = 8;  % Less smoothing

% Check that all inputs are DARTEL imports (rp1/rp2)
% Check that images are similar (no major pathology)

% Start with fewer subjects
% Build initial template with ~20 subjects
% Then warp remaining subjects to this template
```

### Problem 3: Out of Memory

**Symptoms:** MATLAB crashes during processing

**Solutions:**
```matlab
% Process in smaller batches
% Close unnecessary programs
% Increase Java heap size in MATLAB preferences

% For large cohorts, process subjects in groups
for i = 1:5:n_subjects
    batch_subjects = subjects(i:min(i+4, n_subjects));
    % Process batch
end
```

### Problem 4: Results Look Strange

**Symptoms:** Unexpected statistical findings

**Solutions:**
```matlab
% Check smoothing kernel size
% Too large: Loss of spatial specificity
% Too small: Reduced statistical power
% Typical: 8mm FWHM (2-3x voxel size)

% Verify you used MODULATED images
% mwp1* for volume-based VBM
% wp1* for density-based VBM (rarely used)

% Check covariate scaling
% TIV should be in similar scale to GM volumes
% Age in years is fine

% Inspect design matrix
spm_DesRep('DesMtx', xX);
```

### Problem 5: Reproducing Published Results

**Symptoms:** Cannot replicate older VBM8 study

**Solutions:**
```matlab
% Check VBM8 version used
% Methods should report version number
% r445, r435, etc.

% Match preprocessing parameters:
% - Smoothing kernel
% - Modulation (yes/no)
% - Template (custom or standard)
% - Covariates included

% Check SPM version
% SPM8 vs SPM12 can produce slightly different results

% Contact original authors for exact batch scripts
```

## Best Practices

### Preprocessing

1. **Visual QC is mandatory** - Automated processing can fail silently
2. **Check image orientation** - Ensure rough alignment to MNI
3. **Use appropriate smoothing** - 8mm FWHM typical for VBM
4. **Always include TIV as covariate** - Control for overall brain size
5. **Create study-specific template** - For cohorts with unusual anatomy

### Statistical Analysis

1. **Use modulated images** - For volume-based VBM (most common)
2. **Control for confounds** - Age, sex, TIV at minimum
3. **Apply proper correction** - FWE or FDR for multiple comparisons
4. **Check assumptions** - Normality, homogeneity of variance
5. **Report effect sizes** - Not just p-values

### Quality Control

1. **Inspect all segmentations** - No substitute for visual check
2. **Check tissue volume distributions** - Identify outliers
3. **Verify registration** - Overlay on template
4. **Document exclusions** - Keep record of failed cases
5. **Assess motion artifacts** - Exclude or correct if severe

### Reproducibility

1. **Save batch scripts** - Document exact parameters used
2. **Report VBM8 and SPM versions** - Results can vary slightly
3. **Share preprocessing code** - Enable replication
4. **Archive templates** - If custom template created
5. **Consider migration to CAT12** - For new studies

## Resources

### Official Documentation

- **VBM8 Website:** http://dbm.neuro.uni-jena.de/vbm/
- **VBM8 Manual:** http://dbm.neuro.uni-jena.de/vbm8/VBM8-Manual.pdf
- **SPM Documentation:** https://www.fil.ion.ucl.ac.uk/spm/doc/
- **DARTEL Paper:** Ashburner (2007) "A fast diffeomorphic image registration algorithm"

### Key Publications

- **VBM Review:** Ashburner & Friston (2000) "Voxel-based morphometry" Neuroimage
- **VBM8 Method:** Gaser et al. (2013) described in various publications
- **DARTEL:** Ashburner (2007) NeuroImage
- **VBM Tutorial:** Mechelli et al. (2005) "Voxel-based morphometry of the human brain" Curr Protoc Neurosci

### Learning Resources

- **SPM Course:** https://www.fil.ion.ucl.ac.uk/spm/course/
- **VBM Tutorial Videos:** Available on YouTube
- **Example Scripts:** http://dbm.neuro.uni-jena.de/vbm/download/

### Community Support

- **SPM Mailing List:** https://www.fil.ion.ucl.ac.uk/spm/support/
- **Neurostars Forum:** https://neurostars.org/ (tag: vbm)

### Migration Path

- **CAT12 Website:** http://www.neuro.uni-jena.de/cat/ (Recommended upgrade)

## Citation

```bibtex
@software{VBM8,
  title = {VBM8 Toolbox},
  author = {Gaser, Christian},
  year = {2013},
  url = {http://dbm.neuro.uni-jena.de/vbm/},
  note = {Structural Brain Mapping Group, University of Jena}
}

@article{Ashburner2007,
  title = {A fast diffeomorphic image registration algorithm},
  author = {Ashburner, John},
  journal = {NeuroImage},
  volume = {38},
  number = {1},
  pages = {95--113},
  year = {2007},
  doi = {10.1016/j.neuroimage.2007.07.007}
}
```

## Related Tools

- **CAT12** - Modern successor to VBM8 with surface-based morphometry
- **SPM12** - Statistical Parametric Mapping platform (required for VBM8)
- **FreeSurfer** - Surface-based morphometry alternative
- **FSL** - FMRIB Software Library with VBM tools (FSL-VBM)
- **ANTs** - Advanced normalization for morphometry
- **DARTEL** - High-dimensional registration (integrated in VBM8)
- **MarsBaR** - ROI toolbox for SPM (extract VBM values)
- **xjView** - Results viewer for SPM/VBM analyses

---

**Skill Type:** Structural MRI Analysis
**Difficulty Level:** Intermediate
**Prerequisites:** MATLAB, SPM12, Basic MRI analysis knowledge
**Typical Use Cases:** Voxel-based morphometry, structural group comparisons, aging studies, disease characterization
**Note:** VBM8 is a legacy tool; consider CAT12 for new projects
