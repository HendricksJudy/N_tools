# CAT12 - Computational Anatomy Toolbox

## Overview

CAT12 (Computational Anatomy Toolbox) is a comprehensive SPM12 extension for computational anatomy, specializing in voxel-based morphometry (VBM), surface-based morphometry (SBM), and region-of-interest analysis of structural MRI data. It provides state-of-the-art preprocessing with improved segmentation, spatial registration, and surface reconstruction algorithms optimized for group studies and clinical applications.

**Website:** http://www.neuro.uni-jena.de/cat/
**Platform:** MATLAB (requires SPM12)
**Language:** MATLAB/C
**License:** GPL-2.0

## Key Features

- Advanced tissue segmentation (6 tissue classes)
- Longitudinal processing pipeline
- Surface-based morphometry (cortical thickness, gyrification, sulcal depth)
- DARTEL and geodesic shooting registration
- Integrated quality control with IQR scores
- Total intracranial volume (TIV) estimation
- Automated processing with expert defaults
- Batch processing for large studies
- Integrated statistical analysis
- ROI extraction with multiple atlases
- Cross-sectional and longitudinal designs

## Installation

### Prerequisites

```matlab
% Install SPM12 first
% Download from: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/

% Add SPM12 to MATLAB path
addpath('/path/to/spm12');
spm('defaults', 'fmri');
```

### Installing CAT12

```matlab
% Download CAT12 from: http://www.neuro.uni-jena.de/cat12/

% Extract to SPM12 toolbox directory
% /path/to/spm12/toolbox/cat12/

% Add to path
addpath('/path/to/spm12/toolbox/cat12');

% Verify installation
cat12('version')

% Check CAT12 settings
cat12('expert')
```

### Compilation (Optional)

```matlab
% For better performance, compile CAT12 mex files
cd(fullfile(spm('dir'), 'toolbox', 'cat12'));
cat_install_atlases;  % Download additional atlases
```

## Basic VBM Analysis

### Single Subject Preprocessing

```matlab
%% CAT12 VBM Preprocessing - Single Subject

% Clear workspace
clear all;
close all;

% Initialize SPM
spm('defaults', 'pet');
spm_jobman('initcfg');

% Setup batch structure
matlabbatch{1}.spm.tools.cat.estwrite.data = {
    '/data/sub-01/anat/sub-01_T1w.nii,1'
};

% Preprocessing options (use CAT12 defaults)
matlabbatch{1}.spm.tools.cat.estwrite.nproc = 0;  % 0 = auto-detect cores

% Segmentation options
matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm = {
    fullfile(spm('dir'), 'tpm', 'TPM.nii')
};
matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';
matlabbatch{1}.spm.tools.cat.estwrite.opts.biasstr = 0.5;  % Bias correction strength

% Extended options for initial segmentation
matlabbatch{1}.spm.tools.cat.estwrite.extopts.APP = 1070;  % Affine preprocessing
matlabbatch{1}.spm.tools.cat.estwrite.extopts.LASstr = 0.5;  % Local adaptive segmentation
matlabbatch{1}.spm.tools.cat.estwrite.extopts.gcutstr = 2;  % Skull-stripping strength

% Surface options
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface = 1;  % Extract surfaces

% Registration method
matlabbatch{1}.spm.tools.cat.estwrite.extopts.regstr = 0;  % 0 = shooting, 0.5 = optimized shooting

% Output options
matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 1;  % Save surfaces
matlabbatch{1}.spm.tools.cat.estwrite.output.ROI = 1;  % Save ROI measurements

% Gray matter
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 1;  % Modulated (preserve volume)
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.warped = 1;  % MNI space

% White matter
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.warped = 1;

% Run batch
spm_jobman('run', matlabbatch);

% Output files will be in same directory as input:
% - mwp1*.nii (modulated warped GM)
% - mwp2*.nii (modulated warped WM)
% - lh.central.sub-01_T1w.gii (left hemisphere surface)
% - rh.central.sub-01_T1w.gii (right hemisphere surface)
% - cat_sub-01_T1w.xml (QC report)
```

### Quality Control

```matlab
%% Check preprocessing quality

% Generate QC report
cat_stat_check_SPM('/data/sub-01/anat/cat_sub-01_T1w.xml');

% Or check all subjects
xml_files = spm_select('FPList', '/data', '^cat_.*\.xml$');
cat_stat_check_SPM(xml_files);

% Extract quality measures
QC = cat_stat_check_SPM(xml_files, 'noplot');

% Image Quality Rating (IQR)
% A+/A (>90%) - Excellent
% B (80-90%) - Good
% C (70-80%) - Acceptable
% D (<70%) - Poor, consider excluding

% Display ratings
fprintf('Subject\tIQR\tRating\n');
for i = 1:numel(QC)
    fprintf('Sub-%02d\t%.2f\t%s\n', i, QC(i).IQR, QC(i).rating);
end

% Identify poor quality scans
poor_scans = find([QC.IQR] < 70);
if ~isempty(poor_scans)
    fprintf('Warning: %d scans with IQR < 70\n', numel(poor_scans));
end
```

## Batch Processing Multiple Subjects

```matlab
%% CAT12 Batch Processing Script

clear all;
spm('defaults', 'pet');
spm_jobman('initcfg');

% Define subjects
subjects = 1:20;
data_dir = '/data';

% Build file list
T1_files = cell(numel(subjects), 1);
for i = 1:numel(subjects)
    T1_files{i} = sprintf('%s/sub-%02d/anat/sub-%02d_T1w.nii', ...
                          data_dir, subjects(i), subjects(i));
end

% Setup batch
matlabbatch{1}.spm.tools.cat.estwrite.data = T1_files;

% Use CAT12 defaults (expert mode = 0)
matlabbatch{1}.spm.tools.cat.estwrite.nproc = 4;  % Parallel processing

% Standard preprocessing settings
matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm = {
    fullfile(spm('dir'), 'tpm', 'TPM.nii')
};
matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';

% Output: modulated warped GM and WM
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.warped = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.warped = 1;

% Surface reconstruction
matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROI = 1;

% Save and run batch
save('cat12_preprocessing_batch.mat', 'matlabbatch');
spm_jobman('run', matlabbatch);

% Generate quality report
xml_files = spm_select('FPListRec', data_dir, '^cat_.*\.xml$');
cat_stat_check_SPM(xml_files);
```

## Surface-Based Morphometry

### Extract Surface Measures

```matlab
%% Surface-Based Analysis

% After preprocessing, extract cortical measures

% Collect surface files
surf_dir = '/data';
subjects = 1:20;

% Left hemisphere thickness
lh_thickness = cell(numel(subjects), 1);
for i = 1:numel(subjects)
    lh_thickness{i} = sprintf('%s/sub-%02d/surf/lh.thickness.sub-%02d_T1w', ...
                              surf_dir, subjects(i), subjects(i));
end

% Right hemisphere thickness
rh_thickness = cell(numel(subjects), 1);
for i = 1:numel(subjects)
    rh_thickness{i} = sprintf('%s/sub-%02d/surf/rh.thickness.sub-%02d_T1w', ...
                              surf_dir, subjects(i), subjects(i));
end

% Smooth surface data
matlabbatch{1}.spm.tools.cat.stools.surfresamp.data_surf = lh_thickness;
matlabbatch{1}.spm.tools.cat.stools.surfresamp.merge_hemi = 1;  % Combine hemispheres
matlabbatch{1}.spm.tools.cat.stools.surfresamp.mesh32k = 1;  % Resample to 32k mesh
matlabbatch{1}.spm.tools.cat.stools.surfresamp.fwhm_surf = 15;  % 15mm smoothing

spm_jobman('run', matlabbatch);

% Output: s15.mesh.thickness.resampled.sub-01_T1w.gii
```

### Surface Statistics

```matlab
%% Statistical analysis of cortical thickness

% Collect smoothed surface files
surf_files = spm_select('FPListRec', '/data', '^s15\.mesh\.thickness\.resampled\..*\.gii$');

% Setup factorial design
matlabbatch{1}.spm.stats.factorial_design.dir = {'/results/surface_stats'};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = cellstr(surf_files);

% Add TIV as covariate
TIV_values = cat_get_TIV(xml_files);
matlabbatch{1}.spm.stats.factorial_design.cov.c = TIV_values;
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'TIV';
matlabbatch{1}.spm.stats.factorial_design.cov.iCFI = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.iCC = 1;

% Model estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat = {'/results/surface_stats/SPM.mat'};

% Contrast
matlabbatch{3}.spm.stats.con.spmmat = {'/results/surface_stats/SPM.mat'};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Positive effect';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;

% Results
matlabbatch{4}.spm.stats.results.spmmat = {'/results/surface_stats/SPM.mat'};
matlabbatch{4}.spm.stats.results.conspec.contrasts = 1;
matlabbatch{4}.spm.stats.results.conspec.threshdesc = 'FWE';
matlabbatch{4}.spm.stats.results.conspec.thresh = 0.05;
matlabbatch{4}.spm.stats.results.conspec.extent = 0;

spm_jobman('run', matlabbatch);
```

## Longitudinal Analysis

```matlab
%% Longitudinal VBM Processing

% For subjects with multiple timepoints
subjects = {'sub-01', 'sub-02'};
timepoints = {'ses-01', 'ses-02', 'ses-03'};

for subj_idx = 1:numel(subjects)
    subj = subjects{subj_idx};

    % Collect all timepoints for this subject
    tp_files = cell(numel(timepoints), 1);
    for tp_idx = 1:numel(timepoints)
        tp_files{tp_idx} = sprintf('/data/%s/%s/anat/%s_%s_T1w.nii', ...
                                   subj, timepoints{tp_idx}, subj, timepoints{tp_idx});
    end

    % Longitudinal preprocessing
    matlabbatch{1}.spm.tools.cat.long.longmodel.timepoints = tp_files;
    matlabbatch{1}.spm.tools.cat.long.longmodel.longTPM = 1;  % Use longitudinal model

    % Output options
    matlabbatch{1}.spm.tools.cat.long.output.GM.mod = 1;
    matlabbatch{1}.spm.tools.cat.long.output.GM.warped = 1;

    % Run
    spm_jobman('run', matlabbatch);
    clear matlabbatch;
end

% Longitudinal analysis produces:
% - avg_*.nii: Average image
% - wp1avg_*.nii: Processed average GM
% - Individual timepoint files with consistent preprocessing
```

## ROI Analysis

```matlab
%% Extract ROI values from multiple atlases

% After preprocessing, extract ROI measurements
xml_files = spm_select('FPListRec', '/data', '^cat_.*\.xml$');

% Available atlases in CAT12:
% - Neuromorphometrics
% - LPBA40
% - Hammers
% - AAL3
% - Brainnetome

% Extract ROI values
cat_roi_fun(xml_files, 'atlas', 'neuromorphometrics');

% Read ROI file
roi_file = 'catROI_neuromorphometrics.txt';
roi_data = cat_io_csv(roi_file);

% Extract specific ROI (e.g., hippocampus)
hippocampus_left = roi_data.GMV_Hippocampus_L;
hippocampus_right = roi_data.GMV_Hippocampus_R;

% Compute total hippocampal volume
total_hippocampus = hippocampus_left + hippocampus_right;

% Normalize by TIV
TIV = roi_data.TIV;
normalized_hippocampus = total_hippocampus ./ TIV * 1000;

% Statistical test
[h, p, ci, stats] = ttest2(normalized_hippocampus(1:10), ...
                           normalized_hippocampus(11:20));
fprintf('T-test: t=%.2f, p=%.4f\n', stats.tstat, p);
```

## Two-Sample T-Test (VBM)

```matlab
%% Group comparison: Patients vs Controls

spm('defaults', 'pet');
spm_jobman('initcfg');

% Collect GM images
controls_GM = spm_select('FPList', '/data/controls', '^mwp1.*\.nii$');
patients_GM = spm_select('FPList', '/data/patients', '^mwp1.*\.nii$');

% Setup factorial design (two-sample t-test)
matlabbatch{1}.spm.stats.factorial_design.dir = {'/results/vbm_ttest'};
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = cellstr(controls_GM);
matlabbatch{1}.spm.tools.factorial_design.des.t2.scans2 = cellstr(patients_GM);
matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0;  % Independent
matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;  % Unequal variance
matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0;

% Add covariates (TIV, age)
% Get TIV values
controls_xml = spm_select('FPList', '/data/controls', '^cat_.*\.xml$');
patients_xml = spm_select('FPList', '/data/patients', '^cat_.*\.xml$');
TIV_all = cat_get_TIV([cellstr(controls_xml); cellstr(patients_xml)]);

matlabbatch{1}.spm.stats.factorial_design.cov.c = TIV_all;
matlabbatch{1}.spm.stats.factorial_design.cov.cname = 'TIV';
matlabbatch{1}.spm.stats.factorial_design.cov.iCFI = 1;
matlabbatch{1}.spm.stats.factorial_design.cov.iCC = 1;

% Masking (explicit mask with absolute threshold)
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;  % Implicit mask
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};

% Global calculation
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

% Estimate
matlabbatch{2}.spm.stats.fmri_est.spmmat = {'/results/vbm_ttest/SPM.mat'};
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

% Contrasts
matlabbatch{3}.spm.stats.con.spmmat = {'/results/vbm_ttest/SPM.mat'};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Controls > Patients';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'Patients > Controls';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1];

% Results (FWE corrected)
matlabbatch{4}.spm.stats.results.spmmat = {'/results/vbm_ttest/SPM.mat'};
matlabbatch{4}.spm.stats.results.conspec.contrasts = 1;
matlabbatch{4}.spm.stats.results.conspec.threshdesc = 'FWE';
matlabbatch{4}.spm.stats.results.conspec.thresh = 0.05;
matlabbatch{4}.spm.stats.results.conspec.extent = 100;  % Cluster extent

% Run analysis
spm_jobman('run', matlabbatch);
```

## Advanced Options

### Expert Mode Settings

```matlab
%% Use expert settings for fine control

% Enable expert mode
cat_get_defaults('extopts.expertgui', 2);

% Custom preprocessing
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.APP = 1070;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.NCstr = -Inf;  % Noise correction
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.LASstr = 0.5;  % Local adaptive seg
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.gcutstr = 2;  % Skull-stripping
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.cleanupstr = 0.5;  % Cleanup

% Registration options
matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.shooting.shootingtpm = {
    fullfile(spm('dir'), 'toolbox', 'cat12', 'templates_MNI152NLin2009cAsym', 'Template_0_GS.nii')
};
matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.shooting.regstr = 0.5;

% Surface options
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.pbtres = 0.5;  % Resolution
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.scale_cortex = 0.7;  % Cortex scaling
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.add_parahipp = 0.1;  % Parahippocampal
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.close_parahipp = 1;
```

## Integration with Claude Code

When helping users with CAT12:

1. **Check Installation:**
   ```matlab
   which cat12
   which spm
   cat12('version')
   ```

2. **Common Issues:**
   - SPM12 not in path or wrong version
   - Memory errors with large datasets
   - Poor segmentation quality (check image quality)
   - Convergence issues in registration
   - Missing atlases (run `cat_install_atlases`)

3. **Best Practices:**
   - Always run quality control after preprocessing
   - Use TIV as covariate in group analyses
   - Check IQR scores, exclude scans with IQR < 70
   - Use modulated images for VBM (preserves volume)
   - Smooth surface data (15mm recommended)
   - Use appropriate atlas for ROI analysis
   - Save batch files for reproducibility

4. **Output Files:**
   - `mwp1*.nii`: Modulated warped GM (VBM analysis)
   - `mwp2*.nii`: Modulated warped WM
   - `wp1*.nii`: Warped GM (non-modulated)
   - `lh.central.*.gii`: Left hemisphere central surface
   - `lh.thickness.*`: Cortical thickness
   - `cat_*.xml`: QC report with IQR scores
   - `catROI_*.txt`: ROI measurements

## Troubleshooting

**Problem:** "Cannot find TPM.nii"
**Solution:** Ensure SPM12 is properly installed and in MATLAB path

**Problem:** Poor segmentation quality (low IQR)
**Solution:** Check image quality, try adjusting bias correction strength, use expert mode settings

**Problem:** Out of memory errors
**Solution:** Process fewer subjects in parallel (`nproc`), increase MATLAB memory, or use single-threaded processing

**Problem:** Surface reconstruction fails
**Solution:** Check skull-stripping quality, adjust `gcutstr` parameter, or disable surface extraction temporarily

**Problem:** Registration does not converge
**Solution:** Try optimized shooting (`regstr = 0.5`), check affine preprocessing quality

## Resources

- Website: http://www.neuro.uni-jena.de/cat/
- Manual: http://www.neuro.uni-jena.de/cat12/CAT12-Manual.pdf
- Forum: https://www.nitrc.org/forum/?group_id=1115
- SPM Mailing List: https://www.fil.ion.ucl.ac.uk/spm/support/
- Tutorial: http://www.neuro.uni-jena.de/cat12/CAT12-Tutorial.pdf

## Citation

```bibtex
@misc{cat12,
  title={Computational Anatomy Toolbox - CAT12},
  author={Gaser, Christian and Dahnke, Robert and Thompson, Paul M and Kurth, Florian and Luders, Eileen},
  year={2022},
  url={http://www.neuro.uni-jena.de/cat/}
}

@article{gaser2022cat,
  title={CAT--A Computational Anatomy Toolbox for the Analysis of Structural MRI Data},
  author={Gaser, Christian and Dahnke, Robert and Thompson, Paul M and Kurth, Florian and Luders, Eileen and Alzheimer's Disease Neuroimaging Initiative},
  journal={bioRxiv},
  year={2022}
}
```

## Related Tools

- **SPM12:** Statistical Parametric Mapping (required)
- **FreeSurfer:** Alternative surface reconstruction
- **FSL:** Complementary neuroimaging analysis
- **ANTs:** Advanced normalization
- **DARTEL:** SPM's registration tool (predecessor)
