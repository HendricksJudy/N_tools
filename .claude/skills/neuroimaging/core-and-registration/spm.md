# SPM (Statistical Parametric Mapping)

## Overview

SPM is one of the most widely used software packages for analyzing functional neuroimaging data (fMRI, PET, SPECT, EEG, and MEG). Developed at the Wellcome Centre for Human Neuroimaging at UCL, it implements statistical parametric mapping to characterize differences in brain activity.

**Website:** https://www.fil.ion.ucl.ac.uk/spm/
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** GNU GPL v2

## Key Features

- fMRI time-series analysis
- Volumetric segmentation and spatial normalization
- Dynamic causal modeling (DCM)
- Multivariate pattern analysis
- VBM (Voxel-Based Morphometry)
- Statistical inference with Random Field Theory
- Batch processing system
- Extensive toolbox ecosystem (CAT12, CONN, etc.)

## Installation

### Requirements
- MATLAB R2018b or later
- Image Processing Toolbox (recommended)
- At least 8GB RAM
- 2GB disk space

### Installation Steps

1. Download SPM12 from the official website
2. Extract the archive to your preferred location
3. Add SPM to MATLAB path:

```matlab
addpath('/path/to/spm12')
savepath
```

4. Verify installation:
```matlab
spm
```

## Common Workflows

### fMRI Analysis Pipeline

```matlab
% Initialize SPM
spm('defaults', 'fmri')
spm_jobman('initcfg')

% 1. Realignment (motion correction)
matlabbatch{1}.spm.spatial.realign.estimate.data = {files};

% 2. Coregistration
matlabbatch{2}.spm.spatial.coreg.estimate.ref = {structural};
matlabbatch{2}.spm.spatial.coreg.estimate.source = {mean_functional};

% 3. Segmentation
matlabbatch{3}.spm.spatial.preproc.channel.vols = {structural};

% 4. Normalization
matlabbatch{4}.spm.spatial.normalise.write.subj.def = {deformation_field};

% 5. Smoothing
matlabbatch{5}.spm.spatial.smooth.data = {normalized_images};
matlabbatch{5}.spm.spatial.smooth.fwhm = [8 8 8];

% Run batch
spm_jobman('run', matlabbatch);
```

### First-Level GLM Analysis

```matlab
% Specify model
matlabbatch{1}.spm.stats.fmri_spec.dir = {output_dir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2; % TR in seconds
matlabbatch{1}.spm.stats.fmri_spec.sess.scans = {functional_images};
matlabbatch{1}.spm.stats.fmri_spec.sess.cond.name = 'task';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond.onset = onsets;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond.duration = durations;

% Estimate model
matlabbatch{2}.spm.stats.fmri_est.spmmat = {fullfile(output_dir, 'SPM.mat')};

% Define contrasts
matlabbatch{3}.spm.stats.con.spmmat = {fullfile(output_dir, 'SPM.mat')};
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'task > baseline';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 0];

spm_jobman('run', matlabbatch);
```

### VBM (Voxel-Based Morphometry)

```matlab
% Use CAT12 toolbox for modern VBM
% Or use standard SPM segmentation
matlabbatch{1}.spm.spatial.preproc.channel.vols = {t1_images};
matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {grey_matter_prior};
matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [1 1];

spm_jobman('run', matlabbatch);
```

## Integration with Claude Code

When helping users with SPM:

1. **Check MATLAB Environment:**
   ```matlab
   ver % Check MATLAB version
   which spm % Verify SPM is in path
   ```

2. **Batch System:** SPM12 uses a batch system - help users build matlabbatch structures

3. **File Organization:** SPM expects specific file naming conventions (e.g., `sub-01_task-rest_bold.nii`)

4. **Memory Management:** Large datasets may require adjusting MATLAB memory settings

5. **Common Issues:**
   - Path problems: Use `addpath` with full paths
   - Missing files: Verify file existence before processing
   - NIfTI headers: Ensure proper orientation and voxel sizes

## Best Practices

- Always work on copies of your data
- Use BIDS format for data organization
- Document your preprocessing steps
- Save matlabbatch structures for reproducibility
- Check results visually at each step
- Use SPM's batch system for automation
- Keep SPM updated to the latest version

## Common Commands

```matlab
% Display image
spm_check_registration(image_file)

% Read NIfTI
V = spm_vol(nifti_file);
Y = spm_read_vols(V);

% Write NIfTI
V_new = V; % Copy header
V_new.fname = 'output.nii';
spm_write_vol(V_new, new_data);

% View results
spm_results_ui('Setup')

% Render activation on surface
spm_render(structural, overlays)
```

## Troubleshooting

**Problem:** "Undefined function or variable 'spm'"
**Solution:** Add SPM to MATLAB path and save

**Problem:** Out of memory errors
**Solution:** Process data in smaller batches or increase Java heap space

**Problem:** Coregistration fails
**Solution:** Check image orientations, manually set origin if needed

## Resources

- Official SPM Documentation: https://www.fil.ion.ucl.ac.uk/spm/doc/
- SPM Manual: https://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf
- SPM Mailing List: https://www.fil.ion.ucl.ac.uk/spm/support/
- Wikibooks SPM Tutorial: https://en.wikibooks.org/wiki/SPM
- YouTube Tutorial Series: Multiple available

## Related Tools

- **CAT12:** Computational Anatomy Toolbox for SPM
- **CONN:** Functional connectivity toolbox
- **DPABI:** Data Processing & Analysis for Brain Imaging
- **MarsBaR:** ROI analysis toolbox
- **SnPM:** Statistical non-parametric mapping

## Citation

```bibtex
@book{friston2007spm,
  title={Statistical Parametric Mapping: The Analysis of Functional Brain Images},
  author={Friston, Karl and Ashburner, John and Kiebel, Stefan and Nichols, Thomas and Penny, Will},
  publisher={Academic Press},
  year={2007}
}
```
