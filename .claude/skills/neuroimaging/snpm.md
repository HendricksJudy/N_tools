# SnPM (Statistical nonParametric Mapping)

## Overview

Statistical nonParametric Mapping (SnPM) is a toolbox for SPM that uses permutation testing to make statistical inferences about neuroimaging data. It provides a nonparametric alternative to the parametric methods in SPM, offering valid inference when parametric assumptions (e.g., normality, equal variance) are violated. SnPM is particularly valuable for small sample sizes, non-normal data, and complex experimental designs.

**Website:** http://warwick.ac.uk/snpm
**Platform:** MATLAB with SPM (Windows/macOS/Linux)
**Language:** MATLAB
**License:** GNU General Public License

## Key Features

- Permutation-based statistical inference
- Valid for any sample size (including small n)
- No assumptions about data distribution
- Integrates seamlessly with SPM
- Multiple design types (one-sample, two-sample, paired, ANOVA)
- Variance smoothing for improved sensitivity
- Pseudo t-statistics for better localization
- Cluster-based inference
- TFCE-like enhancement methods
- Compatible with SPM preprocessing pipelines

## Installation

### Requirements

```matlab
% SPM12 (required)
% MATLAB R2012b or later
% Statistics and Machine Learning Toolbox (recommended)
```

### Download and Setup

```bash
# Download SnPM13
# https://github.com/SnPM-toolbox/SnPM-devel/releases

# Extract to SPM toolbox directory
unzip snpm13.zip -d /path/to/spm12/toolbox/
```

```matlab
% Add SnPM to MATLAB path (automatically done by SPM)
addpath('/path/to/spm12');
spm('defaults', 'fmri');

% Launch SnPM
snpm
```

### Verify Installation

```matlab
% Check SnPM version
which snpm_ui
% Should return: /path/to/spm12/toolbox/snpm13/snpm_ui.m

% Launch to verify
snpm
```

## Basic Workflow

### Step 1: Setup Analysis

```matlab
% Launch SnPM
snpm

% GUI Steps:
% 1. Click "Specify"
% 2. Select design type:
%    - MultiSub: One-sample t-test
%    - MultiSub: Simple regression
%    - MultiSub: 2-group unpaired t-test
%    - MultiSub: Paired t-test
%    - 2x2 ANOVA (within-subject)
%    - And more...
```

### Step 2: Specify Design

```matlab
% Example: Two-sample t-test (GUI)
% 1. Select scans for Group 1
% 2. Select scans for Group 2
% 3. Specify variance smoothing (recommended: 6mm)
% 4. Choose number of permutations (5000-10000)
% 5. Specify output directory
% 6. Run specification
```

### Step 3: Compute Permutations

```matlab
% After design specification, compute permutations
% Click "Compute" button in SnPM window
% Or run from command line:
cd /path/to/output/directory
snpm_cp('SnPMcfg.mat');

% This creates permuted datasets and computes statistics
% Can take considerable time for many permutations
```

### Step 4: Inference

```matlab
% View results
% Click "Inference" in SnPM window

% Set threshold:
% - FWE-corrected p-value (e.g., 0.05)
% - Uncorrected p-value
% - Cluster extent threshold

% Create results table and save outputs
```

## Command-Line Usage

### One-Sample T-Test

```matlab
% One-sample t-test (test if activation > 0)
clear job;

% Specify design
job.DesignName = 'MultiSub: One Sample T test on diffs/contrasts';
job.dir = {'/output/snpm_onesample'};

% Input images
scans = {
    '/data/sub-01/con_0001.nii,1'
    '/data/sub-02/con_0001.nii,1'
    '/data/sub-03/con_0001.nii,1'
    % ... add all subjects
};
job.P = scans;

% Analysis parameters
job.Variance.VarSm = 6;  % Variance smoothing (mm)
job.nPerm = 5000;        % Number of permutations
job.vFWHM = [6 6 6];     % Variance smoothness (estimated or specified)
job.bVolm = 1;           % Volume analysis
job.ST.ST_later = -1;    % Compute later
job.masking.tm.tm_none = 1;  % No threshold masking
job.masking.im = 1;      % Implicit mask
job.masking.em = {''};   % No explicit mask
job.globalc.g_omit = 1;  % No global normalization
job.globalm.gmsca.gmsca_no = 1;  % No grand mean scaling
job.globalm.glonorm = 1; % No global normalization

% Run specification
snpm_run_Specify(job);

% Compute permutations
snpm_cp(fullfile('/output/snpm_onesample', 'SnPMcfg.mat'));
```

### Two-Sample T-Test (Unpaired)

```matlab
clear job;

% Design specification
job.DesignName = 'MultiSub: Two Sample T test; 2 Groups: Different Variance';
job.dir = {'/output/snpm_twosample'};

% Group 1 scans
group1_scans = {
    '/data/controls/sub-01/con_0001.nii,1'
    '/data/controls/sub-02/con_0001.nii,1'
    % ... add all control subjects
};

% Group 2 scans
group2_scans = {
    '/data/patients/sub-01/con_0001.nii,1'
    '/data/patients/sub-02/con_0001.nii,1'
    % ... add all patient subjects
};

job.P = [group1_scans; group2_scans];
job.n1 = length(group1_scans);  % Number in Group 1
job.n2 = length(group2_scans);  % Number in Group 2

% Parameters
job.Variance.VarSm = 6;
job.nPerm = 5000;
job.vFWHM = [8 8 8];
job.bVolm = 1;
job.masking.tm.tm_none = 1;
job.masking.im = 1;
job.globalc.g_omit = 1;

% Run
snpm_run_Specify(job);
snpm_cp(fullfile('/output/snpm_twosample', 'SnPMcfg.mat'));
```

### Paired T-Test

```matlab
clear job;

% Paired design (e.g., pre-post intervention)
job.DesignName = 'MultiSub: Paired T test; 1 Group: 2 Conditions';
job.dir = {'/output/snpm_paired'};

% Pre-intervention scans
pre_scans = {
    '/data/sub-01/pre/con_0001.nii,1'
    '/data/sub-02/pre/con_0001.nii,1'
    % ...
};

% Post-intervention scans
post_scans = {
    '/data/sub-01/post/con_0001.nii,1'
    '/data/sub-02/post/con_0001.nii,1'
    % ...
};

job.P = [pre_scans; post_scans];
job.nSubj = length(pre_scans);  % Number of subjects

% Parameters
job.Variance.VarSm = 6;
job.nPerm = 5000;
job.vFWHM = [6 6 6];
job.bVolm = 1;
job.masking.tm.tm_none = 1;
job.masking.im = 1;
job.globalc.g_omit = 1;

% Run
snpm_run_Specify(job);
snpm_cp(fullfile('/output/snpm_paired', 'SnPMcfg.mat'));
```

### Simple Regression

```matlab
clear job;

% Correlation with continuous variable (e.g., age, symptom severity)
job.DesignName = 'MultiSub: Simple Regression (correlation)';
job.dir = {'/output/snpm_regression'};

% Scans
scans = {
    '/data/sub-01/con_0001.nii,1'
    '/data/sub-02/con_0001.nii,1'
    % ... all subjects
};
job.P = scans;

% Covariate of interest (e.g., age)
age = [25; 30; 28; 35; 32; 27; 29; 31; 26; 33];
job.Covariates = age;

% Parameters
job.Variance.VarSm = 6;
job.nPerm = 5000;
job.vFWHM = [6 6 6];
job.bVolm = 1;
job.masking.tm.tm_none = 1;
job.masking.im = 1;
job.globalc.g_omit = 1;

% Run
snpm_run_Specify(job);
snpm_cp(fullfile('/output/snpm_regression', 'SnPMcfg.mat'));
```

## Inference and Results

### Setting Thresholds

```matlab
% Load SnPM results
cd /output/snpm_analysis
load SnPM.mat

% Setup inference
clear job;
job.SnPMmat = {fullfile(pwd, 'SnPM.mat')};

% FWE-corrected threshold
job.Thr.Clus.ClusSize.CFth = 0.05;  % Cluster-forming threshold (FWE)
job.Thr.Clus.ClusSize.ClusSig.FWEthC = 0.05;  % Cluster FWE

% Or voxel-wise FWE
job.Thr.Vox.VoxSig.FWEth = 0.05;

% Extent threshold
job.Tsign = 1;  % Positive effects (use -1 for negative)
job.WriteFiltImg.name = 'SnPM_filtered.nii';

% Run inference
snpm_run_Results(job);
```

### Extract Results

```matlab
% After running inference through GUI or command line
% Results are saved in:
% - SnPM_pp.img: Corrected p-values
% - SnPM_filtered.img: Thresholded results

% View in SPM
spm_check_registration('SnPM_filtered.nii');

% Extract peak coordinates
[Y, XYZ] = spm_read_vols(spm_vol('SnPM_filtered.nii'));
[peaks, peak_vals] = find_peaks(Y, XYZ);
```

## Advanced Features

### Variance Smoothing

```matlab
% Variance smoothing improves sensitivity
% Recommended: 1-2 × spatial smoothness of data

% If data smoothed at 8mm FWHM:
job.Variance.VarSm = 8;  % Match data smoothing

% Or estimate from residuals:
job.Variance.VarSm = 0;  % Automatic estimation

% Can specify anisotropic smoothing:
job.Variance.VarSm_XYZ = [6 6 8];  % Different by dimension
```

### Cluster-Based Inference

```matlab
% Primary threshold
primary_threshold = 0.001;  % Uncorrected p-value

% Inference on cluster extent
job.Thr.Clus.ClusSize.CFth = primary_threshold;
job.Thr.Clus.ClusSize.ClusSig.FWEthC = 0.05;  % Cluster-level FWE

% Or cluster mass
job.Thr.Clus.ClusMass.CFth = primary_threshold;
job.Thr.Clus.ClusMass.ClusSig.FWEthC = 0.05;
```

### Pseudo T-Statistics

```matlab
% SnPM uses pseudo t-statistics
% Better spatial specificity than MaxT

% Enable pseudo-t (default in most designs)
job.usePseudo = 1;

% Pseudo-t combines:
% - Standardized effect size
% - Variance estimates
% - More powerful than simple permutation
```

## Batch Processing

### Process Multiple Contrasts

```matlab
% Batch script for multiple contrasts
contrasts = {
    'Task1 vs Baseline'
    'Task2 vs Baseline'
    'Task1 vs Task2'
};

con_nums = [1, 2, 3];

for c = 1:length(contrasts)
    fprintf('Processing: %s\n', contrasts{c});

    % Gather contrast images
    scans = {};
    for s = 1:n_subjects
        con_file = sprintf('/data/sub-%02d/con_%04d.nii,1', s, con_nums(c));
        scans{end+1} = con_file;
    end

    % Setup job
    job = struct();
    job.DesignName = 'MultiSub: One Sample T test on diffs/contrasts';
    job.dir = {sprintf('/output/snpm_con%d', con_nums(c))};
    job.P = scans;
    job.Variance.VarSm = 6;
    job.nPerm = 5000;
    job.vFWHM = [6 6 6];
    job.bVolm = 1;
    job.masking.tm.tm_none = 1;
    job.masking.im = 1;
    job.globalc.g_omit = 1;

    % Run specification and computation
    snpm_run_Specify(job);
    snpm_cp(fullfile(job.dir{1}, 'SnPMcfg.mat'));

    fprintf('Completed: %s\n\n', contrasts{c});
end
```

### Parallel Permutation Computation

```matlab
% For very large datasets, parallelize permutation computation

% Setup parallel pool
parpool(4);  % Use 4 cores

% Modify SnPM to use parfor (requires editing snpm_cp.m)
% Or split permutations across multiple machines

% Example: Compute 10000 permutations in chunks
n_perms_total = 10000;
n_chunks = 10;
perms_per_chunk = n_perms_total / n_chunks;

for chunk = 1:n_chunks
    % Compute chunk
    snpm_cp_chunk(SnPMcfg, perms_per_chunk, chunk);
end

% Combine results
snpm_combine_chunks();
```

## Integration with SPM Pipeline

### After First-Level Analysis

```matlab
% Typical workflow:
% 1. Preprocess in SPM (realignment, normalization, smoothing)
% 2. First-level GLM in SPM
% 3. Extract contrast images
% 4. Group-level inference in SnPM

% After SPM first-level:
clear job;

% Gather all contrast images
con_list = spm_select('FPList', '/data/*/con_0001.nii');

% SnPM group analysis
job.DesignName = 'MultiSub: One Sample T test on diffs/contrasts';
job.dir = {'/output/snpm_group'};
job.P = cellstr(con_list);
job.Variance.VarSm = 6;
job.nPerm = 5000;

% Run
snpm_run_Specify(job);
snpm_cp(fullfile(job.dir{1}, 'SnPMcfg.mat'));
```

### VBM Analysis

```matlab
% Voxel-based morphometry with SnPM

% After CAT12 or SPM VBM preprocessing
% Gather modulated GM images

gm_images = spm_select('FPListRec', '/data', '^mwp1.*\.nii$');

% Two-group comparison
n_controls = 25;
n_patients = 25;

job.DesignName = 'MultiSub: Two Sample T test; 2 Groups: Different Variance';
job.dir = {'/output/snpm_vbm'};
job.P = cellstr(gm_images);
job.n1 = n_controls;
job.n2 = n_patients;
job.Variance.VarSm = 8;  % Larger smoothing for VBM
job.nPerm = 10000;

% Add TIV as covariate
load('tiv_values.mat');  % Total intracranial volume
job.Covariates = tiv;

% Run
snpm_run_Specify(job);
snpm_cp(fullfile(job.dir{1}, 'SnPMcfg.mat'));
```

## Integration with Claude Code

When helping users with SnPM:

1. **Check SPM Installation:**
   ```matlab
   which spm
   spm('ver')  % Should show SPM12
   ```

2. **Verify Data Dimensions:**
   ```matlab
   % All images must have same dimensions
   V = spm_vol(char(scans));
   dims = cat(1, V.dim);
   assert(length(unique(dims(:,1))) == 1);  % Check consistency
   ```

3. **Choose Number of Permutations:**
   - Small samples (n < 20): 5000-10000 permutations
   - Medium samples (n = 20-50): 5000 permutations
   - Large samples (n > 50): 5000 permutations (accuracy saturates)
   - Maximum possible permutations may be limited by sample size

4. **Variance Smoothing:**
   - Default: 6mm is reasonable
   - Match to data smoothing if known
   - Larger smoothing = more sensitivity, less specificity

5. **Common Issues:**
   - Dimension mismatch between images
   - Incorrect number of subjects specified
   - Insufficient permutations for desired precision
   - Memory issues with large datasets

## Troubleshooting

**Problem:** "Dimensions differ between images"
**Solution:** Ensure all images are in same space (check SPM preprocessing)

**Problem:** SnPM runs very slowly
**Solution:** Reduce number of permutations initially for testing, use coarser mask

**Problem:** "Not enough permutations possible"
**Solution:** Small sample size limits unique permutations; use all available or increase n

**Problem:** No significant results
**Solution:** Check data quality, verify preprocessing, try different variance smoothing, increase sample size

**Problem:** Results differ from SPM parametric tests
**Solution:** Expected - SnPM more conservative for small n, parametric more liberal when assumptions hold

## Best Practices

1. **Sample Size:**
   - SnPM valid for any n, but power increases with larger samples
   - For small n (< 15), SnPM preferred over parametric tests
   - Report exact n for each group/condition

2. **Permutations:**
   - Use at least 5000 permutations
   - More permutations = more precise p-values
   - Report number used

3. **Variance Smoothing:**
   - Generally improves sensitivity
   - Try 6-8mm for standard fMRI
   - Can test sensitivity to this parameter

4. **Masking:**
   - Use explicit mask to restrict search volume
   - Reduces computational burden
   - Improves power by reducing multiple comparisons

5. **Reporting:**
   - Report SnPM version
   - Report number of permutations
   - Report variance smoothing used
   - Report cluster-forming and cluster-level thresholds
   - Justify choice of nonparametric vs parametric

## Resources

- **Website:** http://warwick.ac.uk/snpm
- **GitHub:** https://github.com/SnPM-toolbox/SnPM-devel
- **Manual:** http://warwick.ac.uk/snpm/snpm13manual.pdf
- **Mailing List:** https://www.jiscmail.ac.uk/SNPM
- **SPM:** https://www.fil.ion.ucl.ac.uk/spm/

## Citation

```bibtex
@article{nichols2002nonparametric,
  title={Nonparametric permutation tests for functional neuroimaging: a primer with examples},
  author={Nichols, Thomas E and Holmes, Andrew P},
  journal={Human brain mapping},
  volume={15},
  number={1},
  pages={1--25},
  year={2002},
  publisher={Wiley Online Library}
}

@article{ridgway2012snpm13,
  title={SnPM13 Manual},
  author={Ridgway, Gerard and others},
  year={2012},
  url={http://warwick.ac.uk/snpm}
}
```

## Related Tools

- **SPM12:** Parametric statistical parametric mapping
- **PALM:** More flexible permutation testing (FSL)
- **Randomise:** FSL permutation tool
- **NBS:** Network-based statistic for connectivity
- **CAT12:** VBM preprocessing for SnPM
