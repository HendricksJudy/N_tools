# NBS (Network-Based Statistic)

## Overview

The Network-Based Statistic (NBS) is a nonparametric statistical test for identifying connected components in brain networks that differ significantly between groups. Unlike traditional mass-univariate approaches that test each connection independently, NBS controls the family-wise error rate by identifying clusters of interconnected edges, offering greater statistical power for detecting distributed network effects.

**Website:** https://www.nitrc.org/projects/nbs/
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** GNU General Public License

## Key Features

- Network-level statistical inference
- Family-wise error rate control via permutation testing
- Detection of connected components (subnetworks)
- Two-sample t-tests and F-tests
- One-sample t-tests (vs. zero or other value)
- Paired t-tests for repeated measures
- ANCOVA with continuous and categorical covariates
- Flexible contrast specification
- Multiple comparison strategies (extent and intensity)
- Integration with common neuroimaging formats

## Installation

### Requirements

```matlab
% MATLAB R2012b or later
% Statistics and Machine Learning Toolbox
% No additional dependencies required
```

### Download and Setup

```bash
# Download from NITRC
# https://www.nitrc.org/projects/nbs/

# Extract archive
unzip NBS1.2.zip -d /path/to/NBS
```

```matlab
% Add to MATLAB path
addpath('/path/to/NBS');
savepath;

% Launch NBS GUI
NBS
```

## Conceptual Background

### What is NBS?

NBS addresses the multiple comparison problem in network analysis by:

1. **Thresholding:** Apply initial test statistic threshold to identify suprathreshold connections
2. **Component Identification:** Find connected components (subnetworks) in thresholded network
3. **Permutation Testing:** Assess significance of component size using null distribution
4. **FWER Control:** Control family-wise error rate for network-level inference

### When to Use NBS?

✅ **Use NBS when:**
- Testing for group differences in connectivity networks
- Expecting distributed network effects (not isolated connections)
- Need to control family-wise error rate
- Have sufficient sample size (typically n > 20 per group)

❌ **Don't use NBS when:**
- Looking for isolated connection differences (use FDR instead)
- Sample size is very small (n < 15)
- Effect is expected to be diffuse across entire network

## Basic Usage (GUI)

### Step 1: Prepare Data

```matlab
% Data format: 4D matrix
% Dimensions: [N x N x 1 x S]
%   N = number of nodes
%   S = number of subjects

% Example: Load connectivity matrices
subjects = {'sub-01', 'sub-02', 'sub-03'};
n_nodes = 90;  % e.g., AAL parcellation
n_subjects = length(subjects);

% Initialize 4D array
connectivity_group1 = zeros(n_nodes, n_nodes, 1, n_subjects);

% Load individual connectivity matrices
for s = 1:n_subjects
    data = load(sprintf('/data/%s_connectivity.mat', subjects{s}));
    connectivity_group1(:,:,1,s) = data.connectivity;
end

% Save for NBS
save('group1_connectivity.mat', 'connectivity_group1');
```

### Step 2: Launch NBS GUI

```matlab
% Start NBS
NBS

% GUI workflow:
% 1. "Load Matrices" → Select group1_connectivity.mat
% 2. "Load Matrices" → Select group2_connectivity.mat
% 3. "Design Matrix" → Specify experimental design
% 4. "Contrast" → Define contrast of interest
% 5. "Test Statistic" → Choose t-test or F-test
% 6. "Threshold" → Set primary threshold (e.g., 3.0)
% 7. "Permutations" → Set number (e.g., 5000)
% 8. "Run NBS" → Execute analysis
```

### Step 3: Interpret Results

```matlab
% NBS outputs:
% - Significant components (subnetworks)
% - Component size (number of edges)
% - P-value for each component
% - Test statistics for each edge
% - Edge lists for significant components
```

## Command-Line Usage

### Two-Sample T-Test

```matlab
% Load connectivity matrices
load('group1_connectivity.mat');  % [N x N x 1 x S1]
load('group2_connectivity.mat');  % [N x N x 1 x S2]

% Setup NBS parameters
nbs = struct();
nbs.test = 'ttest';           % Two-sample t-test
nbs.thresh = 3.0;             % Primary threshold (t-statistic)
nbs.alpha = 0.05;             % Significance level
nbs.perms = 5000;             % Number of permutations
nbs.exchange = [];            % Leave empty for unpaired test

% Create design matrix
% Group 1: coded as 1
% Group 2: coded as 0
n1 = size(connectivity_group1, 4);
n2 = size(connectivity_group2, 4);
nbs.design = [ones(n1, 1); zeros(n2, 1)];

% Create contrast (Group1 > Group2)
nbs.contrast = [1, -1];

% Combine connectivity matrices
nbs.matrices = cat(4, connectivity_group1, connectivity_group2);

% Run NBS
[nbs_result] = NBSrun(nbs);

% Display results
fprintf('Number of significant components: %d\n', nbs_result.n);
if nbs_result.n > 0
    for c = 1:nbs_result.n
        fprintf('Component %d: %d edges, p = %.4f\n', ...
                c, nbs_result.comp_size(c), nbs_result.pval(c));
    end
end
```

### One-Sample T-Test

```matlab
% Test if connectivity is significantly different from zero
nbs = struct();
nbs.test = 'onesample';
nbs.thresh = 3.5;
nbs.alpha = 0.05;
nbs.perms = 5000;

% Design matrix (column of ones)
n_subjects = size(connectivity, 4);
nbs.design = ones(n_subjects, 1);
nbs.contrast = 1;
nbs.matrices = connectivity;

% Run analysis
[nbs_result] = NBSrun(nbs);
```

### F-Test (Multiple Groups)

```matlab
% Three-group comparison (e.g., Controls, Patients1, Patients2)
nbs = struct();
nbs.test = 'ftest';
nbs.thresh = 10.0;  % F-statistic threshold
nbs.alpha = 0.05;
nbs.perms = 5000;

% Design matrix (one-hot encoding)
n_ctrl = 20;
n_pat1 = 20;
n_pat2 = 20;

nbs.design = [
    ones(n_ctrl, 1),  zeros(n_ctrl, 1), zeros(n_ctrl, 1);
    zeros(n_pat1, 1), ones(n_pat1, 1),  zeros(n_pat1, 1);
    zeros(n_pat2, 1), zeros(n_pat2, 1), ones(n_pat2, 1)
];

% Contrast for omnibus F-test
nbs.contrast = [1, 0, 0; 0, 1, 0; 0, 0, 1];

% Combine all groups
nbs.matrices = cat(4, conn_ctrl, conn_pat1, conn_pat2);

% Run NBS
[nbs_result] = NBSrun(nbs);
```

### Paired T-Test

```matlab
% Within-subject comparison (e.g., pre-post intervention)
nbs = struct();
nbs.test = 'ttest';
nbs.thresh = 3.0;
nbs.alpha = 0.05;
nbs.perms = 5000;

% Exchange blocks for paired test
% Each pair of subjects should be adjacent in matrix
n_subjects = 20;
nbs.exchange = reshape(1:(2*n_subjects), 2, n_subjects)';

% Design matrix
nbs.design = [ones(n_subjects, 1); zeros(n_subjects, 1)];
nbs.contrast = [1, -1];

% Combine pre and post data
nbs.matrices = cat(4, conn_pre, conn_post);

% Run paired NBS
[nbs_result] = NBSrun(nbs);
```

## ANCOVA with Covariates

### Continuous Covariates

```matlab
% Control for age and sex
age = [25, 30, 28, 35, 40, ...]';  % Age for all subjects
sex = [1, 0, 1, 0, 1, ...]';        % Sex (1=M, 0=F)

% Design matrix: [Group, Age, Sex]
nbs.design = [
    [ones(n1, 1); zeros(n2, 1)], ...  % Group
    age, ...                           % Age covariate
    sex                                % Sex covariate
];

% Contrast for group effect (controlling for age and sex)
nbs.contrast = [1, -1, 0, 0];

% Run NBS
[nbs_result] = NBSrun(nbs);
```

### Interaction Effects

```matlab
% Test group × age interaction
% Design: [Group, Age, Group×Age]
group = [ones(n1, 1); zeros(n2, 1)];
age_centered = age - mean(age);
interaction = group .* age_centered;

nbs.design = [group, age_centered, interaction];

% Contrast for interaction
nbs.contrast = [0, 0, 1];

% Run NBS
[nbs_result] = NBSrun(nbs);
```

## Advanced Options

### NBSE (Extent-Based) vs. NBSI (Intensity-Based)

```matlab
% Standard NBS (extent-based)
% Tests: component size (number of edges)
nbs.method = 'run';  % Default
nbs.statistic = 'extent';

% Intensity-based NBS
% Tests: sum of suprathreshold statistics within component
nbs.statistic = 'intensity';

% Run with intensity
[nbs_result] = NBSrun(nbs);
```

### Multiple Primary Thresholds

```matlab
% Test range of thresholds
thresholds = 2.5:0.5:4.0;
results_all = cell(length(thresholds), 1);

for t = 1:length(thresholds)
    nbs.thresh = thresholds(t);
    [nbs_result] = NBSrun(nbs);
    results_all{t} = nbs_result;

    fprintf('Threshold %.1f: %d components\n', ...
            thresholds(t), nbs_result.n);
end
```

### Constrained NBS

```matlab
% Test only subset of connections
% Create mask (1 = test, 0 = exclude)
mask = ones(n_nodes, n_nodes);

% Exclude interhemispheric connections
mask(1:45, 46:90) = 0;
mask(46:90, 1:45) = 0;

% Apply mask to connectivity matrices
nbs.matrices = nbs.matrices .* repmat(mask, [1, 1, 1, size(nbs.matrices, 4)]);

% Run NBS
[nbs_result] = NBSrun(nbs);
```

## Visualization

### Extract Significant Edges

```matlab
% Get edges from significant component
if nbs_result.n > 0
    % Component 1 (most significant)
    comp1_edges = nbs_result.con_mat{1};  % Adjacency matrix

    % Find edge indices
    [row, col] = find(comp1_edges);
    n_edges = length(row);

    fprintf('Significant component has %d edges:\n', n_edges);
    for e = 1:n_edges
        fprintf('  Edge: %d - %d\n', row(e), col(e));
    end

    % Get test statistics for significant edges
    test_stats = nbs_result.test_stat .* comp1_edges;
end
```

### Visualize Network Component

```matlab
% Use BrainNet Viewer or similar
% Prepare edge file
edge_matrix = nbs_result.con_mat{1};  % Binary adjacency matrix

% Weight edges by test statistic
weighted_edges = edge_matrix .* nbs_result.test_stat;

% Save for visualization
save('nbs_component1.mat', 'weighted_edges');

% Load in BrainNet Viewer
% BrainNet('BrainMesh.nv', 'nodes.node', 'nbs_component1.edge');
```

### Plot Component Statistics

```matlab
% Plot component sizes and p-values
if nbs_result.n > 0
    figure;

    subplot(1,2,1);
    bar(1:nbs_result.n, nbs_result.comp_size);
    xlabel('Component');
    ylabel('Number of Edges');
    title('Component Sizes');

    subplot(1,2,2);
    bar(1:nbs_result.n, nbs_result.pval);
    hold on;
    plot([0, nbs_result.n+1], [0.05, 0.05], 'r--', 'LineWidth', 2);
    xlabel('Component');
    ylabel('P-value');
    title('Component Significance');
    ylim([0, 0.1]);
    legend('P-value', '\alpha = 0.05');
end
```

## Batch Processing

```matlab
% Batch NBS analysis for multiple contrasts
clear; clc;

% Load data
load('all_subjects_connectivity.mat');  % [N x N x 1 x S]

% Define contrasts
contrasts = struct();
contrasts(1).name = 'HC > Patient';
contrasts(1).design = [ones(20,1); zeros(20,1)];
contrasts(1).contrast = [1, -1];

contrasts(2).name = 'Patient > HC';
contrasts(2).design = [ones(20,1); zeros(20,1)];
contrasts(2).contrast = [-1, 1];

% NBS parameters
nbs_params = struct();
nbs_params.test = 'ttest';
nbs_params.thresh = 3.0;
nbs_params.alpha = 0.05;
nbs_params.perms = 5000;
nbs_params.matrices = all_connectivity;

% Run all contrasts
results = cell(length(contrasts), 1);

for c = 1:length(contrasts)
    fprintf('\nRunning: %s\n', contrasts(c).name);

    nbs_params.design = contrasts(c).design;
    nbs_params.contrast = contrasts(c).contrast;

    [results{c}] = NBSrun(nbs_params);

    fprintf('Found %d significant components\n', results{c}.n);
    if results{c}.n > 0
        for comp = 1:results{c}.n
            fprintf('  Component %d: %d edges, p = %.4f\n', ...
                    comp, results{c}.comp_size(comp), results{c}.pval(comp));
        end
    end
end

% Save batch results
save('nbs_batch_results.mat', 'results', 'contrasts');
```

## Integration with Other Tools

### From CONN Toolbox

```matlab
% Export connectivity matrices from CONN
% In CONN GUI: Results > Export to MATLAB

% Load CONN output
load('CONN_results.mat');

% Extract ROI-to-ROI matrices
n_subjects = length(CONN_x.subjects);
n_rois = size(CONN_x.Z, 1);

nbs_matrices = zeros(n_rois, n_rois, 1, n_subjects);
for s = 1:n_subjects
    nbs_matrices(:,:,1,s) = CONN_x.Z(:,:,s);
end

% Proceed with NBS analysis
```

### From BCT (Brain Connectivity Toolbox)

```matlab
% Threshold connectivity matrices using BCT
load('raw_connectivity.mat');

% Apply proportional threshold
density = 0.15;  % Keep top 15% of connections

for s = 1:size(connectivity, 4)
    mat = connectivity(:,:,1,s);
    mat_thresh = threshold_proportional(mat, density);
    connectivity(:,:,1,s) = mat_thresh;
end

% Run NBS on thresholded data
```

### Export to GraphVar

```matlab
% Save NBS results for visualization in GraphVar
if nbs_result.n > 0
    comp1 = nbs_result.con_mat{1};

    % Save as GraphVar-compatible format
    save('nbs_comp1_for_graphvar.mat', 'comp1');
end
```

## Integration with Claude Code

When helping users with NBS:

1. **Check Data Dimensions:**
   ```matlab
   size(connectivity)  % Should be [N x N x 1 x S]
   % N = nodes, S = subjects
   ```

2. **Verify Design Matrix:**
   ```matlab
   % Number of rows should match number of subjects
   assert(size(nbs.design, 1) == size(nbs.matrices, 4));
   ```

3. **Choose Appropriate Threshold:**
   - Too low: Many spurious components, reduced power
   - Too high: May miss true effects
   - Typical range: t = 2.5 to 4.0 for t-tests
   - Can test multiple thresholds as sensitivity analysis

4. **Sample Size Considerations:**
   - Minimum: ~15-20 subjects per group
   - Recommended: 25+ subjects per group
   - Increase permutations for small samples (10,000+)

5. **Common Issues:**
   - Negative correlations: Decide to keep, remove, or absolute value
   - Self-connections: Set diagonal to zero
   - Missing data: Impute or exclude subjects

## Effect Size and Power Analysis

### Computing Effect Sizes

```matlab
% Calculate effect sizes for significant components
if nbs_result.n > 0
    comp1_edges = nbs_result.con_mat{1};
    [row, col] = find(comp1_edges);

    % Extract connectivity values for edges in component
    group1_comp_edges = zeros(length(row), size(connectivity_group1, 4));
    group2_comp_edges = zeros(length(row), size(connectivity_group2, 4));

    for e = 1:length(row)
        group1_comp_edges(e, :) = squeeze(connectivity_group1(row(e), col(e), 1, :));
        group2_comp_edges(e, :) = squeeze(connectivity_group2(row(e), col(e), 1, :));
    end

    % Mean connectivity across component edges
    mean_comp_group1 = mean(group1_comp_edges, 1);
    mean_comp_group2 = mean(group2_comp_edges, 1);

    % Cohen's d for component
    pooled_std = sqrt((std(mean_comp_group1)^2 + std(mean_comp_group2)^2) / 2);
    cohens_d = (mean(mean_comp_group1) - mean(mean_comp_group2)) / pooled_std;

    fprintf('Component 1 effect size (Cohen''s d): %.3f\n', cohens_d);
end
```

### Sample Size Estimation

```matlab
% Estimate required sample size for desired power
alpha = 0.05;
power = 0.80;
expected_effect_size = 0.5;  % Cohen's d

% Use power analysis (requires Statistics Toolbox)
n_required = sampsizepwr('t', [0, 1], expected_effect_size, power, [], 'Alpha', alpha);

fprintf('Required sample size per group: %.0f\n', ceil(n_required));
fprintf('Total sample size needed: %.0f\n', ceil(n_required * 2));
```

## Multi-Site Studies

### Site as Covariate

```matlab
% Multi-site data with site as covariate
% Site coding: Site1=1, Site2=2, Site3=3
site = [ones(10,1); 2*ones(10,1); 3*ones(10,1); ...
        ones(10,1); 2*ones(10,1); 3*ones(10,1)];

% Group coding
group = [ones(30,1); zeros(30,1)];

% Create design with site dummies
site_dummy1 = double(site == 1);
site_dummy2 = double(site == 2);
% Site 3 is reference

nbs.design = [group, site_dummy1, site_dummy2];

% Contrast for group effect (controlling for site)
nbs.contrast = [1, -1, 0, 0];

% Run NBS
[nbs_result] = NBSrun(nbs);
```

### ComBat Harmonization

```matlab
% Apply ComBat harmonization before NBS
% Requires ComBat MATLAB implementation

% Reshape connectivity matrices for ComBat
n_subjects = size(all_connectivity, 4);
n_edges = n_nodes * (n_nodes - 1) / 2;

% Extract upper triangle
conn_vectorized = zeros(n_subjects, n_edges);
for s = 1:n_subjects
    mat = all_connectivity(:, :, 1, s);
    conn_vectorized(s, :) = mat(triu(true(n_nodes), 1));
end

% Apply ComBat
% batch = site labels
% mod = covariates to preserve (e.g., group, age)
conn_harmonized = combat(conn_vectorized', batch, mod)';

% Reshape back to matrices
all_connectivity_harmonized = zeros(n_nodes, n_nodes, 1, n_subjects);
for s = 1:n_subjects
    mat = zeros(n_nodes, n_nodes);
    mat(triu(true(n_nodes), 1)) = conn_harmonized(s, :);
    mat = mat + mat';
    all_connectivity_harmonized(:, :, 1, s) = mat;
end

% Proceed with NBS on harmonized data
```

## Python Implementation (NBS-Predict)

### Installation

```bash
# Install NBS-Predict (Python implementation)
pip install nbs-predict

# Or from source
git clone https://github.com/ColeLab/NetworkBasedStatistic.git
cd NetworkBasedStatistic
pip install -e .
```

### Basic Python Usage

```python
import numpy as np
from nbs import nbs_bct

# Load connectivity matrices
# Shape: (n_subjects, n_nodes, n_nodes)
group1_conn = np.load('group1_connectivity.npy')
group2_conn = np.load('group2_connectivity.npy')

# Combine groups
all_conn = np.concatenate([group1_conn, group2_conn], axis=0)

# Create group labels
n1 = group1_conn.shape[0]
n2 = group2_conn.shape[0]
group_labels = np.array([1]*n1 + [2]*n2)

# Run NBS
thresh = 3.0
k = 5000  # Number of permutations

# Two-sample t-test
pval, adj, null = nbs_bct(all_conn, group_labels, thresh, k, tail='both')

# Extract significant components
if adj is not None:
    n_components = len(np.unique(adj)) - 1  # Exclude 0
    print(f'Significant components: {n_components}')

    for comp in range(1, n_components + 1):
        comp_size = np.sum(adj == comp)
        print(f'Component {comp}: {comp_size} edges, p = {pval[comp-1]:.4f}')
```

### Python with Covariates

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load subject data
subjects_df = pd.read_csv('subjects.csv')
age = subjects_df['age'].values
sex = subjects_df['sex'].values  # 0=F, 1=M

# Regress out covariates from each edge
n_subjects = all_conn.shape[0]
n_nodes = all_conn.shape[1]

conn_corrected = np.zeros_like(all_conn)

for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        # Extract edge values
        edge_values = all_conn[:, i, j]

        # Create covariate matrix
        X = np.column_stack([age, sex])

        # Fit regression
        reg = LinearRegression()
        reg.fit(X, edge_values)

        # Get residuals
        residuals = edge_values - reg.predict(X)

        # Store corrected values (symmetric)
        conn_corrected[:, i, j] = residuals
        conn_corrected[:, j, i] = residuals

# Run NBS on corrected connectivity
pval, adj, null = nbs_bct(conn_corrected, group_labels, thresh, k)
```

## Validation and Sensitivity Analysis

### Cross-Validation with Data Splitting

```matlab
% Split-half validation
n_total = size(nbs.matrices, 4);
n_half = floor(n_total / 2);

% Random split
rand_idx = randperm(n_total);
split1_idx = rand_idx(1:n_half);
split2_idx = rand_idx(n_half+1:end);

% NBS on split 1
nbs_split1 = nbs;
nbs_split1.matrices = nbs.matrices(:, :, :, split1_idx);
nbs_split1.design = nbs.design(split1_idx, :);
[result1] = NBSrun(nbs_split1);

% NBS on split 2
nbs_split2 = nbs;
nbs_split2.matrices = nbs.matrices(:, :, :, split2_idx);
nbs_split2.design = nbs.design(split2_idx, :);
[result2] = NBSrun(nbs_split2);

% Compare significant edges
if result1.n > 0 && result2.n > 0
    edges1 = result1.con_mat{1};
    edges2 = result2.con_mat{1};

    % Overlap
    overlap = edges1 .* edges2;
    overlap_pct = nnz(overlap) / max(nnz(edges1), nnz(edges2)) * 100;

    fprintf('Edge overlap between splits: %.1f%%\n', overlap_pct);
end
```

### Threshold Sensitivity Analysis

```matlab
% Test range of thresholds and plot results
thresholds = 2.0:0.2:4.5;
n_thresholds = length(thresholds);

results_sensitivity = struct();
results_sensitivity.thresholds = thresholds;
results_sensitivity.n_components = zeros(n_thresholds, 1);
results_sensitivity.largest_comp_size = zeros(n_thresholds, 1);
results_sensitivity.min_pval = ones(n_thresholds, 1);

for t = 1:n_thresholds
    nbs.thresh = thresholds(t);
    [result] = NBSrun(nbs);

    results_sensitivity.n_components(t) = result.n;

    if result.n > 0
        results_sensitivity.largest_comp_size(t) = max(result.comp_size);
        results_sensitivity.min_pval(t) = min(result.pval);
    end
end

% Plot sensitivity results
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot(thresholds, results_sensitivity.n_components, 'b-o', 'LineWidth', 2);
xlabel('Threshold');
ylabel('Number of Components');
title('Components vs. Threshold');
grid on;

subplot(1, 3, 2);
plot(thresholds, results_sensitivity.largest_comp_size, 'r-o', 'LineWidth', 2);
xlabel('Threshold');
ylabel('Largest Component Size');
title('Component Size vs. Threshold');
grid on;

subplot(1, 3, 3);
semilogy(thresholds, results_sensitivity.min_pval, 'g-o', 'LineWidth', 2);
hold on;
plot([min(thresholds), max(thresholds)], [0.05, 0.05], 'k--', 'LineWidth', 1.5);
xlabel('Threshold');
ylabel('Minimum P-value');
title('Significance vs. Threshold');
legend('Min p-value', '\alpha = 0.05');
grid on;

sgtitle('NBS Threshold Sensitivity Analysis');
```

## Troubleshooting

**Problem:** No significant components found
**Solution:** Try lower threshold, check effect size, increase sample size, verify data quality

**Problem:** Too many components
**Solution:** Increase primary threshold, check for preprocessing issues (motion, etc.)

**Problem:** "Dimensions do not match" error
**Solution:** Ensure design matrix rows = number of subjects in matrices

**Problem:** Results not reproducible
**Solution:** Increase number of permutations (5000-10000), use fixed random seed

**Problem:** Very small p-values (p < 0.0001)
**Solution:** Increase permutations to get more precise p-value estimates

## Best Practices

1. **Preprocessing:**
   - Remove poor quality subjects before NBS
   - Control for motion in fMRI connectivity
   - Verify data normality (transform if needed)

2. **Threshold Selection:**
   - Report results for multiple thresholds
   - Use conservative threshold (3.0-3.5) as primary
   - Check robustness across threshold range

3. **Permutations:**
   - Use minimum 5000 permutations
   - Use 10000+ for publication-quality results
   - More permutations = more precise p-values

4. **Reporting:**
   - Report primary threshold used
   - Report number of permutations
   - Report all significant components (not just largest)
   - Visualize significant components
   - Report component size and p-value

## Resources

- **NITRC Page:** https://www.nitrc.org/projects/nbs/
- **Original Paper:** Zalesky et al. (2010). NeuroImage.
- **User Guide:** Included in download package
- **Forum:** https://www.nitrc.org/forum/?group_id=448
- **Tutorial:** https://www.nitrc.org/docman/?group_id=448

## Citation

```bibtex
@article{zalesky2010network,
  title={Network-based statistic: identifying differences in brain networks},
  author={Zalesky, Andrew and Fornito, Alex and Bullmore, Edward T},
  journal={Neuroimage},
  volume={53},
  number={4},
  pages={1197--1207},
  year={2010},
  publisher={Elsevier}
}
```

## Related Tools

- **Brain Connectivity Toolbox (BCT):** Graph theory metrics
- **CONN Toolbox:** Functional connectivity preprocessing
- **GraphVar:** Comprehensive network analysis
- **GRETNA:** Graph theoretical network analysis
- **BrainNet Viewer:** Network visualization
- **NBS-Predict:** Machine learning extension of NBS
