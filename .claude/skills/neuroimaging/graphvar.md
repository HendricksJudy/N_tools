# GraphVar

## Overview

GraphVar is a user-friendly MATLAB toolbox for comprehensive graph theoretical analysis of brain networks. It provides a graphical interface for constructing connectivity matrices, computing graph metrics, and performing statistical analyses across experimental groups. GraphVar is particularly designed for researchers who want to perform network neuroscience analyses without extensive programming.

**Website:** https://www.nitrc.org/projects/graphvar/
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** GNU General Public License v3

## Key Features

- Graphical user interface for workflow-based analysis
- Connectivity matrix construction (structural and functional)
- Comprehensive graph metrics (global and nodal)
- Group-level statistical comparisons
- Network-Based Statistic (NBS) integration
- Machine learning classification
- Multiple thresholding strategies
- Batch processing capabilities
- Publication-ready visualization
- Support for multiple parcellation schemes

## Installation

### Requirements

```matlab
% MATLAB R2014b or later
% Statistics and Machine Learning Toolbox
% Signal Processing Toolbox (optional)
% Image Processing Toolbox (optional)

% Check MATLAB version
version
```

### Download and Setup

```bash
# Download from NITRC
# https://www.nitrc.org/projects/graphvar/

# Extract to desired location
unzip GraphVar_v2.0.zip -d /path/to/GraphVar

# Add to MATLAB path
```

```matlab
% In MATLAB
addpath(genpath('/path/to/GraphVar'));
savepath;  % Save path permanently

% Launch GraphVar
GraphVar
```

## Data Preparation

### Input Data Formats

GraphVar accepts multiple input types:

1. **Connectivity Matrices** (.mat, .txt, .csv)
2. **Time Series** (.mat, .txt)
3. **Structural Networks** (e.g., from tractography)
4. **Functional Networks** (e.g., from fMRI)

### Organizing Data

```matlab
% Directory structure
% /project/
%   ├── Group1/
%   │   ├── subject001.mat
%   │   ├── subject002.mat
%   │   └── ...
%   ├── Group2/
%   │   ├── subject001.mat
%   │   └── ...
%   └── covariates.txt

% Connectivity matrix format (MATLAB)
% Each .mat file should contain a variable 'connectivity'
% Size: [N x N] where N = number of nodes

% Example: Create connectivity matrix
connectivity = corrcoef(time_series');  % Pearson correlation
connectivity = abs(connectivity);       % Absolute values
save('subject001.mat', 'connectivity');
```

### Subject Information File

```matlab
% Create subject list file (CSV or TXT)
% Format: SubjectID, Group, Age, Sex, ...

% subjects.txt
% subject001,1,25,M
% subject002,1,28,F
% subject003,2,30,M
% subject004,2,27,F
```

## Basic Workflow

### Step 1: Load Data

```matlab
% Launch GraphVar
GraphVar

% GUI Steps:
% 1. Click "Load Data"
% 2. Select data type (e.g., "Connectivity Matrices")
% 3. Browse to data directory
% 4. Select file format (.mat, .txt, etc.)
% 5. Load subject information file
```

### Step 2: Construct Networks

```matlab
% Network construction options in GUI:

% For functional connectivity:
% - Pearson correlation
% - Partial correlation
% - Mutual information
% - Coherence

% For structural connectivity:
% - Tract counts
% - Fractional anisotropy weighted
% - Streamline density
```

### Step 3: Apply Thresholding

```matlab
% Thresholding strategies:

% 1. Proportional threshold (keep top X% of edges)
% Range: 0.05 to 0.50 in steps of 0.05

% 2. Absolute threshold (keep edges above value)
% Range: correlation values (e.g., 0.1 to 0.5)

% 3. Statistical threshold (based on significance)
% Range: p-values (e.g., p < 0.05)

% 4. Minimum Spanning Tree (MST)
% Keeps strongest connected network
```

### Step 4: Compute Graph Metrics

```matlab
% Global metrics:
% - Clustering coefficient
% - Characteristic path length
% - Global efficiency
% - Local efficiency
% - Modularity
% - Assortativity
% - Small-worldness
% - Rich club coefficient

% Nodal metrics:
% - Node degree
% - Node strength
% - Betweenness centrality
% - Closeness centrality
% - Eigenvector centrality
% - Participation coefficient
% - Within-module degree
% - Local efficiency
```

### Step 5: Statistical Analysis

```matlab
% Group comparison options:
% - Two-sample t-test
% - ANOVA
% - ANCOVA (with covariates)
% - Repeated measures ANOVA
% - Permutation testing
% - FDR correction
% - Bonferroni correction
```

## Command-Line Usage

### Loading and Processing Data

```matlab
% Initialize GraphVar project
cfg = [];
cfg.project_name = 'MyNetworkStudy';
cfg.project_dir = '/path/to/project';

% Load connectivity matrices
cfg.data_dir = '/path/to/connectivity/matrices';
cfg.data_format = 'mat';
cfg.matrix_variable = 'connectivity';

% Load subject information
cfg.subject_file = '/path/to/subjects.txt';
cfg.group_variable = 'Group';
cfg.covariates = {'Age', 'Sex'};

% Process data
GraphVar_Process(cfg);
```

### Compute Network Metrics

```matlab
% Set network parameters
cfg.threshold_type = 'proportional';
cfg.threshold_range = 0.05:0.05:0.50;
cfg.binary = false;  % Use weighted networks

% Global metrics
cfg.metrics.global = {
    'clustering_coefficient', ...
    'characteristic_path_length', ...
    'global_efficiency', ...
    'modularity', ...
    'small_worldness'
};

% Nodal metrics
cfg.metrics.nodal = {
    'degree', ...
    'betweenness', ...
    'local_efficiency'
};

% Compute
results = GraphVar_ComputeMetrics(cfg);
```

### Statistical Comparison

```matlab
% Two-group comparison
cfg.stats.test = 'ttest2';
cfg.stats.group1 = [1];  % Group indices
cfg.stats.group2 = [2];
cfg.stats.alpha = 0.05;
cfg.stats.correction = 'fdr';

% Run statistics
stats_results = GraphVar_Statistics(cfg);

% Display significant results
disp('Significant differences in global metrics:');
disp(stats_results.global.significant);

disp('Significant differences in nodal metrics:');
disp(stats_results.nodal.significant);
```

## Advanced Features

### Network-Based Statistic (NBS)

```matlab
% NBS analysis for network-level inference
cfg_nbs = [];
cfg_nbs.method = 'NBS';
cfg_nbs.threshold = 3.0;  % T-statistic threshold
cfg_nbs.nperm = 5000;     % Number of permutations
cfg_nbs.alpha = 0.05;

% Run NBS
nbs_results = GraphVar_NBS(connectivity_group1, connectivity_group2, cfg_nbs);

% Visualize NBS components
GraphVar_VisualizeNBS(nbs_results);
```

### Machine Learning Classification

```matlab
% Configure classifier
cfg_ml = [];
cfg_ml.method = 'SVM';  % or 'LDA', 'RandomForest'
cfg_ml.features = 'graph_metrics';  % or 'connectivity', 'both'
cfg_ml.cv_folds = 10;  % Cross-validation folds
cfg_ml.feature_selection = true;

% Train classifier
ml_results = GraphVar_MachineLearning(cfg_ml);

% Display classification performance
fprintf('Accuracy: %.2f%%\n', ml_results.accuracy * 100);
fprintf('Sensitivity: %.2f%%\n', ml_results.sensitivity * 100);
fprintf('Specificity: %.2f%%\n', ml_results.specificity * 100);
fprintf('AUC: %.2f\n', ml_results.auc);
```

### Modular Organization Analysis

```matlab
% Detect modules/communities
cfg_mod = [];
cfg_mod.algorithm = 'louvain';  % or 'newman', 'infomap'
cfg_mod.gamma = 1.0;  % Resolution parameter
cfg_mod.iterations = 100;

% Compute modules
[modules, Q] = GraphVar_Modularity(connectivity, cfg_mod);

% Module metrics
mod_metrics = GraphVar_ModuleMetrics(connectivity, modules);

% Participation coefficient
PC = mod_metrics.participation_coefficient;

% Within-module degree z-score
WMD = mod_metrics.within_module_degree;
```

### Rich Club Analysis

```matlab
% Rich club coefficient
cfg_rc = [];
cfg_rc.k_range = 1:50;  % Degree range
cfg_rc.nperm = 1000;    % Permutations for null model

% Compute rich club
rc_results = GraphVar_RichClub(connectivity, cfg_rc);

% Plot rich club curve
figure;
plot(rc_results.k, rc_results.phi, 'b-', 'LineWidth', 2);
hold on;
plot(rc_results.k, rc_results.phi_norm, 'r-', 'LineWidth', 2);
xlabel('Degree (k)');
ylabel('Rich Club Coefficient');
legend('Raw \phi(k)', 'Normalized \phi_{norm}(k)');
title('Rich Club Organization');
```

## Batch Processing

### Process Multiple Subjects

```matlab
% Batch processing script
clear; clc;

% Define subjects
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04'};
groups = [1, 1, 2, 2];  % Group assignments

% Parameters
threshold_range = 0.10:0.05:0.40;
metrics = {'degree', 'clustering', 'efficiency'};

% Initialize results
all_results = struct();

% Loop through subjects
for s = 1:length(subjects)
    fprintf('Processing %s...\n', subjects{s});

    % Load connectivity matrix
    load(sprintf('/data/%s_connectivity.mat', subjects{s}));

    % Threshold and compute metrics
    for t = 1:length(threshold_range)
        thresh = threshold_range(t);

        % Apply threshold
        net = threshold_proportional(connectivity, thresh);

        % Compute metrics
        all_results.(subjects{s}).degree(t) = mean(degrees_und(net));
        all_results.(subjects{s}).clustering(t) = mean(clustering_coef_bu(net));
        all_results.(subjects{s}).efficiency(t) = efficiency_bin(net);
    end
end

% Save batch results
save('batch_network_results.mat', 'all_results');
```

### Area Under the Curve (AUC) Analysis

```matlab
% Integrate metrics across thresholds using AUC
subjects = fieldnames(all_results);

for s = 1:length(subjects)
    subj = subjects{s};

    % Compute AUC for each metric
    auc_degree = trapz(threshold_range, all_results.(subj).degree);
    auc_clustering = trapz(threshold_range, all_results.(subj).clustering);
    auc_efficiency = trapz(threshold_range, all_results.(subj).efficiency);

    % Store AUC values
    auc_results.(subj).degree = auc_degree;
    auc_results.(subj).clustering = auc_clustering;
    auc_results.(subj).efficiency = auc_efficiency;
end

% Group comparison using AUC
group1_degree = [];
group2_degree = [];

for s = 1:length(subjects)
    if groups(s) == 1
        group1_degree = [group1_degree; auc_results.(subjects{s}).degree];
    else
        group2_degree = [group2_degree; auc_results.(subjects{s}).degree];
    end
end

[h, p] = ttest2(group1_degree, group2_degree);
fprintf('AUC degree comparison: p = %.4f\n', p);
```

## Visualization

### Network Visualization

```matlab
% Load BrainNet Viewer for 3D visualization
% (GraphVar integrates with BrainNet Viewer)

% Prepare node file
node_file = '/path/to/nodes.node';
% Format: X Y Z Color Size Label

% Prepare edge file
edge_file = '/path/to/edges.edge';
% Format: connectivity matrix

% Generate visualization
GraphVar_Visualize3D(node_file, edge_file);
```

### Statistical Maps

```matlab
% Visualize nodal statistics
cfg_vis = [];
cfg_vis.colormap = 'jet';
cfg_vis.threshold = 0.05;  % p-value threshold
cfg_vis.correction = 'fdr';

% Plot nodal differences
GraphVar_PlotNodalStats(stats_results, cfg_vis);

% Create circular graph
GraphVar_CircularPlot(connectivity, modules);
```

### Generate Reports

```matlab
% Automatic report generation
cfg_report = [];
cfg_report.format = 'pdf';  % or 'html'
cfg_report.include_figures = true;
cfg_report.output_dir = '/path/to/reports';

% Generate comprehensive report
GraphVar_GenerateReport(results, stats_results, cfg_report);
```

## Integration with Claude Code

When helping users with GraphVar:

1. **Check Installation:**
   ```matlab
   which GraphVar
   % Should return: /path/to/GraphVar/GraphVar.m
   ```

2. **Verify Data Format:**
   ```matlab
   % Load and check connectivity matrix
   data = load('subject001.mat');
   disp(fieldnames(data));  % Check variable names
   size(data.connectivity)  % Should be N x N
   ```

3. **Common Workflows:**
   - Functional connectivity: Time series → Correlation → Threshold → Graph metrics
   - Structural connectivity: Tractography → Connectivity matrix → Threshold → Graph metrics
   - Group analysis: Individual metrics → Statistical comparison → Visualization

4. **Best Practices:**
   - Use multiple thresholds (not single)
   - Apply AUC to summarize across thresholds
   - Correct for multiple comparisons (FDR or Bonferroni)
   - Validate findings with permutation testing
   - Report network density alongside metrics

## Troubleshooting

**Problem:** "Undefined function or variable 'GraphVar'"
**Solution:** Add GraphVar directory to MATLAB path with `addpath(genpath('/path/to/GraphVar'))`

**Problem:** Connectivity matrices have different sizes across subjects
**Solution:** Ensure all subjects use the same parcellation scheme (same number of ROIs)

**Problem:** Negative correlations in functional connectivity
**Solution:** Decide on strategy: absolute values, threshold at 0, or keep negatives (controversial)

**Problem:** Network becomes disconnected after thresholding
**Solution:** Use lower threshold or ensure sufficient network density (typically >5-10%)

**Problem:** No significant group differences
**Solution:** Check sample size, effect size, use AUC analysis, try different metrics/thresholds

## Resources

- **NITRC Project:** https://www.nitrc.org/projects/graphvar/
- **User Manual:** Included in download package
- **Publication:** Kruschwitz et al. (2015). "GraphVar: A user-friendly toolbox for comprehensive graph theoretical network analysis." *Journal of Neuroscience Methods*.
- **Forum:** https://www.nitrc.org/forum/?group_id=696
- **Tutorial:** https://www.nitrc.org/docman/?group_id=696

## Citation

```bibtex
@article{kruschwitz2015graphvar,
  title={GraphVar: A user-friendly toolbox for comprehensive graph theoretical network analysis},
  author={Kruschwitz, Johann D and List, Denise and Waller, Lars and Rubinov, Mikail and Walter, Henrik},
  journal={Journal of Neuroscience Methods},
  volume={245},
  pages={107--115},
  year={2015},
  publisher={Elsevier}
}
```

## Related Tools

- **Brain Connectivity Toolbox (BCT):** Core graph theory functions
- **GRETNA:** Alternative MATLAB graph analysis toolbox
- **NBS:** Network-Based Statistic for mass-univariate testing
- **BrainNet Viewer:** 3D network visualization
- **CONN Toolbox:** Functional connectivity preprocessing and analysis
- **NetworkX:** Python alternative for graph analysis
