# GRETNA - Graph Theoretical Network Analysis

## Overview

GRETNA (Graph Theoretical Network Analysis) is a comprehensive MATLAB-based toolbox for analyzing brain networks using graph theory. It provides an intuitive GUI and extensive functions for constructing functional and structural brain networks, calculating graph metrics, performing statistical analyses, and visualizing network properties. GRETNA is widely used for studying brain connectivity in both task-based and resting-state fMRI, as well as structural connectivity from DTI.

**Website:** https://www.nitrc.org/projects/gretna/
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** GPL-3.0

## Key Features

- Functional connectivity network construction
- Structural connectivity from DTI
- Comprehensive graph theory metrics (local and global)
- Multiple thresholding strategies
- Small-world analysis
- Module detection and hub identification
- Network-based statistics (NBS) integration
- Statistical comparison between groups
- Rich Club analysis
- Network visualization with BrainNet Viewer
- Batch processing capabilities
- Integration with SPM and DPABI
- Support for parcellation schemes (AAL, Dosenbach, Power, etc.)

## Installation

```matlab
% Download GRETNA from: https://www.nitrc.org/projects/gretna/

% Extract to desired location
unzip GretnaV2.0.zip

% Add to MATLAB path
addpath(genpath('/path/to/GRETNA_V2.0'));

% Verify installation
which gretna

% Launch GUI
gretna

% Check dependencies
% Required: MATLAB R2014a or later
% Required: SPM12 (for preprocessing)
% Optional: Brain Connectivity Toolbox
% Optional: BrainNet Viewer (for visualization)
```

## Data Preparation

### Preprocessing

```matlab
%% Preprocess fMRI data before GRETNA analysis

% Use SPM, DPABI, or other tools for:
% 1. Slice timing correction
% 2. Realignment
% 3. Normalization to MNI space
% 4. Smoothing (optional, typically 4-6mm)
% 5. Detrending
% 6. Band-pass filtering (0.01-0.1 Hz)
% 7. Nuisance regression (WM, CSF, motion)

% GRETNA expects preprocessed 4D NIfTI files organized as:
% data/
%   sub-01/func/filtered_4d.nii
%   sub-02/func/filtered_4d.nii
%   ...
```

### Define ROIs/Nodes

```matlab
%% Define network nodes using atlas

% GRETNA includes common atlases:
% - AAL (Automated Anatomical Labeling) - 90/116 regions
% - AAL3 - 166 regions
% - Dosenbach - 160 ROIs
% - Power - 264 ROIs
% - Brainnetome - 246 regions
% - Schaefer (100/200/300/400 parcels)

% Load atlas
atlas_file = '/path/to/GRETNA/Atlas/AAL_61x73x61_YCG.nii';

% Or use custom atlas
custom_atlas = '/path/to/my_parcellation.nii';
% Atlas should be integer-labeled ROI mask in MNI space
```

## Basic Workflow - Functional Connectivity

### 1. Extract Time Series

```matlab
%% Launch GRETNA GUI
gretna

% Or programmatically extract ROI time series
data_dir = '/data/preprocessed';
subjects = {'sub-01', 'sub-02', 'sub-03'};
atlas_file = '/path/to/GRETNA/Atlas/AAL_61x73x61_YCG.nii';

for i = 1:length(subjects)
    subj = subjects{i};

    % Input 4D fMRI
    func_file = fullfile(data_dir, subj, 'func', 'filtered_4d.nii');

    % Output time series
    output_file = fullfile(data_dir, subj, 'ROISignals_AAL.mat');

    % Extract using GRETNA function
    gretna_RUN_ROI_Extraction(func_file, atlas_file, output_file);

    fprintf('Extracted time series for %s\n', subj);
end

% ROISignals_AAL.mat contains:
% - ROISignals: matrix of timepoints × regions
% - ROInames: cell array of region names
```

### 2. Calculate Correlation Matrix

```matlab
%% Calculate functional connectivity (correlation)

for i = 1:length(subjects)
    subj = subjects{i};

    % Load time series
    load(fullfile(data_dir, subj, 'ROISignals_AAL.mat'), 'ROISignals');

    % Calculate Pearson correlation
    R = corr(ROISignals);

    % Fisher z-transform
    Z = 0.5 * log((1 + R) ./ (1 - R));
    Z(isinf(Z)) = 0;
    Z(isnan(Z)) = 0;

    % Remove diagonal (self-connections)
    Z(logical(eye(size(Z)))) = 0;

    % Save
    save(fullfile(data_dir, subj, 'FC_matrix.mat'), 'Z', 'R');

    % Visualize
    figure;
    imagesc(Z);
    colorbar;
    title(sprintf('%s Functional Connectivity', subj));
    axis square;
end
```

### 3. Threshold and Binarize

```matlab
%% Threshold correlation matrix

% Load connectivity matrix
load('FC_matrix.mat', 'Z');

% Method 1: Absolute threshold
threshold = 0.3;  % Keep connections with |r| > 0.3
A = abs(Z) > threshold;

% Method 2: Sparsity threshold (keep top X% of connections)
sparsity = 0.1;  % Keep 10% strongest connections
threshold = gretna_threshold(Z, sparsity, 'sparsity');
A = abs(Z) > threshold;

% Method 3: Proportional threshold (density)
density = 0.15;  % 15% connection density
A = gretna_threshold_network(abs(Z), 1, 'Density', density);

% Method 4: Multiple thresholds (for AUC analysis)
densities = 0.05:0.01:0.50;  % 5% to 50% in steps of 1%
for d = 1:length(densities)
    A_range{d} = gretna_threshold_network(abs(Z), 1, 'Density', densities(d));
end

% Binarize
A_binary = double(A);

% Weighted (keep original values)
A_weighted = Z .* A;
```

### 4. Calculate Graph Metrics

```matlab
%% Calculate graph theory metrics

% Binary undirected network
A = gretna_threshold_network(abs(Z), 1, 'Density', 0.1);

% Global metrics
global_efficiency = gretna_node_global_efficiency(A, 'Binary');
local_efficiency = gretna_node_local_efficiency(A, 'Binary');
clustering_coef = gretna_node_clustcoef(A);
characteristic_path = gretna_charpath(A);
modularity = gretna_modularity(A);
small_worldness = gretna_small_world(A);

% Nodal metrics (per ROI)
degree = sum(A, 2);  % Node degree
betweenness = gretna_node_betweenness(A);
nodal_efficiency = gretna_node_efficiency(A);
nodal_clustering = gretna_node_clustcoef(A);

% Hub identification (high degree, high betweenness)
hubs = find(degree > mean(degree) + std(degree));

fprintf('Network characteristics:\n');
fprintf('  Global efficiency: %.3f\n', mean(global_efficiency));
fprintf('  Clustering coefficient: %.3f\n', mean(clustering_coef));
fprintf('  Characteristic path length: %.3f\n', characteristic_path);
fprintf('  Modularity: %.3f\n', modularity(1));
fprintf('  Small-worldness: %.3f\n', small_worldness);
fprintf('  Number of hubs: %d\n', length(hubs));
```

## Advanced Analysis

### Area Under Curve (AUC)

```matlab
%% Calculate metrics across range of densities

% Range of network densities
densities = 0.05:0.01:0.40;
n_densities = length(densities);
n_nodes = size(Z, 1);

% Initialize storage
Cp = zeros(n_densities, 1);  % Clustering coefficient
Lp = zeros(n_densities, 1);  % Path length
Eglob = zeros(n_densities, 1);  % Global efficiency
Eloc = zeros(n_nodes, n_densities);  % Local efficiency

% Calculate at each density
for d = 1:n_densities
    A = gretna_threshold_network(abs(Z), 1, 'Density', densities(d));

    % Global metrics
    Cp(d) = mean(gretna_node_clustcoef(A));
    Lp(d) = gretna_charpath(A);
    Eglob(d) = mean(gretna_node_global_efficiency(A, 'Binary'));

    % Nodal metrics
    Eloc(:, d) = gretna_node_local_efficiency(A, 'Binary');
end

% Calculate AUC (area under curve)
AUC_Cp = trapz(densities, Cp);
AUC_Lp = trapz(densities, Lp);
AUC_Eglob = trapz(densities, Eglob);
AUC_Eloc = trapz(densities, Eloc, 2);

% Visualize
figure;
subplot(2,2,1); plot(densities, Cp); title('Clustering'); xlabel('Density');
subplot(2,2,2); plot(densities, Lp); title('Path Length'); xlabel('Density');
subplot(2,2,3); plot(densities, Eglob); title('Global Efficiency'); xlabel('Density');
subplot(2,2,4); plot(densities, mean(Eloc,1)); title('Mean Local Efficiency'); xlabel('Density');
```

### Module Detection

```matlab
%% Detect community structure

% Threshold network
A = gretna_threshold_network(abs(Z), 1, 'Density', 0.15);

% Weighted network
W = Z .* A;

% Module detection (Louvain algorithm)
[modules, Q] = gretna_modularity(W, 'Weighted');

% modules: vector of module assignments for each node
% Q: modularity value (quality of partition)

% Number of modules
n_modules = max(modules);
fprintf('Detected %d modules (Q = %.3f)\n', n_modules, Q(1));

% Participation coefficient (inter-module connections)
PC = gretna_participation_coef(W, modules);

% Within-module degree z-score
WMD = gretna_module_degree_zscore(W, modules);

% Hub classification (Guimerà & Amaral, 2005)
% Provincial hubs: high WMD, low PC (within-module)
% Connector hubs: high WMD, high PC (between-module)

provincial_hubs = find(WMD > 2.5 & PC < 0.3);
connector_hubs = find(WMD > 2.5 & PC > 0.3);
```

### Rich Club Analysis

```matlab
%% Identify rich club organization

% Rich club = densely interconnected high-degree nodes

% Degree distribution
k = sum(A, 2);

% Rich club coefficient
k_range = min(k):max(k);
rich_club = zeros(length(k_range), 1);

for i = 1:length(k_range)
    k_level = k_range(i);

    % Nodes with degree > k_level
    rich_nodes = find(k > k_level);

    if length(rich_nodes) > 1
        % Subnetwork of rich nodes
        A_rich = A(rich_nodes, rich_nodes);

        % Rich club coefficient
        E_rich = sum(A_rich(:)) / 2;  % Number of edges
        N_rich = length(rich_nodes);
        possible_edges = N_rich * (N_rich - 1) / 2;

        rich_club(i) = E_rich / possible_edges;
    end
end

% Normalized rich club (compare to random networks)
n_rand = 100;
rich_club_rand = zeros(length(k_range), n_rand);

for r = 1:n_rand
    % Generate random network with same degree distribution
    A_rand = gretna_gen_random_network(A, 'Maslov', 20);
    k_rand = sum(A_rand, 2);

    for i = 1:length(k_range)
        rich_nodes = find(k_rand > k_range(i));
        if length(rich_nodes) > 1
            A_rich = A_rand(rich_nodes, rich_nodes);
            E_rich = sum(A_rich(:)) / 2;
            N_rich = length(rich_nodes);
            rich_club_rand(i, r) = E_rich / (N_rich * (N_rich - 1) / 2);
        end
    end
end

% Normalized rich club
phi_norm = rich_club ./ mean(rich_club_rand, 2);

% Plot
figure;
plot(k_range, phi_norm, 'LineWidth', 2);
hold on;
plot([min(k_range) max(k_range)], [1 1], 'r--');
xlabel('Degree (k)'); ylabel('Normalized Rich Club (φ_{norm})');
title('Rich Club Organization');
```

## Group Comparison

### Two-Sample T-Test

```matlab
%% Compare network metrics between groups

% Groups
controls = {'sub-01', 'sub-02', 'sub-03'};
patients = {'sub-04', 'sub-05', 'sub-06'};

n_controls = length(controls);
n_patients = length(patients);

% Extract metrics for each group
metrics_controls = zeros(n_controls, n_nodes);
metrics_patients = zeros(n_patients, n_nodes);

for i = 1:n_controls
    load(fullfile(data_dir, controls{i}, 'FC_matrix.mat'), 'Z');
    A = gretna_threshold_network(abs(Z), 1, 'Density', 0.15);
    metrics_controls(i, :) = gretna_node_efficiency(A);
end

for i = 1:n_patients
    load(fullfile(data_dir, patients{i}, 'FC_matrix.mat'), 'Z');
    A = gretna_threshold_network(abs(Z), 1, 'Density', 0.15);
    metrics_patients(i, :) = gretna_node_efficiency(A);
end

% Two-sample t-test at each node
[h, p, ci, stats] = ttest2(metrics_controls, metrics_patients);

% FDR correction
p_fdr = gretna_FDR(p, 0.05);

% Significant nodes
sig_nodes = find(p_fdr < 0.05);
fprintf('Found %d significantly different nodes (FDR < 0.05)\n', length(sig_nodes));
```

### Network-Based Statistic (NBS)

```matlab
%% Identify subnetworks with group differences

% NBS identifies connected components with differences

% Connectivity matrices for each group
Z_controls = zeros(n_nodes, n_nodes, n_controls);
Z_patients = zeros(n_nodes, n_nodes, n_patients);

% Load data...
% (populate Z_controls and Z_patients)

% Run NBS
thresh = 3.0;  % T-statistic threshold
n_perm = 5000;  % Number of permutations

[pval, adj, null] = gretna_NBS(Z_controls, Z_patients, thresh, n_perm);

% pval: p-value for each component
% adj: adjacency of significant component(s)
% null: null distribution

% Significant subnetworks
sig_components = find(pval < 0.05);
fprintf('Found %d significant subnetworks\n', length(sig_components));
```

## Visualization

### BrainNet Viewer Integration

```matlab
%% Visualize networks with BrainNet Viewer

% Create node file (ROI coordinates and values)
% Format: X Y Z Color Size Label

% Load atlas coordinates
load('/path/to/GRETNA/Atlas/AAL_coordinates.mat');  % MNI coordinates

% Node sizes based on degree
degree = sum(A, 2);
node_sizes = 1 + (degree - min(degree)) / (max(degree) - min(degree)) * 5;

% Create node file
node_file = 'nodes.node';
fid = fopen(node_file, 'w');
for i = 1:n_nodes
    fprintf(fid, '%g %g %g %g %g %s\n', ...
            coords(i,1), coords(i,2), coords(i,3), ...
            degree(i), node_sizes(i), roi_names{i});
end
fclose(fid);

% Create edge file (connectivity matrix)
edge_file = 'edges.edge';
dlmwrite(edge_file, A, 'delimiter', '\t');

% Visualize with BrainNet Viewer
BrainNet_MapCfg('BrainMesh_ICBM152.nv', node_file, edge_file, 'network.png');
```

### GRETNA Built-in Visualization

```matlab
%% Use GRETNA's visualization functions

% Plot connectivity matrix
gretna_plot_matrix(Z, 'Correlation Matrix');

% Plot degree distribution
degree = sum(A, 2);
gretna_plot_histogram(degree, 'Degree Distribution');

% Circular graph
gretna_plot_circular(A, modules);

% Glass brain
gretna_plot_glass_brain(A, coords);
```

## Batch Processing

```matlab
%% Complete pipeline for multiple subjects

subjects = dir('data/sub-*');
subjects = {subjects.name};

atlas_file = '/path/to/atlas.nii';
output_dir = '/results/network_analysis';

% Parameters
densities = 0.05:0.01:0.40;

for s = 1:length(subjects)
    subj = subjects{s};

    fprintf('Processing %s...\n', subj);

    % 1. Extract time series
    func_file = fullfile('data', subj, 'func', 'filtered_4d.nii');
    ts_file = fullfile(output_dir, subj, 'timeseries.mat');
    gretna_RUN_ROI_Extraction(func_file, atlas_file, ts_file);

    % 2. Calculate connectivity
    load(ts_file, 'ROISignals');
    R = corr(ROISignals);
    Z = atanh(R);
    Z(isinf(Z)) = 0;

    % 3. Calculate metrics across densities
    AUC_metrics = gretna_calculate_AUC(Z, densities);

    % 4. Save
    save(fullfile(output_dir, subj, 'results.mat'), 'Z', 'AUC_metrics');

    fprintf('%s complete\n\n', subj);
end
```

## Integration with Claude Code

When helping users with GRETNA:

1. **Check Installation:**
   ```matlab
   which gretna
   gretna  % Should launch GUI
   ```

2. **Common Issues:**
   - SPM12 not in path
   - Atlas files missing
   - Negative path lengths (disconnected network)
   - Memory errors with large matrices
   - Threshold too strict (no edges)

3. **Best Practices:**
   - Use AUC analysis (multiple thresholds)
   - Fisher z-transform correlations
   - Compare to random networks for normalization
   - Use FDR correction for multiple comparisons
   - Document threshold selection rationale
   - Visualize results with BrainNet Viewer
   - Check network connectivity before metrics
   - Report both binary and weighted metrics

4. **Parameter Selection:**
   - Density: 5-40% typical range
   - Threshold: Balance sensitivity/specificity
   - Permutations: 5000-10000 for NBS
   - Atlas: Match to research question

## Troubleshooting

**Problem:** Inf or NaN in metrics
**Solution:** Check for disconnected networks, remove self-connections, verify matrix symmetry

**Problem:** All nodes classified as hubs
**Solution:** Threshold may be too high, adjust hub criteria, check degree distribution

**Problem:** Module detection unstable
**Solution:** Run multiple times (stochastic algorithm), increase network density, use consensus clustering

**Problem:** Small-worldness < 1
**Solution:** Network may not be small-world, check preprocessing, adjust threshold

**Problem:** Negative path lengths
**Solution:** Network disconnected, reduce threshold, check for isolated nodes

## Resources

- Website: https://www.nitrc.org/projects/gretna/
- Manual: Included with download
- Forum: https://www.nitrc.org/forum/?group_id=1155
- Publications: http://www.gaolaboratory.org/
- Brain Connectivity Toolbox: https://sites.google.com/site/bctnet/

## Citation

```bibtex
@article{wang2015gretna,
  title={GRETNA: a graph theoretical network analysis toolbox for imaging connectomics},
  author={Wang, Jinhui and Wang, Xindi and Xia, Mingrui and Liao, Xuhong and Evans, Alan and He, Yong},
  journal={Frontiers in human neuroscience},
  volume={9},
  pages={386},
  year={2015}
}
```

## Related Tools

- **Brain Connectivity Toolbox (BCT):** MATLAB/Python graph theory
- **BrainNet Viewer:** Network visualization
- **CONN:** Functional connectivity analysis
- **GraphVar:** Multivariate graph analysis
- **NetworkX:** Python network analysis library
- **NBS:** Network-Based Statistic
