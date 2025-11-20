# BRAPH2 (BRain Analysis using graPH theory 2)

## Overview

BRAPH2 (BRain Analysis using graPH theory 2) is a comprehensive MATLAB software for brain network analysis that combines graph theory, statistics, machine learning, and deep learning. It provides both a graphical user interface and extensive scripting capabilities for analyzing structural, functional, and diffusion MRI connectivity data. BRAPH2 implements over 100 graph measures, supports multi-layer and multiplex networks, and offers advanced statistical comparisons and machine learning for network-based classification.

**Website:** https://braph.org/
**GitHub:** https://github.com/braph-software/BRAPH-2
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB (Object-Oriented)
**License:** GNU General Public License v3.0

## Key Features

- 100+ graph theory measures (degree, centrality, clustering, modularity, etc.)
- Multi-layer and multiplex network analysis
- Structural, functional, and DTI connectivity support
- Graphical user interface for interactive analysis
- Object-oriented scripting interface
- Group comparisons with permutation testing
- Machine learning for network classification
- Deep learning integration
- Longitudinal network analysis
- Network visualization with brain anatomy
- Publication-quality figures
- Extensive documentation and tutorials
- Modular and extensible architecture

## Installation

### Requirements

```matlab
% MATLAB R2020b or later recommended
% Statistics and Machine Learning Toolbox
% Deep Learning Toolbox (optional, for deep learning features)
% Parallel Computing Toolbox (optional, for speedup)
```

### Download and Setup

```bash
# Clone from GitHub
git clone https://github.com/braph-software/BRAPH-2.git
cd BRAPH-2

# Or download ZIP from https://braph.org/
unzip BRAPH-2-main.zip
```

```matlab
% Add BRAPH2 to MATLAB path
addpath(genpath('/path/to/BRAPH-2'));
savepath;

% Verify installation
braph2  % Launch BRAPH2 GUI

% Check version
fprintf('BRAPH2 version: %s\n', BRAPH2.VERSION);
```

## Conceptual Background

### Graph Types in BRAPH2

BRAPH2 supports multiple graph types:

- **Binary Undirected (BU):** Unweighted, symmetric networks
- **Binary Directed (BD):** Unweighted, asymmetric networks
- **Weighted Undirected (WU):** Weighted, symmetric networks
- **Weighted Directed (WD):** Weighted, asymmetric networks
- **Multilayer:** Multiple layers with inter-layer connections
- **Multiplex:** Multiple layers with node correspondence

### Data Types

- **Structural connectivity:** DTI tractography, morphometric similarity
- **Functional connectivity:** Resting-state fMRI, task fMRI
- **Multiplex:** Combining structural and functional layers
- **Longitudinal:** Time series of networks

## GUI-Based Analysis

### Step 1: Launch GUI

```matlab
% Start BRAPH2 graphical interface
braph2
```

### Step 2: Import Data via GUI

```matlab
% GUI Workflow:
% 1. File → New → Connectivity Data
% 2. Select data type (Structural/Functional)
% 3. Load connectivity matrices (*.txt, *.xlsx, *.mat)
% 4. Import subject information
% 5. Define groups
% 6. Set parcellation/atlas
```

### Step 3: Create Graph

```matlab
% GUI Workflow:
% 1. Analysis → Create Graph
% 2. Select graph type (BU, BD, WU, WD)
% 3. Set thresholds (density, absolute)
% 4. Configure graph properties
% 5. Visualize graph structure
```

### Step 4: Compute Measures

```matlab
% GUI Workflow:
% 1. Analysis → Calculate Measures
% 2. Select measures (degree, betweenness, clustering, etc.)
% 3. Compute for all subjects
% 4. View results in table/plot
% 5. Export results
```

### Step 5: Group Comparison

```matlab
% GUI Workflow:
% 1. Analysis → Compare Groups
% 2. Select groups to compare
% 3. Choose statistical test (permutation, parametric)
% 4. Set number of permutations (e.g., 1000)
% 5. Run comparison
% 6. View significant differences
% 7. Generate figures
```

## Scripting Interface

### Basic Graph Creation

```matlab
% Load connectivity matrix
connectivity = readmatrix('connectivity_matrix.txt');
n_nodes = size(connectivity, 1);

% Create graph object (Weighted Undirected)
graph = GraphWU('B', connectivity);

% Calculate all available measures
measure_list = graph.getMeasureList();
fprintf('Available measures: %d\n', length(measure_list));

% Compute specific measure (e.g., degree)
degree = graph.measure('Degree');
degree_values = degree.getValue();

fprintf('Node degrees:\n');
disp(degree_values);
```

### Multi-Subject Analysis

```matlab
% Load multiple subjects
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04'};
n_subjects = length(subjects);
n_nodes = 90;  % AAL parcellation

% Initialize connectivity cell array
conn_matrices = cell(n_subjects, 1);

for s = 1:n_subjects
    % Load connectivity matrix for each subject
    filename = sprintf('data/%s_connectivity.txt', subjects{s});
    conn_matrices{s} = readmatrix(filename);
end

% Create graph for each subject
graphs = cell(n_subjects, 1);
for s = 1:n_subjects
    graphs{s} = GraphWU('B', conn_matrices{s});
end

% Compute measures across subjects
all_degrees = zeros(n_subjects, n_nodes);
for s = 1:n_subjects
    degree_measure = graphs{s}.measure('Degree');
    all_degrees(s, :) = degree_measure.getValue();
end

% Average degree across subjects
mean_degree = mean(all_degrees, 1);
std_degree = std(all_degrees, 0, 1);

fprintf('Mean node degree: %.2f ± %.2f\n', mean(mean_degree), mean(std_degree));
```

### Thresholding Strategies

```matlab
% Load raw connectivity matrix
conn_raw = readmatrix('connectivity_raw.txt');

% 1. Absolute threshold
threshold_abs = 0.3;
conn_abs = conn_raw .* (conn_raw > threshold_abs);
graph_abs = GraphWU('B', conn_abs);

% 2. Proportional threshold (keep top X% of connections)
density = 0.15;  % Keep 15% strongest connections
conn_sorted = sort(conn_raw(:), 'descend');
threshold_prop = conn_sorted(round(density * numel(conn_raw)));
conn_prop = conn_raw .* (conn_raw > threshold_prop);
graph_prop = GraphWU('B', conn_prop);

% 3. Binary threshold
conn_binary = double(conn_raw > threshold_abs);
graph_binary = GraphBU('B', conn_binary);

fprintf('Absolute threshold: %d edges\n', nnz(conn_abs)/2);
fprintf('Proportional threshold: %d edges\n', nnz(conn_prop)/2);
fprintf('Binary threshold: %d edges\n', nnz(conn_binary)/2);
```

## Comprehensive Graph Measures

### Centrality Measures

```matlab
% Create graph
graph = GraphWU('B', connectivity);

% Degree centrality
degree = graph.measure('Degree');
degree_vals = degree.getValue();

% Strength (weighted degree)
strength = graph.measure('Strength');
strength_vals = strength.getValue();

% Betweenness centrality
betweenness = graph.measure('Betweenness');
betweenness_vals = betweenness.getValue();

% Closeness centrality
closeness = graph.measure('Closeness');
closeness_vals = closeness.getValue();

% Eigenvector centrality
eigenvector = graph.measure('EigenvectorCentrality');
eigenvector_vals = eigenvector.getValue();

% Identify hubs (top 10% by degree)
degree_threshold = prctile(degree_vals, 90);
hub_nodes = find(degree_vals > degree_threshold);

fprintf('Hub nodes: ');
disp(hub_nodes');
```

### Clustering and Modularity

```matlab
% Clustering coefficient
clustering = graph.measure('Clustering');
clustering_vals = clustering.getValue();
global_clustering = mean(clustering_vals);

fprintf('Global clustering coefficient: %.4f\n', global_clustering);

% Transitivity
transitivity = graph.measure('Transitivity');
transitivity_val = transitivity.getValue();

fprintf('Transitivity: %.4f\n', transitivity_val);

% Modularity (community detection)
modularity = graph.measure('Modularity');
modularity_val = modularity.getValue();
community_structure = modularity.getCommunityStructure();

fprintf('Modularity Q: %.4f\n', modularity_val);
fprintf('Number of communities: %d\n', max(community_structure));

% Within-module degree z-score
within_module_z = graph.measure('WithinModuleDegreeZscore');
within_module_z_vals = within_module_z.getValue();

% Participation coefficient
participation = graph.measure('ParticipationCoefficient');
participation_vals = participation.getValue();
```

### Path Length and Efficiency

```matlab
% Characteristic path length
path_length = graph.measure('PathLength');
path_length_val = path_length.getValue();

fprintf('Characteristic path length: %.4f\n', path_length_val);

% Global efficiency
global_eff = graph.measure('GlobalEfficiency');
global_eff_val = global_eff.getValue();

fprintf('Global efficiency: %.4f\n', global_eff_val);

% Local efficiency
local_eff = graph.measure('LocalEfficiency');
local_eff_vals = local_eff.getValue();
mean_local_eff = mean(local_eff_vals);

fprintf('Mean local efficiency: %.4f\n', mean_local_eff);

% Nodal efficiency
nodal_eff = graph.measure('NodalEfficiency');
nodal_eff_vals = nodal_eff.getValue();
```

### Small-World Analysis

```matlab
% Compute small-world properties
clustering = graph.measure('Clustering');
C = mean(clustering.getValue());

path_length = graph.measure('PathLength');
L = path_length.getValue();

% Generate random graph for comparison
n_random = 100;
C_rand = zeros(n_random, 1);
L_rand = zeros(n_random, 1);

for r = 1:n_random
    % Create random graph with same density
    rand_graph = graph.randomize();

    C_rand(r) = mean(rand_graph.measure('Clustering').getValue());
    L_rand(r) = rand_graph.measure('PathLength').getValue();
end

% Small-world metrics
gamma = C / mean(C_rand);  % Normalized clustering
lambda = L / mean(L_rand);  % Normalized path length
sigma = gamma / lambda;     % Small-world index

fprintf('Small-world analysis:\n');
fprintf('  Clustering ratio (gamma): %.4f\n', gamma);
fprintf('  Path length ratio (lambda): %.4f\n', lambda);
fprintf('  Small-world index (sigma): %.4f\n', sigma);

if sigma > 1
    fprintf('  Network exhibits small-world properties\n');
end
```

### Rich Club Analysis

```matlab
% Rich club coefficient
rich_club = graph.measure('RichClub');
rich_club_vals = rich_club.getValue();

% Get rich club coefficient for different degree thresholds
k_levels = 1:20;  % Degree levels
phi = zeros(length(k_levels), 1);

for i = 1:length(k_levels)
    phi(i) = rich_club_vals(k_levels(i));
end

% Plot rich club curve
figure;
plot(k_levels, phi, 'b-o', 'LineWidth', 2);
xlabel('Degree threshold (k)');
ylabel('Rich club coefficient \phi(k)');
title('Rich Club Analysis');
grid on;

% Compare to random networks
phi_rand = zeros(length(k_levels), n_random);
for r = 1:n_random
    rand_graph = graph.randomize();
    rc_rand = rand_graph.measure('RichClub');
    phi_rand(:, r) = rc_rand.getValue()(k_levels);
end

phi_rand_mean = mean(phi_rand, 2);
phi_rand_std = std(phi_rand, 0, 2);

% Normalized rich club coefficient
phi_norm = phi ./ phi_rand_mean;

hold on;
plot(k_levels, phi_rand_mean, 'r--', 'LineWidth', 1.5);
legend('Observed', 'Random');
```

## Group Comparison Analysis

### Two-Group Comparison

```matlab
% Load data for two groups
group1_files = {'ctrl_01.txt', 'ctrl_02.txt', 'ctrl_03.txt'};
group2_files = {'patient_01.txt', 'patient_02.txt', 'patient_03.txt'};

% Load connectivity matrices
group1_graphs = cell(length(group1_files), 1);
group2_graphs = cell(length(group2_files), 1);

for s = 1:length(group1_files)
    conn = readmatrix(group1_files{s});
    group1_graphs{s} = GraphWU('B', conn);
end

for s = 1:length(group2_files)
    conn = readmatrix(group2_files{s});
    group2_graphs{s} = GraphWU('B', conn);
end

% Compute measure for all subjects (e.g., degree)
measure_name = 'Degree';

group1_measures = zeros(length(group1_graphs), n_nodes);
group2_measures = zeros(length(group2_graphs), n_nodes);

for s = 1:length(group1_graphs)
    m = group1_graphs{s}.measure(measure_name);
    group1_measures(s, :) = m.getValue();
end

for s = 1:length(group2_graphs)
    m = group2_graphs{s}.measure(measure_name);
    group2_measures(s, :) = m.getValue();
end

% Statistical comparison (permutation test)
n_perms = 1000;
p_values = zeros(n_nodes, 1);
t_stats = zeros(n_nodes, 1);

for node = 1:n_nodes
    [~, p_values(node), ~, stats] = ttest2(group1_measures(:, node), ...
                                            group2_measures(:, node));
    t_stats(node) = stats.tstat;
end

% FDR correction
p_fdr = mafdr(p_values, 'BHFDR', true);

% Find significant nodes
sig_nodes = find(p_fdr < 0.05);

fprintf('Significant nodes (FDR < 0.05): %d\n', length(sig_nodes));
fprintf('Node indices: ');
disp(sig_nodes');
```

### Permutation-Based Group Comparison

```matlab
% Permutation test for group differences
n_perm = 5000;
all_subjects = [group1_measures; group2_measures];
n1 = size(group1_measures, 1);
n2 = size(group2_measures, 1);
n_total = n1 + n2;

% Observed difference
obs_diff = mean(group1_measures, 1) - mean(group2_measures, 1);

% Permutation distribution
perm_diff = zeros(n_perm, n_nodes);

for perm = 1:n_perm
    % Randomly permute group labels
    perm_idx = randperm(n_total);
    perm_group1 = all_subjects(perm_idx(1:n1), :);
    perm_group2 = all_subjects(perm_idx(n1+1:end), :);

    perm_diff(perm, :) = mean(perm_group1, 1) - mean(perm_group2, 1);
end

% Calculate p-values
p_perm = zeros(n_nodes, 1);
for node = 1:n_nodes
    p_perm(node) = sum(abs(perm_diff(:, node)) >= abs(obs_diff(node))) / n_perm;
end

% Significant nodes
sig_nodes_perm = find(p_perm < 0.05);

fprintf('Permutation test:\n');
fprintf('  Significant nodes (p < 0.05): %d\n', length(sig_nodes_perm));
```

## Multi-Layer Network Analysis

### Creating Multiplex Networks

```matlab
% Load structural and functional connectivity
struct_conn = readmatrix('structural_connectivity.txt');
func_conn = readmatrix('functional_connectivity.txt');

% Create multiplex graph (2 layers)
% Layer 1: Structural
% Layer 2: Functional

multiplex_graph = MultiplexWU();
multiplex_graph.addLayer(struct_conn, 'Structural');
multiplex_graph.addLayer(func_conn, 'Functional');

% Set interlayer connections (node correspondence)
multiplex_graph.setInterlayerConnections('diagonal');  % One-to-one correspondence

% Compute multiplex measures
% Multilayer degree
ml_degree = multiplex_graph.measure('MultiplexDegree');
ml_degree_vals = ml_degree.getValue();

% Overlapping degree (nodes with high degree in multiple layers)
overlap_degree = multiplex_graph.measure('OverlappingDegree');
overlap_vals = overlap_degree.getValue();

% Participation coefficient across layers
ml_participation = multiplex_graph.measure('MultiplexParticipation');
ml_part_vals = ml_participation.getValue();

fprintf('Multiplex analysis:\n');
fprintf('  Mean multilayer degree: %.2f\n', mean(ml_degree_vals));
fprintf('  Mean overlapping degree: %.2f\n', mean(overlap_vals));
```

### Layer-Specific Analysis

```matlab
% Extract individual layers
layer1_graph = multiplex_graph.getLayer(1);
layer2_graph = multiplex_graph.getLayer(2);

% Compute measures for each layer
layer1_degree = layer1_graph.measure('Degree').getValue();
layer2_degree = layer2_graph.measure('Degree').getValue();

% Correlation between layers
layer_correlation = corr(layer1_degree, layer2_degree);

fprintf('Structural-functional degree correlation: %.3f\n', layer_correlation);

% Identify nodes with high degree in both layers
high_degree_both = find((layer1_degree > median(layer1_degree)) & ...
                        (layer2_degree > median(layer2_degree)));

fprintf('Nodes with high degree in both layers: %d\n', length(high_degree_both));
```

## Machine Learning for Classification

### Feature Extraction

```matlab
% Extract graph features for classification
n_subjects_total = n1 + n2;
features = [];

measure_names = {'Degree', 'Clustering', 'Betweenness', 'LocalEfficiency'};

for m = 1:length(measure_names)
    measure_vals = zeros(n_subjects_total, n_nodes);

    % Group 1
    for s = 1:n1
        meas = group1_graphs{s}.measure(measure_names{m});
        measure_vals(s, :) = meas.getValue();
    end

    % Group 2
    for s = 1:n2
        meas = group2_graphs{s}.measure(measure_names{m});
        measure_vals(n1 + s, :) = meas.getValue();
    end

    features = [features, measure_vals];
end

% Create labels
labels = [ones(n1, 1); 2*ones(n2, 1)];

fprintf('Feature matrix: %d subjects x %d features\n', size(features));
```

### Support Vector Machine Classification

```matlab
% Split into training and testing
cv = cvpartition(n_subjects_total, 'HoldOut', 0.3);
train_idx = training(cv);
test_idx = test(cv);

X_train = features(train_idx, :);
y_train = labels(train_idx);
X_test = features(test_idx, :);
y_test = labels(test_idx);

% Train SVM
svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', 'Standardize', true);

% Predict on test set
y_pred = predict(svm_model, X_test);

% Evaluate performance
accuracy = sum(y_pred == y_test) / length(y_test);
confusion_mat = confusionmat(y_test, y_pred);

fprintf('Classification results:\n');
fprintf('  Accuracy: %.2f%%\n', accuracy * 100);
fprintf('  Confusion matrix:\n');
disp(confusion_mat);
```

### Cross-Validation

```matlab
% 10-fold cross-validation
k_folds = 10;
cv = cvpartition(n_subjects_total, 'KFold', k_folds);

accuracies = zeros(k_folds, 1);
sensitivities = zeros(k_folds, 1);
specificities = zeros(k_folds, 1);

for fold = 1:k_folds
    % Get train/test indices
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);

    % Train model
    model = fitcsvm(features(train_idx, :), labels(train_idx), ...
                    'KernelFunction', 'rbf', 'Standardize', true);

    % Predict
    pred = predict(model, features(test_idx, :));
    true_labels = labels(test_idx);

    % Metrics
    accuracies(fold) = sum(pred == true_labels) / length(true_labels);

    % Sensitivity and specificity (assuming binary classification)
    tp = sum((pred == 2) & (true_labels == 2));
    tn = sum((pred == 1) & (true_labels == 1));
    fp = sum((pred == 2) & (true_labels == 1));
    fn = sum((pred == 1) & (true_labels == 2));

    sensitivities(fold) = tp / (tp + fn);
    specificities(fold) = tn / (tn + fp);
end

fprintf('Cross-validation results:\n');
fprintf('  Mean accuracy: %.2f%% (±%.2f%%)\n', mean(accuracies)*100, std(accuracies)*100);
fprintf('  Mean sensitivity: %.2f%% (±%.2f%%)\n', mean(sensitivities)*100, std(sensitivities)*100);
fprintf('  Mean specificity: %.2f%% (±%.2f%%)\n', mean(specificities)*100, std(specificities)*100);
```

## Visualization

### Network Visualization

```matlab
% Create graph
graph = GraphWU('B', connectivity);

% Get coordinates (assume parcellation coordinates available)
coordinates = readmatrix('parcellation_coordinates.txt');  % [x, y, z]

% Compute node sizes based on degree
degree = graph.measure('Degree').getValue();
node_sizes = degree / max(degree) * 50;  % Scale for visualization

% Plot network
figure('Position', [100, 100, 800, 600]);

% Plot edges
adjacency = graph.getAdjacency();
[i, j] = find(triu(adjacency, 1));

for e = 1:length(i)
    plot3([coordinates(i(e),1), coordinates(j(e),1)], ...
          [coordinates(i(e),2), coordinates(j(e),2)], ...
          [coordinates(i(e),3), coordinates(j(e),3)], ...
          'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
    hold on;
end

% Plot nodes
scatter3(coordinates(:,1), coordinates(:,2), coordinates(:,3), ...
         node_sizes, degree, 'filled');
colormap('jet');
colorbar;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Brain Network Visualization');
axis equal;
view(3);
grid on;
```

### Circular Network Plot

```matlab
% Create circular layout
n_nodes = size(connectivity, 1);
theta = linspace(0, 2*pi, n_nodes+1);
theta = theta(1:n_nodes);

x = cos(theta);
y = sin(theta);

% Plot connections
figure('Position', [100, 100, 800, 800]);
adjacency = connectivity > 0.3;  % Threshold

for i = 1:n_nodes
    for j = i+1:n_nodes
        if adjacency(i, j)
            plot([x(i), x(j)], [y(i), y(j)], 'Color', [0.7, 0.7, 0.7, 0.3], ...
                 'LineWidth', connectivity(i,j)*2);
            hold on;
        end
    end
end

% Plot nodes
scatter(x, y, 100, 'filled', 'MarkerFaceColor', 'b');

% Add labels
for i = 1:n_nodes
    text(x(i)*1.1, y(i)*1.1, sprintf('%d', i), 'HorizontalAlignment', 'center');
end

axis equal off;
title('Circular Network Visualization');
```

### Connectivity Matrix Heatmap

```matlab
% Plot connectivity matrix
figure;
imagesc(connectivity);
colormap('jet');
colorbar;
axis square;
xlabel('Node');
ylabel('Node');
title('Connectivity Matrix');

% Add community structure if available
modularity = graph.measure('Modularity');
communities = modularity.getCommunityStructure();

% Sort by community
[~, sort_idx] = sort(communities);
conn_sorted = connectivity(sort_idx, sort_idx);

figure;
imagesc(conn_sorted);
colormap('jet');
colorbar;
axis square;
xlabel('Node (sorted by community)');
ylabel('Node (sorted by community)');
title('Connectivity Matrix (Sorted by Community)');
```

## Batch Processing Pipeline

```matlab
% Batch processing for multiple subjects and groups
clear; clc;

% Configuration
data_dir = '/data/connectivity/';
output_dir = '/results/braph2/';
subjects_ctrl = dir(fullfile(data_dir, 'ctrl_*.txt'));
subjects_patient = dir(fullfile(data_dir, 'patient_*.txt'));

% Measures to compute
measures = {'Degree', 'Clustering', 'Betweenness', 'GlobalEfficiency', ...
            'LocalEfficiency', 'Modularity', 'PathLength'};

% Initialize results structure
results = struct();
results.controls = cell(length(subjects_ctrl), 1);
results.patients = cell(length(subjects_patient), 1);

% Process controls
fprintf('Processing controls...\n');
for s = 1:length(subjects_ctrl)
    fprintf('  Subject %d/%d\n', s, length(subjects_ctrl));

    % Load connectivity
    conn = readmatrix(fullfile(data_dir, subjects_ctrl(s).name));

    % Create graph
    graph = GraphWU('B', conn);

    % Compute all measures
    subject_results = struct();
    for m = 1:length(measures)
        measure_obj = graph.measure(measures{m});
        subject_results.(measures{m}) = measure_obj.getValue();
    end

    results.controls{s} = subject_results;
end

% Process patients
fprintf('Processing patients...\n');
for s = 1:length(subjects_patient)
    fprintf('  Subject %d/%d\n', s, length(subjects_patient));

    conn = readmatrix(fullfile(data_dir, subjects_patient(s).name));
    graph = GraphWU('B', conn);

    subject_results = struct();
    for m = 1:length(measures)
        measure_obj = graph.measure(measures{m});
        subject_results.(measures{m}) = measure_obj.getValue();
    end

    results.patients{s} = subject_results;
end

% Save results
save(fullfile(output_dir, 'braph2_batch_results.mat'), 'results');
fprintf('Batch processing complete!\n');
```

## Integration with Other Tools

### From CONN Toolbox

```matlab
% Load CONN output
load('CONN_x.mat');

% Extract connectivity matrices
n_subjects = length(CONN_x.subjects);
n_rois = size(CONN_x.Z, 1);

for s = 1:n_subjects
    % Get subject connectivity
    conn = CONN_x.Z(:,:,s);

    % Convert to BRAPH2 graph
    graph = GraphWU('B', conn);

    % Compute measures
    degree = graph.measure('Degree').getValue();

    % Store or analyze...
end
```

### Export to BrainNet Viewer

```matlab
% Prepare network for visualization
adjacency = graph.getAdjacency();

% Save as edge file
edge_matrix = adjacency;
save('network_edges.edge', 'edge_matrix', '-ascii');

% Save node file (coordinates + attributes)
nodes = [coordinates, degree];  % [x, y, z, degree]
save('network_nodes.node', 'nodes', '-ascii');

% Open in BrainNet Viewer
% BrainNet('BrainMesh_ICBM152.nv', 'network_nodes.node', 'network_edges.edge');
```

## Troubleshooting

**Problem:** Out of memory errors
**Solution:** Process subjects individually, use sparse matrices, reduce number of permutations

**Problem:** Graph measures return NaN or Inf
**Solution:** Check for disconnected components, normalize connectivity matrix, remove self-connections

**Problem:** GUI not launching
**Solution:** Verify MATLAB path, check MATLAB version compatibility, restart MATLAB

**Problem:** Permutation tests are slow
**Solution:** Use Parallel Computing Toolbox, reduce permutations for initial testing, use parfor loops

**Problem:** Community detection gives different results each time
**Solution:** Set random seed, use consensus clustering, run multiple times and average

## Best Practices

1. **Data Quality:**
   - Remove subjects with excessive motion
   - Verify connectivity matrix symmetry
   - Check for outliers and artifacts

2. **Thresholding:**
   - Test multiple thresholds (sensitivity analysis)
   - Report results across threshold range
   - Consider minimum spanning tree approach

3. **Statistics:**
   - Use permutation tests when possible
   - Correct for multiple comparisons (FDR, Bonferroni)
   - Report effect sizes, not just p-values

4. **Reproducibility:**
   - Set random seeds for randomization
   - Document all parameters
   - Save intermediate results
   - Use version control for scripts

## Resources

- **Official Website:** https://braph.org/
- **GitHub Repository:** https://github.com/braph-software/BRAPH-2
- **Documentation:** https://braph.org/documentation/
- **Tutorials:** https://braph.org/tutorials/
- **Forum:** https://braph.org/forum/
- **Publications:** https://braph.org/publications/

## Citation

```bibtex
@article{mijalkov2017braph,
  title={BRAPH: a graph theory software for the analysis of brain connectivity},
  author={Mijalkov, Mite and Kakaei, Ehsan and Pereira, Joana B and Westman, Eric and Volpe, Giovanni},
  journal={PLoS One},
  volume={12},
  number={8},
  pages={e0178798},
  year={2017},
  publisher={Public Library of Science}
}
```

## Related Tools

- **Brain Connectivity Toolbox (BCT):** MATLAB graph metrics
- **NetworkX:** Python graph analysis
- **GraphVar:** Comprehensive network analysis GUI
- **NBS:** Network-based statistics
- **BrainNet Viewer:** 3D network visualization
- **CONN Toolbox:** Functional connectivity analysis
