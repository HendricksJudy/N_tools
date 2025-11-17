# Brain Connectivity Toolbox (BCT)

## Overview

The Brain Connectivity Toolbox (BCT) is a comprehensive collection of MATLAB and Python functions for complex network analysis of brain connectivity data. It provides a wide range of graph theoretical measures for characterizing the topological properties of brain networks, including metrics for segregation, integration, centrality, modularity, and resilience.

**Website:** https://sites.google.com/site/bctnet/
**Platform:** MATLAB/Python (cross-platform)
**Language:** MATLAB (primary), Python (bctpy)
**License:** GNU GPL

## Key Features

- Comprehensive graph theory metrics (100+ functions)
- Weighted and binary network analysis
- Directed and undirected graphs
- Community detection algorithms
- Network randomization and null models
- Rich club analysis
- Network resilience and robustness
- Motif analysis
- Python implementation (bctpy) available
- Integration with neuroimaging tools

## Installation

### MATLAB Version

```matlab
% Download BCT from: https://sites.google.com/site/bctnet/
% Extract to desired location

% Add to MATLAB path
addpath('/path/to/BCT');
addpath('/path/to/BCT/2019_03_03_BCT');  % Version folder

% Verify installation
which clustering_coef_bu  % Test function availability
```

### Python Version (bctpy)

```bash
# Install via pip
pip install bctpy

# Or from GitHub
pip install git+https://github.com/aestrivex/bctpy.git
```

```python
# Verify installation
import bct
print(bct.__version__)
```

## Basic Concepts

### Network Representation

```matlab
% Connectivity matrix (adjacency matrix)
% N x N matrix where N = number of nodes (ROIs/parcels)
% A(i,j) = connection strength between nodes i and j

% Binary undirected network
A_binary = [
    0 1 1 0;
    1 0 1 1;
    1 1 0 1;
    0 1 1 0
];

% Weighted undirected network (e.g., correlation matrix)
A_weighted = [
    1.0  0.7  0.5  0.1;
    0.7  1.0  0.6  0.4;
    0.5  0.6  1.0  0.3;
    0.1  0.4  0.3  1.0
];

% Directed weighted network
A_directed = [
    0    0.5  0.3  0;
    0.4  0    0.6  0.2;
    0.2  0.3  0    0.5;
    0    0.1  0.4  0
];
```

### Thresholding

```matlab
% Threshold by connection strength
threshold = 0.3;
A_thresholded = A_weighted .* (A_weighted > threshold);

% Keep top N% of connections (proportional threshold)
density = 0.2;  % Keep 20% strongest connections
A_prop = threshold_proportional(A_weighted, density);

% Absolute thresholding (keeps specific number of connections)
n_connections = 50;
A_abs = threshold_absolute(A_weighted, n_connections);

% Ensure no self-connections
A_thresholded(1:length(A_thresholded)+1:end) = 0;
```

## Core Network Measures

### Segregation (Clustering)

```matlab
% Clustering coefficient (binary undirected)
C_binary = clustering_coef_bu(A_binary);

% Clustering coefficient (weighted undirected)
C_weighted = clustering_coef_wu(A_weighted);

% Average clustering coefficient
C_avg = mean(C_binary);

% Transitivity (alternative to clustering)
T = transitivity_bu(A_binary);
```

### Integration (Path Length)

```matlab
% Shortest path length (binary)
D = distance_bin(A_binary);  % Distance matrix
charpath = mean(D(~isinf(D) & D~=0));  % Characteristic path length

% Shortest path length (weighted)
% First convert weights to lengths (inverse)
L = weight_conversion(A_weighted, 'lengths');
D_weighted = distance_wei(L);

% Global efficiency
Eglob = efficiency_bin(A_binary);  % Binary
Eglob_weighted = efficiency_wei(A_weighted);  % Weighted
```

### Small-Worldness

```matlab
% Small-world coefficient
% Compute clustering and path length
C = mean(clustering_coef_bu(A_binary));
L = charpath(distance_bin(A_binary));

% Generate random networks for comparison
n_rand = 100;
C_rand = zeros(n_rand, 1);
L_rand = zeros(n_rand, 1);

for i = 1:n_rand
    A_rand = randmio_und(A_binary, 10);  % Randomize while preserving degree
    C_rand(i) = mean(clustering_coef_bu(A_rand));
    L_rand(i) = charpath(distance_bin(A_rand));
end

% Small-world index
gamma = C / mean(C_rand);  % Normalized clustering
lambda = L / mean(L_rand);  % Normalized path length
sigma = gamma / lambda;  % Small-world coefficient

% sigma > 1 indicates small-world properties
fprintf('Small-world coefficient: %.3f\n', sigma);
```

### Centrality Measures

```matlab
% Degree centrality
degree = degrees_und(A_binary);  % Binary undirected
degree_weighted = strengths_und(A_weighted);  % Weighted (strength)
degree_in_out = [degrees_dir(A_directed, 'in'), degrees_dir(A_directed, 'out')];

% Betweenness centrality
BC = betweenness_bin(A_binary);  % Binary
BC_weighted = betweenness_wei(L);  % Weighted

% Eigenvector centrality
EC = eigenvector_centrality_und(A_weighted);

% PageRank centrality
PR = pagerank_centrality(A_directed, 0.85);  % Damping factor = 0.85

% Closeness centrality
CC = mean(D, 2).^-1;  % Inverse of average distance
```

## Community Detection

### Modularity Analysis

```matlab
% Louvain algorithm (community detection)
[Ci, Q] = community_louvain(A_weighted);

% Ci = community assignment vector
% Q = modularity value

fprintf('Number of communities: %d\n', length(unique(Ci)));
fprintf('Modularity: %.3f\n', Q);

% Run multiple times (stochastic algorithm)
n_iterations = 100;
Q_all = zeros(n_iterations, 1);
Ci_all = zeros(length(A_weighted), n_iterations);

for i = 1:n_iterations
    [Ci_all(:,i), Q_all(i)] = community_louvain(A_weighted);
end

% Get consensus partition
Ci_consensus = consensus_und(Ci_all, 0.5, n_iterations);

% Visualize communities
figure;
imagesc(A_weighted(Ci_consensus, Ci_consensus));
colorbar;
title('Network organized by communities');
```

### Participation Coefficient

```matlab
% Within-module degree z-score
Z = module_degree_zscore(A_weighted, Ci);

% Participation coefficient
P = participation_coef(A_weighted, Ci);

% Classify nodes by role
% Connector hubs: high P, high Z
% Provincial hubs: low P, high Z
% Connectors: high P, low Z
% Peripheral: low P, low Z
```

## Rich Club Analysis

```matlab
% Rich club coefficient
[R, Nk, Ek] = rich_club_bu(A_binary);

% Normalized rich club (compare to random)
n_rand = 100;
R_rand = zeros(length(R), n_rand);

for i = 1:n_rand
    A_rand = randmio_und(A_binary, 10);
    R_rand(:,i) = rich_club_bu(A_rand);
end

% Normalized rich club coefficient
R_norm = R ./ mean(R_rand, 2);

% Plot
figure;
plot(R_norm);
xlabel('Degree');
ylabel('Normalized Rich Club Coefficient');
title('Rich Club Analysis');
```

## Network Randomization

### Null Models

```matlab
% Preserve degree distribution
A_rand = randmio_und(A_binary, 100);  % 100 rewiring iterations

% Maslov-Sneppen randomization
A_rand_ms = null_model_und_sign(A_binary);

% Weighted network randomization
A_rand_weighted = null_model_und_sign(A_weighted);

% Check degree preservation
degree_original = degrees_und(A_binary);
degree_random = degrees_und(A_rand);

% Should be identical
isequal(sort(degree_original), sort(degree_random))
```

## Motif Analysis

```matlab
% Count 3-node motifs
[I, Q, F] = motif3funct_bin(A_binary);
% I = intensity
% Q = coherence
% F = frequency

% Count 4-node motifs
[I4, Q4, F4] = motif4funct_bin(A_binary);

% Structural motifs
M = motif3struct_bin(A_binary);
```

## Network Resilience

### Attack Tolerance

```matlab
% Random attack
n_nodes = size(A_binary, 1);
removal_order = randperm(n_nodes);

% Targeted attack (remove high-degree nodes first)
degree = degrees_und(A_binary);
[~, removal_order] = sort(degree, 'descend');

% Simulate sequential node removal
efficiency = zeros(n_nodes, 1);
for i = 1:n_nodes
    nodes_remaining = setdiff(1:n_nodes, removal_order(1:i));
    A_remaining = A_binary(nodes_remaining, nodes_remaining);
    efficiency(i) = efficiency_bin(A_remaining);
end

% Plot vulnerability curve
figure;
plot((1:n_nodes)/n_nodes, efficiency);
xlabel('Proportion of nodes removed');
ylabel('Global efficiency');
title('Network Attack Tolerance');
```

## Python Usage (bctpy)

```python
import numpy as np
import bct

# Load connectivity matrix
A = np.loadtxt('connectivity_matrix.txt')

# Threshold
A_thresh = bct.threshold_proportional(A, 0.2)

# Clustering coefficient
C = bct.clustering_coef_wu(A_thresh)

# Path length
D = bct.distance_wei(bct.weight_conversion(A_thresh, 'lengths'))
charpath = np.mean(D[np.nonzero(D)])

# Community detection
ci, Q = bct.community_louvain(A_thresh)
print(f'Modularity: {Q:.3f}')
print(f'Communities: {len(np.unique(ci))}')

# Betweenness centrality
BC = bct.betweenness_wei(bct.weight_conversion(A_thresh, 'lengths'))

# Rich club
R, Nk, Ek = bct.rich_club_bu(A_thresh > 0)

# Network measures
degree = bct.degrees_und(A_thresh > 0)
strength = bct.strengths_und(A_thresh)
eigenvector_centrality = bct.eigenvector_centrality_und(A_thresh)
```

## Complete Analysis Example

```matlab
%% Network Analysis Pipeline

% Load connectivity matrix
load('connectivity_matrix.mat');  % Variable A

% Basic properties
n_nodes = size(A, 1);
fprintf('Network has %d nodes\n', n_nodes);

% Threshold
A = threshold_proportional(A, 0.2);  % Keep top 20%
A(1:n_nodes+1:end) = 0;  % Remove diagonal

% Basic metrics
C = mean(clustering_coef_wu(A));
L_inv = weight_conversion(A, 'lengths');
D = distance_wei(L_inv);
charpath = mean(D(D~=0 & ~isinf(D)));
Eglob = efficiency_wei(A);

% Degree and strength
degree = degrees_und(A > 0);
strength = strengths_und(A);

% Centrality
BC = betweenness_wei(L_inv);
EC = eigenvector_centrality_und(A);

% Community structure
[Ci, Q] = community_louvain(A);
Z = module_degree_zscore(A, Ci);
P = participation_coef(A, Ci);

% Report
fprintf('\n=== Network Metrics ===\n');
fprintf('Clustering: %.3f\n', C);
fprintf('Path length: %.3f\n', charpath);
fprintf('Global efficiency: %.3f\n', Eglob);
fprintf('Modularity: %.3f\n', Q);
fprintf('Number of communities: %d\n', length(unique(Ci)));

% Visualize
figure;
subplot(2,2,1);
histogram(degree);
title('Degree Distribution');

subplot(2,2,2);
imagesc(A(Ci, Ci));
colorbar;
title('Connectivity Matrix (by community)');

subplot(2,2,3);
scatter(Z, P);
xlabel('Within-module degree z-score');
ylabel('Participation coefficient');
title('Nodal Roles');

subplot(2,2,4);
bar([C, charpath, Eglob, Q]);
set(gca, 'XTickLabel', {'Clustering', 'Path Length', 'Efficiency', 'Modularity'});
title('Summary Metrics');
```

## Integration with Claude Code

When helping users with BCT:

1. **Check Installation:**
   ```matlab
   which clustering_coef_bu
   % or in Python
   import bct; print(bct.__version__)
   ```

2. **Common Issues:**
   - Diagonal elements not set to zero
   - Negative weights not handled properly
   - Wrong network type (binary vs weighted)
   - Not removing disconnected components

3. **Best Practices:**
   - Always zero the diagonal
   - Use appropriate function for network type
   - Run stochastic algorithms multiple times
   - Compare to null models
   - Visualize results
   - Document thresholding strategy

4. **Performance:**
   - Community detection can be slow for large networks
   - Use compiled MEX files if available
   - Consider subsampling for initial exploration

## Troubleshooting

**Problem:** Functions not found
**Solution:** Ensure BCT is in MATLAB path: `addpath('/path/to/BCT')`

**Problem:** Inf or NaN values
**Solution:** Check for disconnected nodes, remove diagonal, handle zero weights

**Problem:** Different results each run
**Solution:** Normal for stochastic algorithms (Louvain), run multiple times and use consensus

**Problem:** Negative weights
**Solution:** Use absolute values or functions designed for signed networks

## Resources

- Website: https://sites.google.com/site/bctnet/
- MATLAB Download: https://www.nitrc.org/projects/bct/
- Python (bctpy): https://github.com/aestrivex/bctpy
- Paper: Rubinov & Sporns (2010) NeuroImage
- Tutorial: https://sites.google.com/site/bctnet/measures

## Citation

```bibtex
@article{rubinov2010complex,
  title={Complex network measures of brain connectivity: uses and interpretations},
  author={Rubinov, Mikail and Sporns, Olaf},
  journal={Neuroimage},
  volume={52},
  number={3},
  pages={1059--1069},
  year={2010},
  publisher={Elsevier}
}
```

## Related Tools

- **CONN Toolbox:** fMRI connectivity analysis
- **NetworkX:** Python graph analysis (general)
- **GRETNA:** Graph-theoretic network analysis
- **BrainNet Viewer:** Network visualization
- **NBS:** Network-based statistics
