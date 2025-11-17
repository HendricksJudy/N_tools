# Brain Connectivity Toolbox (BCT) - Brain Network Analysis

## Overview

**Brain Connectivity Toolbox (BCT)** is the gold-standard toolbox for graph-theoretical analysis of brain networks, providing optimized implementations specifically designed for neuroimaging data. Originally developed in MATLAB by Olaf Sporns, Mikail Rubinov, and colleagues, BCT includes specialized algorithms for weighted and directed brain networks, handling the unique characteristics of functional and structural connectivity. The Python port (bctpy) brings BCT functionality to Python workflows while maintaining compatibility with the MATLAB version.

BCT stands out for its brain-specific optimizations, comprehensive documentation, extensive validation, and widespread adoption in the connectomics community. It provides not just basic graph metrics, but sophisticated measures for modular organization, hub classification, resilience analysis, and null model generation tailored to brain network properties.

**Key Features:**
- Comprehensive graph metrics optimized for brain connectivity
- Degree, strength, and diversity measures for weighted networks
- Clustering coefficients for weighted and directed networks
- Path length measures (binary, weighted, directed)
- Centrality measures (betweenness, closeness, eigenvector, PageRank, subgraph)
- Modularity and community detection (Louvain, spectral, hierarchical)
- Participation coefficient and within-module degree z-score
- Rich club coefficient with statistical testing
- Assortativity and core-periphery structure
- Motif analysis (3-node and 4-node motifs)
- Network resilience and robustness testing
- Null model generation (degree-preserving, weight-preserving)
- Both MATLAB and Python (bctpy) implementations
- Extensive validation and literature support

**Primary Use Cases:**
- Comprehensive characterization of brain network topology
- Module/community detection in functional connectivity
- Hub identification and classification
- Rich club analysis of structural connectivity
- Group comparisons of network properties
- Developmental and disease-related network changes
- Method validation and cross-tool verification

**Official Documentation:** https://sites.google.com/site/bctnet/

---

## Installation

### MATLAB Installation

```matlab
% Download BCT from: https://sites.google.com/site/bctnet/Home/functions

% Add to MATLAB path
addpath('/path/to/BCT');
addpath('/path/to/BCT/'); % Ensure all subdirectories included

% Verify installation
help clustering_coef_bu

% Test with example
n = 100;
A = rand(n, n) > 0.9;  % Random binary network
A = A | A';  % Make symmetric
A(1:n+1:end) = 0;  % Remove diagonal

C = clustering_coef_bu(A);
fprintf('Average clustering: %.3f\n', mean(C));
```

### Python Installation (bctpy)

```bash
# Install bctpy
pip install bctpy

# Or from GitHub for latest version
pip install git+https://github.com/aestrivex/bctpy.git

# Install dependencies
pip install numpy scipy networkx

# Verify installation
python -c "import bct; print(bct.__version__)"
```

### Verify Python Installation

```python
import bct
import numpy as np

# Test with simple network
n = 100
A = (np.random.rand(n, n) > 0.9).astype(float)
A = np.triu(A, 1) + np.triu(A, 1).T  # Symmetric
np.fill_diagonal(A, 0)

# Compute clustering
C = bct.clustering_coef_bu(A)

print(f"Average clustering: {C.mean():.3f}")
print("BCT installation successful")
```

---

## Basic Network Metrics

### Degree and Strength

```python
import bct
import numpy as np

# Load connectivity matrix (functional connectivity from fMRI)
# Shape: (n_regions, n_regions)
n_regions = 100
W = np.random.rand(n_regions, n_regions)
W = (W + W.T) / 2  # Make symmetric
np.fill_diagonal(W, 0)

# Binary threshold
threshold = 0.3
A = (W > threshold).astype(float)

# Degree (binary network)
degree = bct.degrees_und(A)

print(f"Degree statistics:")
print(f"  Mean: {degree.mean():.2f}")
print(f"  Std: {degree.std():.2f}")
print(f"  Max: {degree.max()}")

# Strength (weighted network)
strength = bct.strengths_und(W)

print(f"\nStrength statistics:")
print(f"  Mean: {strength.mean():.3f}")
print(f"  Std: {strength.std():.3f}")
print(f"  Max: {strength.max():.3f}")
```

### Clustering Coefficient

```python
# Binary clustering coefficient
C_binary = bct.clustering_coef_bu(A)

print(f"Binary clustering coefficient: {C_binary.mean():.3f}")

# Weighted clustering coefficient (Onnela et al. 2005)
C_weighted = bct.clustering_coef_wu(W)

print(f"Weighted clustering coefficient: {C_weighted.mean():.3f}")

# Transitivity (global clustering)
T = bct.transitivity_bu(A)

print(f"Transitivity: {T:.3f}")
```

### Path Length and Efficiency

```python
# Binary path length
# Requires connected network
D = bct.distance_bin(A)  # Distance matrix
lambda_bin, _, _, _, _ = bct.charpath(D)

print(f"Characteristic path length (binary): {lambda_bin:.3f}")

# Weighted path length
# Use inverse of weights as distances (higher correlation = shorter distance)
W_dist = bct.weight_conversion(W, 'lengths')  # Convert to distance
D_wei = bct.distance_wei(W_dist)
lambda_wei, _, _, _, _ = bct.charpath(D_wei)

print(f"Characteristic path length (weighted): {lambda_wei:.3f}")

# Global efficiency
E_glob = bct.efficiency_bin(A, local=False)

print(f"Global efficiency: {E_glob:.3f}")

# Local efficiency
E_loc = bct.efficiency_bin(A, local=True)

print(f"Average local efficiency: {E_loc.mean():.3f}")
```

---

## Centrality and Hub Detection

### Betweenness Centrality

```python
# Node betweenness centrality (binary)
BC = bct.betweenness_bin(A)

print(f"Betweenness centrality:")
print(f"  Mean: {BC.mean():.3f}")
print(f"  Max: {BC.max():.3f}")

# Identify top hubs (e.g., top 10%)
hub_threshold = np.percentile(BC, 90)
hubs_BC = np.where(BC > hub_threshold)[0]

print(f"Hub regions (betweenness): {hubs_BC}")

# Edge betweenness
EBC = bct.edge_betweenness_bin(A)

print(f"Max edge betweenness: {EBC.max():.3f}")
```

### Eigenvector Centrality

```python
# Eigenvector centrality
EC = bct.eigenvector_centrality_und(A)

print(f"Eigenvector centrality mean: {EC.mean():.3f}")

# Hubs by eigenvector centrality
hubs_EC = np.where(EC > np.percentile(EC, 90))[0]

print(f"Hub regions (eigenvector): {hubs_EC}")
```

### Subgraph Centrality

```python
# Subgraph centrality (weighted)
CS = bct.subgraph_centrality(A)

print(f"Subgraph centrality mean: {CS.mean():.2e}")
```

### PageRank Centrality

```python
# PageRank (like Google's algorithm)
PR = bct.pagerank_centrality(A, d=0.85)  # d = damping factor

print(f"PageRank mean: {PR.mean():.3f}")

# Compare centrality measures
import pandas as pd

centrality_df = pd.DataFrame({
    'degree': degree,
    'betweenness': BC,
    'eigenvector': EC,
    'pagerank': PR
})

# Correlation between centrality measures
print("\nCentrality correlations:")
print(centrality_df.corr())
```

---

## Modular Organization

### Community Detection (Louvain Algorithm)

```python
# Community detection using Louvain algorithm
# Returns community assignments and modularity quality (Q)

# For weighted undirected networks
ci, Q = bct.community_louvain(W, gamma=1.0, seed=0)

print(f"Number of communities: {len(np.unique(ci))}")
print(f"Modularity (Q): {Q:.3f}")

# Community sizes
community_sizes = np.bincount(ci)
print(f"Community sizes: {community_sizes}")

# Consensus clustering (run multiple times for stability)
n_iter = 100
ci_all = np.zeros((n_iter, n_regions))

for i in range(n_iter):
    ci_temp, _ = bct.community_louvain(W, seed=i)
    ci_all[i, :] = ci_temp

# Agreement matrix
agreement = bct.agreement(ci_all) / n_iter

# Consensus partition
ci_consensus, Q_consensus = bct.community_louvain(agreement)

print(f"Consensus communities: {len(np.unique(ci_consensus))}")
print(f"Consensus modularity: {Q_consensus:.3f}")
```

### Within-Module Degree and Participation Coefficient

```python
# Within-module degree z-score
# Measures how well-connected a node is to other nodes in its module

Z = bct.module_degree_zscore(W, ci, flag=0)  # flag=0 for undirected

print(f"Within-module z-score range: {Z.min():.2f} to {Z.max():.2f}")

# Participation coefficient
# Measures diversity of intermodular connections

P = bct.participation_coef(W, ci, degree='undirected')

print(f"Participation coefficient mean: {P.mean():.3f}")

# Hub classification (Guimerà & Amaral 2005)
# Provincial hubs: high Z, low P (within-module connectors)
# Connector hubs: high Z, high P (between-module connectors)

provincial_hubs = np.where((Z > 2.5) & (P < 0.3))[0]
connector_hubs = np.where((Z > 2.5) & (P > 0.3))[0]

print(f"\nProvincial hubs: {len(provincial_hubs)}")
print(f"Connector hubs: {len(connector_hubs)}")
```

### Visualize Modular Structure

```python
import matplotlib.pyplot as plt

# Sort nodes by module
node_order = np.argsort(ci)

# Reorder connectivity matrix
W_ordered = W[node_order, :][:, node_order]

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(W_ordered, cmap='hot', interpolation='nearest')
plt.colorbar(label='Connection Strength')
plt.title(f'Modular Structure (Q = {Q:.3f})')
plt.xlabel('Region (sorted by module)')
plt.ylabel('Region (sorted by module)')

# Add module boundaries
module_boundaries = np.where(np.diff(np.sort(ci)))[0] + 0.5
for boundary in module_boundaries:
    plt.axhline(boundary, color='cyan', linewidth=2)
    plt.axvline(boundary, color='cyan', linewidth=2)

plt.tight_layout()
plt.savefig('modular_structure.png', dpi=300)
```

---

## Small-World Analysis

### Small-World Propensity

```python
# Compute small-world properties relative to random and lattice networks

# Random null model (preserving degree sequence)
B, R = bct.null_model_und_sign(A, bin_swaps=5, wei_freq=0.1)

# Clustering and path length for observed network
C_obs = bct.clustering_coef_bu(A).mean()
D_obs = bct.distance_bin(A)
L_obs, _, _, _, _ = bct.charpath(D_obs)

# For random network
C_rand = bct.clustering_coef_bu(B).mean()
D_rand = bct.distance_bin(B)
L_rand, _, _, _, _ = bct.charpath(D_rand)

# Normalized metrics
gamma = C_obs / C_rand if C_rand > 0 else 0
lambda_val = L_obs / L_rand if L_rand > 0 else 0

# Small-world coefficient
sigma = gamma / lambda_val if lambda_val > 0 else 0

print(f"Clustering: C = {C_obs:.3f}, C_rand = {C_rand:.3f}, γ = {gamma:.3f}")
print(f"Path length: L = {L_obs:.3f}, L_rand = {L_rand:.3f}, λ = {lambda_val:.3f}")
print(f"Small-world coefficient σ = {sigma:.3f}")

if sigma > 1:
    print("Network exhibits small-world properties")
```

### Small-Worldness (Humphries & Gurney 2008)

```python
# Alternative small-world metric

# Generate multiple random networks for robust estimate
n_rand = 100
C_rand_all = []
L_rand_all = []

for _ in range(n_rand):
    B_temp, _ = bct.null_model_und_sign(A, bin_swaps=5)
    C_rand_all.append(bct.clustering_coef_bu(B_temp).mean())

    D_temp = bct.distance_bin(B_temp)
    L_temp, _, _, _, _ = bct.charpath(D_temp)
    L_rand_all.append(L_temp)

C_rand_mean = np.mean(C_rand_all)
L_rand_mean = np.mean(L_rand_all)

# Small-worldness
SW = (C_obs / C_rand_mean) / (L_obs / L_rand_mean)

print(f"Small-worldness: {SW:.3f}")
print(f"(SW > 1 indicates small-world)")
```

---

## Rich Club Analysis

### Rich Club Coefficient

```python
# Rich club coefficient: tendency of high-degree nodes to connect

# Compute for range of degree thresholds
degrees = bct.degrees_und(A).astype(int)
k_max = int(degrees.max())

k_range = np.arange(1, k_max)
phi = np.zeros(len(k_range))

for idx, k in enumerate(k_range):
    phi[idx] = bct.rich_club_bu(A, k)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, phi, 'o-', linewidth=2, markersize=4)
plt.xlabel('Degree threshold k')
plt.ylabel('Rich club coefficient φ(k)')
plt.title('Rich Club Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rich_club_bct.png', dpi=300)
```

### Normalized Rich Club Coefficient

```python
# Compare to random networks

n_rand = 100
phi_rand = np.zeros((n_rand, len(k_range)))

for i in range(n_rand):
    # Generate random network preserving degree
    B_rand, _ = bct.null_model_und_sign(A, bin_swaps=10)

    for idx, k in enumerate(k_range):
        phi_rand[i, idx] = bct.rich_club_bu(B_rand, k)

# Mean and std of random
phi_rand_mean = phi_rand.mean(axis=0)
phi_rand_std = phi_rand.std(axis=0)

# Normalized rich club
phi_norm = phi / (phi_rand_mean + 1e-10)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, phi_norm, 'o-', linewidth=2, markersize=4, label='Observed')
plt.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Random')
plt.fill_between(k_range,
                 (phi_rand_mean - phi_rand_std) / (phi_rand_mean + 1e-10),
                 (phi_rand_mean + phi_rand_std) / (phi_rand_mean + 1e-10),
                 alpha=0.3, color='red')
plt.xlabel('Degree threshold k')
plt.ylabel('Normalized φ(k)')
plt.title('Normalized Rich Club Coefficient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rich_club_normalized.png', dpi=300)

# Statistical significance
# φ_norm > 1 + 2*std indicates significant rich club
phi_significant = phi_norm > 1 + 2
k_significant = k_range[phi_significant]

if len(k_significant) > 0:
    print(f"Significant rich club at k = {k_significant}")
```

---

## Advanced Network Metrics

### Assortativity

```python
# Degree assortativity coefficient
r = bct.assortativity_bin(A, flag=0)  # flag=0 for undirected

print(f"Degree assortativity: {r:.3f}")

# Positive r: assortative (hubs connect to hubs)
# Negative r: disassortative (hubs connect to low-degree nodes)
# Brain networks typically show disassortative mixing
```

### Core-Periphery Structure

```python
# k-core decomposition
kcore = bct.kcore_bu(A, k=None)  # k=None finds highest k-core

print(f"k-core assignments: {np.unique(kcore)}")

# Nodes in highest k-core
max_k = kcore.max()
core_nodes = np.where(kcore == max_k)[0]

print(f"Core nodes (k={max_k}): {len(core_nodes)}")
```

### Motif Analysis

```python
# 3-node and 4-node motif frequencies

# 3-node motifs (13 possible motifs in directed graphs, fewer in undirected)
# For undirected: only 2 unique 3-node motifs (triangle, open triplet)

# Intensity and coherence of motifs
I, Q, F = bct.motif3funct_bin(A)

print(f"3-node motif intensity: {I.mean():.3f}")
print(f"3-node motif coherence: {Q.mean():.3f}")

# 4-node motifs
# More complex, 199 unique motifs in directed graphs

# For undirected networks, use motif counting
# (Full motif analysis computationally intensive for large networks)
```

---

## Null Models and Statistical Testing

### Degree-Preserving Randomization

```python
# Generate null model preserving degree distribution

# Binary network
B_null, eff = bct.null_model_und_sign(A, bin_swaps=10, wei_freq=0.1)

print(f"Null model generated (efficiency: {eff})")

# Check degree preservation
degree_original = bct.degrees_und(A)
degree_null = bct.degrees_und(B_null)

print(f"Degree correlation: {np.corrcoef(degree_original, degree_null)[0, 1]:.3f}")
```

### Weight-Preserving Randomization

```python
# For weighted networks, preserve weight and degree

W_null = bct.null_model_und_sign(W, bin_swaps=10, wei_freq=0.5)[0]

# Check weight preservation
strength_original = bct.strengths_und(W)
strength_null = bct.strengths_und(W_null)

print(f"Strength correlation: {np.corrcoef(strength_original, strength_null)[0, 1]:.3f}")
```

### Permutation Testing

```python
# Compare network metrics between groups with permutation test

def permutation_test_networks(group1_matrices, group2_matrices, metric_func, n_perm=1000):
    """
    Permutation test for network metrics
    """
    # Compute observed difference
    metrics_g1 = [metric_func(m) for m in group1_matrices]
    metrics_g2 = [metric_func(m) for m in group2_matrices]

    obs_diff = np.mean(metrics_g1) - np.mean(metrics_g2)

    # Permutation distribution
    all_matrices = group1_matrices + group2_matrices
    n1 = len(group1_matrices)
    n_total = len(all_matrices)

    perm_diffs = []

    for _ in range(n_perm):
        # Shuffle group assignments
        perm_idx = np.random.permutation(n_total)
        perm_g1 = [all_matrices[i] for i in perm_idx[:n1]]
        perm_g2 = [all_matrices[i] for i in perm_idx[n1:]]

        # Compute metric
        perm_metrics_g1 = [metric_func(m) for m in perm_g1]
        perm_metrics_g2 = [metric_func(m) for m in perm_g2]

        perm_diff = np.mean(perm_metrics_g1) - np.mean(perm_metrics_g2)
        perm_diffs.append(perm_diff)

    # P-value
    perm_diffs = np.array(perm_diffs)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    return obs_diff, p_value, perm_diffs

# Example: compare clustering between groups
def clustering_metric(conn_matrix):
    A_temp = (conn_matrix > 0.3).astype(float)
    return bct.clustering_coef_bu(A_temp).mean()

# Simulate two groups
group1 = [np.random.rand(50, 50) for _ in range(10)]
group2 = [np.random.rand(50, 50) for _ in range(10)]

obs_diff, p_val, null_dist = permutation_test_networks(group1, group2, clustering_metric, n_perm=100)

print(f"Observed difference: {obs_diff:.3f}")
print(f"P-value: {p_val:.3f}")
```

---

## Integration with Neuroimaging Data

### From fMRI Connectivity to BCT Analysis

```python
import numpy as np
from nilearn import datasets, connectome

# Load atlas
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)

# Load connectivity matrix (from CONN, nilearn, or computed directly)
# For example, simulate
n_rois = 200
timeseries = np.random.randn(250, n_rois)  # 250 TRs

# Compute functional connectivity
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
conn_matrix = correlation_measure.fit_transform([timeseries])[0]

# Threshold
threshold = 0.3
A = (conn_matrix > threshold).astype(float)

# Comprehensive BCT analysis
results = {}

# Basic metrics
results['degree'] = bct.degrees_und(A).mean()
results['clustering'] = bct.clustering_coef_bu(A).mean()

D = bct.distance_bin(A)
results['path_length'], _, _, _, _ = bct.charpath(D)

# Efficiency
results['efficiency'] = bct.efficiency_bin(A, local=False)

# Modularity
ci, Q = bct.community_louvain(conn_matrix)
results['modularity'] = Q
results['n_modules'] = len(np.unique(ci))

# Centrality
BC = bct.betweenness_bin(A)
results['betweenness'] = BC.mean()

print("Network Analysis Results:")
for key, value in results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")
```

### Multi-Subject Analysis Pipeline

```python
import pandas as pd
from pathlib import Path

def analyze_subject_network(conn_matrix, threshold=0.3):
    """Complete BCT analysis for single subject"""

    # Threshold
    A = (conn_matrix > threshold).astype(float)

    # Metrics
    metrics = {}

    # Degree
    metrics['mean_degree'] = bct.degrees_und(A).mean()

    # Clustering
    metrics['clustering'] = bct.clustering_coef_bu(A).mean()

    # Path length
    D = bct.distance_bin(A)
    metrics['path_length'], _, _, _, _ = bct.charpath(D)

    # Efficiency
    metrics['global_efficiency'] = bct.efficiency_bin(A, local=False)
    metrics['local_efficiency'] = bct.efficiency_bin(A, local=True).mean()

    # Modularity
    _, Q = bct.community_louvain(conn_matrix)
    metrics['modularity'] = Q

    # Centralization
    BC = bct.betweenness_bin(A)
    metrics['betweenness_max'] = BC.max()

    # Assortativity
    metrics['assortativity'] = bct.assortativity_bin(A, flag=0)

    return metrics

# Process multiple subjects
subjects = [f'sub-{i:02d}' for i in range(1, 21)]
all_results = []

for subject in subjects:
    print(f"Processing {subject}...")

    # Load connectivity (simulated here)
    conn_matrix = np.random.rand(200, 200)
    conn_matrix = (conn_matrix + conn_matrix.T) / 2

    # Analyze
    metrics = analyze_subject_network(conn_matrix, threshold=0.3)
    metrics['subject'] = subject

    all_results.append(metrics)

# Create DataFrame
results_df = pd.DataFrame(all_results)

print("\nGroup statistics:")
print(results_df.describe())

# Save
results_df.to_csv('bct_network_metrics.csv', index=False)
```

---

## MATLAB Implementation

### Basic BCT Analysis in MATLAB

```matlab
% Load connectivity matrix
load('connectivity_matrix.mat');  % Variable 'W'

% Threshold
threshold = 0.3;
A = double(W > threshold);
A(1:size(A,1)+1:end) = 0;  % Remove diagonal

% Degree
deg = degrees_und(A);
fprintf('Mean degree: %.2f\n', mean(deg));

% Clustering
C = clustering_coef_bu(A);
fprintf('Mean clustering: %.3f\n', mean(C));

% Path length
D = distance_bin(A);
[lambda, efficiency, ecc, radius, diameter] = charpath(D);
fprintf('Characteristic path length: %.3f\n', lambda);

% Community detection
[Ci, Q] = community_louvain(W);
fprintf('Modularity Q: %.3f\n', Q);
fprintf('Number of modules: %d\n', length(unique(Ci)));

% Participation coefficient
P = participation_coef(W, Ci);
fprintf('Mean participation: %.3f\n', mean(P));

% Within-module degree
Z = module_degree_zscore(W, Ci);
fprintf('Z-score range: %.2f to %.2f\n', min(Z), max(Z));
```

---

## Troubleshooting

### bctpy Installation Issues

```bash
# If bctpy install fails
pip install --upgrade pip setuptools wheel
pip install bctpy --no-cache-dir

# Check dependencies
pip install numpy scipy networkx

# Verify
python -c "import bct; print('BCT imported successfully')"
```

### Function Not Found

```python
# Some functions may have different names in bctpy vs MATLAB BCT

# MATLAB: clustering_coef_bu
# Python: bct.clustering_coef_bu

# Check available functions
import bct
print(dir(bct))

# Or check documentation
help(bct.clustering_coef_bu)
```

### Numerical Issues

```python
# For very sparse or dense networks, some metrics may fail

# Check connectivity
n_edges = np.sum(A) / 2
density = n_edges / (n_regions * (n_regions - 1) / 2)

print(f"Network density: {density:.3f}")

# If too sparse, consider lower threshold
# If disconnected, work with largest component
```

---

## Best Practices

### Threshold Selection

1. **Absolute threshold:** Simple but variable edge count
2. **Proportional threshold:** Fixed edge count across subjects
3. **Significance-based:** Only significant connections (requires statistical testing)

```python
# Proportional thresholding
def proportional_threshold(conn_matrix, proportion=0.10):
    """Keep top proportion of connections"""

    n = conn_matrix.shape[0]

    # Get upper triangle (undirected)
    triu_idx = np.triu_indices(n, k=1)
    edge_weights = conn_matrix[triu_idx]

    # Find threshold for top X%
    n_keep = int(len(edge_weights) * proportion)
    threshold_val = np.partition(edge_weights, -n_keep)[-n_keep]

    # Threshold
    A = (conn_matrix >= threshold_val).astype(float)
    np.fill_diagonal(A, 0)

    return A

A_prop = proportional_threshold(conn_matrix, proportion=0.10)
print(f"Edges kept: {np.sum(A_prop) / 2}")
```

### Reporting Network Metrics

Always report:
1. Threshold used (value and method)
2. Resulting network density
3. Whether network is connected
4. Null model used for comparisons
5. Number of permutations for statistical tests

---

## Resources and Further Reading

### Official Documentation

- **BCT Website:** https://sites.google.com/site/bctnet/
- **MATLAB Functions:** https://sites.google.com/site/bctnet/Home/functions
- **bctpy GitHub:** https://github.com/aestrivex/bctpy
- **bctpy Docs:** https://bctpy.readthedocs.io/

### Key Publications

```
Rubinov, M., & Sporns, O. (2010).
Complex network measures of brain connectivity: Uses and interpretations.
NeuroImage, 52(3), 1059-1069.
```

```
Sporns, O., & Betzel, R. F. (2016).
Modular brain networks.
Annual Review of Psychology, 67, 613-640.
```

### Related Tools

- **NetworkX:** General graph theory
- **BRAPH2:** Comprehensive GUI-based analysis
- **NBS:** Network-based statistics
- **nilearn:** Connectivity computation

---

## Summary

**Brain Connectivity Toolbox (BCT)** is the gold standard for brain network analysis:

**Strengths:**
- Brain-optimized algorithms
- Comprehensive validation
- Weighted and directed networks
- Extensive literature support
- Both MATLAB and Python versions
- Wide adoption in connectomics

**Best For:**
- Rigorous brain network characterization
- Modular organization analysis
- Hub identification and classification
- Group comparisons
- Method validation
- Cross-platform analysis (MATLAB/Python)

**Typical Workflow:**
1. Load connectivity matrix from neuroimaging
2. Apply threshold (absolute, proportional, or significance)
3. Compute comprehensive metrics (degree, clustering, modularity, etc.)
4. Compare to null models
5. Statistical testing for group differences
6. Interpret and visualize results

BCT remains the essential tool for graph-theoretical analysis of brain connectivity, providing validated, optimized implementations for the most important network measures in neuroscience.
