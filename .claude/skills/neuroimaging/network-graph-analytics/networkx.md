# NetworkX - Graph Theory and Network Analysis

## Overview

**NetworkX** is the foundational Python library for graph theory and network analysis. While not neuroscience-specific, NetworkX provides comprehensive, efficient implementations of graph algorithms that underpin brain network analysis. Its extensive collection of algorithms for shortest paths, centrality measures, community detection, and network generation models makes it an essential tool for analyzing structural and functional brain connectivity. NetworkX's seamless integration with NumPy, SciPy, pandas, and matplotlib enables powerful neuroimaging workflows.

Many neuroscience-specific tools build upon NetworkX foundations, making it a critical skill for understanding and extending brain network analysis methods. NetworkX supports various graph types (directed, undirected, weighted, multi-graphs) and provides both classic graph algorithms and modern network science techniques.

**Key Features:**
- Comprehensive graph data structures (Graph, DiGraph, MultiGraph, MultiDiGraph)
- 500+ graph algorithms and network measures
- Shortest path algorithms (Dijkstra, Floyd-Warshall, A*, Bellman-Ford)
- Centrality measures (degree, betweenness, closeness, eigenvector, PageRank, Katz)
- Clustering coefficients and transitivity
- Community detection (modularity, Louvain, Girvan-Newman, label propagation)
- Small-world metrics (clustering, characteristic path length, sigma, omega)
- Network efficiency (global, local, nodal)
- Rich club coefficient
- Graph generation models (Erdős-Rényi, Watts-Strogatz, Barabási-Albert)
- Assortativity and mixing patterns
- Graph visualization with matplotlib
- Extensive I/O support (adjacency lists, edge lists, GraphML, GML, pickle)

**Primary Use Cases:**
- Brain connectivity network analysis
- Graph-theoretical characterization of brain organization
- Hub identification via centrality measures
- Community/module detection in brain networks
- Small-world property assessment
- Network topology comparisons
- Foundation for neuroscience-specific tools (BCT, NBS)

**Official Documentation:** https://networkx.org/

---

## Installation

### Install NetworkX

```bash
# Install via pip
pip install networkx

# Install with optional dependencies
pip install networkx[default]

# For visualization
pip install networkx matplotlib

# Verify installation
python -c "import networkx as nx; print(nx.__version__)"
```

### Install Additional Dependencies

```bash
# For neuroimaging integration
pip install numpy scipy pandas nibabel nilearn

# For advanced visualization
pip install seaborn plotly

# For community detection algorithms
pip install python-louvain networkx-metis
```

---

## Basic Graph Operations

### Create Graph from Connectivity Matrix

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Load connectivity matrix (e.g., from resting-state fMRI)
# Shape: (n_regions, n_regions)
n_regions = 100
connectivity_matrix = np.random.rand(n_regions, n_regions)

# Make symmetric (for undirected graph)
connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2

# Set diagonal to zero (no self-loops)
np.fill_diagonal(connectivity_matrix, 0)

# Create weighted undirected graph
G = nx.from_numpy_array(connectivity_matrix)

print(f"Graph info: {nx.info(G)}")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
```

### Thresholding Strategies

```python
# Absolute threshold
threshold_abs = 0.3

# Create binary graph above threshold
adjacency_binary = (connectivity_matrix > threshold_abs).astype(int)
np.fill_diagonal(adjacency_binary, 0)
G_threshold = nx.from_numpy_array(adjacency_binary)

print(f"Edges after absolute threshold: {G_threshold.number_of_edges()}")

# Proportional threshold (keep top X% of connections)
threshold_prop = 0.10  # Keep top 10%
n_keep = int(n_regions * (n_regions - 1) / 2 * threshold_prop)

# Get upper triangle values
triu_indices = np.triu_indices(n_regions, k=1)
edge_weights = connectivity_matrix[triu_indices]

# Find threshold for top X%
threshold_value = np.partition(edge_weights, -n_keep)[-n_keep]

adjacency_prop = (connectivity_matrix >= threshold_value).astype(int)
np.fill_diagonal(adjacency_prop, 0)
G_proportional = nx.from_numpy_array(adjacency_prop)

print(f"Edges after proportional threshold: {G_proportional.number_of_edges()}")
```

### Create Weighted Graph

```python
# Weighted graph preserving edge weights
G_weighted = nx.Graph()

# Add nodes with labels
node_labels = [f'Region_{i}' for i in range(n_regions)]
G_weighted.add_nodes_from(node_labels)

# Add weighted edges
for i in range(n_regions):
    for j in range(i+1, n_regions):
        if connectivity_matrix[i, j] > 0.3:  # Threshold
            G_weighted.add_edge(
                node_labels[i],
                node_labels[j],
                weight=connectivity_matrix[i, j]
            )

print(f"Weighted graph: {G_weighted.number_of_edges()} edges")
```

---

## Centrality Measures

### Degree Centrality (Hub Identification)

```python
import networkx as nx

# Degree centrality: fraction of nodes each node is connected to
degree_centrality = nx.degree_centrality(G_threshold)

# Convert to array for easier manipulation
degree_values = np.array([degree_centrality[i] for i in range(n_regions)])

# Identify hubs (top 10%)
hub_threshold = np.percentile(degree_values, 90)
hubs = np.where(degree_values > hub_threshold)[0]

print(f"Hub regions (top 10% degree): {hubs}")
print(f"Mean degree centrality: {degree_values.mean():.3f}")
print(f"Max degree centrality: {degree_values.max():.3f}")

# Visualize degree distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(degree_values, bins=30, edgecolor='black')
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')
plt.title('Degree Centrality Distribution')

plt.subplot(1, 2, 2)
plt.plot(sorted(degree_values, reverse=True), 'o-')
plt.xlabel('Node Rank')
plt.ylabel('Degree Centrality')
plt.title('Degree Centrality Rank')
plt.tight_layout()
plt.savefig('degree_centrality.png', dpi=300)
```

### Betweenness Centrality (Connector Hubs)

```python
# Betweenness centrality: fraction of shortest paths passing through each node
betweenness = nx.betweenness_centrality(G_threshold)
betweenness_values = np.array([betweenness[i] for i in range(n_regions)])

# Identify connector hubs
connector_threshold = np.percentile(betweenness_values, 90)
connectors = np.where(betweenness_values > connector_threshold)[0]

print(f"Connector hubs (top 10% betweenness): {connectors}")
print(f"Mean betweenness: {betweenness_values.mean():.3f}")

# For weighted graphs, use weight parameter
betweenness_weighted = nx.betweenness_centrality(G_weighted, weight='weight')
```

### Closeness Centrality

```python
# Closeness centrality: inverse of average distance to all other nodes
closeness = nx.closeness_centrality(G_threshold)
closeness_values = np.array([closeness[i] for i in range(n_regions)])

print(f"Mean closeness centrality: {closeness_values.mean():.3f}")
```

### Eigenvector Centrality

```python
# Eigenvector centrality: influence based on connections to influential nodes
try:
    eigenvector = nx.eigenvector_centrality(G_threshold, max_iter=1000)
    eigenvector_values = np.array([eigenvector[i] for i in range(n_regions)])

    print(f"Mean eigenvector centrality: {eigenvector_values.mean():.3f}")
except nx.PowerIterationFailedConvergence:
    print("Eigenvector centrality did not converge")

    # Use alternative: PageRank
    pagerank = nx.pagerank(G_threshold)
    pagerank_values = np.array([pagerank[i] for i in range(n_regions)])
    print(f"Mean PageRank: {pagerank_values.mean():.3f}")
```

### Compare Centrality Measures

```python
import pandas as pd
import seaborn as sns

# Create DataFrame with all centrality measures
centrality_df = pd.DataFrame({
    'degree': degree_values,
    'betweenness': betweenness_values,
    'closeness': closeness_values
})

# Correlation between centrality measures
centrality_corr = centrality_df.corr()

print("Centrality measure correlations:")
print(centrality_corr)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(centrality_corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1)
plt.title('Centrality Measure Correlations')
plt.tight_layout()
plt.savefig('centrality_correlations.png', dpi=300)
```

---

## Clustering and Small-World Properties

### Clustering Coefficient

```python
# Clustering coefficient: fraction of node's neighbors that are also connected
clustering = nx.clustering(G_threshold)
clustering_values = np.array([clustering[i] for i in range(n_regions)])

avg_clustering = nx.average_clustering(G_threshold)

print(f"Average clustering coefficient: {avg_clustering:.3f}")
print(f"Clustering std: {clustering_values.std():.3f}")

# For weighted networks
clustering_weighted = nx.clustering(G_weighted, weight='weight')
avg_clustering_weighted = sum(clustering_weighted.values()) / len(clustering_weighted)

print(f"Average weighted clustering: {avg_clustering_weighted:.3f}")
```

### Characteristic Path Length

```python
# Average shortest path length
# Only defined for connected graphs
if nx.is_connected(G_threshold):
    avg_path_length = nx.average_shortest_path_length(G_threshold)
    print(f"Average path length: {avg_path_length:.3f}")
else:
    # For disconnected graphs, use largest connected component
    largest_cc = max(nx.connected_components(G_threshold), key=len)
    G_connected = G_threshold.subgraph(largest_cc).copy()

    avg_path_length = nx.average_shortest_path_length(G_connected)
    print(f"Average path length (largest component): {avg_path_length:.3f}")
    print(f"Largest component size: {len(largest_cc)}/{n_regions}")
```

### Small-World Coefficient

```python
# Small-world properties: high clustering, short path length
# Compare to random graph with same degree distribution

# Generate random graph
degree_sequence = [G_threshold.degree(n) for n in G_threshold.nodes()]
G_random = nx.configuration_model(degree_sequence)
G_random = nx.Graph(G_random)  # Remove multi-edges and self-loops
G_random.remove_edges_from(nx.selfloop_edges(G_random))

# Compute metrics for random graph
C_random = nx.average_clustering(G_random)
if nx.is_connected(G_random):
    L_random = nx.average_shortest_path_length(G_random)
else:
    largest_cc = max(nx.connected_components(G_random), key=len)
    G_random_connected = G_random.subgraph(largest_cc).copy()
    L_random = nx.average_shortest_path_length(G_random_connected)

# Small-world metrics
C = avg_clustering
L = avg_path_length

# Normalized clustering
gamma = C / C_random

# Normalized path length
lambda_val = L / L_random

# Small-world coefficient (sigma)
sigma = gamma / lambda_val

print(f"Clustering (C): {C:.3f}, Random (C_rand): {C_random:.3f}, γ = {gamma:.3f}")
print(f"Path length (L): {L:.3f}, Random (L_rand): {L_random:.3f}, λ = {lambda_val:.3f}")
print(f"Small-world coefficient (σ): {sigma:.3f}")

# Interpretation: σ > 1 indicates small-world properties
if sigma > 1:
    print("Network exhibits small-world properties")
else:
    print("Network does not exhibit small-world properties")
```

---

## Community Detection

### Greedy Modularity Maximization

```python
# Greedy modularity optimization
from networkx.algorithms import community

communities_greedy = community.greedy_modularity_communities(G_threshold)

# Convert to node-community mapping
community_map = {}
for idx, comm in enumerate(communities_greedy):
    for node in comm:
        community_map[node] = idx

n_communities = len(communities_greedy)
print(f"Number of communities (greedy): {n_communities}")

# Community sizes
comm_sizes = [len(c) for c in communities_greedy]
print(f"Community sizes: {comm_sizes}")

# Modularity quality
modularity = community.modularity(G_threshold, communities_greedy)
print(f"Modularity (Q): {modularity:.3f}")
```

### Louvain Algorithm

```python
# Louvain algorithm (requires python-louvain package)
try:
    import community as community_louvain

    # Detect communities
    partition = community_louvain.best_partition(G_threshold)

    # Number of communities
    n_communities_louvain = len(set(partition.values()))
    print(f"Number of communities (Louvain): {n_communities_louvain}")

    # Modularity
    modularity_louvain = community_louvain.modularity(partition, G_threshold)
    print(f"Modularity (Louvain): {modularity_louvain:.3f}")

    # Community assignments
    communities_louvain = partition

except ImportError:
    print("Install python-louvain: pip install python-louvain")
    communities_louvain = community_map
```

### Visualize Communities

```python
import matplotlib.pyplot as plt

# Layout for visualization
pos = nx.spring_layout(G_threshold, seed=42)

# Color nodes by community
if 'partition' in locals():
    node_colors = [communities_louvain[node] for node in G_threshold.nodes()]
else:
    node_colors = [community_map.get(node, 0) for node in G_threshold.nodes()]

plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G_threshold, pos, node_color=node_colors,
                       cmap='tab20', node_size=100, alpha=0.8)
nx.draw_networkx_edges(G_threshold, pos, alpha=0.2, width=0.5)
plt.title(f'Network Communities (Q = {modularity:.3f})')
plt.axis('off')
plt.tight_layout()
plt.savefig('network_communities.png', dpi=300, bbox_inches='tight')
```

---

## Network Efficiency

### Global and Local Efficiency

```python
# Global efficiency: average inverse shortest path length
def global_efficiency(G):
    """Compute global efficiency"""
    n = len(G)
    if n < 2:
        return 0

    efficiency = 0
    for node in G:
        path_lengths = nx.single_source_shortest_path_length(G, node)
        efficiency += sum(1/length for target, length in path_lengths.items() if length > 0)

    return efficiency / (n * (n - 1))

# Local efficiency: efficiency of node's neighborhood
def local_efficiency(G):
    """Compute local efficiency for each node"""
    local_eff = {}
    for node in G:
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            local_eff[node] = 0
        else:
            subgraph = G.subgraph(neighbors)
            local_eff[node] = global_efficiency(subgraph)

    return local_eff

# Compute efficiencies
E_glob = global_efficiency(G_threshold)
E_local_dict = local_efficiency(G_threshold)
E_local = np.mean(list(E_local_dict.values()))

print(f"Global efficiency: {E_glob:.3f}")
print(f"Average local efficiency: {E_local:.3f}")

# Cost-efficiency
density = nx.density(G_threshold)
print(f"Network density (cost): {density:.3f}")
print(f"Efficiency per cost: {E_glob / density:.3f}")
```

---

## Rich Club Analysis

### Rich Club Coefficient

```python
# Rich club coefficient: tendency of high-degree nodes to connect to each other

def rich_club_coefficient(G, k):
    """
    Compute rich club coefficient for degree threshold k
    """
    # Nodes with degree > k
    rich_nodes = [n for n in G.nodes() if G.degree(n) > k]

    if len(rich_nodes) < 2:
        return 0

    # Edges among rich nodes
    subgraph = G.subgraph(rich_nodes)
    E_rich = subgraph.number_of_edges()

    # Possible edges among rich nodes
    N_rich = len(rich_nodes)
    E_max = N_rich * (N_rich - 1) / 2

    if E_max == 0:
        return 0

    # Rich club coefficient
    phi = E_rich / E_max

    return phi

# Compute for range of k values
degrees = [G_threshold.degree(n) for n in G_threshold.nodes()]
max_degree = max(degrees)

k_range = range(1, max_degree)
phi_values = [rich_club_coefficient(G_threshold, k) for k in k_range]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, phi_values, 'o-', linewidth=2, markersize=4)
plt.xlabel('Degree threshold (k)')
plt.ylabel('Rich club coefficient φ(k)')
plt.title('Rich Club Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rich_club.png', dpi=300)

# Identify rich club nodes (e.g., at k where phi peaks or plateaus)
k_threshold = 15  # Example threshold
rich_club_nodes = [n for n in G_threshold.nodes() if G_threshold.degree(n) > k_threshold]

print(f"Rich club nodes (degree > {k_threshold}): {len(rich_club_nodes)}")
```

### Rich Club Normalized to Random Network

```python
# Compare to random null model
n_rand = 100  # Number of random networks

phi_rand_mean = []

for k in k_range:
    phi_rand_k = []

    for _ in range(n_rand):
        # Generate random graph
        G_rand = nx.configuration_model(degree_sequence)
        G_rand = nx.Graph(G_rand)
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

        # Compute rich club for random graph
        phi_r = rich_club_coefficient(G_rand, k)
        phi_rand_k.append(phi_r)

    phi_rand_mean.append(np.mean(phi_rand_k))

phi_rand_mean = np.array(phi_rand_mean)

# Normalized rich club coefficient
phi_norm = np.array(phi_values) / (phi_rand_mean + 1e-10)

# Plot normalized
plt.figure(figsize=(10, 6))
plt.plot(k_range, phi_norm, 'o-', linewidth=2, markersize=4)
plt.axhline(y=1, color='r', linestyle='--', label='Random network')
plt.xlabel('Degree threshold (k)')
plt.ylabel('Normalized φ(k)')
plt.title('Normalized Rich Club Coefficient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rich_club_normalized.png', dpi=300)

# φ_norm > 1 indicates rich club organization
```

---

## Integration with Neuroimaging Data

### Load Connectivity from nilearn

```python
from nilearn import datasets, connectome
from nilearn.input_data import NiftiLabelsMasker

# Load atlas
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
atlas_img = atlas['maps']
labels = atlas['labels']

# Load fMRI data (from fMRIPrep)
# fmri_file = '/path/to/sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

# Extract timeseries (example with random data)
n_rois = 400
n_timepoints = 200
timeseries = np.random.randn(n_timepoints, n_rois)

# Compute connectivity
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]

# Create NetworkX graph
G_brain = nx.from_numpy_array(connectivity_matrix)

# Add region labels as node attributes
for i, label in enumerate(labels):
    G_brain.nodes[i]['label'] = label
    G_brain.nodes[i]['network'] = label.split('_')[1] if '_' in label else 'unknown'

print(f"Brain network: {G_brain.number_of_nodes()} regions")
```

### Network Analysis Pipeline

```python
# Complete analysis pipeline for brain connectivity

def analyze_brain_network(connectivity_matrix, threshold=0.3):
    """
    Comprehensive brain network analysis
    """
    # Threshold and create graph
    adj_matrix = (connectivity_matrix > threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    G = nx.from_numpy_array(adj_matrix)

    # Ensure connected
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Compute metrics
    metrics = {}

    # Degree
    degree_cent = nx.degree_centrality(G)
    metrics['mean_degree'] = np.mean(list(degree_cent.values()))

    # Clustering
    metrics['clustering'] = nx.average_clustering(G)

    # Path length
    metrics['path_length'] = nx.average_shortest_path_length(G)

    # Efficiency
    metrics['global_efficiency'] = global_efficiency(G)

    # Modularity
    communities = community.greedy_modularity_communities(G)
    metrics['modularity'] = community.modularity(G, communities)
    metrics['n_communities'] = len(communities)

    # Small-world
    degree_seq = [G.degree(n) for n in G.nodes()]
    G_rand = nx.configuration_model(degree_seq)
    G_rand = nx.Graph(G_rand)
    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

    C_rand = nx.average_clustering(G_rand)
    L_rand = nx.average_shortest_path_length(G_rand) if nx.is_connected(G_rand) else float('inf')

    gamma = metrics['clustering'] / C_rand if C_rand > 0 else 0
    lambda_val = metrics['path_length'] / L_rand if L_rand > 0 else 0
    metrics['sigma'] = gamma / lambda_val if lambda_val > 0 else 0

    return metrics, G

# Run analysis
metrics, G_analyzed = analyze_brain_network(connectivity_matrix, threshold=0.3)

print("Brain Network Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.3f}")
```

---

## Graph Visualization

### Basic Network Plot

```python
import matplotlib.pyplot as plt

# Create layout
pos = nx.spring_layout(G_threshold, k=0.5, iterations=50, seed=42)

# Node sizes by degree
degrees = dict(G_threshold.degree())
node_sizes = [300 * degrees[node] / max(degrees.values()) for node in G_threshold.nodes()]

# Node colors by betweenness
node_colors = [betweenness_values[node] for node in G_threshold.nodes()]

# Plot
plt.figure(figsize=(14, 12))
nx.draw_networkx_nodes(G_threshold, pos, node_size=node_sizes,
                       node_color=node_colors, cmap='YlOrRd',
                       alpha=0.8, vmin=0)
nx.draw_networkx_edges(G_threshold, pos, alpha=0.2, width=0.5)
plt.title('Brain Network (node size = degree, color = betweenness)')
plt.axis('off')
plt.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd'), label='Betweenness Centrality')
plt.tight_layout()
plt.savefig('brain_network.png', dpi=300, bbox_inches='tight')
```

### Circular Layout with Communities

```python
# Circular layout grouped by communities

# Sort nodes by community
if 'partition' in locals():
    node_community = [(node, communities_louvain[node]) for node in G_threshold.nodes()]
else:
    node_community = [(node, community_map[node]) for node in G_threshold.nodes()]

node_community_sorted = sorted(node_community, key=lambda x: x[1])
node_order = [n for n, c in node_community_sorted]

# Create circular layout
pos_circular = nx.circular_layout(G_threshold, scale=2)

# Reorder positions
pos_ordered = {node: pos_circular[i] for i, node in enumerate(node_order)}

# Plot
plt.figure(figsize=(14, 14))
nx.draw_networkx_nodes(G_threshold, pos_ordered, node_size=100,
                       node_color=node_colors, cmap='tab20', alpha=0.8)
nx.draw_networkx_edges(G_threshold, pos_ordered, alpha=0.1, width=0.3)
plt.title('Circular Network Layout (grouped by community)')
plt.axis('off')
plt.tight_layout()
plt.savefig('network_circular.png', dpi=300, bbox_inches='tight')
```

---

## Advanced Features

### Assortativity

```python
# Degree assortativity: correlation between degrees of connected nodes
assortativity = nx.degree_assortativity_coefficient(G_threshold)

print(f"Degree assortativity: {assortativity:.3f}")

# Positive: high-degree nodes connect to high-degree nodes (assortative)
# Negative: high-degree nodes connect to low-degree nodes (disassortative)
# Brain networks typically show disassortative mixing
```

### Motif Analysis

```python
# Count triangles (3-node motifs)
triangles = nx.triangles(G_threshold)
total_triangles = sum(triangles.values()) // 3

print(f"Total triangles: {total_triangles}")

# Transitivity (global clustering)
transitivity = nx.transitivity(G_threshold)
print(f"Transitivity: {transitivity:.3f}")
```

### k-Core Decomposition

```python
# k-core: maximal subgraph where all nodes have degree >= k

# Compute core numbers
core_numbers = nx.core_number(G_threshold)

# Maximum k-core
max_k = max(core_numbers.values())
k_core = nx.k_core(G_threshold, k=max_k)

print(f"Maximum k-core: k={max_k}, nodes={k_core.number_of_nodes()}")

# Visualize k-core
pos = nx.spring_layout(G_threshold, seed=42)

plt.figure(figsize=(12, 10))
# Draw full network faintly
nx.draw_networkx_edges(G_threshold, pos, alpha=0.1, width=0.3)
nx.draw_networkx_nodes(G_threshold, pos, node_size=50, node_color='lightgray', alpha=0.5)

# Highlight k-core
nx.draw_networkx_nodes(k_core, pos, node_size=200, node_color='red', alpha=0.8)
nx.draw_networkx_edges(k_core, pos, width=2, alpha=0.5, edge_color='red')

plt.title(f'k-Core Decomposition (k={max_k})')
plt.axis('off')
plt.tight_layout()
plt.savefig('k_core.png', dpi=300, bbox_inches='tight')
```

---

## Batch Processing

### Multi-Subject Network Analysis

```python
import os
from pathlib import Path
import pandas as pd

# Multiple subjects
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
connectivity_dir = Path('/data/connectivity_matrices')

results = []

for subject in subjects:
    print(f"Processing {subject}...")

    # Load connectivity matrix
    conn_file = connectivity_dir / f'{subject}_connectivity.npy'
    # conn_matrix = np.load(conn_file)

    # For example, use random
    conn_matrix = np.random.rand(100, 100)
    conn_matrix = (conn_matrix + conn_matrix.T) / 2
    np.fill_diagonal(conn_matrix, 0)

    # Analyze
    metrics, G = analyze_brain_network(conn_matrix, threshold=0.3)

    # Store results
    metrics['subject'] = subject
    results.append(metrics)

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\nGroup-level metrics:")
print(results_df.describe())

# Save
results_df.to_csv('network_metrics_all_subjects.csv', index=False)
```

---

## Troubleshooting

### Graph Not Connected

```python
# Check connectivity
if not nx.is_connected(G):
    print("Graph is not connected")

    # Number of connected components
    n_components = nx.number_connected_components(G)
    print(f"Number of components: {n_components}")

    # Work with largest component
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    G_connected = G.subgraph(largest_component).copy()

    print(f"Largest component: {len(largest_component)}/{len(G)} nodes")

    # Use G_connected for analyses requiring connectivity
```

### Memory Issues with Large Graphs

```python
# For very large graphs, use sparse matrices
from scipy import sparse

# Create sparse adjacency matrix
adj_sparse = sparse.csr_matrix(connectivity_matrix > 0.3)

# NetworkX can work with sparse matrices
G_sparse = nx.from_scipy_sparse_array(adj_sparse)

# Compute metrics efficiently
degree_dict = dict(G_sparse.degree())
```

---

## Best Practices

### Thresholding Strategy

1. **Absolute threshold:**
   - Simple, interpretable
   - But different subjects may have different edge counts

2. **Proportional threshold:**
   - Ensures same number of edges across subjects
   - Good for group comparisons

3. **Minimum Spanning Tree (MST) + threshold:**
   - Guarantees connectivity
   - Add strongest edges beyond MST

```python
# MST-based thresholding
from scipy.sparse.csgraph import minimum_spanning_tree

# Compute MST (on inverted weights for maximum spanning tree of correlation)
mst = minimum_spanning_tree(-connectivity_matrix).toarray()
mst = -mst  # Convert back

# Add additional strong edges
threshold = 0.5
additional_edges = (connectivity_matrix > threshold) & (mst == 0)

final_network = (mst != 0) | additional_edges
```

### Null Model Selection

- **Configuration model:** Preserves degree distribution
- **Lattice model:** For spatial networks
- **Erdős-Rényi:** Random graph with same density

---

## Resources and Further Reading

### Official Documentation

- **NetworkX Docs:** https://networkx.org/documentation/stable/
- **Tutorial:** https://networkx.org/documentation/stable/tutorial.html
- **Reference:** https://networkx.org/documentation/stable/reference/index.html

### Related Tools

- **Brain Connectivity Toolbox (BCT):** Brain-specific metrics
- **graph-tool:** High-performance graph library
- **igraph:** Alternative graph library
- **nilearn:** Neuroimaging connectivity analysis

### Key Publications

```
Hagberg, A., Swart, P., & Chult, D. S. (2008).
Exploring network structure, dynamics, and function using NetworkX.
Los Alamos National Lab, Report LA-UR-08-05495.
```

---

## Summary

**NetworkX** is the foundational tool for graph analysis in Python:

**Strengths:**
- Comprehensive graph algorithms
- Easy to use and well-documented
- Excellent integration with NumPy, pandas
- Flexible graph types (directed, weighted, multi-graphs)
- Active development and community

**Best For:**
- Brain network analysis
- Graph-theoretical characterization
- Custom algorithm development
- Integration with neuroimaging pipelines
- Foundation for specialized tools

**Typical Workflow:**
1. Load connectivity matrix from fMRI/DTI
2. Threshold and create NetworkX graph
3. Compute centrality, clustering, path length
4. Detect communities
5. Compare to null models
6. Visualize and interpret

NetworkX provides the essential foundation for brain network analysis, enabling rigorous graph-theoretical characterization of structural and functional connectivity.

## Citation

```bibtex
@inproceedings{hagberg2008exploring,
  title={Exploring network structure, dynamics, and function using NetworkX},
  author={Hagberg, Aric and Schult, Daniel and Swart, Pieter},
  booktitle={Proceedings of the 7th Python in Science Conference},
  pages={11--15},
  year={2008}
}
```
