# Connectome Viewer - Brain Network Visualization Platform

## Overview

Connectome Viewer is a visualization and analysis platform for brain connectivity networks, developed as part of the Connectome Mapping Toolkit (CMTK). It provides interactive 3D visualization of structural and functional connectivity matrices alongside cortical surfaces, white matter tractography, and brain parcellations. Originally designed to support early Human Connectome Project (HCP) efforts and the Connectome File Format (CFF), Connectome Viewer pioneered many visualization approaches for whole-brain networks that are now standard in modern tools like Connectome Workbench and nilearn.

While Connectome Viewer is now considered legacy software (development slowed after 2014), it remains valuable for analyzing older connectome datasets, understanding the evolution of network neuroscience visualization, and educational purposes. The tool integrates FreeSurfer cortical surfaces, TrackVis/MRtrix tractography, and NetworkX graph analysis, offering a comprehensive platform for exploring multi-scale brain networks from voxels to parcellated regions. For modern connectome research, users are encouraged to migrate to Connectome Workbench (for HCP data) or Python-based tools (nilearn, BrainNetViewer).

**Official Repository:** https://github.com/LTS5/connectomeviewer (archived)
**Documentation:** http://www.cmtk.org/users/tutorials (legacy)
**CMTK Website:** http://www.cmtk.org

### Key Features

- **3D Network Visualization:** Interactive network graphs with anatomical node positioning
- **Surface Integration:** FreeSurfer cortical surfaces with parcellation overlay
- **Tractography Display:** White matter fiber tracts from TrackVis/MRtrix
- **Connectivity Matrices:** Heatmap and graph representations of structural/functional connectivity
- **Graph Theory Metrics:** Degree, clustering coefficient, shortest path, modularity
- **Multi-Scale Analysis:** Voxel, ROI, and atlas-based connectivity
- **CFF Format:** Connectome File Format for standardized data exchange
- **Python Scripting:** Programmatic access via Python API
- **NetworkX Integration:** Graph analysis using NetworkX library
- **Legacy Data Support:** Read older connectome datasets and formats

### Applications (Historical Context)

- Legacy connectome dataset visualization (pre-2015)
- Educational demonstrations of network neuroscience concepts
- Understanding connectome visualization history
- Transitioning older analysis pipelines to modern tools
- CMTK workflow visualization
- Connectome File Format (CFF) data exploration

### Modern Alternatives

- **Connectome Workbench:** HCP-standard visualization (CIFTI format)
- **nilearn:** Python-based connectivity visualization and analysis
- **BrainNet Viewer:** MATLAB-based network visualization
- **FSLeyes:** Modern NIfTI/GIFTI viewer with network support
- **Gephi:** General network visualization (non-brain-specific)

### Citation

```bibtex
@article{Gerhard2011ConnectomeViewer,
  title={The connectome viewer toolkit: an open source framework to manage, analyze, and visualize connectomes},
  author={Gerhard, Stephan and Daducci, Alessandro and Lemkaddem, Alia and
          Meuli, Reto and Thiran, Jean-Philippe and Hagmann, Patric},
  journal={Frontiers in Neuroinformatics},
  volume={5},
  pages={3},
  year={2011},
  publisher={Frontiers}
}

@article{Hagmann2008CMTK,
  title={Mapping the structural core of human cerebral cortex},
  author={Hagmann, Patric and Cammoun, Leila and Gigandet, Xavier and others},
  journal={PLoS Biology},
  volume={6},
  number={7},
  pages={e159},
  year={2008}
}
```

---

## Installation

**Important Note:** Connectome Viewer depends on Python 2.7 (deprecated since 2020). For modern systems, Docker containerization is strongly recommended.

### Docker Installation (Recommended)

```bash
# Create Dockerfile for Connectome Viewer
cat > Dockerfile <<'EOF'
FROM ubuntu:16.04

# Install Python 2.7 and dependencies
RUN apt-get update && apt-get install -y \
    python2.7 python-pip python-numpy python-scipy \
    python-matplotlib python-vtk6 python-networkx \
    python-traits python-traitsui mayavi2 git

# Install Connectome Viewer
RUN pip install cfflib connectomeviewer

# Set up display forwarding
ENV DISPLAY=:0

CMD ["/bin/bash"]
EOF

# Build Docker image
docker build -t connectome-viewer:legacy .

# Run with X11 forwarding (Linux/macOS)
xhost +local:docker
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/connectome_data:/data \
  connectome-viewer:legacy

# Inside container:
python -c "from cviewer.main import main; main()"
```

### Legacy Installation (Python 2.7 Systems)

```bash
# WARNING: Only for legacy systems with Python 2.7

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install python2.7 python-pip python-numpy \
  python-scipy python-matplotlib python-vtk6 \
  python-networkx python-traits python-traitsui \
  mayavi2

# Install Connectome Viewer
pip install cfflib
pip install connectomeviewer

# Test installation
python -c "from cviewer.main import main; main()"
```

### Dependencies

- **Python 2.7** (legacy, deprecated)
- **NumPy/SciPy:** Numerical computation
- **Matplotlib:** Plotting
- **VTK 6:** 3D visualization backend
- **Mayavi:** 3D scientific visualization
- **Traits/TraitsUI:** GUI framework
- **NetworkX:** Graph analysis
- **cfflib:** Connectome File Format library

### Testing Installation

```python
# Test Connectome Viewer import
from cviewer.main import main
from cfflib import load

print("Connectome Viewer successfully installed!")

# Launch GUI (if display available)
# main()
```

---

## Data Formats

### Connectome File Format (CFF)

```python
# CFF is HDF5-based container for connectome data

# CFF file structure:
connectome.cff
├── surfaces/         # FreeSurfer surfaces (GIFTI)
├── volumes/          # NIfTI volumes
├── tracks/           # Tractography (TrackVis .trk)
├── networks/         # Connectivity matrices (GraphML)
├── scripts/          # Analysis scripts
└── metadata.xml      # Dataset description

# Load CFF file
from cfflib import load
cfile = load('/path/to/connectome.cff')

# Access contents
print(cfile.get_connectome_network())  # Networks
print(cfile.get_connectome_surface())  # Surfaces
print(cfile.get_connectome_track())    # Tractography
```

### FreeSurfer Surface Formats

```python
# Connectome Viewer reads FreeSurfer surfaces

# Supported formats:
# - .pial (pial surface)
# - .white (white matter surface)
# - .inflated (inflated surface)
# - .sphere (spherical surface)
# - .annot (parcellation/annotation)

# Surface files location (FreeSurfer output):
# $SUBJECTS_DIR/sub-01/surf/
# ├── lh.pial
# ├── lh.white
# ├── lh.inflated
# ├── rh.pial
# └── ...

# Add surfaces to CFF
from cfflib import *
cfile = create()
cfile.add_connectome_surface(
    'lh.pial',
    fileformat='Gifti',
    name='Left Hemisphere Pial'
)
```

### TrackVis Tractography (.trk)

```python
# Load white matter fiber tracks

# TrackVis .trk file contains:
# - 3D fiber coordinates
# - Scalar properties (FA, color)
# - Header with dimensions and voxel size

# Generate .trk from MRtrix or TrackVis
# Example (MRtrix3):
# tckconvert tracks.tck tracks.trk

# Add to CFF
cfile.add_connectome_track(
    'tracks.trk',
    name='Whole Brain Tractography'
)
```

### Connectivity Matrices

```python
# NetworkX-compatible graph formats

# Supported formats:
# - GraphML (XML-based, recommended)
# - GML (graph modeling language)
# - NumPy .npy (adjacency matrix)
# - CSV (comma-separated connectivity matrix)

# Create connectivity matrix (NumPy)
import numpy as np
n_regions = 68  # Desikan-Killiany atlas
conn_matrix = np.random.rand(n_regions, n_regions)
conn_matrix = (conn_matrix + conn_matrix.T) / 2  # Symmetric

# Save as NumPy
np.save('connectivity.npy', conn_matrix)

# Convert to GraphML for Connectome Viewer
import networkx as nx
G = nx.from_numpy_array(conn_matrix)
nx.write_graphml(G, 'connectivity.graphml')
```

---

## Surface Visualization

### Loading FreeSurfer Surfaces

```python
# Visualize cortical surfaces in Connectome Viewer

# Step 1: Launch Connectome Viewer
from cviewer.main import main
main()

# Step 2: Load CFF file (GUI)
# File > Open CFF File
# Select connectome.cff

# Step 3: View surface
# In left panel, expand "Surfaces"
# Double-click surface to visualize

# Step 4: Adjust visualization
# - Color: Click color button to change
# - Opacity: Adjust transparency slider
# - Representation: Surface, Wireframe, or Points
```

### Cortical Parcellations

```python
# Display brain atlases on cortical surfaces

# Common atlases:
# - Desikan-Killiany (68 regions)
# - Destrieux (148 regions)
# - Custom parcellations (.annot files)

# Load parcellation
# In Connectome Viewer GUI:
# Surfaces > Right-click surface > Load Annotation
# Select .annot file from FreeSurfer

# Example: Desikan-Killiany
# File: $SUBJECTS_DIR/sub-01/label/lh.aparc.annot

# Programmatic loading:
from cfflib import *
cfile = load('connectome.cff')
surf = cfile.get_connectome_surface()[0]
surf.load_annotation('lh.aparc.annot')
```

### Overlay Activation Maps

```python
# Display functional activation or connectivity values on surface

# Step 1: Prepare overlay data
# - Scalar values per vertex (FreeSurfer .curv format)
# - Or per-ROI values

# Step 2: Load overlay in Connectome Viewer
# Surfaces > Surface > Load Scalar Data
# Select .curv or .mgh file

# Step 3: Adjust colormap
# - Colormap: Hot, Cool, Jet, etc.
# - Range: Min/Max values for color mapping
# - Threshold: Hide values below threshold

# Example overlay: Cortical thickness
# File: lh.thickness (FreeSurfer output)
```

### Surface Colormapping

```python
# Customize surface appearance

# Colormap options:
# - Gray: Anatomical surfaces
# - Spectral: Activation maps
# - RdBu: Positive/negative values
# - Hot: Increasing activation

# Apply colormap:
# Surfaces > Surface > Colormap > Select colormap

# Adjust color range:
# Surfaces > Surface > Color Range
# Min: 0, Max: 100 (example for percentage values)
```

---

## Tractography Visualization

### Loading TrackVis Files

```python
# Visualize white matter fiber tracts

# Step 1: Generate .trk file
# Using MRtrix3:
tckgen dwi.mif \
  -algorithm iFOD2 \
  -seed_image wm.nii.gz \
  -select 10000 \
  tracks.tck
tckconvert tracks.tck tracks.trk

# Step 2: Load in Connectome Viewer
# File > Open CFF (with tracks) or
# Tracks > Load Track File > Select tracks.trk

# Step 3: Visualize
# Tracks panel > Check "Visible" checkbox
# 3D view displays fiber tracts
```

### Fiber Coloring and Filtering

```python
# Customize tract appearance

# Color by direction (DTI convention):
# - Red: Left-Right (X)
# - Green: Anterior-Posterior (Y)
# - Blue: Inferior-Superior (Z)

# In Connectome Viewer:
# Tracks > Track > Color Mode > Direction

# Filter tracts by length:
# Tracks > Track > Length Filter
# Min Length: 20mm
# Max Length: 200mm
# Removes very short/long (likely noise) fibers

# Subsample for performance:
# Tracks > Track > Downsample
# Factor: 0.1 (show 10% of fibers)
```

### Tract Density Maps

```python
# Create maps of fiber density

# Tract density: Number of fibers passing through each voxel

# Generate density map (external tool, then visualize)
# Using MRtrix3:
tckmap tracks.trk \
  -template dwi.nii.gz \
  tract_density.nii.gz

# Load in Connectome Viewer as volume:
# Volumes > Load Volume > tract_density.nii.gz

# Overlay density on anatomical:
# Volumes > Overlay > Select density map
# Adjust opacity and colormap
```

### Integration with Surfaces

```python
# Display tracts alongside cortical surfaces

# Step 1: Load both surfaces and tracts in CFF

# Step 2: Enable both visualizations
# Surfaces > Surface > Visible: ON
# Tracks > Track > Visible: ON

# Step 3: Adjust transparency
# Surfaces > Opacity: 0.5 (semi-transparent)
# Tracks > Opacity: 1.0 (opaque)

# Use case: Show connections between cortical ROIs
# with underlying white matter pathways
```

---

## Network Visualization

### Connectivity Matrix as Network Graph

```python
# Visualize connectivity as 3D network graph

# Step 1: Prepare connectivity matrix
import numpy as np
import networkx as nx

# Load structural connectivity (e.g., from DSI Studio)
conn_matrix = np.load('connectivity.npy')  # Shape: (68, 68)

# Create NetworkX graph
G = nx.from_numpy_array(conn_matrix)

# Add node positions (ROI coordinates in MNI space)
roi_coords = np.loadtxt('roi_centers.txt')  # Shape: (68, 3)
for i, (x, y, z) in enumerate(roi_coords):
    G.nodes[i]['xyz'] = [x, y, z]

# Save as GraphML
nx.write_graphml(G, 'network.graphml')

# Step 2: Load in Connectome Viewer
# Networks > Load Network > network.graphml

# Step 3: Visualize
# Networks > Network > 3D Visualization
# Nodes positioned at anatomical coordinates
# Edges colored by connection strength
```

### Node Positioning (Anatomical Coordinates)

```python
# Position graph nodes at brain ROI locations

# Extract ROI centers from atlas
# Method 1: FreeSurfer label centers
mri_segstats --i aparc+aseg.nii.gz \
  --seg aparc+aseg.nii.gz \
  --sum stats.txt \
  --ctab $FREESURFER_HOME/FreeSurferColorLUT.txt

# Method 2: Python (nibabel)
import nibabel as nib
import numpy as np

atlas = nib.load('aparc+aseg.nii.gz').get_fdata()
roi_ids = np.unique(atlas)[1:]  # Exclude 0 (background)

roi_centers = []
for roi_id in roi_ids:
    coords = np.argwhere(atlas == roi_id)
    center = coords.mean(axis=0)
    roi_centers.append(center)

# Convert voxel to MNI coordinates (apply affine)
affine = nib.load('aparc+aseg.nii.gz').affine
roi_centers_mni = nib.affines.apply_affine(affine, roi_centers)

# Save for network visualization
np.savetxt('roi_centers_mni.txt', roi_centers_mni)
```

### Edge Thickness by Connection Strength

```python
# Visualize connection weights as edge thickness

# In NetworkX graph, edge weight determines thickness
G = nx.Graph()
G.add_edge(0, 1, weight=0.5)  # Weak connection
G.add_edge(2, 3, weight=0.9)  # Strong connection

# Connectome Viewer renders:
# - Thick edges: High weight
# - Thin edges: Low weight

# Threshold weak connections:
threshold = 0.3
G_thresh = nx.Graph()
for u, v, data in G.edges(data=True):
    if data['weight'] > threshold:
        G_thresh.add_edge(u, v, weight=data['weight'])

nx.write_graphml(G_thresh, 'network_thresholded.graphml')
```

### Interactive Graph Manipulation

```python
# Explore network interactively in 3D

# Connectome Viewer 3D controls:
# - Rotate: Click and drag
# - Zoom: Scroll wheel
# - Pan: Shift + drag
# - Select node: Click node (highlights connections)

# Node/edge properties:
# - Click node: Show degree, clustering coefficient
# - Click edge: Show connection weight

# Filter by degree:
# Networks > Network > Degree Filter
# Min Degree: 5 (hide low-degree nodes)
```

---

## Graph Theory Analysis

### Computing Network Metrics

```python
# Calculate graph theory measures using NetworkX

import networkx as nx
import numpy as np

# Load connectivity matrix
conn_matrix = np.load('connectivity.npy')
G = nx.from_numpy_array(conn_matrix)

# Degree (number of connections per node)
degree = dict(G.degree(weight='weight'))
print(f"Degree: {degree}")

# Clustering coefficient (local connectivity density)
clustering = nx.clustering(G, weight='weight')
print(f"Clustering: {clustering}")

# Shortest path length (network integration)
try:
    avg_path_length = nx.average_shortest_path_length(G, weight='weight')
    print(f"Average Path Length: {avg_path_length:.3f}")
except nx.NetworkXError:
    print("Graph is not connected")

# Modularity (community structure)
from networkx.algorithms import community
communities = community.greedy_modularity_communities(G)
modularity = community.modularity(G, communities)
print(f"Modularity: {modularity:.3f}")
```

### Hub Identification

```python
# Identify highly connected brain regions (hubs)

import numpy as np
import networkx as nx

G = nx.from_numpy_array(conn_matrix)

# Method 1: Degree centrality
degree_centrality = nx.degree_centrality(G)
top_hubs_degree = sorted(degree_centrality.items(),
                          key=lambda x: x[1], reverse=True)[:10]
print("Top 10 hubs by degree:")
for node, centrality in top_hubs_degree:
    print(f"  Node {node}: {centrality:.3f}")

# Method 2: Betweenness centrality (bridging nodes)
betweenness = nx.betweenness_centrality(G, weight='weight')
top_hubs_betweenness = sorted(betweenness.items(),
                               key=lambda x: x[1], reverse=True)[:10]

# Method 3: Eigenvector centrality (connected to other hubs)
eigenvector = nx.eigenvector_centrality(G, weight='weight')

# Visualize hubs in Connectome Viewer:
# Assign node sizes based on centrality
for node in G.nodes():
    G.nodes[node]['size'] = degree_centrality[node] * 100

nx.write_graphml(G, 'network_with_hubs.graphml')
```

### Community Detection

```python
# Detect modules/communities in brain networks

import networkx as nx
from networkx.algorithms import community

# Louvain method (modularity optimization)
communities = community.greedy_modularity_communities(G)

# Assign community labels to nodes
community_map = {}
for comm_id, comm in enumerate(communities):
    for node in comm:
        community_map[node] = comm_id

# Add to graph
for node, comm_id in community_map.items():
    G.nodes[node]['community'] = comm_id

# Save for visualization
nx.write_graphml(G, 'network_communities.graphml')

# In Connectome Viewer:
# Color nodes by community membership
# Networks > Network > Color by Attribute > community
```

### Exporting Metrics to CSV

```python
# Export network metrics for statistical analysis

import pandas as pd
import networkx as nx

# Compute multiple metrics
metrics = {
    'degree': dict(G.degree(weight='weight')),
    'clustering': nx.clustering(G, weight='weight'),
    'betweenness': nx.betweenness_centrality(G, weight='weight'),
    'eigenvector': nx.eigenvector_centrality(G, weight='weight')
}

# Create DataFrame
df = pd.DataFrame(metrics)
df.index.name = 'node_id'

# Add ROI labels (if available)
roi_labels = ['lh_bankssts', 'lh_caudalanteriorcingulate', ...]  # 68 labels
df['roi_label'] = roi_labels

# Export
df.to_csv('network_metrics.csv')
print(df.head())
```

---

## Multi-Modal Integration

### Combining Structural and Functional Connectivity

```python
# Compare structure-function correspondence

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load connectivity matrices
sc_matrix = np.load('structural_connectivity.npy')  # DTI tractography
fc_matrix = np.load('functional_connectivity.npy')  # Resting-state fMRI

# Flatten upper triangle (exclude diagonal)
idx = np.triu_indices_from(sc_matrix, k=1)
sc_flat = sc_matrix[idx]
fc_flat = fc_matrix[idx]

# Correlation between SC and FC
corr, pval = pearsonr(sc_flat, fc_flat)
print(f"SC-FC Correlation: r={corr:.3f}, p={pval:.3e}")

# Visualize
plt.figure(figsize=(6, 6))
plt.scatter(sc_flat, fc_flat, alpha=0.3, s=10)
plt.xlabel('Structural Connectivity (fiber count)')
plt.ylabel('Functional Connectivity (correlation)')
plt.title(f'Structure-Function Coupling (r={corr:.2f})')
plt.tight_layout()
plt.savefig('sc_fc_correlation.png', dpi=300)
```

### Overlay fMRI Activation on Networks

```python
# Display task activation on connectivity network

# Step 1: Extract activation values for ROIs
from nilearn import datasets, plotting
from nilearn.image import resample_to_img
import nibabel as nib

# Load activation map (SPM/FSL output)
activation = nib.load('task_activation_zmap.nii.gz')

# Load atlas
atlas = datasets.fetch_atlas_aal()
atlas_img = nib.load(atlas.maps)

# Extract mean activation per ROI
from nilearn.maskers import NiftiLabelsMasker
masker = NiftiLabelsMasker(atlas_img, standardize=False)
roi_activations = masker.fit_transform(activation).flatten()

# Step 2: Add to network graph
G = nx.read_graphml('network.graphml')
for i, activation_val in enumerate(roi_activations):
    G.nodes[i]['activation'] = float(activation_val)

nx.write_graphml(G, 'network_with_activation.graphml')

# Step 3: Visualize in Connectome Viewer
# Color nodes by activation strength
```

---

## Python Scripting

### Programmatic Data Loading

```python
# Automate Connectome Viewer workflows with Python

from cfflib import load, create
import os

# Load existing CFF file
cff_file = load('/path/to/connectome.cff')

# Access networks
networks = cff_file.get_connectome_network()
print(f"Found {len(networks)} networks")

# Access surfaces
surfaces = cff_file.get_connectome_surface()
print(f"Found {len(surfaces)} surfaces")

# Access tracks
tracks = cff_file.get_connectome_track()
print(f"Found {len(tracks)} tractography datasets")
```

### Automated Visualization

```python
# Generate visualizations programmatically

# Note: Full automation limited in legacy Connectome Viewer
# For batch processing, use modern tools (nilearn, Workbench)

# Example: Create screenshot of network (pseudocode)
from cviewer.main import ConnectomeViewerWindow

# Launch viewer
window = ConnectomeViewerWindow()

# Load data
window.load_cff('/path/to/connectome.cff')

# Render network
window.render_network(network_id=0)

# Save screenshot
window.save_screenshot('network_view.png')
```

---

## Migration to Modern Tools

### Transitioning to Connectome Workbench

```bash
# Convert CFF data to CIFTI format for Workbench

# Extract components from CFF:
# 1. Surfaces (GIFTI) - already Workbench-compatible
# 2. Connectivity matrices - convert to .pconn.nii (CIFTI)
# 3. Scalar overlays - convert to .pscalar.nii

# Example: Convert surface to Workbench format
# (GIFTI surfaces work directly in wb_view)
wb_view lh.pial.gii

# For connectivity, use CIFTI format:
# See Connectome Workbench documentation
```

### Exporting Data for Modern Viewers

```python
# Export CFF contents to standard formats

from cfflib import load
import nibabel as nib
import numpy as np

# Load CFF
cff = load('connectome.cff')

# Export connectivity matrix to CSV
networks = cff.get_connectome_network()
conn_matrix = networks[0].data
np.savetxt('connectivity.csv', conn_matrix, delimiter=',')

# Export surfaces (already GIFTI format)
surfaces = cff.get_connectome_surface()
# Surfaces can be loaded in FreeSurfer, Workbench, FSLeyes

# Export tractography to .trk (for TrackVis, MI-Brain)
tracks = cff.get_connectome_track()
# Already in .trk format, copy to output directory
```

### Alternative Tools

- **nilearn (Python):**
  ```python
  from nilearn import plotting
  plotting.plot_connectome(conn_matrix, roi_coords)
  ```

- **BrainNet Viewer (MATLAB):**
  ```matlab
  BrainNet_MapCfg('BrainMesh_ICBM152.nv', ...
                   'Node_AAL90.node', ...
                   'Edge_connectome.edge', ...
                   'Options.mat');
  ```

- **Gephi (General networks):**
  Import GraphML, visualize with force-directed layout

---

## Troubleshooting

### Python 2.7 Compatibility Issues

**Problem:** Python 2.7 no longer supported on modern systems

**Solution:** Use Docker containerization
```bash
# See Installation section for Docker approach
docker run -it connectome-viewer:legacy
```

### Mayavi Rendering Problems

**Problem:** Mayavi 3D visualization not working

**Solutions:**
```bash
# Enable offscreen rendering
export ETS_TOOLKIT=null

# Or use VTK backend
export QT_API=pyqt

# Check VTK installation
python -c "import vtk; print(vtk.VTK_VERSION)"
```

### File Format Conversion Errors

**Problem:** Cannot load .trk or surfaces

**Solutions:**
```bash
# Verify .trk file integrity (TrackVis)
track_info tracks.trk

# Regenerate .trk from .tck (MRtrix)
tckconvert -force tracks.tck tracks.trk

# Check surface format (FreeSurfer)
mris_info lh.pial
```

### Docker Display Issues

**Problem:** Cannot display GUI in Docker container

**Solutions:**
```bash
# Linux: Enable X11 forwarding
xhost +local:docker

# macOS: Install XQuartz
brew install --cask xquartz
# Start XQuartz, enable "Allow connections from network clients"
# export DISPLAY=host.docker.internal:0

# Windows: Use VcXsrv or Xming
# Set DISPLAY environment variable to host IP
```

---

## Best Practices

### When to Use Connectome Viewer vs. Alternatives

**Use Connectome Viewer for:**
- Legacy CFF datasets
- Historical/educational purposes
- Understanding network visualization evolution

**Use Modern Alternatives for:**
- New research projects → Connectome Workbench, nilearn
- HCP data → Connectome Workbench (CIFTI format)
- Publication-quality figures → nilearn, BrainNet Viewer
- Cross-platform compatibility → nilearn (Python)

### Legacy Data Handling

- **Preserve Original Data:** Keep CFF files for archival
- **Export Standard Formats:** Convert to GraphML, GIFTI, NIfTI
- **Document Processing:** Record conversion steps for reproducibility
- **Validate:** Check converted data against original

### Citation and Reproducibility

- **Cite Original Tool:** Gerhard et al. (2011) paper
- **Specify Version:** Connectome Viewer version used
- **Archive Code:** Save Python scripts for CFF processing
- **Share Data:** Use modern formats (BIDS, CIFTI) for sharing

---

## References

1. **Connectome Viewer:**
   - Gerhard et al. (2011). The connectome viewer toolkit. *Front Neuroinform*, 5:3.
   - CMTK website: http://www.cmtk.org

2. **Connectome Mapping:**
   - Hagmann et al. (2008). Mapping the structural core of human cerebral cortex. *PLoS Biol*, 6(7):e159.
   - Sporns et al. (2005). The human connectome: A structural description of the human brain. *PLoS Comp Biol*, 1(4):e42.

3. **Modern Alternatives:**
   - Marcus et al. (2011). Human Connectome Project informatics. *NeuroImage*, 80:195-204.
   - Abraham et al. (2014). Machine learning for neuroimaging with scikit-learn. *Front Neuroinform*, 8:14.

4. **Graph Theory:**
   - Rubinov & Sporns (2010). Complex network measures of brain connectivity. *NeuroImage*, 52:1059-1069.
   - Bullmore & Sporns (2009). Complex brain networks: graph theoretical analysis. *Nat Rev Neurosci*, 10:186-198.

**Resources:**
- CMTK: http://www.cmtk.org
- GitHub (archived): https://github.com/LTS5/connectomeviewer
- CFF Specification: http://www.connectomics.org/cff/
- NetworkX: https://networkx.org/
