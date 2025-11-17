# BrainNet Viewer - Brain Network Visualization

## Overview

BrainNet Viewer is a MATLAB-based toolbox for visualizing structural and functional connectivity networks on brain surfaces or volumes. Developed for publication-quality figures, it provides intuitive controls for displaying nodes (brain regions), edges (connections), and overlays on glass brains, cortical surfaces, or volumetric slices. It integrates seamlessly with connectivity analysis tools like GRETNA, BCT, and CONN.

**Website:** https://www.nitrc.org/projects/bnv/
**Platform:** MATLAB (Windows/macOS/Linux)
**Language:** MATLAB
**License:** GPL-2.0

## Key Features

- High-quality brain network visualization
- 3D surface rendering (glass brain, cortical surfaces)
- Node and edge customization (size, color, shape)
- Multiple view layouts (sagittal, coronal, axial, 3D)
- Support for weighted and directed networks
- Module/community visualization
- Overlay statistical maps and parcellations
- Export publication-ready images
- Batch processing for multiple networks
- Integration with connectivity analysis tools
- Customizable color schemes and transparency
- FreeSurfer and GIFTI surface support

## Installation

```matlab
% Download from: https://www.nitrc.org/projects/bnv/

% Extract to desired location
unzip BrainNetViewer_*.zip

% Add to MATLAB path
addpath(genpath('/path/to/BrainNetViewer'));

% Verify installation
which BrainNet

% Launch GUI
BrainNet

% Check for required files
% BrainMesh_*.nv - Brain surface meshes
% Node*.node - Example node files
% Edge*.edge - Example edge files
```

## File Formats

### Node File (.node)

```
# Format: X Y Z Color Size Label
# One row per node/ROI
# Coordinates in MNI space (mm)

0 -52 18 1 3 PCC
-45 -67 36 2 2.5 Angular_L
45 -67 36 2 2.5 Angular_R
-41 -21 54 3 2 Motor_L
41 -21 54 3 2 Motor_R

# Columns:
# 1-3: X, Y, Z coordinates (MNI)
# 4: Color (node value for colormap)
# 5: Size (sphere radius or proportional)
# 6: Label (optional, ROI name)
```

### Edge File (.edge)

```
# Format: Connectivity matrix (NxN)
# Symmetric for undirected networks
# Values represent connection strength

0    0.5  0.3  0.1  0
0.5  0    0.6  0.4  0.2
0.3  0.6  0    0.5  0.3
0.1  0.4  0.5  0    0.4
0    0.2  0.3  0.4  0

# For directed networks: A(i,j) = connection from i to j
# For binary: Use 1 (connected) or 0 (not connected)
# For weighted: Use correlation, coherence, etc.
```

### Surface File (.nv)

```
# Built-in surfaces:
BrainMesh_ICBM152.nv - MNI152 surface (default)
BrainMesh_Ch2.nv - Colin27 surface
BrainMesh_Ch2WithCerebellum.nv - With cerebellum

# Custom surfaces:
# FreeSurfer: lh.pial, rh.pial (need conversion)
# GIFTI: *.surf.gii
```

## Basic Usage

### Launch GUI

```matlab
% Start BrainNet Viewer
BrainNet

% GUI workflow:
% 1. File → Load File
% 2. Select Surface file (.nv)
% 3. Select Node file (.node) - optional
% 4. Select Edge file (.edge) - optional
% 5. Adjust visualization options
% 6. File → Save Image

% Or load from command line
BrainNet('BrainMesh_ICBM152.nv', 'MyNetwork.node', 'MyNetwork.edge')
```

### Simple Network Visualization

```matlab
%% Create simple network visualization

% Surface mesh
surf_file = 'BrainMesh_ICBM152.nv';

% Create node file
nodes = [
    0 -52 18 1 5 'PCC';
    -45 -67 36 1 4 'Angular_L';
    45 -67 36 1 4 'Angular_R';
    -46 40 8 2 4 'DLPFC_L';
    46 40 8 2 4 'DLPFC_R'
];

% Save node file
fid = fopen('dmn_nodes.node', 'w');
for i = 1:size(nodes, 1)
    fprintf(fid, '%g %g %g %g %g %s\n', nodes{i,:});
end
fclose(fid);

% Create edge file (connectivity matrix)
edges = [
    0   0.8 0.8 0.3 0.3;
    0.8 0   0.7 0.2 0.2;
    0.8 0.7 0   0.2 0.2;
    0.3 0.2 0.2 0   0.6;
    0.3 0.2 0.2 0.6 0
];

dlmwrite('dmn_edges.edge', edges, 'delimiter', '\t');

% Visualize
BrainNet(surf_file, 'dmn_nodes.node', 'dmn_edges.edge');
```

## Creating Files from Analysis

### From Correlation Matrix

```matlab
%% Convert GRETNA/BCT output to BrainNet format

% Load connectivity matrix and ROI coordinates
load('FC_matrix.mat', 'Z');  % Correlation matrix
load('AAL_coordinates.mat', 'coords', 'roi_names');  % MNI coordinates

n_rois = size(Z, 1);

% Threshold correlation matrix
threshold = 0.3;
edges = abs(Z) .* (abs(Z) > threshold);
edges(logical(eye(n_rois))) = 0;  % Remove diagonal

% Node properties based on degree
degree = sum(edges > 0, 2);
node_sizes = 1 + (degree / max(degree)) * 5;  % Scale 1-6
node_colors = degree;  % Color by degree

% Create node file
fid = fopen('network_nodes.node', 'w');
for i = 1:n_rois
    fprintf(fid, '%g %g %g %g %g %s\n', ...
        coords(i,1), coords(i,2), coords(i,3), ...
        node_colors(i), node_sizes(i), roi_names{i});
end
fclose(fid);

% Save edge file
dlmwrite('network_edges.edge', edges, 'delimiter', '\t');

% Visualize
BrainNet('BrainMesh_ICBM152.nv', 'network_nodes.node', 'network_edges.edge');
```

### From Module Detection

```matlab
%% Visualize network modules/communities

% Load module assignments
load('network_analysis.mat', 'modules', 'Z', 'coords');

% Color nodes by module
n_modules = max(modules);
node_colors = modules;  % Each module gets unique color

% Node sizes by within-module degree
module_degree = zeros(size(modules));
for m = 1:n_modules
    module_nodes = find(modules == m);
    subnetwork = Z(module_nodes, module_nodes);
    module_degree(module_nodes) = sum(abs(subnetwork) > 0.3, 2);
end

node_sizes = 1 + (module_degree / max(module_degree)) * 4;

% Create node file
fid = fopen('modules_nodes.node', 'w');
for i = 1:length(modules)
    fprintf(fid, '%g %g %g %g %g Module%d_ROI%d\n', ...
        coords(i,1), coords(i,2), coords(i,3), ...
        node_colors(i), node_sizes(i), modules(i), i);
end
fclose(fid);

% Edge file (show only strong connections)
edges = abs(Z) .* (abs(Z) > 0.5);
dlmwrite('modules_edges.edge', edges, 'delimiter', '\t');

% Visualize
BrainNet('BrainMesh_ICBM152.nv', 'modules_nodes.node', 'modules_edges.edge');
```

## Advanced Options

### Customizing Nodes

```matlab
%% Node display options (in GUI or .mat configuration)

% Node settings:
% - Size: 0-10 (sphere radius)
% - Color: Mapped to colormap
% - Shape: Sphere, cube, none
% - Transparency: 0 (transparent) to 1 (opaque)

% Example configuration
cfg.nodes.size = 5;  % Base size
cfg.nodes.colormap = 'jet';  % Color scheme
cfg.nodes.transparency = 0.8;
cfg.nodes.draw = 1;  % Show nodes
cfg.nodes.fontsize = 12;  % Label font size
cfg.nodes.showlabels = 0;  % Hide/show labels

% Apply via GUI: Options → Node
```

### Customizing Edges

```matlab
%% Edge display options

% Edge settings:
% - Width: Proportional to connection strength
% - Color: Single color or mapped to strength
% - Transparency
% - Threshold: Show only edges above value

% Configuration
cfg.edges.color = [0 0 1];  % Blue
cfg.edges.sizethr = [0.3 1.0];  % Show edges 0.3-1.0
cfg.edges.widththr = [0.5 3];  % Width range
cfg.edges.transparency = 0.7;
cfg.edges.draw = 1;

% Via GUI: Options → Edge
% - Size threshold: Filter weak connections
% - Width mapping: Scale by weight
% - Color: Single or gradient
```

### View Layouts

```matlab
%% Different visualization layouts

% In GUI: Layout → Select view

% Available views:
% - Full: 3D rotatable
% - Sagittal: Left/right side view
% - Axial: Top/bottom view
% - Coronal: Front/back view
% - Medium: 6-panel (3 slices + 3D)
% - Large: 8-panel comprehensive view

% Programmatically set layout
cfg.view.layout = 4;  % 1=Full, 2=Sag, 3=Axi, 4=Cor, 5=Med, 6=Large
cfg.view.angle = [0 90];  % Azimuth, elevation
```

### Surface Options

```matlab
%% Surface rendering options

% Surface settings:
cfg.surface.vertex = 0.5;  % Vertex color
cfg.surface.edge = 'none';  % Edge rendering
cfg.surface.lighting = 'gouraud';  % Lighting model
cfg.surface.alpha = 0.1;  % Transparency (0=transparent, 1=opaque)

% Via GUI: Options → Surface
% - Vertex: Surface color
% - Alpha: Transparency for see-through
% - Lighting: Shading quality
```

## Batch Processing

### Generate Multiple Views

```matlab
%% Create multiple views of same network

surf_file = 'BrainMesh_ICBM152.nv';
node_file = 'network_nodes.node';
edge_file = 'network_edges.edge';

% Define views
views = {'sagittal_left', 'sagittal_right', 'axial_top', 'coronal_front'};
angles = {[-90 0], [90 0], [0 90], [0 0]};

for v = 1:length(views)
    % Load network
    BrainNet(surf_file, node_file, edge_file);

    % Set view angle
    cfg.view.angle = angles{v};
    BrainNet_MapCfg(surf_file, node_file, edge_file, cfg);

    % Save
    print(sprintf('network_%s.png', views{v}), '-dpng', '-r300');
    close;
end
```

### Compare Multiple Networks

```matlab
%% Visualize networks from multiple subjects/conditions

subjects = {'sub-01', 'sub-02', 'sub-03'};
surf_file = 'BrainMesh_ICBM152.nv';

for s = 1:length(subjects)
    subj = subjects{s};

    % Load subject's network
    node_file = sprintf('%s_nodes.node', subj);
    edge_file = sprintf('%s_edges.edge', subj);

    % Visualize
    BrainNet_MapCfg(surf_file, node_file, edge_file);

    % Save
    saveas(gcf, sprintf('%s_network.png', subj));
    close;
end
```

## Configuration Files

### Save and Load Settings

```matlab
%% Save current visualization settings

% In GUI: File → Save Configuration
% Saves as .mat file with all settings

% Load configuration
load('myconfig.mat', 'cfg');

% Apply to new network
BrainNet_MapCfg('BrainMesh_ICBM152.nv', ...
                'new_nodes.node', ...
                'new_edges.edge', ...
                cfg, ...
                'output.png');
```

### Complete Configuration Example

```matlab
%% Full configuration structure

cfg = struct();

% Surface
cfg.surf.file = 'BrainMesh_ICBM152.nv';
cfg.surf.alpha = 0.1;
cfg.surf.lighting = 'gouraud';

% Nodes
cfg.node.draw = 1;
cfg.node.size = 5;
cfg.node.colormap = 'jet';
cfg.node.transparency = 0.9;
cfg.node.fontsize = 10;

% Edges
cfg.edge.draw = 1;
cfg.edge.sizethr = [0.3 1.0];
cfg.edge.widththr = [0.5 3];
cfg.edge.color = [0 0.5 1];
cfg.edge.transparency = 0.7;

% View
cfg.view.layout = 1;  % Full 3D
cfg.view.angle = [37.5 30];
cfg.view.axis = 'off';

% Apply
BrainNet_MapCfg(cfg.surf.file, node_file, edge_file, cfg, 'output.png');
```

## Publication-Quality Figures

### High-Resolution Export

```matlab
%% Export high-quality images

% Method 1: Via GUI
% File → Save Image
% Set resolution (300-600 DPI recommended)

% Method 2: Programmatic export
BrainNet_MapCfg(surf_file, node_file, edge_file, cfg);

% Save as high-res PNG
print('figure1.png', '-dpng', '-r600');

% Save as vector (for editing)
print('figure1.svg', '-dsvg');
print('figure1.eps', '-depsc');

% Save as PDF
print('figure1.pdf', '-dpdf');
```

### Multi-Panel Figures

```matlab
%% Create multi-panel figure

figure('Position', [100 100 1200 800]);

% Panel 1: Lateral view
subplot(2,2,1);
BrainNet_MapCfg(surf_file, node_file, edge_file, cfg);
title('Lateral View');

% Panel 2: Medial view
subplot(2,2,2);
cfg.view.angle = [90 0];
BrainNet_MapCfg(surf_file, node_file, edge_file, cfg);
title('Medial View');

% Panel 3: Dorsal view
subplot(2,2,3);
cfg.view.angle = [0 90];
BrainNet_MapCfg(surf_file, node_file, edge_file, cfg);
title('Dorsal View');

% Panel 4: Connectivity matrix
subplot(2,2,4);
edges = load('network_edges.edge');
imagesc(edges);
colorbar;
title('Connectivity Matrix');

% Save
print('multipanel_figure.png', '-dpng', '-r300');
```

## Integration with Other Tools

### From CONN Toolbox

```matlab
%% Visualize CONN results

% Extract CONN connectivity matrix and coordinates
% (After running CONN analysis)

% Get ROI coordinates from CONN
ROIs = conn_module('get', 'rois');
coords = zeros(length(ROIs), 3);
for i = 1:length(ROIs)
    coords(i,:) = ROIs(i).coordinates;
end

% Get connectivity matrix
Z = conn_module('results');

% Create BrainNet files
% ... (use code from previous examples)
```

### From FreeSurfer

```matlab
%% Use FreeSurfer surfaces

% Convert FreeSurfer surface to BrainNet format
% (Requires FreeSurfer in path)

% Load FreeSurfer surface
[vertices, faces] = freesurfer_read_surf('lh.pial');

% Create .nv file (BrainNet format)
% Note: May need custom conversion script
% Or use built-in MNI surfaces
```

## Tips and Tricks

### Color Schemes

```matlab
% Useful colormaps for networks:
% - 'jet': Rainbow (avoid for accessibility)
% - 'hot': Red-yellow
% - 'cool': Blue-purple
% - 'spring': Magenta-yellow
% - 'parula': MATLAB default (perceptually uniform)

% For modules: Use discrete colors
module_colors = [
    1 0 0;    % Red
    0 1 0;    % Green
    0 0 1;    % Blue
    1 1 0;    % Yellow
    1 0 1;    % Magenta
    0 1 1     % Cyan
];

% Map modules to colors
node_rgb = module_colors(modules, :);
```

### Performance Optimization

```matlab
% For large networks (>100 nodes):
% 1. Increase edge threshold (show fewer connections)
% 2. Reduce edge transparency
% 3. Use simpler surface mesh
% 4. Disable node labels
% 5. Export to file rather than displaying

% Batch mode (no display)
BrainNet_MapCfg(surf, nodes, edges, cfg, 'output.png');
close all;  % Close figure immediately
```

## Integration with Claude Code

When helping users with BrainNet Viewer:

1. **Check Installation:**
   ```matlab
   which BrainNet
   BrainNet  % Should open GUI
   ```

2. **Common Issues:**
   - Node coordinates outside brain (check MNI vs. Talairach)
   - Missing surface files
   - Edge matrix wrong size (must match nodes)
   - Labels not showing (check fontsize, overlap)
   - Rendering slow (too many edges)

3. **Best Practices:**
   - Use MNI coordinates for nodes
   - Threshold edges to show meaningful connections
   - Size nodes by importance (degree, betweenness)
   - Color by module or metric
   - Export high resolution (300+ DPI)
   - Save configuration for reproducibility
   - Use transparency for overlapping structures
   - Check view angle for optimal presentation

4. **File Preparation:**
   - Verify node coordinates are in MNI space
   - Ensure edge matrix is symmetric (undirected)
   - Use consistent formatting (tab-delimited)
   - Include meaningful labels
   - Scale node sizes appropriately

## Troubleshooting

**Problem:** Nodes not visible
**Solution:** Check coordinates in MNI space (-90 to 90), adjust node size and transparency

**Problem:** Edges overwhelming display
**Solution:** Increase threshold, reduce transparency, show only top N% connections

**Problem:** Surface too opaque
**Solution:** Reduce surface alpha (0.05-0.2), adjust lighting

**Problem:** Labels overlapping
**Solution:** Reduce number of labels, increase font size, adjust node positions slightly

**Problem:** Export image cut off
**Solution:** Adjust figure size before export, use larger margins

## Resources

- Website: https://www.nitrc.org/projects/bnv/
- Manual: Included in download (BrainNetViewer_Manual.pdf)
- Forum: https://www.nitrc.org/forum/?group_id=51
- Example data: Included with installation
- YouTube: BrainNet Viewer Tutorials

## Citation

```bibtex
@article{xia2013brainnet,
  title={BrainNet Viewer: a network visualization tool for human brain connectomics},
  author={Xia, Mingrui and Wang, Jinhui and He, Yong},
  journal={PloS one},
  volume={8},
  number={7},
  pages={e68910},
  year={2013}
}
```

## Related Tools

- **GRETNA:** Network analysis (creates BrainNet input)
- **Brain Connectivity Toolbox:** Graph metrics
- **CONN:** Functional connectivity analysis
- **Connectome Workbench:** HCP visualization
- **CircularGraph:** Alternative network visualization
- **Gephi:** General network visualization
