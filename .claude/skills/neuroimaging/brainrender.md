# Brainrender

## Overview

**Brainrender** is a modern Python package for creating high-quality 3D renderings of brain anatomy, regions, and data. Developed as part of the BrainGlobe initiative, Brainrender provides a programmatic interface for generating publication-quality figures and animations using anatomical atlases across multiple species. Built on VTK and vedo for rendering, it offers powerful visualization capabilities while maintaining an intuitive Python API.

Brainrender excels at atlas-based visualization, allowing researchers to display specific brain regions, overlay connectivity data, visualize cell distributions, and create custom scenes programmatically. The tool supports multiple atlases (Allen Mouse, Human, Zebrafish, and more), making it valuable for systems neuroscience, comparative neuroanatomy, and multi-species circuit studies.

**Key Use Cases:**
- Atlas-based brain region visualization
- Connectivity and circuit mapping
- Cell distribution and morphology rendering
- Cross-species comparative neuroanatomy
- Publication figure creation
- Animated brain visualizations
- Injection site and projection mapping
- Custom mesh and data overlay

**Official Website:** https://brainglobe.info/brainrender.html
**Documentation:** https://docs.brainglobe.info/brainrender/
**Source Code:** https://github.com/brainglobe/brainrender

---

## Key Features

- **Multi-Atlas Support:** Allen Brain Atlas (mouse, human), WHS rat, zebrafish, and more
- **Region Visualization:** Display specific anatomical structures from atlases
- **Connectivity Rendering:** Visualize connections between brain regions
- **Cell Distribution:** Render cell locations and densities
- **Neuron Morphology:** Display reconstructed neuron tracings
- **Custom Meshes:** Import and render custom 3D objects
- **Programmatic Control:** Full Python API for reproducible visualizations
- **Publication Quality:** High-resolution rendering for papers
- **Video Export:** Create animations and rotating views
- **Jupyter Integration:** Interactive rendering in notebooks
- **Flexible Styling:** Custom colors, transparency, and materials
- **Camera Control:** Precise viewpoint positioning
- **Multi-Panel Figures:** Composite visualizations
- **Open Source:** MIT licensed, community-driven development
- **Active Development:** Regular updates and new features

---

## Installation

### Using Pip (Recommended)

```bash
# Install brainrender
pip install brainrender

# Install optional dependencies
pip install jupyter vedo[all]

# Verify installation
python -c "import brainrender; print(brainrender.__version__)"
```

### Using Conda

```bash
# Create conda environment
conda create -n brainrender python=3.9
conda activate brainrender

# Install from conda-forge
conda install -c conda-forge brainrender

# Or install with pip in conda environment
pip install brainrender
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/brainglobe/brainrender.git
cd brainrender

# Install in development mode
pip install -e .[dev]

# Run tests
pytest
```

### Atlas Download

```bash
# Atlases are downloaded automatically on first use
# Or manually download specific atlas

from bg_atlasapi import list_atlases, show_atlases

# List available atlases
available = list_atlases()
print(available)

# Download specific atlas
from bg_atlasapi.bg_atlas import BG Atlas
atlas = BGAtlas("allen_mouse_25um")
```

---

## Basic Usage

### Create Simple Scene

```python
from brainrender import Scene

# Create scene with default atlas (Allen Mouse 25um)
scene = Scene()

# Add whole brain mesh
scene.add_brain_region("root", alpha=0.3)

# Show scene
scene.render()
```

### Specify Atlas

```python
from brainrender import Scene

# Use specific atlas
scene = Scene(atlas_name="allen_mouse_25um")

# Or use different species
scene_human = Scene(atlas_name="allen_human_500um")
scene_rat = Scene(atlas_name="whs_sd_rat_39um")

scene.render()
```

### Add Brain Regions

```python
from brainrender import Scene

scene = Scene()

# Add specific brain region
scene.add_brain_region("HIP", alpha=0.8, color="skyblue")

# Add multiple regions
regions = ["CA1", "CA2", "CA3", "DG"]
for region in regions:
    scene.add_brain_region(region)

scene.render()
```

---

## Atlas Navigation

### List Available Regions

```python
from brainrender import Scene

scene = Scene(atlas_name="allen_mouse_25um")

# Get all available regions
regions = scene.atlas.get_structure_ancestors("CA1")
print(f"CA1 ancestors: {regions}")

# Get region hierarchy
descendants = scene.atlas.get_structure_descendants("HIP")
print(f"Hippocampus subregions: {descendants}")
```

### Search for Regions

```python
# Search for regions by name
scene = Scene()

# Find regions matching pattern
matches = scene.atlas.get_structures_by_name("cortex")
print(f"Found {len(matches)} regions matching 'cortex'")

for match in matches[:5]:
    print(f"  - {match['acronym']}: {match['name']}")
```

### Region Information

```python
# Get detailed information about a region
scene = Scene()

region_info = scene.atlas.get_structure_by_acronym("CA1")
print(f"Name: {region_info['name']}")
print(f"ID: {region_info['id']}")
print(f"Parent: {region_info['structure_id_path']}")
```

---

## Region Visualization

### Single Region with Custom Style

```python
from brainrender import Scene

scene = Scene()

# Add region with custom styling
hippocampus = scene.add_brain_region(
    "HIP",
    alpha=0.7,
    color="deepskyblue",
    silhouette=False
)

# Add context (transparent brain)
scene.add_brain_region("root", alpha=0.1)

scene.render()
```

### Multiple Regions with Different Colors

```python
from brainrender import Scene

scene = Scene()

# Define regions and colors
regions_colors = {
    "HIP": "skyblue",
    "TH": "salmon",
    "STR": "lightgreen",
    "CB": "orchid"
}

# Add all regions
for region, color in regions_colors.items():
    scene.add_brain_region(region, alpha=0.8, color=color)

# Render with labels
scene.render(camera="sagittal")
```

### Hierarchical Region Display

```python
from brainrender import Scene

scene = Scene()

# Add parent region (transparent)
scene.add_brain_region("HIP", alpha=0.2, color="gray")

# Add subregions (opaque)
subregions = ["CA1", "CA2", "CA3", "DG"]
colors = ["red", "blue", "green", "yellow"]

for region, color in zip(subregions, colors):
    scene.add_brain_region(region, alpha=0.9, color=color)

scene.render()
```

### Region Silhouettes

```python
from brainrender import Scene

scene = Scene()

# Add region with silhouette outline
scene.add_brain_region(
    "HIP",
    alpha=0.6,
    color="skyblue",
    silhouette=True,
    silhouette_kwargs={'lw': 2, 'color': 'black'}
)

scene.render()
```

---

## Connectivity Visualization

### Allen Connectivity Data

```python
from brainrender import Scene

scene = Scene()

# Add source region
scene.add_brain_region("VISp", alpha=0.5, color="steelblue")

# Add connectivity streamlines
scene.add_streamlines(
    "VISp",
    color="red",
    alpha=0.8,
    radius=10
)

# Add transparent brain for context
scene.add_brain_region("root", alpha=0.1)

scene.render(camera="three_quarters")
```

### Custom Connections

```python
from brainrender import Scene
import numpy as np

scene = Scene()

# Define source and target regions
source = "VISp"
target = "SC"

# Add both regions
scene.add_brain_region(source, alpha=0.7, color="blue")
scene.add_brain_region(target, alpha=0.7, color="red")

# Add connection line
source_center = scene.atlas.get_region_CenterOfMass(source)
target_center = scene.atlas.get_region_CenterOfMass(target)

# Create line between centers
from vedo import Line
connection = Line(source_center, target_center, c="green", lw=5)
scene.add(connection)

scene.render()
```

### Network Visualization

```python
from brainrender import Scene

scene = Scene()

# Define network nodes
nodes = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm"]

# Add nodes
for node in nodes:
    scene.add_brain_region(node, alpha=0.8)

# Add connections (example)
scene.add_streamlines("VISp", color="red", alpha=0.5, radius=8)

scene.render()
```

---

## Cell and Neuron Visualization

### Cell Scatter Plot

```python
from brainrender import Scene
import numpy as np

scene = Scene()

# Add brain region
scene.add_brain_region("HIP", alpha=0.3)

# Generate random cell coordinates (example)
# In practice, load from experimental data
n_cells = 500
bounds = scene.atlas.get_region_bounds("CA1")

cells = np.random.uniform(
    low=[bounds[0], bounds[2], bounds[4]],
    high=[bounds[1], bounds[3], bounds[5]],
    size=(n_cells, 3)
)

# Add cells as points
scene.add_cells(
    cells,
    color="red",
    radius=20,
    alpha=0.8
)

scene.render()
```

### Cell Density Heatmap

```python
from brainrender import Scene
import numpy as np

scene = Scene()

# Add brain region
region = scene.add_brain_region("HIP", alpha=0.2)

# Simulate cell density (replace with real data)
# Color vertices by density
n_vertices = len(region.points())
density = np.random.exponential(scale=1.0, size=n_vertices)

# Apply colormap to region
region.cmap("Reds", density, vmin=0, vmax=5)

scene.add(region)
scene.render()
```

### Neuron Morphology

```python
from brainrender import Scene
from brainrender.actors import Neuron

scene = Scene()

# Add neuron from SWC file
# (Neuronal reconstruction file format)
neuron = scene.add_neuron(
    "path/to/neuron.swc",
    color="darkblue",
    alpha=0.8,
    neurite_radius=8
)

# Add surrounding region for context
scene.add_brain_region("CA1", alpha=0.2)

scene.render()
```

---

## Custom Meshes and Data

### Load Custom Mesh

```python
from brainrender import Scene
from vedo import Mesh, load

scene = Scene()

# Load custom mesh (OBJ, STL, VTK, etc.)
custom_mesh = load("path/to/custom_mesh.obj")

# Style the mesh
custom_mesh.c("yellow").alpha(0.7)

# Add to scene
scene.add(custom_mesh)

# Add brain for reference
scene.add_brain_region("root", alpha=0.1)

scene.render()
```

### Overlay Volume Data

```python
from brainrender import Scene
from vedo import Volume
import nibabel as nib
import numpy as np

scene = Scene()

# Load volumetric data (e.g., activity map)
img = nib.load("activity_map.nii.gz")
data = img.get_fdata()

# Create volume visualization
volume = Volume(data).cmap("hot").alpha([0, 0.5, 0.9])

scene.add(volume)
scene.add_brain_region("root", alpha=0.1)

scene.render()
```

### Add Injection Site

```python
from brainrender import Scene
from vedo import Sphere

scene = Scene()

# Add brain region
scene.add_brain_region("VISp", alpha=0.5)

# Define injection site
injection_coords = [5400, 2600, 4500]  # Example coordinates

# Create sphere at injection site
injection = Sphere(
    pos=injection_coords,
    r=100,
    c="lime",
    alpha=0.9
)

scene.add(injection)
scene.render()
```

---

## Camera and View Control

### Predefined Camera Views

```python
from brainrender import Scene

scene = Scene()
scene.add_brain_region("root", alpha=0.4)

# Available camera positions:
# 'frontal', 'sagittal', 'top', 'three_quarters'

scene.render(camera="sagittal")
```

### Custom Camera Position

```python
from brainrender import Scene

scene = Scene()
scene.add_brain_region("HIP", alpha=0.7)

# Set custom camera position
scene.render(
    camera={
        'pos': (10000, 5000, -15000),
        'focalPoint': (7000, 4000, 5000),
        'viewup': (0, -1, 0)
    }
)
```

### Zoom and Rotate

```python
from brainrender import Scene

scene = Scene()
scene.add_brain_region("HIP", alpha=0.7)

# Get camera before rendering
scene.render(interactive=False, zoom=1.5)

# Rotate view
scene.rotate(azimuth=45, elevation=30)

# Continue rendering
scene.render()
```

---

## Export and Figures

### Save Screenshot

```python
from brainrender import Scene

scene = Scene()
scene.add_brain_region("HIP", alpha=0.7, color="skyblue")
scene.add_brain_region("TH", alpha=0.7, color="salmon")

# Render and save
scene.render(interactive=False)
scene.screenshot(filename="brain_figure.png")

# Or specify path
scene.screenshot(filename="/path/to/output/figure.png")
```

### High-Resolution Export

```python
from brainrender import Scene

scene = Scene()
scene.add_brain_region("HIP", alpha=0.7)

# Render at high resolution
scene.render(interactive=False)

# Save high-res image
scene.screenshot(
    filename="high_res_brain.png",
    scale=3  # 3x resolution
)
```

### Create Animation

```python
from brainrender import Scene, Animation

# Create scene
scene = Scene()
scene.add_brain_region("HIP", alpha=0.7)

# Create animation
anim = Animation(scene, "/path/to/output", "rotation_video")

# Add frames with rotation
for angle in range(0, 360, 2):
    scene.rotate(azimuth=2)
    anim.add_frame()

# Save video
anim.make_video(fps=30, format="mp4")
```

### Multi-Panel Figure

```python
from brainrender import Scene
from vedo import show

# Create multiple scenes
scene1 = Scene()
scene1.add_brain_region("HIP", alpha=0.7, color="skyblue")

scene2 = Scene()
scene2.add_brain_region("TH", alpha=0.7, color="salmon")

# Render in grid layout
show(
    scene1.render(interactive=False),
    scene2.render(interactive=False),
    N=2,  # 2 panels
    axes=0
)
```

---

## Integration with Claude Code

Brainrender integrates well with automated workflows:

```python
# brainrender_pipeline.py - Automated brain visualization

from brainrender import Scene
from pathlib import Path
import pandas as pd

class BrainVisualizer:
    """Wrapper for Brainrender in automated pipelines."""

    def __init__(self, atlas="allen_mouse_25um", output_dir="figures"):
        self.atlas = atlas
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def visualize_regions(self, regions, colors=None, filename="regions.png"):
        """Visualize multiple brain regions."""

        scene = Scene(atlas_name=self.atlas)

        # Use default colors if not provided
        if colors is None:
            colors = ["skyblue", "salmon", "lightgreen", "orchid", "yellow"]

        # Add regions
        for i, region in enumerate(regions):
            color = colors[i % len(colors)]
            scene.add_brain_region(region, alpha=0.7, color=color)

        # Add transparent brain
        scene.add_brain_region("root", alpha=0.1)

        # Render and save
        scene.render(interactive=False)
        output_path = self.output_dir / filename
        scene.screenshot(str(output_path))

        print(f"Saved: {output_path}")

    def batch_visualize(self, region_list_file):
        """Batch visualize regions from CSV file."""

        # Load region list
        df = pd.read_csv(region_list_file)

        for idx, row in df.iterrows():
            regions = row['regions'].split(';')
            output_name = f"{row['group_name']}.png"

            self.visualize_regions(regions, filename=output_name)

# Usage
visualizer = BrainVisualizer(atlas="allen_mouse_25um")

# Visualize hippocampal subregions
visualizer.visualize_regions(
    regions=["CA1", "CA2", "CA3", "DG"],
    filename="hippocampus_subregions.png"
)
```

**Connectivity Analysis Visualization:**

```python
# connectivity_visualization.py
from brainrender import Scene
import pandas as pd

def visualize_connectivity_matrix(connectivity_df, output_file):
    """Visualize connectivity from matrix."""

    scene = Scene()

    # Get unique regions
    regions = list(set(
        connectivity_df['source'].tolist() +
        connectivity_df['target'].tolist()
    ))

    # Add all regions
    for region in regions:
        scene.add_brain_region(region, alpha=0.6)

    # Add connections
    for _, row in connectivity_df.iterrows():
        if row['strength'] > 0.5:  # Threshold
            source_center = scene.atlas.get_region_CenterOfMass(row['source'])
            target_center = scene.atlas.get_region_CenterOfMass(row['target'])

            from vedo import Line
            connection = Line(
                source_center,
                target_center,
                c="red",
                lw=row['strength'] * 10,
                alpha=0.7
            )
            scene.add(connection)

    scene.render(interactive=False)
    scene.screenshot(output_file)

# Load connectivity data
conn_df = pd.read_csv('connectivity_matrix.csv')
visualize_connectivity_matrix(conn_df, 'connectivity_network.png')
```

---

## Integration with Other Tools

### Allen SDK Integration

```python
# Use Allen SDK to fetch data, Brainrender to visualize
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from brainrender import Scene

# Initialize Allen SDK
mcc = MouseConnectivityCache()

# Get experiments for a structure
structure = "VISp"
experiments = mcc.get_experiments(structure_ids=[structure])

# Visualize with Brainrender
scene = Scene()
scene.add_brain_region(structure, alpha=0.7)
scene.add_streamlines(structure, color="red", alpha=0.6)

scene.render()
```

### Integration with Analysis Results

```python
# Visualize statistical results on brain regions
from brainrender import Scene
import pandas as pd
import numpy as np

# Load statistical results (region-wise t-values)
results = pd.read_csv('statistical_results.csv')

scene = Scene()

# Color regions by t-value
from vedo import colorMap

for _, row in results.iterrows():
    region = row['region']
    t_value = row['t_statistic']

    # Map t-value to color
    color = colorMap(t_value, "RdBu_r", vmin=-5, vmax=5)

    # Add region
    scene.add_brain_region(region, alpha=0.8, color=color)

scene.render()
```

### Combine with Neuronal Tracing

```python
# Visualize traced neurons from registration
from brainrender import Scene
import pandas as pd

scene = Scene()

# Load neuron trace coordinates
traces = pd.read_csv('registered_neurons.csv')

# Add brain region
scene.add_brain_region("CA1", alpha=0.3)

# Add traced neurons
for neuron_id in traces['neuron_id'].unique():
    neuron_data = traces[traces['neuron_id'] == neuron_id]
    coords = neuron_data[['x', 'y', 'z']].values

    # Add as line
    from vedo import Line
    neuron_line = Line(coords, c="blue", lw=2, alpha=0.7)
    scene.add(neuron_line)

scene.render()
```

---

## Advanced Techniques

### Custom Shaders and Materials

```python
from brainrender import Scene
from vedo import Mesh

scene = Scene()

# Add region
region_actor = scene.add_brain_region("HIP", alpha=0.7)

# Apply custom material properties
region_actor.phong()  # Phong shading
region_actor.metallic(0.5)  # Metallic appearance
region_actor.specular(0.8)  # Specular highlights

scene.render()
```

### Clipping Planes

```python
from brainrender import Scene

scene = Scene()

# Add brain
brain = scene.add_brain_region("root", alpha=0.8)

# Add clipping plane
from vedo import Plane
clip_plane = Plane(pos=(7000, 4000, 5000), normal=(1, 0, 0))

# Clip brain
brain_clipped = brain.cutWithPlane(origin=clip_plane.center, normal=clip_plane.normal)

scene.add(brain_clipped)
scene.render()
```

### Depth Peeling (Transparency)

```python
from brainrender import Scene

scene = Scene()

# Add multiple transparent layers
scene.add_brain_region("root", alpha=0.2)
scene.add_brain_region("HIP", alpha=0.5)
scene.add_brain_region("CA1", alpha=0.8)

# Enable depth peeling for better transparency
scene.render(
    interactive=True,
    depthpeeling=True  # Better transparency rendering
)
```

---

## Troubleshooting

### Problem 1: Atlas Download Fails

**Symptoms:** Error downloading atlas data

**Solution:**
```python
# Manually download atlas
from bg_atlasapi.bg_atlas import BGAtlas

# Specify cache directory
import os
os.environ['BRAINGLOBE_DATA_DIR'] = '/path/to/atlas/cache'

# Download specific atlas
atlas = BGAtlas("allen_mouse_25um", check_latest=False)
```

### Problem 2: Rendering Issues

**Symptoms:** Blank window or visualization errors

**Solution:**
```python
# Try different backend
import vedo
vedo.settings.default_backend = 'vtk'  # or 'k3d', '2d'

# Check VTK installation
import vtk
print(vtk.vtkVersion.GetVTKVersion())

# Update graphics drivers if issues persist
```

### Problem 3: Region Not Found

**Symptoms:** KeyError when adding region

**Solution:**
```python
# Check region acronym
from brainrender import Scene
scene = Scene()

# Search for region
matches = scene.atlas.get_structures_by_name("hippocampus")
for match in matches:
    print(f"{match['acronym']}: {match['name']}")

# Use correct acronym
scene.add_brain_region("HIP")  # Not "hippocampus"
```

### Problem 4: Out of Memory

**Symptoms:** Crash with large visualizations

**Solution:**
```python
# Reduce mesh resolution
scene = Scene(atlas_name="allen_mouse_100um")  # Lower resolution

# Reduce number of actors
# Combine regions when possible
# Use hemisphere-specific regions: "HIP-lh", "HIP-rh"
```

### Problem 5: Interactive Mode Not Working

**Symptoms:** Window closes immediately

**Solution:**
```python
# Use interactive=True
scene.render(interactive=True)

# Or in Jupyter
scene.render(interactive=False)  # For notebook display

# Check matplotlib backend
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg'
```

---

## Best Practices

### 1. Scene Organization

- **Start simple:** Add brain first, then overlay data
- **Use transparency:** Make background structures semi-transparent
- **Color choice:** Use distinct colors for different regions
- **Hierarchical display:** Show parent regions with children
- **Context:** Always include reference structures

### 2. Performance

- **Atlas resolution:** Use appropriate resolution (25um, 50um, 100um)
- **Limit actors:** Combine regions when possible
- **Simplify meshes:** Decimate for faster rendering
- **Batch processing:** Pre-render and save images
- **Close scenes:** Release resources after rendering

### 3. Publication Figures

- **High resolution:** Use scale parameter for screenshots
- **Consistent views:** Use same camera angles across figures
- **Clear colors:** Avoid ambiguous color schemes
- **Labels:** Add annotations in post-processing
- **Multiple views:** Provide different perspectives

### 4. Reproducibility

- **Script everything:** Use Python scripts, not interactive
- **Version control:** Track Brainrender and atlas versions
- **Document atlas:** Record atlas name and version used
- **Save parameters:** Store camera positions and colors
- **Seed random:** For any random data generation

### 5. Cross-Species Work

- **Check atlas compatibility:** Verify region names across species
- **Standardize coordinates:** Account for different coordinate systems
- **Document species:** Clearly label which atlas is used
- **Comparative views:** Create side-by-side visualizations

---

## Resources

### Official Documentation

- **BrainGlobe Website:** https://brainglobe.info/
- **Brainrender Docs:** https://docs.brainglobe.info/brainrender/
- **GitHub Repository:** https://github.com/brainglobe/brainrender
- **Issue Tracker:** https://github.com/brainglobe/brainrender/issues

### Tutorials and Examples

- **User Guide:** https://docs.brainglobe.info/brainrender/usage/
- **Example Gallery:** Available in GitHub repository
- **Video Tutorials:** On BrainGlobe YouTube channel
- **Jupyter Notebooks:** Example notebooks in docs

### Community

- **Zulip Chat:** https://brainglobe.zulipchat.com/
- **Forum:** https://forum.image.sc/ (tag: brainglobe)
- **Twitter:** @brain_globe

### Related Projects

- **BrainGlobe Atlas API:** https://github.com/brainglobe/bg-atlasapi
- **Cellfinder:** Cell detection tool
- **brainreg:** Registration tool

---

## Citation

```bibtex
@article{claudi2021brainrender,
  title={Visualizing anatomically registered data with Brainrender},
  author={Claudi, Federico and Tyson, Adam L and Petrucco, Luigi and Margrie, Troy W and Portugues, Ruben and Branco, Tiago},
  journal={eLife},
  volume={10},
  pages={e65751},
  year={2021},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.65751}
}
```

---

## Related Tools

- **Surfice:** Surface visualization (see `surfice.md`)
- **PyCortex:** Web-based cortical visualization (see `pycortex.md`)
- **Connectome Workbench:** HCP visualization (see `connectome-workbench.md`)
- **BrainNet Viewer:** Network visualization (see `brainnet-viewer.md`)
- **Mango:** Multi-image viewer (see `mango.md`)
- **FSLeyes:** FSL viewer (see `fsleyes.md`)
- **Allen SDK:** Allen Brain Atlas data access
- **vedo:** Python VTK wrapper (rendering engine)
- **BrainGlobe Atlas API:** Atlas management

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**Brainrender Version Covered:** 2.x
**Maintainer:** Claude Code Neuroimaging Skills
