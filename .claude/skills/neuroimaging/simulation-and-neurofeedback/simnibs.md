# SimNIBS: Simulation of Non-Invasive Brain Stimulation

## Overview

SimNIBS (Simulation of Non-invasive Brain Stimulation) is a free, open-source software package for realistic simulations of electric fields induced by transcranial magnetic stimulation (TMS) and transcranial direct current/alternating current stimulation (tDCS/tACS). It uses finite element methods (FEM) to compute electric field distributions in individualized head models, enabling optimization of stimulation parameters and prediction of effects.

**Key Features:**
- **Realistic Head Modeling**: Automatic head mesh generation from MRI
- **TMS Simulation**: Multiple coil types and positions
- **tDCS/tACS Simulation**: Electrode montage optimization
- **Multi-Subject Analysis**: Group-level E-field maps
- **Optimization Algorithms**: Automated coil/electrode placement
- **Python + MATLAB APIs**: Flexible scripting
- **Integration**: FreeSurfer, FSL, SPM compatible
- **Visualization**: gmsh, Paraview, FreeSurfer integration

**Website:** https://simnibs.github.io/simnibs/

**Citation:** Thielscher, A., et al. (2015). Field modeling for transcranial magnetic stimulation: A useful tool to understand the physiological effects of TMS? *IEEE EMBS*, 222-225.

## Installation

### Standard Installation

```bash
# Download SimNIBS installer
wget https://github.com/simnibs/simnibs/releases/download/v4.0.0/simnibs_installer_linux.tar.gz

# Extract
tar -xzf simnibs_installer_linux.tar.gz

# Run installer
cd simnibs_installer
./install

# Follow prompts (default installation recommended)
# Installation directory: ~/SimNIBS-4.0

# Add to PATH (add to ~/.bashrc)
export PATH="${PATH}:${HOME}/SimNIBS-4.0/bin"

# Test installation
simnibs --version
```

### FreeSurfer Integration (Recommended)

```bash
# SimNIBS works best with FreeSurfer for cortical surfaces

# Install FreeSurfer (if not already installed)
# Download from: https://surfer.nmr.mgh.harvard.edu/

# Set FreeSurfer home
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# SimNIBS will automatically detect FreeSurfer installation
```

### Python Environment

```bash
# SimNIBS includes its own Python environment
# Activate it for scripting

source ~/SimNIBS-4.0/miniconda3/bin/activate simnibs_env

# Install additional packages if needed
pip install nilearn nibabel
```

## Creating Head Models

### Automatic Head Mesh Generation (headreco)

```bash
# Generate head model from T1w MRI (basic)
headreco all --sub sub-01 --t1 /path/to/sub-01_T1w.nii.gz

# With T2w for improved skull segmentation
headreco all --sub sub-01 \
  --t1 /path/to/sub-01_T1w.nii.gz \
  --t2 /path/to/sub-01_T2w.nii.gz

# Output structure:
# m2m_sub-01/
#   ├── sub-01.msh          # Tetrahedral head mesh
#   ├── T1fs_conform.nii.gz # Reoriented T1
#   ├── toMNI/              # MNI transformations
#   └── segment/            # Tissue segmentations

# Processing time: 4-8 hours (parallel processing)
```

### Head Model with Custom FreeSurfer

```bash
# If you already have FreeSurfer reconstruction
headreco all --sub sub-01 \
  --t1 /path/to/sub-01_T1w.nii.gz \
  --fs /path/to/freesurfer/sub-01

# This uses existing pial and white matter surfaces
```

### Quality Control

```bash
# Open mesh in gmsh for visual inspection
gmsh m2m_sub-01/sub-01.msh

# Check tissue segmentations
fsleyes m2m_sub-01/final_tissues.nii.gz

# Verify cortical surfaces
freeview -v m2m_sub-01/T1fs_conform.nii.gz \
  -f m2m_sub-01/surf/lh.pial:edgecolor=red \
  -f m2m_sub-01/surf/rh.pial:edgecolor=red

# Common issues:
# - Skull segmentation errors (use T2 to improve)
# - Neck inclusion (usually OK if below cerebellum)
# - Surface reconstruction failures (check FreeSurfer log)
```

### Using Standard MNI Head Model

```bash
# For group studies or when individual MRI unavailable
# SimNIBS includes MNI152 head model

# Copy template
cp -r $SIMNIBSDIR/simnibs_env/simnibs/resources/templates/MNI152_5layer .

# Use in simulations (specify in Python/MATLAB scripts)
```

## TMS Simulation

### Basic Motor Cortex TMS

```python
# Python script for M1 stimulation

from simnibs import sim_struct, run_simnibs

# Initialize session
s = sim_struct.SESSION()
s.subpath = 'm2m_sub-01'  # Head model directory
s.pathfem = 'tms_motor'    # Output directory

# TMS list
tms = s.add_tmslist()
tms.fnamecoil = 'Magstim_70mm_Fig8.ccd'  # Coil file

# Position 1: M1 hand area
pos = tms.add_position()
pos.centre = 'C3'  # 10-20 EEG position
pos.pos_ydir = 'CP3'  # Coil orientation
pos.distance = 4.0  # mm from scalp

# Run simulation
run_simnibs(s)

# Output: E-field maps in tms_motor/
```

### Multiple TMS Coil Positions

```python
# Simulate TMS at multiple positions

from simnibs import sim_struct, run_simnibs

s = sim_struct.SESSION()
s.subpath = 'm2m_sub-01'
s.pathfem = 'tms_dlpfc_grid'

tms = s.add_tmslist()
tms.fnamecoil = 'Magstim_70mm_Fig8.ccd'

# Grid of positions over DLPFC
positions = [
    {'centre': [48, 38, 42], 'name': 'DLPFC_anterior'},
    {'centre': [44, 32, 48], 'name': 'DLPFC_central'},
    {'centre': [40, 26, 52], 'name': 'DLPFC_posterior'}
]

for p in positions:
    pos = tms.add_position()
    pos.centre = p['centre']  # MNI coordinates
    pos.pos_ydir = 'anterior'
    pos.distance = 4.0
    pos.name = p['name']

run_simnibs(s)

# Compare E-fields across positions
```

### Coil Types and Orientations

```bash
# Available coil files in SimNIBS

# Figure-8 coils (focal stimulation)
Magstim_70mm_Fig8.ccd
Dantec_MagProR30.ccd
MagVenture_MC_B70.ccd

# Circular coils (diffuse stimulation)
Magstim_50mm_Circular.ccd

# Custom coils can be created

# Coil orientation examples:
```

```python
# Tangential to cortex (automatic)
pos.didt = 1e6  # A/s (stimulator intensity)

# Specific orientation
pos.matsimnibs = [[1, 0, 0, 50],   # x-axis: AP direction
                  [0, 1, 0, 30],   # y-axis: ML direction
                  [0, 0, 1, 60],   # z-axis: IS direction
                  [0, 0, 0, 1]]    # (position in mm)

# Perpendicular to gyrus (needs surface normal)
pos.type = 'perpendicular'
```

### Analyzing TMS Results

```python
# Extract E-field at target region

from simnibs import read_msh
import numpy as np

# Load mesh with results
mesh = read_msh('tms_motor/sub-01_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh')

# Get E-field magnitude in gray matter
gm_indices = mesh.elm.tag1 == 2  # Gray matter label
e_field_gm = mesh.field['E'].value[gm_indices]
e_field_mag_gm = mesh.field['E'].norm()[gm_indices]

# Peak E-field
peak_e = np.max(e_field_mag_gm)
print(f"Peak E-field in GM: {peak_e:.2f} V/m")

# Mean E-field in top 1%
threshold = np.percentile(e_field_mag_gm, 99)
target_indices = e_field_mag_gm > threshold
mean_target_e = np.mean(e_field_mag_gm[target_indices])
print(f"Mean E-field in target (top 1%): {mean_target_e:.2f} V/m")

# Determine stimulated region
stimulated_coords = mesh.nodes.node_coord[target_indices]
centroid = np.mean(stimulated_coords, axis=0)
print(f"Centroid of stimulation: {centroid}")
```

## tDCS Simulation

### Basic Montage Setup

```python
# Simple anodal tDCS over left DLPFC

from simnibs import sim_struct, run_simnibs

s = sim_struct.SESSION()
s.subpath = 'm2m_sub-01'
s.pathfem = 'tdcs_dlpfc'

# tDCS list
tdcs = s.add_tdcslist()
tdcs.currents = [1e-3, -1e-3]  # 1 mA (anode, cathode)

# Electrode 1: Anode over F3 (DLPFC)
electrode1 = tdcs.add_electrode()
electrode1.centre = 'F3'  # 10-20 position
electrode1.shape = 'rect'
electrode1.dimensions = [50, 50]  # mm
electrode1.thickness = 4  # Sponge + saline

# Electrode 2: Cathode over right supraorbital
electrode2 = tdcs.add_electrode()
electrode2.centre = 'Fp2'
electrode2.shape = 'rect'
electrode2.dimensions = [50, 50]
electrode2.thickness = 4

run_simnibs(s)

# Output: E-field and current density maps
```

### HD-tDCS (4×1 Ring Montage)

```python
# High-definition tDCS with center electrode + 4 return electrodes

s = sim_struct.SESSION()
s.subpath = 'm2m_sub-01'
s.pathfem = 'hdtdcs_m1'

tdcs = s.add_tdcslist()

# Center electrode: 2 mA anode
# 4 return electrodes: 0.5 mA cathode each
tdcs.currents = [2e-3, -0.5e-3, -0.5e-3, -0.5e-3, -0.5e-3]

# Center over C3 (M1)
center = tdcs.add_electrode()
center.centre = 'C3'
center.shape = 'ellipse'
center.dimensions = [12, 12]  # Small diameter

# Ring electrodes (2cm radius from center)
ring_offsets = [
    (0, 20),    # Anterior
    (20, 0),    # Right
    (0, -20),   # Posterior
    (-20, 0)    # Left
]

for offset in ring_offsets:
    ring_el = tdcs.add_electrode()
    ring_el.centre = 'C3'
    ring_el.pos_offset = offset  # mm offset
    ring_el.shape = 'ellipse'
    ring_el.dimensions = [12, 12]

run_simnibs(s)
```

### Multi-Electrode Optimization

```python
# Optimize electrode positions for targeted stimulation

from simnibs.optimization import optimize_tdcs
import numpy as np

# Define target region (e.g., left DLPFC from fMRI activation)
target_mesh = read_msh('m2m_sub-01/sub-01.msh')
target_coords = np.loadtxt('dlpfc_target_coords.txt')  # From fMRI

# Candidate electrode positions (10-10 system)
electrode_positions = ['F3', 'F4', 'Fz', 'C3', 'C4', 'P3', 'P4', 'Fp1', 'Fp2']

# Optimize
result = optimize_tdcs(
    leadfield='m2m_sub-01/leadfield/leadfield.hdf5',
    target=target_coords,
    max_total_current=2e-3,  # 2 mA total
    max_el_current=1e-3,     # 1 mA per electrode
    max_n_electrodes=5        # Use up to 5 electrodes
)

print("Optimal currents:")
for pos, current in zip(electrode_positions, result.currents):
    if abs(current) > 1e-6:
        print(f"  {pos}: {current*1e3:.2f} mA")

print(f"Focality: {result.focality:.3f}")
print(f"Target E-field: {result.target_e_field:.2f} V/m")
```

## Advanced Coil/Electrode Placement

### Anatomical Targeting

```python
# Target specific cortical region by name

from simnibs import sim_struct

s = sim_struct.SESSION()
s.subpath = 'm2m_sub-01'
s.pathfem = 'tms_motor_hand'

tms = s.add_tmslist()
tms.fnamecoil = 'Magstim_70mm_Fig8.ccd'

pos = tms.add_position()

# Target hand knob (anatomical landmark)
pos.centre = 'lh.BA4a'  # Left hemisphere, BA4a
pos.pos_ydir = 'perpendicular'  # Perpendicular to cortical surface
pos.distance = 4.0

run_simnibs(s)
```

### Functional Targeting from fMRI

```python
# Target peak fMRI activation

from simnibs import sim_struct, mni2subject_coords
import nibabel as nib
import numpy as np

# Find peak activation in MNI space
fmri_zmap = nib.load('motor_task_zstat.nii.gz')
fmri_data = fmri_zmap.get_fdata()
peak_idx = np.unravel_index(np.argmax(fmri_data), fmri_data.shape)

# Convert to MNI coordinates
affine = fmri_zmap.affine
mni_coords = nib.affines.apply_affine(affine, peak_idx)
print(f"MNI peak: {mni_coords}")

# Transform to subject space
subject_coords = mni2subject_coords(
    mni_coords,
    'm2m_sub-01'
)
print(f"Subject space: {subject_coords}")

# Create TMS simulation
s = sim_struct.SESSION()
s.subpath = 'm2m_sub-01'
s.pathfem = 'tms_fmri_peak'

tms = s.add_tmslist()
tms.fnamecoil = 'Magstim_70mm_Fig8.ccd'

pos = tms.add_position()
pos.centre = subject_coords.tolist()
pos.distance = 4.0

run_simnibs(s)
```

### Coil Orientation Optimization

```python
# Find optimal coil angle for maximum E-field

from simnibs import sim_struct, run_simnibs
import numpy as np

# Test multiple orientations
angles = np.arange(0, 180, 15)  # 0-180 degrees in 15-deg steps
results = []

for angle in angles:
    s = sim_struct.SESSION()
    s.subpath = 'm2m_sub-01'
    s.pathfem = f'tms_rotation_{angle:03d}'

    tms = s.add_tmslist()
    tms.fnamecoil = 'Magstim_70mm_Fig8.ccd'

    pos = tms.add_position()
    pos.centre = [-37, -21, 58]  # M1 hand area
    pos.distance = 4.0

    # Rotate coil
    pos.angle = angle  # Degrees

    run_simnibs(s)

    # Load and analyze
    mesh = read_msh(f'tms_rotation_{angle:03d}/sub-01_TMS_1-0001.msh')
    gm_e_field = mesh.field['E'].norm()[mesh.elm.tag1 == 2]
    peak_e = np.max(gm_e_field)

    results.append({'angle': angle, 'peak_e': peak_e})

# Find optimal angle
best = max(results, key=lambda x: x['peak_e'])
print(f"Optimal angle: {best['angle']}° (E = {best['peak_e']:.1f} V/m)")
```

## Group Analysis

### Multi-Subject Simulations

```python
# Run TMS simulations for multiple subjects

from simnibs import sim_struct, run_simnibs
import glob

subjects = glob.glob('m2m_sub-*')

for subject in subjects:
    sub_id = subject.replace('m2m_', '')

    s = sim_struct.SESSION()
    s.subpath = subject
    s.pathfem = f'{sub_id}_tms_dlpfc'

    tms = s.add_tmslist()
    tms.fnamecoil = 'Magstim_70mm_Fig8.ccd'

    pos = tms.add_position()
    pos.centre = 'F3'  # Standard DLPFC position
    pos.pos_ydir = 'Fz'
    pos.distance = 4.0

    run_simnibs(s)

    print(f"Completed: {sub_id}")
```

### Group-Level E-Field Maps

```python
# Average E-fields across subjects in MNI space

from simnibs import msh2nii, transformations
import nibabel as nib
import numpy as np

subjects = ['sub-01', 'sub-02', 'sub-03']
e_field_maps = []

for sub in subjects:
    # Convert mesh to NIfTI
    msh2nii.mesh2nifti(
        f'{sub}_tms_dlpfc/{sub}_TMS_1-0001.msh',
        f'{sub}_efield.nii.gz',
        'E'  # E-field magnitude
    )

    # Transform to MNI
    transformations.warp_volume(
        f'{sub}_efield.nii.gz',
        f'm2m_{sub}/toMNI/Conform2MNI.mat',
        f'{sub}_efield_mni.nii.gz'
    )

    # Load
    img = nib.load(f'{sub}_efield_mni.nii.gz')
    e_field_maps.append(img.get_fdata())

# Compute group average
group_mean = np.mean(e_field_maps, axis=0)
group_std = np.std(e_field_maps, axis=0)

# Save group maps
nib.save(nib.Nifti1Image(group_mean, img.affine), 'group_mean_efield.nii.gz')
nib.save(nib.Nifti1Image(group_std, img.affine), 'group_std_efield.nii.gz')

# Visualize variability
print(f"Mean peak E-field: {np.max(group_mean):.1f} V/m")
print(f"Std of peak E-field: {np.max(group_std):.1f} V/m")
```

### Inter-Subject Variability Analysis

```python
# Quantify inter-individual differences

import pandas as pd

def compute_subject_metrics(subject):
    mesh = read_msh(f'{subject}_tms_dlpfc/{subject}_TMS_1-0001.msh')

    # Gray matter E-field
    gm_mask = mesh.elm.tag1 == 2
    e_gm = mesh.field['E'].norm()[gm_mask]

    return {
        'subject': subject,
        'peak_e': np.max(e_gm),
        'mean_e': np.mean(e_gm),
        'median_e': np.median(e_gm),
        'volume_above_50': np.sum(e_gm > 50)  # Volume with E > 50 V/m
    }

# Collect metrics
metrics = [compute_subject_metrics(s) for s in subjects]
df = pd.DataFrame(metrics)

print(df)
print(f"\nCoefficient of variation (peak): {df['peak_e'].std() / df['peak_e'].mean():.2f}")
```

## Electrode/Coil Design

### Custom TMS Coil

```python
# Create custom coil design

from simnibs.simulation import coil

# Define coil windings
coil_file = coil.Coil()

# Add windings (figure-8 example)
coil_file.add_element(
    position=[0, 0, 0],      # mm
    direction=[1, 0, 0],     # x-direction
    radius=35,                # mm
    turns=10
)

coil_file.add_element(
    position=[0, 70, 0],
    direction=[-1, 0, 0],     # Opposite direction
    radius=35,
    turns=10
)

# Save
coil_file.write('custom_fig8.ccd')
```

### tDCS Electrode Shapes

```python
# Custom electrode geometries

tdcs = s.add_tdcslist()

# Rectangular electrode
rect_el = tdcs.add_electrode()
rect_el.shape = 'rect'
rect_el.dimensions = [50, 70]  # Width × Height mm

# Elliptical electrode
ellipse_el = tdcs.add_electrode()
ellipse_el.shape = 'ellipse'
ellipse_el.dimensions = [40, 60]  # Major × Minor axes mm

# Custom polygon electrode (e.g., for avoiding scalp defects)
custom_el = tdcs.add_electrode()
custom_el.shape = 'custom'
custom_el.vertices = [[0, 0], [30, 0], [30, 40], [15, 50], [0, 40]]  # mm
```

## Integration with Neuroimaging

### Combining with DTI Tractography

```python
# Visualize E-field along white matter tracts

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking

# Load tractography
tracts = load_tractogram('tractography.trk')

# Load E-field mesh
e_mesh = read_msh('tms_simulation/sub-01_TMS.msh')

# Sample E-field along tracts
def sample_e_field_on_tract(tract, mesh):
    """Sample E-field magnitude along tract"""
    e_values = []
    for point in tract:
        # Find nearest mesh element
        distances = np.linalg.norm(mesh.nodes.node_coord - point, axis=1)
        nearest_node = np.argmin(distances)
        e_value = mesh.field['E'].norm()[nearest_node]
        e_values.append(e_value)
    return e_values

# Analyze CST specifically
cst_tracts = select_tracts_by_roi(tracts, 'corticospinal_tract_roi.nii.gz')

for tract in cst_tracts:
    e_along_tract = sample_e_field_on_tract(tract, e_mesh)
    print(f"Peak E-field in CST: {np.max(e_along_tract):.1f} V/m")
```

### Integration with fMRI Connectivity

```python
# Target based on connectivity patterns

from nilearn import connectome
import nibabel as nib

# Compute seed-based connectivity
seed_roi = 'dlpfc_seed.nii.gz'
resting_fmri = 'sub-01_task-rest_bold.nii.gz'

# Find voxels most connected to seed
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

seed_masker = NiftiLabelsMasker(labels_img=seed_roi)
seed_ts = seed_masker.fit_transform(resting_fmri)

brain_masker = NiftiMasker()
brain_ts = brain_masker.fit_transform(resting_fmri)

# Correlation
correlation = np.corrcoef(seed_ts.T, brain_ts.T)[0, 1:]

# Threshold and find peak
conn_map = brain_masker.inverse_transform(correlation)
peak_conn_voxel = np.unravel_index(np.argmax(conn_map.get_fdata()), conn_map.shape)

# Use as TMS target
peak_mni = nib.affines.apply_affine(conn_map.affine, peak_conn_voxel)
subject_coords = mni2subject_coords(peak_mni, 'm2m_sub-01')

# Simulate TMS at connectivity-defined target
# ... (use subject_coords as pos.centre)
```

## Visualization

### E-Field on Cortical Surface

```python
# Overlay E-field on FreeSurfer surface

from simnibs.msh import mesh_io
import os

# Convert mesh field to FreeSurfer overlay
mesh = read_msh('tms_simulation/sub-01_TMS.msh')

# Map to FreeSurfer surface
mesh_io.write_curv(
    'lh.E_field.mgh',
    mesh.field['E'].norm(),
    mesh
)

# View in Freeview
os.system('''
freeview -f m2m_sub-01/surf/lh.pial:overlay=lh.E_field.mgh:overlay_threshold=50,150
''')
```

### Creating Publication Figures

```python
# Export for visualization

from simnibs import msh2nii
import matplotlib.pyplot as plt
from nilearn import plotting

# Convert to NIfTI
msh2nii.mesh2nifti(
    'tms_simulation/sub-01_TMS.msh',
    'e_field.nii.gz',
    'E',
    reference='m2m_sub-01/T1fs_conform.nii.gz'
)

# Plot with nilearn
fig = plotting.plot_stat_map(
    'e_field.nii.gz',
    bg_img='m2m_sub-01/T1fs_conform.nii.gz',
    threshold=50,
    vmax=150,
    cmap='hot',
    title='TMS E-field (V/m)',
    display_mode='ortho'
)

plt.savefig('tms_efield_fig.png', dpi=300)
```

## Troubleshooting

### Head Model Generation Failures

**Problem:** Tissue segmentation errors

```bash
# Solution 1: Provide T2w for better skull segmentation
headreco all --sub sub-01 --t1 t1.nii.gz --t2 t2.nii.gz

# Solution 2: Manual correction
# Edit final_tissues.nii.gz in FSLeyes
# Label values: 0=background, 1=WM, 2=GM, 3=CSF, 4=bone, 5=scalp
# Then regenerate mesh:
headreco --sub sub-01 --steps mesh

# Solution 3: Use CAT12 segmentation (alternative)
headreco all --sub sub-01 --t1 t1.nii.gz --cat12
```

**Problem:** FreeSurfer reconstruction fails

```bash
# Check FreeSurfer log
cat m2m_sub-01/recon-all.log

# Common fix: Manual edits to control points
# Follow FreeSurfer wiki instructions

# Or skip FreeSurfer surfaces (use only volume mesh)
headreco all --sub sub-01 --t1 t1.nii.gz --noFS
```

### Mesh Quality Issues

```bash
# Check mesh quality
simnibs_python -m simnibs.cli.check_mesh m2m_sub-01/sub-01.msh

# Visualize in gmsh
gmsh m2m_sub-01/sub-01.msh

# Look for:
# - Inverted elements (negative volumes)
# - Self-intersections
# - Poor aspect ratios

# Fix mesh quality
simnibs_python -m simnibs.cli.improve_mesh m2m_sub-01/sub-01.msh
```

### Simulation Convergence Problems

```python
# Increase solver iterations if not converging

s = sim_struct.SESSION()
s.fnamehead = 'm2m_sub-01/sub-01.msh'
s.solver_options = 'petsc -ksp_max_it 1000'

# Use different solver
s.solver_options = 'pardiso'  # Direct solver (slower but robust)
```

## Best Practices

### MRI Acquisition

**Recommended Parameters:**
- **T1w**: 1mm isotropic, MPRAGE/MP2RAGE
- **T2w** (optional): 1mm isotropic, matched to T1
- **Minimize artifacts**: Motion, metal (dental work)
- **Coverage**: Include full head (top to C2)

### Reporting Simulation Parameters

**Essential Information:**
- SimNIBS version
- Head model generation method (with/without T2, FreeSurfer version)
- Coil model and position (coordinates or anatomical landmark)
- Stimulator intensity (dI/dt for TMS, current for tDCS)
- Tissue conductivities (if non-default)
- Solver settings

### Validation

```python
# Compare to measurements (if available)

# TMS: Validate against motor threshold or MEP amplitude
# tDCS: Validate against current density measurements (agar phantom)

# Common validation metrics:
# - Peak E-field location vs. empirical hotspot
# - E-field magnitude vs. stimulation threshold
# - Spatial extent vs. spread of effects
```

## References

- **SimNIBS Overview**: Thielscher, A., et al. (2015). Field modeling for transcranial magnetic stimulation. *IEEE EMBS*, 222-225.
- **Head Modeling**: Nielsen, J. D., et al. (2018). Automatic skull segmentation from MR images for realistic volume conductor models of the head. *NeuroImage*, 170, 447-455.
- **TMS Physics**: Opitz, A., et al. (2011). How the brain tissue shapes the electric field induced by transcranial magnetic stimulation. *NeuroImage*, 58(3), 849-859.
- **tDCS Optimization**: Ruffini, G., et al. (2014). Optimization of multifocal transcranial current stimulation. *Clinical Neurophysiology*, 125(9), 1847-1857.
- **Documentation**: https://simnibs.github.io/simnibs/
- **Tutorials**: https://simnibs.github.io/simnibs/build/html/tutorial/tutorial.html
- **Forum**: https://www.nitrc.org/forum/?group_id=1029

## Related Tools

- **ROAST:** Simpler TES field modeling
- **OpenMEEG:** Forward modeling for MEG/EEG
- **TVB:** Network simulations that can ingest SIMNIBS fields
