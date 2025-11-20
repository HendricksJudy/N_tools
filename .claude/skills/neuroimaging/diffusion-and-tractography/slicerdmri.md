# SlicerDMRI

## Overview

**SlicerDMRI** is a comprehensive diffusion MRI extension for 3D Slicer that provides an integrated environment for diffusion data visualization, processing, tractography, and quantitative analysis. Combining GUI-based interaction with advanced diffusion methods, SlicerDMRI offers tensor estimation, deterministic and probabilistic tractography, interactive fiber bundle editing, and connectivity analysis within the powerful 3D Slicer platform.

SlicerDMRI is particularly valuable for clinical applications, neurosurgical planning, and interactive data exploration. It integrates with the broader 3D Slicer ecosystem, enabling multi-modal analysis combining diffusion with structural, functional, and other imaging modalities.

**Key Use Cases:**
- Interactive diffusion MRI exploration
- Clinical tract visualization and analysis
- Neurosurgical fiber tracking for planning
- DTI scalar map calculation and visualization
- Tractography-based connectivity analysis
- Teaching and demonstration
- Multi-modal image integration
- ROI-based tract filtering

**Official Website:** https://dmri.slicer.org/
**Documentation:** https://dmri.slicer.org/docs/
**Source Code:** https://github.com/SlicerDMRI/SlicerDMRI

---

## Key Features

- **Integrated Pipeline:** Complete diffusion analysis in 3D Slicer
- **DTI Fitting:** Tensor estimation and scalar maps (FA, MD, AD, RD)
- **Tractography:** Deterministic, probabilistic, and UKF methods
- **Interactive Visualization:** 3D fiber bundle rendering and exploration
- **ROI-Based Filtering:** Extract specific bundles with ROI placement
- **Fiber Bundle Editing:** Interactive streamline selection and cleaning
- **Connectivity Analysis:** Generate connectivity matrices from parcellations
- **DWI Registration:** Motion and distortion correction
- **Multi-Fiber Models:** UKF tractography for crossing fibers
- **FreeSurfer Integration:** Use parcellations for connectivity
- **DICOM Support:** Import diffusion DICOM data
- **Batch Processing:** Command-line interface for automation
- **Python Scripting:** Full programmatic control in Slicer
- **Cross-Platform:** Windows, macOS, Linux
- **Open Source:** Part of 3D Slicer ecosystem

---

## Installation

### Install 3D Slicer

```bash
# Download 3D Slicer from:
# https://download.slicer.org/

# Linux
wget https://download.slicer.org/...
tar -xzvf Slicer-X.Y.Z-linux.tar.gz
cd Slicer-X.Y.Z-linux
./Slicer

# macOS: Download DMG and install

# Windows: Download installer and run
```

### Install SlicerDMRI Extension

```
1. Launch 3D Slicer
2. View → Extension Manager
3. Search for "SlicerDMRI"
4. Click "Install"
5. Restart Slicer
```

### Verify Installation

```
After restart:
- Modules menu → Diffusion → should list SlicerDMRI modules:
  - DTI Estimation
  - Tractography Display
  - Fiber Bundle
  - etc.
```

---

## Load DWI Data

### Load NIFTI DWI

```
1. File → Add Data
2. Select DWI NIFTI file (.nii.gz)
3. Check "Show Options"
4. Select "DWI Volume" as description
5. Add bval and bvec files
6. Click OK

Or use Diffusion Weighted DICOM Import module
```

### Load DICOM DWI

```
1. Modules → DICOM
2. Import DICOM files
3. Load diffusion series
4. SlicerDMRI automatically detects gradient info
```

### Python Script to Load DWI

```python
import slicer

# Load DWI volume
dwi_file = '/path/to/dwi.nii.gz'
bval_file = '/path/to/dwi.bval'
bvec_file = '/path/to/dwi.bvec'

# Use DWI NRRD Module
logic = slicer.modules.dwinrrdconverter.logic()
dwiNode = logic.LoadDWIFromNIFTI(dwi_file, bval_file, bvec_file)

print(f"Loaded DWI: {dwiNode.GetName()}")
```

---

## DTI Estimation

### Compute DTI via GUI

```
1. Modules → Diffusion → Diffusion Tensor Estimation
2. Input DWI Volume: Select loaded DWI
3. Output DTI Volume: Create new volume
4. Output Baseline Volume: Create new volume (optional)
5. Estimation Parameters:
   - Estimation Method: Weighted Least Squares
   - Use brain mask (if available)
6. Click "Apply"

Outputs:
- DTI tensor image
- FA, MD, Color FA maps
```

### Compute DTI with Python

```python
import slicer

# Get DWI node
dwiNode = slicer.util.getNode('DWI')

# Create output nodes
dtiNode = slicer.vtkMRMLDiffusionTensorVolumeNode()
slicer.mrmlScene.AddNode(dtiNode)
dtiNode.SetName('DTI')

# Run estimation
params = {
    'inputVolume': dwiNode.GetID(),
    'outputTensor': dtiNode.GetID(),
    'estimationMethod': 'wls'  # Weighted least squares
}

dtiEstimation = slicer.modules.dtiestimation
slicer.cli.run(dtiEstimation, None, params, wait_for_completion=True)

print("DTI estimation complete")
```

### Generate Scalar Maps

```
1. Modules → Diffusion → DTI

2Scalar Maps
2. Input DTI Volume: Select DTI tensor
3. Output Scalars:
   - FA (Fractional Anisotropy)
   - MD (Mean Diffusivity)
   - AD (Axial Diffusivity)
   - RD (Radial Diffusivity)
   - Color FA
4. Click "Apply"
```

---

## Tractography

### Deterministic Tractography

```
1. Modules → Diffusion → Tractography → Seeding
2. Input:
   - Input Volume: DTI tensor
   - Seeding Region: Select ROI or use whole brain
3. Parameters:
   - Stopping Criteria: FA < 0.15
   - Min Length: 20 mm
   - Max Length: 200 mm
   - Step size: 0.5 mm
4. Output Fiber Bundle: Create new
5. Click "Apply"

Tractography generates fiber bundle (FiberBundleNode)
```

### Python Tractography

```python
import slicer

# Get DTI node
dtiNode = slicer.util.getNode('DTI')

# Create seed ROI
seedNode = slicer.vtkMRMLAnnotationROINode()
slicer.mrmlScene.AddNode(seedNode)
seedNode.SetXYZ(0, 0, 0)
seedNode.SetRadiusXYZ(30, 30, 30)

# Create output fiber bundle
fiberNode = slicer.vtkMRMLFiberBundleNode()
slicer.mrmlScene.AddNode(fiberNode)
fiberNode.SetName('Tracts')

# Run tractography
params = {
    'inputVolume': dtiNode.GetID(),
    'seedingRegion': seedNode.GetID(),
    'outputFiber': fiberNode.GetID(),
    'stoppingCriteriaFA': 0.15,
    'minimumLength': 20,
    'maximumLength': 200,
    'stepSize': 0.5
}

tractography = slicer.modules.tractography
slicer.cli.run(tractography, None, params, wait_for_completion=True)

print(f"Generated {fiberNode.GetNumberOfFibers()} fibers")
```

### UKF Tractography (Multi-Fiber)

```
1. Modules → Diffusion → UKF Tractography
2. Input DWI: Original DWI volume
3. Seeding: Select ROI
4. Parameters:
   - Number of tensors: 2 (for crossing fibers)
   - FA threshold: 0.15
   - Seed spacing: 2 mm
5. Output: Create fiber bundle
6. Click "Apply"

UKF handles crossing fibers better than DTI tractography
```

---

## Interactive Fiber Editing

### ROI-Based Filtering

```
1. Load or create tractography
2. Create ROI (Markups → ROI)
3. Position ROI to select fibers
4. Modules → Diffusion → Tractography Display
5. Select fiber bundle
6. Add ROI as "Positive" (fibers passing through)
   or "Negative" (fibers to exclude)
7. Filtered fibers highlighted in 3D view
```

### Extract Fibers Through ROI

```python
import slicer

# Get fiber bundle and ROI
fiberNode = slicer.util.getNode('Tracts')
roiNode = slicer.util.getNode('ROI')

# Extract fibers
logic = slicer.modules.tractographydisplay.logic()
extractedFibers = logic.ExtractFibersThroughROI(fiberNode, roiNode)

# Save extracted fibers
extractedNode = slicer.vtkMRMLFiberBundleNode()
slicer.mrmlScene.AddNode(extractedNode)
extractedNode.SetName('Extracted_Fibers')
extractedNode.CopyFibers(extractedFibers)

print(f"Extracted {extractedNode.GetNumberOfFibers()} fibers")
```

### Clean and Refine Bundles

```
1. Select fiber bundle in scene
2. Tractography Display module:
   - Adjust length threshold
   - Filter by curvature
   - Remove outliers
3. Save refined bundle
```

---

## Connectivity Analysis

### Generate Connectivity Matrix

```
1. Load FreeSurfer parcellation or create ROI set
2. Load tractography
3. Modules → Diffusion → Connectivity Analysis
4. Input Fiber Bundle: Tractography
5. Input Parcellation: FreeSurfer or atlas
6. Output Matrix: Create new table
7. Click "Apply"

Outputs connectivity matrix (ROI x ROI)
```

### Python Connectivity Matrix

```python
import slicer
import numpy as np

# Get fiber bundle and parcellation
fiberNode = slicer.util.getNode('Tracts')
labelNode = slicer.util.getNode('Parcellation')

# Run connectivity analysis
params = {
    'inputFiberBundle': fiberNode.GetID(),
    'inputParcellation': labelNode.GetID(),
    'outputMatrix': 'ConnectivityMatrix'
}

connectivity = slicer.modules.connectivityanalysis
slicer.cli.run(connectivity, None, params, wait_for_completion=True)

# Load resulting matrix
matrixNode = slicer.util.getNode('ConnectivityMatrix')
matrix = slicer.util.arrayFromModelPolyData(matrixNode)

print(f"Connectivity matrix shape: {matrix.shape}")
```

### Visualize Connectivity

```
1. Load connectivity matrix
2. Modules → Data → Tables
3. View matrix as table or heatmap
4. Export to CSV for external analysis
```

---

## Integration with FreeSurfer

### Load FreeSurfer Parcellation

```
1. Modules → FreeSurfer → FreeSurfer Importer
2. Select FreeSurfer subject directory
3. Import:
   - T1 volume
   - Aparc parcellation
   - White matter surface
4. Use parcellation for connectivity analysis
```

### Python FreeSurfer Integration

```python
import slicer
import os

# FreeSurfer subject directory
fs_dir = '/path/to/freesurfer/subjects/sub-01'

# Load T1
t1_file = os.path.join(fs_dir, 'mri', 'T1.mgz')
t1_node = slicer.util.loadVolume(t1_file)

# Load aparc parcellation
aparc_file = os.path.join(fs_dir, 'mri', 'aparc+aseg.mgz')
parcNode = slicer.util.loadLabelVolume(aparc_file)

# Now use parcellation for connectivity
```

---

## Batch Processing

### Command-Line Tractography

```bash
# Run Slicer in batch mode
Slicer --no-main-window --python-script tractography_script.py

# tractography_script.py:
import slicer

# Load DWI
dwiNode = slicer.util.loadVolume('/data/sub-01/dwi.nii.gz')

# Run DTI estimation
dtiNode = slicer.vtkMRMLDiffusionTensorVolumeNode()
slicer.mrmlScene.AddNode(dtiNode)

params = {
    'inputVolume': dwiNode.GetID(),
    'outputTensor': dtiNode.GetID()
}
slicer.cli.run(slicer.modules.dtiestimation, None, params, wait_for_completion=True)

# Run tractography
# ...

# Save results
slicer.util.saveNode(dtiNode, '/data/sub-01/dti.nrrd')

# Exit
slicer.app.quit()
```

### Batch Script for Multiple Subjects

```python
# batch_slicerdmri.py
import slicer
import sys
import os

def process_subject(subject_dir, output_dir):
    """Process single subject."""

    # Load DWI
    dwi_file = os.path.join(subject_dir, 'dwi.nii.gz')
    dwiNode = slicer.util.loadVolume(dwi_file)

    # DTI estimation
    dtiNode = slicer.vtkMRMLDiffusionTensorVolumeNode()
    slicer.mrmlScene.AddNode(dtiNode)

    params = {'inputVolume': dwiNode.GetID(), 'outputTensor': dtiNode.GetID()}
    slicer.cli.run(slicer.modules.dtiestimation, None, params, wait_for_completion=True)

    # Tractography
    fiberNode = slicer.vtkMRMLFiberBundleNode()
    slicer.mrmlScene.AddNode(fiberNode)

    params = {
        'inputVolume': dtiNode.GetID(),
        'outputFiber': fiberNode.GetID(),
        'stoppingCriteriaFA': 0.15
    }
    slicer.cli.run(slicer.modules.tractography, None, params, wait_for_completion=True)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    slicer.util.saveNode(dtiNode, os.path.join(output_dir, 'dti.nrrd'))
    slicer.util.saveNode(fiberNode, os.path.join(output_dir, 'tracts.vtk'))

    print(f"Processed: {subject_dir}")

# Process all subjects
subjects = ['sub-01', 'sub-02', 'sub-03']
for subject in subjects:
    process_subject(f'/data/{subject}', f'/data/derivatives/slicerdmri/{subject}')

slicer.app.quit()
```

---

## Integration with Claude Code

```python
# slicerdmri_automation.py - Automated SlicerDMRI pipeline

import subprocess
import os
from pathlib import Path

class SlicerDMRIPipeline:
    """Automated SlicerDMRI processing."""

    def __init__(self, slicer_path='/path/to/Slicer'):
        self.slicer_path = slicer_path

    def run_slicer_script(self, script_content, subject_id):
        """Run Python script in Slicer."""

        # Write script to temp file
        script_file = f'/tmp/slicer_script_{subject_id}.py'
        with open(script_file, 'w') as f:
            f.write(script_content)

        # Run Slicer
        cmd = [
            self.slicer_path,
            '--no-main-window',
            '--python-script', script_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Cleanup
        os.remove(script_file)

        return result

    def process_tractography(self, dwi_file, output_dir):
        """Generate tractography for subject."""

        script = f"""
import slicer
import os

# Load DWI
dwiNode = slicer.util.loadVolume('{dwi_file}')

# DTI estimation
dtiNode = slicer.vtkMRMLDiffusionTensorVolumeNode()
slicer.mrmlScene.AddNode(dtiNode)

params = {{'inputVolume': dwiNode.GetID(), 'outputTensor': dtiNode.GetID()}}
slicer.cli.run(slicer.modules.dtiestimation, None, params, wait_for_completion=True)

# Tractography
fiberNode = slicer.vtkMRMLFiberBundleNode()
slicer.mrmlScene.AddNode(fiberNode)

params = {{
    'inputVolume': dtiNode.GetID(),
    'outputFiber': fiberNode.GetID(),
    'stoppingCriteriaFA': 0.15
}}
slicer.cli.run(slicer.modules.tractography, None, params, wait_for_completion=True)

# Save
os.makedirs('{output_dir}', exist_ok=True)
slicer.util.saveNode(dtiNode, '{output_dir}/dti.nrrd')
slicer.util.saveNode(fiberNode, '{output_dir}/tracts.vtk')

slicer.app.quit()
"""

        subject_id = Path(dwi_file).parent.parent.name
        result = self.run_slicer_script(script, subject_id)

        if result.returncode == 0:
            print(f"Success: {subject_id}")
        else:
            print(f"Error: {result.stderr}")

# Usage
pipeline = SlicerDMRIPipeline(slicer_path='/Applications/Slicer.app/Contents/MacOS/Slicer')

pipeline.process_tractography(
    dwi_file='/data/sub-01/dwi/dwi.nii.gz',
    output_dir='/data/derivatives/slicerdmri/sub-01'
)
```

---

## Troubleshooting

### Problem 1: Extension Not Installing

**Symptoms:** SlicerDMRI not appearing in Extension Manager

**Solution:**
```
1. Check Slicer version (need recent stable)
2. Restart Slicer after install
3. Check Extensions → Manage Extensions
4. Manual install: Download from GitHub and add manually
```

### Problem 2: Tractography Generates No Fibers

**Symptoms:** Empty fiber bundle

**Solution:**
```
1. Check FA threshold (may be too high)
2. Verify seed ROI placement
3. Check DTI quality
4. Reduce stopping criteria
5. Increase max length
```

### Problem 3: Slicer Crashes During Processing

**Symptoms:** Slicer closes unexpectedly

**Solution:**
```
1. Increase memory allocation
2. Process smaller ROIs
3. Reduce number of seeds
4. Check data quality and format
```

---

## Best Practices

### 1. Data Quality

- Use preprocessed DWI data
- Ensure proper gradient table
- Quality control before tractography
- Sufficient SNR for reliable tracking

### 2. Tractography Parameters

- Start with conservative FA threshold (0.15-0.2)
- Adjust based on tissue and goals
- Use multiple ROIs for specific bundles
- Validate tracts anatomically

### 3. Clinical Workflow

- Load all modalities (T1, fMRI, etc.)
- Register to common space
- Use anatomical landmarks for guidance
- Document ROI placement

### 4. Reproducibility

- Save Slicer scene (.mrml)
- Record all parameters
- Script workflows when possible
- Version SlicerDMRI extension

---

## Resources

### Official Documentation

- **Website:** https://dmri.slicer.org/
- **Documentation:** https://dmri.slicer.org/docs/
- **GitHub:** https://github.com/SlicerDMRI/SlicerDMRI
- **3D Slicer:** https://www.slicer.org/

### Tutorials

- **SlicerDMRI Tutorials:** https://dmri.slicer.org/tutorials/
- **3D Slicer Tutorials:** https://www.slicer.org/wiki/Documentation/Nightly/Training

### Community

- **Slicer Forum:** https://discourse.slicer.org/
- **GitHub Issues:** https://github.com/SlicerDMRI/SlicerDMRI/issues

---

## Citation

```bibtex
@article{norton2017slicerdmri,
  title={SlicerDMRI: Open source diffusion MRI software for brain cancer research},
  author={Norton, Isaiah and Essayed, Walid I and Zhang, Fan and Pujol, Sonia and Yarmarkovich, Alex and Golby, Alexandra J and Kindlmann, Gordon and Wassermann, Demian and Estepar, Raul San Jose and Rathi, Yogesh and others},
  journal={Cancer research},
  volume={77},
  number={21},
  pages={e101--e103},
  year={2017},
  publisher={AACR}
}
```

---

## Related Tools

- **3D Slicer:** Platform (visualization, registration)
- **DIPY:** Python diffusion library (see `dipy.md`)
- **MRtrix3:** Command-line diffusion tools (see `mrtrix3.md`)
- **TractoFlow:** Automated pipeline (see `tractoflow.md`)
- **DSI Studio:** Alternative GUI tool (see `dsistudio.md`)
- **FreeSurfer:** Parcellation provider (see `freesurfer.md`)

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**SlicerDMRI Version Covered:** Latest
**Maintainer:** Claude Code Neuroimaging Skills
