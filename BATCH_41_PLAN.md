# Batch 41 Plan: Legacy/Commercial Viewers

## Overview

**Theme:** Legacy/Commercial Viewers
**Focus:** Specialized medical imaging viewers for clinical and research applications
**Target:** 2 skills, 1,000-1,100 lines

**Current Progress:** 127/133 skills (95.5%)
**After Batch 40:** 127/133 skills (95.5%)
**After Batch 41:** 129/133 skills (97.0%)

This batch addresses specialized visualization tools that serve niche but important roles in neuroimaging: OsiriX for clinical DICOM viewing and Connectome Viewer for legacy network visualization. While both are older or platform-specific tools, they remain relevant for certain workflows and provide historical context for neuroimaging visualization development.

## Rationale

Visualization tools have evolved significantly over the past two decades:

- **Clinical Viewers:** OsiriX pioneered modern DICOM viewing on macOS
- **Research Viewers:** Connectome Viewer bridged early network visualization needs
- **Legacy Support:** Many datasets and workflows still reference these tools
- **Historical Context:** Understanding tool evolution aids method selection
- **Platform-Specific Solutions:** macOS users often prefer native applications

This batch provides documentation for tools that, while not cutting-edge, continue to serve specific user communities and offer valuable lessons in neuroimaging visualization.

## Skills to Create

### 1. OsiriX (500-550 lines, 18-20 examples)

**Overview:**
OsiriX is a macOS-native DICOM image viewer and processing platform widely used in clinical radiology and medical research. Originally open-source (now commercial as OsiriX MD), it provides advanced 2D/3D/4D visualization, multi-planar reconstruction (MPR), maximum intensity projection (MIP), volume rendering, and extensive plugin architecture. OsiriX excels at handling large DICOM datasets, supporting PACS integration, and offering an intuitive interface for radiologists and clinicians. The open-source fork Horos continues the original OsiriX development model.

**Key Features:**
- Native macOS application with Metal acceleration
- DICOM standard compliance and PACS integration
- 2D viewer with MPR (multi-planar reconstruction)
- 3D/4D volume rendering (MIP, VR, surface rendering)
- ROI tools and quantitative measurements
- Image fusion and registration
- Extensive plugin ecosystem
- DICOM send/receive (C-STORE, C-FIND, C-MOVE)
- Report generation and annotation
- Support for all imaging modalities (CT, MRI, PET, ultrasound)
- Horos as open-source alternative

**Target Audience:**
- Clinical radiologists and neurologists
- Medical researchers using macOS
- Hospitals with macOS workstations
- Users requiring PACS integration
- Researchers needing DICOM viewing without conversion

**Main Sections:**

1. **Overview** (40 lines)
   - Introduction to OsiriX and Horos
   - History (open-source to commercial transition)
   - Clinical vs. research applications
   - OsiriX MD vs. Horos comparison
   - Citation information

2. **Installation** (60 lines)
   - OsiriX MD (commercial, macOS only)
   - Horos (open-source, macOS only)
   - System requirements
   - DICOM database setup
   - PACS configuration

3. **DICOM Management** (70 lines, 2-3 examples)
   - Importing DICOM files
   - Database organization
   - PACS query/retrieve
   - DICOM send/export
   - Example: Import study from PACS

4. **2D Viewing and MPR** (80 lines, 3-4 examples)
   - Basic 2D navigation
   - Window/level adjustment
   - Multi-planar reconstruction
   - Synchronized viewing
   - Example: MPR brain imaging

5. **3D/4D Visualization** (90 lines, 3-4 examples)
   - Volume rendering
   - Maximum intensity projection (MIP)
   - Surface rendering
   - 4D time-series viewing
   - Example: 3D brain rendering
   - Example: 4D fMRI viewing

6. **ROI and Measurements** (70 lines, 2-3 examples)
   - Drawing ROIs (circle, polygon, freehand)
   - Volume measurements
   - Intensity statistics
   - Length/angle measurements
   - Example: Tumor volume quantification

7. **Image Processing** (60 lines, 2-3 examples)
   - Image fusion (overlay PET on MRI)
   - Registration tools
   - Filtering and enhancement
   - Example: PET/MRI fusion

8. **Plugins and Extensions** (50 lines, 1-2 examples)
   - Plugin manager
   - Popular plugins (Nifti export, advanced segmentation)
   - Developing custom plugins

9. **Export and Interoperability** (50 lines, 1-2 examples)
   - Export to NIfTI, JPEG, TIFF
   - Integration with neuroimaging pipelines
   - DICOM anonymization

10. **Horos Differences** (40 lines)
    - Feature comparison with OsiriX
    - Open-source advantages
    - Community development

11. **Troubleshooting** (40 lines)
    - DICOM import issues
    - PACS connection problems
    - Performance optimization
    - Metal rendering issues

12. **Best Practices** (30 lines)
    - Database management
    - DICOM workflow optimization
    - Clinical vs. research usage
    - Privacy and anonymization

13. **References** (20 lines)
    - OsiriX publications
    - DICOM standards
    - Radiology imaging

**Code Examples:**
- DICOM import (GUI walkthrough)
- MPR setup (screenshots)
- 3D rendering (configuration)
- ROI measurement (step-by-step)
- Plugin installation (instructions)

**Integration Points:**
- PACS systems for clinical imaging
- DICOM converters (dcm2niix)
- FreeSurfer for anatomical overlay
- Export to FSL/SPM via NIfTI conversion

---

### 2. Connectome Viewer (500-550 lines, 18-20 examples)

**Overview:**
Connectome Viewer is a visualization and analysis platform for brain connectivity networks, developed as part of the Connectome Mapping Toolkit (CMTK). It provides interactive 3D visualization of structural and functional connectivity matrices alongside cortical surfaces and white matter tracts. Originally developed to support the Human Connectome Project data formats, Connectome Viewer enables exploration of network graphs, tract-based connectivity, and multimodal neuroimaging data integration. While development has slowed (legacy status), it pioneered many visualization approaches now standard in modern tools like Connectome Workbench.

**Key Features:**
- Interactive 3D network visualization
- Integration with FreeSurfer surfaces
- DTI tractography visualization
- Connectivity matrix exploration
- Graph theory metrics (degree, clustering, path length)
- Multi-scale network analysis (voxel, ROI, atlas levels)
- Integration with TrackVis fiber formats
- Python scripting interface
- CFF (Connectome File Format) support
- Annotation and region labeling
- Legacy support for early connectome studies

**Target Audience:**
- Researchers analyzing legacy connectome datasets
- Users of Connectome Mapping Toolkit (CMTK)
- Educators teaching network neuroscience history
- Researchers maintaining older analysis pipelines
- Users transitioning to modern tools (Workbench, ConnectomeViewer2)

**Main Sections:**

1. **Overview** (40 lines)
   - Introduction to Connectome Viewer
   - History and legacy status
   - Relation to CMTK and HCP
   - Modern alternatives (Connectome Workbench)
   - Citation information

2. **Installation** (70 lines)
   - Python 2.7 dependency (legacy)
   - Required packages (TraitsUI, Mayavi, NetworkX)
   - Docker containerization (recommended for modern systems)
   - Testing installation
   - Platform compatibility (Linux, macOS, Windows)

3. **Data Formats** (70 lines, 2-3 examples)
   - CFF (Connectome File Format)
   - FreeSurfer surface formats
   - TrackVis (.trk) fiber files
   - Connectivity matrices (GraphML, NumPy)
   - Example: Loading CFF file

4. **Surface Visualization** (80 lines, 3-4 examples)
   - Loading FreeSurfer surfaces
   - Cortical parcellations
   - Overlay activation maps
   - Surface colormapping
   - Example: Visualize Desikan-Killiany atlas

5. **Tractography Visualization** (80 lines, 3-4 examples)
   - Loading .trk files from TrackVis
   - Fiber coloring and filtering
   - Tract density maps
   - Integration with surfaces
   - Example: Display major white matter tracts

6. **Network Visualization** (90 lines, 3-4 examples)
   - Connectivity matrix as network graph
   - Node positioning (anatomical coordinates)
   - Edge thickness by connection strength
   - Interactive graph manipulation
   - Example: Structural connectivity network
   - Example: Functional connectivity network

7. **Graph Theory Analysis** (70 lines, 2-3 examples)
   - Computing network metrics (degree, clustering)
   - Shortest path analysis
   - Community detection
   - Exporting metrics to CSV
   - Example: Compute hub regions

8. **Multi-Modal Integration** (60 lines, 2-3 examples)
   - Combining structural and functional connectivity
   - Overlay fMRI activation on networks
   - DTI metrics on tract endpoints
   - Example: Structure-function correspondence

9. **Python Scripting** (60 lines, 2-3 examples)
   - Programmatic data loading
   - Automated visualization
   - Batch processing networks
   - Example: Script to load and visualize connectome

10. **CFF File Format** (50 lines, 1-2 examples)
    - Creating CFF files
    - CFF structure and contents
    - Conversion from other formats

11. **Migration to Modern Tools** (50 lines)
    - Transitioning to Connectome Workbench
    - Exporting data for modern viewers
    - Alternative tools (nilearn, BrainNetViewer)

12. **Troubleshooting** (40 lines)
    - Python 2.7 compatibility issues
    - Mayavi rendering problems
    - File format conversion errors
    - Docker workarounds

13. **Best Practices** (30 lines)
    - When to use Connectome Viewer vs. alternatives
    - Legacy data handling
    - Citation and reproducibility

14. **References** (20 lines)
    - CMTK publications
    - Connectome Viewer papers
    - CFF format specifications

**Code Examples:**
- Load CFF file (Python)
- Visualize surface (Python)
- Display tractography (Python)
- Network graph (Python)
- Graph metrics (Python)

**Integration Points:**
- FreeSurfer for surface generation
- TrackVis for tractography
- CMTK for connectome mapping
- NetworkX for graph analysis
- Migration to Connectome Workbench

---

## Implementation Checklist

### Per-Skill Requirements
- [ ] 500-550 lines per skill
- [ ] 18-20 code examples per skill
- [ ] Consistent section structure
- [ ] Installation instructions
- [ ] Basic usage examples
- [ ] Advanced features
- [ ] Integration examples
- [ ] Troubleshooting section
- [ ] Best practices
- [ ] References with citations

### Quality Assurance
- [ ] All examples accurate
- [ ] Screenshots/GUI walkthroughs where appropriate
- [ ] Legacy status clearly indicated
- [ ] Migration paths to modern tools
- [ ] Clear explanations
- [ ] Common issues covered
- [ ] Complete references

### Batch Requirements
- [ ] Total lines: 1,000-1,100
- [ ] Total examples: 36-40
- [ ] Consistent markdown formatting
- [ ] Cross-referencing
- [ ] Visualization tools focus

## Timeline

1. **OsiriX**: 500-550 lines, 18-20 examples
2. **Connectome Viewer**: 500-550 lines, 18-20 examples

**Estimated Total:** 1,000-1,100 lines, 36-40 examples

## Context & Connections

### Visualization Evolution

**Clinical Viewing (OsiriX):**
```
DICOM Images → OsiriX/Horos → 2D/3D Viewing → Clinical Interpretation
       ↓              ↓             ↓                  ↓
    PACS          macOS GUI    Measurements      Radiology Reports
```

**Network Visualization (Connectome Viewer):**
```
Connectome Data → CFF Format → Connectome Viewer → Network Exploration
       ↓               ↓              ↓                    ↓
  DTI + fMRI      Surfaces      3D Networks          Graph Metrics
                  + Tracts
```

### Complementary Tools

**Already Covered:**
- **Connectome Workbench**: Modern replacement for Connectome Viewer
- **FSLeyes**: Alternative DICOM/NIfTI viewer (cross-platform)
- **FreeSurfer**: Surface generation for both tools
- **TrackVis**: Tractography for Connectome Viewer

**New Capabilities:**
- **OsiriX**: macOS-native clinical DICOM viewing
- **Connectome Viewer**: Legacy connectome visualization

### Migration Paths

**From OsiriX:**
- Export to NIfTI → FSLeyes, ITK-SNAP, or other viewers
- DICOM → dcm2niix → neuroimaging pipelines

**From Connectome Viewer:**
- Migrate to Connectome Workbench for HCP data
- Use nilearn for connectivity visualization in Python
- BrainNet Viewer for network visualization

## Expected Impact

### Research Community
- Support legacy dataset analysis
- Historical context for tool evolution
- Transition guidance to modern alternatives

### Clinical Applications
- macOS-based radiology workflows (OsiriX)
- DICOM viewing without platform constraints
- Integration with hospital PACS systems

### Education
- Understanding DICOM standards and clinical imaging
- Evolution of connectome visualization
- Comparison of legacy vs. modern approaches

## Conclusion

Batch 41 addresses legacy and platform-specific visualization tools:

1. **OsiriX** provides clinical-grade DICOM viewing for macOS users
2. **Connectome Viewer** documents early connectome visualization approaches

By completing this batch, the N_tools collection will reach **129/133 skills (97.0%)**, with comprehensive coverage extending to specialized and legacy visualization platforms.

These tools represent important historical milestones in neuroimaging visualization and continue to serve niche user communities, particularly in clinical radiology (OsiriX) and legacy connectome research (Connectome Viewer).
