# Batch 19: Visualization & Rendering Tools - Planning Document

## Overview

**Batch Theme:** Visualization & Rendering Tools
**Batch Number:** 19
**Number of Skills:** 4
**Current Progress:** 68/133 skills completed (51.1%)
**After Batch 19:** 72/133 skills (54.1%)

## Rationale

Batch 19 focuses on advanced **visualization and rendering tools** for neuroimaging data. While we've covered basic visualization capabilities within processing pipelines (FSLeyes, ITK-SNAP, Connectome Workbench, BrainNet Viewer), this batch addresses specialized tools for:

- **Surface and volume rendering** with advanced graphics capabilities
- **Web-based interactive visualization** for presentations and publications
- **3D brain rendering** for publication-quality figures
- **Multi-modal image viewing** with extensive format support

These tools are essential for:
- **Scientific communication:** Creating publication-quality figures and videos
- **Interactive exploration:** Web-based visualizations for data sharing
- **Quality control:** Advanced visual inspection of processing results
- **Presentation:** Professional brain renderings for talks and papers
- **Atlas visualization:** Mapping data to 3D brain structures

## Tools in This Batch

### 1. Surfice
**Website:** https://www.nitrc.org/projects/surfice/
**Platform:** Windows/macOS/Linux
**Language:** Pascal/OpenGL
**Priority:** High

**Overview:**
Surfice is a powerful surface and volume visualization tool developed by Chris Rorden. It provides high-performance rendering of brain surfaces, tractography, and volumetric data using modern OpenGL. Surfice is particularly popular for creating publication-quality figures and videos with smooth surfaces, transparent overlays, and advanced lighting.

**Key Capabilities:**
- Surface mesh rendering (FreeSurfer, GIfTI, OBJ, PLY formats)
- Volume rendering with ray-casting
- Tractography visualization (TRK, TCK formats)
- Lesion mapping and overlap visualization
- Multi-modal overlay (statistical maps, parcellations)
- Scripting support (Python-like syntax)
- High-quality screenshot and video export
- Custom color maps and lighting
- Cross-platform GPU acceleration

**Target Audience:**
- Researchers creating publication figures
- Lesion-symptom mapping studies
- Tractography visualization
- Surface-based statistical mapping
- Clinical presentation of results

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**
1. Installation (binary downloads for all platforms)
2. Basic surface rendering (pial, white matter, inflated)
3. Volume overlay on surfaces (statistical maps, lesions)
4. Tractography visualization with surfaces
5. Lesion overlap and subtraction mapping
6. Scripting with Python interface
7. Custom colormaps and transparency
8. Camera positioning and lighting
9. Export high-resolution images and videos
10. Batch processing multiple subjects
11. Integration with FreeSurfer outputs
12. Integration with MRtrix3 tractograms
13. Multi-panel figure creation
14. Command-line usage for automation

**Example Workflows:**
- Render FreeSurfer pial surface with activation overlay
- Visualize fiber bundles on transparent brain
- Create lesion overlap map across subjects
- Generate rotating brain video for presentation
- Multi-subject composite figure

**Integration Points:**
- FreeSurfer surfaces
- MRtrix3 tractography
- FSL statistical maps
- SPM results
- Custom meshes and volumes

---

### 2. Mango (Multi-image Analysis GUI)
**Website:** http://ric.uthscsa.edu/mango/
**Platform:** Windows/macOS/Linux
**Language:** Java
**Priority:** Medium-High

**Overview:**
Mango is a versatile medical image viewer developed at the Research Imaging Institute (RIT). It provides comprehensive viewing capabilities for a wide range of neuroimaging formats, with tools for ROI drawing, image registration, surface rendering, and plugin extensibility. Mango is particularly useful for manual quality control, ROI definition, and multi-modal image comparison.

**Key Capabilities:**
- Multi-format support (NIFTI, DICOM, ANALYZE, MINC, etc.)
- 3D volume rendering
- Multi-planar reformatting (orthogonal, oblique views)
- ROI drawing and editing tools
- Image registration and fusion
- Surface visualization
- Built-in tools (histograms, coordinates, measurements)
- Plugin architecture for extensions
- Scripting support (JavaScript, Jython)
- Cross-platform Java-based GUI

**Target Audience:**
- Clinical researchers
- Manual ROI definition
- Multi-modal image comparison
- Quality control workflows
- DICOM to NIFTI conversion
- Basic image processing

**Estimated Lines:** 650-700
**Estimated Code Examples:** 25-28

**Key Topics to Cover:**
1. Installation (Java-based, cross-platform)
2. Loading and viewing images (multiple formats)
3. Multi-planar visualization (axial, sagittal, coronal)
4. ROI drawing and management
5. Image registration and overlay
6. Surface extraction and rendering
7. Coordinate system handling
8. Image intensity windowing and LUTs
9. Measurement tools (distance, angles, volumes)
10. DICOM handling and conversion
11. Batch processing with scripts
12. Plugin development
13. Export formats (images, movies, ROIs)
14. Integration with other tools

**Example Workflows:**
- Manual hippocampus segmentation
- QC multi-modal registration
- Extract lesion ROIs from clinical scans
- Compare pre/post treatment scans
- Create teaching materials with annotations

**Integration Points:**
- DICOM from scanners
- FSL/SPM/ANTs outputs
- FreeSurfer volumes
- Manual ROIs to analysis pipelines

---

### 3. PyCortex
**Website:** https://gallantlab.github.io/pycortex/
**Platform:** Python/Web (all OS)
**Language:** Python/JavaScript (WebGL)
**Priority:** High

**Overview:**
PyCortex is a Python library for interactive web-based visualization of cortical surface data. Developed by the Gallant Lab at UC Berkeley, it creates stunning interactive visualizations that can be embedded in Jupyter notebooks, web pages, or standalone HTML files. PyCortex is particularly powerful for visualizing fMRI results, retinotopic maps, and any cortical surface data with full interactivity (rotation, zooming, layer selection).

**Key Capabilities:**
- Web-based interactive 3D brain visualization
- FreeSurfer and custom surface support
- Flattened cortex visualization
- Volume-to-surface projection
- Interactive data exploration (overlays, crosshairs, transparency)
- Multiple views (inflated, fiducial, flat, sphere)
- Custom colormaps and overlays
- Jupyter notebook integration
- Standalone HTML export (shareable visualizations)
- Subject database management
- Retinotopic mapping tools

**Target Audience:**
- fMRI researchers
- Retinotopy and visual neuroscience
- Interactive data exploration
- Web-based result sharing
- Publication supplements
- Teaching and outreach

**Estimated Lines:** 750-800
**Estimated Code Examples:** 30-35

**Key Topics to Cover:**
1. Installation (pip install, dependencies)
2. Subject database setup (FreeSurfer import)
3. Loading cortical surfaces
4. Volume-to-surface mapping
5. Creating 2D flat maps
6. Vertex data visualization
7. Interactive webshow in notebooks
8. Custom colormaps and transparency
9. Multiple overlay layers
10. ROI definition and visualization
11. Retinotopic mapping tools
12. Export standalone HTML visualizations
13. Screenshot and video capture
14. Advanced WebGL customization
15. Integration with Nilearn and NiBabel
16. Group analysis visualization

**Example Workflows:**
- Visualize fMRI activation on inflated cortex
- Create interactive retinotopic map
- Share web-based results with collaborators
- Flat map visualization for publications
- Multi-subject overlay comparison

**Integration Points:**
- FreeSurfer subjects
- Nilearn statistical maps
- fMRIPrep surface outputs
- Custom vertex data from analyses

---

### 4. Brainrender
**Website:** https://github.com/brainglobe/brainrender
**Platform:** Python (all OS)
**Language:** Python (VTK/vedo)
**Priority:** High

**Overview:**
Brainrender is a modern Python package for creating high-quality 3D renderings of brain anatomy, regions, and data. Part of the BrainGlobe initiative, it provides a programmatic interface for generating publication-quality figures and animations with anatomical atlases, connectivity data, and custom meshes. Brainrender uses VTK/vedo for rendering and supports multiple species and atlases.

**Key Capabilities:**
- Atlas-based brain rendering (Allen Mouse, Human, etc.)
- Region and structure visualization
- Connectivity and tractography rendering
- Cell and neuron morphology
- Injection site and projection visualization
- Programmatic scene composition
- Publication-quality rendering
- Video and animation export
- Jupyter notebook integration
- Custom mesh and data overlay
- Cross-species atlas support

**Target Audience:**
- Systems neuroscience researchers
- Connectivity and circuit studies
- Atlas-based analysis visualization
- Multi-species comparative studies
- Publication figure creation
- Programmatic visualization workflows

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**
1. Installation (pip install brainrender)
2. Atlas selection and loading
3. Basic scene creation and rendering
4. Brain region visualization
5. Connectivity and tractography
6. Cell and neuron morphology rendering
7. Custom mesh import
8. Data overlay on atlas regions
9. Camera positioning and lighting
10. Color schemes and transparency
11. Screenshots and video export
12. Jupyter notebook integration
13. Programmatic animation creation
14. Multi-panel figure composition
15. Integration with atlasAPI
16. Cross-species visualization

**Example Workflows:**
- Render specific brain regions from Allen atlas
- Visualize connectivity between regions
- Create rotating brain animation
- Overlay experimental data on atlas
- Multi-panel anatomical figure

**Integration Points:**
- BrainGlobe atlases
- Allen Brain Atlas
- Connectivity databases
- Custom mesh formats (OBJ, STL, VTK)
- Tractography from various sources

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **Surfice** (most widely used for publication figures)
   - **PyCortex** (Python-based, high priority for fMRI)
   - **Brainrender** (modern Python tool, growing adoption)
   - **Mango** (utility viewer, broad format support)

2. **Comprehensive Coverage:**
   - Each skill: 650-800 lines
   - 25-35 code examples per skill
   - Real-world visualization workflows
   - Integration with processing pipelines

3. **Consistent Structure:**
   - Overview and key features
   - Installation (all platforms)
   - Basic visualization examples
   - Advanced rendering techniques
   - Scripting and automation
   - Export and sharing
   - Integration with other tools
   - Troubleshooting
   - Best practices
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Cross-platform installation
   - Dependency management
   - Environment setup

2. **Basic Visualization** (5-7)
   - Loading data
   - Simple rendering
   - View manipulation
   - Color mapping

3. **Advanced Rendering** (6-8)
   - Multi-modal overlays
   - Transparency and lighting
   - Camera control
   - Custom visualizations

4. **Scripting & Automation** (4-6)
   - Batch processing
   - Programmatic control
   - Parameter sweeps
   - Reproducible figures

5. **Export & Sharing** (3-4)
   - High-resolution images
   - Video creation
   - Web export (PyCortex)
   - Multi-panel figures

6. **Integration Examples** (4-6)
   - FreeSurfer integration
   - fMRIPrep outputs
   - Statistical map overlays
   - Atlas integration

7. **Real-World Workflows** (3-5)
   - Publication figure creation
   - Quality control visualization
   - Data exploration
   - Result presentation

### Cross-Tool Integration

All skills will demonstrate integration with:
- **FreeSurfer:** Surface and volume inputs
- **fMRIPrep/QSIPrep:** Preprocessed outputs
- **FSL/SPM/ANTs:** Statistical maps
- **MRtrix3:** Tractography (Surfice, Brainrender)
- **Nilearn:** Python-based analysis (PyCortex, Brainrender)
- **Claude Code:** Automated visualization pipelines

### Quality Targets

- **Minimum lines per skill:** 650
- **Target lines per skill:** 700-800
- **Minimum code examples:** 25
- **Target code examples:** 28-35
- **Total batch lines:** ~2,800-3,000
- **Total code examples:** ~110-125

---

## Batch Statistics

### Projected Totals

| Tool | Est. Lines | Est. Examples | Priority |
|------|-----------|---------------|----------|
| Surfice | 700-750 | 28-32 | High |
| Mango | 650-700 | 25-28 | Medium-High |
| PyCortex | 750-800 | 30-35 | High |
| Brainrender | 700-750 | 28-32 | High |
| **TOTAL** | **2,800-3,000** | **111-127** | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Surface visualization (Surfice, PyCortex)
- ✓ Volume rendering (Surfice, Mango)
- ✓ Web-based visualization (PyCortex)
- ✓ Programmatic rendering (PyCortex, Brainrender)
- ✓ Atlas-based visualization (Brainrender)
- ✓ Multi-format viewing (Mango)
- ✓ Tractography rendering (Surfice, Brainrender)
- ✓ Interactive exploration (PyCortex, Brainrender)

**Language Coverage:**
- Python: PyCortex, Brainrender (2/4)
- Native/OpenGL: Surfice (1/4)
- Java: Mango (1/4)

**Platform Coverage:**
- Cross-platform: All 4 tools
- Web-based: PyCortex
- Desktop GUI: Surfice, Mango
- Python library: PyCortex, Brainrender

---

## Strategic Importance

### Fills Critical Gap

Previous batches focused on:
- Data processing and analysis pipelines
- Statistical modeling and machine learning
- Quality control and preprocessing
- Workflow management

**Batch 19 adds:**
- Advanced visualization and rendering
- Publication-quality figure creation
- Interactive data exploration
- Web-based result sharing

### Complementary Skills

**Works with existing skills:**
- **FreeSurfer:** Provides surfaces for visualization
- **fMRIPrep:** Outputs for cortical visualization
- **FSL/SPM:** Statistical maps to overlay
- **MRtrix3:** Tractography for 3D rendering
- **Nilearn:** Analysis results to visualize
- **Connectome Workbench:** Alternative visualization approaches

### User Benefits

1. **Publication Workflow:**
   - Create figures with Surfice/Brainrender
   - Interactive supplements with PyCortex
   - QC visualizations with Mango

2. **Data Exploration:**
   - Interactive web visualizations
   - Programmatic visualization pipelines
   - Multi-modal comparison

3. **Communication:**
   - Professional renderings for presentations
   - Shareable web visualizations
   - Educational materials

4. **Automation:**
   - Scripted figure generation
   - Batch visualization
   - Reproducible plots

---

## Dependencies and Prerequisites

### Software Prerequisites

**Surfice:**
- OpenGL 3.3+ capable GPU
- No other dependencies (standalone)

**Mango:**
- Java Runtime Environment (JRE) 8+
- No other dependencies

**PyCortex:**
- Python 3.7+
- NumPy, SciPy, matplotlib
- NiBabel
- Tornado (web server)
- Pillow (image processing)
- Optional: Mayavi for 3D

**Brainrender:**
- Python 3.7+
- vedo (VTK wrapper)
- VTK 9.0+
- NumPy, pandas
- BrainGlobe atlasAPI
- Optional: napari for interactive viewing

### Data Prerequisites

**Common to all:**
- Brain surface meshes (GIfTI, FreeSurfer, OBJ)
- Volumetric images (NIFTI, ANALYZE)
- Statistical maps and overlays
- Atlases and parcellations

**Tool-specific:**
- **Surfice:** TRK/TCK tractography files
- **Mango:** DICOM support for clinical data
- **PyCortex:** FreeSurfer subject directory
- **Brainrender:** BrainGlobe atlas data

### Knowledge Prerequisites

Users should understand:
- Basic neuroimaging concepts
- Coordinate systems and spaces
- Surface vs. volume representations
- Statistical mapping basics
- Python (for PyCortex, Brainrender)

---

## Learning Outcomes

After completing Batch 19 skills, users will be able to:

1. **Create Publication Figures:**
   - Generate high-quality brain renderings
   - Overlay statistical maps on surfaces
   - Create multi-panel composite figures
   - Export publication-ready images

2. **Interactive Visualization:**
   - Build web-based brain viewers
   - Share interactive results
   - Explore data in 3D
   - Embed visualizations in notebooks

3. **Programmatic Workflows:**
   - Automate figure generation
   - Batch process visualizations
   - Script reproducible plots
   - Integrate with analysis pipelines

4. **Multi-Modal Visualization:**
   - Combine surfaces and volumes
   - Overlay tractography
   - Visualize atlas regions
   - Compare across modalities

5. **Data Sharing:**
   - Export standalone HTML visualizations
   - Create videos and animations
   - Generate shareable figures
   - Build interactive supplements

---

## Relationship to Existing Skills

### Builds Upon:
- **FreeSurfer** (Batch 1): Provides surface meshes
- **FSL** (Batch 1): Statistical maps to visualize
- **SPM** (Batch 1): Volumetric results
- **fMRIPrep** (Batch 4): Preprocessed functional data
- **MRtrix3** (Batch 2): Tractography files
- **Nilearn** (Batch 2): Python-based analysis
- **Connectome Workbench** (Batch 11): Alternative surface visualization

### Complements:
- **ITK-SNAP** (Batch 13): Segmentation and manual editing
- **FSLeyes** (Batch 13): FSL-specific visualization
- **BrainNet Viewer** (Batch 12): Network visualization
- **MRIcron** (Batch 13): Basic volume viewing

### Enables:
- Professional scientific communication
- Interactive data exploration
- Publication-quality figures
- Web-based result sharing
- Automated visualization pipelines

---

## Expected Challenges and Solutions

### Challenge 1: Platform-Specific Installation
**Issue:** Different installation methods across tools
**Solution:** Provide detailed platform-specific instructions with troubleshooting

### Challenge 2: Graphics Requirements
**Issue:** GPU/OpenGL requirements (Surfice)
**Solution:** Document minimum requirements, provide fallback options

### Challenge 3: Python Environment Management
**Issue:** Dependency conflicts (PyCortex, Brainrender)
**Solution:** Recommend conda environments, provide environment.yml files

### Challenge 4: Data Format Complexity
**Issue:** Different mesh and volume formats across tools
**Solution:** Include comprehensive format conversion examples

### Challenge 5: Scripting vs. GUI
**Issue:** Some tools GUI-focused (Mango), others Python-only (PyCortex)
**Solution:** Provide both interactive and scripted workflows where applicable

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Test commands to verify successful installation
   - Version checking
   - Dependency validation

2. **Basic Functionality Tests:**
   - Load sample data
   - Create simple visualization
   - Export output

3. **Integration Tests:**
   - Load FreeSurfer surfaces
   - Overlay statistical maps
   - Combine with other tool outputs

4. **Example Data:**
   - Links to test datasets
   - Sample visualization scripts
   - Expected outputs

---

## Timeline Estimate

**Per Skill:**
- Research and planning: 15-20 min
- Writing and examples: 40-60 min
- Review and refinement: 10-15 min
- **Total per skill:** ~65-95 min

**Total Batch 19:**
- 4 skills × 80 min average = ~320 min (~5-6 hours)
- Includes documentation, examples, and testing

**Can be completed in:** 1-2 extended sessions

---

## Success Criteria

Batch 19 will be considered successful when:

✓ All 4 skills created with 650-800 lines each
✓ Total of 110+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - Basic to advanced visualization examples
  - Integration with major tools (FreeSurfer, fMRIPrep, etc.)
  - Scripting and automation examples
  - Publication-quality figure workflows
  - Troubleshooting section
  - Best practices
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 72/133 (54.1%)

---

## Next Batches Preview

### Batch 20: Advanced Diffusion Modeling
- DMIPY (diffusion microstructure)
- MDT (microstructure diffusion toolbox)
- Recobundles (fiber bundle recognition)
- SlicerDMRI (3D Slicer diffusion module)

### Batch 21: PET Imaging & Reconstruction
- AMIDE (medical imaging data examiner)
- NiftyPET (PET reconstruction and processing)
- SIRF (synergistic image reconstruction framework)
- STIR (software for tomographic image reconstruction)

### Batch 22: Multimodal Gradients & Transcriptomics
- BrainSpace (gradient analysis)
- neuromaps (brain annotations)
- abagen (Allen Brain Atlas genetics)
- BrainStat (statistical analysis)

---

## Conclusion

Batch 19 represents a strategic shift toward **visualization and communication** of neuroimaging results. While previous batches focused on data acquisition, processing, and analysis, Batch 19 enables researchers to:

- **Communicate findings** through publication-quality figures
- **Share results** via interactive web visualizations
- **Explore data** with advanced 3D rendering
- **Automate visualization** within analysis pipelines

These tools are essential for the final steps of the research workflow: presentation, publication, and dissemination of results. By covering Surfice, Mango, PyCortex, and Brainrender, we provide comprehensive visualization capabilities across different use cases, programming interfaces, and output formats.

**Status After Batch 19:** 72/133 skills (54.1% complete)

---

**Document Version:** 1.0
**Created:** 2025-11-15
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 4 skills, ~2,800-3,000 lines, ~110-125 examples
