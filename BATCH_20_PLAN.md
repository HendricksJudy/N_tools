# Batch 20: Advanced Diffusion Modeling - Planning Document

## Overview

**Batch Theme:** Advanced Diffusion Modeling & Microstructure
**Batch Number:** 20
**Number of Skills:** 4
**Current Progress:** 71/133 skills completed (53.4%)
**After Batch 20:** 75/133 skills (56.4%)

## Rationale

Batch 20 focuses on **advanced diffusion MRI modeling and microstructure imaging** tools. While previous batches covered foundational diffusion processing (DIPY, MRtrix3, DSI Studio, TractSeg, DTI-TK, TORTOISE, TractoFlow, QSIPrep), this batch addresses specialized tools for:

- **Microstructure modeling** beyond standard DTI
- **Biophysical tissue models** (NODDI, CHARMED, AxCaliber, etc.)
- **Fiber bundle recognition** and atlas-based tractography
- **Advanced diffusion analysis** with multi-compartment models
- **Clinical diffusion imaging** with integrated visualization

These tools are essential for:
- **Quantitative tissue characterization:** Extract microstructural parameters
- **Clinical research:** Disease biomarker development
- **Neurite imaging:** Axon diameter, density, dispersion quantification
- **Reproducible tractography:** Standardized bundle extraction
- **Advanced modeling:** Beyond tensor to multi-compartment models

## Tools in This Batch

### 1. DMIPY (Diffusion Microstructure Imaging in Python)
**Website:** https://dmipy.readthedocs.io/
**Platform:** Python
**Language:** Python
**Priority:** High

**Overview:**
DMIPY is a comprehensive Python framework for estimating microstructural features from diffusion MRI data using multi-compartment models. It provides modular building blocks for creating custom tissue models, implementing state-of-the-art biophysical models (NODDI, CHARMED, AxCaliber, etc.), and performing robust parameter estimation with multiple optimization strategies.

**Key Capabilities:**
- Multi-compartment tissue modeling
- Biophysical models (NODDI, CHARMED, ActiveAx, AxCaliber, SMTMC-in-vivo)
- Spherical mean technique (SMT) for rotationally invariant modeling
- Multi-shell multi-tissue (MSMT) modeling
- Microstructure fingerprinting
- Custom model construction from compartments
- GPU acceleration support
- Integration with DIPY and MRtrix3

**Target Audience:**
- Researchers studying tissue microstructure
- Method developers creating new models
- Clinical researchers quantifying disease effects
- Advanced diffusion MRI users

**Estimated Lines:** 750-800
**Estimated Code Examples:** 30-35

**Key Topics to Cover:**
1. Installation (pip, conda, dependencies)
2. Basic model building from compartments
3. Standard models (DTI, Ball-Stick, NODDI)
4. Multi-compartment model construction
5. Parameter estimation and optimization
6. Spherical mean technique (SMT)
7. Multi-shell data processing
8. Microstructure parameter maps
9. Model comparison and selection
10. GPU acceleration
11. Integration with DIPY gradient tables
12. Batch processing workflows
13. Visualization of parameter maps
14. Custom model development

**Example Workflows:**
- Estimate NODDI parameters from multi-shell data
- Build custom 3-compartment model
- Compare ball-stick vs. NODDI models
- Generate microstructure parameter maps
- SMT-based rotationally invariant analysis

**Integration Points:**
- DIPY (gradient tables, data loading)
- MRtrix3 (preprocessing outputs)
- NiBabel (file I/O)
- FSL (brain masks)

---

### 2. MDT (Microstructure Diffusion Toolbox)
**Website:** https://mdt-toolbox.readthedocs.io/
**Platform:** Python/OpenCL
**Language:** Python with OpenCL acceleration
**Priority:** High

**Overview:**
MDT (Microstructure Diffusion Toolbox) is a GPU/OpenCL-accelerated framework for fitting microstructure models to diffusion MRI data. It implements numerous biophysical models (NODDI, CHARMED, Tensor, Ball&Stick, etc.) with fast parallel processing on GPUs or CPUs. MDT emphasizes ease of use with sensible defaults while providing flexibility for advanced users to define custom models.

**Key Capabilities:**
- GPU/OpenCL acceleration for fast fitting
- 20+ pre-implemented microstructure models
- Cascade model fitting for initialization
- Multi-shell and multi-tissue support
- Protocol simulation and optimization
- Model comparison tools
- Custom model definition
- Uncertainty quantification via sampling
- Command-line and Python API
- Automatic brain masking

**Target Audience:**
- Clinical researchers needing fast processing
- Method developers with GPU resources
- Large-scale diffusion studies
- Quantitative microstructure imaging

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**
1. Installation (pip, OpenCL drivers, GPU setup)
2. Basic model fitting (DTI, Ball&Stick)
3. Advanced models (NODDI, CHARMED, ActiveAx)
4. Cascade fitting for complex models
5. Protocol optimization
6. Custom model definition
7. GPU vs CPU processing
8. Batch processing multiple subjects
9. Parameter map visualization
10. Uncertainty estimation
11. Model comparison metrics
12. Integration with preprocessing pipelines
13. Command-line usage
14. Python API for automation

**Example Workflows:**
- Fit NODDI model with GPU acceleration
- Compare multiple models on same data
- Define and fit custom tissue model
- Batch process large dataset
- Optimize acquisition protocol

**Integration Points:**
- DIPY (preprocessing, gradient schemes)
- MRtrix3 (data preparation)
- FSL (brain extraction)
- QSIPrep (preprocessed data)

---

### 3. Recobundles (Recognition of Bundles)
**Website:** https://dipy.org/documentation/latest/examples_built/bundle_extraction/
**Platform:** Python (part of DIPY)
**Language:** Python
**Priority:** Medium-High

**Overview:**
Recobundles is an atlas-based fiber bundle recognition method integrated into DIPY. It uses a template-based approach to automatically identify and extract white matter bundles from whole-brain tractography. The method is based on streamline-based registration and clustering, enabling reproducible bundle extraction without manual ROI placement.

**Key Capabilities:**
- Automatic bundle extraction from tractography
- Atlas-based bundle recognition
- Streamline-based registration (SLR)
- Bundle-specific analysis
- Multiple bundle atlases (RecobundlesX, etc.)
- Robust to individual anatomical variability
- Integration with DIPY tractography
- Reproducible bundle definition
- Quality control metrics
- Multi-subject bundle comparison

**Target Audience:**
- Researchers studying specific white matter tracts
- Connectomics studies requiring standardized bundles
- Clinical studies comparing patient groups
- Reproducible tractography analyses

**Estimated Lines:** 650-700
**Estimated Code Examples:** 26-30

**Key Topics to Cover:**
1. Installation (DIPY installation)
2. Load tractography and atlas bundles
3. Streamline-based registration (SLR)
4. Bundle recognition and extraction
5. Multiple bundle extraction
6. Quality control and validation
7. Bundle-specific metrics (FA, MD along tract)
8. Atlas management and selection
9. Custom bundle atlas creation
10. Integration with TractoFlow/QSIPrep outputs
11. Batch processing workflows
12. Visualization of extracted bundles
13. Statistical analysis of bundles
14. Cross-subject bundle comparison

**Example Workflows:**
- Extract arcuate fasciculus bilaterally
- Recognize 20 major bundles from whole-brain tractography
- Calculate FA profiles along extracted bundles
- Create custom bundle atlas from expert segmentation
- Compare bundle properties across subjects

**Integration Points:**
- DIPY (tractography, registration)
- MRtrix3 (whole-brain tracking)
- TractoFlow (preprocessing pipeline)
- TractSeg (alternative bundle segmentation)
- Surfice (bundle visualization)

---

### 4. SlicerDMRI
**Website:** https://dmri.slicer.org/
**Platform:** 3D Slicer Extension
**Language:** Python/C++ (3D Slicer)
**Priority:** Medium-High

**Overview:**
SlicerDMRI is a comprehensive diffusion MRI analysis extension for 3D Slicer, providing an integrated environment for diffusion data visualization, processing, tractography, and quantitative analysis. It combines GUI-based interaction with advanced diffusion methods including tensor estimation, tractography, tract-specific analysis, and connectivity analysis, all within the powerful 3D Slicer platform.

**Key Capabilities:**
- Integrated diffusion MRI pipeline in 3D Slicer
- Tensor fitting and visualization
- Deterministic and probabilistic tractography
- Interactive fiber bundle editing
- Tractography visualization and ROI-based filtering
- DTI scalar map calculation (FA, MD, AD, RD)
- DWI registration and motion correction
- Multi-fiber model support (UKF tractography)
- Connectivity matrix generation
- Interactive 3D visualization
- Integration with Slicer ecosystem
- DICOM import for diffusion data
- Batch processing with command-line interface

**Target Audience:**
- Clinical researchers needing GUI-based analysis
- Interactive diffusion data exploration
- Teaching and demonstration
- Combining diffusion with other imaging modalities
- Neurosurgical planning

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**
1. Installation (3D Slicer + SlicerDMRI extension)
2. Load and visualize DWI data
3. Tensor estimation and scalar maps
4. Deterministic tractography
5. UKF (unscented Kalman filter) tractography
6. Interactive ROI-based tract filtering
7. Fiber bundle editing and cleaning
8. Connectivity matrix computation
9. Integration with FreeSurfer parcellation
10. DWI registration and preprocessing
11. Batch processing via CLI
12. Python scripting in Slicer
13. Export results to standard formats
14. Multi-modal integration (structural, functional)

**Example Workflows:**
- Interactive tractography from motor cortex to spinal cord
- Extract arcuate fasciculus with ROI filtering
- Generate connectome from parcellation
- Clinical fiber tracking for surgical planning
- Multi-fiber tractography with UKF

**Integration Points:**
- 3D Slicer (platform and modules)
- FreeSurfer (parcellations for connectivity)
- DIPY (algorithms)
- DICOM (clinical data import)
- FSL/MRtrix3 (external preprocessing)

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **DMIPY** (most comprehensive Python framework)
   - **MDT** (GPU-accelerated, widely used)
   - **Recobundles** (DIPY-integrated, reproducible bundles)
   - **SlicerDMRI** (GUI-based, clinical applications)

2. **Comprehensive Coverage:**
   - Each skill: 650-800 lines
   - 26-35 code examples per skill
   - Real-world microstructure workflows
   - Integration with preprocessing pipelines

3. **Consistent Structure:**
   - Overview and key features
   - Installation (including dependencies)
   - Basic model fitting examples
   - Advanced modeling techniques
   - Parameter estimation and optimization
   - Batch processing
   - Integration with other tools
   - Troubleshooting
   - Best practices
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Package installation
   - Dependency setup (OpenCL for MDT, 3D Slicer for SlicerDMRI)
   - Environment configuration

2. **Basic Modeling** (6-8)
   - Load diffusion data
   - Simple model fitting (DTI, Ball-Stick)
   - Parameter map generation
   - Basic visualization

3. **Advanced Models** (6-8)
   - Multi-compartment models (NODDI, CHARMED)
   - Model construction from components
   - Cascade fitting strategies
   - Custom model definition

4. **Parameter Estimation** (4-6)
   - Optimization algorithms
   - Initialization strategies
   - Uncertainty quantification
   - GPU acceleration

5. **Bundle Analysis** (Recobundles, SlicerDMRI) (4-6)
   - Bundle extraction
   - ROI-based filtering
   - Tract-specific metrics
   - Visualization

6. **Integration Examples** (4-6)
   - DIPY integration
   - MRtrix3 preprocessing
   - QSIPrep outputs
   - Batch workflows

7. **Real-World Workflows** (4-6)
   - Complete analysis pipelines
   - Clinical applications
   - Research studies
   - Quality control

### Cross-Tool Integration

All skills will demonstrate integration with:
- **DIPY:** Gradient tables, data structures, preprocessing
- **MRtrix3:** Preprocessing, response functions
- **QSIPrep:** Preprocessed multi-shell data
- **FSL:** Brain masks, registration
- **TractoFlow:** Tractography outputs
- **NiBabel:** File I/O
- **Claude Code:** Automated microstructure pipelines

### Quality Targets

- **Minimum lines per skill:** 650
- **Target lines per skill:** 700-800
- **Minimum code examples:** 26
- **Target code examples:** 28-35
- **Total batch lines:** ~2,800-3,100
- **Total code examples:** ~112-129

---

## Batch Statistics

### Projected Totals

| Tool | Est. Lines | Est. Examples | Priority |
|------|-----------|---------------|----------|
| DMIPY | 750-800 | 30-35 | High |
| MDT | 700-750 | 28-32 | High |
| Recobundles | 650-700 | 26-30 | Medium-High |
| SlicerDMRI | 700-750 | 28-32 | Medium-High |
| **TOTAL** | **2,800-3,000** | **112-129** | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Microstructure modeling (DMIPY, MDT)
- ✓ Biophysical tissue models (NODDI, CHARMED, etc.)
- ✓ Multi-compartment fitting (DMIPY, MDT)
- ✓ Bundle recognition (Recobundles)
- ✓ Interactive tractography (SlicerDMRI)
- ✓ GPU acceleration (MDT)
- ✓ Clinical applications (SlicerDMRI)
- ✓ Reproducible methods (Recobundles)

**Language/Platform Coverage:**
- Python: DMIPY, MDT, Recobundles (3/4)
- 3D Slicer: SlicerDMRI (1/4)
- GPU: MDT (OpenCL)
- DIPY-integrated: Recobundles

**Modeling Approaches:**
- Multi-compartment: DMIPY, MDT
- Atlas-based: Recobundles
- Interactive: SlicerDMRI
- Biophysical: DMIPY, MDT

---

## Strategic Importance

### Fills Critical Gap

Previous diffusion batches covered:
- Basic diffusion processing (DIPY, MRtrix3, DSI Studio)
- Preprocessing pipelines (QSIPrep, TractoFlow, TORTOISE)
- Bundle segmentation (TractSeg)
- Registration (DTI-TK)

**Batch 20 adds:**
- Advanced microstructure modeling beyond DTI
- Biophysical tissue characterization
- Reproducible bundle extraction
- Clinical-focused diffusion analysis
- GPU-accelerated fitting

### Complementary Skills

**Works with existing skills:**
- **DIPY:** Foundation for gradient handling, basic modeling
- **MRtrix3:** Preprocessing, multi-shell acquisition
- **QSIPrep:** Standardized preprocessing pipeline
- **TractoFlow:** Tractography generation
- **TractSeg:** Alternative bundle segmentation
- **FSL:** Brain extraction, registration
- **Surfice/Brainrender:** Bundle visualization

### User Benefits

1. **Microstructure Quantification:**
   - Estimate NODDI parameters
   - Quantify axon diameter and density
   - Multi-compartment tissue modeling
   - Disease biomarker extraction

2. **Reproducible Tractography:**
   - Standardized bundle extraction
   - Atlas-based methods
   - Cross-subject comparison
   - Reduced manual ROI dependence

3. **Clinical Translation:**
   - GUI-based analysis (SlicerDMRI)
   - Fast GPU processing (MDT)
   - Surgical planning (SlicerDMRI)
   - Quantitative biomarkers

4. **Advanced Research:**
   - Custom model development (DMIPY)
   - Method comparison
   - Multi-shell optimization
   - Microstructure fingerprinting

---

## Dependencies and Prerequisites

### Software Prerequisites

**DMIPY:**
- Python 3.7+
- NumPy, SciPy, cvxpy
- DIPY (for gradient tables)
- NiBabel (file I/O)
- matplotlib (visualization)
- Optional: Numba for acceleration

**MDT:**
- Python 3.7+
- OpenCL drivers (GPU or CPU)
- PyOpenCL
- NumPy, SciPy, matplotlib
- NiBabel, DIPY
- Optional: CUDA for NVIDIA GPUs

**Recobundles:**
- DIPY 1.4+
- NumPy, SciPy
- NiBabel
- Scikit-learn (for clustering)
- matplotlib (visualization)

**SlicerDMRI:**
- 3D Slicer 4.11+
- SlicerDMRI extension
- Optional: FreeSurfer for parcellations
- Optional: DIPY, MRtrix3 for external processing

### Data Prerequisites

**Common to all:**
- Multi-shell diffusion MRI data (recommended)
- Single-shell also supported (limited models)
- Brain masks
- Gradient tables (bval, bvec)
- Preprocessed data (motion-corrected, eddy-corrected)

**Tool-specific:**
- **DMIPY/MDT:** Multi-shell recommended for advanced models
- **Recobundles:** Whole-brain tractography (TRK or TCK)
- **SlicerDMRI:** Raw or preprocessed DWI data

### Knowledge Prerequisites

Users should understand:
- Basic diffusion MRI concepts (b-value, gradient directions)
- Tensor model and limitations
- Multi-shell acquisition
- White matter anatomy (for bundle analysis)
- Python programming (for DMIPY, MDT, Recobundles)

---

## Learning Outcomes

After completing Batch 20 skills, users will be able to:

1. **Fit Advanced Models:**
   - Estimate NODDI parameters
   - Fit multi-compartment models
   - Compare different tissue models
   - Quantify microstructure properties

2. **Optimize Processing:**
   - Use GPU acceleration for fast fitting
   - Implement cascade fitting strategies
   - Batch process large datasets
   - Optimize acquisition protocols

3. **Extract Bundles Reproducibly:**
   - Use atlas-based bundle recognition
   - Extract standardized white matter tracts
   - Perform tract-specific analysis
   - Compare bundles across subjects

4. **Develop Custom Models:**
   - Build models from compartments (DMIPY)
   - Define custom tissue models (MDT)
   - Create bundle atlases (Recobundles)
   - Implement novel modeling approaches

5. **Clinical Applications:**
   - Perform interactive tractography (SlicerDMRI)
   - Generate connectivity matrices
   - Surgical planning with fiber tracking
   - Extract quantitative biomarkers

---

## Relationship to Existing Skills

### Builds Upon:
- **DIPY** (Batch 2): Foundation for diffusion processing
- **MRtrix3** (Batch 2): Multi-shell preprocessing
- **QSIPrep** (Batch 4): Preprocessing pipeline
- **TractoFlow** (Batch 18): Automated tractography
- **TractSeg** (Batch 9): Bundle segmentation
- **DSI Studio** (Batch 8): Alternative diffusion analysis
- **DTI-TK** (Batch 17): DTI registration

### Complements:
- **TORTOISE** (Batch 9): Preprocessing alternative
- **FSL** (Batch 1): Brain extraction, registration
- **Surfice** (Batch 19): Bundle visualization
- **Brainrender** (Batch 19): 3D visualization

### Enables:
- Quantitative microstructure imaging
- Reproducible tractography studies
- Clinical biomarker development
- Advanced diffusion modeling research
- GPU-accelerated processing

---

## Expected Challenges and Solutions

### Challenge 1: OpenCL Setup (MDT)
**Issue:** OpenCL driver installation varies by platform
**Solution:** Provide detailed platform-specific installation guides with troubleshooting

### Challenge 2: Multi-Shell Data Requirements
**Issue:** Advanced models require multi-shell data
**Solution:** Include examples with both single and multi-shell data, explain limitations

### Challenge 3: Computational Intensity
**Issue:** Model fitting can be slow without GPU
**Solution:** Document GPU setup, provide progress monitoring, suggest batch processing

### Challenge 4: 3D Slicer Learning Curve (SlicerDMRI)
**Issue:** Users unfamiliar with 3D Slicer interface
**Solution:** Include GUI navigation guide, screenshots, step-by-step tutorials

### Challenge 5: Bundle Atlas Availability (Recobundles)
**Issue:** Need appropriate atlas for study population
**Solution:** Document available atlases, explain custom atlas creation

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Test commands for successful installation
   - Dependency checking
   - GPU detection (MDT)
   - Version compatibility

2. **Basic Functionality Tests:**
   - Load sample data
   - Fit simple model
   - Generate parameter maps
   - Export results

3. **Integration Tests:**
   - Load DIPY gradient tables
   - Process QSIPrep outputs
   - Integrate with MRtrix3 data
   - Visualize results

4. **Example Data:**
   - Links to test datasets
   - Sample diffusion data
   - Expected outputs
   - Validation metrics

---

## Timeline Estimate

**Per Skill:**
- Research and planning: 15-20 min
- Writing and examples: 45-60 min
- Review and refinement: 10-15 min
- **Total per skill:** ~70-95 min

**Total Batch 20:**
- 4 skills × 82 min average = ~328 min (~5.5 hours)
- Includes documentation, examples, and testing

**Can be completed in:** 1-2 extended sessions

---

## Success Criteria

Batch 20 will be considered successful when:

✓ All 4 skills created with 650-800 lines each
✓ Total of 112+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - Basic to advanced modeling examples
  - GPU/acceleration setup (MDT)
  - Integration with DIPY, MRtrix3, QSIPrep
  - Batch processing workflows
  - Clinical application examples
  - Troubleshooting section
  - Best practices
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 75/133 (56.4%)

---

## Next Batches Preview

### Batch 21: PET Imaging & Reconstruction
- AMIDE (medical imaging data examiner)
- NiftyPET (PET reconstruction and processing)
- SIRF (synergistic image reconstruction framework)
- STIR (software for tomographic image reconstruction)

### Batch 22: Spectroscopy & Metabolite Imaging
- Osprey (MRS processing and quantification)
- TARQUIN (MRS analysis - already done!)
- FID-A (MRS toolkit - already done!)
- jMRUI (Java-based MRS - already done!)
*Note: May need different composition*

### Batch 23: Multimodal Gradients & Transcriptomics
- BrainSpace (gradient analysis)
- neuromaps (brain annotations)
- abagen (Allen Brain Atlas genetics)
- BrainStat (statistical analysis)

---

## Conclusion

Batch 20 represents a strategic deepening of **diffusion MRI capabilities** beyond basic tensor modeling. While previous batches established foundational processing (DIPY, MRtrix3, QSIPrep, TractoFlow), Batch 20 enables researchers to:

- **Quantify tissue microstructure** with biophysical models
- **Extract reproducible fiber bundles** using atlas-based methods
- **Accelerate processing** with GPU computing
- **Translate to clinical applications** with integrated visualization

These tools are at the forefront of diffusion MRI research, enabling:
- Disease biomarker development
- Microstructure quantification in clinical trials
- Standardized tractography across studies
- Advanced tissue characterization

By covering DMIPY, MDT, Recobundles, and SlicerDMRI, we provide comprehensive coverage of advanced diffusion analysis from research-focused Python libraries to clinical GUI applications.

**Status After Batch 20:** 75/133 skills (56.4% complete)

---

**Document Version:** 1.0
**Created:** 2025-11-15
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 4 skills, ~2,800-3,000 lines, ~112-129 examples
