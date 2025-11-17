# Batch 29: Advanced Surface Analysis & Cortical Morphometry - Planning Document

## Overview

**Batch Theme:** Advanced Surface Analysis & Cortical Morphometry
**Batch Number:** 29
**Number of Skills:** 3
**Current Progress:** 101/133 skills completed (75.9%)
**After Batch 29:** 104/133 skills (78.2%)

## Rationale

Batch 29 focuses on **advanced cortical surface extraction, reconstruction, and morphometric analysis** pipelines that provide alternatives and complements to FreeSurfer. While FreeSurfer (Batch 1) is the most widely-used surface analysis tool, CIVET, BrainSuite, and Mindboggle offer unique advantages including different parcellation schemes, morphometric measurements, and quality control approaches. These tools enable:

- **Alternative surface reconstruction** pipelines with different algorithms
- **Population-specific templates** and normalization approaches
- **Advanced morphometric measurements** beyond thickness and area
- **Automated cortical labeling** with multiple parcellation schemes
- **Quality control frameworks** for surface-based analyses
- **Cross-validation** of FreeSurfer results
- **Specialized analyses** for clinical and developmental populations

**Key Scientific Advances:**
- Non-linear surface registration for precise anatomical correspondence
- Multi-resolution surface-based statistics
- Comprehensive cortical feature extraction (thickness, curvature, gyrification, depth)
- Automated quality assessment of surface reconstructions
- Population-specific brain templates
- Harmonized multi-site cortical analysis protocols

**Applications:**
- Developmental neuroscience (cortical maturation)
- Aging studies (cortical atrophy patterns)
- Clinical neuroimaging (disease effects on cortex)
- Large-scale consortia (ENIGMA, UK Biobank)
- Multi-site harmonization
- Methods comparison and validation
- Cortical feature extraction for machine learning

---

## Tools in This Batch

### 1. CIVET
**Website:** https://www.bic.mni.mcgill.ca/ServicesSoftware/CIVET
**Platform:** Linux (command-line pipeline)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
CIVET (Cortical Imaging VElocimetry Toolkit) is a comprehensive pipeline for fully automated structural MRI analysis developed at the Montreal Neurological Institute (MNI). CIVET performs cortical surface extraction, cortical thickness measurement, non-linear registration to stereotaxic space, and tissue classification using a unified processing framework. It excels at multi-site studies through standardized protocols and provides extensive quality control metrics.

**Key Capabilities:**
- Fully automated T1w MRI processing pipeline
- Cortical surface extraction (white and pial surfaces)
- Cortical thickness measurement at each vertex
- Non-linear registration to MNI-ICBM152 template
- Tissue classification (GM, WM, CSF)
- Lobar and regional parcellation (AAL, DKT)
- Surface-based smoothing and registration
- Quality control image generation
- Multi-subject batch processing
- Longitudinal processing support
- Integration with RMINC for statistical analysis
- Population-specific template creation
- Pediatric and aging template support
- CSV output of morphometric measurements

**Target Audience:**
- Researchers conducting cortical morphometry studies
- Multi-site consortium investigators (ENIGMA)
- Developmental neuroscientists
- Clinical researchers studying cortical pathology
- Users wanting alternatives to FreeSurfer
- Large-scale population studies

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - CIVET installation (container vs native)
   - License and registration
   - Verify installation and dependencies
   - Directory structure and organization

2. **Basic Pipeline Execution**
   - Run CIVET on single subject
   - Input requirements (T1w NIfTI)
   - Pipeline stages overview
   - Output directory structure

3. **Pipeline Configuration**
   - Template selection (ICBM152, pediatric, aging)
   - Pipeline options and parameters
   - Processing flags and customization
   - Mask and lesion handling

4. **Output Files and Interpretation**
   - Surface files (white, gray, mid-surface)
   - Thickness maps
   - Volume files (classified tissue)
   - Transform files
   - Quality control images

5. **Quality Control**
   - Visual QC protocol
   - Automated QC metrics
   - Common failure modes
   - Manual correction procedures

6. **Surface-Based Analysis**
   - Surface registration and resampling
   - Cortical thickness analysis
   - Surface area and volume measurements
   - Gyrification index

7. **Batch Processing**
   - Process multiple subjects
   - Parallel execution
   - Job submission for HPC
   - Error handling and logging

8. **Statistical Analysis with RMINC**
   - Load CIVET outputs in R
   - Vertex-wise statistics
   - ROI-based analysis
   - Visualization of results

9. **Longitudinal Analysis**
   - Multi-timepoint processing
   - Within-subject registration
   - Longitudinal cortical change
   - Atrophy rate calculation

10. **Integration and Comparison**
    - Compare with FreeSurfer results
    - ENIGMA protocol compatibility
    - Export to surface viewers
    - Data sharing formats

11. **Advanced Features**
    - Custom templates
    - Population-specific normalization
    - Lesion masking
    - Multi-modal integration

12. **Troubleshooting**
    - Common errors and solutions
    - Quality issues and fixes
    - Performance optimization
    - Support and resources

**Example Workflows:**
- Process ADNI dataset for cortical thickness analysis
- Developmental trajectory of cortical maturation
- Multi-site harmonization with ENIGMA protocol
- Longitudinal atrophy measurement in aging
- Clinical vs. control cortical comparison

---

### 2. BrainSuite
**Website:** http://brainsuite.org/
**GitHub:** https://github.com/bids-apps/BrainSuite
**Platform:** Linux/macOS/Windows (GUI + command-line)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
BrainSuite is a comprehensive software suite for processing and analyzing structural and diffusion MRI data with emphasis on cortical surface modeling, registration, and parcellation. Developed at UCLA, BrainSuite provides both a user-friendly GUI and command-line tools for surface extraction, cortical labeling using multiple atlases, and integration with diffusion tractography. It stands out for its anatomical registration accuracy and multi-modal integration capabilities.

**Key Capabilities:**
- Cortical surface extraction (inner, pial, mid-cortical)
- Brain extraction and bias correction (BFC, BSE)
- Cortical parcellation with BrainSuite Labeling Protocol (BCI-DNI)
- Subcortical structure segmentation
- Non-linear registration (BrainSuite Volume Registration - SVR)
- Surface registration (Curve-Based Registration - CBR)
- Diffusion MRI processing and tractography integration
- Multi-atlas labeling and fusion
- Interactive GUI for manual editing and QC
- Batch processing via BrainSuite Dashboard
- Statistical analysis tools
- Integration with FSL, FreeSurfer outputs
- BIDS App available
- Population atlas creation

**Target Audience:**
- Researchers needing interactive surface editing
- Multi-modal imaging studies (T1 + DTI)
- Users preferring GUI-based workflows
- Clinical applications requiring precise labeling
- Method developers testing atlas-based segmentation
- Teaching and training environments

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - Download and install BrainSuite
   - GUI vs command-line tools
   - MATLAB runtime (if needed)
   - Verify installation

2. **GUI-Based Processing**
   - Launch BrainSuite GUI
   - Load T1w image
   - Run Cortical Surface Extraction Sequence
   - Interactive viewing and editing
   - Save outputs

3. **Surface Extraction Pipeline**
   - Brain Surface Extractor (BSE)
   - Bias Field Corrector (BFC)
   - Partial Volume Classifier (PVC)
   - Cerebrum labeling
   - Surface generation (inner, pial, mid)
   - Topology correction

4. **Cortical Labeling**
   - BrainSuite labeling protocol (BCI-DNI atlas)
   - Apply cortical labels
   - Brainstem and cerebellar labels
   - Subcortical structure segmentation
   - Custom atlas application

5. **Surface Registration**
   - Sulcal curve extraction
   - Curve-based registration to atlas
   - Surface-constrained volumetric registration (SVReg)
   - Registration quality assessment

6. **Command-Line Tools**
   - bse (brain extraction)
   - bfc (bias correction)
   - pvc (tissue classification)
   - cerebroextract
   - cortex.svreg.sh (registration)

7. **Batch Processing**
   - BrainSuite Dashboard for batch jobs
   - Process multiple subjects
   - Automated workflow execution
   - Quality control automation

8. **Diffusion MRI Integration**
   - Load diffusion data
   - Run BDP (BrainSuite Diffusion Pipeline)
   - Tractography generation
   - Surface-constrained tractography
   - Connectivity matrices

9. **Quality Control and Editing**
   - Visual inspection workflow
   - Manual correction tools
   - Mask editing
   - Surface refinement
   - QC metrics

10. **Statistical Analysis**
    - Extract ROI statistics
    - Surface-based statistics
    - Group comparison tools
    - Visualization of results

11. **BrainSuite BIDS App**
    - Run via Docker/Singularity
    - BIDS dataset processing
    - Automated quality reports

12. **Integration and Export**
    - Export to FreeSurfer format
    - Import FreeSurfer surfaces
    - FSL integration
    - Data sharing formats

**Example Workflows:**
- Interactive cortical surface editing for clinical cases
- Multi-atlas brain parcellation
- Diffusion-weighted MRI tractography with cortical endpoints
- Batch processing of research cohort
- Population atlas creation

---

### 3. Mindboggle
**Website:** https://mindboggle.info/
**GitHub:** https://github.com/nipy/mindboggle
**Platform:** Python (Docker container)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
Mindboggle is an open-source software package for automated feature extraction and labeling of human brain MRI data, with particular emphasis on cortical shape analysis and anatomical labeling harmonization. Developed by Arno Klein and collaborators, Mindboggle improves upon FreeSurfer outputs by applying advanced labeling algorithms, extracting detailed shape features, and computing comprehensive morphometric statistics. It serves as both an analysis tool and a resource for anatomical labels and shape databases.

**Key Capabilities:**
- Automated anatomical labeling refinement
- Comprehensive shape feature extraction (depth, curvature, thickness, area, volume, travel depth, geodesic depth)
- Cortical labeling with multiple protocols (DKT, Desikan-Killiany)
- Sulcal fundus and gyral crest extraction
- Shape-based registration and correspondence
- Quality metrics for anatomical labels
- Harmonized multi-atlas labeling
- Integration with FreeSurfer, ANTS outputs
- Docker container for reproducibility
- Label statistics and morphometry tables
- Visualization tools for shape features
- Database of anatomical shapes for comparison

**Target Audience:**
- Researchers extracting detailed cortical features
- Anatomical labeling quality improvement
- Shape-based morphometry studies
- FreeSurfer users wanting refined labels
- Method developers in cortical parcellation
- Large-scale morphometry projects

**Estimated Lines:** 650-700
**Estimated Code Examples:** 22-26

**Key Topics to Cover:**

1. **Installation and Setup**
   - Docker installation (recommended)
   - Build from source (advanced)
   - Verify installation
   - Input requirements

2. **Input Preparation**
   - FreeSurfer recon-all outputs
   - ANTs cortical thickness pipeline outputs
   - Required files and directory structure
   - BIDS compatibility

3. **Running Mindboggle**
   - Basic Docker command
   - Input/output specification
   - Processing options
   - Resource requirements (memory, time)

4. **Output Files and Structure**
   - Labeled surfaces
   - Shape feature maps
   - Morphometry tables (CSV)
   - VTK files for visualization
   - HTML quality reports

5. **Anatomical Labeling**
   - DKT cortical labeling protocol
   - Label refinement algorithms
   - Sulcal and gyral identification
   - Label quality metrics
   - Manual label comparison

6. **Shape Feature Extraction**
   - Cortical depth (geodesic, travel)
   - Mean and Gaussian curvature
   - Surface area (pial, white, mid)
   - Cortical thickness
   - Sulcal depth and width
   - Gyrification measures

7. **Morphometry Tables**
   - Load and interpret CSV outputs
   - Per-label statistics
   - Per-vertex measures
   - Fundus-based measurements
   - Volume and area statistics

8. **Quality Control**
   - Visual inspection of labels
   - Shape feature visualization
   - Quality metrics interpretation
   - Compare with FreeSurfer labels
   - Identify problematic regions

9. **Visualization**
   - VTK surface rendering
   - Shape feature overlays
   - Label boundary visualization
   - Export for external viewers (FreeView, Paraview)

10. **Batch Processing**
    - Process multiple subjects
    - Parallel execution strategies
    - HPC submission scripts
    - Aggregate statistics across subjects

11. **Integration with Analysis**
    - Load morphometry data in Python/R
    - Statistical analysis of shape features
    - Group comparisons
    - Correlation with behavior/genetics

12. **Advanced Usage**
    - Custom labeling protocols
    - Shape database queries
    - Registration based on shape
    - Method validation studies

**Example Workflows:**
- Refine FreeSurfer labels for improved anatomical accuracy
- Extract comprehensive shape features for machine learning
- Compare sulcal patterns across populations
- Harmonize labels across scanners/sites
- Create population-specific anatomical atlases

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **CIVET** - MNI pipeline for large-scale studies (new, 700-750 lines)
   - **BrainSuite** - UCLA interactive surface analysis (new, 700-750 lines)
   - **Mindboggle** - Advanced labeling and shape analysis (new, 650-700 lines)

2. **Comprehensive Coverage:**
   - Each skill: 650-750 lines
   - 22-28 code examples per skill
   - Real-world cortical analysis workflows
   - Integration with existing tools (FreeSurfer, ANTs)

3. **Consistent Structure:**
   - Overview and key features
   - Installation (container + native where applicable)
   - Basic pipeline execution
   - Surface extraction and registration
   - Quality control procedures
   - Morphometric analysis
   - Batch processing
   - Statistical analysis integration
   - Troubleshooting
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Container installation
   - Native installation
   - Verification

2. **Basic Pipeline** (6-8)
   - Single subject processing
   - Default parameters
   - Output inspection
   - Quality checking

3. **Surface Analysis** (6-8)
   - Surface extraction
   - Registration
   - Thickness measurement
   - Feature extraction
   - Parcellation

4. **Quality Control** (3-5)
   - Visual QC
   - Automated metrics
   - Manual corrections
   - Comparison with standards

5. **Batch Processing** (3-5)
   - Multi-subject workflows
   - Parallel execution
   - HPC integration
   - Error handling

6. **Statistical Analysis** (2-4)
   - Extract measurements
   - Group comparisons
   - ROI-based statistics
   - Visualization

7. **Integration** (2-4)
   - FreeSurfer comparison
   - ANTs integration
   - Export to other tools
   - Data sharing

### Cross-Tool Integration

All skills will demonstrate integration with:
- **FreeSurfer (Batch 1):** Comparison, validation, alternative pipelines
- **ANTs (Batch 1):** Registration, normalization
- **SurfStat (Batch 24):** Statistical analysis of surfaces
- **TemplateFlow (Batch 28):** Standard templates
- **fMRIPrep (Batch 5):** Anatomical preprocessing
- **Visualization:** Paraview, FreeView, BrainNet Viewer

### Quality Targets

- **Minimum lines per skill:** 650
- **Target lines per skill:** 650-750
- **Minimum code examples:** 22
- **Target code examples:** 22-28
- **Total batch lines:** ~2,050-2,200
- **Total code examples:** ~68-82

---

## Batch Statistics

### Projected Totals

| Tool | Est. Lines | Est. Examples | Priority | Status |
|------|-----------|---------------|----------|---------|
| CIVET | 700-750 | 24-28 | High | Create new |
| BrainSuite | 700-750 | 24-28 | High | Create new |
| Mindboggle | 650-700 | 22-26 | High | Create new |
| **TOTAL** | **2,050-2,200** | **70-82** | - | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Cortical surface extraction (all tools)
- ✓ Cortical thickness measurement (all tools)
- ✓ Surface-based registration (all tools)
- ✓ Anatomical labeling (all tools)
- ✓ Quality control (all tools)
- ✓ Shape feature extraction (Mindboggle)
- ✓ Multi-modal integration (BrainSuite)
- ✓ Batch processing (all tools)

**Platform Coverage:**
- Linux: All tools (3/3)
- macOS: BrainSuite (1/3)
- Windows: BrainSuite (1/3)
- GUI: BrainSuite (1/3)
- Command-line: All tools (3/3)
- Container: CIVET, Mindboggle, BrainSuite (3/3)

**Application Areas:**
- Developmental neuroscience: All tools
- Aging studies: All tools
- Clinical neuroimaging: All tools
- Large-scale consortia: CIVET, Mindboggle
- Multi-site studies: All tools
- Methods comparison: All tools

---

## Strategic Importance

### Fills Critical Gap

Previous batches have covered:
- FreeSurfer (Batch 1): Gold standard surface analysis
- SurfStat (Batch 24): Surface-based statistics
- CAT12 (Batch 8): VBM and surface-based morphometry

**Batch 29 adds:**
- **Alternative surface pipelines** with different algorithms
- **Interactive editing** capabilities (BrainSuite)
- **Advanced shape analysis** (Mindboggle)
- **Multi-site harmonization** protocols (CIVET)
- **Quality improvement** over FreeSurfer (Mindboggle)
- **Multi-modal integration** (BrainSuite diffusion)

### Complementary Skills

**Works with existing skills:**
- **FreeSurfer (Batch 1):** Alternative/complementary pipelines
- **ANTs (Batch 1):** Registration integration
- **SurfStat (Batch 24):** Statistical analysis of surfaces
- **CAT12 (Batch 8):** VBM and surface morphometry
- **TemplateFlow (Batch 28):** Standard templates
- **fMRIPrep (Batch 5):** Anatomical derivatives

### User Benefits

1. **Methodological Diversity:**
   - Multiple surface extraction algorithms
   - Cross-validation of results
   - Method comparison studies
   - Choose best tool for specific data

2. **Enhanced Features:**
   - More detailed shape measurements
   - Improved anatomical labeling
   - Interactive quality control
   - Multi-modal integration

3. **Large-Scale Studies:**
   - Optimized for multi-site data
   - Standardized protocols (ENIGMA)
   - Batch processing efficiency
   - Quality harmonization

4. **Research Flexibility:**
   - Population-specific templates
   - Custom parcellation schemes
   - Advanced morphometric features
   - Integration with other modalities

---

## Dependencies and Prerequisites

### Software Prerequisites

**CIVET:**
- Linux operating system
- Perl
- Significant disk space (~10GB per subject)
- Registration required (free for academic use)

**BrainSuite:**
- Linux/macOS/Windows
- MATLAB Runtime (for some tools)
- Docker/Singularity (for BIDS App)

**Mindboggle:**
- Docker (recommended)
- Or: Python 3.6+, VTK, ANTs, FreeSurfer (for source install)

### Data Prerequisites

**Common to all:**
- T1-weighted MRI (preferably 1mm isotropic)
- Skull-stripped or will be stripped by pipeline
- NIfTI format

**Tool-specific:**
- **CIVET:** Raw T1w (will perform skull-stripping)
- **BrainSuite:** Any orientation (will reorient)
- **Mindboggle:** FreeSurfer or ANTs outputs

### Knowledge Prerequisites

Users should understand:
- Neuroanatomy basics
- Surface-based vs volumetric analysis
- Quality control principles
- Basic command-line usage
- Brain templates and normalization
- Cortical parcellation concepts

---

## Learning Outcomes

After completing Batch 29 skills, users will be able to:

1. **Process Structural MRI:**
   - Extract cortical surfaces with multiple tools
   - Measure cortical thickness and area
   - Apply anatomical parcellations
   - Perform surface-based registration

2. **Quality Control:**
   - Identify common surface reconstruction failures
   - Apply manual corrections where needed
   - Use automated QC metrics
   - Compare results across methods

3. **Morphometric Analysis:**
   - Extract ROI-based measurements
   - Compute vertex-wise statistics
   - Analyze shape features
   - Perform group comparisons

4. **Method Comparison:**
   - Compare FreeSurfer, CIVET, BrainSuite results
   - Understand strengths of each tool
   - Choose appropriate method for data
   - Cross-validate findings

5. **Large-Scale Processing:**
   - Batch process cohorts efficiently
   - Implement quality harmonization
   - Use HPC resources effectively
   - Manage large datasets

---

## Relationship to Existing Skills

### Builds Upon:
- **FreeSurfer (Batch 1):** Standard for comparison
- **ANTs (Batch 1):** Registration methods
- **TemplateFlow (Batch 28):** Brain templates
- **fMRIPrep (Batch 5):** Anatomical preprocessing
- **CAT12 (Batch 8):** Alternative morphometry

### Complements:
- **SurfStat (Batch 24):** Statistical analysis
- **BrainSpace (Batch 27):** Gradient analysis
- **Visualization tools:** Display surfaces and results
- **Network analysis:** Parcellation-based connectivity

### Enables:
- Multi-method cortical morphometry
- Robust surface-based analyses
- Large-scale consortium studies
- Method development and validation
- Clinical surface analysis
- Population neuroscience

---

## Expected Challenges and Solutions

### Challenge 1: Long Processing Times
**Issue:** Surface extraction can take 6-24 hours per subject
**Solution:** Parallel processing, HPC clusters, optimize parameters, use faster alternatives where appropriate

### Challenge 2: Quality Control Burden
**Issue:** Manual QC of surfaces is time-consuming
**Solution:** Automated QC metrics, prioritize failed cases, batch visualization tools

### Challenge 3: Learning Multiple Tools
**Issue:** Each tool has different conventions and workflows
**Solution:** Consistent documentation structure, comparison tables, unified examples

### Challenge 4: Storage Requirements
**Issue:** Surface outputs require significant disk space
**Solution:** Guidance on essential vs optional outputs, compression strategies, cleanup scripts

### Challenge 5: Installation Complexity
**Issue:** Some tools have complex dependencies
**Solution:** Container-based approaches, clear installation guides, troubleshooting sections

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Software version checking
   - Dependency validation
   - Test data processing

2. **Basic Functionality Tests:**
   - Single subject processing
   - Output file verification
   - Quality metric extraction

3. **Integration Tests:**
   - FreeSurfer comparison
   - Atlas application
   - Statistical analysis workflow

4. **Example Data:**
   - Links to public datasets (OASIS, ADNI)
   - Expected output examples
   - Quality benchmarks

---

## Timeline Estimate

**Per Skill:**
- CIVET: 70-85 min (new, comprehensive pipeline)
- BrainSuite: 70-85 min (new, GUI + CLI coverage)
- Mindboggle: 60-75 min (new, shape analysis focus)

**Total Batch 29:**
- ~3.5-4 hours total
- Can be completed in 1-2 extended sessions

---

## Success Criteria

Batch 29 will be considered successful when:

✓ All 3 skills created with 650-750 lines each
✓ Total of 70+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - Complete pipeline execution examples
  - Surface extraction and registration workflows
  - Quality control procedures
  - Morphometric analysis examples
  - Batch processing templates
  - Integration with FreeSurfer/ANTs
  - Statistical analysis examples
  - Troubleshooting section
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 104/133 (78.2%)

---

## Next Batches Preview

### Batch 30: Meta-Analysis & Coordinate-Based Methods
- NeuroSynth (large-scale automated meta-analysis)
- NiMARE (neuroimaging meta-analysis research environment)
- NeuroQuery (meta-analytic brain decoding)

### Batch 31: Specialized Diffusion Methods
- DESIGNER (comprehensive diffusion preprocessing)
- Advanced microstructure models
- Diffusion simulation and validation tools

### Batch 32: Clinical & Lesion Analysis
- Lesion analysis tools
- Clinical diagnostic pipelines
- Specialized clinical populations

---

## Conclusion

Batch 29 provides **advanced cortical surface analysis** capabilities beyond FreeSurfer, enabling researchers to leverage multiple surface reconstruction algorithms, perform detailed shape analysis, and ensure robust results through cross-method validation. By covering:

- **CIVET** - MNI standardized pipeline for large-scale studies
- **BrainSuite** - Interactive multi-modal surface analysis
- **Mindboggle** - Advanced anatomical labeling and shape features

This batch enables researchers to:
- **Choose optimal methods** for their specific data and questions
- **Cross-validate** cortical findings across multiple pipelines
- **Extract advanced features** beyond standard morphometry
- **Participate in consortia** using standardized protocols
- **Improve anatomical precision** with refined labeling
- **Integrate modalities** (structural + diffusion)

These tools are critical for:
- Developmental and aging neuroscience
- Clinical cortical morphometry
- Large-scale population studies (ENIGMA, UK Biobank)
- Multi-site harmonization
- Method development and validation
- Comprehensive cortical phenotyping

By providing multiple surface analysis pipelines with unique strengths, Batch 29 positions users to conduct robust, validated cortical morphometry research with methodological rigor and flexibility.

**Status After Batch 29:** 104/133 skills (78.2% complete - approaching 80%!)

---

**Document Version:** 1.0
**Created:** 2025-11-17
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 3 skills, ~2,050-2,200 lines, ~70-82 examples
