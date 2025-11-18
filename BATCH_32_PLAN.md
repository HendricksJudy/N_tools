# Batch 32: Arterial Spin Labeling (ASL) & Perfusion Imaging - Planning Document

## Overview

**Batch Theme:** Arterial Spin Labeling (ASL) & Perfusion Imaging
**Batch Number:** 32
**Number of Skills:** 3
**Current Progress:** 108/133 skills completed (81.2%)
**After Batch 32:** 111/133 skills (83.5%)

## Rationale

Batch 32 focuses on **Arterial Spin Labeling (ASL) and perfusion imaging analysis**, a critical but specialized neuroimaging modality that quantifies cerebral blood flow (CBF) non-invasively without contrast agents. While structural (T1/T2) and functional (BOLD fMRI) imaging are well-covered, ASL provides unique information about brain hemodynamics, vascular health, and tissue perfusion that is increasingly important for aging, cerebrovascular disease, and neurodegenerative studies. These tools enable:

- **Non-invasive CBF quantification** without contrast agents (unlike DSC/DCE perfusion)
- **Absolute perfusion measurements** in mL/100g/min
- **Multiple ASL sequences** (pCASL, CASL, PASL) support
- **Vascular territory mapping** and arterial transit time analysis
- **Clinical applications** (stroke, dementia, tumors, epilepsy)
- **Integration** with structural and functional MRI
- **Standardized processing** following ASL White Paper recommendations

**Key Scientific Advances:**
- Quantitative perfusion mapping without radiation or contrast
- Multi-timepoint ASL for arterial transit time quantification
- Partial volume correction for accurate gray matter CBF
- ASL-BOLD calibration for quantitative fMRI
- Vascular territory and collateral flow assessment
- Longitudinal perfusion tracking in disease progression
- BIDS-compliant ASL preprocessing pipelines

**Applications:**
- Cerebrovascular disease (stroke, moyamoya, stenosis)
- Neurodegenerative diseases (Alzheimer's, frontotemporal dementia)
- Brain tumors (perfusion characterization, grading)
- Epilepsy (seizure focus localization)
- Aging and cognitive decline studies
- Pharmacological interventions (vasodilators, cognitive enhancers)
- Calibrated fMRI and neurovascular coupling

---

## Tools in This Batch

### 1. ExploreASL
**Website:** https://sites.google.com/view/exploreasl
**GitHub:** https://github.com/ExploreASL/ExploreASL
**Platform:** MATLAB (cross-platform)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
ExploreASL is a comprehensive, open-source pipeline for processing and analyzing multi-center ASL datasets. Developed by the ExploreASL consortium, it provides fully automated processing from raw ASL images to quantified CBF maps, with extensive quality control, multi-site harmonization support, and integration with structural MRI. ExploreASL follows ASL White Paper recommendations and supports all major ASL acquisition schemes (pCASL, CASL, PASL, multi-PLD).

**Key Capabilities:**
- End-to-end ASL processing pipeline
- Support for pCASL, CASL, PASL sequences
- Multi-PLD (post-labeling delay) analysis
- Automated quality control with visual and quantitative metrics
- CBF quantification following ASL consensus paper
- Partial volume correction (PVC)
- Registration to structural MRI (T1w)
- Spatial normalization to standard space (MNI)
- ROI-based CBF extraction
- Vascular territory mapping
- Multi-center harmonization tools
- BIDS compatibility
- Integration with CAT12, SPM
- Longitudinal ASL processing
- Comprehensive QC reports with figures

**Target Audience:**
- ASL researchers (novice to expert)
- Multi-center consortium members
- Clinical researchers (stroke, dementia)
- Aging and cerebrovascular studies
- Pharmacological imaging researchers
- Anyone needing automated ASL processing

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - MATLAB installation and configuration
   - ExploreASL download and setup
   - SPM12 dependency
   - Directory structure
   - Data organization (BIDS format)

2. **Data Preparation**
   - BIDS format for ASL
   - Required and optional files
   - JSON sidecar parameters
   - Multi-PLD data organization
   - M0 images and calibration

3. **Basic Processing**
   - Initialize ExploreASL
   - Run full pipeline
   - Pipeline stages (Structural, ASL, Population)
   - Output directory structure
   - CBF quantification parameters

4. **ASL Sequences**
   - pCASL (pseudo-continuous ASL)
   - CASL (continuous ASL)
   - PASL (pulsed ASL)
   - Multi-PLD ASL
   - Configure sequence-specific parameters

5. **Quality Control**
   - Visual QC reports
   - Automated QC metrics
   - Motion detection and correction
   - Registration quality assessment
   - Identify and exclude bad data

6. **CBF Quantification**
   - Kinetic model parameters
   - Blood-brain partition coefficient
   - Labeling efficiency
   - Partial volume correction
   - M0 calibration

7. **Spatial Processing**
   - Registration to T1w
   - Normalization to MNI space
   - Smoothing options
   - Atlas-based ROI extraction
   - Vascular territory mapping

8. **Advanced Analysis**
   - Multi-PLD analysis (arterial transit time)
   - Cerebrovascular reactivity
   - Longitudinal processing
   - Group-level statistics
   - Multi-site harmonization

9. **Integration**
   - SPM integration
   - CAT12 for segmentation
   - FSL tools compatibility
   - Export to statistical software

10. **Population Analysis**
    - Group statistics
    - ROI-based comparisons
    - Age-related perfusion changes
    - Clinical group differences

11. **Batch Processing**
    - Process multiple subjects
    - Parallel execution
    - HPC cluster submission
    - Error handling

12. **Troubleshooting**
    - Common errors
    - Parameter optimization
    - Quality issues
    - Support and resources

**Example Workflows:**
- Process multi-center Alzheimer's disease study
- Stroke perfusion analysis with vascular territories
- Multi-PLD analysis for arterial transit time
- Longitudinal perfusion changes in aging
- Pediatric ASL with age-specific templates

---

### 2. BASIL (Bayesian Arterial Spin Labeling)
**Website:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BASIL
**GitHub:** Part of FSL
**Platform:** Python/FSL (Linux, macOS)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
BASIL (Bayesian Arterial Spin Labeling) is FSL's tool for quantitative analysis of ASL MRI data using Bayesian inference. It provides advanced modeling of ASL data including estimation of perfusion, arterial transit time, arterial blood volume, and partial volume correction. BASIL uses variational Bayesian inference to provide probabilistic estimates with uncertainty quantification, making it particularly suitable for multi-PLD ASL and research requiring rigorous statistical modeling.

**Key Capabilities:**
- Bayesian inference for ASL quantification
- Single and multi-PLD ASL analysis
- Perfusion (CBF) estimation
- Arterial transit time (ATT) mapping
- Arterial blood volume (aBV) estimation
- Partial volume correction
- Spatial regularization for noise reduction
- Model selection and comparison
- Uncertainty quantification (posterior variance)
- Integration with FSL pipeline (BET, FAST, FLIRT, FNIRT)
- BASIL GUI for interactive analysis
- Command-line scripting for batch processing
- Support for pCASL, CASL, PASL
- Calibration with M0 or reference tissue

**Target Audience:**
- FSL users analyzing ASL data
- Researchers needing rigorous statistical modeling
- Multi-PLD ASL users
- Clinical researchers (stroke, tumors)
- Method developers
- Users requiring uncertainty quantification

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - FSL installation
   - BASIL as part of FSL
   - Verify installation
   - Set up FSL environment

2. **BASIL Basics**
   - Bayesian inference for ASL
   - Kinetic models
   - Prior distributions
   - Variational Bayes approximation

3. **BASIL GUI**
   - Launch GUI
   - Load ASL data
   - Configure acquisition parameters
   - Set analysis options
   - Run analysis and view results

4. **Command-Line Processing**
   - oxford_asl command
   - Basic single-PLD analysis
   - Multi-PLD analysis
   - Configure kinetic model
   - Output files

5. **Acquisition Parameters**
   - Labeling scheme (pCASL, CASL, PASL)
   - Bolus duration
   - Post-labeling delay(s)
   - Readout parameters
   - M0 calibration image

6. **Perfusion Quantification**
   - CBF estimation
   - Units and calibration
   - Absolute vs relative perfusion
   - Spatial regularization
   - Noise modeling

7. **Multi-PLD Analysis**
   - Arterial transit time estimation
   - Temporal model
   - Dispersion modeling
   - Macrovascular signal correction

8. **Partial Volume Correction**
   - Tissue segmentation (FAST)
   - PV correction methods
   - Gray matter CBF
   - White matter CBF

9. **Advanced Options**
   - Arterial blood volume (aBV)
   - Exchange model
   - Model comparison
   - Custom priors
   - Spatial priors

10. **Integration with FSL**
    - Preprocessing (BET, MCFLIRT)
    - Registration (FLIRT, FNIRT)
    - Segmentation (FAST)
    - Statistical analysis (feat, randomise)

11. **Quality Control**
    - Visual inspection
    - Calibration quality
    - Model fit assessment
    - Outlier detection

12. **Batch Processing**
    - Script-based workflows
    - Parallel processing
    - Error handling
    - HPC integration

**Example Workflows:**
- Single-PLD pCASL perfusion quantification
- Multi-PLD ATT mapping in cerebrovascular disease
- Partial volume corrected CBF in cortical regions
- Longitudinal perfusion analysis
- Group-level perfusion differences

---

### 3. ASLPrep
**Website:** https://aslprep.readthedocs.io/
**GitHub:** https://github.com/PennLINC/aslprep
**Platform:** Python (BIDS App, Docker/Singularity)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
ASLPrep is a robust, BIDS-compliant preprocessing pipeline for Arterial Spin Labeling MRI, inspired by fMRIPrep. Developed at PennLINC, ASLPrep provides automated, reproducible preprocessing of ASL data including motion correction, registration, normalization, CBF quantification, and quality control. It generates publication-ready visual reports, handles multiple ASL sequences, and integrates seamlessly with BIDS datasets and downstream analysis tools.

**Key Capabilities:**
- BIDS-compliant ASL preprocessing
- Automated robust processing pipeline
- Support for pCASL, CASL, PASL
- Single and multi-PLD ASL
- Motion correction and outlier detection
- Registration to T1w and standard spaces
- CBF quantification (multiple methods)
- Partial volume correction
- Confound extraction
- Comprehensive HTML reports
- Docker and Singularity containers
- FreeSurfer integration (optional)
- Multiple output spaces (native, T1w, MNI152)
- BIDS derivatives structure
- Quality metrics and figures
- Reproducible across platforms

**Target Audience:**
- Researchers using BIDS datasets
- Users familiar with fMRIPrep workflow
- Multi-site consortium members
- Clinical researchers needing reproducibility
- Users wanting automated preprocessing
- HPC users (container-based)

**Estimated Lines:** 650-700
**Estimated Code Examples:** 22-26

**Key Topics to Cover:**

1. **Installation and Setup**
   - Docker installation
   - Singularity installation
   - Build from source (advanced)
   - Verify installation

2. **BIDS Dataset Preparation**
   - ASL BIDS format
   - Required files and metadata
   - JSON sidecar parameters
   - Multi-session organization
   - M0 and calibration images

3. **Basic Execution**
   - Run ASLPrep with Docker
   - Run with Singularity
   - Basic command-line options
   - Participant-level analysis
   - Group-level (not applicable)

4. **Output Structure**
   - BIDS derivatives format
   - Preprocessed ASL timeseries
   - CBF maps
   - Confounds TSV
   - Quality control figures
   - HTML reports

5. **CBF Quantification**
   - Quantification methods
   - M0 calibration
   - Kinetic model parameters
   - Partial volume correction
   - Output units

6. **Processing Options**
   - Output spaces (native, T1w, MNI152)
   - Skip FreeSurfer option
   - Confound extraction
   - Smoothing
   - Resampling resolution

7. **Quality Control**
   - HTML visual reports
   - Interactive anatomical views
   - Registration quality
   - Motion assessment
   - CBF map quality

8. **Advanced Options**
   - FreeSurfer integration
   - Custom templates
   - Low memory mode
   - Ignore fieldmaps
   - Multi-echo ASL (if supported)

9. **Integration with Analysis**
   - Load preprocessed data
   - Use confounds
   - ROI extraction
   - Statistical analysis
   - Connectivity analysis (if applicable)

10. **Batch Processing**
    - Process multiple subjects
    - Parallel execution
    - HPC cluster usage
    - Resource management
    - Error recovery

11. **Comparison with Other Tools**
    - vs ExploreASL
    - vs BASIL
    - vs in-house pipelines
    - When to use which tool

12. **Troubleshooting**
    - Common errors
    - Memory issues
    - Container problems
    - BIDS validation errors
    - Support resources

**Example Workflows:**
- Preprocess BIDS ASL dataset
- Multi-subject batch processing on HPC
- Extract CBF for ROI analysis
- Longitudinal ASL preprocessing
- Integration with fMRIPrep structural outputs

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **ExploreASL** - MATLAB pipeline for multi-center ASL (new, 700-750 lines)
   - **BASIL** - FSL Bayesian ASL analysis (new, 700-750 lines)
   - **ASLPrep** - BIDS-compliant preprocessing (new, 650-700 lines)

2. **Comprehensive Coverage:**
   - Each skill: 650-750 lines
   - 22-28 code examples per skill
   - Real-world ASL analysis workflows
   - Integration with existing tools

3. **Consistent Structure:**
   - Overview and key features
   - Installation (MATLAB/FSL/Docker)
   - ASL basics and concepts
   - Basic processing workflows
   - CBF quantification
   - Quality control
   - Advanced analysis
   - Batch processing
   - Integration with other tools
   - Troubleshooting
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Software installation
   - Dependency setup
   - Verification

2. **Data Preparation** (3-4)
   - BIDS organization
   - Parameter configuration
   - M0 handling

3. **Basic Processing** (6-8)
   - Single subject analysis
   - CBF quantification
   - Quality control
   - Output inspection

4. **Advanced Analysis** (6-8)
   - Multi-PLD analysis
   - Partial volume correction
   - Vascular territories
   - Group analysis

5. **Quality Control** (3-5)
   - Visual QC
   - Quantitative metrics
   - Troubleshooting
   - Validation

6. **Batch Processing** (2-4)
   - Multi-subject workflows
   - Parallel execution
   - HPC integration
   - Error handling

7. **Integration** (2-4)
   - Structural MRI
   - fMRI analysis
   - Statistical frameworks
   - Visualization

### Cross-Tool Integration

All skills will demonstrate integration with:
- **FSL**: Registration, segmentation
- **SPM**: Statistical analysis
- **FreeSurfer**: Cortical parcellation
- **fMRIPrep (Batch 5):** Structural preprocessing
- **ANTs**: Registration and normalization
- **Nilearn (Batch 4):** Analysis and visualization

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
| ExploreASL | 700-750 | 24-28 | High | Create new |
| BASIL | 700-750 | 24-28 | High | Create new |
| ASLPrep | 650-700 | 22-26 | High | Create new |
| **TOTAL** | **2,050-2,200** | **70-82** | - | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ ASL preprocessing and quantification (all tools)
- ✓ CBF mapping (all tools)
- ✓ Multi-PLD analysis (ExploreASL, BASIL)
- ✓ Partial volume correction (all tools)
- ✓ Quality control (all tools)
- ✓ BIDS compliance (ASLPrep, ExploreASL)
- ✓ Multi-center harmonization (ExploreASL)
- ✓ Bayesian inference (BASIL)

**Platform Coverage:**
- MATLAB: ExploreASL (1/3)
- Python/FSL: BASIL, ASLPrep (2/3)
- Docker/Singularity: ASLPrep (1/3)
- GUI: BASIL (1/3)
- Command-line: All tools (3/3)

**Application Areas:**
- Cerebrovascular disease: All tools
- Neurodegenerative disease: All tools
- Brain tumors: All tools
- Aging studies: All tools
- Multi-center studies: ExploreASL, ASLPrep
- Research (advanced modeling): BASIL

---

## Strategic Importance

### Fills Critical Gap

Previous batches have covered:
- Structural MRI (FreeSurfer, ANTs, CAT12)
- Functional MRI (fMRIPrep, AFNI, FSL, SPM)
- Diffusion MRI (MRtrix3, QSIPrep, DIPY)
- Spectroscopy (Osprey, TARQUIN, FID-A)
- PET imaging (NiftyPET, STIR)

**Batch 32 adds:**
- **ASL/Perfusion imaging** - major neuroimaging modality
- **Non-invasive CBF quantification** without contrast
- **Vascular health assessment** for aging and disease
- **Multi-PLD analysis** for hemodynamic parameters
- **Standardized ASL processing** following consensus guidelines
- **Clinical applications** in stroke, dementia, tumors

### Complementary Skills

**Works with existing skills:**
- **FSL (existing):** BASIL integration, preprocessing
- **SPM (existing):** ExploreASL backend, statistics
- **fMRIPrep (Batch 5):** Structural preprocessing for ASL
- **FreeSurfer (existing):** Cortical parcellation for CBF ROIs
- **ANTs (existing):** Registration and normalization
- **Nilearn (Batch 4):** Analysis and visualization

### User Benefits

1. **Complete ASL Workflow:**
   - From raw data to quantified CBF
   - Multiple tool options
   - Standardized processing

2. **Clinical Research:**
   - Stroke perfusion assessment
   - Dementia biomarkers
   - Tumor characterization
   - Epilepsy localization

3. **Multi-Center Studies:**
   - Harmonized processing
   - Quality control frameworks
   - BIDS compliance

4. **Research Flexibility:**
   - MATLAB vs Python options
   - GUI vs command-line
   - Basic vs advanced modeling

---

## Dependencies and Prerequisites

### Software Prerequisites

**ExploreASL:**
- MATLAB R2017b or later
- SPM12
- Sufficient disk space

**BASIL:**
- FSL 6.0 or later
- Linux or macOS
- Python (for scripting)

**ASLPrep:**
- Docker or Singularity
- BIDS-formatted dataset
- FreeSurfer license (optional)

### Data Prerequisites

**Common to all:**
- ASL MRI data (pCASL, CASL, or PASL)
- M0 calibration image (recommended)
- Structural T1w image
- Acquisition parameters (labeling duration, PLD, etc.)

**Tool-specific:**
- **ExploreASL:** BIDS or legacy format
- **BASIL:** NIfTI format
- **ASLPrep:** BIDS format required

### Knowledge Prerequisites

Users should understand:
- ASL acquisition basics
- Perfusion physiology
- CBF quantification principles
- MRI preprocessing concepts
- Quality control importance

---

## Learning Outcomes

After completing Batch 32 skills, users will be able to:

1. **Process ASL Data:**
   - Preprocess ASL timeseries
   - Quantify CBF maps
   - Apply quality control
   - Handle different ASL sequences

2. **Quantify Perfusion:**
   - Calculate absolute CBF
   - Apply kinetic models
   - Perform M0 calibration
   - Correct for partial volume

3. **Advanced Analysis:**
   - Estimate arterial transit time
   - Map vascular territories
   - Analyze multi-PLD data
   - Compare clinical groups

4. **Choose Appropriate Tools:**
   - Understand tool differences
   - Select based on data/goals
   - Integrate multiple tools
   - Troubleshoot effectively

5. **Clinical Applications:**
   - Assess stroke perfusion
   - Evaluate dementia biomarkers
   - Characterize tumor perfusion
   - Monitor treatment response

---

## Relationship to Existing Skills

### Builds Upon:
- **FSL (existing):** BASIL integration
- **SPM (existing):** ExploreASL backend
- **fMRIPrep (Batch 5):** Preprocessing framework
- **FreeSurfer (existing):** Structural parcellation

### Complements:
- **AFNI/SPM/FSL:** Statistical analysis
- **ANTs (existing):** Registration
- **Nilearn (Batch 4):** Visualization
- **MRIQC (existing):** Quality assessment

### Enables:
- Comprehensive perfusion studies
- Clinical ASL research
- Multi-center ASL consortia
- Aging and cerebrovascular research
- Combined ASL-BOLD studies
- Pharmacological interventions

---

## Expected Challenges and Solutions

### Challenge 1: ASL Signal Quality
**Issue:** ASL has inherently low SNR (~1-2%)
**Solution:** Emphasize averaging, quality control, proper acquisition parameters

### Challenge 2: Parameter Complexity
**Issue:** Many acquisition and quantification parameters
**Solution:** Clear parameter tables, recommended defaults, sensitivity analysis

### Challenge 3: Multi-Center Variability
**Issue:** ASL varies across sites more than structural MRI
**Solution:** Harmonization guidelines, quality metrics, standardization protocols

### Challenge 4: Limited ASL Knowledge
**Issue:** Users may be unfamiliar with ASL basics
**Solution:** Include background sections, recommended readings, conceptual explanations

### Challenge 5: Software Diversity
**Issue:** Different platforms (MATLAB, FSL, Docker)
**Solution:** Clear installation guides, troubleshooting, platform-specific tips

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Software version checking
   - Dependency validation
   - Test data processing

2. **Basic Functionality Tests:**
   - Single subject processing
   - CBF quantification
   - Quality metrics

3. **Integration Tests:**
   - Structural MRI integration
   - Statistical analysis
   - ROI extraction

4. **Example Data:**
   - Public ASL datasets
   - Expected CBF values
   - Quality benchmarks

---

## Timeline Estimate

**Per Skill:**
- ExploreASL: 70-85 min (new, comprehensive MATLAB pipeline)
- BASIL: 70-85 min (new, Bayesian modeling complexity)
- ASLPrep: 60-75 min (new, BIDS App framework)

**Total Batch 32:**
- ~3.5-4 hours total
- Can be completed in 1-2 extended sessions

---

## Success Criteria

Batch 32 will be considered successful when:

✓ All 3 skills created with 650-750 lines each
✓ Total of 70+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - ASL background and concepts
  - Complete processing workflows
  - CBF quantification examples
  - Quality control procedures
  - Advanced analysis options
  - Batch processing templates
  - Integration examples
  - Troubleshooting section
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 111/133 (83.5%)

---

## Next Batches Preview

### Batch 33: Real-Time fMRI & Neurofeedback
- OpenNFT (real-time neurofeedback)
- Real-time processing infrastructure
- Neurofeedback experimental design

### Batch 34: Advanced Diffusion Microstructure
- DESIGNER (comprehensive diffusion preprocessing)
- Advanced microstructure models
- Validation and simulation tools

### Batch 35: Specialized Clinical Tools
- Lesion segmentation tools
- Clinical diagnostic pipelines
- Population-specific processing

---

## Conclusion

Batch 32 provides **comprehensive ASL and perfusion imaging analysis** capabilities, filling a major gap in the neuroimaging skills catalog. By covering:

- **ExploreASL** - Automated multi-center ASL processing
- **BASIL** - Bayesian ASL quantification with advanced modeling
- **ASLPrep** - BIDS-compliant ASL preprocessing pipeline

This batch enables researchers to:
- **Quantify brain perfusion** non-invasively
- **Assess vascular health** in aging and disease
- **Analyze multi-PLD data** for hemodynamic parameters
- **Process multi-center studies** with standardized methods
- **Apply clinical ASL** in stroke, dementia, tumors, epilepsy
- **Integrate perfusion** with structural and functional MRI

These tools are critical for:
- Cerebrovascular disease research
- Neurodegenerative disease biomarkers
- Brain tumor characterization
- Aging and cognitive decline studies
- Clinical trials with perfusion endpoints
- Pharmacological imaging
- Surgical planning and monitoring

By providing automated (ExploreASL), advanced (BASIL), and BIDS-compliant (ASLPrep) ASL processing, Batch 32 establishes comprehensive infrastructure for perfusion imaging research and clinical applications.

**Status After Batch 32:** 111/133 skills (83.5% complete - over 5/6 done!)

---

**Document Version:** 1.0
**Created:** 2025-11-17
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 3 skills, ~2,050-2,200 lines, ~70-82 examples
