# Batch 26: Advanced Diffusion Preprocessing & Microstructure - Planning Document

## Overview

**Batch Theme:** Advanced Diffusion Preprocessing & Microstructure
**Batch Number:** 26
**Number of Skills:** 4
**Current Progress:** 95/133 skills completed (71.4%)
**After Batch 26:** 99/133 skills (74.4%)

## Rationale

Batch 26 focuses on **advanced diffusion MRI preprocessing and microstructure-informed tractography**, providing specialized tools for state-of-the-art diffusion analysis. This batch addresses critical needs in diffusion neuroimaging:

- **Advanced preprocessing** to remove artifacts (noise, Gibbs ringing, distortions)
- **Comprehensive diffusion processing** from raw DWI to tensor metrics
- **Complete tractography pipelines** for reproducible structural connectivity
- **Microstructure-informed tractography** that respects tissue properties
- **Quality control** specific to diffusion data

These tools enable researchers to move beyond basic DTI to conduct **cutting-edge diffusion analysis** with proper artifact correction, advanced modeling, and biologically-informed tractography.

**Key Scientific Advances:**
- Advanced denoising and artifact correction improve data quality substantially
- Gibbs ringing removal prevents spurious peaks in fiber orientation distributions
- Distortion correction enables accurate registration and anatomical localization
- Microstructure-informed tractography reduces false positives and increases specificity
- Complete automated pipelines ensure reproducibility across studies

**Applications:**
- High-quality structural connectivity for network neuroscience
- Clinical diffusion biomarkers with minimal artifact contamination
- Microstructure mapping (NODDI, DIAMOND, etc.) requires clean data
- Tractography for presurgical planning
- Multi-site harmonization through consistent preprocessing
- Longitudinal studies requiring robust preprocessing

---

## Tools in This Batch

### 1. DESIGNER
**Website:** https://github.com/NYU-DiffusionMRI/DESIGNER
**GitHub:** https://github.com/NYU-DiffusionMRI/DESIGNER
**Platform:** Python/MATLAB
**Priority:** High

**Overview:**
DESIGNER (Diffusion parameter EStImation with Gibbs and NoisE Removal) is a state-of-the-art preprocessing pipeline developed at NYU for advanced diffusion MRI data processing. It implements cutting-edge methods for denoising (Marchenko-Pastur PCA), Gibbs ringing removal, distortion correction, and motion correction in a unified, automated pipeline. DESIGNER is specifically designed to prepare diffusion data for advanced modeling techniques (NODDI, SMT, DKI) by maximally removing artifacts while preserving biological signal.

**Key Capabilities:**
- Marchenko-Pastur PCA denoising (removes thermal noise optimally)
- Gibbs ringing artifact removal (prevents spurious fiber orientations)
- Distortion correction (EPI, eddy current, motion)
- Rician bias correction
- B1 field inhomogeneity correction
- Outlier detection and removal
- Automated quality control metrics
- Preparation for DKI, NODDI, SMT, and other advanced models
- Integration with MRtrix3 and FSL
- Python and MATLAB implementations
- Batch processing capabilities
- Comprehensive preprocessing reports

**Target Audience:**
- Diffusion MRI researchers
- Microstructure imaging specialists
- Clinical neuroimaging with high-quality requirements
- Multi-site harmonization studies
- Advanced diffusion modeling (DKI, NODDI)

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install DESIGNER (Python/MATLAB versions)
   - Dependencies (FSL, MRtrix3, ANTs)
   - Verify installation
   - Data organization requirements

2. **Basic Preprocessing Pipeline**
   - Load raw diffusion data
   - Run full DESIGNER pipeline
   - Inspect preprocessing outputs
   - Quality control visualizations

3. **Step-by-Step Processing**
   - Denoising with MP-PCA
   - Gibbs ringing removal
   - Distortion correction (topup, eddy)
   - Motion correction
   - Rician bias correction

4. **Quality Control**
   - Automated QC metrics
   - Visual inspection tools
   - SNR maps before/after denoising
   - Residual analysis
   - Motion parameters

5. **Advanced Options**
   - B1 field correction
   - Outlier detection and exclusion
   - Custom denoising parameters
   - Selective preprocessing steps
   - Multi-shell optimization

6. **Integration with Modeling**
   - Prepare data for DKI fitting
   - Export for NODDI analysis
   - Output for SMT models
   - Integration with MRtrix3 downstream

7. **Batch Processing**
   - Multi-subject automation
   - BIDS-compatible workflows
   - Cluster computing
   - Containerization (Docker/Singularity)

8. **Troubleshooting**
   - Common preprocessing failures
   - Handling partial datasets
   - Parameter tuning for noisy data
   - Validation against ground truth

**Example Workflows:**
- Complete preprocessing of HCP-style diffusion acquisition
- Prepare multi-shell data for NODDI modeling
- Denoise and correct clinical DTI data
- Multi-site preprocessing harmonization
- Quality control pipeline for large datasets

**Integration Points:**
- **MRtrix3:** Downstream tractography and modeling
- **FSL:** Distortion and motion correction
- **ANTs:** Registration and normalization
- **NODDI Toolbox:** Microstructure fitting
- **DSI Studio:** Alternative modeling and tractography

---

### 2. TORTOISE
**Website:** https://tortoise.nibib.nih.gov/
**GitHub:** https://github.com/eurotomania/TORTOISEV4
**Platform:** Linux/macOS (command-line)
**Priority:** High

**Overview:**
TORTOISE (Tolerably Obsessive Registration and Tensor Optimization Indolent Software Ensemble) is a comprehensive diffusion MRI processing software developed at the NIH. TORTOISE is renowned for its sophisticated distortion correction, including correction for eddy current distortions, susceptibility-induced distortions, and motion artifacts. It implements advanced tensor fitting, fiber tracking, and diffusion parameter mapping with particular emphasis on data quality and accuracy. TORTOISE is widely used in clinical research and developmental studies.

**Key Capabilities:**
- Advanced distortion correction (DIFF_PREP, DIFF_CALC, DR_BUDDI)
- Eddy current and motion correction
- Susceptibility-induced distortion correction (structural-driven)
- Gradient nonlinearity correction
- DTI, DKI, and NODDI fitting
- Probabilistic and deterministic tractography
- Tensor-based registration
- ROI-based analysis
- White matter atlas integration
- Comprehensive quality control tools
- MAPMRI implementation
- Batch processing scripts
- Integration with structural MRI for registration

**Target Audience:**
- Clinical diffusion researchers
- Developmental neuroscience (neonatal, pediatric imaging)
- Precision tractography for presurgical planning
- Longitudinal diffusion studies
- Multi-modal integration studies

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**

1. **Installation and Setup**
   - Download and install TORTOISE
   - Set up environment variables
   - Verify dependencies
   - Data import and format conversion

2. **Distortion Correction Workflow**
   - DIFF_PREP for preprocessing
   - DR_BUDDI for structural-diffusion registration
   - Eddy current correction
   - Motion correction
   - Susceptibility correction

3. **Tensor Estimation**
   - DIFF_CALC for DTI fitting
   - Compute FA, MD, AD, RD maps
   - Export scalar maps
   - Quality assessment

4. **Advanced Diffusion Models**
   - Diffusion Kurtosis Imaging (DKI)
   - NODDI parameter estimation
   - MAPMRI fitting
   - Model comparison

5. **Fiber Tracking**
   - Deterministic tractography
   - Probabilistic tractography
   - Multi-ROI tracking
   - Tract-specific analysis
   - White matter bundle extraction

6. **Registration and Normalization**
   - Tensor-based registration (DTI-TK integration)
   - Registration to structural MRI
   - Normalization to atlas space
   - Group template creation

7. **ROI Analysis**
   - Define ROIs on FA maps
   - Atlas-based analysis
   - Tract-based statistics
   - Export measurements

8. **Batch Processing and Automation**
   - Scripting TORTOISE workflows
   - Multi-subject processing
   - Integration with cluster computing
   - Quality control pipelines

**Example Workflows:**
- Neonatal diffusion preprocessing with advanced distortion correction
- DTI analysis with atlas-based ROI quantification
- Presurgical tractography for tumor resection planning
- Longitudinal DTI study with tensor-based registration
- Multi-modal structural-diffusion integration

**Integration Points:**
- **FSL:** Complementary analysis tools
- **DTI-TK:** Tensor-based registration
- **FreeSurfer:** Anatomical segmentation for ROIs
- **ANTs:** Alternative registration
- **TrackVis:** Tractography visualization

---

### 3. TractoFlow
**Website:** https://tractoflow-documentation.readthedocs.io/
**GitHub:** https://github.com/scilus/tractoflow
**Platform:** Nextflow pipeline (cross-platform)
**Priority:** Medium-High

**Overview:**
TractoFlow is a fully automated and reproducible tractography pipeline developed by the Sherbrooke Connectivity Imaging Lab (SCIL). Built using Nextflow for workflow management and containerized with Docker/Singularity, TractoFlow implements state-of-the-art diffusion preprocessing, fiber orientation distribution estimation, and tractography in a BIDS-compatible, reproducible framework. It integrates best practices from MRtrix3, FSL, and ANTs into a cohesive pipeline that runs from raw DWI to whole-brain tractography with comprehensive quality control.

**Key Capabilities:**
- Complete DWI-to-tractography pipeline (fully automated)
- BIDS-compatible input/output
- Advanced denoising and preprocessing
- Multi-shell, multi-tissue constrained spherical deconvolution (MSMT-CSD)
- Anatomically-constrained tractography (ACT)
- Particle filtering tractography (PFT)
- Seeding strategies (WM interface, FA-based, tissue-based)
- Comprehensive quality control reports
- Reproducible via Nextflow and containers
- Parallelization for HPC environments
- Modular architecture (can run individual steps)
- Integration with downstream connectivity analysis

**Target Audience:**
- Researchers needing reproducible tractography pipelines
- Multi-site connectome studies
- Large-scale tractography projects
- Labs without extensive diffusion expertise
- Studies requiring BIDS compliance and reproducibility

**Estimated Lines:** 650-700
**Estimated Code Examples:** 26-30

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install Nextflow
   - Install Docker/Singularity
   - Download TractoFlow container
   - BIDS data organization
   - Configuration files

2. **Running Complete Pipeline**
   - Basic TractoFlow command
   - Input/output structure
   - Pipeline execution monitoring
   - Resource allocation (CPU, memory)
   - Resume interrupted runs

3. **Preprocessing Steps**
   - Denoising (MP-PCA)
   - Gibbs ringing removal
   - Distortion correction (topup, eddy)
   - Motion correction
   - Bias field correction
   - Brain extraction

4. **Fiber Orientation Estimation**
   - Single-shell CSD
   - Multi-shell, multi-tissue CSD (MSMT-CSD)
   - Response function estimation
   - FOD visualization
   - Peak extraction

5. **Anatomically-Constrained Tractography**
   - ACT tissue segmentation (5TT)
   - Generate tractography
   - Seeding strategies
   - Anatomical constraints
   - Filtering parameters

6. **Quality Control**
   - Automated QC reports
   - Visual inspection of preprocessing
   - Tractogram statistics
   - Coverage maps
   - Identify failed subjects

7. **Customization**
   - Modify pipeline parameters
   - Custom configuration files
   - Skip specific steps
   - Add custom processing
   - Integration with other tools

8. **Downstream Analysis**
   - Connectivity matrix generation
   - Tractometry analysis
   - Integration with network analysis
   - Export for visualization
   - BIDS derivatives structure

**Example Workflows:**
- Process HCP-style multi-shell diffusion data
- Generate whole-brain tractography for connectome analysis
- Multi-site diffusion study with standardized pipeline
- Quality-controlled tractography for large cohorts
- Reproducible structural connectivity matrices

**Integration Points:**
- **MRtrix3:** Underlying algorithms
- **FSL:** Preprocessing tools
- **ANTs:** Registration
- **FreeSurfer:** Anatomical segmentation
- **NetworkX/BCT:** Connectivity analysis downstream

---

### 4. COMMIT
**Website:** https://github.com/daducci/COMMIT
**GitHub:** https://github.com/daducci/COMMIT
**Platform:** Python
**Priority:** Medium-High

**Overview:**
COMMIT (Convex Optimization Modeling for Microstructure Informed Tractography) is a framework for microstructure-informed tractography filtering and quantification. Developed by Alessandro Daducci, COMMIT addresses the fundamental challenge of tractography validation by fitting the diffusion signal using tractography streamlines weighted by microstructure compartments. This approach filters false-positive connections, assigns microstructure-informed weights to streamlines, and provides biologically-interpretable metrics of white matter organization. COMMIT2 extends this with global tractography optimization.

**Key Capabilities:**
- Tractography filtering based on signal fit
- Microstructure-informed streamline weighting
- Multi-compartment models (intra-axonal, extra-axonal, isotropic)
- Removes false-positive connections
- Assigns connectivity weights based on biology
- Volume fraction estimation per streamline
- Integration with any tractography algorithm
- COMMIT2: global tractography optimization
- Reproduces diffusion signal from tractography
- Quality metrics for tractogram evaluation
- Applicable to clinical and research tractography
- Python implementation with C++ core

**Target Audience:**
- Structural connectivity researchers
- Network neuroscience (weighted connectomes)
- Tractography validation and optimization
- Clinical applications requiring high specificity
- Connectome-based biomarker studies

**Estimated Lines:** 650-700
**Estimated Code Examples:** 26-30

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install COMMIT (Python package)
   - Dependencies (DIPY, AMICO, etc.)
   - Data requirements
   - Tractography format conversion

2. **Basic COMMIT Workflow**
   - Load diffusion data and tractography
   - Build COMMIT dictionary (stick, ball, zeppelin)
   - Fit COMMIT model
   - Filter tractography based on weights
   - Extract streamline weights

3. **Multi-Compartment Models**
   - Stick (intra-axonal) compartment
   - Ball (isotropic) compartment
   - Zeppelin (extra-axonal) compartment
   - Customize compartment models
   - Interpret compartment contributions

4. **Tractography Filtering**
   - Weight-based filtering (remove zero-weight streamlines)
   - Threshold selection strategies
   - Evaluate filtering impact
   - Compare to unfiltered tractography
   - Validation against ground truth

5. **Connectivity Matrix Construction**
   - Generate weighted connectivity matrices
   - Microstructure-weighted connections
   - Compare to binary/count-based matrices
   - Integration with network analysis
   - Statistical analysis of weighted networks

6. **COMMIT2: Global Optimization**
   - Global tractography optimization
   - Differences from COMMIT
   - Computational requirements
   - Convergence criteria
   - Advanced applications

7. **Quality Assessment**
   - Signal reconstruction quality (RMSE)
   - Explained variance per voxel
   - Streamline contribution maps
   - Identify poorly-explained regions
   - Optimize tractography parameters

8. **Integration and Validation**
   - Work with MRtrix3, DIPY, DSI Studio tractography
   - Compare COMMIT-filtered to other methods
   - Validation studies
   - Clinical applications
   - Multi-subject batch processing

**Example Workflows:**
- Filter whole-brain tractography to remove false positives
- Generate microstructure-weighted connectomes
- Validate tractography quality via signal reconstruction
- Optimize tractography parameters using COMMIT feedback
- Clinical connectivity with high specificity requirements

**Integration Points:**
- **MRtrix3:** Tractography input
- **DIPY:** Tractography and modeling
- **DSI Studio:** Alternative tractography source
- **BCT/NetworkX:** Weighted network analysis
- **AMICO:** Related microstructure modeling

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **DESIGNER** (advanced preprocessing foundation)
   - **TORTOISE** (comprehensive diffusion processing)
   - **TractoFlow** (reproducible tractography pipeline)
   - **COMMIT** (microstructure-informed optimization)

2. **Comprehensive Coverage:**
   - Each skill: 650-750 lines
   - 26-32 code examples per skill
   - Real-world diffusion analysis workflows
   - Integration across preprocessing → modeling → tractography

3. **Consistent Structure:**
   - Overview and key features
   - Installation (multi-platform where applicable)
   - Basic preprocessing/processing
   - Advanced techniques and parameters
   - Quality control and validation
   - Visualization techniques
   - Batch processing and automation
   - Integration with diffusion ecosystem
   - Troubleshooting
   - Best practices
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Package/software installation
   - Dependency verification
   - Basic setup and configuration

2. **Basic Processing** (6-8)
   - Load diffusion data
   - Run basic preprocessing/analysis
   - Generate standard outputs
   - Visualize results

3. **Advanced Processing** (6-8)
   - Advanced preprocessing options
   - Parameter optimization
   - Multi-shell handling
   - Custom pipelines

4. **Quality Control** (4-6)
   - Automated QC
   - Visual inspection
   - Quantitative metrics
   - Outlier detection

5. **Integration with Ecosystem** (4-6)
   - Load/export to MRtrix3, FSL, DIPY
   - BIDS compatibility
   - Multi-tool workflows
   - Downstream connectivity analysis

6. **Visualization** (3-5)
   - Preprocessing QC plots
   - Tractography visualization
   - Quality metrics visualization
   - Publication figures

7. **Batch Processing** (3-5)
   - Multi-subject automation
   - Cluster computing
   - Containerization
   - Reproducible workflows

### Cross-Tool Integration

All skills will demonstrate integration with:
- **Data sources:** BIDS datasets, HCP, clinical acquisitions
- **Preprocessing:** DESIGNER → TORTOISE → TractoFlow pipeline
- **Modeling:** DKI, NODDI, MAPMRI
- **Tractography:** MRtrix3, DIPY, DSI Studio
- **Validation:** COMMIT filtering and weighting
- **Connectivity:** NetworkX, BCT from Batch 23
- **Visualization:** MRtrix3view, TrackVis, MI-Brain

### Quality Targets

- **Minimum lines per skill:** 650
- **Target lines per skill:** 700-750
- **Minimum code examples:** 26
- **Target code examples:** 28-32
- **Total batch lines:** ~2,700-2,900
- **Total code examples:** ~108-124

---

## Batch Statistics

### Projected Totals

| Tool | Est. Lines | Est. Examples | Priority |
|------|-----------|---------------|----------|
| DESIGNER | 700-750 | 28-32 | High |
| TORTOISE | 700-750 | 28-32 | High |
| TractoFlow | 650-700 | 26-30 | Medium-High |
| COMMIT | 650-700 | 26-30 | Medium-High |
| **TOTAL** | **2,700-2,900** | **108-124** | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Advanced preprocessing and denoising (DESIGNER)
- ✓ Comprehensive diffusion processing (TORTOISE)
- ✓ Reproducible tractography pipelines (TractoFlow)
- ✓ Microstructure-informed tractography (COMMIT)
- ✓ Multi-shell, multi-tissue analysis
- ✓ Quality control throughout pipeline
- ✓ Clinical and research applications

**Language/Platform Coverage:**
- Python: DESIGNER, TractoFlow, COMMIT (3/4)
- MATLAB: DESIGNER, TORTOISE (2/4)
- Command-line: TORTOISE, TractoFlow (2/4)
- Nextflow: TractoFlow (1/4)
- Containers: TractoFlow, DESIGNER (2/4)

**Application Areas:**
- Clinical diffusion imaging: All tools
- Research tractography: All tools
- Multi-site studies: DESIGNER, TractoFlow, COMMIT
- Microstructure imaging: DESIGNER, TORTOISE, COMMIT
- Network neuroscience: TractoFlow, COMMIT + Batch 23 tools

---

## Strategic Importance

### Fills Critical Gap

Previous batches have covered:
- Basic diffusion tools (MRtrix3, DIPY - Batch 1)
- DTI/DWI specialized analysis (TORTOISE, TractSeg - Batch 11)
- Advanced diffusion modeling (DMIPY, MDT - Batch 20)
- Network analysis (Batch 23)

**Batch 26 adds:**
- **State-of-the-art preprocessing** to maximize data quality
- **Advanced artifact removal** (Gibbs, noise, distortions)
- **Complete reproducible pipelines** for tractography
- **Microstructure-informed tractography** for biological validity
- **Quality control** specific to diffusion imaging

### Complementary Skills

**Works with existing skills:**
- **MRtrix3 (Batch 1):** DESIGNER/TractoFlow use MRtrix3 tools
- **DIPY (Batch 1):** COMMIT integrates with DIPY tractography
- **DSI Studio (Batch 11):** Alternative downstream analysis
- **DMIPY/MDT (Batch 20):** Advanced models require DESIGNER preprocessing
- **NetworkX/BCT (Batch 23):** COMMIT provides weighted connectomes

### User Benefits

1. **Data Quality:**
   - State-of-the-art denoising increases SNR substantially
   - Gibbs removal prevents artifacts in microstructure fitting
   - Proper distortion correction enables accurate anatomy

2. **Reproducibility:**
   - TractoFlow provides containerized, version-controlled pipelines
   - Automated QC flags problematic datasets
   - BIDS compatibility ensures standardization

3. **Biological Validity:**
   - COMMIT filters false-positive connections
   - Microstructure-informed weights improve network analysis
   - Reduces bias in connectivity studies

4. **Clinical Translation:**
   - TORTOISE widely used in clinical research
   - High-quality preprocessing enables biomarker discovery
   - Presurgical tractography with validation

---

## Dependencies and Prerequisites

### Software Prerequisites

**DESIGNER:**
- Python 3.7+ or MATLAB R2017+
- FSL 6.0+
- MRtrix3 3.0+
- ANTs 2.3+

**TORTOISE:**
- Linux or macOS
- FSL (for some functions)
- Matlab Runtime (if not using MATLAB)

**TractoFlow:**
- Nextflow 20.07+
- Docker or Singularity
- Java 8+
- BIDS-formatted data

**COMMIT:**
- Python 3.7+
- NumPy, SciPy, Cython
- DIPY (optional, for tractography)
- AMICO (for related modeling)

### Data Prerequisites

**Common to all:**
- Raw diffusion-weighted images (DWI)
- B-values and B-vectors
- Anatomical T1w image (for registration)
- Multi-shell data (preferred for advanced methods)

**Tool-specific:**
- **DESIGNER:** Multi-shell recommended, at least 30 directions
- **TORTOISE:** Any DWI acquisition, works with single-shell
- **TractoFlow:** BIDS-formatted DWI dataset
- **COMMIT:** Existing tractography streamlines + DWI data

### Knowledge Prerequisites

Users should understand:
- Diffusion MRI basics (b-values, gradients, FA, MD)
- Common artifacts (eddy currents, susceptibility distortion, Gibbs ringing)
- Fiber orientation and tractography concepts
- Basic neuroanatomy
- Python or MATLAB scripting
- Linux command-line basics

---

## Learning Outcomes

After completing Batch 26 skills, users will be able to:

1. **Advanced Preprocessing:**
   - Apply state-of-the-art denoising (MP-PCA)
   - Remove Gibbs ringing artifacts
   - Correct for distortions comprehensively
   - Assess preprocessing quality quantitatively

2. **Complete Diffusion Processing:**
   - Process raw DWI through complete pipelines
   - Generate DTI, DKI, and NODDI parameters
   - Perform quality control at each step
   - Troubleshoot common failures

3. **Reproducible Tractography:**
   - Run containerized, automated tractography
   - Implement anatomically-constrained tractography
   - Generate BIDS-compliant derivatives
   - Ensure reproducibility across sites and time

4. **Microstructure-Informed Analysis:**
   - Filter tractography with COMMIT
   - Generate biologically-weighted connectomes
   - Validate tractography quality
   - Integrate microstructure and connectivity

5. **Quality Assurance:**
   - Implement comprehensive QC pipelines
   - Identify and handle problematic datasets
   - Quantify preprocessing improvements
   - Document data quality

6. **Integration:**
   - Chain preprocessing → modeling → tractography
   - Integrate with network analysis (Batch 23)
   - Export to visualization tools
   - Build reproducible multi-step workflows

---

## Relationship to Existing Skills

### Builds Upon:
- **MRtrix3 (Batch 1):** DESIGNER and TractoFlow use MRtrix3 extensively
- **DIPY (Batch 1):** COMMIT integrates with DIPY tractography
- **FSL (Batch 1):** DESIGNER uses FSL for distortion correction
- **ANTs (Batch 1):** Registration for preprocessing
- **DTI-TK (Batch 11):** Tensor-based registration complements TORTOISE
- **DMIPY/MDT (Batch 20):** Advanced models require clean data from DESIGNER

### Complements:
- **TractSeg (Batch 11):** Automated bundle segmentation of COMMIT-filtered tractography
- **DSI Studio (Batch 11):** Alternative tractography for COMMIT
- **NetworkX/BCT (Batch 23):** Weighted network analysis of COMMIT connectomes
- **NBS (Batch 23):** Statistical testing of filtered connectivity

### Enables:
- High-quality structural connectomics
- Microstructure imaging with minimal artifacts
- Reproducible multi-site diffusion studies
- Clinical diffusion biomarkers
- Validated tractography for network neuroscience
- Precision medicine applications in white matter disease

---

## Expected Challenges and Solutions

### Challenge 1: Computational Resources
**Issue:** Advanced preprocessing is computationally intensive
**Solution:** Provide timing estimates, parallelization strategies, cluster examples

### Challenge 2: Software Dependencies
**Issue:** Complex dependency chains (FSL, MRtrix3, ANTs, etc.)
**Solution:** Document containerized solutions, provide installation troubleshooting

### Challenge 3: Parameter Selection
**Issue:** Many parameters to optimize for preprocessing
**Solution:** Provide recommended defaults, parameter sensitivity analysis, dataset-specific guidance

### Challenge 4: Quality Control
**Issue:** Identifying failed preprocessing steps
**Solution:** Comprehensive QC sections, visual inspection guides, quantitative thresholds

### Challenge 5: Integration Complexity
**Issue:** Chaining multiple tools can be complex
**Solution:** End-to-end workflow examples, common pitfalls section, scripting templates

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Software installation tests
   - Dependency checking
   - Version compatibility
   - Basic functionality verification

2. **Basic Functionality Tests:**
   - Process example dataset
   - Generate expected outputs
   - Visualize results
   - Compare to reference values

3. **Cross-Tool Validation:**
   - DESIGNER → MRtrix3 compatibility
   - TORTOISE → COMMIT pipeline
   - TractoFlow → Network analysis
   - Verify consistent results

4. **Example Data:**
   - Links to test datasets (HCP, PING, etc.)
   - Expected processing times
   - Reference output values
   - Interpretation guidance

---

## Timeline Estimate

**Per Skill:**
- Research and planning: 15-20 min
- Writing and examples: 45-55 min
- Review and refinement: 10-15 min
- **Total per skill:** ~70-90 min

**Total Batch 26:**
- 4 skills × 80 min average = ~320 min (~5.3 hours)
- Includes documentation, examples, and testing

**Can be completed in:** 1-2 extended sessions

---

## Success Criteria

Batch 26 will be considered successful when:

✓ All 4 skills created with 650-750 lines each
✓ Total of 108+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - Basic to advanced processing examples
  - Quality control procedures
  - Parameter optimization guidance
  - Batch processing examples
  - Integration with diffusion ecosystem
  - Cross-tool workflows
  - Troubleshooting section
  - Best practices
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 99/133 (74.4%)

---

## Next Batches Preview

### Batch 27: Lesion Analysis & Clinical Tools
- LST (Lesion Segmentation Toolbox)
- BIANCA (Brain Intensity AbNormality Classification Algorithm)
- Clinica (clinical neuroimaging platform)
- Additional clinical-focused tools

### Batch 28: ASL & Perfusion Imaging
- ASLtbx (ASL processing toolbox)
- BASIL (Bayesian ASL modeling)
- ExploreASL (ASL processing pipeline)
- Perfusion quantification tools

### Batch 29: Advanced Statistics & Machine Learning
- PRoNTo (Pattern Recognition for Neuroimaging Toolbox)
- Remaining specialized statistical methods
- Machine learning integration tools

---

## Conclusion

Batch 26 provides **comprehensive advanced diffusion preprocessing and microstructure-informed tractography** capabilities, completing the diffusion neuroimaging toolkit. By covering:

- **State-of-the-art preprocessing** (DESIGNER)
- **Comprehensive diffusion processing** (TORTOISE)
- **Reproducible tractography pipelines** (TractoFlow)
- **Microstructure-informed optimization** (COMMIT)

This batch enables researchers to:
- **Maximize diffusion data quality** through advanced preprocessing
- **Remove critical artifacts** that confound microstructure and tractography
- **Generate reproducible tractography** with containerized pipelines
- **Validate and weight tractography** based on biological plausibility
- **Build high-quality connectomes** for network neuroscience

These tools are critical for:
- Structural connectomics and network neuroscience
- Microstructure imaging (NODDI, DKI, etc.)
- Clinical diffusion biomarkers
- Multi-site harmonization
- Presurgical tractography planning
- White matter disease characterization

By providing access to cutting-edge diffusion preprocessing and tractography validation methods, Batch 26 positions users to conduct state-of-the-art research in diffusion neuroimaging with maximal data quality and biological validity.

**Status After Batch 26:** 99/133 skills (74.4% complete - approaching three-quarters!)

---

**Document Version:** 1.0
**Created:** 2025-11-17
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 4 skills, ~2,700-2,900 lines, ~108-124 examples
