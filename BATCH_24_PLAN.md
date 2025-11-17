# Batch 24: Specialized Statistical Tools - Planning Document

## Overview

**Batch Theme:** Specialized Statistical Tools
**Batch Number:** 24
**Number of Skills:** 4
**Current Progress:** 87/133 skills completed (65.4%)
**After Batch 24:** 91/133 skills (68.4%)

## Rationale

Batch 24 focuses on **advanced statistical inference methods** for neuroimaging data, providing specialized tools for permutation-based testing, surface-based statistics, and electrophysiological data analysis. This batch addresses the critical need for robust statistical methods that:

- **Handle complex experimental designs** (repeated measures, multi-modal, hierarchical)
- **Provide valid inference** when parametric assumptions are violated
- **Control for multiple comparisons** across diverse data types
- **Support specialized data structures** (cortical surfaces, time-frequency, EEG/MEG)
- **Enable rigorous statistical testing** with proper Type I error control

These tools complement parametric methods (SPM, FSL) by offering nonparametric alternatives and specialized frameworks for data types that require tailored statistical approaches.

**Key Scientific Advances:**
- Permutation testing eliminates distributional assumptions
- Surface-based statistics preserve anatomical topology
- Linear mixed-effects models for hierarchical data
- TFCE provides topology-preserving cluster inference
- Multivariate analysis across imaging modalities
- Proper statistical inference for electrophysiological data

**Applications:**
- Small sample studies where parametric assumptions fail
- Surface-based cortical thickness and curvature analysis
- Multi-modal integration (structural + functional + genetics)
- Longitudinal and repeated-measures designs
- EEG/MEG event-related potential and oscillation analysis
- Clinical trials with complex within-subject factors

---

## Tools in This Batch

### 1. PALM (Permutation Analysis of Linear Models)
**Website:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM
**GitHub:** https://github.com/andersonwinkler/PALM
**Platform:** MATLAB/Octave, Standalone
**Priority:** High
**Current Status:** Exists (743 lines) - Ready

**Overview:**
PALM is the most versatile and powerful tool for permutation-based inference in neuroimaging. Developed by Anderson Winkler at NIH, PALM extends beyond simple permutation tests to handle extraordinarily complex designs including repeated measures, multi-level hierarchies, non-exchangeable data, and multi-modal integration. PALM's flexibility makes it the go-to tool when parametric methods are inappropriate or when design complexity exceeds what standard tools can handle.

**Key Capabilities:**
- Permutation and sign-flipping inference
- Complex GLM designs (repeated measures, within-subject factors)
- Exchangeability blocks for non-independent data
- Multi-modal Canonical Correlation Analysis (mCCA)
- Threshold-Free Cluster Enhancement (TFCE)
- Variance group analysis (test for variance differences)
- Tail approximation for acceleration (millions of permutations)
- Multiple comparison correction (FWER, FDR, cluster)
- Support for volumetric, surface, and TBSS data
- Non-Parametric Combination (NPC) for joint testing
- Integration with FSL, FreeSurfer, and SPM outputs
- Robust to outliers and non-normal distributions

**Target Audience:**
- Researchers with complex experimental designs
- Small sample studies
- Multi-modal imaging studies
- Longitudinal and developmental research
- Genetic-imaging associations
- Clinical trials with repeated measures

**Estimated Lines:** 750-800 (current: 743)
**Estimated Code Examples:** 30-35

**Key Topics Covered:**
1. Installation and basic usage
2. Simple permutation tests (two-sample, one-sample, paired)
3. Complex designs (repeated measures, within-subject factors)
4. Exchangeability blocks for hierarchical data
5. Multi-modal CCA for integrated analysis
6. TFCE for cluster inference
7. Variance group analysis
8. Acceleration with tail approximation
9. Surface-based analysis (FreeSurfer integration)
10. TBSS skeleton analysis
11. Batch processing pipelines

**Current Status:** Skill exists and is comprehensive. May benefit from minor enhancements (~50-100 additional lines for advanced CCA examples and NPC).

---

### 2. SnPM (Statistical nonParametric Mapping)
**Website:** http://warwick.ac.uk/snpm
**GitHub:** https://github.com/SnPM-toolbox/SnPM-devel
**Platform:** MATLAB (SPM toolbox)
**Priority:** High
**Current Status:** Exists (619 lines) - Needs Expansion to 650-700

**Overview:**
Statistical nonParametric Mapping (SnPM) is the original and most widely-used permutation testing toolbox for SPM. Developed by Thomas Nichols and Andrew Holmes, SnPM provides a nonparametric alternative to SPM's parametric methods while maintaining full integration with the SPM ecosystem. SnPM is particularly valuable when sample sizes are small, data distributions are non-normal, or when robust inference is required without strong assumptions.

**Key Capabilities:**
- Permutation-based inference within SPM framework
- No distributional assumptions required
- Valid for any sample size (including very small n)
- Multiple experimental designs (one-sample, two-sample, paired, ANOVA, regression)
- Variance smoothing for improved sensitivity
- Pseudo t-statistics for better spatial localization
- Cluster-based inference with permutation
- Seamless integration with SPM preprocessing
- Suprathreshold cluster analysis
- Multiple comparison correction via permutation
- Compatible with SPM contrast manager
- Supports volumetric fMRI, VBM, and PET data

**Target Audience:**
- SPM users needing nonparametric inference
- Small sample neuroimaging studies
- Studies with non-normal data distributions
- Researchers wanting robust statistical methods
- Clinical studies with limited participants
- Quality-controlled inference for publications

**Estimated Lines:** 650-700 (current: 619, needs +30-80 lines)
**Estimated Code Examples:** 26-30

**Key Topics to Cover:**
1. Installation within SPM
2. Basic permutation tests (one-sample, two-sample)
3. Paired t-tests and within-subject designs
4. ANOVA and multi-group comparisons
5. Regression designs with covariates
6. Variance smoothing configuration
7. Cluster inference and extent thresholds
8. Multi-subject fMRI group analysis
9. VBM analysis with SnPM
10. Integration with SPM preprocessing
11. Result visualization and reporting
12. Batch scripting for automation

**Enhancement Needed:** Add ~50-80 lines covering:
- Advanced variance smoothing examples
- Multi-level factorial designs
- Comparison with parametric SPM results
- Detailed batch scripting examples
- Integration with CAT12 for VBM
- Publication-quality result reporting

---

### 3. SurfStat
**Website:** http://www.math.mcgill.ca/keith/surfstat/
**Platform:** MATLAB
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
SurfStat is a MATLAB toolbox specifically designed for statistical analysis of surfaces, particularly cortical surfaces from structural MRI. Developed by Keith Worsley at McGill University, SurfStat implements linear mixed-effects models and random field theory for cortical surface data. It excels at analyzing cortical thickness, surface area, curvature, and other surface-based morphometric measures while properly accounting for the topology and smoothness of cortical surfaces.

**Key Capabilities:**
- Linear mixed-effects models for surface data
- Random field theory for multiple comparison correction
- Cortical thickness and surface area analysis
- Support for FreeSurfer and CIVET surfaces
- Fixed-effects and random-effects models
- Longitudinal and repeated-measures designs
- Cluster detection with RFT-based p-values
- Surface-based smoothing (geodesic and heat kernel)
- Integration with FreeSurfer parcellations
- Visualization on inflated and pial surfaces
- Support for bilateral (left/right hemisphere) analysis
- Covariate modeling (age, sex, scanner, etc.)
- Contrast specification and inference
- Multivariate analysis across vertices

**Target Audience:**
- Cortical morphometry researchers
- FreeSurfer users needing statistical analysis
- Developmental neuroscience (longitudinal cortical development)
- Aging studies (cortical thinning analysis)
- Clinical studies (patient vs. control cortical differences)
- Surface-based fMRI analysis

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**

1. **Installation and Setup**
   - Download and install SurfStat
   - Load FreeSurfer surfaces
   - Data format requirements
   - Surface mesh structures

2. **Loading Surface Data**
   - Import FreeSurfer cortical thickness
   - Load curvature and surface area
   - Combine left and right hemispheres
   - Handle missing data

3. **Basic Linear Models**
   - Two-group comparison (patients vs. controls)
   - One-sample t-tests
   - Correlation with continuous variables
   - ANCOVA with covariates

4. **Surface Smoothing**
   - Geodesic smoothing (along surface)
   - Heat kernel smoothing
   - FWHM specification
   - Optimal smoothing for cortical thickness

5. **Statistical Inference**
   - Random field theory for correction
   - Cluster-based inference
   - Peak detection and localization
   - Effect size estimation

6. **Longitudinal Analysis**
   - Repeated-measures designs
   - Linear mixed-effects models
   - Within-subject factors
   - Age × Group interactions

7. **Visualization**
   - Plot statistical maps on surfaces
   - Inflated and pial surface rendering
   - Cluster highlighting
   - Publication-quality figures

8. **Advanced Models**
   - Multi-level random effects
   - Vertex-wise multivariate analysis
   - Integration with FreeSurfer parcellations
   - ROI-based summary statistics

9. **Batch Processing**
   - Multi-subject analysis pipelines
   - Automated contrast testing
   - Export results to tables
   - Reproducible workflows

10. **Integration and Best Practices**
    - FreeSurfer quality control
    - Covariate selection strategies
    - Multiple comparison considerations
    - Troubleshooting common issues

**Example Workflows:**
- Cross-sectional cortical thickness comparison (patients vs. controls)
- Longitudinal developmental trajectories (cortical maturation)
- Cortical thickness correlation with cognitive scores
- Age-related cortical thinning in healthy aging
- Disease progression effects on cortical morphometry

**Integration Points:**
- **FreeSurfer:** Primary source of surface data and parcellations
- **CIVET:** Alternative surface generation pipeline
- **BrainSpace:** Surface-based gradient analysis
- **ENIGMA:** Large-scale cortical analysis protocols
- **CAT12:** Surface-based morphometry

---

### 4. LIMO EEG
**Website:** https://github.com/LIMO-EEG-Toolbox/limo_tools
**GitHub:** https://github.com/LIMO-EEG-Toolbox/limo_tools
**Platform:** MATLAB (EEGLAB plugin)
**Priority:** Medium-High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
LIMO EEG (LInear MOdeling of EEG) is a comprehensive toolbox for hierarchical linear modeling of electrophysiological data (EEG/MEG). Developed by Cyril Pernet and colleagues, LIMO implements advanced statistical methods specifically designed for the temporal and spectral characteristics of EEG/MEG data. It handles single-trial analysis, massive univariate testing, and complex experimental designs while properly controlling for multiple comparisons across electrodes, time points, and frequencies.

**Key Capabilities:**
- Hierarchical linear models for EEG/MEG data
- Single-trial analysis (avoiding trial averaging artifacts)
- Massive univariate testing (electrode × time × frequency)
- Cluster-based permutation testing
- Threshold-Free Cluster Enhancement (TFCE) for EEG
- Multiple comparison correction (FWE, FDR, cluster)
- Integration with EEGLAB preprocessing
- Event-related potential (ERP) analysis
- Time-frequency analysis (ERSP, ITC)
- Repeated-measures and factorial designs
- Covariate effects (age, reaction time, etc.)
- Group-level random-effects inference
- Source-space analysis support
- Robust regression methods
- Support for continuous and categorical predictors

**Target Audience:**
- EEG/MEG researchers
- Cognitive neuroscience (ERPs, oscillations)
- Clinical electrophysiology
- EEGLAB users needing advanced statistics
- Single-trial analysis researchers
- Multi-level experimental designs

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install LIMO as EEGLAB plugin
   - Verify dependencies
   - Configure for first use
   - Integration with EEGLAB

2. **Data Preparation**
   - Import EEGLAB datasets
   - Organize single-trial data
   - Define experimental design
   - Set up design matrix

3. **First-Level Analysis**
   - Single-subject linear models
   - ERP analysis (electrode × time)
   - Time-frequency analysis (electrode × time × frequency)
   - Model specification and estimation

4. **Group-Level Analysis**
   - Combine subjects for random-effects
   - One-sample t-tests
   - Two-sample comparisons
   - ANOVA designs

5. **Statistical Inference**
   - Cluster-based permutation testing
   - TFCE for EEG data
   - Multiple comparison correction
   - Threshold selection strategies

6. **Covariate Analysis**
   - Continuous predictors (age, reaction time)
   - Categorical factors (condition, group)
   - Interaction effects
   - Nuisance regression

7. **Time-Frequency Analysis**
   - Event-related spectral perturbation (ERSP)
   - Inter-trial coherence (ITC)
   - Phase-amplitude coupling
   - Statistical testing across frequencies

8. **Visualization**
   - Topographic maps over time
   - Time-frequency plots
   - Cluster visualization
   - Effect size maps

9. **Advanced Models**
   - Repeated-measures within subjects
   - Mixed-effects hierarchical models
   - Robust regression for outliers
   - Multi-factor designs

10. **Batch Processing**
    - Automated first-level analysis
    - Group-level pipelines
    - Result export and reporting
    - Reproducible workflows

**Example Workflows:**
- ERP component analysis (N170, P300, N400)
- Group differences in neural oscillations
- Correlation between ERPs and behavior
- Development of ERP responses across age
- Clinical vs. control EEG differences

**Integration Points:**
- **EEGLAB:** Preprocessing and data management
- **FieldTrip:** Alternative EEG/MEG analysis
- **Brainstorm:** Source reconstruction integration
- **MNE-Python:** Cross-platform validation
- **SPM:** Statistical methods adaptation

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **PALM** - Already complete (minor enhancements optional)
   - **SnPM** - Expand existing skill (+50-80 lines)
   - **SurfStat** - Create comprehensive new skill (700-750 lines)
   - **LIMO EEG** - Create comprehensive new skill (700-750 lines)

2. **Comprehensive Coverage:**
   - Each skill: 650-750 lines
   - 26-32 code examples per skill
   - Real-world neuroimaging workflows
   - Integration with preprocessing pipelines

3. **Consistent Structure:**
   - Overview and key features
   - Installation (MATLAB/Octave/Standalone)
   - Basic statistical tests
   - Advanced experimental designs
   - Multiple comparison correction methods
   - Visualization techniques
   - Batch processing workflows
   - Integration with neuroimaging tools
   - Troubleshooting common issues
   - Best practices for robust inference
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Package installation
   - Verification
   - Basic configuration

2. **Basic Statistical Tests** (6-8)
   - One-sample, two-sample t-tests
   - Paired comparisons
   - Simple regression
   - Basic visualizations

3. **Complex Experimental Designs** (6-8)
   - Repeated measures
   - Factorial designs (ANOVA)
   - Mixed effects models
   - Hierarchical structures
   - Covariate inclusion

4. **Multiple Comparison Correction** (4-6)
   - FWER control methods
   - FDR procedures
   - Cluster-based inference
   - TFCE applications
   - Random field theory

5. **Modality-Specific Applications** (4-6)
   - Surface-based analysis (SurfStat)
   - ERP/oscillation analysis (LIMO)
   - Multi-modal integration (PALM)
   - Variance analysis (PALM)

6. **Visualization** (3-5)
   - Statistical maps
   - Surface rendering (SurfStat)
   - Topographic plots (LIMO)
   - Time-series plots
   - Publication figures

7. **Batch Processing** (3-5)
   - Multi-subject pipelines
   - Automated workflows
   - Result aggregation
   - Reproducible scripts

### Cross-Tool Integration

All skills will demonstrate integration with:
- **Preprocessing:** SPM, FSL, fMRIPrep, FreeSurfer, EEGLAB
- **Data formats:** NIfTI, GIFTI, CIFTI, .set (EEGLAB)
- **Atlases:** AAL, Desikan-Killiany, Destrieux, HCP
- **Visualization:** FSLeyes, FreeView, EEGLAB topoplot
- **Validation:** Cross-tool comparison where applicable

### Quality Targets

- **Minimum lines per skill:** 650
- **Target lines per skill:** 700-750
- **Minimum code examples:** 26
- **Target code examples:** 28-32
- **Total batch lines:** ~2,750-3,000
- **Total code examples:** ~112-128

---

## Batch Statistics

### Projected Totals

| Tool | Est. Lines | Est. Examples | Priority | Status |
|------|-----------|---------------|----------|---------|
| PALM | 750-800 | 30-35 | High | Exists (743) |
| SnPM | 650-700 | 26-30 | High | Exists (619, expand) |
| SurfStat | 700-750 | 28-32 | High | Create new |
| LIMO EEG | 700-750 | 28-32 | Medium-High | Create new |
| **TOTAL** | **2,800-3,000** | **112-129** | - | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Permutation-based inference (PALM, SnPM)
- ✓ Surface-based statistics (SurfStat)
- ✓ Electrophysiological analysis (LIMO EEG)
- ✓ Complex experimental designs (all tools)
- ✓ Multiple comparison correction (all tools)
- ✓ Linear mixed-effects models (SurfStat, LIMO)
- ✓ Multi-modal integration (PALM)

**Platform Coverage:**
- MATLAB: All tools (4/4)
- Octave: PALM (1/4)
- Standalone: PALM (1/4)
- SPM integration: SnPM (1/4)
- EEGLAB integration: LIMO EEG (1/4)

**Application Areas:**
- Structural MRI: PALM, SnPM, SurfStat
- Functional MRI: PALM, SnPM
- Diffusion MRI: PALM
- Cortical surfaces: SurfStat
- EEG/MEG: LIMO EEG
- Multi-modal: PALM

---

## Strategic Importance

### Fills Critical Gap

Previous batches have covered:
- Core preprocessing and analysis platforms
- Connectivity and network analysis
- Parametric statistical methods (SPM, FSL built-ins)

**Batch 24 adds:**
- **Permutation-based inference** when parametric assumptions fail
- **Surface-based statistics** for cortical morphometry
- **Complex design handling** (repeated measures, hierarchies)
- **Electrophysiological statistics** for EEG/MEG
- **Robust inference** for small samples and non-normal data

### Complementary Skills

**Works with existing skills:**
- **FreeSurfer (Batch 1):** Provides surface data for SurfStat
- **SPM (Batch 1):** Integrated with SnPM
- **EEGLAB (Batch 9):** Provides data for LIMO EEG
- **FSL (Batch 1):** PALM designed for FSL outputs
- **fMRIPrep (Batch 5):** Preprocessed data for group statistics

### User Benefits

1. **Robust Statistical Inference:**
   - Valid inference without distributional assumptions
   - Proper Type I error control for complex designs
   - Handles small sample sizes appropriately

2. **Specialized Data Types:**
   - Cortical surface analysis with proper topology
   - Single-trial EEG/MEG analysis
   - Multi-modal data integration

3. **Complex Designs:**
   - Repeated measures and longitudinal studies
   - Multi-level hierarchical models
   - Within-subject and between-subject factors

4. **Publication Quality:**
   - Rigorous statistical methods accepted by reviewers
   - Proper multiple comparison correction
   - Transparent and reproducible analyses

---

## Dependencies and Prerequisites

### Software Prerequisites

**PALM:**
- MATLAB R2010b+ or Octave 3.8+
- FSL (optional, for preprocessing)
- FreeSurfer (optional, for surfaces)

**SnPM:**
- MATLAB R2012b+
- SPM12
- Statistics and Machine Learning Toolbox (recommended)

**SurfStat:**
- MATLAB R2013a+
- FreeSurfer (for surface generation)
- Statistics and Machine Learning Toolbox

**LIMO EEG:**
- MATLAB R2016a+
- EEGLAB 14.0+
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox

### Data Prerequisites

**Common to all:**
- Preprocessed neuroimaging data
- Subject demographics and covariates
- Experimental design specification

**Tool-specific:**
- **PALM:** Any format (NIfTI, GIFTI, CSV for non-imaging)
- **SnPM:** SPM-compatible NIfTI files
- **SurfStat:** FreeSurfer surface files (.thickness, .area, etc.)
- **LIMO EEG:** EEGLAB .set files with epoch information

### Knowledge Prerequisites

Users should understand:
- General linear model basics
- Multiple comparison problem
- Permutation testing concepts
- Experimental design (repeated measures, factorial)
- Basic MATLAB programming
- Domain-specific knowledge (EEG for LIMO, surfaces for SurfStat)

---

## Learning Outcomes

After completing Batch 24 skills, users will be able to:

1. **Perform Robust Statistical Tests:**
   - Implement permutation-based inference
   - Handle complex experimental designs
   - Choose appropriate correction methods
   - Validate assumptions

2. **Analyze Specialized Data:**
   - Cortical thickness and surface morphometry
   - Single-trial EEG/MEG responses
   - Multi-modal imaging integration
   - Time-frequency representations

3. **Handle Complex Designs:**
   - Repeated-measures within subjects
   - Hierarchical multi-level models
   - Factorial designs with interactions
   - Covariate effects and confounds

4. **Control Multiple Comparisons:**
   - FWER control with permutation
   - FDR procedures
   - Cluster-based inference
   - TFCE applications
   - Random field theory

5. **Integrate with Workflows:**
   - Combine preprocessing and statistical testing
   - Automate batch processing
   - Visualize and report results
   - Ensure reproducibility

---

## Relationship to Existing Skills

### Builds Upon:
- **SPM (Batch 1):** SnPM integrates with SPM
- **FreeSurfer (Batch 1):** SurfStat analyzes FreeSurfer outputs
- **EEGLAB (Batch 9):** LIMO integrates with EEGLAB
- **FSL (Batch 1):** PALM designed for FSL workflows
- **fMRIPrep (Batch 5):** Preprocessed data for group stats

### Complements:
- **NBS (Batch 23):** Network-based permutation statistics
- **BrainStat (Batch 22):** Alternative surface-based stats
- **CAT12 (Batch 8):** Surface-based morphometry preprocessing
- **CONN (Batch 13):** Functional connectivity for PALM

### Enables:
- Rigorous statistical inference for publications
- Small sample neuroimaging studies
- Longitudinal and developmental research
- Multi-modal data integration
- Cortical morphometry analysis
- EEG/MEG cognitive neuroscience

---

## Expected Challenges and Solutions

### Challenge 1: Understanding Permutation Theory
**Issue:** Users may not grasp when/why permutations are valid
**Solution:** Clear explanations, flowcharts for method selection, comparison with parametric methods

### Challenge 2: Complex Design Specification
**Issue:** Exchangeability blocks and design matrices can be confusing
**Solution:** Multiple worked examples, templates for common designs, troubleshooting guide

### Challenge 3: Computational Demands
**Issue:** Permutation tests can be very slow
**Solution:** Guidance on acceleration (tail approximation, parallelization), realistic test counts

### Challenge 4: Surface Data Format
**Issue:** Surface file formats and vertex correspondence
**Solution:** Detailed FreeSurfer integration examples, data loading templates

### Challenge 5: EEG-Specific Concepts
**Issue:** Single-trial analysis and design matrices unfamiliar to EEG users
**Solution:** EEG-specific examples, EEGLAB integration workflows, conceptual explanations

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Software availability checks
   - Version checking
   - Dependency validation

2. **Basic Functionality Tests:**
   - Simple permutation test examples
   - Known results verification
   - Comparison with parametric methods where applicable

3. **Complex Design Examples:**
   - Repeated-measures validation
   - Mixed-effects model examples
   - Multi-modal integration tests

4. **Example Data:**
   - Links to public datasets
   - Sample analysis scripts
   - Expected results and interpretations

---

## Timeline Estimate

**Per Skill:**
- PALM: Already complete (optional 30-60 min for enhancements)
- SnPM: Expansion 30-40 min
- SurfStat: New creation 60-75 min
- LIMO EEG: New creation 60-75 min

**Total Batch 24:**
- ~2.5-4 hours total
- Can be completed in 1-2 extended sessions

---

## Success Criteria

Batch 24 will be considered successful when:

✓ All 4 skills created/updated with 650-750 lines each
✓ Total of 112+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - Basic to advanced statistical examples
  - Complex experimental design workflows
  - Multiple comparison correction methods
  - Visualization examples
  - Batch processing templates
  - Integration with preprocessing tools
  - Troubleshooting section
  - Best practices for inference
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 91/133 (68.4%)

---

## Next Batches Preview

### Batch 25: Quality Control & Validation
- MRIQC (automated quality metrics)
- Visual QC tools
- Quality assessment frameworks
- Artifact detection methods

### Batch 26: Advanced Diffusion Methods
- DESIGNER (comprehensive diffusion preprocessing)
- Advanced microstructure models
- Diffusion simulation tools
- Specialized tractography methods

### Batch 27: Multivariate & Machine Learning
- PyMVPA (multivariate pattern analysis)
- Nilearn estimators
- PRoNTo (pattern recognition)
- Decoding and encoding models

---

## Conclusion

Batch 24 provides **specialized statistical inference** capabilities for neuroimaging and electrophysiology, completing the advanced statistics toolkit. By covering:

- **Permutation-based inference** (PALM, SnPM)
- **Surface-based statistics** (SurfStat)
- **EEG/MEG linear modeling** (LIMO EEG)
- **Complex experimental designs** (all tools)

This batch enables researchers to:
- **Perform robust statistical tests** without parametric assumptions
- **Handle complex designs** with repeated measures and hierarchies
- **Analyze specialized data** (surfaces, EEG, multi-modal)
- **Control Type I errors** properly across modalities
- **Publish rigorous results** with accepted statistical methods

These tools are critical for:
- Small sample neuroimaging studies
- Cortical morphometry research
- Cognitive EEG/MEG experiments
- Longitudinal and developmental studies
- Multi-modal data integration
- Clinical trials with complex designs

By providing access to the most powerful and flexible statistical tools, Batch 24 positions users to conduct rigorous, publication-quality research across diverse neuroimaging and electrophysiological applications.

**Status After Batch 24:** 91/133 skills (68.4% complete - over two-thirds!)

---

**Document Version:** 1.0
**Created:** 2025-11-17
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 4 skills, ~2,800-3,000 lines, ~112-129 examples
