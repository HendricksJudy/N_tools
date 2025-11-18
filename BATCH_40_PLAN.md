# Batch 40 Plan: GLM & Statistical Extensions

## Overview

**Theme:** GLM & Statistical Extensions
**Focus:** Advanced statistical modeling and functional connectivity extensions
**Target:** 2 skills, 1,000-1,100 lines

**Current Progress:** 125/133 skills (93.9%)
**After Batch 39:** 125/133 skills (93.9%)
**After Batch 40:** 127/133 skills (95.5%)

This batch addresses advanced statistical modeling for neuroimaging, focusing on general linear models (GLM) and functional connectivity extensions. These tools provide sophisticated statistical analysis capabilities essential for task-based and resting-state fMRI studies.

## Rationale

Statistical analysis is the foundation of neuroimaging research:

- **GLM Framework:** Essential for task-based fMRI analysis
- **First-level Analysis:** Within-subject modeling of BOLD responses
- **Second-level Analysis:** Group-level statistical inference
- **Resting-State Extensions:** Advanced functional connectivity metrics
- **Clinical Applications:** Enhanced biomarker discovery

This batch provides comprehensive coverage of statistical modeling tools that complement existing preprocessing and analysis pipelines.

## Skills to Create

### 1. Nilearn GLM (500-550 lines, 18-20 examples)

**Overview:**
Nilearn's GLM module provides a complete framework for statistical analysis of fMRI data in Python. It implements both first-level (single-subject) and second-level (group-level) general linear models with support for various experimental designs, confound regression, and statistical inference. Unlike MATLAB-based alternatives (SPM, FSL), Nilearn GLM offers a pure Python solution with excellent integration into scientific Python workflows, making it ideal for reproducible research and custom analysis pipelines.

**Key Features:**
- First-level GLM for single-subject analysis
- Second-level GLM for group statistics
- Flexible design matrix construction
- Canonical HRF and derivatives
- Parametric modulation
- Confound regression (motion, physiological)
- Contrast computation and inference
- Multiple comparison correction (FWE, FDR, cluster)
- Statistical maps visualization
- Integration with fMRIPrep outputs

**Target Audience:**
- Researchers performing task-based fMRI analysis
- Python users seeking alternatives to SPM/FSL
- Scientists building custom analysis pipelines
- Educators teaching fMRI statistics
- Developers creating neuroimaging applications

**Main Sections:**

1. **Overview** (40 lines)
   - Introduction to GLM in fMRI
   - Nilearn GLM architecture
   - Advantages over MATLAB tools
   - Applications and use cases
   - Citation information

2. **Installation** (50 lines)
   - Installing Nilearn with GLM dependencies
   - Verifying installation
   - Required data formats (NIfTI, TSV)
   - Example dataset download

3. **First-Level Analysis** (100 lines, 4-5 examples)
   - Design matrix construction
   - Event-related designs
   - Block designs
   - Mixed designs
   - Confound regression
   - Example: Task fMRI GLM
   - Example: Design matrix visualization

4. **HRF Modeling** (70 lines, 2-3 examples)
   - Canonical HRF
   - Temporal derivatives
   - Dispersion derivatives
   - FIR models
   - Example: HRF comparison

5. **Contrast Estimation** (80 lines, 3-4 examples)
   - T-contrasts
   - F-contrasts
   - Conjunction analysis
   - Example: Multiple contrasts

6. **Second-Level Analysis** (90 lines, 3-4 examples)
   - One-sample t-test
   - Two-sample t-test
   - Paired t-test
   - ANOVA designs
   - Covariates and nuisance variables
   - Example: Group analysis

7. **Statistical Inference** (80 lines, 2-3 examples)
   - Multiple comparison correction
   - Family-wise error (FWE)
   - False discovery rate (FDR)
   - Cluster-level inference
   - Example: Thresholding statistical maps

8. **Integration with fMRIPrep** (60 lines, 1-2 examples)
   - Loading fMRIPrep outputs
   - Using confound files
   - Complete preprocessing-to-statistics workflow

9. **Advanced Features** (50 lines, 1-2 examples)
   - Parametric modulation
   - Temporal filtering in GLM
   - High-pass filtering
   - AR(1) autocorrelation

10. **Troubleshooting** (40 lines)
    - Design matrix singularity
    - Collinearity issues
    - Memory management
    - Numerical stability

11. **Best Practices** (30 lines)
    - Design efficiency
    - Confound selection
    - Statistical power
    - Result interpretation

12. **References** (20 lines)
    - Nilearn GLM papers
    - fMRI statistics literature
    - HRF modeling

**Code Examples:**
- Design matrix creation (Python)
- First-level GLM (Python)
- Second-level analysis (Python)
- Statistical thresholding (Python)
- fMRIPrep integration (Python)

**Integration Points:**
- fMRIPrep for preprocessing
- Nilearn for visualization
- NiBabel for data I/O
- Pandas for design matrices
- NumPy/SciPy for computation

---

### 2. RESTplus (500-550 lines, 18-20 examples)

**Overview:**
RESTplus is an enhanced version of the Resting-State fMRI Data Analysis Toolkit (REST), providing advanced tools for analyzing resting-state functional connectivity and brain activity. RESTplus extends the original REST toolbox with additional metrics for dynamic functional connectivity, amplitude of low-frequency fluctuation (ALFF/fALFF), regional homogeneity (ReHo), degree centrality (DC), and voxel-mirrored homotopic connectivity (VMHC). It includes a user-friendly GUI and command-line interface, making it accessible for both interactive analysis and batch processing.

**Key Features:**
- Extended resting-state metrics beyond basic REST
- Dynamic functional connectivity (sliding window)
- ALFF/fALFF enhancements
- ReHo with multiple neighborhood sizes
- Degree centrality (weighted/unweighted)
- VMHC (voxel-mirrored homotopic connectivity)
- Global signal regression options
- Seed-based correlation analysis
- Preprocessing integration with DPARSF
- MATLAB and standalone versions
- Batch processing capabilities

**Target Audience:**
- Researchers analyzing resting-state fMRI
- Clinical researchers studying brain disorders
- Users of original REST/DPABI toolbox
- Neuroscientists studying functional connectivity dynamics
- Researchers requiring GUI-based analysis

**Main Sections:**

1. **Overview** (40 lines)
   - Introduction to RESTplus
   - Evolution from REST/DPABI
   - New features and enhancements
   - Applications in research
   - Citation information

2. **Installation** (60 lines)
   - MATLAB version installation
   - Standalone version (no MATLAB required)
   - Dependencies (SPM12)
   - GUI launch
   - Testing installation

3. **Preprocessing Integration** (70 lines, 2-3 examples)
   - DPARSF preprocessing pipeline
   - Nuisance regression strategies
   - Global signal regression
   - Example: Preprocessing configuration

4. **ALFF/fALFF Analysis** (80 lines, 3-4 examples)
   - Amplitude of low-frequency fluctuation
   - Fractional ALFF
   - Frequency band selection
   - Z-score standardization
   - Example: ALFF computation
   - Example: Group comparison

5. **Regional Homogeneity (ReHo)** (70 lines, 2-3 examples)
   - Kendall's coefficient of concordance
   - Neighborhood definitions (7, 19, 27 voxels)
   - Smoothing considerations
   - Example: ReHo analysis

6. **Degree Centrality** (80 lines, 3-4 examples)
   - Weighted degree centrality
   - Binary degree centrality
   - Correlation thresholds
   - Example: DC mapping
   - Example: Hub identification

7. **Voxel-Mirrored Homotopic Connectivity** (70 lines, 2-3 examples)
   - VMHC methodology
   - Symmetric templates
   - Interhemispheric connectivity
   - Example: VMHC computation

8. **Dynamic Functional Connectivity** (80 lines, 3-4 examples)
   - Sliding window analysis
   - Window length and step size
   - Variability measures
   - Example: Dynamic FC analysis
   - Example: Temporal dynamics

9. **Seed-Based Analysis** (60 lines, 2-3 examples)
   - ROI-to-whole-brain correlation
   - Multiple seed regions
   - Fisher's z-transformation
   - Example: Seed-based FC

10. **Batch Processing** (50 lines, 1-2 examples)
    - Scripting multiple subjects
    - Parameter files
    - Parallel processing

11. **Statistical Analysis** (50 lines, 1-2 examples)
    - Group-level statistics
    - Multiple comparison correction
    - Integration with SPM

12. **Troubleshooting** (40 lines)
    - Memory issues with large datasets
    - Parameter selection
    - Result interpretation
    - Common errors

13. **Best Practices** (30 lines)
    - Metric selection
    - Preprocessing choices
    - Statistical considerations
    - Quality control

14. **References** (20 lines)
    - RESTplus publications
    - Original REST/DPABI papers
    - Resting-state metrics literature

**Code Examples:**
- ALFF computation (MATLAB)
- ReHo analysis (MATLAB)
- Degree centrality (MATLAB)
- VMHC analysis (MATLAB)
- Batch processing (MATLAB)

**Integration Points:**
- DPARSF for preprocessing
- SPM12 for spatial processing
- REST/DPABI ecosystem
- GRETNA for network analysis
- Statistical analysis tools

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
- [ ] All code examples functional
- [ ] Syntax accuracy
- [ ] Real-world workflows
- [ ] Practical integration
- [ ] Clear explanations
- [ ] Common issues covered
- [ ] Complete references

### Batch Requirements
- [ ] Total lines: 1,000-1,100
- [ ] Total examples: 36-40
- [ ] Consistent markdown formatting
- [ ] Cross-referencing
- [ ] Statistical analysis focus

## Timeline

1. **Nilearn GLM**: 500-550 lines, 18-20 examples
2. **RESTplus**: 500-550 lines, 18-20 examples

**Estimated Total:** 1,000-1,100 lines, 36-40 examples

## Context & Connections

### Statistical Framework

**Task-Based Analysis (Nilearn GLM):**
```
fMRI Data → Preprocessing → Design Matrix → GLM → Contrasts → Statistical Maps
    ↓           ↓              ↓           ↓        ↓            ↓
  BOLD      fMRIPrep      Events/HRF  Regression  t/F-tests   Inference
```

**Resting-State Extensions (RESTplus):**
```
Resting fMRI → Preprocessing → Metrics → Statistical Analysis → Biomarkers
      ↓             ↓            ↓              ↓                  ↓
    BOLD        DPARSF    ALFF/ReHo/DC    Group Comparison    Clinical Apps
```

### Complementary Tools

**Already Covered:**
- **Nilearn**: Data loading and visualization (uses GLM module)
- **REST**: Original resting-state toolkit (RESTplus extends this)
- **DPABI**: Preprocessing for resting-state (integrates with RESTplus)
- **fMRIPrep**: Preprocessing pipeline (outputs work with Nilearn GLM)

**New Capabilities:**
- **Nilearn GLM**: Pure Python statistical modeling
- **RESTplus**: Enhanced resting-state metrics

## Expected Impact

### Research Community
- Python-based task fMRI analysis workflow
- Advanced resting-state biomarkers
- Reproducible statistical pipelines
- Dynamic connectivity analysis

### Clinical Applications
- Biomarker discovery for brain disorders
- Pre-surgical functional mapping
- Treatment response prediction
- Disease progression tracking

### Education
- Teaching fMRI statistics in Python
- Understanding GLM framework
- Resting-state metrics interpretation

## Conclusion

Batch 40 addresses statistical analysis by documenting two essential tools:

1. **Nilearn GLM** enables comprehensive fMRI statistical analysis in Python
2. **RESTplus** provides enhanced resting-state connectivity metrics

By completing this batch, the N_tools collection will reach **127/133 skills (95.5%)**, with comprehensive coverage of both task-based and resting-state statistical methods.

These tools bridge the gap between preprocessing and scientific interpretation, enabling researchers to extract meaningful patterns from neuroimaging data.
