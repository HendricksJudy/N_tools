# Batch 30: Physiological Noise Correction & Advanced fMRI Preprocessing - Planning Document

## Overview

**Batch Theme:** Physiological Noise Correction & Advanced fMRI Preprocessing
**Batch Number:** 30
**Number of Skills:** 3
**Current Progress:** 104/133 skills completed (78.2%)
**After Batch 30:** 107/133 skills (80.5%)

## Rationale

Batch 30 focuses on **advanced physiological noise correction and specialized fMRI preprocessing** techniques that go beyond standard pipelines to improve data quality and sensitivity. While tools like fMRIPrep provide excellent general preprocessing, these specialized tools address specific sources of noise that can significantly impact functional connectivity, task activation detection, and statistical power. They enable:

- **Multi-echo fMRI denoising** leveraging TE-dependent vs TE-independent signals
- **Physiological noise modeling** from cardiac and respiratory fluctuations
- **Low-frequency oscillation detection** and systemic noise removal
- **Improved sensitivity** in functional connectivity and task fMRI
- **Data quality enhancement** beyond motion correction and smoothing
- **Integration** with existing preprocessing pipelines
- **Validation** of BOLD signal authenticity

**Key Scientific Advances:**
- Multi-echo ICA for automatic denoising without manual classification
- Rapid detection of physiological signals in BOLD data
- Model-based removal of cardiac and respiratory artifacts
- Improved signal-to-noise ratio in functional imaging
- Better separation of BOLD vs non-BOLD components
- Enhanced detection of neural signals

**Applications:**
- Multi-echo fMRI acquisition and analysis
- Resting-state functional connectivity (noise reduction)
- Task fMRI with improved sensitivity
- High-field fMRI (7T, 9.4T) with enhanced artifacts
- Clinical fMRI with motion-prone populations
- Presurgical mapping requiring high sensitivity
- Pharmacological fMRI with subtle BOLD changes

---

## Tools in This Batch

### 1. tedana
**Website:** https://tedana.readthedocs.io/
**GitHub:** https://github.com/ME-ICA/tedana
**Platform:** Python (cross-platform)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
tedana (TE-Dependent ANAlysis) is a Python library for denoising multi-echo functional MRI data using independent component analysis. Based on the ME-ICA (Multi-Echo Independent Component Analysis) framework, tedana leverages the different T2* decay rates of BOLD (neural) vs non-BOLD (artifact) signals to automatically classify and remove noise components. This approach significantly improves tSNR and functional connectivity detection without requiring physiological recordings.

**Key Capabilities:**
- Multi-echo fMRI preprocessing and combination
- TE-dependent vs TE-independent component classification
- Automatic denoising without manual component selection
- T2*/S0 mapping from multi-echo data
- Improved temporal SNR (tSNR) over single-echo
- BIDS-compliant input/output
- Integration with fMRIPrep outputs
- Denoising metrics and quality reports
- Manual component classification option
- Optimal echo combination algorithms
- Component selection algorithms (kundu, minimal, manual)
- Comprehensive HTML reports with component visualizations
- Python API for custom workflows

**Target Audience:**
- Researchers using multi-echo fMRI acquisition
- Resting-state connectivity researchers
- Task fMRI researchers wanting improved sensitivity
- Clinical neuroimagers in motion-prone populations
- High-field MRI users (7T+)
- Anyone wanting automatic, principled denoising

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install via pip/conda
   - Verify installation
   - Dependencies (nilearn, nibabel, scikit-learn)
   - BIDS compatibility

2. **Multi-Echo Data Requirements**
   - Multi-echo acquisition parameters
   - Recommended TEs (minimum 3 echoes)
   - Data structure and organization
   - BIDS format for multi-echo

3. **Basic Workflow**
   - Run tedana on multi-echo data
   - Specify echo times
   - Output directory structure
   - Denoised timeseries output

4. **Echo Combination**
   - Optimal combination (OC) algorithm
   - T2* and S0 map generation
   - PAID (Posse) method
   - ME-DN (denoised) method

5. **Component Classification**
   - TE-dependent (BOLD) vs TE-independent (noise)
   - Kundu decision tree (default)
   - Minimal classification
   - Manual classification mode
   - Component metrics (kappa, rho)

6. **Quality Reports**
   - HTML report interpretation
   - Component visualization
   - Metrics plots
   - Denoising efficacy assessment

7. **Integration with fMRIPrep**
   - Use fMRIPrep preprocessed multi-echo data
   - Apply tedana to fMRIPrep outputs
   - BIDS derivatives structure
   - Combine with confound regression

8. **Advanced Options**
   - Custom component selection
   - Low-motion vs high-motion datasets
   - T2* fitting approaches
   - Memory optimization for large datasets

9. **Quality Control**
   - tSNR improvement assessment
   - Component classification validation
   - Compare denoised vs raw connectivity
   - Edge case handling

10. **Statistical Analysis**
    - Use denoised data in GLM
    - Functional connectivity with tedana outputs
    - Compare with single-echo analysis
    - Power analysis and sensitivity improvements

11. **Batch Processing**
    - Process multiple subjects
    - BIDS App usage
    - HPC submission scripts
    - Aggregate QC metrics

12. **Troubleshooting**
    - Common errors and solutions
    - Suboptimal component classification
    - Memory issues
    - Multi-echo acquisition optimization

**Example Workflows:**
- Denoise resting-state fMRI for improved connectivity
- Task fMRI with enhanced activation detection
- High-field (7T) fMRI artifact removal
- Clinical populations with motion artifacts
- Pharmacological fMRI with subtle BOLD changes

---

### 2. RapidTide
**Website:** https://rapidtide.readthedocs.io/
**GitHub:** https://github.com/bbfrederick/rapidtide
**Platform:** Python (cross-platform)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
RapidTide is a suite of Python tools for detecting, characterizing, and removing physiological noise sources in fMRI data, with particular emphasis on systemic low-frequency oscillations (sLFOs) related to blood flow and respiration. Unlike traditional approaches requiring external physiological recordings, RapidTide can detect these signals directly from the BOLD data itself using cross-correlation and spectral analysis. This enables retrospective cleaning of data and identification of physiological confounds.

**Key Capabilities:**
- Rapid detection of systemic low-frequency oscillations
- Time-delay mapping of physiological signals
- Automatic regressor generation from BOLD data
- No external physiological recordings required
- Cardiac and respiratory signal detection
- Blood arrival time mapping
- Improved functional connectivity after denoising
- Integration with existing preprocessing pipelines
- Voxel-wise delay maps
- Significance testing for delays
- Multiple filtering options
- GLM-based noise removal
- Quality metrics and visualizations

**Target Audience:**
- Resting-state fMRI researchers
- Researchers without physiological recordings
- Clinical imaging without cardiac/respiratory monitoring
- Cerebrovascular researchers (blood arrival timing)
- Anyone with low-frequency noise contamination
- Multi-site studies with inconsistent physio recording

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install via pip/conda
   - Verify installation
   - Command-line tools overview
   - Python API usage

2. **Physiological Noise Sources**
   - Systemic low-frequency oscillations (sLFOs)
   - Cardiac pulsation effects
   - Respiratory effects
   - Blood arrival time variations
   - Global signal contamination

3. **Basic Workflow**
   - Run rapidtide on fMRI data
   - Specify input parameters
   - Generate delay maps
   - Extract regressors

4. **Delay Mapping**
   - Time-delay calculation
   - Delay maps interpretation
   - Significance testing
   - Physiological interpretation (arterial, venous)

5. **Regressor Generation**
   - Automatic probe regressor creation
   - Voxel-wise regressors
   - Optimal regressor extraction
   - Integration with GLM

6. **Denoising Application**
   - Apply regressors to remove noise
   - GLM-based cleaning
   - Before/after comparison
   - Connectivity improvement assessment

7. **Advanced Filtering**
   - Frequency band specification
   - Cardiac vs respiratory separation
   - Multiple frequency components
   - Custom filter design

8. **Quality Metrics**
   - Signal-to-noise improvement
   - Correlation strength maps
   - Significance maps
   - Validation metrics

9. **Comparison with Other Methods**
   - vs Global Signal Regression (GSR)
   - vs CompCor
   - vs ICA-AROMA
   - Complementary usage

10. **Integration with Preprocessing**
    - Use with fMRIPrep outputs
    - Combine with motion correction
    - Order of operations
    - Multi-stage denoising

11. **Batch Processing**
    - Process multiple subjects
    - Automated workflows
    - HPC submission
    - Quality control aggregation

12. **Advanced Applications**
    - Cerebrovascular assessment
    - Blood arrival time analysis
    - Multi-echo integration
    - Real-time applications

**Example Workflows:**
- Retrospective physiological noise removal
- Resting-state connectivity enhancement
- Blood arrival time mapping for cerebrovascular studies
- Multi-site data harmonization (physio-denoising)
- Task fMRI improvement without external recordings

---

### 3. PhysIO (SPM PhysIO Toolbox)
**Website:** https://www.tnu.ethz.ch/en/software/tapas/documentations/physio-toolbox
**GitHub:** https://github.com/translationalneuromodeling/tapas
**Platform:** MATLAB (SPM toolbox)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
The PhysIO Toolbox (part of the TAPAS suite) is a comprehensive MATLAB/SPM toolbox for model-based physiological noise correction in fMRI. PhysIO creates nuisance regressors from cardiac and respiratory recordings (or estimate them from the data) using the RETROICOR method and variants. It handles multiple physiological recording formats, provides extensive quality control, and integrates seamlessly with SPM for statistical analysis. This toolbox represents the gold standard for model-based physiological noise correction when recordings are available.

**Key Capabilities:**
- RETROICOR (RETROspective Image-based CORrection)
- Respiratory volume per time (RVT) regressors
- Heart rate variability (HRV) regressors
- Multiple physiological recording format support
- Automatic peak detection (cardiac, respiratory)
- Manual peak correction GUI
- Multiple vendor support (Siemens, Philips, GE)
- Integration with SPM first-level analysis
- Comprehensive quality control plots
- Slice-timing correction integration
- Support for simultaneous multi-slice (SMS)
- Model order optimization
- Batch processing via SPM batch system
- Extensive documentation and tutorials

**Target Audience:**
- SPM users with physiological recordings
- Researchers with Siemens/Philips/GE physio logs
- Task fMRI researchers wanting maximum sensitivity
- Clinical fMRI requiring robust denoising
- High-field fMRI users
- Precision fMRI and layer fMRI researchers

**Estimated Lines:** 650-700
**Estimated Code Examples:** 22-26

**Key Topics to Cover:**

1. **Installation and Setup**
   - Download TAPAS toolbox
   - Add to MATLAB path
   - SPM integration
   - Verify installation

2. **Physiological Recording Formats**
   - Siemens DICOM physiological logs
   - Philips SCANPHYSLOG files
   - GE physiological data
   - Custom text formats
   - BIDS physiological recordings

3. **Basic Workflow**
   - Load physiological data
   - Synchronize with fMRI acquisition
   - Create RETROICOR regressors
   - Save regressors for SPM

4. **RETROICOR Model**
   - Cardiac phase regressors
   - Respiratory phase regressors
   - Fourier expansion orders
   - Model order selection
   - Interaction terms

5. **Additional Regressors**
   - Respiratory Volume per Time (RVT)
   - Heart Rate Variability (HRV)
   - Global signal regressors
   - Motion parameters integration

6. **Peak Detection and QC**
   - Automatic cardiac peak detection
   - Automatic respiratory peak detection
   - Manual correction GUI
   - Quality control plots
   - Validation metrics

7. **Slice Timing Considerations**
   - Slice-specific regressors
   - Simultaneous multi-slice (SMS)
   - Integration with SPM slice-timing
   - Temporal upsampling

8. **SPM Integration**
   - Add regressors to SPM design matrix
   - First-level GLM with PhysIO
   - Batch processing in SPM
   - Compare with/without physio correction

9. **Quality Control**
   - Physio recording quality assessment
   - Synchronization verification
   - Regressor effectiveness
   - Before/after comparison

10. **Advanced Options**
    - Custom model specifications
    - Multi-echo fMRI
    - High temporal resolution data
    - Real-time applications

11. **Batch Processing**
    - SPM batch system integration
    - Process multiple subjects
    - Automated QC
    - Script-based workflows

12. **Comparison and Validation**
    - vs data-driven methods (ICA-AROMA, tedana)
    - vs RapidTide
    - When to use model-based vs data-driven
    - Best practices

**Example Workflows:**
- Model-based physiological noise correction for task fMRI
- High-field fMRI denoising with cardiac/respiratory logs
- Layer fMRI with precision denoising
- Simultaneous multi-slice acquisition preprocessing
- Multi-site studies with standardized physio correction

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **tedana** - Multi-echo denoising (new, 700-750 lines)
   - **RapidTide** - Data-driven physiological noise detection (new, 700-750 lines)
   - **PhysIO** - Model-based physiological correction (new, 650-700 lines)

2. **Comprehensive Coverage:**
   - Each skill: 650-750 lines
   - 22-28 code examples per skill
   - Real-world denoising workflows
   - Integration with existing preprocessing pipelines

3. **Consistent Structure:**
   - Overview and key features
   - Installation (pip/conda/MATLAB)
   - Physiological noise background
   - Basic workflow execution
   - Advanced denoising techniques
   - Quality control procedures
   - Integration with fMRIPrep/SPM
   - Batch processing
   - Comparison with alternative methods
   - Troubleshooting
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Package installation
   - Dependency verification
   - Basic testing

2. **Basic Denoising** (6-8)
   - Single subject processing
   - Default parameters
   - Output interpretation
   - Quality assessment

3. **Advanced Denoising** (6-8)
   - Custom parameters
   - Multi-stage workflows
   - Optimization strategies
   - Special cases (high-motion, clinical)

4. **Quality Control** (3-5)
   - Visual QC
   - Quantitative metrics
   - Before/after comparison
   - Validation approaches

5. **Integration** (3-5)
   - fMRIPrep integration
   - SPM integration
   - Combine multiple methods
   - Statistical analysis

6. **Batch Processing** (2-4)
   - Multi-subject workflows
   - Automated QC
   - HPC scripts
   - BIDS Apps

7. **Analysis and Comparison** (2-4)
   - Connectivity analysis
   - Task activation analysis
   - Method comparison
   - Sensitivity analysis

### Cross-Tool Integration

All skills will demonstrate integration with:
- **fMRIPrep (Batch 5):** Preprocessed data as input
- **AFNI/FSL/SPM:** Statistical analysis pipelines
- **Nilearn (Batch 4):** Connectivity and visualization
- **Conn (existing):** Functional connectivity toolbox
- **ICA-AROMA (existing):** Complementary denoising
- **MRIQC (existing):** Quality assessment

### Quality Targets

- **Minimum lines per skill:** 650
- **Target lines per skill:** 650-750
- **Minimum code examples:** 22
- **Target code examples:** 22-28
- **Total batch lines:** ~2,050-2,200
- **Total code examples:** ~70-80

---

## Batch Statistics

### Projected Totals

| Tool | Est. Lines | Est. Examples | Priority | Status |
|------|-----------|---------------|----------|---------|
| tedana | 700-750 | 24-28 | High | Create new |
| RapidTide | 700-750 | 24-28 | High | Create new |
| PhysIO | 650-700 | 22-26 | High | Create new |
| **TOTAL** | **2,050-2,200** | **70-82** | - | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Multi-echo fMRI denoising (tedana)
- ✓ Data-driven physiological noise detection (RapidTide)
- ✓ Model-based physiological correction (PhysIO)
- ✓ Systemic low-frequency oscillation removal (RapidTide)
- ✓ RETROICOR methodology (PhysIO)
- ✓ Cardiac/respiratory noise modeling (all tools)
- ✓ No external recordings required (tedana, RapidTide)
- ✓ External recordings supported (PhysIO)

**Platform Coverage:**
- Python: tedana, RapidTide (2/3)
- MATLAB: PhysIO (1/3)
- Command-line: All tools (3/3)
- GUI: PhysIO (1/3 - peak correction)
- BIDS compatible: tedana, RapidTide (2/3)

**Application Areas:**
- Multi-echo fMRI: tedana
- Resting-state fMRI: All tools
- Task fMRI: All tools
- High-field fMRI: All tools
- Clinical imaging: All tools
- Without physio recordings: tedana, RapidTide
- With physio recordings: PhysIO

---

## Strategic Importance

### Fills Critical Gap

Previous batches have covered:
- fMRIPrep (Batch 5): General preprocessing
- ICA-AROMA (existing): Motion-related ICA denoising
- CONN (existing): Functional connectivity with denoising options
- AFNI/FSL/SPM: Basic preprocessing and denoising

**Batch 30 adds:**
- **Multi-echo specific denoising** (tedana)
- **Physiological signal detection** without recordings (RapidTide)
- **Model-based physiological correction** (PhysIO/RETROICOR)
- **Advanced noise characterization** beyond motion
- **Improved sensitivity** for subtle BOLD signals
- **Data-driven vs model-based** comparison capabilities

### Complementary Skills

**Works with existing skills:**
- **fMRIPrep (Batch 5):** Enhanced preprocessing pipeline
- **CONN (existing):** Improved connectivity analysis
- **AFNI/FSL/SPM:** Enhanced first-level analysis
- **Nilearn (Batch 4):** Better connectivity estimation
- **MRIQC (existing):** Quality assessment integration
- **ICA-AROMA (existing):** Complementary denoising

### User Benefits

1. **Improved Data Quality:**
   - Higher temporal SNR
   - Better sensitivity to neural signals
   - Reduced false positives from artifacts

2. **Methodological Flexibility:**
   - Multi-echo vs single-echo approaches
   - Data-driven vs model-based methods
   - Choose optimal approach for data type

3. **Retrospective Analysis:**
   - Improve existing datasets
   - No physiological recordings required (some methods)
   - Enhance legacy data

4. **Research Applications:**
   - Subtle BOLD effects (pharmacological fMRI)
   - High-field artifacts (7T, 9.4T)
   - Clinical populations (motion, compliance)
   - Precision mapping (layer fMRI, presurgical)

---

## Dependencies and Prerequisites

### Software Prerequisites

**tedana:**
- Python 3.8+
- nibabel, nilearn, scikit-learn
- numpy, scipy, pandas

**RapidTide:**
- Python 3.7+
- numpy, scipy, nibabel
- matplotlib for visualizations

**PhysIO:**
- MATLAB R2017a or later
- SPM12
- TAPAS toolbox

### Data Prerequisites

**tedana:**
- Multi-echo fMRI (minimum 3 echoes)
- Echo times (TE) values
- Preprocessed or raw multi-echo data
- BIDS format recommended

**RapidTide:**
- fMRI timeseries (single or multi-echo)
- Preferably preprocessed (motion corrected)
- No external physiological recordings required

**PhysIO:**
- fMRI data
- Cardiac and respiratory recordings
- Synchronization information (scan start trigger)
- Supported file formats (Siemens, Philips, GE, or custom)

### Knowledge Prerequisites

Users should understand:
- fMRI physics and BOLD signal
- Sources of physiological noise (cardiac, respiratory)
- T2* decay and multi-echo acquisition
- Basic preprocessing concepts
- GLM and confound regression
- Quality control principles

---

## Learning Outcomes

After completing Batch 30 skills, users will be able to:

1. **Apply Multi-Echo Denoising:**
   - Acquire and process multi-echo fMRI
   - Use tedana for automatic denoising
   - Interpret component classifications
   - Assess denoising efficacy

2. **Detect Physiological Noise:**
   - Use RapidTide for noise detection
   - Interpret delay maps
   - Generate data-driven regressors
   - Remove systemic oscillations

3. **Model Physiological Signals:**
   - Apply RETROICOR with PhysIO
   - Create cardiac/respiratory regressors
   - Integrate with SPM analysis
   - Quality control physiological data

4. **Choose Optimal Methods:**
   - Compare multi-echo vs single-echo denoising
   - Select data-driven vs model-based approaches
   - Understand when to use each method
   - Combine complementary techniques

5. **Enhance Analysis Quality:**
   - Improve functional connectivity detection
   - Increase task activation sensitivity
   - Reduce false positives from artifacts
   - Validate denoising effectiveness

---

## Relationship to Existing Skills

### Builds Upon:
- **fMRIPrep (Batch 5):** Basic preprocessing foundation
- **Nilearn (Batch 4):** Analysis and connectivity
- **SPM/AFNI/FSL:** Statistical analysis frameworks
- **MRIQC (existing):** Quality assessment

### Complements:
- **ICA-AROMA (existing):** Motion-related denoising
- **CONN (existing):** Connectivity analysis
- **Nilearn (Batch 4):** Functional connectivity estimation
- **XCPEngine (existing):** Comprehensive denoising pipelines

### Enables:
- High-quality functional connectivity analysis
- Sensitive task activation detection
- Clinical fMRI in challenging populations
- Multi-echo fMRI research
- High-field fMRI studies
- Pharmacological fMRI
- Precision fMRI (layer-specific, presurgical)

---

## Expected Challenges and Solutions

### Challenge 1: Multi-Echo Acquisition Learning Curve
**Issue:** Users may not be familiar with multi-echo acquisition
**Solution:** Provide acquisition parameter guidance, explain TE selection, show benefits quantitatively

### Challenge 2: Method Selection Confusion
**Issue:** Multiple denoising approaches can be confusing
**Solution:** Decision trees, comparison tables, clear use-case guidelines

### Challenge 3: Physiological Recording Availability
**Issue:** Not all sites collect cardiac/respiratory data
**Solution:** Emphasize tedana and RapidTide for retrospective analysis

### Challenge 4: Integration Complexity
**Issue:** Combining multiple denoising steps can be complex
**Solution:** Clear workflow diagrams, order of operations guidance, example pipelines

### Challenge 5: Validation Uncertainty
**Issue:** Difficult to know if denoising helped or hurt
**Solution:** Multiple QC metrics, before/after comparisons, validation approaches

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Software version checking
   - Dependency validation
   - Test data processing

2. **Basic Functionality Tests:**
   - Single subject denoising
   - Output verification
   - Quality metric extraction

3. **Integration Tests:**
   - fMRIPrep integration
   - Statistical analysis workflow
   - Connectivity analysis

4. **Example Data:**
   - Links to multi-echo datasets (OpenNeuro)
   - Example physiological recordings
   - Expected improvements quantified

---

## Timeline Estimate

**Per Skill:**
- tedana: 70-85 min (new, multi-echo focus)
- RapidTide: 70-85 min (new, delay mapping complexity)
- PhysIO: 60-75 min (new, MATLAB/SPM toolbox)

**Total Batch 30:**
- ~3.5-4 hours total
- Can be completed in 1-2 extended sessions

---

## Success Criteria

Batch 30 will be considered successful when:

✓ All 3 skills created with 650-750 lines each
✓ Total of 70+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - Physiological noise background
  - Complete denoising workflows
  - Quality control procedures
  - Integration with preprocessing pipelines
  - Before/after comparison examples
  - Batch processing templates
  - Method comparison guidance
  - Troubleshooting section
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 107/133 (80.5% - passed 80%!)

---

## Next Batches Preview

### Batch 31: Deep Learning for Neuroimaging
- TorchIO (medical image preprocessing for PyTorch)
- PyRadiomics (radiomics feature extraction)
- NeuroHarmonize (deep learning harmonization)

### Batch 32: Perfusion & ASL Imaging
- ExploreASL (ASL processing pipeline)
- BASIL (Bayesian ASL analysis from FSL)
- ASL-specific quantification tools

### Batch 33: Real-time fMRI & Neurofeedback
- OpenNFT (real-time neurofeedback)
- Real-time processing pipelines
- Neurofeedback experimental design

---

## Conclusion

Batch 30 provides **advanced physiological noise correction and specialized fMRI preprocessing** capabilities that significantly enhance data quality beyond standard pipelines. By covering:

- **tedana** - Multi-echo ICA denoising
- **RapidTide** - Data-driven physiological noise detection
- **PhysIO** - Model-based RETROICOR correction

This batch enables researchers to:
- **Improve data quality** through advanced denoising
- **Choose optimal methods** for their acquisition and research question
- **Enhance sensitivity** for subtle BOLD signals
- **Work retrospectively** with existing data (no physio recordings required for some methods)
- **Validate results** through multiple complementary approaches
- **Apply cutting-edge methods** in multi-echo and physiological denoising

These tools are critical for:
- Multi-echo fMRI research
- Resting-state functional connectivity
- Task fMRI requiring maximum sensitivity
- High-field fMRI (7T+)
- Clinical populations with artifacts
- Pharmacological fMRI
- Precision fMRI (layer-specific, presurgical mapping)

By providing data-driven (tedana, RapidTide) and model-based (PhysIO) approaches, Batch 30 equips users with comprehensive tools for physiological noise correction, enabling more robust and sensitive neuroimaging research.

**Status After Batch 30:** 107/133 skills (80.5% complete - passed 80% milestone!)

---

**Document Version:** 1.0
**Created:** 2025-11-17
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 3 skills, ~2,050-2,200 lines, ~70-82 examples
