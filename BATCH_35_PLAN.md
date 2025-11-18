# Batch 35 Plan: Specialized Clinical & Developmental Neuroimaging Pipelines

## Overview

**Theme:** Specialized Clinical & Developmental Neuroimaging Pipelines
**Focus:** Clinical neurodegenerative disease analysis, infant/pediatric preprocessing, and modular workflow design
**Target:** 3 new skills, 1,900-2,050 total lines

**Current Progress:** 114/133 skills (85.7%)
**After Batch 34:** 117/133 skills (87.9%)
**After Batch 35:** 120/133 skills (90.2%)

This batch addresses critical gaps in specialized neuroimaging pipelines designed for specific populations and research domains. These tools extend beyond general-purpose preprocessing to provide targeted workflows for clinical neurodegenerative disease research, infant brain development, and flexible modular pipeline construction.

## Rationale

While general preprocessing pipelines like fMRIPrep and QSIPrep are well-established, certain research domains require specialized approaches:
- **Clinical Research:** Longitudinal multi-modal analysis for Alzheimer's, Parkinson's, and other neurodegenerative diseases
- **Developmental Neuroscience:** Infant and pediatric brains require different algorithms, templates, and processing strategies
- **Custom Workflows:** Researchers need flexible, modular tools to build tailored pipelines beyond standardized approaches

This batch provides comprehensive coverage of specialized preprocessing ecosystems that complement the core pipelines.

## Skills to Create

### 1. Clinica (700-750 lines, 24-28 examples)

**Overview:**
Clinica is a software platform for clinical neuroimaging research, particularly focused on neurodegenerative diseases (Alzheimer's disease, Parkinson's disease, frontotemporal dementia). It provides standardized pipelines for processing multi-modal neuroimaging data (T1w, DWI, PET, fMRI) and extracting clinically relevant biomarkers for longitudinal studies and clinical trials.

**Key Features:**
- BIDS-compatible clinical neuroimaging pipelines
- Longitudinal analysis workflows (ADNI, AIBL, OASIS)
- Multi-modal biomarker extraction (cortical thickness, hippocampal volume, amyloid load)
- Integration with FreeSurfer, SPM, FSL, ANTs, and PETPVC
- Machine learning classification pipelines
- Statistical analysis and visualization
- Support for large clinical datasets
- Reproducible and containerized execution

**Target Audience:**
- Clinical researchers studying neurodegenerative diseases
- Alzheimer's Disease Neuroimaging Initiative (ADNI) users
- Pharmaceutical companies conducting clinical trials
- Neurologists analyzing longitudinal patient data

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to Clinica and clinical neuroimaging
   - Neurodegenerative disease biomarkers
   - Supported modalities and workflows
   - BIDS organization for clinical data
   - Citation information

2. **Installation** (80 lines)
   - Conda/pip installation
   - Docker and Singularity containers
   - Dependencies (FreeSurfer, SPM, FSL, ANTs)
   - Installing specific pipelines
   - Setting up environment variables
   - Testing installation with example data

3. **BIDS Conversion for Clinical Data** (100 lines, 3-4 examples)
   - Converting ADNI data to BIDS
   - Handling longitudinal sessions
   - Clinical metadata organization
   - Participant demographics and clinical scores
   - Example: ADNI to BIDS conversion

4. **Anatomical Pipelines** (120 lines, 4-5 examples)
   - T1w linear processing (volume-based features)
   - T1w FreeSurfer cross-sectional and longitudinal
   - Regional volume extraction (hippocampus, amygdala)
   - Cortical thickness measurements
   - Example: Longitudinal FreeSurfer for AD progression

5. **DWI Processing** (100 lines, 3-4 examples)
   - DWI preprocessing pipeline
   - DTI and NODDI modeling
   - White matter hyperintensity segmentation
   - Tractography and connectivity
   - Example: DWI biomarkers in Alzheimer's disease

6. **PET Processing** (120 lines, 4-5 examples)
   - PET-to-T1 registration
   - Partial volume correction (PETPVC)
   - SUVR quantification (amyloid, tau, FDG)
   - Regional PET analysis (Centiloid scale)
   - Example: Amyloid PET quantification in preclinical AD

7. **fMRI Processing** (90 lines, 3-4 examples)
   - Task-based fMRI preprocessing
   - Resting-state fMRI preprocessing
   - Functional connectivity analysis
   - Integration with clinical variables
   - Example: Default mode network in MCI

8. **Machine Learning Pipelines** (100 lines, 3-4 examples)
   - Feature extraction across modalities
   - SVM classification (AD vs. controls)
   - Cross-validation and model evaluation
   - Multimodal biomarker integration
   - Example: Predicting MCI-to-AD conversion

9. **Statistical Analysis** (80 lines, 2-3 examples)
   - Group-level statistics
   - Longitudinal mixed-effects models
   - Controlling for age, sex, education
   - Visualization of results
   - Example: Cortical thinning rate in AD

10. **Quality Control** (60 lines, 1-2 examples)
    - Automated QC metrics
    - Visual QC reports
    - Outlier detection
    - Multi-site harmonization considerations

11. **HPC and Batch Processing** (70 lines, 2-3 examples)
    - Running Clinica on computing clusters
    - Parallel processing strategies
    - Large cohort management (ADNI, UK Biobank)
    - Example: Processing 500 subjects on HPC

12. **Troubleshooting** (60 lines)
    - Common installation issues
    - Pipeline failures and recovery
    - Handling missing modalities
    - Memory and computational requirements
    - BIDS validation errors

13. **Best Practices** (50 lines)
    - Clinical data organization
    - Longitudinal study design
    - Multi-modal integration strategies
    - Reporting clinical biomarkers
    - Reproducibility and version control

14. **References** (20 lines)
    - Clinica publications
    - Clinical neuroimaging standards
    - Disease-specific biomarker papers

**Code Examples:**
- ADNI to BIDS conversion (bash)
- T1w-linear pipeline (bash)
- FreeSurfer longitudinal pipeline (bash)
- PET SUVR quantification (bash)
- Machine learning classification (Python)

**Integration Points:**
- FreeSurfer for cortical reconstruction
- SPM for statistical analysis
- ANTs for registration
- FSL for preprocessing
- PETPVC for PET partial volume correction
- Scikit-learn for machine learning
- BIDS for data organization

---

### 2. Nibabies (650-700 lines, 22-26 examples)

**Overview:**
Nibabies (NeuroImaging Baby BIDS Application for Infant Examination Suite) is an extension of fMRIPrep specifically designed for preprocessing infant and pediatric neuroimaging data. It addresses unique challenges of the developing brain including rapid age-related anatomical changes, smaller brain size, lower tissue contrast, and the need for age-specific templates and segmentation algorithms.

**Key Features:**
- Age-specific infant brain templates (0-24 months)
- Adapted tissue segmentation for developing brains
- Motion correction optimized for pediatric data
- Susceptibility distortion correction
- Surface reconstruction for infants
- Age-appropriate spatial normalization
- BIDS-compatible input and derivatives
- Comprehensive visual QC reports

**Target Audience:**
- Developmental neuroscientists studying early brain development
- Pediatric neurologists analyzing clinical infant imaging
- Researchers investigating neurodevelopmental disorders
- Longitudinal infant cohort studies (Baby Connectome Project, HEALthy Brain and Child Development)

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to infant neuroimaging challenges
   - Differences from adult preprocessing
   - Age-specific considerations (0-2 years)
   - Nibabies vs. fMRIPrep adaptations
   - Citation information

2. **Installation** (80 lines)
   - Docker and Singularity installation
   - FreeSurfer infant module setup
   - Python package installation
   - Age-specific template download
   - Testing with example data
   - Version compatibility

3. **BIDS Organization for Infant Data** (90 lines, 3-4 examples)
   - Organizing infant T1w/T2w data
   - Session naming for longitudinal infant studies
   - Age metadata (gestational age, postnatal age)
   - Handling sedation and sleep states
   - Example: BIDS structure for 0-24 month cohort

4. **Anatomical Preprocessing** (120 lines, 4-5 examples)
   - Infant brain extraction
   - Age-specific tissue segmentation (GM, WM, CSF)
   - T1w and T2w registration
   - Bias field correction
   - Surface reconstruction (infant-adapted)
   - Example: Preprocessing 6-month-old T1w/T2w

5. **Spatial Normalization** (100 lines, 3-4 examples)
   - Age-specific template selection
   - Infant brain atlases (UNC, NKI)
   - Non-linear registration to infant templates
   - Temporal alignment across ages
   - Example: Normalizing newborn to 2-week template

6. **fMRI Preprocessing** (110 lines, 3-4 examples)
   - Motion correction for naturally sleeping infants
   - Susceptibility distortion correction (fieldmap-based)
   - Slice-timing correction
   - Confound extraction (motion, physiological)
   - Smoothing recommendations
   - Example: Resting-state fMRI in 12-month-old

7. **Quality Control** (90 lines, 2-3 examples)
   - Visual QC reports interpretation
   - Motion assessment in infant fMRI
   - Segmentation quality metrics
   - Age-specific QC criteria
   - Example: Flagging poor-quality infant scans

8. **Common Infant-Specific Issues** (80 lines, 2-3 examples)
   - Handling motion artifacts
   - Incomplete skull coverage
   - Fontanelles and skull development
   - Low tissue contrast (especially 6-12 months)
   - Example: Troubleshooting failed brain extraction

9. **Advanced Options** (70 lines, 2-3 examples)
   - Custom infant templates
   - Multi-echo fMRI for infants
   - High-motion data strategies
   - Integration with custom workflows
   - Example: Using BCP templates for neonates

10. **Integration with Downstream Analysis** (60 lines, 1-2 examples)
    - Functional connectivity in infants
    - Cortical surface analysis
    - Integration with developmental trajectory modeling
    - Example: Infant seed-based connectivity

11. **Troubleshooting** (60 lines)
    - Segmentation failures in low-contrast ages
    - Motion-related issues
    - Template mismatch problems
    - Memory and computational requirements
    - Container-related issues

12. **Best Practices** (50 lines)
    - Acquisition protocols for infants
    - Age-appropriate quality standards
    - Longitudinal infant study design
    - Reporting infant preprocessing
    - Ethical considerations (sedation, imaging time)

13. **References** (20 lines)
    - Nibabies publications
    - Infant brain development papers
    - Infant template studies

**Code Examples:**
- Basic Nibabies command (bash)
- BIDS validation for infant data (bash)
- Anatomical-only preprocessing (bash)
- Full fMRI preprocessing (bash)
- Using custom infant templates (bash)

**Integration Points:**
- fMRIPrep (parent software)
- FreeSurfer infant module
- ANTs for registration
- FSL for preprocessing
- BIDS for data organization
- Infant brain atlases (UNC, NKI, MNI infant)

---

### 3. fMRIflows (550-600 lines, 18-22 examples)

**Overview:**
fMRIflows is a modular, flexible workflow framework for fMRI preprocessing and analysis. Unlike monolithic pipelines, it provides building blocks that users can combine to create custom workflows tailored to specific research questions, datasets, or computational environments. Built on Nipype, it emphasizes transparency, reproducibility, and adaptability.

**Key Features:**
- Modular preprocessing components (motion correction, registration, smoothing, etc.)
- Mix-and-match workflow construction
- Support for multiple software backends (FSL, SPM, AFNI, ANTs)
- Integration with fMRIPrep outputs
- Custom quality control modules
- Flexible confound regression strategies
- HPC and local execution modes
- Extensive visualization and reporting

**Target Audience:**
- Researchers needing custom preprocessing workflows
- Method developers testing new approaches
- Multi-site studies requiring harmonized but flexible pipelines
- Advanced users who want full control over preprocessing steps

**Main Sections:**

1. **Overview** (40 lines)
   - Introduction to modular workflow design
   - Advantages over monolithic pipelines
   - Nipype integration
   - Use cases for fMRIflows
   - Citation information

2. **Installation** (70 lines)
   - Python package installation
   - Docker/Singularity containers
   - Backend software dependencies (FSL, AFNI, ANTs)
   - Testing installation
   - Configuration files

3. **Basic Workflow Construction** (100 lines, 3-4 examples)
   - Understanding workflow modules
   - Building a simple preprocessing workflow
   - Connecting nodes in Nipype
   - Specifying inputs and outputs
   - Example: Minimal motion correction + smoothing workflow

4. **Preprocessing Modules** (110 lines, 4-5 examples)
   - Motion correction (MCFLIRT, AFNI 3dvolreg)
   - Slice-timing correction
   - Registration (functional to anatomical)
   - Spatial normalization
   - Smoothing and filtering
   - Example: Multi-step preprocessing workflow

5. **Integration with fMRIPrep** (80 lines, 2-3 examples)
   - Using fMRIPrep derivatives as input
   - Custom post-fMRIPrep processing
   - Combining fMRIPrep confounds with custom modules
   - Example: Task fMRI denoising after fMRIPrep

6. **Confound Regression** (90 lines, 3-4 examples)
   - Motion parameter regression
   - CompCor strategies (aCompCor, tCompCor)
   - Global signal regression
   - Scrubbing and censoring
   - Custom confound models
   - Example: Comparing different denoising strategies

7. **Quality Control Modules** (80 lines, 2-3 examples)
   - Motion QC metrics (FD, DVARS)
   - Registration quality assessment
   - Carpet plots and timeseries visualization
   - Custom QC reports
   - Example: Automated QC workflow

8. **Advanced Workflow Customization** (90 lines, 3-4 examples)
   - Iterative workflows (parameter optimization)
   - Conditional processing (different paths for different data)
   - Parallel processing across subjects
   - Integration with custom scripts
   - Example: Multi-echo fMRI with custom T2* fitting

9. **HPC Execution** (70 lines, 2-3 examples)
   - SLURM integration
   - SGE and PBS support
   - Resource management
   - Distributed processing
   - Example: Running 100 subjects on cluster

10. **Troubleshooting** (60 lines)
    - Workflow failures and debugging
    - Node connection errors
    - Backend software issues
    - Memory and disk management
    - Common Nipype errors

11. **Best Practices** (50 lines)
    - Workflow design principles
    - Reproducibility considerations
    - Version control for workflows
    - Documentation standards
    - Testing custom workflows

12. **References** (20 lines)
    - fMRIflows publications
    - Nipype documentation
    - Modular workflow design papers

**Code Examples:**
- Simple preprocessing workflow (Python)
- Custom confound regression (Python)
- fMRIPrep integration (Python)
- Quality control workflow (Python)
- HPC execution script (Python/bash)

**Integration Points:**
- Nipype for workflow management
- fMRIPrep for standardized preprocessing
- FSL, AFNI, SPM, ANTs as processing backends
- BIDS for data organization
- Custom Python analysis scripts

---

## Implementation Checklist

### Per-Skill Requirements
- [ ] 550-750 lines per skill
- [ ] 18-28 code examples per skill
- [ ] Consistent section structure
- [ ] Installation instructions for multiple platforms
- [ ] Basic usage examples
- [ ] Advanced features
- [ ] Quality control guidance
- [ ] Integration with other neuroimaging tools
- [ ] Troubleshooting section
- [ ] Best practices
- [ ] References with proper citations

### Quality Assurance
- [ ] All code examples are tested and functional
- [ ] Command syntax is accurate
- [ ] File paths use proper conventions
- [ ] Examples demonstrate real-world workflows
- [ ] Integration examples are practical
- [ ] Troubleshooting covers common issues
- [ ] References are complete and up-to-date

### Batch Requirements
- [ ] Total lines: 1,900-2,050
- [ ] Total examples: 64-76
- [ ] Consistent markdown formatting
- [ ] Proper cross-referencing between skills
- [ ] All three skills address specialized pipeline domains

## Timeline

1. **Clinica**: 700-750 lines, 24-28 examples
2. **Nibabies**: 650-700 lines, 22-26 examples
3. **fMRIflows**: 550-600 lines, 18-22 examples

**Estimated Total:** 1,900-2,050 lines, 64-76 examples

## Context & Connections

### Pipeline Specialization Framework

**Clinical Neuroimaging (Clinica):**
```
ADNI/Clinical Data → BIDS Conversion → Clinica Pipelines → Biomarkers
        ↓                   ↓                  ↓               ↓
    Longitudinal      Multi-modal      FreeSurfer/SPM   Clinical Outcomes
```

**Developmental Neuroimaging (Nibabies):**
```
Infant T1w/T2w → Age-specific Templates → Nibabies → Development Metrics
      ↓                 ↓                     ↓              ↓
  0-24 months    Infant Atlases        fMRI Preprocessing   Trajectories
```

**Custom Workflows (fMRIflows):**
```
Research Question → Modular Components → Custom Workflow → Tailored Analysis
       ↓                   ↓                    ↓                ↓
   Specific Needs    FSL/AFNI/SPM       Nipype Pipeline    Optimized Results
```

### Complementary Tools

**Already Covered:**
- **fMRIPrep**: General-purpose adult preprocessing (Nibabies parent)
- **QSIPrep**: Diffusion preprocessing (similar to Clinica DWI)
- **FreeSurfer**: Cortical reconstruction (used by Clinica and Nibabies)
- **Nipype**: Workflow engine (used by fMRIflows)
- **BIDS**: Data organization standard (all three pipelines)

**New Capabilities:**
- **Clinica**: First clinical neurodegenerative disease-focused pipeline
- **Nibabies**: First infant/pediatric-specific preprocessing pipeline
- **fMRIflows**: First fully modular, customizable preprocessing framework

### Research Applications

**Clinica Use Cases:**
- Alzheimer's disease progression studies
- Parkinson's disease biomarker development
- Frontotemporal dementia characterization
- Multi-modal clinical trial endpoints
- Longitudinal cognitive decline prediction

**Nibabies Use Cases:**
- Early brain development trajectories
- Neurodevelopmental disorder detection
- Effects of prenatal/perinatal factors
- Infant brain-behavior relationships
- Longitudinal infant cohort studies

**fMRIflows Use Cases:**
- Testing novel preprocessing strategies
- Comparing denoising methods
- Multi-site harmonization studies
- Task-specific optimized preprocessing
- Integration of experimental techniques

## Technical Specifications

### Clinica
- **Platform**: Cross-platform (Linux, macOS)
- **Language**: Python
- **Dependencies**: FreeSurfer, SPM12, FSL, ANTs, PETPVC
- **Input**: BIDS-organized T1w, T2w, DWI, PET, fMRI
- **Output**: CAPS (Clinica Analyzed Processed Structure)
- **Containers**: Docker, Singularity
- **Supported Datasets**: ADNI, AIBL, OASIS, custom clinical data

### Nibabies
- **Platform**: Cross-platform (via containers)
- **Language**: Python
- **Dependencies**: FreeSurfer (infant mode), ANTs, FSL
- **Input**: BIDS-organized infant T1w, T2w, fMRI
- **Output**: fMRIPrep-compatible derivatives
- **Templates**: UNC 0-1-2 years, NKI infant, custom
- **Age Range**: 0-24 months (primarily)
- **Containers**: Docker, Singularity

### fMRIflows
- **Platform**: Cross-platform
- **Language**: Python (Nipype-based)
- **Dependencies**: Nipype, FSL, AFNI, SPM, ANTs (flexible)
- **Input**: NIfTI, BIDS, fMRIPrep derivatives
- **Output**: User-defined (flexible)
- **Execution**: Local, HPC (SLURM, SGE, PBS)
- **Extensibility**: Custom modules and workflows

## Learning Path

### Clinical Researcher Path
1. Start with **Clinica** for ADNI/clinical cohort analysis
2. Learn longitudinal biomarker workflows
3. Integrate with clinical outcome measures

### Developmental Scientist Path
1. Master **Nibabies** for infant data preprocessing
2. Understand age-specific templates and QC
3. Apply to longitudinal developmental studies

### Methods Developer Path
1. Learn **fMRIflows** for custom workflow construction
2. Experiment with different processing strategies
3. Develop and validate novel preprocessing approaches

### Multi-Domain Researcher Path
1. Use **fMRIPrep** for general adult data
2. Apply **Nibabies** for pediatric extensions
3. Use **Clinica** for clinical populations
4. Customize with **fMRIflows** when needed

## Success Metrics

- [ ] All three skills cover complete specialized workflows
- [ ] Installation instructions work on multiple platforms
- [ ] Code examples run without errors
- [ ] Integration examples connect to existing pipelines
- [ ] Troubleshooting addresses real user issues
- [ ] Documentation enables self-directed learning
- [ ] Clinical, developmental, and custom use cases clearly explained

## Detailed Application Scenarios

### Clinica: Alzheimer's Disease Longitudinal Study

**Study Design:**
Multi-site longitudinal study tracking 500 participants (150 healthy controls, 200 MCI, 150 AD) over 3 years with annual visits.

**Workflow:**
```
Data Collection (each timepoint):
- 3T MRI: T1w, T2w, DWI, rs-fMRI
- Amyloid PET (Florbetapir)
- Tau PET (Flortaucipir)
- FDG PET
- Clinical assessments: MMSE, CDR, neuropsychological battery

Clinica Processing:
1. Convert to BIDS:
   - clinica convert adni-to-bids <input> <output>
   - Organize longitudinal sessions (ses-M00, ses-M12, ses-M24, ses-M36)

2. T1w-linear pipeline:
   - Regional volumes (hippocampus, amygdala, entorhinal cortex)
   - Tissue segmentation
   - MNI normalization
   - Clinica run t1-linear <bids> <caps>

3. FreeSurfer cross-sectional:
   - Cortical thickness maps
   - Subcortical segmentation
   - Surface-based analysis
   - Clinica run t1-freesurfer <bids> <caps>

4. FreeSurfer longitudinal:
   - Unbiased within-subject template
   - Longitudinal cortical thickness
   - Atrophy rates
   - Clinica run t1-freesurfer-longitudinal <bids> <caps>

5. DWI preprocessing:
   - Eddy current and motion correction
   - DTI model fitting (FA, MD, RD, AD)
   - NODDI fitting (ICVF, ISOVF, ODI)
   - Clinica run dwi-preprocessing-using-t1 <bids> <caps>

6. PET processing:
   - Amyloid PET SUVR (Centiloid scale):
     - clinica run pet-linear <bids> <caps> amyloid
   - Tau PET SUVR (temporal meta-ROI):
     - clinica run pet-linear <bids> <caps> tau
   - FDG PET SUVR (pons reference):
     - clinica run pet-linear <bids> <caps> fdg

7. rs-fMRI preprocessing:
   - Motion correction, registration, normalization
   - Confound regression
   - Functional connectivity (DMN, executive, salience networks)
   - Clinica run fmri-preprocessing <bids> <caps>

8. Machine learning classification:
   - Extract multi-modal features (volumes, cortical thickness, PET SUVR, DTI, FC)
   - Train SVM classifier: MCI stable vs. MCI converter
   - Cross-validation with site as stratification variable
   - Clinica run machinelearning-classification <caps> <ml_output>

Outputs:
- Longitudinal hippocampal atrophy rate: 4.2%/year in AD vs. 1.1%/year in controls
- Cortical thinning in temporal and parietal regions
- Amyloid positivity: 80% of AD, 60% of MCI, 25% of controls
- Tau accumulation in medial temporal lobe predicts conversion
- Reduced DMN connectivity in MCI and AD
- Multimodal classifier: 82% accuracy in predicting MCI-to-AD conversion

Clinical Relevance:
- Identify biomarker signatures for early detection
- Predict disease progression trajectories
- Stratify participants for clinical trials
- Monitor treatment effects in therapeutic trials
```

### Nibabies: Baby Connectome Project Analysis

**Study Design:**
Longitudinal study of 100 healthy full-term infants scanned at 1, 3, 6, 12, and 24 months.

**Workflow:**
```
Data Collection (each timepoint):
- 3T MRI during natural sleep
- T1w: 1mm isotropic MPRAGE
- T2w: 1mm isotropic TSE
- rs-fMRI: TR=2s, 10 minutes (600 volumes)
- DWI: 64 directions, b=1000 s/mm²

BIDS Organization:
sub-001/
  ses-01month/
    anat/
      sub-001_ses-01month_T1w.nii.gz
      sub-001_ses-01month_T2w.nii.gz
    func/
      sub-001_ses-01month_task-rest_bold.nii.gz
  ses-03month/
  ses-06month/
  ses-12month/
  ses-24month/

Nibabies Preprocessing (for each session):

1. Anatomical preprocessing:
   nibabies <bids_dir> <output_dir> participant \
     --participant-label 001 \
     --age-months 1 \
     --output-spaces MNIInfant:cohort-1:res-2 anat \
     --fs-license-file <license>

   Processing steps:
   - T1w and T2w co-registration
   - 1-month infant template selection (UNC 0-1-2)
   - Infant brain extraction (age-appropriate thresholds)
   - Tissue segmentation (GM, WM, CSF for developing brain)
   - Surface reconstruction (infant-adapted FreeSurfer)
   - Non-linear registration to MNI infant template

2. Functional preprocessing:
   nibabies <bids_dir> <output_dir> participant \
     --participant-label 001 \
     --age-months 6 \
     --task-id rest \
     --output-spaces MNIInfant:cohort-1 \
     --fs-license-file <license>

   Processing steps:
   - Head motion correction (infant-optimized MCFLIRT)
   - Susceptibility distortion correction (if fieldmap available)
   - Boundary-based registration to T1w
   - Resampling to infant template space
   - CompCor confound extraction
   - Motion parameter computation (FD, DVARS)

3. Quality Control:
   - Review nibabies HTML reports
   - Assess motion: exclude runs with >20% high-motion volumes (FD > 0.3mm)
   - Segmentation quality: visual inspection of GM/WM boundaries
   - Registration quality: check functional-anatomical alignment
   - Age-specific metrics: CSF volume, myelination patterns

4. Downstream Analysis (outside Nibabies):

   a) Functional connectivity development:
      - Extract timeseries from infant brain parcellations
      - Denoise with confound regression
      - Compute seed-based connectivity (sensorimotor, visual, DMN seeds)
      - Track FC changes from 1 to 24 months

   b) Surface-based cortical analysis:
      - Cortical thickness developmental trajectories
      - Surface area expansion
      - Gyrification index changes

   c) Longitudinal within-subject analysis:
      - Template creation for each infant
      - Track hippocampal volume growth
      - Cortical thickness maturation curves

Expected Results:
- Sensorimotor connectivity emerges early (1-3 months)
- Default mode network matures gradually (12-24 months)
- Cortical thickness increases then plateaus
- Regional heterochrony in gray matter development
- Individual differences predict later cognitive outcomes

Challenges Addressed:
- Low tissue contrast at 6-12 months (Nibabies uses T1w+T2w)
- High motion in awake infants (handled by robust motion correction)
- Need for age-specific templates (UNC infant atlases)
- Longitudinal template evolution (age-appropriate for each visit)
```

### fMRIflows: Custom Multi-Echo fMRI Denoising Comparison

**Research Question:**
Which multi-echo denoising strategy (TEDANA, ME-ICA, or optimal combination) provides the best sensitivity for detecting task-related activation in a motor task?

**Workflow:**
```
Dataset:
- 30 participants
- Multi-echo fMRI: 3 echoes (TE=14, 28, 42 ms), TR=2s
- Motor task: finger tapping vs. rest (5 runs per subject)
- Already preprocessed with fMRIPrep (motion correction, registration done)

fMRIflows Custom Workflow:

1. Setup modular workflow components:

from fmriflows import Workflow
from fmriflows.nodes import (
    MultiEchoMerge, TedanaDenoising, OptimalCombination,
    ConfoundRegression, Smoothing, GLMAnalysis
)
from nipype.interfaces import fsl, afni

# Create base workflow
wf = Workflow(name='multiecho_comparison', base_dir='/scratch/mecomp')

2. Define preprocessing branches:

# Branch A: TEDANA denoising
tedana_node = TedanaDenoising(
    tedpca='kundu',
    tedort=True,
    out_dir='tedana_output'
)

# Branch B: Optimal combination only (T2* weighted)
optcom_node = OptimalCombination(method='t2s')

# Branch C: Optimal combination + standard confound regression
optcom_confound_node = ConfoundRegression(
    confounds=['motion_params', 'csf', 'white_matter'],
    confounds_file='fmriprep_confounds.tsv'
)

# Branch D: Manual ME-ICA implementation
meica_node = afni.MEICA(
    tedana_is_good=True,
    mixing_model='logistic'
)

3. Connect parallel processing streams:

# Iterate over strategies
for strategy in ['tedana', 'optcom', 'optcom_confound', 'meica']:

    # Apply denoising
    wf.connect([(input_node, denoising_nodes[strategy], [
        ('echo1', 'echo_1'),
        ('echo2', 'echo_2'),
        ('echo3', 'echo_3')
    ])])

    # Optional smoothing
    smooth_node = Smoothing(fwhm=5.0)

    # First-level GLM
    glm_node = GLMAnalysis(
        design_matrix='motor_task.tsv',
        contrasts={'FingerTapping': [1, 0]}
    )

    # Connect to group-level
    wf.connect([(glm_node, group_node, [
        ('cope', f'copes_{strategy}')
    ])])

4. Quality Control Module:

qc_node = QualityControl(
    metrics=['tsnr', 'dvars', 'kappa_rho_ratio'],
    carpet_plot=True,
    output_report=True
)

# Compare QC metrics across strategies
wf.connect([(denoising_nodes, qc_node, [
    ('denoised_data', 'input_func')
])])

5. Statistical Comparison:

# Group-level analysis for each strategy
for strategy in strategies:
    group_glm = fsl.FLAMEO(
        cope_file=f'sub-*/copes_{strategy}.nii.gz',
        var_cope_file=f'sub-*/varcopes_{strategy}.nii.gz',
        design_file='group_design.mat',
        t_con_file='group_contrasts.con',
        run_mode='fe'  # Fixed effects
    )

    # Threshold
    threshold_node = fsl.Cluster(
        threshold=3.1,  # p < 0.001 uncorrected
        pthreshold=0.05,  # cluster-level FWE
        out_threshold_file=f'thresh_{strategy}.nii.gz'
    )

6. Execute on HPC:

wf.run(
    plugin='SLURM',
    plugin_args={
        'sbatch_args': '--time=24:00:00 --mem=32G --cpus-per-task=4',
        'max_jobs': 50
    }
)

7. Results Aggregation:

import pandas as pd
import matplotlib.pyplot as plt

# Extract activation statistics
results = {
    'tedana': extract_cluster_stats('thresh_tedana.nii.gz'),
    'optcom': extract_cluster_stats('thresh_optcom.nii.gz'),
    'optcom_confound': extract_cluster_stats('thresh_optcom_confound.nii.gz'),
    'meica': extract_cluster_stats('thresh_meica.nii.gz')
}

# Compare tSNR
tsnr_comparison = pd.DataFrame({
    strategy: calculate_tsnr(f'denoised_{strategy}')
    for strategy in strategies
})

# Plot results
fig, axes = plt.subplots(2, 2)
for ax, strategy in zip(axes.flat, strategies):
    plot_activation_map(f'thresh_{strategy}.nii.gz', ax=ax)
    ax.set_title(f'{strategy}: {results[strategy]["num_voxels"]} voxels')

Expected Findings:
- TEDANA removes most noise but may over-denoise subtle activations
- Optimal combination alone: highest tSNR but includes physiological noise
- Optimal combination + confound regression: good balance
- ME-ICA: similar to TEDANA but different component classification

Advantages of fMRIflows:
- Easy to implement and compare multiple strategies in parallel
- Modular components can be reused for other projects
- Transparent processing steps
- Integration with fMRIPrep outputs
- HPC scalability for parameter exploration
```

## Alternative Tools and Comparisons

### Clinica Alternatives

**FreeSurfer + Custom Scripts**
- Pros: Widely used, flexible
- Cons: Requires extensive scripting, no standardized clinical pipeline
- **Clinica advantage**: Integrated multi-modal clinical workflows, BIDS-native

**PETSurfer**
- Pros: PET-specific pipeline within FreeSurfer
- Cons: Limited to PET, less comprehensive
- **Clinica advantage**: Multi-modal integration, machine learning

**SPM12 CAT12 Longitudinal**
- Pros: Excellent longitudinal VBM
- Cons: Not designed for clinical trials, limited PET
- **Clinica advantage**: Clinical trial focus, ADNI support, multi-modal

**Custom ADNI Processing**
- Pros: Tailored to specific research question
- Cons: Labor-intensive, not standardized
- **Clinica advantage**: Standardized, validated, reproducible

### Nibabies Alternatives

**FreeSurfer Infant Module**
- Pros: Longitudinal cortical reconstruction
- Cons: Anatomical only, no fMRI preprocessing
- **Nibabies advantage**: Full fMRI preprocessing, BIDS derivatives

**Developing HCP Pipeline (dHCP)**
- Pros: High-quality neonatal processing
- Cons: Requires specific acquisition, less flexible
- **Nibabies advantage**: Works with standard clinical protocols

**Manual Infant Processing (FSL/ANTS)**
- Pros: Full control
- Cons: Time-consuming, requires expertise
- **Nibabies advantage**: Automated, standardized, validated

**BabySurfaceNet (deep learning segmentation)**
- Pros: Fast segmentation
- Cons: Segmentation only, no full preprocessing
- **Nibabies advantage**: Complete preprocessing pipeline

### fMRIflows Alternatives

**fMRIPrep + Custom Scripts**
- Pros: Standardized preprocessing + flexibility
- Cons: Requires Nipype knowledge for extensions
- **fMRIflows advantage**: Pre-built modular components

**Nipype Workflows from Scratch**
- Pros: Maximum flexibility
- Cons: Requires extensive Nipype expertise
- **fMRIflows advantage**: Ready-to-use modules, less boilerplate

**C-PAC (Configurable Pipeline)**
- Pros: Many built-in options, GUI
- Cons: Less modular, harder to customize deeply
- **fMRIflows advantage**: True modularity, easier custom workflows

**CONN Toolbox**
- Pros: Integrated preprocessing and connectivity
- Cons: MATLAB, less flexible preprocessing
- **fMRIflows advantage**: Python-based, fully modular

## Regulatory and Clinical Considerations

### Clinica

**Research Use:**
- Validated on ADNI and other clinical datasets
- Widely used in Alzheimer's disease research
- Publications demonstrate reliability

**Clinical Translation:**
- Not FDA-approved diagnostic software
- Research tool for clinical trials
- Outputs support research endpoints
- Can inform clinical decision-making (physician discretion)

**Validation:**
- Compared to established pipelines (FreeSurfer, SPM)
- Multi-site reproducibility assessed
- Used in pharma-sponsored trials

### Nibabies

**Research Use:**
- Validated on Baby Connectome Project data
- Infant brain development research
- Neurodevelopmental disorder studies

**Clinical Translation:**
- Research software, not clinical diagnostic tool
- Used in clinical research settings
- IRB approval required for pediatric imaging
- Ethical considerations (sedation, imaging duration)

**Validation:**
- Tested on multi-site infant datasets
- Age-specific QC metrics validated
- Compared to manual infant processing

### fMRIflows

**Research Use:**
- Methodological tool for custom workflows
- Transparency for reproducibility
- Supports open science practices

**Clinical Translation:**
- Research-only software
- Workflow validation user's responsibility
- Not intended for clinical diagnostics

## Sample Datasets and Learning Resources

### Clinica

**Example Datasets:**
1. **ADNI (Alzheimer's Disease Neuroimaging Initiative)**
   - URL: http://adni.loni.usc.edu/
   - Content: Longitudinal T1w, DWI, PET, clinical data
   - Size: 2000+ participants
   - Access: Application required

2. **OASIS-3 (Open Access Series of Imaging Studies)**
   - URL: https://www.oasis-brains.org/
   - Content: Longitudinal aging and AD data
   - Size: 1000+ participants
   - Access: Free with registration

3. **Clinica Test Data**
   - Included with installation
   - Small sample for testing pipelines

**Tutorials:**
- Official Clinica documentation: https://aramislab.paris.inria.fr/clinica/docs/
- Hands-on workshops at OHBM, AAIC
- Video tutorials on YouTube

**Key Publications:**
- Routier et al. (2021). "Clinica: An open-source software platform for reproducible clinical neuroscience studies." *Frontiers in Neuroinformatics*

### Nibabies

**Example Datasets:**
1. **Baby Connectome Project (BCP)**
   - URL: https://nda.nih.gov/edit_collection.html?id=2848
   - Content: Infant MRI from 0-5 years
   - Access: NIMH Data Archive application

2. **Developing Human Connectome Project (dHCP)**
   - URL: http://www.developingconnectome.org/
   - Content: Neonatal MRI (20-44 weeks gestational age)
   - Access: Open access with registration

3. **Nibabies Test Data**
   - Included with documentation
   - Example infant scans for testing

**Tutorials:**
- Official documentation: https://nibabies.readthedocs.io/
- fMRIPrep tutorials (parent software)
- OHBM educational courses

**Key Publications:**
- Goncalves et al. (in prep). "Nibabies: Preprocessing pipeline for infant neuroimaging."

### fMRIflows

**Example Datasets:**
1. **OpenNeuro Datasets**
   - URL: https://openneuro.org/
   - Content: Various fMRI datasets
   - Test custom workflows on real data

2. **fMRIPrep Example Data**
   - Use fMRIPrep outputs as starting point
   - Test post-fMRIPrep custom processing

**Tutorials:**
- Official fMRIflows documentation
- Nipype tutorials: https://nipype.readthedocs.io/
- Custom workflow examples on GitHub

**Key Publications:**
- Custom workflow methodology papers
- Nipype papers (Gorgolewski et al., 2011, 2018)

## Integration Scenarios

### Scenario 1: Multi-Pipeline Longitudinal Study

```
Study: Tracking brain changes from infancy to adulthood

Data Acquisition:
- Infants (0-2 years): T1w, T2w, rs-fMRI
- Children (2-12 years): T1w, rs-fMRI, DTI
- Adolescents/Adults (12-60 years): T1w, rs-fMRI, DTI
- Older adults (60+ years): T1w, rs-fMRI, DTI, cognitive testing

Processing Pipeline:
         ↓
Age 0-2 years → Nibabies
   - Infant-specific templates
   - T1w+T2w processing
   - Age-appropriate segmentation
         ↓
Age 2-60 years → fMRIPrep / QSIPrep
   - Standard adult-optimized processing
   - MNI152 normalization
   - Surface-based analysis
         ↓
Age 60+ with MCI/AD → Clinica
   - Neurodegenerative biomarkers
   - Longitudinal FreeSurfer
   - Clinical outcome correlation
         ↓
Custom Analysis → fMRIflows
   - Harmonize data across age groups
   - Custom connectivity metrics
   - Lifespan developmental trajectories
```

### Scenario 2: Multi-Modal Clinical Trial

```
Trial: Testing anti-amyloid therapy in preclinical AD

Participants:
- 200 amyloid-positive cognitively normal older adults
- Randomized to drug vs. placebo
- 18-month trial duration
- Assessments at baseline, 6, 12, 18 months

Clinica Workflow:

Baseline:
1. BIDS conversion:
   - clinica convert adni-to-bids
2. Anatomical processing:
   - clinica run t1-linear (regional volumes)
   - clinica run t1-freesurfer (cortical thickness)
3. Amyloid PET:
   - clinica run pet-linear (SUVR Centiloid)
4. Tau PET:
   - clinica run pet-linear (temporal meta-ROI)
5. rs-fMRI:
   - clinica run fmri-preprocessing
   - Connectivity analysis (DMN, executive, salience)
6. DWI:
   - clinica run dwi-preprocessing-using-t1
   - DTI metrics in vulnerable regions

Follow-up (6, 12, 18 months):
1. Repeat all imaging
2. Longitudinal FreeSurfer:
   - clinica run t1-freesurfer-longitudinal
   - Track cortical thinning rates
3. Compute change scores:
   - Amyloid PET Centiloid change
   - Hippocampal atrophy rate
   - DTI decline in fornix

Statistical Analysis:
1. Extract multi-modal biomarkers
2. Mixed-effects models:
   - Time × Treatment interaction
   - Control for age, sex, APOE4
3. Clinica machine learning:
   - Predict cognitive decline
   - Identify responders vs. non-responders

Primary Endpoint:
- Reduction in amyloid PET Centiloid (drug vs. placebo)

Secondary Endpoints:
- Hippocampal atrophy rate
- Cortical thickness change
- Tau PET accumulation
- Functional connectivity preservation
```

## Batch Summary Statistics

| Skill | Lines | Examples | Primary Domain | Secondary Domains |
|-------|-------|----------|----------------|-------------------|
| Clinica | 700-750 | 24-28 | Clinical Neuroimaging | Neurodegeneration, Multi-modal |
| Nibabies | 650-700 | 22-26 | Infant/Pediatric | Development, Longitudinal |
| fMRIflows | 550-600 | 18-22 | Custom Workflows | Modularity, Flexibility |
| **Total** | **1,900-2,050** | **64-76** | **Specialized Pipelines** | **Multi-disciplinary** |

## Expected Impact

### Research Community
- **Clinical Trials**: Standardized biomarker extraction for neurodegenerative disease trials
- **Developmental Science**: Rigorous infant brain preprocessing for large cohorts
- **Methods Development**: Flexible platform for testing novel preprocessing strategies

### Clinical Practice
- **Neurology**: Quantitative biomarkers for disease monitoring and treatment planning
- **Pediatrics**: Developmental trajectory assessment in clinical populations
- **Precision Medicine**: Individualized processing for optimal clinical decision-making

### Education and Training
- **Clinical Researchers**: Learn standardized clinical neuroimaging workflows
- **Developmental Scientists**: Master infant-specific preprocessing challenges
- **Methodologists**: Build and test custom neuroimaging pipelines

## Emerging Trends

### Clinical Neuroimaging
- **AI biomarkers**: Deep learning for early detection and progression prediction
- **Multi-modal fusion**: Integrating imaging with genomics, proteomics, digital biomarkers
- **Harmonization**: Pooling multi-site data for large-scale trials
- **Real-time biomarkers**: Rapid processing for clinical decision support

### Developmental Neuroimaging
- **Fetal imaging**: Extending preprocessing to in-utero data
- **Preterm infants**: Specialized templates for prematurity
- **Neurodevelopmental disorders**: Early detection of autism, ADHD
- **Precision medicine**: Individualized developmental trajectories

### Workflow Design
- **Containerization**: Reproducible pipelines across computing environments
- **Cloud computing**: Scalable processing for large datasets
- **Provenance tracking**: Complete audit trail for reproducibility
- **Modular standards**: Interoperable components across pipelines

## Conclusion

Batch 35 addresses critical gaps in neuroimaging pipelines by documenting three specialized tools:

1. **Clinica** enables standardized, reproducible clinical neuroimaging analysis for neurodegenerative disease research and clinical trials
2. **Nibabies** provides infant-specific preprocessing to rigorously analyze the rapidly developing brain from 0-24 months
3. **fMRIflows** offers modular, flexible workflow construction for researchers needing customized preprocessing beyond standardized pipelines

By completing this batch, the N_tools neuroimaging skill collection will reach **120/133 skills (90.2%)**, crossing the 90% completion threshold with comprehensive coverage extending from general-purpose tools to highly specialized clinical, developmental, and custom workflow domains.

These pipelines represent the cutting edge of population-specific and application-specific neuroimaging, ensuring researchers have the right tools for their unique datasets and scientific questions.
