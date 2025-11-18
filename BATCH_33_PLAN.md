# Batch 33 Plan: Clinical & Surgical Neuroimaging

## Overview

**Theme:** Clinical & Surgical Neuroimaging
**Focus:** Deep brain stimulation targeting, surgical planning, and lesion-symptom mapping
**Target:** 3 new skills, 2,050-2,200 total lines

**Current Progress:** 111/133 skills (83.5%)
**After Batch 33:** 114/133 skills (85.7%)

This batch fills critical gaps in clinical and surgical neuroimaging tools, focusing on applications in neurosurgery (DBS targeting), general medical imaging (surgical planning), and clinical neuroscience (lesion analysis). These tools bridge research and clinical practice.

## Rationale

While research-focused neuroimaging tools are well-represented, clinical applications require specialized software for:
- **Neurosurgical Planning:** DBS electrode placement, trajectory optimization
- **Surgical Visualization:** Multi-modal image fusion, 3D rendering, surgical guidance
- **Lesion Analysis:** Voxel-based lesion-symptom mapping (VLSM), clinical outcome prediction

This batch provides comprehensive coverage of clinical neuroimaging workflows.

## Skills to Create

### 1. Lead-DBS (700-750 lines, 24-28 examples)

**Overview:**
Lead-DBS is a MATLAB toolbox for deep brain stimulation (DBS) electrode localization, visualization, and connectivity analysis. It's the gold standard for reconstructing DBS electrode positions and predicting clinical outcomes based on structural and functional connectivity.

**Key Features:**
- DBS electrode reconstruction from CT/MRI
- Automated and manual electrode localization
- VTA (Volume of Tissue Activated) modeling
- Connectivity-based DBS targeting
- Group analysis and sweet spot mapping
- Integration with multiple brain atlases
- Support for multiple DBS systems (Medtronic, Boston Scientific, Abbott)

**Target Audience:**
- Neurosurgeons planning DBS procedures
- Researchers studying DBS mechanisms
- Clinical neurologists optimizing DBS parameters

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to Lead-DBS and DBS concepts
   - Electrode types and manufacturers
   - Citation information

2. **Installation** (80 lines)
   - MATLAB installation and requirements
   - SPM12 dependency setup
   - Lead-DBS download and configuration
   - Atlas installation
   - Docker/standalone versions

3. **Basic Electrode Reconstruction** (120 lines, 4-5 examples)
   - Importing pre-DBS MRI and post-DBS CT
   - Automated electrode detection
   - Manual refinement of electrode positions
   - Multi-modal coregistration
   - Normalization to MNI space
   - Example: Single subject electrode localization

4. **Advanced Electrode Localization** (100 lines, 3-4 examples)
   - Manual correction tools
   - Directional lead reconstruction
   - Bilateral electrode handling
   - Quality control checks
   - Example: Directional DBS lead (Boston Scientific Cartesia)

5. **Volume of Tissue Activated (VTA) Modeling** (100 lines, 3-4 examples)
   - VTA simulation methods (SimBio, OSS-DBS)
   - Stimulation parameter configuration
   - Electric field visualization
   - Overlap with anatomical structures
   - Example: VTA for STN-DBS at clinical parameters

6. **Connectivity Analysis** (120 lines, 4-5 examples)
   - Structural connectivity (normative connectomes)
   - Functional connectivity (resting-state fMRI)
   - Discriminative fiber tracking
   - Connectivity-based prediction of outcomes
   - Example: Predicting tremor improvement from connectivity

7. **Group Analysis & Sweet Spot Mapping** (100 lines, 3-4 examples)
   - Creating group projects
   - Clinical outcome correlation
   - Statistical mapping (sweet spots vs. sour spots)
   - Cross-validation
   - Example: STN sweet spot for Parkinson's motor improvement

8. **Integration with Atlases** (80 lines, 2-3 examples)
   - Available atlases (DISTAL, Horn2017, Ewert2017)
   - Custom atlas import
   - Probabilistic vs. deterministic atlases
   - Visualization options

9. **Troubleshooting** (70 lines)
   - Common coregistration issues
   - Metal artifacts in CT
   - Normalization failures
   - Connectivity data preparation
   - MATLAB memory issues

10. **Best Practices** (60 lines)
    - Pre-operative MRI acquisition
    - Post-operative CT protocols
    - Quality control workflow
    - Reporting standards

11. **References** (20 lines)
    - Key Lead-DBS papers
    - DBS targeting literature
    - Connectivity resources

**Code Examples:**
- Basic electrode reconstruction (MATLAB)
- Directional lead localization (MATLAB)
- VTA modeling with custom parameters (MATLAB)
- Structural connectivity analysis (MATLAB)
- Sweet spot mapping script (MATLAB)

**Integration Points:**
- SPM for image processing
- FSL for preprocessing
- ANTs for normalization
- Connectome Workbench for visualization
- FreeSurfer for cortical parcellations

---

### 2. 3D Slicer (700-750 lines, 24-28 examples)

**Overview:**
3D Slicer is a free, open-source platform for medical image analysis, visualization, and surgical planning. It provides a comprehensive environment for multi-modal image fusion, segmentation, registration, and quantitative analysis with extensive plugin support.

**Key Features:**
- Multi-modal 3D/4D visualization
- Image segmentation (manual, semi-automatic, automatic)
- Image registration and fusion
- Surgical planning and guidance
- Tractography and DTI analysis
- Radiotherapy planning
- Extensive extension ecosystem
- Python scripting API

**Target Audience:**
- Neurosurgeons planning resections
- Radiologists performing quantitative analysis
- Researchers developing imaging pipelines
- Clinical engineers building custom workflows

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to 3D Slicer capabilities
   - Application domains
   - Extension ecosystem
   - Citation information

2. **Installation** (100 lines)
   - Download and installation (Windows, Mac, Linux)
   - Extension Manager setup
   - Python console configuration
   - Data loading and DICOM import
   - SlicerMorph, SlicerDMRI, SlicerRT extensions

3. **Basic Visualization** (100 lines, 3-4 examples)
   - Loading DICOM and NIfTI data
   - 3D volume rendering
   - Multi-planar reconstruction (MPR)
   - Window/level adjustment
   - Example: Multi-modal visualization (T1, T2, FLAIR)

4. **Image Segmentation** (120 lines, 4-5 examples)
   - Manual segmentation (Segment Editor)
   - Thresholding and region growing
   - Level tracing and painting
   - Smoothing and island removal
   - 3D model creation from segmentations
   - Example: Brain tumor segmentation

5. **Image Registration** (100 lines, 3-4 examples)
   - Linear registration (affine, rigid)
   - Non-linear registration (BSpline)
   - Registration quality assessment
   - Transform application
   - Example: CT-to-MRI registration for surgical planning

6. **Surgical Planning** (120 lines, 4-5 examples)
   - Multi-modal fusion (MRI + CT + PET)
   - Trajectory planning
   - Distance measurements and annotations
   - Vascular segmentation and rendering
   - Tumor resection planning
   - Example: Neurosurgical resection planning with vessel avoidance

7. **Advanced Modules** (100 lines, 3-4 examples)
   - SlicerDMRI: Tractography visualization
   - SlicerRT: Radiotherapy planning
   - SegmentStatistics: Quantitative analysis
   - MarkupsToModel: Surgical guides
   - Example: DTI tractography with tumor overlay

8. **Python Scripting** (100 lines, 3-4 examples)
   - Accessing Slicer Python console
   - Loading and manipulating volumes
   - Automated segmentation workflows
   - Batch processing
   - Custom module development
   - Example: Automated volume measurement script

9. **Integration & Data Export** (80 lines, 2-3 examples)
   - DICOM export
   - STL export for 3D printing
   - NIfTI conversion
   - Integration with neuroimaging pipelines
   - Example: Exporting surgical plan to neuronavigation system

10. **Troubleshooting** (70 lines)
    - Memory management for large datasets
    - DICOM import issues
    - Registration failures
    - Extension conflicts
    - Performance optimization

11. **Best Practices** (60 lines)
    - Organizing multi-modal data
    - Segmentation validation
    - Registration QC
    - Version control for extensions
    - Documentation standards

12. **References** (20 lines)
    - 3D Slicer publications
    - Extension documentation
    - Tutorial resources

**Code Examples:**
- Loading and visualizing multi-modal data (Python)
- Automated tumor segmentation (Python)
- Registration workflow (Python)
- Batch volume measurements (Python)
- Creating 3D models from segmentations (Python)

**Integration Points:**
- FreeSurfer for cortical parcellation
- FSL for preprocessing
- ANTs for registration
- SlicerDMRI for diffusion analysis
- DICOM servers for clinical integration

---

### 3. NiiStat (650-700 lines, 22-26 examples)

**Overview:**
NiiStat is a statistical analysis package for neuroimaging designed for simplicity and reproducibility. It specializes in voxel-based lesion-symptom mapping (VLSM), region-of-interest analysis, and clinical neuroimaging statistics, with emphasis on lesion studies and stroke research.

**Key Features:**
- Voxel-based lesion-symptom mapping (VLSM)
- Mass univariate statistics with proper corrections
- Non-parametric Liebermeister test
- Brunner-Munzel rank-order tests
- Integration with clinical data
- Automated quality control
- False discovery rate (FDR) correction

**Target Audience:**
- Clinical researchers studying stroke and brain lesions
- Cognitive neurologists mapping lesion-deficit relationships
- Rehabilitation scientists predicting outcomes

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to lesion-symptom mapping
   - VLSM concepts and methods
   - Comparison to other tools (NPM, VLSM toolbox)
   - Citation information

2. **Installation** (70 lines)
   - MATLAB installation
   - SPM12 dependency
   - NiiStat download and setup
   - Example data download
   - Command-line usage

3. **Data Preparation** (100 lines, 3-4 examples)
   - Lesion mask creation
   - Binary vs. probabilistic lesions
   - Normalization to MNI space
   - Quality control for lesion masks
   - Clinical data CSV format
   - Example: Preparing stroke lesion dataset

4. **Basic VLSM Analysis** (120 lines, 4-5 examples)
   - Running univariate VLSM
   - Liebermeister test for binary outcomes
   - T-tests for continuous outcomes
   - Cluster-based thresholding
   - Example: Aphasia severity mapping

5. **Advanced Statistical Methods** (100 lines, 3-4 examples)
   - Brunner-Munzel tests
   - Non-parametric permutation tests
   - Covariate control (age, lesion volume)
   - FDR and Bonferroni corrections
   - Example: Controlling for lesion size in VLSM

6. **Region-of-Interest Analysis** (90 lines, 3-4 examples)
   - Atlas-based ROI extraction
   - Custom ROI definition
   - ROI-symptom correlations
   - Multi-ROI comparisons
   - Example: Comparing frontal vs. temporal lesion effects

7. **Visualization** (80 lines, 2-3 examples)
   - Statistical maps overlaid on templates
   - Glass brain projections
   - MRIcroGL integration
   - Creating publication-quality figures
   - Example: VLSM results visualization

8. **Batch Processing** (70 lines, 2-3 examples)
   - Scripting multiple analyses
   - Parameter sweeps
   - Cross-validation
   - Example: Analyzing multiple behavioral scores

9. **Quality Control** (60 lines, 1-2 examples)
   - Lesion coverage maps
   - Sample size per voxel
   - Outlier detection
   - Registration quality checks

10. **Integration with Other Tools** (60 lines, 1-2 examples)
    - SPM for preprocessing
    - FSL for normalization
    - Exporting results for meta-analysis
    - Example: NiiStat to NiMARE workflow

11. **Troubleshooting** (50 lines)
    - Small sample size issues
    - Multiple comparison corrections
    - Unbalanced lesion distributions
    - Memory limitations

12. **Best Practices** (40 lines)
    - Sample size recommendations
    - Lesion quality standards
    - Statistical reporting
    - Reproducibility practices

13. **References** (20 lines)
    - VLSM methodology papers
    - Lesion-symptom mapping reviews
    - Statistical best practices

**Code Examples:**
- Basic VLSM with continuous outcome (MATLAB)
- Binary outcome analysis (MATLAB)
- Covariate control (MATLAB)
- ROI extraction and analysis (MATLAB)
- Batch processing script (MATLAB)

**Integration Points:**
- SPM for preprocessing
- FSL for normalization
- MRIcroGL for visualization
- ANTs for registration
- FreeSurfer for parcellations

---

## Implementation Checklist

### Per-Skill Requirements
- [ ] 650-750 lines per skill
- [ ] 22-28 code examples per skill
- [ ] Consistent section structure
- [ ] Installation instructions for multiple platforms
- [ ] Basic usage examples
- [ ] Advanced features
- [ ] Quality control guidance
- [ ] Batch processing examples
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
- [ ] Total lines: 2,050-2,200
- [ ] Total examples: 70-82
- [ ] Consistent markdown formatting
- [ ] Proper cross-referencing between skills
- [ ] All three skills address clinical applications

## Timeline

1. **Lead-DBS**: 700-750 lines, 24-28 examples (neurosurgical DBS)
2. **3D Slicer**: 700-750 lines, 24-28 examples (surgical planning platform)
3. **NiiStat**: 650-700 lines, 22-26 examples (lesion-symptom mapping)

**Estimated Total:** 2,050-2,200 lines, 70-82 examples

## Context & Connections

### Clinical Workflow Integration

**DBS Workflow (Lead-DBS):**
```
Pre-op MRI → Surgical Planning → Post-op CT → Lead-DBS Reconstruction
    ↓              ↓                  ↓              ↓
FreeSurfer    3D Slicer          ANTs/FSL     Connectivity Analysis
```

**Surgical Planning (3D Slicer):**
```
Multi-modal Acquisition → Registration → Segmentation → Planning
       ↓                       ↓              ↓           ↓
   MRI/CT/PET            Elastix/ANTs    FreeSurfer   Trajectory
```

**Lesion Analysis (NiiStat):**
```
Lesion Masks → Normalization → VLSM → Statistical Maps
     ↓              ↓            ↓           ↓
ITK-SNAP        ANTs/SPM    NiiStat    MRIcroGL
```

### Complementary Tools

**Already Covered:**
- **FreeSurfer**: Cortical parcellation for DBS targeting
- **ANTs**: Registration for all three workflows
- **SPM**: Statistical analysis and preprocessing
- **FSL**: Preprocessing and normalization
- **ITK-SNAP**: Lesion mask creation

**New Capabilities:**
- **Lead-DBS**: First DBS-specific targeting and connectivity tool
- **3D Slicer**: First comprehensive surgical planning platform
- **NiiStat**: First dedicated lesion-symptom mapping tool

### Research Applications

**Lead-DBS Use Cases:**
- Optimizing DBS targets for Parkinson's disease
- Predicting clinical outcomes from connectivity
- Understanding mechanisms of DBS therapy
- Retrospective analysis of DBS cohorts

**3D Slicer Use Cases:**
- Pre-surgical tumor resection planning
- Multi-modal image fusion for epilepsy surgery
- Vascular malformation visualization
- Radiotherapy planning
- Custom surgical guide design (3D printing)

**NiiStat Use Cases:**
- Stroke lesion-symptom mapping
- Predicting aphasia recovery
- Traumatic brain injury outcome prediction
- Identifying critical regions for cognitive functions
- Meta-analyses of lesion studies

## Technical Specifications

### Lead-DBS
- **Platform**: MATLAB (R2017a or newer)
- **Dependencies**: SPM12, Statistical Toolbox
- **Input**: Pre-op MRI (T1/T2), Post-op CT
- **Output**: Electrode coordinates, VTA models, connectivity matrices
- **Atlases**: DISTAL, CIT168, ATAG
- **Formats**: NIfTI, DICOM

### 3D Slicer
- **Platform**: Cross-platform (Windows, macOS, Linux)
- **Language**: C++ core, Python scripting
- **Input**: DICOM, NIfTI, NRRD, etc.
- **Output**: DICOM, NIfTI, STL, OBJ, PLY
- **Extensions**: 300+ available extensions
- **API**: Python, C++

### NiiStat
- **Platform**: MATLAB (R2014b or newer)
- **Dependencies**: SPM12, Statistics Toolbox
- **Input**: Lesion masks (NIfTI), clinical CSV
- **Output**: Statistical maps (NIfTI), figures
- **Methods**: Liebermeister, Brunner-Munzel, permutation
- **Corrections**: FDR, Bonferroni, cluster-based

## Learning Path

### Beginner Path (Clinical Users)
1. Start with **3D Slicer** for visualization and basic segmentation
2. Learn **NiiStat** for lesion analysis (if applicable)
3. Advance to **Lead-DBS** for DBS-specific workflows

### Researcher Path
1. Master **NiiStat** for lesion-symptom mapping
2. Use **3D Slicer** for surgical planning research
3. Apply **Lead-DBS** for DBS mechanism studies

### Neurosurgeon Path
1. Learn **3D Slicer** for general surgical planning
2. Master **Lead-DBS** for DBS electrode placement
3. Integrate both into clinical workflow

## Success Metrics

- [ ] All three skills cover complete clinical workflows
- [ ] Installation instructions work on multiple platforms
- [ ] Code examples run without errors
- [ ] Integration examples connect to existing neuroimaging tools
- [ ] Troubleshooting addresses real user issues
- [ ] Documentation enables self-directed learning

## Notes

**Clinical Relevance:**
This batch uniquely focuses on clinical applications rather than pure research tools. All three skills directly support patient care:
- Lead-DBS improves DBS targeting and outcomes
- 3D Slicer enables safer surgical planning
- NiiStat advances stroke rehabilitation research

**Tool Maturity:**
All three tools are mature, widely used, and actively maintained:
- Lead-DBS: 1000+ citations, annual updates
- 3D Slicer: 20+ years development, large community
- NiiStat: Validated in numerous lesion studies

**Complementarity:**
These tools complement the existing neuroimaging ecosystem:
- Lead-DBS extends FreeSurfer and connectivity tools to DBS
- 3D Slicer bridges clinical DICOM and research NIfTI worlds
- NiiStat fills the lesion analysis gap between SPM and FSL

**Target Audience Expansion:**
This batch expands our user base to include:
- Neurosurgeons and surgical planners
- Clinical neurologists
- Rehabilitation researchers
- Medical physicists
- Neuroradiologists

By completing this batch, we provide comprehensive coverage of clinical neuroimaging applications.

## Detailed Clinical Use Cases

### Lead-DBS: DBS Targeting Optimization

**Case 1: Parkinson's Disease STN-DBS**
```
Workflow:
1. Pre-operative 3T MRI (T1w, T2w, SWI for STN visualization)
2. Surgical planning with stereotactic coordinates
3. Post-operative CT showing electrode placement
4. Lead-DBS reconstruction:
   - Coregister post-op CT to pre-op MRI
   - Normalize to MNI space using ANTs
   - Detect electrodes automatically
   - Refine contact positions manually
5. VTA modeling at clinical settings (130 Hz, 60 μs, 2.5V)
6. Overlay VTA on STN atlas to verify targeting
7. Correlation with clinical outcomes (UPDRS-III improvement)

Expected Outcome: Identify whether electrode is in dorsolateral (motor) or ventromedial (limbic) STN
```

**Case 2: Essential Tremor VIM-DBS Connectivity Analysis**
```
Workflow:
1. Pre-operative diffusion MRI (multi-shell, 64+ directions)
2. Normative connectome from healthy subjects
3. Lead-DBS reconstruction of VIM electrodes
4. Structural connectivity analysis:
   - Tractography from VTA to cortical targets
   - Identify connections to motor cortex
5. Functional connectivity using normative resting-state data
6. Discriminative fiber analysis comparing good vs. poor responders
7. Sweet spot mapping across patient cohort

Expected Outcome: Connectivity profile predicting >80% tremor suppression
```

**Case 3: OCD DBS Retrospective Analysis**
```
Workflow:
1. Multi-center retrospective data (30+ patients)
2. Multiple targets (ALIC, STN, NAcc)
3. Group analysis in Lead-DBS:
   - Standardize all electrodes to MNI space
   - Calculate individual VTAs
   - Correlate with Y-BOCS improvement
   - Statistical mapping (sweet spots)
4. Fiber connectivity to prefrontal cortex
5. Cross-validation to test predictive model

Expected Outcome: Optimal stimulation volume and connectivity profile
```

### 3D Slicer: Surgical Planning Workflows

**Case 1: Glioblastoma Resection Planning**
```
Data:
- Pre-operative T1 post-contrast MRI
- T2 FLAIR
- DTI for tractography
- MR angiography (TOF)
- Functional MRI (motor/language mapping)

Workflow:
1. Import all sequences into 3D Slicer
2. Register all modalities to T1 post-contrast
3. Segment tumor (contrast-enhancing + edema)
4. Segment critical structures:
   - Motor/language cortex from fMRI
   - Corticospinal tract from DTI
   - Major vessels from MRA
5. Create 3D rendering with all structures
6. Simulate surgical approach angles
7. Identify safe entry corridor
8. Measure distances to eloquent areas
9. Export to neuronavigation system

Expected Outcome: Maximized resection with preserved function
```

**Case 2: Epilepsy Surgery with SEEG Planning**
```
Data:
- 3T MRI (T1, T2, FLAIR)
- PET (FDG or 11C-flumazenil)
- MEG/EEG source localization

Workflow:
1. Co-register MRI, PET, and MEG data in Slicer
2. Segment suspected seizure focus
3. Plan SEEG electrode trajectories:
   - Avoid vessels (MRA/CTA segmentation)
   - Sample suspected zones
   - Ensure safe entry points
4. Simulate electrode paths in 3D
5. Verify coverage of seizure onset zone
6. Export trajectories to Robotic SEEG system
7. Post-implant CT fusion for verification

Expected Outcome: Optimal coverage with minimal electrodes
```

**Case 3: AVM Embolization Planning**
```
Data:
- MRI (T1, T2)
- MR angiography (time-of-flight)
- CT angiography
- DSA (digital subtraction angiography)

Workflow:
1. Import all vascular imaging modalities
2. Segment AVM nidus
3. Segment feeding arteries and draining veins
4. Create 3D vascular model
5. Identify approach for endovascular catheter
6. Measure distances and angles
7. Plan embolization stages
8. Export STL for 3D printing (physical model)

Expected Outcome: Optimized embolization strategy
```

### NiiStat: Lesion-Symptom Mapping Studies

**Case 1: Acute Stroke Aphasia Prediction**
```
Sample:
- 150 acute stroke patients (left hemisphere)
- Lesion masks from CT/MRI
- WAB (Western Aphasia Battery) scores at 3 months

Workflow:
1. Create binary lesion masks in ITK-SNAP
2. Normalize all lesions to MNI space (ANTs)
3. Verify lesion quality (coverage maps)
4. Prepare clinical CSV with:
   - Subject ID
   - Lesion volume
   - Age, education
   - WAB aphasia quotient (outcome)
5. Run NiiStat VLSM:
   - Control for age and lesion volume
   - Brunner-Munzel test
   - FDR correction (q < 0.05)
   - Minimum 10 patients per voxel
6. Visualize results in MRIcroGL
7. Extract peak coordinates
8. ROI analysis in significant clusters

Expected Outcome: Identify critical regions for language recovery
```

**Case 2: TBI and Executive Function**
```
Sample:
- 80 TBI patients with focal lesions
- T1w MRI with lesion segmentation
- Executive function battery (Stroop, WCST, TMT)

Workflow:
1. Prepare lesion masks (FreeSurfer + manual editing)
2. Normalize to MNI (ANTs SyN)
3. Create composite executive function score (z-scored)
4. NiiStat VLSM analysis:
   - Continuous outcome (executive score)
   - Covariate: time since injury
   - Permutation testing (5000 iterations)
   - Cluster-based thresholding
5. Overlay results on frontal lobe parcellation
6. Compare lesion effects in DLPFC vs. OFC
7. Cross-validation (leave-one-out)

Expected Outcome: Dissociate frontal regions critical for executive control
```

**Case 3: Spatial Neglect After Right Hemisphere Stroke**
```
Sample:
- 120 right hemisphere stroke patients
- Catherine Bergego Scale (neglect severity)
- Binary outcome: neglect present/absent

Workflow:
1. Binary lesion masks from clinical MRI
2. Normalize using cost-function masking (FSL FLIRT)
3. Quality control: check parietal coverage
4. NiiStat Liebermeister test (binary outcome)
5. FDR correction + cluster threshold (k > 50 voxels)
6. Overlay on right hemisphere atlas (JHU)
7. ROI analysis: compare TPJ vs. IPS lesion effects
8. Logistic regression with lesion volume

Expected Outcome: Identify neural substrates of spatial attention
```

## Alternative Tools and Comparisons

### Lead-DBS Alternatives

**PaCER (Precise and Convenient Electrode Reconstruction)**
- Pros: Fast automated reconstruction
- Cons: Less manual control, fewer atlases
- Use case: High-throughput research studies
- **Lead-DBS advantage**: More atlases, better connectivity integration

**DBSproc**
- Pros: Fully automated pipeline
- Cons: Limited to specific electrode models
- Use case: Standardized clinical protocols
- **Lead-DBS advantage**: Multi-manufacturer support, manual refinement

**SureTune (Medtronic)**
- Pros: Clinical-grade, FDA-approved
- Cons: Proprietary, limited research features
- Use case: Clinical DBS programming
- **Lead-DBS advantage**: Research flexibility, connectivity analysis

### 3D Slicer Alternatives

**Brainlab iPlan**
- Pros: Clinically validated, integration with navigation
- Cons: Expensive, proprietary
- Use case: Clinical neurosurgery
- **3D Slicer advantage**: Free, open-source, extensible

**Osirix/Horos**
- Pros: Excellent DICOM handling, Mac-optimized
- Cons: Limited advanced analysis
- Use case: Radiology workstations
- **3D Slicer advantage**: More analysis tools, cross-platform

**FSL/FSLeyes + ITK-SNAP**
- Pros: Research-standard neuroimaging tools
- Cons: Not designed for surgical planning
- Use case: Research preprocessing and QC
- **3D Slicer advantage**: Clinical workflow integration

**MRIcroGL**
- Pros: Fast rendering, scripting
- Cons: Limited segmentation and registration
- Use case: Visualization and quality control
- **3D Slicer advantage**: Comprehensive surgical planning suite

### NiiStat Alternatives

**NPM (Non-Parametric Mapping)**
- Pros: Established VLSM tool, MRIcron integration
- Cons: Older interface, limited to Liebermeister test
- Use case: Basic VLSM analyses
- **NiiStat advantage**: More statistical methods, better SPM integration

**VLSM2 Toolbox**
- Pros: SPM-based, familiar interface
- Cons: Less actively maintained
- Use case: SPM users doing VLSM
- **NiiStat advantage**: More modern, better documentation

**SPM + SnPM**
- Pros: Flexible general linear model
- Cons: Not optimized for lesion data
- Use case: General voxel-wise statistics
- **NiiStat advantage**: Lesion-specific methods, better small-sample handling

**Lesion Quantification Toolkit (LQT)**
- Pros: Automated lesion quantification
- Cons: Limited statistical methods
- Use case: Lesion volumetry and location
- **NiiStat advantage**: Comprehensive VLSM statistical framework

## Regulatory and Clinical Validation

### Lead-DBS

**Research Use:**
- Widely used in peer-reviewed research (1000+ citations)
- Validated against manual measurements (submillimeter accuracy)
- Compared to clinical planning systems (PaCER, SureTune)

**Clinical Translation:**
- Not FDA-cleared for clinical use (research tool)
- Results can inform clinical decision-making
- Many centers use for retrospective analysis

**Best Practices:**
- Always verify automated results manually
- Use clinical-grade imaging protocols
- Document all processing steps
- Compare to manufacturer's programming software

### 3D Slicer

**Research Use:**
- 20+ years of development and validation
- 5000+ citations in scientific literature
- Used in hundreds of imaging studies

**Clinical Use:**
- Not FDA-cleared as medical device (general-purpose software)
- Individual extensions may have regulatory clearance
- Used as research software in clinical settings
- Outputs can inform clinical decisions (physician responsibility)

**Quality Management:**
- ISO 13485 considerations for clinical use
- Validation and verification protocols
- Standard operating procedures recommended
- Regular software updates and testing

**Regulatory Status:**
- SlicerRT: Validated for research radiotherapy planning
- Certain institutions have internal validation for clinical use
- Always follow local institutional guidelines

### NiiStat

**Research Use:**
- Validated statistical methods (published algorithms)
- Compared to established tools (NPM, SPM)
- Used in clinical neuroscience research

**Clinical Translation:**
- Research tool, not diagnostic software
- Results support clinical research, not individual diagnosis
- Appropriate for group studies and validation

**Statistical Rigor:**
- Implements published correction methods
- Permutation testing for robustness
- Power analysis recommendations provided

## Sample Datasets and Learning Resources

### Lead-DBS

**Example Datasets:**
1. **LEAD-DBS Tutorial Data**
   - URL: https://www.lead-dbs.org/helpsupport/knowledge-base/download-install/
   - Content: Sample STN-DBS case with pre/post-op imaging
   - Size: ~2 GB

2. **OpenNeuro DBS Datasets**
   - URL: https://openneuro.org/ (search "DBS")
   - Content: Multi-center DBS imaging data
   - Example: ds003434 (PD STN-DBS)

**Tutorials:**
- Official Lead-DBS tutorials: https://www.lead-dbs.org/helpsupport/knowledge-base/
- YouTube video series by Lead-DBS team
- Hands-on workshops at conferences (Movement Disorders Society, SfN)

**Publications for Learning:**
- Horn et al. (2019). "Lead-DBS v2: Towards a comprehensive pipeline for DBS imaging." *NeuroImage*
- Ewert et al. (2018). "Toward defining deep brain stimulation targets in MNI space: A subcortical atlas based on multimodal MRI." *NeuroImage*

### 3D Slicer

**Example Datasets:**
1. **Slicer Sample Data**
   - Built-in: Sample Data module in Slicer
   - Content: MRI, CT, PET, DTI examples
   - Usage: File → Download Sample Data

2. **SlicerMorph Example Data**
   - URL: https://github.com/SlicerMorph/SampleData
   - Content: 3D morphometric analysis examples

3. **Medical Segmentation Decathlon**
   - URL: http://medicaldecathlon.com/
   - Content: 10 segmentation tasks with ground truth
   - Size: 50+ GB total

**Tutorials:**
- Official tutorials: https://www.slicer.org/wiki/Documentation/4.10/Training
- Slicer YouTube channel: https://www.youtube.com/c/3DSlicer
- Perk Lab tutorials (surgical planning): http://perk.cs.queensu.ca/
- SlicerDMRI tutorials: http://dmri.slicer.org/

**Online Courses:**
- NA-MIC (National Alliance for Medical Image Computing) training materials
- Slicer Project Weeks (virtual/in-person events)

### NiiStat

**Example Datasets:**
1. **NiiStat Example Data**
   - Included with software download
   - Content: Sample lesion masks and clinical data
   - Tutorial scripts included

2. **Stroke Lesion Datasets (OpenNeuro)**
   - Example: ATLAS dataset (Liew et al., 2018)
   - URL: http://fcon_1000.projects.nitrc.org/indi/retro/atlas.html
   - Content: 304 stroke lesions with clinical data

**Tutorials:**
- Official NiiStat documentation
- Chris Rorden's blog posts: https://www.mccauslandcenter.sc.edu/crnl/
- VLSM tutorial papers

**Key Publications:**
- Rorden et al. (2007). "Improving lesion-symptom mapping." *Journal of Cognitive Neuroscience*
- Bates et al. (2003). "Voxel-based lesion-symptom mapping." *Nature Neuroscience*

## Integration Scenarios

### Scenario 1: Complete DBS Research Pipeline

```
Preprocessing (FreeSurfer, FSL):
- Cortical reconstruction
- Diffusion preprocessing
- Registration to MNI
         ↓
Clinical Data Import (Lead-DBS):
- Load pre-op MRI
- Load post-op CT
- Automated electrode detection
         ↓
Verification (3D Slicer):
- Visual QC of electrode position
- Multi-modal visualization
- Manual refinement if needed
         ↓
Analysis (Lead-DBS):
- VTA modeling
- Connectivity analysis
- Group sweet spot mapping
         ↓
Visualization (Connectome Workbench):
- Cortical connectivity maps
- Publication-quality figures
```

### Scenario 2: Stroke Lesion-Outcome Prediction

```
Lesion Segmentation (3D Slicer):
- Load acute stroke MRI/CT
- Manual segmentation using Segment Editor
- Quality control and smoothing
- Export as NIfTI
         ↓
Preprocessing (ANTs):
- Lesion cost-function masking
- Normalization to MNI
- Quality verification
         ↓
Statistical Analysis (NiiStat):
- Prepare clinical data CSV
- VLSM with outcome measures
- Control for confounds
- Multiple comparison correction
         ↓
Visualization (MRIcroGL, FSLeyes):
- Overlay statistical maps
- Glass brain projections
- Regional quantification
         ↓
ROI Analysis (FSL, nilearn):
- Extract values from significant clusters
- Secondary analyses
- Cross-validation
```

### Scenario 3: Multi-Modal Surgical Planning

```
Image Acquisition:
- Structural MRI (T1, T2, FLAIR)
- Functional MRI (task-based)
- DTI (64+ directions)
- MR angiography
         ↓
Preprocessing (fMRIPrep, QSIPrep):
- Anatomical processing
- fMRI preprocessing
- DTI preprocessing
- Surface reconstruction
         ↓
Integration (3D Slicer):
- Import all modalities
- Multi-modal registration
- Tumor segmentation
- Vessel segmentation
- Tractography visualization (SlicerDMRI)
- Functional activation overlays
         ↓
Planning:
- 3D rendering of all structures
- Surgical approach simulation
- Safe corridor identification
- Distance measurements
         ↓
Export:
- DICOM for neuronavigation
- STL for 3D printing
- Documentation for surgical report
```

## Batch Summary Statistics

| Skill | Lines | Examples | Primary Domain | Secondary Domains |
|-------|-------|----------|----------------|-------------------|
| Lead-DBS | 700-750 | 24-28 | DBS Neurosurgery | Connectivity, Movement Disorders |
| 3D Slicer | 700-750 | 24-28 | Surgical Planning | Multi-modal Fusion, Segmentation |
| NiiStat | 650-700 | 22-26 | Lesion Analysis | Stroke, Cognitive Neuroscience |
| **Total** | **2,050-2,200** | **70-82** | **Clinical Neuroimaging** | **Multi-disciplinary** |

## Expected Impact

### Research Community
- **DBS Research**: Enables standardized electrode localization and connectivity analysis
- **Stroke Research**: Facilitates lesion-symptom mapping studies with proper statistics
- **Surgical Research**: Supports retrospective and prospective surgical planning studies

### Clinical Practice
- **Neurosurgery**: Improves DBS targeting precision and surgical planning safety
- **Neurology**: Enhances understanding of lesion-deficit relationships
- **Radiology**: Provides advanced visualization and quantification tools

### Education and Training
- **Medical Students**: Learn neuroanatomy through 3D visualization
- **Residents**: Practice surgical planning in safe environment
- **Researchers**: Access validated tools for clinical neuroscience

## Conclusion

Batch 33 addresses critical gaps in clinical neuroimaging by providing comprehensive documentation for three essential tools spanning neurosurgery, surgical planning, and lesion analysis. These skills will enable users to:

1. **Optimize DBS therapy** through precise electrode localization and connectivity-based targeting
2. **Plan safer surgeries** using multi-modal image fusion and 3D visualization
3. **Map lesion-symptom relationships** with rigorous statistical methods

By completing this batch, the N_tools neuroimaging skill collection will reach **114/133 skills (85.7%)**, with robust coverage of both research and clinical applications.
