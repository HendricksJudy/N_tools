# Batch 34 Plan: Specialized Imaging Modalities

## Overview

**Theme:** Specialized Imaging Modalities & Non-Invasive Brain Stimulation
**Focus:** Spinal cord MRI, real-time fMRI/neurofeedback, and TMS/tDCS simulation
**Target:** 3 new skills, 2,050-2,200 total lines

**Current Progress:** 111/133 skills (83.5%)
**After Batch 33:** 114/133 skills (85.7%)
**After Batch 34:** 117/133 skills (87.9%)

This batch addresses critical gaps in specialized neuroimaging modalities beyond standard brain MRI. These tools enable analysis of the spinal cord, real-time brain activity feedback, and non-invasive brain stimulation planning—all emerging and clinically important applications.

## Rationale

While brain imaging tools are comprehensively covered, several specialized modalities require dedicated software:
- **Spinal Cord MRI:** Unique anatomical challenges (small structure, motion artifacts, partial volume)
- **Real-Time fMRI:** Closed-loop neurofeedback for psychiatric/neurological disorders
- **Brain Stimulation:** TMS/tDCS targeting and electric field modeling for therapy

This batch provides essential tools for researchers and clinicians working beyond conventional brain imaging.

## Skills to Create

### 1. Spinal Cord Toolbox (SCT) (700-750 lines, 24-28 examples)

**Overview:**
Spinal Cord Toolbox (SCT) is a comprehensive software for processing spinal cord MRI data. It addresses unique challenges of spinal cord imaging including automatic segmentation, registration to standardized templates, extraction of quantitative metrics, and atlas-based analysis.

**Key Features:**
- Automated spinal cord segmentation
- Gray and white matter segmentation
- Registration to PAM50 template (spinal cord atlas)
- Vertebral labeling
- Metric extraction (cross-sectional area, diffusion, MTR, etc.)
- Multi-parametric quantification
- Quality control reporting
- Command-line and Python API

**Target Audience:**
- Researchers studying spinal cord injury
- Clinicians analyzing MS lesions in the cord
- Neurosurgeons planning spinal interventions
- Motor neuron disease researchers

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to spinal cord imaging challenges
   - SCT capabilities and workflow
   - Supported modalities (T1w, T2w, T2*w, DWI, MTR, fMRI)
   - Citation information

2. **Installation** (80 lines)
   - Installation methods (conda, pip, Docker)
   - FSL integration
   - Dependencies and setup
   - Batch launcher configuration
   - Testing installation

3. **Basic Spinal Cord Segmentation** (120 lines, 4-5 examples)
   - Automatic segmentation (DeepSeg models)
   - Manual correction with FSLeyes integration
   - Vertebral level detection
   - Quality control
   - Example: T2w spinal cord segmentation

4. **Gray/White Matter Segmentation** (100 lines, 3-4 examples)
   - Gray matter segmentation methods
   - White matter tract parcellation
   - Cross-sectional area computation
   - Example: GM/WM segmentation from T2*w

5. **Registration to PAM50 Template** (120 lines, 4-5 examples)
   - Template-based registration
   - Straightening the spinal cord
   - Warping to template space
   - Inverse transformation
   - Example: Multi-subject registration workflow

6. **Metric Extraction** (100 lines, 3-4 examples)
   - Cross-sectional area (CSA)
   - DTI metrics per vertebral level
   - Magnetization transfer ratio (MTR)
   - Lesion quantification
   - Example: CSA extraction per vertebral level

7. **Advanced Analysis** (100 lines, 3-4 examples)
   - Atlas-based analysis (white matter tracts)
   - Shape analysis
   - Multi-parametric correlation
   - Longitudinal studies
   - Example: Dorsal column FA analysis

8. **Quality Control & Visualization** (80 lines, 2-3 examples)
   - SCT QC report generation
   - FSLeyes plugin for visualization
   - Manual correction workflows
   - Batch QC review

9. **Batch Processing** (70 lines, 2-3 examples)
   - BIDS-compatible processing
   - Processing multiple subjects
   - HPC integration
   - Example: Batch script for multi-subject CSA

10. **Integration with Other Tools** (60 lines, 1-2 examples)
    - FSL for preprocessing
    - ANTs for custom registration
    - Python scripting with SCT API
    - Example: SCT + FSL diffusion pipeline

11. **Troubleshooting** (60 lines)
    - Segmentation failures
    - Registration issues
    - Handling artifacts (motion, CSF pulsation)
    - Cervical vs. thoracic challenges

12. **Best Practices** (40 lines)
    - Acquisition protocols
    - Quality control workflow
    - Reporting standards
    - Multi-center harmonization

13. **References** (20 lines)
    - SCT papers
    - Spinal cord imaging reviews
    - Atlas papers

**Code Examples:**
- Basic segmentation (bash)
- Vertebral labeling (bash)
- Registration to template (bash)
- CSA extraction (bash)
- Batch processing script (bash/Python)

**Integration Points:**
- FSL for preprocessing
- FSLeyes for visualization and manual corrections
- ANTs for advanced registration
- BIDS for data organization
- Python/Nipype for workflow integration

---

### 2. OpenNFT (650-700 lines, 22-26 examples)

**Overview:**
OpenNFT (Open NeuroFeedback Training) is an open-source platform for real-time fMRI neurofeedback experiments. It enables closed-loop paradigms where subjects receive feedback based on their ongoing brain activity, used for therapeutic interventions and cognitive neuroscience research.

**Key Features:**
- Real-time fMRI preprocessing
- Multiple feedback modalities (visual, auditory)
- ROI-based and multivariate pattern analysis
- Dynamic connectivity feedback
- Integration with stimulus presentation software
- Offline simulation and testing
- Support for major scanner vendors (Siemens, GE, Philips)

**Target Audience:**
- Neurofeedback researchers
- Clinical researchers treating psychiatric disorders
- Cognitive neuroscientists studying self-regulation
- Pain researchers

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to real-time fMRI and neurofeedback
   - OpenNFT architecture and workflow
   - Use cases (depression, PTSD, pain, addiction)
   - Citation information

2. **Installation** (90 lines)
   - Software dependencies (Python, MATLAB, SPM)
   - Installation on experiment computer
   - Scanner-side configuration
   - Setting up DICOM receiver
   - Testing with example data

3. **Setting Up a Neurofeedback Experiment** (120 lines, 4-5 examples)
   - Experimental design considerations
   - Defining regions of interest (ROIs)
   - Baseline acquisition
   - Neurofeedback protocol setup
   - Example: Amygdala down-regulation for anxiety

4. **Real-Time Preprocessing** (100 lines, 3-4 examples)
   - Motion correction
   - Spatial smoothing
   - Temporal filtering
   - Online GLM computation
   - Example: Real-time preprocessing pipeline

5. **Feedback Computation Methods** (100 lines, 3-4 examples)
   - ROI mean activation
   - Percent signal change
   - Multivariate pattern analysis (MVPA)
   - Connectivity-based feedback
   - Example: SVM-based emotion classification

6. **Feedback Presentation** (80 lines, 2-3 examples)
   - Visual feedback (thermometer, virtual reality)
   - Auditory feedback
   - Integration with PsychoPy/Presentation
   - Timing and synchronization
   - Example: Visual thermometer feedback

7. **Quality Control During Acquisition** (70 lines, 2-3 examples)
   - Real-time motion monitoring
   - Signal quality assessment
   - Online adjustments
   - Experimenter interface

8. **Offline Analysis** (80 lines, 2-3 examples)
   - Post-hoc verification of neurofeedback
   - Learning curves analysis
   - Transfer effects assessment
   - Example: Analyzing neurofeedback success

9. **Advanced Features** (60 lines, 1-2 examples)
   - Dynamic causal modeling (DCM) based feedback
   - Multi-region connectivity feedback
   - Intermittent feedback protocols
   - Example: PPI-based connectivity neurofeedback

10. **Troubleshooting** (50 lines)
    - DICOM transfer issues
    - Timing delays and latency
    - Artifact handling
    - Scanner-specific problems

11. **Best Practices** (40 lines)
    - Participant instruction and training
    - Control conditions
    - Blinding and sham feedback
    - Reporting standards

12. **References** (20 lines)
    - OpenNFT publications
    - Neurofeedback methodology
    - Clinical applications

**Code Examples:**
- Basic ROI-based feedback (Python/MATLAB)
- MVPA classifier training (Python)
- Real-time GLM (MATLAB)
- Custom feedback display (Python)

**Integration Points:**
- SPM for preprocessing
- PsychoPy for stimulus presentation
- FSL for ROI definition
- scikit-learn for MVPA
- Turbo-BrainVoyager as alternative platform

---

### 3. SimNIBS (700-750 lines, 24-28 examples)

**Overview:**
SimNIBS (Simulation of Non-invasive Brain Stimulation) is a software for realistic simulations of electric fields induced by transcranial magnetic stimulation (TMS) and transcranial direct current stimulation (tDCS). It helps optimize coil/electrode placement and predict stimulation effects.

**Key Features:**
- Realistic head models from MRI
- TMS coil modeling (figure-8, circular, custom)
- tDCS electrode montage optimization
- Electric field simulation (FEM)
- Group-level analysis
- Automated anatomical targeting
- Integration with neuronavigation systems
- MATLAB and Python APIs

**Target Audience:**
- TMS/tDCS researchers
- Clinical neurologists using brain stimulation
- Neuroscientists studying cortical excitability
- Rehabilitation researchers

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to non-invasive brain stimulation
   - TMS vs. tDCS mechanisms
   - Importance of electric field modeling
   - Citation information

2. **Installation** (80 lines)
   - Installation on Windows, macOS, Linux
   - FreeSurfer integration
   - MNI template download
   - MATLAB/Python setup
   - GUI vs. scripting modes

3. **Creating Head Models** (120 lines, 4-5 examples)
   - Automatic head mesh generation (mri2mesh)
   - Tissue segmentation (skin, skull, CSF, GM, WM)
   - Mesh quality control
   - Using standard templates (SimNIBS MNI head)
   - Example: Generate personalized head model from T1w

4. **TMS Simulation** (120 lines, 4-5 examples)
   - Coil positioning (manual, MNI coordinates)
   - Coil types and orientations
   - Electric field calculation
   - Visualization on cortical surface
   - Example: Motor cortex M1-HAND stimulation

5. **tDCS Simulation** (100 lines, 3-4 examples)
   - Electrode placement and sizing
   - Montage optimization for targeted stimulation
   - Current flow modeling
   - Safety considerations (current density limits)
   - Example: Left DLPFC anodal tDCS for depression

6. **Advanced Coil Placement** (100 lines, 3-4 examples)
   - Anatomical targeting (gyrus-based)
   - Functional targeting (fMRI activation sites)
   - Neuronavigation system integration
   - Optimization algorithms
   - Example: TMS coil placement over language areas

7. **Group Analysis** (80 lines, 2-3 examples)
   - Multi-subject simulations
   - Variability analysis across subjects
   - Group-level electric field maps
   - Example: Inter-subject variability in motor cortex TMS

8. **Electrode/Coil Optimization** (80 lines, 2-3 examples)
   - Optimizing tDCS montages for target regions
   - Multi-electrode optimization (HD-tDCS)
   - Constrained optimization
   - Example: 4x1 HD-tDCS montage optimization

9. **Integration with Neuroimaging** (70 lines, 2-3 examples)
   - Using fMRI activations for targeting
   - DTI tractography integration
   - Lesion mapping
   - Example: Targeting stroke lesion penumbra

10. **Quality Control & Visualization** (60 lines, 1-2 examples)
    - Mesh visualization in gmsh
    - Field distribution inspection
    - Exporting results to FreeSurfer/FSL
    - Creating publication figures

11. **Python/MATLAB Scripting** (70 lines, 2-3 examples)
    - Automated batch processing
    - Custom analysis pipelines
    - Parameter sweeps
    - Example: Coil orientation optimization script

12. **Troubleshooting** (50 lines)
    - Mesh generation failures
    - Segmentation errors
    - Electrode placement issues
    - Convergence problems in simulations

13. **Best Practices** (40 lines)
    - MRI acquisition recommendations
    - Validation against measurements
    - Reporting simulation parameters
    - Interpretation of field strengths

14. **References** (20 lines)
    - SimNIBS papers
    - TMS/tDCS physics
    - Clinical applications

**Code Examples:**
- Create head model (Python)
- Simple TMS simulation (Python)
- tDCS montage setup (Python)
- Batch processing (Python)
- Coil optimization (MATLAB)

**Integration Points:**
- FreeSurfer for cortical reconstruction
- FSL for image preprocessing
- MNE-Python for TMS-EEG analysis
- SPM for functional targeting
- Neuronavigation systems (Brainsight, Localite)

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
- [ ] All three skills address specialized modalities

## Timeline

1. **Spinal Cord Toolbox (SCT)**: 700-750 lines, 24-28 examples
2. **OpenNFT**: 650-700 lines, 22-26 examples
3. **SimNIBS**: 700-750 lines, 24-28 examples

**Estimated Total:** 2,050-2,200 lines, 70-82 examples

## Context & Connections

### Research Domain Integration

**Spinal Cord Research:**
```
Acquisition → Preprocessing → SCT Analysis
     ↓              ↓              ↓
MRI Protocol   FSL/ANTs    Segmentation → Metrics → Atlas Mapping
```

**Neurofeedback Research:**
```
fMRI Acquisition → OpenNFT (Real-time) → Behavioral Effects
       ↓                ↓                      ↓
    Scanner      Preprocessing +         Transfer tests
                   Feedback
```

**Brain Stimulation Research:**
```
T1w MRI → SimNIBS Head Model → TMS/tDCS Simulation → Clinical Application
   ↓            ↓                    ↓                      ↓
FreeSurfer   Tissue Seg.      E-field maps         Neuronavigation
```

### Complementary Tools

**Already Covered:**
- **FreeSurfer**: Cortical reconstruction for SimNIBS
- **FSL**: Preprocessing for SCT and OpenNFT
- **SPM**: Coregistration and GLM for OpenNFT
- **ANTs**: Registration for spinal cord
- **Python/MATLAB**: Scripting platforms for all tools

**New Capabilities:**
- **SCT**: First dedicated spinal cord analysis tool
- **OpenNFT**: First real-time fMRI/neurofeedback platform
- **SimNIBS**: First brain stimulation modeling tool

### Research Applications

**SCT Use Cases:**
- Spinal cord injury (SCI) quantification
- Multiple sclerosis lesion tracking
- Motor neuron disease (ALS) biomarkers
- Cervical spondylotic myelopathy assessment
- Atlas-based white matter tract analysis

**OpenNFT Use Cases:**
- Depression treatment (amygdala down-regulation)
- PTSD therapy (fear extinction enhancement)
- Chronic pain management (pain network modulation)
- Motor rehabilitation (motor cortex upregulation)
- Cognitive training (working memory enhancement)

**SimNIBS Use Cases:**
- Optimizing TMS targets for depression (DLPFC)
- Stroke rehabilitation planning (M1 stimulation)
- Individualized tDCS montages
- Understanding stimulation mechanisms
- Multi-site protocol standardization

## Technical Specifications

### Spinal Cord Toolbox (SCT)
- **Platform**: Cross-platform (Linux, macOS, Windows)
- **Language**: Python core, C++ for processing
- **Dependencies**: FSL, ANTs (optional)
- **Input**: T1w, T2w, T2*w, DWI, MTR, fMRI
- **Output**: Segmentations, metrics CSV, QC reports
- **Atlas**: PAM50 (probabilistic anatomical model)
- **Formats**: NIfTI, BIDS-compatible

### OpenNFT
- **Platform**: Cross-platform (primarily Linux/Windows)
- **Language**: Python and MATLAB
- **Dependencies**: SPM, PsychoPy, scikit-learn
- **Input**: Real-time DICOM from scanner
- **Output**: Feedback signals, preprocessed data
- **Latency**: ~2-3 TRs typical delay
- **Scanners**: Siemens, GE, Philips (with DICOM export)

### SimNIBS
- **Platform**: Cross-platform (Linux, macOS, Windows)
- **Language**: Python core, MATLAB API
- **Dependencies**: FreeSurfer (recommended), gmsh
- **Input**: T1w (T2w optional for better skull seg)
- **Output**: Electric field maps, mesh models
- **Methods**: Finite Element Method (FEM)
- **Coils**: Figure-8, circular, custom geometries
- **Electrodes**: Standard pads, HD-tDCS, custom montages

## Learning Path

### Beginner Path (Clinical Researchers)
1. Start with **SimNIBS** for TMS/tDCS planning (most accessible)
2. Learn **SCT** if working with spinal cord data
3. Advance to **OpenNFT** for neurofeedback studies (requires real-time setup)

### Advanced Researcher Path
1. Master **SCT** for comprehensive spinal cord analysis
2. Learn **OpenNFT** for closed-loop experiments
3. Apply **SimNIBS** for mechanistic understanding of stimulation

### Clinical Translation Path
1. Use **SimNIBS** for personalized TMS/tDCS targeting
2. Implement **OpenNFT** for therapeutic neurofeedback trials
3. Apply **SCT** for spinal cord injury/MS monitoring

## Success Metrics

- [ ] All three skills cover complete specialized workflows
- [ ] Installation instructions work on multiple platforms
- [ ] Code examples run without errors
- [ ] Integration examples connect to existing neuroimaging tools
- [ ] Troubleshooting addresses real user issues
- [ ] Documentation enables self-directed learning
- [ ] Clinical and research applications clearly explained

## Detailed Application Scenarios

### SCT: Tracking MS Progression in Spinal Cord

**Clinical Scenario:**
Multiple sclerosis patient with cervical cord involvement, longitudinal monitoring of spinal cord atrophy and lesion burden.

**Workflow:**
```
Month 0, 12, 24: 3T MRI acquisition
- 3D T1w (1mm isotropic)
- 3D T2w (1mm isotropic)
- PSIR (for lesion detection)
- DTI (2mm, 30 directions)

SCT Processing:
1. Spinal cord segmentation (sct_deepseg_sc)
2. Vertebral labeling (sct_label_vertebrae)
3. Registration to PAM50 (sct_register_to_template)
4. CSA computation per level (sct_process_segmentation)
5. Lesion segmentation and registration
6. DTI metric extraction in white matter tracts

Outputs:
- CSA at C2-C3 (primary outcome)
- Lesion volume and distribution
- FA/MD in dorsal columns and CST
- Annual atrophy rate

Clinical Decision:
- CSA decline >2% per year → escalate therapy
- New lesions → reassess disease-modifying treatment
```

### OpenNFT: Neurofeedback for Depression

**Research/Clinical Scenario:**
Treatment-resistant depression study using real-time fMRI neurofeedback to down-regulate amygdala activity.

**Workflow:**
```
Session Design:
- 6 neurofeedback sessions over 3 weeks
- Each session: baseline, 4 neurofeedback runs, transfer
- Total scan time: ~45 minutes per session

OpenNFT Setup:
1. Localizer run: identify amygdala ROI (emotional faces task)
2. Baseline run: establish resting activity
3. Neurofeedback runs:
   - Real-time preprocessing (2-3 TR delay)
   - Extract bilateral amygdala ROI signal
   - Calculate % signal change from baseline
   - Present thermometer feedback (down = success)
   - 30-second regulate blocks alternating with rest
4. Transfer run: self-regulation without feedback

Analysis:
- Learning curve: amygdala downregulation over sessions
- Transfer effects: regulation without feedback
- Clinical outcomes: BDI-II, HDRS pre/post
- Whole-brain changes: offline GLM analysis

Expected Results:
- Improved amygdala regulation across sessions
- Reduced depression scores in responders
- Altered amygdala-PFC connectivity
```

### SimNIBS: Optimizing tDCS for Stroke Recovery

**Clinical Scenario:**
Stroke patient with right hand motor deficit, planning personalized tDCS to enhance left motor cortex excitability during rehabilitation.

**Workflow:**
```
MRI Acquisition:
- T1w 3D (1mm isotropic)
- T2w (for better skull segmentation)
- fMRI: right hand movement attempt
- DTI: for lesion and tract mapping

SimNIBS Processing:
1. Create personalized head model (mri2mesh):
   - Tissue segmentation (5 tissue model)
   - Mesh generation and quality check
   - FreeSurfer cortical surface import

2. Target definition:
   - Peak fMRI activation in left M1 (hand knob)
   - Or use MNI coordinates with registration

3. tDCS Simulation:
   - Conventional montage: anode over M1, cathode over supraorbital
   - Alternative: 4x1 HD-tDCS centered on M1
   - Current: 2 mA, electrode size: 5x5 cm (conventional)

4. Optimization:
   - Maximize E-field in target (M1)
   - Minimize current spread to lesion
   - Constraint: <2 A/m² current density

5. Output Analysis:
   - E-field magnitude at target
   - Field distribution visualization
   - Safety assessment

Clinical Implementation:
- Use optimized montage for 10 sessions
- Concurrent with motor therapy
- Monitor for adverse effects
- Assess motor improvement (ARAT, 9-HPT)

Expected Outcome:
- Personalized montage delivers 20-30% higher E-field to target
- Improved therapy outcomes vs. standard montage
- Individual variability quantified
```

## Alternative Tools and Comparisons

### SCT Alternatives

**Manual Segmentation (ITK-SNAP/FSLeyes)**
- Pros: Full control, works for any anatomy
- Cons: Time-consuming, operator-dependent, no standardization
- **SCT advantage**: Automated, standardized, template-based metrics

**JIM (Xinapse Systems)**
- Pros: Good spinal cord tools, commercial support
- Cons: Proprietary, expensive
- **SCT advantage**: Free, open-source, better template

**volBrain (online service)**
- Pros: Web-based, no installation
- Cons: Limited to basic segmentation, privacy concerns
- **SCT advantage**: Local processing, comprehensive analysis

### OpenNFT Alternatives

**Turbo-BrainVoyager**
- Pros: Commercial support, excellent real-time capabilities
- Cons: Expensive, proprietary, Windows-only
- **OpenNFT advantage**: Free, open-source, cross-platform

**AFNI (real-time plugin)**
- Pros: Part of AFNI suite, well-documented
- Cons: Limited neurofeedback features, less user-friendly
- **OpenNFT advantage**: Designed for neurofeedback, better interfaces

**Custom MATLAB/Python Scripts**
- Pros: Full flexibility
- Cons: Requires extensive programming, reinventing wheel
- **OpenNFT advantage**: Ready-to-use framework, validated

### SimNIBS Alternatives

**ROAST (Realistic vOlumetric Approach to Simulate TES)**
- Pros: SPM-based, automated
- Cons: tDCS only (no TMS), less flexible
- **SimNIBS advantage**: TMS + tDCS, more features, better maintained

**COMETS (Computational MEG and TMS Simulator)**
- Pros: Includes TMS-MEG modeling
- Cons: Less user-friendly, limited documentation
- **SimNIBS advantage**: Better documentation, active development

**Commercial Software (e.g., SimNIBS ancestors)**
- Pros: May have clinical certification
- Cons: Expensive, closed-source
- **SimNIBS advantage**: Free, validated, widely adopted

## Regulatory and Clinical Considerations

### SCT
**Research Use:**
- Widely validated for spinal cord morphometry
- Used in numerous clinical trials (MS, SCI)
- Metrics correlate with clinical outcomes

**Clinical Translation:**
- Not a diagnostic device (research software)
- Outputs can support clinical decision-making
- Used in clinical research settings worldwide
- Quality control essential for clinical use

**Validation:**
- Multi-site reproducibility studies
- Comparison to manual segmentation (Dice >0.9)
- Longitudinal reliability assessed

### OpenNFT
**Research Use:**
- Research-grade software for neurofeedback studies
- Multiple proof-of-concept studies published
- Not FDA-approved for therapeutic use

**Clinical Translation:**
- Experimental therapeutic approach
- IRB approval required for clinical studies
- Informed consent emphasizing experimental nature
- Some clinics use under research protocols

**Safety:**
- Standard MRI safety protocols apply
- Psychological effects monitoring
- Trained experimenters required

### SimNIBS
**Research Use:**
- Standard tool for TMS/tDCS field modeling
- Extensive validation against measurements
- Used to plan stimulation in research studies

**Clinical Use:**
- Not a medical device (simulation software)
- Outputs inform clinical TMS/tDCS application
- Actual stimulation devices are FDA-cleared separately
- Simulations support treatment planning

**Validation:**
- E-field predictions validated vs. measurements
- Multi-site comparison studies
- Integration with clinical neuronavigation systems

## Sample Datasets and Tutorials

### SCT

**Example Datasets:**
1. **SCT Course Data**
   - URL: https://github.com/spinalcordtoolbox/sct_course
   - Content: Example datasets for all major SCT functions
   - Modalities: T1w, T2w, T2*w, DWI, MTR

2. **Multi-Subject Spine Dataset**
   - URL: https://github.com/spine-generic
   - Content: Multi-center, multi-vendor spine MRI
   - Size: ~50 subjects

**Tutorials:**
- Official SCT documentation: https://spinalcordtoolbox.com/
- Video tutorials on YouTube
- Hands-on workshops at ISMRM, OHBM

**Key Publications:**
- De Leener et al. (2017). "SCT: Spinal Cord Toolbox." *NeuroImage*
- Gros et al. (2019). "Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions." *NeuroImage*

### OpenNFT

**Example Datasets:**
1. **OpenNFT Tutorial Data**
   - URL: https://github.com/OpenNFT/OpenNFT
   - Content: Example real-time fMRI data
   - Includes: ROI masks, protocols

2. **Offline Testing Data**
   - Provided with installation
   - Simulates real-time acquisition for testing

**Tutorials:**
- Official manual: https://opennft.org/
- Setup guides for different scanners
- Example protocols (emotion regulation, motor imagery)

**Key Publications:**
- Koush et al. (2017). "OpenNFT: An open-source Python/Matlab framework for real-time fMRI neurofeedback training." *NeuroImage*
- Sulzer et al. (2013). "Real-time fMRI neurofeedback: Progress and challenges." *NeuroImage*

### SimNIBS

**Example Datasets:**
1. **SimNIBS Example Data (Ernie)**
   - Included with installation
   - Complete head model from T1w/T2w
   - Example TMS and tDCS simulations

2. **MNI Head Models**
   - Available through SimNIBS
   - Group average templates

**Tutorials:**
- Official tutorials: https://simnibs.github.io/simnibs/
- Video series on YouTube
- Example scripts (Python and MATLAB)
- Workshops at brain stimulation conferences

**Key Publications:**
- Thielscher et al. (2015). "Field modeling for transcranial magnetic stimulation." *Brain Stimulation*
- Saturnino et al. (2019). "SimNIBS 2.1: A comprehensive pipeline for individualized electric field modeling." *Brain Stimulation*

## Integration Scenarios

### Scenario 1: Comprehensive Spinal Cord Study

```
Data Acquisition:
- Multi-contrast spine MRI (T1w, T2w, T2*w, DWI)
- BIDS organization

Preprocessing (FSL/ANTs):
- Basic quality control
- Motion correction (if needed)
- BIDS validation
         ↓
SCT Analysis:
- Automated segmentation (cord, GM, WM)
- Vertebral labeling
- Registration to PAM50
- Metric extraction (CSA, DTI metrics)
- Atlas-based analysis
         ↓
Statistical Analysis (Python/R):
- Group comparisons
- Correlation with clinical scores
- Longitudinal modeling
         ↓
Visualization (Python/matplotlib):
- Publication figures
- QC reports for all subjects
```

### Scenario 2: Neurofeedback Clinical Trial

```
Trial Design:
- RCT: real vs. sham neurofeedback
- Target: anxiety disorder (amygdala regulation)

Pre-Treatment (Week 0):
- Structural MRI (T1w for anatomy)
- Localizer fMRI (identify amygdala ROI)
- Clinical assessments (STAI, HDRS)
         ↓
Treatment Phase (Weeks 1-3):
- OpenNFT real-time neurofeedback sessions
- 2 sessions per week
- Continuous performance monitoring
- Weekly clinical assessments
         ↓
Post-Treatment (Week 4):
- Transfer fMRI (regulation without feedback)
- Clinical outcome measures
- Follow-up scans at 3, 6 months
         ↓
Offline Analysis:
- fMRIPrep for high-quality preprocessing
- Learning curves (OpenNFT logs)
- Whole-brain changes (SPM/FSL)
- Clinical outcome correlation
- Connectivity analysis (amygdala-PFC)
```

### Scenario 3: Personalized Brain Stimulation

```
Patient Assessment:
- Stroke with motor deficit
- 6 months post-stroke

MRI Acquisition:
- T1w, T2w (for SimNIBS head model)
- fMRI: affected hand movement attempt
- DTI: corticospinal tract mapping
         ↓
Target Definition:
- Lesion segmentation (3D Slicer)
- Motor activation peak (fMRI analysis)
- CST reconstruction (DiPy/MRtrix)
         ↓
SimNIBS Simulation:
- Create personalized head model
- TMS coil positioning over M1 residual activation
- Multiple tDCS montage simulations
- Optimization for focal M1 targeting
         ↓
Treatment Planning:
- Select optimal montage (highest E-field in target)
- Generate neuronavigation coordinates
- Plan 10-session intervention
         ↓
Treatment Delivery:
- Combined with physical therapy
- Neuronavigation-guided TMS or tDCS
- Monitor adverse effects
         ↓
Outcome Assessment:
- Motor function tests (ARAT, grip strength)
- Repeat fMRI (cortical reorganization)
- Repeat SimNIBS (validate field delivery)
```

## Batch Summary Statistics

| Skill | Lines | Examples | Primary Domain | Secondary Domains |
|-------|-------|----------|----------------|-------------------|
| SCT | 700-750 | 24-28 | Spinal Cord Imaging | MS, SCI, ALS, Myelopathy |
| OpenNFT | 650-700 | 22-26 | Real-Time fMRI | Neurofeedback, Psychiatry, Pain |
| SimNIBS | 700-750 | 24-28 | Brain Stimulation | TMS, tDCS, Neuromodulation |
| **Total** | **2,050-2,200** | **70-82** | **Specialized Modalities** | **Multi-disciplinary** |

## Expected Impact

### Research Community
- **Spinal Cord Research**: Standardized analysis for SCI, MS, and motor neuron diseases
- **Neurofeedback Research**: Accessible platform for closed-loop fMRI studies
- **Brain Stimulation**: Mechanistic understanding and optimization of TMS/tDCS

### Clinical Practice
- **Neurology**: Objective spinal cord metrics for MS and SCI monitoring
- **Psychiatry**: Neurofeedback as emerging therapeutic tool
- **Rehabilitation**: Personalized brain stimulation for stroke recovery

### Education and Training
- **Graduate Students**: Learn specialized neuroimaging modalities
- **Clinicians**: Understand brain stimulation mechanisms
- **Researchers**: Design neurofeedback and stimulation studies

## Emerging Trends

### Spinal Cord Imaging
- **7T MRI**: Ultra-high-resolution spinal cord imaging
- **Functional spinal cord fMRI**: Mapping sensory/motor pathways
- **qMRI**: Quantitative MRI for tissue characterization
- **AI segmentation**: Deep learning for automated analysis

### Neurofeedback
- **EEG-fMRI neurofeedback**: Multimodal approaches
- **Decoded neurofeedback**: Implicit regulation paradigms
- **Home-based neurofeedback**: Using EEG instead of fMRI
- **Personalized protocols**: Adapting to individual brain networks

### Brain Stimulation
- **Multi-focal tDCS**: Optimized multi-electrode arrays
- **Temporal interference stimulation**: Deep brain targets non-invasively
- **Closed-loop TMS**: Triggered by brain state
- **Combination therapies**: Stimulation + neurofeedback + behavior

## Conclusion

Batch 34 addresses critical gaps in neuroimaging by documenting three essential tools for specialized modalities:

1. **Spinal Cord Toolbox** enables standardized spinal cord MRI analysis, filling a major gap as spinal cord research grows
2. **OpenNFT** provides an accessible platform for real-time fMRI neurofeedback, an emerging therapeutic approach
3. **SimNIBS** allows individualized brain stimulation modeling, essential for optimizing TMS/tDCS interventions

By completing this batch, the N_tools neuroimaging skill collection will reach **117/133 skills (87.9%)**, with comprehensive coverage extending beyond brain imaging to spinal cord, real-time applications, and non-invasive stimulation.

These tools represent the cutting edge of clinical neuroscience, translating advanced neuroimaging into therapeutic applications and expanding the boundaries of what neuroimaging can achieve.
