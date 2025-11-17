# Batch 26: Computational Neuroscience & Brain Simulation - Planning Document

## Overview

**Batch Theme:** Computational Neuroscience & Brain Simulation
**Batch Number:** 26
**Number of Skills:** 2
**Current Progress:** 93/133 skills completed (69.9%)
**After Batch 26:** 95/133 skills (71.4%)

## Rationale

Batch 26 focuses on **computational neuroscience and whole-brain simulation tools** that bridge neuroimaging data with biophysical models and advanced machine learning. These tools provide:

- **Whole-brain network modeling** and simulation
- **Biophysically realistic brain dynamics**
- **Advanced machine learning** for neuroimaging analysis
- **Functional connectivity modeling** and validation
- **Multi-scale brain simulation** from neurons to networks
- **Data-driven brain modeling** using empirical neuroimaging

**Key Scientific Advances:**
- Link structural connectivity to functional dynamics
- Predict brain activity patterns from network models
- Advanced statistical learning for neuroimaging
- Validate computational models against empirical data
- Multi-modal integration of imaging and simulation

**Applications:**
- Computational psychiatry and disease modeling
- Predicting brain dynamics from structural connectivity
- Advanced machine learning on neuroimaging data
- Virtual brain surgery planning and outcome prediction
- Multi-scale brain modeling and simulation
- Research integrating data analysis with computational models

---

## Tools in This Batch

### 1. The Virtual Brain (TVB)
**Website:** https://www.thevirtualbrain.org/
**GitHub:** https://github.com/the-virtual-brain/tvb-root
**Platform:** Python (Windows/macOS/Linux)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
The Virtual Brain (TVB) is an open-source neuroinformatics platform for simulating large-scale brain network dynamics using empirically-derived structural connectivity. TVB enables researchers to build personalized brain models from subject-specific neuroimaging data (structural/diffusion MRI, fMRI, MEG/EEG), simulate brain dynamics using biophysical models, and validate predictions against empirical recordings.

**Key Capabilities:**
- Whole-brain network simulation with biophysical models
- Integration of structural connectivity (DTI/tractography) into simulations
- Multiple neural mass models (Wilson-Cowan, Kuramoto, FitzHugh-Nagumo, etc.)
- Forward modeling for fMRI, MEG, EEG signals
- Parameter exploration and optimization
- Personalized brain models from individual connectivity
- Interactive GUI and Python scripting API
- Multi-scale modeling from regions to neurons
- Disease modeling (epilepsy, stroke, tumors, neurodegenerative)
- Virtual lesion studies and surgical planning

**Target Audience:**
- Computational neuroscientists modeling brain dynamics
- Clinical researchers studying disease mechanisms
- Theoreticians linking structure to function
- Researchers validating network models against data

**Estimated Lines:** 650-700
**Estimated Code Examples:** 20-24

**Key Topics to Cover:**
1. **Installation and Setup**
   - Install TVB (pip, Docker, source)
   - Launch TVB GUI and web interface
   - Python API setup and environment

2. **Building Connectivity Models**
   - Import structural connectivity matrices
   - Load default and custom atlases
   - Integrate DTI tractography data
   - Define region parcellations

3. **Neural Mass Models**
   - Wilson-Cowan oscillators
   - Generic 2D oscillator
   - Kuramoto model
   - Epileptor and seizure modeling
   - Custom model development

4. **Simulation Configuration**
   - Set integration parameters
   - Configure monitors (BOLD, EEG, MEG)
   - Noise and stochasticity
   - Coupling functions

5. **Running Simulations**
   - Launch whole-brain simulations
   - Parameter sweeps and exploration
   - GPU acceleration
   - Batch simulation management

6. **Analysis and Visualization**
   - Time series analysis
   - Functional connectivity from simulations
   - 3D brain visualization
   - Compare simulated vs empirical data

7. **Personalized Brain Models**
   - Subject-specific connectivity
   - Patient-specific disease models
   - Virtual lesions and surgery
   - Outcome prediction

8. **Integration & Advanced Use**
   - Link to neuroimaging pipelines
   - Export results for further analysis
   - Multi-modal data integration
   - Troubleshooting and optimization

**Example Workflows:**
- Simulate resting-state fMRI from structural connectivity
- Model epileptic seizure propagation in patient brain
- Virtual lesion studies for surgical planning
- Parameter optimization to fit empirical MEG data

---

### 2. BrainIAK (Brain Imaging Analysis Kit)
**Website:** https://brainiak.org/
**GitHub:** https://github.com/brainiak/brainiak
**Platform:** Python (Linux/macOS)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
BrainIAK (Brain Imaging Analysis Kit) is a Python package for advanced fMRI analysis developed by Intel and Princeton Neuroscience Institute. It provides high-performance implementations of state-of-the-art algorithms for multivariate pattern analysis, real-time fMRI, functional connectivity, and machine learning on neuroimaging data. BrainIAK emphasizes computational efficiency with MPI parallelization and is optimized for HPC environments.

**Key Capabilities:**
- Searchlight analysis with MPI parallelization
- Full correlation matrix analysis (FCMA)
- Shared response modeling (SRM) for hyperalignment
- Real-time fMRI analysis and neurofeedback
- Inter-subject correlation (ISC) and inter-subject functional connectivity (ISFC)
- Representational similarity analysis (RSA)
- Event segmentation and time-series analysis
- Template-based rotation for small sample sizes
- Multi-voxel pattern analysis (MVPA)
- HPC-optimized with MPI and Cython

**Target Audience:**
- Researchers performing multivariate fMRI analysis
- Cognitive neuroscientists using MVPA methods
- Real-time fMRI and neurofeedback studies
- Large-scale neuroimaging studies with HPC resources
- Method developers needing efficient implementations

**Estimated Lines:** 650-700
**Estimated Code Examples:** 20-24

**Key Topics to Cover:**
1. **Installation and Setup**
   - Install BrainIAK via pip/conda
   - MPI setup for parallelization
   - HPC environment configuration
   - Dependencies (NumPy, SciPy, scikit-learn, NiBabel)

2. **Searchlight Analysis**
   - Whole-brain searchlight MVPA
   - MPI parallelization for speed
   - Custom searchlight functions
   - Statistical inference

3. **Shared Response Modeling (SRM)**
   - Functional alignment across subjects
   - Hyperalignment for inter-subject analysis
   - Dimensionality reduction
   - Template construction

4. **Inter-Subject Correlation (ISC)**
   - ISC for naturalistic stimuli
   - Inter-subject functional connectivity (ISFC)
   - Statistical testing
   - Time-segment analysis

5. **Real-Time fMRI**
   - Real-time preprocessing
   - Incremental GLM
   - Neurofeedback implementation
   - Online pattern classification

6. **Event Segmentation**
   - Hidden Markov models for events
   - Boundary detection in time series
   - Cross-subject event alignment
   - Hierarchical event models

7. **Representational Similarity Analysis (RSA)**
   - Compute representational dissimilarity matrices
   - Compare neural representations
   - Model-based RSA
   - Searchlight RSA

8. **Integration & Workflows**
   - Preprocessing with fMRIPrep
   - HPC job submission
   - Batch processing multiple subjects
   - Visualization and reporting

**Example Workflows:**
- Whole-brain searchlight decoding with MPI
- Shared response model for movie-watching fMRI
- Real-time neurofeedback experiment
- Inter-subject correlation during naturalistic viewing

---

## Success Criteria
- Two new skills authored with comprehensive coverage of installation, core methods, workflows, and troubleshooting
- Each skill includes ~650-700 lines with ~20-24 code examples
- TVB skill emphasizes whole-brain simulation, biophysical models, and personalized brain modeling
- BrainIAK skill emphasizes advanced MVPA methods, HPC optimization, and real-time fMRI
- Clear integration examples with neuroimaging pipelines (fMRIPrep, tractography)
- Practical workflows demonstrating computational modeling and advanced machine learning
