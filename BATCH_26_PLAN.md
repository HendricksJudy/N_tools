# Batch 26: Advanced Diffusion & DMRI - Planning Document

## Overview

**Batch Theme:** Advanced Diffusion & DMRI
**Batch Number:** 26
**Number of Skills:** 2
**Current Progress:** 93/133 skills completed (69.9%)
**After Batch 26:** 95/133 skills (71.4%)

## Rationale

Batch 26 focuses on **advanced diffusion MRI processing and tractography pipelines** to deliver robust end-to-end diffusion workflows. These tools provide:

- **Integrated pipelines** that automate preprocessing, modeling, and tractography
- **Clinically relevant visualization** for neurosurgical planning and research
- **Reproducible containerized workflows** that scale from desktops to clusters
- **Quality control and reporting** to assess diffusion data fidelity
- **Interoperability with BIDS and 3D Slicer ecosystems**

**Key Scientific Advances:**
- Streamlined diffusion workflows reduce manual errors and variability
- Advanced modeling and tracking improve white matter characterization
- Interactive 3D visualization accelerates quality review and clinical communication
- Containerization ensures reproducibility across sites

**Applications:**
- Clinical neurosurgical planning and connectomics
- Research studies requiring standardized tractography
- Large-scale diffusion processing on HPC or cloud systems
- Teaching and demonstrations using 3D Slicer

---

## Tools in This Batch

### 1. SlicerDMRI (Diffusion MRI Extension for 3D Slicer)
**Website:** https://dmri.slicer.org/
**GitHub:** https://github.com/SlicerDMRI/SlicerDMRI
**Platform:** 3D Slicer extension (Windows/macOS/Linux)
**Priority:** Medium
**Current Status:** Does Not Exist - Need to Create

**Overview:**
SlicerDMRI is the official diffusion MRI toolkit for 3D Slicer. It delivers interactive visualization, preprocessing, tractography, and quantitative analysis within a clinical-grade 3D environment. The extension supports diffusion tensor imaging (DTI), diffusion spectrum imaging (DSI), and higher-order models, enabling neurosurgical planning and research-grade tract analysis.

**Key Capabilities:**
- DWI preprocessing (eddy correction, bias field correction, gradient table handling)
- Tensor and higher-order model fitting (DTI, HARDI, multi-shell)
- Deterministic and probabilistic tractography
- Interactive 3D visualization and tract editing
- Region-of-interest seeding and labelmap-based selection
- Quantitative tract metrics (FA, MD, RD, AD) and tract profiles
- Tract clustering and cleaning utilities
- Integration with 3D Slicer segmentation and registration modules
- Export to common formats (VTK, NIfTI, MRML)

**Target Audience:**
- Clinical and research users needing interactive diffusion visualization
- Neurosurgeons and neuroradiologists for preoperative planning
- Researchers performing tract-based analyses within 3D Slicer

**Estimated Lines:** 600-650
**Estimated Code Examples:** 18-22

**Key Topics to Cover:**
1. **Installation and Setup**
   - Install 3D Slicer
   - Add SlicerDMRI extension via Extension Manager
   - Verify diffusion toolbars and modules

2. **Data Preparation**
   - Import DWI, b-values, and b-vectors
   - Gradient table checks and reorientation
   - DWI denoising and bias correction

3. **Model Fitting**
   - Tensor fitting (FA, MD, eigenvectors)
   - HARDI and multi-shell reconstruction options
   - Handling multi-b-value datasets

4. **Tractography**
   - Deterministic and probabilistic seeding
   - ROI-based seeding and exclusion
   - Tract editing, pruning, and cleaning

5. **Visualization**
   - 3D tract rendering and slice overlays
   - Color-by-orientation and scalar overlays
   - Interactive tract selection and labeling

6. **Quantification & Export**
   - Tract profiles and region statistics
   - Export tracts and scalar volumes to NIfTI/VTK
   - Screenshots and 3D scenes for reporting

7. **Integration & Workflows**
   - Registration with structural MRI
   - Using Slicer modules for segmentation/labeling
   - Pipeline templates for clinical review

8. **Quality Control & Troubleshooting**
   - Motion/artifact detection
   - Gradient inconsistencies
   - Common tractography pitfalls and parameter tuning

**Example Workflows:**
- Pre-surgical corticospinal tract planning
- Tract-based FA/MD analysis for research cohorts
- Interactive QC of diffusion acquisitions and tracts

---

### 2. TractoFlow (Reproducible Tractography Pipeline)
**Website:** https://tractoflow-documentation.readthedocs.io/
**GitHub:** https://github.com/scilus/tractoflow
**Platform:** Nextflow + Singularity/Docker
**Priority:** Medium
**Current Status:** Does Not Exist - Need to Create

**Overview:**
TractoFlow is a fully automated and reproducible diffusion MRI processing pipeline built with Nextflow. It integrates best-practice steps—denoising, Gibbs ringing removal, bias correction, motion/eddy correction, susceptibility correction, model fitting, tractography, and bundle segmentation—using Singularity/Docker containers for consistent results across HPC and cloud environments.

**Key Capabilities:**
- BIDS-friendly inputs and organized outputs
- Automated preprocessing (denoising, Gibbs, bias, motion/eddy, susceptibility)
- Multi-shell support for advanced modeling
- FOD estimation and anatomically constrained tractography
- Whole-brain tractography with filtering (SIFT/SIFT2-style)
- Bundle segmentation and recognition
- Quality control reports and logs
- HPC- and cloud-ready execution via Nextflow
- Reproducible containerized environment (Singularity/Docker)

**Target Audience:**
- Researchers needing standardized, reproducible diffusion pipelines
- Labs running large diffusion cohorts on HPC or cloud
- Users migrating from ad-hoc scripts to containerized workflows

**Estimated Lines:** 600-650
**Estimated Code Examples:** 18-22

**Key Topics to Cover:**
1. **Installation and Setup**
   - Install Nextflow
   - Configure Singularity or Docker
   - Download TractoFlow pipelines and test datasets

2. **Input Preparation**
   - Expected directory structure (BIDS or custom)
   - DWI, bval, bvec validation
   - Optional fieldmaps for susceptibility correction

3. **Running the Pipeline**
   - Basic Nextflow command with minimal parameters
   - Specifying container engine and resource requests
   - Multi-subject batch runs on HPC/cloud

4. **Processing Stages**
   - Denoising, Gibbs removal, bias correction
   - Motion and eddy current correction
   - Susceptibility correction (topup/fieldmaps)
   - Model fitting (DTI, FOD)
   - Tractography generation and filtering

5. **Outputs & QC**
   - Directory layout and key derivatives
   - QC reports/logs and what to inspect
   - Exporting tracts and metrics for downstream analysis

6. **Advanced Configuration**
   - Custom resource settings and parallelization
   - Selecting models and tracking parameters
   - Resuming pipelines and caching

7. **Integration & Scaling**
   - Coupling with fMRIPrep or anatomical pipelines
   - Cloud execution patterns
   - Data management tips for large cohorts

8. **Troubleshooting**
   - Container/runtime issues
   - Common diffusion failures (bvec errors, motion)
   - Performance tuning

**Example Workflows:**
- End-to-end reproducible processing of a 50-subject multi-shell study
- HPC batch execution with Singularity and SLURM
- Tractography outputs for connectome construction or bundle analysis

---

## Success Criteria
- Two new skills authored with coverage of installation, core workflows, QC, and troubleshooting.
- Each skill includes ~600-650 lines with ~18-22 examples, emphasizing reproducibility and QC.
- Clear guidance on HPC/cloud execution for TractoFlow and interactive 3D workflows for SlicerDMRI.
