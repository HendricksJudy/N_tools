# Batch 21: PET Imaging & Reconstruction - Planning Document

## Overview

**Batch Theme:** PET Imaging & Reconstruction
**Batch Number:** 21
**Number of Skills:** 4
**Current Progress:** 75/133 skills completed (56.4%)
**After Batch 21:** 79/133 skills (59.4%)

## Rationale

Batch 21 focuses on **PET (Positron Emission Tomography) imaging and reconstruction** tools. While previous batches have covered structural MRI, functional MRI, and diffusion MRI extensively, PET imaging has been relatively underrepresented. This batch addresses specialized tools for:

- **PET image reconstruction** from raw data
- **PET/MR synergistic reconstruction** combining modalities
- **PET data processing and analysis** with versatile viewers
- **Tomographic image reconstruction** with advanced algorithms
- **CUDA-accelerated processing** for fast PET workflows

These tools are essential for:
- **Molecular imaging research:** Amyloid, tau, FDG, receptor imaging
- **Clinical PET studies:** Oncology, neurology, cardiology
- **PET/MR multi-modal imaging:** Combined structural-molecular
- **Quantitative PET analysis:** SUVR, kinetic modeling
- **Method development:** Novel reconstruction algorithms

## Tools in This Batch

### 1. AMIDE (Amide's a Medical Image Data Examiner)
**Website:** http://amide.sourceforge.net/
**Platform:** Linux/macOS/Windows
**Language:** C (GTK+ GUI)
**Priority:** High

**Overview:**
AMIDE is a free, open-source medical imaging viewer and analysis tool with particular strength in PET, SPECT, and multi-modal imaging. It provides comprehensive viewing capabilities for various formats (DICOM, ANALYZE, NIFTI), ROI drawing and analysis, co-registration, and advanced visualization. AMIDE is particularly valuable for PET analysis with built-in tools for SUV calculation, time-activity curves, and kinetic modeling.

**Key Capabilities:**
- Multi-format support (DICOM, NIFTI, ANALYZE, Interfile)
- PET, SPECT, CT, MRI visualization
- 3D volume rendering
- Multi-planar reformatting
- ROI drawing and analysis
- Image registration and fusion
- Time-activity curve analysis
- SUV (Standardized Uptake Value) calculation
- Kinetic modeling
- DICOM import/export
- Scripting support (XML-based)
- Cross-platform GUI

**Target Audience:**
- PET/SPECT researchers
- Clinical nuclear medicine
- Multi-modal imaging studies
- ROI-based PET quantification
- Teaching and demonstration

**Estimated Lines:** 650-700
**Estimated Code Examples:** 26-30

**Key Topics to Cover:**
1. Installation (Linux, macOS, Windows)
2. Load and visualize PET data
3. Multi-modal image registration (PET-MR, PET-CT)
4. ROI drawing for PET analysis
5. SUV calculation and normalization
6. Time-activity curve extraction
7. Dynamic PET analysis
8. Kinetic modeling
9. 3D volume rendering
10. Multi-planar visualization
11. DICOM handling
12. Batch processing with XML scripts
13. Export results and images
14. Integration with other tools

**Example Workflows:**
- Calculate SUVr for amyloid PET
- Extract time-activity curves from dynamic FDG PET
- Register PET to MRI for anatomical localization
- ROI-based quantification across timepoints
- Multi-subject PET analysis

**Integration Points:**
- FreeSurfer (anatomical ROIs)
- FSL (image processing)
- SPM (statistical analysis)
- DICOM from scanners

---

### 2. NiftyPET
**Website:** https://niftypet.readthedocs.io/
**Platform:** Python/CUDA
**Language:** Python with CUDA acceleration
**Priority:** High

**Overview:**
NiftyPET is a GPU-accelerated Python package for high-throughput PET image reconstruction and analysis. Developed at UCL, it provides fast, accurate image reconstruction from PET raw data with support for attenuation correction, scatter correction, and motion correction. NiftyPET leverages NVIDIA CUDA for massive speedups and integrates with standard Python scientific libraries.

**Key Capabilities:**
- GPU-accelerated PET reconstruction (CUDA)
- List-mode and sinogram reconstruction
- Attenuation correction (μ-maps from CT or MR)
- Scatter and randoms correction
- Motion correction
- Time-of-flight (TOF) reconstruction
- Point spread function (PSF) modeling
- Iterative reconstruction (OSEM, MLEM)
- Siemens Biograph mMR support
- Python API for automation
- Integration with NiftyReg for registration
- DICOM and NIFTI I/O

**Target Audience:**
- PET physics and methodology research
- High-throughput PET studies
- PET/MR imaging
- GPU computing for PET
- Advanced reconstruction algorithms

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**
1. Installation (CUDA requirements, GPU setup)
2. Load PET raw data (list-mode, sinograms)
3. Basic OSEM reconstruction
4. Attenuation correction from CT/MR
5. Scatter correction
6. Motion correction
7. TOF reconstruction
8. PSF modeling
9. Iterative reconstruction parameters
10. GPU optimization
11. Batch reconstruction
12. Integration with NiftyReg
13. Quality control
14. Export reconstructed images

**Example Workflows:**
- Reconstruct dynamic FDG PET with motion correction
- PET/MR reconstruction with MR-based attenuation
- High-resolution amyloid PET with PSF
- Batch reconstruct large study
- Compare reconstruction algorithms

**Integration Points:**
- NiftyReg (registration)
- FSL (preprocessing)
- SPM (analysis)
- FreeSurfer (anatomical segmentation)

---

### 3. SIRF (Synergistic Image Reconstruction Framework)
**Website:** https://www.ccppetmr.ac.uk/sites/sirf
**Platform:** Python/C++
**Language:** Python and C++
**Priority:** High

**Overview:**
SIRF is a comprehensive framework for PET and MR image reconstruction developed by the Collaborative Computational Project in Positron Emission Tomography and Magnetic Resonance imaging (CCP PET-MR). It provides a unified Python interface to multiple reconstruction engines (STIR for PET, Gadgetron for MR) enabling synergistic multi-modal reconstruction and joint PET-MR optimization.

**Key Capabilities:**
- Unified interface to STIR (PET) and Gadgetron (MR)
- Synergistic PET-MR reconstruction
- Motion-corrected reconstruction
- Multi-modal prior-based reconstruction
- Iterative reconstruction algorithms
- Python scripting for workflows
- Integration with registration tools
- Support for PET and MR raw data
- Joint estimation of activity and motion
- GPU acceleration support
- Educational framework for reconstruction
- Open-source and extensible

**Target Audience:**
- PET-MR researchers
- Reconstruction algorithm developers
- Multi-modal imaging studies
- Motion correction research
- Method development and teaching

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**
1. Installation (SIRF, STIR, Gadgetron dependencies)
2. Load PET data with SIRF
3. Basic PET reconstruction with STIR engine
4. Load MR data
5. MR reconstruction with Gadgetron
6. Attenuation correction
7. Motion estimation and correction
8. Synergistic PET-MR reconstruction
9. Prior-based reconstruction
10. Registration within SIRF
11. Python scripting workflows
12. Optimization and parameters
13. Quality control and validation
14. Integration with analysis tools

**Example Workflows:**
- Reconstruct PET with MR-derived motion
- Joint PET-MR reconstruction with anatomical prior
- Motion-corrected dynamic PET
- Compare reconstruction algorithms
- Educational reconstruction demonstrations

**Integration Points:**
- STIR (PET reconstruction engine)
- Gadgetron (MR reconstruction)
- NiftyReg (registration)
- SPM (analysis)

---

### 4. STIR (Software for Tomographic Image Reconstruction)
**Website:** http://stir.sourceforge.net/
**Platform:** C++
**Language:** C++ with Python bindings
**Priority:** Medium-High

**Overview:**
STIR is a comprehensive, open-source library for tomographic image reconstruction with particular focus on PET and SPECT. It provides a wide range of reconstruction algorithms (FBP, OSEM, OSSPS), correction methods (attenuation, scatter, randoms), and utilities for sinogram processing. STIR is highly flexible, supporting multiple scanner geometries and enabling method development.

**Key Capabilities:**
- Multiple reconstruction algorithms (FBP, MLEM, OSEM, OSSPS)
- Attenuation, scatter, and randoms correction
- Multiple scanner geometries
- Sinogram and list-mode data
- Forward and back-projection
- Reconstruction with PSF modeling
- Parameterized reconstruction (parfile-based)
- Python bindings (via SIRF)
- Extensible framework for development
- Utilities for data processing
- Monte Carlo scatter simulation
- Support for TOF data

**Target Audience:**
- PET/SPECT reconstruction research
- Algorithm developers
- Scanner calibration and QC
- Method validation
- Advanced PET physics

**Estimated Lines:** 650-700
**Estimated Code Examples:** 26-30

**Key Topics to Cover:**
1. Installation (C++ compilation, dependencies)
2. Data formats and conversion
3. Parameter files (parfiles) structure
4. FBP reconstruction
5. OSEM reconstruction
6. Attenuation correction
7. Scatter simulation and correction
8. Randoms estimation
9. PSF modeling
10. Sinogram processing
11. Forward and back-projection
12. Custom algorithm development
13. Integration with SIRF (Python)
14. Quality control

**Example Workflows:**
- Basic PET reconstruction with OSEM
- Full quantitative reconstruction with corrections
- Compare reconstruction algorithms
- Develop custom reconstruction method
- Monte Carlo scatter estimation

**Integration Points:**
- SIRF (Python interface)
- Scanner data formats
- Analysis tools (SPM, FSL)
- Simulation tools (GATE)

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **AMIDE** (most user-friendly, GUI-based)
   - **NiftyPET** (Python, popular for research)
   - **SIRF** (comprehensive framework)
   - **STIR** (foundation library)

2. **Comprehensive Coverage:**
   - Each skill: 650-750 lines
   - 26-32 code examples per skill
   - Real-world PET analysis workflows
   - Integration with neuroimaging pipelines

3. **Consistent Structure:**
   - Overview and key features
   - Installation (including GPU for NiftyPET)
   - Basic reconstruction/analysis
   - Advanced techniques
   - Quantification methods
   - Batch processing
   - Integration with other tools
   - Troubleshooting
   - Best practices
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Cross-platform installation
   - GPU setup (NiftyPET)
   - Dependency management
   - Verification

2. **Basic Usage** (6-8)
   - Load PET data
   - Simple reconstruction/viewing
   - Basic quantification
   - Export results

3. **Advanced Reconstruction** (6-8)
   - Attenuation correction
   - Scatter correction
   - Motion correction
   - Iterative algorithms
   - TOF/PSF modeling

4. **Quantification** (4-6)
   - SUV calculation
   - ROI analysis
   - Time-activity curves
   - Kinetic modeling

5. **Multi-Modal Integration** (3-5)
   - PET-MR registration
   - PET-CT fusion
   - Anatomical ROIs
   - Prior-based reconstruction

6. **Automation** (4-6)
   - Batch processing
   - Python scripting
   - Pipeline integration
   - Reproducible workflows

7. **Real-World Applications** (3-5)
   - Amyloid imaging analysis
   - FDG metabolism studies
   - Clinical quantification
   - Research workflows

### Cross-Tool Integration

All skills will demonstrate integration with:
- **FreeSurfer:** Anatomical segmentation for ROIs
- **FSL:** Image processing and registration
- **SPM:** Statistical analysis of PET
- **NiBabel:** File I/O in Python
- **DICOM:** Scanner data import
- **Claude Code:** Automated PET pipelines

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
| AMIDE | 650-700 | 26-30 | High |
| NiftyPET | 700-750 | 28-32 | High |
| SIRF | 700-750 | 28-32 | High |
| STIR | 650-700 | 26-30 | Medium-High |
| **TOTAL** | **2,700-2,900** | **108-124** | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ PET image viewing and analysis (AMIDE)
- ✓ GPU-accelerated reconstruction (NiftyPET)
- ✓ Synergistic PET-MR reconstruction (SIRF)
- ✓ Tomographic reconstruction library (STIR)
- ✓ Quantitative PET analysis (SUV, kinetic modeling)
- ✓ Multi-modal PET imaging
- ✓ Clinical and research applications

**Language/Platform Coverage:**
- C/GTK+: AMIDE (1/4)
- Python/CUDA: NiftyPET (1/4)
- Python/C++: SIRF (1/4)
- C++ with Python: STIR (1/4)

**Application Areas:**
- Clinical viewing: AMIDE
- Research reconstruction: NiftyPET, SIRF, STIR
- Multi-modal: All tools
- GPU acceleration: NiftyPET, SIRF (optional)

---

## Strategic Importance

### Fills Critical Gap

Previous batches have focused heavily on:
- Structural MRI (FreeSurfer, ANTs, CAT12, etc.)
- Functional MRI (SPM, FSL, AFNI, fMRIPrep, etc.)
- Diffusion MRI (DIPY, MRtrix3, QSIPrep, TractoFlow, DMIPY, MDT, etc.)

**Batch 21 adds:**
- PET imaging and reconstruction
- Molecular imaging capabilities
- Multi-modal PET-MR analysis
- Quantitative nuclear medicine
- Tomographic reconstruction methods

### Complementary Skills

**Works with existing skills:**
- **FreeSurfer:** Anatomical ROIs for PET quantification
- **FSL:** Image processing and registration
- **SPM:** Statistical analysis of PET (already has PET support)
- **ANTs:** Registration for PET-MR alignment
- **NiBabel:** File I/O for PET data
- **Clinica:** Clinical PET pipelines (already covered)

### User Benefits

1. **Clinical PET Analysis:**
   - Amyloid imaging for Alzheimer's
   - FDG metabolism studies
   - Tumor imaging quantification
   - Multi-timepoint comparison

2. **Research Applications:**
   - Novel tracer development
   - Receptor imaging
   - Tau PET studies
   - PET-MR protocols

3. **Method Development:**
   - Reconstruction algorithm development
   - Motion correction methods
   - Quantification validation
   - Scanner calibration

4. **Multi-Modal Integration:**
   - PET-MR synergistic reconstruction
   - Anatomically-guided PET analysis
   - Combined structural-molecular imaging
   - Multi-modal biomarkers

---

## Dependencies and Prerequisites

### Software Prerequisites

**AMIDE:**
- GTK+ 2.x or 3.x
- libvolpack (for 3D rendering)
- Optional: FFMPEG for video export
- Cross-platform (Linux, macOS, Windows)

**NiftyPET:**
- Python 3.7+
- NVIDIA GPU with CUDA 9.0+
- CUDA toolkit
- NumPy, SciPy, NiBabel
- NiftyReg (for registration)

**SIRF:**
- Python 3.6+
- STIR 4.0+
- Gadgetron (for MR)
- CMake, C++ compiler
- Optional: CUDA for GPU
- NiftyReg, SPM registration

**STIR:**
- C++ compiler (GCC, Clang, MSVC)
- CMake
- Optional: Python for bindings
- Optional: CUDA for GPU acceleration
- ITK (optional, for additional I/O)

### Data Prerequisites

**Common to all:**
- PET raw data (list-mode or sinograms) for reconstruction
- Reconstructed PET images (DICOM, NIFTI) for analysis
- Anatomical images (CT, MR) for registration and attenuation
- Scanner-specific calibration data

**Tool-specific:**
- **AMIDE:** DICOM or NIFTI PET images
- **NiftyPET:** Siemens mMR raw data or generic formats
- **SIRF:** PET raw data compatible with STIR
- **STIR:** Sinogram or list-mode data

### Knowledge Prerequisites

Users should understand:
- Basic PET physics (annihilation, coincidence detection)
- Reconstruction concepts (sinograms, back-projection, OSEM)
- Attenuation and scatter correction
- SUV calculation and normalization
- Multi-modal registration concepts
- Python programming (for NiftyPET, SIRF)
- C++ basics (for STIR development)

---

## Learning Outcomes

After completing Batch 21 skills, users will be able to:

1. **Analyze PET Data:**
   - View and quantify PET images
   - Calculate SUV and SUVr
   - Extract ROI statistics
   - Perform time-activity curve analysis

2. **Reconstruct PET Images:**
   - Perform basic OSEM reconstruction
   - Apply attenuation correction
   - Implement scatter correction
   - Use GPU acceleration

3. **Multi-Modal Analysis:**
   - Register PET to MR/CT
   - Use anatomical ROIs for PET quantification
   - Perform synergistic reconstruction
   - Combine structural and molecular information

4. **Advanced Techniques:**
   - Motion-corrected reconstruction
   - TOF and PSF modeling
   - Kinetic modeling
   - Algorithm development

5. **Automation:**
   - Batch process PET studies
   - Create reproducible pipelines
   - Script quantification workflows
   - Integrate with neuroimaging analyses

---

## Relationship to Existing Skills

### Builds Upon:
- **SPM** (Batch 1): PET analysis capabilities
- **FSL** (Batch 1): Registration tools
- **FreeSurfer** (Batch 1): Anatomical segmentation
- **ANTs** (Batch 3): Advanced registration
- **NiBabel** (Batch 2): File I/O
- **Clinica** (Batch 18): Clinical PET pipelines

### Complements:
- **Structural imaging:** MR for PET attenuation and ROIs
- **fMRI:** Multi-modal functional-molecular
- **Spectroscopy:** Metabolic imaging comparison
- **Visualization:** Surfice, Brainrender for PET rendering

### Enables:
- Quantitative molecular imaging
- PET-MR multi-modal studies
- Clinical nuclear medicine analysis
- PET reconstruction research
- Amyloid/tau/FDG imaging pipelines

---

## Expected Challenges and Solutions

### Challenge 1: GPU Requirements (NiftyPET)
**Issue:** CUDA and NVIDIA GPU required
**Solution:** Document requirements, provide CPU alternatives, container solutions

### Challenge 2: Raw Data Access
**Issue:** PET raw data often proprietary, scanner-specific
**Solution:** Focus on reconstructed data analysis, provide example datasets

### Challenge 3: Complex Dependencies (SIRF)
**Issue:** Multiple packages (STIR, Gadgetron) required
**Solution:** Detailed installation guide, Docker containers, step-by-step setup

### Challenge 4: Physics Knowledge Gap
**Issue:** PET reconstruction requires understanding of physics
**Solution:** Include background sections, explain corrections, provide references

### Challenge 5: Limited PET Adoption
**Issue:** Fewer users than MRI-focused tools
**Solution:** Emphasize clinical applications, multi-modal benefits, growing field

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Test commands for successful installation
   - GPU detection (NiftyPET)
   - Version checking
   - Dependency validation

2. **Basic Functionality Tests:**
   - Load sample PET data
   - Perform reconstruction or viewing
   - Calculate SUV
   - Export results

3. **Integration Tests:**
   - PET-MR registration
   - ROI-based quantification
   - Multi-modal workflows
   - Pipeline integration

4. **Example Data:**
   - Links to example PET datasets
   - Sample reconstruction scripts
   - Expected outputs
   - Validation metrics

---

## Timeline Estimate

**Per Skill:**
- Research and planning: 15-20 min
- Writing and examples: 40-50 min
- Review and refinement: 10-15 min
- **Total per skill:** ~65-85 min

**Total Batch 21:**
- 4 skills × 75 min average = ~300 min (~5 hours)
- Includes documentation, examples, and testing

**Can be completed in:** 1-2 extended sessions

---

## Success Criteria

Batch 21 will be considered successful when:

✓ All 4 skills created with 650-750 lines each
✓ Total of 108+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions (GPU for NiftyPET)
  - Basic to advanced PET analysis/reconstruction
  - SUV and quantification examples
  - Multi-modal integration (PET-MR, PET-CT)
  - Batch processing workflows
  - Clinical application examples
  - Troubleshooting section
  - Best practices
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 79/133 (59.4%)

---

## Next Batches Preview

### Batch 22: Multimodal Gradients & Transcriptomics
- BrainSpace (gradient analysis and manifolds)
- neuromaps (brain annotation maps)
- abagen (Allen Brain Atlas gene expression)
- BrainStat (statistical analysis toolbox)

### Batch 23: Network Analysis & Connectivity
- NetworkX (graph analysis in Python)
- Brain Connectivity Toolbox (BCT)
- BRAPH2 (brain graph analysis)
- NBS-Predict (network-based statistics)

### Batch 24: Specialized Statistical Tools
- PALM (permutation analysis)
- SnPM (statistical nonparametric mapping)
- SurfStat (surface-based statistics)
- LIMO EEG (linear modeling)

---

## Conclusion

Batch 21 represents an important expansion into **PET imaging and reconstruction**, a domain that has been underrepresented in previous batches despite its clinical and research importance. While we've extensively covered MRI (structural, functional, diffusion), Batch 21 enables researchers to:

- **Analyze molecular imaging data** with amyloid, tau, FDG, and other tracers
- **Reconstruct PET images** from raw data with advanced corrections
- **Integrate PET with MR** for synergistic multi-modal analysis
- **Quantify PET biomarkers** for clinical and research applications
- **Develop reconstruction methods** with open-source frameworks

These tools are critical for:
- Alzheimer's disease imaging (amyloid, tau PET)
- Oncological PET studies
- Receptor imaging and pharmacology
- PET-MR research protocols
- Clinical nuclear medicine

By covering AMIDE (viewing/analysis), NiftyPET (GPU reconstruction), SIRF (synergistic framework), and STIR (reconstruction library), we provide comprehensive PET capabilities from clinical analysis to advanced method development.

**Status After Batch 21:** 79/133 skills (59.4% complete - approaching 60%!)

---

**Document Version:** 1.0
**Created:** 2025-11-15
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 4 skills, ~2,700-2,900 lines, ~108-124 examples
