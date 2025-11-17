# Batch 31: Deep Learning & Medical Imaging Preprocessing for AI - Planning Document

## Overview

**Batch Theme:** Deep Learning & Medical Imaging Preprocessing for AI
**Batch Number:** 31
**Number of Skills:** 3
**Current Progress:** 105/133 skills completed (78.9%)
**After Batch 31:** 108/133 skills (81.2%)

## Rationale

Batch 31 focuses on **deep learning infrastructure and preprocessing tools** that enable AI/ML workflows in neuroimaging. While tools like MONAI, nnU-Net, and SynthSeg provide powerful deep learning models, this batch addresses the critical preprocessing, feature engineering, and harmonization steps required for successful deep learning applications. These tools enable:

- **PyTorch-native medical image preprocessing** with domain-specific augmentations
- **Radiomics feature extraction** for classical and hybrid ML approaches
- **Multi-site harmonization** to remove scanner/site effects before ML training
- **Reproducible ML pipelines** with standardized preprocessing
- **Integration** with deep learning frameworks (PyTorch, TensorFlow)
- **Quality assurance** for ML-ready datasets
- **Domain adaptation** and transfer learning support

**Key Scientific Advances:**
- GPU-accelerated medical image preprocessing
- Domain-specific data augmentation for medical imaging
- Quantitative radiomic biomarkers for ML
- Statistical harmonization preserving biological variance
- End-to-end differentiable preprocessing pipelines
- Batch processing for large-scale ML datasets
- Cross-site model generalization

**Applications:**
- Deep learning model training (segmentation, classification, prediction)
- Radiomics and quantitative imaging biomarkers
- Multi-site/multi-scanner studies
- Transfer learning and domain adaptation
- Federated learning preparation
- Clinical decision support systems
- Precision medicine and patient stratification

---

## Tools in This Batch

### 1. TorchIO
**Website:** https://torchio.readthedocs.io/
**GitHub:** https://github.com/fepegar/torchio
**Platform:** Python (PyTorch-based, cross-platform)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
TorchIO is a Python library for efficient loading, preprocessing, augmentation, and patch-based sampling of 3D medical images in deep learning pipelines. Built on PyTorch, it provides GPU-accelerated transforms specifically designed for medical imaging (unlike generic computer vision libraries), supports 4D data (3D + time), handles metadata preservation, and integrates seamlessly with PyTorch DataLoaders for training neural networks.

**Key Capabilities:**
- Medical-specific data augmentation (elastic deformation, bias field, motion artifacts)
- Efficient 3D/4D image loading and preprocessing
- Patch-based sampling for large volumes
- GPU-accelerated transforms
- DICOM and NIfTI support with metadata preservation
- Intensity normalization and standardization
- Queue-based patch sampling for memory efficiency
- Integration with PyTorch DataLoader
- Preprocessing pipeline composition
- Reproducible random transforms
- Label map handling for segmentation
- Multi-modal image registration
- Batch preprocessing utilities

**Target Audience:**
- Deep learning researchers training on medical images
- Developers building medical imaging AI models
- Radiomics researchers needing preprocessing
- Clinical AI developers
- Multi-site study coordinators
- Anyone using PyTorch for medical imaging

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install via pip/conda
   - PyTorch integration
   - GPU support verification
   - Basic usage examples

2. **Loading Medical Images**
   - Subject and Image classes
   - Load NIfTI and DICOM
   - Multi-modal images
   - Metadata handling
   - Label maps

3. **Preprocessing Transforms**
   - Resampling and resizing
   - Cropping and padding
   - Intensity normalization (z-score, min-max, histogram)
   - Bias field correction simulation
   - Motion artifact simulation
   - Ghosting and spike artifacts

4. **Data Augmentation**
   - Spatial transforms (affine, elastic deformation)
   - Intensity transforms (gamma, blur, noise)
   - Medical-specific augmentations
   - Random transforms with reproducibility
   - Compose multiple transforms

5. **Patch-Based Sampling**
   - GridSampler for sliding window
   - UniformSampler for random patches
   - WeightedSampler for label-based sampling
   - Queue for efficient memory usage
   - Aggregator for patch reconstruction

6. **PyTorch Integration**
   - Create Dataset and DataLoader
   - Training loop integration
   - Batch processing
   - Multi-GPU support

7. **Advanced Preprocessing**
   - Histogram standardization
   - Label remapping
   - One-hot encoding
   - Keep largest component
   - Ensure shape divisibility

8. **Quality Control**
   - Visualize transforms
   - Check preprocessing pipeline
   - Validate augmentations
   - Debug data loading

9. **Batch Processing**
   - Process datasets offline
   - Save preprocessed data
   - Parallel processing
   - Handle large cohorts

10. **Integration with Deep Learning**
    - nnU-Net preprocessing
    - MONAI compatibility
    - Custom model training
    - Transfer learning workflows

11. **Best Practices**
    - Pipeline design
    - Augmentation strategies
    - Memory optimization
    - Reproducibility

12. **Troubleshooting**
    - Common errors
    - Memory issues
    - Performance optimization
    - GPU utilization

**Example Workflows:**
- Brain tumor segmentation preprocessing
- Multi-site harmonization preprocessing
- 4D fMRI data augmentation
- Patch-based model training
- Cross-validation data splits

---

### 2. PyRadiomics
**Website:** https://pyradiomics.readthedocs.io/
**GitHub:** https://github.com/AIM-Harvard/pyradiomics
**Platform:** Python (cross-platform)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
PyRadiomics is a Python package for extracting Radiomics features from medical imaging. Developed at Harvard Medical School, it computes hundreds of quantitative imaging features including shape, intensity, and texture descriptors that can be used as biomarkers for machine learning models. PyRadiomics is highly configurable, validated against multiple phantom datasets, and widely used in oncology, neurology, and precision medicine research.

**Key Capabilities:**
- Extract 100+ standardized radiomic features
- Shape-based features (volume, surface area, sphericity, etc.)
- First-order intensity statistics (mean, variance, skewness, etc.)
- Texture features (GLCM, GLRLM, GLSZM, GLDM, NGTDM)
- Filter-based features (Wavelet, LoG, Gradient, etc.)
- Multi-region feature extraction
- Configurable feature extraction
- Image preprocessing (resampling, normalization, discretization)
- Batch feature extraction
- IBSI (Image Biomarker Standardization Initiative) compliant
- Integration with scikit-learn
- Feature reproducibility testing
- Export to CSV/JSON

**Target Audience:**
- Radiomics researchers
- Medical imaging ML practitioners
- Oncology researchers (tumor characterization)
- Neurology researchers (lesion analysis)
- Clinical researchers developing biomarkers
- Anyone building ML models on medical images

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install via pip
   - Verify installation
   - Dependencies
   - Example data

2. **Basic Feature Extraction**
   - Load image and mask
   - Extract all features
   - Feature types overview
   - Interpret results

3. **Shape Features**
   - Volumetric measurements
   - Surface area and density
   - Sphericity and compactness
   - Elongation and flatness
   - Mesh-based features

4. **First-Order Features**
   - Intensity statistics
   - Histogram analysis
   - Energy and entropy
   - Robust metrics (MAD, IQR)

5. **Texture Features**
   - Gray Level Co-occurrence Matrix (GLCM)
   - Gray Level Run Length Matrix (GLRLM)
   - Gray Level Size Zone Matrix (GLSZM)
   - Gray Level Dependence Matrix (GLDM)
   - Neighboring Gray Tone Difference Matrix (NGTDM)

6. **Image Filters**
   - Wavelet decomposition features
   - Laplacian of Gaussian (LoG)
   - Gradient magnitude
   - Local Binary Pattern (LBP)
   - Square and exponential filters

7. **Preprocessing Configuration**
   - Resampling
   - Intensity normalization
   - Discretization (binning)
   - Mask validation
   - Distance maps

8. **Batch Feature Extraction**
   - Process multiple subjects
   - Parallel extraction
   - Handle errors gracefully
   - Aggregate features across cohort

9. **Feature Selection for ML**
   - Export features for scikit-learn
   - Feature correlation analysis
   - Dimensionality reduction
   - Feature importance

10. **Quality Control**
    - IBSI compliance testing
    - Feature reproducibility
    - Phantom validation
    - Cross-scanner stability

11. **Integration with ML Pipelines**
    - Scikit-learn integration
    - Feature normalization
    - Cross-validation
    - Model interpretation

12. **Advanced Applications**
    - Multi-region analysis
    - Longitudinal feature tracking
    - Radiogenomics
    - Treatment response prediction

**Example Workflows:**
- Brain tumor radiomics for glioma grading
- Alzheimer's disease hippocampal features
- Lesion characterization in MS
- Multi-site radiomic biomarker validation
- Integration with survival analysis

---

### 3. NeuroHarmonize
**Website:** https://github.com/rpomponio/neuroHarmonize
**GitHub:** https://github.com/rpomponio/neuroHarmonize
**Platform:** Python (cross-platform)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
NeuroHarmonize is a Python implementation of the ComBat harmonization method for removing scanner and site effects from neuroimaging data while preserving biological variance. Originally developed for genomics batch effect correction, ComBat has been adapted and validated for multi-site neuroimaging studies. NeuroHarmonize provides empirical Bayes harmonization with support for covariates, COMBAT-GAM for non-linear effects, and integration with neuroimaging pipelines.

**Key Capabilities:**
- ComBat harmonization for multi-site/scanner effects
- Empirical Bayes shrinkage estimation
- Preserve biological variance (age, sex, diagnosis)
- Support for covariates and interaction terms
- COMBAT-GAM for non-linear site effects
- Voxel-wise and ROI-based harmonization
- Handle missing data
- Cross-validation support
- Statistical validation metrics
- Integration with scikit-learn
- Save/load harmonization models
- Apply to new data (out-of-sample harmonization)
- Batch processing for large datasets

**Target Audience:**
- Multi-site study coordinators
- Meta-analysis researchers
- Machine learning researchers (multi-scanner data)
- ENIGMA consortium members
- Clinical trial coordinators
- Anyone combining data from multiple scanners/sites

**Estimated Lines:** 650-700
**Estimated Code Examples:** 22-26

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install via pip/GitHub
   - Dependencies
   - Verify installation
   - Example data

2. **Understanding Harmonization**
   - Site/scanner effects in neuroimaging
   - ComBat methodology
   - When to harmonize
   - What not to harmonize

3. **Basic Harmonization**
   - Prepare data (features × subjects)
   - Specify site labels
   - Run ComBat harmonization
   - Interpret results

4. **Covariate Specification**
   - Biological covariates (age, sex)
   - Diagnosis/group preservation
   - Continuous vs categorical
   - Interaction terms

5. **COMBAT-GAM**
   - Non-linear site effects
   - Smooth covariate modeling
   - When to use GAM vs standard ComBat
   - Parameter tuning

6. **Validation and Assessment**
   - Check site effect removal
   - Verify biological variance preservation
   - Statistical tests (ANOVA, t-tests)
   - Visualization (PCA, distributions)

7. **ROI-Based Harmonization**
   - FreeSurfer thickness/volume
   - Subcortical volumes
   - Cortical area
   - DTI metrics (FA, MD)

8. **Voxel-Wise Harmonization**
   - Whole-brain voxel harmonization
   - Memory considerations
   - Parallel processing
   - Save harmonized images

9. **Out-of-Sample Harmonization**
   - Train harmonization model
   - Save model parameters
   - Apply to new subjects
   - Use cases (new sites, test sets)

10. **Integration with ML Pipelines**
    - Harmonize before ML
    - Cross-validation considerations
    - Feature engineering post-harmonization
    - Model generalization

11. **Quality Control**
    - Assess harmonization quality
    - Residual site effects
    - Biological signal preservation
    - Over-harmonization risks

12. **Best Practices**
    - When to harmonize
    - Covariate selection
    - Handling small samples
    - Reporting harmonization
    - Limitations and caveats

**Example Workflows:**
- Multi-site cortical thickness harmonization
- DTI metrics across scanners
- Subcortical volume harmonization for ENIGMA
- ML model training on multi-site data
- Longitudinal harmonization across scanner upgrades

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **TorchIO** - PyTorch medical image preprocessing (new, 700-750 lines)
   - **PyRadiomics** - Radiomics feature extraction (new, 700-750 lines)
   - **NeuroHarmonize** - Multi-site harmonization (new, 650-700 lines)

2. **Comprehensive Coverage:**
   - Each skill: 650-750 lines
   - 22-28 code examples per skill
   - Real-world deep learning workflows
   - Integration with existing tools

3. **Consistent Structure:**
   - Overview and key features
   - Installation (pip/conda)
   - Basic usage and concepts
   - Core functionality
   - Advanced features
   - Quality control
   - Integration with ML/DL frameworks
   - Batch processing
   - Best practices
   - Troubleshooting
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Package installation
   - Dependency verification
   - GPU setup (TorchIO)

2. **Basic Usage** (6-8)
   - Load data
   - Core functionality
   - Simple examples
   - Output interpretation

3. **Advanced Features** (6-8)
   - Complex workflows
   - Customization
   - Integration
   - Optimization

4. **Quality Control** (3-5)
   - Validation
   - Visualization
   - Statistical assessment
   - Reproducibility

5. **ML/DL Integration** (3-5)
   - PyTorch integration
   - scikit-learn pipelines
   - Model training
   - Cross-validation

6. **Batch Processing** (2-4)
   - Multi-subject workflows
   - Parallel processing
   - Large-scale processing
   - Error handling

7. **Best Practices** (2-4)
   - Recommended workflows
   - Parameter selection
   - Common pitfalls
   - Reporting standards

### Cross-Tool Integration

All skills will demonstrate integration with:
- **PyTorch**: Deep learning framework
- **MONAI (existing)**: Medical imaging deep learning
- **nnU-Net (existing)**: Segmentation framework
- **scikit-learn**: Machine learning
- **Nilearn (Batch 4)**: Neuroimaging ML
- **fMRIPrep (Batch 5)**: Preprocessing outputs
- **FreeSurfer (existing)**: ROI extraction

### Quality Targets

- **Minimum lines per skill:** 650
- **Target lines per skill:** 650-750
- **Minimum code examples:** 22
- **Target code examples:** 22-28
- **Total batch lines:** ~2,050-2,200
- **Total code examples:** ~68-82

---

## Batch Statistics

### Projected Totals

| Tool | Est. Lines | Est. Examples | Priority | Status |
|------|-----------|---------------|----------|---------|
| TorchIO | 700-750 | 24-28 | High | Create new |
| PyRadiomics | 700-750 | 24-28 | High | Create new |
| NeuroHarmonize | 650-700 | 22-26 | High | Create new |
| **TOTAL** | **2,050-2,200** | **70-82** | - | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Medical image preprocessing (TorchIO)
- ✓ Data augmentation for deep learning (TorchIO)
- ✓ Radiomics feature extraction (PyRadiomics)
- ✓ Multi-site harmonization (NeuroHarmonize)
- ✓ PyTorch integration (TorchIO)
- ✓ ML feature engineering (PyRadiomics, NeuroHarmonize)
- ✓ Batch processing (all tools)

**Platform Coverage:**
- Python: All tools (3/3)
- PyTorch: TorchIO (1/3)
- GPU-accelerated: TorchIO (1/3)
- Platform-independent: All tools (3/3)

**Application Areas:**
- Deep learning model training: TorchIO
- Radiomics research: PyRadiomics
- Multi-site studies: NeuroHarmonize
- Clinical AI development: All tools
- Precision medicine: PyRadiomics, NeuroHarmonize
- Transfer learning: TorchIO, NeuroHarmonize

---

## Strategic Importance

### Fills Critical Gap

Previous batches have covered:
- MONAI (existing): Deep learning framework
- nnU-Net (existing): Segmentation models
- SynthSeg/SynthSR (existing): Synthetic image generation
- Nilearn (Batch 4): Classical ML for neuroimaging

**Batch 31 adds:**
- **PyTorch-native preprocessing** for medical images
- **Domain-specific augmentation** unavailable in computer vision libraries
- **Standardized radiomics** for feature-based ML
- **Multi-site harmonization** for robust models
- **End-to-end ML pipelines** from preprocessing to deployment
- **Quality assurance** for AI-ready datasets

### Complementary Skills

**Works with existing skills:**
- **MONAI (existing):** Deep learning infrastructure
- **nnU-Net (existing):** Segmentation training
- **Nilearn (Batch 4):** Classical ML and analysis
- **fMRIPrep (Batch 5):** Preprocessed data as input
- **FreeSurfer (existing):** ROI definition for radiomics
- **ANTs (existing):** Registration and normalization

### User Benefits

1. **Deep Learning Enablement:**
   - GPU-accelerated preprocessing
   - Medical-specific augmentation
   - Memory-efficient patch sampling
   - PyTorch-native workflows

2. **Feature Engineering:**
   - Standardized radiomic features
   - IBSI-compliant extraction
   - Quantitative biomarkers
   - Feature reproducibility

3. **Multi-Site Robustness:**
   - Remove scanner effects
   - Preserve biological variance
   - Enable meta-analyses
   - Improve model generalization

4. **Research Quality:**
   - Reproducible pipelines
   - Validated methodologies
   - Standardized reporting
   - Best practices guidance

---

## Dependencies and Prerequisites

### Software Prerequisites

**TorchIO:**
- Python ≥ 3.7
- PyTorch ≥ 1.1
- SimpleITK, nibabel
- NumPy, SciPy

**PyRadiomics:**
- Python ≥ 3.6
- NumPy, SciPy
- SimpleITK
- Six, pykwalify, pywavelets

**NeuroHarmonize:**
- Python ≥ 3.6
- NumPy, pandas
- scikit-learn
- statsmodels

### Data Prerequisites

**Common to all:**
- Medical images in standard formats (NIfTI, DICOM)
- Appropriate metadata
- Quality-controlled inputs

**Tool-specific:**
- **TorchIO:** 3D/4D images, optional label maps
- **PyRadiomics:** Image + segmentation mask
- **NeuroHarmonize:** Feature table with site labels and covariates

### Knowledge Prerequisites

Users should understand:
- Basic Python programming
- Medical image formats
- Deep learning concepts (for TorchIO)
- Machine learning basics
- Statistical concepts (harmonization)
- Multi-site study design

---

## Learning Outcomes

After completing Batch 31 skills, users will be able to:

1. **Prepare Data for Deep Learning:**
   - Load and preprocess 3D medical images
   - Apply domain-specific augmentations
   - Create patch-based datasets
   - Build PyTorch DataLoaders

2. **Extract Radiomic Features:**
   - Compute standardized features
   - Configure feature extraction
   - Validate feature reproducibility
   - Integrate with ML pipelines

3. **Harmonize Multi-Site Data:**
   - Apply ComBat harmonization
   - Preserve biological variance
   - Validate harmonization quality
   - Handle out-of-sample data

4. **Build ML/DL Pipelines:**
   - End-to-end preprocessing
   - Feature engineering
   - Model training workflows
   - Quality assurance

5. **Ensure Reproducibility:**
   - Standardized preprocessing
   - Documented pipelines
   - Validated methods
   - Best practices compliance

---

## Relationship to Existing Skills

### Builds Upon:
- **MONAI (existing):** Deep learning framework
- **Nilearn (Batch 4):** ML for neuroimaging
- **fMRIPrep (Batch 5):** Preprocessed inputs
- **FreeSurfer (existing):** ROI segmentation

### Complements:
- **nnU-Net (existing):** Segmentation models
- **SynthSeg/SynthSR (existing):** Synthetic data
- **ANTs (existing):** Registration
- **BrainSpace (Batch 27):** Gradients and manifolds

### Enables:
- Robust deep learning model training
- Radiomics-based precision medicine
- Multi-site AI/ML studies
- Transfer learning across scanners
- Clinical AI deployment
- Federated learning preparation

---

## Expected Challenges and Solutions

### Challenge 1: GPU Memory Management
**Issue:** Large 3D volumes can exceed GPU memory
**Solution:** Patch-based sampling, batch size tuning, gradient accumulation, mixed precision

### Challenge 2: Feature Selection Complexity
**Issue:** PyRadiomics extracts 100+ features, risk of overfitting
**Solution:** Feature selection guidance, cross-validation, domain knowledge, dimensionality reduction

### Challenge 3: Harmonization Validation
**Issue:** Difficult to verify biological variance preservation
**Solution:** Multiple validation approaches, statistical testing, visualization, phantom studies

### Challenge 4: Computational Cost
**Issue:** Batch processing can be time-consuming
**Solution:** Parallel processing, GPU utilization, offline preprocessing, efficiency tips

### Challenge 5: Integration Complexity
**Issue:** Connecting multiple tools in pipelines
**Solution:** Clear workflow examples, integration patterns, troubleshooting guides

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Package import tests
   - Dependency checking
   - GPU availability (TorchIO)

2. **Basic Functionality Tests:**
   - Load example data
   - Run core functions
   - Verify outputs

3. **Integration Tests:**
   - PyTorch DataLoader (TorchIO)
   - scikit-learn pipeline (PyRadiomics, NeuroHarmonize)
   - End-to-end workflow

4. **Example Data:**
   - Public datasets
   - Toy examples
   - Expected outputs
   - Validation benchmarks

---

## Timeline Estimate

**Per Skill:**
- TorchIO: 70-85 min (new, comprehensive preprocessing)
- PyRadiomics: 70-85 min (new, many feature types)
- NeuroHarmonize: 60-75 min (new, focused harmonization)

**Total Batch 31:**
- ~3.5-4 hours total
- Can be completed in 1-2 extended sessions

---

## Success Criteria

Batch 31 will be considered successful when:

✓ All 3 skills created with 650-750 lines each
✓ Total of 70+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - Conceptual background
  - Basic and advanced usage
  - Quality control procedures
  - Integration with ML/DL frameworks
  - Batch processing examples
  - Best practices guidance
  - Troubleshooting section
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 108/133 (81.2%)

---

## Next Batches Preview

### Batch 32: Perfusion & ASL Imaging
- ExploreASL (ASL processing pipeline)
- BASIL (Bayesian ASL analysis from FSL)
- ASL-specific quantification tools

### Batch 33: Real-Time fMRI & Neurofeedback
- OpenNFT (real-time neurofeedback)
- Real-time processing infrastructure
- Neurofeedback experimental design

### Batch 34: Advanced Diffusion Microstructure
- DESIGNER (diffusion preprocessing)
- Advanced microstructure models
- Validation and simulation tools

---

## Conclusion

Batch 31 provides **deep learning infrastructure and preprocessing tools** that are essential for modern AI/ML neuroimaging research. By covering:

- **TorchIO** - PyTorch-native medical image preprocessing and augmentation
- **PyRadiomics** - Standardized radiomics feature extraction
- **NeuroHarmonize** - Multi-site harmonization with ComBat

This batch enables researchers to:
- **Build robust DL pipelines** with medical-specific preprocessing
- **Extract quantitative biomarkers** for ML models
- **Harmonize multi-site data** for better generalization
- **Accelerate AI research** with GPU-optimized tools
- **Ensure reproducibility** through standardized methods
- **Deploy clinical AI** with validated preprocessing

These tools are critical for:
- Deep learning model development
- Radiomics and precision medicine
- Multi-site and meta-analysis studies
- Clinical AI applications
- Transfer learning across scanners
- Federated learning preparation

By providing PyTorch-native preprocessing (TorchIO), standardized feature extraction (PyRadiomics), and statistical harmonization (NeuroHarmonize), Batch 31 establishes the foundational infrastructure for cutting-edge AI/ML neuroimaging research.

**Status After Batch 31:** 108/133 skills (81.2% complete - over 4/5 done!)

---

**Document Version:** 1.0
**Created:** 2025-11-17
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 3 skills, ~2,050-2,200 lines, ~70-82 examples
