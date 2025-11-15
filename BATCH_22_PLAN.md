# Batch 22: Multimodal Gradients & Transcriptomics - Planning Document

## Overview

**Batch Theme:** Multimodal Gradients & Transcriptomics
**Batch Number:** 22
**Number of Skills:** 4
**Current Progress:** 79/133 skills completed (59.4%)
**After Batch 22:** 83/133 skills (62.4%)

## Rationale

Batch 22 represents a strategic shift from imaging reconstruction to **integrative neuroscience**, focusing on tools that bridge multiple data modalities. This batch addresses the growing importance of:

- **Gradient-based manifold representations** of brain organization
- **Multi-modal brain mapping** integrating structure, function, genetics
- **Transcriptomic-neuroimaging integration** linking genes to brain phenotypes
- **Advanced statistical frameworks** for surface and volume-based analysis

These tools enable researchers to move beyond univariate analyses to explore the **organizational principles** of the brain across multiple scales and modalities, from genes to networks to behavior.

**Key Scientific Advances:**
- Gradient analysis reveals smooth transitions in brain organization (not discrete modules)
- Brain annotation maps link imaging to genetics, pharmacology, development
- Transcriptomics-imaging integration connects molecular mechanisms to macroscale patterns
- Unified statistical frameworks for hypothesis testing across modalities

**Applications:**
- Understanding brain organization via connectivity gradients
- Linking genetic variation to brain structure and function
- Multi-modal biomarker discovery
- Cross-species comparative neuroscience
- Transcriptomic signatures of neuropsychiatric disorders

---

## Tools in This Batch

### 1. BrainSpace
**Website:** https://brainspace.readthedocs.io/
**GitHub:** https://github.com/MICA-MNI/BrainSpace
**Platform:** Python/MATLAB
**Priority:** High

**Overview:**
BrainSpace is a comprehensive toolbox for gradient-based analysis of brain organization, developed at the Montreal Neurological Institute (MNI). It enables researchers to identify and analyze continuous spatial transitions (gradients) in brain connectivity, cortical microstructure, gene expression, and other features. BrainSpace implements manifold learning techniques (diffusion maps, Laplacian eigenmaps) to reveal the low-dimensional organizational principles underlying high-dimensional brain data.

**Key Capabilities:**
- Gradient computation using manifold learning (diffusion maps, PCA, Laplacian eigenmaps)
- Gradient alignment across individuals, modalities, and species
- Statistical testing for gradient associations
- Null model generation (spin tests, spatial permutations)
- Surface and volumetric gradient analysis
- Integration with connectivity matrices, morphometry, gene expression
- Visualization of gradients on cortical surfaces
- Cross-modal gradient comparison (e.g., structure vs. function)
- Python and MATLAB implementations
- Integration with standard neuroimaging formats

**Target Audience:**
- Connectivity researchers
- Systems neuroscience
- Multi-modal neuroimaging studies
- Comparative neuroanatomy
- Network neuroscience

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**

1. **Installation and Setup**
   - Python and MATLAB versions
   - Dependencies (scikit-learn, nibabel, vtk)
   - Surface data requirements

2. **Basic Gradient Analysis**
   - Load connectivity matrix
   - Compute gradients with diffusion maps
   - Visualize gradients on cortical surfaces
   - Interpret gradient meanings

3. **Advanced Gradient Techniques**
   - Multiple embedding approaches (diffusion, Laplacian, PCA)
   - Kernel selection and parameter tuning
   - Gradient alignment across subjects (Procrustes)
   - Cross-modal gradient alignment

4. **Statistical Analysis**
   - Spin tests for spatial permutations
   - Null models for gradient correlations
   - Gradient-phenotype associations
   - Significance testing

5. **Multi-Modal Integration**
   - Structural connectivity gradients
   - Functional connectivity gradients
   - Morphometric gradients (cortical thickness, curvature)
   - Gene expression gradients

6. **Visualization**
   - Surface plotting with BrainSpace viewers
   - 2D gradient scatter plots
   - Gradient density maps
   - Export for publication

7. **Real-World Applications**
   - Sensorimotor-transmodal gradient
   - Hierarchical brain organization
   - Individual differences in gradients
   - Clinical gradient alterations

8. **Integration**
   - Load data from HCP, fMRIPrep
   - Integration with connectome workbench
   - Export to FreeSurfer surfaces
   - BIDS-compatible workflows

**Example Workflows:**
- Compute principal connectivity gradient from resting-state fMRI
- Align gradients across subjects for group analysis
- Correlate genetic PC1 with functional gradient
- Compare gradients between healthy and patient groups
- Cross-species gradient comparison (human-macaque)

**Integration Points:**
- **Connectome Workbench:** Surface visualization
- **FreeSurfer:** Cortical surfaces and parcellations
- **HCP Pipelines:** Connectivity matrices
- **neuromaps:** Brain annotation overlays
- **abagen:** Gene expression data

---

### 2. neuromaps
**Website:** https://netneurolab.github.io/neuromaps/
**GitHub:** https://github.com/netneurolab/neuromaps
**Platform:** Python
**Priority:** High

**Overview:**
neuromaps is a Python toolbox for accessing, transforming, and analyzing a comprehensive collection of brain annotation maps. Developed by the Network Neuroscience Lab, it provides standardized access to diverse brain maps including receptor densities, gene expression, metabolic profiles, developmental trajectories, and more. neuromaps enables researchers to contextualize their findings by comparing custom brain maps to established reference annotations through spatial correlation and statistical testing.

**Key Capabilities:**
- Access to 50+ curated brain annotation maps (receptors, neurotransmitters, metabolism, genetics)
- Standardized spatial transformations between surfaces and volumes
- Surface-based and volume-based analysis
- Spatial null models for statistical testing (spin tests, spatial autocorrelation-preserving permutations)
- Map transformations between templates (fsaverage, fslr, MNI152)
- Comparison of user maps to reference annotations
- Significance testing with multiple comparison correction
- Visualization tools for brain maps
- Integration with parcellations (Desikan-Killiany, Schaefer, etc.)
- Data from multiple sources (PET atlases, Allen Brain Atlas, histology)

**Target Audience:**
- Multi-modal neuroimaging researchers
- Systems neuroscience
- Neuropsychopharmacology
- Developmental neuroscience
- Comparative neuroscience

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**

1. **Installation and Data Access**
   - Install neuromaps
   - Browse available annotations
   - Download brain maps
   - Understand map metadata

2. **Working with Brain Maps**
   - Load receptor density maps
   - Load gene expression maps
   - Load metabolic maps
   - Visualize on surfaces

3. **Spatial Transformations**
   - Transform fsaverage to fslr
   - Surface to volume transformations
   - MNI152 volumetric resampling
   - Parcellation-based aggregation

4. **Statistical Testing**
   - Spatial null models (spin tests)
   - Variogram matching for spatial autocorrelation
   - Significance testing for correlations
   - Multiple comparison correction

5. **Comparing Custom Maps**
   - Load user brain map (e.g., cortical thickness)
   - Correlate with receptor maps
   - Test significance with null models
   - Identify strongest associations

6. **Multi-Modal Correlation**
   - Structure-function associations
   - Gene expression-connectivity correlations
   - Receptor-metabolism relationships
   - Developmental trajectories

7. **Practical Applications**
   - Contextualize fMRI activation maps
   - Link structural changes to neurotransmitter systems
   - Identify genetic correlates of brain features
   - Cross-modal validation

8. **Integration**
   - BrainSpace gradients with neuromaps annotations
   - abagen gene expression integration
   - FreeSurfer morphometry comparisons
   - HCP data analysis

**Example Workflows:**
- Correlate cortical thinning pattern with serotonin 5HT1a receptor density
- Compare fMRI activation to gene expression profiles
- Link connectivity gradient to metabolic profile
- Contextualize patient brain map with normative annotations
- Multi-receptor correlation analysis

**Integration Points:**
- **BrainSpace:** Gradient-annotation correlations
- **abagen:** Gene expression maps
- **FreeSurfer:** Morphometric maps
- **Connectome Workbench:** Surface visualization
- **FSL/SPM:** Statistical maps

---

### 3. abagen
**Website:** https://abagen.readthedocs.io/
**GitHub:** https://github.com/rmarkello/abagen
**Platform:** Python
**Priority:** High

**Overview:**
abagen is a Python toolbox for working with the Allen Human Brain Atlas (AHBA) gene expression data in the context of neuroimaging. It provides standardized preprocessing pipelines to generate parcellated gene expression matrices from the AHBA microarray data, enabling researchers to link genetic signatures to brain structure, function, and connectivity. abagen handles the complex preprocessing steps (probe selection, sample assignment, normalization) and provides robust methods for integrating transcriptomic data with neuroimaging phenotypes.

**Key Capabilities:**
- Automated download of Allen Human Brain Atlas data
- Standardized preprocessing pipeline for AHBA microarray data
- Probe selection and filtering (intensity-based, differential stability)
- Sample-to-parcel assignment with multiple strategies
- Within-donor and across-donor normalization
- Missing data interpolation
- Generation of parcellated gene expression matrices
- Integration with custom brain parcellations
- Quality control and reproducibility checks
- Gene set enrichment analysis
- Spatial correlation with brain maps
- Support for all 6 AHBA donor brains

**Target Audience:**
- Imaging genetics researchers
- Systems neuroscience
- Transcriptomics-neuroimaging integration
- Network neuroscience
- Psychiatric genetics

**Estimated Lines:** 700-750
**Estimated Code Examples:** 28-32

**Key Topics to Cover:**

1. **Installation and Data Setup**
   - Install abagen
   - Download AHBA data
   - Understand AHBA structure (donors, samples, probes)
   - Data organization

2. **Basic Gene Expression Processing**
   - Fetch AHBA data for all donors
   - Default preprocessing pipeline
   - Generate parcellated expression matrix
   - Quality control checks

3. **Advanced Preprocessing**
   - Probe selection strategies (differential stability, intensity)
   - Sample filtering (tissue quality, distance thresholds)
   - Normalization methods (scaled robust sigmoid, mixed effects)
   - Missing data handling

4. **Parcellation Integration**
   - Use with Desikan-Killiany atlas
   - Custom parcellations (Schaefer, Gordon, etc.)
   - Subcortical region handling
   - Hemisphere-specific processing

5. **Gene-Brain Phenotype Correlation**
   - Correlate gene expression with cortical thickness
   - Link gene expression to connectivity
   - Identify genes associated with brain features
   - Statistical significance testing

6. **Gene Set Analysis**
   - Gene ontology enrichment
   - Pathway analysis
   - Disease-associated gene sets
   - Interpretation of results

7. **Multi-Modal Integration**
   - Gene expression gradients with BrainSpace
   - Receptor maps from neuromaps
   - Structural/functional imaging associations
   - Cross-modal validation

8. **Reproducibility**
   - Documenting preprocessing choices
   - Sensitivity analyses
   - Donor-specific effects
   - Replication strategies

**Example Workflows:**
- Generate parcellated gene expression for Schaefer atlas
- Identify genes correlated with cortical thickness
- Link gene expression to resting-state networks
- Gene enrichment for autism-associated changes
- Transcriptomic signatures of brain gradients

**Integration Points:**
- **BrainSpace:** Gene expression gradients
- **neuromaps:** Cross-modal comparisons
- **FreeSurfer:** Morphometric correlations
- **fMRIPrep/HCP:** Functional connectivity-gene associations
- **Enigma toolbox:** Disease-gene associations

---

### 4. BrainStat
**Website:** https://brainstat.readthedocs.io/
**GitHub:** https://github.com/MICA-MNI/BrainStat
**Platform:** Python/MATLAB
**Priority:** Medium-High

**Overview:**
BrainStat is a unified statistical analysis toolbox for neuroimaging data, developed at MNI. It provides a comprehensive framework for both surface-based and volume-based statistical analyses, implementing linear models, mixed effects models, multiple comparison correction, and advanced techniques like Random Field Theory. BrainStat integrates seamlessly with common neuroimaging outputs (FreeSurfer, fMRIPrep, HCP) and provides both vertex-wise and parcel-wise analysis capabilities.

**Key Capabilities:**
- Linear models for neuroimaging data (GLM)
- Mixed effects models for hierarchical data
- Surface-based statistical analysis (FreeSurfer surfaces)
- Volume-based analysis (NIfTI images)
- Multiple comparison correction (FDR, FWE, cluster-based)
- Random Field Theory for cluster correction
- Context decoding for functional maps
- Meta-analysis integration
- Visualization of statistical maps on surfaces
- Python and MATLAB implementations
- Integration with BrainSpace for gradient analysis
- Parallelization for large datasets

**Target Audience:**
- Neuroimaging statisticians
- Group comparison studies
- Longitudinal neuroimaging
- Multi-site studies
- Method developers

**Estimated Lines:** 650-700
**Estimated Code Examples:** 26-30

**Key Topics to Cover:**

1. **Installation and Setup**
   - Python and MATLAB versions
   - Dependencies
   - Data preparation

2. **Basic Linear Models**
   - Vertex-wise GLM on cortical surfaces
   - T-tests and F-tests
   - Covariate adjustment
   - Model diagnostics

3. **Surface-Based Analysis**
   - Load FreeSurfer surfaces and data
   - Vertex-wise statistical maps
   - Hemisphere handling
   - Surface smoothing

4. **Volume-Based Analysis**
   - Voxel-wise analysis of NIfTI data
   - Masking strategies
   - Interpretation of results

5. **Multiple Comparison Correction**
   - FDR correction
   - FWE correction
   - Cluster-based correction
   - Random Field Theory

6. **Mixed Effects Models**
   - Repeated measures analysis
   - Multi-level modeling
   - Random effects specification
   - Longitudinal data

7. **Advanced Features**
   - Context decoding (NeuroSynth integration)
   - Meta-analysis
   - Effect size estimation
   - Power analysis

8. **Integration**
   - BrainSpace gradient analysis
   - FreeSurfer morphometry
   - fMRIPrep outputs
   - HCP data

**Example Workflows:**
- Group comparison of cortical thickness (patients vs. controls)
- Longitudinal analysis of brain development
- Correlation with behavioral measures
- Multi-site data harmonization
- Meta-analytic conjunction analysis

**Integration Points:**
- **FreeSurfer:** Surface morphometry analysis
- **BrainSpace:** Statistical testing of gradients
- **fMRIPrep:** Functional activation statistics
- **HCP:** Multi-modal statistical analysis
- **NeuroSynth:** Context decoding

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **BrainSpace** (foundational gradient analysis)
   - **neuromaps** (brain annotation access)
   - **abagen** (gene expression integration)
   - **BrainStat** (statistical framework)

2. **Comprehensive Coverage:**
   - Each skill: 650-750 lines
   - 26-32 code examples per skill
   - Real-world integrative neuroscience workflows
   - Multi-modal analysis examples

3. **Consistent Structure:**
   - Overview and key features
   - Installation (Python/MATLAB where applicable)
   - Basic analysis workflows
   - Advanced techniques
   - Multi-modal integration
   - Statistical testing
   - Batch processing
   - Integration with other tools
   - Troubleshooting
   - Best practices
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Package installation
   - Data downloads
   - Dependency setup
   - Verification

2. **Basic Analysis** (6-8)
   - Load neuroimaging data
   - Compute basic measures (gradients, correlations, statistics)
   - Simple visualizations
   - Export results

3. **Advanced Techniques** (6-8)
   - Manifold learning (BrainSpace)
   - Spatial null models (neuromaps)
   - Preprocessing pipelines (abagen)
   - Mixed models (BrainStat)

4. **Multi-Modal Integration** (4-6)
   - Structure-function associations
   - Gene-imaging correlations
   - Gradient-annotation comparisons
   - Cross-modal validation

5. **Statistical Testing** (4-6)
   - Spatial permutation tests
   - Multiple comparison correction
   - Effect size estimation
   - Significance thresholds

6. **Visualization** (3-5)
   - Surface plotting
   - Scatter plots with annotations
   - Statistical maps
   - Publication-quality figures

7. **Batch Processing** (3-5)
   - Multi-subject analysis
   - Group-level statistics
   - Automated pipelines
   - Reproducible workflows

### Cross-Tool Integration

All skills will demonstrate integration with:
- **FreeSurfer:** Surfaces and morphometry
- **Connectome Workbench:** Visualization
- **fMRIPrep/HCP:** Functional and structural data
- **Python ecosystem:** NumPy, pandas, matplotlib, scikit-learn
- **Statistical packages:** statsmodels, scipy

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
| BrainSpace | 700-750 | 28-32 | High |
| neuromaps | 700-750 | 28-32 | High |
| abagen | 700-750 | 28-32 | High |
| BrainStat | 650-700 | 26-30 | Medium-High |
| **TOTAL** | **2,750-2,950** | **108-126** | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Gradient analysis and manifold learning (BrainSpace)
- ✓ Brain annotation maps and multi-modal atlases (neuromaps)
- ✓ Transcriptomics-neuroimaging integration (abagen)
- ✓ Statistical analysis framework (BrainStat)
- ✓ Spatial null models and permutation testing
- ✓ Cross-modal correlation and validation
- ✓ Surface and volume-based analysis

**Language/Platform Coverage:**
- Python: All 4 tools (4/4)
- MATLAB: BrainSpace, BrainStat (2/4)
- R integration: Via reticulate for all Python tools

**Application Areas:**
- Systems neuroscience: All tools
- Imaging genetics: abagen, neuromaps, BrainSpace
- Network neuroscience: BrainSpace, BrainStat
- Comparative neuroanatomy: BrainSpace, neuromaps
- Clinical research: All tools

---

## Strategic Importance

### Fills Critical Gap

Previous batches have focused on:
- Image acquisition, preprocessing, and reconstruction
- Single-modality analyses (structural, functional, diffusion, PET)

**Batch 22 adds:**
- **Integrative analysis** across modalities
- **Molecular-to-macroscale** connections (genes to brain)
- **Gradient-based representations** of brain organization
- **Spatial statistics** and null model frameworks
- **Multi-modal contextualization** of findings

### Complementary Skills

**Works with existing skills:**
- **FreeSurfer:** Provides surfaces and morphometry for gradients/statistics
- **fMRIPrep:** Functional connectivity for gradient analysis
- **HCP Pipelines:** Multi-modal data for integration
- **Connectome Workbench:** Visualization of gradients and maps
- **SPM/FSL:** Statistical maps for contextualization

### User Benefits

1. **Gradient-Based Analysis:**
   - Move beyond discrete modules to continuous organization
   - Reveal hierarchical brain structure
   - Individual differences in brain organization
   - Cross-species comparative analyses

2. **Multi-Modal Integration:**
   - Link structure, function, genetics, neurochemistry
   - Contextualize findings with reference annotations
   - Cross-validate discoveries across modalities
   - Mechanistic insights from molecular data

3. **Transcriptomic-Imaging:**
   - Identify genetic drivers of brain phenotypes
   - Link disease genes to imaging alterations
   - Understand molecular basis of connectivity
   - Personalized medicine applications

4. **Robust Statistics:**
   - Proper spatial null models
   - Multiple comparison correction
   - Mixed effects for hierarchical data
   - Reproducible analytical frameworks

---

## Dependencies and Prerequisites

### Software Prerequisites

**BrainSpace:**
- Python 3.6+ (for Python version)
- MATLAB R2018a+ (for MATLAB version)
- NumPy, SciPy, scikit-learn
- VTK for visualization
- nibabel for neuroimaging I/O

**neuromaps:**
- Python 3.7+
- nilearn, nibabel
- matplotlib, seaborn
- Internet connection for map downloads

**abagen:**
- Python 3.6+
- pandas, NumPy, SciPy
- nibabel, nilearn
- scikit-learn
- Internet connection for AHBA data download

**BrainStat:**
- Python 3.6+ or MATLAB R2018a+
- NumPy, SciPy, pandas
- statsmodels
- BrainSpace (for gradient integration)

### Data Prerequisites

**Common to all:**
- Cortical surfaces (FreeSurfer fsaverage or HCP fslr)
- Parcellations (Desikan-Killiany, Schaefer, etc.)
- Neuroimaging data (structural, functional, diffusion)

**Tool-specific:**
- **BrainSpace:** Connectivity matrices, feature maps
- **neuromaps:** None (annotations downloaded automatically)
- **abagen:** None (AHBA data downloaded automatically)
- **BrainStat:** Statistical design matrices, contrasts

### Knowledge Prerequisites

Users should understand:
- Basic neuroimaging concepts
- Linear algebra (for gradients and manifolds)
- Statistics (correlation, regression, multiple testing)
- Python programming
- Neuroanatomy (cortical organization)
- Optional: Graph theory, genetics basics

---

## Learning Outcomes

After completing Batch 22 skills, users will be able to:

1. **Gradient Analysis:**
   - Compute connectivity gradients from fMRI data
   - Align gradients across subjects and modalities
   - Interpret principal gradients (sensorimotor-transmodal)
   - Test gradient-phenotype associations

2. **Brain Annotation Integration:**
   - Access receptor, neurotransmitter, metabolic maps
   - Correlate custom brain maps with reference annotations
   - Apply spatial null models for significance testing
   - Contextualize findings with multi-modal atlases

3. **Transcriptomic-Imaging Integration:**
   - Generate parcellated gene expression from AHBA
   - Correlate gene expression with brain features
   - Identify genes associated with imaging phenotypes
   - Perform gene set enrichment analysis

4. **Statistical Analysis:**
   - Run vertex-wise GLMs on surfaces
   - Apply proper multiple comparison correction
   - Perform mixed effects analyses
   - Estimate effect sizes and power

5. **Multi-Modal Workflows:**
   - Integrate structure, function, genetics, neurochemistry
   - Cross-validate findings across modalities
   - Build mechanistic models from multi-modal data
   - Create publication-quality integrative figures

---

## Relationship to Existing Skills

### Builds Upon:
- **FreeSurfer** (Batch 1): Surface generation and morphometry
- **fMRIPrep** (Batch 5): Preprocessed functional data for connectivity
- **HCP Pipelines** (Batch 12): Multi-modal data
- **Connectome Workbench** (Batch 12): Surface visualization
- **CONN** (Batch 13): Connectivity matrices for gradients

### Complements:
- **Network analysis tools:** Gradient representations of networks
- **Statistical tools (SPM, FSL):** Additional statistical frameworks
- **Visualization tools:** Gradient and annotation displays

### Enables:
- Integrative multi-modal neuroscience
- Molecular-to-macroscale analyses
- Gradient-based understanding of brain organization
- Transcriptomic biomarker discovery
- Cross-species comparative studies

---

## Expected Challenges and Solutions

### Challenge 1: Abstract Concepts (Gradients, Manifolds)
**Issue:** Gradients and manifold learning are mathematically complex
**Solution:** Clear conceptual explanations, visual examples, interpretation guides

### Challenge 2: Data Access and Downloads
**Issue:** AHBA data is large; neuromaps requires internet
**Solution:** Document download procedures, provide example datasets, offline alternatives

### Challenge 3: Spatial Statistics
**Issue:** Spatial autocorrelation requires specialized null models
**Solution:** Explain rationale, provide ready-to-use functions, interpret results

### Challenge 4: Multi-Modal Integration Complexity
**Issue:** Combining multiple data types is complex
**Solution:** Step-by-step workflows, integration examples, troubleshooting

### Challenge 5: Interpretation
**Issue:** Gradient and gene-imaging results require careful interpretation
**Solution:** Interpretation guidelines, literature references, common pitfalls

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Package import tests
   - Data download checks
   - Dependency validation
   - Version checking

2. **Basic Functionality Tests:**
   - Load example data
   - Run basic analysis
   - Generate output
   - Visualize results

3. **Integration Tests:**
   - Load FreeSurfer data
   - Process fMRIPrep outputs
   - Cross-tool workflows
   - Multi-modal examples

4. **Example Data:**
   - Links to example datasets
   - Sample analysis scripts
   - Expected outputs
   - Interpretation examples

---

## Timeline Estimate

**Per Skill:**
- Research and planning: 15-20 min
- Writing and examples: 45-55 min
- Review and refinement: 10-15 min
- **Total per skill:** ~70-90 min

**Total Batch 22:**
- 4 skills × 80 min average = ~320 min (~5.3 hours)
- Includes documentation, examples, and testing

**Can be completed in:** 1-2 extended sessions

---

## Success Criteria

Batch 22 will be considered successful when:

✓ All 4 skills created with 650-750 lines each
✓ Total of 108+ code examples across batch
✓ Each skill includes:
  - Comprehensive installation instructions
  - Basic to advanced analysis workflows
  - Multi-modal integration examples
  - Statistical testing procedures
  - Spatial null model applications
  - Visualization examples
  - Batch processing capabilities
  - Integration with FreeSurfer, fMRIPrep, HCP
  - Troubleshooting section
  - Best practices
  - Citations and resources

✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 83/133 (62.4%)

---

## Next Batches Preview

### Batch 23: Network Analysis & Connectivity
- NetworkX (graph analysis in Python)
- Brain Connectivity Toolbox (BCT)
- BRAPH2 (brain graph analysis)
- NBS-Predict (network-based statistics)

### Batch 24: Specialized Statistical Tools
- PALM (permutation analysis)
- SnPM (statistical nonparametric mapping)
- SurfStat (surface-based statistics)
- LIMO EEG (linear modeling for EEG/MEG)

### Batch 25: Quality Control & Visualization
- MRIQC (automated quality control)
- QC-Automation tools
- Nilearn plotting utilities
- Additional visualization frameworks

---

## Conclusion

Batch 22 represents a pivotal shift toward **integrative neuroscience**, moving beyond single-modality analyses to embrace the multi-scale, multi-modal nature of brain organization. By providing tools for:

- **Gradient-based representations** (BrainSpace)
- **Multi-modal annotations** (neuromaps)
- **Transcriptomic integration** (abagen)
- **Unified statistics** (BrainStat)

This batch enables researchers to:
- Discover **organizational principles** of the brain via gradients
- **Contextualize findings** with reference annotations
- **Link genes to brain phenotypes** for mechanistic insights
- Apply **robust statistical frameworks** with spatial null models

These tools are critical for:
- Systems neuroscience and connectomics
- Imaging genetics and precision medicine
- Understanding brain development and aging
- Identifying biomarkers for neuropsychiatric disorders
- Cross-species comparative neuroscience

By bridging molecular, cellular, and systems-level data, Batch 22 skills position users at the forefront of integrative neuroscience, enabling discoveries that span from genes to networks to behavior.

**Status After Batch 22:** 83/133 skills (62.4% complete - past 60% milestone!)

---

**Document Version:** 1.0
**Created:** 2025-11-15
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 4 skills, ~2,750-2,950 lines, ~108-126 examples
