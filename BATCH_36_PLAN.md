# Batch 36 Plan: Surface Mapping, Gradients & Brain Annotation

## Overview

**Theme:** Surface Mapping, Gradients & Brain Annotation
**Focus:** Cortical gradient analysis, surface-based statistics, brain annotation comparison, and genetics integration
**Target:** 4 new skills, 2,400-2,600 total lines

**Current Progress:** 120/133 skills (90.2%)
**After Batch 35:** 120/133 skills (90.2%)
**After Batch 36:** 124/133 skills (93.2%)

This batch addresses advanced surface-based analysis methods that go beyond traditional morphometry. These tools enable researchers to characterize the brain's functional and structural organization using gradient-based approaches, perform sophisticated surface statistics, compare diverse brain annotations across datasets, and integrate neuroimaging with genetic expression data from the Allen Brain Atlas.

## Rationale

While basic surface analysis tools (FreeSurfer, Connectome Workbench) are well-established, emerging research paradigms require specialized approaches:

- **Gradient Analysis:** Understanding brain organization through continuous gradients rather than discrete parcellations
- **Surface Statistics:** Robust statistical methods for cortical surface data with proper multiple comparison correction
- **Cross-Dataset Integration:** Comparing and integrating diverse brain maps (functional, structural, genetic, molecular)
- **Imaging-Genetics:** Linking neuroimaging phenotypes to underlying genetic architecture via gene expression

This batch provides comprehensive coverage of cutting-edge surface-based analysis and multi-modal brain annotation integration.

## Skills to Create

### 1. BrainSpace (650-700 lines, 22-26 examples)

**Overview:**
BrainSpace is a Python toolbox for identifying and analyzing gradients of brain organization. Rather than treating the brain as discrete modules, gradient analysis reveals continuous transitions in functional connectivity, microstructural properties, and other features across the cortical surface. BrainSpace implements manifold learning techniques to identify these gradients and provides tools for visualization, statistical analysis, and relating gradients to behavior and genetics.

**Key Features:**
- Gradient decomposition using diffusion map embedding and other manifold learning techniques
- Alignment of gradients across individuals and datasets
- Gradient visualization on cortical surfaces
- Null model generation for statistical testing
- Integration with parcellations and annotations
- Support for multiple surface formats (FreeSurfer, GIFTI, CIFTI)
- Gradient comparison across species and development
- MATLAB and Python implementations

**Target Audience:**
- Researchers studying functional brain organization
- Cognitive neuroscientists exploring brain-behavior relationships
- Developmental scientists tracking gradient maturation
- Computational neuroscientists modeling brain networks

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to gradient analysis and manifold learning
   - Advantages over discrete parcellations
   - Theoretical foundations (diffusion maps, Laplacian eigenmaps)
   - BrainSpace capabilities
   - Citation information

2. **Installation** (70 lines)
   - Python package installation
   - MATLAB toolbox installation
   - Dependencies (Nilearn, VTK, PyVista)
   - Surface data requirements
   - Example dataset download
   - Testing installation

3. **Computing Functional Gradients** (120 lines, 4-5 examples)
   - Building connectivity matrices from fMRI
   - Gradient decomposition using diffusion maps
   - Selecting number of gradients
   - Interpreting gradient components
   - Example: Principal gradient from resting-state fMRI
   - Example: Task-based connectivity gradients

4. **Gradient Alignment** (100 lines, 3-4 examples)
   - Procrustes alignment across subjects
   - Joint alignment for group analysis
   - Aligning gradients across datasets
   - Handling missing data
   - Example: Group-level gradient averaging

5. **Visualization** (100 lines, 3-4 examples)
   - Surface plotting with gradient values
   - Interactive 3D visualization
   - Scatterplots in gradient space
   - Customizing color maps
   - Example: Plotting principal gradient on fsaverage

6. **Statistical Analysis** (90 lines, 3-4 examples)
   - Null models for gradient-based statistics
   - Spin tests for spatial autocorrelation
   - Correlating gradients with annotations
   - Vertex-wise gradient-behavior associations
   - Example: Gradient-cognition correlations

7. **Parcellation and ROI Analysis** (80 lines, 2-3 examples)
   - Computing gradients within parcellations
   - Gradient-based parcellation refinement
   - Comparing gradients across parcellations
   - Example: Schaefer parcellation gradient analysis

8. **Advanced Applications** (70 lines, 2-3 examples)
   - Microstructural gradients (myelin, thickness)
   - Multi-modal gradient fusion
   - Developmental gradient trajectories
   - Cross-species gradient comparison
   - Example: Myelin-function gradient coupling

9. **Integration with Neuroimaging Tools** (60 lines, 1-2 examples)
   - Loading FreeSurfer surfaces
   - CIFTI format support (HCP data)
   - Integration with Nilearn
   - Exporting results to Connectome Workbench

10. **Troubleshooting** (50 lines)
    - Convergence issues in diffusion maps
    - Handling noisy connectivity matrices
    - Surface mesh compatibility
    - Memory requirements for large datasets

11. **Best Practices** (40 lines)
    - Data quality requirements
    - Choosing gradient parameters
    - Statistical testing considerations
    - Interpretation guidelines

12. **References** (20 lines)
    - BrainSpace papers
    - Gradient analysis methodology
    - Applications in neuroscience

**Code Examples:**
- Computing connectivity gradients (Python)
- Gradient alignment (Python)
- Surface visualization (Python)
- Spin test null models (Python)
- Parcellation-based gradients (Python)

**Integration Points:**
- FreeSurfer for cortical surfaces
- Nilearn for connectivity analysis
- Connectome Workbench for visualization
- SciPy/scikit-learn for manifold learning
- VTK/PyVista for 3D rendering

---

### 2. BrainStat (650-700 lines, 22-26 examples)

**Overview:**
BrainStat is a comprehensive toolbox for statistical analysis of neuroimaging data, with particular emphasis on cortical surface-based analysis. It provides robust methods for linear modeling, multiple comparison correction, effect size computation, and visualization. BrainStat implements best practices for surface-based statistics including cluster-based thresholding, random field theory corrections, and mixed-effects models for longitudinal data.

**Key Features:**
- Linear mixed-effects models for surface and volume data
- Multiple comparison correction (FDR, RFT, cluster-based)
- Flexible contrast specification
- Effect size maps and confidence intervals
- Cross-validation and prediction
- Integration with common surface formats
- Python and MATLAB interfaces
- Comprehensive statistical diagnostics

**Target Audience:**
- Researchers performing group-level surface analysis
- Clinical neuroscientists comparing patient populations
- Longitudinal study investigators
- Methods developers validating statistical approaches

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to surface-based statistics
   - Challenges of cortical data (spatial autocorrelation, non-stationarity)
   - BrainStat statistical framework
   - Comparison to SPM, FSL, SurfStat
   - Citation information

2. **Installation** (70 lines)
   - Python package installation
   - MATLAB toolbox installation
   - Dependencies
   - Surface mesh requirements
   - Example dataset download

3. **Linear Models for Surface Data** (120 lines, 4-5 examples)
   - Specifying design matrices
   - Contrasts and t-tests
   - F-tests for multiple contrasts
   - Handling covariates (age, sex, ICV)
   - Example: Group comparison on cortical thickness
   - Example: Age-cognition interaction

4. **Mixed-Effects Models** (100 lines, 3-4 examples)
   - Longitudinal data analysis
   - Within-subject repeated measures
   - Random intercepts and slopes
   - Model comparison
   - Example: Longitudinal cortical thinning

5. **Multiple Comparison Correction** (110 lines, 3-4 examples)
   - False discovery rate (FDR)
   - Family-wise error rate (FWER)
   - Random field theory (RFT)
   - Cluster-based thresholding
   - Example: Cortex-wide thickness differences with RFT

6. **Effect Sizes and Confidence Intervals** (80 lines, 2-3 examples)
   - Cohen's d maps
   - Partial eta-squared
   - Confidence interval computation
   - Interpreting effect sizes on surfaces
   - Example: Effect size maps for patient-control comparison

7. **Predictive Modeling** (90 lines, 3-4 examples)
   - Cross-validated prediction
   - Feature selection
   - Model performance metrics
   - Out-of-sample generalization
   - Example: Predicting cognitive scores from cortical morphometry

8. **Advanced Statistical Tests** (70 lines, 2-3 examples)
   - Non-parametric permutation tests
   - Multi-modal fusion (thickness + myelin)
   - Mediation analysis
   - Example: Permutation testing for small samples

9. **Visualization** (60 lines, 1-2 examples)
   - Statistical maps on surfaces
   - Thresholded overlays
   - Effect size visualization
   - Exporting figures

10. **Integration with Processing Pipelines** (50 lines, 1-2 examples)
    - FreeSurfer outputs
    - CIVET data
    - HCP-style CIFTI
    - Custom surface data

11. **Troubleshooting** (40 lines)
    - Convergence issues
    - Multicollinearity
    - Unbalanced designs
    - Memory management

12. **Best Practices** (40 lines)
    - Sample size considerations
    - Choosing correction methods
    - Reporting statistical results
    - Replication and validation

13. **References** (20 lines)
    - BrainStat papers
    - Statistical methodology
    - Surface-based analysis reviews

**Code Examples:**
- GLM on surface data (Python)
- Mixed-effects longitudinal model (Python)
- FDR and RFT correction (Python)
- Effect size computation (Python)
- Cross-validated prediction (Python)

**Integration Points:**
- FreeSurfer for cortical morphometry
- CIVET for surface extraction
- Nibabel for surface I/O
- Statsmodels for linear modeling
- Scikit-learn for prediction

---

### 3. neuromaps (600-650 lines, 20-24 examples)

**Overview:**
neuromaps is a Python toolbox for comparing and integrating diverse brain maps (annotations) across different coordinate systems, surface spaces, and data modalities. It provides standardized transformations, spatial null models for statistical testing, and a curated collection of brain annotations including receptor maps, gene expression, metabolism, and functional networks. This enables researchers to relate their findings to the broader landscape of brain organization.

**Key Features:**
- Standardized brain annotation database (40+ maps)
- Spatial transformations between surfaces and volumes
- Surface-to-surface resampling (fsaverage, fsLR, CIVET)
- Volume-to-surface projection methods
- Spatial null models accounting for spatial autocorrelation
- Annotation comparison and correlation
- Visualization and quality control
- Reproducible workflows

**Target Audience:**
- Researchers contextualizing findings with existing annotations
- Multi-modal integration studies
- Neurobiological interpretation of neuroimaging results
- Meta-analysts comparing findings across studies

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to brain annotation integration
   - Challenges of cross-dataset comparison
   - neuromaps annotation library
   - Use cases and applications
   - Citation information

2. **Installation** (60 lines)
   - Python package installation
   - Dependencies
   - Downloading annotation library
   - Data storage and caching
   - Testing installation

3. **Accessing Brain Annotations** (100 lines, 3-4 examples)
   - Browsing available annotations
   - Loading receptor density maps
   - Accessing gene expression data
   - Metabolic maps (PET tracers)
   - Example: Fetching serotonin receptor maps
   - Example: Loading functional gradients

4. **Surface Transformations** (110 lines, 4-5 examples)
   - Resampling between fsaverage resolutions
   - fsaverage to fsLR transformation (HCP)
   - CIVET to FreeSurfer conversion
   - Handling hemispheric data
   - Example: Transform Schaefer parcellation to fsLR

5. **Volume-to-Surface Projection** (90 lines, 3-4 examples)
   - Projection methods (nearest, linear, trilinear)
   - Ribbon-constrained mapping
   - Subcortical data handling
   - Example: Project volumetric PET to surface

6. **Spatial Null Models** (100 lines, 3-4 examples)
   - Spin-based permutation tests
   - Variogram matching
   - Accounting for spatial autocorrelation
   - Computing p-values
   - Example: Test correlation between annotations with spin test

7. **Comparing Brain Annotations** (90 lines, 3-4 examples)
   - Correlation analysis
   - Partial correlations
   - Dominance analysis
   - Example: Relate myelin map to receptor distributions
   - Example: Compare task activation to gene expression

8. **Visualization** (70 lines, 2-3 examples)
   - Plotting annotations on surfaces
   - Correlation matrices
   - Scatterplots with spatial nulls
   - Example: Visualize annotation comparison results

9. **Custom Annotation Integration** (60 lines, 1-2 examples)
   - Adding user-defined maps
   - Format requirements
   - Metadata specification
   - Sharing annotations

10. **Troubleshooting** (40 lines)
    - Surface mesh mismatches
    - Missing data handling
    - Transformation errors
    - Memory issues with large datasets

11. **Best Practices** (40 lines)
    - Choosing appropriate transformations
    - Statistical testing guidelines
    - Interpreting correlations
    - Reporting methods

12. **References** (20 lines)
    - neuromaps papers
    - Brain annotation datasets
    - Null model methodology

**Code Examples:**
- Fetch and plot annotations (Python)
- Surface resampling (Python)
- Volume-to-surface projection (Python)
- Spin test for correlation (Python)
- Multi-annotation comparison (Python)

**Integration Points:**
- BrainSpace for gradient analysis
- FreeSurfer for surface formats
- Nilearn for volumetric data
- Connectome Workbench for CIFTI
- Matplotlib/Plotly for visualization

---

### 4. abagen (600-650 lines, 20-24 examples)

**Overview:**
abagen (Allen Brain Atlas genetics) is a Python toolbox for integrating neuroimaging data with gene expression data from the Allen Human Brain Atlas (AHBA). It provides standardized workflows for assigning microarray samples to brain regions, processing gene expression data, and correlating imaging-derived phenotypes with transcriptional profiles. This enables imaging-transcriptomics analyses to identify genetic mechanisms underlying neuroimaging findings.

**Key Features:**
- Allen Human Brain Atlas data fetching and processing
- Microarray sample assignment to brain parcellations
- Gene expression normalization and quality control
- Probe selection and annotation
- Imaging-transcriptomics correlation
- Spatial null models for gene-brain associations
- Support for multiple parcellations
- Reproducible gene expression matrices

**Target Audience:**
- Imaging-genetics researchers
- Systems neuroscientists studying brain organization
- Psychiatric and neurological disease researchers
- Neuropharmacology and drug target discovery

**Main Sections:**

1. **Overview** (50 lines)
   - Introduction to Allen Human Brain Atlas
   - Imaging-transcriptomics rationale
   - abagen workflow and capabilities
   - Limitations and considerations
   - Citation information

2. **Installation** (60 lines)
   - Python package installation
   - Dependencies
   - AHBA data download (automatic)
   - Parcellation files
   - Testing installation

3. **Allen Human Brain Atlas Basics** (80 lines, 2-3 examples)
   - AHBA structure and samples
   - 6 donor brains, microarray coverage
   - Gene expression data format
   - Anatomical annotations
   - Example: Exploring AHBA samples

4. **Assigning Samples to Parcellations** (110 lines, 4-5 examples)
   - Sample-to-region assignment strategies
   - Handling incomplete coverage
   - Multi-resolution parcellations
   - Quality control for assignments
   - Example: Assign samples to Desikan-Killiany atlas
   - Example: Custom parcellation assignment

5. **Gene Expression Processing** (100 lines, 3-4 examples)
   - Probe selection (highest correlation, least variance)
   - Expression normalization (SRS, scaled robust sigmoid)
   - Inter-subject normalization
   - Filtering genes and samples
   - Example: Create normalized expression matrix

6. **Imaging-Transcriptomics Analysis** (110 lines, 4-5 examples)
   - Correlating brain maps with gene expression
   - Identifying genes associated with neuroimaging phenotypes
   - Partial Least Squares (PLS) analysis
   - Gene set enrichment
   - Example: Genes correlated with cortical thickness
   - Example: Functional connectivity-gene associations

7. **Spatial Null Models** (90 lines, 3-4 examples)
   - Spatial autocorrelation in gene expression
   - Spin-based permutations for gene-imaging correlations
   - Correcting for distance-dependent effects
   - Example: Null model for imaging-transcriptomics

8. **Gene Set Enrichment Analysis** (80 lines, 2-3 examples)
   - Over-representation analysis
   - Gene Ontology (GO) enrichment
   - KEGG pathway analysis
   - Cell-type specificity
   - Example: Enriched biological processes

9. **Advanced Applications** (70 lines, 2-3 examples)
   - Receptor-specific gene expression
   - Disease-associated gene sets
   - Developmental gene expression
   - Example: Dopaminergic gene expression patterns

10. **Visualization** (60 lines, 1-2 examples)
    - Gene expression on brain surfaces
    - Manhattan plots for gene associations
    - Heatmaps of expression matrices
    - Example: Visualize top genes on cortex

11. **Troubleshooting** (50 lines)
    - Missing data handling
    - Parcellation coverage issues
    - Memory requirements
    - Reproducibility considerations

12. **Best Practices** (40 lines)
    - Sample size and power
    - Multiple comparison correction
    - Replication in independent datasets
    - Biological interpretation

13. **References** (20 lines)
    - abagen papers
    - Allen Human Brain Atlas
    - Imaging-transcriptomics reviews
    - Gene enrichment resources

**Code Examples:**
- Fetch AHBA data (Python)
- Assign samples to parcellation (Python)
- Process gene expression (Python)
- Correlate imaging with genes (Python)
- Gene set enrichment (Python)

**Integration Points:**
- Allen Brain Atlas API
- FreeSurfer for parcellations
- Nilearn for imaging data
- Pandas for data manipulation
- Enrichr/GOATOOLS for enrichment
- neuromaps for spatial nulls

---

## Implementation Checklist

### Per-Skill Requirements
- [ ] 600-700 lines per skill
- [ ] 20-26 code examples per skill
- [ ] Consistent section structure
- [ ] Installation instructions for Python/MATLAB
- [ ] Basic usage examples
- [ ] Advanced features and applications
- [ ] Statistical testing guidance
- [ ] Integration with neuroimaging ecosystem
- [ ] Troubleshooting section
- [ ] Best practices
- [ ] References with proper citations

### Quality Assurance
- [ ] All code examples are tested and functional
- [ ] Command/function syntax is accurate
- [ ] Examples demonstrate real-world workflows
- [ ] Integration examples are practical
- [ ] Statistical methods properly explained
- [ ] Troubleshooting covers common issues
- [ ] References are complete and up-to-date

### Batch Requirements
- [ ] Total lines: 2,400-2,600
- [ ] Total examples: 84-102
- [ ] Consistent markdown formatting
- [ ] Proper cross-referencing between skills
- [ ] All four skills address surface/annotation domains

## Timeline

1. **BrainSpace**: 650-700 lines, 22-26 examples
2. **BrainStat**: 650-700 lines, 22-26 examples
3. **neuromaps**: 600-650 lines, 20-24 examples
4. **abagen**: 600-650 lines, 20-24 examples

**Estimated Total:** 2,500-2,700 lines, 84-100 examples

## Context & Connections

### Analysis Framework Integration

**Gradient Analysis (BrainSpace):**
```
Connectivity Matrix → Manifold Learning → Gradients → Statistical Analysis
        ↓                    ↓                 ↓              ↓
    fMRI/DWI          Diffusion Maps    Visualization   Null Models
```

**Surface Statistics (BrainStat):**
```
Cortical Metrics → Linear Models → Correction → Effect Sizes
       ↓               ↓              ↓           ↓
  Thickness/Area    Mixed Effects    FDR/RFT    Visualization
```

**Annotation Integration (neuromaps):**
```
Brain Map → Transform → Compare → Statistical Test
    ↓          ↓          ↓           ↓
  Any Space  Standard   Correlate   Spin Test
```

**Imaging-Genetics (abagen):**
```
Neuroimaging → Parcellation → Gene Expression → Enrichment
     ↓             ↓              ↓                ↓
  Brain Map    ROI Values      AHBA Data      GO/KEGG
```

### Complementary Tools

**Already Covered:**
- **FreeSurfer**: Cortical surface generation (used by all four tools)
- **Nilearn**: Connectivity analysis (input to BrainSpace)
- **Connectome Workbench**: Surface visualization (output from all tools)
- **SPM/FSL**: Statistical analysis (complemented by BrainStat)

**New Capabilities:**
- **BrainSpace**: First gradient-based brain organization analysis
- **BrainStat**: First comprehensive surface statistics toolbox
- **neuromaps**: First standardized annotation comparison framework
- **abagen**: First imaging-transcriptomics integration tool

### Research Applications

**BrainSpace Use Cases:**
- Mapping functional hierarchies in cortex
- Developmental gradient maturation studies
- Disease-related gradient alterations
- Cross-species gradient comparisons

**BrainStat Use Cases:**
- Group comparisons in cortical morphometry
- Longitudinal disease progression analysis
- Cognitive neuroscience surface-based GLMs
- Multi-site harmonized statistics

**neuromaps Use Cases:**
- Relating findings to receptor distributions
- Comparing task activation to gene expression
- Validating parcellations against cytoarchitecture
- Meta-analytic annotation integration

**abagen Use Cases:**
- Identifying genetic basis of brain networks
- Disease-associated gene expression patterns
- Drug target discovery from imaging
- Neurotransmitter system mapping

## Technical Specifications

### BrainSpace
- **Platform**: Python 3.6+, MATLAB R2017a+
- **Dependencies**: NumPy, SciPy, scikit-learn, VTK, Nilearn
- **Input**: Connectivity matrices, surface meshes, annotations
- **Output**: Gradient maps, aligned gradients, statistics
- **Methods**: Diffusion maps, Procrustes alignment, spin tests

### BrainStat
- **Platform**: Python 3.7+, MATLAB R2018b+
- **Dependencies**: NumPy, SciPy, Statsmodels, Nibabel, BrainSpace
- **Input**: Surface data (thickness, area, etc.), design matrices
- **Output**: Statistical maps, p-values, effect sizes
- **Methods**: Linear models, mixed-effects, RFT, cluster correction

### neuromaps
- **Platform**: Python 3.7+
- **Dependencies**: Nibabel, Nilearn, SciPy, NetNeuroTools
- **Input**: Surface/volume maps, parcellations
- **Output**: Transformed maps, correlation matrices, p-values
- **Annotations**: 40+ brain maps (receptors, genes, metabolism, networks)
- **Formats**: FreeSurfer, GIFTI, CIFTI, NIfTI

### abagen
- **Platform**: Python 3.6+
- **Dependencies**: Pandas, Nibabel, Nilearn, SciPy
- **Input**: Parcellations, neuroimaging maps
- **Output**: Gene expression matrices, gene-imaging correlations
- **Data**: Allen Human Brain Atlas (6 donors, ~20,000 genes)
- **Methods**: Sample assignment, normalization, PLS, enrichment

## Learning Path

### Beginner Path (Surface Analysis)
1. Start with **BrainStat** for basic surface statistics
2. Learn **neuromaps** for contextualizing findings
3. Explore **BrainSpace** for gradient analysis

### Advanced Researcher Path
1. Master **BrainSpace** for gradient decomposition
2. Use **BrainStat** for robust statistical inference
3. Integrate findings with **neuromaps** and **abagen**

### Imaging-Genetics Path
1. Learn surface analysis with **FreeSurfer + BrainStat**
2. Use **abagen** for gene expression integration
3. Apply **neuromaps** for multi-modal validation

### Multi-Modal Integration Path
1. Use **BrainSpace** for functional gradients
2. Apply **neuromaps** to compare with structural/molecular maps
3. Use **abagen** to identify genetic underpinnings
4. Statistical validation with **BrainStat**

## Success Metrics

- [ ] All four skills cover complete analysis workflows
- [ ] Installation instructions work for Python and MATLAB
- [ ] Code examples run without errors
- [ ] Integration examples connect existing tools
- [ ] Statistical methods properly explained
- [ ] Troubleshooting addresses real issues
- [ ] Documentation enables self-directed learning
- [ ] Research applications clearly demonstrated

## Detailed Application Scenarios

### BrainSpace: Functional Gradient Analysis

**Study Design:**
Map the principal gradient of functional connectivity and relate it to cognition across 500 healthy adults.

**Workflow:**
```
Data: HCP resting-state fMRI (400 parcels, 1200 timepoints)

1. Compute connectivity matrix:
   - Extract timeseries from Schaefer 400 parcellation
   - Calculate Pearson correlations
   - Group average connectivity

2. BrainSpace gradient decomposition:
   - Diffusion map embedding
   - Extract top 3 gradients
   - Procrustes alignment across subjects

3. Statistical analysis:
   - Correlate gradient 1 with cognitive scores
   - Spin test for spatial null model
   - Vertex-wise gradient-cognition associations

4. Interpretation:
   - Gradient 1: sensorimotor-to-transmodal axis
   - Higher gradient scores → better executive function
   - Relates to receptor distribution (via neuromaps)

Expected Results:
- Principal gradient: unimodal sensorimotor → heteromodal DMN
- Correlation with fluid intelligence: r=0.45 (p<0.001, spin-corrected)
- Gradient stability across individuals
```

### BrainStat: Longitudinal Cortical Thinning

**Study Design:**
Analyze cortical thinning trajectories in 200 older adults (annual scans × 5 years) with and without cognitive decline.

**Workflow:**
```
Data: FreeSurfer outputs, 200 subjects, 5 timepoints

1. Load cortical thickness data:
   - Extract thickness at each vertex
   - Quality control for segmentation

2. Specify mixed-effects model:
   - Fixed effects: time, group, time×group
   - Random effects: subject (intercept, slope)
   - Covariates: age, sex, education, ICV

3. Fit model with BrainStat:
   - Vertex-wise analysis (327,684 vertices)
   - F-test for time×group interaction
   - Random field theory correction (p<0.05)

4. Effect size computation:
   - Cohen's d maps for group differences
   - Annual atrophy rate per group

5. Post-hoc analyses:
   - Cluster localization
   - Relation to cognitive decline rate

Expected Results:
- Significant time×group interaction in temporal and parietal regions
- Declining group: 2.5%/year thickness loss
- Stable group: 0.8%/year thickness loss
- Clusters predict cognitive decline (AUC=0.78)
```

### neuromaps: Multi-Modal Integration

**Study Design:**
Relate a novel task-based activation map to existing receptor, metabolic, and genetic brain annotations.

**Workflow:**
```
Data: Custom task fMRI z-map (working memory), 100 subjects

1. Prepare custom annotation:
   - Group-level working memory activation (fsaverage)
   - Threshold and QC

2. Fetch neuromaps annotations:
   - Serotonin receptor (5-HT1a) density
   - Dopamine receptor (D1, D2) density
   - FDG PET metabolism
   - Functional gradient (principal)
   - Gene expression (cognitive genes)

3. Transform all to common space:
   - Resample to fsaverage5
   - Vertex-wise alignment

4. Compute correlations:
   - Pearson r for each annotation pair
   - Spin test null models (10,000 permutations)

5. Interpret results:
   - Strong correlation with D1 receptors (r=0.62, p<0.001)
   - Moderate correlation with FDG metabolism (r=0.48, p<0.01)
   - Alignment with functional gradient axis

Expected Insights:
- Working memory activation follows dopaminergic architecture
- Overlaps with high-metabolism regions
- Aligns with heteromodal gradient regions
```

### abagen: Imaging-Transcriptomics

**Study Design:**
Identify genes whose expression patterns correlate with cortical thickness in Alzheimer's disease.

**Workflow:**
```
Data: 150 AD patients, 150 controls
Cortical thickness maps (FreeSurfer, Desikan-Killiany 68 parcels)

1. Compute group difference map:
   - AD vs. Control t-statistic per region
   - Effect size (Cohen's d)

2. Process AHBA data with abagen:
   - Assign samples to DK68 parcellation
   - Normalize gene expression (SRS normalization)
   - Filter low-variance genes
   - Result: 68 regions × 15,633 genes

3. Imaging-transcriptomics correlation:
   - Correlate AD-related thickness changes with gene expression
   - Spatial null model (10,000 spins)
   - FDR correction across genes

4. Gene set enrichment:
   - Top 500 genes → GO enrichment
   - KEGG pathway analysis
   - Cell-type deconvolution

5. Validation:
   - Replicate in independent dataset
   - Compare to AD genetic risk genes

Expected Results:
- 1,247 genes significantly correlated with AD thickness (FDR<0.05)
- Enriched: synaptic transmission, mitochondrial function, immune response
- Overlap with APOE, APP pathways
- Neuronal and microglial cell types
```

## Alternative Tools and Comparisons

### BrainSpace Alternatives

**Custom Manifold Learning:**
- Pros: Full control over parameters
- Cons: Requires coding expertise, no alignment tools
- **BrainSpace advantage**: Standardized workflow, gradient alignment

**Functional Parcellations:**
- Pros: Discrete labels, easier interpretation
- Cons: Misses continuous gradients
- **BrainSpace advantage**: Captures continuous organization

### BrainStat Alternatives

**SurfStat (MATLAB):**
- Pros: Established, well-documented
- Cons: MATLAB-only, less actively developed
- **BrainStat advantage**: Python support, modern features

**FreeSurfer's mri_glmfit:**
- Pros: Integrated with FreeSurfer
- Cons: Limited model types, basic corrections
- **BrainStat advantage**: Mixed-effects, advanced corrections

**SPM on Surfaces:**
- Pros: Familiar SPM interface
- Cons: Primarily volume-focused
- **BrainStat advantage**: Surface-optimized, better RFT

### neuromaps Alternatives

**Manual Data Integration:**
- Pros: Full control
- Cons: Labor-intensive, non-standardized
- **neuromaps advantage**: Standardized transformations, curated library

**BrainMap/Neurosynth:**
- Pros: Meta-analytic coordinates
- Cons: Limited annotation types
- **neuromaps advantage**: Diverse annotations (genes, receptors, metabolism)

### abagen Alternatives

**Manual AHBA Processing:**
- Pros: Custom workflows
- Cons: Complex, error-prone
- **abagen advantage**: Standardized best practices

**Allen Brain Map API:**
- Pros: Direct data access
- Cons: Requires custom processing pipeline
- **abagen advantage**: Preprocessing, parcellation assignment

## Emerging Trends

### Gradient-Based Analysis
- Multi-modal gradient fusion (structure + function)
- Gradient dynamics (time-varying gradients in task fMRI)
- Gradient perturbations in disease
- Cross-species gradient conservation

### Surface Statistics
- Machine learning on surfaces (geometric deep learning)
- Bayesian hierarchical models
- Multi-resolution analysis
- Real-time surface-based neurofeedback

### Brain Annotation
- Single-cell resolution atlases
- Multimodal parcellations (functional + molecular)
- Temporal dynamics annotations
- Individual-specific annotations

### Imaging-Genetics
- Polygenic risk scores correlated with imaging
- eQTL mapping in brain
- Spatial transcriptomics integration
- Drug target discovery from imaging-genetics

## Conclusion

Batch 36 addresses critical gaps in surface-based analysis and multi-modal integration by documenting four essential tools:

1. **BrainSpace** enables gradient-based characterization of brain organization, revealing continuous functional and structural hierarchies
2. **BrainStat** provides robust statistical methods for surface data with proper corrections and effect sizes
3. **neuromaps** standardizes brain annotation comparison and integration across diverse data modalities
4. **abagen** bridges neuroimaging and genetics through Allen Brain Atlas gene expression integration

By completing this batch, the N_tools neuroimaging skill collection will reach **124/133 skills (93.2%)**, with comprehensive coverage extending from basic surface analysis to advanced gradient mapping, rigorous statistics, and multi-modal annotation integration including genetics.

These tools represent the cutting edge of integrative neuroscience, enabling researchers to contextualize their findings within the broader landscape of brain organization, relate imaging phenotypes to molecular and genetic substrates, and perform sophisticated surface-based analyses with proper statistical rigor.
