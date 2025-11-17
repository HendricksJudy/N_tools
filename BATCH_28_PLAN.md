# Batch 28: Reproducibility & Workflow Infrastructure - Planning Document

## Overview

**Batch Theme:** Reproducibility & Workflow Infrastructure
**Batch Number:** 28
**Number of Skills:** 3 new skills (2 existing skills verified)
**Current Progress:** 98/133 skills completed (73.7%)
**After Batch 28:** 101/133 skills (75.9%)

## Rationale

Batch 28 focuses on **reproducibility infrastructure and workflow standardization** for neuroimaging research. While powerful workflow engines (Pydra, Snakebids) already exist in our collection, this batch completes the reproducibility ecosystem by adding container generation, tool standardization, and template management capabilities. These tools enable:

- **Containerized environments** for perfect reproducibility
- **Standardized tool descriptors** for cross-platform execution
- **Brain template management** for consistent spatial normalization
- **Automated container generation** from recipes
- **Tool validation** and quality assurance
- **Reproducible compute environments** across institutions

**Key Scientific Advances:**
- Eliminate "works on my machine" problems
- Share exact computational environments
- Validate tool installations automatically
- Standardize brain templates and atlases
- Enable multi-site reproducibility
- Facilitate method sharing and replication

**Applications:**
- Multi-site neuroimaging consortia
- Computational reproducibility for publications
- Cloud and HPC deployment
- Tool development and distribution
- BIDS Apps creation and deployment
- Standardized preprocessing across studies
- Template-based spatial normalization

---

## Existing Skills (Verified)

### Pydra (Already Exists - 981 lines)
**Status:** Complete
**Description:** Next-generation dataflow engine (Nipype successor)
**Key Features:** Task graphs, lazy evaluation, caching, provenance tracking

### Snakebids (Already Exists - 939 lines)
**Status:** Complete
**Description:** BIDS + Snakemake integration for neuroimaging workflows
**Key Features:** BIDS-aware workflow management, reproducible pipelines

---

## Tools in This Batch (3 New Skills)

### 1. NeuroDocker
**Website:** https://github.com/ReproNim/neurodocker
**GitHub:** https://github.com/ReproNim/neurodocker
**Platform:** Python (generates Docker/Singularity containers)
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
NeuroDocker is a command-line tool for generating custom Docker and Singularity containers for neuroimaging, developed by ReproNim. It simplifies the creation of reproducible containerized environments by providing a high-level interface for installing neuroimaging software packages (FSL, FreeSurfer, AFNI, SPM, ANTs, etc.) with proper dependencies, environment variables, and configurations. NeuroDocker ensures bit-for-bit reproducibility across institutions and computing environments.

**Key Capabilities:**
- Generate Dockerfiles and Singularity recipes
- Install neuroimaging tools with single commands
- Handle complex dependencies automatically
- Configure environment variables (FSLDIR, FREESURFER_HOME, etc.)
- Support Debian, Ubuntu, CentOS base images
- Minimize container size with multi-stage builds
- Integration with NeuroDebian repository
- Jupyter notebook environments
- BIDS Apps compatible containers
- Version pinning for exact reproducibility
- Miniconda and Python environment setup
- GPU support (CUDA, cuDNN) for deep learning
- Reproducible research environments

**Target Audience:**
- Method developers creating BIDS Apps
- Researchers ensuring reproducibility
- HPC administrators deploying neuroimaging software
- Multi-site consortia standardizing environments
- Cloud computing users
- Anyone needing reproducible computational environments

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install NeuroDocker via pip
   - Docker and Singularity prerequisites
   - Basic usage and command structure
   - Version checking

2. **Basic Container Generation**
   - Generate simple Dockerfile
   - Install single neuroimaging package
   - Build Docker image
   - Test container

3. **Installing Neuroimaging Software**
   - FSL installation
   - FreeSurfer installation
   - AFNI installation
   - ANTs installation
   - SPM12 installation
   - Multiple packages in one container

4. **Environment Configuration**
   - Environment variables
   - PATH modifications
   - Library paths (LD_LIBRARY_PATH)
   - MATLAB runtime configuration

5. **Base Image Selection**
   - Debian vs Ubuntu vs CentOS
   - Neurodebian base images
   - Minimal vs full images
   - Version pinning

6. **Python and Conda Environments**
   - Miniconda installation
   - Python package management
   - Nilearn, Nipype, PyMVPA installation
   - Virtual environments in containers

7. **Advanced Features**
   - Multi-stage builds for size optimization
   - GPU support (CUDA)
   - Jupyter notebook servers
   - Non-root user configuration
   - Custom scripts and entrypoints

8. **Singularity Containers**
   - Generate Singularity recipes
   - Convert Docker to Singularity
   - Build Singularity images
   - HPC deployment

9. **BIDS Apps Containers**
   - BIDS Apps specification
   - Create BIDS-compatible container
   - Entrypoint scripts
   - Validation and testing

10. **Reproducibility Best Practices**
    - Version pinning strategies
    - Container registries (Docker Hub, Singularity Hub)
    - Container metadata
    - Documentation

11. **Integration and Deployment**
    - Use with Pydra and Snakebids
    - HPC batch systems
    - Cloud deployment (AWS, GCP)
    - Container orchestration

12. **Troubleshooting**
    - Build failures
    - Package conflicts
    - Size optimization
    - Common errors

**Example Workflows:**
- Create fMRIPrep-compatible analysis container
- Build FreeSurfer + Python analysis environment
- Generate multi-tool neuroimaging container
- Deploy reproducible pipeline to HPC cluster
- Create BIDS App from scratch

---

### 2. Boutiques
**Website:** https://boutiques.github.io/
**GitHub:** https://github.com/boutiques/boutiques
**Platform:** Python
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
Boutiques is a framework for describing, validating, and executing command-line tools in a platform-independent manner. Developed at McGill University, Boutiques uses JSON schema descriptors that capture all aspects of a tool's interface (inputs, outputs, parameters, constraints), enabling automatic validation, execution on diverse platforms (local, HPC, cloud), and integration with workflow engines. It ensures tools are correctly installed, properly invoked, and produce expected outputs.

**Key Capabilities:**
- JSON-based tool descriptors
- Automatic input validation
- Command-line invocation generation
- Container integration (Docker, Singularity)
- Cross-platform execution
- Tool testing and validation
- Automatic documentation generation
- Integration with workflow engines (Pydra, Nipype)
- Error checking and debugging
- Tool publishing and sharing (Zenodo integration)
- Schema validation
- BIDS Apps integration
- Provenance tracking
- Tool versioning

**Target Audience:**
- Tool developers ensuring correct usage
- Pipeline developers integrating tools
- Workflow engine users
- BIDS Apps developers
- HPC users standardizing tool invocations
- Researchers validating tool installations

**Estimated Lines:** 700-750
**Estimated Code Examples:** 24-28

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install Boutiques via pip
   - Command-line interface overview
   - Basic concepts (descriptors, invocations)

2. **Tool Descriptors**
   - JSON descriptor structure
   - Define tool name and version
   - Specify inputs (files, parameters, flags)
   - Specify outputs
   - Command-line template

3. **Input Types and Validation**
   - File inputs
   - String, number, boolean parameters
   - Choice parameters (enums)
   - Lists and arrays
   - Value constraints (min, max, regex)

4. **Output Specification**
   - Output files
   - Path templates
   - Optional vs required outputs
   - Output validation

5. **Creating Descriptors**
   - Describe existing tool (e.g., FSL bet)
   - Validate descriptor schema
   - Test descriptor
   - Export documentation

6. **Executing Tools**
   - Create invocation JSON
   - Validate invocation
   - Execute locally
   - Execute in container
   - Capture outputs

7. **Container Integration**
   - Docker container execution
   - Singularity container execution
   - Container image specification
   - Volume mounts

8. **Advanced Descriptor Features**
   - Parameter groups
   - Conditional parameters (requires, disables)
   - Output file naming templates
   - Environment variables
   - Custom error messages

9. **Tool Testing**
   - Define test invocations
   - Automated testing
   - Validate tool installation
   - Regression testing

10. **Publishing and Sharing**
    - Publish to Zenodo
    - Tool registry
    - Versioning strategies
    - Citation generation

11. **Integration with Workflows**
    - Use with Pydra
    - Use with Nipype
    - BIDS Apps integration
    - Workflow composition

12. **Best Practices**
    - Descriptor design principles
    - Documentation standards
    - Versioning and updates
    - Error handling

**Example Workflows:**
- Create descriptor for FSL tool
- Validate and execute FreeSurfer command
- Integrate Boutiques tool into Pydra workflow
- Test tool across Docker and Singularity
- Publish tool descriptor to Zenodo

---

### 3. TemplateFlow
**Website:** https://www.templateflow.org/
**GitHub:** https://github.com/templateflow/templateflow
**Platform:** Python
**Priority:** High
**Current Status:** Does Not Exist - Need to Create

**Overview:**
TemplateFlow is a framework for managing, distributing, and using neuroimaging templates and atlases. Developed by the NiPreps community, TemplateFlow provides a centralized repository of standardized brain templates (MNI152, MNI305, fsaverage, etc.) with consistent metadata, ensuring spatial normalization uses validated, version-controlled references. It eliminates ambiguity about which template variant is used and enables reproducible spatial normalization.

**Key Capabilities:**
- Centralized repository of brain templates
- Programmatic template access via Python API
- Automatic template downloading and caching
- Template metadata (resolution, modality, cohort)
- Multiple coordinate systems (MNI, fsaverage, fsnative)
- Parcellation and atlas integration
- Template versioning and provenance
- BIDS-compatible naming conventions
- Support for pediatric, aging, and clinical templates
- Surface and volumetric templates
- Symmetric and asymmetric variants
- Integration with fMRIPrep, Nilearn, ANTs
- Custom template addition

**Target Audience:**
- fMRIPrep and preprocessing pipeline users
- Researchers performing spatial normalization
- Method developers needing standardized templates
- Multi-site studies requiring template consistency
- Anyone analyzing group-level neuroimaging data

**Estimated Lines:** 650-700
**Estimated Code Examples:** 22-26

**Key Topics to Cover:**

1. **Installation and Setup**
   - Install templateflow via pip
   - Configure template cache directory
   - Check available templates
   - Update template repository

2. **Template Basics**
   - What are brain templates?
   - Coordinate systems (MNI, Talairach, fsaverage)
   - Template resolutions
   - Modalities (T1w, T2w, PD, FLAIR)

3. **Accessing Templates**
   - List available templates
   - Query template metadata
   - Download specific template
   - Load template into Python

4. **Common Templates**
   - MNI152NLin2009cAsym (fMRIPrep default)
   - MNI152NLin6Asym
   - MNI152NLin2009aSym
   - OASIS30ANTs
   - fsaverage (surface)
   - fsLR (HCP surface)

5. **Template Metadata**
   - Resolution (1mm, 2mm)
   - Cohort (adult, pediatric, aging)
   - Modality
   - Symmetry (symmetric vs asymmetric)
   - Template versions

6. **Parcellations and Atlases**
   - Access brain parcellations
   - Schaefer parcellations
   - DKT atlas
   - Destrieux atlas
   - Harvard-Oxford atlas

7. **Surface Templates**
   - fsaverage surface spaces
   - fsLR (HCP) surfaces
   - fsnative individual surfaces
   - Surface resolution (fsaverage5, fsaverage6)

8. **Integration with Analysis Pipelines**
   - Use with fMRIPrep
   - Use with ANTsPy
   - Use with Nilearn
   - Custom normalization workflows

9. **Template Selection Guidelines**
   - Choose template for study population
   - Resolution considerations
   - Symmetric vs asymmetric
   - Legacy vs modern templates

10. **Custom Templates**
    - Add custom template to TemplateFlow
    - Template metadata specification
    - Validation and testing
    - Share custom templates

11. **Reproducibility Practices**
    - Document template usage
    - Version pinning
    - Template provenance
    - Citation generation

12. **Advanced Usage**
    - Programmatic template queries
    - Batch template downloads
    - Cache management
    - Template conversion utilities

**Example Workflows:**
- Download and visualize MNI152 template
- Access Schaefer parcellation for network analysis
- Use fsaverage template for surface-based analysis
- Integrate TemplateFlow with fMRIPrep
- Create custom study-specific template

---

## Implementation Strategy

### Development Approach

1. **Skill Creation Order:**
   - **NeuroDocker** - Container generation (new, 700-750 lines)
   - **Boutiques** - Tool descriptors (new, 700-750 lines)
   - **TemplateFlow** - Template management (new, 650-700 lines)
   - **Pydra** - Already exists (981 lines) ✓
   - **Snakebids** - Already exists (939 lines) ✓

2. **Comprehensive Coverage:**
   - Each new skill: 650-750 lines
   - 22-28 code examples per skill
   - Real-world reproducibility workflows
   - Integration examples across tools

3. **Consistent Structure:**
   - Overview and key features
   - Installation
   - Basic usage and examples
   - Advanced features
   - Integration with neuroimaging tools
   - Best practices for reproducibility
   - Troubleshooting
   - Resources and citations

### Code Example Categories

**For Each Tool:**

1. **Installation Examples** (2-3)
   - Package installation
   - Setup and configuration
   - Verification

2. **Basic Usage** (6-8)
   - Simple examples
   - Common use cases
   - Essential commands
   - Output inspection

3. **Advanced Features** (6-8)
   - Complex configurations
   - Optimization techniques
   - Custom extensions
   - Integration workflows

4. **Neuroimaging Applications** (4-6)
   - FSL integration
   - FreeSurfer integration
   - fMRIPrep integration
   - Custom pipelines

5. **Reproducibility** (3-5)
   - Version control
   - Documentation
   - Sharing and publishing
   - Validation

6. **Integration** (3-5)
   - Workflow engines
   - HPC deployment
   - Cloud platforms
   - BIDS Apps

### Cross-Tool Integration

All skills will demonstrate integration with:
- **Workflow engines:** Pydra, Snakebids, Nipype
- **Containers:** Docker, Singularity
- **Preprocessing:** fMRIPrep, QSIPrep, C-PAC
- **Core tools:** FSL, FreeSurfer, AFNI, ANTs
- **Standards:** BIDS, BIDS Apps
- **Platforms:** HPC, cloud, local

### Quality Targets

- **Minimum lines per skill:** 650
- **Target lines per skill:** 650-750
- **Minimum code examples:** 22
- **Target code examples:** 22-28
- **Total batch lines (new skills):** ~2,050-2,200
- **Total code examples (new skills):** ~68-82

---

## Batch Statistics

### Projected Totals

| Tool | Est. Lines | Est. Examples | Priority | Status |
|------|-----------|---------------|----------|---------|
| NeuroDocker | 700-750 | 24-28 | High | Create new |
| Boutiques | 700-750 | 24-28 | High | Create new |
| TemplateFlow | 650-700 | 22-26 | High | Create new |
| Pydra | 981 | ~30 | - | Already exists ✓ |
| Snakebids | 939 | ~28 | - | Already exists ✓ |
| **TOTAL (new)** | **2,050-2,200** | **70-82** | - | - |

### Coverage Analysis

**Domain Coverage:**
- ✓ Container generation (NeuroDocker)
- ✓ Tool standardization (Boutiques)
- ✓ Template management (TemplateFlow)
- ✓ Workflow engines (Pydra, Snakebids - existing)
- ✓ Reproducibility infrastructure (all tools)
- ✓ Cross-platform deployment (all tools)

**Platform Coverage:**
- Docker: NeuroDocker, Boutiques (2/3)
- Singularity: NeuroDocker, Boutiques (2/3)
- Python: All tools (3/3)
- HPC: NeuroDocker, Boutiques (2/3)
- Cloud: NeuroDocker, Boutiques (2/3)

**Application Areas:**
- Reproducible research: All tools
- Multi-site studies: NeuroDocker, TemplateFlow
- Tool development: NeuroDocker, Boutiques
- Workflow automation: Integration with Pydra, Snakebids
- BIDS Apps: NeuroDocker, Boutiques
- Spatial normalization: TemplateFlow

---

## Strategic Importance

### Fills Critical Gap

Previous batches have covered:
- Preprocessing pipelines (fMRIPrep, QSIPrep)
- Analysis methods (statistics, ML, connectivity)
- Workflow engines (Pydra, Snakebids)

**Batch 28 adds:**
- **Container generation** for perfect reproducibility
- **Tool standardization** via descriptors
- **Template management** for spatial consistency
- **Deployment infrastructure** for multi-platform execution
- **Validation frameworks** for quality assurance

### Complementary Skills

**Works with existing skills:**
- **Pydra (Batch 28):** Workflow execution with containers
- **Snakebids (Batch 28):** BIDS-aware pipelines with containers
- **fMRIPrep (Batch 5):** Uses TemplateFlow templates
- **All preprocessing tools:** Run in NeuroDocker containers
- **BIDS Validator (Batch 4):** Ensures input data quality

### User Benefits

1. **Perfect Reproducibility:**
   - Identical environments across sites
   - Version-controlled software stacks
   - Eliminate installation issues

2. **Simplified Deployment:**
   - One container for all tools
   - Easy HPC and cloud deployment
   - Standardized tool invocations

3. **Quality Assurance:**
   - Validate tool installations
   - Test tool functionality
   - Ensure correct usage

4. **Standardization:**
   - Consistent brain templates
   - Reproducible spatial normalization
   - Multi-site harmonization

---

## Dependencies and Prerequisites

### Software Prerequisites

**NeuroDocker:**
- Python 3.6+
- Docker (for Docker containers)
- Singularity (for Singularity containers)

**Boutiques:**
- Python 3.6+
- Docker or Singularity (optional, for execution)
- jsonschema

**TemplateFlow:**
- Python 3.6+
- requests (for template download)
- NiBabel (for NIfTI handling)

### Knowledge Prerequisites

Users should understand:
- Basic command-line usage
- Container concepts (Docker/Singularity)
- Neuroimaging file formats (NIfTI)
- Brain templates and spatial normalization
- JSON format (for Boutiques)

---

## Learning Outcomes

After completing Batch 28 skills, users will be able to:

1. **Generate Containers:**
   - Create custom neuroimaging containers
   - Install multiple tools in one container
   - Optimize container size
   - Deploy to registries

2. **Standardize Tools:**
   - Write tool descriptors
   - Validate tool invocations
   - Integrate tools into workflows
   - Test tool functionality

3. **Manage Templates:**
   - Access standard brain templates
   - Select appropriate templates for studies
   - Integrate templates into pipelines
   - Ensure spatial normalization reproducibility

4. **Ensure Reproducibility:**
   - Create reproducible compute environments
   - Version control software stacks
   - Document computational environments
   - Share reproducible pipelines

---

## Relationship to Existing Skills

### Builds Upon:
- **Pydra (Batch 28):** Workflow engine using containers
- **Snakebids (Batch 28):** Pipeline management
- **Nipype (Batch 2):** Original workflow engine
- **BIDS Validator (Batch 4):** Data validation
- **fMRIPrep (Batch 5):** Uses TemplateFlow

### Complements:
- **All preprocessing tools:** Can run in containers
- **All analysis tools:** Benefit from reproducibility
- **DataLad (Batch 4):** Version control for data
- **Quality control tools:** Validation frameworks

### Enables:
- Perfectly reproducible neuroimaging studies
- Multi-site consortia with identical environments
- Easy method sharing and replication
- Cloud and HPC deployment
- BIDS Apps development
- Standardized spatial normalization

---

## Expected Challenges and Solutions

### Challenge 1: Container Build Complexity
**Issue:** Neuroimaging software has complex dependencies
**Solution:** NeuroDocker abstracts complexity, clear examples, troubleshooting guides

### Challenge 2: Container Size
**Issue:** Containers with multiple tools can be very large
**Solution:** Multi-stage builds, selective installation, optimization examples

### Challenge 3: Learning JSON Descriptors
**Issue:** Boutiques descriptors have learning curve
**Solution:** Templates, progressive examples, validation tools

### Challenge 4: Template Selection
**Issue:** Many templates available, unclear which to use
**Solution:** Decision flowcharts, population-specific guidance, best practices

### Challenge 5: Platform Differences
**Issue:** Docker vs Singularity, HPC vs cloud variations
**Solution:** Platform-specific examples, conversion tools, compatibility notes

---

## Testing and Validation

Each skill will include:

1. **Installation Verification:**
   - Software installation tests
   - Container build tests
   - Template download tests

2. **Basic Functionality Tests:**
   - Simple container generation
   - Tool descriptor validation
   - Template access

3. **Integration Tests:**
   - Multi-tool containers
   - Workflow integration
   - Cross-platform execution

4. **Example Data:**
   - Public container recipes
   - Example Boutiques descriptors
   - Template visualization examples

---

## Timeline Estimate

**Per Skill:**
- NeuroDocker: 70-85 min (new, comprehensive)
- Boutiques: 70-85 min (new, JSON + validation)
- TemplateFlow: 60-75 min (new, template access)

**Total Batch 28:**
- ~3.5-4 hours total for new skills
- Can be completed in 1-2 extended sessions

---

## Success Criteria

Batch 28 will be considered successful when:

✓ All 3 new skills created with 650-750 lines each
✓ Total of 68+ code examples across new skills
✓ Each skill includes:
  - Comprehensive installation instructions
  - Basic to advanced usage examples
  - Neuroimaging tool integration
  - Reproducibility best practices
  - Container/descriptor/template workflows
  - Troubleshooting section
  - Integration with existing tools (Pydra, Snakebids, fMRIPrep)
  - Citations and resources

✓ Pydra and Snakebids verified as complete
✓ All examples tested and validated
✓ Cross-references to related skills complete
✓ Committed and pushed to repository
✓ Progress updated to 101/133 (75.9%)

---

## Next Batches Preview

### Batch 29: Advanced Surface Analysis
- CIVET (cortical surface extraction pipeline)
- BrainSuite (surface modeling suite)
- Mindboggle (brain morphometry and labeling)

### Batch 30: Meta-Analysis Tools (Remaining)
- NeuroSynth (automated meta-analysis) - if not already complete
- NiMARE (meta-analysis research environment) - if not already complete
- NeuroQuery (meta-analytic brain decoding)

### Batch 31: Deep Learning Extensions
- Additional deep learning tools for neuroimaging
- Transfer learning frameworks
- Specialized segmentation methods

---

## Conclusion

Batch 28 completes the **reproducibility and workflow infrastructure** for neuroimaging research, providing essential tools for containerization, tool standardization, and template management. By covering:

- **Container generation** (NeuroDocker)
- **Tool standardization** (Boutiques)
- **Template management** (TemplateFlow)
- **Workflow engines** (Pydra, Snakebids - existing)

This batch enables researchers to:
- **Achieve perfect reproducibility** across institutions
- **Standardize computational environments** for multi-site studies
- **Simplify software deployment** to HPC and cloud platforms
- **Validate tool installations** automatically
- **Ensure spatial normalization consistency** with versioned templates
- **Share methods** via containers and descriptors
- **Create BIDS Apps** for community distribution

These tools are critical for:
- Reproducible neuroimaging research
- Multi-site consortia (ABCD, UK Biobank, HCP)
- Method development and sharing
- Cloud and HPC deployment
- Publication requirements
- Open science initiatives

By providing comprehensive reproducibility infrastructure, Batch 28 positions users to conduct rigorous, reproducible neuroimaging research with standardized tools, validated environments, and version-controlled templates.

**Status After Batch 28:** 101/133 skills (75.9% complete - crossing 75%!)

---

**Document Version:** 1.0
**Created:** 2025-11-17
**Batch Status:** Planning Complete, Ready for Implementation
**Estimated Completion:** 3 new skills, ~2,050-2,200 lines, ~68-82 examples
