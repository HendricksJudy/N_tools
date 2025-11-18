# Batch 43 Plan: Additional Meta-Analysis Tools

## Overview

**Theme:** Additional Meta-Analysis Tools
**Focus:** Meta-analytic brain decoding and database interfaces
**Target:** 2 skills, 1,100-1,200 lines

**Current Progress:** 130/133 skills (97.7%)
**After Batch 42:** 130/133 skills (97.7%)
**After Batch 43:** 132/133 skills (99.2%)

This batch addresses the final meta-analysis tools for neuroimaging: BrainMap Sleuth for accessing the BrainMap database and NeuroQuery for automated meta-analytic decoding. These tools enable researchers to contextualize their findings within the broader neuroimaging literature, predict brain activation patterns from cognitive terms, and perform large-scale meta-analyses without manually collecting studies.

## Rationale

Meta-analysis tools synthesize findings across thousands of neuroimaging studies:

- **Literature Contextualization:** Compare new findings to existing literature
- **Automated Meta-Analysis:** Generate activation maps from text queries
- **Brain Decoding:** Predict cognitive functions from activation patterns (reverse inference)
- **Hypothesis Generation:** Explore brain-cognition relationships
- **Open Science:** Leverage community-curated databases (BrainMap, NeuroVault)

This batch completes the meta-analysis coverage in N_tools, complementing earlier tools (NeuroSynth, NiMARE, GingerALE) with database-specific interfaces and modern decoding approaches.

## Skills to Create

### 1. BrainMap Sleuth (550-600 lines, 18-20 examples)

**Overview:**
BrainMap Sleuth is the desktop application interface to the BrainMap database, one of the largest manually curated repositories of functional neuroimaging coordinates. The BrainMap database contains metadata from >3,500 papers with >100,000 activation coordinates, categorized by cognitive paradigms (behavioral domains, paradigm classes) and experimental contrasts. Sleuth allows researchers to query the database, filter studies by experimental design criteria, and export coordinates for meta-analysis with GingerALE or other tools. While somewhat dated in interface, BrainMap remains valuable for high-quality, manually curated coordinate data with detailed experimental annotations.

**Key Features:**
- Access to BrainMap coordinate database (>100,000 coordinates)
- Query by behavioral domain (e.g., emotion, memory, language)
- Filter by paradigm class (e.g., n-back, Stroop, face perception)
- Subject demographics filtering (age, gender, handedness)
- Imaging parameters filtering (field strength, analysis software)
- Coordinate export for GingerALE meta-analysis
- Study metadata (journal, year, sample size)
- Manual curation ensures high quality
- Java-based cross-platform application
- Integration with BrainMap taxonomy

**Target Audience:**
- Researchers performing coordinate-based meta-analyses
- Scientists needing high-quality curated activation data
- GingerALE users requiring coordinate input
- Educators teaching meta-analysis methods
- Researchers contextualizing individual study findings

**Main Sections:**

1. **Overview** (40 lines)
   - Introduction to BrainMap database
   - Sleuth application purpose
   - Comparison with NeuroSynth/NeuroQuery
   - Curation vs. automated extraction
   - Citation information

2. **Installation** (60 lines)
   - Downloading Sleuth (Java application)
   - Java Runtime Environment requirements
   - Database installation and updates
   - First launch and setup
   - Platform compatibility

3. **BrainMap Database Structure** (70 lines, 2-3 examples)
   - Behavioral domains taxonomy
   - Paradigm classes
   - Coordinate spaces (Talairach, MNI)
   - Metadata fields
   - Example: Browse database structure

4. **Querying by Behavioral Domain** (80 lines, 3-4 examples)
   - Searching for specific domains
   - Combining multiple domains
   - Excluding domains
   - Example: Query working memory studies
   - Example: Emotion processing studies

5. **Filtering by Paradigm Class** (70 lines, 2-3 examples)
   - Paradigm class taxonomy
   - Selecting specific tasks
   - Example: n-back working memory tasks

6. **Advanced Filtering** (80 lines, 3-4 examples)
   - Subject demographics (age, gender)
   - Imaging parameters (field strength, modality)
   - Publication metadata (journal, year)
   - Sample size criteria
   - Example: 3T fMRI studies in adults

7. **Coordinate Export** (70 lines, 2-3 examples)
   - Export formats (text, XML)
   - Coordinate space conversion (Talairach↔MNI)
   - Preparing data for GingerALE
   - Example: Export to GingerALE format

8. **Integration with GingerALE** (60 lines, 1-2 examples)
   - Workflow: Sleuth → GingerALE
   - Creating focus files
   - Running ALE meta-analysis

9. **Database Statistics** (50 lines, 1-2 examples)
   - Viewing database coverage
   - Study count by domain
   - Publication trends

10. **Troubleshooting** (40 lines)
    - Java compatibility issues
    - Database update problems
    - Query returning no results

11. **Best Practices** (30 lines)
    - Query design strategies
    - When to use BrainMap vs. NeuroSynth
    - Quality vs. quantity trade-offs

12. **References** (20 lines)
    - BrainMap papers
    - Meta-analysis methodology

**Code Examples:**
- Launch Sleuth (command line)
- Query construction (GUI screenshots/descriptions)
- Export coordinates (GUI)
- GingerALE integration (workflow)

**Integration Points:**
- GingerALE for ALE meta-analysis
- Mango for coordinate visualization
- NiMARE for Python-based meta-analysis
- Comparison with NeuroSynth

---

### 2. NeuroQuery (550-600 lines, 18-20 examples)

**Overview:**
NeuroQuery is a modern meta-analytic tool that generates statistical brain maps from free-form text queries using machine learning trained on the neuroimaging literature. Unlike traditional meta-analysis requiring manual study selection, NeuroQuery automatically extracts relationships between terms (e.g., "working memory", "cognitive control") and brain coordinates from >13,000 studies. It produces predictive activation maps, associated terms, and study lists for any neuroscience query. NeuroQuery offers both a web interface (https://neuroquery.org) and Python API, enabling researchers to rapidly explore brain-cognition relationships, generate hypotheses, and decode brain activation patterns through reverse inference.

**Key Features:**
- Text-to-brain mapping from >13,000 neuroimaging studies
- Free-form natural language queries
- Predictive statistical brain maps (z-scores)
- Associated terms and synonyms
- Study lists with relevance scores
- Reverse inference (decode brain patterns)
- Python API for programmatic access
- Web interface for interactive exploration
- Semantic similarity search
- Integration with NeuroVault activation maps
- Open-source and continuously updated

**Target Audience:**
- Researchers exploring brain-cognition relationships
- Scientists performing hypothesis generation
- Educators demonstrating meta-analytic principles
- Developers integrating meta-analysis into workflows
- Reviewers contextualizing manuscript findings

**Main Sections:**

1. **Overview** (40 lines)
   - Introduction to NeuroQuery
   - Machine learning for meta-analysis
   - Comparison with NeuroSynth
   - Web vs. Python API
   - Citation information

2. **Web Interface Usage** (80 lines, 3-4 examples)
   - Accessing https://neuroquery.org
   - Text query interface
   - Interpreting brain maps
   - Viewing associated terms
   - Example: Query "working memory"
   - Example: Query "default mode network"

3. **Python API Installation** (60 lines)
   - Installing neuroquery package
   - Downloading trained models
   - Loading datasets
   - Testing installation

4. **Basic Queries** (80 lines, 3-4 examples)
   - Single term queries
   - Multi-word queries
   - Viewing predicted activation maps
   - Example: Python query for "fear"
   - Example: Complex query "emotion regulation"

5. **Interpreting Results** (90 lines, 3-4 examples)
   - Z-score brain maps
   - Statistical thresholding
   - Associated terms and weights
   - Relevant studies
   - Example: Analyze query output

6. **Reverse Inference (Brain Decoding)** (80 lines, 2-3 examples)
   - Predicting cognitive terms from activation
   - Using coordinate sets as input
   - Interpreting decoder output
   - Example: Decode activation pattern

7. **Advanced Queries** (70 lines, 2-3 examples)
   - Boolean operators (AND, OR, NOT)
   - Weighted term queries
   - Custom vocabularies
   - Example: "attention NOT memory"

8. **Visualization** (80 lines, 3-4 examples)
   - Plotting brain maps with nilearn
   - Interactive viewers
   - Exporting NIfTI images
   - Example: Create publication figure

9. **Comparing with NeuroSynth** (60 lines, 1-2 examples)
   - Methodology differences
   - When to use NeuroQuery vs. NeuroSynth
   - Complementary strengths

10. **Integration with Research Workflows** (60 lines, 1-2 examples)
    - Hypothesis generation
    - Result contextualization
    - Literature search augmentation

11. **Troubleshooting** (40 lines)
    - Model download issues
    - Empty query results
    - Memory limitations

12. **Best Practices** (30 lines)
    - Query formulation tips
    - Interpreting predictive maps
    - Limitations and caveats

13. **References** (20 lines)
    - NeuroQuery papers
    - Meta-analytic decoding methods

**Code Examples:**
- Web interface usage (screenshots/descriptions)
- Python API queries (Python)
- Visualization (Python)
- Reverse inference (Python)
- Custom analysis pipelines (Python)

**Integration Points:**
- nilearn for visualization
- NeuroSynth for comparison
- NiMARE for comprehensive meta-analysis
- NeuroVault for activation map databases

---

## Implementation Checklist

### Per-Skill Requirements
- [ ] 550-600 lines per skill
- [ ] 18-20 code examples per skill
- [ ] Consistent section structure
- [ ] Installation instructions
- [ ] Basic and advanced usage
- [ ] Visualization examples
- [ ] Integration examples
- [ ] Troubleshooting section
- [ ] Best practices
- [ ] References with citations

### Quality Assurance
- [ ] All examples functional
- [ ] Web interface descriptions accurate
- [ ] Python code tested
- [ ] Clear methodology explanations
- [ ] Practical workflows
- [ ] Common issues covered
- [ ] Complete references

### Batch Requirements
- [ ] Total lines: 1,100-1,200
- [ ] Total examples: 36-40
- [ ] Consistent markdown formatting
- [ ] Cross-referencing meta-analysis tools
- [ ] Focus on practical applications

## Timeline

1. **BrainMap Sleuth**: 550-600 lines, 18-20 examples
2. **NeuroQuery**: 550-600 lines, 18-20 examples

**Estimated Total:** 1,100-1,200 lines, 36-40 examples

## Context & Connections

### Meta-Analysis Ecosystem

**Database-Driven (BrainMap Sleuth):**
```
BrainMap Database → Sleuth Query → Coordinate Export → GingerALE → ALE Meta-Analysis
        ↓               ↓              ↓                 ↓              ↓
   Curated Data   Filter Criteria   Text Files      Statistical    Thresholded
   (>100k coords)  (Domain/Task)                      Modeling         Maps
```

**Text-Driven (NeuroQuery):**
```
Text Query → ML Model → Predicted Brain Map + Associated Terms + Studies
     ↓          ↓             ↓                      ↓                ↓
"working    Trained on   Z-score NIfTI           Related        Relevant
 memory"    13k studies                          Concepts         Papers
```

### Complementary Tools

**Already Covered:**
- **NeuroSynth:** Automated meta-analysis (term-based)
- **NiMARE:** Comprehensive Python meta-analysis framework
- **GingerALE:** ALE meta-analysis (uses BrainMap/Sleuth coordinates)
- **nilearn:** Visualization for meta-analysis results

**New Capabilities:**
- **BrainMap Sleuth:** Curated database access, detailed experimental filtering
- **NeuroQuery:** Natural language queries, predictive modeling, reverse inference

### Tool Comparison

| Feature | NeuroSynth | NeuroQuery | BrainMap Sleuth |
|---------|-----------|------------|-----------------|
| Data Source | Automated extraction | ML on literature | Manual curation |
| Study Count | ~15,000 | ~13,000 | ~3,500 |
| Query Type | Keywords | Natural language | Structured taxonomy |
| Output | Statistical maps | Predictive maps | Coordinates |
| Reverse Inference | Yes | Yes | No |
| Quality | Variable | High | Very high |
| Update Frequency | Regular | Regular | Periodic |

## Expected Impact

### Research Community
- Rapid hypothesis generation from text queries
- High-quality curated coordinates for meta-analysis
- Brain-cognition relationship exploration
- Literature contextualization

### Methodological Advances
- Predictive modeling of brain activation
- Semantic meta-analysis
- Integration of natural language processing with neuroimaging

### Education
- Demonstrating meta-analytic principles
- Exploring brain-behavior relationships
- Understanding database curation vs. automation

## Conclusion

Batch 43 completes the meta-analysis coverage in N_tools by documenting two complementary approaches:

1. **BrainMap Sleuth** provides access to manually curated, high-quality activation coordinates with detailed experimental annotations
2. **NeuroQuery** enables natural language meta-analytic queries with predictive brain mapping

By completing this batch, the N_tools collection will reach **132/133 skills (99.2%)**, with comprehensive coverage of neuroimaging meta-analysis methods spanning:
- Automated extraction (NeuroSynth)
- Comprehensive frameworks (NiMARE)
- Statistical modeling (GingerALE, AES-SDM)
- Database access (BrainMap Sleuth)
- Predictive modeling (NeuroQuery)

These tools democratize access to the neuroimaging literature, enabling researchers to contextualize findings, generate hypotheses, and perform large-scale meta-analyses without manually collecting hundreds of studies.
