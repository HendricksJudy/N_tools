# BrainMap Sleuth - BrainMap Database Query Interface

## Overview

BrainMap Sleuth is the desktop application interface to the BrainMap database, one of the largest and most rigorously curated repositories of functional neuroimaging activation coordinates in the world. The BrainMap database contains manually extracted metadata from over 3,500 peer-reviewed neuroimaging studies, representing more than 100,000 activation coordinates organized by standardized behavioral domains, paradigm classes, and experimental contrasts. Unlike automated meta-analysis tools that use text mining (e.g., NeuroSynth), BrainMap relies on expert human curation to ensure accuracy and consistency of experimental annotations.

Sleuth provides a graphical interface for querying the BrainMap database using sophisticated filtering criteria including cognitive domains (emotion, memory, language, etc.), experimental paradigms (n-back, Stroop, face perception), subject demographics (age, gender, handedness), and imaging parameters (field strength, analysis software). Query results can be exported as coordinate files for activation likelihood estimation (ALE) meta-analysis using GingerALE or other coordinate-based meta-analysis tools. While the interface is somewhat dated, BrainMap Sleuth remains the gold standard for accessing high-quality, manually curated neuroimaging coordinates with rich experimental context.

**Official Website:** http://www.brainmap.org
**Download:** http://www.brainmap.org/sleuth/
**Documentation:** http://www.brainmap.org/sleuth/manual.html

### Key Features

- **Access to BrainMap Database:** >100,000 activation coordinates from >3,500 studies
- **Manual Curation:** Expert-annotated experimental metadata
- **Behavioral Domain Taxonomy:** Standardized cognitive categorization
- **Paradigm Class Filtering:** Specific task-based queries (n-back, Stroop, etc.)
- **Demographic Filtering:** Age, gender, handedness criteria
- **Imaging Parameters:** Field strength, modality, analysis software
- **Coordinate Export:** Text format for GingerALE meta-analysis
- **Space Conversion:** Talairach ↔ MNI coordinate transformation
- **Study Metadata:** Journal, year, sample size, contrast details
- **Cross-Platform:** Java-based application (Windows, macOS, Linux)
- **Regular Updates:** Database updated periodically with new studies

### Applications

- Coordinate-based meta-analysis (ALE, MKDA)
- Literature review and hypothesis generation
- Contextualizing individual study findings
- Identifying canonical activation patterns for cognitive processes
- Educational demonstrations of meta-analysis
- Comparing task-specific vs. domain-general activation

### BrainMap vs. NeuroSynth

**BrainMap (Sleuth):**
- **Pros:** Manually curated, high quality, detailed experimental annotations
- **Cons:** Smaller database, periodic updates, requires manual query

**NeuroSynth:**
- **Pros:** Automated extraction, >15,000 studies, continuous updates
- **Cons:** Automated extraction less precise, keyword-based only

**Recommendation:** Use BrainMap for high-quality curated coordinates; NeuroSynth for rapid exploratory analysis

### Citation

```bibtex
@article{Fox2005BrainMap,
  title={BrainMap taxonomy of experimental design: description and evaluation},
  author={Fox, Peter T and Lancaster, Jack L and Laird, Angela R and Eickhoff, Simon B},
  journal={Human Brain Mapping},
  volume={25},
  number={1},
  pages={185--198},
  year={2005},
  publisher={Wiley}
}

@article{Laird2005BrainMapDatabase,
  title={ALE meta-analysis: controlling the false discovery rate and performing
         statistical contrasts},
  author={Laird, Angela R and Fox, P Mickle and Price, Cathy J and others},
  journal={Human Brain Mapping},
  volume={25},
  number={1},
  pages={155--164},
  year={2005}
}
```

---

## Installation

### Downloading Sleuth

```bash
# Visit BrainMap website
# http://www.brainmap.org/sleuth/

# Download Sleuth application (Java-based)
# File: Sleuth_X.X.jar (cross-platform)

# Alternative: Platform-specific installers
# - Windows: Sleuth_Setup.exe
# - macOS: Sleuth.dmg
# - Linux: Sleuth.tar.gz

# Example download (Linux/macOS):
cd ~/Applications
wget http://www.brainmap.org/sleuth/downloads/Sleuth_2.4.jar
```

### Java Runtime Environment Requirements

```bash
# Sleuth requires Java Runtime Environment (JRE) 8 or later

# Check Java version:
java -version
# Required: java version "1.8" or higher

# Install Java if needed:

# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install default-jre

# macOS (Homebrew):
brew install openjdk@11

# Windows:
# Download from https://www.java.com/en/download/
```

### Database Installation and Updates

```bash
# First launch will download BrainMap database (~50 MB)

# Launch Sleuth:
java -jar Sleuth_2.4.jar

# On first launch:
# 1. Accept license agreement
# 2. Database will auto-download
# 3. Progress bar shows download status
# 4. Database saved to ~/.sleuth/database/

# Manual database update:
# Help > Check for Database Updates
# Downloads latest BrainMap database if available

# Database location:
ls ~/.sleuth/database/
# Output: brainmap.db (SQLite database)
```

### First Launch and Setup

```bash
# Launch Sleuth:
java -jar Sleuth_2.4.jar

# Initial setup wizard:
# 1. Select coordinate space preference (Talairach or MNI)
# 2. Set export directory for query results
# 3. Configure GingerALE integration (optional)

# Preferences:
# Edit > Preferences
# - Default coordinate space: Talairach or MNI152
# - Export directory: ~/BrainMapQueries/
# - GingerALE path: /Applications/GingerALE.app
```

### Platform Compatibility

```text
Operating Systems:
- Windows 7/8/10/11 (32-bit and 64-bit)
- macOS 10.10+ (Intel and Apple Silicon via Rosetta)
- Linux (Ubuntu, Fedora, CentOS)

Requirements:
- Java Runtime Environment 8+
- 2 GB RAM minimum, 4 GB recommended
- 500 MB disk space for database
- Internet connection for database updates
```

---

## BrainMap Database Structure

### Behavioral Domains Taxonomy

BrainMap uses a hierarchical taxonomy of behavioral domains:

```text
Level 1: Main Domains
├── Action
│   ├── Execution
│   ├── Inhibition
│   ├── Observation
│   └── Preparation
├── Cognition
│   ├── Attention
│   ├── Language
│   ├── Memory
│   ├── Reasoning
│   └── Working Memory
├── Emotion
│   ├── Anger
│   ├── Anxiety
│   ├── Disgust
│   ├── Fear
│   ├── Happiness
│   └── Sadness
├── Interoception
│   ├── Hunger
│   ├── Pain
│   └── Thirst
└── Perception
    ├── Audition
    ├── Gustation
    ├── Olfaction
    ├── Somesthesis
    └── Vision

Each domain can have sub-domains (up to 3 levels deep)
Example: Cognition → Memory → Working Memory → Spatial
```

### Paradigm Classes

Paradigm classes describe specific experimental tasks:

```text
Common Paradigm Classes:
- n-back (working memory load task)
- Stroop (cognitive control/interference)
- Go/No-Go (response inhibition)
- Emotional faces (face emotion processing)
- Word generation (verbal fluency)
- Cued recall (episodic memory retrieval)
- Semantic decision (semantic processing)
- Reward processing (monetary incentive delay)
- Oddball (attentional target detection)
- Passive viewing (stimulus presentation without task)

Each study coded with one or more paradigm classes
Enables precise task-based filtering
```

### Coordinate Spaces

```bash
# BrainMap stores coordinates in two spaces:

# 1. Talairach Space (historical standard)
# - Based on Talairach-Tournoux atlas (1988)
# - Single subject template
# - Most older studies use Talairach

# 2. MNI152 Space (modern standard)
# - Montreal Neurological Institute template
# - 152-subject average brain
# - Most recent studies use MNI

# Sleuth can convert between spaces:
# icbm2tal (MNI → Talairach)
# tal2icbm (Talairach → MNI)

# Conversion handled automatically during export
```

### Metadata Fields

```text
Each BrainMap experiment includes:

Study Information:
- Authors, journal, year, title, PubMed ID
- Number of subjects, groups, conditions

Subject Demographics:
- Age (mean, range)
- Gender distribution
- Handedness (right, left, mixed)
- Clinical status (healthy, patient population)

Imaging Parameters:
- Field strength (1.5T, 3T, 7T)
- Imaging modality (fMRI, PET)
- Analysis software (SPM, FSL, AFNI, etc.)

Experimental Design:
- Behavioral domains (cognitive processes)
- Paradigm class (specific task)
- Stimulus modality
- Response modality
- Experimental contrast

Coordinates:
- X, Y, Z in Talairach or MNI space
- Coordinate space explicitly labeled
```

---

## Querying by Behavioral Domain

### Searching for Specific Domains

```bash
# Sleuth GUI workflow:

# Step 1: Launch Sleuth
java -jar Sleuth_2.4.jar

# Step 2: Open Search panel
# View > Search > Behavioral Domain Search

# Step 3: Select domain(s)
# Navigate tree structure:
# Cognition > Working Memory
# Check "Working Memory" checkbox

# Step 4: Execute search
# Click "Search" button
# Results panel shows matching experiments

# Results display:
# - Number of experiments: 523
# - Number of subjects: 8,456
# - Number of coordinates: 4,231

# View coordinates:
# Results > View Coordinates Table
# Columns: X, Y, Z, Study, Contrast
```

### Combining Multiple Domains

```bash
# Query: Working Memory AND Emotion
# (Studies involving both cognitive and emotional processing)

# Method 1: Boolean AND (intersection)
# 1. Select "Cognition > Working Memory"
# 2. Select "Emotion" (any sub-domain)
# 3. Set combination mode: AND
# 4. Search

# Result: Studies coded with BOTH domains
# Example: Emotional n-back task

# Method 2: Boolean OR (union)
# 1. Select "Cognition > Attention"
# 2. Select "Cognition > Working Memory"
# 3. Set combination mode: OR
# 4. Search

# Result: Studies with EITHER domain
# Broader query, more results
```

### Excluding Domains

```bash
# Query: Cognition but NOT Language
# (Cognitive tasks excluding language processing)

# Workflow:
# 1. Select "Cognition" (all sub-domains)
# 2. Right-click "Language" > Exclude
# 3. Search

# Result: Cognitive studies without language component
# Useful for isolating specific processes

# Example use case:
# "Working Memory NOT Language"
# Isolates spatial/visual working memory tasks
# Excludes verbal working memory (n-back with letters)
```

### Example: Query Working Memory Studies

```bash
# Practical example: Meta-analysis of working memory

# Step-by-step:
# 1. Launch Sleuth
# 2. Behavioral Domain Search
# 3. Select: Cognition > Memory > Working Memory
# 4. Search

# Review results:
# - 500+ experiments
# - View study details (click on experiment)
# - Check paradigm classes (mostly n-back)
# - Inspect coordinates

# Refine query if needed:
# Add constraint: Paradigm Class = "n-back"
# Further filter: Age = Adults only
# Result: Canonical working memory network
```

---

## Filtering by Paradigm Class

### Paradigm Class Taxonomy

```bash
# Access paradigm class filter:
# View > Search > Paradigm Class Search

# Browse paradigm classes:
# - Alphabetically sorted
# - Searchable text field
# - Description tooltips

# Common paradigms:
# - n-back: 280 experiments
# - Stroop: 145 experiments
# - Go/No-Go: 98 experiments
# - Emotional faces: 215 experiments
# - Word generation: 187 experiments
```

### Selecting Specific Tasks

```bash
# Example: n-back working memory task

# Query setup:
# 1. Paradigm Class Search
# 2. Select "n-back"
# 3. Search

# Results:
# - 280 experiments using n-back
# - Typically 2-back or 3-back load
# - Mix of spatial and verbal n-back

# View activation pattern:
# Results > Export Coordinates > Open in Mango
# Expected pattern:
# - Dorsolateral prefrontal cortex (DLPFC)
# - Posterior parietal cortex
# - Anterior cingulate cortex

# Export for GingerALE meta-analysis:
# Results > Export > GingerALE Format
# File: nback_coordinates.txt
```

### Example: Stroop Task Studies

```bash
# Query Stroop interference studies

# Sleuth query:
# 1. Paradigm Class = "Stroop"
# 2. Optional: Add Behavioral Domain = "Cognition > Attention"
# 3. Search

# Results:
# - 145 Stroop experiments
# - Cognitive control network activation

# Typical activation:
# - Anterior cingulate cortex (ACC)
# - Dorsolateral prefrontal cortex
# - Inferior frontal gyrus

# Export and analyze:
# File > Export > stroop_studies.txt
# Open in GingerALE for ALE meta-analysis
```

---

## Advanced Filtering

### Subject Demographics

```bash
# Filter by age, gender, handedness

# Age filtering:
# View > Search > Subject Criteria
# Age range slider:
# - Min: 18, Max: 65 (adults only)
# - Excludes pediatric and elderly studies

# Gender filtering:
# Gender ratio:
# - Female only
# - Male only
# - Mixed (default)

# Handedness:
# - Right-handed only (most common)
# - Left-handed
# - Mixed

# Example: Adult right-handed working memory
# 1. Behavioral Domain = Working Memory
# 2. Age: 18-65
# 3. Handedness: Right
# 4. Search
```

### Imaging Parameters

```bash
# Filter by scanner and analysis software

# Field strength:
# View > Search > Imaging Criteria
# Select:
# - 1.5 Tesla
# - 3 Tesla
# - 7 Tesla
# - PET scanner

# Imaging modality:
# - fMRI (BOLD)
# - PET (H2O-15, FDG)
# - Combined (multi-modal)

# Analysis software:
# - SPM (Statistical Parametric Mapping)
# - FSL (FMRIB Software Library)
# - AFNI
# - BrainVoyager
# - Other

# Example: 3T fMRI studies only
# Field Strength = 3T
# Modality = fMRI
```

### Publication Metadata

```bash
# Filter by journal, year, citation count

# Publication year:
# View > Search > Publication Criteria
# Year range: 2010 - 2023 (recent studies)

# Journal filtering:
# - High-impact journals only
# - Specific journals (e.g., NeuroImage, JNeuro)

# Sample size:
# Minimum subjects: 20
# Excludes small underpowered studies

# Example: Recent large-sample emotion studies
# 1. Behavioral Domain = Emotion
# 2. Year: 2015-2023
# 3. Min subjects: 30
# 4. Search
```

### Example: 3T fMRI Studies in Adults

```bash
# Comprehensive query combining multiple criteria

# Query definition:
# Behavioral Domain: Cognition > Working Memory
# Paradigm Class: n-back
# Age: 18-65 years
# Field Strength: 3 Tesla
# Modality: fMRI
# Year: 2010-2023
# Min subjects: 20

# Execute search:
# Results: ~150 experiments
# High-quality, recent, well-powered studies

# Export for meta-analysis:
# File > Export > GingerALE Format
# Filename: nback_3T_adults_recent.txt

# This focused query provides canonical activation pattern
# for adult working memory at 3T
```

---

## Coordinate Export

### Export Formats

```bash
# Sleuth supports multiple export formats:

# 1. GingerALE format (most common)
# Plain text file with:
# // Reference: Author (Year)
# // Subjects: N
# X Y Z

# Example:
# // Reference: Smith et al. (2015)
# // Subjects: 24
# -42 28 32
# 44 30 28
# ...

# 2. XML format
# Structured data with full metadata

# 3. CSV format
# Spreadsheet-compatible
# Columns: Study, X, Y, Z, Space, Subjects

# 4. BrainMap format (internal)
# For BrainMap database submissions
```

### Coordinate Space Conversion

```bash
# Convert between Talairach and MNI during export

# Export dialog:
# File > Export Coordinates
# Coordinate Space dropdown:
# - Talairach
# - MNI (icbm152)
# - Original (as reported in paper)

# Automatic conversion:
# If study reported MNI, can export as Talairach
# Uses icbm2tal transformation
# Vice versa: tal2icbm

# Recommendation:
# Export in MNI for modern meta-analyses
# GingerALE handles both spaces
```

### Preparing Data for GingerALE

```bash
# Workflow: Sleuth query → GingerALE meta-analysis

# Step 1: Query in Sleuth
# Example: Working memory studies

# Step 2: Review results
# Check: >17 experiments (recommended minimum for ALE)

# Step 3: Export
# File > Export > GingerALE Format
# Filename: working_memory.txt
# Space: MNI

# Step 4: Open in GingerALE
java -jar GingerALE.jar

# Step 5: Load coordinate file
# File > Open > working_memory.txt

# Step 6: Run ALE analysis
# GingerALE performs activation likelihood estimation
# Generates statistical maps of consistent activation

# Result: Meta-analytic activation map
# Identifies brain regions consistently active across studies
```

### Example: Export to GingerALE Format

```bash
# Complete export workflow

# After Sleuth query with 50 experiments:

# Export coordinates:
# 1. Results panel: Click "Export"
# 2. Format: GingerALE
# 3. Coordinate space: MNI
# 4. Filename: emotion_regulation.txt
# 5. Save

# Verify export:
cat emotion_regulation.txt

# Output example:
# // Reference: Ochsner et al. (2004)
# // Subjects: 15
# -42 10 32
# 42 12 28
# -8 18 48
#
# // Reference: Wager et al. (2008)
# // Subjects: 24
# -44 14 30
# ...

# File ready for GingerALE ALE meta-analysis
```

---

## Integration with GingerALE

### Workflow: Sleuth → GingerALE

```bash
# Complete meta-analysis workflow

# Phase 1: Query in Sleuth
# 1. Define research question
#    Example: "What brain regions support working memory?"
# 2. Construct query
#    Behavioral Domain: Working Memory
#    Paradigm Class: n-back
#    Age: Adults
# 3. Execute search → 200 experiments

# Phase 2: Export coordinates
# 1. File > Export > GingerALE format
# 2. working_memory_coordinates.txt

# Phase 3: ALE meta-analysis in GingerALE
# 1. Launch GingerALE
# 2. File > Open > working_memory_coordinates.txt
# 3. Algorithm: ALE (default)
# 4. Threshold: p < 0.05 FWE-corrected
# 5. Minimum cluster size: 200 mm³
# 6. Run

# Phase 4: Interpret results
# ALE output:
# - Statistical maps (NIfTI format)
# - Cluster tables (peak coordinates, ALE values)
# - Visualization in Mango or MRIcron

# Expected working memory network:
# - Bilateral DLPFC (BA 9/46)
# - Posterior parietal cortex (BA 7/40)
# - Anterior cingulate (BA 32)
# - Pre-SMA
```

### Creating Focus Files

```bash
# GingerALE "focus file" = exported Sleuth coordinates

# Focus file requirements:
# - Plain text format
# - Study headers with metadata
# - Coordinate triplets (X Y Z)
# - Consistent coordinate space

# Sleuth export automatically creates valid focus files

# Manual editing (if needed):
# Add study that's not in BrainMap:
# // Reference: NewStudy et al. (2024)
# // Subjects: 30
# -45 28 32
# 48 30 28

# Multiple focus files:
# Can create separate files for contrasts
# Example:
# - attention_increase.txt
# - attention_decrease.txt
# Then run ALE contrast in GingerALE
```

---

## Database Statistics

### Viewing Database Coverage

```bash
# Check BrainMap database statistics

# Menu: Tools > Database Statistics

# Overall statistics:
# - Total studies: 3,542
# - Total experiments: 15,234
# - Total subjects: 127,856
# - Total coordinates: 102,341

# Coverage by modality:
# - fMRI: 85%
# - PET: 15%

# Field strength distribution:
# - 1.5T: 45%
# - 3T: 52%
# - 7T: 3%

# Coordinate space:
# - Talairach: 38%
# - MNI: 62%

# Publication years:
# - Range: 1985 - 2023
# - Peak: 2010-2015
```

### Study Count by Domain

```bash
# View study counts for each behavioral domain

# Tools > Domain Coverage

# Top domains by study count:
# 1. Cognition (5,421 experiments)
#    - Language: 1,234
#    - Working Memory: 823
#    - Attention: 756
# 2. Perception (4,102 experiments)
#    - Vision: 2,876
#    - Audition: 543
# 3. Emotion (2,156 experiments)
#    - Fear: 678
#    - Happiness: 334
# 4. Action (1,987 experiments)
#    - Execution: 1,234

# Use statistics to assess:
# - Sufficient data for meta-analysis (>17 studies)
# - Domain coverage gaps
# - Popular research areas
```

---

## Troubleshooting

### Java Compatibility Issues

**Problem:** Sleuth won't launch or crashes

**Solution:**
```bash
# Check Java version
java -version

# Ensure Java 8 or later installed
# If using Java 11+, may need compatibility flag:
java --illegal-access=permit -jar Sleuth_2.4.jar

# macOS-specific:
# If Java not found, install from:
brew install openjdk@11

# Set JAVA_HOME:
export JAVA_HOME=/Library/Java/JavaVirtualMachines/openjdk-11.jdk/Contents/Home
```

### Database Update Problems

**Problem:** Database won't update or download fails

**Solution:**
```bash
# Manual database download:
# 1. Visit http://www.brainmap.org/sleuth/database/
# 2. Download latest brainmap.db
# 3. Replace existing database:
mv ~/Downloads/brainmap.db ~/.sleuth/database/

# Check database integrity:
# Help > Validate Database
# Sleuth will check for corruption

# Reset database:
# Delete corrupted database:
rm -rf ~/.sleuth/database/
# Relaunch Sleuth to re-download
```

### Query Returning No Results

**Problem:** Search returns zero experiments

**Troubleshooting:**
```bash
# Check query criteria:
# 1. Too restrictive?
#    Example: "7T + Emotion + Age 65+" may have no studies
# 2. Typo in paradigm class search?
# 3. Incompatible combinations?

# Solution: Relax criteria
# Start broad, then narrow:
# - First: Single behavioral domain
# - Then add: Age restriction
# - Finally: Paradigm class

# Check database version:
# Help > About
# Ensure database is recent (updates 1-2 times/year)

# Example troubleshooting:
# Query: "Working Memory + 7T" → 0 results
# Relaxed: "Working Memory + 3T" → 200 results
# Conclusion: Insufficient 7T studies in database
```

---

## Best Practices

### Query Design Strategies

- **Start Broad:** Begin with behavioral domain only, then add filters
- **Check Sample Size:** Aim for >17 studies for reliable ALE meta-analysis
- **Consider Paradigm Homogeneity:** Mix of tasks may obscure specific patterns
- **Age Appropriateness:** Separate pediatric, adult, elderly if studying development
- **Modern Standards:** Prefer 3T fMRI, recent studies when possible
- **Document Queries:** Save query criteria for reproducibility

### When to Use BrainMap vs. NeuroSynth

**Use BrainMap (Sleuth) when:**
- Need high-quality curated coordinates
- Require detailed experimental filtering (specific paradigms)
- Performing formal coordinate-based meta-analysis (ALE)
- Quality over quantity preferred

**Use NeuroSynth when:**
- Need rapid exploratory analysis
- Want broad literature coverage (>15,000 studies)
- Interested in reverse inference (brain decoding)
- Prefer automated continuous updates

**Use Both:**
- BrainMap for primary meta-analysis
- NeuroSynth for validation and broader context

### Quality vs. Quantity Trade-offs

- **BrainMap Advantages:** Manual curation ensures accuracy, detailed task coding
- **BrainMap Limitations:** Smaller database, periodic updates (lag behind literature)
- **Recommendation:** For publication-quality meta-analysis, use BrainMap curated coordinates
- **Validation:** Consider comparing BrainMap results with NeuroSynth automated meta-analysis

---

## References

1. **BrainMap Database:**
   - Fox & Lancaster (2002). Mapping context and content: the BrainMap model. *Nat Rev Neurosci*, 3:319-321.
   - Laird et al. (2005). BrainMap: The BrainMap database of functional neuroimaging experiments. *Neuroinformatics*, 3:65-78.

2. **Behavioral Domains:**
   - Fox et al. (2005). BrainMap taxonomy of experimental design. *Hum Brain Mapp*, 25:185-198.

3. **ALE Meta-Analysis:**
   - Eickhoff et al. (2009). Coordinate-based activation likelihood estimation meta-analysis of neuroimaging data. *NeuroImage*, 25:1140-1151.
   - Turkeltaub et al. (2012). Minimizing within-experiment and within-group effects in ALE meta-analyses. *Hum Brain Mapp*, 33:1-13.

4. **Comparison with Automated Methods:**
   - Yarkoni et al. (2011). Large-scale automated synthesis of human functional neuroimaging data. *Nat Methods*, 8:665-670.

5. **Coordinate Spaces:**
   - Lancaster et al. (2007). Bias between MNI and Talairach coordinates analyzed using the ICBM-152 brain template. *Hum Brain Mapp*, 28:1194-1205.

**Official Resources:**
- BrainMap: http://www.brainmap.org
- Sleuth Manual: http://www.brainmap.org/sleuth/manual.html
- GingerALE: http://www.brainmap.org/ale/
- Taxonomy: http://www.brainmap.org/taxonomy/
