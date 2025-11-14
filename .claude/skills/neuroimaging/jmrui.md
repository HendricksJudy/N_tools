# jMRUI

## Overview

jMRUI (Java-based Magnetic Resonance User Interface) is a comprehensive software package for time-domain analysis of Magnetic Resonance Spectroscopy (MRS) and Spectroscopic Imaging (MRSI) data. Written in Java for cross-platform compatibility, jMRUI provides sophisticated quantitation methods including AMARES, QUEST, and AQSES, along with extensive preprocessing and visualization capabilities.

**Website:** http://www.jmrui.eu/
**Platform:** Java (Windows/macOS/Linux)
**Language:** Java
**License:** Free for research use

## Key Features

- Time-domain quantitation (AMARES, QUEST, AQSES)
- Frequency-domain analysis
- Advanced preprocessing tools
- MRSI support (2D and 3D)
- Multiple vendor format support
- Prior knowledge incorporation
- Hankel SVD (HSVD) filtering
- Quality control tools
- Publication-quality visualization
- Simulation capabilities
- Batch processing
- Integration with MATLAB

## Installation

### Requirements

```bash
# Java Runtime Environment (JRE) 8 or later
java -version

# Verify Java is installed
# Should show version 1.8.0 or higher
```

### Download and Install

```bash
# Download from website
# http://www.jmrui.eu/downloads/

# Linux/macOS
unzip jMRUI_v6.0beta.zip
cd jMRUI_v6.0beta
chmod +x jMRUI.sh

# Launch
./jMRUI.sh

# Windows
# Extract zip file
# Double-click jMRUI.bat or jMRUI.exe
```

### First Launch

```bash
# jMRUI will create configuration directory
# ~/.jmrui/ (Linux/macOS)
# C:\Users\<username>\.jmrui\ (Windows)

# Contains:
# - User preferences
# - Prior knowledge files
# - Temporary files
# - Processing histories
```

## Data Formats

### Supported Vendors

```bash
# jMRUI can read:
# - Siemens: DICOM, .rda
# - Philips: .spar/.sdat, DICOM
# - GE: P-files
# - Bruker: ser/fid
# - Varian: .fid
# - jMRUI native: .mrui, .txt
# - Generic ASCII
# - DICOM MRS
```

### jMRUI Data Structure

```bash
# jMRUI uses its own data format (.mrui)
# Stores:
# - Time-domain signals
# - Acquisition parameters
# - Processing history
# - Metadata
# - Annotations
```

## Basic Workflow

### Step 1: Load Data

```bash
# Launch jMRUI
./jMRUI.sh

# In GUI:
# File → Open → Select format → Browse to file
# Supported formats in dropdown menu

# For MRSI data:
# File → Open MRSI → Select format → Browse to file
```

### Step 2: Inspect Spectrum

```bash
# After loading, inspect in time and frequency domains
# Toggle between:
# - Time domain (FID)
# - Frequency domain (spectrum)
# - Real/Imaginary/Magnitude/Phase

# Use View menu:
# View → Time Domain
# View → Frequency Domain
# View → Display Options
```

### Step 3: Preprocessing

```bash
# Common preprocessing steps:

# 1. Truncation (remove post-acquisition points)
# Preprocessing → Truncation

# 2. Apodization (line broadening)
# Preprocessing → Apodization → Exponential

# 3. Zero filling
# Preprocessing → Zero Filling

# 4. Phase correction
# Preprocessing → Phase Correction → Manual/Automatic

# 5. Frequency shift
# Preprocessing → Frequency Shift

# 6. Water removal (HLSVD)
# Preprocessing → HLSVD Filter
```

### Step 4: Quantitation

```bash
# Choose quantitation method:

# AMARES (Advanced Method for Accurate, Robust, and Efficient Spectral fitting)
# Quantitation → AMARES → Set prior knowledge → Fit

# QUEST (QUantitation based on QUantum ESTimation)
# Quantitation → QUEST → Load basis set → Fit

# AQSES (Automated Quantitation of Short Echo time MRS)
# Quantitation → AQSES → Configure → Fit
```

## AMARES Quantitation

### Create Prior Knowledge

```bash
# AMARES requires prior knowledge definition
# Specify for each metabolite peak:
# - Frequency (ppm)
# - Damping (linewidth)
# - Amplitude
# - Phase
# - Constraints on parameters

# Example: NAA singlet at 2.01 ppm
# Frequency: 2.01 ppm (fixed or allowed to vary ±0.05)
# Damping: 5 Hz (allowed to vary 2-15 Hz)
# Amplitude: Free
# Phase: Constrained to 0 or free
```

### AMARES Workflow

```bash
# 1. Load spectrum
# 2. Create or load prior knowledge file
# 3. Set fitting range (e.g., 1.8-2.2 ppm for NAA)
# 4. Run AMARES
# 5. Inspect fit quality
# 6. Extract results (amplitudes, frequencies, linewidths)
# 7. Export results
```

### Prior Knowledge Example

```text
# Prior knowledge for brain metabolites
# Format: MetabName Freq Damp Amp Phase Constraints

NAA_2.01
  Freq: 2.01 [1.96, 2.06]   # ppm, allowed range
  Damp: 5 [2, 15]            # Hz, allowed range
  Amp:  1 [0, 10]            # Free, positive
  Phase: 0 [-30, 30]         # degrees

Cr_3.03
  Freq: 3.03 [2.98, 3.08]
  Damp: 5 [2, 15]
  Amp:  1 [0, 10]
  Phase: 0 [-30, 30]

Cho_3.20
  Freq: 3.20 [3.15, 3.25]
  Damp: 5 [2, 15]
  Amp:  1 [0, 10]
  Phase: 0 [-30, 30]
```

## QUEST Quantitation

### Load Basis Set

```bash
# QUEST uses simulated basis sets
# Load pre-computed basis or create new one

# In jMRUI:
# Quantitation → QUEST → Load Basis Set
# Browse to .txt basis set file

# Basis set must match:
# - Echo time
# - Field strength
# - Sequence type
```

### QUEST Workflow

```bash
# 1. Load in vivo spectrum
# 2. Load basis set
# 3. Define metabolites to include
# 4. Set baseline model (splines, polynomial)
# 5. Set constraints
# 6. Run QUEST
# 7. View fitted components
# 8. Export concentrations
```

### Interpret QUEST Results

```bash
# QUEST provides:
# - Metabolite amplitudes (arbitrary units)
# - Standard deviations (Cramér-Rao bounds)
# - Baseline
# - Residual
# - Fit quality metrics

# Good fit:
# - Low residual
# - CRLBs < 20%
# - Reasonable baseline
```

## HLSVD Water Removal

### Hankel-Lanczos SVD

```bash
# HLSVD removes water and lipid signals
# Based on singular value decomposition

# In jMRUI:
# Preprocessing → HLSVD Filter

# Parameters:
# - Frequency range (e.g., 4.4-5.0 ppm for water)
# - Number of components (typically 10-30)
# - Threshold (automatic or manual)
```

### HLSVD Example

```bash
# Remove water peak at 4.68 ppm

# 1. Load spectrum
# 2. Preprocessing → HLSVD Filter
# 3. Set frequency range: 4.4-5.0 ppm
# 4. Set components: 20
# 5. Run filter
# 6. Inspect result
# 7. Apply if satisfactory

# Can iteratively remove:
# - Water (4.4-5.0 ppm)
# - Lipids (0.8-1.5 ppm)
```

## MRSI Analysis

### Load MRSI Data

```bash
# Load 2D or 3D MRSI dataset
# File → Open MRSI → Select format

# jMRUI displays:
# - Spatial grid
# - Individual voxel spectra
# - Summary images (metabolite maps)
```

### Process Individual Voxels

```bash
# Select voxel(s) in spatial grid
# Process selected voxels:
# - Preprocessing
# - Quantitation
# - Quality control

# Batch process all voxels:
# MRSI → Batch Process → Select method
# - Apply same processing to all voxels
# - Generate metabolite maps
# - Export results per voxel
```

### Create Metabolite Maps

```bash
# After quantitation:
# MRSI → Create Maps → Select metabolites

# Maps show spatial distribution of:
# - NAA
# - Cho
# - Cr
# - Ratios (e.g., Cho/NAA)
# - Quality metrics

# Export maps as images or matrices
```

## Simulation

### Simulate Metabolite Signals

```bash
# jMRUI includes simulation tools

# Simulation → Create Signal
# Define:
# - Frequency (ppm)
# - Amplitude
# - Linewidth (Hz)
# - Phase
# - Time points
# - Spectral width

# Can simulate:
# - Single peaks
# - Multiplets
# - J-coupled systems
# - Complex metabolites
```

### Create Basis Sets

```bash
# Simulate basis set for QUEST

# For each metabolite:
# 1. Define spin system
# 2. Simulate at specific TE
# 3. Add appropriate linewidth
# 4. Save as basis function

# Combine all metabolites into basis set
# Export in jMRUI format for QUEST
```

## Batch Processing

### Process Multiple Spectra

```bash
# Batch mode for processing many spectra with same parameters

# 1. Load first spectrum
# 2. Define processing pipeline:
#    - Preprocessing steps
#    - Quantitation method
#    - Prior knowledge or basis set
# 3. Save processing template
# 4. Apply to all spectra in batch

# Batch → Load Template → Select Files → Process All
```

### Export Batch Results

```bash
# After batch processing:
# Results → Export All

# Export formats:
# - CSV (metabolite concentrations)
# - Text tables
# - MATLAB .mat files
# - XML format

# Typical output:
# Subject, NAA, Cr, Cho, mI, Glu, ...
# sub-01, 10.5, 8.2, 2.1, 7.3, 12.1, ...
# sub-02, 11.2, 8.5, 1.9, 6.8, 11.5, ...
```

## Integration with Other Tools

### Export to MATLAB

```bash
# Export data for further analysis in MATLAB

# File → Export → MATLAB format
# Creates .mat file with:
# - Time-domain data
# - Frequency-domain spectrum
# - Parameters
# - Results

# In MATLAB:
# load('spectrum.mat');
# plot(real(spectrum));
```

### Import from Osprey/FID-A

```bash
# Convert to jMRUI format

# From Osprey (MATLAB):
# io_writejmrui(MRSCont.processed.metab{1}, 'subject01.txt');

# From FID-A (MATLAB):
# io_writejmrui(in, 'subject01.txt');

# Load in jMRUI:
# File → Open → jMRUI text format → Select file
```

### Export to LCModel

```bash
# Convert jMRUI data to LCModel RAW format

# File → Export → LCModel RAW
# Saves as .RAW file compatible with LCModel
```

## Quality Control

### Visual Inspection

```bash
# Always visually inspect:
# 1. Raw time-domain signal (FID)
#    - Check for artifacts
#    - Verify decay pattern
#    - Look for discontinuities

# 2. Frequency-domain spectrum
#    - Baseline flatness
#    - Peak shapes
#    - Water suppression
#    - Lipid contamination

# 3. Fitted spectrum
#    - Residual should be noise-like
#    - No systematic patterns
#    - Good fit to all metabolites
```

### Quality Metrics

```bash
# jMRUI provides:
# - SNR (signal-to-noise ratio)
# - Linewidth (FWHM)
# - CRLBs (Cramér-Rao lower bounds)
# - Fit residual

# View metrics:
# Results → Quality Metrics

# Good quality criteria:
# - SNR > 10 (preferably > 20)
# - FWHM < 0.1 ppm (< 13 Hz @ 3T)
# - CRLBs < 20% for main metabolites
```

## Advanced Features

### Constrained Fitting

```bash
# AMARES allows sophisticated constraints

# Link parameters across peaks:
# - Force same linewidth for all singlets
# - Constrain relative amplitudes
# - Fix phase relationships
# - Set multiplet patterns

# Example: Cr doublet at 3.03 ppm
# - Two peaks with fixed frequency separation
# - Equal amplitudes
# - Same linewidth
# - Same phase
```

### Multi-Component Analysis

```bash
# Separate overlapping signals

# Example: Lac doublet at 1.31 ppm
# - Overlaps with lipids
# - Use AMARES with prior knowledge
# - Constrain Lac doublet pattern
# - Allow lipid baseline

# Can distinguish:
# - Lactate from lipids
# - Glutamate from glutamine
# - GABA from overlapping signals
```

### Spectral Editing Analysis

```bash
# Process MEGA-PRESS or other edited sequences

# 1. Load edit-ON and edit-OFF
# 2. Calculate difference spectrum
# 3. Quantify edited metabolites (GABA, GSH)
# 4. Account for co-editing

# jMRUI handles:
# - Difference spectrum creation
# - Phase correction across conditions
# - Fitting of edited signals
```

## Scripting and Automation

### Command-Line Interface

```bash
# jMRUI supports batch processing via command line

# Example: Process all files in directory
java -jar jMRUI.jar \
  --batch \
  --input /data/*.txt \
  --preprocessing preprocessing_template.xml \
  --quantitation AMARES \
  --priorknowledge NAA_Cr_Cho.prior \
  --output /results/

# Template files define:
# - Processing steps
# - Parameters
# - Methods
```

### MATLAB Integration

```bash
# Call jMRUI from MATLAB

# 1. Export from MATLAB to jMRUI format
% io_writejmrui(spectrum, 'temp.txt');

# 2. Process in jMRUI (command line)
% system('java -jar jMRUI.jar --process temp.txt');

# 3. Import results back to MATLAB
% results = load('temp_results.mat');
```

## Integration with Claude Code

When helping users with jMRUI:

1. **Check Java Installation:**
   ```bash
   java -version
   # Need Java 8 or later
   ```

2. **Common Workflow:**
   - Load data → Preprocess → Define prior knowledge → Quantify → Export

3. **Prior Knowledge Files:**
   - Critical for AMARES success
   - Start with provided examples
   - Adjust based on your data

4. **Quality Checks:**
   - Always inspect fits visually
   - Check residuals for patterns
   - Verify CRLBs are reasonable

## Troubleshooting

**Problem:** jMRUI won't launch
**Solution:** Check Java version (need 8+), verify JAR file integrity, check file permissions

**Problem:** Cannot read vendor files
**Solution:** Verify format selection, check DICOM compliance, convert to jMRUI text format

**Problem:** AMARES fit fails
**Solution:** Adjust prior knowledge constraints, reduce number of peaks, improve SNR, check frequency range

**Problem:** Very high CRLBs
**Solution:** Inspect spectrum quality, simplify model, add constraints, check for overlapping peaks

**Problem:** HLSVD removes too much signal
**Solution:** Reduce number of components, narrow frequency range, adjust threshold

## Best Practices

1. **Save original data** before preprocessing
2. **Use appropriate prior knowledge** for your field strength and sequence
3. **Start simple** (fewer metabolites) then add complexity
4. **Always inspect fits visually** - don't rely only on numbers
5. **Document all processing steps** and parameters
6. **Export processing history** for reproducibility
7. **Validate with phantoms** or known concentrations
8. **Report jMRUI version** and methods used in publications

## Performance Tips

1. **Memory:** Increase Java heap size for large MRSI datasets
   ```bash
   java -Xmx4g -jar jMRUI.jar
   ```

2. **Speed:** Use simpler baseline models for batch processing

3. **Storage:** Compress MRSI datasets, archive intermediate results

## Resources

- **Website:** http://www.jmrui.eu/
- **Documentation:** http://www.jmrui.eu/documentation/
- **Tutorials:** http://www.jmrui.eu/tutorials/
- **Forum:** http://www.jmrui.eu/forum/
- **Mailing List:** jmrui-users@lists.sourceforge.net
- **Publications:** http://www.jmrui.eu/publications/

## Citation

```bibtex
@article{naressi2001javabased,
  title={Java-based graphical user interface for the MRUI quantitation package},
  author={Naressi, Alberto and Couturier, Christophe and Devos, Jan Martin and Janssen, Michel and Mangeat, Claude and De Beer, Rudy and Graveron-Demilly, Danielle},
  journal={Magnetic Resonance Materials in Physics, Biology and Medicine},
  volume={12},
  number={2-3},
  pages={141--152},
  year={2001},
  publisher={Springer}
}

@article{stefan2009amares,
  title={Quantitation of magnetic resonance spectroscopy signals: the jMRUI software package},
  author={Stefan, Diana and Cesare, Francesca Di and Andrasescu, Adrian and Popa, Elena and Lazariev, Andriy and Vescovo, Emanuele and Strbak, Otto and Williams, Simon and Starcuk, Zenon and Cabanas, Maria and others},
  journal={Measurement Science and Technology},
  volume={20},
  number={10},
  pages={104035},
  year={2009},
  publisher={IOP Publishing}
}
```

## Related Tools

- **LCModel:** Commercial frequency-domain quantification
- **Osprey:** Comprehensive MRS processing suite
- **FID-A:** MATLAB toolkit for MRS
- **TARQUIN:** Automatic MRS quantification
- **Gannet:** GABA analysis toolkit
- **QUEST:** Quantum estimation (part of jMRUI)
- **AMARES:** Advanced method (part of jMRUI)
