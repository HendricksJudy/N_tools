# LCModel - Linear Combination Model for MRS Quantification

## Overview

LCModel (Linear Combination of Model spectra) is commercial software for automated, quantitative analysis of in vivo proton magnetic resonance spectroscopy (MRS) data. Developed by Stephen Provencher in the 1990s, LCModel revolutionized MRS analysis by providing fully automated metabolite quantification through linear combination fitting of individual metabolite basis spectra. The software analyzes frequency-domain spectra from single-voxel or chemical shift imaging (CSI) experiments, fitting the measured spectrum as a weighted sum of individual metabolite spectra while accounting for baseline, macromolecules, and lineshape variations.

LCModel provides absolute metabolite concentrations (mM or institutional units via water referencing), Cramér-Rao lower bounds (CRLB) for uncertainty estimation, and comprehensive diagnostic plots in PDF/PostScript format. It supports all major MRI vendors (Siemens, GE, Philips, Bruker), multiple field strengths (1.5T to 7T), and various pulse sequences (PRESS, STEAM, MEGA-PRESS). LCModel is considered the gold-standard reference method for single-voxel MRS quantification in clinical research, with thousands of publications using it to study brain tumors, epilepsy, neurodegenerative diseases, psychiatric disorders, and metabolic conditions.

**Official Website:** http://www.lcmodel.com
**Documentation:** http://s-provencher.com/pages/lcm-manual.shtml
**Licensing:** Commercial (academic and commercial licenses available)

### Key Features

- **Linear Combination Modeling:** Fits spectra as weighted sum of basis spectra
- **Automated Baseline Correction:** Adaptive spline baseline with regularization
- **Absolute Quantification:** Metabolite concentrations in mM (water referencing)
- **Cramér-Rao Lower Bounds (CRLB):** Uncertainty estimates for each metabolite
- **Multi-Vendor Support:** Siemens, GE, Philips, Bruker formats
- **Basis Set Library:** Pre-simulated basis sets for common protocols (1.5T-7T)
- **Water Referencing:** Eddy current and receiver gain correction
- **Quality Metrics:** SNR, linewidth (FWHM), fitting residuals
- **Batch Processing:** Analyze hundreds of spectra automatically
- **Comprehensive Reports:** PDF/PostScript with fitted spectra and diagnostics
- **Edited MRS:** MEGA-PRESS (GABA, GSH), HERMES support
- **Macromolecule Handling:** Built-in macromolecule and lipid models

### Applications

- **Clinical MRS:** Brain tumor characterization, epilepsy focus localization
- **Neurodegenerative Diseases:** Alzheimer's, Parkinson's, ALS metabolic profiling
- **Psychiatry:** Schizophrenia, depression glutamate/GABA quantification
- **Metabolic Disorders:** Hepatic encephalopathy, mitochondrial disease
- **Pharmaceutical Research:** MRS biomarkers for drug development
- **Neuroscience Research:** Neurotransmitter quantification (glutamate, GABA, glutamine)
- **Sports Medicine:** Lactate quantification, metabolic fatigue studies

### Citation

```bibtex
@article{Provencher1993LCModel,
  title={Estimation of metabolite concentrations from localized in vivo
         proton NMR spectra},
  author={Provencher, Stephen W},
  journal={Magnetic Resonance in Medicine},
  volume={30},
  number={6},
  pages={672--679},
  year={1993},
  publisher={Wiley}
}

@article{Provencher2001LCModel,
  title={Automatic quantitation of localized in vivo 1H spectra with LCModel},
  author={Provencher, Stephen W},
  journal={NMR in Biomedicine},
  volume={14},
  number={4},
  pages={260--264},
  year={2001},
  publisher={Wiley}
}
```

---

## Installation and Licensing

### Obtaining License

LCModel requires a commercial license:

```bash
# Academic License:
# - Contact: http://www.lcmodel.com
# - Cost: ~$400 USD (perpetual, single user)
# - Free for developing countries (contact Dr. Provencher)

# Commercial License:
# - For pharmaceutical/industry use
# - Contact Dr. Provencher for pricing

# Request License:
# Email: stephen@s-provencher.com
# Provide:
# - Name and affiliation
# - Intended use (academic/commercial)
# - Operating system (Linux/macOS/Windows)

# License includes:
# - LCModel software (all platforms)
# - Basis set library (1.5T, 3T, 7T)
# - User manual
# - Email support
# - Free updates for 1 year
```

### Installation on Linux

```bash
# Download LCModel package (after receiving license)
# File: lcmodel-linux.tar

# Extract to /opt or home directory
sudo mkdir -p /opt/lcmodel
sudo tar -xvf lcmodel-linux.tar -C /opt/lcmodel

# Install license key
# Copy license file to LCModel directory
sudo cp ~/license.dat /opt/lcmodel/.lcmodel/license

# Set environment variables in ~/.bashrc
echo 'export LCMODEL=/opt/lcmodel' >> ~/.bashrc
echo 'export PATH=$LCMODEL/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Test installation
lcmodel --version
# Output: LCModel Version 6.3-1L
```

### Installation on macOS

```bash
# Download macOS package: lcmodel-mac.dmg

# Mount DMG and copy to Applications
open lcmodel-mac.dmg
cp -R /Volumes/LCModel/LCModel.app /Applications/

# Install license
mkdir -p ~/.lcmodel
cp ~/license.dat ~/.lcmodel/license

# Command-line access (optional)
echo 'export PATH="/Applications/LCModel.app/Contents/Resources:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Launch GUI
open /Applications/LCModel.app
```

### Installation on Windows

```bash
# Download Windows installer: lcmodel-setup.exe

# Run installer (Administrator rights)
# Default install location: C:\LCModel

# Install license
# Copy license.dat to C:\LCModel\.lcmodel\

# Add to PATH (System Properties > Environment Variables)
# Add C:\LCModel\bin to PATH

# Test in Command Prompt
lcmodel --version
```

### Testing Installation

```bash
# Test with example data (included with LCModel)
cd $LCMODEL/example

# Run LCModel on example spectrum
lcmodel < press_te30.CONTROL

# Check output
ls press_te30.PS  # PostScript output
ls press_te30.PDF # PDF output (if ps2pdf available)

# If successful, LCModel is ready for use
```

---

## Data Preparation

### Exporting Spectra from Scanner

Different scanner vendors use different raw data formats:

```bash
# Siemens (RDA format - recommended):
# On scanner console:
# - Open spectrum in spectroscopy evaluation
# - File > Export > Raw Data Amplitude (RDA)
# Output: spectrum.rda (text file with header)

# GE (P-file format):
# Access P-file from /usr/g/mrraw/ on scanner
# Copy P000000.7 file
# Convert with GE tools or use Gannet/Osprey

# Philips (SPAR/SDAT format):
# Export both files:
# - spectrum.SPAR (parameters)
# - spectrum.SDAT (data)

# Bruker (jcamp-dx format):
# Export from Bruker TopSpin software
# Includes acquisition and processing parameters
```

### Format Conversion to LCModel

```bash
# LCModel native format: .RAW (binary)
# Most users convert from vendor formats using:

# Option 1: spec2nii (Python tool)
pip install spec2nii

# Convert Siemens RDA to LCModel RAW
spec2nii lcmodel -f spectrum.rda -o output_dir/

# Output: spectrum.RAW (data) + spectrum.h2o (water reference)

# Option 2: Osprey (MATLAB toolbox)
# Osprey includes LCModel export functionality
# See Osprey documentation for workflow

# Option 3: Manual conversion (advanced users)
# Write custom script to create .RAW file
# Format specification in LCModel manual
```

### Water Reference Acquisition

```bash
# Water reference for absolute quantification:
# - Same voxel position as metabolite spectrum
# - No water suppression
# - Usually 2-4 averages

# On Siemens scanner:
# 1. Acquire metabolite spectrum (128-256 averages)
# 2. Turn off water suppression
# 3. Acquire water reference (2-4 averages)
# 4. Export both as RDA files

# File naming convention:
# - metabolite.rda (suppressed water)
# - metabolite_w.rda (water reference)
```

### Quality Checks Before Quantification

```bash
# Inspect raw spectra before LCModel analysis

# Check for:
# 1. Adequate SNR (>10 for NAA peak)
# 2. Minimal lipid contamination
# 3. Proper water suppression (residual water < 10x NAA)
# 4. No frequency drift
# 5. Proper phasing

# Visual inspection (FSL-MRS or TARQUIN):
fsleyes metabolite.nii.gz

# Or convert to NIfTI with spec2nii:
spec2nii nifti -f metabolite.rda -o metabolite.nii.gz
```

---

## Basis Sets

### What Are Basis Sets?

```text
Basis sets are libraries of simulated individual metabolite spectra used by
LCModel to fit measured in vivo spectra. Each basis spectrum represents one
metabolite (e.g., NAA, Cr, Cho) acquired under specific conditions:

- Field strength (1.5T, 3T, 7T)
- Echo time (TE = 30ms, 144ms, etc.)
- Pulse sequence (PRESS, STEAM, MEGA-PRESS)

LCModel fits the measured spectrum as:
  Measured = c1*NAA + c2*Cr + c3*Cho + ... + Baseline + Noise

Where c1, c2, c3 are concentrations to be determined.
```

### Selecting Appropriate Basis Set

```bash
# Basis set selection criteria:
# 1. Field strength must match (1.5T, 3T, 7T)
# 2. Echo time (TE) must match (±5ms acceptable)
# 3. Pulse sequence must match (PRESS vs. STEAM)

# Example: 3T brain MRS with PRESS TE=30ms
# Use basis set: press_te30_3t.BASIS

# Common basis sets (included with LCModel):
# - press_te30_3t.BASIS (3T PRESS TE=30ms, most common)
# - press_te144_3t.BASIS (3T PRESS TE=144ms, long TE)
# - steam_te20_3t.BASIS (3T STEAM TE=20ms, short TE)
# - mega_press_3t.BASIS (3T MEGA-PRESS for GABA editing)
# - press_te30_7t.BASIS (7T PRESS TE=30ms)

# Basis set location:
$LCMODEL/basis/
```

### Standard Basis Sets

```bash
# View metabolites included in basis set
strings $LCMODEL/basis/press_te30_3t.BASIS | grep METABO

# Typical metabolites in short TE (30ms) basis:
# - NAA (N-acetylaspartate)
# - Cr (creatine)
# - PCr (phosphocreatine)
# - Cho (choline)
# - mI (myo-inositol)
# - Glu (glutamate)
# - Gln (glutamine)
# - GABA (gamma-aminobutyric acid)
# - GSH (glutathione)
# - Tau (taurine)
# - Lac (lactate)
# - Ala (alanine)
# - Asp (aspartate)
# - MM (macromolecules)
# - Lip (lipids)

# Long TE (144ms) typically includes fewer metabolites:
# - NAA, Cr, Cho, Lac (singlets that persist at long TE)
```

### Custom Basis Set Generation

```bash
# Advanced: Generate custom basis set using FID-A or Osprey

# Requirements:
# - Pulse sequence timing parameters
# - B0 field strength
# - Individual metabolite structures

# Using FID-A (MATLAB):
# 1. Define sequence parameters
seq.te = 30;  % ms
seq.fieldStrength = 3;  % Tesla
seq.sequence = 'PRESS';

# 2. Simulate basis set
basis = sim_press(seq);

# 3. Export to LCModel format
io_writelcm(basis, 'custom_basis.BASIS');

# Note: Most users should use standard basis sets
# Custom basis sets require expert knowledge
```

---

## Control File Configuration

The control file (.CONTROL) specifies all parameters for LCModel analysis.

### Basic Control File

```bash
# Create file: spectrum.CONTROL

 $LCMODL
 TITLE='Subject 001 - Posterior Cingulate Cortex'
 KEY=210387309
 OWNER='University MRS Lab'
 DELTAT=0.0005
 NUNFIL=2048
 FILBAS='/opt/lcmodel/basis/press_te30_3t.BASIS'
 FILRAW='spectrum.RAW'
 FILPS='spectrum.PS'
 LTABLE=7
 LCSV=11
 LCOORD=9
 $END
```

### Parameter Descriptions

```text
Essential Parameters:

FILBAS  = Basis set file path
FILRAW  = Raw data file (spectrum to analyze)
FILPS   = PostScript output file
TITLE   = Descriptive title for output
KEY     = License key (provided with license)
OWNER   = License owner name
DELTAT  = Dwell time in seconds (1 / spectral width in Hz)
NUNFIL  = Number of data points (must be power of 2)

Output Options:

LTABLE  = Output tables (7 = full output)
LCSV    = CSV output for metabolite concentrations
LCOORD  = Output coordinates and dimensions
LPRINT  = Print level (6 = detailed diagnostic output)

Water Referencing:

FILH2O  = Water reference file
DOWS    = Include water scaling (T = true)
ATTH2O  = Assumed water attenuation factor
WCONC   = Water concentration (mM, default 35880 for brain)

Frequency/Phase:

SDDEGZ  = Starting phase (degrees, auto if omitted)
SDDEGP  = Phase increment
SHIFMN  = Minimum frequency shift (ppm)
SHIFMX  = Maximum frequency shift (ppm)
```

### Control File with Water Referencing

```bash
# spectrum_water.CONTROL

 $LCMODL
 TITLE='Subject 001 - PCC with Water Referencing'
 KEY=210387309
 OWNER='University MRS Lab'
 DELTAT=0.0005
 NUNFIL=2048
 FILBAS='/opt/lcmodel/basis/press_te30_3t.BASIS'
 FILRAW='spectrum.RAW'
 FILH2O='spectrum_water.RAW'
 FILPS='spectrum.PS'
 DOWS=T
 ATTH2O=1.0
 WCONC=35880
 LTABLE=7
 LCSV=11
 LCOORD=9
 $END

# Parameters:
# FILH2O: Water reference spectrum (unsuppressed)
# DOWS: Enable water scaling
# ATTH2O: Water attenuation (T1/T2 relaxation, typically 0.7-1.0)
# WCONC: Water concentration in voxel (mM)
#        Gray matter ~43000 mM
#        White matter ~36000 mM
#        Mixed tissue ~35880 mM (default)
```

### Batch Processing Control File

```bash
# batch_template.CONTROL (template for multiple subjects)

 $LCMODL
 TITLE='BATCH_TITLE'
 KEY=210387309
 OWNER='University MRS Lab'
 DELTAT=0.0005
 NUNFIL=2048
 FILBAS='/opt/lcmodel/basis/press_te30_3t.BASIS'
 FILRAW='BATCH_RAW'
 FILH2O='BATCH_H2O'
 FILPS='BATCH_PS'
 DOWS=T
 LTABLE=7
 LCSV=11
 $END

# Use with batch script (see Running LCModel section)
```

---

## Running LCModel

### Command-Line Execution

```bash
# Basic usage: redirect control file to lcmodel
lcmodel < spectrum.CONTROL

# Output files created:
# - spectrum.PS (PostScript with plots)
# - spectrum.CSV (metabolite concentrations)
# - spectrum.COORD (voxel coordinates)
# - spectrum.PRINT (detailed fitting log)

# Convert PostScript to PDF (optional):
ps2pdf spectrum.PS spectrum.PDF
```

### GUI Interface (LCMgui)

```bash
# Launch LCModel GUI (if available)
lcmgui

# GUI workflow:
# 1. File > New Analysis
# 2. Select basis set (FILBAS)
# 3. Select raw data (FILRAW)
# 4. Optional: Select water reference (FILH2O)
# 5. Set output file (FILPS)
# 6. Enter KEY and OWNER
# 7. Run > Start Analysis
# 8. View results in PDF viewer

# GUI generates control file automatically
```

### Batch Processing Multiple Subjects

```bash
# Batch script: process_all.sh

#!/bin/bash

# Subject list
subjects=(sub-001 sub-002 sub-003 sub-004 sub-005)

# Template control file
template="batch_template.CONTROL"

for sub in "${subjects[@]}"; do
    echo "Processing $sub..."

    # Create subject-specific control file
    sed -e "s|BATCH_TITLE|$sub PCC|g" \
        -e "s|BATCH_RAW|${sub}_pcc.RAW|g" \
        -e "s|BATCH_H2O|${sub}_pcc_water.RAW|g" \
        -e "s|BATCH_PS|${sub}_pcc.PS|g" \
        $template > ${sub}.CONTROL

    # Run LCModel
    lcmodel < ${sub}.CONTROL

    # Convert to PDF
    ps2pdf ${sub}_pcc.PS ${sub}_pcc.PDF

    echo "$sub complete"
done

echo "All subjects processed"
```

### Monitoring Progress

```bash
# LCModel outputs progress to STDOUT during analysis

# Typical output:
# LCModel 6.3-1L
# Reading control file...
# Reading basis set...
# Reading raw data...
# Fitting spectrum...
# Iteration 1: RMS = 0.0123
# Iteration 2: RMS = 0.0098
# Converged after 15 iterations
# Writing output files...
# Done

# Check for errors:
grep -i error *.PRINT

# Check for warnings:
grep -i warning *.PRINT
```

---

## Interpreting Results

### PDF/PostScript Output

The LCModel output PDF contains multiple pages:

```text
Page 1: Fitted Spectrum
- Blue line: Measured spectrum
- Red line: LCModel fit
- Green line: Baseline
- Gray line: Residuals (measured - fit)
- Individual metabolite contributions shown

Page 2: Concentration Table
- Metabolite names (NAA, Cr, Cho, etc.)
- Concentrations (mM or institutional units)
- SD% (Cramér-Rao Lower Bounds)

Page 3: Diagnostics
- SNR (Signal-to-Noise Ratio)
- FWHM (Full Width at Half Maximum, Hz)
- Data shift (Hz)
- Phase corrections

Pages 4+: Individual Metabolite Fits
- Each metabolite's contribution to total fit
```

### Metabolite Concentrations and CRLB

```bash
# Extract concentrations from CSV output

# spectrum.CSV contains:
# Metabolite, Conc, %SD
# NAA, 12.5, 3
# Cr, 8.2, 4
# Cho, 2.1, 5
# ...

# Cramér-Rao Lower Bounds (%SD):
# - Theoretical minimum uncertainty
# - Lower %SD = more reliable estimate
# - %SD < 20%: Good quality (publishable)
# - %SD 20-50%: Moderate quality (interpret cautiously)
# - %SD > 50%: Poor quality (exclude from analysis)

# Common metabolites and typical CRLB:
# NAA: 2-5% (strong signal, easy to quantify)
# Cr: 3-6%
# Cho: 4-8%
# mI: 5-10%
# Glu: 8-15% (complex multiplet)
# GABA: 10-20% (low concentration)
# GSH: 15-30% (low concentration, overlapping peaks)
```

### Quality Metrics

```bash
# SNR (Signal-to-Noise Ratio):
# - NAA peak height / RMS noise
# - Typical values: 10-50 for single-voxel MRS
# - SNR > 10: Acceptable
# - SNR > 20: Good
# - SNR > 30: Excellent

# FWHM (Linewidth):
# - Full width at half maximum of spectral peaks (Hz)
# - Reflects B0 homogeneity
# - Typical values at 3T:
#   * FWHM < 0.05 ppm (~6 Hz): Excellent
#   * FWHM 0.05-0.10 ppm (~6-13 Hz): Good
#   * FWHM > 0.10 ppm (>13 Hz): Poor (check shimming)

# Residuals:
# - Difference between measured and fitted spectrum
# - Should be noise-like (no systematic patterns)
# - Large residuals indicate poor fit (artifact, missing metabolites)
```

### Assessing Fit Quality

```bash
# Visual inspection of PDF output:

# Good fit:
# - Residuals (gray line) are noise-like
# - All major peaks fitted well
# - CRLB < 20% for metabolites of interest
# - Baseline smooth and reasonable

# Poor fit:
# - Large systematic residuals
# - CRLB > 30% for multiple metabolites
# - Baseline unrealistic (extreme curvature)
# - Lipid contamination (large peaks at 0.9, 1.3 ppm)

# Automatic quality check script:
# check_quality.py

import pandas as pd
import sys

csv_file = sys.argv[1]
df = pd.read_csv(csv_file, sep=r'\s+', header=None, names=['Metab', 'Conc', 'SD'])

# Filter metabolites with SD < 20%
good_metabs = df[df['SD'] < 20]
print(f"Metabolites with SD < 20%: {len(good_metabs)}/{len(df)}")
print(good_metabs)

# Check key metabolites
for metab in ['NAA', 'Cr', 'Cho']:
    row = df[df['Metab'] == metab]
    if not row.empty:
        sd = row['SD'].values[0]
        print(f"{metab}: SD = {sd}% {'(PASS)' if sd < 20 else '(FAIL)'}")
```

### Extracting Concentrations to CSV

```bash
# Parse LCModel CSV output for statistical analysis

# Python script: extract_results.py

import pandas as pd
import glob

# Collect all subject CSV files
csv_files = glob.glob('sub-*_pcc.CSV')

results = []
for csv_file in csv_files:
    subject = csv_file.split('_')[0]  # Extract subject ID

    # Read LCModel CSV (space-separated, no header)
    df = pd.read_csv(csv_file, sep=r'\s+', header=None,
                      names=['Metabolite', 'Concentration', 'SD'])

    # Convert to wide format
    row = {'Subject': subject}
    for _, r in df.iterrows():
        metab = r['Metabolite']
        row[f'{metab}_conc'] = r['Concentration']
        row[f'{metab}_SD'] = r['SD']

    results.append(row)

# Create master dataframe
master_df = pd.DataFrame(results)
master_df.to_csv('all_subjects_metabolites.csv', index=False)
print(f"Extracted results for {len(results)} subjects")
```

---

## Quality Control

### CRLB Thresholds

```bash
# Standard quality criteria for metabolite inclusion

# Conservative (strict QC):
# - Include only metabolites with CRLB < 10%
# - Appropriate for critical metabolites (NAA, Cr, Cho)

# Standard (typical QC):
# - Include metabolites with CRLB < 20%
# - Most common threshold in literature

# Lenient (exploratory):
# - Include metabolites with CRLB < 30%
# - Use for preliminary analysis, report with caution

# Apply QC in Python:
df_qc = df[df['SD'] < 20]  # Keep only CRLB < 20%
```

### Linewidth (FWHM) Criteria

```bash
# Extract FWHM from LCModel PRINT file

grep "FWHM" spectrum.PRINT
# Output: FWHM = 0.055 ppm = 7.0 Hz

# Quality thresholds (3T):
# FWHM < 0.05 ppm (~6 Hz): Excellent
# FWHM 0.05-0.08 ppm (~6-10 Hz): Good
# FWHM 0.08-0.12 ppm (~10-15 Hz): Acceptable
# FWHM > 0.12 ppm (>15 Hz): Poor (exclude or re-shim)

# Automated QC script with FWHM check
awk '/FWHM/ {if ($4 > 0.12) print "WARNING: FWHM too large"}' spectrum.PRINT
```

### SNR Requirements

```bash
# Extract SNR from PRINT file
grep "S/N" spectrum.PRINT
# Output: S/N = 25

# Typical SNR requirements:
# SNR > 5: Minimum for qualitative analysis
# SNR > 10: Acceptable for quantification
# SNR > 20: Good quality
# SNR > 30: Excellent quality

# Automated SNR check
awk '/S\/N/ {if ($3 < 10) print "WARNING: Low SNR"}' spectrum.PRINT
```

### Automated QC Pipeline

```bash
#!/bin/bash
# qc_pipeline.sh: Automated quality control for LCModel outputs

subject=$1
print_file="${subject}.PRINT"
csv_file="${subject}.CSV"

echo "Quality Control for $subject"
echo "================================"

# Extract FWHM
fwhm=$(grep "FWHM" $print_file | awk '{print $4}')
echo "FWHM: $fwhm ppm"
if (( $(echo "$fwhm > 0.12" | bc -l) )); then
    echo "  WARNING: FWHM exceeds threshold"
fi

# Extract SNR
snr=$(grep "S/N" $print_file | awk '{print $3}')
echo "SNR: $snr"
if (( $(echo "$snr < 10" | bc -l) )); then
    echo "  WARNING: SNR below threshold"
fi

# Check CRLB for key metabolites
echo "Metabolite CRLB:"
awk '/NAA/ || /Cr/ || /Cho/ {printf "  %s: %s%%\n", $1, $3}' $csv_file

echo "================================"
```

---

## Advanced Features

### Edited MRS (MEGA-PRESS for GABA)

```bash
# MEGA-PRESS: Editing sequence for GABA quantification

# Requires two acquisitions:
# 1. ON: Editing pulse at 1.9 ppm (GABA C3)
# 2. OFF: No editing pulse

# Scanner exports DIFF spectrum (ON - OFF)

# LCModel control file for MEGA-PRESS:
 $LCMODL
 TITLE='GABA-edited MEGA-PRESS'
 KEY=210387309
 OWNER='University MRS Lab'
 DELTAT=0.0005
 NUNFIL=2048
 FILBAS='/opt/lcmodel/basis/mega_press_3t.BASIS'
 FILRAW='diff_spectrum.RAW'
 FILPS='diff_spectrum.PS'
 LTABLE=7
 $END

# Typical GABA CRLB: 10-20% (acceptable)
# Co-edited: Glutathione (GSH) at 2.95 ppm
```

### Macromolecule Handling

```bash
# Macromolecules (MM): Broad baseline components from proteins/lipids

# LCModel includes MM in basis set by default

# Options for MM handling:

# 1. Use built-in MM (standard)
# Default behavior, MM included in .BASIS file

# 2. Measured MM spectrum (advanced)
# Acquire inversion recovery spectrum (TI = 0.7s)
# Adds FILM2O parameter to control file:
FILMM='measured_mm.RAW'

# 3. Exclude MM (not recommended)
# Only for special cases (e.g., long TE where MM negligible)
# Requires custom basis set without MM
```

### Lipid Contamination Correction

```bash
# Lipid signals from skull/subcutaneous fat can contaminate brain spectra

# Check for lipid contamination in PDF output:
# - Large peaks at 0.9 ppm (methyl)
# - Large peaks at 1.3 ppm (methylene)

# LCModel handles lipids via basis set (Lip09, Lip13, etc.)

# If severe lipid contamination:
# 1. Re-position voxel away from skull
# 2. Use PRESS instead of STEAM (better localization)
# 3. Apply outer volume suppression (OVS) bands
# 4. Use lipid regularization in LCModel:
NSIMUL=13  # Increase lipid regularization

# Exclude spectra with dominant lipid peaks (>50% of NAA)
```

---

## Integration with MRS Pipelines

### Osprey Workflow Integration

```bash
# Osprey: Comprehensive MRS processing toolbox (MATLAB)
# Includes LCModel as quantification backend

# Osprey workflow:
# 1. Load raw data (all vendor formats)
# 2. Preprocess (frequency/phase correction, averaging)
# 3. Quantify with LCModel
# 4. Visualize and QC

# Osprey job file (jobFile.m):
LCMcontrol.autoPP = 1;  % Automatic phasing
LCMcontrol.FILBAS = 'press_te30_3t.BASIS';
LCMcontrol.owner = 'University MRS Lab';
LCMcontrol.key = 210387309;

# Run Osprey with LCModel:
runOspreyJob(jobFile);

# Osprey calls LCModel and parses results automatically
```

### TARQUIN Comparison

```bash
# TARQUIN: Alternative open-source quantification tool

# Compare LCModel vs. TARQUIN on same spectrum:

# LCModel:
lcmodel < spectrum.CONTROL

# TARQUIN:
tarquin --input spectrum.rda \
        --format rda \
        --output_csv tarquin_results.csv

# Compare concentrations:
# LCModel typically provides tighter CRLB (more robust fitting)
# TARQUIN is open-source and free (good for exploratory work)
```

### Gannet (GABA Editing) Compatibility

```bash
# Gannet: MATLAB toolbox for GABA-edited MRS (MEGA-PRESS)

# Gannet can export difference spectra to LCModel

# Gannet workflow:
# 1. Load MEGA-PRESS data (ON/OFF)
# 2. Preprocess (alignment, averaging)
# 3. Export DIFF spectrum to .RAW format
# 4. Process with LCModel using MEGA-PRESS basis set

# Gannet export to LCModel:
GannetLCMExport(MRS_struct);

# Then run LCModel on exported .RAW file
```

---

## Troubleshooting

### Common Fitting Errors

**Error: "Concentration of X is negative"**

**Cause:** Poor fitting, wrong basis set, or metabolite not present

**Solution:**
```bash
# Check basis set matches acquisition parameters
# Inspect fit visually in PDF
# If metabolite truly absent, exclude from CHOMIT list
```

**Error: "FSUM > 0.15" (poor fit quality)**

**Cause:** Bad data quality, wrong basis set, or artifact

**Solution:**
```bash
# Check SNR and linewidth
# Verify basis set correct for field strength and TE
# Inspect for lipid contamination or artifacts
# Try adjusting DKNTMN parameter (baseline stiffness)
```

### Poor Baseline Fits

**Problem:** Baseline extremely curved or unrealistic

**Solution:**
```bash
# Adjust baseline stiffness:
DKNTMN=0.15  # Default is 0.15
# Increase to 0.25 for stiffer baseline
# Decrease to 0.10 for more flexible baseline

# Add to control file:
 $LCMODL
 ...
 DKNTMN=0.20
 $END
```

### Frequency Drift Issues

**Problem:** Spectrum shifted in frequency, poor fit

**Solution:**
```bash
# Allow LCModel to search for optimal frequency:
SHIFMN=-0.5  # Minimum shift (ppm)
SHIFMX=0.5   # Maximum shift (ppm)

# Or manually set shift:
PPMST=4.65  # Chemical shift reference (NAA at 2.01 ppm)
```

### License Problems

**Error: "License file not found" or "Invalid license"**

**Solution:**
```bash
# Check license file location
ls ~/.lcmodel/license  # macOS/Linux
ls C:\LCModel\.lcmodel\license  # Windows

# Verify KEY and OWNER in control file match license
grep -i key ~/.lcmodel/license
grep -i owner ~/.lcmodel/license

# Contact Dr. Provencher if license expired or invalid
```

---

## Best Practices

### Acquisition Protocol Recommendations

- **Short TE (20-35ms):** Maximize metabolite yield, detect Glu/Gln, suitable for most applications
- **Long TE (135-144ms):** Simplified spectra (NAA, Cr, Cho, Lac), less lipid contamination
- **Voxel Size:** 8-27 cm³ (larger = better SNR, less spatial specificity)
- **Averages:** 128-256 for single-voxel (adjust for SNR ~20-30)
- **Water Suppression:** VAPOR or CHESS (residual water < 10× NAA)
- **Shimming:** Automatic + manual adjustment (target FWHM < 10 Hz at 3T)
- **Outer Volume Suppression (OVS):** Reduce lipid contamination

### Basis Set Selection

- **Match acquisition parameters exactly:** Field strength, TE (±5ms acceptable), sequence type
- **Use validated basis sets:** Included LCModel basis sets or published custom sets
- **Document basis set source:** Essential for reproducibility

### Quality Control Criteria

- **SNR ≥ 10:** Minimum for quantitative analysis
- **FWHM < 0.12 ppm:** Indicates adequate B0 shimming
- **CRLB < 20%:** Standard threshold for metabolite inclusion
- **Visual inspection:** Always check fitted spectrum for artifacts

### Reporting Standards

When reporting LCModel results, include:

1. **LCModel version:** e.g., "LCModel Version 6.3-1L"
2. **Basis set:** e.g., "press_te30_3t.BASIS"
3. **Quality criteria:** e.g., "Metabolites with CRLB > 20% excluded"
4. **Concentrations:** Report mean ± SD, with units (mM or institutional units)
5. **Water referencing:** Specify ATTH2O and WCONC if used
6. **Cite LCModel:** Provencher 1993 and 2001 papers

---

## References

1. **LCModel Methodology:**
   - Provencher (1993). Estimation of metabolite concentrations from localized in vivo proton NMR spectra. *Magn Reson Med*, 30(6):672-679.
   - Provencher (2001). Automatic quantitation of localized in vivo 1H spectra with LCModel. *NMR Biomed*, 14(4):260-264.

2. **MRS Reviews:**
   - Oz et al. (2014). Clinical proton MR spectroscopy in central nervous system disorders. *Radiology*, 270(3):658-679.
   - Near et al. (2020). Preprocessing, analysis and quantification in single-voxel magnetic resonance spectroscopy. *NMR Biomed*, e4257.

3. **Quality Control:**
   - Wilson et al. (2019). Methodological consensus on clinical proton MRS of the brain. *Magn Reson Med*, 82(2):527-550.

4. **GABA Editing:**
   - Mullins et al. (2014). Current practice in the use of MEGA-PRESS spectroscopy for the detection of GABA. *NeuroImage*, 86:43-52.

5. **Integration:**
   - Oeltzschner et al. (2020). Osprey: An open-source, end-to-end, versatile, and fully automated spectroscopy pipeline. *Magn Reson Med*, 83(2):433-447.

**Official Resources:**
- LCModel Website: http://www.lcmodel.com
- User Manual: http://s-provencher.com/pages/lcm-manual.shtml
- MRS Consensus Paper: https://doi.org/10.1002/mrm.27742
