# TARQUIN

## Overview

TARQUIN (Totally Automatic Robust Quantitation in NMR) is a fast, fully-automated tool for MRS quantification. Written in C++, TARQUIN provides sophisticated analysis of magnetic resonance spectroscopy data with minimal user input, making it ideal for large-scale studies and clinical applications. It uses advanced algorithms for baseline estimation, lineshape modeling, and metabolite quantification.

**Website:** https://tarquin.sourceforge.net/
**Platform:** Linux/macOS/Windows
**Language:** C++
**License:** GPLv3

## Key Features

- Fully automatic quantification
- No user interaction required
- Fast processing (seconds per spectrum)
- Robust to noise and artifacts
- Automatic water referencing
- Tissue composition correction
- Multiple file format support
- Built-in basis sets for common sequences
- Quality control metrics
- CSV and text output formats
- Command-line and GUI interfaces
- Cross-platform compatibility

## Installation

### Linux (Ubuntu/Debian)

```bash
# Add repository
sudo add-apt-repository ppa:martin-wilson/tarquin
sudo apt-get update

# Install TARQUIN
sudo apt-get install tarquin

# Verify installation
tarquin --version
```

### macOS

```bash
# Download from SourceForge
# https://sourceforge.net/projects/tarquin/files/

# Extract and install
tar -xzf tarquin-*.tar.gz
cd tarquin-*
sudo cp tarquin /usr/local/bin/

# Verify
tarquin --version
```

### Windows

```bash
# Download Windows installer from SourceForge
# https://sourceforge.net/projects/tarquin/files/

# Run installer
# TARQUIN will be added to PATH automatically

# Verify in command prompt
tarquin --version
```

### Build from Source

```bash
# Install dependencies
sudo apt-get install cmake g++ libboost-all-dev libfftw3-dev

# Clone repository
git clone https://git.code.sf.net/p/tarquin/code tarquin-code
cd tarquin-code

# Build
mkdir build && cd build
cmake ..
make
sudo make install

# Test
tarquin --version
```

## Supported File Formats

TARQUIN can read:

- **Siemens:** .rda, .dat (TWIX)
- **Philips:** .sdat/.spar
- **GE:** .7 (P-files)
- **Varian/Agilent:** .fid
- **Bruker:** ser/fid
- **jMRUI:** .txt, .mrui
- **LCModel:** .raw
- **NIfTI-MRS:** .nii/.nii.gz
- **Generic text:** .txt, .csv

## Basic Usage

### Simple Quantification

```bash
# Most basic usage - metabolite spectrum only
tarquin --input subject01_metab.rda \
        --format rda \
        --output_txt subject01_results.txt \
        --output_csv subject01_results.csv

# With water reference for absolute quantification
tarquin --input subject01_metab.rda \
        --input_w subject01_water.rda \
        --format rda \
        --output_txt subject01_results.txt

# Results include:
# - Metabolite concentrations
# - Quality metrics (SNR, linewidth, FWHM)
# - Fitting residuals
# - Reference information
```

### Specify Sequence Parameters

```bash
# Provide echo time if not in header
tarquin --input subject01.rda \
        --echo 30 \
        --output_txt results.txt

# Specify field strength
tarquin --input subject01.rda \
        --field_strength 3 \
        --echo 30 \
        --output_txt results.txt
```

### Choose Basis Set

```bash
# TARQUIN includes built-in basis sets for:
# - PRESS at 1.5T and 3T (TE = 30, 35, 144ms)
# - MEGA-PRESS for GABA

# Specify basis set
tarquin --input subject01.rda \
        --basis_3t_press_te30 \
        --output_txt results.txt

# Or provide custom basis
tarquin --input subject01.rda \
        --basis_lcm my_basis.BASIS \
        --output_txt results.txt
```

## Water Referencing

### Absolute Quantification

```bash
# Use water reference for absolute concentrations (mM)
tarquin --input metab.rda \
        --input_w water.rda \
        --format rda \
        --output_txt results.txt

# TARQUIN automatically:
# - Scales metabolite signal to water
# - Accounts for water concentration (55.5 M)
# - Applies tissue composition corrections
# - Reports institutional units (i.u.) or mM
```

### Water Suppression Factor

```bash
# If water unsuppressed scan unavailable,
# estimate from residual water in metabolite spectrum

tarquin --input metab.rda \
        --auto_ref \
        --output_txt results.txt

# Or specify water suppression factor manually
tarquin --input metab.rda \
        --w_att 1000 \
        --output_txt results.txt
```

## Tissue Correction

### With Segmentation Data

```bash
# Provide tissue segmentation for correction
tarquin --input metab.rda \
        --input_w water.rda \
        --input_seg tissue_fractions.csv \
        --output_txt results.txt

# tissue_fractions.csv format:
# GM,WM,CSF
# 0.60,0.30,0.10

# TARQUIN applies:
# - Tissue-specific water content
# - Relaxation corrections
# - CSF partial volume correction
```

### Automatic Tissue Estimation

```bash
# TARQUIN can estimate tissue composition
# from water signal if no segmentation provided
tarquin --input metab.rda \
        --input_w water.rda \
        --auto_tissue \
        --output_txt results.txt
```

## Advanced Options

### Quality Control Filtering

```bash
# Set quality thresholds
tarquin --input metab.rda \
        --min_snr 10 \
        --max_fwhm 0.15 \
        --output_txt results.txt

# Skip processing if QC fails
# Useful for batch processing
```

### Baseline Modeling

```bash
# Control baseline flexibility
tarquin --input metab.rda \
        --baseline_order 2 \
        --output_txt results.txt

# Options:
# --baseline_order N : Polynomial order (0-4)
# --no_baseline      : Disable baseline
# --dkntmn N        : Dynamic knot minimum
```

### Metabolite Selection

```bash
# Fit only specific metabolites
tarquin --input metab.rda \
        --metab NAA,Cr,Cho,Glu,Glx \
        --output_txt results.txt

# Useful for:
# - Reducing fitting time
# - Avoiding overfitting
# - Focus on metabolites of interest
```

### Frequency Range

```bash
# Limit fitting range (ppm)
tarquin --input metab.rda \
        --ppm_start 0.2 \
        --ppm_end 4.2 \
        --output_txt results.txt

# Exclude problematic regions
# Improve fitting stability
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# Batch processing script

# Input directory
INPUT_DIR="/data/subjects"
OUTPUT_DIR="/results"

# Find all metabolite files
for metab_file in ${INPUT_DIR}/sub-*/mrs/*_metab.rda; do
    # Extract subject ID
    subj=$(basename $(dirname $(dirname ${metab_file})))

    echo "Processing ${subj}..."

    # Find corresponding water file
    water_file="${metab_file/_metab/_water}"

    # Output files
    out_txt="${OUTPUT_DIR}/${subj}_results.txt"
    out_csv="${OUTPUT_DIR}/${subj}_results.csv"
    out_fit="${OUTPUT_DIR}/${subj}_fit.txt"

    # Run TARQUIN
    tarquin --input ${metab_file} \
            --input_w ${water_file} \
            --format rda \
            --output_txt ${out_txt} \
            --output_csv ${out_csv} \
            --output_fit ${out_fit} \
            --echo 30 \
            --basis_3t_press_te30

    echo "  Completed ${subj}"
done

echo "Batch processing complete!"
```

### Combine Results

```bash
# Combine all CSV results into single file

# Create header from first file
head -n 1 /results/sub-01_results.csv > /results/all_subjects.csv

# Append data from all files
for csv_file in /results/sub-*_results.csv; do
    tail -n +2 ${csv_file} >> /results/all_subjects.csv
done

echo "Combined results saved to all_subjects.csv"
```

## Output Formats

### Text Output

```bash
# Default text output includes:
# - Metabolite concentrations (i.u. or mM)
# - Standard deviations (Cramér-Rao lower bounds)
# - Quality metrics
# - Fit parameters

tarquin --input metab.rda \
        --output_txt results.txt

# Example output:
# NAA: 10.5 +/- 0.8 mM (CRLB: 3%)
# Cr: 8.2 +/- 0.6 mM (CRLB: 4%)
# Cho: 2.1 +/- 0.3 mM (CRLB: 7%)
# SNR: 25.3
# FWHM: 0.08 ppm (10.2 Hz @ 3T)
```

### CSV Output

```bash
# Structured CSV for statistical analysis
tarquin --input metab.rda \
        --output_csv results.csv

# Columns include:
# - Metabolite names
# - Concentrations
# - CRLBs (%)
# - Quality metrics
```

### Fit Details

```bash
# Save detailed fitting results
tarquin --input metab.rda \
        --output_fit fit_details.txt \
        --output_basis basis_contributions.txt

# fit_details.txt contains:
# - Fitted spectrum
# - Individual metabolite contributions
# - Baseline
# - Residual

# basis_contributions.txt:
# - Amplitude of each basis function
```

### PDF Report

```bash
# Generate visual report (if compiled with PDF support)
tarquin --input metab.rda \
        --output_pdf report.pdf

# Report includes:
# - Fitted spectrum
# - Residual
# - Metabolite table
# - Quality metrics
```

## Integration with Other Tools

### From Osprey

```bash
# Export from Osprey to TARQUIN format
# In MATLAB:
# io_writetarquin(MRSCont.processed.metab{1}, 'subject01.txt');

# Then run TARQUIN
tarquin --input subject01.txt \
        --format txt \
        --output_txt results.txt
```

### From FID-A

```bash
# Export from FID-A
# In MATLAB:
# io_writetarquin(in, 'subject01.txt');

tarquin --input subject01.txt \
        --format txt \
        --output_txt results.txt
```

### To Statistical Software

```bash
# TARQUIN CSV output can be directly imported to R, Python, SPSS

# R example:
# data <- read.csv('results.csv')
# model <- lm(NAA ~ Group + Age, data=data)

# Python example:
# import pandas as pd
# data = pd.read_csv('results.csv')
# from scipy import stats
# t, p = stats.ttest_ind(data[data['Group']==1]['NAA'],
#                         data[data['Group']==2]['NAA'])
```

## GUI Mode

### Launch GUI

```bash
# Start TARQUIN GUI
tarquin_gui

# Or on Windows, double-click tarquin_gui.exe
```

### GUI Features

- Drag-and-drop file loading
- Interactive basis set selection
- Real-time fitting visualization
- Quality metrics display
- Batch processing queue
- Export options panel

## Quality Control

### Interpretation of Metrics

```bash
# Run TARQUIN and check quality
tarquin --input metab.rda \
        --output_txt results.txt

# Key quality metrics:

# SNR (Signal-to-Noise Ratio)
# Good: > 20
# Acceptable: 10-20
# Poor: < 10

# FWHM (Full Width at Half Maximum)
# Good: < 0.08 ppm (< 10 Hz @ 3T)
# Acceptable: 0.08-0.12 ppm
# Poor: > 0.12 ppm

# CRLB (Cramér-Rao Lower Bounds)
# Reliable: < 20%
# Marginal: 20-40%
# Unreliable: > 40%

# Baseline Amplitude
# Low: < 20% of metabolite peak
# Acceptable: 20-40%
# High: > 40% (may indicate artifacts)
```

### Automated QC Filtering

```bash
# Filter by quality in batch script
#!/bin/bash

for file in /data/*.rda; do
    # Run TARQUIN
    tarquin --input ${file} \
            --output_txt results.txt

    # Extract SNR
    snr=$(grep "SNR" results.txt | awk '{print $2}')

    # Check threshold
    if (( $(echo "$snr > 15" | bc -l) )); then
        echo "PASS: ${file} (SNR = ${snr})"
        # Process further
    else
        echo "FAIL: ${file} (SNR = ${snr})"
        # Skip or flag
    fi
done
```

## MEGA-PRESS / Edited Spectroscopy

### GABA Analysis

```bash
# Process MEGA-PRESS for GABA
tarquin --input mega_diff.rda \
        --format rda \
        --basis_mega_press_3t \
        --output_txt gaba_results.txt

# Difference spectrum (edit-ON minus edit-OFF)
# TARQUIN quantifies:
# - GABA+
# - Glx (overlapping)
# - Other edited metabolites
```

### Separate ON/OFF Spectra

```bash
# If ON and OFF need separate processing
tarquin --input mega_on.rda \
        --output_txt on_results.txt

tarquin --input mega_off.rda \
        --output_txt off_results.txt

# Calculate difference manually or use tool
# to combine results
```

## Integration with Claude Code

When helping users with TARQUIN:

1. **Check Installation:**
   ```bash
   tarquin --version
   tarquin --help
   ```

2. **Test with Sample Data:**
   ```bash
   # Download example from website
   tarquin --input example.rda \
           --format rda \
           --output_txt test.txt
   ```

3. **Common Issues:**
   - File format not recognized → Check format flag
   - Poor quantification → Inspect spectrum quality
   - Missing water reference → Use --auto_ref
   - Basis set mismatch → Verify TE and field strength

4. **Optimization:**
   - Use appropriate basis set for your sequence
   - Provide water reference for absolute quantification
   - Include tissue segmentation if available
   - Set reasonable QC thresholds for studies

## Troubleshooting

**Problem:** "Unable to read file"
**Solution:** Check file format, specify --format flag, verify file permissions

**Problem:** Very high CRLBs for all metabolites
**Solution:** Check SNR, inspect raw spectrum quality, verify basis set matches acquisition

**Problem:** Unrealistic concentrations
**Solution:** Verify water reference, check tissue corrections, inspect for artifacts

**Problem:** TARQUIN crashes
**Solution:** Check input file integrity, reduce baseline order, limit metabolite list

**Problem:** Results differ from LCModel
**Solution:** Different algorithms expected to give slightly different results; compare quality metrics

## Best Practices

1. **Always use water reference** when possible for absolute quantification
2. **Inspect spectra visually** before automated processing
3. **Set quality thresholds** appropriate for your study
4. **Save fit files** for quality control review
5. **Use consistent basis sets** across subjects
6. **Document TARQUIN version** used
7. **Report quality metrics** (SNR, FWHM, CRLB) in publications
8. **Validate against phantoms** or known concentrations

## Performance

### Speed Comparison

```bash
# TARQUIN is very fast:
# Single spectrum: < 5 seconds
# Batch of 100: < 5 minutes
# Much faster than LCModel or manual analysis

# Benchmark test
time tarquin --input test.rda --output_txt results.txt

# Typical output:
# real    0m2.4s
# user    0m2.1s
# sys     0m0.3s
```

## Resources

- **Website:** https://tarquin.sourceforge.net/
- **SourceForge:** https://sourceforge.net/projects/tarquin/
- **Documentation:** https://tarquin.sourceforge.net/doc/
- **Forum:** https://sourceforge.net/p/tarquin/discussion/
- **Publication:** https://doi.org/10.1002/mrm.22579

## Citation

```bibtex
@article{wilson2011tarquin,
  title={A constrained least-squares approach to the automated quantitation of in vivo 1H magnetic resonance spectroscopy data},
  author={Wilson, Martin and Reynolds, Glyn and Kauppinen, Risto A and Arvanitis, Theodoros N and Peet, Andrew C},
  journal={Magnetic Resonance in Medicine},
  volume={65},
  number={1},
  pages={1--12},
  year={2011},
  publisher={Wiley Online Library}
}
```

## Related Tools

- **LCModel:** Commercial gold-standard quantification
- **Osprey:** Comprehensive MRS processing suite
- **FID-A:** MATLAB toolkit for MRS processing
- **jMRUI:** Time-domain MRS analysis
- **AMARES:** Advanced Method for Accurate, Robust, and Efficient Spectral fitting
- **INSPECTOR:** Web-based MRS quality control
