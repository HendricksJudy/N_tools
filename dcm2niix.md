# dcm2niix - DICOM to NIfTI Converter

## Overview

dcm2niix is a cross-platform, open-source command-line tool for converting medical images from DICOM format to NIfTI format, the standard used by most neuroimaging analysis software. Developed by Chris Rorden and maintained actively by the neuroimaging community, dcm2niix has become the de facto standard for neuroimaging DICOM conversion, replacing older tools like dcm2nii and mricron due to its superior handling of modern DICOM variants, multi-vendor support, and BIDS (Brain Imaging Data Structure) integration.

dcm2niix excels at handling complex imaging scenarios including multi-echo fMRI, diffusion-weighted imaging (automatically extracting bval/bvec gradient files), phase/magnitude image separation, mosaic and multi-frame DICOM formats, and vendor-specific quirks from Siemens, GE, Philips, Canon, and Bruker scanners. The tool automatically generates BIDS-compliant JSON sidecars containing critical metadata, corrects image orientation issues, and provides extensive options for file naming, anonymization, and format selection. Its speed, reliability, and comprehensive feature set make it an essential tool in every neuroimaging researcher's workflow.

**Official Website:** https://www.nitrc.org/projects/dcm2nii/
**Repository:** https://github.com/rordenlab/dcm2niix
**Documentation:** https://github.com/rordenlab/dcm2niix/blob/master/README.md

### Key Features

- **Multi-Vendor Support:** Siemens, GE, Philips, Canon, Bruker, UIH, and more
- **BIDS Compliance:** Automatic JSON sidecar generation with metadata
- **Diffusion MRI:** Extracts bval/bvec files for DTI/DWI analysis
- **Multi-Echo fMRI:** Handles multi-echo sequences with echo separation
- **Phase/Magnitude:** Separates complex data into phase and magnitude images
- **Orientation Correction:** Automatically fixes vendor-specific orientation issues
- **Gzip Compression:** Creates .nii.gz files to save disk space
- **Batch Processing:** Recursive directory search for automatic conversion
- **Anonymization:** Remove patient identifiable information
- **Flexible Naming:** Custom output filename templates
- **Fast:** Written in C++, highly optimized for speed
- **Cross-Platform:** Windows, macOS, Linux, with minimal dependencies

### Applications

- Converting scanner DICOM data to analysis-ready NIfTI format
- Creating BIDS-compliant neuroimaging datasets
- Multi-site study harmonization (consistent format across vendors)
- Clinical scan conversion for research analysis
- Automated preprocessing pipeline integration
- Quality control and metadata verification
- Teaching DICOM/NIfTI concepts

### Why dcm2niix vs. Other Converters?

- **vs. dcm2nii (older version):** dcm2niix handles modern DICOM better, faster, more features
- **vs. mricron:** dcm2niix is command-line optimized for batch processing
- **vs. MATLAB dicomread:** dcm2niix preserves metadata and handles multi-volume
- **vs. FreeSurfer mri_convert:** dcm2niix better vendor support and BIDS compliance
- **Recommendation:** dcm2niix is the current gold standard for neuroimaging

### Citation

```bibtex
@article{Li2016dcm2niix,
  title={The first step for neuroimaging data analysis: DICOM to NIfTI conversion},
  author={Li, Xiangrui and Morgan, Paul S and Ashburner, John and Smith, Jolinda and Rorden, Christopher},
  journal={Journal of Neuroscience Methods},
  volume={264},
  pages={47--56},
  year={2016},
  publisher={Elsevier}
}
```

---

## Installation

### Pre-Compiled Binaries

```bash
# Download from GitHub releases
# https://github.com/rordenlab/dcm2niix/releases

# Windows:
# Download dcm2niix_win.zip
# Extract and add to PATH

# macOS:
# Download dcm2niix_mac.zip
# Extract to /usr/local/bin/

# Linux:
# Download dcm2niix_lnx.zip
wget https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip
unzip dcm2niix_lnx.zip
sudo mv dcm2niix /usr/local/bin/
sudo chmod +x /usr/local/bin/dcm2niix
```

### Package Managers

```bash
# Conda (recommended for research environments)
conda install -c conda-forge dcm2niix

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install dcm2niix

# macOS (Homebrew)
brew install dcm2niix

# Arch Linux
sudo pacman -S dcm2niix

# Verify installation
dcm2niix -h
# Should display help message with version
```

### Docker Container

```bash
# Pull official container
docker pull rordenlab/dcm2niix:latest

# Run conversion in container
docker run --rm -v /path/to/dicom:/data rordenlab/dcm2niix \
  dcm2niix -o /data/output /data/input

# Create alias for convenient use
echo 'alias dcm2niix="docker run --rm -v $(pwd):/data rordenlab/dcm2niix dcm2niix"' >> ~/.bashrc
```

### Compiling from Source

```bash
# Clone repository
git clone https://github.com/rordenlab/dcm2niix.git
cd dcm2niix

# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake

# Build with CMake
mkdir build && cd build
cmake ..
make

# Install
sudo make install

# Test
dcm2niix -v
# Output: dcm2niix version v1.0.20220720
```

### Testing Installation

```bash
# Check version
dcm2niix -v

# Display help
dcm2niix -h

# Test with sample data (if available)
dcm2niix -o /tmp/test ~/DICOM_sample/

# Verify output
ls /tmp/test/
# Should show .nii.gz and .json files
```

---

## Basic Usage

### Command-Line Syntax

```bash
# Basic syntax
dcm2niix [options] <input_directory>

# Simplest conversion
dcm2niix /path/to/dicom

# Specify output directory
dcm2niix -o /path/to/output /path/to/dicom

# Common options:
# -o <dir>   : output directory
# -f <format>: output filename format
# -z y       : compress output (.nii.gz)
# -b y       : create BIDS JSON sidecar
# -ba y      : anonymize BIDS sidecar
```

### Converting Single Series

```bash
# Convert single DICOM series
cd /path/to/dicom/series1
dcm2niix -o ~/nifti ./

# Output files:
# - series1.nii.gz (NIfTI image)
# - series1.json (metadata sidecar)

# Specify output filename
dcm2niix -o ~/nifti -f sub-01_T1w ./

# Output:
# - sub-01_T1w.nii.gz
# - sub-01_T1w.json
```

### Converting Entire Directory

```bash
# Recursively convert all DICOM series in directory tree
dcm2niix -o ~/nifti ~/DICOM_study/

# Directory structure:
# ~/DICOM_study/
# ├── sub-01/
# │   ├── anat/
# │   │   └── [T1 DICOM files]
# │   └── func/
# │       └── [fMRI DICOM files]
# └── sub-02/
#     └── ...

# dcm2niix automatically detects series and converts each

# Output in ~/nifti/:
# - sub-01_T1w.nii.gz
# - sub-01_T1w.json
# - sub-01_task-rest_bold.nii.gz
# - sub-01_task-rest_bold.json
# - sub-02_T1w.nii.gz
# - ...
```

### Output File Naming

```bash
# Default naming uses DICOM metadata:
# <ProtocolName>_<SeriesNumber>_<ImageType>.nii.gz

# Custom filename template with -f option
# Placeholders:
# %f = folder name
# %i = patient ID
# %n = patient name
# %p = protocol name
# %s = series number
# %t = time

# Example: BIDS-style naming
dcm2niix -f "sub-%i_ses-01_%p" -o ~/bids/sub-01/anat/ ~/DICOM/sub-01/anat/

# Output: sub-01_ses-01_T1w.nii.gz

# Include series number and protocol
dcm2niix -f "%s_%p" ~/DICOM/

# Output: 3_T1_MPRAGE.nii.gz
```

---

## Advanced Options

### Output Format Options

```bash
# Compression control
dcm2niix -z y ~/DICOM/  # Compress (.nii.gz) - recommended
dcm2niix -z n ~/DICOM/  # No compression (.nii)
dcm2niix -z i ~/DICOM/  # Internal compression (pigz if available)

# BIDS JSON sidecar
dcm2niix -b y ~/DICOM/  # Create JSON with metadata
dcm2niix -b n ~/DICOM/  # No JSON sidecar

# Combine options
dcm2niix -z y -b y -o ~/nifti ~/DICOM/
# Creates .nii.gz + .json for each series
```

### Anonymization

```bash
# Anonymize patient information in BIDS JSON
dcm2niix -ba y ~/DICOM/

# Result: JSON sidecar omits:
# - PatientName
# - PatientID
# - PatientBirthDate
# - PatientSex (kept by default, remove with additional flags)

# Full anonymization (experimental)
dcm2niix -ba y -z y -f "%s_%p" ~/DICOM/
# Removes patient info from filenames and metadata
```

### Conflict Resolution

```bash
# Handle duplicate output filenames
# -d <n>: Behavior when file exists

# -d n: Don't create duplicate
dcm2niix -d n ~/DICOM/
# If file exists, skip conversion

# -d y: Create duplicate with suffix
dcm2niix -d y ~/DICOM/
# Creates: series1.nii.gz, series1a.nii.gz, series1b.nii.gz

# -d o: Overwrite existing
dcm2niix -d o ~/DICOM/
# Overwrites previous conversion
```

### Custom Naming Scheme

```bash
# Complex naming template example
dcm2niix -f "sub-%i_ses-%t_run-%s_%p" ~/DICOM/

# Placeholders explained:
# %f = folder name containing DICOMs
# %i = patient ID from DICOM
# %n = patient name
# %m = manufacturer
# %p = protocol name
# %s = series number
# %t = time (HHMMSS)
# %d = description

# Example output:
# sub-001_ses-143022_run-003_T1_MPRAGE.nii.gz
```

### BIDS-Compliant Conversion

```bash
# Create BIDS-compliant dataset structure

# Setup BIDS directory
mkdir -p ~/bids_dataset/{sub-01,sub-02}/{anat,func,dwi}

# Convert anatomical
dcm2niix \
  -z y \
  -b y \
  -ba y \
  -f "sub-01_T1w" \
  -o ~/bids_dataset/sub-01/anat/ \
  ~/DICOM/sub-01/T1/

# Convert functional
dcm2niix \
  -z y \
  -b y \
  -ba y \
  -f "sub-01_task-rest_bold" \
  -o ~/bids_dataset/sub-01/func/ \
  ~/DICOM/sub-01/fMRI/

# Convert diffusion
dcm2niix \
  -z y \
  -b y \
  -ba y \
  -f "sub-01_dwi" \
  -o ~/bids_dataset/sub-01/dwi/ \
  ~/DICOM/sub-01/DTI/

# Result: BIDS-compliant dataset ready for fMRIPrep, QSIPrep, etc.
```

---

## Multi-Vendor Support

### Siemens

```bash
# Siemens DICOM (including enhanced DICOM)
dcm2niix ~/DICOM/siemens/

# Handles:
# - Classic DICOM (IMA files)
# - Enhanced DICOM (newer format)
# - Mosaic format (fMRI, DTI)
# - CSA headers (for diffusion gradients)

# Siemens mosaic to 4D volume
# Automatically detects and converts mosaic

# Example: DTI with gradients
dcm2niix -o ~/output ~/DICOM/siemens_dti/
# Creates:
# - dwi.nii.gz (4D volume)
# - dwi.bval (b-values)
# - dwi.bvec (gradient directions)
# - dwi.json (metadata)
```

### GE

```bash
# GE DICOM and P-files
dcm2niix ~/DICOM/ge/

# Handles:
# - GE DICOM
# - P-files (raw data, if compiled with support)
# - Multi-frame DICOM

# GE slice timing correction
# Properly extracts slice timing from GE metadata
# Check JSON sidecar: "SliceTiming" field

# Example: GE fMRI
dcm2niix -b y ~/DICOM/ge_fmri/
cat output.json | grep -A 10 SliceTiming
# Verify slice timing extracted correctly
```

### Philips

```bash
# Philips DICOM and PAR/REC
dcm2niix ~/DICOM/philips/

# Handles:
# - Philips DICOM
# - PAR/REC format (older Philips format)
# - Philips enhanced DICOM

# PAR/REC conversion
dcm2niix ~/DICOM/philips/*.PAR

# Philips diffusion
# Automatically extracts bval/bvec from Philips DICOM
dcm2niix ~/DICOM/philips_dti/
# Creates dwi.nii.gz, dwi.bval, dwi.bvec
```

### Vendor-Specific Quirks

```text
Siemens:
- Mosaic format requires special handling (dcm2niix does automatically)
- Enhanced DICOM stores metadata differently (handled)
- CSA headers for diffusion (parsed correctly)

GE:
- Slice timing in custom tags (dcm2niix extracts)
- P-files require special compilation flag
- Variable TR in fMRI (detected and handled)

Philips:
- PAR/REC format (dcm2niix supports directly)
- Scaling factors in metadata (applied correctly)
- Diffusion in private tags (extracted)

Canon/Toshiba:
- Relatively standard DICOM (well supported)

Bruker:
- Research scanner format (supported)
- ParaVision datasets (can convert)
```

---

## Diffusion MRI Conversion

### Extracting bval/bvec Files

```bash
# DTI/DWI conversion automatically extracts gradient information
dcm2niix -o ~/dwi ~/DICOM/dti_scan/

# Output files:
# - dwi.nii.gz (4D volume: b0 + diffusion directions)
# - dwi.bval (b-values, space-separated)
# - dwi.bvec (gradient directions, 3 rows x N columns)
# - dwi.json (metadata)

# Verify bval file
cat dwi.bval
# Example: 0 1000 1000 1000 ... (b0 followed by diffusion b-values)

# Verify bvec file
cat dwi.bvec
# Example:
# 0.0000 0.9876 -0.5432 ...  (x-components)
# 0.0000 0.1234 0.8765 ...   (y-components)
# 0.0000 0.0987 0.1234 ...   (z-components)
```

### Multi-Shell DTI

```bash
# Multi-shell diffusion (multiple b-values)
dcm2niix ~/DICOM/multishell_dti/

# Example bval output:
# 0 0 0 1000 1000 1000 2000 2000 2000
# (3 b0s, 3 directions at b=1000, 3 directions at b=2000)

# Verify number of directions
wc -w dwi.bval
# Should match number of volumes in NIfTI

# Check 4D dimensions
fslinfo dwi.nii.gz | grep dim4
# dim4  9  (matches bval count)
```

### Handling Gradient Directions

```bash
# dcm2niix corrects gradient directions for image orientation
# Output bvec is in scanner coordinate system

# Verify gradient table orientation
# Use FSL's fdt_rotate_bvecs if needed (usually not necessary)

# Visual inspection of gradients
# Use FSLeyes or other viewer:
fsleyes dwi.nii.gz

# Check JSON for additional info
cat dwi.json | grep -i "diffusion"
# Shows: DiffusionScheme, b-value, etc.
```

### Example: DTI Conversion Pipeline

```bash
#!/bin/bash
# Complete DTI conversion pipeline

DICOM_DIR="~/DICOM/sub-01/DTI"
OUTPUT_DIR="~/bids/sub-01/dwi"

# Convert with dcm2niix
dcm2niix \
  -z y \
  -b y \
  -ba y \
  -f "sub-01_dwi" \
  -o ${OUTPUT_DIR} \
  ${DICOM_DIR}

# Verify outputs
echo "Checking DTI conversion outputs:"
ls -lh ${OUTPUT_DIR}/

# Check dimensions match
n_vols=$(fslinfo ${OUTPUT_DIR}/sub-01_dwi.nii.gz | grep "dim4" | awk '{print $2}')
n_bvals=$(wc -w < ${OUTPUT_DIR}/sub-01_dwi.bval)

if [ ${n_vols} -eq ${n_bvals} ]; then
    echo "✓ Volume count matches b-values: ${n_vols}"
else
    echo "✗ ERROR: Mismatch! Volumes: ${n_vols}, b-values: ${n_bvals}"
fi

# Ready for QSIPrep or other DTI analysis
echo "DTI conversion complete. Ready for analysis."
```

---

## Multi-Echo and Complex Data

### Multi-Echo fMRI

```bash
# Multi-echo fMRI (multiple TEs per volume)
dcm2niix ~/DICOM/multiecho_fmri/

# dcm2niix automatically separates echoes
# Output files (if 3 echoes):
# - func_e1.nii.gz (echo 1, shortest TE)
# - func_e1.json
# - func_e2.nii.gz (echo 2)
# - func_e2.json
# - func_e3.nii.gz (echo 3, longest TE)
# - func_e3.json

# Check echo times in JSON
cat func_e1.json | grep EchoTime
# "EchoTime": 0.015  (15ms)
cat func_e2.json | grep EchoTime
# "EchoTime": 0.030  (30ms)
cat func_e3.json | grep EchoTime
# "EchoTime": 0.045  (45ms)

# Ready for multi-echo processing (tedana, fMRIPrep with --me-output-echos)
```

### Phase and Magnitude Separation

```bash
# Complex MRI data (phase and magnitude)
dcm2niix ~/DICOM/fieldmap/

# Separates phase and magnitude automatically
# Output:
# - fieldmap_ph.nii.gz (phase image)
# - fieldmap_ph.json
# - fieldmap_magnitude.nii.gz (magnitude image)
# - fieldmap_magnitude.json

# Use for:
# - B0 field mapping (phase for distortion correction)
# - Susceptibility-weighted imaging (SWI)
# - Quantitative susceptibility mapping (QSM)

# BIDS fieldmap convention
dcm2niix \
  -f "sub-01_phasediff" \
  -o ~/bids/sub-01/fmap/ \
  ~/DICOM/fieldmap/
```

### Real and Imaginary Components

```bash
# Real/imaginary (for complex reconstruction)
# Less common, but dcm2niix supports

# Output (if present):
# - image_real.nii.gz
# - image_imaginary.nii.gz

# Combine to magnitude and phase:
# magnitude = sqrt(real^2 + imag^2)
# phase = atan2(imag, real)

# Most users work with magnitude/phase directly
```

---

## BIDS Integration

### BIDS Directory Structure

```bash
# Standard BIDS structure created with dcm2niix

bids_dataset/
├── sub-01/
│   ├── anat/
│   │   ├── sub-01_T1w.nii.gz
│   │   └── sub-01_T1w.json
│   ├── func/
│   │   ├── sub-01_task-rest_bold.nii.gz
│   │   └── sub-01_task-rest_bold.json
│   ├── dwi/
│   │   ├── sub-01_dwi.nii.gz
│   │   ├── sub-01_dwi.bval
│   │   ├── sub-01_dwi.bvec
│   │   └── sub-01_dwi.json
│   └── fmap/
│       ├── sub-01_phasediff.nii.gz
│       └── sub-01_magnitude1.nii.gz
├── sub-02/
│   └── ...
└── dataset_description.json

# dcm2niix handles NIfTI + JSON creation
# User provides proper BIDS naming with -f option
```

### JSON Sidecar Metadata

```bash
# BIDS JSON contains critical metadata
cat sub-01_task-rest_bold.json

# Example JSON content:
{
  "EchoTime": 0.03,
  "RepetitionTime": 2.0,
  "FlipAngle": 90,
  "SliceTiming": [0, 0.5, 1.0, 1.5, ...],
  "PhaseEncodingDirection": "j-",
  "EffectiveEchoSpacing": 0.00051,
  "TotalReadoutTime": 0.0459,
  "Manufacturer": "Siemens",
  "ManufacturersModelName": "Prisma",
  "MagneticFieldStrength": 3,
  "ImagingFrequency": 123.2,
  ...
}

# This metadata is essential for:
# - fMRIPrep preprocessing
# - Slice timing correction
# - Distortion correction (SDC)
# - Quality control
```

### Integration with HeuDiConv

```bash
# HeuDiConv uses dcm2niix as conversion backend

# Install HeuDiConv
pip install heudiconv

# Create heuristic file (defines BIDS naming)
# heuristic.py example available in HeuDiConv docs

# Run HeuDiConv (calls dcm2niix internally)
heudiconv \
  -d /path/to/DICOM/{subject}/*/*.dcm \
  -o /path/to/bids \
  -f heuristic.py \
  -s 01 02 03 \
  -c dcm2niix \
  -b

# HeuDiConv automatically:
# - Organizes DICOM by subject
# - Calls dcm2niix for conversion
# - Renames files per BIDS convention
# - Creates dataset_description.json
```

### Integration with Dcm2Bids

```bash
# Dcm2Bids: Another BIDS converter using dcm2niix

# Install
pip install dcm2bids

# Create configuration file
dcm2bids_scaffold -o ~/bids_dataset

# Edit config file: code/dcm2bids_config.json
# Define BIDS naming rules

# Run conversion
dcm2bids \
  -d ~/DICOM/sub-01/ \
  -p 01 \
  -c ~/bids_dataset/code/dcm2bids_config.json \
  -o ~/bids_dataset/

# Dcm2Bids handles:
# - Calling dcm2niix
# - BIDS renaming per config
# - Sidecar organization
```

---

## Batch Processing

### Recursive Directory Search

```bash
# dcm2niix automatically searches subdirectories
dcm2niix -o ~/output ~/DICOM_study/

# Processes all series found in directory tree
# Useful for converting entire studies at once
```

### Scripting for Multiple Subjects

```bash
#!/bin/bash
# Batch conversion script for multiple subjects

DICOM_ROOT="~/DICOM_raw"
OUTPUT_ROOT="~/nifti_converted"

# Subject list
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05)

for sub in "${subjects[@]}"; do
    echo "Processing ${sub}..."

    # Create output directory
    mkdir -p ${OUTPUT_ROOT}/${sub}

    # Convert all series for this subject
    dcm2niix \
        -z y \
        -b y \
        -ba y \
        -f "${sub}_%p_%s" \
        -o ${OUTPUT_ROOT}/${sub} \
        ${DICOM_ROOT}/${sub}/

    echo "${sub} complete."
done

echo "All subjects converted."
```

### Parallel Processing

```bash
# Use GNU parallel for faster batch conversion

# Install GNU parallel
sudo apt-get install parallel

# Create subject list
ls -1d ~/DICOM/sub-* > subjects.txt

# Parallel conversion function
convert_subject() {
    sub=$(basename $1)
    echo "Processing ${sub}..."
    dcm2niix -z y -b y -o ~/nifti/${sub} $1
}

# Export function for parallel
export -f convert_subject

# Run in parallel (4 subjects at a time)
parallel -j 4 convert_subject :::: subjects.txt

echo "Parallel conversion complete."
```

---

## Metadata and Quality Control

### Inspecting JSON Sidecars

```bash
# View JSON metadata
cat sub-01_T1w.json | jq .

# Extract specific fields
echo "TR: $(jq -r .RepetitionTime sub-01_bold.json)"
echo "TE: $(jq -r .EchoTime sub-01_bold.json)"

# Check slice timing
jq -r .SliceTiming sub-01_bold.json

# Verify acquisition parameters
jq -r '.RepetitionTime, .EchoTime, .FlipAngle' sub-01_bold.json
```

### Verifying Orientations

```bash
# Check NIfTI orientation
fslinfo sub-01_T1w.nii.gz

# Output includes:
# qform_name: NIFTI_XFORM_SCANNER_ANAT
# sform_name: NIFTI_XFORM_SCANNER_ANAT

# Visualize to verify correct orientation
fsleyes sub-01_T1w.nii.gz &

# Check if left/right are correct:
# - Left side of brain should appear on RIGHT side of image (radiological)
# - Or configure viewer for neurological convention

# dcm2niix typically handles orientation correctly
# Issues rare but possible with unusual DICOM
```

### Checking for Conversion Errors

```bash
# dcm2niix outputs warnings/errors to stdout
dcm2niix ~/DICOM/ 2>&1 | tee conversion.log

# Check log for warnings
grep -i warning conversion.log
grep -i error conversion.log

# Common warnings:
# - "Unable to determine slice timing" (some sequences don't encode this)
# - "Duplicate series" (handled with -d option)

# Verify all expected series converted
ls -1 *.nii.gz | wc -l
# Compare to number of DICOM series expected
```

---

## Troubleshooting

### Common Conversion Errors

**Error: "Unable to open DICOM file"**

**Solution:**
```bash
# Check file permissions
chmod 644 *.dcm

# Verify files are valid DICOM
dcminfo file.dcm

# Check for compressed DICOM (dcm2niix handles most)
# If issues persist, try decompressing first
```

**Error: "Slice order ambiguous"**

**Solution:**
```bash
# Some sequences don't encode slice timing clearly
# dcm2niix may not extract SliceTiming field

# Workaround: Manually add to JSON after conversion
# Or obtain from scanner protocol
```

### Orientation Issues

**Problem:** Left/right flipped or incorrect orientation

**Solution:**
```bash
# Visualize in FSLeyes
fsleyes image.nii.gz

# Check qform/sform
fslinfo image.nii.gz

# If orientation wrong, may need to manually fix
# (Rare with dcm2niix, usually DICOM issue)

# Use FSL fslorient to check/fix
fslorient -getorient image.nii.gz
fslorient -setqformcode 1 image.nii.gz  # Mark as scanner coords
```

### Missing Slices

**Problem:** Incomplete volume (missing slices)

**Solution:**
```bash
# Check DICOM source
ls *.dcm | wc -l
# Should match expected slice count

# dcm2niix warns if slices missing:
grep -i "missing" conversion.log

# If slices truly missing, contact scanner operator
# May need to re-export from scanner
```

### Vendor-Specific Problems

**Siemens Enhanced DICOM:**
```bash
# Some tools don't handle enhanced DICOM
# dcm2niix does, but verify

dcm2niix -v  # Check version (ensure recent)
# Upgrade if old version
```

**GE Slice Timing:**
```bash
# GE doesn't always encode slice timing
# Check JSON for SliceTiming field

# If missing, may need to:
# 1. Contact GE support for timing info
# 2. Use fMRIPrep's --ignore-slicetiming
```

**Philips Scaling:**
```bash
# Philips uses scaling factors
# dcm2niix applies automatically

# Verify intensity ranges reasonable
fslstats image.nii.gz -R
# Should be typical for MRI (0-4000 range)
```

---

## Best Practices

### Pre-Conversion Organization

- **Organize DICOM by subject/session** before conversion
- **Verify DICOM completeness** (all series present)
- **Check for DICOM errors** at scanner (missing slices, artifacts)
- **Document scanner protocol** (sequence parameters for reference)

### Naming Conventions

- **Use BIDS naming** for compatibility with pipelines (fMRIPrep, QSIPrep)
- **Include subject/session ID** in filename template
- **Avoid spaces and special characters** in filenames
- **Be consistent** across all conversions

### Quality Control Steps

1. **Verify conversion count:** Number of NIfTI files matches expected series
2. **Check JSON metadata:** TR, TE, slice timing present
3. **Visualize images:** Inspect for artifacts, orientation errors
4. **Validate BIDS:** Use BIDS Validator if creating BIDS dataset
5. **Archive original DICOM:** Keep DICOM for reference (conversions can be re-done)

### BIDS Compliance Tips

- **Use dcm2niix with -b y -ba y** for BIDS JSON creation
- **Follow BIDS naming convention exactly** (sub-, ses-, task-, etc.)
- **Create dataset_description.json** (required for BIDS)
- **Validate with BIDS Validator:** https://bids-standard.github.io/bids-validator/
- **Document any BIDS extensions** in README

---

## References

1. **dcm2niix:**
   - Li et al. (2016). The first step for neuroimaging data analysis: DICOM to NIfTI conversion. *J Neurosci Methods*, 264:47-56.
   - https://github.com/rordenlab/dcm2niix

2. **NIfTI Format:**
   - Cox et al. (2004). A (sort of) new image data format standard: NIfTI-1. *NeuroImage*, 22:e1440.

3. **BIDS:**
   - Gorgolewski et al. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. *Sci Data*, 3:160044.

4. **DICOM Standard:**
   - NEMA PS3 / ISO 12052. Digital Imaging and Communications in Medicine (DICOM) Standard.

**Official Resources:**
- dcm2niix GitHub: https://github.com/rordenlab/dcm2niix
- NITRC Project: https://www.nitrc.org/projects/dcm2nii/
- BIDS Specification: https://bids-specification.readthedocs.io/
- NIfTI Format: https://nifti.nimh.nih.gov/
