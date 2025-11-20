# HeuDiConv

## Overview

HeuDiConv (Heuristic DICOM Converter) is a flexible DICOM-to-BIDS converter that uses heuristics to automatically organize and convert neuroimaging data from DICOM format to the Brain Imaging Data Structure (BIDS) format. It provides powerful customization through Python-based heuristic files and integrates seamlessly with DataLad for version control.

**Website:** https://github.com/nipy/heudiconv
**Platform:** Linux/macOS (Docker/Singularity containers)
**Language:** Python
**License:** Apache 2.0

## Key Features

- Flexible DICOM to BIDS conversion
- Customizable heuristics (Python-based)
- Automatic metadata extraction
- Integration with DataLad for version control
- Handles complex scan protocols
- Supports multiple modalities (anat, func, dwi, fmap)
- Defacing capability for anatomical images
- Docker and Singularity containers available
- Dry-run mode for testing
- Comprehensive logging

## Installation

### Using pip

```bash
# Install HeuDiConv
pip install heudiconv

# Additional dependencies
pip install dcm2niix  # Or install system package

# Verify installation
heudiconv --version
```

### Using Docker (Recommended)

```bash
# Pull Docker image
docker pull nipy/heudiconv:latest

# Run
docker run --rm -it -v /path/to/data:/data:ro \
  nipy/heudiconv:latest --version
```

### Using Singularity

```bash
# Build from Docker
singularity build heudiconv.sif docker://nipy/heudiconv:latest

# Run
singularity run heudiconv.sif --version
```

## Basic Usage

### Exploring DICOM Data

```bash
# List available scan series
heudiconv \
  -d /data/dicom/{subject}/*/*.dcm \
  -s sub001 \
  -c none \
  -f convertall \
  -o /output

# Output shows:
# - Series descriptions
# - Number of files
# - Sequence information
# Helps design heuristic
```

### Simple Conversion

```bash
# Convert with built-in heuristic
heudiconv \
  -d /data/dicom/{subject}/*/*.dcm \
  -s sub001 sub002 sub003 \
  -c dcm2niix \
  -b \
  -f convertall \
  -o /output/bids

# Options:
# -d: DICOM file pattern
# -s: Subject IDs
# -c: Converter (dcm2niix recommended)
# -b: Create BIDS
# -f: Heuristic file
# -o: Output directory
```

### With Custom Heuristic

```bash
# Use custom heuristic file
heudiconv \
  -d /data/dicom/{subject}/*/*.dcm \
  -s sub001 \
  -c dcm2niix \
  -b \
  -f /path/to/myheuristic.py \
  -o /output/bids \
  --overwrite
```

## Creating Heuristic Files

### Basic Heuristic Template

```python
# myheuristic.py
import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo, metadata):
    """
    Heuristic evaluator for determining which runs belong where

    Parameters:
    -----------
    seqinfo : list of namedtuples
        Sequence info from DICOM headers
    metadata : list of dicts
        Additional metadata

    Returns:
    --------
    info : dict
        Dictionary mapping file templates to file lists
    """
    # Define output file templates
    t1w = create_key('sub-{subject}/anat/sub-{subject}_T1w')
    t2w = create_key('sub-{subject}/anat/sub-{subject}_T2w')

    func_rest = create_key('sub-{subject}/func/sub-{subject}_task-rest_bold')
    func_task = create_key('sub-{subject}/func/sub-{subject}_task-{task}_run-{item:02d}_bold')

    dwi = create_key('sub-{subject}/dwi/sub-{subject}_dwi')

    fmap_magnitude = create_key('sub-{subject}/fmap/sub-{subject}_magnitude{item}')
    fmap_phasediff = create_key('sub-{subject}/fmap/sub-{subject}_phasediff')

    # Initialize dictionary
    info = {
        t1w: [], t2w: [],
        func_rest: [], func_task: [],
        dwi: [],
        fmap_magnitude: [], fmap_phasediff: []
    }

    # Iterate through scans
    for idx, s in enumerate(seqinfo):
        # Anatomical
        if ('t1' in s.protocol_name.lower() and
            s.dim3 > 100):
            info[t1w].append(s.series_id)

        if ('t2' in s.protocol_name.lower() and
            s.dim3 > 100):
            info[t2w].append(s.series_id)

        # Functional
        if 'rest' in s.protocol_name.lower():
            info[func_rest].append(s.series_id)

        if 'task' in s.protocol_name.lower():
            # Extract task name from protocol
            task_name = s.protocol_name.split('_')[1]
            info[func_task].append({'item': s.series_id, 'task': task_name})

        # Diffusion
        if 'dti' in s.protocol_name.lower() or 'dwi' in s.protocol_name.lower():
            info[dwi].append(s.series_id)

        # Fieldmaps
        if 'field_mapping' in s.protocol_name.lower():
            if 'M' in s.image_type:  # Magnitude
                info[fmap_magnitude].append(s.series_id)
            elif 'P' in s.image_type:  # Phase
                info[fmap_phasediff].append(s.series_id)

    return info
```

### Advanced Heuristic with Sessions

```python
# heuristic_sessions.py
import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo, metadata):
    """Heuristic with session support"""

    # Session information from seqinfo
    # Assumes session info in study description or date

    t1w = create_key('sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T1w')
    func = create_key('sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-{task}_run-{item:02d}_bold')

    info = {t1w: [], func: []}

    for s in seqinfo:
        # Extract session from study description
        if hasattr(s, 'study_description'):
            session = s.study_description.split('_')[0]  # e.g., "baseline_study"
        else:
            session = '01'

        if 't1' in s.protocol_name.lower():
            info[t1w].append({'item': s.series_id, 'session': session})

        if 'task' in s.protocol_name.lower():
            task = extract_task_name(s.protocol_name)
            run = extract_run_number(s.series_description)
            info[func].append({
                'item': run,
                'task': task,
                'session': session
            })

    return info

def extract_task_name(protocol):
    """Extract task name from protocol"""
    # Implement your logic
    if 'nback' in protocol.lower():
        return 'nback'
    elif 'rest' in protocol.lower():
        return 'rest'
    return 'unknown'

def extract_run_number(description):
    """Extract run number from series description"""
    import re
    match = re.search(r'run[_-]?(\d+)', description, re.IGNORECASE)
    return int(match.group(1)) if match else 1
```

## Docker Usage

### Basic Docker Command

```bash
# Convert DICOM to BIDS
docker run --rm -it \
  -v /path/to/dicom:/data:ro \
  -v /path/to/output:/output \
  -v /path/to/heuristic.py:/heuristic.py:ro \
  nipy/heudiconv:latest \
  -d /data/{subject}/*/*.dcm \
  -s sub001 sub002 \
  -c dcm2niix \
  -b \
  -f /heuristic.py \
  -o /output \
  --overwrite
```

### With DataLad Integration

```bash
# Initialize DataLad dataset
datalad create -c bids bids_dataset
cd bids_dataset

# Run HeuDiConv with DataLad
datalad run \
  -m "Convert DICOM to BIDS for sub-001" \
  -i /source/dicom/sub001 \
  -o . \
  "docker run --rm \
    -v /source/dicom:/data:ro \
    -v $(pwd):/output \
    -v /heuristic.py:/heuristic.py:ro \
    nipy/heudiconv:latest \
    -d /data/{subject}/*/*.dcm \
    -s sub001 \
    -c dcm2niix \
    -b \
    -f /heuristic.py \
    -o /output"
```

## Advanced Features

### Defacing Anatomical Images

```bash
# Deface T1w images for anonymization
heudiconv \
  -d /data/dicom/{subject}/*/*.dcm \
  -s sub001 \
  -c dcm2niix \
  -b \
  -f myheuristic.py \
  -o /output/bids \
  --minmeta \
  --anon-cmd "pydeface {item}"
```

### Handling Multiple Sessions

```bash
# Convert data with sessions
heudiconv \
  -d /data/dicom/{subject}/{session}/*/*.dcm \
  -s sub001 \
  -ss ses01 ses02 \
  -c dcm2niix \
  -b \
  -f heuristic_sessions.py \
  -o /output/bids
```

### Custom Metadata

```python
# In heuristic file
def custom_seqinfo(wrapper, series_files):
    """Add custom metadata extraction"""
    info = wrapper(series_files)

    # Add custom fields
    if hasattr(info, 'protocol_name'):
        info.custom_field = extract_custom_info(info.protocol_name)

    return info

# Define in heuristic
infotodict.custom_seqinfo = custom_seqinfo
```

## IntendedFor Fieldmaps

```python
# Specify IntendedFor in heuristic
def infotodict(seqinfo, metadata):
    # ... define keys ...

    info = {fmap: [], func: []}

    for s in seqinfo:
        if 'fieldmap' in s.protocol_name.lower():
            # Store with IntendedFor information
            info[fmap].append({
                'item': s.series_id,
                'IntendedFor': [
                    'func/sub-{subject}_task-rest_bold.nii.gz',
                    'func/sub-{subject}_task-task_bold.nii.gz'
                ]
            })

    return info
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# batch_convert.sh

DICOM_DIR=/data/dicom
OUTPUT_DIR=/data/bids
HEURISTIC=/scripts/myheuristic.py

subjects=(sub001 sub002 sub003 sub004 sub005)

for subj in "${subjects[@]}"; do
    echo "Processing $subj"

    heudiconv \
      -d ${DICOM_DIR}/${subj}/*/*.dcm \
      -s $subj \
      -c dcm2niix \
      -b \
      -f $HEURISTIC \
      -o $OUTPUT_DIR \
      --overwrite

    # Validate
    bids-validator ${OUTPUT_DIR}
done
```

### Parallel Processing

```bash
# Using GNU parallel
parallel -j 4 \
  heudiconv \
    -d /data/dicom/{1}/*/*.dcm \
    -s {1} \
    -c dcm2niix \
    -b \
    -f myheuristic.py \
    -o /output/bids \
  ::: sub001 sub002 sub003 sub004
```

## Troubleshooting Conversions

### Debug Mode

```bash
# Enable verbose logging
heudiconv \
  -d /data/dicom/{subject}/*/*.dcm \
  -s sub001 \
  -c none \
  -f myheuristic.py \
  -o /output \
  --debug

# Check .heudiconv directory
ls -la /output/.heudiconv/sub001/
# Contains:
# - dicominfo.tsv (DICOM header summary)
# - filegroup.txt (file groupings)
```

### Testing Heuristic

```bash
# Dry run (no conversion)
heudiconv \
  -d /data/dicom/{subject}/*/*.dcm \
  -s sub001 \
  -c none \
  -f myheuristic.py \
  -o /output

# Review output in .heudiconv/sub001/
# Check if series are correctly identified
```

### Common Issues

```python
# Issue: Series not being detected

# Debug in heuristic
def infotodict(seqinfo, metadata):
    # Print debug info
    for s in seqinfo:
        print(f"Series {s.series_id}: {s.protocol_name}")
        print(f"  Dimensions: {s.dim1}x{s.dim2}x{s.dim3}")
        print(f"  Image type: {s.image_type}")

    # ... rest of heuristic ...
```

## Integration with Claude Code

When helping users with HeuDiConv:

1. **Check Installation:**
   ```bash
   heudiconv --version
   dcm2niix --version
   ```

2. **Common Issues:**
   - dcm2niix not installed
   - DICOM pattern doesn't match files
   - Heuristic logic errors
   - File permissions in Docker
   - Metadata extraction failures

3. **Best Practices:**
   - Start with dry run (`-c none`)
   - Test heuristic on one subject first
   - Use descriptive protocol names in scanner
   - Validate output with bids-validator
   - Keep heuristic files under version control
   - Document scan protocol assumptions
   - Use DataLad for reproducibility

4. **Quick Start:**
   ```bash
   # 1. Explore DICOM
   heudiconv -d /data/{subject}/*/*.dcm -s sub001 -c none -f convertall -o /tmp

   # 2. Check .heudiconv/sub001/dicominfo.tsv

   # 3. Create heuristic based on series info

   # 4. Test conversion
   heudiconv -d /data/{subject}/*/*.dcm -s sub001 -c dcm2niix -b -f heuristic.py -o /output

   # 5. Validate
   bids-validator /output
   ```

## Troubleshooting

**Problem:** "No DICOM files found"
**Solution:** Check DICOM pattern with `ls`, ensure correct `{subject}` placeholder

**Problem:** Heuristic not matching any series
**Solution:** Check dicominfo.tsv, review protocol_name matching logic

**Problem:** Missing metadata in JSON
**Solution:** Use `--minmeta` flag, check DICOM headers have required fields

**Problem:** Docker permission errors
**Solution:** Add `--user $(id -u):$(id -g)` to docker run command

**Problem:** dcm2niix conversion fails
**Solution:** Check DICOM integrity, try different dcm2niix version

## Resources

- GitHub: https://github.com/nipy/heudiconv
- Documentation: https://heudiconv.readthedocs.io/
- Example Heuristics: https://github.com/nipy/heudiconv/tree/master/heudiconv/heuristics
- BIDS Specification: https://bids-specification.readthedocs.io/
- Forum: https://neurostars.org/ (tag: heudiconv)

## Example Heuristics Repository

```bash
# Clone example heuristics
git clone https://github.com/nipy/heudiconv.git
cd heudiconv/heudiconv/heuristics

# Available examples:
# - convertall.py - Convert everything
# - reproin.py - ReproIn naming scheme
# - bids_*.py - Various BIDS examples
```

## Citation

```bibtex
@misc{heudiconv,
  title = {HeuDiConv},
  author = {Halchenko, Yaroslav O. and Gorgolewski, Krzysztof J. and others},
  year = {2021},
  url = {https://github.com/nipy/heudiconv},
  doi = {10.5281/zenodo.1012598}
}
```

## Related Tools

- **dcm2niix:** DICOM to NIfTI converter
- **Dcm2Bids:** Alternative DICOM to BIDS converter
- **BIDScoin:** GUI-based BIDS conversion
- **BIDS Validator:** Validate converted data
- **DataLad:** Version control for datasets
- **PyBIDS:** Python library for BIDS
