# BIDS Validator

## Overview

The BIDS Validator is an essential tool for validating Brain Imaging Data Structure (BIDS) formatted datasets. It checks that neuroimaging datasets conform to the BIDS specification, ensuring compatibility with BIDS-compliant analysis tools like fMRIPrep, QSIPrep, and MRIQC.

**Website:** https://github.com/bids-standard/bids-validator
**Platform:** Cross-platform (Node.js/Web/Docker)
**Language:** JavaScript/TypeScript
**License:** MIT

## Key Features

- Comprehensive BIDS specification compliance checking
- File naming validation
- Required file detection
- JSON sidecar validation
- Data type specific rules (anat, func, dwi, fmap, etc.)
- Command-line and web-based interfaces
- Integration with BIDS Apps
- Detailed error and warning messages
- Continuous integration support
- Docker container available

## Installation

### Node.js (Recommended)

```bash
# Install globally
npm install -g bids-validator

# Or using yarn
yarn global add bids-validator

# Verify installation
bids-validator --version
```

### Docker

```bash
# Pull Docker image
docker pull bids/validator

# Run validator
docker run -ti --rm -v /path/to/dataset:/data:ro \
  bids/validator /data
```

### Python (via bids-validator wrapper)

```bash
# Install Python wrapper
pip install bids-validator
```

```python
from bids_validator import BIDSValidator

validator = BIDSValidator()
is_bids = validator.is_bids('/path/to/dataset')
```

## Basic Usage

### Command Line

```bash
# Validate a BIDS dataset
bids-validator /path/to/bids_dataset

# Verbose output
bids-validator /path/to/bids_dataset --verbose

# Ignore specific warnings
bids-validator /path/to/bids_dataset \
  --ignoreWarnings

# Ignore specific error codes
bids-validator /path/to/bids_dataset \
  --ignoreNiftiHeaders

# Generate JSON output
bids-validator /path/to/bids_dataset \
  --json
```

### Web Interface

Visit: https://bids-standard.github.io/bids-validator/

- Drag and drop dataset folder
- Select files from browser
- Completely client-side (no data uploaded)
- Immediate validation feedback

## BIDS Dataset Structure

### Minimal Valid BIDS Dataset

```
my_dataset/
├── dataset_description.json
├── participants.tsv
├── README
├── CHANGES
└── sub-01/
    └── anat/
        ├── sub-01_T1w.nii.gz
        └── sub-01_T1w.json
```

### Required Files

```bash
# dataset_description.json (required)
{
  "Name": "My Dataset",
  "BIDSVersion": "1.8.0",
  "DatasetType": "raw",
  "License": "CC0",
  "Authors": ["Author One", "Author Two"]
}

# participants.tsv (required)
participant_id	age	sex
sub-01	25	F
sub-02	30	M

# README (required)
# Brief description of the dataset
# Citation information
# Contact information

# CHANGES (optional but recommended)
# Version history
1.0.0 2024-01-15
  - Initial release
```

## Validation Output

### Successful Validation

```bash
$ bids-validator /data/my_dataset

This dataset appears to be BIDS compatible.

	Summary:                Available Tasks:
	83 Files, 2.51GB        	rest
	2 - Subjects
	1 - Session

	If you have any questions, please post on https://neurostars.org/tags/bids.
```

### Errors and Warnings

```bash
$ bids-validator /data/my_dataset

1: [ERR] Files with such naming scheme are not part of BIDS specification. This error is most commonly caused by typos in file names that make them not BIDS compatible. Please consult the specification and make sure your files are named correctly. (code: 1 - NOT_INCLUDED)
		./sub-01/anat/sub-01_t1w.nii.gz  # Wrong: should be T1w not t1w

2: [WARN] The recommended file /README is missing. See Section 03 (Modality agnostic files) of the BIDS specification. (code: 99 - README_FILE_MISSING)

Please visit https://neurostars.org/search?q=NOT_INCLUDED for existing conversations about this issue.
```

## Common Validation Issues

### File Naming

```bash
# INCORRECT
sub-01_t1w.nii.gz           # Wrong case
sub-1_T1w.nii.gz            # Missing zero padding
sub-01_anat_T1w.nii.gz      # Extra field
sub01_T1w.nii.gz            # Missing hyphen

# CORRECT
sub-01_T1w.nii.gz
sub-02_T1w.nii.gz
sub-01_ses-01_T1w.nii.gz
```

### Missing Sidecar JSON

```bash
# Each .nii.gz file should have a corresponding .json

# INCORRECT (missing JSON)
sub-01/func/sub-01_task-rest_bold.nii.gz

# CORRECT
sub-01/func/sub-01_task-rest_bold.nii.gz
sub-01/func/sub-01_task-rest_bold.json
```

### Required JSON Fields

```json
// For functional data
{
  "TaskName": "rest",
  "RepetitionTime": 2.0,
  "EchoTime": 0.03,
  "FlipAngle": 90
}

// For fieldmaps
{
  "IntendedFor": ["ses-01/func/sub-01_task-rest_bold.nii.gz"],
  "EchoTime1": 0.00492,
  "EchoTime2": 0.00738
}

// For DWI
{
  "PhaseEncodingDirection": "j-",
  "TotalReadoutTime": 0.05
}
```

## Configuration

### .bidsignore File

```bash
# Create .bidsignore in dataset root
# Similar to .gitignore

# Ignore specific files
*_bad.nii.gz
*.swp
.DS_Store

# Ignore directories
scratch/
backup/
sourcedata/
```

### Config File

```json
// .bids-validator-config.json
{
  "ignore": [
    "INCONSISTENT_PARAMETERS"
  ],
  "warn": [
    "NO_T1W"
  ],
  "error": [],
  "ignoredFiles": [
    "/sub-01/anat/sub-01_T1w_backup.nii.gz"
  ]
}
```

## Integration with Pipelines

### Pre-Processing Check

```bash
#!/bin/bash
# Run validator before preprocessing

BIDS_DIR=/data/bids_dataset

# Validate
bids-validator $BIDS_DIR --json > validation_report.json

# Check if valid
if [ $? -eq 0 ]; then
    echo "Dataset is valid, proceeding with preprocessing"
    fmriprep $BIDS_DIR /output participant
else
    echo "Dataset validation failed, check validation_report.json"
    exit 1
fi
```

### CI/CD Integration

```yaml
# .github/workflows/validate.yml
name: Validate BIDS

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '16'
      - name: Install BIDS Validator
        run: npm install -g bids-validator
      - name: Validate dataset
        run: bids-validator ./ --ignoreWarnings
```

## Python API

```python
from bids_validator import BIDSValidator

# Initialize validator
validator = BIDSValidator()

# Check if path is BIDS
is_bids = validator.is_bids('/path/to/dataset')
print(f"Is BIDS: {is_bids}")

# Validate specific file
is_valid = validator.is_bids('/path/to/dataset/sub-01/anat/sub-01_T1w.nii.gz')

# Custom validation
import subprocess
import json

# Run validator and capture output
result = subprocess.run(
    ['bids-validator', '/path/to/dataset', '--json'],
    capture_output=True,
    text=True
)

validation = json.loads(result.stdout)

# Check for errors
if validation['issues']['errors']:
    print("Errors found:")
    for error in validation['issues']['errors']:
        print(f"  {error['reason']}")
        for file in error['files']:
            print(f"    {file['file']['relativePath']}")
```

## Common Workflows

### Validating New Dataset

```bash
# 1. Check basic structure
tree -L 2 my_dataset/

# 2. Validate
bids-validator my_dataset/ --verbose

# 3. Fix issues iteratively
# ... make corrections ...

# 4. Re-validate
bids-validator my_dataset/

# 5. Document version
echo "1.0.0 $(date +%Y-%m-%d)" >> my_dataset/CHANGES
echo "  - Initial BIDS version" >> my_dataset/CHANGES
```

### Batch Validation

```bash
# Validate multiple datasets
for dataset in /data/studies/*/; do
    echo "Validating $dataset"
    bids-validator "$dataset" > "${dataset}/validation_report.txt"
done

# Check which datasets passed
grep -l "appears to be BIDS compatible" /data/studies/*/validation_report.txt
```

### Automated Fixing

```python
import json
import os
from pathlib import Path

def fix_json_sidecars(bids_dir):
    """Add required fields to JSON sidecars"""
    bids_path = Path(bids_dir)

    # Find all functional NIfTI files
    for nii_file in bids_path.rglob('*_bold.nii.gz'):
        json_file = nii_file.with_suffix('').with_suffix('.json')

        if json_file.exists():
            with open(json_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Add required fields if missing
        if 'TaskName' not in metadata:
            # Extract from filename
            task = nii_file.stem.split('task-')[1].split('_')[0]
            metadata['TaskName'] = task

        if 'RepetitionTime' not in metadata:
            print(f"WARNING: Missing TR for {nii_file}")
            # metadata['RepetitionTime'] = 2.0  # Don't guess!

        # Save updated JSON
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)

fix_json_sidecars('/path/to/bids_dataset')
```

## Advanced Usage

### Ignore Specific Warnings

```bash
# Ignore all warnings
bids-validator /data --ignoreWarnings

# Ignore specific warning codes
bids-validator /data \
  --config.ignore="INCONSISTENT_PARAMETERS" \
  --config.ignore="PARTICIPANT_ID_MISMATCH"
```

### Custom Rules

```json
// custom-rules.json
{
  "ignore": [],
  "warn": ["CUSTOM_WARNING"],
  "error": ["CUSTOM_ERROR"],
  "ignoredFiles": [],
  "custom_rules": {
    "CUSTOM_WARNING": {
      "severity": "warning",
      "reason": "Custom validation failed"
    }
  }
}
```

### Output Formats

```bash
# JSON output for parsing
bids-validator /data --json > validation.json

# Verbose output
bids-validator /data --verbose

# Include file list
bids-validator /data --verbose --json
```

## Integration with Claude Code

When helping users with BIDS Validator:

1. **Check Installation:**
   ```bash
   bids-validator --version
   node --version  # Requires Node.js
   ```

2. **Common Issues:**
   - Node.js not installed
   - Case-sensitive file naming (especially on Windows)
   - Missing required files
   - Incorrect JSON formatting
   - Wrong file extensions

3. **Best Practices:**
   - Validate early and often
   - Use web validator for quick checks
   - Keep .bidsignore for non-BIDS files
   - Document all validation errors
   - Use CI/CD for automated validation
   - Check validator version compatibility

4. **Quick Fixes:**
   - Use consistent naming (uppercase: T1w, BOLD, DWI)
   - Zero-pad subject numbers (sub-01, not sub-1)
   - Include all required metadata in JSON
   - Create participants.tsv with required columns
   - Add dataset_description.json with BIDSVersion

## Troubleshooting

**Problem:** "command not found: bids-validator"
**Solution:** Install with `npm install -g bids-validator` or use Docker

**Problem:** Many "NOT_INCLUDED" errors
**Solution:** Check file naming scheme, ensure correct case (T1w not t1w)

**Problem:** JSON validation errors
**Solution:** Validate JSON with `jsonlint` or online validator

**Problem:** IntendedFor field errors
**Solution:** Use relative paths from dataset root, check file exists

**Problem:** Validator hangs on large datasets
**Solution:** Use `--ignoreNiftiHeaders` flag, check for corrupted files

## Resources

- GitHub: https://github.com/bids-standard/bids-validator
- Web Validator: https://bids-standard.github.io/bids-validator/
- BIDS Specification: https://bids-specification.readthedocs.io/
- BIDS Starter Kit: https://github.com/bids-standard/bids-starter-kit
- Forum: https://neurostars.org/tags/bids

## BIDS Specification Versions

```bash
# Check which BIDS version validator supports
bids-validator --version

# Common BIDS versions
# 1.0.0 - Initial release
# 1.4.0 - Added PET
# 1.6.0 - Added genetics
# 1.7.0 - Added microscopy
# 1.8.0 - Current stable

# Validate against specific version
# (Use matching validator version)
```

## Related Tools

- **HeuDiConv:** Convert DICOM to BIDS
- **Dcm2Bids:** Alternative DICOM converter
- **BIDScoin:** Interactive BIDS conversion
- **PyBIDS:** Python library for BIDS datasets
- **BIDS Apps:** Analysis tools for BIDS data
