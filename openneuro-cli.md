# OpenNeuro CLI: Command-Line Interface for OpenNeuro

## Overview

OpenNeuro CLI is a command-line interface for interacting with OpenNeuro.org, a free and open platform for sharing BIDS-formatted neuroimaging data. The CLI enables researchers to programmatically upload datasets, download public datasets, manage dataset metadata, control access permissions, and automate data sharing workflows. OpenNeuro hosts thousands of neuroimaging datasets following the Brain Imaging Data Structure (BIDS) standard, making it a critical resource for open science, meta-analyses, methods validation, and collaborative research.

The OpenNeuro platform democratizes access to neuroimaging data by providing permanent, version-controlled dataset storage with DOI assignment for citation. The CLI complements the web interface by enabling scripted workflows for large-scale data sharing, automated downloads for computational pipelines, and integration with high-performance computing environments where GUI access is limited.

**Key Features:**
- Download public datasets with resumable transfers
- Upload BIDS datasets with version control
- Manage dataset metadata and descriptions
- Control dataset access (public vs. private)
- Automated snapshot creation and DOI assignment
- Query dataset information and statistics
- Integration with DataLad for versioned data management
- Support for large datasets (>100 GB) with efficient transfers
- Automated BIDS validation before upload

**Primary Use Cases:**
- Downloading public datasets for research or meta-analysis
- Uploading and sharing new neuroimaging datasets
- Automating dataset downloads in processing pipelines
- Batch downloading multiple datasets
- Managing dataset versions and snapshots
- Collaborative data sharing with controlled access
- Integrating OpenNeuro with institutional data repositories

**Citation:**
```
Markiewicz, C. J., Gorgolewski, K. J., Feingold, F., Blair, R., Halchenko, Y. O.,
Miller, E., ... & Poldrack, R. A. (2021). The OpenNeuro resource for sharing of
neuroscience data. eLife, 10, e71774.
```

## Installation

### Installation via pip (Recommended)

```bash
# Install OpenNeuro CLI
pip install openneuro-cli

# Verify installation
openneuro --version

# Expected output: openneuro-cli, version 0.17.0
```

### Installation via conda

```bash
# Create environment with OpenNeuro CLI
conda create -n openneuro python=3.9
conda activate openneuro

# Install via pip
pip install openneuro-cli
```

### Installation from Source

```bash
# Clone repository
git clone https://github.com/OpenNeuroOrg/openneuro-cli.git
cd openneuro-cli

# Install in development mode
pip install -e .

# Test installation
openneuro --help
```

### Authentication Setup

```bash
# Create OpenNeuro account at https://openneuro.org

# Generate API key:
# 1. Log in to OpenNeuro.org
# 2. Go to Settings → Obtain an API Key
# 3. Copy the generated key

# Set API key as environment variable (recommended)
export OPENNEURO_API_KEY="your_api_key_here"

# Or store in config file
mkdir -p ~/.openneuro
echo "api_key: your_api_key_here" > ~/.openneuro/config.yml

# Test authentication
openneuro datasets --my-datasets
```

## Downloading Datasets

### Basic Dataset Download

**Example 1: Download Complete Public Dataset**

```bash
# Download dataset by accession number
openneuro download ds000001

# Downloads to: ./ds000001/
# - All BIDS files
# - Dataset description
# - Participant metadata
# - README and CHANGES files

# The dataset ds000001 is "Balloon Analog Risk-taking Task"
# Contains: 16 subjects, task fMRI data
```

**Example 2: Download to Specific Directory**

```bash
# Download to custom location
openneuro download ds000001 --output /data/openneuro/ds000001

# With verbose output
openneuro download ds000001 \
  --output /data/openneuro/ds000001 \
  --verbose

# Progress shows:
# - Files being downloaded
# - Transfer speed
# - Remaining time
# - Total size
```

**Example 3: Download Specific Version (Snapshot)**

```bash
# List available versions
openneuro datasets --dataset ds000001

# Download specific snapshot version
openneuro download ds000001:1.0.0

# Or latest version explicitly
openneuro download ds000001:latest

# Snapshots are immutable and have DOIs
# Important for reproducibility
```

### Selective Downloads

**Example 4: Download Only Specific Subjects**

```bash
# Download only subjects 01, 02, and 03
openneuro download ds000001 \
  --include "sub-01/*" \
  --include "sub-02/*" \
  --include "sub-03/*"

# Exclude specific subjects
openneuro download ds000001 \
  --exclude "sub-04/*" \
  --exclude "sub-05/*"

# Download only anatomical data
openneuro download ds000001 \
  --include "*/anat/*.nii.gz"
```

**Example 5: Download Only Derivatives**

```bash
# Download preprocessed data (if available)
openneuro download ds000001 \
  --include "derivatives/*"

# Common derivatives:
# - derivatives/fmriprep/  (fMRIPrep outputs)
# - derivatives/freesurfer/ (FreeSurfer outputs)
# - derivatives/mriqc/ (Quality control metrics)

# Download specific derivative pipeline
openneuro download ds000001 \
  --include "derivatives/fmriprep/*"
```

**Example 6: Resume Interrupted Downloads**

```bash
# Downloads automatically resume from where they stopped
# If download interrupted:

# Simply re-run the same command
openneuro download ds000001

# CLI detects existing files and skips them
# Only missing or incomplete files are downloaded

# Force re-download (overwrite existing)
openneuro download ds000001 --force
```

## Uploading Datasets

### Creating and Uploading New Dataset

**Example 7: Upload BIDS Dataset**

```bash
# Prerequisites:
# 1. Dataset organized in BIDS format
# 2. BIDS validation passed
# 3. OpenNeuro account with API key

# Validate BIDS structure first
bids-validator /path/to/my_dataset

# Create new dataset on OpenNeuro
openneuro create /path/to/my_dataset

# Output: Dataset created: ds00XXXX
# Returns new dataset accession number

# Upload will:
# - Validate BIDS structure
# - Upload all files
# - Create initial version
# - Dataset initially private
```

**Example 8: Upload with Metadata**

```bash
# Create dataset with full metadata
openneuro create /path/to/my_dataset \
  --dataset-name "My Awesome fMRI Study" \
  --dataset-description "Task fMRI during cognitive control" \
  --authors "Smith J, Doe J, Johnson M" \
  --license "CC0" \
  --acknowledgements "Funded by NIH grant R01..."

# Metadata fields:
# - Name: Short descriptive title
# - Description: Detailed dataset description
# - Authors: Comma-separated list
# - License: CC0, PDDL, CC-BY-4.0
# - Acknowledgements: Funding, support
# - References: Related publications
```

**Example 9: Update Existing Dataset**

```bash
# Upload new data to existing dataset
openneuro upload ds00XXXX /path/to/updated_dataset

# This creates a new draft version
# Original version remains unchanged

# Add new subjects to existing dataset
openneuro upload ds00XXXX \
  --include "sub-17/*" \
  --include "sub-18/*"

# Update only specific files
openneuro upload ds00XXXX \
  --include "derivatives/fmriprep/*"
```

### Managing Dataset Versions

**Example 10: Create Snapshot (Immutable Version)**

```bash
# Create snapshot from draft version
openneuro snapshot ds00XXXX

# Snapshots:
# - Are immutable (cannot be changed)
# - Receive version number (e.g., 1.0.0)
# - Get assigned a DOI for citation
# - Can be referenced in publications

# Specify tag for snapshot
openneuro snapshot ds00XXXX --tag "initial-release"

# Snapshot versioning follows semantic versioning:
# - Major: Incompatible changes
# - Minor: Backward-compatible additions
# - Patch: Backward-compatible fixes
```

**Example 11: Delete Draft Changes**

```bash
# Discard uncommitted changes
openneuro dataset --dataset ds00XXXX --delete-draft

# Reverts dataset to last snapshot
# Use with caution - cannot be undone

# Check draft status before deleting
openneuro dataset --dataset ds00XXXX
```

## Dataset Management

### Querying Dataset Information

**Example 12: List Your Datasets**

```bash
# List all your datasets
openneuro datasets --my-datasets

# Output:
# ds00123 - My fMRI Study (private, draft)
# ds00456 - Resting State Cohort (public, 1.0.0)
# ds00789 - Longitudinal Study (public, 2.1.0)

# List public datasets (first 50)
openneuro datasets --public

# Search for specific datasets
openneuro datasets --search "resting state"

# Filter by modality
openneuro datasets --modality fMRI
openneuro datasets --modality DWI
```

**Example 13: Get Dataset Details**

```bash
# Get detailed information about dataset
openneuro dataset --dataset ds000001

# Returns:
# - Dataset name and description
# - Authors
# - Number of subjects
# - Data size
# - Modalities (T1w, bold, dwi, etc.)
# - Available versions/snapshots
# - DOI (if snapshot)
# - License
# - Download count

# Get as JSON for parsing
openneuro dataset --dataset ds000001 --json > ds000001_info.json
```

**Example 14: View Dataset Files**

```bash
# List all files in dataset
openneuro files --dataset ds000001

# List with sizes
openneuro files --dataset ds000001 --sizes

# Output files to text file
openneuro files --dataset ds000001 > ds000001_files.txt

# Count files and total size
openneuro files --dataset ds000001 --sizes | \
  awk '{sum+=$1; count++} END {print count " files, " sum/1e9 " GB"}'
```

### Access Control

**Example 15: Making Dataset Public**

```bash
# Change dataset visibility to public
openneuro dataset --dataset ds00XXXX --public

# Dataset becomes publicly downloadable
# Cannot be made private again once public

# Check current visibility
openneuro dataset --dataset ds00XXXX

# Best practice:
# 1. Keep private during data collection/QC
# 2. Make public after publication acceptance
# 3. Create snapshot before making public
```

**Example 16: Managing Permissions**

```bash
# Add collaborator by email
openneuro permissions --dataset ds00XXXX \
  --add collaborator@university.edu \
  --role editor

# Roles:
# - viewer: Can view private dataset
# - editor: Can upload/modify files
# - admin: Can change permissions, make public

# List current permissions
openneuro permissions --dataset ds00XXXX --list

# Remove collaborator
openneuro permissions --dataset ds00XXXX \
  --remove collaborator@university.edu
```

## Batch Operations

**Example 17: Download Multiple Datasets**

```bash
# Create list of datasets to download
cat > datasets.txt << EOF
ds000001
ds000002
ds000003
ds000004
ds000005
EOF

# Batch download
while read dataset; do
  echo "Downloading $dataset"
  openneuro download $dataset --output /data/openneuro/$dataset
done < datasets.txt

# With error handling
while read dataset; do
  if openneuro download $dataset --output /data/openneuro/$dataset; then
    echo "✓ $dataset completed"
  else
    echo "✗ $dataset failed" >> failed_downloads.txt
  fi
done < datasets.txt
```

**Example 18: Automated Pipeline Integration**

```python
# Python script for automated dataset processing
import subprocess
import os

def download_and_process(dataset_id, output_dir):
    """Download dataset and run processing pipeline."""

    # Download dataset
    download_cmd = [
        'openneuro', 'download', dataset_id,
        '--output', output_dir
    ]

    print(f"Downloading {dataset_id}...")
    result = subprocess.run(download_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Download failed: {result.stderr}")
        return False

    # Run processing (e.g., fMRIPrep)
    bids_dir = os.path.join(output_dir, dataset_id)
    derivatives_dir = os.path.join(output_dir, 'derivatives')

    process_cmd = [
        'fmriprep',
        bids_dir,
        derivatives_dir,
        'participant',
        '--fs-license-file', '/license.txt'
    ]

    print(f"Processing {dataset_id}...")
    result = subprocess.run(process_cmd)

    return result.returncode == 0

# Download and process multiple datasets
datasets = ['ds000001', 'ds000002', 'ds000003']

for ds in datasets:
    success = download_and_process(ds, '/data/openneuro')
    if success:
        print(f"✓ {ds} completed successfully")
    else:
        print(f"✗ {ds} failed")
```

## Advanced Features

**Example 19: Using DataLad for Version Control**

```bash
# OpenNeuro integrates with DataLad
# DataLad provides Git-like version control for data

# Install DataLad
pip install datalad

# Clone dataset as DataLad repository
datalad install -s https://github.com/OpenNeuroDatasets/ds000001.git

# Download specific files on-demand
cd ds000001
datalad get sub-01/anat/*

# Update to newer version
datalad update --merge

# Benefits:
# - Efficient storage (only download what you need)
# - Full version history
# - Reproducible data provenance
```

**Example 20: Programmatic API Access**

```python
# Use OpenNeuro GraphQL API directly
import requests

# GraphQL endpoint
url = "https://openneuro.org/crn/graphql"

# Query for dataset info
query = """
query {
  dataset(id: "ds000001") {
    id
    name
    description
    created
    modified
    public
    latestSnapshot {
      tag
      created
    }
    analytics {
      downloads
      views
    }
  }
}
"""

response = requests.post(url, json={'query': query})
data = response.json()

print(f"Dataset: {data['data']['dataset']['name']}")
print(f"Description: {data['data']['dataset']['description']}")
print(f"Downloads: {data['data']['dataset']['analytics']['downloads']}")
```

**Example 21: Automated Metadata Updates**

```python
# Update dataset metadata programmatically
import json

# Prepare metadata
metadata = {
    "Name": "Updated Study Name",
    "BIDSVersion": "1.6.0",
    "Authors": [
        "Smith, John",
        "Doe, Jane",
        "Johnson, Michael"
    ],
    "License": "CC0",
    "Acknowledgements": "Funded by NIH grant R01MH123456",
    "ReferencesAndLinks": [
        "Smith et al. (2023). Journal of Neuroscience.",
        "https://example.com/project"
    ],
    "DatasetDOI": "10.18112/openneuro.ds00XXXX.v1.0.0"
}

# Save as dataset_description.json
with open('dataset_description.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Upload updated metadata
# openneuro upload ds00XXXX --include "dataset_description.json"
```

## Quality Control and Validation

**Example 22: Pre-Upload BIDS Validation**

```bash
# Always validate before upload
# Install BIDS validator
npm install -g bids-validator

# Validate dataset
bids-validator /path/to/my_dataset

# Common validation errors:
# - Missing required files (dataset_description.json)
# - Incorrect file naming
# - Missing or malformed JSON sidecars
# - Invalid BIDS version

# Fix errors before uploading
# OpenNeuro will reject invalid datasets

# Validate with specific BIDS version
bids-validator /path/to/my_dataset --bids-version 1.6.0
```

## Troubleshooting

### Authentication Issues

```bash
# If authentication fails:

# 1. Verify API key is set
echo $OPENNEURO_API_KEY

# 2. Check API key validity
openneuro datasets --my-datasets

# If error "Unauthorized":
# - Generate new API key on OpenNeuro.org
# - Update environment variable

# 3. Clear cached credentials
rm -rf ~/.openneuro/

# 4. Re-authenticate
export OPENNEURO_API_KEY="new_api_key"
```

### Download Failures

```bash
# If downloads fail or hang:

# 1. Check internet connection
ping openneuro.org

# 2. Check disk space
df -h /path/to/download/location

# 3. Use smaller chunks (for unstable connections)
openneuro download ds000001 \
  --include "sub-01/*"  # One subject at a time

# 4. Check dataset availability
openneuro dataset --dataset ds000001

# 5. Try different time (server may be busy)
```

### Upload Issues

```bash
# If uploads fail:

# 1. Validate BIDS first
bids-validator /path/to/dataset

# 2. Check file permissions
ls -la /path/to/dataset

# 3. Check dataset size limits
# OpenNeuro supports datasets up to 200 GB
# For larger datasets, contact support

# 4. Upload in chunks
openneuro upload ds00XXXX --include "sub-01/*"
openneuro upload ds00XXXX --include "sub-02/*"
# etc.

# 5. Check API rate limits
# Wait a few minutes between large uploads
```

## Best Practices

**Data Sharing:**
- Validate with bids-validator before uploading
- Include comprehensive dataset_description.json
- Write clear README with study description
- Add CHANGES file for version history
- Defaceanatomical images before uploading
- Remove or anonymize participant metadata
- Use snapshots for publications (citable DOIs)

**Downloading Data:**
- Always specify version/snapshot for reproducibility
- Check dataset license before use
- Cite datasets in publications (use DOI)
- Download to fast storage (SSD) for processing
- Keep dataset_description.json with downloaded data

**Dataset Management:**
- Keep datasets private during data collection
- Create snapshots at major milestones
- Make public after publication acceptance
- Use meaningful version tags
- Document changes in CHANGES file

**Collaboration:**
- Add collaborators before making public
- Use appropriate permission levels
- Communicate before making datasets public
- Plan snapshot timing with co-authors

## Integration with Analysis Tools

**fMRIPrep:**
```bash
# Download dataset and run fMRIPrep
openneuro download ds000001 --output /data

fmriprep /data/ds000001 /data/derivatives participant \
  --fs-license-file /license.txt \
  --output-spaces MNI152NLin2009cAsym:res-2 \
  --nthreads 8
```

**MRIQC:**
```bash
# Run quality control on downloaded dataset
openneuro download ds000001 --output /data

mriqc /data/ds000001 /data/mriqc participant \
  --nprocs 8 \
  --mem_gb 16
```

**BIDS Apps:**
```bash
# Run any BIDS App on OpenNeuro dataset
openneuro download ds000001 --output /data

docker run -it --rm \
  -v /data/ds000001:/data:ro \
  -v /data/derivatives:/out \
  bids/freesurfer \
  /data /out participant \
  --participant_label 01 02 03
```

**Meta-Analysis:**
```python
# Download multiple datasets for meta-analysis
import subprocess
import pandas as pd

datasets = ['ds000001', 'ds000002', 'ds000003']
output_dir = '/data/meta_analysis'

for ds in datasets:
    # Download dataset
    subprocess.run([
        'openneuro', 'download', ds,
        '--output', f'{output_dir}/{ds}'
    ])

    # Extract participant metadata
    participants = pd.read_csv(
        f'{output_dir}/{ds}/participants.tsv',
        sep='\t'
    )

    # Combine metadata for meta-analysis
    # ...
```

## References

**OpenNeuro:**
- Markiewicz et al. (2021). The OpenNeuro resource for sharing of neuroscience data. *eLife*, 10, e71774.
- Poldrack & Gorgolewski (2014). Making big data open: Data sharing in neuroimaging. *Nature Neuroscience*, 17(11), 1510-1517.

**BIDS:**
- Gorgolewski et al. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. *Scientific Data*, 3, 160044.

**DataLad:**
- Halchenko et al. (2021). DataLad: Distributed system for joint management of code, data, and their relationship. *Journal of Open Source Software*, 6(63), 3262.

**Data Sharing:**
- Poldrack et al. (2017). Scanning the horizon: Towards transparent and reproducible neuroimaging research. *Nature Reviews Neuroscience*, 18(2), 115-126.

**Online Resources:**
- OpenNeuro Platform: https://openneuro.org
- OpenNeuro CLI Documentation: https://github.com/OpenNeuroOrg/openneuro-cli
- BIDS Specification: https://bids-specification.readthedocs.io/
- DataLad Documentation: http://docs.datalad.org/
- OpenNeuro GraphQL API: https://docs.openneuro.org/
