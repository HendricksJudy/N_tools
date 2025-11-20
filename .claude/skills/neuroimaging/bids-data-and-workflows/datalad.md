# DataLad

## Overview

DataLad is a free and open-source distributed data management system that provides a unified interface for managing, sharing, and version controlling datasets of any size. Built on top of git and git-annex, DataLad enables reproducible data science by tracking data, code, and computational environments together while efficiently handling large files.

**Website:** https://www.datalad.org/
**Platform:** Cross-platform (Linux/macOS/Windows)
**Language:** Python
**License:** MIT

## Key Features

- Version control for large datasets (TB-scale)
- Lightweight metadata tracking (no full file duplication)
- Decentralized data distribution
- Automated data retrieval on demand
- Integration with data repositories (OpenNeuro, OSF, etc.)
- BIDS dataset support
- Provenance tracking for reproducibility
- Command-line and Python API
- Integration with containers (Docker/Singularity)
- Seamless collaboration and sharing

## Installation

### Linux/macOS

```bash
# Using pip
pip install datalad

# Using conda
conda install -c conda-forge datalad

# Ubuntu/Debian
sudo apt-get install datalad

# macOS with Homebrew
brew install datalad
```

### Windows

```bash
# Using conda (recommended)
conda install -c conda-forge datalad

# Or pip
pip install datalad
```

### Additional Dependencies

```bash
# git-annex (required)
# Linux
sudo apt-get install git-annex

# macOS
brew install git-annex

# Verify installation
datalad --version
git annex version
```

## Core Concepts

### Datasets

```bash
# Create a new dataset
datalad create my_dataset
cd my_dataset

# Dataset is a git repository + git-annex
# .git/ - contains version control
# .datalad/ - contains DataLad configuration

# Add files (small files go to git, large to git-annex)
echo "README content" > README.md
datalad save -m "Add README"

# Add large file (automatically annexed)
cp /data/large_scan.nii.gz data/
datalad save -m "Add scan data"

# Check status
datalad status
```

### Installing Datasets

```bash
# Clone/install a dataset (gets metadata only)
datalad clone https://github.com/OpenNeuroDatasets/ds000001.git

# Or from OpenNeuro
datalad clone ///openneuro/ds000001

cd ds000001

# Get specific files when needed
datalad get sub-01/anat/sub-01_T1w.nii.gz

# Get all files in a directory
datalad get sub-01/

# Get everything
datalad get .

# Drop files to save space (keeps metadata)
datalad drop sub-01/anat/sub-01_T1w.nii.gz
```

## Working with Neuroimaging Data

### BIDS Dataset Management

```bash
# Create BIDS dataset
datalad create -c bids my_bids_dataset
cd my_bids_dataset

# Structure is created automatically
ls -la
# dataset_description.json
# README
# CHANGES
# participants.tsv

# Add subject data
mkdir -p sub-01/anat
cp /source/T1.nii.gz sub-01/anat/sub-01_T1w.nii.gz

# Save with meaningful message
datalad save -m "Add sub-01 anatomical scan"

# Add metadata
datalad meta-add dataset_description.json \
  -c Name="My Study" \
  -c BIDSVersion="1.6.0"
```

### Subdatasets

```bash
# Install dataset with subdatasets
datalad clone https://github.com/psychoinformatics-de/studyforrest-data-phase2.git
cd studyforrest-data-phase2

# List subdatasets
datalad subdatasets

# Install specific subdataset
datalad get -n sub-01  # -n: no data, just metadata

# Get data from subdataset
datalad get sub-01/ses-movie/func/
```

## Provenance and Reproducibility

### Run Command with Provenance

```bash
# Run analysis with full provenance tracking
datalad run \
  -m "Skull strip sub-01" \
  -i sub-01/anat/sub-01_T1w.nii.gz \
  -o sub-01/anat/sub-01_T1w_brain.nii.gz \
  "bet sub-01/anat/sub-01_T1w.nii.gz sub-01/anat/sub-01_T1w_brain.nii.gz"

# DataLad tracks:
# - Input files
# - Output files
# - Command executed
# - Environment

# Rerun the command
datalad rerun

# Rerun on different subject
datalad rerun --script - <<EOF
sub=02
bet sub-${sub}/anat/sub-${sub}_T1w.nii.gz \
    sub-${sub}/anat/sub-${sub}_T1w_brain.nii.gz
EOF
```

### Containers Integration

```bash
# Add container to dataset
datalad containers-add fmriprep \
  --url docker://nipreps/fmriprep:latest

# Run analysis in container
datalad containers-run \
  -m "Run fMRIPrep on sub-01" \
  --container-name fmriprep \
  -i bids_dataset/ \
  -o derivatives/fmriprep/ \
  "bids_dataset/ derivatives/fmriprep/ participant --participant-label 01"

# Container image is tracked in dataset
# Analysis is fully reproducible
```

## Data Sharing and Collaboration

### Sibling Repositories

```bash
# Create sibling on GitHub
datalad create-sibling-github my_dataset \
  --github-organization my-org \
  --access-protocol ssh

# Or GitLab
datalad create-sibling-gitlab my_dataset \
  --site https://gitlab.com

# Push dataset
datalad push --to github

# Configure special remote for large files (e.g., S3)
git annex initremote s3-storage \
  type=S3 \
  encryption=none \
  bucket=my-data-bucket \
  public=yes

# Publish large files to S3
datalad push --to s3-storage
```

### Downloading from OpenNeuro

```bash
# Install OpenNeuro dataset
datalad clone ///openneuro/ds000114

cd ds000114

# Get specific subject
datalad get sub-01/

# Process and save derivatives
mkdir -p derivatives/fmriprep
datalad run -m "Run fMRIPrep sub-01" \
  -i sub-01/ \
  -o derivatives/fmriprep/sub-01/ \
  "fmriprep sub-01 derivatives/fmriprep participant --participant-label 01"

# Share your derivatives
datalad push
```

## Python API

### Basic Operations

```python
import datalad.api as dl

# Create dataset
ds = dl.create('my_dataset')

# Add file
ds.save(path='data/file.txt', message='Add data file')

# Get file
dl.get('data/large_file.nii.gz', dataset='.')

# Clone dataset
ds = dl.clone(source='https://github.com/user/dataset.git',
              path='local_dataset')

# Install subdatasets
ds.get(path='sub-01', get_data=False)  # Metadata only
ds.get(path='sub-01')  # With data
```

### Running Analyses

```python
import datalad.api as dl

# Run command with provenance
ds = dl.Dataset('.')

ds.run(
    cmd='bet {inputs} {outputs} -f 0.5',
    inputs=['sub-01/anat/sub-01_T1w.nii.gz'],
    outputs=['sub-01/anat/sub-01_T1w_brain.nii.gz'],
    message='Brain extraction sub-01'
)

# Run with container
ds.containers_run(
    cmd='mrconvert {inputs} {outputs}',
    container_name='mrtrix3',
    inputs=['sub-01/dwi/sub-01_dwi.nii.gz'],
    outputs=['sub-01/dwi/sub-01_dwi.mif'],
    message='Convert to MIF format'
)
```

### Batch Processing

```python
import datalad.api as dl
from pathlib import Path

ds = dl.Dataset('.')

# Process all subjects
subjects = [d.name for d in Path('bids_dataset').glob('sub-*')]

for sub in subjects:
    ds.run(
        cmd=f'fmriprep bids_dataset/ derivatives/ participant --participant-label {sub[4:]}',
        inputs=[f'bids_dataset/{sub}/'],
        outputs=[f'derivatives/fmriprep/{sub}/'],
        message=f'Process {sub}'
    )
```

## Advanced Features

### Configuration

```bash
# Configure git-annex backend for large files
datalad create -c text2git my_dataset
cd my_dataset

# Only text files go to git, binaries to annex
echo "*.txt annex.largefiles=nothing" > .gitattributes
echo "*.nii.gz annex.largefiles=anything" >> .gitattributes
datalad save -m "Configure annex rules"

# Set default location for annexed files
git annex wanted here "include=sub-*/anat/*"
git annex wanted origin "exclude=sub-*/func/*"
```

### Metadata Management

```bash
# Extract metadata
datalad meta-extract -d . extractors.bids

# Aggregate metadata
datalad meta-aggregate -d .

# Search metadata
datalad search "T1w"
datalad search --mode grep "task-rest"
```

### Working Offline

```bash
# Get all data you need
datalad get sub-01/

# Work offline
# ... make changes ...

# When back online, sync
datalad save -m "Processed sub-01"
datalad push
```

## Common Workflows

### Complete Analysis Pipeline

```bash
# 1. Install source dataset
datalad clone ///openneuro/ds000114 sourcedata
cd sourcedata
datalad get sub-01/

# 2. Create analysis superdataset
cd ..
datalad create analysis
cd analysis

# 3. Add source data as subdataset
datalad clone -d . ../sourcedata inputs/rawdata

# 4. Add container
datalad containers-add fmriprep \
  --url docker://nipreps/fmriprep:latest

# 5. Run preprocessing
datalad containers-run \
  -m "Preprocess sub-01" \
  --container-name fmriprep \
  -i inputs/rawdata/sub-01/ \
  -o outputs/fmriprep/sub-01/ \
  "inputs/rawdata/ outputs/fmriprep/ participant --participant-label 01"

# 6. Share results
datalad create-sibling-github analysis-results
datalad push --to github
```

### Collaborative Project

```bash
# Collaborator A: Create and share
datalad create shared_project
cd shared_project
datalad create-sibling-github shared_project --access-protocol ssh

# Add data and push
cp data/* .
datalad save -m "Initial data"
datalad push --to github

# Collaborator B: Clone and contribute
datalad clone git@github.com:user/shared_project.git
cd shared_project
datalad get .

# Add analysis
mkdir analysis
# ... do work ...
datalad save -m "Add analysis"
datalad push

# Collaborator A: Get updates
datalad update --merge
datalad get analysis/
```

## Integration with Claude Code

When helping users with DataLad:

1. **Check Installation:**
   ```bash
   datalad --version
   git annex version
   ```

2. **Common Issues:**
   - git-annex not installed
   - SSH keys not configured for remotes
   - Large files not being annexed
   - Merge conflicts with annexed files

3. **Best Practices:**
   - Use meaningful commit messages
   - Configure .gitattributes for file types
   - Use `datalad run` for reproducibility
   - Regular `datalad save` after changes
   - Test on small datasets first
   - Use subdatasets for organization
   - Document provenance

4. **Performance:**
   - Use `--jobs` for parallel operations
   - Configure git-annex for faster operations
   - Use `drop` to manage disk space
   - Shallow clones for large histories

## Troubleshooting

**Problem:** "git-annex not found"
**Solution:** Install git-annex: `sudo apt-get install git-annex` or `brew install git-annex`

**Problem:** Large files committed to git instead of annex
**Solution:** Configure `.gitattributes` with `annex.largefiles` rules

**Problem:** Cannot get files from remote
**Solution:** Check remote configuration, SSH keys, network connectivity

**Problem:** Merge conflicts with annexed files
**Solution:** Use `git annex sync` or `datalad update --merge`

**Problem:** Out of disk space
**Solution:** Use `datalad drop` to remove file content while keeping metadata

## Resources

- Website: https://www.datalad.org/
- Handbook: http://handbook.datalad.org/
- Documentation: https://docs.datalad.org/
- GitHub: https://github.com/datalad/datalad
- YouTube: DataLad Tutorial Series
- Forum: https://neurostars.org/ (tag: datalad)

## Use Cases

- **Data sharing:** Share large datasets efficiently
- **Version control:** Track changes to data over time
- **Reproducibility:** Record complete computational provenance
- **Collaboration:** Enable team-based data analysis
- **Data publication:** Publish datasets with full history
- **Pipeline development:** Build reproducible analysis pipelines

## Citation

```bibtex
@software{datalad,
  title = {DataLad},
  author = {Halchenko, Yaroslav O. and Hanke, Michael and others},
  year = {2021},
  url = {https://www.datalad.org},
  doi = {10.5281/zenodo.808846}
}
```

## Related Tools

- **git-annex:** Underlying technology for large file management
- **BIDS:** Brain Imaging Data Structure
- **OpenNeuro:** Neuroimaging data repository
- **OSF:** Open Science Framework
- **ReproNim:** Center for Reproducible Neuroimaging Computation
