# Snakebids - BIDS-Aware Snakemake Workflows

## Overview

Snakebids is a Python framework for creating BIDS-compliant neuroimaging pipelines using Snakemake. Developed by Khan Lab at Western University, Snakebids automatically parses BIDS datasets and generates Snakemake wildcards from BIDS entities (subject, session, acquisition, etc.), enabling scalable, reproducible workflows that seamlessly integrate with the BIDS ecosystem. It provides a template structure for building BIDS apps with minimal boilerplate while leveraging Snakemake's powerful workflow management capabilities.

**Website:** https://github.com/akhanf/snakebids
**Platform:** Python/Snakemake (Cross-platform)
**License:** MIT
**Key Application:** BIDS app development, reproducible neuroimaging pipelines, HPC deployment

### Why Snakebids?

**Advantages:**
- **Automatic BIDS parsing** - No manual file pattern matching needed
- **Entity-based wildcards** - Use {subject}, {session}, etc. naturally
- **Config-driven** - Flexible pipeline configuration
- **Snakemake integration** - Leverage mature workflow engine
- **Container support** - Singularity/Docker per rule
- **HPC-ready** - SLURM, PBS, SGE cluster profiles
- **BIDS app template** - Quick pipeline development

**Key Concepts:**
- BidsComponent: Automatic input specification
- BidsPartialComponent: Flexible BIDS queries
- Config-driven input selection
- Participant/group analysis levels
- Derivative organization

## Key Features

- **BIDS-aware input handling** - Automatic entity extraction
- **Snakemake workflow engine** - Parallel, reproducible execution
- **Template project structure** - Scaffolding for BIDS apps
- **Config file flexibility** - YAML-based configuration
- **Container integration** - Per-rule Singularity/Docker
- **Cluster profiles** - HPC job submission
- **Participant/group levels** - Standard BIDS app structure
- **Derivative organization** - BIDS-compliant outputs
- **Command-line interface** - Auto-generated from config
- **Quality control hooks** - Integrate QC steps
- **Modular rules** - Reusable workflow components
- **Version control friendly** - Git-compatible structure

## Installation

### Basic Installation

```bash
# Install Snakebids
pip install snakebids

# Or with all extras
pip install snakebids[all]

# Install Snakemake (required)
pip install snakemake

# Verify installation
snakebids --version
```

### With Container Support

```bash
# For Singularity support
pip install snakebids[singularity]
conda install -c conda-forge singularity

# For both Singularity and visualization
pip install snakebids[all]
```

### Create New Project

```bash
# Create BIDS app from template
snakebids create my_bids_app

# Navigate to project
cd my_bids_app

# Project structure created:
# my_bids_app/
#   ├── config/
#   │   └── snakebids.yml
#   ├── workflow/
#   │   ├── Snakefile
#   │   └── rules/
#   ├── my_bids_app/
#   │   └── run.py
#   └── setup.py
```

## Project Structure

### Standard Snakebids Layout

```bash
my_pipeline/
├── config/
│   ├── snakebids.yml          # Main configuration
│   └── config.yml             # Additional settings
├── workflow/
│   ├── Snakefile              # Main workflow file
│   ├── rules/                 # Modular rule files
│   │   ├── preprocessing.smk
│   │   └── analysis.smk
│   └── scripts/               # Helper scripts
├── my_pipeline/
│   ├── __init__.py
│   └── run.py                 # CLI entry point
├── resources/                 # Templates, atlases
├── results/                   # Output directory
└── setup.py                   # Package installation
```

## Configuration

### Basic Config (snakebids.yml)

```yaml
# config/snakebids.yml
bids_dir: '/path/to/bids/dataset'
output_dir: 'results'

# BIDS input specification
pybids_inputs:
  t1w:
    filters:
      suffix: 'T1w'
      extension: '.nii.gz'
      datatype: 'anat'
    wildcards:
      - subject
      - session
      - acquisition
      - run

# Parse arguments
parse_args:
  bids_dir:
    help: 'Input BIDS dataset directory'
  output_dir:
    help: 'Output directory for derivatives'
  
# Analysis levels
analysis_levels:
  - participant
  - group

# Participant selection
participants:
  - all  # Or specific: ['sub-01', 'sub-02']

# Exclude participants
exclude_participants: []

# Container settings
singularity:
  fsl: 'docker://brainlife/fsl:6.0.4'
  freesurfer: 'docker://freesurfer/freesurfer:7.3.2'
```

### Advanced Config

```yaml
# Multi-modal inputs
pybids_inputs:
  # T1-weighted
  t1w:
    filters:
      suffix: 'T1w'
      extension: '.nii.gz'
      datatype: 'anat'
    wildcards:
      - subject
      - session
  
  # T2-weighted  
  t2w:
    filters:
      suffix: 'T2w'
      extension: '.nii.gz'
      datatype: 'anat'
    wildcards:
      - subject
      - session
  
  # Bold fMRI
  bold:
    filters:
      suffix: 'bold'
      extension: '.nii.gz'
      datatype: 'func'
    wildcards:
      - subject
      - session
      - task
      - run

# Derivatives as input
pybids_derivatives:
  fmriprep:
    filters:
      suffix: 'preproc'
      desc: 'preproc'
      space: 'MNI152NLin2009cAsym'
    wildcards:
      - subject
      - session
```

## Workflow Development

### Basic Snakefile

```python
# workflow/Snakefile
from snakebids import bids

# Load config
configfile: 'config/snakebids.yml'

# Generate inputs from BIDS
inputs = bids(**config['pybids_inputs'])

# Define all outputs
rule all:
    input:
        expand(
            bids(
                root=config['output_dir'],
                datatype='anat',
                suffix='brain.nii.gz',
                **inputs['t1w'].wildcards
            ),
            **inputs['t1w'].zip_lists
        )

# Brain extraction rule
rule brain_extract:
    input:
        t1=inputs['t1w'].path
    output:
        brain=bids(
            root=config['output_dir'],
            datatype='anat',
            suffix='brain.nii.gz',
            **inputs['t1w'].wildcards
        )
    container:
        config['singularity']['fsl']
    shell:
        "bet {input.t1} {output.brain} -f 0.5 -m"
```

### Using BIDS Wildcards

```python
# Access BIDS entities as wildcards
rule process_subject:
    input:
        t1=inputs['t1w'].path
    output:
        processed=bids(
            root=config['output_dir'],
            datatype='anat',
            suffix='processed.nii.gz',
            subject='{subject}',
            session='{session}'
        )
    shell:
        """
        echo "Processing {wildcards.subject} session {wildcards.session}"
        # Process T1w image
        """
```

### Multi-Modal Workflow

```python
# Combine T1w and T2w processing
rule coregister_t2_to_t1:
    input:
        t1=inputs['t1w'].path,
        t2=inputs['t2w'].path
    output:
        t2_coreg=bids(
            root=config['output_dir'],
            datatype='anat',
            suffix='T2w',
            space='T1w',
            desc='coreg',
            subject='{subject}',
            session='{session}'
        )
    container:
        config['singularity']['fsl']
    shell:
        """
        flirt -in {input.t2} -ref {input.t1} \
              -out {output.t2_coreg} -omat transform.mat
        """
```

## Execution

### Local Execution

```bash
# Run pipeline locally
snakebids run /path/to/bids/dataset results participant

# Or using snakemake directly
cd my_pipeline
snakemake --cores 4

# Dry run to check
snakemake -n

# Specific subjects
snakemake --config participants=['sub-01','sub-02'] --cores 2
```

### Cluster Execution (SLURM)

```bash
# Create SLURM profile
mkdir -p profiles/slurm

# profiles/slurm/config.yaml
cat > profiles/slurm/config.yaml << EOF
cluster: "sbatch -p general -c {threads} --mem={resources.mem_mb}M -t {resources.time}"
jobs: 100
latency-wait: 60
default-resources:
  - mem_mb=4000
  - time=60
EOF

# Run on cluster
snakemake --profile profiles/slurm --jobs 50
```

### With Singularity Containers

```bash
# Enable Singularity
snakemake --use-singularity --singularity-args "-B /data"

# Cluster with containers
snakemake --profile profiles/slurm --use-singularity --jobs 100
```

## Complete Example: T1w Preprocessing

### Config File

```yaml
# config/snakebids.yml
bids_dir: '/data/bids_dataset'
output_dir: 'derivatives/preprocessing'

pybids_inputs:
  t1w:
    filters:
      suffix: 'T1w'
      extension: '.nii.gz'
      datatype: 'anat'
    wildcards:
      - subject
      - session
      - acquisition

parse_args:
  bids_dir:
    help: 'BIDS dataset directory'
  output_dir:
    help: 'Output directory'

participants: all

singularity:
  fsl: 'docker://brainlife/fsl:6.0.4'
  ants: 'docker://antsx/ants:latest'
```

### Workflow

```python
# workflow/Snakefile
from snakebids import bids

configfile: 'config/snakebids.yml'

# Load BIDS inputs
inputs = bids(**config['pybids_inputs'])

# All outputs
rule all:
    input:
        # Brain-extracted
        expand(
            bids(
                root=config['output_dir'],
                datatype='anat',
                suffix='brain.nii.gz',
                desc='bet',
                **inputs['t1w'].wildcards
            ),
            **inputs['t1w'].zip_lists
        ),
        # Bias-corrected
        expand(
            bids(
                root=config['output_dir'],
                datatype='anat',
                suffix='T1w',
                desc='n4',
                **inputs['t1w'].wildcards
            ),
            **inputs['t1w'].zip_lists
        ),
        # Normalized to MNI
        expand(
            bids(
                root=config['output_dir'],
                datatype='anat',
                suffix='T1w',
                space='MNI152NLin2009cAsym',
                **inputs['t1w'].wildcards
            ),
            **inputs['t1w'].zip_lists
        )

# Step 1: Bias correction
rule n4_bias_correction:
    input:
        t1=inputs['t1w'].path
    output:
        corrected=bids(
            root=config['output_dir'],
            datatype='anat',
            suffix='T1w',
            desc='n4',
            **inputs['t1w'].wildcards
        )
    container:
        config['singularity']['ants']
    threads: 1
    resources:
        mem_mb=4000,
        time=30
    shell:
        "N4BiasFieldCorrection -i {input.t1} -o {output.corrected}"

# Step 2: Brain extraction
rule brain_extraction:
    input:
        t1=rules.n4_bias_correction.output.corrected
    output:
        brain=bids(
            root=config['output_dir'],
            datatype='anat',
            suffix='brain.nii.gz',
            desc='bet',
            **inputs['t1w'].wildcards
        ),
        mask=bids(
            root=config['output_dir'],
            datatype='anat',
            suffix='mask.nii.gz',
            desc='brain',
            **inputs['t1w'].wildcards
        )
    container:
        config['singularity']['fsl']
    threads: 1
    resources:
        mem_mb=4000,
        time=15
    shell:
        "bet {input.t1} {output.brain} -m -f 0.5"

# Step 3: Registration to MNI
rule register_to_mni:
    input:
        moving=rules.brain_extraction.output.brain,
        template='/templates/MNI152_T1_1mm_brain.nii.gz'
    output:
        warped=bids(
            root=config['output_dir'],
            datatype='anat',
            suffix='T1w',
            space='MNI152NLin2009cAsym',
            **inputs['t1w'].wildcards
        ),
        affine=bids(
            root=config['output_dir'],
            datatype='anat',
            suffix='affine.mat',
            from_='T1w',
            to='MNI152NLin2009cAsym',
            mode='image',
            **inputs['t1w'].wildcards
        )
    container:
        config['singularity']['ants']
    threads: 4
    resources:
        mem_mb=8000,
        time=60
    shell:
        """
        antsRegistrationSyN.sh -d 3 -f {input.template} \
          -m {input.moving} -o transform_ -n {threads}
        mv transform_Warped.nii.gz {output.warped}
        mv transform_0GenericAffine.mat {output.affine}
        """
```

## Advanced Features

### Aggregate Rules

```python
# Combine results across subjects
rule group_analysis:
    input:
        # Collect all subject results
        brains=expand(
            bids(
                root=config['output_dir'],
                datatype='anat',
                suffix='brain.nii.gz',
                **inputs['t1w'].wildcards
            ),
            **inputs['t1w'].zip_lists
        )
    output:
        report=bids(
            root=config['output_dir'],
            suffix='report.html',
            desc='group'
        )
    script:
        "scripts/generate_group_report.py"
```

### Quality Control

```python
# QC after brain extraction
rule qc_brain_extraction:
    input:
        original=inputs['t1w'].path,
        brain=rules.brain_extraction.output.brain,
        mask=rules.brain_extraction.output.mask
    output:
        qc_img=bids(
            root=config['output_dir'],
            datatype='qc',
            suffix='brainextraction.png',
            **inputs['t1w'].wildcards
        )
    script:
        "scripts/qc_brain_extraction.py"
```

### Custom Filters

```python
# Filter by specific BIDS entities
from snakebids import bids

# Only process certain acquisitions
inputs_filtered = bids(
    filters={
        'suffix': 'T1w',
        'acquisition': ['MPRAGE', 'TFE'],  # Only these
        'extension': '.nii.gz'
    },
    wildcards=['subject', 'session', 'acquisition']
)
```

## Command-Line Interface

### Auto-Generated CLI

```python
# my_pipeline/run.py
#!/usr/bin/env python3
from snakebids.app import SnakeBidsApp

def main():
    app = SnakeBidsApp('config/snakebids.yml')
    app.run_snakemake()

if __name__ == '__main__':
    main()
```

### Usage

```bash
# Install pipeline
pip install -e .

# Run via CLI
my-pipeline /data/bids results participant --cores 8

# Specific participants
my-pipeline /data/bids results participant \
  --participant-label sub-01 sub-02 --cores 4

# With containers
my-pipeline /data/bids results participant \
  --use-singularity --cores 8

# Dry run
my-pipeline /data/bids results participant -n
```

## Deployment

### BIDS App Packaging

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='my-bids-pipeline',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'snakebids>=0.8.0',
        'snakemake>=7.0.0',
    ],
    entry_points={
        'console_scripts': [
            'my-pipeline=my_pipeline.run:main',
        ],
    },
)
```

### Docker Container

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pipeline
COPY . /src
WORKDIR /src
RUN pip install -e .

# Entry point
ENTRYPOINT ["my-pipeline"]
```

### Build and Publish

```bash
# Build Docker image
docker build -t my-pipeline:latest .

# Test locally
docker run -it --rm \
  -v /data/bids:/data \
  -v /data/output:/output \
  my-pipeline:latest /data /output participant

# Push to Docker Hub
docker tag my-pipeline:latest username/my-pipeline:latest
docker push username/my-pipeline:latest
```

## Integration with Claude Code

Snakebids enables Claude-assisted BIDS app development:

### Pipeline Generation

```markdown
**Prompt to Claude:**
"Create Snakebids pipeline for diffusion preprocessing:
1. BIDS input: DWI with bval/bvec
2. Steps: Denoising, eddy correction, bias correction
3. Outputs: Preprocessed DWI in BIDS derivatives
4. Use FSL in Singularity container
5. Include QC visualizations
Provide complete config and Snakefile."
```

### Cluster Configuration

```markdown
**Prompt to Claude:**
"Set up Snakebids for SLURM cluster:
- Pipeline: FreeSurfer recon-all on 100 subjects
- Resources: 8 cores, 16GB RAM, 24 hours per subject
- Use Singularity container
- Parallel job submission (max 20 concurrent)
- Error handling and job restart
Provide SLURM profile and execution commands."
```

### Multi-Modal Integration

```markdown
**Prompt to Claude:**
"Build Snakebids workflow combining:
- T1w → FreeSurfer
- T2w → Coregister to T1w
- FLAIR → Lesion segmentation
- Bold → fMRIPrep derivatives as input
Use appropriate containers for each step.
Include aggregate QC report at end."
```

## Integration with Other Tools

### With fMRIPrep Derivatives

```yaml
# Use fMRIPrep outputs as input
pybids_derivatives:
  fmriprep_bold:
    database_path: /data/derivatives/fmriprep
    filters:
      suffix: 'bold'
      desc: 'preproc'
      space: 'MNI152NLin2009cAsym'
    wildcards:
      - subject
      - session
      - task
```

### With FreeSurfer

```python
# Run FreeSurfer in workflow
rule freesurfer_recon:
    input:
        t1=inputs['t1w'].path
    output:
        done=touch(bids(
            root=config['output_dir'],
            datatype='freesurfer',
            suffix='recon.done',
            subject='{subject}'
        ))
    container:
        config['singularity']['freesurfer']
    threads: 4
    resources:
        mem_mb=16000,
        time=1440  # 24 hours
    shell:
        """
        recon-all -s {wildcards.subject} -i {input.t1} -all \
          -sd {config[output_dir]}/freesurfer
        """
```

### With Custom Python Scripts

```python
# workflow/scripts/custom_analysis.py
import sys
import nibabel as nib
import numpy as np

# Snakemake provides inputs/outputs automatically
input_file = snakemake.input['brain']
output_file = snakemake.output['metrics']

# Load image
img = nib.load(input_file)
data = img.get_fdata()

# Compute metrics
metrics = {
    'mean': float(np.mean(data[data > 0])),
    'std': float(np.std(data[data > 0])),
    'volume': int(np.sum(data > 0))
}

# Save results
import json
with open(output_file, 'w') as f:
    json.dump(metrics, f, indent=2)
```

## Troubleshooting

### Problem 1: BIDS Entities Not Found

**Symptoms:** No files match filter

**Solutions:**
```bash
# Validate BIDS dataset first
pip install bids-validator
bids-validator /data/bids_dataset

# Check what PyBIDS finds
python -c "
from bids import BIDSLayout
layout = BIDSLayout('/data/bids_dataset')
print(layout.get(suffix='T1w'))
"

# Adjust filters in config
# Maybe 'datatype' should be omitted or acquisition is different
```

### Problem 2: Container Binding Issues

**Symptoms:** File not found in container

**Solutions:**
```bash
# Ensure paths are bound
snakemake --use-singularity \
  --singularity-args "-B /data:/data -B /scratch:/scratch"

# Or set in profile
# config.yaml:
singularity-args: "-B /data:/data"
```

### Problem 3: Cluster Jobs Fail

**Symptoms:** SLURM jobs error out

**Solutions:**
```yaml
# Increase resources in rules
rule heavy_processing:
    resources:
        mem_mb=32000,  # Increase memory
        time=480       # 8 hours

# Check cluster logs
# Usually in .snakemake/slurm_logs/
```

### Problem 4: Wildcards Not Expanding

**Symptoms:** Output paths have literal {subject}

**Solutions:**
```python
# Ensure wildcards are properly propagated
rule example:
    output:
        # Correct: wildcards defined
        bids(suffix='output.nii.gz', subject='{subject}')
    
    # NOT:
    # bids(suffix='output.nii.gz', subject=config['subject'])
```

## Best Practices

1. **Start with template** - Use `snakebids create` for proper structure
2. **Validate BIDS** - Run bids-validator on input dataset
3. **Test locally first** - Use small subset before cluster
4. **Use containers** - Ensures reproducibility
5. **Version control config** - Track pipeline versions
6. **Document requirements** - List BIDS entities needed
7. **Include QC steps** - Visual and quantitative checks
8. **Modular rules** - Separate rules in different files
9. **Resource profiles** - Different for local/cluster
10. **Error handling** - Use try/except in scripts

## Resources

### Official Documentation

- **GitHub:** https://github.com/akhanf/snakebids
- **Documentation:** https://snakebids.readthedocs.io/
- **Examples:** https://github.com/akhanf/snakebids/tree/main/examples

### Snakemake Resources

- **Snakemake:** https://snakemake.readthedocs.io/
- **Tutorial:** https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html

### BIDS Resources

- **BIDS Specification:** https://bids-specification.readthedocs.io/
- **PyBIDS:** https://bids-standard.github.io/pybids/

### Community Support

- **GitHub Issues:** https://github.com/akhanf/snakebids/issues
- **Snakemake Forum:** https://stackoverflow.com/questions/tagged/snakemake

## Citation

```bibtex
@software{snakebids,
  title = {Snakebids: BIDS integration into Snakemake workflows},
  author = {Khan, Ali R and others},
  year = {2021},
  url = {https://github.com/akhanf/snakebids},
  note = {Python package}
}
```

## Related Tools

- **Snakemake** - Workflow management system (underlying engine)
- **PyBIDS** - BIDS dataset querying (used by Snakebids)
- **Boutiques** - Tool descriptor framework
- **Nipype** - Python workflows for neuroimaging
- **Pydra** - Next-generation workflow engine
- **Nextflow** - Data-driven workflow language
- **BIDS Apps** - Standard BIDS application framework

---

**Skill Type:** Workflow Framework
**Difficulty Level:** Intermediate
**Prerequisites:** Python, Snakemake basics, BIDS format knowledge
**Typical Use Cases:** BIDS app development, reproducible pipelines, multi-subject processing, HPC deployment
