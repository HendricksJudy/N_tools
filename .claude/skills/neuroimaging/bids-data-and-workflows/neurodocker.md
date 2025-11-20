# NeuroDocker - Reproducible Neuroimaging Containers

## Overview

**NeuroDocker** is a command-line tool for generating custom Docker and Singularity containers tailored for neuroimaging research. Developed by the ReproNim (Reproducible Neuroimaging) initiative, NeuroDocker simplifies the creation of reproducible containerized environments by providing a high-level interface for installing neuroimaging software packages (FSL, FreeSurfer, AFNI, SPM, ANTs, etc.) with proper dependencies, environment variables, and configurations automatically handled.

NeuroDocker ensures bit-for-bit computational reproducibility across different institutions, computing platforms, and time periods, eliminating "works on my machine" problems and enabling seamless deployment to HPC clusters, cloud platforms, and local workstations. It generates optimized Dockerfiles and Singularity recipes that can be version-controlled, shared, and rebuilt to recreate exact computational environments.

**Key Features:**
- Generate Dockerfiles and Singularity recipes from command-line
- Install neuroimaging tools with single commands
- Automatic dependency resolution and configuration
- Environment variable setup (FSLDIR, FREESURFER_HOME, etc.)
- Support for multiple base images (Debian, Ubuntu, CentOS, Neurodebian)
- Miniconda and Python package management
- Multi-stage builds for size optimization
- GPU support (CUDA, cuDNN) for deep learning
- Jupyter notebook server configuration
- BIDS Apps compatible container generation
- Version pinning for exact reproducibility
- Integration with container registries

**Primary Use Cases:**
- Create reproducible research environments
- Build BIDS Apps for method distribution
- Deploy pipelines to HPC and cloud
- Standardize environments across multi-site studies
- Package analysis workflows for sharing
- Develop neuroimaging software with consistent dependencies
- Teaching and training with pre-configured environments

**Official Documentation:** https://github.com/ReproNim/neurodocker

---

## Installation

### Install NeuroDocker

```bash
# Install via pip (recommended)
pip install neurodocker

# Or install from GitHub for latest development version
pip install git+https://github.com/ReproNim/neurodocker.git

# Verify installation
neurodocker --version

# Show available commands
neurodocker --help
```

### Prerequisites

```bash
# For Docker containers, install Docker
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER  # Add user to docker group
# Log out and back in for group changes

# Verify Docker
docker --version
docker run hello-world

# For Singularity containers, install Singularity
# (Installation varies by platform)
# Check: https://sylabs.io/guides/latest/user-guide/

# Verify Singularity
singularity --version
```

---

## Basic Container Generation

### Generate Simple Dockerfile

```bash
# Generate Dockerfile for Ubuntu with FSL
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --fsl version=6.0.5 \
    > Dockerfile

# View generated Dockerfile
cat Dockerfile

# Build Docker image
docker build -t my-fsl:6.0.5 .

# Run container
docker run -it --rm my-fsl:6.0.5 bash

# Test FSL installation inside container
# $ fsl
# $ bet
```

### Generate and Build in One Step

```bash
# Generate, build, and tag in pipeline
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --freesurfer version=7.2.0 \
    | docker build -t my-freesurfer:7.2.0 -

# Run FreeSurfer container
docker run -it --rm my-freesurfer:7.2.0 bash
# $ recon-all --version
```

### Save Dockerfile for Version Control

```bash
# Generate and save Dockerfile
neurodocker generate docker \
    --pkg-manager apt \
    --base-image neurodebian:bullseye \
    --afni version=latest \
    --install git vim \
    > Dockerfile

# Commit to version control
git add Dockerfile
git commit -m "Add AFNI container recipe"

# Anyone can rebuild identical container
docker build -t afni-container .
```

---

## Installing Neuroimaging Software

### FSL Installation

```bash
# FSL with specific version
neurodocker generate docker \
    --pkg-manager apt \
    --base-image debian:bullseye-slim \
    --fsl version=6.0.5 method=binaries \
    > Dockerfile.fsl

# FSL sets up:
# - FSLDIR environment variable
# - PATH to include FSL binaries
# - Required libraries
```

### FreeSurfer Installation

```bash
# FreeSurfer requires license file
# First, get license from https://surfer.nmr.mgh.harvard.edu/registration.html

neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --freesurfer version=7.2.0 \
    > Dockerfile.freesurfer

# Build with license
docker build -t freesurfer:7.2.0 -f Dockerfile.freesurfer .

# Run with license file mounted
docker run -it --rm \
    -v /path/to/license.txt:/opt/freesurfer/license.txt \
    freesurfer:7.2.0 bash
```

### AFNI Installation

```bash
# AFNI latest version
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --afni version=latest method=binaries \
    > Dockerfile.afni

# AFNI configures:
# - PATH with AFNI binaries
# - Required dependencies (tcsh, libglu1, etc.)
```

### ANTs Installation

```bash
# ANTs (Advanced Normalization Tools)
neurodocker generate docker \
    --pkg-manager apt \
    --base-image debian:bullseye \
    --ants version=2.3.5 method=binaries \
    > Dockerfile.ants

# ANTs sets:
# - ANTSPATH environment variable
# - PATH to include ANTs scripts
```

### SPM12 with MATLAB Runtime

```bash
# SPM12 Standalone (includes MATLAB Runtime)
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --spm12 version=r7771 method=binaries \
    > Dockerfile.spm

# SPM12 installation:
# - Downloads MATLAB Runtime (large download ~1GB)
# - Configures SPM standalone
# - Sets up environment variables
```

### Multiple Tools in One Container

```bash
# Comprehensive neuroimaging container
neurodocker generate docker \
    --pkg-manager apt \
    --base-image neurodebian:bullseye \
    --fsl version=6.0.5 \
    --freesurfer version=7.2.0 \
    --afni version=latest \
    --ants version=2.3.5 \
    --install git vim curl wget \
    > Dockerfile.multiTool

# Build large multi-tool container
docker build -t neuro-tools:latest -f Dockerfile.multiTool .

# Container size will be several GB
# Consider multi-stage builds for optimization (see below)
```

---

## Python and Conda Environments

### Install Miniconda

```bash
# Container with Miniconda
neurodocker generate docker \
    --pkg-manager apt \
    --base-image debian:bullseye-slim \
    --miniconda \
        version=latest \
        env_name=neuro \
        conda_install="python=3.9 numpy scipy pandas" \
    > Dockerfile.conda

# Activates 'neuro' environment by default
```

### Install Python Packages

```bash
# Neuroimaging Python stack
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --miniconda version=latest \
        env_name=neuroimaging \
        conda_install="python=3.9 numpy scipy scikit-learn" \
        pip_install="nibabel nilearn nipype" \
    > Dockerfile.python

# Build and test
docker build -t neuro-python:latest -f Dockerfile.python .
docker run -it --rm neuro-python:latest python -c "import nilearn; print(nilearn.__version__)"
```

### Combine Miniconda with Neuroimaging Tools

```bash
# FSL + Python environment
neurodocker generate docker \
    --pkg-manager apt \
    --base-image neurodebian:bullseye \
    --fsl version=6.0.5 \
    --miniconda \
        version=latest \
        env_name=analysis \
        conda_install="python=3.9 matplotlib seaborn" \
        pip_install="nipype nilearn nibabel" \
    > Dockerfile.fsl-python

# Now you have FSL command-line tools + Python libraries
```

---

## Environment Configuration

### Set Environment Variables

```bash
# Custom environment variables
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --env MYVAR=myvalue \
    --env DATA_DIR=/data \
    --install curl \
    > Dockerfile

# Multiple environment variables for configuration
neurodocker generate docker \
    --pkg-manager apt \
    --base-image debian:bullseye \
    --env OMP_NUM_THREADS=4 \
    --env MKL_NUM_THREADS=4 \
    --env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 \
    --fsl version=6.0.5 \
    > Dockerfile.parallel
```

### Run Custom Commands

```bash
# Execute custom commands during build
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --run "mkdir -p /data /output" \
    --run "chmod 777 /data /output" \
    --install git \
    > Dockerfile.custom

# Install from GitHub
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --install git \
    --run "cd /opt && git clone https://github.com/myrepo/mytool.git" \
    --run "cd /opt/mytool && pip install ." \
    > Dockerfile.github
```

### Non-Root User Configuration

```bash
# Create non-root user for security
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --user neurouser \
    --fsl version=6.0.5 \
    > Dockerfile.nonroot

# Container runs as 'neurouser' instead of root
# Important for HPC and security-conscious environments
```

---

## Advanced Features

### Multi-Stage Builds for Size Optimization

```bash
# Multi-stage build to reduce final image size
# Stage 1: Build environment
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 as builder \
    --install git build-essential cmake \
    --run "git clone https://github.com/example/tool.git /tmp/tool" \
    --run "cd /tmp/tool && mkdir build && cd build && cmake .. && make" \
    > Dockerfile.stage1

# Stage 2: Runtime environment (smaller)
cat >> Dockerfile.stage1 <<'EOF'
FROM ubuntu:20.04
COPY --from=builder /tmp/tool/build/bin /usr/local/bin
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*
EOF

docker build -t optimized-tool:latest -f Dockerfile.stage1 .
```

### GPU Support with CUDA

```bash
# NVIDIA CUDA base for GPU computing
neurodocker generate docker \
    --pkg-manager apt \
    --base-image nvidia/cuda:11.4.0-runtime-ubuntu20.04 \
    --miniconda \
        version=latest \
        env_name=gpu \
        conda_install="python=3.9" \
        pip_install="torch torchvision tensorflow-gpu" \
    > Dockerfile.gpu

# Build GPU-enabled container
docker build -t neuro-gpu:latest -f Dockerfile.gpu .

# Run with GPU access
docker run --gpus all -it --rm neuro-gpu:latest python -c "import torch; print(torch.cuda.is_available())"
```

### Jupyter Notebook Server

```bash
# Container with Jupyter for interactive analysis
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --miniconda \
        version=latest \
        env_name=jupyter \
        conda_install="python=3.9 jupyter matplotlib" \
        pip_install="nibabel nilearn" \
    --run "mkdir /notebooks" \
    --workdir /notebooks \
    --cmd "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root" \
    > Dockerfile.jupyter

# Build and run
docker build -t neuro-jupyter:latest -f Dockerfile.jupyter .
docker run -p 8888:8888 -v $(pwd):/notebooks neuro-jupyter:latest

# Access Jupyter at http://localhost:8888
```

---

## Singularity Containers

### Generate Singularity Recipe

```bash
# Generate Singularity definition file
neurodocker generate singularity \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --fsl version=6.0.5 \
    --ants version=2.3.5 \
    > fsl-ants.def

# View Singularity recipe
cat fsl-ants.def

# Build Singularity image (requires sudo or fakeroot)
sudo singularity build fsl-ants.sif fsl-ants.def

# Run Singularity container
singularity exec fsl-ants.sif bet
```

### Singularity for HPC

```bash
# HPC-optimized Singularity container
neurodocker generate singularity \
    --pkg-manager apt \
    --base-image neurodebian:bullseye \
    --freesurfer version=7.2.0 \
    --ants version=2.3.5 \
    --miniconda \
        version=latest \
        env_name=analysis \
        conda_install="python=3.9" \
        pip_install="nipype nibabel" \
    > hpc-neuro.def

# Build
sudo singularity build hpc-neuro.sif hpc-neuro.def

# Use on HPC (no sudo needed)
# Run as current user, bind directories
singularity exec \
    --bind /data:/data \
    --bind /scratch:/scratch \
    hpc-neuro.sif \
    python /data/my_analysis.py
```

### Convert Docker to Singularity

```bash
# Build Docker image first
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --fsl version=6.0.5 \
    | docker build -t my-fsl:latest -

# Convert Docker image to Singularity
singularity build fsl.sif docker-daemon://my-fsl:latest

# Or pull from Docker Hub
singularity build fsl.sif docker://myusername/my-fsl:latest
```

---

## BIDS Apps Creation

### BIDS App Container Structure

```bash
# BIDS App compliant container
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --fsl version=6.0.5 \
    --ants version=2.3.5 \
    --miniconda \
        version=latest \
        env_name=bidsapp \
        conda_install="python=3.9" \
        pip_install="pybids nibabel numpy" \
    --copy run.py /run.py \
    --entrypoint "python /run.py" \
    > Dockerfile.bidsapp

# BIDS App should accept:
# - /bids_dir (input BIDS dataset)
# - /output_dir (output directory)
# - analysis_level (participant or group)
```

### BIDS App Entrypoint Script

```python
# run.py - BIDS App entrypoint
#!/usr/bin/env python3
"""
BIDS App entrypoint script
Usage: docker run -v /data:/bids_dir -v /output:/output_dir \
       myapp:latest /bids_dir /output_dir participant
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='My BIDS App')
    parser.add_argument('bids_dir', type=Path, help='BIDS dataset directory')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    parser.add_argument('analysis_level', choices=['participant', 'group'],
                        help='Analysis level')
    parser.add_argument('--participant-label', nargs='+',
                        help='Process specific participants')
    args = parser.parse_args()

    # Validate BIDS dataset
    from bids import BIDSLayout
    layout = BIDSLayout(args.bids_dir)

    # Run analysis
    if args.analysis_level == 'participant':
        subjects = args.participant_label or layout.get_subjects()
        for sub in subjects:
            print(f"Processing sub-{sub}")
            # Your analysis here

if __name__ == '__main__':
    main()
```

### Build and Test BIDS App

```bash
# Build BIDS App
docker build -t mybidsapp:1.0.0 -f Dockerfile.bidsapp .

# Test with BIDS dataset
docker run -it --rm \
    -v /path/to/bids_dataset:/bids_dir:ro \
    -v /path/to/output:/output_dir \
    mybidsapp:1.0.0 \
    /bids_dir /output_dir participant \
    --participant-label 01 02 03

# Validate BIDS App compliance
# https://bids-apps.neuroimaging.io/
```

---

## Integration with Workflows

### Use with Pydra

```python
# Use NeuroDocker container in Pydra workflow
from pydra import Workflow, mark
from pydra.engine.specs import DockerSpec

# Define task with container
@mark.task
@mark.annotate({'return': {'out_file': str}})
def bet_task(in_file: str) -> str:
    """FSL BET brain extraction"""
    import subprocess
    out_file = in_file.replace('.nii.gz', '_brain.nii.gz')
    subprocess.run(['bet', in_file, out_file])
    return out_file

# Configure to run in Docker
bet_task.docker_spec = DockerSpec(
    image='my-fsl:6.0.5',
    container_type='docker'
)

# Create workflow
wf = Workflow(name='extraction', input_spec=['t1w'])
wf.add(bet_task(name='bet', in_file=wf.lzin.t1w))
wf.set_output([('brain', wf.bet.lzout.out_file)])

# Execute
result = wf(t1w='/data/sub-01_T1w.nii.gz')
```

### Container in Snakemake Pipeline

```python
# Snakefile using NeuroDocker container
rule brain_extraction:
    input:
        t1w = "data/{subject}/anat/{subject}_T1w.nii.gz"
    output:
        brain = "derivatives/{subject}/anat/{subject}_T1w_brain.nii.gz"
    container:
        "docker://my-fsl:6.0.5"
    shell:
        "bet {input.t1w} {output.brain}"

# Run with Singularity
# snakemake --use-singularity

# Snakemake automatically converts Docker to Singularity
```

---

## Reproducibility Best Practices

### Version Pinning

```bash
# Pin exact versions for reproducibility
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --fsl version=6.0.5 \
    --freesurfer version=7.2.0 \
    --ants version=2.3.5 \
    --miniconda \
        version=4.10.3 \
        env_name=analysis \
        conda_install="python=3.9.7 numpy=1.21.2 scipy=1.7.1" \
        pip_install="nibabel==3.2.1 nilearn==0.8.1" \
    > Dockerfile.pinned

# Document versions in README
echo "# Versions" > VERSION.txt
echo "FSL: 6.0.5" >> VERSION.txt
echo "FreeSurfer: 7.2.0" >> VERSION.txt
echo "ANTs: 2.3.5" >> VERSION.txt
```

### Container Metadata

```bash
# Add metadata labels
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --label maintainer="your.email@institution.edu" \
    --label version="1.0.0" \
    --label description="fMRI preprocessing container" \
    --fsl version=6.0.5 \
    > Dockerfile.labeled

# View metadata
docker inspect my-container:latest | grep -A 5 Labels
```

### Document and Share

```bash
# Create comprehensive documentation
cat > README.md <<'EOF'
# My Neuroimaging Container

## Building
```bash
docker build -t my-container:1.0.0 .
```

## Running
```bash
docker run -v /data:/data my-container:1.0.0 [command]
```

## Included Software
- FSL 6.0.5
- ANTs 2.3.5
- Python 3.9 with nibabel, nilearn

## Citation
If you use this container, please cite...
EOF

# Push to Docker Hub
docker tag my-container:1.0.0 username/my-container:1.0.0
docker push username/my-container:1.0.0

# Or save as tar for sharing
docker save my-container:1.0.0 | gzip > my-container-1.0.0.tar.gz
```

---

## Deployment Scenarios

### Local Workstation

```bash
# Build container locally
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --fsl version=6.0.5 \
    | docker build -t fsl:local -

# Run analysis
docker run -it --rm \
    -v $(pwd)/data:/data \
    -v $(pwd)/output:/output \
    fsl:local \
    bet /data/T1.nii.gz /output/T1_brain.nii.gz
```

### HPC Cluster with Singularity

```bash
# Generate Singularity recipe
neurodocker generate singularity \
    --pkg-manager apt \
    --base-image neurodebian:bullseye \
    --freesurfer version=7.2.0 \
    > freesurfer.def

# Build on login node or build server
singularity build freesurfer.sif freesurfer.def

# Submit job to cluster
cat > job.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=recon
#SBATCH --time=24:00:00
#SBATCH --mem=16G

singularity exec \
    --bind /scratch/$USER:/data \
    freesurfer.sif \
    recon-all -s sub-01 -i /data/T1.nii.gz -all
EOF

sbatch job.sh
```

### Cloud Deployment (AWS)

```bash
# Push container to Amazon ECR
# 1. Authenticate
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789.dkr.ecr.us-east-1.amazonaws.com

# 2. Tag container
docker tag my-container:latest \
    123456789.dkr.ecr.us-east-1.amazonaws.com/my-container:latest

# 3. Push
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/my-container:latest

# 4. Use in AWS Batch or ECS
```

---

## Optimization and Troubleshooting

### Reduce Container Size

```bash
# Use slim base images
neurodocker generate docker \
    --pkg-manager apt \
    --base-image debian:bullseye-slim \  # Smaller than ubuntu
    --fsl version=6.0.5 \
    > Dockerfile.slim

# Clean up after installations
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --fsl version=6.0.5 \
    --run "apt-get clean && rm -rf /var/lib/apt/lists/*" \
    > Dockerfile.clean

# Check image size
docker images | grep my-container
```

### Debug Build Failures

```bash
# Build with verbose output
docker build --progress=plain -t my-container:debug -f Dockerfile .

# Build up to failing step
# If build fails at step 5, build up to step 4
# Modify Dockerfile to comment out step 5 and beyond
docker build -t debug-container .
docker run -it debug-container bash
# Manually try failing command to debug
```

### Handle License Files

```bash
# For tools requiring licenses (FreeSurfer, FSL)

# Option 1: Mount license at runtime
docker run -v /path/to/license.txt:/opt/freesurfer/license.txt ...

# Option 2: Copy during build (not recommended for public images)
# Dockerfile:
# COPY license.txt /opt/freesurfer/license.txt

# Option 3: Use environment variable
docker run -e FS_LICENSE=$(cat license.txt) ...
```

### Memory and Build Resources

```bash
# Increase Docker memory limit for large builds
# Docker Desktop: Settings > Resources > Memory

# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t my-container .

# Clean up build cache
docker builder prune
```

---

## Example: Complete fMRIPrep-Compatible Container

```bash
# Container for fMRI preprocessing
neurodocker generate docker \
    --pkg-manager apt \
    --base-image ubuntu:20.04 \
    --fsl version=6.0.5 \
    --freesurfer version=7.2.0 \
    --ants version=2.3.5 \
    --miniconda \
        version=latest \
        env_name=fmri \
        conda_install="python=3.9 numpy scipy pandas" \
        pip_install="nibabel nilearn nipype pybids" \
    --install git vim curl \
    --env OMP_NUM_THREADS=8 \
    --user fmriuser \
    --workdir /work \
    > Dockerfile.fmri

# Build
docker build -t fmri-preproc:1.0.0 -f Dockerfile.fmri .

# Test
docker run -it --rm \
    -v $(pwd)/data:/data \
    -v $(pwd)/output:/output \
    fmri-preproc:1.0.0 \
    python -c "import nilearn; print('Success!')"
```

---

## Related Tools and Integration

**Workflow Engines:**
- **Pydra** (Batch 28): Execute tasks in NeuroDocker containers
- **Snakebids** (Batch 28): Snakemake with BIDS and containers
- **Nipype** (Batch 2): Interface neuroimaging tools in containers

**Complementary Tools:**
- **Boutiques** (Batch 28): Validate tools in containers
- **TemplateFlow** (Batch 28): Brain templates for containerized pipelines
- **BIDS Validator** (Batch 4): Validate inputs to BIDS Apps

**Preprocessing:**
- **fMRIPrep** (Batch 5): Available as BIDS App container
- **QSIPrep** (Batch 6): Diffusion preprocessing container
- **All neuroimaging tools:** Installable via NeuroDocker

---

## References

- Kurtzer, G. M., et al. (2017). Singularity: Scientific containers for mobility of compute. *PLoS ONE*, 12(5), e0177459.
- Merkel, D. (2014). Docker: lightweight Linux containers for consistent development and deployment. *Linux Journal*, 2014(239), 2.
- Ghosh, S. S., et al. (2017). A very simple, re-executable neuroimaging publication. *F1000Research*, 6, 124.
- Gorgolewski, K. J., et al. (2017). BIDS apps: Improving ease of use, accessibility, and reproducibility of neuroimaging data analysis methods. *PLoS Computational Biology*, 13(3), e1005209.

**Official Repository:** https://github.com/ReproNim/neurodocker
**Documentation:** https://github.com/ReproNim/neurodocker/blob/master/README.md
**ReproNim:** http://www.repronim.org/
**BIDS Apps:** https://bids-apps.neuroimaging.io/
