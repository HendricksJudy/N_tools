# TractoFlow

## Overview

**TractoFlow** is a fully automated, production-ready diffusion MRI preprocessing and tractography pipeline developed by the SCIL (Sherbrooke Connectivity Imaging Lab). Built using Nextflow and containerized with Singularity or Docker, TractoFlow implements state-of-the-art diffusion processing methods from denoising through fiber tractography, with comprehensive quality control at every step.

TractoFlow is designed for robustness and reproducibility in clinical and research settings. It handles single-shell and multi-shell diffusion data, applies rigorous preprocessing (including denoising, Gibbs ringing removal, motion/eddy current correction, bias field correction), performs advanced diffusion modeling (DTI, fODF), and generates anatomically-informed whole-brain tractography. The pipeline outputs detailed quality control reports and standardized derivatives suitable for downstream connectomics and white matter analysis.

**Key Use Cases:**
- Automated diffusion MRI preprocessing for clinical trials
- Multi-subject tractography for research studies
- Standardized white matter analysis pipelines
- HPC cluster processing of large dMRI datasets
- Quality-controlled tractogram generation
- Reproducible diffusion preprocessing workflows

**Official Website:** https://tractoflow-documentation.readthedocs.io/
**Source Code:** https://github.com/scilus/tractoflow
**Documentation:** https://tractoflow-documentation.readthedocs.io/en/latest/

---

## Key Features

- **Fully Automated Pipeline:** End-to-end processing from raw DWI to tractography with no manual intervention
- **State-of-the-Art Methods:** MP-PCA denoising, Gibbs ringing removal, FSL Eddy, N4 bias correction
- **Multi-Shell Support:** Handles single-shell, multi-shell, and mixed acquisition protocols
- **Container-Based:** Complete reproducibility with Singularity or Docker containers
- **HPC Ready:** Seamless execution on SLURM, PBS, SGE, and other cluster schedulers
- **Quality Control:** Automated QC metrics and visual reports at every processing step
- **Nextflow Engine:** Robust workflow management with resume capability and error handling
- **Anatomical Registration:** T1-to-DWI registration for seeding and masking
- **Advanced Tractography:** Probabilistic tracking with particle filtering and anatomical priors
- **Comprehensive Outputs:** DTI metrics, fODF maps, tractograms, and connectivity-ready derivatives
- **Resume Capability:** Restart failed runs from the last successful step
- **Resource Management:** Configurable CPU/memory allocation per process
- **Validation:** Extensively validated on public datasets (HCP, MASSIVE)
- **Active Development:** Regular updates with community-driven enhancements
- **Open Source:** MIT licensed, community contributions welcome

---

## Installation

### Install Nextflow

Nextflow is required to run TractoFlow:

```bash
# Install Nextflow (requires Java 8+)
curl -s https://get.nextflow.io | bash

# Make executable and move to PATH
chmod +x nextflow
sudo mv nextflow /usr/local/bin/

# Verify installation
nextflow -version
```

### Install Singularity (Recommended)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y singularity-container

# Or install from source (latest version)
export VERSION=3.8.7
wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce-${VERSION}.tar.gz
tar -xzf singularity-ce-${VERSION}.tar.gz
cd singularity-ce-${VERSION}
./mconfig
make -C builddir
sudo make -C builddir install

# Verify
singularity --version
```

### Install Docker (Alternative)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io

# Add user to docker group
sudo usermod -aG docker $USER

# Verify
docker --version
```

### Download TractoFlow

```bash
# Clone the repository
git clone https://github.com/scilus/tractoflow.git
cd tractoflow

# Or use Nextflow to pull directly
nextflow pull scilus/tractoflow
```

### Test Installation

```bash
# Download test dataset
wget https://scil.usherbrooke.ca/en/tractoflow_data/test_data.zip
unzip test_data.zip

# Run test with Singularity
nextflow run tractoflow/main.nf \
  --input test_data \
  -profile singularity \
  -resume

# Or with Docker
nextflow run tractoflow/main.nf \
  --input test_data \
  -profile docker \
  -resume
```

---

## Pipeline Overview

TractoFlow consists of multiple processing stages executed in sequence:

### Preprocessing Stages

```bash
# 1. Denoising (MP-PCA)
#    - Marchenko-Pastur PCA denoising
#    - Reduces thermal noise while preserving signal

# 2. Gibbs Ringing Removal
#    - Removes Gibbs artifacts from DWI
#    - Improves image quality

# 3. Topup (if reverse phase-encoding available)
#    - Corrects susceptibility distortions
#    - Uses FSL Topup

# 4. Eddy Current and Motion Correction
#    - FSL Eddy for motion and distortion correction
#    - Outlier detection and replacement

# 5. N4 Bias Field Correction
#    - Corrects intensity inhomogeneity
#    - Uses ANTs N4

# 6. Brain Extraction
#    - Automated brain mask generation
#    - Uses FSL BET or custom methods

# 7. Resampling
#    - Resample to isotropic voxels (default: 1mm)
#    - Improves tractography accuracy

# 8. DTI Metrics
#    - FA, MD, AD, RD, color FA
#    - Uses Dipy

# 9. fODF Estimation
#    - Fiber orientation distribution function
#    - Constrained spherical deconvolution (CSD)
#    - Uses MRtrix3 or Dipy

# 10. Tractography
#    - Probabilistic fiber tracking
#    - Particle filtering tractography (PFT)
#    - Anatomically-informed seeding
```

### Pipeline DAG Visualization

```bash
# Generate pipeline diagram
nextflow run tractoflow/main.nf \
  --input test_data \
  -with-dag flowchart.png
```

---

## Input Requirements

### Directory Structure

TractoFlow expects a specific input structure:

```
input_directory/
├── sub-01/
│   ├── dwi.nii.gz          # DWI 4D volume (required)
│   ├── bval                # b-values (required)
│   ├── bvec                # b-vectors (required)
│   ├── rev_b0.nii.gz       # Reverse phase-encoded b0 (optional, for Topup)
│   └── t1.nii.gz           # T1-weighted image (optional, recommended)
├── sub-02/
│   ├── dwi.nii.gz
│   ├── bval
│   ├── bvec
│   └── t1.nii.gz
└── ...
```

### Prepare Input Data

```bash
# Create input directory structure
mkdir -p tractoflow_input/sub-01

# Copy and rename files
cp /path/to/dwi.nii.gz tractoflow_input/sub-01/dwi.nii.gz
cp /path/to/dwi.bval tractoflow_input/sub-01/bval
cp /path/to/dwi.bvec tractoflow_input/sub-01/bvec
cp /path/to/t1.nii.gz tractoflow_input/sub-01/t1.nii.gz

# Optionally add reverse phase-encoded b0
cp /path/to/rev_b0.nii.gz tractoflow_input/sub-01/rev_b0.nii.gz
```

### Validate Input

```python
#!/usr/bin/env python3
# validate_tractoflow_input.py

import nibabel as nib
import numpy as np
from pathlib import Path

def validate_subject(subject_dir):
    """Validate TractoFlow input for a subject."""

    errors = []

    # Check required files
    dwi_file = subject_dir / "dwi.nii.gz"
    bval_file = subject_dir / "bval"
    bvec_file = subject_dir / "bvec"

    if not dwi_file.exists():
        errors.append(f"Missing DWI: {dwi_file}")
    if not bval_file.exists():
        errors.append(f"Missing bval: {bval_file}")
    if not bvec_file.exists():
        errors.append(f"Missing bvec: {bvec_file}")

    if errors:
        return errors

    # Load DWI
    dwi_img = nib.load(dwi_file)
    dwi_data = dwi_img.get_fdata()

    # Load bvals and bvecs
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file)

    # Check dimensions
    if dwi_data.ndim != 4:
        errors.append(f"DWI must be 4D, got {dwi_data.ndim}D")

    n_vols = dwi_data.shape[3] if dwi_data.ndim == 4 else 0

    if len(bvals) != n_vols:
        errors.append(f"Bvals count ({len(bvals)}) != DWI volumes ({n_vols})")

    if bvecs.shape[1] != n_vols:
        errors.append(f"Bvecs columns ({bvecs.shape[1]}) != DWI volumes ({n_vols})")

    if bvecs.shape[0] != 3:
        errors.append(f"Bvecs must have 3 rows, got {bvecs.shape[0]}")

    # Check for b0 volumes
    n_b0 = np.sum(bvals < 50)
    if n_b0 == 0:
        errors.append("No b0 volumes found (bval < 50)")

    return errors

def main():
    import sys
    input_dir = Path(sys.argv[1])

    for subject_dir in sorted(input_dir.glob("sub-*")):
        print(f"Validating {subject_dir.name}...")
        errors = validate_subject(subject_dir)

        if errors:
            print(f"  ❌ Errors found:")
            for error in errors:
                print(f"    - {error}")
        else:
            print(f"  ✓ Valid")

if __name__ == "__main__":
    main()
```

---

## Basic Usage

### Local Execution

```bash
# Run TractoFlow on local machine with Singularity
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  -profile singularity \
  -resume

# Use Docker instead
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  -profile docker \
  -resume
```

### Process Specific Subjects

```bash
# Process only selected subjects
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --subjects sub-01 sub-03 \
  -profile singularity \
  -resume
```

### Custom Configuration

```bash
# Use custom configuration file
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  -c custom_config.config \
  -profile singularity \
  -resume
```

**Example Configuration (custom_config.config):**

```groovy
// Resource allocation
process {
    withName: 'Eddy' {
        cpus = 8
        memory = '16 GB'
        time = '4h'
    }

    withName: 'Tractography' {
        cpus = 4
        memory = '8 GB'
        time = '2h'
    }
}

// Singularity settings
singularity {
    enabled = true
    autoMounts = true
    runOptions = '--bind /data:/data'
}

// Working directory
workDir = '/scratch/tractoflow_work'
```

---

## Preprocessing Configuration

### Denoising Options

```bash
# Disable denoising (not recommended)
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --no_denoising \
  -profile singularity

# Adjust denoising extent parameter
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --extent 7 \
  -profile singularity
```

### Topup and Eddy

```bash
# Skip Topup (if no reverse phase-encoding)
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --no_topup \
  -profile singularity

# Eddy with outlier replacement
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --eddy_cmd eddy_cuda \
  -profile singularity
```

### Resampling

```bash
# Custom isotropic resolution (default: 1mm)
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --isotropic_resolution 1.5 \
  -profile singularity
```

---

## Tractography Options

### Basic Tractography Parameters

```bash
# Custom tractography settings
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --number_of_seeds 10 \
  --seeding_strategy npv \
  --tracking_algorithm prob \
  --step_size 0.5 \
  --theta 20 \
  --min_length 20 \
  --max_length 200 \
  -profile singularity
```

**Parameter Explanations:**

- `--number_of_seeds`: Seeds per voxel (default: 10)
- `--seeding_strategy`: `npv` (per voxel) or `nt` (total seeds)
- `--tracking_algorithm`: `prob` (probabilistic) or `det` (deterministic)
- `--step_size`: Step size in mm (default: 0.5)
- `--theta`: Maximum angle between steps in degrees (default: 20)
- `--min_length`: Minimum streamline length in mm (default: 20)
- `--max_length`: Maximum streamline length in mm (default: 200)

### Particle Filtering Tractography (PFT)

```bash
# Enable PFT for anatomically-informed tracking
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --pft \
  -profile singularity
```

### Custom fODF Parameters

```bash
# CSD with custom response function
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --sh_order 8 \
  --basis descoteaux07 \
  -profile singularity
```

---

## HPC Cluster Execution

### SLURM Cluster

```bash
# Run on SLURM cluster
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  -profile slurm,singularity \
  -resume
```

**Custom SLURM Profile (slurm_config.config):**

```groovy
// SLURM configuration
process {
    executor = 'slurm'
    queue = 'compute'
    clusterOptions = '--account=myproject'

    withLabel: 'high_memory' {
        memory = '32 GB'
        cpus = 16
        time = '8h'
    }

    withLabel: 'standard' {
        memory = '8 GB'
        cpus = 4
        time = '4h'
    }
}

executor {
    queueSize = 100
    submitRateLimit = '10 sec'
}

singularity {
    enabled = true
    autoMounts = true
    cacheDir = '/shared/singularity_cache'
}
```

### PBS Cluster

```groovy
// PBS configuration
process {
    executor = 'pbs'
    queue = 'batch'

    withName: 'Eddy' {
        cpus = 8
        memory = '16 GB'
        time = '6h'
        clusterOptions = '-l walltime=6:00:00'
    }
}
```

### Submit to Cluster

```bash
# Submit as SLURM job
sbatch << EOF
#!/bin/bash
#SBATCH --job-name=tractoflow
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

module load nextflow singularity

nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  -profile slurm,singularity \
  -resume
EOF
```

---

## Output Organization

### Output Directory Structure

```
output_directory/
├── sub-01/
│   ├── Bet_DWI/
│   │   ├── sub-01__dwi_bet_mask.nii.gz
│   │   └── sub-01__dwi_bet.nii.gz
│   ├── DTI_Metrics/
│   │   ├── sub-01__fa.nii.gz
│   │   ├── sub-01__md.nii.gz
│   │   ├── sub-01__ad.nii.gz
│   │   ├── sub-01__rd.nii.gz
│   │   ├── sub-01__colored_fa.nii.gz
│   │   └── sub-01__tensor.nii.gz
│   ├── FODF_Metrics/
│   │   ├── sub-01__fodf.nii.gz
│   │   ├── sub-01__peaks.nii.gz
│   │   └── sub-01__nufo.nii.gz
│   ├── Register_T1/
│   │   ├── sub-01__t1_warped.nii.gz
│   │   └── sub-01__output0GenericAffine.mat
│   ├── Tracking/
│   │   ├── sub-01__tracking.trk
│   │   └── sub-01__tracking_pft.trk
│   ├── Preprocessing/
│   │   ├── sub-01__dwi_denoised.nii.gz
│   │   ├── sub-01__dwi_eddy_corrected.nii.gz
│   │   └── sub-01__dwi_preprocessed.nii.gz
│   └── QC/
│       └── (quality control images)
└── reports/
    ├── execution_report.html
    ├── execution_timeline.html
    └── execution_trace.txt
```

### Access Outputs

```python
#!/usr/bin/env python3
# load_tractoflow_outputs.py

import nibabel as nib
from pathlib import Path

def load_subject_outputs(output_dir, subject):
    """Load TractoFlow outputs for a subject."""

    subj_dir = Path(output_dir) / subject
    outputs = {}

    # DTI metrics
    dti_dir = subj_dir / "DTI_Metrics"
    outputs['fa'] = nib.load(dti_dir / f"{subject}__fa.nii.gz")
    outputs['md'] = nib.load(dti_dir / f"{subject}__md.nii.gz")
    outputs['colored_fa'] = nib.load(dti_dir / f"{subject}__colored_fa.nii.gz")

    # fODF
    fodf_dir = subj_dir / "FODF_Metrics"
    outputs['fodf'] = nib.load(fodf_dir / f"{subject}__fodf.nii.gz")

    # Tractography
    from dipy.io.streamline import load_tractogram
    tracking_dir = subj_dir / "Tracking"
    tracking_file = tracking_dir / f"{subject}__tracking.trk"
    outputs['tractogram'] = load_tractogram(tracking_file, 'same')

    return outputs

# Load outputs
outputs = load_subject_outputs("/path/to/output", "sub-01")
print(f"FA shape: {outputs['fa'].shape}")
print(f"Number of streamlines: {len(outputs['tractogram'])}")
```

---

## Quality Control

### Automated QC Reports

TractoFlow generates comprehensive QC reports automatically:

```bash
# QC reports are in output_directory/reports/
ls output_directory/reports/

# execution_report.html - Resource usage and timing
# execution_timeline.html - Visual timeline of processes
# execution_trace.txt - Detailed execution trace
```

### Visual QC with mrview

```bash
# View FA map
mrview output_directory/sub-01/DTI_Metrics/sub-01__fa.nii.gz

# View tractography
mrview output_directory/sub-01/DTI_Metrics/sub-01__fa.nii.gz \
  -tractography.load output_directory/sub-01/Tracking/sub-01__tracking.trk
```

### Custom QC Script

```python
#!/usr/bin/env python3
# qc_tractoflow.py

import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd

def compute_qc_metrics(output_dir, subject):
    """Compute QC metrics for TractoFlow outputs."""

    subj_dir = Path(output_dir) / subject
    metrics = {'subject': subject}

    # Load FA
    fa_file = subj_dir / "DTI_Metrics" / f"{subject}__fa.nii.gz"
    fa_data = nib.load(fa_file).get_fdata()

    # Load brain mask
    mask_file = subj_dir / "Bet_DWI" / f"{subject}__dwi_bet_mask.nii.gz"
    mask_data = nib.load(mask_file).get_fdata().astype(bool)

    # FA statistics
    fa_brain = fa_data[mask_data]
    metrics['fa_mean'] = fa_brain.mean()
    metrics['fa_std'] = fa_brain.std()
    metrics['fa_median'] = np.median(fa_brain)

    # Check for outliers
    metrics['fa_outliers'] = np.sum(fa_brain > 1.0)  # FA should be <= 1

    # Load tractogram
    from dipy.io.streamline import load_tractogram
    trk_file = subj_dir / "Tracking" / f"{subject}__tracking.trk"
    tractogram = load_tractogram(trk_file, 'same')

    metrics['n_streamlines'] = len(tractogram)
    lengths = [len(s) for s in tractogram.streamlines]
    metrics['streamline_length_mean'] = np.mean(lengths)
    metrics['streamline_length_std'] = np.std(lengths)

    return metrics

def main():
    import sys
    output_dir = sys.argv[1]

    results = []
    for subject_dir in sorted(Path(output_dir).glob("sub-*")):
        subject = subject_dir.name
        print(f"Processing {subject}...")
        metrics = compute_qc_metrics(output_dir, subject)
        results.append(metrics)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("tractoflow_qc.csv", index=False)
    print(f"\nQC metrics saved to tractoflow_qc.csv")
    print(df)

if __name__ == "__main__":
    main()
```

---

## Integration with Claude Code

TractoFlow integrates seamlessly with Claude Code for automated batch processing:

```python
# tractoflow_batch.py - Automated TractoFlow execution

import subprocess
from pathlib import Path
import logging
import time

def run_tractoflow(input_dir, output_dir, profile='singularity', resume=True):
    """Execute TractoFlow pipeline."""

    cmd = [
        'nextflow', 'run', 'scilus/tractoflow',
        '--input', str(input_dir),
        '--output', str(output_dir),
        '-profile', profile
    ]

    if resume:
        cmd.append('-resume')

    logging.info(f"Running: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        duration = time.time() - start_time
        logging.info(f"Pipeline completed in {duration:.1f} seconds")

        return result

    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline failed: {e.stderr}")
        raise

def monitor_progress(output_dir):
    """Monitor TractoFlow progress."""

    output_path = Path(output_dir)

    subjects = list(output_path.glob("sub-*"))
    total = len(subjects)

    for i, subj_dir in enumerate(subjects, 1):
        # Check if tracking completed
        tracking_file = subj_dir / "Tracking" / f"{subj_dir.name}__tracking.trk"

        if tracking_file.exists():
            logging.info(f"✓ {subj_dir.name} completed ({i}/{total})")
        else:
            logging.info(f"⏳ {subj_dir.name} in progress ({i}/{total})")

# Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run_tractoflow(
        input_dir="/data/diffusion_input",
        output_dir="/data/tractoflow_output",
        profile='singularity',
        resume=True
    )

    monitor_progress("/data/tractoflow_output")
```

**Integration with Study Management:**

```python
#!/usr/bin/env python3
# integrate_tractoflow_study.py

import pandas as pd
from pathlib import Path
import subprocess

class TractoFlowStudy:
    """Manage TractoFlow processing for a study."""

    def __init__(self, study_dir):
        self.study_dir = Path(study_dir)
        self.input_dir = self.study_dir / "tractoflow_input"
        self.output_dir = self.study_dir / "tractoflow_output"

    def prepare_inputs(self, bids_dir):
        """Convert BIDS to TractoFlow input format."""

        from bids import BIDSLayout
        layout = BIDSLayout(bids_dir)

        subjects = layout.get_subjects()
        for subject in subjects:
            subj_input_dir = self.input_dir / f"sub-{subject}"
            subj_input_dir.mkdir(parents=True, exist_ok=True)

            # Get DWI file
            dwi_files = layout.get(
                subject=subject,
                datatype='dwi',
                suffix='dwi',
                extension='nii.gz'
            )

            if dwi_files:
                dwi_file = dwi_files[0]

                # Copy DWI and sidecar files
                subprocess.run([
                    'cp', dwi_file.path,
                    subj_input_dir / 'dwi.nii.gz'
                ])

                # Copy bval and bvec
                bval_file = dwi_file.path.replace('.nii.gz', '.bval')
                bvec_file = dwi_file.path.replace('.nii.gz', '.bvec')

                subprocess.run(['cp', bval_file, subj_input_dir / 'bval'])
                subprocess.run(['cp', bvec_file, subj_input_dir / 'bvec'])

            # Get T1
            t1_files = layout.get(
                subject=subject,
                datatype='anat',
                suffix='T1w',
                extension='nii.gz'
            )

            if t1_files:
                subprocess.run([
                    'cp', t1_files[0].path,
                    subj_input_dir / 't1.nii.gz'
                ])

        print(f"Prepared {len(subjects)} subjects")

    def run_pipeline(self):
        """Execute TractoFlow."""

        run_tractoflow(
            self.input_dir,
            self.output_dir,
            profile='singularity'
        )

    def extract_metrics(self):
        """Extract DTI metrics for all subjects."""

        results = []
        for subj_dir in self.output_dir.glob("sub-*"):
            subject = subj_dir.name

            # Load FA
            fa_file = subj_dir / "DTI_Metrics" / f"{subject}__fa.nii.gz"
            if fa_file.exists():
                fa_img = nib.load(fa_file)
                fa_data = fa_img.get_fdata()

                results.append({
                    'subject': subject,
                    'fa_mean': fa_data.mean(),
                    'fa_std': fa_data.std()
                })

        df = pd.DataFrame(results)
        df.to_csv(self.study_dir / "dti_metrics.csv", index=False)
        return df

# Usage
study = TractoFlowStudy("/data/my_study")
study.prepare_inputs("/data/my_study_BIDS")
study.run_pipeline()
metrics = study.extract_metrics()
```

---

## Integration with Other Tools

### MRtrix3 Integration

```bash
# Use TractoFlow outputs with MRtrix3
mrconvert \
  output_directory/sub-01/FODF_Metrics/sub-01__fodf.nii.gz \
  sub-01_fodf.mif

# Perform tractography with MRtrix3
tckgen \
  sub-01_fodf.mif \
  sub-01_mrtrix_tracks.tck \
  -seed_image output_directory/sub-01/Bet_DWI/sub-01__dwi_bet_mask.nii.gz \
  -select 1000000
```

### DIPY Integration

```python
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import Streamlines
import nibabel as nib

# Load TractoFlow tractogram
tractogram = load_tractogram(
    "output_directory/sub-01/Tracking/sub-01__tracking.trk",
    reference='same'
)

# Filter short streamlines
from dipy.tracking.streamline import length
lengths = length(tractogram.streamlines)
long_streamlines = tractogram.streamlines[lengths > 30]

# Save filtered tractogram
filtered_tractogram = Streamlines(long_streamlines)
save_tractogram(
    filtered_tractogram,
    "sub-01_filtered.trk",
    affine=tractogram.affine,
    vox_size=tractogram.voxel_sizes
)
```

### Connectome Generation

```python
#!/usr/bin/env python3
# generate_connectome_from_tractoflow.py

import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import connectivity_matrix

def generate_connectome(tractoflow_output, subject, atlas_file):
    """Generate connectivity matrix from TractoFlow outputs."""

    # Load tractogram
    trk_file = (
        f"{tractoflow_output}/{subject}/Tracking/"
        f"{subject}__tracking.trk"
    )
    tractogram = load_tractogram(trk_file, 'same')

    # Load atlas
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata().astype(int)

    # Generate connectivity matrix
    M, grouping = connectivity_matrix(
        tractogram.streamlines,
        affine=tractogram.affine,
        label_volume=atlas_data,
        return_mapping=True,
        mapping_as_streamlines=False
    )

    return M, grouping

# Usage
M, grouping = generate_connectome(
    "/path/to/tractoflow_output",
    "sub-01",
    "/path/to/atlas.nii.gz"
)

print(f"Connectivity matrix shape: {M.shape}")
np.save("sub-01_connectome.npy", M)
```

---

## Troubleshooting

### Problem 1: Nextflow Out of Memory

**Symptoms:** Pipeline crashes with Java heap space error

**Solution:**
```bash
# Increase Nextflow memory
export NXF_OPTS='-Xms1g -Xmx4g'

# Or set in config
env {
    NXF_OPTS = '-Xms1g -Xmx4g'
}
```

### Problem 2: Singularity Container Not Found

**Symptoms:** Error downloading or building container

**Solution:**
```bash
# Pre-pull container
singularity pull library://scilus/tractoflow/tractoflow:latest

# Or specify cache directory
export SINGULARITY_CACHEDIR=/path/to/cache

# Use local container
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  -with-singularity /path/to/container.sif
```

### Problem 3: Eddy Fails with CUDA Error

**Symptoms:** FSL Eddy crashes on GPU

**Solution:**
```bash
# Use CPU version of Eddy
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  --eddy_cmd eddy_openmp \
  -profile singularity
```

### Problem 4: Insufficient Disk Space

**Symptoms:** Pipeline fails with disk space error

**Solution:**
```bash
# Use different work directory
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  -work-dir /scratch/work \
  -profile singularity

# Clean work directory after completion
nextflow clean -f
```

### Problem 5: Resume Not Working

**Symptoms:** Pipeline restarts from beginning

**Solution:**
```bash
# Ensure work directory is preserved
# Use -resume flag (with single dash)
nextflow run tractoflow/main.nf \
  --input /path/to/input \
  --output /path/to/output \
  -resume \
  -profile singularity

# Check .nextflow.log for details
cat .nextflow.log
```

---

## Best Practices

### 1. Data Preparation

- **Validate inputs:** Check DWI dimensions, bvals, bvecs before running
- **Consistent naming:** Use standardized subject IDs
- **Quality check:** Visual inspection of raw DWI data
- **Reverse phase-encoding:** Include for Topup when available
- **T1 images:** Always include for better registration and PFT

### 2. Pipeline Execution

- **Start small:** Test on 1-2 subjects before full dataset
- **Use resume:** Always include `-resume` flag
- **Monitor resources:** Check CPU/memory usage
- **Working directory:** Use fast scratch filesystem
- **Container caching:** Pre-download containers before large runs

### 3. Quality Control

- **Visual inspection:** Review outputs for each subject
- **Automated metrics:** Use QC scripts to detect outliers
- **Compare subjects:** Look for systematic differences
- **Check logs:** Review Nextflow reports for errors
- **Validate tractograms:** Ensure anatomically plausible streamlines

### 4. HPC Optimization

- **Resource allocation:** Match to pipeline requirements
- **Parallel submission:** Let Nextflow manage job submission
- **Queue selection:** Use appropriate cluster queues
- **Time limits:** Set generous walltime for Eddy and tracking
- **Checkpointing:** Preserve work directories for resume

### 5. Reproducibility

- **Version tracking:** Record TractoFlow, Nextflow, container versions
- **Configuration files:** Save all pipeline parameters
- **Random seeds:** Set for reproducible tractography
- **Container snapshots:** Archive exact container versions used
- **Documentation:** Log any manual interventions

---

## Resources

### Official Documentation

- **TractoFlow Documentation:** https://tractoflow-documentation.readthedocs.io/
- **GitHub Repository:** https://github.com/scilus/tractoflow
- **SCIL Lab:** https://scil.usherbrooke.ca/

### Publications

- **TractoFlow Paper:** Theaud et al. (2020) "TractoFlow: A robust, efficient and reproducible diffusion MRI pipeline leveraging Nextflow & Singularity" *NeuroImage*
- **Nextflow:** Di Tommaso et al. (2017) "Nextflow enables reproducible computational workflows"

### Community Support

- **GitHub Issues:** https://github.com/scilus/tractoflow/issues
- **SCIL Contact:** Email scil@usherbrooke.ca

### Related Resources

- **Nextflow Documentation:** https://www.nextflow.io/docs/latest/
- **Singularity Documentation:** https://sylabs.io/docs/
- **FSL Eddy:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy

---

## Citation

```bibtex
@article{theaud2020tractoflow,
  title={TractoFlow: A robust, efficient and reproducible diffusion MRI pipeline leveraging Nextflow \& Singularity},
  author={Theaud, Guillaume and Houde, Jean-Christophe and Bor{\'e}, Arnaud and Rheault, Fran{\c{c}}ois and Morency, Felix and Descoteaux, Maxime},
  journal={NeuroImage},
  volume={218},
  pages={116889},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.neuroimage.2020.116889}
}
```

---

## Related Tools

- **MRtrix3:** Advanced diffusion MRI processing (see `mrtrix3.md`)
- **DIPY:** Diffusion imaging in Python (see `dipy.md`)
- **QSIPrep:** Alternative diffusion preprocessing (see `qsiprep.md`)
- **DSI Studio:** Diffusion MRI analysis (see `dsistudio.md`)
- **FSL:** FMRIB Software Library with Eddy and Topup (see `fsl.md`)
- **ANTs:** Advanced normalization tools (see `ants.md`)
- **Clinica:** Clinical neuroimaging platform (see `clinica.md`)
- **Pydra:** Workflow engine alternative (see `pydra.md`)
- **Snakebids:** BIDS workflow framework (see `snakebids.md`)
- **Nextflow:** Workflow management system

---

**Skill Version:** 1.0
**Last Updated:** 2025-11
**TractoFlow Version Covered:** 2.x
**Maintainer:** Claude Code Neuroimaging Skills
