# Pydra - Next-Generation Dataflow Engine

## Overview

Pydra is a lightweight, Python-based dataflow engine for computational graph construction and execution. Developed as the successor to NiPype (NiPype 2.0), Pydra introduces modern design patterns including lazy evaluation, content-addressable caching, and type-safe interfaces. Unlike traditional workflow systems, Pydra treats tasks as pure functions with explicit inputs and outputs, enabling sophisticated provenance tracking, efficient result reuse, and parallel execution across various compute backends (local, HPC clusters, cloud).

**Website:** https://pydra.readthedocs.io/
**Platform:** Python (Cross-platform)
**License:** Apache 2.0
**Key Application:** Reproducible neuroimaging workflows, parallel processing, container-based pipelines

### Why Pydra?

**Advantages over NiPype 1.x:**
- **Lazy evaluation** - Tasks execute only when outputs needed
- **Hash-based caching** - Automatic result reuse based on inputs
- **Type hints** - Better error detection and IDE support
- **Modern Python** - Python 3.7+ with dataclasses
- **Simplified API** - Cleaner task/workflow construction
- **Better parallelization** - Splitters/combiners for data parallelism

**Key Concepts:**
- Tasks as pure functions with explicit dependencies
- Directed acyclic graphs (DAGs) for workflows
- Content-addressable storage for results
- Container execution (Docker, Singularity) as first-class citizens

## Key Features

- **Lazy evaluation** - Efficient execution planning
- **Content-addressable caching** - Hash-based result reuse
- **Type-safe interfaces** - Python type hints for inputs/outputs
- **Splitters and combiners** - Data parallelism patterns
- **Container support** - Docker and Singularity integration
- **Multiple backends** - Local, SLURM, Dask, SGE
- **State management** - Track parameter variations
- **Provenance tracking** - Complete computational history
- **Plugin architecture** - Extend with custom functionality
- **Audit trails** - Reproducibility and debugging
- **Python 3.7+** - Modern language features
- **Active development** - NIPY community support

## Installation

### Basic Installation

```bash
# Install Pydra
pip install pydra

# Or with all extras
pip install pydra[all]

# For development
pip install pydra[dev]
```

### With Container Support

```bash
# For Docker support
pip install pydra[docker]

# For Singularity support
pip install pydra[singularity]

# Both
pip install pydra[container]
```

### With Cluster Support

```bash
# For SLURM
pip install pydra[slurm]

# For Dask distributed
pip install pydra[dask]

# For all backends
pip install pydra[all]
```

### Verify Installation

```python
import pydra
print(f"Pydra version: {pydra.__version__}")

# Test basic functionality
from pydra import Workflow
wf = Workflow(name="test", input_spec=["x"])
print("Pydra installed successfully!")
```

## Core Concepts

### Tasks

Tasks are the basic units of computation:

```python
from pydra import Workflow
from pydra.mark import task, annotate

# Define a simple task using function
@task
@annotate({"return": {"result": int}})
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

# Create task instance
add_task = add(x=5, y=3)

# Execute
result = add_task()
print(f"Result: {result.output.result}")  # 8
```

### Workflows

Workflows connect multiple tasks:

```python
from pydra import Workflow

# Create workflow
wf = Workflow(name="math_wf", input_spec=["a", "b", "c"])

# Add tasks
wf.add(add(name="add1", x=wf.lzin.a, y=wf.lzin.b))
wf.add(add(name="add2", x=wf.add1.lzout.result, y=wf.lzin.c))

# Set output
wf.set_output([("final", wf.add2.lzout.result)])

# Execute workflow
wf.inputs.a = 10
wf.inputs.b = 20
wf.inputs.c = 5

result = wf()
print(f"Final result: {result.output.final}")  # 35
```

### Lazy Inputs and Outputs

```python
# Lazy inputs: wf.lzin.input_name
# Lazy outputs: task_name.lzout.output_name

wf = Workflow(name="example", input_spec=["input_file"])

# Connect workflow input to task
wf.add(process_task(name="proc", input_file=wf.lzin.input_file))

# Connect task output to another task
wf.add(analyze_task(name="analyze", 
                    data=wf.proc.lzout.output_data))

# Lazy evaluation: tasks execute only when results needed
```

## Shell Command Tasks

### Basic Shell Commands

```python
from pydra.tasks.shell import ShellCommandTask

# Simple command
task = ShellCommandTask(
    name="list_files",
    executable="ls",
    args="-lh"
)

result = task()
print(result.output.stdout)
```

### With Input/Output Specification

```python
# Brain extraction with FSL bet
bet_task = ShellCommandTask(
    name="bet",
    executable="bet",
    input_spec={
        "input_file": str,
        "output_file": str,
        "frac": float
    },
    output_spec={
        "brain": str,
        "mask": str
    },
    args="",  # Built from inputs
)

# Build command from inputs
def bet_args(input_file, output_file, frac=0.5):
    return f"{input_file} {output_file} -f {frac} -m"

bet_task.args = bet_args

# Execute
bet_task.inputs.input_file = "T1w.nii.gz"
bet_task.inputs.output_file = "T1w_brain.nii.gz"
bet_task.inputs.frac = 0.5

result = bet_task()
```

### Neuroimaging Tools

```python
# FreeSurfer recon-all
from pydra.tasks.shell import ShellCommandTask

recon_task = ShellCommandTask(
    name="recon_all",
    executable="recon-all",
    input_spec={
        "subject_id": str,
        "input_file": str,
        "directive": str
    },
    args=""
)

def recon_args(subject_id, input_file, directive="all"):
    return f"-s {subject_id} -i {input_file} -{directive}"

recon_task.inputs.subject_id = "sub-01"
recon_task.inputs.input_file = "/data/sub-01_T1w.nii.gz"
recon_task.inputs.directive = "all"

# Execute (takes hours)
result = recon_task()
```

## Function Tasks

### Python Function as Task

```python
from pydra.mark import task, annotate
import nibabel as nib
import numpy as np

@task
@annotate({"return": {"mean_intensity": float}})
def compute_mean(image_path: str) -> float:
    """Compute mean intensity of brain image."""
    img = nib.load(image_path)
    data = img.get_fdata()
    return float(np.mean(data[data > 0]))

# Use in workflow
mean_task = compute_mean(image_path="brain.nii.gz")
result = mean_task()
print(f"Mean intensity: {result.output.mean_intensity}")
```

### With Multiple Outputs

```python
@task
@annotate({"return": {"mean": float, "std": float, "min": float, "max": float}})
def image_stats(image_path: str) -> dict:
    """Compute image statistics."""
    img = nib.load(image_path)
    data = img.get_fdata()
    mask = data > 0
    
    return {
        "mean": float(np.mean(data[mask])),
        "std": float(np.std(data[mask])),
        "min": float(np.min(data[mask])),
        "max": float(np.max(data[mask]))
    }

stats_task = image_stats(image_path="T1w.nii.gz")
result = stats_task()
print(f"Stats: {result.output}")
```

## Splitters and Combiners

### Map Over Inputs

```python
from pydra import Workflow

# Process multiple subjects in parallel
wf = Workflow(name="multi_subject", input_spec=["subjects"])

# Split over subjects
wf.split("subjects")

# Add task that processes each subject
wf.add(process_subject(
    name="proc",
    subject_id=wf.lzin.subjects
))

# Combine results
wf.combine("subjects")

# Execute with multiple subjects
wf.inputs.subjects = ["sub-01", "sub-02", "sub-03", "sub-04"]
results = wf()

# Access results for each subject
for i, subj in enumerate(wf.inputs.subjects):
    print(f"{subj}: {results[i].output}")
```

### Multiple Splitters

```python
# Split over subjects and sessions
wf = Workflow(
    name="multi_subject_session",
    input_spec=["subjects", "sessions"]
)

# Split over both dimensions
wf.split(("subjects", "sessions"))

# Process each combination
wf.add(process_data(
    name="proc",
    subject=wf.lzin.subjects,
    session=wf.lzin.sessions
))

wf.combine(("subjects", "sessions"))

# Execute
wf.inputs.subjects = ["sub-01", "sub-02"]
wf.inputs.sessions = ["ses-01", "ses-02"]
results = wf()

# Results: 2 subjects × 2 sessions = 4 results
```

## Caching

### Automatic Caching

```python
# Pydra automatically caches based on input hash
import time

@task
@annotate({"return": {"result": int}})
def slow_computation(x: int) -> int:
    time.sleep(5)  # Simulate slow operation
    return x * x

# First run: takes 5 seconds
task1 = slow_computation(x=10)
result1 = task1()  # Slow

# Second run with same input: instant
task2 = slow_computation(x=10)
result2 = task2()  # Fast! Uses cached result

# Different input: runs computation
task3 = slow_computation(x=20)
result3 = task3()  # Slow again
```

### Cache Directory

```python
from pydra import Workflow
from pathlib import Path

# Set cache directory
cache_dir = Path("/data/pydra_cache")
cache_dir.mkdir(exist_ok=True)

wf = Workflow(
    name="cached_wf",
    input_spec=["input_file"],
    cache_dir=cache_dir
)

# All tasks in workflow will use this cache
```

### Disable Caching

```python
# Disable cache for specific task
task = process_data(
    name="no_cache",
    input_file="data.nii.gz",
    cache_dir=None  # No caching
)
```

## Container Execution

### Docker Tasks

```python
from pydra.tasks.docker import DockerTask

# Run FSL bet in Docker container
bet_docker = DockerTask(
    name="bet_docker",
    executable="bet",
    image="brainlife/fsl:6.0.4",
    input_spec={
        "input_file": str,
        "output_file": str
    },
    args=""
)

def bet_args_func(input_file, output_file):
    return f"{input_file} {output_file} -f 0.5 -m"

bet_docker.args = bet_args_func

# Bind mount directories
bet_docker.inputs.input_file = "/data/T1w.nii.gz"
bet_docker.inputs.output_file = "/data/T1w_brain.nii.gz"

# Execute in container
result = bet_docker(
    bindings={
        "/local/data": ("/data", "rw")  # (host_path, container_path, mode)
    }
)
```

### Singularity Tasks

```python
from pydra.tasks.singularity import SingularityTask

# Run in Singularity container
freesurfer_task = SingularityTask(
    name="recon",
    executable="recon-all",
    image="/containers/freesurfer-7.3.2.sif",
    input_spec={
        "subject": str,
        "input_file": str
    },
    args=lambda subject, input_file: f"-s {subject} -i {input_file} -all"
)

freesurfer_task.inputs.subject = "sub-01"
freesurfer_task.inputs.input_file = "/data/T1w.nii.gz"

result = freesurfer_task(
    bindings={
        "/local/data": ("/data", "ro"),
        "/local/subjects": ("/subjects", "rw")
    }
)
```

## Parallel Execution

### Local Multiprocessing

```python
from pydra import Workflow

wf = Workflow(name="parallel_wf", input_spec=["subjects"])
wf.split("subjects")

wf.add(process_subject(name="proc", subject=wf.lzin.subjects))
wf.combine("subjects")

wf.inputs.subjects = [f"sub-{i:02d}" for i in range(1, 21)]

# Execute with multiple processes
result = wf(plugin="cf", plugin_args={"n_procs": 8})
```

### SLURM Cluster

```python
# Submit to SLURM cluster
from pydra import Workflow

wf = Workflow(name="slurm_wf", input_spec=["subjects"])
wf.split("subjects")
wf.add(heavy_processing(name="proc", subject=wf.lzin.subjects))
wf.combine("subjects")

wf.inputs.subjects = [f"sub-{i:02d}" for i in range(1, 101)]

# SLURM plugin configuration
slurm_config = {
    "sbatch_args": "-p normal -n 1 -c 4 --mem=16G -t 4:00:00"
}

result = wf(plugin="slurm", plugin_args=slurm_config)
```

### Dask Distributed

```python
from dask.distributed import Client
from pydra import Workflow

# Start Dask cluster
client = Client()

wf = Workflow(name="dask_wf", input_spec=["files"])
wf.split("files")
wf.add(process_file(name="proc", file=wf.lzin.files))
wf.combine("files")

wf.inputs.files = [f"file_{i}.nii.gz" for i in range(100)]

# Execute on Dask cluster
result = wf(plugin="dask")

client.close()
```

## Complete Example: fMRI Preprocessing

### FSL FEAT Workflow

```python
from pydra import Workflow
from pydra.tasks.shell import ShellCommandTask
from pydra.mark import task, annotate
import nibabel as nib

# Create workflow
preproc_wf = Workflow(
    name="fmri_preproc",
    input_spec=["func_file", "anat_file", "output_dir"]
)

# 1. Brain extraction on anatomical
bet_anat = ShellCommandTask(
    name="bet_anat",
    executable="bet",
    args=lambda input, output: f"{input} {output} -f 0.5 -m"
)
preproc_wf.add(bet_anat(
    name="bet",
    input=preproc_wf.lzin.anat_file,
    output=f"{preproc_wf.lzin.output_dir}/anat_brain.nii.gz"
))

# 2. Motion correction
mcflirt = ShellCommandTask(
    name="mcflirt",
    executable="mcflirt",
    args=lambda input, output: f"-in {input} -out {output} -plots"
)
preproc_wf.add(mcflirt(
    name="mc",
    input=preproc_wf.lzin.func_file,
    output=f"{preproc_wf.lzin.output_dir}/func_mc.nii.gz"
))

# 3. Brain extraction on functional
bet_func = ShellCommandTask(
    name="bet_func",
    executable="bet",
    args=lambda input, output: f"{input} {output} -f 0.3 -F"
)
preproc_wf.add(bet_func(
    name="bet_func",
    input=preproc_wf.mc.lzout.output,
    output=f"{preproc_wf.lzin.output_dir}/func_brain.nii.gz"
))

# 4. Smoothing
susan = ShellCommandTask(
    name="susan",
    executable="susan",
    args=lambda input, bt, output: f"{input} {bt} 0.75 3 1 1 {output}"
)
preproc_wf.add(susan(
    name="smooth",
    input=preproc_wf.bet_func.lzout.output,
    bt=2000,
    output=f"{preproc_wf.lzin.output_dir}/func_smooth.nii.gz"
))

# Set workflow output
preproc_wf.set_output([
    ("anat_brain", preproc_wf.bet.lzout.output),
    ("func_preprocessed", preproc_wf.smooth.lzout.output)
])

# Execute
preproc_wf.inputs.func_file = "/data/sub-01_bold.nii.gz"
preproc_wf.inputs.anat_file = "/data/sub-01_T1w.nii.gz"
preproc_wf.inputs.output_dir = "/data/derivatives/sub-01"

result = preproc_wf()
```

## State Management

### Track Parameter Variations

```python
from pydra import Workflow

wf = Workflow(name="param_sweep", input_spec=["input_file", "frac_values"])

# Split over parameter values
wf.split("frac_values")

# Brain extraction with varying threshold
wf.add(bet_task(
    name="bet",
    input_file=wf.lzin.input_file,
    frac=wf.lzin.frac_values
))

wf.combine("frac_values")

# Test multiple thresholds
wf.inputs.input_file = "T1w.nii.gz"
wf.inputs.frac_values = [0.3, 0.4, 0.5, 0.6, 0.7]

results = wf()

# Compare results
for i, frac in enumerate(wf.inputs.frac_values):
    print(f"frac={frac}: {results[i].output}")
```

## Debugging and Provenance

### Enable Logging

```python
import logging
from pydra import Workflow

# Configure logging
logging.basicConfig(level=logging.DEBUG)

wf = Workflow(name="debug_wf", input_spec=["input"])
# ... add tasks

# Execution will print detailed logs
result = wf()
```

### Inspect Task Outputs

```python
# Check task outputs before full workflow execution
task = process_data(input_file="test.nii.gz")

# Dry run
print(f"Task inputs: {task.inputs}")
print(f"Task command: {task.cmdline}")

# Execute
result = task()

# Inspect
print(f"Return code: {result.output.return_code}")
print(f"stdout: {result.output.stdout}")
print(f"stderr: {result.output.stderr}")
```

### Provenance

```python
# Pydra automatically tracks provenance
result = wf()

# Access provenance
print(f"Task hash: {task.checksum}")
print(f"Cache directory: {task.cache_dir}")
print(f"Output directory: {task.output_dir}")

# Audit files stored in cache
# .pydra_provenance.json contains full execution history
```

## Integration with Claude Code

Pydra enables Claude-assisted workflow development:

### Workflow Generation

```markdown
**Prompt to Claude:**
"Create a Pydra workflow for multi-subject fMRI preprocessing:
1. Brain extraction (FSL bet)
2. Motion correction (mcflirt)  
3. Slice timing correction
4. Spatial smoothing (5mm FWHM)
5. Temporal filtering (0.01-0.1 Hz)
Process 20 subjects in parallel using 8 cores.
Include quality control outputs."
```

### Container Pipeline

```markdown
**Prompt to Claude:**
"Build Pydra workflow using containers:
- fMRIPrep (Docker) for preprocessing
- FreeSurfer (Singularity) for anatomical
- Custom Python task for QC metrics
Execute on SLURM cluster with appropriate resources.
Include caching and error handling."
```

### Migration from NiPype

```markdown
**Prompt to Claude:**
"Convert this NiPype workflow to Pydra:
[paste NiPype code]
Maintain same functionality but use:
- Lazy evaluation
- Type hints
- Modern splitters/combiners
- Container execution
Document improvements."
```

## Integration with Other Tools

### With NiBabel

```python
import nibabel as nib
from pydra.mark import task, annotate

@task
@annotate({"return": {"output_file": str}})
def resample_image(input_file: str, output_file: str, voxel_size: tuple) -> str:
    """Resample image to new voxel size."""
    img = nib.load(input_file)
    
    # Resample (using nibabel operations)
    from scipy.ndimage import zoom
    zoom_factors = [old / new for old, new in zip(img.header.get_zooms(), voxel_size)]
    
    new_data = zoom(img.get_fdata(), zoom_factors, order=1)
    new_img = nib.Nifti1Image(new_data, img.affine)
    
    nib.save(new_img, output_file)
    return output_file

# Use in workflow
resample = resample_image(
    input_file="T1w.nii.gz",
    output_file="T1w_2mm.nii.gz",
    voxel_size=(2.0, 2.0, 2.0)
)
```

### With ANTs

```python
from pydra.tasks.shell import ShellCommandTask

# ANTs registration
ants_reg = ShellCommandTask(
    name="ants_registration",
    executable="antsRegistrationSyN.sh",
    input_spec={
        "dimension": int,
        "fixed": str,
        "moving": str,
        "output_prefix": str
    },
    args=lambda d, f, m, o: f"-d {d} -f {f} -m {m} -o {o}"
)

ants_reg.inputs.dimension = 3
ants_reg.inputs.fixed = "template.nii.gz"
ants_reg.inputs.moving = "subject.nii.gz"
ants_reg.inputs.output_prefix = "sub_to_template_"

result = ants_reg()
```

### With BIDS

```python
from bids import BIDSLayout
from pydra import Workflow
from pathlib import Path

# Load BIDS dataset
layout = BIDSLayout("/data/bids_dataset")

# Get all T1w images
t1w_files = layout.get(datatype='anat', suffix='T1w', extension='nii.gz')

# Create workflow
wf = Workflow(name="bids_processing", input_spec=["t1w_files"])
wf.split("t1w_files")

# Process each T1w
wf.add(process_t1w(name="proc", t1w=wf.lzin.t1w_files))

wf.combine("t1w_files")

# Execute
wf.inputs.t1w_files = [f.path for f in t1w_files]
results = wf(plugin="cf", plugin_args={"n_procs": 4})
```

## Troubleshooting

### Problem 1: Task Not Executing

**Symptoms:** Task seems to do nothing

**Solutions:**
```python
# Pydra uses lazy evaluation
# Must call task to execute:
result = task()  # Don't forget ()!

# Or workflow
result = wf()

# Check if cached
print(f"Cache dir: {task.cache_dir}")
# Delete cache to force re-execution
```

### Problem 2: Import Errors in Tasks

**Symptoms:** Module not found inside task function

**Solutions:**
```python
# Imports must be inside task function
@task
def process_data(input_file: str):
    # Import HERE, not at module level
    import nibabel as nib
    import numpy as np
    
    img = nib.load(input_file)
    # ... process
    
    return result
```

### Problem 3: Container Path Issues

**Symptoms:** File not found in container

**Solutions:**
```python
# Ensure bindings are correct
task = DockerTask(
    name="task",
    image="...",
    # ...
)

result = task(
    bindings={
        "/local/data": ("/data", "rw"),  # Local:Container
    }
)

# Use container paths in task inputs
task.inputs.input_file = "/data/file.nii.gz"  # Container path!
```

### Problem 4: SLURM Jobs Not Submitting

**Symptoms:** Jobs stay in queue or fail

**Solutions:**
```python
# Check SLURM configuration
plugin_args = {
    "sbatch_args": "-p partition -n 1 -c 4 --mem=16G -t 2:00:00",
    "poll_interval": 5  # Check job status every 5 seconds
}

# Test with single task first
result = task(plugin="slurm", plugin_args=plugin_args)

# Check job logs in cache directory
```

## Best Practices

1. **Use type hints** - Enables better error checking
2. **Import inside tasks** - Avoid serialization issues
3. **Test locally first** - Before cluster submission
4. **Leverage caching** - Speeds up development
5. **Use containers** - Reproducible environments
6. **Split appropriately** - Balance parallelism and overhead
7. **Monitor cache size** - Can grow large
8. **Document workflows** - Explain task connections
9. **Version control** - Track workflow changes
10. **Start simple** - Add complexity incrementally

## Resources

### Official Documentation

- **Website:** https://pydra.readthedocs.io/
- **GitHub:** https://github.com/nipype/pydra
- **Tutorial:** https://pydra.readthedocs.io/en/latest/tutorial.html
- **API Reference:** https://pydra.readthedocs.io/en/latest/api.html

### Key Publications

- **NiPype:** Gorgolewski et al. (2011) "Nipype: A flexible, lightweight and extensible neuroimaging data processing framework" Front. Neuroinform.

### Learning Resources

- **Examples:** https://github.com/nipype/pydra/tree/master/pydra/engine/tests
- **Workshops:** OHBM Pydra tutorials
- **Video Tutorials:** NeuroStars Pydra discussions

### Community Support

- **GitHub Issues:** https://github.com/nipype/pydra/issues
- **NeuroStars:** https://neurostars.org/ (tag: pydra)
- **NIPY Slack:** https://nipy.org/community.html

## Citation

```bibtex
@software{pydra,
  title = {Pydra: Lightweight dataflow engine for computational graphs},
  author = {{Pydra Developers}},
  year = {2020},
  url = {https://github.com/nipype/pydra},
  note = {Python package}
}

@article{Gorgolewski2011,
  title = {Nipype: A flexible, lightweight and extensible neuroimaging data processing framework in Python},
  author = {Gorgolewski, Krzysztof and Burns, Christopher D and Madison, Cindee and Clark, Dav and Halchenko, Yaroslav O and Waskom, Michael L and Ghosh, Satrajit S},
  journal = {Frontiers in Neuroinformatics},
  volume = {5},
  pages = {13},
  year = {2011},
  doi = {10.3389/fninf.2011.00013}
}
```

## Related Tools

- **NiPype** - Original neuroimaging workflow framework (predecessor)
- **Snakemake** - Python-based workflow management
- **Nextflow** - DSL for data-driven workflows
- **Apache Airflow** - Workflow orchestration platform
- **Luigi** - Python workflow engine (Spotify)
- **Prefect** - Modern workflow orchestration
- **Dask** - Parallel computing library
- **Nipype1** - Legacy version of Pydra

---

**Skill Type:** Workflow Engine
**Difficulty Level:** Intermediate to Advanced
**Prerequisites:** Python 3.7+, Basic neuroimaging knowledge, Understanding of dataflow concepts
**Typical Use Cases:** Reproducible neuroimaging pipelines, parallel processing, container-based workflows, multi-subject analysis
