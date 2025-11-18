# fMRIflows: Modular fMRI Preprocessing Workflow Framework

## Overview

fMRIflows is a flexible, modular framework for constructing custom fMRI preprocessing and analysis workflows. Unlike monolithic pipelines (fMRIPrep, SPM, AFNI proc.py) that follow fixed processing sequences, fMRIflows provides building blocks that researchers can combine and configure to create tailored workflows optimized for specific research questions, datasets, or methodological innovations.

Built on Nipype (Neuroimaging in Python Pipelines and Interfaces), fMRIflows enables users to mix and match processing steps from different software packages (FSL, AFNI, SPM, ANTs), implement custom algorithms, test alternative preprocessing strategies, and maintain full transparency and reproducibility. This modularity is particularly valuable for methods development, comparing preprocessing approaches, and adapting workflows to non-standard datasets.

**Key Features:**
- Modular preprocessing components (motion correction, registration, smoothing, etc.)
- Mix-and-match backends (FSL, AFNI, SPM, ANTs, custom Python)
- Integration with fMRIPrep outputs for custom post-processing
- Flexible confound regression strategies
- Quality control modules with custom metrics
- HPC-compatible parallel execution
- Reproducible workflow graphs and provenance tracking
- Extensive customization without reinventing the wheel

**Primary Use Cases:**
- Testing novel preprocessing strategies and comparing methods
- Custom workflows for non-standard data (multi-echo, real-time, task-based with complex designs)
- Post-fMRIPrep custom processing (specialized denoising, additional confound regression)
- Multi-site studies requiring harmonized but adaptable pipelines
- Method development and validation studies
- Educational purposes (learning preprocessing step-by-step)

**Citation:**
```
fMRIflows uses Nipype as its core engine:
Gorgolewski, K. J., et al. (2011). Nipype: A flexible, lightweight and extensible
neuroimaging data processing framework in Python. Frontiers in Neuroinformatics, 5, 13.
```

## Installation

### Python Package Installation

```bash
# Create dedicated environment
conda create -n fmriflows python=3.9
conda activate fmriflows

# Install fMRIflows
pip install fmriflows

# Or install from GitHub for latest version
pip install git+https://github.com/fmriflows/fmriflows.git

# Install Nipype (core dependency)
pip install nipype
```

### Backend Software Dependencies

fMRIflows requires neuroimaging software depending on which modules you use:

```bash
# FSL (for MCFLIRT, FLIRT, FNIRT, etc.)
# Download from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh

# AFNI (for 3dvolreg, 3dTshift, etc.)
# Download from: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html
export PATH=/usr/local/afni:$PATH

# ANTs (for registration)
# Download from: https://github.com/ANTsX/ANTs
export ANTSPATH=/usr/local/ants/bin
export PATH=$ANTSPATH:$PATH

# SPM12 with MATLAB (optional)
# Download from: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
```

### Docker Installation (All-in-One)

```bash
# Pull Docker image with all dependencies
docker pull fmriflows/fmriflows:latest

# Run fMRIflows in container
docker run -it --rm \
  -v /path/to/data:/data \
  fmriflows/fmriflows:latest \
  python /data/my_workflow.py
```

### Testing Installation

```bash
# Test Nipype and backend interfaces
python -c "from nipype.interfaces import fsl; print(fsl.Info.version())"
python -c "from nipype.interfaces import afni; print(afni.Info.version())"
python -c "from nipype.interfaces import ants; print('ANTs available')"
```

## Basic Workflow Construction

**Example 1: Minimal Motion Correction Workflow**

```python
from nipype import Workflow, Node
from nipype.interfaces import fsl
import os

# Create workflow
wf = Workflow(name='motion_correction_workflow', base_dir='/tmp/workflows')

# Define input node
from nipype import Node, IdentityInterface
inputnode = Node(IdentityInterface(fields=['func']), name='inputnode')
inputnode.inputs.func = '/path/to/sub-001_task-rest_bold.nii.gz'

# Add motion correction node (FSL MCFLIRT)
mcflirt = Node(fsl.MCFLIRT(
    save_plots=True,
    stats_imgs=True
), name='mcflirt')

# Connect input to motion correction
wf.connect([(inputnode, mcflirt, [('func', 'in_file')])])

# Run workflow
wf.run()

# Outputs in: /tmp/workflows/motion_correction_workflow/mcflirt/
# - *_mcf.nii.gz (motion-corrected functional)
# - *_mcf.par (motion parameters)
```

**Example 2: Simple Preprocessing Pipeline**

```python
from nipype import Workflow, Node
from nipype.interfaces import fsl, afni
from nipype.interfaces.utility import IdentityInterface, Function

# Create workflow
preproc = Workflow(name='simple_preprocessing', base_dir='/scratch/workflows')

# Input node
inputnode = Node(IdentityInterface(fields=['func', 'anat']), name='inputnode')

# 1. Slice timing correction (AFNI)
slicetime = Node(afni.TShift(
    tr=2.0,
    tpattern='altplus',
    outputtype='NIFTI_GZ'
), name='slicetime')

# 2. Motion correction (FSL)
motion_correct = Node(fsl.MCFLIRT(
    mean_vol=True,
    save_plots=True
), name='motion_correct')

# 3. Smoothing (FSL)
smooth = Node(fsl.SUSAN(
    fwhm=5.0,
    brightness_threshold=2000.0
), name='smooth')

# 4. High-pass filtering (FSL)
highpass = Node(fsl.TemporalFilter(
    highpass_sigma=50.0  # 100s cutoff at TR=2s
), name='highpass')

# Connect nodes
preproc.connect([
    (inputnode, slicetime, [('func', 'in_file')]),
    (slicetime, motion_correct, [('out_file', 'in_file')]),
    (motion_correct, smooth, [('out_file', 'in_file')]),
    (smooth, highpass, [('smoothed_file', 'in_file')])
])

# Set inputs
inputnode.inputs.func = '/data/sub-001_task-rest_bold.nii.gz'

# Run
preproc.run(plugin='MultiProc', plugin_args={'n_procs': 4})
```

**Example 3: Visualizing Workflow Graph**

```python
# Generate workflow visualization
preproc.write_graph(graph2use='colored', format='png', simple_form=True)

# Outputs:
# - graph.png (visual representation of workflow)
# - Shows node connections and data flow
# - Useful for debugging and documentation
```

## Preprocessing Modules

**Example 4: Motion Correction with Multiple Backends**

```python
from nipype import Workflow, Node, MapNode
from nipype.interfaces import fsl, afni, spm

# Compare motion correction algorithms
wf = Workflow(name='motion_comparison', base_dir='/tmp/workflows')

# FSL MCFLIRT
mcflirt = Node(fsl.MCFLIRT(ref_vol=0), name='mcflirt')

# AFNI 3dvolreg
volreg = Node(afni.Volreg(
    zpad=4,
    outputtype='NIFTI_GZ'
), name='volreg')

# SPM Realign
realign = Node(spm.Realign(
    register_to_mean=True
), name='spm_realign')

# Run all three in parallel for comparison
from nipype import IdentityInterface
inputnode = Node(IdentityInterface(fields=['func']), name='input')

wf.connect([
    (inputnode, mcflirt, [('func', 'in_file')]),
    (inputnode, volreg, [('func', 'in_file')]),
    (inputnode, realign, [('func', 'in_files')])
])

# Quality metrics to compare performance
from nipype.algorithms.rapidart import ArtifactDetect

qc_mcflirt = Node(ArtifactDetect(
    mask_type='file',
    parameter_source='FSL',
    norm_threshold=1.0,
    use_differences=[True, False]
), name='qc_mcflirt')

wf.connect([(mcflirt, qc_mcflirt, [('out_file', 'realigned_files')])])
```

**Example 5: Registration to Anatomical**

```python
from nipype.interfaces import fsl, ants

# Boundary-based registration (FSL)
bbr = Node(fsl.FLIRT(
    dof=6,
    cost='bbr',
    schedule='/usr/local/fsl/etc/flirtsch/bbr.sch'
), name='bbr_registration')

# Requires white matter segmentation
from nipype.interfaces import fsl
segment = Node(fsl.FAST(
    number_classes=3,
    output_biascorrected=True
), name='segment_anat')

# Alternative: ANTs registration
ants_reg = Node(ants.Registration(
    dimension=3,
    transforms=['Rigid'],
    metric=['MI'],
    metric_weight=[1.0],
    radius_or_number_of_bins=[32],
    convergence_threshold=[1e-6],
    number_of_iterations=[[1000, 500, 250]],
    smoothing_sigmas=[[2, 1, 0]],
    shrink_factors=[[4, 2, 1]],
    output_warped_image=True
), name='ants_registration')
```

**Example 6: Spatial Normalization Options**

```python
# FSL FNIRT (non-linear registration to MNI)
fnirt = Node(fsl.FNIRT(
    fieldcoeff_file=True,
    config_file='T1_2_MNI152_2mm'
), name='fnirt_to_mni')

# ANTs SyN (symmetric normalization)
syn = Node(ants.Registration(
    dimension=3,
    transforms=['Rigid', 'Affine', 'SyN'],
    metric=['MI', 'MI', 'CC'],
    metric_weight=[1.0, 1.0, 1.0],
    radius_or_number_of_bins=[32, 32, 4],
    number_of_iterations=[[1000, 500, 250, 100],
                          [1000, 500, 250, 100],
                          [100, 70, 50, 20]],
    convergence_threshold=[1e-6, 1e-6, 1e-6],
    smoothing_sigmas=[[3, 2, 1, 0],
                     [3, 2, 1, 0],
                     [3, 2, 1, 0]],
    shrink_factors=[[8, 4, 2, 1],
                   [8, 4, 2, 1],
                   [8, 4, 2, 1]],
    output_warped_image=True
), name='ants_syn')
```

## Integration with fMRIPrep

**Example 7: Post-fMRIPrep Custom Processing**

```python
from nipype import Workflow, Node, IdentityInterface
from nipype.interfaces import fsl
import glob

# Start with fMRIPrep outputs
fmriprep_dir = '/path/to/fmriprep/derivatives'
subject = 'sub-001'

# Create workflow for additional processing
post_fmriprep = Workflow(name='post_fmriprep_custom', base_dir='/scratch')

# Input: fMRIPrep preprocessed data
inputnode = Node(IdentityInterface(fields=['preproc_bold', 'confounds']), name='input')

# Find fMRIPrep outputs
preproc_file = f'{fmriprep_dir}/{subject}/func/{subject}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confounds_file = f'{fmriprep_dir}/{subject}/func/{subject}_task-rest_desc-confounds_timeseries.tsv'

inputnode.inputs.preproc_bold = preproc_file
inputnode.inputs.confounds = confounds_file

# Custom denoising beyond fMRIPrep
# Example: Additional spatial smoothing
smooth = Node(fsl.SUSAN(fwhm=6.0), name='additional_smoothing')

post_fmriprep.connect([(inputnode, smooth, [('preproc_bold', 'in_file')])])

# Custom confound regression (see next example)
```

**Example 8: Custom Confound Regression**

```python
from nipype import Node, Function
import pandas as pd
import numpy as np
from nilearn.image import clean_img

def custom_confound_regression(in_file, confounds_file, confound_vars):
    """Custom confound regression with specific variables."""
    import pandas as pd
    from nilearn.image import clean_img
    import os

    # Load confounds
    confounds = pd.read_csv(confounds_file, sep='\t')

    # Select specific confounds
    confounds_subset = confounds[confound_vars].fillna(0)

    # Apply regression
    cleaned = clean_img(
        in_file,
        confounds=confounds_subset.values,
        detrend=False,  # fMRIPrep already detrended
        standardize=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0
    )

    # Save
    out_file = os.path.abspath('denoised_bold.nii.gz')
    cleaned.to_filename(out_file)
    return out_file

# Create node with custom function
denoise = Node(Function(
    input_names=['in_file', 'confounds_file', 'confound_vars'],
    output_names=['out_file'],
    function=custom_confound_regression
), name='custom_denoise')

# Specify confound model
denoise.inputs.confound_vars = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'csf', 'white_matter',
    'global_signal'  # Optional: GSR
]

# Connect to workflow
post_fmriprep.connect([
    (inputnode, denoise, [('preproc_bold', 'in_file'),
                          ('confounds', 'confounds_file')])
])
```

## Confound Regression Strategies

**Example 9: Comparing Denoising Strategies**

```python
from nipype import Workflow, Node, Function, IdentityInterface
import itertools

# Define different confound models
confound_models = {
    'minimal': ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
    'standard': ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                 'csf', 'white_matter'],
    'with_gsr': ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                 'csf', 'white_matter', 'global_signal'],
    'acompcor': ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                 'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02',
                 'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05']
}

# Create separate denoise node for each strategy
wf = Workflow(name='compare_denoising', base_dir='/tmp')

for model_name, confounds in confound_models.items():
    denoise_node = Node(Function(
        input_names=['in_file', 'confounds_file', 'confound_vars'],
        output_names=['out_file'],
        function=custom_confound_regression
    ), name=f'denoise_{model_name}')

    denoise_node.inputs.confound_vars = confounds

    # Connect to input
    wf.connect([(inputnode, denoise_node, [
        ('preproc_bold', 'in_file'),
        ('confounds', 'confounds_file')
    ])])

# Run and compare QC metrics across strategies
```

**Example 10: Scrubbing High-Motion Volumes**

```python
def scrub_volumes(in_file, confounds_file, fd_threshold=0.5, dvars_threshold=1.5):
    """Remove high-motion volumes via interpolation or censoring."""
    import pandas as pd
    import numpy as np
    import nibabel as nib
    import os

    # Load confounds
    confounds = pd.read_csv(confounds_file, sep='\t')

    # Identify high-motion volumes
    fd = confounds['framewise_displacement'].fillna(0).values
    dvars = confounds['std_dvars'].fillna(0).values

    high_motion = (fd > fd_threshold) | (dvars > dvars_threshold)

    # Also remove 1 pre and 2 post high-motion volumes
    scrub_mask = np.zeros_like(high_motion, dtype=bool)
    for idx in np.where(high_motion)[0]:
        scrub_mask[max(0, idx-1):min(len(high_motion), idx+3)] = True

    # Load functional data
    img = nib.load(in_file)
    data = img.get_fdata()

    # Option 1: Remove volumes (censoring)
    clean_data = data[:, :, :, ~scrub_mask]

    # Option 2: Interpolate (preserves volume count)
    # from scipy.interpolate import interp1d
    # ... interpolation code ...

    # Save
    out_img = nib.Nifti1Image(clean_data, img.affine, img.header)
    out_file = os.path.abspath('scrubbed_bold.nii.gz')
    out_img.to_filename(out_file)

    return out_file, scrub_mask.sum()

scrubbing = Node(Function(
    input_names=['in_file', 'confounds_file', 'fd_threshold'],
    output_names=['out_file', 'n_scrubbed'],
    function=scrub_volumes
), name='scrubbing')

scrubbing.inputs.fd_threshold = 0.5  # mm
```

## Quality Control Modules

**Example 11: Custom QC Metrics**

```python
from nipype import Node, Function
import nibabel as nib
import numpy as np

def compute_tsnr(in_file, mask_file=None):
    """Compute temporal signal-to-noise ratio."""
    import nibabel as nib
    import numpy as np
    import os

    img = nib.load(in_file)
    data = img.get_fdata()

    if mask_file:
        mask = nib.load(mask_file).get_fdata().astype(bool)
    else:
        mask = np.ones(data.shape[:3], dtype=bool)

    # tSNR = mean / std over time
    mean_img = data[mask].mean(axis=0)
    std_img = data[mask].std(axis=0)
    tsnr = mean_img / (std_img + 1e-10)

    # Summary statistics
    tsnr_mean = np.mean(tsnr)
    tsnr_median = np.median(tsnr)

    # Save tSNR image
    tsnr_img = np.zeros(data.shape[:3])
    tsnr_img[mask] = tsnr
    tsnr_nii = nib.Nifti1Image(tsnr_img, img.affine)

    out_file = os.path.abspath('tsnr.nii.gz')
    tsnr_nii.to_filename(out_file)

    return out_file, float(tsnr_mean), float(tsnr_median)

tsnr_node = Node(Function(
    input_names=['in_file', 'mask_file'],
    output_names=['tsnr_file', 'tsnr_mean', 'tsnr_median'],
    function=compute_tsnr
), name='compute_tsnr')
```

**Example 12: Carpet Plot Visualization**

```python
def create_carpet_plot(in_file, confounds_file, mask_file, out_file='carpet_plot.png'):
    """Generate carpet plot for visual QC."""
    import nibabel as nib
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from nilearn import plotting

    # Load data
    img = nib.load(in_file)
    confounds = pd.read_csv(confounds_file, sep='\t')

    # Create carpet plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Carpet (voxels × time)
    plotting.plot_carpet(img, mask_img=mask_file, axes=axes[0])

    # Plot 2: Confounds (FD, DVARS, Global signal)
    time = np.arange(len(confounds))
    axes[1].plot(time, confounds['framewise_displacement'], label='FD (mm)')
    axes[1].axhline(0.5, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Volume')
    axes[1].set_ylabel('FD (mm)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)

    return out_file

carpet = Node(Function(
    input_names=['in_file', 'confounds_file', 'mask_file'],
    output_names=['carpet_plot'],
    function=create_carpet_plot
), name='carpet_plot')
```

## Advanced Customization

**Example 13: Multi-Echo fMRI Workflow**

```python
from nipype import Workflow, Node, MapNode
from nipype.interfaces.afni import TEDana

# Multi-echo preprocessing workflow
me_workflow = Workflow(name='multiecho_processing', base_dir='/scratch')

# Input: multiple echo times
inputnode = Node(IdentityInterface(fields=['echo1', 'echo2', 'echo3']), name='input')

# Motion correction for each echo
mcflirt_echoes = MapNode(fsl.MCFLIRT(
    ref_vol=0,
    save_mats=True
), iterfield=['in_file'], name='mcflirt_echoes')

# Apply motion parameters from first echo to all
# (ensures consistent motion correction)

# TEDANA: TE-dependent analysis for denoising
tedana = Node(TEDana(
    tedpca='kundu',
    tedort=True,
    out_dir='tedana_output'
), name='tedana')

# Connect echoes
me_workflow.connect([
    (inputnode, mcflirt_echoes, [
        ('echo1', 'in_file'),
        ('echo2', 'in_file'),
        ('echo3', 'in_file')
    ]),
    (mcflirt_echoes, tedana, [('out_file', 'in_files')])
])
```

**Example 14: Custom Temporal Filtering**

```python
from scipy.signal import butter, filtfilt
import nibabel as nib
import numpy as np

def custom_bandpass_filter(in_file, tr, lowcut=0.01, highcut=0.1):
    """Apply custom bandpass filter."""
    img = nib.load(in_file)
    data = img.get_fdata()

    # Design Butterworth bandpass filter
    nyquist = 0.5 / tr
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(4, [low, high], btype='band')

    # Apply filter to each voxel timeseries
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if data[i, j, k, :].std() > 0:
                    filtered_data[i, j, k, :] = filtfilt(b, a, data[i, j, k, :])

    # Save
    out_img = nib.Nifti1Image(filtered_data, img.affine, img.header)
    out_file = 'filtered_bold.nii.gz'
    out_img.to_filename(out_file)

    return out_file

filter_node = Node(Function(
    input_names=['in_file', 'tr', 'lowcut', 'highcut'],
    output_names=['out_file'],
    function=custom_bandpass_filter
), name='bandpass_filter')

filter_node.inputs.tr = 2.0
filter_node.inputs.lowcut = 0.01  # Hz
filter_node.inputs.highcut = 0.1  # Hz
```

## HPC Execution

**Example 15: SLURM Plugin for Parallel Processing**

```python
from nipype import Workflow, Node, IdentityInterface
from nipype.interfaces import fsl
import glob

# Create workflow
wf = Workflow(name='parallel_preprocessing', base_dir='/scratch')

# Process multiple subjects in parallel
subjects = [f'sub-{i:03d}' for i in range(1, 51)]

# Use iterables to parallelize across subjects
inputnode = Node(IdentityInterface(fields=['subject_id']), name='input')
inputnode.iterables = ('subject_id', subjects)

# Preprocessing nodes (same as before)
mcflirt = Node(fsl.MCFLIRT(), name='motion_correct')
smooth = Node(fsl.SUSAN(fwhm=5.0), name='smooth')

wf.connect([
    (inputnode, mcflirt, [('subject_id', 'in_file')]),
    (mcflirt, smooth, [('out_file', 'in_file')])
])

# Execute on SLURM cluster
wf.run(plugin='SLURM', plugin_args={
    'sbatch_args': '--time=2:00:00 --mem=8G --cpus-per-task=2',
    'max_jobs': 20  # Limit concurrent jobs
})
```

**Example 16: SGE and PBS Support**

```python
# For SGE clusters
wf.run(plugin='SGE', plugin_args={
    'qsub_args': '-l h_vmem=8G -pe smp 2',
    'max_jobs': 100
})

# For PBS/Torque clusters
wf.run(plugin='PBS', plugin_args={
    'qsub_args': '-l nodes=1:ppn=2,mem=8gb,walltime=02:00:00'
})

# For local multiprocessing
wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})
```

**Example 17: Monitoring Workflow Execution**

```bash
# Check workflow status
ls /scratch/parallel_preprocessing/*/

# View individual node outputs
cat /scratch/parallel_preprocessing/motion_correct/_subject_id_sub-001/result*.pklz

# Check for errors
grep -r "ERROR" /scratch/parallel_preprocessing/

# Workflow can be resumed if interrupted
# Nipype caches completed nodes
```

## Troubleshooting

**Common Workflow Issues:**

**Node Connection Errors:**
```python
# Error: Output 'out_file' not defined in interface

# Fix: Check interface documentation
from nipype.interfaces import fsl
help(fsl.MCFLIRT)  # Shows all inputs/outputs

# Ensure output names match
wf.connect([(mcflirt, smooth, [
    ('out_file', 'in_file')  # 'out_file' must exist in MCFLIRT outputs
])])
```

**Memory Issues:**
```python
# Limit memory for specific nodes
mcflirt.interface.mem_gb = 4  # Limit to 4 GB

# Or set globally
from nipype import config
config.set('execution', 'hash_method', 'timestamp')
config.set('execution', 'use_relative_paths', True)
```

**Backend Software Errors:**
```bash
# Verify FSL is in PATH
which fsl

# Check environment variables
echo $FSLDIR
echo $ANTSPATH

# Test interfaces directly
python -c "from nipype.interfaces import fsl; fsl.MCFLIRT().version"
```

## Best Practices

**Workflow Design:**
- Start simple, add complexity incrementally
- Test on single subject before scaling to cohort
- Use descriptive node names for clarity
- Visualize workflow graphs (`workflow.write_graph()`)
- Cache intermediate results in working directory

**Reproducibility:**
- Version control workflow scripts (Git)
- Document all parameter choices
- Save workflow graphs with each analysis
- Use containers (Docker/Singularity) for production
- Record software versions in provenance

**Performance Optimization:**
- Use working directory on fast disk (SSD, /scratch)
- Enable Nipype caching to avoid recomputation
- Parallelize independent processing streams
- Monitor resource usage and adjust node-level limits
- Clean up working directories after successful runs

**Testing and Validation:**
- Compare outputs to established pipelines (fMRIPrep, SPM)
- Visual QC at each major processing step
- Quantitative metrics (tSNR, motion, registration quality)
- Validate custom functions on toy datasets

## Integration with Analysis Tools

**Nilearn:**
```python
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMapsMasker

# Use fMRIflows preprocessed data with Nilearn
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

masker = NiftiMapsMasker(atlas.maps, standardize=True)
timeseries = masker.fit_transform('preprocessed_bold.nii.gz')

# Connectivity analysis
from nilearn.connectome import ConnectivityMeasure
conn_measure = ConnectivityMeasure(kind='correlation')
corr_matrix = conn_measure.fit_transform([timeseries])[0]
```

**Custom Python Analysis:**
```python
# Export preprocessed data for custom analysis
import nibabel as nib
import pandas as pd

# Load fMRIflows outputs
func = nib.load('denoised_bold.nii.gz')
confounds = pd.read_csv('confounds.tsv', sep='\t')

# Your custom analysis code
# ...
```

## Example: Complete Custom Workflow

**Example 18: Full Task fMRI Preprocessing**

```python
from nipype import Workflow, Node, IdentityInterface
from nipype.interfaces import fsl, afni
from nipype.interfaces.utility import Function

# Create comprehensive task fMRI workflow
task_wf = Workflow(name='task_fmri_preprocessing', base_dir='/scratch')

# Input
inputnode = Node(IdentityInterface(fields=[
    'func', 'anat', 'events_file'
]), name='input')

# 1. Slice-timing correction
slicetime = Node(afni.TShift(tr=2.0, tpattern='altplus'), name='slicetime')

# 2. Motion correction
mcflirt = Node(fsl.MCFLIRT(save_plots=True), name='motion')

# 3. Register to anatomical
bbr = Node(fsl.FLIRT(dof=6, cost='bbr'), name='register')

# 4. Normalize to MNI
fnirt = Node(fsl.FNIRT(config_file='T1_2_MNI152_2mm'), name='normalize')

# 5. Smooth
smooth = Node(fsl.SUSAN(fwhm=5.0), name='smooth')

# 6. High-pass filter
highpass = Node(fsl.TemporalFilter(highpass_sigma=50), name='highpass')

# Connect all nodes
task_wf.connect([
    (inputnode, slicetime, [('func', 'in_file')]),
    (slicetime, mcflirt, [('out_file', 'in_file')]),
    (mcflirt, bbr, [('out_file', 'in_file')]),
    (inputnode, bbr, [('anat', 'reference')]),
    (bbr, fnirt, [('out_file', 'in_file')]),
    (fnirt, smooth, [('warped_file', 'in_file')]),
    (smooth, highpass, [('smoothed_file', 'in_file')])
])

# Run workflow
task_wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
```

## References

**Nipype:**
- Gorgolewski et al. (2011). Nipype: A flexible, lightweight and extensible neuroimaging data processing framework in Python. *Frontiers in Neuroinformatics*, 5, 13.
- Gorgolewski et al. (2018). Nipype: A flexible, lightweight and extensible neuroimaging data processing framework in Python (2018 update). *F1000Research*, 7, 560.

**Preprocessing Methods:**
- Lindquist (2008). The statistical analysis of fMRI data. *Statistical Science*, 23(4), 439-464.
- Hallquist et al. (2013). The nuisance of nuisance regression: Spectral misspecification in a common approach to resting-state fMRI preprocessing. *NeuroImage*, 82, 208-225.

**Confound Regression:**
- Satterthwaite et al. (2013). An improved framework for confound regression and filtering for control of motion artifact in the preprocessing of resting-state functional connectivity data. *NeuroImage*, 64, 240-256.
- Ciric et al. (2017). Benchmarking of participant-level confound regression strategies for the control of motion artifact in studies of functional connectivity. *NeuroImage*, 154, 174-187.

**Software Backends:**
- FSL: Jenkinson et al. (2012). FSL. *NeuroImage*, 62(2), 782-790.
- AFNI: Cox (1996). AFNI: Software for analysis and visualization of functional magnetic resonance neuroimages. *Computers and Biomedical Research*, 29(3), 162-173.
- ANTs: Avants et al. (2011). A reproducible evaluation of ANTs similarity metric performance in brain image registration. *NeuroImage*, 54(3), 2033-2044.

**Online Resources:**
- Nipype Documentation: https://nipype.readthedocs.io/
- Nipype Tutorial: https://miykael.github.io/nipype_tutorial/
- fMRIflows GitHub: https://github.com/fmriflows/fmriflows
- Neuroimaging Workflows Gallery: https://github.com/niflows
