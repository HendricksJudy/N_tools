# NiPyPe (Neuroimaging in Python Pipelines and Interfaces)

## Overview

NiPyPe is a Python framework for building reproducible, reusable neuroimaging analysis workflows. It provides uniform interfaces to multiple neuroimaging software packages (FSL, SPM, AFNI, FreeSurfer, ANTs, etc.) and enables construction of complex pipelines with automatic dependency management, parallel execution, and provenance tracking.

**Website:** https://nipype.readthedocs.io/
**Platform:** Cross-platform (Linux/macOS/Windows)
**Language:** Python
**License:** Apache 2.0

## Key Features

- Unified interface to neuroimaging tools (FSL, SPM, AFNI, ANTs, FreeSurfer)
- Workflow graph construction with automatic dependency resolution
- Parallel execution (multiprocessing, SGE, SLURM, PBS)
- Provenance tracking and reproducibility
- Data caching and reuse
- Plugin architecture for extensibility
- Integration with BIDS datasets
- Visualization of workflows

## Installation

```bash
# Basic installation
pip install nipype

# With all dependencies
pip install nipype[all]

# Specific interfaces
pip install nipype[fsl]
pip install nipype[spm]
pip install nipype[afni]

# Development version
pip install git+https://github.com/nipy/nipype.git
```

### Verify Installation

```python
import nipype
print(nipype.__version__)

# Check available interfaces
from nipype.interfaces import fsl, spm, afni, ants, freesurfer
```

## Basic Concepts

### Interfaces

Interfaces wrap command-line tools or Python functions:

```python
from nipype.interfaces import fsl

# BET interface
bet = fsl.BET()
bet.inputs.in_file = 'struct.nii.gz'
bet.inputs.out_file = 'struct_brain.nii.gz'
bet.inputs.mask = True
bet.inputs.frac = 0.5

# Run interface
result = bet.run()
print(result.outputs)
```

### Nodes

Nodes wrap interfaces for use in workflows:

```python
from nipype import Node

# Create node
bet_node = Node(fsl.BET(), name='brain_extraction')
bet_node.inputs.frac = 0.5
bet_node.inputs.mask = True
```

### Workflows

Workflows connect nodes to create pipelines:

```python
from nipype import Workflow

# Create workflow
preproc = Workflow(name='preprocessing', base_dir='/output')

# Add nodes
preproc.add_nodes([bet_node, realign_node, smooth_node])

# Connect nodes
preproc.connect([
    (bet_node, realign_node, [('out_file', 'in_file')]),
    (realign_node, smooth_node, [('out_file', 'in_file')])
])
```

## Complete fMRI Preprocessing Example

```python
from nipype import Workflow, Node, MapNode
from nipype.interfaces import fsl, utility as niu
from nipype.interfaces.io import SelectFiles, DataSink

# Set up workflow
preproc_wf = Workflow(name='fmri_preproc', base_dir='/output')

# 1. Brain extraction
bet = Node(fsl.BET(frac=0.3, mask=True), name='brain_extraction')

# 2. Motion correction
mcflirt = Node(fsl.MCFLIRT(mean_vol=True, save_plots=True),
               name='motion_correction')

# 3. Smoothing
smooth = Node(fsl.Smooth(fwhm=6.0), name='smoothing')

# 4. High-pass filtering
highpass = Node(fsl.TemporalFilter(highpass_sigma=50.0),
                name='temporal_filter')

# 5. Intensity normalization
meanfunc = Node(fsl.ImageMaths(op_string='-Tmean',
                                suffix='_mean'),
                name='mean_image')

intnorm = Node(fsl.ImageMaths(suffix='_intnorm'),
               name='intensity_normalization')

# Connect nodes
preproc_wf.connect([
    (bet, mcflirt, [('mask_file', 'ref_file')]),
    (mcflirt, smooth, [('out_file', 'in_file')]),
    (smooth, highpass, [('smoothed_file', 'in_file')]),
    (highpass, meanfunc, [('out_file', 'in_file')]),
    (highpass, intnorm, [('out_file', 'in_file')]),
    (meanfunc, intnorm, [('out_file', 'op_string')])
])

# Run workflow
preproc_wf.run()
```

## Multi-Subject Processing

```python
from nipype import Workflow, Node, MapNode
from nipype.interfaces import io, utility as niu
from nipype.interfaces import fsl

# Subject list
subject_list = ['sub-01', 'sub-02', 'sub-03']

# Infosource - iterate over subjects
infosource = Node(niu.IdentityInterface(fields=['subject_id']),
                  name='infosource')
infosource.iterables = [('subject_id', subject_list)]

# SelectFiles - locate files
templates = {
    'func': '{subject_id}/func/{subject_id}_task-rest_bold.nii.gz',
    'anat': '{subject_id}/anat/{subject_id}_T1w.nii.gz'
}

selectfiles = Node(SelectFiles(templates, base_directory='/data/BIDS'),
                   name='selectfiles')

# Processing nodes
bet = Node(fsl.BET(frac=0.3, functional=True), name='bet')
mcflirt = Node(fsl.MCFLIRT(), name='mcflirt')
smooth = Node(fsl.Smooth(fwhm=6), name='smooth')

# DataSink - save outputs
datasink = Node(DataSink(base_directory='/output/derivatives'),
                name='datasink')

# Create workflow
multi_subj_wf = Workflow(name='multi_subject', base_dir='/output')

# Connect
multi_subj_wf.connect([
    (infosource, selectfiles, [('subject_id', 'subject_id')]),
    (selectfiles, bet, [('func', 'in_file')]),
    (bet, mcflirt, [('out_file', 'in_file')]),
    (mcflirt, smooth, [('out_file', 'in_file')]),
    (smooth, datasink, [('smoothed_file', 'preprocessed.@func')]),
    (infosource, datasink, [('subject_id', 'container')])
])

# Run in parallel
multi_subj_wf.run('MultiProc', plugin_args={'n_procs': 4})
```

## SPM Integration

```python
from nipype.interfaces import spm
from nipype import Node, Workflow

# Configure SPM
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/path/to/spm12')
MatlabCommand.set_default_matlab_cmd('matlab -nodesktop -nosplash')

# Realignment
realign = Node(spm.Realign(), name='realignment')
realign.inputs.register_to_mean = True

# Coregistration
coreg = Node(spm.Coregister(), name='coregistration')

# Segmentation
segment = Node(spm.Segment(), name='segmentation')
segment.inputs.tissue_prob_maps = [
    '/path/to/spm12/tpm/TPM.nii'
]

# Normalize
normalize = Node(spm.Normalize12(), name='normalization')

# Smoothing
smooth = Node(spm.Smooth(), name='smoothing')
smooth.inputs.fwhm = [8, 8, 8]

# Create SPM workflow
spm_wf = Workflow(name='spm_preproc')
spm_wf.connect([
    (realign, coreg, [('mean_image', 'source')]),
    (coreg, segment, [('coregistered_source', 'data')]),
    (segment, normalize, [('transformation_mat', 'deformation_file')]),
    (realign, normalize, [('realigned_files', 'apply_to_files')]),
    (normalize, smooth, [('normalized_files', 'in_files')])
])
```

## ANTs Integration

```python
from nipype.interfaces import ants

# Brain extraction
brain_extract = Node(ants.BrainExtraction(), name='brain_extraction')
brain_extract.inputs.dimension = 3
brain_extract.inputs.brain_template = 'template.nii.gz'
brain_extract.inputs.brain_probability_mask = 'template_mask.nii.gz'

# Registration
registration = Node(ants.Registration(), name='registration')
registration.inputs.fixed_image = 'template.nii.gz'
registration.inputs.transforms = ['Rigid', 'Affine', 'SyN']
registration.inputs.metric = ['MI', 'MI', 'CC']
registration.inputs.metric_weight = [1.0, 1.0, 1.0]
registration.inputs.radius_or_number_of_bins = [32, 32, 4]
registration.inputs.convergence_threshold = [1e-6, 1e-6, 1e-6]
registration.inputs.convergence_window_size = [10, 10, 10]
registration.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 3
registration.inputs.shrink_factors = [[8, 4, 2, 1]] * 3

# Apply transforms
apply_transforms = Node(ants.ApplyTransforms(), name='apply_transforms')
apply_transforms.inputs.dimension = 3
apply_transforms.inputs.interpolation = 'Linear'
```

## Parallel Execution

```python
# Local multiprocessing
workflow.run(plugin='MultiProc', plugin_args={'n_procs': 8})

# SGE cluster
workflow.run(plugin='SGE', plugin_args={'qsub_args': '-q long.q -pe smp 4'})

# SLURM
workflow.run(plugin='SLURM', plugin_args={'sbatch_args': '--time=24:00:00'})

# PBS
workflow.run(plugin='PBS', plugin_args={'qsub_args': '-l nodes=1:ppn=4'})

# Linear (no parallelization, for debugging)
workflow.run(plugin='Linear')
```

## Visualization

```python
# Create workflow graph
workflow.write_graph(graph2use='colored', format='png', simple_form=True)

# Detailed graph
workflow.write_graph(graph2use='flat', format='svg')

# View in browser
from nipype.pipeline.plugins import DebugPlugin
workflow.run(plugin=DebugPlugin())
```

## MapNode for Parallel Processing

```python
from nipype import MapNode

# Process multiple files in parallel
smooth_multi = MapNode(
    fsl.Smooth(fwhm=6),
    name='smooth_multiple',
    iterfield=['in_file']
)

# Will process each file independently
smooth_multi.inputs.in_file = [
    'func_run1.nii.gz',
    'func_run2.nii.gz',
    'func_run3.nii.gz'
]
```

## Custom Functions and Interfaces

```python
from nipype.interfaces.utility import Function

def custom_threshold(in_file, threshold=0.5):
    """Custom thresholding function"""
    import nibabel as nib
    import numpy as np
    import os

    img = nib.load(in_file)
    data = img.get_fdata()
    data[data < threshold] = 0

    out_file = os.path.abspath('thresholded.nii.gz')
    nib.save(nib.Nifti1Image(data, img.affine), out_file)

    return out_file

# Create Function node
threshold_node = Node(
    Function(
        input_names=['in_file', 'threshold'],
        output_names=['out_file'],
        function=custom_threshold
    ),
    name='custom_threshold'
)
threshold_node.inputs.threshold = 0.5
```

## Iterables and Parameterization

```python
# Iterate over parameters
smooth_node = Node(fsl.Smooth(), name='smoothing')
smooth_node.iterables = [
    ('fwhm', [4, 6, 8]),  # Try different smoothing kernels
    ('output_type', ['NIFTI', 'NIFTI_GZ'])
]

# Synchronized iteration
preproc = Node(fsl.SUSAN(), name='susan')
preproc.iterables = [
    ('brightness_threshold', [1000, 2000]),
    ('fwhm', [4, 6])
]
preproc.synchronize = True  # Pairs: (1000,4), (2000,6)
```

## Caching and Reuse

```python
from nipype.caching import Memory

# Enable caching
mem = Memory('/output/cache')

# Cached execution
bet_cached = mem.cache(fsl.BET)
result = bet_cached(in_file='struct.nii.gz', frac=0.3)

# Rerunning with same inputs uses cache
result2 = bet_cached(in_file='struct.nii.gz', frac=0.3)  # Uses cached result
```

## Integration with Claude Code

When helping users with NiPyPe:

1. **Check Setup:**
   ```python
   import nipype
   from nipype.interfaces import fsl
   fsl.Info.version()  # Check FSL availability
   ```

2. **Common Issues:**
   - Tool not in PATH
   - MATLAB/SPM configuration
   - Memory errors with large datasets
   - Hash collisions in caching

3. **Debugging:**
   ```python
   # Enable logging
   from nipype import config, logging
   config.enable_debug_mode()
   logging.update_logging(config)

   # Or use config file
   config.set('execution', 'stop_on_first_crash', True)
   config.set('execution', 'remove_unnecessary_outputs', False)
   ```

4. **Best Practices:**
   - Use base_dir for workflow outputs
   - Enable crash file generation
   - Start with small test dataset
   - Visualize workflow before running

## Troubleshooting

**Problem:** "Command not found"
**Solution:** Ensure tool is in PATH or set interface paths

**Problem:** Hash collision errors
**Solution:** Clear workflow directory or update input specifications

**Problem:** Out of memory
**Solution:** Reduce n_procs or use chunking with MapNode

**Problem:** SPM MATLAB errors
**Solution:** Configure MATLAB paths correctly, check SPM version

## Resources

- Documentation: https://nipype.readthedocs.io/
- Examples: https://nipype.readthedocs.io/en/latest/examples.html
- GitHub: https://github.com/nipy/nipype
- Forum: https://neurostars.org/ (tag: nipype)

## Related Tools

- **Pydra:** Next-generation workflow engine
- **BIDS:** Brain Imaging Data Structure
- **fMRIPrep:** NiPyPe-based preprocessing pipeline
- **Niworkflows:** Shared workflow components
