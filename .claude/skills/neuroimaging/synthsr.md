# SynthSR - Deep Learning MRI Super-Resolution

## Overview

SynthSR is a deep learning tool for enhancing the resolution of clinical MRI scans, developed by the CERVL lab (Computational Radiology Laboratory at MGH/Harvard Medical School and MIT). Using convolutional neural networks trained with domain randomization, SynthSR can upscale low-resolution or thick-slice clinical scans to isotropic 1mm resolution, regardless of the input contrast (T1, T2, FLAIR, CT, etc.). Unlike traditional super-resolution methods that require matched training data, SynthSR's domain randomization approach makes it robust and generalizable across contrasts, field strengths, and acquisition protocols.

**Website:** https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR
**Platform:** Python/TensorFlow (Linux/macOS/Windows)
**License:** FreeSurfer License
**Key Application:** Clinical scan enhancement, legacy data improvement, preprocessing for analysis pipelines

### Why Super-Resolution?

Clinical MRI scans often have:
- **Anisotropic resolution** - High in-plane resolution (e.g., 0.5×0.5mm) but thick slices (5-7mm)
- **Low resolution** - Faster scans with larger voxels (e.g., 1.5×1.5×5mm)
- **Motion artifacts** - Leading to degraded image quality
- **Limited contrast** - Single modality available

SynthSR addresses these issues by producing isotropic 1mm resolution images suitable for:
- Quantitative analysis (volumetry, morphometry)
- Improved visualization
- Input to other processing tools (FreeSurfer, SynthSeg)
- Multi-site harmonization

## Key Features

- **Contrast-agnostic** - Works on any MRI contrast without retraining
- **Isotropic 1mm output** - Standard resolution for analysis
- **Domain randomization training** - Robust to acquisition variability
- **No paired training data required** - Generalizes across protocols
- **Fast processing** - Minutes per volume (CPU) or seconds (GPU)
- **FreeSurfer integration** - Seamless workflow with recon-all
- **Multiple field strengths** - Works with 1.5T, 3T, 7T data
- **Clinical scan support** - Handles thick slices, low resolution
- **Pathology robust** - Works with lesions, atrophy, artifacts
- **Multi-contrast** - Process T1, T2, FLAIR, PD, CT, etc.
- **Easy installation** - pip install, no compilation needed
- **Batch processing** - Process multiple subjects efficiently
- **Quality control tools** - Built-in QC metrics
- **Open source** - Available on GitHub
- **Well-validated** - Published and extensively tested

## Installation

### Prerequisites

SynthSR requires Python 3.6+ and TensorFlow:

```bash
# Check Python version
python3 --version  # Should be 3.6 or higher
```

### Install via pip (Recommended)

```bash
# Install SynthSR
pip install SynthSR

# Verify installation
python3 -c "import SynthSR; print('SynthSR installed successfully')"
```

### Install from Source

```bash
# Clone repository
cd ~/software
git clone https://github.com/freesurfer/freesurfer.git
cd freesurfer/SynthSR

# Install
pip install -e .
```

### FreeSurfer Integration

```bash
# SynthSR is included in FreeSurfer 7.3+
# After FreeSurfer installation:
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# SynthSR command becomes available
mri_synthsr --help
```

### GPU Support (Optional but Recommended)

```bash
# Install TensorFlow with GPU support for faster processing
pip install tensorflow-gpu

# Verify GPU availability
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Basic Usage

### Simple Super-Resolution

Enhance a single scan:

```bash
# Basic usage
mri_synthsr \
  --i input_lowres.nii.gz \
  --o output_1mm.nii.gz

# Or using Python
python3 -c "from SynthSR.predict import predict; \
            predict(path_images='input_lowres.nii.gz', \
                   path_predictions='output_1mm.nii.gz')"
```

### Typical Execution

```bash
# Example: Enhance thick-slice T1w image
mri_synthsr \
  --i thick_slice_T1.nii.gz \
  --o enhanced_T1_1mm.nii.gz

# Processing time:
# CPU: ~3-5 minutes
# GPU: ~30-60 seconds

# Output: Isotropic 1mm resolution T1w image
```

### With Custom Resolution

```bash
# Specify output resolution (default: 1mm)
mri_synthsr \
  --i input.nii.gz \
  --o output_0.5mm.nii.gz \
  --target_res 0.5  # Output at 0.5mm isotropic

# Or coarser resolution
mri_synthsr \
  --i input.nii.gz \
  --o output_2mm.nii.gz \
  --target_res 2  # Output at 2mm isotropic
```

## Multi-Contrast Support

### T1-weighted Enhancement

```bash
# Typical clinical T1w: 1×1×5mm → 1×1×1mm
mri_synthsr \
  --i clinical_T1w.nii.gz \
  --o enhanced_T1w_1mm.nii.gz

# Works regardless of:
# - Field strength (1.5T, 3T, 7T)
# - Sequence (MPRAGE, SPGR, etc.)
# - Acquisition parameters
```

### T2-weighted Enhancement

```bash
# T2w clinical scan
mri_synthsr \
  --i clinical_T2w.nii.gz \
  --o enhanced_T2w_1mm.nii.gz

# Useful for:
# - Lesion visualization
# - Multi-contrast segmentation
# - Clinical assessment
```

### FLAIR Enhancement

```bash
# FLAIR for lesion detection
mri_synthsr \
  --i clinical_FLAIR.nii.gz \
  --o enhanced_FLAIR_1mm.nii.gz

# Improved resolution helps:
# - White matter lesion detection
# - Multiple sclerosis monitoring
# - Small vessel disease assessment
```

### CT Enhancement (Experimental)

```bash
# Even works on CT scans
mri_synthsr \
  --i head_CT.nii.gz \
  --o enhanced_CT_1mm.nii.gz

# Note: Primarily trained on MRI, CT support is limited
```

## Batch Processing

### Process Multiple Subjects

```bash
#!/bin/bash
# batch_synthsr.sh

# BIDS dataset structure
bids_dir="/path/to/bids/dataset"
output_dir="/path/to/output"

# Loop through subjects
for subj_dir in ${bids_dir}/sub-*/; do
    subj=$(basename ${subj_dir})
    echo "Processing ${subj}..."

    # Input T1w
    input_t1="${subj_dir}/anat/${subj}_T1w.nii.gz"

    # Check if file exists
    if [ -f "${input_t1}" ]; then
        # Output
        output_t1="${output_dir}/${subj}/${subj}_T1w_synthsr.nii.gz"
        mkdir -p "$(dirname ${output_t1})"

        # Run SynthSR
        mri_synthsr --i "${input_t1}" --o "${output_t1}"

        echo "${subj}: Success"
    else
        echo "${subj}: T1w not found, skipping"
    fi
done
```

### Parallel Processing

```bash
# GNU Parallel for faster batch processing
parallel -j 4 mri_synthsr --i {} --o {.}_synthsr.nii.gz ::: input_dir/*.nii.gz

# -j 4: Process 4 subjects simultaneously
# Adjust based on available CPU/GPU resources
```

### Python Batch Script

```python
import os
from pathlib import Path
from SynthSR.predict import predict

# Input directory
input_dir = Path('/path/to/input')
output_dir = Path('/path/to/output')
output_dir.mkdir(exist_ok=True)

# Find all NIfTI files
input_files = list(input_dir.glob('*.nii.gz'))

print(f"Found {len(input_files)} files to process")

# Process each file
for input_file in input_files:
    print(f"Processing {input_file.name}...")

    output_file = output_dir / f"{input_file.stem}_synthsr.nii.gz"

    try:
        predict(
            path_images=str(input_file),
            path_predictions=str(output_file),
            target_res=1.0,  # 1mm isotropic
            cpu=False  # Use GPU if available
        )
        print(f"  Success: {output_file.name}")
    except Exception as e:
        print(f"  Error: {e}")

print("Batch processing complete!")
```

## Integration with FreeSurfer

### Preprocessing for recon-all

SynthSR enhances low-quality inputs before FreeSurfer processing:

```bash
# Step 1: Enhance clinical scan with SynthSR
mri_synthsr \
  --i clinical_T1w_lowres.nii.gz \
  --o T1w_enhanced_1mm.nii.gz

# Step 2: Run FreeSurfer on enhanced image
recon-all \
  -i T1w_enhanced_1mm.nii.gz \
  -s subject01 \
  -all

# Benefits:
# - Improved segmentation accuracy
# - Better surface reconstruction
# - More reliable cortical thickness estimates
```

### Direct Integration

```bash
# FreeSurfer 7.3+ can call SynthSR internally
recon-all \
  -i clinical_T1w_lowres.nii.gz \
  -s subject01 \
  -synthsr \
  -all

# FreeSurfer will:
# 1. Run SynthSR first
# 2. Use enhanced image for processing
# 3. Save both original and enhanced
```

## Integration with SynthSeg

Combine super-resolution with robust segmentation:

```bash
# Pipeline: SynthSR → SynthSeg

# Step 1: Enhance resolution
mri_synthsr \
  --i lowres_scan.nii.gz \
  --o enhanced_1mm.nii.gz

# Step 2: Segment with SynthSeg
mri_synthseg \
  --i enhanced_1mm.nii.gz \
  --o segmentation.nii.gz \
  --vol volumes.csv \
  --qc qc_scores.csv

# Or direct (SynthSeg can handle low-res, but SynthSR helps)
mri_synthseg \
  --i lowres_scan.nii.gz \
  --o segmentation.nii.gz \
  --vol volumes.csv
```

## Python API

### Basic Usage

```python
from SynthSR.predict import predict

# Enhance single image
predict(
    path_images='input_lowres.nii.gz',
    path_predictions='output_1mm.nii.gz',
    target_res=1.0,  # Target resolution in mm
    cpu=False  # Use GPU if available
)
```

### Advanced Options

```python
import numpy as np
import nibabel as nib
from SynthSR.predict import predict

# Load and process image
input_img = nib.load('input.nii.gz')
input_data = input_img.get_fdata()

# Predict with custom parameters
output_data = predict(
    path_images='input.nii.gz',
    path_predictions='output.nii.gz',
    target_res=1.0,  # 1mm isotropic
    cpu=False,  # Use GPU
    # Additional options:
    # crop=192,  # Crop size (default: None for no cropping)
    # n_neutral_labels=18,  # Number of neutral labels
)

print(f"Input shape: {input_data.shape}")
print(f"Output resolution: 1mm isotropic")
```

### Batch Processing with Progress

```python
from SynthSR.predict import predict
from tqdm import tqdm
import glob

# Find all input files
input_files = glob.glob('/path/to/input/*.nii.gz')

# Process with progress bar
for input_file in tqdm(input_files, desc="Processing"):
    output_file = input_file.replace('input', 'output').replace('.nii.gz', '_synthsr.nii.gz')

    try:
        predict(
            path_images=input_file,
            path_predictions=output_file,
            target_res=1.0,
            cpu=False
        )
    except Exception as e:
        print(f"Failed on {input_file}: {e}")
```

## Quality Control

### Visual Comparison

```python
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load original and enhanced
orig = nib.load('original.nii.gz').get_fdata()
enhanced = nib.load('enhanced_synthsr.nii.gz').get_fdata()

# Select middle slice
slice_idx = orig.shape[2] // 2

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(orig[:, :, slice_idx], cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(enhanced[:, :, slice_idx], cmap='gray')
axes[1].set_title('SynthSR Enhanced')
axes[1].axis('off')

# Difference
diff = enhanced[:, :, slice_idx] - orig[:, :, slice_idx]
axes[2].imshow(diff, cmap='RdBu_r')
axes[2].set_title('Difference')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('synthsr_qc.png', dpi=150)
plt.show()
```

### Resolution Verification

```python
import nibabel as nib

# Check output resolution
img = nib.load('output_synthsr.nii.gz')
voxel_size = img.header.get_zooms()
print(f"Voxel size: {voxel_size}")

# Should be (1.0, 1.0, 1.0) for 1mm isotropic
assert all(abs(v - 1.0) < 0.01 for v in voxel_size[:3]), "Resolution not 1mm isotropic!"
```

### Automated QC Metrics

```python
import nibabel as nib
import numpy as np
from scipy import ndimage

def compute_qc_metrics(original_path, enhanced_path):
    """Compute quality metrics for SynthSR output."""

    # Load images
    orig_img = nib.load(original_path)
    enh_img = nib.load(enhanced_path)

    orig_data = orig_img.get_fdata()
    enh_data = enh_img.get_fdata()

    # Resample original to match enhanced for comparison
    from scipy.ndimage import zoom
    orig_voxsize = orig_img.header.get_zooms()
    enh_voxsize = enh_img.header.get_zooms()
    zoom_factors = np.array(orig_voxsize) / np.array(enh_voxsize)
    orig_resampled = zoom(orig_data, zoom_factors, order=1)

    # Crop to same size
    min_shape = [min(s1, s2) for s1, s2 in zip(orig_resampled.shape, enh_data.shape)]
    orig_crop = orig_resampled[:min_shape[0], :min_shape[1], :min_shape[2]]
    enh_crop = enh_data[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Compute metrics
    # 1. Signal-to-Noise Ratio (SNR)
    signal = np.mean(enh_crop[enh_crop > 0])
    noise = np.std(enh_crop[enh_crop > 0])
    snr = signal / noise if noise > 0 else 0

    # 2. Contrast-to-Noise Ratio (CNR)
    # Assume top 25% intensity is signal, bottom 25% is background
    sorted_vals = np.sort(enh_crop[enh_crop > 0])
    signal_region = sorted_vals[-len(sorted_vals)//4:]
    background_region = sorted_vals[:len(sorted_vals)//4]
    cnr = (np.mean(signal_region) - np.mean(background_region)) / np.std(background_region)

    # 3. Normalized correlation with original
    correlation = np.corrcoef(orig_crop.flatten(), enh_crop.flatten())[0, 1]

    return {
        'SNR': snr,
        'CNR': cnr,
        'Correlation': correlation,
        'Output_shape': enh_data.shape,
        'Output_voxsize': enh_voxsize[:3]
    }

# Example usage
metrics = compute_qc_metrics('original.nii.gz', 'enhanced_synthsr.nii.gz')
print("QC Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
```

## Clinical Applications

### Legacy Data Enhancement

```bash
# Enhance old low-resolution clinical scans for modern analysis

# 1990s scan: 1.5×1.5×7mm
mri_synthsr \
  --i legacy_1990s_T1.nii.gz \
  --o legacy_enhanced_1mm.nii.gz

# Now compatible with modern pipelines:
# - FreeSurfer
# - VBM analysis
# - Deep learning segmentation
```

### Multi-Site Harmonization

```bash
# Different sites have different protocols
# SynthSR brings all to common 1mm resolution

# Site A: High-res protocol (already 1mm)
mri_synthsr --i siteA_1mm.nii.gz --o siteA_synthsr.nii.gz

# Site B: Clinical protocol (1×1×5mm)
mri_synthsr --i siteB_5mm.nii.gz --o siteB_synthsr.nii.gz

# Site C: Fast protocol (1.5×1.5×3mm)
mri_synthsr --i siteC_3mm.nii.gz --o siteC_synthsr.nii.gz

# All outputs: 1mm isotropic, harmonized
```

### Pediatric Imaging

```bash
# Children often have motion artifacts and lower resolution scans
# SynthSR improves analysis feasibility

mri_synthsr \
  --i pediatric_T1_motion.nii.gz \
  --o pediatric_enhanced.nii.gz

# Enables:
# - More accurate segmentation
# - Better morphometry
# - Improved developmental analysis
```

### Elderly/Dementia Studies

```bash
# Atrophied brains and clinical protocols
# SynthSR maintains pathology while improving resolution

mri_synthsr \
  --i alzheimers_patient_clinical.nii.gz \
  --o alzheimers_enhanced.nii.gz

# Preserves:
# - Atrophy patterns
# - Ventricular enlargement
# - While improving resolution
```

## Integration with Claude Code

SynthSR integrates naturally into Claude-assisted workflows:

### Automated Pipeline Generation

```markdown
**Prompt to Claude:**
"Create a processing pipeline that:
1. Takes a directory of clinical T1w scans (mixed resolutions)
2. Enhances all with SynthSR to 1mm
3. Runs FreeSurfer on enhanced images
4. Generates QC report comparing before/after
5. Extracts volumetric measures
Include error handling and progress tracking."
```

### Quality Control Automation

```markdown
**Prompt to Claude:**
"Generate Python script for SynthSR QC:
1. Load original and enhanced images
2. Create side-by-side comparison mosaics
3. Compute SNR, CNR, and sharpness metrics
4. Generate HTML report with thumbnails
5. Flag subjects with low correlation (<0.8)
Include visualization code."
```

### Batch Processing Helper

```markdown
**Prompt to Claude:**
"Write a SLURM job array script to process 200 subjects with SynthSR:
- Each job processes one subject
- Uses GPU nodes
- Includes resource requests (1 GPU, 8GB RAM, 30 min)
- Logs output and errors
- Restarts failed jobs automatically"
```

## Integration with Other Tools

### FSL

```bash
# SynthSR as preprocessing for FSL pipelines

# Enhance T1w
mri_synthsr --i T1w.nii.gz --o T1w_synthsr.nii.gz

# FSL brain extraction
bet T1w_synthsr.nii.gz T1w_brain.nii.gz -f 0.5

# FSL FAST segmentation
fast -t 1 -n 3 -o T1w_brain T1w_brain.nii.gz
```

### ANTs

```bash
# Use enhanced images for better registration

# Enhance both images
mri_synthsr --i moving.nii.gz --o moving_synthsr.nii.gz
mri_synthsr --i fixed.nii.gz --o fixed_synthsr.nii.gz

# ANTs registration
antsRegistrationSyN.sh \
  -d 3 \
  -f fixed_synthsr.nii.gz \
  -m moving_synthsr.nii.gz \
  -o moving_to_fixed_
```

### SPM

```matlab
% Use SynthSR-enhanced images in SPM

% After SynthSR enhancement
enhanced_files = spm_select('FPList', '/path/to/enhanced', '^.*_synthsr\.nii$');

% SPM preprocessing
matlabbatch{1}.spm.spatial.preproc.channel.vols = cellstr(enhanced_files);
% ... rest of SPM batch

spm_jobman('run', matlabbatch);
```

### CAT12

```matlab
% VBM with SynthSR-enhanced clinical scans

% After SynthSR
enhanced_scans = cellstr(spm_select('FPList', '/enhanced', '^.*_synthsr\.nii$'));

% CAT12 processing
matlabbatch{1}.spm.tools.cat.estwrite.data = enhanced_scans;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 1;

spm_jobman('run', matlabbatch);
```

## Troubleshooting

### Problem 1: Out of Memory (CPU)

**Symptoms:** Process killed or memory error

**Solutions:**
```bash
# Reduce processing size by cropping
mri_synthsr \
  --i large_volume.nii.gz \
  --o enhanced.nii.gz \
  --crop 256  # Crop to 256³ voxels

# Or process on smaller subvolumes
# Or use machine with more RAM
```

### Problem 2: GPU Not Detected

**Symptoms:** Slow processing, CPU usage only

**Solutions:**
```bash
# Check TensorFlow GPU installation
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, reinstall TensorFlow GPU
pip uninstall tensorflow
pip install tensorflow-gpu

# Check CUDA installation
nvidia-smi

# Set GPU explicitly in Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
```

### Problem 3: Artifacts in Output

**Symptoms:** Unrealistic structures, checkerboard patterns

**Solutions:**
```bash
# Usually due to extreme input quality issues
# Try preprocessing first:

# 1. Intensity normalization
fslmaths input.nii.gz -inm 1000 input_norm.nii.gz

# 2. Then SynthSR
mri_synthsr --i input_norm.nii.gz --o enhanced.nii.gz

# 3. Check input orientation
fslreorient2std input.nii.gz input_reoriented.nii.gz
```

### Problem 4: Wrong Output Resolution

**Symptoms:** Output not 1mm isotropic

**Solutions:**
```bash
# Explicitly set target resolution
mri_synthsr \
  --i input.nii.gz \
  --o output.nii.gz \
  --target_res 1.0

# Verify output
fslinfo output.nii.gz | grep pixdim
# Should show: pixdim1 = 1.000000
```

### Problem 5: Very Different Intensity Distribution

**Symptoms:** Output intensities drastically different from input

**Solutions:**
```python
# SynthSR normalizes intensities
# This is expected and usually beneficial
# If needed, rescale to match original range

import nibabel as nib
import numpy as np

# Load images
orig = nib.load('original.nii.gz')
enhanced = nib.load('enhanced_synthsr.nii.gz')

orig_data = orig.get_fdata()
enh_data = enhanced.get_fdata()

# Rescale enhanced to match original range
orig_min, orig_max = np.percentile(orig_data[orig_data > 0], [1, 99])
enh_min, enh_max = np.percentile(enh_data[enh_data > 0], [1, 99])

enh_rescaled = (enh_data - enh_min) / (enh_max - enh_min) * (orig_max - orig_min) + orig_min

# Save rescaled
nib.save(nib.Nifti1Image(enh_rescaled, enhanced.affine), 'enhanced_rescaled.nii.gz')
```

## Best Practices

### When to Use SynthSR

1. **Clinical scans with thick slices** - Primary use case
2. **Legacy data** - Bring old scans to modern standards
3. **Multi-site studies** - Harmonize resolution differences
4. **Preprocessing for analysis** - Before FreeSurfer, VBM, etc.
5. **Poor quality scans** - Motion, low SNR can be improved

### When NOT to Use SynthSR

1. **Already high-quality 1mm isotropic** - No benefit, adds processing time
2. **Extreme pathology** - Massive lesions may confuse the model
3. **Non-brain imaging** - Trained on brain MRI
4. **Time-critical analysis** - If original quality sufficient
5. **Preserving exact original intensities** - SynthSR modifies intensities

### Workflow Recommendations

1. **Always visually inspect** - QC both input and output
2. **Keep original data** - Never delete source scans
3. **Document enhancement** - Note SynthSR use in methods
4. **Compare pipelines** - Test with and without SynthSR
5. **Use consistent version** - For reproducibility

### Processing Parameters

1. **Default 1mm target** - Appropriate for most analyses
2. **GPU recommended** - 10-30× faster than CPU
3. **Batch processing** - More efficient for large datasets
4. **Adequate disk space** - Output files same or larger than input
5. **Version tracking** - Record SynthSR version used

## Resources

### Official Documentation

- **SynthSR Website:** https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR
- **GitHub Repository:** https://github.com/freesurfer/freesurfer (SynthSR subdirectory)
- **FreeSurfer Wiki:** https://surfer.nmr.mgh.harvard.edu/
- **Installation Guide:** https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR#Installation

### Key Publications

- **SynthSR Paper:** Iglesias et al. (2021) "Joint super-resolution and synthesis of 1 mm isotropic MP-RAGE volumes from clinical MRI exams with scans of different orientation, resolution and contrast" NeuroImage
- **Domain Randomization:** Billot et al. (2023) "Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets" PNAS

### Learning Resources

- **Tutorial:** https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR#Tutorial
- **Example Data:** Provided with FreeSurfer installation
- **Video Demo:** FreeSurfer YouTube channel

### Community Support

- **FreeSurfer Mailing List:** https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSupport
- **GitHub Issues:** https://github.com/freesurfer/freesurfer/issues
- **Neurostars Forum:** https://neurostars.org/ (tag: freesurfer, synthsr)

## Citation

```bibtex
@article{Iglesias2021,
  title = {Joint super-resolution and synthesis of 1 mm isotropic MP-RAGE
           volumes from clinical MRI exams with scans of different orientation,
           resolution and contrast},
  author = {Iglesias, Juan Eugenio and Billot, Benjamin and Balbastre, Yaël and
            Tabley, Azadeh and Conklin, John and Gonz{\'a}lez, Raúl and
            Lev, Michael and Serrano-Pozo, Alberto and Frosch, Matthew and
            Augustinack, Jean and others},
  journal = {NeuroImage},
  volume = {237},
  pages = {118206},
  year = {2021},
  doi = {10.1016/j.neuroimage.2021.118206}
}
```

## Related Tools

- **SynthSeg** - Robust segmentation (same research group, complementary)
- **FreeSurfer** - Cortical surface reconstruction and analysis
- **SynthStrip** - Skull-stripping from same group
- **Super-Resolution CNN** - Alternative deep learning super-resolution
- **ANTs** - Image registration and normalization
- **FSL SUSAN** - Smoothing while preserving edges
- **CAT12** - Structural brain analysis
- **DeepResolve** - Alternative MRI super-resolution tool

---

**Skill Type:** Image Enhancement/Preprocessing
**Difficulty Level:** Beginner to Intermediate
**Prerequisites:** Python 3.6+, Basic neuroimaging knowledge, TensorFlow
**Typical Use Cases:** Clinical scan enhancement, legacy data improvement, multi-site harmonization, preprocessing for analysis
