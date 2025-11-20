# MRtrix3

## Overview

MRtrix3 is a comprehensive software package for diffusion MRI analysis, providing cutting-edge algorithms for tractography, tissue microstructure modeling, and structural connectivity analysis. It is particularly known for its advanced fiber tracking methods and constrained spherical deconvolution.

**Website:** https://www.mrtrix.org/
**Platform:** Linux/macOS/Windows (WSL)
**Language:** C++, with Python scripting
**License:** Mozilla Public License 2.0

## Key Features

- Constrained Spherical Deconvolution (CSD)
- Probabilistic tractography
- Anatomically-Constrained Tractography (ACT)
- Spherical-deconvolution Informed Filtering of Tractograms (SIFT)
- Fixel-based analysis
- Multi-shell multi-tissue CSD
- Advanced preprocessing tools
- DEC maps and connectivity matrices
- Interactive visualization (mrview)

## Installation

### From Package Managers

```bash
# Ubuntu/Debian
sudo apt-get install mrtrix3

# macOS (Homebrew)
brew install mrtrix3

# Or from source
git clone https://github.com/MRtrix3/mrtrix3.git
cd mrtrix3
./configure
./build
./set_path
```

### Verify Installation

```bash
mrinfo --version
dwi2tensor --help
```

## Preprocessing Workflow

### Basic DWI Preprocessing

```bash
# 1. Convert DICOM to MRtrix format
mrconvert dicoms/ dwi.mif

# 2. Denoise
dwidenoise dwi.mif dwi_denoised.mif -noise noise.mif

# 3. Remove Gibbs ringing
mrdegibbs dwi_denoised.mif dwi_degibbs.mif

# 4. Motion and distortion correction
dwifslpreproc dwi_degibbs.mif dwi_preproc.mif \
    -rpe_header \
    -eddy_options " --slm=linear --data_is_shelled" \
    -eddyqc_all eddy_qc

# 5. Bias field correction
dwibiascorrect ants dwi_preproc.mif dwi_corrected.mif \
    -bias bias_field.mif

# 6. Brain mask estimation
dwi2mask dwi_corrected.mif mask.mif
```

### Multi-Shell Processing

```bash
# Check shells
mrinfo -dwgrad dwi.mif

# Combine b0 volumes
dwiextract dwi.mif - -bzero | mrmath - mean mean_b0.mif -axis 3

# Response function estimation (multi-tissue)
dwi2response dhollander dwi_corrected.mif \
    wm_response.txt gm_response.txt csf_response.txt \
    -voxels voxels.mif

# Or for single-shell (tournier method)
dwi2response tournier dwi_corrected.mif wm_response.txt
```

## Fiber Orientation Distribution (FOD)

```bash
# Multi-shell multi-tissue CSD
dwi2fod msmt_csd dwi_corrected.mif \
    wm_response.txt wm_fod.mif \
    gm_response.txt gm.mif \
    csf_response.txt csf.mif \
    -mask mask.mif

# Single-shell CSD
dwi2fod csd dwi_corrected.mif \
    wm_response.txt wm_fod.mif \
    -mask mask.mif

# Intensity normalization across subjects
mtnormalise wm_fod.mif wm_fod_norm.mif \
    gm.mif gm_norm.mif \
    csf.mif csf_norm.mif \
    -mask mask.mif
```

## Tractography

### Probabilistic Tractography

```bash
# Whole-brain tractography
tckgen wm_fod_norm.mif tracks_10M.tck \
    -algorithm iFOD2 \
    -seed_image mask.mif \
    -select 10M \
    -cutoff 0.06 \
    -maxlength 250 \
    -minlength 10

# Anatomically-Constrained Tractography (ACT)
5ttgen fsl T1.nii.gz 5tt.mif
tckgen wm_fod_norm.mif tracks_ACT.tck \
    -algorithm iFOD2 \
    -act 5tt.mif \
    -backtrack \
    -crop_at_gmwmi \
    -seed_dynamic wm_fod_norm.mif \
    -select 10M

# Seed-based tracking
tckgen wm_fod_norm.mif seed_tracks.tck \
    -algorithm iFOD2 \
    -seed_image roi_seed.mif \
    -include roi_target.mif \
    -select 1000

# Exclusion regions
tckgen wm_fod_norm.mif tracks.tck \
    -algorithm iFOD2 \
    -seed_image mask.mif \
    -exclude lesion_mask.mif \
    -select 5M
```

### Track Filtering (SIFT)

```bash
# SIFT2 (spherical-deconvolution informed filtering)
tcksift2 tracks_10M.tck wm_fod_norm.mif sift_weights.txt \
    -act 5tt.mif \
    -out_mu mu.txt \
    -out_coeffs coeffs.txt

# SIFT (original, reduces track count)
tcksift tracks_10M.tck wm_fod_norm.mif tracks_1M_sift.tck \
    -term_number 1M \
    -act 5tt.mif
```

### Track Processing

```bash
# Filter tracks by length
tckedit tracks.tck filtered_tracks.tck \
    -minlength 10 -maxlength 250

# Extract subset
tckedit tracks.tck subset.tck -number 100k

# Concatenate track files
tckedit tracks1.tck tracks2.tck combined.tck

# Track statistics
tckstats tracks.tck -dump stats.txt
```

## Connectivity Analysis

### Parcellation-based Connectome

```bash
# Convert FreeSurfer parcellation to MRtrix format
labelconvert aparc+aseg.mgz \
    FreeSurferColorLUT.txt \
    fs_default.txt \
    parcellation.mif

# Register parcellation to DWI space
mrtransform parcellation.mif parcellation_dwi.mif \
    -interp nearest \
    -from T1_to_dwi.txt \
    -inverse

# Generate connectome
tck2connectome tracks.tck parcellation_dwi.mif connectome.csv \
    -tck_weights_in sift_weights.txt \
    -symmetric \
    -zero_diagonal \
    -scale_invnodevol

# Additional metrics
tck2connectome tracks.tck parcellation_dwi.mif \
    mean_FA.csv \
    -scale_file fa.mif \
    -stat_edge mean

# View connectivity matrix
mrview connectome.csv
```

## Fixel-Based Analysis

```bash
# Compute fiber density and cross-section
# 1. Warp FODs to template
mrregister wm_fod.mif template_fod.mif \
    -mask1 mask.mif \
    -nl_warp subject2template_warp.mif \
    template2subject_warp.mif

# 2. Compute fixel metrics
fod2fixel wm_fod.mif fixels/ -mask mask.mif

# 3. Warp fixels to template
fixelcorrespondence fixels/fd.mif \
    template_fixels fixels_warped/ \
    subject2template_warp.mif

# 4. Statistical analysis
fixelcfestats fixels_warped/ design.txt contrast.txt \
    tractogram.tck output_stats/
```

## Visualization

```bash
# View DWI
mrview dwi.mif

# View with overlays
mrview T1.mif -overlay mask.mif

# View FODs
mrview dwi.mif -odf.load_sh wm_fod.mif

# View tractography
mrview T1.mif -tractography.load tracks.tck

# View with multiple modalities
mrview dwi.mif \
    -overlay fa.mif \
    -tractography.load tracks.tck \
    -odf.load_sh wm_fod.mif

# Create screenshot
mrview dwi.mif -mode 2 \
    -tractography.load tracks.tck \
    -capture.folder screenshots/ \
    -capture.prefix tract_ \
    -capture.grab \
    -exit
```

## Tensor Metrics

```bash
# Fit tensor
dwi2tensor dwi_corrected.mif tensor.mif -mask mask.mif

# Compute FA
tensor2metric tensor.mif -fa fa.mif -mask mask.mif

# Compute multiple metrics
tensor2metric tensor.mif \
    -fa fa.mif \
    -ad ad.mif \
    -rd rd.mif \
    -adc md.mif \
    -vector eigenvec.mif \
    -mask mask.mif

# Directionally-Encoded Color (DEC) map
dwi2tensor dwi.mif - | tensor2metric - -vec - | \
    mrconvert - dec.mif -coord 3 0:2
```

## Registration

```bash
# Registration to T1
mrregister mean_b0.mif T1.nii.gz \
    -type rigid \
    -rigid dwi_to_T1.txt \
    -mask1 mask.mif

# Apply transformation
mrtransform dwi.mif dwi_in_T1.mif \
    -linear dwi_to_T1.txt \
    -template T1.nii.gz

# Nonlinear registration
mrregister moving.mif fixed.mif \
    -type nonlinear \
    -nl_warp warp.mif \
    -nl_warp_full nl_warp_full.mif
```

## Integration with Claude Code

When helping users with MRtrix3:

1. **File Formats:** MRtrix uses .mif/.mih (native) but can read NIfTI

2. **Check Data:**
   ```bash
   mrinfo dwi.mif # Check headers
   mrinfo -dwgrad dwi.mif # Check gradients
   ```

3. **Memory Usage:** Large datasets may require significant RAM

4. **Parallelization:** Many commands support `-nthreads` option

5. **Common Issues:**
   - Missing or incorrect gradient tables
   - Insufficient b-values for CSD
   - Memory limitations
   - FSL not available for dwifslpreproc

## Best Practices

- Always denoise and remove Gibbs artifacts first
- Visually inspect results at each step
- Use ACT for anatomically-informed tractography
- Apply SIFT/SIFT2 for quantitative analysis
- Use multi-shell acquisitions for msmt_csd
- Keep gradient tables with data
- Document all processing steps

## Troubleshooting

**Problem:** "dwi2response failed"
**Solution:** Check sufficient voxels in each tissue type, try alternative algorithm

**Problem:** Poor tractography results
**Solution:** Check FOD quality, adjust cutoff/parameters, use ACT

**Problem:** "FSL not found" during dwifslpreproc
**Solution:** Install and configure FSL

## Resources

- MRtrix3 Documentation: https://mrtrix.readthedocs.io/
- Community Forum: https://community.mrtrix.org/
- GitHub: https://github.com/MRtrix3/mrtrix3
- Tutorial: https://osf.io/fkyht/

## Related Tools

- **MRView:** Interactive viewer
- **MRtrix3Tissue:** Enhanced tissue segmentation
- **Population Template:** Template building scripts
- **Connectome Workbench:** For surface visualization

## Citation

```bibtex
@article{tournier2019mrtrix3,
  title={MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation},
  author={Tournier, J.-Donald and Smith, Robert and Raffelt, David and others},
  journal={NeuroImage},
  volume={202},
  pages={116137},
  year={2019},
  doi={10.1016/j.neuroimage.2019.116137}
}
```
