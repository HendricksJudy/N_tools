# FSL (FMRIB Software Library)

## Overview

FSL is a comprehensive library of analysis tools for fMRI, MRI, and DTI brain imaging data. Developed at the Oxford Centre for Functional MRI of the Brain (FMRIB), it provides both command-line tools and GUI applications for various neuroimaging analyses.

**Website:** https://fsl.fmrib.ox.ac.uk/
**Platform:** Linux/macOS/Windows (via WSL)
**Language:** C++, Shell, Python
**License:** Custom academic license (free for academic use)

## Key Features

- **FEAT:** fMRI analysis tool
- **MELODIC:** ICA for fMRI denoising
- **BET:** Brain extraction
- **FLIRT/FNIRT:** Linear and non-linear registration
- **FAST:** Tissue segmentation
- **FDT:** Diffusion MRI analysis
- **TBSS:** Tract-Based Spatial Statistics
- **FIRST:** Subcortical structure segmentation
- **FSLeyes:** Modern image viewer and editor

## Installation

### Linux/macOS

```bash
# Download FSL installer
wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py

# Run installer
python fslinstaller.py

# Add to shell configuration (~/.bashrc or ~/.zshrc)
export FSLDIR=/usr/local/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}
```

### Verify Installation

```bash
flirt -version
bet -h
```

## Common Workflows

### Brain Extraction

```bash
# Basic brain extraction
bet input.nii.gz output_brain.nii.gz

# With skull image output
bet input.nii.gz output_brain.nii.gz -s

# Fractional intensity threshold (0-1)
bet input.nii.gz output_brain.nii.gz -f 0.5

# For functional images (faster, less aggressive)
bet input.nii.gz output_brain.nii.gz -F
```

### Motion Correction

```bash
# MCFLIRT - motion correction
mcflirt -in input_4d.nii.gz -out mcf_output -plots

# View motion parameters
fsl_tsplot -i mcf_output.par -t 'MCFLIRT estimated rotations (radians)' -u 1 --start=1 --finish=3 -a x,y,z -w 640 -h 144 -o rot.png

# Generate mean functional image
fslmaths input_4d.nii.gz -Tmean mean_func.nii.gz
```

### Registration

```bash
# Linear registration with FLIRT
flirt -in functional.nii.gz \
      -ref structural.nii.gz \
      -out func_to_struct.nii.gz \
      -omat func_to_struct.mat \
      -dof 6 # Degrees of freedom (6, 7, 9, or 12)

# Apply transformation
flirt -in input.nii.gz \
      -ref reference.nii.gz \
      -out output.nii.gz \
      -init transform.mat \
      -applyxfm

# Non-linear registration with FNIRT
fnirt --in=input.nii.gz \
      --ref=reference.nii.gz \
      --iout=output.nii.gz \
      --cout=coef_output
```

### Tissue Segmentation

```bash
# FAST - tissue type segmentation
fast -t 1 -n 3 -o output_prefix input.nii.gz

# -t: image type (1=T1, 2=T2, 3=PD)
# -n: number of tissue classes (usually 3)
# -o: output basename

# Output files:
# output_prefix_seg.nii.gz - segmentation
# output_prefix_pve_0/1/2.nii.gz - partial volume estimates
```

### fMRI Analysis with FEAT

```bash
# Create FEAT design file (can use GUI or manually)
Feat &

# Or run from command line with existing design
feat design.fsf

# Higher-level analysis
feat_model design
flameo --copefile=design.grp \
        --covsplitfile=design.con \
        --designfile=design.mat \
        --ld=stats \
        --runmode=flame1 \
        --td=stats
```

### Diffusion MRI Analysis

```bash
# Correct for eddy currents and motion
eddy_correct diffusion.nii.gz diffusion_edc.nii.gz 0

# Or use modern eddy tool
eddy --imain=diffusion.nii.gz \
     --mask=nodif_brain_mask.nii.gz \
     --acqp=acqparams.txt \
     --index=index.txt \
     --bvecs=bvecs \
     --bvals=bvals \
     --out=eddy_corrected

# Fit diffusion tensors
dtifit -k diffusion_edc.nii.gz \
       -o dti \
       -m nodif_brain_mask.nii.gz \
       -r bvecs \
       -b bvals

# Probabilistic tractography
probtrackx2 -x seed_mask.nii.gz \
            -l --onewaycondition \
            -c 0.2 \
            -S 2000 \
            --steplength=0.5 \
            -P 5000 \
            --forcedir --opd \
            -s bedpostX_output/merged \
            -m nodif_brain_mask.nii.gz \
            --dir=output_dir
```

### ICA Denoising with MELODIC

```bash
# Run MELODIC
melodic -i input_4d.nii.gz \
        -o melodic_output \
        --nobet \
        --tr=2.0 \
        -d 0 # Automatic dimensionality estimation

# Manual cleanup (identify noise components)
# Then remove them with fsl_regfilt
fsl_regfilt -i input_4d.nii.gz \
            -o denoised.nii.gz \
            -d melodic_output/melodic_mix \
            -f "1,2,5,8" # Component numbers to remove
```

## Useful FSL Utilities

```bash
# View image
fsleyes image.nii.gz &

# Get image information
fslinfo image.nii.gz

# Image statistics
fslstats image.nii.gz -M -S # Mean and standard deviation
fslstats image.nii.gz -R # Min and max

# Math operations
fslmaths input.nii.gz -add 100 output.nii.gz
fslmaths input.nii.gz -mul 2 output.nii.gz
fslmaths input.nii.gz -thr 50 output.nii.gz # Threshold

# Merge volumes
fslmerge -t output_4d vol1.nii.gz vol2.nii.gz vol3.nii.gz

# Split 4D into 3D volumes
fslsplit input_4d.nii.gz output_prefix -t

# Change orientation
fslreorient2std input.nii.gz output.nii.gz

# Swap dimensions
fslswapdim input.nii.gz x -y z output.nii.gz
```

## Integration with Claude Code

When helping users with FSL:

1. **Environment Check:**
   ```bash
   echo $FSLDIR
   which fsl
   ```

2. **File Formats:** FSL primarily uses NIfTI (.nii.gz)

3. **Parallel Processing:** Many FSL tools don't parallelize well - consider processing subjects separately

4. **Memory Requirements:** Large datasets may need significant RAM

5. **Common Pitfalls:**
   - FSL environment not sourced
   - Incorrect file paths
   - Missing brain masks
   - Incompatible image orientations

## Best Practices

- Always visually inspect results (use fsleyes)
- Keep original data untouched
- Use descriptive output names
- Document your analysis steps
- Check image orientations before registration
- Use brain masks where appropriate
- Save transformation matrices for reproducibility

## Performance Tips

```bash
# Run multiple subjects in parallel
for subj in sub-*; do
    (bet ${subj}/T1.nii.gz ${subj}/T1_brain.nii.gz) &
done
wait

# Use FSL's parallel processing
# Set FSLPARALLEL in environment
export FSLPARALLEL=1
```

## Troubleshooting

**Problem:** "fsl: command not found"
**Solution:** Source FSL configuration: `source ${FSLDIR}/etc/fslconf/fsl.sh`

**Problem:** Registration produces poor results
**Solution:** Check image orientations, manually verify starting position, adjust cost function

**Problem:** BET removes too much/little brain
**Solution:** Adjust fractional intensity threshold (-f parameter)

## Resources

- FSL Course Materials: https://fsl.fmrib.ox.ac.uk/fslcourse/
- FSL Wiki: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
- Email Support: https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=fsl
- FSLeyes Documentation: https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/

## Related Tools

- **FSLeyes:** Modern FSL viewer
- **Featquery:** ROI analysis for FEAT results
- **Randomise:** Permutation testing for statistics
- **PALM:** Permutation Analysis of Linear Models
- **AutoPtx:** Automated probabilistic tractography

## Citation

```bibtex
@article{jenkinson2012fsl,
  title={FSL},
  author={Jenkinson, Mark and Beckmann, Christian F. and Behrens, Timothy E. J. and Woolrich, Mark W. and Smith, Stephen M.},
  journal={NeuroImage},
  volume={62},
  number={2},
  pages={782--790},
  year={2012},
  doi={10.1016/j.neuroimage.2011.09.015}
}
```
