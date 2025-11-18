# BrainSuite - Cortical Surface Modeling and Analysis

## Overview

**BrainSuite** is a comprehensive software suite for processing, analyzing, and visualizing structural and diffusion MRI data, with particular emphasis on cortical surface modeling, anatomical labeling, and multi-modal integration. Developed at the UCLA Laboratory of Neuro Imaging, BrainSuite provides both an intuitive graphical user interface and powerful command-line tools for automated brain extraction, cortical surface generation, registration, and parcellation using multiple anatomical atlases.

BrainSuite excels at combining anatomical MRI with diffusion tractography, enabling structural connectivity analysis anchored to precise cortical labels. The software supports interactive editing for challenging clinical cases, automated batch processing for research cohorts, and is available as a BIDS App for containerized deployment.

**Key Features:**
- Interactive GUI and command-line interface
- Cortical surface extraction (inner/white, pial, mid-cortical)
- Brain extraction (BSE - Brain Surface Extractor)
- Bias field correction (BFC)
- Tissue classification (PVC - Partial Volume Classifier)
- Cortical labeling with BrainSuite Labeling Protocol (BCI-DNI atlas)
- Subcortical structure segmentation
- Non-linear volumetric registration (SVReg)
- Surface-based registration (Curve-Based Registration)
- Diffusion MRI processing and tractography (BDP)
- Surface-constrained tractography
- Multi-atlas labeling and fusion
- Population atlas creation
- Integration with FSL, FreeSurfer outputs
- BIDS App for containerized processing
- Statistical analysis tools

**Primary Use Cases:**
- Interactive cortical surface editing and refinement
- Multi-modal imaging (T1w + diffusion MRI)
- Structural connectivity analysis
- Clinical neuroimaging requiring precise labeling
- Cortical morphometry studies
- Teaching and training environments
- Method development and validation
- Population atlas creation
- Surface-based statistics

**Official Documentation:** http://brainsuite.org/

---

## Installation and Setup

### Download and Install

```bash
# Download BrainSuite from official website
# http://brainsuite.org/download/

# Linux installation
wget http://brainsuite.org/downloads/BrainSuite21a_linux.tar.gz
tar -xzvf BrainSuite21a_linux.tar.gz
cd BrainSuite21a

# Add to PATH
export BRAINS

UITE_PATH=/opt/BrainSuite21a
export PATH=$BRAINSUITE_PATH/bin:$PATH

# Add to ~/.bashrc for persistence
echo "export BRAINSUITE_PATH=/opt/BrainSuite21a" >> ~/.bashrc
echo 'export PATH=$BRAINSUITE_PATH/bin:$PATH' >> ~/.bashrc

# macOS installation
# Download .dmg and drag to Applications
# Or use command line:
# hdiutil mount BrainSuite21a_macos.dmg
# cp -r /Volumes/BrainSuite21a/BrainSuite21a.app /Applications/

# Windows installation
# Download installer and run
# BrainSuite21a_windows.exe
```

### Verify Installation

```bash
# Check BrainSuite installation
bse --version
# BrainSuite21a

# Launch GUI
BrainSuite &

# Check MATLAB Runtime (if needed for some tools)
# Some BrainSuite tools require MATLAB Runtime R2019b
# Download from MathWorks if not included
```

### BIDS App Installation

```bash
# BrainSuite BIDS App via Docker
docker pull bids/brainsuite

# Or Singularity
singularity pull docker://bids/brainsuite

# Verify BIDS App
docker run -it --rm bids/brainsuite --version
```

---

## GUI-Based Processing

### Launch BrainSuite GUI

```bash
# Start BrainSuite GUI
BrainSuite &

# GUI will open with menu bar:
# - File: Load images, save
# - Cortex: Cortical Surface Extraction Sequence
# - Cerebrum: Cerebrum labeling
# - SVReg: Surface-Volume Registration
# - BDP: BrainSuite Diffusion Pipeline
```

### Load T1w Image

```bash
# In GUI:
# File > Open Volume > Select T1w NIfTI file

# Or from command line:
BrainSuite /data/sub-01_T1w.nii.gz &

# BrainSuite will load image in viewer
# Axial, Coronal, Sagittal views
# 3D surface rendering panel
```

### Run Cortical Surface Extraction Sequence

```bash
# In GUI:
# Cortex > Cortical Surface Extraction Sequence (CSE)

# This opens dialog with sequence steps:
# 1. BSE (Brain Surface Extractor) - skull stripping
# 2. BFC (Bias Field Corrector) - inhomogeneity correction
# 3. PVC (Partial Volume Classifier) - tissue segmentation
# 4. Cerebro - cerebrum mask generation
# 5. Cortex - inner cortical surface
# 6. Scrubmask - surface cleanup
# 7. Tca - topology correction
# 8. Dewisp - remove thin strands
# 9. Pial - outer cortical surface

# Click "Run All" to execute full sequence
# Or run steps individually for fine control

# Processing takes 20-60 minutes
# Progress shown in status bar
```

### Interactive Viewing and Editing

```bash
# After processing, surfaces displayed in 3D panel

# View controls:
# - Rotate: Left click + drag
# - Zoom: Scroll wheel
# - Pan: Middle click + drag

# Toggle display:
# - Inner surface (white matter)
# - Pial surface
# - Mid-cortical surface
# - Brain mask
# - Tissue labels

# Edit brain mask:
# - Tools > Mask Editor
# - Paint tool: Add to mask
# - Erase tool: Remove from mask
# - Smooth tool: Smooth boundaries
# - Apply edits and rerun surface extraction
```

---

## Surface Extraction Pipeline

### Brain Surface Extractor (BSE)

```bash
# Skull stripping - command line
bse \
    --auto \
    --trim \
    -i /data/sub-01_T1w.nii.gz \
    -o /output/sub-01_brain.nii.gz \
    --mask /output/sub-01_brain.mask.nii.gz

# Options:
# --auto: Automatic parameter selection
# --trim: Crop image to brain
# --diffusionIterations: Edge detection iterations (default: 3)
# --diffusionConstant: Diffusion smoothing (default: 25)
# --edgeDetectionConstant: Edge threshold (default: 0.64)

# GUI equivalent: Cortex > BSE (Brain Surface Extractor)
```

### Bias Field Correction (BFC)

```bash
# Correct intensity inhomogeneity
bfc \
    -i /output/sub-01_brain.nii.gz \
    -o /output/sub-01_brain.bfc.nii.gz

# Options:
# --iterate: Number of iterations (default: 3)
# --biasEstimateSpacing: Control point spacing (default: 20mm)
# --biasEstimateConvergenceThreshold: Convergence (default: 0.001)
# --histogramRadius: Histogram smoothing (default: 5)

# BFC improves tissue classification accuracy
```

### Partial Volume Classifier (PVC)

```bash
# Tissue classification: GM, WM, CSF
pvc \
    -i /output/sub-01_brain.bfc.nii.gz \
    -o /output/sub-01_brain.bfc.pvc.label.nii.gz \
    -f /output/sub-01_brain.bfc.pvc.frac.nii.gz

# Output:
# .label.nii.gz: Discrete labels (0=bg, 1=CSF, 2=GM, 3=WM)
# .frac.nii.gz: Fractional tissue volumes (3 volumes)

# Labels used for surface generation
```

### Cerebrum Extraction

```bash
# Separate cerebrum from cerebellum/brainstem
cerebroextract \
    -i /output/sub-01_brain.bfc.nii.gz \
    -o /output/sub-01.cerebrum.mask.nii.gz \
    --pvc /output/sub-01_brain.bfc.pvc.label.nii.gz

# Cerebrum mask used for cortical surface extraction
# Ensures cortical surfaces don't include cerebellum
```

### Inner Cortical Surface Generation

```bash
# Generate white matter (inner cortical) surface
cortex \
    --pvc /output/sub-01_brain.bfc.pvc.label.nii.gz \
    --cerebrum /output/sub-01.cerebrum.mask.nii.gz \
    --output /output/sub-01.inner.cortex.dfs

# Output: .dfs format (BrainSuite surface format)
# Surface positioned at GM/WM interface
```

### Topology Correction

```bash
# Correct topological defects (holes, handles)
tca \
    -i /output/sub-01.inner.cortex.dfs \
    -o /output/sub-01.inner.cortex.tca.dfs

# Ensures cortical surface has spherical topology
# Required for subsequent analysis
```

### Pial Surface Generation

```bash
# Generate pial (outer cortical) surface
pial \
    --cortex /output/sub-01.inner.cortex.tca.dfs \
    --tissue /output/sub-01_brain.bfc.pvc.label.nii.gz \
    --output /output/sub-01.pial.cortex.dfs

# Pial surface at GM/CSF boundary
# Expands from inner surface to brain boundary
```

---

## Cortical Labeling

### BrainSuite Labeling Protocol (BCI-DNI)

```bash
# Apply BrainSuite BCI-DNI atlas
# Cortical and subcortical labels

# Requires SVReg registration first (see below)

# After SVReg, apply labels:
applyLabels \
    --subjectFile /output/sub-01_T1w.nii.gz \
    --atlasFile $BRAINSUITE_PATH/atlas/BCI-DNI_brain.nii.gz \
    --labelsFile $BRAINSUITE_PATH/atlas/BCI-DNI_brain.label.nii.gz \
    --deformation /output/sub-01.svreg.inv.map.nii.gz \
    --output /output/sub-01.BCI-DNI.label.nii.gz

# BCI-DNI atlas includes:
# - 98 cortical regions (49 per hemisphere)
# - Subcortical structures (thalamus, basal ganglia, etc.)
# - Brainstem and cerebellum labels
```

### View Labels in GUI

```bash
# Load labeled volume in BrainSuite GUI
BrainSuite /data/sub-01_T1w.nii.gz &

# Load label file:
# File > Open Label Volume > Select .label.nii.gz

# Labels displayed as colored overlay
# Click on region to see label name
# Toggle label visibility
```

### Extract Label Statistics

```bash
# Compute volume for each labeled region
labelstats \
    --image /output/sub-01_T1w.nii.gz \
    --labels /output/sub-01.BCI-DNI.label.nii.gz \
    --output /output/sub-01_label_stats.csv

# CSV contains:
# Label_ID, Label_Name, Volume_mm3, Mean_Intensity, Std_Intensity

# Use in further analysis:
import pandas as pd
stats = pd.read_csv('/output/sub-01_label_stats.csv')
print(stats[['Label_Name', 'Volume_mm3']])
```

---

## Surface Registration

### Surface-Constrained Volume Registration (SVReg)

```bash
# Non-linear volumetric registration to atlas
# Constrained by cortical surface correspondence

# Full SVReg command:
svreg.sh \
    /output/sub-01 \
    $BRAINSUITE_PATH/atlas/BCI-DNI_brain

# This performs:
# 1. Affine registration
# 2. Surface-based curve matching
# 3. Volume deformation guided by surfaces
# 4. Refinement and inverse map generation

# Outputs:
# sub-01.svreg.map.nii.gz: Forward deformation
# sub-01.svreg.inv.map.nii.gz: Inverse deformation
# sub-01.svreg.label.nii.gz: Transferred atlas labels

# Processing time: 30-90 minutes
```

### Curve-Based Surface Registration

```bash
# Register cortical surfaces based on sulcal curves
# More accurate than volume-only registration

# Extract sulcal curves
curvatureStats \
    --surface /output/sub-01.pial.cortex.dfs \
    --output /output/sub-01.pial.cortex.curves.dfs

# Register to atlas curves
surfaceRegister \
    --subject /output/sub-01.pial.cortex.curves.dfs \
    --atlas $BRAINSUITE_PATH/atlas/BCI-DNI_cortex.curves.dfs \
    --output /output/sub-01.registered.dfs

# Registration enables:
# - Vertex-wise correspondence across subjects
# - Surface-based statistics
# - Accurate anatomical matching
```

---

## Command-Line Batch Processing

### Process Single Subject (Full Pipeline)

```bash
# Run complete BrainSuite pipeline
# One command for all processing steps

bse.sh /data/sub-01_T1w.nii.gz

# This script runs:
# bse, bfc, pvc, cerebro, cortex, scrubmask, tca, dewisp, pial
# Outputs saved in same directory as input

# Or with explicit output directory:
brainsuite.sh \
    --input /data/sub-01_T1w.nii.gz \
    --output /output/sub-01

# Full pipeline takes 30-90 minutes
```

### BrainSuite Dashboard for Batch Processing

```bash
# BrainSuite Dashboard: GUI for batch processing

# Launch Dashboard:
BrainSuiteDashboard &

# In Dashboard:
# 1. Click "Add Subjects" - select T1w images
# 2. Select processing stages:
#    - Cortical Surface Extraction
#    - Cortical Labeling (SVReg)
#    - Diffusion Processing (if DWI available)
# 3. Set parameters (or use defaults)
# 4. Click "Start Processing"

# Dashboard handles:
# - Job queuing
# - Progress monitoring
# - Error handling
# - Quality control checks
# - Result organization
```

### Batch Processing Script

```bash
# Process multiple subjects with shell script
cat > process_brainsuite_batch.sh <<'EOF'
#!/bin/bash

# Subject list
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05)

for subject in "${subjects[@]}"; do
    echo "Processing $subject..."

    # Run full pipeline
    bse.sh /data/${subject}_T1w.nii.gz

    # Run SVReg if successful
    if [ -f /data/${subject}.pial.cortex.dfs ]; then
        svreg.sh /data/$subject $BRAINSUITE_PATH/atlas/BCI-DNI_brain
    fi

    echo "$subject complete"
done
EOF

chmod +x process_brainsuite_batch.sh
./process_brainsuite_batch.sh
```

---

## Diffusion MRI Integration

### BrainSuite Diffusion Pipeline (BDP)

```bash
# Process diffusion MRI data
# Requires T1w already processed with BrainSuite

# BDP performs:
# - Distortion correction
# - Registration to T1w
# - Tensor estimation
# - Fiber tracking
# - Connectivity analysis

# Run BDP
bdp.sh \
    --t1 /output/sub-01_T1w.nii.gz \
    --dwi /data/sub-01_dwi.nii.gz \
    --bval /data/sub-01_dwi.bval \
    --bvec /data/sub-01_dwi.bvec \
    --output /output/sub-01_BDP

# Outputs:
# - Corrected DWI
# - DTI maps (FA, MD, AD, RD)
# - Fiber orientation distributions
# - Tractography
```

### Surface-Constrained Tractography

```bash
# Generate tractography constrained by cortical surfaces
# Fibers must start/end at cortical GM

# After BDP, run surface-constrained tracking:
bdp_track_surface \
    --t1 /output/sub-01_T1w.nii.gz \
    --inner /output/sub-01.inner.cortex.dfs \
    --pial /output/sub-01.pial.cortex.dfs \
    --fib /output/sub-01_BDP/sub-01.fib.gz \
    --output /output/sub-01_surface_tracks.trk

# Ensures tractography:
# - Terminates at cortical surface
# - Doesn't enter CSF or skull
# - Provides cortical connectivity
```

### Structural Connectivity Matrix

```bash
# Compute connectivity matrix between cortical regions
# Based on tractography and cortical parcellation

connectivityMatrix \
    --tracts /output/sub-01_surface_tracks.trk \
    --labels /output/sub-01.BCI-DNI.label.nii.gz \
    --output /output/sub-01_connectivity_matrix.mat

# Output: NxN matrix
# N = number of cortical regions
# Values = connection strength (fiber count/density)

# Analyze in MATLAB/Python:
import scipy.io
conn = scipy.io.loadmat('/output/sub-01_connectivity_matrix.mat')
print(conn['connectivity'].shape)  # e.g., (98, 98)
```

---

## Quality Control and Manual Editing

### Visual Inspection

```bash
# Load subject in GUI for QC
BrainSuite /output/sub-01_T1w.nii.gz &

# Check key steps:
# 1. Brain extraction - all brain, no skull
# 2. Tissue classification - GM/WM boundary clear
# 3. Inner surface - follows white matter
# 4. Pial surface - follows gray matter boundary
# 5. Labels - anatomically accurate

# Common issues:
# - Incomplete skull stripping
# - Dura included in brain
# - Incorrect tissue labels
# - Surface topology errors
# - Misaligned labels
```

### Edit Brain Mask

```bash
# If skull stripping failed:
# Tools > Mask Editor

# Paint tool: Add missing brain regions
# - Adjust brush size
# - Click/drag to paint
# - Include all brain tissue

# Erase tool: Remove non-brain regions
# - Remove skull fragments
# - Remove dura mater
# - Remove blood vessels outside brain

# Apply edits:
# - File > Save Mask
# - Cortex > Run from PVC (rerun from tissue classification)
```

### Correct Tissue Classification

```bash
# If GM/WM boundary incorrect:
# Tools > Label Editor

# Manual label correction:
# - Load PVC labels
# - Paint correct tissue type
# - Save corrected labels
# - Rerun surface generation

# Or adjust PVC parameters and rerun:
pvc \
    --sameGaussianParameters \
    --spatialPrior 0.2 \
    -i /output/sub-01_brain.bfc.nii.gz \
    -o /output/sub-01_brain.bfc.pvc.corrected.label.nii.gz
```

---

## Statistical Analysis

### Extract Morphometric Measures

```bash
# Surface area calculation
surfaceArea \
    --surface /output/sub-01.pial.cortex.dfs \
    --output /output/sub-01_surface_area.txt

# Cortical thickness
corticalThickness \
    --inner /output/sub-01.inner.cortex.dfs \
    --pial /output/sub-01.pial.cortex.dfs \
    --output /output/sub-01_thickness.txt

# Mean thickness per label
corticalThicknessStats \
    --thickness /output/sub-01_thickness.txt \
    --labels /output/sub-01.BCI-DNI.label.nii.gz \
    --output /output/sub-01_thickness_by_region.csv
```

### Group Analysis

```R
# Load morphometric data for all subjects
library(data.table)

# Read all subject thickness CSVs
subjects <- c("sub-01", "sub-02", "sub-03", ...)
all_data <- list()

for (subj in subjects) {
    data <- fread(paste0("/output/", subj, "_thickness_by_region.csv"))
    data$Subject <- subj
    all_data[[subj]] <- data
}

thickness_df <- rbindlist(all_data)

# Add covariates
covariates <- fread("subject_demographics.csv")
thickness_df <- merge(thickness_df, covariates, by="Subject")

# Statistical analysis
# Compare groups
library(lme4)
model <- lmer(Thickness ~ Group + Age + Sex + (1|Region),
              data=thickness_df)
summary(model)

# Plot results
library(ggplot2)
ggplot(thickness_df, aes(x=Region, y=Thickness, fill=Group)) +
    geom_boxplot() +
    theme(axis.text.x = element_text(angle=90, hjust=1))
```

---

## BIDS App Usage

### Run BrainSuite BIDS App

```bash
# Process BIDS dataset with BrainSuite BIDS App
docker run -it --rm \
    -v /data/bids_dataset:/bids_dir:ro \
    -v /output:/output_dir \
    bids/brainsuite \
    /bids_dir /output_dir participant \
    --participant_label 01 02 03

# Options:
# --stages: cse (cortical surface extraction), svreg (registration), bdp (diffusion)
# --atlas: BCI-DNI, Desikan-Killiany, or custom
# --modeling: Tensor, GQI, etc. (for diffusion)

# Example: Only cortical surface extraction
docker run -it --rm \
    -v /data/bids_dataset:/bids_dir:ro \
    -v /output:/output_dir \
    bids/brainsuite \
    /bids_dir /output_dir participant \
    --stages CSE \
    --skip-bse-qa  # Skip QA step for automation
```

### BIDS App Output Structure

```bash
# BrainSuite BIDS App outputs in BIDS derivatives format
/output_dir/
└── brainsuite/
    ├── dataset_description.json
    └── sub-01/
        └── anat/
            ├── sub-01_T1w.bse.nii.gz      # Brain extracted
            ├── sub-01_T1w.pvc.label.nii.gz # Tissue labels
            ├── sub-01_left.inner.cortex.dfs # Surfaces
            ├── sub-01_left.pial.cortex.dfs
            └── sub-01.BCI-DNI.label.nii.gz # Parcellation
```

---

## Integration with Other Tools

### Export to FreeSurfer Format

```bash
# Convert BrainSuite surface to FreeSurfer format
dfs2surf \
    --input /output/sub-01.pial.cortex.left.dfs \
    --output /output/sub-01.lh.pial

# Now compatible with FreeSurfer tools:
# - FreeView visualization
# - FreeSurfer statistics
# - SurfStat analysis
```

### Import FreeSurfer Surfaces

```bash
# Convert FreeSurfer surface to BrainSuite DFS
surf2dfs \
    --input /freesurfer/sub-01/surf/lh.pial \
    --output /output/sub-01_fs.pial.left.dfs

# View in BrainSuite GUI
BrainSuite --surface /output/sub-01_fs.pial.left.dfs
```

### FSL Integration

```bash
# Use FSL-processed images as input
# BrainSuite works with any NIfTI

# Example: Use FSL BET output
bet /data/sub-01_T1w.nii.gz /output/sub-01_brain -R -f 0.5

# Continue with BrainSuite pipeline
bfc -i /output/sub-01_brain.nii.gz -o /output/sub-01_brain.bfc.nii.gz
# ... continue pipeline
```

---

## Advanced Usage

### Multi-Atlas Labeling

```bash
# Use multiple atlases and fuse labels
# Improves labeling accuracy

# Register subject to multiple atlases
for atlas in Atlas1 Atlas2 Atlas3; do
    svreg.sh \
        /output/sub-01 \
        $BRAINSUITE_PATH/atlas/$atlas
done

# Fuse labels via majority voting
fuseLabels \
    --label1 /output/sub-01.Atlas1.label.nii.gz \
    --label2 /output/sub-01.Atlas2.label.nii.gz \
    --label3 /output/sub-01.Atlas3.label.nii.gz \
    --output /output/sub-01.fused.label.nii.gz \
    --method majority

# Or weighted fusion based on registration quality
```

### Population Atlas Creation

```bash
# Create custom atlas from cohort
# Useful for specialized populations

# 1. Process all subjects with BrainSuite
# 2. Register all to common space
# 3. Average volumes and surfaces

# Average registered T1w images
averageVolumes \
    --input /output/sub-*/sub-*.svreg.T1.nii.gz \
    --output /atlas/population_T1.nii.gz

# Average surfaces
averageSurfaces \
    --input /output/sub-*/sub-*.registered.pial.dfs \
    --output /atlas/population.pial.dfs

# Use custom atlas for new subjects
svreg.sh /output/new-sub /atlas/population_T1
```

---

## Troubleshooting

### Common Issues

```bash
# Issue: BSE fails to remove all skull
# Solution: Adjust edge detection parameters
bse --edgeDetectionConstant 0.5 \  # More aggressive (default: 0.64)
    --diffusionIterations 5 \       # More smoothing
    -i input.nii.gz -o output.nii.gz

# Issue: Topology errors in surfaces
# Solution: TCA may need more iterations
tca --maxIterations 10000 \  # Default: 50000
    -i surface.dfs -o corrected.dfs

# Issue: SVReg takes too long
# Solution: Use faster parameters
svreg.sh --fast /output/sub-01 $BRAINSUITE_PATH/atlas/BCI-DNI_brain

# Issue: GUI crashes on large datasets
# Solution: Downsample image first
mri_convert --voxsize 1 1 1 \
    input.nii.gz downsampled.nii.gz
```

### Performance Optimization

```bash
# Speed up processing:

# 1. Skip optional steps
# Only generate surfaces needed for analysis

# 2. Use lower resolution
# Downsample to 1mm isotropic if higher res

# 3. Parallel processing
# Run multiple subjects in parallel
for i in {1..4}; do
    bse.sh sub-0${i}_T1w.nii.gz &
done
wait

# 4. Use BIDS App with parallel option
docker run ... --n_cpus 4

# Typical processing times:
# CSE: 30-60 min
# SVReg: 60-120 min
# BDP: 120-240 min
```

---

## Related Tools and Integration

**Surface Analysis:**
- **FreeSurfer** (Batch 1): Alternative pipeline, interoperable
- **CIVET** (Batch 29): MNI pipeline for comparison
- **Mindboggle** (Batch 29): Shape analysis and labeling

**Diffusion MRI:**
- **MRtrix3** (Batch 1): Advanced tractography
- **DSI Studio** (Batch 10): Diffusion analysis
- **QSIPrep** (Batch 6): Preprocessing pipeline

**Statistics:**
- **SurfStat** (Batch 24): Surface-based statistics

---

## References

- Shattuck, D. W., & Leahy, R. M. (2002). BrainSuite: an automated cortical surface identification tool. *Medical Image Analysis*, 6(2), 129-142.
- Joshi, A. A., et al. (2007). Surface-constrained volumetric brain registration using harmonic mappings. *IEEE Transactions on Medical Imaging*, 26(12), 1657-1669.
- Pantazis, D., et al. (2010). Comparison of landmark-based and automatic methods for cortical surface registration. *NeuroImage*, 49(3), 2479-2493.
- Choi, Y. Y., et al. (2020). The Human Connectome Project for Disordered Emotional States: Protocol and rationale for a research domain criteria study of brain connectivity in young adults. *NeuroImage*, 214, 116715.

**Official Website:** http://brainsuite.org/
**Documentation:** http://brainsuite.org/documentation/
**Forums:** http://forums.brainsuite.org/
**BIDS App:** https://github.com/bids-apps/BrainSuite
**Tutorials:** http://brainsuite.org/tutorials/
