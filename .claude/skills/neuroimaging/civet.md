# CIVET - Cortical Imaging VElocimetry Toolkit

## Overview

**CIVET** (Cortical Imaging VElocimetry Toolkit) is a fully automated structural MRI processing pipeline developed at the Montreal Neurological Institute (MNI) for comprehensive cortical surface extraction, cortical thickness measurement, and morphometric analysis. CIVET provides a standardized, reproducible framework for processing large cohorts with emphasis on multi-site studies, quality control, and integration with statistical analysis tools.

CIVET performs brain extraction, tissue classification, cortical surface generation (white and pial surfaces), non-linear registration to MNI stereotaxic space, and cortical thickness computation at each vertex. The pipeline is optimized for robustness across diverse datasets and is widely used in international consortia (ENIGMA, CCNA, PREVENT-AD) for harmonized cortical morphometry analysis.

**Key Features:**
- Fully automated T1w MRI processing pipeline
- Cortical surface extraction (inner/white, outer/pial, mid-cortical)
- Vertex-wise cortical thickness measurement (tlink method)
- Non-linear registration to MNI-ICBM152 template
- Tissue classification (gray matter, white matter, CSF)
- Lobar and regional parcellation (AAL, DKT)
- Surface-based smoothing and blurring
- Quality control image generation
- Longitudinal processing support
- Batch processing for large cohorts
- Integration with RMINC for statistical analysis in R
- Pediatric and aging-specific templates
- Extensive output metrics (CSV tables)
- ENIGMA protocol compatibility

**Primary Use Cases:**
- Large-scale cortical morphometry studies
- Multi-site consortium analyses (ENIGMA)
- Developmental neuroscience (cortical maturation)
- Aging and neurodegeneration research
- Clinical cortical pathology studies
- Method comparison and validation
- Population neuroscience studies
- Longitudinal cortical change analysis

**Official Documentation:** https://www.bic.mni.mcgill.ca/ServicesSoftware/CIVET

---

## Installation and Setup

### Container Installation (Recommended)

```bash
# CIVET is available as a Singularity container
# Download from MNI or institutional repository

# Example: Pull CIVET container
singularity pull civet-2.1.1.sif docker://mcin/civet:2.1.1

# Verify installation
singularity exec civet-2.1.1.sif CIVET_Processing_Pipeline -version

# Expected output: CIVET-2.1.1
```

### Native Installation

```bash
# CIVET requires registration (free for academic use)
# Register at: https://www.bic.mni.mcgill.ca/ServicesSoftware/RegisterCIVET

# After receiving download link, extract:
tar -xzvf CIVET-2.1.1-Linux-x86_64.tar.gz

# Add to PATH
export CIVET_HOME=/opt/CIVET-2.1.1
export PATH=$CIVET_HOME/bin:$PATH
export PERL5LIB=$CIVET_HOME/perl:$PERL5LIB

# Add to ~/.bashrc for persistence
echo "export CIVET_HOME=/opt/CIVET-2.1.1" >> ~/.bashrc
echo "export PATH=\$CIVET_HOME/bin:\$PATH" >> ~/.bashrc
echo "export PERL5LIB=\$CIVET_HOME/perl:\$PERL5LIB" >> ~/.bashrc

# Verify installation
CIVET_Processing_Pipeline -version
```

### Directory Structure Setup

```bash
# Create project directory structure
mkdir -p civet_project/{input,output,logs}

# Input directory for T1w images
# output directory for CIVET results
# logs directory for processing logs

cd civet_project
```

---

## Basic Pipeline Execution

### Run CIVET on Single Subject

```bash
# Basic CIVET command
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix sub-01 \
    -N3-distance 50 \
    sub-01_T1w.nii.gz \
    -run

# Parameters:
# -sourcedir: Directory containing input T1w image
# -targetdir: Output directory for results
# -prefix: Subject identifier
# -N3-distance: N3 bias correction spacing (mm)
# -run: Execute the pipeline (omit for dry-run)

# Processing takes ~6-12 hours per subject
# Monitor progress in output/logs/
```

### Basic Pipeline with Common Options

```bash
# Recommended options for standard processing
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix sub-01 \
    -N3-distance 50 \
    -lsq6 \
    -lsq12 \
    -no-surfaces \
    -thickness tlaplace 30 \
    sub-01_T1w.nii.gz \
    -run

# Options explained:
# -lsq6: Use 6-parameter linear registration
# -lsq12: Use 12-parameter linear registration
# -no-surfaces: Skip intermediate surface files (saves space)
# -thickness tlaplace 30: Laplacian thickness with 30mm kernel
```

### Check Pipeline Status

```bash
# Monitor processing
tail -f output/logs/sub-01_*.log

# Check for completion
ls output/sub-01/surfaces/

# Key output files indicate successful completion:
# sub-01_gray_surface_left_81920.obj  (white matter surface)
# sub-01_gray_surface_right_81920.obj
# sub-01_white_surface_left_81920.obj (pial surface)
# sub-01_white_surface_right_81920.obj
```

---

## Pipeline Configuration

### Template Selection

```bash
# Default: ICBM152 adult template (1mm)
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix sub-01 \
    -template 1.00 \
    sub-01_T1w.nii.gz \
    -run

# Pediatric template (for ages 4.5-18)
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix child-01 \
    -template 0.50 \
    -lsq6 -lsq12 \
    child-01_T1w.nii.gz \
    -run

# Template options:
# 1.00: Adult template (default)
# 0.50: Pediatric template
```

### Processing Options

```bash
# High-quality processing (slower)
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix sub-01 \
    -N3-distance 25 \
    -thickness tlaplace 10 \
    -resample-surfaces \
    sub-01_T1w.nii.gz \
    -run

# Fast processing (less accurate)
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix sub-01 \
    -N3-distance 100 \
    -no-surfaces \
    -no-VBM \
    sub-01_T1w.nii.gz \
    -run

# Options:
# -N3-distance: Smaller = better correction, slower
# -resample-surfaces: Generate surfaces at multiple resolutions
# -no-VBM: Skip VBM outputs (faster)
```

### Mask and Lesion Handling

```bash
# Provide brain mask
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix sub-01 \
    -mask input/sub-01_mask.mnc \
    sub-01_T1w.nii.gz \
    -run

# Lesion masking (exclude from processing)
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix patient-01 \
    -mask-cerebellum \
    -lesion input/patient-01_lesion.mnc \
    patient-01_T1w.nii.gz \
    -run
```

---

## Output Files and Interpretation

### Output Directory Structure

```bash
# CIVET creates organized output structure
output/
└── sub-01/
    ├── classify/          # Tissue classification
    ├── final/            # Final processed volumes
    ├── logs/             # Processing logs
    ├── mask/             # Brain masks
    ├── native/           # Native space outputs
    ├── segment/          # Segmentation
    ├── surfaces/         # Cortical surfaces (main outputs)
    ├── temp/             # Temporary files
    ├── thickness/        # Thickness maps
    ├── transforms/       # Registration transforms
    └── verify/           # QC images
```

### Key Output Files

```bash
# Cortical surfaces (surfaces/)
sub-01_gray_surface_left_81920.obj   # White matter surface
sub-01_white_surface_right_81920.obj # Pial surface
sub-01_mid_surface_left_40962.obj    # Mid-cortical surface

# Thickness maps (thickness/)
sub-01_native_rms_rsl_tlink_30mm_left.txt   # Thickness values per vertex
sub-01_native_rms_rsl_tlink_30mm_right.txt

# Classified volumes (classify/)
sub-01_cls_volumes.dat  # Volume of GM, WM, CSF

# Quality control (verify/)
sub-01_*.jpg  # QC images for visual inspection
```

### Extract Thickness Values

```bash
# Thickness files are text: one value per vertex
# 81920 vertices for full resolution
# 40962 vertices for resampled surfaces

# View thickness statistics
cat output/sub-01/thickness/sub-01_native_rms_rsl_tlink_30mm_left.txt | \
    awk '{sum+=$1; sumsq+=$1*$1} END {
        print "Mean:", sum/NR;
        print "Std:", sqrt(sumsq/NR - (sum/NR)^2)
    }'

# Extract thickness for specific vertex
awk 'NR==5000 {print $1}' output/sub-01/thickness/sub-01_native_rms_rsl_tlink_30mm_left.txt
```

---

## Quality Control

### Visual QC Protocol

```bash
# CIVET generates QC images in verify/ directory
cd output/sub-01/verify/

# View key QC images:
# 1. Brain extraction
display sub-01_skull_mask_native.jpg

# 2. Tissue classification
display sub-01_classify.jpg

# 3. White matter surface on T1
display sub-01_gray_surface_native.jpg

# 4. Pial surface on T1
display sub-01_white_surface_native.jpg

# 5. Registration to template
display sub-01_stx_t1.jpg

# Rate quality: Pass / Fail / Warning
# Document issues for failed cases
```

### Automated QC Metrics

```bash
# CIVET provides quantitative QC metrics
# Extract from classify volumes file

# Total brain volume
grep "Total Brain Volume" output/sub-01/classify/sub-01_cls_volumes.dat

# Gray matter volume
grep "Gray Matter Volume" output/sub-01/classify/sub-01_cls_volumes.dat

# Check for outliers
# Typical ranges (adults):
# Total brain: 1000-1600 cm³
# Gray matter: 500-800 cm³
# Thickness: 2.0-4.0 mm (cortical average)

# Extract mean thickness
cat output/sub-01/thickness/sub-01_native_rms_rsl_tlink_30mm_left.txt | \
    awk '{sum+=$1} END {print "Mean thickness:", sum/NR, "mm"}'
```

### Common QC Failures

```bash
# 1. Incomplete skull stripping
# Symptom: Non-brain tissue in surfaces
# Solution: Provide better mask or adjust parameters

# 2. Topology errors in surfaces
# Symptom: Holes or handles in cortical surface
# Check surface topology (should be 2-genus sphere)
check_topology.pl output/sub-01/surfaces/sub-01_gray_surface_left_81920.obj

# 3. Poor registration
# Symptom: Misaligned to template
# Solution: Check input quality, try different registration

# 4. Segmentation errors
# Symptom: Incorrect GM/WM boundary
# View classification: output/sub-01/verify/sub-01_classify.jpg
```

---

## Surface-Based Analysis

### Cortical Thickness Analysis

```bash
# Thickness values are stored per vertex
# Load in R using RMINC package

# Example R script:
cat > analyze_thickness.R <<'EOF'
library(RMINC)

# Load thickness data
thickness_left <- read.table(
  "output/sub-01/thickness/sub-01_native_rms_rsl_tlink_30mm_left.txt"
)

# Basic statistics
mean_thickness <- mean(thickness_left$V1, na.rm=TRUE)
sd_thickness <- sd(thickness_left$V1, na.rm=TRUE)

cat(sprintf("Mean thickness: %.3f ± %.3f mm\n",
    mean_thickness, sd_thickness))

# Histogram
hist(thickness_left$V1, breaks=50,
     main="Cortical Thickness Distribution",
     xlab="Thickness (mm)", col="skyblue")
EOF

Rscript analyze_thickness.R
```

### ROI-Based Measurements

```bash
# CIVET provides AAL atlas parcellation
# Extract mean thickness per ROI

# AAL ROI labels in native space
AAL_FILE="output/sub-01/segment/sub-01_AAL_native.mnc"

# Thickness map in native space
THICK_FILE="output/sub-01/thickness/sub-01_native_rms_tlink_30mm.mnc"

# Extract stats per ROI (requires MINC tools)
mincstats -mask $AAL_FILE -mask_binvalue 1 $THICK_FILE  # Precentral_L
mincstats -mask $AAL_FILE -mask_binvalue 2 $THICK_FILE  # Precentral_R

# Or use RMINC for batch ROI extraction:
cat > roi_thickness.R <<'EOF'
library(RMINC)

# Load volumes
aal <- mincGetVolume("output/sub-01/segment/sub-01_AAL_native.mnc")
thickness <- mincGetVolume("output/sub-01/thickness/sub-01_native_rms_tlink_30mm.mnc")

# AAL labels (1-90 for cortical regions)
roi_stats <- data.frame(ROI=1:90, Mean_Thickness=NA)
for(roi in 1:90) {
  roi_mask <- (aal == roi)
  roi_stats$Mean_Thickness[roi] <- mean(thickness[roi_mask], na.rm=TRUE)
}

write.csv(roi_stats, "sub-01_roi_thickness.csv", row.names=FALSE)
EOF

Rscript roi_thickness.R
```

### Surface Area and Volume

```bash
# Extract surface area from surface file
# Surface files are .obj format

# Count vertices and faces
grep "^V" output/sub-01/surfaces/sub-01_gray_surface_left_81920.obj | wc -l
# Output: 81920 vertices

# Compute total surface area (requires CIVET utilities)
average_surfaces \
    output/sub-01/surfaces/sub-01_gray_surface_left_81920.obj \
    output/sub-01/surfaces/sub-01_white_surface_left_81920.obj \
    output/sub-01/surfaces/sub-01_mid_surface_left_81920.obj

# Extract cortical volumes from classification
grep "Cortical Gray Matter" output/sub-01/classify/sub-01_cls_volumes.dat
```

---

## Batch Processing

### Process Multiple Subjects

```bash
# Create subject list
ls input/*.nii.gz | sed 's|input/||; s|_T1w.nii.gz||' > subjects.txt

# Process all subjects in loop
while read subject; do
    echo "Processing $subject..."

    CIVET_Processing_Pipeline \
        -sourcedir input \
        -targetdir output \
        -prefix $subject \
        -N3-distance 50 \
        -lsq12 \
        ${subject}_T1w.nii.gz \
        -run

done < subjects.txt

# Monitor progress
tail -f output/logs/*_*.log
```

### Parallel Execution

```bash
# Process multiple subjects in parallel
# Use GNU parallel or HPC job array

# GNU parallel example (4 jobs at once)
cat subjects.txt | parallel -j 4 \
    "CIVET_Processing_Pipeline \
        -sourcedir input \
        -targetdir output \
        -prefix {} \
        -N3-distance 50 \
        {}_T1w.nii.gz \
        -run"

# Check for failures
for subject in $(cat subjects.txt); do
    if [ ! -f output/$subject/surfaces/${subject}_gray_surface_left_81920.obj ]; then
        echo "FAILED: $subject"
    fi
done
```

### HPC Job Submission

```bash
# SLURM array job example
cat > run_civet_array.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=civet
#SBATCH --array=1-100
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Get subject from list
subject=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subjects.txt)

# Run CIVET
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix $subject \
    -N3-distance 50 \
    -lsq12 \
    ${subject}_T1w.nii.gz \
    -run
EOF

# Submit job array
sbatch run_civet_array.sh
```

---

## Statistical Analysis with RMINC

### Load Data in R

```R
library(RMINC)

# Load subject list and covariates
subjects <- read.csv("subjects.csv")  # Subject ID, Age, Sex, Group

# Create paths to thickness files
subjects$thickness_left <- paste0(
    "output/", subjects$ID,
    "/thickness/", subjects$ID, "_native_rms_rsl_tlink_30mm_left.txt"
)

# Load thickness data for all subjects
thickness_matrix_left <- matrix(NA,
    nrow=nrow(subjects),
    ncol=81920  # vertices
)

for(i in 1:nrow(subjects)) {
    thickness_matrix_left[i,] <-
        read.table(subjects$thickness_left[i])$V1
}
```

### Vertex-Wise Statistics

```R
# T-test at each vertex: Patients vs Controls
# Vertex-wise comparison

p_values <- numeric(81920)
t_stats <- numeric(81920)

for(v in 1:81920) {
    vertex_data <- thickness_matrix_left[, v]
    test <- t.test(
        vertex_data[subjects$Group == "Patient"],
        vertex_data[subjects$Group == "Control"]
    )
    p_values[v] <- test$p.value
    t_stats[v] <- test$statistic
}

# FDR correction
p_fdr <- p.adjust(p_values, method="fdr")

# Count significant vertices
n_sig <- sum(p_fdr < 0.05)
cat(sprintf("%d vertices significant (FDR < 0.05)\n", n_sig))

# Save results
write.table(
    data.frame(Vertex=1:81920, T=t_stats, P=p_values, P_FDR=p_fdr),
    "group_comparison_left.txt",
    row.names=FALSE
)
```

### Linear Models with Covariates

```R
# Control for age and sex
library(RMINC)

# Fit model at each vertex
coef_age <- numeric(81920)
coef_group <- numeric(81920)
p_group <- numeric(81920)

for(v in 1:81920) {
    vertex_data <- thickness_matrix_left[, v]

    model <- lm(vertex_data ~ Age + Sex + Group, data=subjects)

    coef_age[v] <- coef(model)["Age"]
    coef_group[v] <- coef(model)["GroupPatient"]
    p_group[v] <- summary(model)$coefficients["GroupPatient", "Pr(>|t|)"]
}

# Visualize age effects
hist(coef_age, breaks=50,
     main="Age Effect on Cortical Thickness",
     xlab="Coefficient (mm/year)")
```

---

## Longitudinal Analysis

### Multi-Timepoint Processing

```bash
# Process timepoint 1
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix sub-01_tp1 \
    sub-01_tp1_T1w.nii.gz \
    -run

# Process timepoint 2
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix sub-01_tp2 \
    sub-01_tp2_T1w.nii.gz \
    -run
```

### Within-Subject Registration

```bash
# Register timepoint 2 to timepoint 1
# Use CIVET's surface registration

# Extract surfaces
TP1_SURF="output/sub-01_tp1/surfaces/sub-01_tp1_mid_surface_left_40962.obj"
TP2_SURF="output/sub-01_tp2/surfaces/sub-01_tp2_mid_surface_left_40962.obj"

# Register surfaces (creates correspondence)
surface_register \
    -source $TP2_SURF \
    -target $TP1_SURF \
    -output sub-01_tp2_to_tp1_left.xfm

# Now thickness values at same vertex represent same anatomical location
```

### Compute Longitudinal Change

```R
# Load thickness at both timepoints
tp1_thick <- read.table("output/sub-01_tp1/thickness/sub-01_tp1_native_rms_rsl_tlink_30mm_left.txt")$V1
tp2_thick <- read.table("output/sub-01_tp2/thickness/sub-01_tp2_native_rms_rsl_tlink_30mm_left.txt")$V1

# Compute change (after registration)
thickness_change <- tp2_thick - tp1_thick

# Time interval
time_interval <- 2.5  # years

# Annualized atrophy rate
atrophy_rate <- thickness_change / time_interval

# Mean atrophy
mean(atrophy_rate, na.rm=TRUE)

# Regions with significant atrophy
significant_atrophy <- which(thickness_change < -0.1)  # >0.1mm loss
length(significant_atrophy)
```

---

## ENIGMA Protocol Compatibility

### ENIGMA-Cortical Protocol

```bash
# CIVET is compatible with ENIGMA cortical protocol
# http://enigma.ini.usc.edu/protocols/imaging-protocols/

# Standard ENIGMA CIVET processing:
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix $subject \
    -template 1.00 \
    -lsq12 \
    -no-surfaces \
    -thickness tlaplace 30 \
    ${subject}_T1w.nii.gz \
    -run

# Extract ENIGMA ROI statistics
# CIVET outputs include DKT parcellation
# Compatible with ENIGMA analysis scripts
```

### Extract ENIGMA ROIs

```bash
# ENIGMA cortical ROIs based on DKT atlas
# CIVET provides these automatically

# Example: Extract mean thickness for ENIGMA ROIs
cat > extract_enigma_rois.R <<'EOF'
library(RMINC)

# Load DKT parcellation
dkt <- mincGetVolume("output/sub-01/segment/sub-01_DKT_native.mnc")

# Load thickness
thickness <- mincGetVolume("output/sub-01/thickness/sub-01_native_rms_tlink_30mm.mnc")

# ENIGMA ROI definitions (DKT labels)
enigma_rois <- data.frame(
    ROI = c("bankssts_left", "caudalmiddlefrontal_left", ...),
    Label = c(1, 2, ...)
)

# Extract mean thickness per ROI
enigma_stats <- data.frame(ROI=character(), Mean_Thickness=numeric())
for(i in 1:nrow(enigma_rois)) {
    mask <- (dkt == enigma_rois$Label[i])
    mean_thick <- mean(thickness[mask], na.rm=TRUE)
    enigma_stats <- rbind(enigma_stats,
        data.frame(ROI=enigma_rois$ROI[i], Mean_Thickness=mean_thick))
}

write.csv(enigma_stats, "sub-01_enigma_cortical.csv", row.names=FALSE)
EOF

Rscript extract_enigma_rois.R
```

---

## Integration and Comparison

### Compare with FreeSurfer

```bash
# Compare CIVET and FreeSurfer thickness measurements
# Both provide vertex-wise thickness

# FreeSurfer thickness (resampled to fsaverage)
FS_THICK="/freesurfer/sub-01/surf/lh.thickness"

# CIVET thickness
CIVET_THICK="output/sub-01/thickness/sub-01_native_rms_rsl_tlink_30mm_left.txt"

# Compare in R
cat > compare_civet_freesurfer.R <<'EOF'
# Load data
civet <- read.table("output/sub-01/thickness/sub-01_native_rms_rsl_tlink_30mm_left.txt")$V1
# FreeSurfer data loading would require freesurferformats package

# Correlation (requires vertex correspondence)
cor(civet, freesurfer, use="complete.obs")

# Bland-Altman plot
difference <- civet - freesurfer
average <- (civet + freesurfer) / 2

plot(average, difference,
     xlab="Average Thickness (mm)",
     ylab="Difference (CIVET - FreeSurfer)",
     main="CIVET vs FreeSurfer Comparison")
abline(h=0, col="red")
EOF
```

### Export to Surface Viewers

```bash
# CIVET surfaces are in .obj format
# Can be viewed in:

# 1. Display (from CIVET)
Display output/sub-01/surfaces/sub-01_gray_surface_left_81920.obj

# 2. Convert to GIFTI for other viewers
obj2gii \
    output/sub-01/surfaces/sub-01_gray_surface_left_81920.obj \
    sub-01_white_left.surf.gii

# 3. View in FreeView (FreeSurfer viewer)
# Requires conversion to FreeSurfer format

# 4. Paraview (for advanced visualization)
# Load .obj directly
```

---

## Advanced Features

### Custom Template Creation

```bash
# Create population-specific template from cohort
# Useful for specialized populations (pediatric, disease)

# 1. Process all subjects with CIVET
# 2. Average normalized brains

# Average T1w images in stereotaxic space
mincaverage \
    output/sub-*/final/sub-*_t1_final.mnc \
    custom_template_t1.mnc

# Average surfaces
average_surfaces \
    output/sub-*/surfaces/sub-*_gray_surface_left_81920.obj \
    custom_template_gray_left.obj

# Use custom template for new subjects
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix new-sub \
    -template custom_template_t1.mnc \
    new-sub_T1w.nii.gz \
    -run
```

### Lesion Masking

```bash
# Exclude lesions from surface extraction
# Important for stroke, tumor, MS patients

# Create lesion mask (same space as T1)
# Lesion mask: 1=lesion, 0=normal

# Run CIVET with lesion mask
CIVET_Processing_Pipeline \
    -sourcedir input \
    -targetdir output \
    -prefix patient-01 \
    -lesion input/patient-01_lesion.mnc \
    patient-01_T1w.nii.gz \
    -run

# CIVET will:
# - Exclude lesion from tissue classification
# - Prevent lesion from affecting surfaces
# - Properly handle boundary regions
```

---

## Troubleshooting

### Common Errors

```bash
# Error: "Cannot find input file"
# Solution: Check file paths, use absolute paths

# Error: "Insufficient memory"
# Solution: Increase memory allocation (8GB minimum recommended)

# Error: "Template not found"
# Solution: Verify CIVET installation includes templates
ls $CIVET_HOME/models/

# Error: "Pipeline stopped at stage X"
# Solution: Check logs for specific error
tail -50 output/sub-01/logs/sub-01_process.log
```

### Performance Optimization

```bash
# Speed up processing:

# 1. Use coarser N3 correction
-N3-distance 100  # vs default 50

# 2. Skip optional outputs
-no-surfaces  # Skip intermediate surfaces
-no-VBM       # Skip VBM outputs

# 3. Use lower resolution template (faster registration)
-template 2.00  # vs 1.00

# 4. Parallel subject processing
# Run multiple subjects simultaneously on different cores

# Typical processing times:
# Standard: 8-12 hours
# Fast: 4-6 hours
# High-quality: 16-24 hours
```

### Failed QC Cases

```bash
# If subject fails QC:

# 1. Review verify images
cd output/sub-01/verify/
display *.jpg

# 2. Check for input issues
# - Motion artifacts
# - Low SNR
# - Acquisition problems

# 3. Try different parameters
# - Adjust N3-distance
# - Use different template
# - Provide manual mask

# 4. Document failure
echo "sub-01: Failed - poor skull stripping" >> failed_subjects.txt

# 5. Consider manual intervention or exclusion
```

---

## Output Summary and Reporting

### Generate Summary Statistics

```bash
# Create summary table for all subjects
cat > summarize_civet.R <<'EOF'
library(RMINC)

subjects <- read.csv("subjects.csv")
summary_stats <- data.frame(
    Subject = subjects$ID,
    Total_Brain_Volume = NA,
    GM_Volume = NA,
    WM_Volume = NA,
    Mean_Thickness_Left = NA,
    Mean_Thickness_Right = NA
)

for(i in 1:nrow(subjects)) {
    subject <- subjects$ID[i]

    # Read volumes
    vol_file <- paste0("output/", subject, "/classify/",
                       subject, "_cls_volumes.dat")
    volumes <- read.table(vol_file, header=FALSE)
    summary_stats$Total_Brain_Volume[i] <- volumes[1, 2]
    summary_stats$GM_Volume[i] <- volumes[2, 2]
    summary_stats$WM_Volume[i] <- volumes[3, 2]

    # Read thickness
    thick_l <- paste0("output/", subject, "/thickness/",
                      subject, "_native_rms_rsl_tlink_30mm_left.txt")
    thick_r <- paste0("output/", subject, "/thickness/",
                      subject, "_native_rms_rsl_tlink_30mm_right.txt")

    summary_stats$Mean_Thickness_Left[i] <-
        mean(read.table(thick_l)$V1, na.rm=TRUE)
    summary_stats$Mean_Thickness_Right[i] <-
        mean(read.table(thick_r)$V1, na.rm=TRUE)
}

write.csv(summary_stats, "civet_summary.csv", row.names=FALSE)
print(summary(summary_stats))
EOF

Rscript summarize_civet.R
```

---

## Related Tools and Integration

**Surface Analysis:**
- **FreeSurfer** (Batch 1): Alternative surface pipeline
- **BrainSuite** (Batch 29): Interactive surface analysis
- **Mindboggle** (Batch 29): Advanced shape features

**Statistics:**
- **SurfStat** (Batch 24): Surface-based statistics
- **RMINC**: R interface for MINC/CIVET analysis

**Templates:**
- **TemplateFlow** (Batch 28): Standard brain templates

**Preprocessing:**
- **fMRIPrep** (Batch 5): Anatomical preprocessing alternative

---

## References

- Ad-Dab'bagh, Y., et al. (2006). The CIVET image-processing environment: a fully automated comprehensive pipeline for anatomical neuroimaging research. *Proceedings of the 12th Annual Meeting of the Organization for Human Brain Mapping*, 2266.
- Kim, J. S., et al. (2005). Automated 3-D extraction and evaluation of the inner and outer cortical surfaces using a Laplacian map and partial volume effect classification. *NeuroImage*, 27(1), 210-221.
- Lerch, J. P., & Evans, A. C. (2005). Cortical thickness analysis examined through power analysis and a population simulation. *NeuroImage*, 24(1), 163-173.
- MacDonald, D., et al. (2000). Automated 3-D extraction of inner and outer surfaces of cerebral cortex from MRI. *NeuroImage*, 12(3), 340-356.

**Official Website:** https://www.bic.mni.mcgill.ca/ServicesSoftware/CIVET
**User Guide:** https://www.bic.mni.mcgill.ca/ServicesSoftware/CIVETDocumentation
**RMINC:** https://github.com/Mouse-Imaging-Centre/RMINC
**ENIGMA Protocols:** http://enigma.ini.usc.edu/protocols/imaging-protocols/
