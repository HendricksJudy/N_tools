# MRIQC (MRI Quality Control)

## Overview

MRIQC (MRI Quality Control) is an automated, no-reference image quality assessment tool for structural and functional MRI data. Developed by the NiPreps community (the same team behind fMRIPrep), MRIQC extracts over 60 image quality metrics (IQMs) from raw neuroimaging data without requiring a reference standard or gold standard comparison. It generates comprehensive visual HTML reports and provides objective quality metrics for data inclusion/exclusion decisions, making it essential for quality control in both small and large-scale neuroimaging studies.

**Website:** https://mriqc.readthedocs.io/
**GitHub:** https://github.com/nipreps/mriqc
**Platform:** Python (Docker/Singularity recommended)
**Language:** Python
**License:** Apache License 2.0

## Key Features

- Automated extraction of 60+ image quality metrics
- No-reference quality assessment (no ground truth needed)
- Structural MRI QC (T1w, T2w)
- Functional MRI QC (BOLD)
- Diffusion MRI support (experimental)
- BIDS-compatible input and output
- Individual subject HTML reports
- Group-level quality comparisons
- Machine learning classifier ratings
- Interactive visualizations (mosaic, carpet plots)
- CSV export of all metrics
- Containerized deployment (Docker/Singularity)
- HPC cluster support
- Parallel processing
- Integration with fMRIPrep and other pipelines

## Installation

### Docker Installation (Recommended)

```bash
# Pull latest MRIQC Docker image
docker pull nipreps/mriqc:latest

# Verify installation
docker run -it --rm nipreps/mriqc:latest --version
```

### Singularity Installation

```bash
# Build Singularity image from Docker
singularity build mriqc-latest.simg docker://nipreps/mriqc:latest

# Verify installation
singularity run mriqc-latest.simg --version
```

### Local Python Installation

```bash
# Create conda environment
conda create -n mriqc python=3.9
conda activate mriqc

# Install MRIQC
pip install mriqc

# Verify installation
mriqc --version
```

## Basic Usage

### Run MRIQC on BIDS Dataset

```bash
# Basic command structure
mriqc <bids_dir> <output_dir> participant --participant-label <subject_id>

# Example: Single subject T1w QC
docker run -it --rm \
  -v /data/my_study:/data:ro \
  -v /data/my_study/derivatives/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out participant \
  --participant-label sub-01

# Process all subjects
docker run -it --rm \
  -v /data/my_study:/data:ro \
  -v /data/my_study/derivatives/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out participant

# Generate group report
docker run -it --rm \
  -v /data/my_study:/data:ro \
  -v /data/my_study/derivatives/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out group
```

### Structural T1w Quality Control

```bash
# T1w-specific QC with custom settings
docker run -it --rm \
  -v /data/study:/data:ro \
  -v /data/study/derivatives/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out participant \
  --participant-label sub-01 sub-02 sub-03 \
  --modalities T1w \
  --nprocs 4 \
  --mem_gb 16
```

### Functional BOLD Quality Control

```bash
# BOLD fMRI QC
docker run -it --rm \
  -v /data/study:/data:ro \
  -v /data/study/derivatives/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out participant \
  --participant-label sub-01 \
  --modalities bold \
  --nprocs 8 \
  --mem_gb 32

# Multiple runs/sessions
docker run -it --rm \
  -v /data/study:/data:ro \
  -v /data/study/derivatives/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out participant \
  --modalities T1w bold \
  --session-id ses-01
```

## Image Quality Metrics

### Structural MRI Metrics

```python
# Load and analyze MRIQC JSON output
import json
import pandas as pd

# Load individual subject metrics
with open('sub-01_T1w.json', 'r') as f:
    metrics = json.load(f)

# Key structural metrics
print(f"SNR: {metrics['snr_total']:.2f}")  # Signal-to-noise ratio
print(f"CNR: {metrics['cnr']:.2f}")  # Contrast-to-noise ratio
print(f"FBER: {metrics['fber']:.2f}")  # Foreground-background energy ratio
print(f"EFC: {metrics['efc']:.4f}")  # Entropy focus criterion
print(f"FWHM: {metrics['fwhm_avg']:.2f} mm")  # Smoothness

# Artifact metrics
print(f"INU (Bias field): {metrics['inu_med']:.4f}")
print(f"Qi1 (Artifacts): {metrics['qi_1']:.4f}")
print(f"WM2MAX: {metrics['wm2max']:.4f}")
```

### Functional MRI Metrics

```python
# Load BOLD QC metrics
with open('sub-01_task-rest_bold.json', 'r') as f:
    bold_metrics = json.load(f)

# Temporal quality metrics
print(f"tSNR: {bold_metrics['tsnr']:.2f}")  # Temporal SNR
print(f"DVARS: {bold_metrics['dvars_std']:.4f}")  # Standardized DVARS
print(f"FD: {bold_metrics['fd_mean']:.4f} mm")  # Mean framewise displacement

# Spatial quality
print(f"FWHM (spatial): {bold_metrics['fwhm_avg']:.2f} mm")
print(f"Ghost-to-signal ratio: {bold_metrics['gsr_x']:.4f}")

# Temporal artifacts
print(f"Global correlation: {bold_metrics['gcor']:.4f}")
print(f"Outlier frames: {bold_metrics['aor']:.2f}%")
```

### Load Group Metrics

```python
# Load group CSV file
df = pd.read_csv('group_T1w.tsv', sep='\t')

# Display key metrics
print(df[['bids_name', 'snr_total', 'cnr', 'fber', 'efc']].head())

# Summary statistics
print(df[['snr_total', 'cnr', 'fber']].describe())
```

## Visual Reports

### Understanding Individual Reports

```python
# Individual report structure:
# 1. Summary section with key metrics
# 2. Mosaic view of anatomical images
# 3. Background/foreground separation
# 4. Noise and artifact visualizations
# 5. Metadata and acquisition parameters

# Access reports
# file:///<output_dir>/sub-01_T1w.html
```

### Group Report Interpretation

```python
# Group report includes:
# - Distribution plots for all metrics
# - Outlier detection
# - Interactive violin plots
# - Scatter plots of metric correlations

# Load and plot group distributions
import matplotlib.pyplot as plt
import seaborn as sns

# Load group metrics
df = pd.read_csv('group_T1w.tsv', sep='\t')

# Plot SNR distribution
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, y='snr_total')
plt.title('SNR Distribution Across Subjects')
plt.ylabel('SNR')
plt.axhline(y=10, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.savefig('snr_distribution.png')
```

### Carpet Plots for fMRI

```python
# Carpet plots show:
# - Temporal patterns across all voxels
# - Motion artifacts (visible as vertical bands)
# - Respiratory/cardiac artifacts
# - Global signal fluctuations

# Interpret carpet plot:
# - Vertical lines = motion spikes
# - Horizontal bands = spatial artifacts
# - Smooth gradual changes = physiological noise
```

## Quality Assessment

### Identify Problematic Scans

```python
# Define quality thresholds
thresholds = {
    'snr_total': 10,      # Minimum SNR
    'cnr': 3,             # Minimum CNR
    'fber': 1000,         # Minimum FBER
    'efc': 0.5,           # Maximum EFC (lower is better)
    'qi_1': 0.00001,      # Maximum Qi1 (lower is better)
    'inu_med': 0.8        # Maximum INU (lower is better)
}

# Flag problematic subjects
df = pd.read_csv('group_T1w.tsv', sep='\t')

df['qc_fail'] = (
    (df['snr_total'] < thresholds['snr_total']) |
    (df['cnr'] < thresholds['cnr']) |
    (df['fber'] < thresholds['fber']) |
    (df['efc'] > thresholds['efc']) |
    (df['qi_1'] > thresholds['qi_1']) |
    (df['inu_med'] > thresholds['inu_med'])
)

# List failed subjects
failed = df[df['qc_fail']]
print(f"Failed QC: {len(failed)}/{len(df)} subjects")
print(failed[['bids_name', 'snr_total', 'cnr', 'efc']])

# Save exclusion list
failed['bids_name'].to_csv('exclude_subjects.txt', index=False, header=False)
```

### Machine Learning Classifier

```python
# MRIQC includes ML classifier for T1w quality
# Classifier predicts: accept, doubtful, exclude

# Load predictions from MRIQC output
with open('sub-01_T1w.json', 'r') as f:
    metrics = json.load(f)

# Get classifier prediction
if 'pred_qa' in metrics:
    prediction = metrics['pred_qa']
    print(f"Classifier rating: {prediction}")

# Aggregate classifier predictions
df = pd.read_csv('group_T1w.tsv', sep='\t')

if 'pred_qa' in df.columns:
    print(df['pred_qa'].value_counts())
```

### Manual Rating Interface

```bash
# MRIQC provides web interface for manual rating
# Start rating server
mriqc_clf --load-classifier <classifier.pkl> \
  --rate <group_T1w.tsv>

# Opens web interface at http://localhost:8000
# Manually rate images as: accept, doubtful, exclude
# Saves ratings to database for classifier training
```

## Functional MRI Specific QC

### Motion Assessment

```python
# Load BOLD metrics
df_bold = pd.read_csv('group_bold.tsv', sep='\t')

# Motion quality thresholds
fd_threshold = 0.5  # mm
dvars_threshold = 1.5  # standardized units

# Flag high motion scans
df_bold['high_motion'] = (
    (df_bold['fd_mean'] > fd_threshold) |
    (df_bold['dvars_std'] > dvars_threshold)
)

print(f"High motion: {df_bold['high_motion'].sum()}/{len(df_bold)} runs")

# Plot motion distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df_bold['fd_mean'], bins=30)
axes[0].axvline(fd_threshold, color='r', linestyle='--')
axes[0].set_xlabel('Mean FD (mm)')
axes[0].set_title('Framewise Displacement Distribution')

axes[1].hist(df_bold['dvars_std'], bins=30)
axes[1].axvline(dvars_threshold, color='r', linestyle='--')
axes[1].set_xlabel('DVARS (standardized)')
axes[1].set_title('DVARS Distribution')

plt.tight_layout()
plt.savefig('motion_qc.png')
```

### Temporal Quality

```python
# Temporal SNR assessment
tsnr_threshold = 40  # Typical minimum for task fMRI

df_bold['low_tsnr'] = df_bold['tsnr'] < tsnr_threshold

print(f"Low tSNR: {df_bold['low_tsnr'].sum()}/{len(df_bold)} runs")

# Global correlation (GCOR)
# High GCOR (> 0.3) may indicate motion or physiological artifacts
gcor_threshold = 0.3

df_bold['high_gcor'] = df_bold['gcor'] > gcor_threshold
print(f"High GCOR: {df_bold['high_gcor'].sum()}/{len(df_bold)} runs")
```

### Artifact Detection

```python
# Ghost-to-signal ratio (GSR)
# Detects EPI ghosting artifacts
gsr_threshold = 0.03

df_bold['ghosting'] = df_bold['gsr_x'] > gsr_threshold

# Spike detection (AOR - AFNI Outlier Ratio)
# Percentage of outlier timepoints
aor_threshold = 5  # percent

df_bold['spikes'] = df_bold['aor'] > aor_threshold

# Combined artifact flag
df_bold['artifact_flag'] = (
    df_bold['ghosting'] |
    df_bold['spikes'] |
    df_bold['high_motion']
)

print(f"Artifacts detected: {df_bold['artifact_flag'].sum()}/{len(df_bold)} runs")
```

## Group-Level Analysis

### Compare Quality Across Sites

```python
# Multi-site quality comparison
# Assumes 'site' column in metadata

# Load metadata and merge with QC metrics
metadata = pd.read_csv('participants.tsv', sep='\t')
df_qc = pd.read_csv('group_T1w.tsv', sep='\t')

# Extract subject ID from bids_name
df_qc['participant_id'] = df_qc['bids_name'].str.extract(r'(sub-[^_]+)')

# Merge
df_merged = df_qc.merge(metadata, on='participant_id')

# Compare sites
site_comparison = df_merged.groupby('site')[['snr_total', 'cnr', 'fber']].agg(['mean', 'std'])
print(site_comparison)

# Visualize site differences
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(['snr_total', 'cnr', 'fber']):
    sns.boxplot(data=df_merged, x='site', y=metric, ax=axes[i])
    axes[i].set_title(f'{metric} by Site')
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.savefig('site_qc_comparison.png')
```

### Outlier Detection

```python
# Multivariate outlier detection using Mahalanobis distance
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# Select key metrics
metrics = ['snr_total', 'cnr', 'fber', 'efc', 'fwhm_avg']
X = df[metrics].dropna()

# Compute covariance matrix
cov = np.cov(X.T)
cov_inv = np.linalg.inv(cov)

# Compute mean
mean = X.mean().values

# Mahalanobis distance for each subject
mahal_dist = X.apply(lambda row: mahalanobis(row, mean, cov_inv), axis=1)

# Chi-square threshold (p < 0.01)
threshold = chi2.ppf(0.99, df=len(metrics))

# Flag outliers
outliers = mahal_dist > threshold

print(f"Outliers detected: {outliers.sum()}/{len(outliers)}")
print(df.loc[outliers, ['bids_name'] + metrics])
```

### Quality Distributions

```python
# Plot distributions of key metrics
metrics_to_plot = ['snr_total', 'cnr', 'fber', 'efc', 'fwhm_avg', 'inu_med']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, metric in enumerate(metrics_to_plot):
    df[metric].hist(bins=30, ax=axes[i])
    axes[i].set_title(metric)
    axes[i].set_xlabel(metric)
    axes[i].set_ylabel('Count')

    # Add mean line
    axes[i].axvline(df[metric].mean(), color='r', linestyle='--',
                    label=f'Mean: {df[metric].mean():.2f}')
    axes[i].legend()

plt.tight_layout()
plt.savefig('metric_distributions.png')
```

## Integration with Pipelines

### Pre-fMRIPrep Quality Check

```bash
# Workflow: MRIQC → Review → fMRIPrep

# Step 1: Run MRIQC on raw data
docker run -it --rm \
  -v /data/bids:/data:ro \
  -v /data/derivatives/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out participant

# Step 2: Run group analysis
docker run -it --rm \
  -v /data/bids:/data:ro \
  -v /data/derivatives/mriqc:/out \
  nipreps/mriqc:latest \
  /data /out group

# Step 3: Review reports and identify exclusions
# (Use Python scripts above)

# Step 4: Run fMRIPrep only on passing subjects
for sub in $(cat passing_subjects.txt); do
  docker run -it --rm \
    -v /data/bids:/data:ro \
    -v /data/derivatives/fmriprep:/out \
    nipreps/fmriprep:latest \
    /data /out participant \
    --participant-label $sub
done
```

### Quality Metrics as Regressors

```python
# Use MRIQC metrics as covariates in group analysis

# Load QC metrics
qc_metrics = pd.read_csv('group_T1w.tsv', sep='\t')
qc_metrics['participant_id'] = qc_metrics['bids_name'].str.extract(r'(sub-[^_]+)')

# Load analysis results (e.g., cortical thickness)
analysis_data = pd.read_csv('cortical_thickness.csv')

# Merge
merged = analysis_data.merge(qc_metrics[['participant_id', 'snr_total', 'cnr']],
                               on='participant_id')

# Include QC metrics in statistical model
from statsmodels.formula.api import ols

model = ols('thickness ~ group + age + sex + snr_total + cnr', data=merged).fit()
print(model.summary())
```

## Advanced Configuration

### Custom Settings File

```json
// mriqc_config.json
{
  "bids_dir": "/data",
  "output_dir": "/output",
  "analysis_level": "participant",
  "participant_label": ["01", "02", "03"],
  "session_id": ["ses-01"],
  "run_id": ["1", "2"],
  "task_id": ["rest", "nback"],
  "modalities": ["T1w", "bold"],
  "nprocs": 8,
  "mem_gb": 32,
  "float32": true,
  "ants_nthreads": 4,
  "no_sub": false,
  "verbose_reports": false
}
```

```bash
# Use config file
docker run -it --rm \
  -v /data:/data:ro \
  -v /output:/out \
  -v $(pwd)/mriqc_config.json:/config.json:ro \
  nipreps/mriqc:latest \
  /data /out participant \
  --config /config.json
```

### Memory and CPU Optimization

```bash
# Optimize for HPC cluster
docker run -it --rm \
  -v /data:/data:ro \
  -v /output:/out \
  nipreps/mriqc:latest \
  /data /out participant \
  --participant-label sub-01 \
  --nprocs 16 \
  --mem_gb 64 \
  --ants-nthreads 8 \
  --ants-float \
  --fd-radius 50  # Adjust for non-adult brains
```

### Partial Processing

```bash
# Skip certain processing steps
docker run -it --rm \
  -v /data:/data:ro \
  -v /output:/out \
  nipreps/mriqc:latest \
  /data /out participant \
  --no-sub  # Skip submission to MRIQC web API
```

## HPC Cluster Usage

### SLURM Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=mriqc
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=1-50  # 50 subjects

# Load Singularity
module load singularity

# Get subject ID from array
SUBJECT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subject_list.txt)

# Run MRIQC
singularity run --cleanenv \
  -B /data:/data \
  -B /scratch:/output \
  /containers/mriqc-latest.simg \
  /data /output participant \
  --participant-label ${SUBJECT} \
  --nprocs 16 \
  --mem_gb 64 \
  --work-dir /scratch/work_${SUBJECT}

echo "Completed MRIQC for ${SUBJECT}"
```

### PBS/Torque Script

```bash
#!/bin/bash
#PBS -N mriqc_batch
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -l mem=64gb
#PBS -t 1-50

cd $PBS_O_WORKDIR

SUBJECT=$(sed -n "${PBS_ARRAYID}p" subjects.txt)

singularity run --cleanenv \
  -B /data:/data \
  /containers/mriqc.simg \
  /data /output participant \
  --participant-label ${SUBJECT} \
  --nprocs 16 \
  --mem_gb 64
```

## Multi-Site Studies

### Quality Harmonization

```python
# Compare quality across sites and scanners

# Load metadata
df_qc = pd.read_csv('group_T1w.tsv', sep='\t')
df_meta = pd.read_csv('participants.tsv', sep='\t')

df_qc['participant_id'] = df_qc['bids_name'].str.extract(r'(sub-[^_]+)')
df = df_qc.merge(df_meta, on='participant_id')

# Site and scanner effects
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Test site effect on SNR
model = ols('snr_total ~ C(site) + C(scanner)', data=df).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

# Post-hoc pairwise comparisons
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(df['snr_total'], df['site'])
print(tukey)
```

### Quality Control Dashboard

```python
# Create interactive dashboard with Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df = pd.read_csv('group_T1w.tsv', sep='\t')

# Create dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('SNR Distribution', 'CNR vs SNR',
                    'Motion (if BOLD)', 'Quality Timeline')
)

# SNR distribution
fig.add_trace(
    go.Histogram(x=df['snr_total'], name='SNR'),
    row=1, col=1
)

# CNR vs SNR scatter
fig.add_trace(
    go.Scatter(x=df['snr_total'], y=df['cnr'],
               mode='markers', name='Subjects'),
    row=1, col=2
)

# Update layout
fig.update_layout(height=800, showlegend=True,
                  title_text="MRIQC Quality Dashboard")

fig.write_html('qc_dashboard.html')
print("Dashboard saved to qc_dashboard.html")
```

## Troubleshooting

**Problem:** MRIQC crashes with memory error
**Solution:** Increase `--mem_gb`, reduce `--nprocs`, or process fewer subjects simultaneously

**Problem:** "Not a valid BIDS dataset" error
**Solution:** Run BIDS validator, ensure proper BIDS organization, check for required files

**Problem:** Very slow processing
**Solution:** Use Docker/Singularity instead of local install, increase `--nprocs`, use SSD storage

**Problem:** Missing metrics in output
**Solution:** Check if input images are complete, verify modality specification, check MRIQC logs

**Problem:** Group report fails
**Solution:** Ensure all participant-level runs completed, check for corrupted JSON files, rerun group stage

## Best Practices

1. **Run MRIQC Before Preprocessing:**
   - Identify bad data early
   - Save computational resources
   - Exclude before fMRIPrep

2. **Use Containerization:**
   - Docker or Singularity for reproducibility
   - Consistent environment across systems
   - Easier HPC deployment

3. **Document QC Criteria:**
   - Define thresholds a priori
   - Document exclusions
   - Save QC scripts for reproducibility

4. **Multi-Modal QC:**
   - Run on T1w, T2w, BOLD, DWI
   - Compare across modalities
   - Consistent quality standards

5. **Reporting:**
   - Include MRIQC version
   - Report QC metrics in papers
   - Share QC reports with data
   - Document exclusion criteria

## Resources

- **Documentation:** https://mriqc.readthedocs.io/
- **GitHub:** https://github.com/nipreps/mriqc
- **Paper:** Esteban et al. (2017). PLOS ONE
- **Web API:** https://mriqc.nimh.nih.gov/ (share/compare metrics)
- **Forum:** https://neurostars.org/tag/mriqc
- **Tutorial:** https://mriqc.readthedocs.io/en/latest/tutorials.html

## Citation

```bibtex
@article{esteban2017mriqc,
  title={MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites},
  author={Esteban, Oscar and Birman, Daniel and Schaer, Marie and Koyejo, Oluwasanmi O and Poldrack, Russell A and Gorgolewski, Krzysztof J},
  journal={PLoS ONE},
  volume={12},
  number={9},
  pages={e0184661},
  year={2017},
  publisher={Public Library of Science}
}
```

## Related Tools

- **fMRIPrep:** Preprocessing pipeline (MRIQC is pre-processing QC)
- **QSIPrep:** Diffusion preprocessing with QC
- **BIDS Validator:** Format validation
- **VisualQC:** Manual quality control interface
- **XCP-D:** Post-processing QC for fMRIPrep outputs
- **C-PAC:** Alternative preprocessing with integrated QC
