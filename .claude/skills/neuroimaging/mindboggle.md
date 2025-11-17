# Mindboggle: Automated Brain Labeling and Shape Analysis

## Overview

**Mindboggle** is an open-source brain morphometry platform that improves upon traditional methods by taking the output of popular surface reconstruction pipelines (FreeSurfer, ANTs) and computing a rich set of shape measures for brain structures. Developed by Arno Klein and colleagues, Mindboggle provides automated anatomical brain labeling and extracts hundreds of morphological features from individual brains.

### Key Features

- **Multi-Pipeline Input**: Works with FreeSurfer and ANTs cortical surface outputs
- **Advanced Shape Analysis**: Computes depth, curvature, thickness, area, volume, Laplace-Beltrami spectra, and Zernike moments
- **Anatomical Labeling**: DKT-31 protocol with improved boundary definitions
- **Feature Extraction**: Hundreds of morphological measures per brain region
- **Quality Control**: Visual inspection tools and automated metrics
- **BIDS Integration**: Available as BIDS App for standardized workflows
- **Reproducibility**: Containerized deployment (Docker/Singularity)

### Scientific Foundation

Mindboggle implements validated methods for:
- Surface-based morphometry with improved anatomical precision
- Shape analysis using spectral and polynomial descriptors
- Travel depth calculation for sulcal and gyral identification
- Multi-atlas labeling with manual anatomical definitions

### Primary Use Cases

1. **Cortical morphometry studies** across populations
2. **Shape-based biomarker discovery** for neurological conditions
3. **Anatomical feature extraction** for machine learning
4. **Quality assessment** of surface reconstruction
5. **Multi-site harmonization** with standardized features

---

## Installation

### Docker Installation (Recommended)

```bash
# Pull the latest Mindboggle Docker image
docker pull nipy/mindboggle:latest

# Verify installation
docker run --rm nipy/mindboggle:latest mindboggle --version
```

### Singularity Installation

```bash
# Build Singularity image from Docker Hub
singularity build mindboggle.sif docker://nipy/mindboggle:latest

# Test the container
singularity exec mindboggle.sif mindboggle --version
```

### Python Package Installation

```bash
# Create virtual environment
conda create -n mindboggle python=3.8
conda activate mindboggle

# Install Mindboggle
pip install mindboggle

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y \
    vtk9 \
    cmake \
    libvtk9-dev \
    python3-vtk9

# Verify installation
mindboggle --help
```

### FreeSurfer Setup

Mindboggle requires FreeSurfer output as input:

```bash
# Set FreeSurfer environment
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Verify FreeSurfer is available
which recon-all
```

### ANTS Setup

For ANTs-based workflows:

```bash
# Install ANTs
conda install -c conda-forge ants

# Verify ANTs installation
antsRegistration --version
```

---

## Basic Usage

### Running Mindboggle on FreeSurfer Output

```bash
# Basic Mindboggle command with FreeSurfer input
mindboggle \
    /data/freesurfer_subjects/sub-01 \
    --out /output/mindboggle/sub-01

# This processes both hemispheres and generates:
# - labels/     (anatomical labels)
# - features/   (shape features)
# - shapes/     (surface files)
# - tables/     (CSV feature tables)
```

### Docker Execution

```bash
# Run Mindboggle using Docker
docker run --rm \
    -v /data/freesurfer_subjects:/input:ro \
    -v /output/mindboggle:/output \
    nipy/mindboggle:latest \
    /input/sub-01 \
    --out /output/sub-01
```

### Singularity Execution

```bash
# Run with Singularity (HPC-friendly)
singularity run \
    -B /data/freesurfer_subjects:/input:ro \
    -B /output/mindboggle:/output \
    mindboggle.sif \
    /input/sub-01 \
    --out /output/sub-01
```

### Output Structure

Mindboggle creates a comprehensive output directory:

```
sub-01/
├── labels/
│   ├── left_hemisphere_labels.nii.gz
│   └── right_hemisphere_labels.nii.gz
├── features/
│   ├── left_hemisphere_features.csv
│   └── right_hemisphere_features.csv
├── shapes/
│   ├── left_travel_depth.vtk
│   ├── right_travel_depth.vtk
│   ├── left_curvature.vtk
│   └── right_curvature.vtk
├── tables/
│   ├── thickinthehead_per_freesurfer_cortex_label.csv
│   ├── volumes_per_freesurfer_label.csv
│   └── shapes_per_freesurfer_label.csv
└── artefacts.txt
```

---

## Anatomical Labeling

### DKT-31 Protocol

Mindboggle uses the Desikan-Killiany-Tourville (DKT) protocol with 31 cortical labels per hemisphere:

```python
import nibabel as nib
import numpy as np

# Load Mindboggle labels
labels_img = nib.load('/output/sub-01/labels/left_hemisphere_labels.nii.gz')
labels_data = labels_img.get_fdata()

# Get unique labels
unique_labels = np.unique(labels_data)
print(f"Found {len(unique_labels)} unique labels")

# DKT-31 label names
dkt_labels = {
    1002: 'caudalanteriorcingulate',
    1003: 'caudalmiddlefrontal',
    1005: 'cuneus',
    1006: 'entorhinal',
    1007: 'fusiform',
    1008: 'inferiorparietal',
    1009: 'inferiortemporal',
    # ... (31 cortical regions per hemisphere)
}
```

### Extracting Label Statistics

```python
import pandas as pd

# Load per-label feature table
features_df = pd.read_csv(
    '/output/sub-01/tables/shapes_per_freesurfer_label.csv'
)

# View available features
print(features_df.columns.tolist())
# ['label', 'area', 'volume', 'thickness.mean', 'thickness.std',
#  'travel_depth.mean', 'travel_depth.std', 'geodesic_depth.mean',
#  'mean_curvature.mean', 'freesurfer_curvature.mean', ...]

# Get specific region statistics
middle_frontal = features_df[
    features_df['label.text'] == 'Left-middle-frontal'
]
print(f"Middle frontal cortex area: {middle_frontal['area'].values[0]:.2f} mm²")
print(f"Mean thickness: {middle_frontal['thickness.mean'].values[0]:.2f} mm")
```

### Comparing Labels to FreeSurfer

```python
import nibabel.freesurfer as nbfs

# Load FreeSurfer annotation
fs_labels, fs_ctab, fs_names = nbfs.read_annot(
    '/data/freesurfer_subjects/sub-01/label/lh.aparc.annot'
)

# Load Mindboggle labels
mb_labels_img = nib.load(
    '/output/sub-01/labels/left_hemisphere_labels.nii.gz'
)
mb_labels = mb_labels_img.get_fdata()

# Calculate overlap/agreement
from scipy.stats import contingency

# Create contingency table
overlap = np.histogram2d(
    fs_labels.ravel(),
    mb_labels.ravel(),
    bins=[len(np.unique(fs_labels)), len(np.unique(mb_labels))]
)[0]

print(f"Label overlap matrix shape: {overlap.shape}")
```

---

## Shape Feature Extraction

### Travel Depth

Travel depth measures the distance from the brain surface along the cortical fold:

```python
import vtk
from vtk.util import numpy_support

def load_vtk_surface(vtk_file):
    """Load VTK surface file"""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    return reader.GetOutput()

def extract_scalar_data(polydata, scalar_name='scalars'):
    """Extract scalar values from VTK polydata"""
    scalars = polydata.GetPointData().GetScalars(scalar_name)
    if scalars is None:
        scalars = polydata.GetPointData().GetScalars()
    return numpy_support.vtk_to_numpy(scalars)

# Load travel depth surface
travel_depth_surface = load_vtk_surface(
    '/output/sub-01/shapes/left_travel_depth.vtk'
)

# Extract depth values
depth_values = extract_scalar_data(travel_depth_surface)

print(f"Travel depth range: {depth_values.min():.2f} to {depth_values.max():.2f} mm")
print(f"Mean depth: {depth_values.mean():.2f} mm")

# Identify sulci (negative values) and gyri (positive values)
sulcal_vertices = depth_values < 0
gyral_vertices = depth_values > 0

print(f"Sulcal vertices: {sulcal_vertices.sum()} ({100*sulcal_vertices.sum()/len(depth_values):.1f}%)")
print(f"Gyral vertices: {gyral_vertices.sum()} ({100*gyral_vertices.sum()/len(depth_values):.1f}%)")
```

### Curvature Measures

```python
# Load mean curvature
curvature_surface = load_vtk_surface(
    '/output/sub-01/shapes/left_curvature.vtk'
)
curvature_values = extract_scalar_data(curvature_surface)

# Analyze curvature distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(curvature_values, bins=100, edgecolor='black')
plt.xlabel('Mean Curvature')
plt.ylabel('Frequency')
plt.title('Curvature Distribution')

plt.subplot(1, 2, 2)
plt.hist(depth_values, bins=100, edgecolor='black')
plt.xlabel('Travel Depth (mm)')
plt.ylabel('Frequency')
plt.title('Travel Depth Distribution')

plt.tight_layout()
plt.savefig('/output/sub-01/shape_distributions.png')
```

### Thickness Analysis

```python
# Load cortical thickness table
thickness_df = pd.read_csv(
    '/output/sub-01/tables/thickinthehead_per_freesurfer_cortex_label.csv'
)

# Summary statistics per region
thickness_summary = thickness_df.groupby('label.text')['thickness'].agg([
    'mean', 'std', 'min', 'max', 'count'
]).sort_values('mean', ascending=False)

print("Top 5 thickest regions:")
print(thickness_summary.head())

print("\nTop 5 thinnest regions:")
print(thickness_summary.tail())
```

### Laplace-Beltrami Spectra

```python
# Load spectral shape features
spectra_file = '/output/sub-01/features/left_hemisphere_features.csv'
spectra_df = pd.read_csv(spectra_file)

# Extract Laplace-Beltrami eigenvalues (if available)
lb_columns = [col for col in spectra_df.columns if 'laplace' in col.lower()]

if lb_columns:
    lb_features = spectra_df[lb_columns]
    print(f"Found {len(lb_columns)} Laplace-Beltrami spectral features")
    print(lb_features.describe())
```

### Zernike Moments

```python
# Extract Zernike moment features
zernike_columns = [col for col in spectra_df.columns if 'zernike' in col.lower()]

if zernike_columns:
    zernike_features = spectra_df[zernike_columns]
    print(f"Found {len(zernike_columns)} Zernike moment features")

    # Zernike moments are rotation-invariant shape descriptors
    # Useful for shape-based classification
    print("\nZernike moment statistics:")
    print(zernike_features.describe())
```

---

## Quality Control

### Visual Inspection with Paraview

```bash
# Install ParaView for VTK visualization
sudo apt-get install paraview

# Open travel depth surface
paraview /output/sub-01/shapes/left_travel_depth.vtk

# Or use VTK viewer
vtk_viewer /output/sub-01/shapes/left_travel_depth.vtk
```

### Automated QC Metrics

```python
def compute_qc_metrics(subject_dir):
    """Compute quality control metrics for Mindboggle output"""

    qc_metrics = {}

    # Check for artefacts file
    artefacts_file = f"{subject_dir}/artefacts.txt"
    if os.path.exists(artefacts_file):
        with open(artefacts_file, 'r') as f:
            artefacts = f.read().strip()
            qc_metrics['artefacts_detected'] = len(artefacts) > 0

    # Load feature tables
    shapes_df = pd.read_csv(f"{subject_dir}/tables/shapes_per_freesurfer_label.csv")

    # Check for outlier values
    qc_metrics['mean_thickness'] = shapes_df['thickness.mean'].mean()
    qc_metrics['thickness_std'] = shapes_df['thickness.mean'].std()

    # Flag suspicious thickness values
    suspicious_thickness = (
        (shapes_df['thickness.mean'] < 1.0) |
        (shapes_df['thickness.mean'] > 6.0)
    ).sum()
    qc_metrics['suspicious_thickness_regions'] = suspicious_thickness

    # Check surface area
    qc_metrics['total_cortical_area'] = shapes_df['area'].sum()

    # Flag if total area is outside expected range (typical: 1500-2500 cm²)
    qc_metrics['area_within_normal_range'] = (
        15000 < qc_metrics['total_cortical_area'] < 25000
    )

    return qc_metrics

# Run QC
qc_results = compute_qc_metrics('/output/sub-01')
print("Quality Control Metrics:")
for metric, value in qc_results.items():
    print(f"  {metric}: {value}")
```

### Comparing to Population Norms

```python
import numpy as np
from scipy import stats

def compare_to_norms(subject_features, normative_data):
    """Compare subject features to normative population"""

    z_scores = {}

    for feature in subject_features.columns:
        if feature in normative_data.columns:
            subject_value = subject_features[feature].mean()
            norm_mean = normative_data[feature].mean()
            norm_std = normative_data[feature].std()

            z_score = (subject_value - norm_mean) / norm_std
            z_scores[feature] = z_score

    return pd.DataFrame([z_scores])

# Load normative data (example)
normative_df = pd.read_csv('/data/normative_features.csv')

# Load subject features
subject_df = pd.read_csv(
    '/output/sub-01/tables/shapes_per_freesurfer_label.csv'
)

# Compute z-scores
z_scores = compare_to_norms(subject_df, normative_df)

# Flag outliers (|z| > 2.5)
outlier_features = z_scores.columns[np.abs(z_scores.values[0]) > 2.5]
print(f"Outlier features: {list(outlier_features)}")
```

---

## Batch Processing

### Processing Multiple Subjects

```bash
#!/bin/bash
# batch_mindboggle.sh

SUBJECTS_DIR=/data/freesurfer_subjects
OUTPUT_DIR=/output/mindboggle

# Get list of subjects
subjects=$(ls $SUBJECTS_DIR)

for subject in $subjects; do
    echo "Processing $subject..."

    mindboggle \
        $SUBJECTS_DIR/$subject \
        --out $OUTPUT_DIR/$subject \
        --cpus 4

    if [ $? -eq 0 ]; then
        echo "✓ $subject completed successfully"
    else
        echo "✗ $subject failed"
    fi
done
```

### Parallel Processing with GNU Parallel

```bash
# Create subject list
ls /data/freesurfer_subjects > subjects.txt

# Run in parallel (4 concurrent jobs)
cat subjects.txt | parallel -j 4 \
    "mindboggle \
        /data/freesurfer_subjects/{} \
        --out /output/mindboggle/{} \
        --cpus 2"
```

### Python Batch Processing

```python
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_subject(subject_id, subjects_dir, output_dir):
    """Process single subject with Mindboggle"""

    subject_path = subjects_dir / subject_id
    output_path = output_dir / subject_id

    cmd = [
        'mindboggle',
        str(subject_path),
        '--out', str(output_path),
        '--cpus', '4'
    ]

    logger.info(f"Starting {subject_id}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ {subject_id} completed")
        return {'subject': subject_id, 'status': 'success'}

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {subject_id} failed: {e.stderr}")
        return {'subject': subject_id, 'status': 'failed', 'error': e.stderr}

def batch_process_subjects(subjects_dir, output_dir, max_workers=4):
    """Process multiple subjects in parallel"""

    subjects_dir = Path(subjects_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get subject list
    subjects = [d.name for d in subjects_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(subjects)} subjects to process")

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_subject, subj, subjects_dir, output_dir)
            for subj in subjects
        ]

        for future in futures:
            results.append(future.result())

    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    logger.info(f"Completed: {successful}/{len(subjects)} subjects")

    return results

# Run batch processing
results = batch_process_subjects(
    subjects_dir='/data/freesurfer_subjects',
    output_dir='/output/mindboggle',
    max_workers=4
)
```

---

## Integration with Neuroimaging Tools

### Loading Mindboggle Output in Python

```python
import nibabel as nib
import numpy as np
import pandas as pd

class MindboggleOutput:
    """Helper class to load Mindboggle outputs"""

    def __init__(self, subject_dir):
        self.subject_dir = Path(subject_dir)

    def load_labels(self, hemisphere='left'):
        """Load anatomical labels"""
        label_file = self.subject_dir / 'labels' / f'{hemisphere}_hemisphere_labels.nii.gz'
        return nib.load(str(label_file))

    def load_shape_features(self):
        """Load shape feature tables"""
        features_file = self.subject_dir / 'tables' / 'shapes_per_freesurfer_label.csv'
        return pd.read_csv(features_file)

    def load_thickness(self):
        """Load thickness measurements"""
        thickness_file = self.subject_dir / 'tables' / 'thickinthehead_per_freesurfer_cortex_label.csv'
        return pd.read_csv(thickness_file)

    def load_surface(self, surface_name, hemisphere='left'):
        """Load VTK surface"""
        surface_file = self.subject_dir / 'shapes' / f'{hemisphere}_{surface_name}.vtk'
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(surface_file))
        reader.Update()
        return reader.GetOutput()

# Usage
mb_output = MindboggleOutput('/output/sub-01')
features = mb_output.load_shape_features()
thickness = mb_output.load_thickness()
```

### Integration with Nilearn

```python
from nilearn import plotting, surface
import matplotlib.pyplot as plt

# Load Mindboggle surface for visualization
# Convert VTK to FSL/FreeSurfer format if needed
left_surface = mb_output.load_surface('travel_depth', hemisphere='left')

# Extract coordinates and faces
coords = numpy_support.vtk_to_numpy(
    left_surface.GetPoints().GetData()
)
faces = numpy_support.vtk_to_numpy(
    left_surface.GetPolys().GetData()
).reshape(-1, 4)[:, 1:]  # Skip cell size

# Load scalar data
travel_depth = numpy_support.vtk_to_numpy(
    left_surface.GetPointData().GetScalars()
)

# Create surface mesh for Nilearn
mesh = [coords, faces]

# Plot with Nilearn
fig = plotting.plot_surf(
    mesh,
    surf_map=travel_depth,
    hemi='left',
    view='lateral',
    cmap='coolwarm',
    colorbar=True,
    title='Travel Depth (Mindboggle)'
)
plt.savefig('/output/travel_depth_visualization.png')
```

### Combining with fMRIPrep Derivatives

```python
from bids import BIDSLayout

# Load BIDS dataset
layout = BIDSLayout('/data/bids_dataset', derivatives=[
    '/data/bids_dataset/derivatives/fmriprep',
    '/output/mindboggle'
])

# Get fMRIPrep preprocessed T1w
t1w_preproc = layout.get(
    subject='01',
    datatype='anat',
    suffix='T1w',
    space='MNI152NLin2009cAsym',
    extension='nii.gz',
    return_type='filename'
)[0]

# Get Mindboggle labels
mb_labels = layout.get(
    subject='01',
    datatype='anat',
    suffix='dseg',
    desc='mindboggle',
    extension='nii.gz',
    return_type='filename'
)[0]

# Combine for ROI analysis
import nibabel as nib

t1w_img = nib.load(t1w_preproc)
labels_img = nib.load(mb_labels)

# Extract mean intensity per ROI
t1w_data = t1w_img.get_fdata()
labels_data = labels_img.get_fdata()

roi_intensities = {}
for label in np.unique(labels_data)[1:]:  # Skip background
    mask = labels_data == label
    roi_intensities[int(label)] = t1w_data[mask].mean()

print(f"Extracted intensities for {len(roi_intensities)} ROIs")
```

---

## BIDS App Usage

### Running as BIDS App

```bash
# Prepare BIDS dataset with FreeSurfer derivatives
bids_dir=/data/bids_dataset
derivatives_dir=$bids_dir/derivatives/freesurfer
output_dir=$bids_dir/derivatives/mindboggle

# Run Mindboggle BIDS App
docker run --rm \
    -v $bids_dir:/data:ro \
    -v $output_dir:/output \
    nipy/mindboggle:latest \
    /data \
    /output \
    participant \
    --participant-label 01 02 03 \
    --freesurfer-dir /data/derivatives/freesurfer
```

### BIDS App with Custom Options

```bash
# Run with specific options
docker run --rm \
    -v $bids_dir:/data:ro \
    -v $output_dir:/output \
    nipy/mindboggle:latest \
    /data \
    /output \
    participant \
    --participant-label 01 \
    --freesurfer-dir /data/derivatives/freesurfer \
    --cpus 8 \
    --working-dir /tmp/work
```

### Group-Level Analysis

```python
def aggregate_mindboggle_features(derivatives_dir, subjects):
    """Aggregate Mindboggle features across subjects"""

    all_features = []

    for subject in subjects:
        subject_dir = Path(derivatives_dir) / f'sub-{subject}'
        features_file = subject_dir / 'tables' / 'shapes_per_freesurfer_label.csv'

        if features_file.exists():
            df = pd.read_csv(features_file)
            df['subject'] = subject
            all_features.append(df)

    # Combine all subjects
    combined_df = pd.concat(all_features, ignore_index=True)

    return combined_df

# Load group data
subjects = ['01', '02', '03', '04', '05']
group_features = aggregate_mindboggle_features(
    '/data/bids_dataset/derivatives/mindboggle',
    subjects
)

# Compute group statistics
group_stats = group_features.groupby('label.text').agg({
    'area': ['mean', 'std'],
    'thickness.mean': ['mean', 'std'],
    'travel_depth.mean': ['mean', 'std']
})

print(group_stats)
```

---

## Advanced Analysis

### Feature Selection for Machine Learning

```python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

def prepare_ml_features(group_features, demographics):
    """Prepare Mindboggle features for machine learning"""

    # Pivot features to wide format (one row per subject)
    feature_cols = ['area', 'thickness.mean', 'travel_depth.mean', 'curvature.mean']

    pivot_dfs = []
    for feature in feature_cols:
        pivot = group_features.pivot(
            index='subject',
            columns='label.text',
            values=feature
        )
        pivot.columns = [f"{col}_{feature}" for col in pivot.columns]
        pivot_dfs.append(pivot)

    # Combine all features
    X = pd.concat(pivot_dfs, axis=1)

    # Merge with demographics
    data = X.merge(demographics, left_index=True, right_on='subject')

    return data

# Load demographics
demographics = pd.read_csv('/data/participants.tsv', sep='\t')

# Prepare features
ml_data = prepare_ml_features(group_features, demographics)

# Feature selection
X = ml_data.select_dtypes(include=[np.number]).drop('age', axis=1)
y = (ml_data['age'] > 65).astype(int)  # Binary classification

# Select top 50 features
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} features:")
print(selected_features[:10])
```

### Longitudinal Analysis

```python
def longitudinal_analysis(derivatives_dir, subject, timepoints):
    """Analyze longitudinal changes in Mindboggle features"""

    timepoint_features = []

    for tp in timepoints:
        tp_dir = Path(derivatives_dir) / f'sub-{subject}_ses-{tp}'
        features_file = tp_dir / 'tables' / 'shapes_per_freesurfer_label.csv'

        if features_file.exists():
            df = pd.read_csv(features_file)
            df['timepoint'] = tp
            timepoint_features.append(df)

    # Combine timepoints
    longitudinal_df = pd.concat(timepoint_features, ignore_index=True)

    # Calculate change scores
    pivot_df = longitudinal_df.pivot_table(
        index='label.text',
        columns='timepoint',
        values='thickness.mean'
    )

    # Compute annual change rate
    pivot_df['annual_change'] = (
        (pivot_df[timepoints[-1]] - pivot_df[timepoints[0]]) /
        (len(timepoints) - 1)
    )

    return pivot_df

# Analyze longitudinal changes
timepoints = ['01', '02', '03']  # 3 annual visits
longitudinal_results = longitudinal_analysis(
    '/data/derivatives/mindboggle',
    subject='01',
    timepoints=timepoints
)

# Identify regions with significant atrophy
significant_atrophy = longitudinal_results[
    longitudinal_results['annual_change'] < -0.05  # >0.05 mm/year thinning
].sort_values('annual_change')

print("Regions with significant cortical thinning:")
print(significant_atrophy)
```

---

## Troubleshooting

### Common Issues

**Issue: FreeSurfer subjects directory not found**

```bash
# Verify FreeSurfer output exists
ls -lh /data/freesurfer_subjects/sub-01/

# Check for required files
required_files=(
    "surf/lh.pial"
    "surf/rh.pial"
    "surf/lh.white"
    "surf/rh.white"
    "label/lh.aparc.annot"
    "label/rh.aparc.annot"
)

for file in "${required_files[@]}"; do
    if [ ! -f "/data/freesurfer_subjects/sub-01/$file" ]; then
        echo "Missing: $file"
    fi
done
```

**Issue: VTK import errors**

```python
# Check VTK installation
try:
    import vtk
    print(f"VTK version: {vtk.vtkVersion.GetVTKVersion()}")
except ImportError as e:
    print(f"VTK not available: {e}")
    print("Install with: conda install -c conda-forge vtk")
```

**Issue: Memory errors during processing**

```bash
# Reduce parallelization
mindboggle \
    /data/freesurfer_subjects/sub-01 \
    --out /output/sub-01 \
    --cpus 1 \
    --no-parallel

# Or increase memory limits for Docker
docker run --rm \
    --memory=16g \
    --memory-swap=24g \
    -v /data:/data:ro \
    -v /output:/output \
    nipy/mindboggle:latest \
    /data/freesurfer_subjects/sub-01 \
    --out /output/sub-01
```

### Validation and Testing

```python
def validate_mindboggle_output(output_dir):
    """Validate Mindboggle output completeness"""

    output_path = Path(output_dir)
    issues = []

    # Check required directories
    required_dirs = ['labels', 'features', 'shapes', 'tables']
    for dirname in required_dirs:
        if not (output_path / dirname).exists():
            issues.append(f"Missing directory: {dirname}")

    # Check for label files
    for hemi in ['left', 'right']:
        label_file = output_path / 'labels' / f'{hemi}_hemisphere_labels.nii.gz'
        if not label_file.exists():
            issues.append(f"Missing {hemi} hemisphere labels")

    # Check for feature tables
    required_tables = [
        'shapes_per_freesurfer_label.csv',
        'thickinthehead_per_freesurfer_cortex_label.csv'
    ]
    for table in required_tables:
        if not (output_path / 'tables' / table).exists():
            issues.append(f"Missing table: {table}")

    # Validate feature table content
    if not issues:
        shapes_file = output_path / 'tables' / 'shapes_per_freesurfer_label.csv'
        shapes_df = pd.read_csv(shapes_file)

        # Check for expected number of regions (62 cortical for DKT)
        if len(shapes_df) < 60:
            issues.append(f"Unexpected number of regions: {len(shapes_df)}")

        # Check for NaN values
        nan_count = shapes_df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values in features")

    if issues:
        print("⚠ Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Output validation passed")
        return True

# Validate output
validate_mindboggle_output('/output/sub-01')
```

---

## Multi-Site Harmonization

### ComBat Harmonization

```python
from neuroCombat import neuroCombat
import pandas as pd
import numpy as np

def harmonize_mindboggle_features(features_df, site_info):
    """Harmonize Mindboggle features across sites using ComBat"""

    # Prepare data matrix (subjects x features)
    # Group by subject to get one row per subject
    subject_features = features_df.pivot_table(
        index='subject',
        columns='label.text',
        values='thickness.mean'
    )

    # Prepare batch/site information
    batch = site_info.set_index('subject').loc[subject_features.index, 'site']

    # Prepare covariates (optional: age, sex)
    covars = site_info.set_index('subject').loc[
        subject_features.index, ['age', 'sex']
    ]

    # Run ComBat harmonization
    harmonized_data = neuroCombat(
        dat=subject_features.T.values,  # features x subjects
        batch=batch.values,
        mod=covars.values
    )['data']

    # Convert back to DataFrame
    harmonized_df = pd.DataFrame(
        harmonized_data.T,
        index=subject_features.index,
        columns=subject_features.columns
    )

    return harmonized_df

# Load multi-site data
site_info = pd.read_csv('/data/site_information.tsv', sep='\t')

# Harmonize
harmonized_features = harmonize_mindboggle_features(
    group_features,
    site_info
)

print("✓ Features harmonized across sites")
```

### Site Effect Assessment

```python
from sklearn.decomposition import PCA
import seaborn as sns

def assess_site_effects(features_df, site_info, before_harmonization=True):
    """Assess site effects using PCA"""

    # Merge features with site information
    data = features_df.merge(site_info, on='subject')

    # Extract feature columns
    feature_cols = [col for col in data.columns if col not in ['subject', 'site', 'age', 'sex']]
    X = data[feature_cols].values

    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    # Plot
    plt.figure(figsize=(8, 6))
    for site in data['site'].unique():
        mask = data['site'] == site
        plt.scatter(
            components[mask, 0],
            components[mask, 1],
            label=f'Site {site}',
            alpha=0.6
        )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA of Mindboggle Features ' +
              ('(Before Harmonization)' if before_harmonization else '(After Harmonization)'))
    plt.legend()
    plt.tight_layout()

    status = 'before' if before_harmonization else 'after'
    plt.savefig(f'/output/site_effects_{status}_harmonization.png')

# Assess before and after harmonization
assess_site_effects(group_features, site_info, before_harmonization=True)
assess_site_effects(harmonized_features, site_info, before_harmonization=False)
```

---

## Best Practices

### Recommended Workflow

1. **Quality-Controlled FreeSurfer Processing**
   - Ensure FreeSurfer recon-all completed successfully
   - Manually inspect and edit surfaces if needed
   - Check for motion artifacts

2. **Mindboggle Processing**
   - Run with adequate computational resources (8+ GB RAM per subject)
   - Use containerized version for reproducibility
   - Enable parallelization when processing multiple subjects

3. **Quality Control**
   - Visually inspect travel depth and curvature surfaces
   - Check for outlier feature values
   - Compare to population norms

4. **Feature Extraction and Analysis**
   - Extract relevant features for your research question
   - Consider harmonization for multi-site studies
   - Use appropriate statistical methods

5. **Reporting and Reproducibility**
   - Document Mindboggle version and parameters
   - Share feature extraction code
   - Provide summary statistics

### Performance Optimization

```bash
# Optimize for HPC environments
#!/bin/bash
#SBATCH --job-name=mindboggle
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00

module load singularity

singularity run \
    --cleanenv \
    -B /scratch:/tmp \
    -B /data:/data:ro \
    -B /output:/output \
    /apps/mindboggle.sif \
    /data/freesurfer_subjects/${SUBJECT_ID} \
    --out /output/${SUBJECT_ID} \
    --cpus 8 \
    --working-dir /tmp/${SLURM_JOB_ID}
```

---

## References

### Key Publications

1. Klein, A., Ghosh, S. S., Bao, F. S., et al. (2017). "Mindboggling morphometry of human brains." *PLOS Computational Biology*, 13(2), e1005350.

2. Klein, A., & Tourville, J. (2012). "101 labeled brain images and a consistent human cortical labeling protocol." *Frontiers in Neuroscience*, 6, 171.

3. Klein, A., et al. (2017). "Evaluation of volume-based and surface-based brain image registration methods." *NeuroImage*, 51(1), 214-220.

### Documentation and Resources

- **Official Documentation**: http://mindboggle.info
- **GitHub Repository**: https://github.com/nipy/mindboggle
- **Docker Hub**: https://hub.docker.com/r/nipy/mindboggle
- **BIDS App**: https://github.com/BIDS-Apps/mindboggle
- **Example Datasets**: https://osf.io/nhtur/

### Related Tools

- **FreeSurfer**: Surface reconstruction and cortical parcellation
- **ANTs**: Image registration and normalization
- **BrainVISA**: Morphometry and sulcal analysis
- **CIVET**: Alternative cortical surface pipeline
- **Connectome Workbench**: Surface visualization and analysis

---

## See Also

- **freesurfer.md**: FreeSurfer cortical reconstruction
- **ants.md**: ANTs registration and segmentation
- **fmriprep.md**: fMRI preprocessing with surface outputs
- **nilearn.md**: Machine learning with neuroimaging data
- **civet.md**: CIVET cortical processing pipeline
