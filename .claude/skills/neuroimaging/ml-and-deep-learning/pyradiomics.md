# PyRadiomics: Radiomics Feature Extraction for Medical Imaging

## Overview

**PyRadiomics** is a Python package for extracting Radiomics features from medical imaging, developed at Harvard Medical School. It computes hundreds of quantitative imaging features including shape, intensity, and texture descriptors that can be used as biomarkers for machine learning models. PyRadiomics is highly configurable, validated against multiple phantom datasets, IBSI (Image Biomarker Standardization Initiative) compliant, and widely used in oncology, neurology, and precision medicine research.

### Key Features

- **100+ Standardized Features**: Shape, first-order, and texture features
- **IBSI Compliant**: Follows standardized radiomics definitions
- **Image Filters**: Wavelet, LoG, Gradient, Square, Exponential, LBP
- **Configurable Preprocessing**: Resampling, normalization, discretization
- **Multi-Region Support**: Extract features from multiple ROIs
- **Reproducible**: Validated on digital phantoms
- **Batch Processing**: Efficient multi-subject feature extraction
- **Flexible Output**: Export to CSV, JSON, or Python dictionaries
- **Integration Ready**: Works with scikit-learn, pandas, NumPy
- **Clinical Validation**: Used in 1000+ publications

### Scientific Foundation

Radiomics extracts quantitative features from medical images that characterize:

- **Shape**: Geometric properties of regions (volume, surface, sphericity)
- **Intensity**: First-order statistics (mean, variance, skewness, kurtosis)
- **Texture**: Second-order patterns (homogeneity, contrast, correlation)
- **Higher-Order**: Filtered image characteristics (wavelet coefficients)

These features can capture phenotypic characteristics invisible to human eyes and serve as biomarkers for diagnosis, prognosis, treatment response, and molecular characteristics.

### Primary Use Cases

1. **Tumor Characterization**: Glioma grading, metastasis prediction
2. **Treatment Response**: Predict therapy outcomes
3. **Radiogenomics**: Link imaging to genomic profiles
4. **Disease Classification**: Alzheimer's, MS lesion characterization
5. **Prognostic Models**: Survival prediction
6. **Machine Learning**: Feature engineering for medical imaging ML

---

## Installation

### Using pip (Recommended)

```bash
# Install PyRadiomics
pip install pyradiomics

# Verify installation
python -c "import radiomics; print(radiomics.__version__)"
```

### Using conda

```bash
# Create environment with PyRadiomics
conda create -n pyradiomics-env python=3.9
conda activate pyradiomics-env
pip install pyradiomics

# Verify installation
python -c "import radiomics; print(radiomics.__version__)"
```

### Install from Source

```bash
# Clone repository
git clone https://github.com/AIM-Harvard/pyradiomics.git
cd pyradiomics

# Install
python -m pip install -r requirements.txt
python -m pip install .

# Run tests
python -m pytest tests/
```

### Dependencies

PyRadiomics requires:
- Python ≥ 3.6
- NumPy, SimpleITK
- PyWavelets, scikit-image
- Six, pykwalify (for validation)

---

## Basic Feature Extraction

### Load Image and Mask

```python
from radiomics import featureextractor
import SimpleITK as sitk

# Load image and segmentation mask
image_path = 'sub-01_T1w.nii.gz'
mask_path = 'sub-01_tumor_mask.nii.gz'

# Load with SimpleITK
image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

print(f"Image size: {image.GetSize()}")
print(f"Image spacing: {image.GetSpacing()}")
print(f"Mask labels: {sitk.GetArrayFromImage(mask).max()}")
```

### Extract All Features

```python
# Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Extract features
result = extractor.execute(image_path, mask_path)

# Print all features
print(f"\nExtracted {len(result)} features:")
for key, value in result.items():
    if not key.startswith('diagnostics'):
        print(f"{key}: {value}")
```

### Feature Categories

```python
# Group features by category
shape_features = {}
firstorder_features = {}
texture_features = {}

for key, value in result.items():
    if 'shape' in key:
        shape_features[key] = value
    elif 'firstorder' in key:
        firstorder_features[key] = value
    elif any(tex in key for tex in ['glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']):
        texture_features[key] = value

print(f"\nShape features: {len(shape_features)}")
print(f"First-order features: {len(firstorder_features)}")
print(f"Texture features: {len(texture_features)}")
```

---

## Shape Features

### Volumetric Measurements

```python
# Extract shape features
extractor_shape = featureextractor.RadiomicsFeatureExtractor()
extractor_shape.disableAllFeatures()
extractor_shape.enableFeatureClassByName('shape')

shape_result = extractor_shape.execute(image_path, mask_path)

# Key shape features
shape_dict = {k.split('_')[-1]: v for k, v in shape_result.items()
              if 'shape' in k and not k.startswith('diagnostics')}

print("Shape Features:")
print(f"  Volume (voxels): {shape_dict.get('VoxelVolume', 'N/A')}")
print(f"  Volume (mm³): {shape_dict.get('MeshVolume', 'N/A')}")
print(f"  Surface Area (mm²): {shape_dict.get('SurfaceArea', 'N/A')}")
print(f"  Surface-to-Volume Ratio: {shape_dict.get('SurfaceVolumeRatio', 'N/A')}")
print(f"  Sphericity: {shape_dict.get('Sphericity', 'N/A')}")
print(f"  Compactness1: {shape_dict.get('Compactness1', 'N/A')}")
print(f"  Compactness2: {shape_dict.get('Compactness2', 'N/A')}")
```

### Shape Descriptors

```python
# Additional shape descriptors
print("\nShape Descriptors:")
print(f"  Maximum 3D Diameter: {shape_dict.get('Maximum3DDiameter', 'N/A'):.2f} mm")
print(f"  Maximum 2D Diameter (Slice): {shape_dict.get('Maximum2DDiameterSlice', 'N/A'):.2f} mm")
print(f"  Elongation: {shape_dict.get('Elongation', 'N/A'):.3f}")
print(f"  Flatness: {shape_dict.get('Flatness', 'N/A'):.3f}")
print(f"  Spherical Disproportion: {shape_dict.get('SphericalDisproportion', 'N/A'):.3f}")

# Interpretation
elongation = shape_dict.get('Elongation', 0)
if elongation < 0.5:
    print("  → Highly elongated structure")
elif elongation > 0.9:
    print("  → Nearly spherical structure")
else:
    print("  → Moderately elongated structure")
```

---

## First-Order Features

### Intensity Statistics

```python
# Extract first-order features
extractor_fo = featureextractor.RadiomicsFeatureExtractor()
extractor_fo.disableAllFeatures()
extractor_fo.enableFeatureClassByName('firstorder')

fo_result = extractor_fo.execute(image_path, mask_path)

# Extract feature values
fo_dict = {k.split('_')[-1]: v for k, v in fo_result.items()
           if 'firstorder' in k and not k.startswith('diagnostics')}

print("First-Order Features (Intensity Statistics):")
print(f"  Mean: {fo_dict.get('Mean', 'N/A'):.2f}")
print(f"  Median: {fo_dict.get('Median', 'N/A'):.2f}")
print(f"  Std Dev: {fo_dict.get('StandardDeviation', 'N/A'):.2f}")
print(f"  Variance: {fo_dict.get('Variance', 'N/A'):.2f}")
print(f"  Min: {fo_dict.get('Minimum', 'N/A'):.2f}")
print(f"  Max: {fo_dict.get('Maximum', 'N/A'):.2f}")
print(f"  Range: {fo_dict.get('Range', 'N/A'):.2f}")
```

### Histogram Features

```python
print("\nHistogram Features:")
print(f"  Skewness: {fo_dict.get('Skewness', 'N/A'):.3f}")
print(f"  Kurtosis: {fo_dict.get('Kurtosis', 'N/A'):.3f}")
print(f"  Entropy: {fo_dict.get('Entropy', 'N/A'):.3f}")
print(f"  Uniformity: {fo_dict.get('Uniformity', 'N/A'):.3f}")

# Interpretation
skewness = fo_dict.get('Skewness', 0)
if skewness > 0:
    print("  → Right-skewed distribution (tail extends right)")
elif skewness < 0:
    print("  → Left-skewed distribution (tail extends left)")
else:
    print("  → Symmetric distribution")
```

### Energy and Robust Features

```python
print("\nEnergy and Robust Features:")
print(f"  Energy: {fo_dict.get('Energy', 'N/A'):.2f}")
print(f"  Total Energy: {fo_dict.get('TotalEnergy', 'N/A'):.2f}")
print(f"  Root Mean Squared: {fo_dict.get('RootMeanSquared', 'N/A'):.2f}")
print(f"  Mean Absolute Deviation (MAD): {fo_dict.get('MeanAbsoluteDeviation', 'N/A'):.2f}")
print(f"  Robust Mean: {fo_dict.get('RobustMeanAbsoluteDeviation', 'N/A'):.2f}")
print(f"  Interquartile Range: {fo_dict.get('InterquartileRange', 'N/A'):.2f}")
print(f"  10th Percentile: {fo_dict.get('10Percentile', 'N/A'):.2f}")
print(f"  90th Percentile: {fo_dict.get('90Percentile', 'N/A'):.2f}")
```

---

## Texture Features

### Gray Level Co-occurrence Matrix (GLCM)

```python
# Extract GLCM features
extractor_glcm = featureextractor.RadiomicsFeatureExtractor()
extractor_glcm.disableAllFeatures()
extractor_glcm.enableFeatureClassByName('glcm')

glcm_result = extractor_glcm.execute(image_path, mask_path)

glcm_dict = {k.split('_')[-1]: v for k, v in glcm_result.items()
             if 'glcm' in k and not k.startswith('diagnostics')}

print("GLCM Features (Texture):")
print(f"  Contrast: {glcm_dict.get('Contrast', 'N/A'):.3f}")
print(f"  Correlation: {glcm_dict.get('Correlation', 'N/A'):.3f}")
print(f"  Energy (ASM): {glcm_dict.get('JointEnergy', 'N/A'):.3f}")
print(f"  Homogeneity: {glcm_dict.get('JointAverage', 'N/A'):.3f}")
print(f"  Entropy: {glcm_dict.get('JointEntropy', 'N/A'):.3f}")
print(f"  Dissimilarity: {glcm_dict.get('Dissimilarity', 'N/A'):.3f}")

# High contrast = heterogeneous texture
# High correlation = linear relationship between voxels
# High energy = uniform texture
# High entropy = complex/random texture
```

### Gray Level Run Length Matrix (GLRLM)

```python
# Extract GLRLM features
extractor_glrlm = featureextractor.RadiomicsFeatureExtractor()
extractor_glrlm.disableAllFeatures()
extractor_glrlm.enableFeatureClassByName('glrlm')

glrlm_result = extractor_glrlm.execute(image_path, mask_path)

glrlm_dict = {k.split('_')[-1]: v for k, v in glrlm_result.items()
              if 'glrlm' in k and not k.startswith('diagnostics')}

print("\nGLRLM Features (Run Length):")
print(f"  Short Run Emphasis: {glrlm_dict.get('ShortRunEmphasis', 'N/A'):.3f}")
print(f"  Long Run Emphasis: {glrlm_dict.get('LongRunEmphasis', 'N/A'):.3f}")
print(f"  Gray Level Non-Uniformity: {glrlm_dict.get('GrayLevelNonUniformity', 'N/A'):.3f}")
print(f"  Run Length Non-Uniformity: {glrlm_dict.get('RunLengthNonUniformity', 'N/A'):.3f}")
print(f"  Run Percentage: {glrlm_dict.get('RunPercentage', 'N/A'):.3f}")
print(f"  Run Variance: {glrlm_dict.get('RunVariance', 'N/A'):.3f}")
```

### Gray Level Size Zone Matrix (GLSZM)

```python
# Extract GLSZM features
extractor_glszm = featureextractor.RadiomicsFeatureExtractor()
extractor_glszm.disableAllFeatures()
extractor_glszm.enableFeatureClassByName('glszm')

glszm_result = extractor_glszm.execute(image_path, mask_path)

glszm_dict = {k.split('_')[-1]: v for k, v in glszm_result.items()
              if 'glszm' in k and not k.startswith('diagnostics')}

print("\nGLSZM Features (Size Zone):")
print(f"  Small Area Emphasis: {glszm_dict.get('SmallAreaEmphasis', 'N/A'):.3f}")
print(f"  Large Area Emphasis: {glszm_dict.get('LargeAreaEmphasis', 'N/A'):.3f}")
print(f"  Gray Level Non-Uniformity: {glszm_dict.get('GrayLevelNonUniformity', 'N/A'):.3f}")
print(f"  Size Zone Non-Uniformity: {glszm_dict.get('SizeZoneNonUniformity', 'N/A'):.3f}")
print(f"  Zone Percentage: {glszm_dict.get('ZonePercentage', 'N/A'):.3f}")
```

### GLDM and NGTDM

```python
# Gray Level Dependence Matrix (GLDM)
extractor_gldm = featureextractor.RadiomicsFeatureExtractor()
extractor_gldm.disableAllFeatures()
extractor_gldm.enableFeatureClassByName('gldm')

gldm_result = extractor_gldm.execute(image_path, mask_path)

# Neighboring Gray Tone Difference Matrix (NGTDM)
extractor_ngtdm = featureextractor.RadiomicsFeatureExtractor()
extractor_ngtdm.disableAllFeatures()
extractor_ngtdm.enableFeatureClassByName('ngtdm')

ngtdm_result = extractor_ngtdm.execute(image_path, mask_path)

print("\nGLDM and NGTDM Features extracted")
print(f"GLDM features: {sum(1 for k in gldm_result.keys() if 'gldm' in k and not k.startswith('diagnostics'))}")
print(f"NGTDM features: {sum(1 for k in ngtdm_result.keys() if 'ngtdm' in k and not k.startswith('diagnostics'))}")
```

---

## Image Filters

### Wavelet Decomposition

```python
# Extract wavelet features
extractor_wavelet = featureextractor.RadiomicsFeatureExtractor()

# Enable wavelet
extractor_wavelet.enableImageTypeByName('Wavelet')
extractor_wavelet.disableAllFeatures()
extractor_wavelet.enableFeatureClassByName('firstorder')

wavelet_result = extractor_wavelet.execute(image_path, mask_path)

# Wavelet creates 8 decompositions (HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL)
wavelet_features = {k: v for k, v in wavelet_result.items()
                    if 'wavelet' in k and not k.startswith('diagnostics')}

print(f"\nWavelet Features: {len(wavelet_features)}")
print("Example wavelet features:")
for i, (key, value) in enumerate(list(wavelet_features.items())[:5]):
    print(f"  {key}: {value:.3f}")
```

### Laplacian of Gaussian (LoG)

```python
# Extract LoG features
extractor_log = featureextractor.RadiomicsFeatureExtractor()

# Enable LoG with different sigma values
extractor_log.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0]})
extractor_log.disableAllFeatures()
extractor_log.enableFeatureClassByName('firstorder')
extractor_log.enableFeatureClassByName('glcm')

log_result = extractor_log.execute(image_path, mask_path)

log_features = {k: v for k, v in log_result.items()
                if 'log-sigma' in k and not k.startswith('diagnostics')}

print(f"\nLoG Features: {len(log_features)}")
print("LoG features at different scales (sigma 1.0, 2.0, 3.0)")
```

### Gradient and Other Filters

```python
# Square filter (enhances high intensities)
extractor_square = featureextractor.RadiomicsFeatureExtractor()
extractor_square.enableImageTypeByName('Square')
extractor_square.disableAllFeatures()
extractor_square.enableFeatureClassByName('firstorder')

square_result = extractor_square.execute(image_path, mask_path)

# Square root filter
extractor_sqrt = featureextractor.RadiomicsFeatureExtractor()
extractor_sqrt.enableImageTypeByName('SquareRoot')
extractor_sqrt.disableAllFeatures()
extractor_sqrt.enableFeatureClassByName('firstorder')

sqrt_result = extractor_sqrt.execute(image_path, mask_path)

# Gradient magnitude
extractor_gradient = featureextractor.RadiomicsFeatureExtractor()
extractor_gradient.enableImageTypeByName('Gradient')
extractor_gradient.disableAllFeatures()
extractor_gradient.enableFeatureClassByName('firstorder')

gradient_result = extractor_gradient.execute(image_path, mask_path)

print("\nFilter-Based Features:")
print(f"  Square filter features: {sum(1 for k in square_result if 'square' in k)}")
print(f"  Square root features: {sum(1 for k in sqrt_result if 'squareroot' in k)}")
print(f"  Gradient features: {sum(1 for k in gradient_result if 'gradient' in k)}")
```

---

## Preprocessing Configuration

### Configure Extractor with YAML

```python
import yaml

# Create parameter file
params = {
    'imageType': {
        'Original': {},
        'Wavelet': {},
        'LoG': {'sigma': [1.0, 2.0, 3.0]}
    },
    'featureClass': {
        'shape': None,
        'firstorder': None,
        'glcm': None,
        'glrlm': None,
        'glszm': None
    },
    'setting': {
        'binWidth': 25,  # Intensity discretization
        'resampledPixelSpacing': [1, 1, 1],  # Resample to 1mm isotropic
        'interpolator': 'sitkBSpline',
        'normalize': True,
        'normalizeScale': 100
    }
}

# Save to YAML
with open('radiomics_params.yaml', 'w') as f:
    yaml.dump(params, f)

# Load configuration
extractor_config = featureextractor.RadiomicsFeatureExtractor('radiomics_params.yaml')

# Extract with configuration
config_result = extractor_config.execute(image_path, mask_path)

print(f"Extracted {len(config_result)} features with custom config")
```

### Resampling and Normalization

```python
# Configure preprocessing
extractor_preproc = featureextractor.RadiomicsFeatureExtractor()

# Resampling
extractor_preproc.settings['resampledPixelSpacing'] = [1.0, 1.0, 1.0]  # 1mm isotropic
extractor_preproc.settings['interpolator'] = sitk.sitkBSpline

# Normalization
extractor_preproc.settings['normalize'] = True
extractor_preproc.settings['normalizeScale'] = 100

# Intensity discretization
extractor_preproc.settings['binWidth'] = 25  # Fixed bin width
# OR
# extractor_preproc.settings['binCount'] = 32  # Fixed number of bins

result_preproc = extractor_preproc.execute(image_path, mask_path)

print("Preprocessing configuration applied")
```

### Mask Validation

```python
# Configure mask validation
extractor_validate = featureextractor.RadiomicsFeatureExtractor()

# Minimum number of voxels in ROI
extractor_validate.settings['minimumROIDimensions'] = 2
extractor_validate.settings['minimumROISize'] = 50  # Minimum 50 voxels

# Voxel array shift (for padding)
extractor_validate.settings['voxelArrayShift'] = 0

# Distance to boundary
extractor_validate.settings['geometryTolerance'] = 0.0001

print("Mask validation configured")
```

---

## Batch Feature Extraction

### Process Multiple Subjects

```python
import pandas as pd
from pathlib import Path

def batch_extract_features(subjects_data, extractor, output_csv):
    """
    Extract features for multiple subjects

    subjects_data: List of dicts with 'subject_id', 'image', 'mask'
    """

    all_features = []

    for subject in subjects_data:
        subject_id = subject['subject_id']
        image_path = subject['image']
        mask_path = subject['mask']

        print(f"Processing {subject_id}...")

        try:
            # Extract features
            result = extractor.execute(image_path, mask_path)

            # Add subject ID
            result['subject_id'] = subject_id

            # Remove diagnostics
            features = {k: v for k, v in result.items()
                       if not k.startswith('diagnostics')}

            all_features.append(features)
            print(f"  ✓ Extracted {len(features)} features")

        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    # Move subject_id to first column
    cols = ['subject_id'] + [col for col in df.columns if col != 'subject_id']
    df = df[cols]

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved features to {output_csv}")

    return df

# Example usage
subjects = [
    {'subject_id': 'sub-01', 'image': 'sub-01_T1w.nii.gz', 'mask': 'sub-01_mask.nii.gz'},
    {'subject_id': 'sub-02', 'image': 'sub-02_T1w.nii.gz', 'mask': 'sub-02_mask.nii.gz'},
    {'subject_id': 'sub-03', 'image': 'sub-03_T1w.nii.gz', 'mask': 'sub-03_mask.nii.gz'},
]

# Create extractor
batch_extractor = featureextractor.RadiomicsFeatureExtractor()
batch_extractor.disableAllFeatures()
batch_extractor.enableFeatureClassByName('shape')
batch_extractor.enableFeatureClassByName('firstorder')

# Extract features
features_df = batch_extract_features(subjects, batch_extractor, 'radiomics_features.csv')
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def extract_single_subject(subject, params_file=None):
    """Extract features for single subject (for parallel execution)"""

    try:
        if params_file:
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
        else:
            extractor = featureextractor.RadiomicsFeatureExtractor()

        result = extractor.execute(subject['image'], subject['mask'])
        result['subject_id'] = subject['subject_id']

        # Remove diagnostics
        features = {k: v for k, v in result.items()
                   if not k.startswith('diagnostics')}

        return {'status': 'success', 'features': features}

    except Exception as e:
        return {'status': 'failed', 'subject_id': subject['subject_id'], 'error': str(e)}

def parallel_batch_extract(subjects, params_file=None, max_workers=4):
    """Extract features in parallel"""

    extract_func = partial(extract_single_subject, params_file=params_file)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_func, subject) for subject in subjects]

        for future in futures:
            result = future.result()
            if result['status'] == 'success':
                results.append(result['features'])
            else:
                print(f"✗ {result['subject_id']}: {result['error']}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    return df

# Run parallel extraction
features_df_parallel = parallel_batch_extract(subjects, max_workers=4)
features_df_parallel.to_csv('radiomics_features_parallel.csv', index=False)
```

---

## Feature Selection for Machine Learning

### Load and Prepare Features

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load features
features_df = pd.read_csv('radiomics_features.csv')

# Load outcomes (example: binary classification)
outcomes = pd.read_csv('outcomes.csv')  # Contains 'subject_id' and 'diagnosis'

# Merge features with outcomes
data = features_df.merge(outcomes, on='subject_id')

# Separate features and target
X = data.drop(['subject_id', 'diagnosis'], axis=1)
y = data['diagnosis']

# Convert to numeric (handle any non-numeric columns)
X = X.apply(pd.to_numeric, errors='coerce')

# Handle missing values
X = X.fillna(X.median())

print(f"Features shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")
```

### Correlation Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
correlation_matrix = X.corr()

# Find highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"Highly correlated feature pairs (|r| > 0.9): {len(high_corr_pairs)}")

# Plot correlation heatmap (subset)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix.iloc[:20, :20], annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix (subset)')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=150)
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Univariate feature selection
selector = SelectKBest(f_classif, k=20)  # Select top 20 features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()

print(f"Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features[:10]):
    print(f"  {i+1}. {feat}")

# Feature importance scores
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("\nTop 10 features by F-statistic:")
print(feature_scores.head(10))
```

### Train ML Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\nClassification Results:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))
```

---

## Quality Control and IBSI Compliance

### IBSI Compliance Testing

```python
# PyRadiomics is IBSI compliant
# Test on IBSI phantom data

# Enable verbose logging for compliance checking
import logging

logger = logging.getLogger('radiomics')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Extract features with logging
extractor_ibsi = featureextractor.RadiomicsFeatureExtractor()
result_ibsi = extractor_ibsi.execute(image_path, mask_path)

print("IBSI compliance: PyRadiomics follows IBSI feature definitions")
```

### Feature Reproducibility

```python
# Test feature reproducibility
def test_reproducibility(image_path, mask_path, n_runs=5):
    """Test if features are reproducible"""

    extractor = featureextractor.RadiomicsFeatureExtractor()

    results = []
    for i in range(n_runs):
        result = extractor.execute(image_path, mask_path)
        features = {k: v for k, v in result.items() if not k.startswith('diagnostics')}
        results.append(features)

    # Check if all runs produce identical results
    reference = results[0]
    reproducible = True

    for i, result in enumerate(results[1:], start=1):
        for key in reference.keys():
            if isinstance(reference[key], (int, float)):
                if not np.isclose(reference[key], result[key]):
                    print(f"Run {i}: {key} differs: {reference[key]} vs {result[key]}")
                    reproducible = False

    if reproducible:
        print("✓ Features are perfectly reproducible")
    else:
        print("✗ Some features vary across runs")

    return reproducible

# Test
test_reproducibility(image_path, mask_path)
```

---

## Advanced Applications

### Multi-Region Analysis

```python
# Extract features from multiple ROIs in same scan
def extract_multi_region_features(image_path, mask_path, label_values):
    """Extract features for each label in mask"""

    extractor = featureextractor.RadiomicsFeatureExtractor()

    all_features = []

    for label_value in label_values:
        print(f"Extracting features for label {label_value}...")

        # Configure extractor for specific label
        extractor.settings['label'] = label_value

        result = extractor.execute(image_path, mask_path, label=label_value)

        features = {k: v for k, v in result.items() if not k.startswith('diagnostics')}
        features['label'] = label_value

        all_features.append(features)

    return pd.DataFrame(all_features)

# Example: Extract features for tumor core (label=1) and edema (label=2)
multi_region_features = extract_multi_region_features(
    image_path='tumor_image.nii.gz',
    mask_path='tumor_segmentation.nii.gz',
    label_values=[1, 2]
)

print(multi_region_features)
```

### Longitudinal Feature Tracking

```python
# Track radiomics features over time
def longitudinal_radiomics(subject_id, timepoints):
    """
    Extract features at multiple timepoints

    timepoints: List of dicts with 'timepoint', 'image', 'mask'
    """

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')

    longitudinal_features = []

    for tp in timepoints:
        result = extractor.execute(tp['image'], tp['mask'])
        features = {k: v for k, v in result.items() if not k.startswith('diagnostics')}
        features['subject_id'] = subject_id
        features['timepoint'] = tp['timepoint']
        longitudinal_features.append(features)

    df = pd.DataFrame(longitudinal_features)

    # Compute changes over time
    if len(df) > 1:
        for col in df.columns:
            if col not in ['subject_id', 'timepoint'] and pd.api.types.is_numeric_dtype(df[col]):
                df[f'{col}_change'] = df[col].diff()
                df[f'{col}_pct_change'] = df[col].pct_change() * 100

    return df

# Example
timepoints = [
    {'timepoint': 'baseline', 'image': 'sub-01_tp1_T1w.nii.gz', 'mask': 'sub-01_tp1_mask.nii.gz'},
    {'timepoint': '6month', 'image': 'sub-01_tp2_T1w.nii.gz', 'mask': 'sub-01_tp2_mask.nii.gz'},
    {'timepoint': '12month', 'image': 'sub-01_tp3_T1w.nii.gz', 'mask': 'sub-01_tp3_mask.nii.gz'},
]

longitudinal_df = longitudinal_radiomics('sub-01', timepoints)
print("\nLongitudinal feature changes:")
print(longitudinal_df[['timepoint', 'original_shape_Volume', 'original_shape_Volume_change']])
```

---

## Troubleshooting

### Common Issues

```python
# Issue: Empty ROI or ROI too small
# Check ROI size before extraction

mask_array = sitk.GetArrayFromImage(mask)
roi_size = (mask_array > 0).sum()

print(f"ROI size: {roi_size} voxels")

if roi_size < 50:
    print("⚠ Warning: ROI is very small (<50 voxels)")
    print("Some features may not be reliable")

# Issue: All features return NaN
# Check image and mask alignment

image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

print(f"Image size: {image.GetSize()}")
print(f"Mask size: {mask.GetSize()}")
print(f"Image spacing: {image.GetSpacing()}")
print(f"Mask spacing: {mask.GetSpacing()}")
print(f"Image origin: {image.GetOrigin()}")
print(f"Mask origin: {mask.GetOrigin()}")

if image.GetSize() != mask.GetSize():
    print("✗ Image and mask size mismatch!")
```

### Debugging Feature Extraction

```python
# Enable debug logging
import logging

logger = logging.getLogger('radiomics')
logger.setLevel(logging.DEBUG)

# Extract with detailed logging
extractor_debug = featureextractor.RadiomicsFeatureExtractor()

try:
    result_debug = extractor_debug.execute(image_path, mask_path)
    print("✓ Feature extraction successful")
except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
```

---

## Best Practices

### Recommended Workflow

```python
# 1. Quality control of input data
# 2. Standardize preprocessing (resampling, normalization)
# 3. Configure feature extraction consistently
# 4. Extract features in batch
# 5. Quality check extracted features
# 6. Feature selection before ML
# 7. Report parameters used

# Example comprehensive workflow
params_recommended = {
    'imageType': {'Original': {}, 'Wavelet': {}},
    'featureClass': {'shape': None, 'firstorder': None, 'glcm': None},
    'setting': {
        'binWidth': 25,
        'resampledPixelSpacing': [1, 1, 1],
        'interpolator': 'sitkBSpline',
        'normalize': True
    }
}

print("Recommended workflow:")
print("1. Use consistent preprocessing across all subjects")
print("2. Document all parameters")
print("3. Test on subset before full cohort")
print("4. Validate feature reproducibility")
print("5. Use IBSI-compliant features")
```

---

## References

### Key Publications

1. van Griethuysen, J. J., et al. (2017). "Computational Radiomics System to Decode the Radiographic Phenotype." *Cancer Research*, 77(21), e104-e107.

2. Zwanenburg, A., et al. (2020). "The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping." *Radiology*, 295(2), 328-338.

### Documentation and Resources

- **Documentation**: https://pyradiomics.readthedocs.io/
- **GitHub**: https://github.com/AIM-Harvard/pyradiomics
- **IBSI**: https://theibsi.github.io/
- **Example Data**: Available in repository
- **Tutorials**: https://pyradiomics.readthedocs.io/en/latest/usage.html

### Related Tools

- **TorchIO**: Medical image preprocessing for DL
- **NeuroHarmonize**: Multi-site harmonization
- **Nilearn**: ML for neuroimaging
- **scikit-learn**: Machine learning framework
- **SimpleITK**: Medical image processing

---

## See Also

- **torchio.md**: Medical image preprocessing
- **neuroharmonize.md**: Multi-site harmonization
- **nilearn.md**: Machine learning for neuroimaging
- **monai.md**: Deep learning for medical imaging
- **nnu-net.md**: Segmentation framework

## Citation

```bibtex
@article{van2017computational,
  title={Computational radiomics system to decode the radiographic phenotype},
  author={van Griethuysen, Joost and Fedorov, Andriy and Parmar, Chintan and others},
  journal={Cancer Research},
  volume={77},
  number={21},
  pages={e104--e107},
  year={2017},
  doi={10.1158/0008-5472.CAN-17-0339}
}
```
