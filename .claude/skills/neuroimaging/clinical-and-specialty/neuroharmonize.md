# NeuroHarmonize: ComBat Harmonization for Multi-Site Neuroimaging

## Overview

**NeuroHarmonize** is a Python implementation of the ComBat harmonization method for removing scanner and site effects from neuroimaging data while preserving biological variance. Originally developed for genomics batch effect correction and adapted for neuroimaging by Johnson, Fortin, and colleagues, ComBat uses empirical Bayes to harmonize data across sites. NeuroHarmonize provides ROI-based and voxel-wise harmonization with support for covariates, COMBAT-GAM for non-linear effects, and out-of-sample harmonization for new data.

### Key Features

- **ComBat Harmonization**: Empirical Bayes batch effect removal
- **Preserve Biological Variance**: Maintain age, sex, diagnosis effects
- **Covariate Support**: Model biological and technical variables
- **COMBAT-GAM**: Non-linear site effect modeling with smooth terms
- **ROI-Based**: Harmonize FreeSurfer, cortical thickness, DTI metrics
- **Voxel-Wise**: Whole-brain harmonization for images
- **Out-of-Sample**: Apply trained model to new subjects/sites
- **Missing Data**: Handle incomplete covariate information
- **Statistical Validation**: Built-in QC and visualization tools
- **Integration Ready**: Works with pandas, scikit-learn, neuroimaging pipelines

### Scientific Foundation

Multi-site neuroimaging studies face systematic differences due to:
- **Scanner manufacturer** (Siemens, Philips, GE)
- **Field strength** (1.5T, 3T, 7T)
- **Pulse sequences** and acquisition parameters
- **Scanner software** versions and upgrades
- **Site-specific protocols**

ComBat addresses these by:
1. **Modeling site effects** as additive and multiplicative shifts
2. **Estimating parameters** using empirical Bayes shrinkage
3. **Removing site variance** while preserving biological associations
4. **Standardizing distributions** across sites

### Primary Use Cases

1. **Multi-Site Studies**: ENIGMA, ADNI, UK Biobank consortia
2. **Meta-Analysis**: Combine data from multiple datasets
3. **Machine Learning**: Train robust models across scanners
4. **Longitudinal Studies**: Handle scanner upgrades
5. **Clinical Trials**: Multi-center harmonization
6. **Data Pooling**: Increase sample size with heterogeneous data

---

## Installation

### Using pip

```bash
# Install NeuroHarmonize
pip install neuroHarmonize

# Verify installation
python -c "import neuroHarmonize; print('NeuroHarmonize installed successfully')"
```

### From GitHub

```bash
# Clone repository
git clone https://github.com/rpomponio/neuroHarmonize.git
cd neuroHarmonize

# Install
pip install -e .

# Verify
python -c "from neuroHarmonize import harmonizationLearn; print('Installation successful')"
```

### Dependencies

NeuroHarmonize requires:
- Python ≥ 3.6
- NumPy, pandas
- scikit-learn
- statsmodels (for COMBAT-GAM)
- matplotlib, seaborn (for visualization)

---

## Understanding Harmonization

### When to Harmonize

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load multi-site data
# Example: cortical thickness from 3 sites
data = pd.DataFrame({
    'subject_id': [f'sub-{i:03d}' for i in range(1, 151)],
    'site': ['Site_A']*50 + ['Site_B']*50 + ['Site_C']*50,
    'age': np.random.randint(20, 80, 150),
    'sex': np.random.choice(['M', 'F'], 150),
    'diagnosis': np.random.choice(['control', 'patient'], 150),
    'thickness_left_frontal': np.random.randn(150) * 0.3 + 2.5
})

# Add site-specific biases
data.loc[data['site'] == 'Site_A', 'thickness_left_frontal'] += 0.3  # Site A higher
data.loc[data['site'] == 'Site_B', 'thickness_left_frontal'] -= 0.2  # Site B lower

# Visualize site effects
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=data, x='site', y='thickness_left_frontal')
plt.title('Cortical Thickness by Site (Before Harmonization)')
plt.ylabel('Thickness (mm)')

# Statistical test for site effect
from scipy import stats
sites = data.groupby('site')['thickness_left_frontal'].apply(list)
f_stat, p_value = stats.f_oneway(*sites)
plt.text(0.5, 0.95, f'ANOVA p={p_value:.4f}', transform=plt.gca().transAxes)

print(f"Site effect p-value: {p_value:.4f}")
if p_value < 0.05:
    print("→ Significant site effect detected, harmonization recommended")
```

### ComBat Methodology Overview

```python
# ComBat model:
# Y_ijv = α_v + X_ij·β_v + γ_i + δ_i·ε_ijv
#
# Where:
# - Y_ijv: Feature v for subject j at site i
# - α_v: Overall mean for feature v
# - X_ij·β_v: Covariate effects (age, sex, diagnosis)
# - γ_i: Additive site effect (location shift)
# - δ_i: Multiplicative site effect (scale change)
# - ε_ijv: Residual error

# Empirical Bayes shrinkage pools information across features
# to obtain stable site effect estimates

print("ComBat Harmonization Steps:")
print("1. Standardize data to common mean/variance")
print("2. Estimate site-specific location (γ) and scale (δ) parameters")
print("3. Pool information across features using empirical Bayes")
print("4. Remove site effects while preserving covariates")
print("5. Back-transform to original scale")
```

---

## Basic Harmonization

### Prepare Data

```python
from neuroHarmonize import harmonizationLearn
import pandas as pd
import numpy as np

# Load features (ROI-based: subjects × features)
# Example: FreeSurfer cortical thickness for multiple ROIs
features = pd.DataFrame({
    'lh_superiorfrontal_thickness': np.random.randn(100) * 0.3 + 2.5,
    'lh_middletemporal_thickness': np.random.randn(100) * 0.3 + 2.8,
    'rh_superiorfrontal_thickness': np.random.randn(100) * 0.3 + 2.5,
    'rh_middletemporal_thickness': np.random.randn(100) * 0.3 + 2.8,
})

# Covariates DataFrame
covars = pd.DataFrame({
    'SITE': ['Site_A']*30 + ['Site_B']*40 + ['Site_C']*30,
    'AGE': np.random.randint(20, 80, 100),
    'SEX': np.random.choice([0, 1], 100),  # 0=F, 1=M
})

print(f"Features shape: {features.shape}")
print(f"Covariates shape: {covars.shape}")
print(f"Sites: {covars['SITE'].unique()}")
```

### Run ComBat Harmonization

```python
# Harmonize data
# Note: AGE and SEX will be preserved as biological effects
model, harmonized_data = harmonizationLearn(features.values, covars)

# Convert back to DataFrame
harmonized_df = pd.DataFrame(
    harmonized_data,
    columns=features.columns,
    index=features.index
)

print(f"\nHarmonized data shape: {harmonized_df.shape}")
print(f"Original mean: {features.mean().mean():.3f}")
print(f"Harmonized mean: {harmonized_df.mean().mean():.3f}")
```

### Visualize Harmonization Effect

```python
# Compare before/after for single feature
feature_name = 'lh_superiorfrontal_thickness'

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before harmonization
ax1 = axes[0]
for site in covars['SITE'].unique():
    site_mask = covars['SITE'] == site
    ax1.scatter(covars.loc[site_mask, 'AGE'],
                features.loc[site_mask, feature_name],
                label=site, alpha=0.6)
ax1.set_xlabel('Age')
ax1.set_ylabel('Thickness (mm)')
ax1.set_title('Before Harmonization')
ax1.legend()

# After harmonization
ax2 = axes[1]
for site in covars['SITE'].unique():
    site_mask = covars['SITE'] == site
    ax2.scatter(covars.loc[site_mask, 'AGE'],
                harmonized_df.loc[site_mask, feature_name],
                label=site, alpha=0.6)
ax2.set_xlabel('Age')
ax2.set_ylabel('Thickness (mm)')
ax2.set_title('After Harmonization')
ax2.legend()

plt.tight_layout()
plt.savefig('harmonization_comparison.png', dpi=150)
print("Saved harmonization comparison plot")
```

---

## Covariate Specification

### Biological Covariates

```python
# Specify covariates to preserve
# Typically: age, sex, diagnosis/group

# Continuous covariate (AGE)
# Categorical covariate (SEX, DIAGNOSIS)

covars_full = pd.DataFrame({
    'SITE': ['Site_A']*30 + ['Site_B']*40 + ['Site_C']*30,
    'AGE': np.random.randint(20, 80, 100),
    'SEX': np.random.choice([0, 1], 100),
    'DIAGNOSIS': np.random.choice(['Control', 'Patient'], 100)
})

# One-hot encode categorical variables (except SITE)
covars_encoded = pd.get_dummies(
    covars_full,
    columns=['SEX', 'DIAGNOSIS'],
    drop_first=True  # Avoid multicollinearity
)

print("Covariates with encoding:")
print(covars_encoded.head())

# Harmonize with biological covariates preserved
model, harmonized_bio = harmonizationLearn(features.values, covars_encoded)

print("\n✓ Harmonized while preserving AGE, SEX, and DIAGNOSIS effects")
```

### Interaction Terms

```python
# Model interaction between site and age
# Useful if site effects vary with age

# Create age × site interaction
covars_interaction = covars.copy()
covars_interaction['AGE_squared'] = covars_interaction['AGE'] ** 2

print("Covariates with interaction term:")
print(covars_interaction.head())

# Note: Advanced interactions should be created before harmonization
# NeuroHarmonize will preserve these effects if included in covars
```

---

## COMBAT-GAM (Non-Linear Effects)

### When to Use GAM

```python
# Use COMBAT-GAM when:
# 1. Site effects vary non-linearly with covariates
# 2. Age effects are non-linear
# 3. More complex site-by-covariate interactions

from neuroHarmonize import harmonizationLearn

# Run COMBAT-GAM with smooth age effect
# Note: Requires statsmodels

# Specify smooth terms (example syntax)
# smooth_terms = ['AGE']  # Smooth age effect

# For now, NeuroHarmonize uses standard ComBat
# GAM extension requires additional configuration

print("COMBAT-GAM:")
print("- Models non-linear relationships")
print("- Uses generalized additive models")
print("- Better for complex age-site interactions")
print("- Requires more data per site")
```

---

## Validation and Assessment

### Statistical Tests

```python
from scipy import stats

def assess_harmonization(data_before, data_after, covars, feature_name):
    """Assess harmonization quality"""

    print(f"\nAssessing harmonization for: {feature_name}")

    # Test 1: Site effect before harmonization
    sites_before = [data_before.loc[covars['SITE'] == site, feature_name].values
                   for site in covars['SITE'].unique()]
    f_before, p_before = stats.f_oneway(*sites_before)

    # Test 2: Site effect after harmonization
    sites_after = [data_after.loc[covars['SITE'] == site, feature_name].values
                  for site in covars['SITE'].unique()]
    f_after, p_after = stats.f_oneway(*sites_after)

    print(f"  Site effect before: F={f_before:.2f}, p={p_before:.4f}")
    print(f"  Site effect after:  F={f_after:.2f}, p={p_after:.4f}")

    if p_before < 0.05 and p_after > 0.05:
        print("  ✓ Successfully removed site effect")
    elif p_after < 0.05:
        print("  ⚠ Residual site effect remains")

    # Test 3: Preserve age effect
    from scipy.stats import pearsonr

    age = covars['AGE'].values
    r_before, p_age_before = pearsonr(age, data_before[feature_name])
    r_after, p_age_after = pearsonr(age, data_after[feature_name])

    print(f"  Age correlation before: r={r_before:.3f}, p={p_age_before:.4f}")
    print(f"  Age correlation after:  r={r_after:.3f}, p={p_age_after:.4f}")

    if abs(r_before - r_after) < 0.1:
        print("  ✓ Age effect preserved")
    else:
        print(f"  ⚠ Age effect changed by {abs(r_before - r_after):.3f}")

# Assess harmonization
assess_harmonization(features, harmonized_df, covars, 'lh_superiorfrontal_thickness')
```

### PCA Visualization

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def visualize_harmonization_pca(data_before, data_after, covars):
    """Visualize harmonization effect using PCA"""

    # Standardize for PCA
    scaler = StandardScaler()

    # PCA before harmonization
    data_before_scaled = scaler.fit_transform(data_before)
    pca_before = PCA(n_components=2)
    pc_before = pca_before.fit_transform(data_before_scaled)

    # PCA after harmonization
    data_after_scaled = scaler.fit_transform(data_after)
    pca_after = PCA(n_components=2)
    pc_after = pca_after.fit_transform(data_after_scaled)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before harmonization
    for site in covars['SITE'].unique():
        mask = covars['SITE'] == site
        axes[0].scatter(pc_before[mask, 0], pc_before[mask, 1],
                       label=site, alpha=0.6, s=50)
    axes[0].set_xlabel(f'PC1 ({pca_before.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca_before.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('Before Harmonization')
    axes[0].legend()

    # After harmonization
    for site in covars['SITE'].unique():
        mask = covars['SITE'] == site
        axes[1].scatter(pc_after[mask, 0], pc_after[mask, 1],
                       label=site, alpha=0.6, s=50)
    axes[1].set_xlabel(f'PC1 ({pca_after.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca_after.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('After Harmonization')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('pca_harmonization.png', dpi=150)

    print("\n✓ PCA visualization saved")
    print("  Sites should overlap more after harmonization")

# Visualize
visualize_harmonization_pca(features, harmonized_df, covars)
```

### Distribution Comparison

```python
# Compare distributions before/after harmonization
def plot_distributions(data_before, data_after, covars, feature_name):
    """Plot feature distributions by site"""

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Before harmonization
    for site in covars['SITE'].unique():
        mask = covars['SITE'] == site
        axes[0].hist(data_before.loc[mask, feature_name],
                    alpha=0.5, bins=20, label=site, density=True)
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel('Density')
    axes[0].set_title('Before Harmonization')
    axes[0].legend()

    # After harmonization
    for site in covars['SITE'].unique():
        mask = covars['SITE'] == site
        axes[1].hist(data_after.loc[mask, feature_name],
                    alpha=0.5, bins=20, label=site, density=True)
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel('Density')
    axes[1].set_title('After Harmonization')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=150)

plot_distributions(features, harmonized_df, covars, 'lh_superiorfrontal_thickness')
```

---

## Out-of-Sample Harmonization

### Train Harmonization Model

```python
from neuroHarmonize import harmonizationApply

# Training set (used to learn harmonization parameters)
train_features = features.iloc[:80]
train_covars = covars.iloc[:80]

# Learn harmonization model
model, train_harmonized = harmonizationLearn(
    train_features.values,
    train_covars
)

print(f"Training set: {train_features.shape[0]} subjects")
print(f"Model learned from {train_covars['SITE'].nunique()} sites")

# Save model for later use
import pickle

with open('combat_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✓ Model saved to combat_model.pkl")
```

### Apply to New Data

```python
# Test set (new subjects from same sites)
test_features = features.iloc[80:]
test_covars = covars.iloc[80:]

# Apply pre-trained model
test_harmonized = harmonizationApply(
    test_features.values,
    test_covars,
    model
)

# Convert to DataFrame
test_harmonized_df = pd.DataFrame(
    test_harmonized,
    columns=features.columns,
    index=test_features.index
)

print(f"\nTest set: {test_features.shape[0]} subjects harmonized")
print(f"Using model trained on {train_features.shape[0]} subjects")
```

### Apply to New Site

```python
# New site (unseen during training)
# ComBat can't directly harmonize completely new sites
# Options:
# 1. Retrain model including new site
# 2. Use reference-based harmonization
# 3. Map new site to most similar existing site

# Example: Map new site to existing site
new_site_features = pd.DataFrame({
    'lh_superiorfrontal_thickness': np.random.randn(10) * 0.3 + 2.7,
    'lh_middletemporal_thickness': np.random.randn(10) * 0.3 + 3.0,
    'rh_superiorfrontal_thickness': np.random.randn(10) * 0.3 + 2.7,
    'rh_middletemporal_thickness': np.random.randn(10) * 0.3 + 3.0,
})

new_site_covars = pd.DataFrame({
    'SITE': ['Site_A'] * 10,  # Map to Site_A
    'AGE': np.random.randint(20, 80, 10),
    'SEX': np.random.choice([0, 1], 10)
})

# Apply harmonization
new_site_harmonized = harmonizationApply(
    new_site_features.values,
    new_site_covars,
    model
)

print("\n⚠ Note: New sites require careful validation")
print("  Consider retraining model with new site included")
```

---

## ROI-Based Harmonization

### FreeSurfer Cortical Thickness

```python
# Load FreeSurfer-derived cortical thickness
# Example: aparc.stats output

def load_freesurfer_stats(subjects_dir, subjects, measure='thickness'):
    """Load FreeSurfer statistics for multiple subjects"""

    all_data = []

    for subject in subjects:
        # In practice, parse aparc.stats files
        # Here we simulate the data

        subject_data = {
            'subject_id': subject,
            'lh_superiorfrontal_thickness': np.random.randn() * 0.3 + 2.5,
            'lh_middletemporal_thickness': np.random.randn() * 0.3 + 2.8,
            'rh_superiorfrontal_thickness': np.random.randn() * 0.3 + 2.5,
            'rh_middletemporal_thickness': np.random.randn() * 0.3 + 2.8,
            # ... more ROIs
        }
        all_data.append(subject_data)

    return pd.DataFrame(all_data)

# Load data
freesurfer_data = load_freesurfer_stats('./', [f'sub-{i:03d}' for i in range(1, 101)])

# Prepare for harmonization (drop subject_id)
fs_features = freesurfer_data.drop('subject_id', axis=1)

# Harmonize
model_fs, harmonized_fs = harmonizationLearn(fs_features.values, covars)

print(f"Harmonized {fs_features.shape[1]} FreeSurfer ROIs")
```

### DTI Metrics

```python
# Harmonize diffusion metrics (FA, MD, AD, RD)

dti_data = pd.DataFrame({
    'FA_corpus_callosum': np.random.randn(100) * 0.05 + 0.5,
    'MD_corpus_callosum': np.random.randn(100) * 0.1 + 0.7,
    'FA_cingulum': np.random.randn(100) * 0.05 + 0.45,
    'MD_cingulum': np.random.randn(100) * 0.1 + 0.75,
})

# Harmonize DTI metrics
model_dti, harmonized_dti = harmonizationLearn(dti_data.values, covars)

print(f"Harmonized {dti_data.shape[1]} DTI metrics")
```

---

## Integration with ML Pipelines

### Harmonize Before ML

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Prepare data
X = harmonized_df.copy()
y = np.random.choice([0, 1], len(X))  # Binary classification (example)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Evaluate
print("\nClassification on Harmonized Data:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")
```

### Cross-Validation with Harmonization

```python
from sklearn.model_selection import StratifiedKFold

# Proper cross-validation with harmonization
# IMPORTANT: Harmonize within each fold to avoid data leakage

def cv_with_harmonization(X, y, covars, n_folds=5):
    """Cross-validation with proper harmonization"""

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{n_folds}...")

        # Split data
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        y_test_fold = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]

        covars_train = covars.iloc[train_idx]
        covars_test = covars.iloc[test_idx]

        # Harmonize within fold
        model_fold, X_train_harm = harmonizationLearn(X_train_fold.values, covars_train)
        X_test_harm = harmonizationApply(X_test_fold.values, covars_test, model_fold)

        # Train model
        clf_fold = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_fold.fit(X_train_harm, y_train_fold)

        # Predict
        y_pred_fold = clf_fold.predict(X_test_harm)
        y_pred_proba_fold = clf_fold.predict_proba(X_test_harm)[:, 1]

        # Score
        auc = roc_auc_score(y_test_fold, y_pred_proba_fold)
        fold_scores.append(auc)

        print(f"  Fold {fold + 1} AUC: {auc:.3f}")

    print(f"\nMean CV AUC: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    return fold_scores

# Run CV (with dummy binary target)
y_binary = pd.Series(np.random.choice([0, 1], len(features)))
cv_scores = cv_with_harmonization(features, y_binary, covars, n_folds=5)
```

---

## Quality Control

### Residual Site Effects

```python
def check_residual_site_effects(harmonized_data, covars):
    """Check for remaining site effects after harmonization"""

    print("\nChecking for residual site effects...")

    significant_features = []

    for col in harmonized_data.columns:
        sites = [harmonized_data.loc[covars['SITE'] == site, col].values
                for site in covars['SITE'].unique()]

        f_stat, p_val = stats.f_oneway(*sites)

        if p_val < 0.05:
            significant_features.append((col, p_val))

    if significant_features:
        print(f"⚠ {len(significant_features)} features still show site effects (p<0.05):")
        for feat, p in sorted(significant_features, key=lambda x: x[1])[:5]:
            print(f"  {feat}: p={p:.4f}")
    else:
        print("✓ No significant residual site effects detected")

    return len(significant_features) == 0

# Check residuals
check_residual_site_effects(harmonized_df, covars)
```

### Over-Harmonization Risk

```python
# Check if biological variance was inadvertently removed

def check_biological_preservation(data_before, data_after, covars, covariate='AGE'):
    """Check if biological effects are preserved"""

    print(f"\nChecking preservation of {covariate} effects...")

    correlations_before = []
    correlations_after = []

    for col in data_before.columns:
        r_before, _ = pearsonr(covars[covariate], data_before[col])
        r_after, _ = pearsonr(covars[covariate], data_after[col])

        correlations_before.append(r_before)
        correlations_after.append(r_after)

    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.scatter(correlations_before, correlations_after, alpha=0.6)
    plt.plot([-1, 1], [-1, 1], 'r--', label='Identity line')
    plt.xlabel(f'{covariate} correlation (before)')
    plt.ylabel(f'{covariate} correlation (after)')
    plt.title(f'{covariate} Effect Preservation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('biological_preservation.png', dpi=150)

    # Check for systematic differences
    correlation_change = np.array(correlations_after) - np.array(correlations_before)
    mean_change = np.mean(np.abs(correlation_change))

    print(f"Mean absolute change in {covariate} correlation: {mean_change:.3f}")

    if mean_change < 0.1:
        print(f"✓ {covariate} effects well preserved")
    else:
        print(f"⚠ {covariate} effects may have been altered")

# Check age preservation
check_biological_preservation(features, harmonized_df, covars, 'AGE')
```

---

## Best Practices

### When to Harmonize

```python
print("When to use NeuroHarmonize:")
print("✓ Multi-site data with known scanner/site differences")
print("✓ Combining data from different studies")
print("✓ Training ML models to generalize across sites")
print("✓ Harmonizing before mega-analysis")
print("\nWhen NOT to harmonize:")
print("✗ Single-site data")
print("✗ Site is confounded with diagnosis (all patients from one site)")
print("✗ Very small sample sizes per site (<10 subjects)")
print("✗ Site-specific research questions")
```

### Covariate Selection

```python
print("\nCovariate Selection Guidelines:")
print("1. Always include biological variables of interest:")
print("   - Age, sex, diagnosis/group")
print("2. Include technical variables if known:")
print("   - Scanner field strength, pulse sequence")
print("3. Don't include variables you want to test later")
print("4. Avoid over-modeling (keep it simple)")
print("5. Check for collinearity between covariates")
```

### Reporting Harmonization

```python
print("\nReporting Guidelines:")
print("1. State ComBat/NeuroHarmonize version used")
print("2. List all covariates included in model")
print("3. Report sites and sample sizes per site")
print("4. Show validation of harmonization (PCA, statistical tests)")
print("5. Report any features that failed to harmonize")
print("6. Describe handling of missing data")
print("7. Share harmonization parameters/model if possible")
```

---

## Troubleshooting

### Small Sample Sizes

```python
# Issue: Too few subjects per site (<10)
# Solution: Consider pooling similar sites or using alternative methods

min_site_size = covars.groupby('SITE').size().min()
print(f"Minimum site size: {min_site_size}")

if min_site_size < 10:
    print("⚠ Warning: Some sites have <10 subjects")
    print("Consider:")
    print("  - Pooling similar sites (same scanner model)")
    print("  - Using site as random effect instead of fixed")
    print("  - Alternative harmonization methods")
```

### Convergence Issues

```python
# Issue: ComBat doesn't converge
# Solution: Check data quality, outliers, and model specification

# Check for outliers
from scipy import stats

z_scores = np.abs(stats.zscore(features))
outliers = (z_scores > 3).any(axis=1)

print(f"Subjects with outlier values: {outliers.sum()}")

if outliers.sum() > 0:
    print("Consider removing or investigating outliers before harmonization")
```

---

## References

### Key Publications

1. Fortin, J. P., et al. (2017). "Harmonization of cortical thickness measurements across scanners and sites." *NeuroImage*, 167, 104-120.

2. Johnson, W. E., Li, C., & Rabinovic, A. (2007). "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics*, 8(1), 118-127.

3. Pomponio, R., et al. (2020). "Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan." *NeuroImage*, 208, 116450.

### Documentation and Resources

- **GitHub**: https://github.com/rpomponio/neuroHarmonize
- **Tutorial**: https://github.com/rpomponio/neuroHarmonize/blob/master/README.md
- **ENIGMA Protocols**: http://enigma.ini.usc.edu/protocols/imaging-protocols/
- **ComBat Paper**: Original Johnson et al. 2007 paper

### Related Tools

- **TorchIO**: Medical image preprocessing
- **PyRadiomics**: Radiomics feature extraction
- **Nilearn**: Machine learning for neuroimaging
- **CovBat**: Covariate-adjusted ComBat
- **LongCombat**: Longitudinal harmonization

---

## See Also

- **torchio.md**: Medical image preprocessing
- **pyradiomics.md**: Feature extraction
- **nilearn.md**: Machine learning for neuroimaging
- **fmriprep.md**: Preprocessing pipeline
- **freesurfer.md**: Cortical reconstruction

## Citation

```bibtex
@article{fortin2018harmonization,
  title={Harmonization of multi-site diffusion tensor imaging data},
  author={Fortin, Jean-Philippe and Cullen, N. and Sheline, Y. and others},
  journal={NeuroImage},
  volume={167},
  pages={104--120},
  year={2018},
  doi={10.1016/j.neuroimage.2017.11.024}
}
```
