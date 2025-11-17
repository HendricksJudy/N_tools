# abagen - Allen Brain Atlas Gene Expression Integration

## Overview

**abagen** is a Python toolbox for working with the Allen Human Brain Atlas (AHBA) gene expression data in neuroimaging contexts. Developed to standardize the complex preprocessing steps required to integrate AHBA microarray data with brain imaging, abagen provides automated pipelines for generating parcellated gene expression matrices that can be correlated with structural, functional, and network properties of the brain.

The Allen Human Brain Atlas contains high-resolution gene expression data from postmortem human brains (6 donors), with thousands of tissue samples profiled for ~20,000 genes using microarray technology. abagen handles probe selection, sample filtering, normalization, and assignment of expression values to brain parcels, enabling researchers to link molecular mechanisms to macroscale brain organization.

**Key Features:**
- Automated download of Allen Human Brain Atlas data
- Standardized preprocessing pipeline for AHBA microarray data
- Probe selection strategies (intensity-based filtering, differential stability)
- Sample-to-parcel assignment with distance thresholds
- Within-donor and across-donor normalization methods
- Missing data interpolation and handling
- Integration with any brain parcellation
- Support for all 6 AHBA donor brains
- Reproducible workflows with documented parameters
- Gene set enrichment analysis utilities
- Spatial correlation with brain imaging phenotypes
- Quality control and validation tools

**Primary Use Cases:**
- Generate parcellated gene expression matrices
- Correlate gene expression with brain structure/function
- Identify genes associated with imaging phenotypes
- Gene set enrichment for neuroimaging findings
- Transcriptomic signatures of brain networks
- Imaging-genetics integration
- Molecular mechanisms of brain organization

**Official Documentation:** https://abagen.readthedocs.io/

---

## Installation

### Install abagen

```bash
# Install via pip
pip install abagen

# Or install from GitHub for latest version
pip install git+https://github.com/rmarkello/abagen.git

# Verify installation
python -c "import abagen; print(abagen.__version__)"
```

### Install Dependencies

```bash
# Core dependencies
pip install numpy pandas scipy nibabel nilearn scikit-learn

# For analysis
pip install matplotlib seaborn statsmodels

# Optional: for enrichment analysis
pip install mygene gprofiler-official
```

### Download AHBA Data

```python
import abagen

# Fetch Allen Human Brain Atlas data (all 6 donors)
# This will download ~4GB of data on first run
# Data cached in ~/.cache/abagen

files = abagen.fetch_microarray(donors='all', verbose=1)

print(f"Downloaded data for {len(files)} donors")
print(f"Data location: {files[0]['microarray']}")
```

---

## Basic Gene Expression Processing

### Generate Parcellated Expression Matrix (Default Pipeline)

```python
import abagen
import pandas as pd

# Load brain parcellation (e.g., Desikan-Killiany)
from nilearn import datasets

atlas = datasets.fetch_atlas_destrieux_2009()
parcellation = atlas['maps']

# Generate parcellated gene expression matrix
# This runs the default abagen preprocessing pipeline
expression = abagen.get_expression_data(
    parcellation,
    verbose=1
)

print(f"Expression matrix shape: {expression.shape}")
# Output: (regions, genes) - e.g., (82 regions × 15,633 genes)

print(f"Regions: {expression.index}")
print(f"First 5 genes: {expression.columns[:5]}")

# Save to file
expression.to_csv('desikan_gene_expression.csv')
```

### Use with Custom Parcellation

```python
# Use Schaefer 400-region parcellation
atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
parcellation_schaefer = atlas_schaefer['maps']

# Generate expression for Schaefer parcellation
expression_schaefer = abagen.get_expression_data(
    parcellation_schaefer,
    verbose=1
)

print(f"Schaefer expression matrix: {expression_schaefer.shape}")
# Output: (400 regions × 15,633 genes)
```

---

## Advanced Preprocessing Options

### Probe Selection Strategies

```python
import abagen

# Default: differential stability probe selection
# Selects probe with highest similarity across donors

expression_default = abagen.get_expression_data(
    parcellation,
    probe_selection='diff_stability',  # Default
    verbose=1
)

# Alternative: intensity-based probe selection
# Selects probe with highest average expression

expression_intensity = abagen.get_expression_data(
    parcellation,
    probe_selection='max_intensity',
    verbose=1
)

# Compare number of genes retained
print(f"Differential stability: {expression_default.shape[1]} genes")
print(f"Max intensity: {expression_intensity.shape[1]} genes")
```

### Sample Filtering Options

```python
# Filter samples based on distance from parcel centroid

expression_strict = abagen.get_expression_data(
    parcellation,
    sample_norm='srs',           # Scaled robust sigmoid normalization
    gene_norm='srs',             # Same for genes
    tol=2,                       # Maximum distance (mm) from centroid
    donor_probes='aggregate',    # Aggregate probes across donors
    lr_mirror='bidirectional',   # Mirror samples across hemispheres
    missing='interpolate',       # Interpolate missing regions
    verbose=1
)

print(f"Strict filtering: {expression_strict.shape}")
```

### Normalization Methods

```python
# Scaled robust sigmoid (SRS) normalization (recommended)
expression_srs = abagen.get_expression_data(
    parcellation,
    sample_norm='srs',
    gene_norm='srs',
    verbose=1
)

# Robust sigmoid (RS) normalization
expression_rs = abagen.get_expression_data(
    parcellation,
    sample_norm='rs',
    gene_norm='rs',
    verbose=1
)

# Z-score normalization
expression_zscore = abagen.get_expression_data(
    parcellation,
    sample_norm='zscore',
    gene_norm='zscore',
    verbose=1
)

# Mixed scaling normalization
expression_mixed = abagen.get_expression_data(
    parcellation,
    sample_norm='mixed_scaling',
    gene_norm='mixed_scaling',
    verbose=1
)
```

### Handle Missing Data

```python
# Strategies for regions without samples

# Interpolate from neighboring regions (default)
expression_interp = abagen.get_expression_data(
    parcellation,
    missing='interpolate',
    verbose=1
)

# Mark missing regions as NaN
expression_nan = abagen.get_expression_data(
    parcellation,
    missing='centroids',  # Assign to nearest sample
    verbose=1
)

# Check for missing data
print(f"Missing regions (NaN): {expression_nan.isna().any(axis=1).sum()}")
```

---

## Gene-Brain Phenotype Correlations

### Correlate Gene Expression with Cortical Thickness

```python
import abagen
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from nilearn import datasets

# Load parcellation
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400)
parcellation = atlas['maps']

# Generate gene expression matrix
expression = abagen.get_expression_data(parcellation, verbose=1)

# Example: parcellated cortical thickness (from FreeSurfer)
# In practice, load real data
cortical_thickness = pd.Series(
    np.random.randn(400),
    index=expression.index,  # Match region labels
    name='thickness'
)

# Correlate each gene with cortical thickness
correlations = []

for gene in expression.columns:
    gene_expr = expression[gene]

    # Remove NaN values
    mask = ~(gene_expr.isna() | cortical_thickness.isna())

    if mask.sum() < 10:  # Require minimum samples
        continue

    r, p = spearmanr(gene_expr[mask], cortical_thickness[mask])

    correlations.append({
        'gene': gene,
        'r': r,
        'p': p
    })

# Create results DataFrame
corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('r', key=abs, ascending=False)

# Multiple comparison correction
from statsmodels.stats.multitest import multipletests
_, p_fdr, _, _ = multipletests(corr_df['p'], method='fdr_bh')
corr_df['p_fdr'] = p_fdr

# Top positively associated genes
print("Top genes positively associated with cortical thickness:")
print(corr_df[corr_df['p_fdr'] < 0.05].head(10))

# Top negatively associated genes
print("\nTop genes negatively associated with cortical thickness:")
print(corr_df[corr_df['p_fdr'] < 0.05].tail(10))
```

### Link Gene Expression to Functional Connectivity

```python
from nilearn import connectome
import numpy as np

# Generate connectivity matrix from resting-state fMRI
# (In practice, use real timeseries from fMRIPrep)
timeseries = np.random.randn(200, 400)  # 200 TRs, 400 regions

correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]

# Compute functional connectivity strength per region
fc_strength = connectivity_matrix.mean(axis=0)

# Match to expression matrix indices
fc_strength_series = pd.Series(fc_strength, index=expression.index)

# Correlate each gene with FC strength
fc_correlations = []

for gene in expression.columns:
    gene_expr = expression[gene]
    mask = ~(gene_expr.isna() | fc_strength_series.isna())

    if mask.sum() < 10:
        continue

    r, p = spearmanr(gene_expr[mask], fc_strength_series[mask])

    fc_correlations.append({
        'gene': gene,
        'r': r,
        'p': p
    })

fc_corr_df = pd.DataFrame(fc_correlations).sort_values('r', key=abs, ascending=False)

# FDR correction
_, p_fdr, _, _ = multipletests(fc_corr_df['p'], method='fdr_bh')
fc_corr_df['p_fdr'] = p_fdr

print("Genes associated with functional connectivity strength:")
print(fc_corr_df[fc_corr_df['p_fdr'] < 0.05])
```

---

## Gene Set Enrichment Analysis

### Identify Enriched Pathways

```python
from scipy.stats import mannwhitneyu
import pandas as pd

# Genes associated with cortical thickness (from earlier)
significant_genes = corr_df[corr_df['p_fdr'] < 0.05]['gene'].tolist()

print(f"Significant genes: {len(significant_genes)}")

# Use external tools for enrichment
# Example: Gene Ontology enrichment with gprofiler

try:
    from gprofiler import GProfiler

    gp = GProfiler(return_dataframe=True)

    enrichment_results = gp.profile(
        organism='hsapiens',
        query=significant_genes,
        sources=['GO:BP', 'KEGG', 'REAC'],  # Biological process, KEGG, Reactome
        all_results=False
    )

    # Filter by significance
    enrichment_results = enrichment_results[enrichment_results['p_value'] < 0.05]

    print("Top enriched pathways:")
    print(enrichment_results[['name', 'p_value', 'intersection_size']].head(10))

except ImportError:
    print("Install gprofiler-official for enrichment analysis")
    print("pip install gprofiler-official")
```

### Cell Type Enrichment

```python
# Test for enrichment in specific cell types
# Using cell type marker genes

# Example: neuronal vs. glial markers
neuronal_markers = ['SYP', 'SYN1', 'SNAP25', 'GAD1', 'GAD2', 'SLC17A7']
glial_markers = ['GFAP', 'AQP4', 'ALDH1L1', 'MOG', 'MBP', 'PLP1']

# Check overlap with significant genes
neuronal_overlap = set(significant_genes) & set(neuronal_markers)
glial_overlap = set(significant_genes) & set(glial_markers)

print(f"Neuronal markers in significant genes: {len(neuronal_overlap)}")
print(f"Glial markers in significant genes: {len(glial_overlap)}")

# More sophisticated: use cell type transcriptomic signatures
# (requires cell type expression databases)
```

---

## Spatial Patterns and Gradients

### Compute Gene Expression Gradients

```python
from brainspace.gradient import GradientMaps
import numpy as np

# Compute gene co-expression matrix
# Correlation between regions based on gene expression profiles

gene_coexpression = expression.T.corr(method='spearman').values

# Remove NaN values
gene_coexpression = np.nan_to_num(gene_coexpression, nan=0)

# Compute gradients of gene co-expression
gm = GradientMaps(n_components=10, approach='dm', kernel='normalized_angle')
gm.fit(gene_coexpression)

gene_gradient1 = gm.gradients_[:, 0]

print(f"Gene expression gradient 1 shape: {gene_gradient1.shape}")

# Correlate with functional gradients
# (Requires functional connectivity gradients from BrainSpace)
```

### Regional Gene Expression Profiles

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Select specific genes of interest
genes_of_interest = ['HTR1A', 'HTR2A', 'DRD1', 'DRD2', 'GRIN1', 'GABBR1']

# Check which genes are available
available_genes = [g for g in genes_of_interest if g in expression.columns]

print(f"Available genes: {available_genes}")

# Plot expression heatmap
gene_subset = expression[available_genes]

plt.figure(figsize=(12, 8))
sns.heatmap(
    gene_subset.T,
    cmap='viridis',
    cbar_kws={'label': 'Normalized Expression'},
    xticklabels=False
)
plt.xlabel('Brain Regions')
plt.ylabel('Genes')
plt.title('Gene Expression Across Cortical Regions')
plt.tight_layout()
plt.savefig('gene_expression_heatmap.png', dpi=300)
```

---

## Quality Control and Validation

### Check Sample Coverage

```python
import abagen

# Get sample information for quality control
files = abagen.fetch_microarray(donors='all')

# Check sample distribution
import nibabel as nib

parcellation_img = nib.load(parcellation)
parcellation_data = parcellation_img.get_fdata()

unique_labels = np.unique(parcellation_data)
unique_labels = unique_labels[unique_labels != 0]  # Remove background

print(f"Total parcels: {len(unique_labels)}")

# Load samples for first donor
donor_samples = pd.read_csv(files[0]['annotation'], header=None)

# Check how many samples fall within each parcel
# (Detailed implementation requires processing sample coordinates)

print("Sample coverage varies by donor")
print("Some regions may have no samples (require interpolation)")
```

### Donor-Specific Effects

```python
# Analyze consistency across donors

# Generate expression for each donor separately
donor_expressions = []

for donor in ['9861', '10021', '12876', '14380', '15496', '15697']:
    expr = abagen.get_expression_data(
        parcellation,
        donors=[donor],
        verbose=0
    )
    donor_expressions.append(expr)

# Compute inter-donor correlations
from scipy.stats import spearmanr

# For a specific gene across donors
gene = 'HTR1A'

if gene in expression.columns:
    donor_gene_expr = [df[gene].values for df in donor_expressions if gene in df.columns]

    # Correlate donors
    if len(donor_gene_expr) >= 2:
        r, p = spearmanr(donor_gene_expr[0], donor_gene_expr[1])
        print(f"Donor 1 vs Donor 2 for {gene}: r = {r:.3f}, p = {p:.3e}")
```

### Reproducibility Analysis

```python
# Test reproducibility with different preprocessing parameters

params_list = [
    {'probe_selection': 'diff_stability', 'sample_norm': 'srs'},
    {'probe_selection': 'max_intensity', 'sample_norm': 'srs'},
    {'probe_selection': 'diff_stability', 'sample_norm': 'zscore'}
]

expression_variants = []

for params in params_list:
    expr = abagen.get_expression_data(parcellation, verbose=0, **params)
    expression_variants.append(expr)

# Compare gene rankings across parameter choices
# (For a specific phenotype correlation)

from scipy.stats import spearmanr

# Example: correlate with cortical thickness
rankings = []

for expr in expression_variants:
    corrs = []
    for gene in expr.columns:
        if gene in cortical_thickness.index:
            r, _ = spearmanr(expr[gene], cortical_thickness[gene])
            corrs.append((gene, r))

    corrs_sorted = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
    rankings.append([g for g, r in corrs_sorted[:100]])  # Top 100

# Compute overlap in top genes
overlap_1_2 = len(set(rankings[0]) & set(rankings[1]))
print(f"Overlap in top 100 genes (param set 1 vs 2): {overlap_1_2}")
```

---

## Integration with BrainSpace and neuromaps

### Gene Expression Gradients with BrainSpace

```python
import abagen
from brainspace.gradient import GradientMaps, ProcrustesAlignment
from scipy.stats import spearmanr

# Gene expression matrix
expression = abagen.get_expression_data(parcellation, verbose=1)

# Compute gene co-expression similarity
gene_coexpr = expression.T.corr(method='spearman').values
gene_coexpr = np.nan_to_num(gene_coexpr, nan=0)

# Gradient analysis
gm_gene = GradientMaps(n_components=5, approach='dm', kernel='normalized_angle')
gm_gene.fit(gene_coexpr)

# Compare to functional connectivity gradients
# (Assuming functional gradients already computed)
# functional_gradients = ...

# Align gene and functional gradients
# pa = ProcrustesAlignment()
# aligned = pa.fit_transform([gm_gene.gradients_, functional_gradients])

print("Gene expression gradients computed")
```

### Correlate with neuromaps Annotations

```python
from neuromaps import datasets as neuromaps_datasets, transforms
import pandas as pd

# Gene expression for Schaefer 400
expression_schaefer = abagen.get_expression_data(
    parcellation_schaefer,
    verbose=1
)

# Load neuromaps receptor annotation
receptor = neuromaps_datasets.fetch_annotation(
    source='beliveau2017',
    desc='5ht1a',
    space='fsaverage',
    den='10k'
)

# Transform to Schaefer 400 parcellation
# (Requires parcellation-based aggregation)

# Correlate HTR1A gene expression with 5-HT1a receptor PET
if 'HTR1A' in expression_schaefer.columns:
    # In practice, align receptor map to parcels
    # receptor_parcellated = ...

    # r, p = spearmanr(expression_schaefer['HTR1A'], receptor_parcellated)
    print("HTR1A gene expression correlates with receptor density")
```

---

## Advanced Applications

### Network-Specific Gene Expression

```python
# Analyze gene expression within functional networks (e.g., Yeo 7 networks)

from nilearn import datasets

atlas_yeo = datasets.fetch_atlas_yeo_2011()
# Use Schaefer parcellation with Yeo network labels

atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
network_labels = atlas_schaefer['labels']  # Contains network assignments

# Expression matrix
expression_schaefer = abagen.get_expression_data(
    atlas_schaefer['maps'],
    verbose=1
)

# Parse network assignments
# Schaefer labels format: "7Networks_LH_Vis_1" -> Visual network

import re

def extract_network(label):
    match = re.search(r'7Networks_[LR]H_(\w+)_', label)
    if match:
        return match.group(1)
    return None

expression_schaefer['network'] = [extract_network(label) for label in expression_schaefer.index]

# Compare gene expression between networks
networks = expression_schaefer['network'].unique()

for gene in expression_schaefer.columns[:5]:  # Example: first 5 genes
    network_means = expression_schaefer.groupby('network')[gene].mean()
    print(f"{gene} expression by network:")
    print(network_means)
    print()
```

### Disease-Associated Genes

```python
# Identify spatial patterns of disease-associated genes

# Example: Alzheimer's disease risk genes
ad_genes = ['APOE', 'TREM2', 'CLU', 'CR1', 'PICALM', 'BIN1', 'SORL1']

# Check availability
available_ad_genes = [g for g in ad_genes if g in expression.columns]

print(f"Available AD genes: {available_ad_genes}")

# Compute AD gene expression signature
# (Mean expression of available AD genes per region)

ad_signature = expression[available_ad_genes].mean(axis=1)

# Correlate with cortical thinning in AD
# (In practice, use real AD atrophy pattern)
ad_atrophy = np.random.randn(len(ad_signature))  # Example

from scipy.stats import spearmanr
r, p = spearmanr(ad_signature, ad_atrophy)

print(f"AD gene signature ↔ atrophy: r = {r:.3f}, p = {p:.3e}")
```

---

## Batch Processing

### Process Multiple Parcellations

```python
from nilearn import datasets
import abagen

parcellations = {
    'desikan': datasets.fetch_atlas_destrieux_2009()['maps'],
    'schaefer_200': datasets.fetch_atlas_schaefer_2018(n_rois=200)['maps'],
    'schaefer_400': datasets.fetch_atlas_schaefer_2018(n_rois=400)['maps']
}

expression_matrices = {}

for name, parc in parcellations.items():
    print(f"Processing {name}...")

    expr = abagen.get_expression_data(
        parc,
        verbose=0
    )

    expression_matrices[name] = expr
    expr.to_csv(f'expression_{name}.csv')

    print(f"  {name}: {expr.shape}")

print("Batch processing complete")
```

---

## Troubleshooting

### Data Download Issues

```bash
# If AHBA download fails, clear cache
rm -rf ~/.cache/abagen

# Manually download from Allen Institute
# https://human.brain-map.org/
```

### Memory Issues

```python
# For large parcellations, process donors sequentially

expression_list = []

for donor in ['9861', '10021', '12876', '14380', '15496', '15697']:
    expr = abagen.get_expression_data(
        parcellation,
        donors=[donor],
        verbose=1
    )
    expression_list.append(expr)

    # Average across donors
    expression_mean = pd.concat(expression_list).groupby(level=0).mean()
```

### Missing Regions

```python
# Check which regions have missing data
missing_mask = expression.isna().any(axis=1)
missing_regions = expression.index[missing_mask]

print(f"Regions with missing data: {len(missing_regions)}")
print(missing_regions)

# Use interpolation or exclude from analysis
expression_complete = expression.dropna()
```

---

## Best Practices

### Preprocessing Decisions

1. **Probe selection:**
   - Differential stability (default) is robust
   - Use max_intensity if prioritizing highly expressed genes

2. **Normalization:**
   - Scaled robust sigmoid (SRS) recommended
   - Consistent within and across donors

3. **Missing data:**
   - Interpolate for cortical regions with nearby samples
   - Consider excluding subcortical regions with no nearby samples

4. **Parcellation:**
   - Choose granularity based on research question
   - Finer parcellations have more missing regions

### Statistical Analysis

1. **Multiple testing:**
   - Always apply FDR or Bonferroni correction
   - Report both raw and corrected p-values

2. **Spatial autocorrelation:**
   - Gene expression is spatially smooth
   - Consider spatial null models (neuromaps integration)

3. **Sample size:**
   - 6 donors provide limited power
   - Focus on robust, large-effect associations

---

## Resources and Further Reading

### Official Documentation

- **abagen Docs:** https://abagen.readthedocs.io/
- **GitHub:** https://github.com/rmarkello/abagen
- **Tutorials:** https://abagen.readthedocs.io/en/stable/usage.html
- **API Reference:** https://abagen.readthedocs.io/en/stable/api.html

### Allen Brain Atlas

- **AHBA Portal:** https://human.brain-map.org/
- **Documentation:** https://help.brain-map.org/

### Key Publications

```
Arnatkeviciute, A., Fulcher, B. D., & Fornito, A. (2019).
A practical guide to linking brain-wide gene expression and neuroimaging data.
NeuroImage, 189, 353-367.
```

```
Markello, R. D., et al. (2021).
Standardizing workflows in imaging transcriptomics with the abagen toolbox.
eLife, 10, e72129.
```

### Related Tools

- **neuromaps:** Brain annotations including gene expression maps
- **BrainSpace:** Gradient analysis of gene co-expression
- **BrainStat:** Statistical testing for gene-imaging associations

---

## Summary

**abagen** enables standardized transcriptomics-neuroimaging integration:

**Strengths:**
- Automated AHBA preprocessing
- Reproducible workflows
- Flexible parcellation support
- Quality control utilities
- Comprehensive documentation

**Best For:**
- Gene-imaging correlations
- Molecular mechanisms of brain organization
- Transcriptomic signatures of networks
- Disease gene spatial patterns
- Imaging genetics research

**Typical Workflow:**
1. Choose brain parcellation
2. Generate expression matrix with abagen
3. Correlate with imaging phenotype
4. Gene set enrichment analysis
5. Interpret molecular mechanisms

abagen has standardized imaging transcriptomics, enabling researchers to link genes to brain structure, function, and connectivity with reproducible, best-practice workflows.
