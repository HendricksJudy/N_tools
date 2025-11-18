# NeuroQuery - Meta-Analytic Brain Decoding from Text

## Overview

NeuroQuery is a modern meta-analytic platform that generates predictive brain activation maps from natural language queries using machine learning trained on the neuroimaging literature. Unlike traditional coordinate-based meta-analysis requiring manual study selection, NeuroQuery automatically learns relationships between scientific terms and brain coordinates from over 13,000 neuroimaging studies. Users can enter free-form text queries (e.g., "working memory", "emotion regulation", "default mode network") and instantly receive statistical brain maps showing predicted activation patterns, semantically related terms, and relevant studies ranked by relevance.

Built on advances in natural language processing and neuroinformatics, NeuroQuery combines distributional semantics (word embeddings) with coordinate-based modeling to enable rapid hypothesis generation, literature exploration, and brain-cognition mapping. The tool supports both forward inference (predicting brain patterns from cognitive terms) and reverse inference (decoding cognitive functions from activation coordinates). NeuroQuery is available as an interactive web application (https://neuroquery.org) and a comprehensive Python API for programmatic access, integration into analysis pipelines, and custom model development.

**Website:** https://neuroquery.org
**Repository:** https://github.com/neuroquery/neuroquery
**Documentation:** https://neuroquery.github.io/neuroquery/
**Paper:** https://elifesciences.org/articles/53385

### Key Features

- **Text-to-Brain Mapping:** Natural language queries → statistical brain maps
- **Large Training Set:** >13,000 neuroimaging studies automatically extracted
- **Predictive Modeling:** Supervised learning (ridge regression) on term-coordinate associations
- **Semantic Search:** Identifies related terms and synonyms using word embeddings
- **Study Relevance Ranking:** Lists papers related to query with relevance scores
- **Reverse Inference:** Predict cognitive terms from brain activation patterns (decoding)
- **Interactive Web Interface:** Real-time queries at https://neuroquery.org
- **Python API:** Programmatic access, batch processing, custom analyses
- **NIfTI Export:** Download brain maps for visualization in FSLeyes, nilearn
- **Open Source:** Transparent algorithms, reproducible research
- **Continuous Updates:** Database refreshed with new studies regularly

### Applications

- Hypothesis generation for new experiments
- Literature exploration and review
- Contextualize individual study findings with meta-analytic predictions
- Brain decoding (reverse inference from activation patterns)
- Educational demonstrations of brain-cognition relationships
- Multi-study synthesis without manual coordinate extraction
- Semantic meta-analysis

### NeuroQuery vs. NeuroSynth

**NeuroQuery:**
- **Pros:** Natural language queries, semantic search, predictive modeling, newer algorithm
- **Cons:** Slightly smaller database (~13k vs. ~15k studies)

**NeuroSynth:**
- **Pros:** Larger database, established tool, similar web interface
- **Cons:** Keyword-only queries, older algorithm, less semantic flexibility

**Recommendation:** Both tools are complementary; NeuroQuery excels at semantic queries and prediction, NeuroSynth at broad keyword-based synthesis

### Citation

```bibtex
@article{Dockès2020NeuroQuery,
  title={NeuroQuery, comprehensive meta-analysis of human brain mapping},
  author={Dock{\`e}s, J{\'e}r{\^o}me and Poldrack, Russell A and Primet, Romain and
          G{\"o}kc{\"u}l, Hande and Yarkoni, Tal and Suchanek, Fabian and
          Thirion, Bertrand and Varoquaux, Ga{\"e}l},
  journal={eLife},
  volume={9},
  pages={e53385},
  year={2020},
  publisher={eLife Sciences Publications Limited}
}
```

---

## Web Interface Usage

### Accessing NeuroQuery.org

```bash
# Open web browser and navigate to:
https://neuroquery.org

# No installation required
# No account registration needed
# Free and open access

# Browser compatibility:
# - Chrome/Chromium (recommended)
# - Firefox
# - Safari
# - Edge

# Mobile compatibility:
# - Responsive design works on tablets
# - Limited functionality on smartphones
```

### Text Query Interface

```text
Basic query workflow:

1. Enter query in search box
   Example: "working memory"

2. Click "Search" or press Enter

3. View results:
   - Brain map (3D interactive viewer)
   - Associated terms (semantic neighbors)
   - Relevant studies (ranked by relevance)

4. Interact with brain map:
   - Rotate: Click and drag
   - Zoom: Scroll wheel
   - Slice: Adjust X/Y/Z sliders
   - Threshold: Adjust z-score threshold slider

5. Download results:
   - Brain map: NIfTI format
   - Term weights: CSV format
   - Study list: CSV format
```

### Interpreting Brain Maps

```text
NeuroQuery brain maps show predicted z-scores:

Color Scale:
- Red/Yellow: Positive association (higher z-scores)
- Blue: Negative association (lower z-scores, less common)
- Intensity: Strength of prediction

Statistical Threshold:
- Default: z > 3.0 (roughly p < 0.001)
- Adjustable slider: 2.0 - 6.0
- Higher threshold: More conservative, fewer voxels

Map Interpretation:
- NOT derived from single meta-analysis
- Predictive model trained on all studies
- Shows expected activation for query terms
- Captures general patterns across literature

Example: "working memory" query
Expected activation:
- Dorsolateral prefrontal cortex (DLPFC)
- Posterior parietal cortex
- Anterior cingulate cortex
- Matches canonical working memory network
```

### Viewing Associated Terms

```text
Associated terms show semantic relationships:

Term List:
- Ranked by semantic similarity
- Positive weights: Related concepts
- Negative weights: Contrasting concepts

Example: Query "fear"
Associated terms (positive weights):
- anxiety (0.82)
- threat (0.76)
- amygdala (0.71)
- aversive (0.68)
- emotional (0.65)

Associated terms (negative weights):
- reward (-0.32)
- happy (-0.28)

Use associated terms to:
- Refine queries
- Discover related concepts
- Understand semantic space
- Generate new hypotheses
```

### Example: Query "Working Memory"

```bash
# Complete workflow example

# Step 1: Navigate to https://neuroquery.org

# Step 2: Enter query
Query: "working memory"

# Step 3: Review brain map
# Predicted activation visible in:
# - Bilateral DLPFC (±45, 30, 30)
# - Posterior parietal (±40, -50, 50)
# - Anterior cingulate (0, 20, 40)
# - Lateral premotor cortex

# Step 4: Check associated terms
# Top terms:
# - n-back (0.85)
# - executive function (0.78)
# - cognitive control (0.75)
# - attention (0.68)
# - frontal cortex (0.66)

# Step 5: Review relevant studies
# Top study: "The role of prefrontal cortex in working memory..."
# Relevance: 0.92
# Click study to view PubMed link

# Step 6: Download brain map
# Click "Download map (NIfTI)"
# File: neuroquery_working_memory.nii.gz
# Open in FSLeyes or nilearn for publication figures
```

---

## Python API Installation

### Installing NeuroQuery Package

```bash
# Install via pip
pip install neuroquery

# Install with visualization dependencies
pip install neuroquery[plotting]

# Development installation (from GitHub)
pip install git+https://github.com/neuroquery/neuroquery.git

# Verify installation
python -c "import neuroquery; print(neuroquery.__version__)"
# Output: 0.2.0 (or latest version)
```

### Downloading Trained Models

```python
# Download pre-trained NeuroQuery model and data

from neuroquery import fetch_neuroquery_model

# Download model (~300 MB)
# Includes: trained regression model, vocabulary, coordinates database
model_dir = fetch_neuroquery_model()

print(f"Model downloaded to: {model_dir}")
# Output: /home/user/.cache/neuroquery_data/neuroquery_model

# Model components:
# - corpus_metadata.csv: Study information
# - coordinates.csv: All extracted coordinates
# - vocabulary.csv: Terms and frequencies
# - regression_model.pkl: Trained ridge regression model
# - corpus_tfidf.npz: Term-document matrix
```

### Loading Datasets

```python
# Load NeuroQuery model for querying

from neuroquery import NeuroQueryModel

# Load pre-trained model
model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

print(f"Model vocabulary size: {len(model.get_vocabulary())}")
# Output: ~7,500 unique terms

print(f"Number of studies: {model.n_studies}")
# Output: ~13,000 studies

print(f"Number of coordinates: {model.n_coordinates}")
# Output: ~400,000 coordinates
```

### Testing Installation

```python
# Test query to verify installation

from neuroquery import NeuroQueryModel, fetch_neuroquery_model
import matplotlib.pyplot as plt
from nilearn import plotting

# Load model
model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

# Test query
result = model("memory")

# Check result structure
print(f"Result keys: {result.keys()}")
# Output: dict_keys(['z_map', 'similar_words', 'similar_documents'])

# View similar words
print("Top 5 similar terms:")
for term, weight in result['similar_words'][:5]:
    print(f"  {term}: {weight:.3f}")

# Visualize brain map
plotting.plot_stat_map(result['z_map'], title="Memory", threshold=3.0)
plt.show()

print("NeuroQuery installation successful!")
```

---

## Basic Queries

### Single Term Queries

```python
# Query with single cognitive term

from neuroquery import NeuroQueryModel, fetch_neuroquery_model

# Load model
model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

# Single-term query
result = model("attention")

# Result contains:
# - z_map: Brain activation map (NIfTI image)
# - similar_words: Related terms with weights
# - similar_documents: Relevant studies with scores

# Access z-map
z_map = result['z_map']
print(type(z_map))
# Output: <class 'nibabel.nifti1.Nifti1Image'>

# Get z-map data array
import numpy as np
z_data = z_map.get_fdata()
print(f"Z-map shape: {z_data.shape}")
# Output: (91, 109, 91) - MNI152 2mm space
```

### Multi-Word Queries

```python
# Query with phrases or multiple terms

# Multi-word phrase
result = model("working memory")

# Multiple concepts
result = model("emotion regulation cognitive control")

# Specific paradigm
result = model("n-back task frontal cortex")

# The model treats multi-word queries as a weighted combination
# of individual terms based on learned co-occurrence patterns

# View how query was interpreted
similar_terms = result['similar_words'][:10]
print("Query interpreted as related to:")
for term, weight in similar_terms:
    print(f"  {term}: {weight:.2f}")
```

### Viewing Predicted Activation Maps

```python
# Visualize predicted brain activation

from nilearn import plotting
import matplotlib.pyplot as plt

# Query
result = model("language comprehension")

# Plot brain map (glass brain)
plotting.plot_glass_brain(
    result['z_map'],
    threshold=3.0,
    title="Language Comprehension",
    colorbar=True,
    plot_abs=False
)
plt.show()

# Plot as statistical map (orthogonal slices)
plotting.plot_stat_map(
    result['z_map'],
    threshold=3.0,
    title="Language Comprehension",
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.show()

# Interactive viewer
from nilearn import view
interactive = view.view_img(result['z_map'], threshold=3.0)
interactive.open_in_browser()
```

### Example: Python Query for "Fear"

```python
# Complete query example: emotion processing

from neuroquery import NeuroQueryModel, fetch_neuroquery_model
from nilearn import plotting
import matplotlib.pyplot as plt

# Load model
model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

# Query: fear processing
result = model("fear")

# Print associated terms
print("Terms associated with 'fear':")
for term, weight in result['similar_words'][:10]:
    print(f"  {term}: {weight:.3f}")

# Expected terms:
# - anxiety, threat, amygdala, aversive, emotional

# Print relevant studies
print("\nMost relevant studies:")
for i, (pmid, score) in enumerate(result['similar_documents'][:5]):
    print(f"  {i+1}. PMID {pmid}, relevance: {score:.3f}")

# Visualize activation
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Glass brain
plotting.plot_glass_brain(
    result['z_map'],
    threshold=3.0,
    title="Fear Processing - Meta-Analytic Prediction",
    axes=axes[0],
    colorbar=True
)

# Statistical map
plotting.plot_stat_map(
    result['z_map'],
    threshold=3.0,
    display_mode='z',
    cut_coords=5,
    title="Axial Slices",
    axes=axes[1]
)

plt.tight_layout()
plt.savefig('fear_metaanalysis.png', dpi=300)
plt.show()

# Expected activation:
# - Bilateral amygdala
# - Anterior insula
# - Anterior cingulate cortex
# - Dorsomedial prefrontal cortex
```

---

## Interpreting Results

### Z-Score Brain Maps

```python
# Understanding NeuroQuery z-score maps

# Query result
result = model("reward")
z_map = result['z_map']

# Extract z-values
import numpy as np
z_data = z_map.get_fdata()

# Statistical properties
print(f"Z-score range: {z_data.min():.2f} to {z_data.max():.2f}")
print(f"Mean z-score: {z_data.mean():.2f}")
print(f"Std z-score: {z_data.std():.2f}")

# Voxels above threshold
threshold = 3.0
n_significant = np.sum(z_data > threshold)
print(f"Voxels > {threshold}: {n_significant}")

# Interpretation:
# - Z-scores are predictions, not actual meta-analysis statistics
# - Higher z = stronger predicted association with query terms
# - Threshold of 3.0 ≈ p < 0.001 (interpretive, not inferential)
# - Maps show expected patterns from training data
```

### Statistical Thresholding

```python
# Apply different thresholds to brain maps

from nilearn import plotting
import matplotlib.pyplot as plt

result = model("motor execution")

# Multiple thresholds
thresholds = [2.0, 3.0, 4.0, 5.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, thresh in enumerate(thresholds):
    plotting.plot_glass_brain(
        result['z_map'],
        threshold=thresh,
        title=f"Threshold z > {thresh}",
        axes=axes[i],
        colorbar=True
    )

plt.tight_layout()
plt.show()

# Observations:
# - Lower threshold (2.0): Broad activation, less specific
# - Higher threshold (5.0): Highly specific, core regions only
# - Standard: 3.0 balances sensitivity and specificity
```

### Associated Terms and Weights

```python
# Analyze semantic relationships

result = model("default mode network")

# Get associated terms
similar_terms = result['similar_words']

# Convert to dataframe for analysis
import pandas as pd
df_terms = pd.DataFrame(similar_terms, columns=['term', 'weight'])

# Positive associations (top 10)
print("Positively associated terms:")
print(df_terms.head(10))

# Negative associations (if any)
negative = df_terms[df_terms['weight'] < 0]
if len(negative) > 0:
    print("\nNegatively associated terms:")
    print(negative.head(10))

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(df_terms.head(15)['term'], df_terms.head(15)['weight'])
plt.xlabel('Association Weight')
plt.title('Terms Associated with "Default Mode Network"')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Relevant Studies

```python
# Identify papers most relevant to query

result = model("cognitive control")

# Get study PMIDs and relevance scores
relevant_studies = result['similar_documents'][:20]

# Load metadata to get paper details
from neuroquery import fetch_neuroquery_model
import pandas as pd

model_dir = fetch_neuroquery_model()
metadata = pd.read_csv(f"{model_dir}/corpus_metadata.csv")

# Match PMIDs to metadata
print("Top 10 most relevant studies:")
for i, (pmid, score) in enumerate(relevant_studies[:10]):
    study_info = metadata[metadata['pmid'] == pmid]
    if not study_info.empty:
        title = study_info.iloc[0]['title']
        year = study_info.iloc[0]['publication_year']
        print(f"{i+1}. [{year}] {title[:80]}... (relevance: {score:.3f})")

# Export study list
study_df = pd.DataFrame(relevant_studies, columns=['pmid', 'relevance_score'])
study_df = study_df.merge(metadata, on='pmid', how='left')
study_df.to_csv('cognitive_control_studies.csv', index=False)
```

---

## Reverse Inference (Brain Decoding)

### Predicting Cognitive Terms from Activation

```python
# Reverse inference: brain pattern → cognitive terms

from neuroquery import NeuroQueryModel, fetch_neuroquery_model
import numpy as np
import nibabel as nib

# Load model
model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

# Load a brain activation map (e.g., from your study)
# Example: statistical map from individual fMRI analysis
your_activation = nib.load('my_study_zmap.nii.gz')

# Decode: predict terms from activation pattern
decoded_result = model.decode(your_activation)

# Result contains similar_words (predicted cognitive terms)
print("Predicted cognitive functions:")
for term, weight in decoded_result['similar_words'][:15]:
    print(f"  {term}: {weight:.3f}")

# Interpretation:
# - Terms with high weights are predicted to be related to activation pattern
# - Enables "reverse inference" from brain to cognition
# - Useful for interpreting novel activation patterns
```

### Using Coordinate Sets as Input

```python
# Decode from coordinate list (peak activations)

# Your study's peak coordinates (MNI space)
coordinates = np.array([
    [-42, 28, 32],   # Left DLPFC
    [44, 30, 28],    # Right DLPFC
    [-8, 18, 48],    # Medial frontal
    [0, 52, -8]      # Medial PFC
])

# Create peak activation map
from nilearn import image, datasets

# Get MNI template
template = datasets.load_mni152_template(resolution=2)

# Create sphere ROIs at peak coordinates
from nilearn.input_data import NiftiSpheresMasker
masker = NiftiSpheresMasker(coordinates, radius=10.0)

# Create binary activation map
activation_data = np.zeros(template.shape)
# (Manual approach: set voxels near coordinates to high values)

# Alternative: Use coordinates directly with nilearn
# Convert coordinates to z-score map
from scipy.ndimage import gaussian_filter
coord_map = np.zeros(template.shape)
# Fill in coordinates with z-scores...
# Then decode with model
# decoded = model.decode(coord_map_img)
```

### Interpreting Decoder Output

```python
# Analyze reverse inference results

# Example: decode default mode network activation
dmn_coordinates = np.array([
    [0, -52, 28],     # Posterior cingulate
    [-44, -68, 32],   # Angular gyrus L
    [46, -66, 32],    # Angular gyrus R
    [0, 52, -8],      # Medial PFC
])

# (Assume we created activation map from coordinates)
# decoded = model.decode(dmn_activation_map)

# Analyze predicted terms
# Expected high-weight terms for DMN:
# - "default mode"
# - "resting state"
# - "self-referential"
# - "theory of mind"
# - "mentalizing"

# Compare forward and reverse inference:
# Forward: "default mode network" → brain map
# Reverse: brain map → "default mode network"
# Should be approximately consistent
```

---

## Advanced Queries

### Boolean Operators

```python
# Combine terms with boolean logic (simulated)

# AND: Intersection of concepts
# Query both terms together, model captures overlap
result_and = model("attention working memory")

# OR: Union of concepts
# Query terms separately and combine maps
result1 = model("attention")
result2 = model("working memory")

# Combine z-maps (element-wise maximum)
import numpy as np
from nilearn import image
z_or = image.math_img(
    'np.maximum(img1, img2)',
    img1=result1['z_map'],
    img2=result2['z_map']
)

# NOT: Exclusion (subtract maps)
result_not = image.math_img(
    'img1 - img2',
    img1=result1['z_map'],
    img2=result2['z_map']
)
```

### Weighted Term Queries

```python
# Emphasize certain terms in query

# Implicit weighting via repetition
# "memory memory memory attention"
# Model will weight "memory" more heavily

# Explicit control via custom vocabulary weights (advanced)
# Requires modifying model internals

# Practical approach: Separate queries and combine
result_memory = model("memory")
result_emotion = model("emotion")

# Weighted combination (70% memory, 30% emotion)
from nilearn import image
z_weighted = image.math_img(
    '0.7 * img1 + 0.3 * img2',
    img1=result_memory['z_map'],
    img2=result_emotion['z_map']
)
```

### Example: "Attention NOT Memory"

```python
# Query attention while excluding memory-related activation

from neuroquery import NeuroQueryModel, fetch_neuroquery_model
from nilearn import image, plotting
import matplotlib.pyplot as plt

model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

# Separate queries
result_attention = model("attention")
result_memory = model("memory")

# Subtract memory from attention
z_attention_not_memory = image.math_img(
    'np.maximum(img1 - img2, 0)',  # Keep only positive values
    img1=result_attention['z_map'],
    img2=result_memory['z_map']
)

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

plotting.plot_glass_brain(
    result_attention['z_map'], threshold=3.0,
    title="Attention", axes=axes[0], colorbar=True
)

plotting.plot_glass_brain(
    result_memory['z_map'], threshold=3.0,
    title="Memory", axes=axes[1], colorbar=True
)

plotting.plot_glass_brain(
    z_attention_not_memory, threshold=2.0,
    title="Attention NOT Memory", axes=axes[2], colorbar=True
)

plt.tight_layout()
plt.show()

# Expected result:
# - Reduced activation in medial temporal lobe (memory regions)
# - Preserved activation in dorsal attention network
# - Preserved activation in frontal eye fields
```

---

## Visualization

### Plotting Brain Maps with nilearn

```python
# Comprehensive visualization examples

from neuroquery import NeuroQueryModel, fetch_neuroquery_model
from nilearn import plotting
import matplotlib.pyplot as plt

model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())
result = model("emotion regulation")

# 1. Glass brain (transparent overlay)
plotting.plot_glass_brain(
    result['z_map'],
    threshold=3.0,
    colorbar=True,
    title="Emotion Regulation - Glass Brain"
)
plt.savefig('emotion_glass.png', dpi=300, bbox_inches='tight')

# 2. Statistical map (orthogonal slices)
plotting.plot_stat_map(
    result['z_map'],
    threshold=3.0,
    display_mode='ortho',
    cut_coords=(0, 0, 0),
    title="Emotion Regulation - Orthogonal"
)
plt.savefig('emotion_ortho.png', dpi=300, bbox_inches='tight')

# 3. Multiple slices (axial)
plotting.plot_stat_map(
    result['z_map'],
    threshold=3.0,
    display_mode='z',
    cut_coords=8,
    title="Emotion Regulation - Axial Slices"
)
plt.savefig('emotion_axial.png', dpi=300, bbox_inches='tight')

# 4. Surface projection (cortical surface)
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

plotting.plot_surf_stat_map(
    fsaverage.infl_left,
    result['z_map'],
    hemi='left',
    view='lateral',
    threshold=3.0,
    colorbar=True,
    title="Left Hemisphere - Lateral View"
)
plt.savefig('emotion_surf_left.png', dpi=300, bbox_inches='tight')
```

### Interactive Viewers

```python
# Create interactive HTML visualizations

from nilearn import view

result = model("social cognition")

# Interactive statistical map
interactive_stat = view.view_img(
    result['z_map'],
    threshold=3.0,
    title="Social Cognition Meta-Analysis"
)

# Save as HTML file
interactive_stat.save_as_html('social_cognition_interactive.html')

# Open in browser
interactive_stat.open_in_browser()

# Interactive surface projection
interactive_surf = view.view_img_on_surf(
    result['z_map'],
    threshold=3.0,
    surf_mesh='fsaverage'
)

interactive_surf.save_as_html('social_surf_interactive.html')
```

### Exporting NIfTI Images

```python
# Save brain maps for external visualization

result = model("pain")

# Export z-map as NIfTI
result['z_map'].to_filename('pain_metaanalysis_zmap.nii.gz')

# Threshold and export binary mask
from nilearn import image
thresholded = image.threshold_img(result['z_map'], threshold=3.0)
thresholded.to_filename('pain_mask_z3.nii.gz')

# Export for different software:
# - FSLeyes: Use .nii.gz directly
# - MRIcron: Convert to .nii (uncompressed)
# - Connectome Workbench: Convert to CIFTI (requires resampling)

# Verify export
import nibabel as nib
img = nib.load('pain_metaanalysis_zmap.nii.gz')
print(f"Shape: {img.shape}")
print(f"Affine:\n{img.affine}")
```

---

## Comparing with NeuroSynth

### Methodology Differences

```text
NeuroQuery vs. NeuroSynth:

Training Data:
- NeuroQuery: ~13,000 studies (2020 snapshot)
- NeuroSynth: ~15,000 studies (continually updated)

Text Processing:
- NeuroQuery: Sophisticated NLP, word embeddings, semantic relationships
- NeuroSynth: Simple term frequency in abstracts

Query Flexibility:
- NeuroQuery: Natural language, multi-word phrases, semantic search
- NeuroSynth: Single keywords only

Statistical Model:
- NeuroQuery: Supervised ridge regression, predictive modeling
- NeuroSynth: Chi-square test, association testing

Output Maps:
- NeuroQuery: Predicted z-scores (continuous)
- NeuroSynth: Association z-scores (forward/reverse inference)

Reverse Inference:
- NeuroQuery: Decoding function (brain → terms)
- NeuroSynth: Reverse inference maps (P(term|activation))
```

### When to Use NeuroQuery vs. NeuroSynth

**Use NeuroQuery when:**
- Need semantic/natural language queries ("emotion regulation")
- Want predictive modeling of activation patterns
- Require fine-grained term relationships
- Building hypothesis about brain-cognition mappings

**Use NeuroSynth when:**
- Need largest possible database (>15k studies)
- Want simple keyword-based meta-analysis
- Require established/widely-cited tool
- Seeking reverse inference probabilities

**Use Both:**
- NeuroQuery for primary analysis
- NeuroSynth for validation and cross-checking
- Compare maps to assess robustness

---

## Integration with Research Workflows

### Hypothesis Generation

```python
# Use NeuroQuery to generate hypotheses for new study

# Research question: Where should we expect activation for "cognitive flexibility"?

result = model("cognitive flexibility")

# Examine predicted activation
from nilearn import plotting
plotting.plot_glass_brain(result['z_map'], threshold=3.0)

# Identify predicted regions (extract peaks)
from nilearn.reporting import get_clusters_table
clusters = get_clusters_table(result['z_map'], stat_threshold=3.0, cluster_threshold=100)
print(clusters)

# Use predicted regions as ROIs for new study:
# - Pre-register analysis based on meta-analytic prediction
# - Compare empirical activation to meta-analytic expectation
```

### Result Contextualization

```python
# Compare your study results with meta-analytic prediction

import nibabel as nib

# Load your study's activation map
your_result = nib.load('my_study_cognitive_control_zmap.nii.gz')

# Query NeuroQuery with same term
meta_prediction = model("cognitive control")

# Spatial correlation between your result and meta-analysis
from nilearn.masking import apply_mask, unmask
from scipy.stats import pearsonr

# Mask both images
from nilearn import datasets
mask = datasets.load_mni152_brain_mask()

your_vals = apply_mask(your_result, mask)
meta_vals = apply_mask(meta_prediction['z_map'], mask)

# Compute correlation
r, p = pearsonr(your_vals, meta_vals)
print(f"Spatial correlation with meta-analysis: r = {r:.3f}, p = {p:.3e}")

# Interpretation:
# - High r (>0.3): Your results consistent with literature
# - Low r (<0.1): Your results diverge from typical pattern
#   (Could indicate novel finding or methodological issue)
```

---

## Troubleshooting

### Model Download Issues

**Problem:** fetch_neuroquery_model() fails or times out

**Solution:**
```python
# Manual download and cache
import os
from pathlib import Path

# Download from GitHub releases
# https://github.com/neuroquery/neuroquery_data/releases

# Extract to cache directory
cache_dir = Path.home() / '.cache' / 'neuroquery_data' / 'neuroquery_model'
cache_dir.mkdir(parents=True, exist_ok=True)

# Move downloaded files to cache_dir
# Then load model:
from neuroquery import NeuroQueryModel
model = NeuroQueryModel.from_data_dir(str(cache_dir))
```

### Empty Query Results

**Problem:** Query returns no activation or very weak activation

**Causes:**
- Term not in training vocabulary
- Misspelling
- Very rare concept in literature

**Solution:**
```python
# Check if term in vocabulary
vocab = model.get_vocabulary()
query_term = "neurotransmission"

if query_term in vocab:
    print(f"'{query_term}' in vocabulary")
else:
    print(f"'{query_term}' NOT in vocabulary")
    # Try synonyms or related terms
    # Example: "dopamine" instead of "dopaminergic neurotransmission"
```

### Memory Limitations

**Problem:** Out of memory errors with large batch queries

**Solution:**
```python
# Process queries in smaller batches
queries = ["attention", "memory", "emotion", ...]  # 100 queries

results = []
batch_size = 10

for i in range(0, len(queries), batch_size):
    batch = queries[i:i+batch_size]
    batch_results = [model(q) for q in batch]
    results.extend(batch_results)

    # Clear memory
    import gc
    gc.collect()
```

---

## Best Practices

### Query Formulation Tips

- **Be Specific:** "working memory n-back" better than "memory"
- **Use Standard Terms:** Follow neuroimaging literature conventions
- **Check Vocabulary:** Verify terms exist in model vocabulary
- **Avoid Jargon:** Use widely recognized cognitive terms
- **Natural Language:** Phrases work better than single words

### Interpreting Predictive Maps

- **Not Ground Truth:** Maps are predictions, not definitive meta-analyses
- **Confidence:** Higher z-scores = more confident predictions
- **Validation:** Compare with formal meta-analyses when available
- **Context:** Consider training data bias (over-represented topics)

### Limitations and Caveats

- **Automated Extraction:** Coordinates extracted automatically (some errors possible)
- **Publication Bias:** Training data reflects published literature biases
- **Temporal:** Model snapshot from 2020 (check for updates)
- **Reverse Inference Limits:** Brain-to-cognition mapping not one-to-one
- **Not Causal:** Associations, not causal relationships

---

## References

1. **NeuroQuery:**
   - Dockès et al. (2020). NeuroQuery, comprehensive meta-analysis of human brain mapping. *eLife*, 9:e53385.
   - https://elifesciences.org/articles/53385

2. **Comparison with NeuroSynth:**
   - Yarkoni et al. (2011). Large-scale automated synthesis of human functional neuroimaging data. *Nat Methods*, 8:665-670.

3. **Meta-Analytic Methods:**
   - Poldrack (2011). Inferring mental states from neuroimaging data: from reverse inference to large-scale decoding. *Neuron*, 72:692-697.
   - Wager et al. (2007). Meta-analysis of functional neuroimaging data: current and future directions. *Soc Cogn Affect Neurosci*, 2:150-158.

4. **Natural Language Processing:**
   - Mikolov et al. (2013). Distributed representations of words and phrases. *NeurIPS*, 26.

**Official Resources:**
- Website: https://neuroquery.org
- GitHub: https://github.com/neuroquery/neuroquery
- Documentation: https://neuroquery.github.io/neuroquery/
- Paper: https://elifesciences.org/articles/53385
