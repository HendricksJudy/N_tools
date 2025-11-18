# TorchIO: PyTorch-Based Medical Image Preprocessing and Augmentation

## Overview

**TorchIO** is a Python library for efficient loading, preprocessing, augmentation, and patch-based sampling of 3D medical images in deep learning pipelines. Built on PyTorch, TorchIO provides GPU-accelerated transforms specifically designed for medical imaging (unlike generic computer vision libraries), supports 4D data (3D + time), handles metadata preservation, and integrates seamlessly with PyTorch DataLoaders for training neural networks.

### Key Features

- **Medical-Specific Augmentation**: Elastic deformation, bias field, motion artifacts, ghosting
- **Efficient 3D/4D Processing**: Memory-efficient handling of large medical volumes
- **Patch-Based Sampling**: Grid, uniform, and weighted samplers for training
- **GPU Acceleration**: CUDA-enabled transforms for faster preprocessing
- **PyTorch Integration**: Native Dataset and DataLoader compatibility
- **Metadata Preservation**: DICOM and NIfTI headers maintained through pipeline
- **Reproducible Transforms**: Seeded random augmentations for reproducibility
- **Queue System**: Memory-efficient patch queue for training
- **Label Map Support**: Proper handling of segmentation masks
- **Preprocessing Pipeline**: Composable transforms with easy configuration

### Scientific Foundation

Medical image preprocessing differs fundamentally from natural images:

- **3D/4D Structure**: Volumetric data with spatial relationships
- **Anisotropic Resolution**: Different voxel spacing in x, y, z dimensions
- **Physical Coordinates**: Real-world coordinates (mm) vs voxel indices
- **Domain-Specific Artifacts**: MRI bias field, motion, ghosting, aliasing
- **Large Memory Requirements**: 3D volumes can exceed GPU memory

TorchIO addresses these challenges with medical imaging-specific transforms, efficient memory management, and integration with PyTorch for end-to-end differentiable pipelines.

### Primary Use Cases

1. **Deep Learning Training**: Preprocess and augment data for CNNs
2. **Segmentation Models**: nnU-Net, U-Net, V-Net preprocessing
3. **Classification Tasks**: Brain tumor grading, disease detection
4. **Multi-Modal Imaging**: T1, T2, FLAIR, DWI fusion
5. **4D fMRI**: Temporal augmentation and preprocessing
6. **Patch-Based Learning**: Handle large volumes with limited GPU memory

---

## Installation

### Using pip (Recommended)

```bash
# Install TorchIO
pip install torchio

# Verify installation
python -c "import torchio; print(torchio.__version__)"
```

### Using conda

```bash
# Create environment with TorchIO
conda create -n torchio-env python=3.9
conda activate torchio-env
pip install torchio

# Install PyTorch with CUDA support (for GPU acceleration)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/fepegar/torchio.git
cd torchio

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Verify GPU Support

```python
import torch
import torchio as tio

print(f"TorchIO version: {tio.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

---

## Loading Medical Images

### Subject and Image Classes

```python
import torchio as tio
from pathlib import Path

# Define a subject with multiple images
subject = tio.Subject(
    t1=tio.ScalarImage('sub-01_T1w.nii.gz'),
    t2=tio.ScalarImage('sub-01_T2w.nii.gz'),
    flair=tio.ScalarImage('sub-01_FLAIR.nii.gz'),
    label=tio.LabelMap('sub-01_seg.nii.gz'),  # Segmentation mask
    age=45,  # Additional metadata
    diagnosis='control'
)

# Access images
print(f"T1 shape: {subject.t1.shape}")  # (C, W, H, D)
print(f"T1 spacing: {subject.t1.spacing}")  # Voxel size in mm
print(f"Age: {subject.age}")

# Get image data as numpy array
t1_data = subject.t1.numpy()  # Returns numpy array
t1_tensor = subject.t1.data  # Returns PyTorch tensor

print(f"T1 data type: {type(t1_data)}")
print(f"T1 tensor shape: {t1_tensor.shape}")
```

### Load Multiple Subjects

```python
# Create dataset from multiple subjects
subjects = []

subject_ids = ['01', '02', '03', '04', '05']
for sub_id in subject_ids:
    subject = tio.Subject(
        t1=tio.ScalarImage(f'sub-{sub_id}_T1w.nii.gz'),
        t2=tio.ScalarImage(f'sub-{sub_id}_T2w.nii.gz'),
        label=tio.LabelMap(f'sub-{sub_id}_seg.nii.gz'),
        subject_id=sub_id
    )
    subjects.append(subject)

# Create SubjectsDataset
dataset = tio.SubjectsDataset(subjects)

print(f"Dataset size: {len(dataset)}")
print(f"First subject: {dataset[0]}")
```

### Handle Missing Modalities

```python
# Some subjects may have missing modalities
subject_with_missing = tio.Subject(
    t1=tio.ScalarImage('sub-01_T1w.nii.gz'),
    t2=tio.ScalarImage('sub-01_T2w.nii.gz') if Path('sub-01_T2w.nii.gz').exists() else None,
    label=tio.LabelMap('sub-01_seg.nii.gz'),
)

# Check if modality exists
if subject_with_missing.t2 is not None:
    print("T2 is available")
else:
    print("T2 is missing")
```

---

## Preprocessing Transforms

### Resampling

```python
# Resample to isotropic 1mm resolution
resample = tio.Resample(target=1.0)  # 1mm isotropic

# Apply to subject
resampled_subject = resample(subject)

print(f"Original spacing: {subject.t1.spacing}")
print(f"Resampled spacing: {resampled_subject.t1.spacing}")
print(f"Original shape: {subject.t1.shape}")
print(f"Resampled shape: {resampled_subject.t1.shape}")
```

### Cropping and Padding

```python
# Crop to brain mask
crop_mask = tio.CropOrPad(
    target_shape=(128, 128, 128),
    mask_name='label'  # Use label as mask
)

# Center crop to fixed size
crop_center = tio.CropOrPad((160, 192, 160))

# Apply cropping
cropped_subject = crop_center(subject)

print(f"Cropped shape: {cropped_subject.t1.shape}")
```

### Intensity Normalization

```python
# Z-score normalization (standardization)
znorm = tio.ZNormalization(masking_method='label')  # Normalize only within mask

# Rescale to [0, 1]
rescale = tio.RescaleIntensity(out_min_max=(0, 1))

# Rescale to specific percentiles
rescale_percentile = tio.RescaleIntensity(
    percentiles=(1, 99),  # Clip outliers
    out_min_max=(0, 1)
)

# Apply normalization
normalized_subject = znorm(subject)

print(f"Original intensity range: {subject.t1.data.min():.2f} to {subject.t1.data.max():.2f}")
print(f"Normalized mean: {normalized_subject.t1.data.mean():.2f}")
print(f"Normalized std: {normalized_subject.t1.data.std():.2f}")
```

### Histogram Standardization

```python
# Histogram standardization across dataset
# First, collect landmarks from training set
from torchio.transforms import HistogramStandardization

# Train on dataset
landmarks = HistogramStandardization.train(
    [subject.t1 for subject in subjects],
    output_path='landmarks.npy'
)

# Apply histogram standardization
hist_std = tio.HistogramStandardization({'t1': landmarks})
standardized_subject = hist_std(subject)

print("Histogram standardization applied")
```

---

## Data Augmentation

### Spatial Transforms

```python
# Random affine transformation
random_affine = tio.RandomAffine(
    scales=(0.9, 1.1),  # Random scaling
    degrees=10,  # Random rotation ±10 degrees
    translation=5,  # Random translation ±5mm
    p=0.75  # Probability of applying
)

# Random elastic deformation
random_elastic = tio.RandomElasticDeformation(
    num_control_points=7,  # Coarse deformation
    max_displacement=7.5,  # Maximum displacement in mm
    locked_borders=2,  # Lock image borders
    p=0.5
)

# Random flip
random_flip = tio.RandomFlip(
    axes=('LR',),  # Left-right flip only
    flip_probability=0.5
)

# Apply augmentations
augmented = random_affine(subject)
augmented = random_elastic(augmented)
augmented = random_flip(augmented)

print("Spatial augmentation applied")
```

### Intensity Transforms

```python
# Random bias field (MRI artifact)
random_bias = tio.RandomBiasField(
    coefficients=0.5,  # Strength of bias field
    p=0.5
)

# Random noise
random_noise = tio.RandomNoise(
    mean=0,
    std=(0, 0.1),  # Random std up to 0.1
    p=0.5
)

# Random blur
random_blur = tio.RandomBlur(
    std=(0, 2),  # Random blur std
    p=0.5
)

# Random gamma correction
random_gamma = tio.RandomGamma(
    log_gamma=(-0.3, 0.3),  # Random gamma range
    p=0.5
)

# Apply intensity augmentations
augmented = random_bias(subject)
augmented = random_noise(augmented)
augmented = random_blur(augmented)
augmented = random_gamma(augmented)

print("Intensity augmentation applied")
```

### Medical-Specific Augmentations

```python
# Simulate motion artifacts
random_motion = tio.RandomMotion(
    degrees=10,  # Rotation during acquisition
    translation=10,  # Translation during acquisition
    num_transforms=2,  # Number of motion events
    p=0.25
)

# Simulate ghosting artifacts
random_ghosting = tio.RandomGhosting(
    num_ghosts=(4, 10),  # Number of ghosts
    axes=(0, 1, 2),  # All axes
    intensity=(0.5, 1),  # Ghost intensity
    p=0.25
)

# Simulate spike artifacts (k-space)
random_spike = tio.RandomSpike(
    num_spikes=1,  # Number of spikes
    intensity=(1, 3),  # Spike intensity
    p=0.25
)

# Apply MRI-specific artifacts
augmented = random_motion(subject)
augmented = random_ghosting(augmented)
augmented = random_spike(augmented)

print("MRI artifact simulation applied")
```

### Compose Transforms

```python
# Create preprocessing and augmentation pipeline
transforms = tio.Compose([
    # Preprocessing
    tio.Resample(1.0),  # Isotropic resampling
    tio.CropOrPad((160, 192, 160)),  # Crop/pad to fixed size
    tio.ZNormalization(masking_method='label'),  # Z-score normalization
    # Augmentation (with probabilities)
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, p=0.75),
    tio.RandomElasticDeformation(p=0.5),
    tio.RandomBiasField(p=0.5),
    tio.RandomNoise(std=(0, 0.05), p=0.5),
    tio.RandomFlip(axes=('LR',), flip_probability=0.5),
])

# Apply full pipeline
transformed_subject = transforms(subject)

print("Full preprocessing and augmentation pipeline applied")
```

---

## Patch-Based Sampling

### Grid Sampler (Sliding Window)

```python
# Extract patches in a sliding window manner
patch_size = 64
patch_overlap = 8

grid_sampler = tio.GridSampler(
    subject=subject,
    patch_size=patch_size,
    patch_overlap=patch_overlap
)

print(f"Number of patches: {len(grid_sampler)}")

# Iterate over patches
for i, patch in enumerate(grid_sampler):
    if i < 3:  # Show first 3 patches
        print(f"Patch {i}: {patch.t1.shape}, Location: {patch[tio.LOCATION]}")

# Aggregate patches back to volume
aggregator = tio.GridAggregator(grid_sampler)

# Simulate prediction on patches
for patch in grid_sampler:
    # Your model prediction here
    prediction = patch.t1.data  # Placeholder

    # Add to aggregator
    aggregator.add_batch(prediction, patch[tio.LOCATION])

# Get full volume output
output_tensor = aggregator.get_output_tensor()
print(f"Aggregated output shape: {output_tensor.shape}")
```

### Uniform Sampler (Random Patches)

```python
# Extract random patches from subject
uniform_sampler = tio.UniformSampler(patch_size=64)

# Sample patches
num_patches = 10
for i in range(num_patches):
    patch = uniform_sampler(subject)
    print(f"Random patch {i}: {patch.t1.shape}")
```

### Weighted Sampler (Label-Based)

```python
# Sample patches preferentially from regions with labels
weighted_sampler = tio.WeightedSampler(
    patch_size=64,
    probability_map='label'  # Sample more from labeled regions
)

# Sample patches
for i in range(10):
    patch = weighted_sampler(subject)
    label_voxels = (patch.label.data > 0).sum()
    print(f"Patch {i}: {label_voxels} labeled voxels")
```

### Queue for Efficient Training

```python
# Create queue for efficient patch loading during training
subjects_dataset = tio.SubjectsDataset(subjects, transform=transforms)

patches_queue = tio.Queue(
    subjects_dataset=subjects_dataset,
    max_length=100,  # Queue size
    samples_per_volume=10,  # Patches per subject
    sampler=tio.UniformSampler(patch_size=64),
    num_workers=4,  # Parallel loading
    shuffle_subjects=True,
    shuffle_patches=True
)

print(f"Queue length: {len(patches_queue)}")

# Iterate over patches (used in training loop)
for i, patch in enumerate(patches_queue):
    if i >= 5:
        break
    print(f"Queue patch {i}: {patch.t1.shape}")
```

---

## PyTorch Integration

### Create DataLoader

```python
import torch
from torch.utils.data import DataLoader

# Preprocessing pipeline
preprocessing = tio.Compose([
    tio.Resample(1.0),
    tio.CropOrPad((128, 128, 128)),
    tio.ZNormalization(),
])

# Create dataset
subjects_dataset = tio.SubjectsDataset(subjects, transform=preprocessing)

# Create DataLoader for full volumes
dataloader = DataLoader(
    subjects_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch_idx, batch_subjects in enumerate(dataloader):
    # batch_subjects is a batch of subjects
    t1_batch = batch_subjects['t1'][tio.DATA]  # (B, C, W, H, D)
    labels_batch = batch_subjects['label'][tio.DATA]

    print(f"Batch {batch_idx}: T1 shape {t1_batch.shape}, Labels shape {labels_batch.shape}")

    if batch_idx >= 2:
        break
```

### Training Loop with Patches

```python
# Full training example with patch queue
import torch.nn as nn
import torch.optim as optim

# Define augmentation pipeline
augmentation = tio.Compose([
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, p=0.75),
    tio.RandomElasticDeformation(p=0.5),
    tio.RandomBiasField(p=0.5),
    tio.RandomNoise(std=(0, 0.05), p=0.5),
])

# Combine preprocessing and augmentation
train_transform = tio.Compose([preprocessing, augmentation])

# Create training dataset
train_subjects = tio.SubjectsDataset(subjects, transform=train_transform)

# Create patch queue
train_queue = tio.Queue(
    subjects_dataset=train_subjects,
    max_length=50,
    samples_per_volume=5,
    sampler=tio.UniformSampler(patch_size=64),
    num_workers=4,
    shuffle_subjects=True,
    shuffle_patches=True
)

# Create DataLoader
train_loader = DataLoader(train_queue, batch_size=8)

# Dummy model
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

model = SimpleUNet().cuda() if torch.cuda.is_available() else SimpleUNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        # Get data
        inputs = batch['t1'][tio.DATA]
        targets = batch['label'][tio.DATA].long()

        # Move to GPU
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze(1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if batch_idx >= 20:  # Limit for example
            break

print("Training loop completed")
```

---

## Advanced Preprocessing

### Ensure Shape Divisibility

```python
# Ensure dimensions are divisible by a factor (e.g., for U-Net)
ensure_divisible = tio.EnsureShapeMultiple(8)  # Divisible by 8

subject_divisible = ensure_divisible(subject)
print(f"Original shape: {subject.t1.shape}")
print(f"Adjusted shape: {subject_divisible.t1.shape}")
```

### One-Hot Encoding for Labels

```python
# Convert label map to one-hot encoding
one_hot = tio.OneHot(num_classes=3)  # 3 classes (background, tissue1, tissue2)

subject_onehot = one_hot(subject)
print(f"Label shape before: {subject.label.shape}")
print(f"Label shape after: {subject_onehot.label.shape}")  # (3, W, H, D)
```

### Keep Largest Component

```python
# Keep only the largest connected component in label
keep_largest = tio.KeepLargestComponent()

cleaned_subject = keep_largest(subject)
print("Kept largest connected component in label")
```

### Label Remapping

```python
# Remap label values
# Example: Merge labels 1, 2 -> 1; label 3 -> 2
remap_labels = tio.RemapLabels(remapping={1: 1, 2: 1, 3: 2})

remapped_subject = remap_labels(subject)
print("Label values remapped")
```

---

## Quality Control and Visualization

### Visualize Transforms

```python
import matplotlib.pyplot as plt

# Apply transform
augmented_subject = random_elastic(subject)

# Get middle slices
original_slice = subject.t1.data[0, :, :, subject.t1.shape[-1]//2]
augmented_slice = augmented_subject.t1.data[0, :, :, augmented_subject.t1.shape[-1]//2]

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(original_slice.T, cmap='gray', origin='lower')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(augmented_slice.T, cmap='gray', origin='lower')
axes[1].set_title('Augmented (Elastic Deformation)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('augmentation_comparison.png', dpi=150)
print("Saved augmentation comparison")
```

### Check Transform Effects

```python
# Check multiple augmentations
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

transforms_to_check = [
    ('Original', lambda x: x),
    ('Affine', random_affine),
    ('Elastic', random_elastic),
    ('Bias Field', random_bias),
    ('Motion', random_motion),
    ('Ghosting', random_ghosting),
]

for idx, (name, transform) in enumerate(transforms_to_check):
    ax = axes[idx // 3, idx % 3]

    if name == 'Original':
        img = subject
    else:
        img = transform(subject)

    slice_data = img.t1.data[0, :, :, img.t1.shape[-1]//2]
    ax.imshow(slice_data.T, cmap='gray', origin='lower')
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.savefig('augmentation_gallery.png', dpi=150)
```

### Validate Pipeline

```python
def validate_preprocessing_pipeline(subjects, transform):
    """Validate that preprocessing doesn't introduce errors"""

    issues = []

    for i, subject in enumerate(subjects):
        try:
            # Apply transform
            transformed = transform(subject)

            # Check for NaN/Inf
            for image_name, image in transformed.get_images_dict().items():
                data = image.data

                if torch.isnan(data).any():
                    issues.append(f"Subject {i}, {image_name}: Contains NaN")

                if torch.isinf(data).any():
                    issues.append(f"Subject {i}, {image_name}: Contains Inf")

            # Check shapes match across modalities
            shapes = [img.shape for img in transformed.get_images_dict().values()]
            if len(set(shapes)) > 1:
                issues.append(f"Subject {i}: Inconsistent shapes {shapes}")

        except Exception as e:
            issues.append(f"Subject {i}: Error {str(e)}")

    if issues:
        print("⚠ Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Pipeline validation passed")

    return len(issues) == 0

# Validate
validate_preprocessing_pipeline(subjects[:5], transforms)
```

---

## Batch Processing

### Preprocess Dataset Offline

```python
from pathlib import Path
import nibabel as nib

def batch_preprocess_subjects(subjects, transform, output_dir):
    """Preprocess and save all subjects"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        subject_id = subject.subject_id
        print(f"Processing {subject_id}...")

        # Apply transform
        transformed = transform(subject)

        # Save preprocessed images
        subject_dir = output_dir / f'sub-{subject_id}'
        subject_dir.mkdir(exist_ok=True)

        for image_name, image in transformed.get_images_dict().items():
            output_path = subject_dir / f'{image_name}.nii.gz'
            image.save(output_path)

        print(f"  ✓ Saved to {subject_dir}")

# Run batch preprocessing
preprocessing_pipeline = tio.Compose([
    tio.Resample(1.0),
    tio.CropOrPad((160, 192, 160)),
    tio.ZNormalization(),
])

batch_preprocess_subjects(
    subjects=subjects,
    transform=preprocessing_pipeline,
    output_dir='./preprocessed'
)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def preprocess_single_subject(subject, transform, output_dir):
    """Process single subject (for parallel execution)"""
    try:
        subject_id = subject.subject_id
        transformed = transform(subject)

        subject_dir = Path(output_dir) / f'sub-{subject_id}'
        subject_dir.mkdir(parents=True, exist_ok=True)

        for image_name, image in transformed.get_images_dict().items():
            output_path = subject_dir / f'{image_name}.nii.gz'
            image.save(output_path)

        return {'subject': subject_id, 'status': 'success'}

    except Exception as e:
        return {'subject': subject.subject_id, 'status': 'failed', 'error': str(e)}

# Parallel batch processing
def parallel_batch_preprocess(subjects, transform, output_dir, max_workers=4):
    """Process subjects in parallel"""

    process_func = partial(
        preprocess_single_subject,
        transform=transform,
        output_dir=output_dir
    )

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, subject) for subject in subjects]

        for future in futures:
            result = future.result()
            results.append(result)

            if result['status'] == 'success':
                print(f"✓ {result['subject']}")
            else:
                print(f"✗ {result['subject']}: {result.get('error', 'unknown error')}")

    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nCompleted: {successful}/{len(subjects)} subjects")

    return results

# Run parallel preprocessing
results = parallel_batch_preprocess(
    subjects=subjects,
    transform=preprocessing_pipeline,
    output_dir='./preprocessed_parallel',
    max_workers=4
)
```

---

## Integration with Deep Learning Frameworks

### nnU-Net Style Preprocessing

```python
# Preprocessing similar to nnU-Net
nnunet_transform = tio.Compose([
    # Resampling to median spacing
    tio.Resample(target=(1.5, 1.5, 1.5)),
    # Crop to non-zero region
    tio.CropOrPad(target_shape=(128, 128, 128)),
    # Intensity normalization
    tio.ZNormalization(masking_method='label'),
    # Data augmentation
    tio.RandomAffine(scales=(0.8, 1.2), degrees=15, p=0.5),
    tio.RandomElasticDeformation(p=0.3),
    tio.RandomBiasField(p=0.3),
])

nnunet_subject = nnunet_transform(subject)
print("nnU-Net style preprocessing applied")
```

### MONAI Compatibility

```python
# TorchIO can work alongside MONAI
# Both use PyTorch tensors and can be combined in pipelines

# TorchIO preprocessing
torchio_transform = tio.Compose([
    tio.Resample(1.0),
    tio.CropOrPad((128, 128, 128)),
])

preprocessed = torchio_transform(subject)

# Extract tensor for MONAI
tensor_for_monai = preprocessed.t1.data  # PyTorch tensor

print(f"Tensor ready for MONAI: {tensor_for_monai.shape}")
print(f"Device: {tensor_for_monai.device}")
```

---

## Troubleshooting

### Memory Issues

```python
# For large volumes, use patch-based processing
# Reduce batch size
# Use mixed precision training

# Check memory usage
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Memory usage: {mem.percent}%")
    print(f"Available: {mem.available / 1e9:.2f} GB")

check_memory()

# Use smaller patches
small_patch_sampler = tio.UniformSampler(patch_size=32)  # Smaller patches

# Reduce queue size
small_queue = tio.Queue(
    subjects_dataset=subjects_dataset,
    max_length=20,  # Smaller queue
    samples_per_volume=5,
    sampler=small_patch_sampler
)
```

### Transform Debugging

```python
# Debug transforms one by one
subject_test = subjects[0]

transforms_list = [
    tio.Resample(1.0),
    tio.CropOrPad((128, 128, 128)),
    tio.ZNormalization(),
    tio.RandomAffine(p=1.0),  # Force application
]

for i, transform in enumerate(transforms_list):
    try:
        subject_test = transform(subject_test)
        print(f"✓ Transform {i} ({transform.__class__.__name__}) successful")
        print(f"  Shape: {subject_test.t1.shape}")
    except Exception as e:
        print(f"✗ Transform {i} ({transform.__class__.__name__}) failed: {e}")
        break
```

### Common Errors

```python
# Error: Shape mismatch between images and labels
# Solution: Ensure transforms apply to all images

# Correct approach
transform_all = tio.Compose([
    tio.Resample(1.0),  # Applies to all images in subject
    tio.CropOrPad((128, 128, 128)),
])

# Error: GPU out of memory
# Solution: Reduce batch size or patch size, use CPU for preprocessing

# Use CPU for data loading
dataloader_cpu = DataLoader(
    subjects_dataset,
    batch_size=1,
    num_workers=4,
    pin_memory=False  # Don't pin memory if GPU memory is limited
)
```

---

## Best Practices

### Recommended Workflow

```python
# 1. Preprocessing (deterministic)
preprocessing = tio.Compose([
    tio.Resample(1.0),
    tio.CropOrPad((160, 192, 160)),
    tio.ZNormalization(),
])

# 2. Augmentation (random, only for training)
augmentation = tio.Compose([
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, p=0.75),
    tio.RandomElasticDeformation(p=0.5),
    tio.RandomBiasField(p=0.5),
    tio.RandomNoise(std=(0, 0.05), p=0.5),
    tio.RandomFlip(axes=('LR',), flip_probability=0.5),
])

# 3. Training pipeline (preprocessing + augmentation)
train_transform = tio.Compose([preprocessing, augmentation])

# 4. Validation/test pipeline (preprocessing only)
val_transform = preprocessing

print("Training and validation transforms configured")
```

### Reproducibility

```python
# Set random seed for reproducibility
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# TorchIO respects PyTorch random seed
augmented_1 = random_affine(subject)
set_seed(42)
augmented_2 = random_affine(subject)

# Check if identical
identical = torch.allclose(augmented_1.t1.data, augmented_2.t1.data)
print(f"Reproducible: {identical}")
```

---

## References

### Key Publications

1. Pérez-García, F., et al. (2021). "TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning." *Computer Methods and Programs in Biomedicine*, 208, 106236.

2. Isensee, F., et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 18(2), 203-211.

### Documentation and Resources

- **Documentation**: https://torchio.readthedocs.io/
- **GitHub**: https://github.com/fepegar/torchio
- **Tutorials**: https://torchio.readthedocs.io/tutorials.html
- **Google Colab Examples**: Available in documentation
- **Example Datasets**: Medical Segmentation Decathlon, BraTS

### Related Tools

- **MONAI**: Medical imaging deep learning framework
- **nnU-Net**: Self-configuring segmentation framework
- **Nilearn**: Machine learning for neuroimaging
- **SimpleITK**: Medical image processing
- **PyTorch**: Deep learning framework

---

## See Also

- **monai.md**: Medical imaging deep learning framework
- **nnu-net.md**: Automated segmentation framework
- **nilearn.md**: Machine learning for neuroimaging
- **pyradiomics.md**: Radiomics feature extraction
- **neuroharmonize.md**: Multi-site harmonization
