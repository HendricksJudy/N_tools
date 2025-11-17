# MONAI

## Overview

MONAI (Medical Open Network for AI) is a PyTorch-based open-source framework specifically designed for deep learning in medical imaging. It provides domain-optimized implementations of data loading, preprocessing, augmentation, network architectures, training utilities, and deployment tools tailored for healthcare imaging applications, enabling researchers and practitioners to build production-ready medical AI solutions.

**Website:** https://monai.io/
**Platform:** Python/PyTorch
**Language:** Python
**License:** Apache 2.0

## Key Features

- PyTorch-based framework for medical imaging
- Domain-specific data transformations
- Medical image formats (NIfTI, DICOM, etc.)
- Pre-built network architectures (U-Net, SegResNet, etc.)
- Model Zoo with pre-trained weights
- Auto3DSeg for automated segmentation
- MONAI Label for interactive annotation
- Sliding window inference for large 3D volumes
- GPU acceleration and mixed precision
- Distributed training support
- Production deployment tools (ONNX, TorchScript)
- Comprehensive tutorials and examples
- Active community and regular updates

## Installation

### Core Installation

```bash
# Install core MONAI
pip install monai

# Verify installation
python -c "import monai; print(monai.__version__)"
```

### Full Installation (Recommended)

```bash
# Install with all dependencies
pip install 'monai[all]'

# Includes:
# - nibabel (NIfTI support)
# - scikit-image (image processing)
# - pillow (image I/O)
# - tensorboard (visualization)
# - matplotlib (plotting)
# - tqdm (progress bars)
# - lmdb (caching)
# - psutil (system monitoring)
# - cucim (GPU image processing)
# - openslide (whole slide imaging)
```

### From Source

```bash
# Clone repository
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI

# Install in development mode
pip install -e '.[all]'

# Run tests
python -m pytest tests/
```

### Docker

```bash
# Pull official MONAI Docker image
docker pull projectmonai/monai:latest

# Run with GPU
docker run --gpus all -it \
  -v /path/to/data:/workspace/data \
  projectmonai/monai:latest

# Jupyter notebook version
docker pull projectmonai/monai:latest-jupyter
docker run --gpus all -p 8888:8888 projectmonai/monai:latest-jupyter
```

## Core Components

### Transforms (Preprocessing)

```python
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, ToTensord
)

# Define preprocessing pipeline
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),  # Load NIfTI/DICOM
    EnsureChannelFirstd(keys=["image", "label"]),  # Add channel dim
    Spacingd(keys=["image", "label"],
             pixdim=(1.0, 1.0, 1.0),  # Resample to 1mm isotropic
             mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),  # Standard orientation
    ScaleIntensityRanged(keys=["image"],
                         a_min=-200, a_max=200,  # CT window
                         b_min=0.0, b_max=1.0,  # Normalize to [0,1]
                         clip=True),
    RandCropByPosNegLabeld(keys=["image", "label"],
                           label_key="label",
                           spatial_size=(96, 96, 96),  # Crop size
                           pos=1, neg=1,  # Balance pos/neg samples
                           num_samples=4),  # Samples per image
    ToTensord(keys=["image", "label"])
])
```

### Data Loading

```python
from monai.data import Dataset, DataLoader
from monai.data import CacheDataset  # Cached for speed

# Define data
data_dicts = [
    {"image": "/data/sub01/T1.nii.gz", "label": "/data/sub01/seg.nii.gz"},
    {"image": "/data/sub02/T1.nii.gz", "label": "/data/sub02/seg.nii.gz"},
    # ...
]

# Create dataset
train_ds = CacheDataset(
    data=data_dicts,
    transform=train_transforms,
    cache_rate=1.0,  # Cache all transformed data
    num_workers=4
)

# Create data loader
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### Network Architectures

```python
from monai.networks.nets import UNet, SegResNet, DynUNet

# 3D U-Net
model = UNet(
    spatial_dims=3,
    in_channels=1,  # Single modality (e.g., T1)
    out_channels=2,  # Binary segmentation (background, foreground)
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
)

# SegResNet (state-of-the-art)
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=32,
    dropout_prob=0.2
)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Loss Functions

```python
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss

# Dice loss (common for segmentation)
dice_loss = DiceLoss(
    to_onehot_y=True,  # Convert labels to one-hot
    softmax=True,  # Apply softmax to predictions
    squared_pred=True  # Use squared denominator
)

# Combined Dice + Cross-Entropy
dice_ce_loss = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    lambda_dice=0.5,  # Weight for Dice
    lambda_ce=0.5  # Weight for CE
)

# Tversky loss (for imbalanced data)
tversky_loss = TverskyLoss(
    to_onehot_y=True,
    softmax=True,
    alpha=0.7,  # False negative penalty
    beta=0.3  # False positive penalty
)
```

### Metrics

```python
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# Dice metric
dice_metric = DiceMetric(
    include_background=False,  # Exclude background class
    reduction="mean",
    get_not_nans=False
)

# Hausdorff distance
hausdorff_metric = HausdorffDistanceMetric(
    include_background=False,
    percentile=95  # 95th percentile HD
)
```

## Basic Segmentation Workflow

### Training Loop

```python
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# Setup
device = torch.device("cuda")
model = UNet(...).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training
max_epochs = 100
val_interval = 5

for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0

    for batch_data in train_loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Train loss: {epoch_loss:.4f}")

    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                # Sliding window inference (for large volumes)
                val_outputs = sliding_window_inference(
                    val_inputs, (96, 96, 96), 4, model
                )

                # Compute metric
                dice_metric(y_pred=val_outputs, y=val_labels)

            # Aggregate metric
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"Val Dice: {metric:.4f}")

# Save model
torch.save(model.state_dict(), "best_model.pth")
```

### Inference

```python
from monai.inferers import sliding_window_inference
import nibabel as nib

# Load trained model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Inference on new data
test_data = {"image": "/data/test/sub99/T1.nii.gz"}

test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRanged(keys=["image"],
                         a_min=-200, a_max=200,
                         b_min=0.0, b_max=1.0, clip=True),
    ToTensord(keys=["image"])
])

# Transform and predict
test_input = test_transforms(test_data)["image"].unsqueeze(0).to(device)

with torch.no_grad():
    test_output = sliding_window_inference(
        test_input, (96, 96, 96), 4, model, overlap=0.5
    )

# Post-process
test_output = torch.argmax(test_output, dim=1).cpu().numpy()[0]

# Save result
nib.save(nib.Nifti1Image(test_output, affine=np.eye(4)),
         "prediction.nii.gz")
```

## Pre-trained Models (Model Zoo)

### Available Models

```python
from monai.apps import download_and_extract
from monai.bundle import download, load

# List available bundles
from monai.bundle import get_all_bundles_list
bundles = get_all_bundles_list()
print(bundles)

# Download pre-trained model
download(name="spleen_ct_segmentation", bundle_dir="./models")

# Load and use
from monai.bundle import ConfigParser
config = ConfigParser()
config.read_config("./models/spleen_ct_segmentation/configs/inference.json")

# Get model
model = config.get_parsed_content("network_def")
```

### Fine-tuning Pre-trained Model

```python
from monai.bundle import download, load

# Download pre-trained weights
download(name="spleen_ct_segmentation", bundle_dir="./models")

# Load model
model = load(
    name="spleen_ct_segmentation",
    bundle_dir="./models",
    source="monaihosting",
    progress=True
)

# Fine-tune on your data
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Lower LR

for epoch in range(10):  # Fewer epochs for fine-tuning
    # Training loop as before
    pass
```

## Auto3DSeg (Automated Segmentation)

### Quick Start

```bash
# Install Auto3DSeg
pip install 'monai[auto3dseg]'

# Create Auto3DSeg project
python -m monai.apps.auto3dseg AutoRunner \
    --input /path/to/data \
    --work_dir /path/to/output \
    --modality CT \
    --datalist dataset.json
```

### Auto3DSeg Python API

```python
from monai.apps.auto3dseg import AutoRunner

# Define data list
datalist = {
    "training": [
        {"image": "/data/sub01/img.nii.gz", "label": "/data/sub01/seg.nii.gz"},
        {"image": "/data/sub02/img.nii.gz", "label": "/data/sub02/seg.nii.gz"},
    ],
    "testing": [
        {"image": "/data/test01/img.nii.gz"},
    ]
}

# Create Auto3DSeg runner
runner = AutoRunner(
    work_dir="./auto3dseg_work",
    input=datalist,
    algos="segresnet",  # or "dints", "swinunetr"
    analyze=True,
    train=True,
    hpo=True  # Hyperparameter optimization
)

# Run pipeline
runner.run()

# Inference
runner.infer()
```

## MONAI Label (Interactive Annotation)

### Setup MONAI Label Server

```bash
# Install MONAI Label
pip install monai-label

# Download sample app
monai-label apps --download --name radiology --output apps

# Start server
monai-label start_server \
    --app apps/radiology \
    --studies /path/to/unlabeled/data \
    --conf models segmentation

# Server runs at http://localhost:8000
```

### Integration with 3D Slicer

```bash
# In 3D Slicer:
# 1. Install MONAI Label plugin from Extension Manager
# 2. MONAI Label module appears
# 3. Connect to server: http://localhost:8000
# 4. Select model and click "Next Sample"
# 5. Auto-segment or manually annotate
# 6. Submit annotation (trains model online)
# 7. Repeat for active learning
```

### Custom MONAI Label App

```python
from monai.apps.label import LabelApp
from monai.networks.nets import SegResNet

class MyLabelApp(LabelApp):
    def __init__(self):
        super().__init__(
            app_dir="/path/to/app",
            studies="/path/to/data"
        )

    def init_networks(self):
        return {
            "segmentation": SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2
            )
        }

    def init_infers(self):
        # Define inference strategy
        pass

    def init_trainers(self):
        # Define training strategy
        pass

# Run custom app
app = MyLabelApp()
app.start_server(port=8000)
```

## Advanced Features

### Sliding Window Inference

```python
from monai.inferers import sliding_window_inference

# For large 3D volumes that don't fit in memory
def inference(input_image):
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_image,
            roi_size=(96, 96, 96),  # Window size
            sw_batch_size=4,  # Batch size for windows
            predictor=model,
            overlap=0.5,  # Overlap between windows
            mode="gaussian",  # Blending mode
            sigma_scale=0.125  # Gaussian sigma
        )
    return output
```

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

# Setup
scaler = GradScaler()

# Training loop with mixed precision
for batch_data in train_loader:
    inputs = batch_data["image"].to(device)
    labels = batch_data["label"].to(device)

    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Distributed Training

```python
import torch.distributed as dist
from monai.data import partition_dataset

# Initialize distributed training
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Partition dataset across GPUs
data_partitions = partition_dataset(
    data=data_dicts,
    num_partitions=world_size,
    shuffle=True,
    seed=42
)[rank]

# Create data loader for this rank
train_loader = DataLoader(
    CacheDataset(data_partitions, train_transforms),
    batch_size=2,
    shuffle=True
)

# Wrap model with DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[rank])

# Training proceeds as normal
```

## Deployment

### Export to ONNX

```python
import torch.onnx

# Load trained model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 1, 96, 96, 96).to(device)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Verify
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### TorchScript

```python
# Trace model
model.eval()
traced_model = torch.jit.trace(model, dummy_input)

# Save
traced_model.save("model_traced.pt")

# Load and use
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(new_input)
```

### REST API Deployment

```python
from flask import Flask, request, jsonify
import torch
import numpy as np

app = Flask(__name__)

# Load model
model = UNet(...).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Receive data
    data = request.json
    image_path = data['image_path']

    # Load and preprocess
    # ... preprocessing code ...

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process and return
    result = output.cpu().numpy()
    return jsonify({'prediction': result.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Data Augmentation

### Spatial Augmentations

```python
from monai.transforms import (
    RandRotate90d, RandFlipd, RandAffined,
    RandElasticd, RandZoomd
)

spatial_augmentations = Compose([
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandAffined(
        keys=["image", "label"],
        prob=0.5,
        rotate_range=(0.1, 0.1, 0.1),
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest")
    ),
    RandElasticd(
        keys=["image", "label"],
        prob=0.5,
        sigma_range=(5, 7),
        magnitude_range=(50, 150),
        mode=("bilinear", "nearest")
    )
])
```

### Intensity Augmentations

```python
from monai.transforms import (
    RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd, RandShiftIntensityd
)

intensity_augmentations = Compose([
    RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
    RandGaussianSmoothd(keys=["image"], prob=0.5,
                        sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
    RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
    RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1)
])
```

## Integration with Other Tools

### From BIDS to MONAI

```python
from pathlib import Path

# BIDS dataset
bids_dir = Path("/data/bids_dataset")
subjects = list(bids_dir.glob("sub-*"))

# Convert to MONAI format
data_dicts = []
for subj in subjects:
    subj_id = subj.name
    t1_file = subj / "anat" / f"{subj_id}_T1w.nii.gz"

    # Add segmentation if available
    seg_file = subj / "derivatives" / "seg" / f"{subj_id}_seg.nii.gz"

    if t1_file.exists() and seg_file.exists():
        data_dicts.append({
            "image": str(t1_file),
            "label": str(seg_file)
        })

# Use in MONAI
train_ds = CacheDataset(data_dicts, train_transforms)
```

### Export to Other Formats

```python
import nibabel as nib

# MONAI tensor to NIfTI
def save_prediction(tensor, affine, filename):
    """Save MONAI prediction as NIfTI."""
    array = tensor.cpu().numpy()

    # Remove batch and channel dimensions if present
    if array.ndim == 5:  # [B, C, H, W, D]
        array = array[0, 0]
    elif array.ndim == 4:  # [C, H, W, D]
        array = array[0]

    img = nib.Nifti1Image(array, affine)
    nib.save(img, filename)

# Use
save_prediction(output, affine, "prediction.nii.gz")
```

## Integration with Claude Code

When helping users with MONAI:

1. **Check Installation:**
   ```python
   import monai
   print(monai.__version__)
   print(monai.config.print_config())
   ```

2. **Verify GPU:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

3. **Common Workflow:**
   - Data preparation → Transforms → Dataset → DataLoader → Model → Training → Inference

4. **Best Practices:**
   - Use CacheDataset for faster training
   - Sliding window inference for large volumes
   - Mixed precision for speed and memory
   - Pre-trained models when available

## Troubleshooting

**Problem:** CUDA out of memory
**Solution:** Reduce batch size, use smaller patches, enable mixed precision, clear cache

**Problem:** Slow data loading
**Solution:** Use CacheDataset, increase num_workers, use LMDB backend

**Problem:** Poor segmentation results
**Solution:** Check data quality, try different augmentations, increase training time, use pre-trained weights

**Problem:** Import errors
**Solution:** Install full dependencies with `pip install 'monai[all]'`

**Problem:** Inconsistent results
**Solution:** Set random seeds, use deterministic algorithms, check for data leakage

## Best Practices

1. **Use CacheDataset** for faster iteration during development
2. **Start with pre-trained models** from Model Zoo
3. **Use sliding window inference** for large 3D volumes
4. **Enable mixed precision** for GPU memory and speed
5. **Validate on held-out data** not used during training
6. **Monitor training** with TensorBoard
7. **Use transforms consistently** between train and validation
8. **Save checkpoints regularly** during training
9. **Document hyperparameters** for reproducibility
10. **Test on diverse data** before deployment

## Resources

- **Website:** https://monai.io/
- **GitHub:** https://github.com/Project-MONAI/MONAI
- **Documentation:** https://docs.monai.io/
- **Tutorials:** https://github.com/Project-MONAI/tutorials
- **Model Zoo:** https://github.com/Project-MONAI/model-zoo
- **MONAI Label:** https://github.com/Project-MONAI/MONAILabel
- **Forum:** https://github.com/Project-MONAI/MONAI/discussions
- **Paper:** https://arxiv.org/abs/2211.02701

## Citation

```bibtex
@article{cardoso2022monai,
  title={MONAI: An open-source framework for deep learning in healthcare},
  author={Cardoso, M Jorge and Li, Wenqi and Brown, Richard and Ma, Nic and Kerfoot, Eric and Wang, Yiheng and Murrey, Benjamin and Myronenko, Andriy and Zhao, Can and Yang, Dong and others},
  journal={arXiv preprint arXiv:2211.02701},
  year={2022}
}
```

## Related Tools

- **PyTorch:** Underlying deep learning framework
- **nnU-Net:** Self-configuring segmentation (can be used with MONAI)
- **FastSurfer:** Brain parcellation (similar deep learning approach)
- **SynthSeg:** Robust segmentation (alternative framework)
- **TorchIO:** Medical imaging preprocessing (alternative)
- **NiBabel:** Medical image I/O (used by MONAI)
- **SimpleITK:** Image processing library
- **3D Slicer:** Integration via MONAI Label
