# nnU-Net

## Overview

nnU-Net (no-new-UNet) is a self-configuring framework for deep learning-based biomedical image segmentation. It automatically adapts to any dataset without manual intervention, achieving state-of-the-art performance across diverse medical imaging tasks. By automating preprocessing, network architecture configuration, and training strategies, nnU-Net has won numerous segmentation challenges and become the de facto standard for medical image segmentation.

**Website:** https://github.com/MIC-DKFZ/nnUNet
**Platform:** Python/PyTorch
**Language:** Python
**License:** Apache 2.0

## Key Features

- Fully automated pipeline configuration
- Self-adapting to any dataset
- State-of-the-art segmentation performance
- Multiple configurations (2D, 3D full-res, 3D cascade)
- Automatic preprocessing and augmentation
- Ensemble predictions for optimal results
- Pre-trained model zoo
- GPU and CPU support
- Docker containers available
- Extensive validation and cross-validation
- Works with any medical imaging modality
- Reproducible and well-documented

## Installation

### Prerequisites

```bash
# Python 3.9 or later
python --version

# CUDA for GPU support (highly recommended)
nvidia-smi

# Verify PyTorch with CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Install via pip

```bash
# Create virtual environment (recommended)
conda create -n nnunet python=3.10
conda activate nnunet

# Install nnU-Net
pip install nnunetv2

# Verify installation
nnUNetv2_plan_and_preprocess -h
```

### Install from source

```bash
# Clone repository
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet

# Install in development mode
pip install -e .

# Verify
python -c "import nnunetv2; print(nnunetv2.__version__)"
```

### Docker Installation

```bash
# Pull official Docker image
docker pull ghcr.io/mic-dkfz/nnunet:latest

# Run with GPU support
docker run --gpus all -it \
  -v /path/to/data:/data \
  -v /path/to/results:/results \
  ghcr.io/mic-dkfz/nnunet:latest

# Or build custom image
docker build -t nnunet:custom .
```

## Environment Setup

### Required Directories

```bash
# Set nnU-Net environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Add to ~/.bashrc for persistence
echo 'export nnUNet_raw="/path/to/nnUNet_raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="/path/to/nnUNet_results"' >> ~/.bashrc

# Create directories
mkdir -p $nnUNet_raw
mkdir -p $nnUNet_preprocessed
mkdir -p $nnUNet_results
```

## Dataset Preparation

### nnU-Net Dataset Format

```bash
# Dataset structure
nnUNet_raw/
└── Dataset001_BrainTumor/
    ├── dataset.json
    ├── imagesTr/
    │   ├── case_00000_0000.nii.gz  # Modality 0
    │   ├── case_00000_0001.nii.gz  # Modality 1 (if multi-modal)
    │   ├── case_00001_0000.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── case_00000.nii.gz
    │   ├── case_00001.nii.gz
    │   └── ...
    └── imagesTs/  # Optional test set
        ├── case_test_00000_0000.nii.gz
        └── ...

# Naming convention:
# Images: {CASE_ID}_{MODALITY}.nii.gz
# Labels: {CASE_ID}.nii.gz
```

### Create dataset.json

```python
import json

dataset_json = {
    "channel_names": {
        "0": "T1",
        "1": "T2"  # If multi-modal
    },
    "labels": {
        "background": 0,
        "tumor": 1,
        "edema": 2
    },
    "numTraining": 100,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "NibabelIOWithReorient"
}

# Save dataset.json
with open('nnUNet_raw/Dataset001_BrainTumor/dataset.json', 'w') as f:
    json.dump(dataset_json, f, indent=4)
```

### Convert Your Data to nnU-Net Format

```python
import nibabel as nib
import os
from pathlib import Path

# Source data paths
source_images = "/path/to/your/images"
source_labels = "/path/to/your/labels"

# nnU-Net paths
target_images = Path(os.environ['nnUNet_raw']) / 'Dataset001_BrainTumor/imagesTr'
target_labels = Path(os.environ['nnUNet_raw']) / 'Dataset001_BrainTumor/labelsTr'

target_images.mkdir(parents=True, exist_ok=True)
target_labels.mkdir(parents=True, exist_ok=True)

# Convert files
for i, (img_file, label_file) in enumerate(zip(sorted(os.listdir(source_images)),
                                                  sorted(os.listdir(source_labels)))):
    # Load and save images
    img = nib.load(os.path.join(source_images, img_file))
    nib.save(img, target_images / f"case_{i:05d}_0000.nii.gz")

    # Load and save labels
    label = nib.load(os.path.join(source_labels, label_file))
    nib.save(label, target_labels / f"case_{i:05d}.nii.gz")

print(f"Converted {i+1} cases to nnU-Net format")
```

## Training Pipeline

### Step 1: Experiment Planning and Preprocessing

```bash
# Automatic planning and preprocessing
# Analyzes dataset and configures everything automatically
nnUNetv2_plan_and_preprocess -d DATASET_ID -c CONFIG

# Example: Dataset 001, all configurations
nnUNetv2_plan_and_preprocess -d 1 -c 2d 3d_fullres 3d_lowres

# This creates:
# - Experiment plans
# - Preprocessed data
# - Configuration files

# Check preprocessing output
ls $nnUNet_preprocessed/Dataset001_BrainTumor/
```

### Step 2: Training

```bash
# Train a model
nnUNetv2_train DATASET_ID CONFIG FOLD

# Example: Train 3D full resolution, fold 0
nnUNetv2_train 1 3d_fullres 0 --npz

# Train all 5 folds for cross-validation
for fold in 0 1 2 3 4; do
    nnUNetv2_train 1 3d_fullres $fold --npz
done

# GPU selection
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1 3d_fullres 0

# Multi-GPU training
nnUNetv2_train 1 3d_fullres 0 --npz -num_gpus 2
```

### Training Configurations

```bash
# 2D U-Net (fastest, good for 2D data)
nnUNetv2_train 1 2d 0

# 3D full resolution (most common)
nnUNetv2_train 1 3d_fullres 0

# 3D low resolution (for large images)
nnUNetv2_train 1 3d_lowres 0

# 3D cascade (low-res → high-res refinement)
# First train low-res
nnUNetv2_train 1 3d_lowres 0
# Then train cascade
nnUNetv2_train 1 3d_cascade_fullres 0
```

### Monitor Training

```bash
# Training progress saved to:
# $nnUNet_results/Dataset001_BrainTumor/nnUNetTrainer__nnUNetPlans__CONFIG/fold_X/

# View training log
tail -f $nnUNet_results/Dataset001_BrainTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log*.txt

# Plot training curves
# Logged metrics: train loss, val loss, dice scores
```

## Inference

### Single Model Prediction

```bash
# Predict with single trained model
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER \
    -d DATASET_ID -c CONFIG -f FOLD

# Example
nnUNetv2_predict \
    -i /path/to/test/images \
    -o /path/to/predictions \
    -d 1 -c 3d_fullres -f 0

# Specify checkpoint (default: best)
nnUNetv2_predict -i INPUT -o OUTPUT \
    -d 1 -c 3d_fullres -f 0 \
    -chk checkpoint_final.pth
```

### Ensemble Prediction (Recommended)

```bash
# Ensemble all folds (best performance)
nnUNetv2_predict \
    -i /path/to/test/images \
    -o /path/to/predictions \
    -d 1 -c 3d_fullres -f all

# Ensemble multiple configurations
nnUNetv2_predict \
    -i /path/to/test/images \
    -o /path/to/predictions \
    -d 1 -c 2d 3d_fullres -f all

# Save probability maps (softmax outputs)
nnUNetv2_predict -i INPUT -o OUTPUT \
    -d 1 -c 3d_fullres -f all \
    --save_probabilities
```

### Batch Processing

```python
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Initialize predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_gpu=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

# Load model
predictor.initialize_from_trained_model_folder(
    model_training_output_dir,
    use_folds=(0, 1, 2, 3, 4),
    checkpoint_name='checkpoint_final.pth'
)

# Predict on multiple images
list_of_images = ['image1.nii.gz', 'image2.nii.gz']
output_folder = '/path/to/output'

predictor.predict_from_files(
    list_of_images,
    output_folder,
    save_probabilities=False,
    overwrite=True,
    num_processes_preprocessing=4,
    num_processes_segmentation_export=4
)
```

## Evaluation

### Cross-Validation Results

```bash
# After training all folds, evaluate cross-validation
nnUNetv2_evaluate_folder \
    -ref $nnUNet_raw/Dataset001_BrainTumor/labelsTr \
    -pred $nnUNet_results/Dataset001_BrainTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/cv_niftis_postprocessed \
    -l 1 2  # Label IDs to evaluate

# Results saved as summary.json
# Contains: Dice, IoU, Hausdorff distance, etc.
```

### Custom Evaluation

```python
import numpy as np
import nibabel as nib
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder

# Load prediction and ground truth
pred = nib.load('prediction.nii.gz').get_fdata()
gt = nib.load('ground_truth.nii.gz').get_fdata()

# Compute Dice coefficient
def dice_coefficient(pred, gt, label):
    pred_mask = pred == label
    gt_mask = gt == label

    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    return 2 * intersection / union if union > 0 else 0

# Calculate for each label
labels = [1, 2]  # Tumor, edema
for label in labels:
    dice = dice_coefficient(pred, gt, label)
    print(f"Label {label} Dice: {dice:.4f}")
```

## Advanced Features

### Transfer Learning

```bash
# Use pre-trained weights as initialization
# Download pretrained model from model zoo
# Place in nnUNet_results/DatasetXXX_Name/...

# Fine-tune on new dataset
nnUNetv2_train NEW_DATASET_ID 3d_fullres 0 \
    --pretrained_weights /path/to/pretrained/checkpoint_final.pth

# Continue training from checkpoint
nnUNetv2_train 1 3d_fullres 0 \
    --c  # Continue from latest checkpoint
```

### Custom Preprocessing

```python
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

class CustomPreprocessor(DefaultPreprocessor):
    def run_case_npy(self, data, seg, properties, target_spacing):
        # Custom preprocessing here
        # E.g., additional normalization, artifact removal

        data_processed = super().run_case_npy(
            data, seg, properties, target_spacing
        )

        # Additional processing
        return data_processed
```

### Region-Based Training

```bash
# Train on specific regions/labels only
# Modify dataset.json to specify regions

# Create region-based variant
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --regions 1 2

# Train on regions
nnUNetv2_train 1 3d_fullres 0 --regions 1 2
```

### Custom Architecture

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerCustom(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                        unpack_dataset, device)

    def build_network_architecture(self, architecture_class_name,
                                   arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision):
        # Custom architecture modifications
        network = super().build_network_architecture(...)

        # Modify network here
        return network

# Use custom trainer
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerCustom
```

## Pre-trained Models

### Using Model Zoo

```bash
# List available pretrained models
nnUNetv2_download_pretrained_model -h

# Download specific model
nnUNetv2_download_pretrained_model Dataset001_BrainTumor

# Models downloaded to nnUNet_results/

# Use for inference
nnUNetv2_predict -i INPUT -o OUTPUT \
    -d 1 -c 3d_fullres -f all
```

### Share Your Model

```bash
# Package trained model for sharing
nnUNetv2_export_model_to_zip \
    -d 1 -c 3d_fullres -f all \
    -o my_model.zip

# Include:
# - Trained weights
# - Plans file
# - Dataset fingerprint
# - Preprocessing parameters
```

## GPU Optimization

### Memory Management

```bash
# Reduce batch size if GPU memory insufficient
# Edit plans file or use environment variable
export nnUNet_def_batch_size=2

# Use mixed precision training (enabled by default)
# Reduces memory usage and speeds up training
nnUNetv2_train 1 3d_fullres 0 --npz

# Use gradient checkpointing for very large models
export nnUNet_use_gradient_checkpointing=True
```

### Multi-GPU Training

```bash
# Distributed data parallel training
nnUNetv2_train 1 3d_fullres 0 \
    -num_gpus 4 \
    -device cuda

# Specify GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 1 3d_fullres 0 -num_gpus 4
```

## Integration with Other Tools

### From BIDS to nnU-Net

```python
from pathlib import Path
import shutil

# BIDS structure
bids_dir = Path('/data/bids_dataset')
subjects = list(bids_dir.glob('sub-*'))

# nnU-Net target
nnunet_dir = Path(os.environ['nnUNet_raw']) / 'Dataset001_BrainTumor'
images_dir = nnunet_dir / 'imagesTr'
labels_dir = nnunet_dir / 'labelsTr'

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Convert
for i, subj in enumerate(subjects):
    # Copy T1 image
    t1 = subj / 'anat' / f'{subj.name}_T1w.nii.gz'
    shutil.copy(t1, images_dir / f'case_{i:05d}_0000.nii.gz')

    # Copy segmentation if available
    seg = subj / 'derivatives' / 'seg' / f'{subj.name}_aseg.nii.gz'
    if seg.exists():
        shutil.copy(seg, labels_dir / f'case_{i:05d}.nii.gz')
```

### Export to Other Formats

```python
import nibabel as nib

# Load nnU-Net prediction
pred = nib.load('nnunet_prediction.nii.gz')

# Convert to different format
# E.g., for ITK-SNAP, FSL, etc.
nib.save(pred, 'converted_prediction.nii')

# Create binary masks
pred_data = pred.get_fdata()
tumor_mask = (pred_data == 1).astype(int)
nib.save(nib.Nifti1Image(tumor_mask, pred.affine), 'tumor_mask.nii.gz')
```

### Integration with Nilearn

```python
from nilearn import plotting
import nibabel as nib

# Load prediction and anatomical
pred = nib.load('prediction.nii.gz')
anat = nib.load('T1.nii.gz')

# Visualize
plotting.plot_roi(pred, bg_img=anat, cmap='Set1',
                  title='nnU-Net Segmentation')
plotting.show()
```

## Quality Control

### Visual Inspection

```bash
# Use ITK-SNAP for manual review
itksnap -g image.nii.gz -s prediction.nii.gz

# Or FSLeyes
fsleyes image.nii.gz prediction.nii.gz -cm random

# Batch QC script
for pred in predictions/*.nii.gz; do
    case=$(basename $pred .nii.gz)
    img="images/${case}_0000.nii.gz"

    # Create overlay
    fsleyes render --outfile qc/${case}.png \
        --scene ortho $img $pred -cm random
done
```

### Automated QC Metrics

```python
import numpy as np
import nibabel as nib

def calculate_qc_metrics(pred_path, gt_path=None):
    """Calculate QC metrics for prediction."""
    pred = nib.load(pred_path).get_fdata()

    metrics = {}

    # Volume metrics
    for label in np.unique(pred):
        if label == 0:  # Skip background
            continue
        volume = np.sum(pred == label) * np.prod(pred.shape)
        metrics[f'volume_label_{int(label)}'] = volume

    # If ground truth available
    if gt_path:
        gt = nib.load(gt_path).get_fdata()

        for label in np.unique(gt):
            if label == 0:
                continue

            pred_mask = pred == label
            gt_mask = gt == label

            # Dice
            intersection = np.sum(pred_mask & gt_mask)
            dice = 2 * intersection / (np.sum(pred_mask) + np.sum(gt_mask))
            metrics[f'dice_label_{int(label)}'] = dice

    return metrics
```

## Integration with Claude Code

When helping users with nnU-Net:

1. **Check Installation:**
   ```bash
   python -c "import nnunetv2; print(nnunetv2.__version__)"
   echo $nnUNet_raw
   ```

2. **Verify Dataset Format:**
   ```bash
   ls $nnUNet_raw/DatasetXXX_Name/
   # Should see: imagesTr/, labelsTr/, dataset.json
   ```

3. **Common Workflow:**
   - Prepare data → Plan & preprocess → Train → Predict (ensemble)

4. **GPU Recommendations:**
   - Minimum: 8GB VRAM
   - Recommended: 16GB+ VRAM
   - Can use CPU but much slower

5. **Best Practices:**
   - Always use ensemble prediction
   - Train all 5 folds
   - Use 3d_fullres for most tasks
   - Check cross-validation results

## Troubleshooting

**Problem:** CUDA out of memory during training
**Solution:** Reduce batch size, use smaller patch size, or use 3d_lowres configuration

**Problem:** "Dataset not found" error
**Solution:** Check nnUNet_raw environment variable, verify dataset ID matches folder name

**Problem:** Very slow preprocessing
**Solution:** Use multiple threads (-np flag), check I/O speed, use SSD for data

**Problem:** Poor segmentation quality
**Solution:** Check data quality, ensure labels are correct, try ensemble, increase training time

**Problem:** Predictions not aligned with input
**Solution:** Verify spacing/orientation in dataset.json, check preprocessing parameters

## Best Practices

1. **Always use ensemble prediction** (all 5 folds) for best results
2. **Start with 3d_fullres** configuration for most 3D tasks
3. **Use Docker/Singularity** for reproducibility
4. **Keep raw data separate** from nnU-Net directories
5. **Monitor training** via training logs and validation metrics
6. **Validate on held-out test set** not used during training
7. **Save probability maps** for uncertainty quantification
8. **Use pre-trained models** when available for faster convergence
9. **Document dataset.json** carefully for reproducibility
10. **Version control** your dataset.json and custom code

## Resources

- **GitHub:** https://github.com/MIC-DKFZ/nnUNet
- **Documentation:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/
- **Paper:** https://www.nature.com/articles/s41592-020-01008-z
- **Model Zoo:** https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/model_zoo.md
- **Forum:** https://github.com/MIC-DKFZ/nnUNet/discussions
- **Tutorials:** https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation/tutorials

## Citation

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Related Tools

- **MONAI:** Medical imaging deep learning framework
- **FastSurfer:** Deep learning brain parcellation
- **SynthSeg:** Robust multi-contrast segmentation
- **TotalSegmentator:** Pre-trained nnU-Net for full-body CT
- **HD-BET:** Brain extraction using nnU-Net
- **ITK-SNAP:** Manual annotation and QC
- **3D Slicer:** Visualization and manual editing
