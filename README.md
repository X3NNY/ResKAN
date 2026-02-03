# ResKAN: Residual Kolmogorov-Arnold Network for Seismic Impedance Inversion

## Overview

This repository implements **ResKAN** (Residual Kolmogorov-Arnold Network), a novel deep learning approach for seismic impedance inversion that combines residual blocks and Kolmogorov-Arnold Networks (KANs). ResKAN addresses key challenges in seismic inversion, including high learning cost, insufficient lateral continuity, and large training data requirements.


## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy
- Matplotlib
- scikit-image
- opencv-python
- segyio
- tqdm

See `requirements.txt` for the complete list of dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/x3nny/reskan.git
cd reskan/repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the ResKAN model:
```bash
python scripts/train.py --model ResKAN --dataset 0 --phase 0 --freq 30 --epochs 1000
```

Train baseline models for comparison:
```bash
python scripts/train.py --model CNN --dataset 0 --phase 0 --freq 30 --epochs 1000
python scripts/train.py --model TCN --dataset 0 --phase 0 --freq 30 --epochs 1000
python scripts/train.py --model RNN --dataset 0 --phase 0 --freq 30 --epochs 1000
python scripts/train.py --model RKAN-18 --dataset 0 --phase 0 --freq 30 --epochs 1000
```

**Training Arguments**:
- `--model`: Model type (`CNN`, `TCN`, `RNN`, `RKAN-18`, `ResKAN`)
- `--dataset`: Dataset number (`0`=Marmousi2, `1`=Overthrust, `2`=Overthrust interpolated traces)
- `--phase`: Phase shift (default: `0`)
- `--freq`: Dominant frequency in Hz (default: `30`)
- `--epochs`: Number of training epochs (default: `1000`)
- `--batch-size`: Batch size (default: `64`)
- `--lr`: Learning rate (default: `1e-3`)
- `--weight-decay`: Weight decay (default: `1e-4`)
- `--loss-weight-mse`: MSE loss weight (default: `1.0`)
- `--loss-weight-ce`: Contour loss weight (default: `0.0`)
- `--n-wells`: Number of well logs for training (default: `128`)
- `--device`: Computing device (`cpu` or `cuda`, auto-detected by default)
- `--data-root`: Path to data directory (optional)
- `--out-dir`: Checkpoint output directory (default: `repo/checkpoints`)

### Testing

Evaluate a trained model:
```bash
python scripts/test.py --model ResKAN --dataset 0 --phase 0 --freq 30
```

**Testing Arguments**:
- `--model`: Model type
- `--dataset`: Dataset number
- `--phase`: Phase shift
- `--freq`: Dominant frequency
- `--snr-db`: Signal-to-noise ratio in dB (default: `0`)
- `--checkpoint`: Explicit checkpoint path (optional, overrides default lookup)
- `--ckpt-dir`: Checkpoint directory (default: `repo/checkpoints`)
- `--save-plots`: Save prediction plots (default: `True`)
- `--seed`: Random seed (default: `2024`)
- `--device`: Computing device

### Forward Modeling

Generate synthetic seismic data from impedance models:
```bash
python scripts/forward_model.py --model marmousi2 --phase 0 --freq 30
```

**Forward Modeling Arguments**:
- `--model`: Model name (`marmousi2` or `overthrust`)
- `--phase`: Phase shift
- `--freq`: Dominant frequency
- `--dt`: Time sampling interval in seconds (default: `0.002`)
- `--data-root`: Path to data directory (optional)
- `--output`: Output file path (optional)

## Model Architecture

### ResKAN

The ResKAN architecture consists of two parallel branches:

1. **Branch 1**: Residual blocks with hybrid dilation rates `[1, 2, 5, ...]` (cyclic pattern)
2. **Branch 2**: Residual blocks with exponential dilation rates `[1, 2, 4, 8, ...]`

Both branches extract multi-scale features using dilated convolutions, and their outputs are fused through KAN layers to produce the final impedance prediction. The KAN-based feature fusion replaces traditional fully connected layers, providing stronger feature fusion capabilities while maintaining parameter efficiency.

### Baseline Models

- **CNN**: 1D convolutional neural network for seismic impedance inversion
- **TCN**: Temporal convolutional network with dilated convolutions
- **RNN**: Recurrent neural network (originally named EIIN)
- **RKAN-18**: 18-layer residual KAN network

## Evaluation Metrics

The test script reports the following metrics:

- **MSE** (Mean Squared Error): Measures prediction accuracy
- **SSIM** (Structural Similarity Index): Evaluates structural similarity between predicted and true impedance
- **r²** (Coefficient of Determination): Quantifies the proportion of variance explained
- **PCC** (Pearson Correlation Coefficient): Measures linear correlation

## Results Visualization

After testing, prediction results and error maps are saved in `results/img/ai/`:

- `{ModelName}_pred.png`: Predicted impedance section
- `{ModelName}_diff.png`: Absolute difference between prediction and ground truth

## Datasets

### Marmousi2
- **Location**: `data/marmousi2/`
- **Contents**: Density model, P-wave velocity model, synthetic seismic data
- **Format**: SEGY files for velocity/density, NumPy arrays for seismic data

### Overthrust
- **Location**: `data/overthrust/`
- **Contents**: Impedance model, synthetic seismic data
- **Format**: MATLAB files and NumPy arrays

## Checkpoints

Trained model checkpoints are saved in `checkpoints/` with the naming convention:
```
{ModelName}_{phase}_{freq}.pkl
```

For example:
- `ResKANet5_[1x8x8x8x16x16x16x32x32x32]_[3x2x1]_0_30.pkl`
- `CNN_0_30.pkl`
- `TCN_0_30.pkl`
