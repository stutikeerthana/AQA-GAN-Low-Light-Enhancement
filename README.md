# AQA-GAN: Adaptive Quality-Aware Generative Adversarial Network for Nighttime Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Research Project](https://img.shields.io/badge/Project-Computer%20Vision-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Overview

Low-light image enhancement (LLIE) is a fundamental problem in computer vision with applications in surveillance, autonomous driving, mobile photography, and intelligent vision systems.

Most existing enhancement methods apply a uniform transformation to every image regardless of degradation severity. This often causes:

- Over-enhancement of mildly dark images
- Under-enhancement of severely degraded images
- Color distortion
- Texture loss
- Noise amplification

To address these limitations, we propose **AQA-GAN (Adaptive Quality-Aware Generative Adversarial Network)**, a quality-conditioned image enhancement framework that dynamically adjusts enhancement strength according to the quality level of the input image.

The framework incorporates a lightweight Quality Assessment Module (QAM), a FiLM-conditioned U-Net Generator, and a Quality-Aware PatchGAN Discriminator to produce realistic and perceptually pleasing nighttime image enhancements.

---

## Key Features

✅ Adaptive enhancement based on image quality

✅ Quality Assessment Module (QAM)

✅ FiLM-based feature conditioning

✅ Quality-Aware PatchGAN discriminator

✅ Mixed precision training

✅ Multi-dataset training strategy

✅ Improved texture preservation

✅ Better color consistency

✅ Higher PSNR and SSIM performance

---

## Motivation

Images captured under poor illumination suffer from:

- Low contrast
- Sensor noise
- Uneven lighting
- Color shifts
- Loss of fine details

Traditional enhancement methods such as Histogram Equalization, CLAHE, Retinex, and Wavelet-based techniques often fail under extreme low-light conditions.

Deep learning methods improve restoration quality but typically use a single enhancement strategy for all inputs.

AQA-GAN introduces quality-aware conditioning so that enhancement intensity adapts to the severity of degradation in each image.

---

# Architecture

The proposed framework consists of four major components:

## 1. Quality Assessment Module (QAM)

The QAM estimates image degradation severity using a lightweight NIQE-inspired quality metric.

Each image is classified into:

| Quality Level | Description |
|--------------|-------------|
| 0 | Severe |
| 1 | Moderate |
| 2 | Mild |

The quality level is then used to condition both the Generator and Discriminator.

---

## 2. FiLM-Conditioned U-Net Generator

The Generator is based on a U-Net architecture enhanced with Feature-wise Linear Modulation (FiLM).

### Components

- Encoder
- Bottleneck
- FiLM Conditioning Layer
- Decoder
- Skip Connections

The FiLM layer allows the model to learn different enhancement behaviors for different degradation levels.

---

## 3. Quality-Aware PatchGAN Discriminator

The Discriminator is a modified 70×70 PatchGAN.

It receives:

- Enhanced image
- Quality level information

and learns degradation-aware realism feedback.

This helps the Generator produce more natural-looking outputs.

---

## 4. Composite Loss Function

Training uses a weighted combination of:

### Adversarial Loss

Encourages realistic outputs.

### Charbonnier Loss

Improves pixel-level reconstruction.

### SSIM Loss

Preserves structural information.

### Perceptual Loss

Uses VGG feature matching to preserve high-level visual quality.

---

# Network Pipeline

Input Low-Light Image

↓

Quality Assessment Module (QAM)

↓

Quality Level Prediction

↓

FiLM-Conditioned U-Net Generator

↓

Enhanced Image

↓

Quality-Aware PatchGAN Discriminator

↓

Composite Loss Optimization

---

## Datasets

The model is trained on a combination of:

### LOL-v2 Real Dataset

Contains paired:

- Low-light images
- Ground truth normal-light images

### LSRW Dataset

Contains real-world low-light image pairs from diverse environments.

### Combined Training Strategy

- LOL-v2 Real (Train Split)
- LSRW Dataset

are merged to create a larger and more diverse training corpus.

---

## Data Preprocessing

The following augmentations are applied:

- Resize to 320×320
- Random Crop (256×256)
- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation
- Color Jitter
- Normalization to [-1,1]

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Image Size | 256×256 |
| Batch Size | 4 |
| Learning Rate | 2e-4 |
| Epochs | 150 |
| Optimizer | Adam |
| Mixed Precision | Enabled |
| Random Seed | 42 |

---

## Experimental Results

### Quantitative Comparison

| Method | Type | PSNR ↑ | SSIM ↑ |
|----------|----------|----------|----------|
| Retinex | Classical | 15.8 | 0.48 |
| Wavelet Transform | Classical | 17.3 | 0.54 |
| Random Forest | Shallow ML | 18.6 | 0.59 |
| SRCNN | CNN | 19.8 | 0.63 |
| U-Net | CNN | 20.4 | 0.69 |
| SRGAN | GAN | 20.8 | 0.71 |
| **AQA-GAN** | **Quality-Aware GAN** | **22.6** | **0.82** |

### Key Observations

- Highest PSNR among all evaluated methods
- Highest SSIM among all evaluated methods
- Better texture preservation
- More natural image enhancement
- Reduced over-enhancement artifacts

---

## Sample Results

### Input Image

(Add image here)

```markdown
![Input](images/input.png)
```

### Enhanced Output

```markdown
![Output](images/output.png)
```

### Ground Truth

```markdown
![GroundTruth](images/groundtruth.png)
```

---

## Repository Structure

```text
AQA-GAN-Low-Light-Enhancement
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── code
│   └── aqa.py
│
├── paper
│   ├── Minor_Project_Report.pdf
│   └── Research_Paper.pdf
│
├── images
│   ├── architecture.png
│   ├── sample_outputs.png
│   └── results.png
│
├── checkpoints
│   └── pretrained_weights.pth
│
└── datasets
    └── dataset_links.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AQA-GAN-Low-Light-Enhancement.git

cd AQA-GAN-Low-Light-Enhancement
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train Model

```bash
python aqa.py
```

or

```bash
python aqa.py --mode train
```

### Resume Training

```bash
python aqa.py --mode train --resume checkpoints/model.pth
```

### Test Model

```bash
python aqa.py --mode test
```

### Generate Evaluation Report

```bash
python aqa.py --mode report
```

### Generate Training Plots

```bash
python aqa.py --mode plots
```

---

## Requirements

```text
torch
torchvision
numpy
matplotlib
tqdm
pillow
```

Install using:

```bash
pip install -r requirements.txt
```

---

## Future Work

- Continuous quality conditioning
- Real-time deployment on edge devices
- Video low-light enhancement
- Multi-task learning
- Joint denoising and enhancement
- Super-resolution integration
- Deblurring support
- Mobile deployment optimization

---

## Research Contributions

The major contributions of this work are:

1. Quality Assessment Module (QAM) requiring no additional training.
2. FiLM-conditioned U-Net Generator for adaptive enhancement.
3. Quality-Aware PatchGAN Discriminator.
4. Multi-dataset training protocol using LOL-v2 and LSRW.
5. Comprehensive comparison with classical, CNN, and GAN-based methods.

---

## Authors

### P. Stuti Keerthana

B.Tech Data Science and Artificial Intelligence  
IIIT Naya Raipur

### Ananya R. Nair

B.Tech Data Science and Artificial Intelligence  
IIIT Naya Raipur

### Anushka Anil

B.Tech Data Science and Artificial Intelligence  
IIIT Naya Raipur

---

## Supervisor

Dr. Aruna Shukla

International Institute of Information Technology (IIIT), Naya Raipur

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{aqagan2026,
  title={AQA-GAN: Adaptive Quality-Aware Generative Adversarial Network for Nighttime Image Enhancement},
  author={Keerthana, P. Stuti and Nair, Ananya R. and Anil, Anushka},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

- LOL-v2 Dataset
- LSRW Dataset
- PyTorch
- IIIT Naya Raipur
- Computer Vision Research Community

---

⭐ If you found this project useful, consider starring the repository.
