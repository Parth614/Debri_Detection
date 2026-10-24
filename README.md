# AstroSeg AI: Space Object Detection Using Deep CNN

A lightweight hybrid deep learning framework for **spacecraft image segmentation**, designed to identify spacecraft regions and generate pixel-level segmentation masks while maintaining computational efficiency.

## Overview

AstroSeg AI explores the use of lightweight convolutional neural networks, multi-scale feature fusion, and pixel-level contrastive learning for spacecraft image segmentation. The framework processes spacecraft images and predicts binary masks that distinguish spacecraft pixels from the background.

The project uses a selected subset of the **SPARK-2021 dataset** and combines efficient feature extraction with contextual feature learning to support accurate and computationally efficient segmentation.

## Key Features

- **Lightweight Feature Extraction:** Uses RepViT-M0.9 to extract spatial and semantic features.
- **Multi-Scale Feature Fusion:** Integrates LightGELAN to combine features at different resolutions.
- **Contextual Learning:** Uses Lite-ASPP to capture spatial context at multiple receptive fields.
- **Pixel-Level Contrastive Learning:** Learns more discriminative feature representations.
- **Efficient Segmentation:** Uses a depthwise decoder to generate full-resolution segmentation masks.

## Model Architecture

The proposed pipeline consists of the following components:

```text
Input Spacecraft Image
          |
          v
 Image Preprocessing
          |
          v
   RepViT-M0.9
  Feature Extraction
          |
          v
 Feature Projection
          |
          v
   LightGELAN
  Feature Fusion
          |
          v
     Lite-ASPP
 Contextual Learning
          |
          v
   Depthwise Decoder
          |
          v
  Segmentation Head
          |
          v
 Predicted Binary Mask

Parallel Embedding Head
          |
          v
Pixel-Level Contrastive Learning
```

The training objective combines Dice Loss, Focal Loss, and pixel-level contrastive loss to improve segmentation quality and feature discrimination.

## Dataset

The project uses a selected subset of the **SPARK-2021 spacecraft dataset**.

- **Total samples:** 80,000 image-mask pairs
- **Training set:** 64,000 samples
- **Validation set:** 8,000 samples
- **Test set:** 8,000 samples
- **Number of spacecraft classes:** 10
- **Input resolution:** 320 × 320 pixels
- **Segmentation task:** Binary segmentation of spacecraft against the background

The dataset includes spacecraft categories such as CHEOPS, Lisa Pathfinder, Proba-2, SMART-1, SOHO, Venus Express, and XMM-Newton.

*Note: Ensure that dataset access and redistribution comply with the dataset's licensing and usage terms.*

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Architecture:** RepViT-M0.9, LightGELAN, Lite-ASPP
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy
- **Evaluation:** Dice Score, Intersection over Union (IoU)

## Training Configuration

| Parameter | Configuration |
|---|---|
| Input image size | 320 × 320 |
| Batch size | 16 |
| Maximum epochs | 50 |
| Optimizer | AdamW |
| Loss function | Dice Loss + Focal Loss + Pixel-Level Contrastive Loss |
| Early stopping patience | 7 epochs |
| Gradient clipping | 1.0 |
| Learning-rate scheduling | ReduceLROnPlateau |

Training includes image normalization, mask preprocessing, validation-based monitoring, and early stopping to help control overfitting.

## Preliminary Results

The following results were recorded during the initial training run.

| Metric | Validation Result |
|---|---:|
| Best recorded validation Dice Score | 95.42% |
| Best recorded validation IoU | 91.45% |
| Validation loss | 0.0305 |
| Model parameters | Approximately 4.79 million |
| Estimated model size | Approximately 18.25 MB |

The reported validation metrics were obtained at the sixth completed epoch. **These are preliminary validation results, not final test-set results.** The test-set evaluation and comparative benchmarking remain necessary to establish generalization performance.

The estimated inference latency is approximately 1.81 ms per image under the recorded timing procedure. Actual latency depends on the hardware, batch size, and inference configuration.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AstroSeg-AI
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install torch torchvision numpy opencv-python matplotlib scikit-learn
```

*Install any additional libraries required by the training notebook or implementation.*

## Usage

1. Download or prepare the SPARK-2021 dataset and its corresponding segmentation masks.
2. Configure the dataset paths in the training notebook or script.
3. Run the preprocessing pipeline to resize and normalize images and masks.
4. Train the model using the configured loss functions and optimizer.
5. Evaluate the trained model using Dice Score and IoU.
6. Visualize the predicted masks against the ground-truth masks.

The exact execution commands depend on the scripts and notebooks included in the repository.

## Project Structure

A suggested repository organization is shown below. Adjust the filenames to match the actual files in your repository.

```text
AstroSeg-AI/
├── data/
│   └── README.md
├── notebooks/
│   └── spacecraft_segmentation.ipynb
├── models/
│   └── README.md
├── outputs/
│   └── README.md
├── requirements.txt
├── README.md
└── LICENSE
```

Dataset files and large model checkpoints should generally be excluded from Git if they are too large or subject to redistribution restrictions.

## Future Improvements

- Complete the held-out test-set evaluation.
- Benchmark against U-Net, DeepLabV3+, and other lightweight segmentation models.
- Improve spacecraft boundary segmentation and small-structure recognition.
- Optimize inference speed and memory consumption.
- Evaluate robustness across different backgrounds, orientations, and image conditions.
- Explore deployment on resource-constrained platforms.



## Acknowledgements

The project builds on prior research in spacecraft image datasets, lightweight neural networks, semantic segmentation, and contrastive representation learning. Appreciation is extended to the researchers and dataset contributors whose work supports further investigation in spacecraft computer vision.

## Disclaimer

AstroSeg AI is an academic research project intended for experimentation with spacecraft image segmentation. Its outputs should not be treated as validated operational guidance for real-world space missions.

