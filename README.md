# Transfer_Learning_Classification

Target: U-Net with DenseNet & MobileNetV2 Encoders for Breast Cancer Segmentation
This project investigates and compares the performance of hybrid U-Net architectures for semantic segmentation of breast cancer in DCE-MRI images from the QIN-Breast dataset. Specifically, it uses pretrained DenseNet121 and MobileNetV2 models as feature extraction backbones (encoders) within a standard U-Net decoder structure.

The goal is to leverage the power of transfer learning to achieve robust and accurate tumor segmentation, which is a critical step in cancer diagnosis and treatment planning.

# Table of Contents
1. Objective
2. Models Compared
3. Dataset
4. Repository Structure
5. Setup and Installation
6. Usage
    * Step 1: Prepare the Data
    * Step 2: Run Training
    * Step 3: Evaluate a Model
7. Results

---

## Objective
The primary objective is to build and evaluate two U-Net-based segmentation models:
  * DenseNet121-UNet: Utilizes a pretrained DenseNet121 as the encoder.
  * MobileNetV2-UNet: Utilizes a pretrained MobileNetV2 as the encoder.
The performance of these models will be compared based on standard image segmentation metrics, such as the Dice Coefficient and Intersection over Union (IoU), to determine which backbone provides more effective feature extraction for this specific medical imaging task.

### Models Compared
  U-Net: A convolutional neural network architecture designed for fast and precise biomedical image segmentation. It consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) for precise localization.
  
  DenseNet121: A deep convolutional network where each layer is connected to every other layer in a feed-forward fashion. This dense connectivity encourages feature reuse and strengthens feature propagation, making it highly parameter-efficient.
  
  MobileNetV2: A lightweight deep neural network architecture that is optimized for performance on mobile and resource-constrained devices. It uses depthwise separable convolutions to reduce computational cost.

## Dataset
This project uses the QIN Breast DCE-MRI dataset. This dataset contains dynamic contrast-enhanced magnetic resonance imaging scans of the breast, along with corresponding tumor segmentations.

Important: You must download the dataset yourself and organize it into the structure expected by the scripts. The data directory should be structured as follows:

```
data/
├── train/
│   ├── images/
│   │   ├── patient001_slice01.png
│   │   └── ...
│   └── masks/
│       ├── patient001_slice01_mask.png
│       └── ...
└── test/
    ├── images/
    │   ├── patient050_slice01.png
    │   └── ...
    └── masks/
        ├── patient050_slice01_mask.png
        └── ...
```

## Repository Structure
```
unet-mri-segmentation/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py
├── data/
│   └── (QIN DCE-MRI data)
├── src/
│   ├── dataset.py
│   └── models.py
├── train.py
└── evaluate.py
```

## Setup and Installation

Clone the repository:
```
git clone <your-repository-url>
cd unet-mri-segmentation
```
Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
Install the required packages:
```
pip install -r requirements.txt
```
## Usage
* Step 1: Prepare the Data
Ensure your QIN DCE-MRI dataset is organized in the data/ directory as described above. The scripts expect separate train and test folders, each with images and masks subdirectories.

* Step 2: Run Training
The `train.py` script will train all models specified in the `config.py` file.

To start training, simply run:
```
python train.py
```
The script will create an output/ directory.

Inside `output/`, a separate folder will be created for each model (e.g., densenet121-unet).

Training progress will be logged to the console and a `training.log` file inside the model's directory.

The best model checkpoint (based on validation Dice score) will be saved as `best_model.pth`.

* Step 3: Evaluate a Model
The `evaluate.py` script is used to test a trained model on the test dataset and visualize its predictions.

To evaluate a model, you need to provide the path to its checkpoint file and specify the model architecture.
```
python evaluate.py --checkpoint output/densenet121-unet/best_model.pth --model densenet121-unet
```
The script will calculate and print the average Dice score and IoU on the test set.

It will also save a set of prediction masks in the model's output directory (e.g., in a new predictions/ folder) for visual inspection.

## Results
After running the training and evaluation scripts, the performance of each model will be logged. The primary metric for comparison is the Dice Coefficient on the test set. The `evaluate.py` script provides the final performance measure for each model, allowing for a direct comparison between the DenseNet121 and MobileNetV2 backbones.
