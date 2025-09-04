# config.py

import torch

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 25
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

# --- Dataset and Paths ---
# Adjust these paths based on your local directory structure
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_MASK_DIR = "data/train/masks/"
VAL_IMG_DIR = "data/test/images/"  # Using the test set as our validation set
VAL_MASK_DIR = "data/test/masks/"
OUTPUT_DIR = "output/"

# --- Image Transformations ---
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# --- Model Configuration ---
# List of models to train. The names must match the keys in segmentation-models-pytorch.
# See library documentation for more options.
MODELS_TO_TRAIN = [
    {
        "name": "densenet121-unet",
        "encoder": "densenet121",
        "weights": "imagenet",
    },
    {
        "name": "mobilenet_v2-unet",
        "encoder": "mobilenet_v2",
        "weights": "imagenet",
    },
]

# --- Evaluation Configuration ---
# Number of example images with their masks and predictions to save during evaluation.
NUM_EXAMPLES_TO_SAVE = 5

