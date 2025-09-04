import torch
import os
import segmentation_models_pytorch as smp
from tqdm import tqdm

import config
from src.dataset import get_loaders
from src.utils import (
    load_checkpoint,
    save_predictions_as_imgs,
)

def main():
    """
    Main evaluation script to test the trained models.
    """
    for model_config in config.MODELS_TO_TRAIN:
        model_name = model_config["name"]
        encoder = model_config["encoder"]
        
        print(f"\n===== Evaluating Model: {model_name} =====")
        
        # Setup output directory for evaluation results
        eval_dir = os.path.join(config.OUTPUT_DIR, model_name, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        # Initialize model
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None, # Weights will be loaded from checkpoint
            in_channels=3,
            classes=1,
        ).to(config.DEVICE)
        
        # Get data loaders (only validation/test loader is needed here)
        _, test_loader = get_loaders()
        
        # Load the best checkpoint from the training run
        checkpoint_path = os.path.join(config.OUTPUT_DIR, model_name, "best_checkpoint.pth.tar")
        try:
            load_checkpoint(torch.load(checkpoint_path), model)
        except FileNotFoundError:
            print(f"Error: Checkpoint not found at '{checkpoint_path}'.")
            print("Please run train.py first to generate a model checkpoint.")
            continue

        # Save prediction examples
        print(f"Saving prediction examples to: {eval_dir}")
        save_predictions_as_imgs(
            test_loader, model, folder=eval_dir, device=config.DEVICE
        )
        print("Evaluation complete.")

if __name__ == "__main__":
    main()

