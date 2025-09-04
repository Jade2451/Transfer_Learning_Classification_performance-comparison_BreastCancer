import torch
import torchvision
import os
import logging

import config

def setup_logging(log_dir):
    """Sets up logging to file and console."""
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves the model checkpoint."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """Loads a model checkpoint."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    """
    Calculates Dice score and pixel-wise accuracy for the model on the given data loader.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    accuracy = (num_correct/num_pixels)*100
    dice = dice_score/len(loader)
    
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
    print(f"Dice score: {dice:.4f}")
    
    model.train()
    return accuracy, dice

def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    """
    Saves a batch of images, their ground truth masks, and the model's predictions
    to a specified folder for visual inspection.
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        # Save a limited number of examples
        if idx < config.NUM_EXAMPLES_TO_SAVE:
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{idx}.png")
        else:
            break
            
    model.train()

