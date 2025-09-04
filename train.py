import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import segmentation_models_pytorch as smp

import config
from src.dataset import get_loaders
from src.utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    setup_logging,
)

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    """
    Handles the training loop for one epoch.
    """
    loop = tqdm(loader, leave=True)
    total_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        # Forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)


def main():
    """
    Main training script to train all specified U-Net hybrid models.
    """
    for model_config in config.MODELS_TO_TRAIN:
        model_name = model_config["name"]
        encoder = model_config["encoder"]
        weights = model_config["weights"]
        
        print(f"\n===== Training Model: {model_name} =====")
        
        # Setup logging and output directories
        log_dir = os.path.join(config.OUTPUT_DIR, model_name)
        pred_dir = os.path.join(log_dir, "predictions")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        logger = setup_logging(log_dir)

        # Initialize model
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=1, # Binary segmentation
        ).to(config.DEVICE)
        
        # Loss function and optimizer
        # BCEWithLogitsLoss is more numerically stable than a plain Sigmoid followed by a BCELoss.
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Get data loaders
        train_loader, val_loader = get_loaders()
        
        if config.LOAD_MODEL:
            try:
                load_checkpoint(torch.load(os.path.join(log_dir, "best_checkpoint.pth.tar")), model)
            except FileNotFoundError:
                logger.warning("Checkpoint not found. Starting training from scratch.")

        scaler = torch.cuda.amp.GradScaler()
        best_dice_score = 0.0

        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
            
            # Check accuracy on validation set
            val_accuracy, val_dice = check_accuracy(val_loader, model, device=config.DEVICE)
            
            logger.info(
                f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2f}% | "
                f"Val Dice Score: {val_dice:.4f}"
            )

            # Save model checkpoint if it has the best Dice score so far
            if val_dice > best_dice_score:
                best_dice_score = val_dice
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_dice_score": best_dice_score,
                }
                save_checkpoint(checkpoint, filename=os.path.join(log_dir, "best_checkpoint.pth.tar"))

            # Save some prediction examples for visual inspection
            save_predictions_as_imgs(val_loader, model, folder=pred_dir, device=config.DEVICE)
            
        logger.info(f"Finished training for {model_name}. Best Val Dice Score: {best_dice_score:.4f}")


if __name__ == "__main__":
    main()

