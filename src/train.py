import json
import os
import sys

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import create_model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


def train(config_path="configs/training_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        data_dir=config["dataset"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        train_subset=config["dataset"]["train_subset"],
        val_subset=config["dataset"]["val_subset"],
        num_workers=config["training"]["num_workers"],
    )

    model = create_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = config["output"]["checkpoint_dir"]
    metrics_file = config["output"]["metrics_file"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    with open(metrics_file, "w") as metrics_out:
        for epoch in range(1, epochs + 1):
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
            ):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / total
            train_accuracy = correct / total

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

            metrics_out.write(json.dumps(metrics) + "\n")
            metrics_out.flush()

            print(metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    "best_model.pth",
                )

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "num_classes": config["model"]["num_classes"],
                    },
                    checkpoint_path,
                )

                print(f"Saved best model to {checkpoint_path}")

            else:
                patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    print(f"Training completed on {device}")


if __name__ == "__main__":
    train()
