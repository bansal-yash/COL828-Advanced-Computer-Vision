import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to dataset root",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./checkpoints",
    help="Folder to save models and logs",
)
args = parser.parse_args()

csv_path = os.path.join(args.data_dir, "train.csv")
img_dir = os.path.join(args.data_dir, "orthonet data", "orthonet data")

train_df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df["labels"], random_state=42
)

classes = sorted(train_df["labels"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)


class OrthoNetDataset(Dataset):
    def __init__(self, dataframe, img_dir, class_to_idx, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.images = []
        self.labels = []

        for _, row in self.dataframe.iterrows():
            img_path = os.path.join(self.img_dir, row["filenames"])
            img = Image.open(img_path).convert("RGB")
            self.images.append(img)
            self.labels.append(self.class_to_idx[row["labels"]])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}
    metrics["accuracy"] = (y_true == y_pred).mean()

    top3 = np.argsort(-y_prob, axis=1)[:, :3]
    top3_acc = np.mean([y_true[i] in top3[i] for i in range(len(y_true))])
    metrics["top3_accuracy"] = top3_acc

    metrics["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except:
        metrics["auc_roc"] = None

    return metrics


def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, method, epochs
):
    train_history, val_history = [], []

    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, f"fine_tune_{method}.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_labels, all_preds, all_probs = [], [], []

        for images, labels in tqdm(
            train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
        train_metrics["loss"] = avg_train_loss
        train_history.append(train_metrics)

        model.eval()
        total_loss = 0
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        avg_val_loss = total_loss / len(val_loader)
        val_metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
        val_metrics["loss"] = avg_val_loss
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_metrics['accuracy']:.4f} | "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Best model saved at epoch {epoch+1} with Val Loss={avg_val_loss:.4f}"
            )

    with open("train_metrics_" + method + ".json", "w") as f:
        json.dump(train_history, f, indent=4)
    with open("val_metrics_" + method + ".json", "w") as f:
        json.dump(val_history, f, indent=4)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return train_history, val_history


def plot_metrics(train_history, test_history, metric="accuracy"):
    train_vals = [m[metric] for m in train_history]
    test_vals = [m[metric] for m in test_history]

    plt.figure(figsize=(7, 5))
    plt.plot(train_vals, label=f"Train {metric}")
    plt.plot(test_vals, label=f"Validation {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} over epochs")
    plt.legend()
    plt.show()


def train_method(method, epochs):
    if method == "imagenet21k":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    if method == "clip":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

    if method == "dinov2":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = OrthoNetDataset(
        train_df, img_dir, class_to_idx, transform=train_transform
    )
    val_dataset = OrthoNetDataset(
        val_df, img_dir, class_to_idx, transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    if method == "imagenet21k":
        model = timm.create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=True,
            num_classes=num_classes,
        )

    if method == "clip":
        model = timm.create_model(
            "vit_base_patch16_clip_224.openai", pretrained=True, num_classes=num_classes
        )

    if method == "dinov2":
        model = timm.create_model(
            "vit_base_patch16_224.dino", pretrained=True, num_classes=num_classes
        )

    # for param in model.parameters():
    #     param.requires_grad = False

    # if hasattr(model, "head") and isinstance(model.head, nn.Module):
    #     for param in model.head.parameters():
    #         param.requires_grad = True
    # elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
    #     for param in model.fc.parameters():
    #         param.requires_grad = True

    # trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    # print("Trainable parameters:", trainable_params)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    T_max = epochs
    eta_min = 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )

    train_history, val_history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        method,
        epochs,
    )

    plot_metrics(train_history, val_history, metric="loss")
    plot_metrics(train_history, val_history, metric="accuracy")
    plot_metrics(train_history, val_history, metric="top3_accuracy")
    plot_metrics(train_history, val_history, metric="f1")
    plot_metrics(train_history, val_history, metric="precision")
    plot_metrics(train_history, val_history, metric="recall")
    plot_metrics(train_history, val_history, metric="auc_roc")


if __name__ == "__main__":
    print("Training imagenet21k method")
    train_method("imagenet21k", epochs=100)

    print("Training clip method")
    train_method("clip", epochs=100)

    print("Training dinov2 method")
    train_method("dinov2", epochs=100)
