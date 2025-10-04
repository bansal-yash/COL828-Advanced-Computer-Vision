import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
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

test_root = os.path.join(args.data_dir, "Test")

all_classes = sorted(set(os.listdir(test_root)))

class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
num_classes = len(all_classes)

idx_to_class = {v: k for k, v in class_to_idx.items()}
manufacturers = sorted({cls.split(" - ")[0] for cls in class_to_idx})
manufacturer_to_idx = {m: i for i, m in enumerate(manufacturers)}


class PacemakerDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            manufacturer, _ = class_name.split(" - ", 1)

            for fname in os.listdir(class_path):
                if fname.endswith(".JPG"):
                    img_path = os.path.join(class_path, fname)
                    self.samples.append((img_path, class_name, manufacturer))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, full_label, manufacturer = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[full_label]

        return image, label, manufacturer


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


def test(model, test_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    all_manu_labels, all_manu_preds = [], []

    with torch.no_grad():
        for images, labels, manufacturers_true in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            manu_true = [manufacturer_to_idx[m] for m in manufacturers_true]
            manu_pred = [
                manufacturer_to_idx[idx_to_class[p].split(" - ")[0]] for p in preds
            ]
            all_manu_labels.extend(manu_true)
            all_manu_preds.extend(manu_pred)

    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    metrics["manufacturer_accuracy"] = np.mean(
        np.array(all_manu_preds) == np.array(all_manu_labels)
    )
    print(f"Test Acc ={metrics['accuracy']:.4f}")

    return metrics


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


def test_method(method):
    if method == "imagenet21k":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    if method == "clip":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

    if method == "dinov2":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_dataset = PacemakerDataset(test_root, class_to_idx, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    if method == "imagenet21k":
        model = timm.create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=False,
            num_classes=num_classes,
        )

    if method == "clip":
        model = timm.create_model(
            "vit_base_patch16_clip_224.openai",
            pretrained=False,
            num_classes=num_classes,
        )

    if method == "dinov2":
        model = timm.create_model(
            "vit_base_patch16_224.dino", pretrained=False, num_classes=num_classes
        )

    model.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, f"fine_tune_{method}.pth"), map_location=device
        )
    )
    model = model.to(device)

    test_metrics = test(model, test_loader)
    with open("test_metrics_" + method + ".json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(test_metrics)


if __name__ == "__main__":
    print("Testing imagenet21k method")
    test_method("imagenet21k")

    print("Testing clip method")
    test_method("clip")

    print("Testing dinov2 method")
    test_method("dinov2")
