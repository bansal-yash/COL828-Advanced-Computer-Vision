import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import open_clip
import torch.nn.functional as F


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
args = parser.parse_args()

train_root = os.path.join(args.data_dir, "Train")
test_root = os.path.join(args.data_dir, "Test")

all_classes = sorted(set(os.listdir(train_root)) | set(os.listdir(test_root)))

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


def inference(model, tokenizer, train_loader, test_loader):
    class_prompts = [f"Cardiac implant X-ray by {c} implant" for c in all_classes]
    texts = tokenizer(class_prompts).to(device)
    text_features = model.encode_text(texts)
    text_features = F.normalize(text_features, dim=-1)

    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    all_manu_labels, all_manu_preds = [], []

    with torch.no_grad():
        for images, labels, manufacturers_true in tqdm(
            train_loader, desc="Train Inference"
        ):
            images, labels = images.to(device), labels.to(device)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            logits_per_image = image_features @ text_features.t()

            probs = logits_per_image.softmax(dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            manu_true = [manufacturer_to_idx[m] for m in manufacturers_true]
            manu_pred = [
                manufacturer_to_idx[idx_to_class[p].split(" - ")[0]] for p in preds
            ]
            all_manu_labels.extend(manu_true)
            all_manu_preds.extend(manu_pred)

    train_metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    train_metrics["manufacturer_accuracy"] = np.mean(
        np.array(all_manu_preds) == np.array(all_manu_labels)
    )
    print("Train Metrics:", train_metrics)

    all_labels, all_preds, all_probs = [], [], []
    all_manu_labels, all_manu_preds = [], []

    with torch.no_grad():
        for images, labels, manufacturers_true in tqdm(
            test_loader, desc="Test Inference"
        ):
            images, labels = images.to(device), labels.to(device)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            logits_per_image = image_features @ text_features.t()

            probs = logits_per_image.softmax(dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            manu_true = [manufacturer_to_idx[m] for m in manufacturers_true]
            manu_pred = [
                manufacturer_to_idx[idx_to_class[p].split(" - ")[0]] for p in preds
            ]
            all_manu_labels.extend(manu_true)
            all_manu_preds.extend(manu_pred)

    test_metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    test_metrics["manufacturer_accuracy"] = np.mean(
        np.array(all_manu_preds) == np.array(all_manu_labels)
    )
    print("Test Metrics:", test_metrics)

    return train_metrics, test_metrics


if __name__ == "__main__":
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = PacemakerDataset(
        train_root, class_to_idx, transform=train_transform
    )
    test_dataset = PacemakerDataset(test_root, class_to_idx, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", force_quick_gelu=True
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    model = model.to(device)

    train_metrics, test_metrics = inference(model, tokenizer, train_loader, test_loader)

    print(train_metrics)
    print(test_metrics)
