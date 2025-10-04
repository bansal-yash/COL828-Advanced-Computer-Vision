import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

Tokenizer = _Tokenizer()

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

train_root = os.path.join(args.data_dir, "Train")

all_classes = sorted(set(os.listdir(train_root)))

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


def train_stage_1(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs
):
    train_history, val_history = [], []

    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, f"coop_stage_1.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_labels, all_preds, all_probs = [], [], []

        for images, _, manufacturers_true in tqdm(
            train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"
        ):
            images = images.to(device)
            manu_labels = torch.tensor(
                [manufacturer_to_idx[m] for m in manufacturers_true], dtype=torch.long
            ).to(device)

            outputs = model(images)
            loss = criterion(outputs, manu_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_labels.extend(manu_labels.cpu().numpy())
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
            for images, _, manufacturers_true in val_loader:
                images = images.to(device)
                manu_labels = torch.tensor(
                    [manufacturer_to_idx[m] for m in manufacturers_true],
                    dtype=torch.long,
                ).to(device)

                outputs = model(images)
                loss = criterion(outputs, manu_labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_labels.extend(manu_labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        avg_val_loss = total_loss / len(val_loader)
        val_metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
        val_metrics["loss"] = avg_val_loss
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_metrics['accuracy']:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Best model saved at epoch {epoch+1} with Val Loss={avg_val_loss:.4f}"
            )

    with open("train_metrics_stage_1.json", "w") as f:
        json.dump(train_history, f, indent=4)
    with open("val_metrics_stage_1.json", "w") as f:
        json.dump(val_history, f, indent=4)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return train_history, val_history


def train_stage_2(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs
):
    train_history, val_history = [], []

    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, f"coop_stage_2.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_labels, all_preds, all_probs = [], [], []
        all_manu_labels, all_manu_preds = [], []

        for images, labels, manufacturers_true in tqdm(
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
            manu_true = [manufacturer_to_idx[m] for m in manufacturers_true]
            manu_pred = [
                manufacturer_to_idx[idx_to_class[p].split(" - ")[0]] for p in preds
            ]
            all_manu_labels.extend(manu_true)
            all_manu_preds.extend(manu_pred)

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
        train_metrics["loss"] = avg_train_loss
        train_metrics["manufacturer_accuracy"] = np.mean(
            np.array(all_manu_preds) == np.array(all_manu_labels)
        )
        train_history.append(train_metrics)

        model.eval()
        total_loss = 0
        all_labels, all_preds, all_probs = [], [], []
        all_manu_labels, all_manu_preds = [], []

        with torch.no_grad():
            for images, labels, manufacturers_true in val_loader:
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
                manu_true = [manufacturer_to_idx[m] for m in manufacturers_true]
                manu_pred = [
                    manufacturer_to_idx[idx_to_class[p].split(" - ")[0]] for p in preds
                ]
                all_manu_labels.extend(manu_true)
                all_manu_preds.extend(manu_pred)

        avg_val_loss = total_loss / len(val_loader)
        val_metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
        val_metrics["loss"] = avg_val_loss
        val_metrics["manufacturer_accuracy"] = np.mean(
            np.array(all_manu_preds) == np.array(all_manu_labels)
        )
        val_history.append(val_metrics)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train Acc={train_metrics['accuracy']:.4f}, "
            f"Train Manu Acc={train_metrics['manufacturer_accuracy']:.4f} | "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
            f"Val Manu Acc={val_metrics['manufacturer_accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Best model saved at epoch {epoch+1} with Val Loss={avg_val_loss:.4f}"
            )

    with open("train_metrics_stage_2.json", "w") as f:
        json.dump(train_history, f, indent=4)
    with open("val_metrics_stage_2.json", "w") as f:
        json.dump(val_history, f, indent=4)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return train_history, val_history


class TextEncoder_Coop(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class PromptLearner_Coop(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 10
        ctx_init = None
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            class_specific_prompts = True
            if class_specific_prompts:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(Tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CLIP_Coop(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_Coop(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_Coop(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


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


def train_coop(epochs_stage_1, epochs_stage_2):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    full_dataset = PacemakerDataset(train_root, class_to_idx, transform=None)

    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=[s[1] for s in full_dataset.samples],
        random_state=42,
    )

    train_dataset = PacemakerDataset(
        train_root, class_to_idx, transform=train_transform
    )
    val_dataset = PacemakerDataset(train_root, class_to_idx, transform=val_transform)

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    clip_model_stage_1, _ = clip.load("ViT-B/16", device="cpu")
    model_stage_1 = CLIP_Coop(manufacturers, clip_model_stage_1)

    clip_model_stage_1 = clip_model_stage_1.to(device)
    model_stage_1 = model_stage_1.to(device)

    for name, param in model_stage_1.named_parameters():
        param.requires_grad = False

    for name, param in model_stage_1.named_parameters():
        if "prompt_learner.ctx" in name:
            param.requires_grad = True
            print("Trainable:", name, param.shape)

    trainable = [n for n, p in model_stage_1.named_parameters() if p.requires_grad]
    print("\nFinal trainable params:", trainable)

    criterion_stage_1 = nn.CrossEntropyLoss()
    optimizer_stage_1 = AdamW(model_stage_1.parameters(), lr=1e-3, weight_decay=0.05)

    T_max = epochs_stage_1
    eta_min = 1e-5
    scheduler_stage_1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage_1, T_max=T_max, eta_min=eta_min
    )

    train_history, val_history = train_stage_1(
        model_stage_1,
        train_loader,
        val_loader,
        criterion_stage_1,
        optimizer_stage_1,
        scheduler_stage_1,
        epochs_stage_1,
    )

    plot_metrics(train_history, val_history, metric="loss")
    plot_metrics(train_history, val_history, metric="accuracy")
    plot_metrics(train_history, val_history, metric="top3_accuracy")
    plot_metrics(train_history, val_history, metric="f1")
    plot_metrics(train_history, val_history, metric="precision")
    plot_metrics(train_history, val_history, metric="recall")
    plot_metrics(train_history, val_history, metric="auc_roc")

    model_stage_1.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, f"coop_stage_1.pth"), map_location=device
        )
    )
    manufacturer_prompts = model_stage_1.prompt_learner.ctx.detach().clone()

    clip_model_stage_2, _ = clip.load("ViT-B/16", device="cpu")
    model_stage_2 = CLIP_Coop(all_classes, clip_model_stage_2)

    clip_model_stage_2 = clip_model_stage_2.to(device)
    model_stage_2 = model_stage_2.to(device)

    with torch.no_grad():
        for cls_idx, cls_name in idx_to_class.items():
            manu = cls_name.split(" - ")[0]
            manu_idx = manufacturer_to_idx[manu]

            model_stage_2.prompt_learner.ctx[cls_idx] = manufacturer_prompts[manu_idx]

    for name, param in model_stage_2.named_parameters():
        param.requires_grad = False

    for name, param in model_stage_2.named_parameters():
        if "prompt_learner.ctx" in name:
            param.requires_grad = True
            print("Trainable:", name, param.shape)

    trainable = [n for n, p in model_stage_2.named_parameters() if p.requires_grad]
    print("\nFinal trainable params:", trainable)

    criterion_stage_2 = nn.CrossEntropyLoss()
    optimizer_stage_2 = AdamW(model_stage_2.parameters(), lr=1e-3, weight_decay=0.05)

    T_max = epochs_stage_2
    eta_min = 1e-6
    scheduler_stage_2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage_2, T_max=T_max, eta_min=eta_min
    )

    train_history, val_history = train_stage_2(
        model_stage_2,
        train_loader,
        val_loader,
        criterion_stage_2,
        optimizer_stage_2,
        scheduler_stage_2,
        epochs_stage_2,
    )

    plot_metrics(train_history, val_history, metric="loss")
    plot_metrics(train_history, val_history, metric="accuracy")
    plot_metrics(train_history, val_history, metric="top3_accuracy")
    plot_metrics(train_history, val_history, metric="f1")
    plot_metrics(train_history, val_history, metric="precision")
    plot_metrics(train_history, val_history, metric="recall")
    plot_metrics(train_history, val_history, metric="auc_roc")
    plot_metrics(train_history, val_history, metric="manufacturer_accuracy")


if __name__ == "__main__":
    print("Training coop")
    train_coop(epochs_stage_1=50, epochs_stage_2=100)
