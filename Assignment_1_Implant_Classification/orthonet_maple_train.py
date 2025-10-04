import json
import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from clip_maple import clip
from clip_maple.simple_tokenizer import SimpleTokenizer as _Tokenizer

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


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    train_history, val_history = [], []

    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, f"maple.pth")

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

    with open("train_metrics.json", "w") as f:
        json.dump(train_history, f, indent=4)
    with open("val_metrics.json", "w") as f:
        json.dump(val_history, f, indent=4)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return train_history, val_history


class TextEncoder_Maple(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)

        combined = [
            x,
            compound_prompts_deeper_text,
            0,
        ]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PromptLearner_Maple(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 10
        ctx_init = None
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.compound_prompts_depth = 12

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print("MaPLe design: Multi-modal Prompt Learning")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_ctx, 512))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1
        )

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

    def construct_prompts(self, ctx, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        return (
            prompts,
            self.proj(self.ctx),
            self.compound_prompts_text,
            visual_deep_prompts,
        )


class CLIP_Maple(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner_Maple(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_Maple(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = self.prompt_learner()

        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        )

        aggregated_ctx = shared_ctx.mean(dim=0)

        image_features = self.image_encoder(
            image.type(self.dtype), aggregated_ctx, deep_compound_prompts_vision
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
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


def train_maple(epochs):
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

    train_dataset = OrthoNetDataset(
        train_df, img_dir, class_to_idx, transform=train_transform
    )
    val_dataset = OrthoNetDataset(
        val_df, img_dir, class_to_idx, transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": 10,
    }

    clip_model = clip.build_model(state_dict or model.state_dict(), design_details)

    model = CLIP_Maple(classes, clip_model)

    clip_model = clip_model.to(device)
    model = model.to(device)

    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"

    for name, param in model.named_parameters():
        if name_to_update not in name:
            if "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print("\nFinal trainable params:", trainable)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

    T_max = epochs
    eta_min = 1e-4
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )

    train_history, val_history = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs
    )

    plot_metrics(train_history, val_history, metric="loss")
    plot_metrics(train_history, val_history, metric="accuracy")
    plot_metrics(train_history, val_history, metric="top3_accuracy")
    plot_metrics(train_history, val_history, metric="f1")
    plot_metrics(train_history, val_history, metric="precision")
    plot_metrics(train_history, val_history, metric="recall")
    plot_metrics(train_history, val_history, metric="auc_roc")


if __name__ == "__main__":
    print("Training maple")
    train_maple(epochs=150)
