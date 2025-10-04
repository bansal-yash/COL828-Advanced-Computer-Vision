import json
import os
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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


def test_stage_1(model, test_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, _, manufacturers_true in test_loader:
            images = images.to(device)
            manu_labels = torch.tensor(
                [manufacturer_to_idx[m] for m in manufacturers_true],
                dtype=torch.long,
            ).to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_labels.extend(manu_labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    print(f"Test Acc ={metrics['accuracy']:.4f}")

    return metrics


def test_stage_2(model, test_loader):
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


def test_maple():
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    test_dataset = PacemakerDataset(test_root, class_to_idx, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

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

    model = CLIP_Maple(manufacturers, clip_model)

    model.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, f"maple_stage_1.pth"), map_location=device
        )
    )

    clip_model = clip_model.to(device)
    model = model.to(device)

    test_metrics = test_stage_1(model, test_loader)
    with open("test_metrics_stage_1.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(test_metrics)

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

    model = CLIP_Maple(all_classes, clip_model)

    model.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, f"maple_stage_2.pth"), map_location=device
        )
    )

    clip_model = clip_model.to(device)
    model = model.to(device)

    test_metrics = test_stage_2(model, test_loader)
    with open("test_metrics_stage_2.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    print(test_metrics)


if __name__ == "__main__":
    print("Testing maple")
    test_maple()
