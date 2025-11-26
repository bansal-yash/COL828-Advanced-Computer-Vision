import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Mammography Object Detection Training")
parser.add_argument(
    "--dataset_root",
    type=str,
    default="/mammography",
    help="Path to the dataset root directory",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./trained_models",
    help="Path to save/load trained models",
)
parser.add_argument(
    "--n_ctx", type=int, default=4, help="Number of learnable context tokens"
)

args = parser.parse_args()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

DATASET_ROOT = args.dataset_root
MODEL_PATH = args.model_path
N_CTX = args.n_ctx


class MammographyDataset(Dataset):
    def __init__(self, df, img_dir):
        self.data = df.reset_index(drop=True)
        self.img_dir = img_dir

        self.label_map = {"BENIGN": 0, "MALIGNANT": 1}
        self.data["label"] = self.data["pathology"].map(self.label_map)

        for col in ["xmin", "ymin", "xmax", "ymax"]:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")

        original_width, original_height = image.size

        target_height, target_width = 981, 800

        image = image.resize((target_width, target_height), Image.BILINEAR)

        scale_x = target_width / original_width
        scale_y = target_height / original_height

        label = torch.tensor(row["label"], dtype=torch.long)

        if label == 1:
            scaled_xmin = row["xmin"] * scale_x
            scaled_ymin = row["ymin"] * scale_y
            scaled_xmax = row["xmax"] * scale_x
            scaled_ymax = row["ymax"] * scale_y

            bbox_xywh = torch.tensor(
                [
                    scaled_xmin,
                    scaled_ymin,
                    scaled_xmax - scaled_xmin,
                    scaled_ymax - scaled_ymin,
                ],
                dtype=torch.float32,
            )
            annotations = {
                "image_id": 0,
                "annotations": [
                    {
                        "bbox": bbox_xywh,
                        "area": float(bbox_xywh[2] * bbox_xywh[3]),
                        "category_id": 0,
                        "iscrowd": 0,
                    }
                ],
            }
        else:
            annotations = {"image_id": 0, "annotations": []}

        return image, annotations


def grounding_dino_collate_fn(batch):
    images, annotations = zip(*batch)
    return list(images), list(annotations)


def create_test_dataloaders():
    datasets = ["A", "B", "C"]
    test_loaders = []

    for dataset in datasets:
        test_csv = f"{DATASET_ROOT}/dataset_{dataset}/test/test.csv"
        test_img_dir = f"{DATASET_ROOT}/dataset_{dataset}/test"

        test_df = pd.read_csv(test_csv)
        test_df.columns = [c.strip().lower().replace(" ", "_") for c in test_df.columns]

        test_dataset = MammographyDataset(test_df, test_img_dir)
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            collate_fn=grounding_dino_collate_fn,
        )
        test_loaders.append(test_loader)

    return tuple(test_loaders)


class CoOpContext(nn.Module):
    def __init__(self, device, processor, model, initial_prompt):
        super().__init__()

        tokens = processor.tokenizer(
            initial_prompt, return_tensors="pt", padding=False
        ).to(device)
        input_ids = tokens["input_ids"]

        with torch.no_grad():
            text_embeds = model.model.text_backbone.embeddings.word_embeddings(
                input_ids
            )

        full_embeds = text_embeds[0, :, :]

        self.all_embeddings = nn.Parameter(full_embeds.clone())
        self.initial_token_ids = input_ids[0].clone()

        seq_len = full_embeds.shape[0]
        self.trainable_mask = torch.ones(seq_len, dtype=torch.bool)
        self.trainable_mask[0] = False
        self.trainable_mask[-2] = False
        self.trainable_mask[-1] = False

        def freeze_hook(grad):
            mask = self.trainable_mask.unsqueeze(-1).to(grad.device)
            return grad * mask

        self.hook_handle = self.all_embeddings.register_hook(freeze_hook)

    def forward(self):
        return self.all_embeddings

    def parameters(self, recurse=True):
        return super().parameters(recurse)


def move_labels_to_device(labels, device):
    new_labels = []
    for lbl in labels:
        new_lbl = {}
        for k, v in lbl.items():
            if torch.is_tensor(v):
                new_lbl[k] = v.to(device)
            else:
                new_lbl[k] = v
        new_labels.append(new_lbl)
    return new_labels


def test_one(model, processor, context_module, test_loader, dataset_name):
    model.eval()
    context_module.eval()

    test_map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75]).to(device)
    test_map_metric.warn_on_many_detections = False

    test_map_metric.reset()

    total_test_loss = 0.0

    test_pbar = tqdm(test_loader, desc=f"Testing {dataset_name}")

    with torch.no_grad():
        for images, annotations in test_pbar:
            batch_size = len(images)

            inputs = processor(
                images=images, annotations=annotations, return_tensors="pt"
            ).to(device)
            del inputs["pixel_mask"]

            context_expanded = context_module().unsqueeze(0).expand(batch_size, -1, -1)
            context_token_ids = context_module.initial_token_ids.unsqueeze(0).expand(
                batch_size, -1
            )

            prompt_enc = {}
            prompt_enc["input_ids"] = context_token_ids
            prompt_enc["inputs_embeds"] = context_expanded

            labels = move_labels_to_device(inputs["labels"], device)
            inputs["labels"] = labels

            inputs = inputs | prompt_enc

            outputs = model(**inputs)
            loss = outputs.loss
            loss_dict = outputs.loss_dict

            weight_dict = {
                "loss_ce": 2.0,
                "loss_bbox": model.config.bbox_loss_coefficient,
                "loss_giou": model.config.giou_loss_coefficient,
            }
            enc_weight_dict = {k + "_enc": v for k, v in weight_dict.items()}
            weight_dict.update(enc_weight_dict)
            weight_dict["loss_ce_enc"] = 0
            weight_dict["loss_bbox_enc"] = 0
            weight_dict["loss_giou_enc"] = 0

            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

            total_test_loss += loss.item()

            target_sizes = [img.size[::-1] for img in images]
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0,
                text_threshold=0,
                target_sizes=target_sizes,
            )

            preds = []
            targets = []

            for res, anno in zip(results, annotations):
                preds.append(
                    {
                        "boxes": res["boxes"],
                        "scores": res["scores"],
                        "labels": torch.zeros(
                            len(res["boxes"]), dtype=torch.long, device=device
                        ),
                    }
                )

                if len(anno["annotations"]) > 0:
                    gt_boxes = torch.stack([a["bbox"] for a in anno["annotations"]])
                    gt_boxes_xyxy = torch.zeros_like(gt_boxes)
                    gt_boxes_xyxy[:, 0] = gt_boxes[:, 0]
                    gt_boxes_xyxy[:, 1] = gt_boxes[:, 1]
                    gt_boxes_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
                    gt_boxes_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]

                    targets.append(
                        {
                            "boxes": gt_boxes_xyxy.to(device),
                            "labels": torch.zeros(
                                len(gt_boxes), dtype=torch.long, device=device
                            ),
                        }
                    )
                else:
                    targets.append(
                        {
                            "boxes": torch.empty((0, 4), device=device),
                            "labels": torch.empty(0, dtype=torch.long, device=device),
                        }
                    )

            test_map_metric.update(preds, targets)
            test_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_test_loss = total_test_loss / len(test_loader)
    test_metrics = test_map_metric.compute()

    print(f"\n{'='*70}")
    print(f"  {dataset_name} Results")
    print(f"{'='*70}")
    print(f"  Average Loss       : {avg_test_loss:.6f}")
    print(f"  mAP@50            : {test_metrics['map_50']:.4f}")
    print(f"  mAP@75            : {test_metrics['map_75']:.4f}")
    print(f"  mAP (Overall)     : {test_metrics['map']:.4f}")
    print(f"{'='*70}\n")


def test_all_datasets(model, processor):
    if N_CTX == 4:
        initial_prompt = "malignant tumor cancer."
    elif N_CTX == 1:
        initial_prompt = "cancer."
    else:
        SystemError("Unsupported number of context tokens.")

    context_module = CoOpContext(
        device=device,
        processor=processor,
        model=model,
        initial_prompt=initial_prompt,
    )

    def freeze_hook(grad):
        mask = context_module.trainable_mask.unsqueeze(-1).to(grad.device)
        return grad * mask

    context_module.all_embeddings.register_hook(freeze_hook)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    context_module.all_embeddings.data = checkpoint["context_vectors"]

    test_loader_1, test_loader_2, test_loader_3 = create_test_dataloaders()

    test_one(
        model, processor, context_module, test_loader_1, dataset_name="Test Dataset A"
    )
    test_one(
        model, processor, context_module, test_loader_2, dataset_name="Test Dataset B"
    )
    test_one(
        model, processor, context_module, test_loader_3, dataset_name="Test Dataset C"
    )


if __name__ == "__main__":
    print(device)

    model_id = "IDEA-Research/grounding-dino-base"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    test_all_datasets(model=model, processor=processor)
