import os
import numpy as np
import pandas as pd
import torch
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


def inference_one(model, processor, prompt, test_loader, dataset_name):
    model.eval()

    test_map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75]).to(device)
    test_map_metric.warn_on_many_detections = False
    test_map_metric.reset()

    test_pbar = tqdm(test_loader, desc=f"Testing {dataset_name}")

    tokens = processor.tokenizer(prompt, return_tensors="pt", padding=False).to(device)
    input_ids = tokens["input_ids"]

    with torch.no_grad():
        text_embeds = model.model.text_backbone.embeddings.word_embeddings(input_ids)

    full_embeds = text_embeds[0, :, :]

    with torch.no_grad():
        for images, annotations in test_pbar:
            batch_size = len(images)

            inputs = processor(images=images, return_tensors="pt").to(device)
            del inputs["pixel_mask"]

            prompt_enc = {}
            prompt_enc["input_ids"] = input_ids[0].unsqueeze(0).expand(batch_size, -1)
            prompt_enc["inputs_embeds"] = full_embeds.unsqueeze(0).expand(
                batch_size, -1, -1
            )

            inputs = inputs | prompt_enc

            outputs = model(**inputs)

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
                k = 1000
                top_k = min(k, len(res["scores"]))
                if top_k > 0:
                    sorted_indices = res["scores"].argsort(descending=True)[:top_k]
                    top_boxes = res["boxes"][sorted_indices]
                    top_scores = res["scores"][sorted_indices]
                else:
                    top_boxes = res["boxes"]
                    top_scores = res["scores"]

                preds.append(
                    {
                        "boxes": top_boxes,
                        "scores": top_scores,
                        "labels": torch.zeros(
                            len(top_boxes), dtype=torch.long, device=device
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

    test_metrics = test_map_metric.compute()

    print(f"\n{'='*70}")
    print(f"  {dataset_name} Results")
    print(f"{'='*70}")
    print(f"  mAP@50            : {test_metrics['map_50']:.4f}")
    print(f"  mAP@75            : {test_metrics['map_75']:.4f}")
    print(f"  mAP (Overall)     : {test_metrics['map']:.4f}")
    print(f"{'='*70}\n")


def inference_all_datasets(model, processor, prompt):
    test_loader_1, test_loader_2, test_loader_3 = create_test_dataloaders()

    inference_one(
        model, processor, prompt, test_loader_1, dataset_name="Test Dataset A"
    )
    inference_one(
        model, processor, prompt, test_loader_2, dataset_name="Test Dataset B"
    )
    inference_one(
        model, processor, prompt, test_loader_3, dataset_name="Test Dataset C"
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
    prompt = "malignant tumor cancer."

    inference_all_datasets(model=model, processor=processor, prompt=prompt)
