import os
import json
import torch
import tqdm
import numpy as np
import pandas as pd
import argparse
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_convert
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform):
        self.image_paths = [
            (os.path.join(image_path, file_name), file_name.split(".")[0])
            for file_name in os.listdir(image_path)
            if file_name.endswith(".png") or file_name.endswith(".jpg")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, image_id = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_id


class CocoDetectionV2(torchvision.datasets.CocoDetection):

    def __getitem__(self, idx):
        img, target = super(CocoDetectionV2, self).__getitem__(idx)

        if len(target) > 0:
            boxes = torch.tensor([obj["bbox"] for obj in target], dtype=torch.float32)
            # [x, y, w, h] to [x1, y1, x2, y2]
            boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
            spatial_size = F.get_image_size(img)
            transformed = self.transform(
                {
                    "image": img,
                    "boxes": boxes,
                    "labels": torch.tensor(
                        [obj["category_id"] for obj in target], dtype=torch.int64
                    ),
                }
            )
            img = transformed["image"]
            target_dict = {
                "boxes": transformed["boxes"],
                "labels": transformed["labels"],
                "image_id": (
                    torch.tensor(target[0]["image_id"]) if target else torch.tensor(-1)
                ),
            }
        else:
            target_dict = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor(-1),
            }
            if self.transform is not None:
                img = self.transform(img)

        return img, target_dict


def test_collate_fn(batch):
    images, image_ids = zip(*batch)
    images_list = []
    for img in images:
        images_list.append(img)
    return list(images_list), image_ids


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_dataset(
    batch_size=64,
    train_annotations_path="./nycu-hw2-data/nycu-hw2-data/train.json",
    train_images_path="./nycu-hw2-data/nycu-hw2-data/train",
    valid_annotations_path="./nycu-hw2-data/nycu-hw2-data/valid.json",
    valid_images_path="./nycu-hw2-data/nycu-hw2-data/valid",
    test_images_path="./nycu-hw2-data/nycu-hw2-data/test",
    num_workers=0 if os.name == "nt" else 4,
    verbose=True,
):
    train_transforms = transforms_v2.Compose(
        [
            transforms_v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms_v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0), p=0.2),
            transforms_v2.RandomGrayscale(p=0.1),
            transforms_v2.RandomEqualize(p=0.1),
            transforms_v2.RandomPosterize(bits=4, p=0.1),
            transforms_v2.RandomAutocontrast(p=0.1),
            transforms_v2.RandomInvert(p=0.1),
            transforms_v2.RandomSolarize(threshold=128, p=0.1),
            # Geometric transformations
            transforms_v2.RandomAffine(
                degrees=6,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=10,
                fill=0,
            ),
            transforms_v2.RandomPerspective(distortion_scale=0.1, p=0.1),
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
        ]
    )
    valid_transforms = transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = CocoDetectionV2(
        root=train_images_path,
        annFile=train_annotations_path,
        transform=train_transforms,
    )

    valid_dataset = CocoDetectionV2(
        root=valid_images_path,
        annFile=valid_annotations_path,
        transform=valid_transforms,
    )

    test_dataset = TestDataset(image_path=test_images_path, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=test_collate_fn,
    )

    if verbose:
        print(f"Loaded {len(train_dataset)} train images.")
        print(f"Loaded {len(valid_dataset)} valid images.")
        print(f"Loaded {len(test_dataset)} test images.")

    return train_loader, valid_loader, test_loader


def get_model(
    resnet_or_mobilenet="resnet",
    num_classes=10 + 1,  # 10 classes + background
    trainable_backbone_layers=3,  # trainable_backbone_layers = 4
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if resnet_or_mobilenet == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    elif resnet_or_mobilenet == "resnet":
        model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT", trainable_backbone_layers=trainable_backbone_layers
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=num_classes
    )
    model.to(device)
    return model


def evaluate(
    model,
    data_loader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    iou_threshold=0.5,
    score_threshold=0.3,
):
    total_loss = 0
    coco_predictions = []
    task2_predictions = []
    task2_targets = []
    coco_targets = []

    # 初始化 COCO 对象
    coco = COCO(annotation_file="./nycu-hw2-data/nycu-hw2-data/valid.json")
    coco.dataset["categories"] = [{"id": i + 1, "name": str(i)} for i in range(10)]
    coco.createIndex()

    with torch.no_grad():
        for images, targets in tqdm.tqdm(data_loader, ncols=120, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Validation Loss
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            model.eval()
            outputs = model(images)
            for img_id, (output, target) in enumerate(zip(outputs, targets)):
                # 过滤低置信度预测
                keep = torch.where(output["scores"] > score_threshold)[0]
                boxes = output["boxes"][keep].cpu().numpy()
                scores = output["scores"][keep].cpu().numpy()
                labels = output["labels"][keep].cpu().numpy()

                # 应用NMS
                keep = (
                    torchvision.ops.nms(
                        torch.tensor(boxes).to(device),
                        torch.tensor(scores).to(device),
                        iou_threshold,
                    )
                    .cpu()
                    .numpy()
                )
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                image_id = target["image_id"].item()
                for box, score, label in zip(boxes, scores, labels):
                    coco_pred = {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [
                            box[0],
                            box[1],
                            box[2] - box[0],
                            box[3] - box[1],
                        ],  # 转换回 [x, y, w, h]
                        "score": float(score),
                    }
                    coco_predictions.append(coco_pred)

                # Task1: record targets for mAP
                for box, label in zip(
                    target["boxes"].cpu().numpy(), target["labels"].cpu().numpy()
                ):
                    coco_targets.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        }
                    )

                # Task2: accuracy
                sorted_indices = np.argsort([b[0] for b in boxes])
                pred_digits = "".join([str(int(labels[i])) for i in sorted_indices])
                true_sorted_indices = np.argsort(
                    [b[0] for b in target["boxes"].cpu().numpy()]
                )
                true_digits = "".join(
                    [
                        str(int(target["labels"].cpu().numpy()[i]))
                        for i in true_sorted_indices
                    ]
                )
                task2_predictions.append(pred_digits)
                task2_targets.append(true_digits)

    # 计算 mAP
    if not coco_predictions:
        print("Warning: No valid predictions detected in Validation!")
        map_score = 0.0
    else:
        coco_results = coco.loadRes(coco_predictions)
        coco_eval = COCOeval(coco, coco_results, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_score = coco_eval.stats[0]  # AP@[0.5:0.95]

    # 计算 Task2 准确率
    correct_predictions = sum(
        pred == target for pred, target in zip(task2_predictions, task2_targets)
    )
    task2_accuracy = correct_predictions / len(task2_targets) if task2_targets else 0

    avg_loss = total_loss / len(data_loader)
    print(
        f"Validation Loss: {avg_loss:.4f}, Task1 mAP: {map_score:.4f}, Task2 Acc: {task2_accuracy:.4f}"
    )
    return {"val_loss": avg_loss, "task1_mAP": map_score, "task2_acc": task2_accuracy}


def train_model(
    model,
    train_loader,
    valid_loader,
    num_epochs=40,
    val_epoch_list=[3, 7, 11, 15, 19, 23, 27, 31, 35, 39],
    learning_rate=5e-4,
    weight_decay=0.0005,
    checkpoint_path="./checkpoints",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params=params, lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs, eta_min=1e-6
    )
    scaler = GradScaler(device="cuda")

    best_map = 0.0
    best_accuracy = 0.0
    second_best_map = 0.0
    second_best_accuracy = 0.0
    third_best_map = 0.0
    third_best_accuracy = 0.0

    for epoch in range(num_epochs):
        print("\n\n" + ">" * 50 + f"\nTraining Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for images, targets in tqdm.tqdm(train_loader, ncols=120, desc="Training"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            with autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")
        if epoch in val_epoch_list:
            print(f"Evaluating after epoch {epoch+1}/{num_epochs}...")
            val_results = evaluate(model=model, data_loader=valid_loader)
            val_map = val_results["task1_mAP"]
            val_accuracy = val_results["task2_acc"]
            if val_map > best_map and val_accuracy > best_accuracy:
                best_map = val_map
                best_accuracy = val_accuracy
                print(
                    f"Saving best model with mAP: {best_map:.4f} and accuracy: {best_accuracy:.4f}"
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        checkpoint_path,
                        f"best_map{best_map:.4f}_acc{best_accuracy:.4f}.pth",
                    ),
                )
            elif val_map > second_best_map and val_accuracy > second_best_accuracy:
                second_best_map = val_map
                second_best_accuracy = val_accuracy
                print(
                    f"Saving second best model with mAP: {second_best_map:.4f} and accuracy: {second_best_accuracy:.4f}"
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        checkpoint_path,
                        f"second_map{best_map:.4f}_acc{best_accuracy:.4f}.pth",
                    ),
                )
            elif val_map > third_best_map and val_accuracy > third_best_accuracy:
                third_best_map = val_map
                third_best_accuracy = val_accuracy
                print(
                    f"Saving third best model with mAP: {third_best_map:.4f} and accuracy: {third_best_accuracy:.4f}"
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        checkpoint_path,
                        f"third_map{best_map:.4f}_acc{best_accuracy:.4f}.pth",
                    ),
                )


def visualize_predictions(
    image, boxes, labels, scores, image_id, visualize_save_path="./visualize"
):
    image = image.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x_min,
            y_min - 5,
            f"{int(label)} ({score:.2f})",
            color="red",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )

    ax.axis("off")
    if not os.path.exists(visualize_save_path):
        os.makedirs(visualize_save_path)
    plt.savefig(
        os.path.join(visualize_save_path, f"{image_id}.png"), bbox_inches="tight"
    )


def test_model(
    model,
    test_checkpoint_path,
    test_loader,
    score_threshold=0.3,
    iou_threshold=0.5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    visualize_save_path="./visualize",
    verbose=False,
):
    if test_checkpoint_path == "":
        raise ValueError("Please provide a valid checkpoint path for testing!")
    if not os.path.exists(test_checkpoint_path):
        raise ValueError(f"Checkpoint path {test_checkpoint_path} does not exist!")

    model.load_state_dict(torch.load(test_checkpoint_path, map_location="cpu"))
    model.eval()
    model.to(device)
    print(">" * 50 + "\nBegin testing!")

    coco_predictions = []  # Task1
    csv_predictions = []  # Task2

    with torch.no_grad():
        for images, image_ids in tqdm.tqdm(test_loader, ncols=120, desc="Testing"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for idx, output in enumerate(outputs):
                image_id = image_ids[idx]

                # 过滤低置信度预测
                keep = torch.where(output["scores"] >= score_threshold)[0]
                boxes = output["boxes"][keep].cpu().numpy()
                scores = output["scores"][keep].cpu().numpy()
                labels = output["labels"][keep].cpu().numpy()

                # 应用NMS
                keep = (
                    torchvision.ops.nms(
                        torch.tensor(boxes).to(device),
                        torch.tensor(scores).to(device),
                        iou_threshold,
                    )
                    .cpu()
                    .numpy()
                )
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # Task1
                for box, score, label in zip(boxes, scores, labels):
                    coco_pred = {
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2] - box[0]),
                            float(box[3] - box[1]),
                        ],
                        "score": float(score),
                    }
                    coco_predictions.append(coco_pred)

                # Task2
                sorted_indices = np.argsort([b[0] for b in boxes])  # 按 x 坐标排序
                pred_digits = "".join([str(int(labels[i]) - 1) for i in sorted_indices])
                if len(pred_digits) == 0:
                    pred_digits = "-1"
                csv_predictions.append(
                    {"image_id": int(image_id), "pred_label": pred_digits}
                )
                if verbose:
                    print(f"Image ID: {image_id}, Predicted Digits: {pred_digits}")
                    visualize_predictions(
                        images[0], boxes, labels, scores, image_id, visualize_save_path
                    )

    # pred.json
    with open("pred.json", "w") as f:
        json.dump(coco_predictions, f)
    print("Saved predictions of task1 to pred.json")

    # pred.csv
    df = pd.DataFrame(csv_predictions)
    df.to_csv("pred.csv", index=False)
    print("Saved predictions of task2 to pred.csv")


def args_parser():
    parser = argparse.ArgumentParser(
        description="Faster R-CNN Model for Digit Detection"
    )
    parser.add_argument("-f")
    parser.add_argument("--HistoryManager.hist_file", default=":memory:")

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--resnet_or_mobilenet",
        type=str,
        default="resnet",
        choices=["resnet", "mobilenet"],
        help="Backbone type for Faster R-CNN",
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Score threshold for filtering predictions",
    )
    parser.add_argument(
        "--trainable_backbone_layers",
        type=int,
        default=4,
        help="Number of trainable backbone layers",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--val_epoch_list",
        type=int,
        nargs="+",
        default=[4, 9, 14, 19, 24, 29, 34, 39],
        help="Epochs at which to evaluate the model",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer"
    )

    parser.add_argument(
        "--train_annotations_path",
        type=str,
        default="./nycu-hw2-data/nycu-hw2-data/train.json",
        help="Path to training annotations in COCO format",
    )
    parser.add_argument(
        "--train_images_path",
        type=str,
        default="./nycu-hw2-data/nycu-hw2-data/train",
        help="Path to training images",
    )
    parser.add_argument(
        "--valid_annotations_path",
        type=str,
        default="./nycu-hw2-data/nycu-hw2-data/valid.json",
        help="Path to validation annotations in COCO format",
    )
    parser.add_argument(
        "--valid_images_path",
        type=str,
        default="./nycu-hw2-data/nycu-hw2-data/valid",
        help="Path to validation images",
    )
    parser.add_argument(
        "--test_images_path",
        type=str,
        default="./nycu-hw2-data/nycu-hw2-data/test",
        help="Path to test images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=(0 if os.name == "nt" else 4),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Whether to print detailed logs"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10 + 1,
        help="Number of classes (including background)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to use for training and testing",
    )
    parser.add_argument(
        "--test_checkpoint_path",
        type=str,
        default="",
        help="Path to the model checkpoint for testing",
    )

    parser.add_argument(
        "--visualize_save_path",
        type=str,
        default="./visualize",
        help="Path to save visualized predictions",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints",
        help="Path to save model checkpoints",
    )

    return parser.parse_args()


def main(args):
    train_loader, valid_loader, test_loader = load_dataset(
        batch_size=args.batch_size,
        train_annotations_path=args.train_annotations_path,
        train_images_path=args.train_images_path,
        valid_annotations_path=args.valid_annotations_path,
        valid_images_path=args.valid_images_path,
        test_images_path=args.test_images_path,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )
    model = get_model(
        resnet_or_mobilenet=args.resnet_or_mobilenet,
        num_classes=args.num_classes,
        trainable_backbone_layers=args.trainable_backbone_layers,
        device=args.device,
    )
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=args.num_epochs,
        val_epoch_list=args.val_epoch_list,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )


def test(args):
    _, _, test_loader = load_dataset(
        batch_size=args.batch_size,
        train_annotations_path=args.train_annotations_path,
        train_images_path=args.train_images_path,
        valid_annotations_path=args.valid_annotations_path,
        valid_images_path=args.valid_images_path,
        test_images_path=args.test_images_path,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )
    model = get_model(
        resnet_or_mobilenet=args.resnet_or_mobilenet,
        num_classes=args.num_classes,
        trainable_backbone_layers=args.trainable_backbone_layers,
        device=args.device,
    )
    test_model(
        model=model,
        test_checkpoint_path=args.test_checkpoint_path,
        test_loader=test_loader,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        visualize_save_path=args.visualize_save_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    args = args_parser()
    main(args)
    # test(args)
