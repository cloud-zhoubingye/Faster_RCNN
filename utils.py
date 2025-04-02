import os
import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import load_dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
)


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
