import os
import torch
import argparse


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
        default=0.55,
        help="Score threshold for filtering predictions",
    )
    parser.add_argument(
        "--trainable_backbone_layers",
        type=int,
        default=3,
        help="Number of trainable backbone layers",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=14, help="Number of training epochs"
    )
    parser.add_argument(
        "--val_epoch_list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        metavar="N",
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
