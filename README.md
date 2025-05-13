# Faster R-CNN for digits detection
## Introduction
#### Overview
This project implements a dual-task digit analysis system using Faster R-CNN with PyTorch, combining object detection and sequence recognition capabilities. The system processes images containing multiple digits (0-9) through a configurable pipeline that supports both ResNet50-FPN and MobileNetV3 backbones, featuring COCO-format data loading with augmentation, dynamic batch processing, and multi-scale training. The model architecture includes customizable trainable layers and class-aware RoI heads, optimized via AdamW with cosine annealing learning rate scheduling. Evaluation simultaneously measures detection accuracy (through COCO mAP metrics) and ordered digit sequence recognition (via spatial sorting and sklearn's accuracy_score), with outputs generated in both JSON and CSV formats. The implementation provides full training/inference workflows, visualization tools, and command-line configuration.
#### Tasks
- Task1: Detect each digit in the image. Detect the class and bounding box of each digit in the image, e.g., the bboxes of “4” and “9”. The submission file is a JSON file in COCO format named pred.json. Specifically, it should be a list of labels, where each label is represented as a dictionary with the keys: image_id, bbox, score, and category_id.
- Task2: Recognize the entire digit in the image. Recognize the number of detected digits in the image, e.g., the number “49”. The submission file, pred.csv, should contain two columns: image_id and pred_label. If the model predict no number in an image, it will write -1 in the pred_label.

## How to install
#### Dataset description
You can find it in [Download Here](https://drive.google.com/file/d/13ZOC2mCCtiRCSS-xrmDV9dSyTjirqpSg/view?usp=sharing).  
- RGB images
- Training / Validation: 30,062 / 3,340
- Test: 13,068
- The training and validation labels are JSON files in COCO format. 
- The bounding boxes are described in the format [x_min, y_min, w, h] without normalization. 
- The category id starts from 1.   

#### Environment
You can prepare the code environment with following commands.
```bash
conda create --name yourenvname python=3.12
conda activate yourenvname
pip install -r ./requirements.txt
```

#### Run
Run the code for training with following commands.
```bash
python ./main.py
```
You can modify experiment arguements by ```--param value```, detailed as follows.  
|Parameter|Description|Default Value|
|----------------------------|-----------------------------------------------|-----------------------------------------|
| `--batch_size`             | Batch size for training and testing           | `4`                                     |
| `--resnet_or_mobilenet`    | Backbone type for Faster R-CNN                | `"resnet"`                              |
| `--iou_threshold`          | IoU threshold for Non-Maximum Suppression (NMS) | `0.5`                                   |
| `--score_threshold`        | Score threshold for filtering low-confidence predictions | `0.3`                                   |
| `--trainable_backbone_layers` | Number of trainable layers in the backbone  | `4`                                     |
| `--num_epochs`             | Total number of training epochs               | `40`                                    |
| `--val_epoch_list`         | Epochs at which to evaluate the model         | `[4, 9, 14, 19, 24, 29, 34, 39]`        |
| `--learning_rate`          | Learning rate for the optimizer               | `5e-4`                                  |
| `--weight_decay`           | Weight decay for the optimizer                | `0.0005`                                |
| `--train_annotations_path` | Path to training annotations in COCO format   | `"./nycu-hw2-data/nycu-hw2-data/train.json"` |
| `--train_images_path`      | Path to training images                       | `"./nycu-hw2-data/nycu-hw2-data/train"` |
| `--valid_annotations_path` | Path to validation annotations in COCO format | `"./nycu-hw2-data/nycu-hw2-data/valid.json"` |
| `--valid_images_path`      | Path to validation images                     | `"./nycu-hw2-data/nycu-hw2-data/valid"` |
| `--test_images_path`       | Path to test images                           | `"./nycu-hw2-data/nycu-hw2-data/test"`  |
| `--num_workers`            | Number of workers for data loading            | `0` (Windows) or `4` (other systems)    |
| `--verbose`                | Whether to print detailed logs                | `False`                                 |
| `--num_classes`            | Number of classes (including background)      | `11`                                    |
| `--device`                 | Device to use for training and testing        | `"cuda"` (if available) or `"cpu"`      |
| `--test_checkpoint_path`   | Path to the model checkpoint for testing      | `""`                                    |
| `--visualize_save_path`    | Path to save visualized predictions           | `"./visualize"`                         |
| `--checkpoint_path`        | Path to save model checkpoints                | `"./checkpoints"`                       |

Then test the model.
```bash
python ./test.py --test_checkpoint_path ./checkpoint/your_ckpt_name.pth
```

## Performance snapshot
![alt text](report/image.png)
