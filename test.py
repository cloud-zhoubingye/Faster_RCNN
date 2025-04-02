import os
import json
import torch
import tqdm
import numpy as np
import torchvision
import pandas as pd
from utils import visualize_predictions


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
