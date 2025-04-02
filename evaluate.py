import torch
import tqdm
import numpy as np
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
