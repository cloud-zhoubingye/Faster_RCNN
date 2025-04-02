import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.ops.boxes import box_convert
import torchvision.transforms.v2 as transforms_v2


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
            transforms_v2.RandomApply(
                [transforms_v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1
            ),
            transforms_v2.RandomGrayscale(p=0.1),
            transforms_v2.RandomEqualize(p=0.1),
            transforms_v2.RandomPosterize(bits=4, p=0.1),
            transforms_v2.RandomAutocontrast(p=0.1),
            transforms_v2.RandomInvert(p=0.1),
            transforms_v2.RandomSolarize(threshold=128, p=0.1),
            # # Geometric transformations
            # transforms_v2.RandomAffine(
            #     degrees=6,
            #     translate=(0.1, 0.1),
            #     scale=(0.8, 1.2),
            #     shear=10,
            #     fill=0,
            # ),
            # transforms_v2.RandomPerspective(distortion_scale=0.1, p=0.1),
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
