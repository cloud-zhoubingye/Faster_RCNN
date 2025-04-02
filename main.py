from test import test_model
from utils import get_model
from train import train_model
from dataset import load_dataset
from argparser import args_parser


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
