import os
import torch
import tqdm
from evaluate import evaluate
from torch.amp import autocast, GradScaler


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
