import torch
import torch.nn as nn

from tqdm import tqdm
from utils import draw_translucent_seg_maps
from metrics import IOUEval

def train(
    model,
    train_dataloader,
    device,
    optimizer,
    classes_to_train
):
    print('Training')
    model.train()

    # Use CrossEntropyLoss for UNet outputs
    criterion = nn.CrossEntropyLoss()

    train_running_loss = 0.0
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    prog_bar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )

    counter = 0  # batch counter
    for i, (imgs, masks) in enumerate(prog_bar):
        counter += 1

        imgs = imgs.to(device)       # [B, C, H, W]
        masks = masks.to(device)     # [B, H, W] with class indices

        optimizer.zero_grad()
        logits = model(imgs)         # [B, num_classes, H, W]
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

        # Update IOU metrics
        preds = logits.argmax(dim=1)
        iou_eval.addBatch(preds.data, masks.data)

        prog_bar.set_description(f"Train Loss: {loss.item():.4f}")

    # Average loss over epoch
    train_loss = train_running_loss / counter
    overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
    return train_loss, overall_acc, mIOU


def validate(
    model,
    valid_dataloader,
    device,
    classes_to_train,
    label_colors_list,
    epoch,
    save_dir
):
    print('Validating')
    model.eval()

    # Use same loss for validation
    criterion = nn.CrossEntropyLoss()

    valid_running_loss = 0.0
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    with torch.no_grad():
        prog_bar = tqdm(
            valid_dataloader,
            total=len(valid_dataloader),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        counter = 0  # batch counter

        for i, (imgs, masks) in enumerate(prog_bar):
            counter += 1

            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)      # [B, num_classes, H, W]
            loss = criterion(logits, masks)
            valid_running_loss += loss.item()

            preds = logits.argmax(dim=1)

            # Save the validation segmentation maps for the first batch
            if i == 0:
                # imgs: [B, C, H, W], logits: [B, num_classes, H, W]
                # move to CPU for visualization
                draw_translucent_seg_maps(
                    imgs.cpu(),
                    logits.cpu(),
                    epoch,
                    i,
                    save_dir,
                    label_colors_list,
                )

            # Update IOU metrics
            iou_eval.addBatch(preds.data, masks.data)

            prog_bar.set_description(f"Val Loss: {loss.item():.4f}")

    # Average validation loss
    valid_loss = valid_running_loss / counter
    overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
    return valid_loss, overall_acc, mIOU
