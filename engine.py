# engine.py (modificat)

import torch
import torch.nn as nn
from tqdm import tqdm
from metrics import IOUEval
from utils import draw_translucent_seg_maps

def train_epoch(model, loader, optimizer, device, num_classes):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    iou_eval = IOUEval(num_classes)

    for imgs, masks in tqdm(loader, total=len(loader)):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)  # shape: [B, C, H, W]

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # predicţie
        preds = torch.argmax(logits, dim=1)
        iou_eval.addBatch(preds, masks)

    avg_loss = running_loss / len(loader)
    _, _, _, mIoU = iou_eval.getMetric()
    return avg_loss, mIoU

def validate_epoch(model, loader, device, num_classes, label_colors_list, epoch, save_dir):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    iou_eval = IOUEval(num_classes)

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(loader, total=len(loader))):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)

            loss = criterion(logits, masks)
            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            iou_eval.addBatch(preds, masks)

    if i == 0:
        # scoatem primul element din batch și-i adăugăm dimensiunea batch=1
        sample_img    = imgs[0].unsqueeze(0)    # [1, C, H, W]
        sample_logits = logits[0].unsqueeze(0)  # [1, num_classes, H, W]
        draw_translucent_seg_maps(
            sample_img,
            sample_logits,
            epoch,
            i,
            save_dir,
            label_colors_list,
        )

    avg_loss = running_loss / len(loader)
    _, _, _, mIoU = iou_eval.getMetric()
    return avg_loss, mIoU
