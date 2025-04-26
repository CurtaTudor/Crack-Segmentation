# train.py (modificat)

import os
import torch
import argparse
from datasets import get_data_loaders, get_images  # :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
from model import unet_model
from config import ALL_CLASSES, LABEL_COLORS_LIST
from engine import train_epoch, validate_epoch
from torch.optim.lr_scheduler import MultiStepLR
from utils import SaveBestModel, SaveBestModelIOU, save_plots, save_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, nargs=2, default=[512,416])
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--scheduler-epochs', nargs='+', type=int, default=[30])
    args = parser.parse_args()

    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'valid_preds'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet_model(ALL_CLASSES).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.scheduler:
        scheduler = MultiStepLR(optimizer, milestones=args.scheduler_epochs, gamma=0.1, verbose=True)

    train_imgs, train_masks, valid_imgs, valid_masks = get_images('input')
    train_loader, valid_loader = get_data_loaders(
        train_imgs, train_masks, valid_imgs, valid_masks,
        ALL_CLASSES, ALL_CLASSES, LABEL_COLORS_LIST,
        img_size=args.imgsz, batch_size=args.batch
    )

    save_best_loss = SaveBestModel()
    save_best_iou  = SaveBestModelIOU()

    history = {'train_loss':[], 'train_miou':[], 'valid_loss':[], 'valid_miou':[]}

    for epoch in range(args.epochs):
        tl, tmiou = train_epoch(model, train_loader, optimizer, device, len(ALL_CLASSES))
        vl, vmiou = validate_epoch(model, valid_loader, device, len(ALL_CLASSES), LABEL_COLORS_LIST, epoch, os.path.join(out_dir,'valid_preds'))

        history['train_loss'].append(tl); history['train_miou'].append(tmiou)
        history['valid_loss'].append(vl); history['valid_miou'].append(vmiou)

        save_best_loss(vl, epoch, model, out_dir, name='model_loss')
        save_best_iou(vmiou, epoch, model, out_dir, name='model_iou')

        print(f"[{epoch+1}/{args.epochs}] train_loss={tl:.4f}, train_mIoU={tmiou:.4f} | valid_loss={vl:.4f}, valid_mIoU={vmiou:.4f}")

        if args.scheduler:
            scheduler.step()

    save_plots(
        history['train_loss'], history['valid_loss'],
        [], [],  # dacă nu mai folosim pix_acc
        history['train_miou'], history['valid_miou'],
        out_dir
    )
    save_model(model, out_dir, name='final_model')
    print("TRAINING COMPLETE")
