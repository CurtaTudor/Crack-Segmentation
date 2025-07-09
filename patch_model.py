import os
import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2

mean_nums = [0.485, 0.456, 0.406]
std_nums  = [0.229, 0.224, 0.225]

transforms_dict = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(227),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ]),
    'val': transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)
    ])
}

def load_dataset(data_dir, batch_size):
    train_folder = os.path.join(data_dir, 'train')
    val_folder   = os.path.join(data_dir, 'val')
    train_dataset = datasets.ImageFolder(root=train_folder, transform=transforms_dict['train'])
    val_dataset   = datasets.ImageFolder(root=val_folder,   transform=transforms_dict['val'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, len(train_dataset), len(val_dataset), train_dataset.classes


def build_model(device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 2)
    )
    return model.to(device)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, train_size, val_size, classes = load_dataset(args.data_dir, args.batch_size)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (train_size if phase=='train' else val_size)
            epoch_acc = running_corrects.double() / (train_size if phase=='train' else val_size)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - start_time
    print(f'Training complete in {int(elapsed//60)}m {int(elapsed%60)}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), args.save_path)
    print(f'Model weights saved to {args.save_path}')


def predict_on_crops(model, image_path, height=227, width=227, output_path=None):
    device = next(model.parameters()).device
    idx_to_class = {0: 'Negative', 1: 'Positive'}
    transform = transforms_dict['val']

    img = cv2.imread(image_path)
    h, w, _ = img.shape
    pad_h = (height - h % height) % height
    pad_w = (width  - w % width)  % width
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                    borderType=cv2.BORDER_REFLECT)
    padded_h, padded_w = img_padded.shape[:2]
    out_img = img.copy()

    for i in range(0, padded_h, height):
        for j in range(0, padded_w, width):
            crop = img_padded[i:i+height, j:j+width]
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                _, pred = torch.max(outputs, 1)
            class_name = idx_to_class[pred.item()]
            color = (0,0,255) if class_name=='Positive' else (0,255,0)
            overlay = np.full(crop.shape, color, dtype=np.uint8)
            out_crop = cv2.addWeighted(crop, 0.9, overlay, 0.1, 0)
            h_copy = min(height, h - i)
            w_copy = min(width,  w - j)
            out_img[i:i+h_copy, j:j+w_copy] = out_crop[:h_copy, :w_copy]

    if output_path:
        cv2.imwrite(output_path, out_img)
        print(f'Inference output saved to {output_path}')
    return out_img


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Attempt to build and load state
    model = build_model(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    # If checkpoint is a state_dict, load weights; if a model, use directly
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")
    model.eval()
    predict_on_crops(model, args.input_image, args.crop_h, args.crop_w, args.output_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or inference for crack recognition')
    subparsers = parser.add_subparsers(dest='command')

    # Train command
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--data_dir',    type=str, required=True, help='Path to dataset directory with train/val subfolders')
    parser_train.add_argument('--epochs',      type=int, default=10)
    parser_train.add_argument('--batch_size',  type=int, default=32)
    parser_train.add_argument('--lr',          type=float, default=0.001)
    parser_train.add_argument('--step_size',   type=int, default=3)
    parser_train.add_argument('--gamma',       type=float, default=0.1)
    parser_train.add_argument('--save_path',   type=str, default='model.pth')

    # Inference command
    parser_infer = subparsers.add_parser('infer')
    parser_infer.add_argument('--model_path',  type=str, required=True)
    parser_infer.add_argument('--input_image', type=str, required=True)
    parser_infer.add_argument('--crop_h',      type=int, default=227)
    parser_infer.add_argument('--crop_w',      type=int, default=227)
    parser_infer.add_argument('--output_image',type=str, default='output.jpg')

    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'infer':
        inference(args)
    else:
        parser.print_help()
