import glob
import albumentations as A
import cv2

from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms

def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train/images/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/train/masks/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/valid/images/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/valid/masks/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size[1], img_size[0]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=25),
    ])

def valid_transforms(img_size):
    return A.Compose([
        A.Resize(img_size[1], img_size[0]),
    ])

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, tfms, all_classes, classes_to_train, label_colors_list):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.class_values = set_class_values(all_classes, classes_to_train)
        self.label_colors_list = label_colors_list
        # pentru normalizare (aceleași medii și std-uri ca în SegFormer)
        self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2RGB)

        augmented = self.tfms(image=img, mask=mask)
        img, mask = augmented['image'], augmented['mask']

        # obţinem masca de etichete
        label_mask = get_label_mask(mask, self.class_values, self.label_colors_list)
        label_mask = torch.from_numpy(label_mask).long()

        # transformăm imaginea în tensor și normalizăm
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        img = self.normalize(img)

        return img, label_mask

def get_data_loaders(train_image_paths, train_mask_paths, valid_image_paths, valid_mask_paths,
                     all_classes, classes_to_train, label_colors_list, img_size, batch_size, num_workers=8):
    train_ds = SegmentationDataset(train_image_paths, train_mask_paths, train_transforms(img_size),
                                   all_classes, classes_to_train, label_colors_list)  # :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
    valid_ds = SegmentationDataset(valid_image_paths, valid_mask_paths, valid_transforms(img_size),
                                   all_classes, classes_to_train, label_colors_list)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader