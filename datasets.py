import glob
import albumentations as A
import cv2

from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

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
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=25),
    ], is_check_shapes=False)
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
    ], is_check_shapes=False)
    return valid_image_transform

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # citim și augmentăm
        img = cv2.cvtColor(cv2.imread(self.image_paths[index]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(self.mask_paths[index]), cv2.COLOR_BGR2RGB)
        transformed = self.tfms(image=img, mask=msk)
        img, msk = transformed['image'], transformed['mask']

        # build label mask (0,1,...)
        label_mask = get_label_mask(msk, self.class_values, self.label_colors_list)

        # de la HWC uint8 → CHW float32 [0,1]
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        mask_tensor = torch.from_numpy(label_mask).long()

        return img_tensor, mask_tensor

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=2,
        shuffle=True
    )
    valid_data_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=2,
        shuffle=False
    )

    return train_data_loader, valid_data_loader