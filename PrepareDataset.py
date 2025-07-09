import glob
import albumentations as A
import cv2

from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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

def augment_training(img_size):
    augment_training_image = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=25),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(p=0.2),
        A.RandomCrop(width=img_size[0], height=img_size[1], p=0.5),
        A.ElasticTransform(p=0.2),

    ], check_shape=False)
    return augment_training_image

def augment_validation(img_size):
    augment_validation_image = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
    ], check_shape=False)
    return augment_validation_image

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
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
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image'].astype('uint8')
        mask = transformed['mask']

        mask = get_label_mask(mask, self.class_values, self.label_colors_list).astype('uint8')
        mask = Image.fromarray(mask)
               
        encoded_inputs = self.feature_extractor(
            Image.fromarray(image), 
            mask,
            return_tensors='pt'
        )
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size,
    feature_extractor
):
    train_tfms = augment_training(img_size)
    valid_tfms = augment_validation(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes, 
        feature_extractor
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
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