"""
Creates splits for a segmentation dataset where all images are in one directory
and all masks are in another directory.
"""

import os
import shutil
import random

from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)

SRC_ROOT_DIR = 'input/data_dataset_voc'
SRC_IMAGES = os.path.join(SRC_ROOT_DIR, 'JPEGImages')
SRC_MASKS = os.path.join(SRC_ROOT_DIR, 'SegmentationClassPNG')

DEST_ROOT_DIR = 'input'
TRAIN_IMAGES = os.path.join(DEST_ROOT_DIR, 'train', 'images')
TRAIN_MASKS = os.path.join(DEST_ROOT_DIR, 'train', 'masks')
VALID_IMAGES = os.path.join(DEST_ROOT_DIR, 'valid', 'images')
VALID_MASKS = os.path.join(DEST_ROOT_DIR, 'valid', 'masks')

os.makedirs(TRAIN_IMAGES, exist_ok=True)
os.makedirs(TRAIN_MASKS, exist_ok=True)
os.makedirs(VALID_IMAGES, exist_ok=True)
os.makedirs(VALID_MASKS, exist_ok=True)

VALID_SPLIT = 0.2

ALL_IMAGES = os.listdir(SRC_IMAGES)
ALL_MASKS = os.listdir(SRC_MASKS)

ALL_IMAGES.sort()
ALL_MASKS.sort()

print(ALL_MASKS[:3])
print(ALL_IMAGES[:3])

combined = list(zip(ALL_IMAGES, ALL_MASKS))
random.shuffle(combined)
SHUFFLED_IMAGES, SHUFFLED_MASKS = zip(*combined)

print(SHUFFLED_IMAGES[:3])
print(SHUFFLED_MASKS[:3])

TRAIN_SAMPLES = int(len(SHUFFLED_IMAGES) - VALID_SPLIT*len(SHUFFLED_IMAGES))
VALID_SAMPLES = int(VALID_SPLIT*len(SHUFFLED_IMAGES))

print(TRAIN_SAMPLES, VALID_SAMPLES)

def copy_data(images, masks, split='train'):
    if split == 'train':
        image_dest = TRAIN_IMAGES
        mask_dest = TRAIN_MASKS
    else:
        image_dest = VALID_IMAGES
        mask_dest = VALID_MASKS
    for i, data in tqdm(enumerate(images), total=len(images)):
        shutil.copy(
            src=os.path.join(SRC_IMAGES, images[i]),
            dst=os.path.join(image_dest, images[i])
        )
        shutil.copy(
            src=os.path.join(SRC_MASKS, masks[i]),
            dst=os.path.join(mask_dest, masks[i])
        )

FINAL_TRAIN_IMAGES = SHUFFLED_IMAGES[:TRAIN_SAMPLES]
FINAL_TRAIN_MASKS = SHUFFLED_MASKS[:TRAIN_SAMPLES]
FINAL_VALID_IMAGES = SHUFFLED_IMAGES[-VALID_SAMPLES:]
FINAL_VALID_MASKS = SHUFFLED_MASKS[-VALID_SAMPLES:]

print(len(FINAL_TRAIN_IMAGES))
print(len(FINAL_TRAIN_MASKS))
print(len(FINAL_VALID_IMAGES))
print(len(FINAL_VALID_MASKS))

copy_data(FINAL_TRAIN_IMAGES, FINAL_TRAIN_MASKS, split='train')
copy_data(FINAL_VALID_IMAGES, FINAL_VALID_MASKS, split='valid')