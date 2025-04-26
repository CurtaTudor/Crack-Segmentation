# python infer_image.py --input input/valid/images/ --device cpu --imgsz 1568 1088
# python infer_image.py --input input/edmcrack_images/ --device cpu --imgsz 1568 1088

from transformers import (
    SegformerFeatureExtractor, 
    SegformerForSemanticSegmentation
)
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import (
    draw_segmentation_map, 
    image_overlay,
    predict
)

import argparse
import cv2
import os
import glob
import torch
from model import unet_model

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input image directory',
    default='input/inference_data/images'
)
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu or cuda'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--model',
    default='outputs/model_iou'
)
args = parser.parse_args()

out_dir = 'outputs/inference_results_image'
os.makedirs(out_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = unet_model(['_background_','crack']).to(device).eval()

model.load_state_dict(torch.load('outputs/final_model/pytorch_model.bin'))

image_paths = glob.glob(os.path.join(args.input, '*'))
for image_path in image_paths:
    image = cv2.imread(image_path)

    h, w = image.shape[:2]

    img_size = (w,h)

    image = cv2.resize(image, img_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(image).permute(2,0,1).float()/255.0
    inp = img_tensor.unsqueeze(0).to(device)  # batch dim

    with torch.no_grad():
        logits = model(inp)

    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    preds = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()
    seg_map = draw_segmentation_map(preds, LABEL_COLORS_LIST)
    out = image_overlay(image, seg_map)
    cv2.imshow('Image', out)
    cv2.waitKey(4000)
    
    # Save path.
    image_name = image_path.split(os.path.sep)[-1]
    save_path = os.path.join(
        out_dir, '_'+image_name
    )
    cv2.imwrite(save_path, out)