# python infer_video.py --input input/inference_data/videos/DJI.mp4 --device cpu --imgsz 1568 1088
# python infer_video.py --input input/inference_data/videos/my_video.mp4 --imgsz 1568 1088
# python infer_video.py --input input/inference_data/videos/video_7.mp4 --device cpu

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
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    help='path to the input video',
    default='input/inference_data/videos/video_1.mov'
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
    default='out/outputs/model_iou'
)
args = parser.parse_args()

extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

cap = cv2.VideoCapture(args.input)
if args.imgsz is None:
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
else: 
    frame_width = args.imgsz[0]
    frame_height = args.imgsz[1]
vid_fps = int(cap.get(5))
save_name = args.input.split(os.path.sep)[-1].split('.')[0]
while cap.isOpened:
    ret, frame = cap.read()
    if ret:
        image = frame
        if args.imgsz is not None:
            image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # Get labels.
        labels = predict(model, extractor, image, args.device)
        
        # Get segmentation map.
        seg_map = draw_segmentation_map(
            labels.cpu(), LABEL_COLORS_LIST
        )
        outputs = image_overlay(image, seg_map)
        cv2.imshow('Image', outputs)
        # Press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()
