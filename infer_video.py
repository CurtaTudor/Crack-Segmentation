# python infer_video.py --input input/inference_data/videos/video_3.mov --imgsz 800 600

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

# Parse command-line arguments.
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

# Create output directory if needed.
out_dir = 'out/outputs/inference_results_video'
os.makedirs(out_dir, exist_ok=True)

# Load feature extractor and model from Hugging Face.
extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

# Open the input video.
cap = cv2.VideoCapture(args.input)

# Read the first frame to determine output dimensions after rotation.
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Could not read the first frame from the video.")

# Rotate the first frame 90° clockwise to fix orientation.
first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)

# Set output dimensions.
if args.imgsz is None:
    frame_height, frame_width = first_frame.shape[:2]
else:
    frame_width = args.imgsz[0]
    frame_height = args.imgsz[1]

# Retrieve the input video FPS (fallback to 30 if unavailable).
vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
if vid_fps == 0:
    vid_fps = 30

# Extract the filename (without extension) for saving the output.
save_name = os.path.splitext(os.path.basename(args.input))[0]

# Initialize VideoWriter with corrected dimensions.
out = cv2.VideoWriter(
    f"{out_dir}/{save_name}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'),
    vid_fps,
    (frame_width, frame_height)
)

# Reset the video to start from the first frame.
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
total_fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Rotate the frame 90° clockwise to correct its orientation.
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Resize the frame if custom dimensions are provided.
    if args.imgsz is not None:
        frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Convert frame from BGR (OpenCV default) to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run prediction and measure inference time.
    start_time = time.time()
    labels = predict(model, extractor, image, args.device)
    end_time = time.time()
    
    fps = 1 / (end_time - start_time)
    total_fps += fps

    # Draw the segmentation map and overlay it on the image.
    seg_map = draw_segmentation_map(labels.cpu(), LABEL_COLORS_LIST)
    outputs = image_overlay(image, seg_map)

    # Overlay FPS text on the output frame.
    cv2.putText(
        outputs,
        f"{fps:.1f} FPS",
        (15, 35),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )
    
    # Write the processed frame to the output video and display it.
    out.write(outputs)
    cv2.imshow('Image', outputs)
    
    # Exit early if 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows.
cap.release()
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count if frame_count > 0 else 0
print(f"Average FPS: {avg_fps:.3f}")
