import argparse
import os
import cv2
import sys

# două importuri pentru dialogul de fișiere
import tkinter as tk
from tkinter import filedialog

# --- setup parser (dacă vrei să păstrezi și --device, --imgsz etc) ---
parser = argparse.ArgumentParser()
parser.add_argument(
    '--device',
    default='cuda:0',
    help='compute device, cpu sau cuda'
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    nargs=2,
    help='width height'
)
parser.add_argument(
    '--model',
    default='out/outputs/model_iou',
    help='cale către model'
)
args = parser.parse_args()

# --- inițializare dialog fișier ---
root = tk.Tk()
root.withdraw()  # ascunde fereastra principală TK

file_path = filedialog.askopenfilename(
    title="Selectează o imagine",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
)
if not file_path:
    print("Nicio imagine selectată. Ies din program.")
    sys.exit(0)

# creează directorul de output
out_dir = 'out/outputs/inference_results_image'
os.makedirs(out_dir, exist_ok=True)

# importă modelul și utilitarele
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import draw_segmentation_map, image_overlay, predict

extractor = SegformerFeatureExtractor()
model = SegformerForSemanticSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

# citește și procesează imaginea selectată
image = cv2.imread(file_path)
cv2.imshow('Original', image)
if args.imgsz:
    image = cv2.resize(image, tuple(args.imgsz))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# predict + desenare hartă
labels = predict(model, extractor, image, args.device)
seg_map = draw_segmentation_map(labels.cpu(), LABEL_COLORS_LIST)
output = image_overlay(image, seg_map)

# afișare și salvare
cv2.imshow('Result', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

basename = os.path.basename(file_path)
save_path = os.path.join(out_dir, 'seg_'+basename)
cv2.imwrite(save_path, output)
print(f"Rezultat salvat în: {save_path}")
