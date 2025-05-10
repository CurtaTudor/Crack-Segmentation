import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import draw_segmentation_map, image_overlay, predict

# Dispozitivul pe care rulează modelul ("cuda:0" sau "cpu")
DEVICE = 'cpu'
# Dimensiune fixă la care redimensionăm (sau None pentru original)
IMGSZ = (1568, 1088)
# Calea către modelul salvat
MODEL_PATH = 'out/outputs/model_iou'

class CrackSegApp:
    def __init__(self, root):
        self.root = root
        root.title("Crack Detection - SegFormer")
        root.geometry("900x600")
        root.minsize(600, 400)

        # Încarcă modelul
        self.extractor = SegformerFeatureExtractor()
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
        self.model.to(DEVICE).eval()

        # Buton pentru selectarea imaginii
        btn = tk.Button(root, text="Select Image", command=self.select_image)
        btn.pack(pady=10)

        # Frame pentru a afișa imaginea originală în fereastra principală (opțional)
        frame = tk.Frame(root)
        frame.pack(expand=True, fill='both')
        self.label_orig = tk.Label(frame, compound='top')
        self.label_orig.pack(side='left', padx=10, pady=10, expand=True)

    def select_image(self):
        # Deschide dialogul de fișiere
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*")]
        )
        if not path:
            return

        # Citește și preprocesează imaginea
        image_bgr = cv2.imread(path)
        if IMGSZ:
            image_bgr = cv2.resize(image_bgr, IMGSZ)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Inferență
        labels = predict(self.model, self.extractor, image_bgr, DEVICE)
        seg_map = draw_segmentation_map(labels.cpu(), LABEL_COLORS_LIST)
        #output = image_overlay(image_rgb, seg_map)

        orig_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        seg_bgr  = cv2.cvtColor(seg_map,     cv2.COLOR_RGB2BGR)

        #   blend 1.0 original + 0.5 segmentare
        output = cv2.addWeighted(orig_bgr, 1.0, seg_bgr, 0.5, 0)

        # Afișează originalul în fereastra principală (opțional)
        #self.display_original(image_rgb)
        # Afișează original și rezultat în fereastră nouă
        self.display_result(image_rgb, output)

    def display_original(self, orig_np):
        orig = Image.fromarray(orig_np).resize((400, 400))
        self.photo_orig = ImageTk.PhotoImage(orig)
        self.label_orig.config(image=self.photo_orig, text="Original")

    def display_result(self, orig_np, seg_bgr_np):
        # Creează o fereastră nouă
        top = tk.Toplevel(self.root)
        top.title("Original & Segmentation Result")
        top.geometry("840x440")  # lățime dublă pentru două imagini

        # Prepare and display original image
        orig = Image.fromarray(orig_np).resize((400, 400))
        photo_orig = ImageTk.PhotoImage(orig)
        label_o = tk.Label(top, image=photo_orig, text="Original", compound='top')
        label_o.image = photo_orig  # păstrează referința
        label_o.pack(side='left', padx=10, pady=10)

        # Convert BGR OpenCV image to RGB and display segmentation
        seg_rgb = cv2.cvtColor(seg_bgr_np, cv2.COLOR_BGR2RGB)
        seg = Image.fromarray(seg_rgb).resize((400, 400))
        photo_seg = ImageTk.PhotoImage(seg)
        label_s = tk.Label(top, image=photo_seg, text="Result", compound='top')
        label_s.image = photo_seg  # păstrează referința
        label_s.pack(side='left', padx=10, pady=10)

if __name__ == '__main__':
    root = tk.Tk()
    app = CrackSegApp(root)
    root.mainloop()
