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

        # Frame pentru a afișa imaginile side-by-side
        frame = tk.Frame(root)
        frame.pack()
        self.label_orig = tk.Label(frame)
        self.label_orig.pack(side='left', padx=5, pady=5)
        self.label_pred = tk.Label(frame)
        self.label_pred.pack(side='right', padx=5, pady=5)

        # Label-uri pentru imagini cu text dedesubt
        self.label_orig = tk.Label(frame, compound='top')
        self.label_orig.pack(side='left', padx=10, pady=10, expand=True)
        self.label_pred = tk.Label(frame, compound='top')
        self.label_pred.pack(side='right', padx=10, pady=10, expand=True)

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
        output = image_overlay(image_bgr, seg_map)

        # Afișează originalul și segmentarea
        self.display(image_rgb, output)

    def display(self, orig_np, seg_np):
        # Convertire la PIL
        orig = Image.fromarray(orig_np)
        seg = Image.fromarray(seg_np)
        # Redimensionare pentru afișare
        orig = orig.resize((400, 400))
        seg = seg.resize((400, 400))

        # Transformare în PhotoImage și afișare
        self.photo_orig = ImageTk.PhotoImage(orig)
        self.photo_seg = ImageTk.PhotoImage(seg)
        self.label_orig.config(image=self.photo_orig, text="Original")
        self.label_pred.config(image=self.photo_seg, text="Result")

if __name__ == '__main__':
    root = tk.Tk()
    app = CrackSegApp(root)
    root.mainloop()
