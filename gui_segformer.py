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

        # Frame pentru butoane
        buttons_frame = tk.Frame(root)
        buttons_frame.pack(pady=10)

        # Buton pentru selectarea imaginii
        btn_img = tk.Button(buttons_frame, text="Select Image", command=self.select_image)
        btn_img.pack(side='left', padx=5)

        # Buton pentru selectarea videoclipului
        btn_vid = tk.Button(buttons_frame, text="Select Video", command=self.select_video)
        btn_vid.pack(side='left', padx=5)

        # Frame pentru a afișa imaginea originală în fereastra principală (opțional)
        frame = tk.Frame(root)
        frame.pack(expand=True, fill='both')
        self.label_orig = tk.Label(frame, compound='top')
        self.label_orig.pack(side='left', padx=10, pady=10, expand=True)

    def select_image(self):
        # Deschide dialogul de fișiere pentru imagini
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return

        # Citește și preprocesează imaginea
        image_bgr = cv2.imread(path)
        if IMGSZ:
            image_bgr = cv2.resize(image_bgr, IMGSZ)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Inferență
        labels = predict(self.model, self.extractor, image_rgb, DEVICE)
        seg_map = draw_segmentation_map(labels.cpu(), LABEL_COLORS_LIST)
        output = image_overlay(image_rgb, seg_map)

        # Afișează original și rezultat în fereastră nouă
        self.display_result(image_rgb, output)

    def select_video(self):
        # Deschide dialogul de fișiere pentru videoclipuri
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return

        # Deschide captura video
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30
        delay = int(1000 / fps)

        # Fereastră nouă pentru video
        top = tk.Toplevel(self.root)
        top.title("Video Segmentation Result")
        label_vid = tk.Label(top)
        label_vid.pack()

        def update_frame():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return

            # Redimensionare și conversie la RGB
            if IMGSZ:
                frame = cv2.resize(frame, IMGSZ)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inferență pe fiecare cadru
            labels = predict(self.model, self.extractor, image_rgb, DEVICE)
            seg_map = draw_segmentation_map(labels.cpu(), LABEL_COLORS_LIST)
            output = image_overlay(image_rgb, seg_map)

            # Pregătire pentru afișare
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(output_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label_vid.imgtk = imgtk
            label_vid.config(image=imgtk)

            # Afișează următorul cadru după delay
            label_vid.after(delay, update_frame)

        # Începe bucla de afișare
        update_frame()

    def display_result(self, orig_np, output_bgr_np):
        # Creează o fereastră nouă pentru imagine
        top = tk.Toplevel(self.root)
        top.title("Original & Segmentation Result")
        top.geometry("840x440")

        # Original
        orig = Image.fromarray(orig_np).resize((400, 400))
        photo_orig = ImageTk.PhotoImage(orig)
        label_o = tk.Label(top, image=photo_orig, text="Original", compound='top')
        label_o.image = photo_orig
        label_o.pack(side='left', padx=10, pady=10)

        # Rezultat segmentare
        seg_rgb = cv2.cvtColor(output_bgr_np, cv2.COLOR_BGR2RGB)
        seg = Image.fromarray(seg_rgb).resize((400, 400))
        photo_seg = ImageTk.PhotoImage(seg)
        label_s = tk.Label(top, image=photo_seg, text="Result", compound='top')
        label_s.image = photo_seg
        label_s.pack(side='left', padx=10, pady=10)

if __name__ == '__main__':
    root = tk.Tk()
    app = CrackSegApp(root)
    root.mainloop()
