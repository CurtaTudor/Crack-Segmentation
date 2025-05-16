import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import subprocess, json
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import draw_segmentation_map, image_overlay, predict

# Dispozitivul pe care rulează modelul ("cuda:0" sau "cpu")
DEVICE = 'cpu'
# Dimensiune fixă la care redimensionăm (sau None pentru original)
IMGSZ = (1568, 1088)

def get_rotation(path):

    angle = 0
    cap_test = cv2.VideoCapture(path)
    ret, frame0 = cap_test.read()
    cap_test.release()
    print(frame0.shape[0], frame0.shape[1])
    if frame0.shape[0] < 1000 and frame0.shape[1] < 1000 and frame0.shape[0] < frame0.shape[1]:
        angle = 90
    return angle

def rotate_frame(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

class CrackSegApp:
    def __init__(self, root):
        self.root = root
        root.title("Crack Detection - SegFormer")
        root.geometry("900x600")
        root.minsize(600, 400)

        # Dicționar de modele disponibile
        self.model_paths = {
            'Segformer Model': 'out/outputs/model_iou',
            'Segformer Pothole Model': 'out/outputs_pot/model_iou'
        }
        # Inițializare extractor și model implicit
        self.extractor = SegformerFeatureExtractor()
        default_path = self.model_paths['Segformer Model']
        self.model = SegformerForSemanticSegmentation.from_pretrained(default_path)
        self.model.to(DEVICE).eval()

        # Frame pentru butoane
        buttons_frame = tk.Frame(root)
        buttons_frame.pack(pady=10)

        # Dropdown pentru selecția modelului
        self.selected_model = tk.StringVar()
        self.selected_model.set('Segformer Model')
        model_menu = tk.OptionMenu(
            buttons_frame,
            self.selected_model,
            *self.model_paths.keys(),
            command=self.change_model
        )
        model_menu.pack(side='left', padx=5)

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

    def change_model(self, selection):
        """
        Callback la schimbarea modelului din dropdown.
        """
        model_path = self.model_paths[selection]
        # Încarcă noul model
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(DEVICE).eval()

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
        rotation = get_rotation(path)
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

            frame = rotate_frame(frame, rotation)
            # Redimensionare și conversie la RGB
            MAX_W, MAX_H = 1568, 1088
            h, w = frame.shape[:2]
            if w > MAX_W or h > MAX_H:
                frame = cv2.resize(frame, (MAX_W, MAX_H))
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
        top.geometry("1124x640")

        # Original
        orig = Image.fromarray(orig_np).resize((520, 520))
        photo_orig = ImageTk.PhotoImage(orig)
        label_o = tk.Label(top, image=photo_orig, text="Original", compound='top')
        label_o.image = photo_orig
        label_o.pack(side='left', padx=10, pady=10)

        # Rezultat segmentare
        seg_rgb = cv2.cvtColor(output_bgr_np, cv2.COLOR_BGR2RGB)
        seg = Image.fromarray(seg_rgb).resize((520, 520))
        photo_seg = ImageTk.PhotoImage(seg)
        label_s = tk.Label(top, image=photo_seg, text="Result", compound='top')
        label_s.image = photo_seg
        label_s.pack(side='left', padx=10, pady=10)

if __name__ == '__main__':
    root = tk.Tk()
    app = CrackSegApp(root)
    root.mainloop()
