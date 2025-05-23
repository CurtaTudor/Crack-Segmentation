import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import torch
import segmentation_models_pytorch as smp
from safetensors.torch import load_file
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
        root.title("Crack Detection")
        root.geometry("900x600")
        root.minsize(600, 400)

        # Configurații modele disponibile
        self.model_configs = {
            'Segformer Model': {
                'type': 'segformer',
                'path': 'out/outputs/model_iou'
            },
            'Segformer Pothole Model': {
                'type': 'segformer',
                'path': 'out/outputs_pot/model_iou'
            },
            'UNet Model': {
                'type': 'unet',
                'config': 'out/outputs_unet/final_model/config.json',
                'weights': 'out/outputs_unet/final_model/model.safetensors'
            }
        }

        # Încarcă modelul implicit
        self.selected_model = tk.StringVar()
        self.selected_model.set('Segformer Model')
        self.load_model('Segformer Model')

        # Frame pentru butoane
        buttons_frame = tk.Frame(root)
        buttons_frame.pack(pady=10)

        # Dropdown pentru selecția modelului
        model_menu = tk.OptionMenu(
            buttons_frame,
            self.selected_model,
            *self.model_configs.keys(),
            command=self.change_model
        )
        model_menu.pack(side='left', padx=5)

        # Buton pentru selectarea imaginii
        btn_img = tk.Button(buttons_frame, text="Select Image", command=self.select_image)
        btn_img.pack(side='left', padx=5)

        # Buton pentru selectarea videoclipului
        btn_vid = tk.Button(buttons_frame, text="Select Video", command=self.select_video)
        btn_vid.pack(side='left', padx=5)

        # Frame pentru afișarea imaginii originale
        frame = tk.Frame(root)
        frame.pack(expand=True, fill='both')
        self.label_orig = tk.Label(frame, compound='top')
        self.label_orig.pack(side='left', padx=10, pady=10, expand=True)

    def load_model(self, selection):
        cfg = self.model_configs[selection]
        if cfg['type'] == 'segformer':
            # SegFormer
            self.extractor = SegformerFeatureExtractor()
            self.model = SegformerForSemanticSegmentation.from_pretrained(cfg['path'])
            self.model.to(DEVICE).eval()
        else:
            # UNet
            self.extractor = None
            # Încarcă configurația UNet
            with open(cfg['config'], 'r') as f:
                js = json.load(f)
            # Determină numărul de clase
            if 'id2label' in js:
                num_classes = len(js['id2label'])
            elif 'classes' in js and isinstance(js['classes'], list):
                num_classes = len(js['classes'])
            elif 'classes' in js and isinstance(js['classes'], int):
                num_classes = js['classes']
            elif 'num_labels' in js:
                num_classes = js['num_labels']
            else:
                raise ValueError("config.json trebuie să conțină 'id2label', 'classes' sau 'num_labels'")
            encoder_name = js.get('encoder_name', 'resnet34')
            encoder_weights = js.get('encoder_weights', 'imagenet')
            # Instanțiază UNet
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
                activation=None
            )
            # Încarcă greutăți
            state_dict = load_file(cfg['weights'])
            model.load_state_dict(state_dict)
            model.to(DEVICE).eval()
            self.model = model

    def change_model(self, selection):
        self.load_model(selection)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return

        image_bgr = cv2.imread(path)
        if IMGSZ:
            image_bgr = cv2.resize(image_bgr, IMGSZ)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Inferență
        if self.extractor is not None:
            # SegFormer
            labels = predict(self.model, self.extractor, image_rgb, DEVICE)
            labels_np = labels.cpu()
        else:
            # UNet
            inp = torch.from_numpy(image_rgb.transpose(2,0,1)).float() / 255.0
            inp = inp.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.model(inp)
            labels_np = logits.argmax(dim=1).squeeze().cpu().numpy()

        seg_map = draw_segmentation_map(labels_np, LABEL_COLORS_LIST)
        output = image_overlay(image_rgb, seg_map)

        self.display_result(image_rgb, output)

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        rotation = get_rotation(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = int(1000 / fps)

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
            MAX_W, MAX_H = 1568, 1088
            h, w = frame.shape[:2]
            if w > MAX_W or h > MAX_H:
                frame = cv2.resize(frame, (MAX_W, MAX_H))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inferență pe fiecare cadru
            if self.extractor is not None:
                labels = predict(self.model, self.extractor, image_rgb, DEVICE)
                labels_np = labels.cpu()
            else:
                inp = torch.from_numpy(image_rgb.transpose(2,0,1)).float() / 255.0
                inp = inp.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = self.model(inp)
                labels_np = logits.argmax(dim=1).squeeze().cpu().numpy()

            seg_map = draw_segmentation_map(labels_np, LABEL_COLORS_LIST)
            output = image_overlay(image_rgb, seg_map)

            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(output_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label_vid.imgtk = imgtk
            label_vid.config(image=imgtk)
            label_vid.after(delay, update_frame)

        update_frame()

    def display_result(self, orig_np, output_bgr_np):
        top = tk.Toplevel(self.root)
        top.title("Original & Segmentation Result")
        top.geometry("1124x640")

        orig = Image.fromarray(orig_np).resize((520, 520))
        photo_orig = ImageTk.PhotoImage(orig)
        label_o = tk.Label(top, image=photo_orig, text="Original", compound='top')
        label_o.image = photo_orig
        label_o.pack(side='left', padx=10, pady=10)

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
