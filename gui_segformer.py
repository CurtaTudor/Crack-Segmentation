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
from Configuration import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import draw_segmentation_map, image_overlay, predict
from patch_model import build_model, predict_on_crops

# Device ("cuda:0" sau "cpu")
DEVICE = 'cpu'
# Image Size
IMGSZ = (1088, 1568)

def get_rotation(path):
    angle = 0
    cap_test = cv2.VideoCapture(path)
    ret, frame0 = cap_test.read()
    cap_test.release()
    if frame0 is not None and frame0.shape[0] < 1000 and frame0.shape[1] < 1000 and frame0.shape[0] < frame0.shape[1]:
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

        self.model_configs = {
            'Segformer Model': {'type': 'segformer', 'path': 'out/outputs/model_iou'},
            'Segformer Pothole Model': {'type': 'segformer', 'path': 'out/outputs_pot/model_iou'},
            'UNet Model': {'type': 'unet', 'config': 'out/outputs_unet/model_iou/config.json', 'weights': 'out/outputs_unet/model_iou/model.safetensors'},
            'ResNet50 Classifier': {
                'type': 'resnet50',
                'model_path': 'out/outputs_resnet/model.pth',
                'crop_h': 32,
                'crop_w': 32
            }
        }

        self.base_extractor = SegformerFeatureExtractor()
        self.base_model = SegformerForSemanticSegmentation.from_pretrained(self.model_configs['Segformer Model']['path'])
        self.base_model.to(DEVICE).eval()
        self.pot_extractor = SegformerFeatureExtractor()
        self.pot_model = SegformerForSemanticSegmentation.from_pretrained(self.model_configs['Segformer Pothole Model']['path'])
        self.pot_model.to(DEVICE).eval()

        self.selected_model = tk.StringVar(value='Segformer Model')
        self.load_model('Segformer Model')

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.OptionMenu(btn_frame, self.selected_model, *self.model_configs.keys(), command=self.change_model).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Select Image", command=self.select_image).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Select Video", command=self.select_video).pack(side='left', padx=5)

        frame = tk.Frame(root)
        frame.pack(expand=True, fill='both')
        self.label_orig = tk.Label(frame, compound='top')
        self.label_orig.pack(side='left', padx=10, pady=10, expand=True)

    def load_model(self, selection):
        cfg = self.model_configs[selection]
        if cfg['type'] == 'segformer':
            self.extractor = (self.base_extractor if selection=='Segformer Model' else self.pot_extractor)
            self.model = (self.base_model if selection=='Segformer Model' else self.pot_model)
        elif cfg['type'] == 'resnet50':
            model = build_model(DEVICE)
            checkpoint = torch.load(cfg['model_path'], map_location=DEVICE)
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            elif isinstance(checkpoint, torch.nn.Module):
                model = checkpoint.to(DEVICE)
            else:
                raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")
            model.to(DEVICE).eval()
            self.model = model
            self.extractor = None
        else:
            with open(cfg['config'], 'r') as f:
                js = json.load(f)
            var = js.get('id2label') or js.get('classes') or js.get('num_labels')
            if isinstance(var, int):
                num_classes = var
            else:
                num_classes = len(var)
            model = smp.Unet(
                encoder_name=js.get('encoder_name', 'resnet34'),
                encoder_weights=js.get('encoder_weights', 'imagenet'),
                in_channels=3,
                classes=num_classes,
                activation=None
            )
            model.load_state_dict(load_file(cfg['weights']))
            model.to(DEVICE).eval()
            self.extractor=None
            self.model=model

    def change_model(self, selection): self.load_model(selection)

    def select_image(self):
        path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp"),("All files","*.*")])
        if not path: return
        img_bgr=cv2.imread(path)
        if IMGSZ: img_bgr=cv2.resize(img_bgr,IMGSZ)
        img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        sel=self.selected_model.get()
        if sel=='Segformer Model':
            lb=predict(self.base_model,self.base_extractor,img_rgb,DEVICE).cpu().numpy()
            lp=predict(self.pot_model,self.pot_extractor,img_rgb,DEVICE).cpu().numpy()
            seg=draw_segmentation_map(lb,LABEL_COLORS_LIST)
            seg[lp>0]=[255,0,0]
        elif sel=='Segformer Pothole Model':
            lp=predict(self.pot_model,self.pot_extractor,img_rgb,DEVICE).cpu().numpy()
            seg=np.zeros_like(img_rgb,dtype=np.uint8)
            seg[lp>0]=[255,0,0]
        elif sel == 'ResNet50 Classifier':
            out_bgr = predict_on_crops(self.model, path,
                                   height=self.model_configs[sel]['crop_h'],
                                   width=self.model_configs[sel]['crop_w'])
            self.display_result(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),out_bgr)
            return            
        else:
            inp=torch.from_numpy(img_rgb.transpose(2,0,1)).float()/255.0
            inp=inp.unsqueeze(0).to(DEVICE)
            lbls=self.model(inp).argmax(1).squeeze().cpu().numpy()
            seg=draw_segmentation_map(lbls,LABEL_COLORS_LIST)
        output=image_overlay(img_rgb,seg)
        self.display_result(img_rgb,output)

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files","*.mp4 *.mov *.avi *.mkv"),("All files","*.*")]
        )
        if not path:
            return

        # dialog pentru salvare rezultat
        save_path = filedialog.asksaveasfilename(
            title="Save result as...",
            defaultextension=".mp4",
            filetypes=[("MP4 Video","*.mp4"),("All files","*.*")]
        )
        if not save_path:
            return

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = int(1000 / fps)

        # variabilă pentru VideoWriter
        self.writer = None

        self.frame_count = 0
        top = tk.Toplevel(self.root)
        top.title("Video Segmentation Result")
        lbl = tk.Label(top)
        lbl.pack()

        def update():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                if self.writer:
                    self.writer.release()
                return

            self.frame_count += 1

            # resize dacă e nevoie
            h, w = frame.shape[:2]
            if w > IMGSZ[0] or h > IMGSZ[1]:
                frame = cv2.resize(frame, IMGSZ)
                h, w = IMGSZ

            # inițializează writer la primul frame
            if self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # inferență
            sel = self.selected_model.get()
            if sel == 'Segformer Model':
                with torch.no_grad():
                    base = predict(self.base_model, self.base_extractor, img_rgb, DEVICE).cpu().numpy()
                    lp=predict(self.pot_model,self.pot_extractor,img_rgb,DEVICE).cpu().numpy()
                seg_map = draw_segmentation_map(base, LABEL_COLORS_LIST)
                seg_map[lp>0]=[255,0,0]
            elif sel == 'Segformer Pothole Model':
                pot = predict(self.pot_model, self.pot_extractor, img_rgb, DEVICE).cpu().numpy()
                seg_map = np.zeros_like(img_rgb, dtype=np.uint8)
                seg_map[pot > 0] = [255, 0, 0]
            else:
                inp = torch.from_numpy(img_rgb.transpose(2,0,1)).float()/255.0
                inp = inp.unsqueeze(0).to(DEVICE)
                lbls = self.model(inp).argmax(1).squeeze().cpu().numpy()
                seg_map = draw_segmentation_map(lbls, LABEL_COLORS_LIST)

            out_bgr = image_overlay(img_rgb, seg_map)
            # scrie frame-ul în fișier
            self.writer.write(out_bgr)

            # afișează în UI (conversie la RGB pentru Tkinter)
            display = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(display)
            imgtk = ImageTk.PhotoImage(im)
            lbl.imgtk = imgtk
            lbl.config(image=imgtk)

            lbl.after(delay, update)

        update()


    def display_result(self, orig_np, output_bgr_np):
        sel = self.selected_model.get()
        if sel != 'ResNet50 Classifier':
            legend = [('Crack', (0, 255, 0)), ('Pothole', (0, 0, 255))]
        else:
            legend = []

        if legend:        
            box_w, box_h = 40, 40
            padding = 20
            font_scale = 1.0
            thickness = 3
    
            start_x = padding
            start_y = padding
            legend_w = box_w + padding + 150
            legend_h = len(legend) * (box_h + padding) + padding - 20
        
            overlay = output_bgr_np.copy()
            cv2.rectangle(overlay,
                        (start_x - padding//2, start_y - padding//2),
                        (start_x + legend_w, start_y + legend_h),
                        (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.7, output_bgr_np, 0.3, 0, output_bgr_np)
    
            cv2.rectangle(output_bgr_np,
                        (start_x - padding//2, start_y - padding//2),
                        (start_x + legend_w, start_y + legend_h),
                        (0, 0, 0), thickness)
    
            for idx, (lbl, col) in enumerate(legend):
                y = start_y + idx * (box_h + padding)
                cv2.rectangle(output_bgr_np,
                            (start_x, y),
                            (start_x + box_w, y + box_h),
                            col, -1)
                cv2.putText(output_bgr_np, lbl,
                            (start_x + box_w + padding, y + box_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        top = tk.Toplevel(self.root)
        top.title("Inference result")
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
