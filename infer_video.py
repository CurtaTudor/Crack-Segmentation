# python infer_video.py --input input/inference_data/videos/11_Trim.mp4 --device cpu --imgsz 448 448

import argparse
import os
import json

import cv2
import torch
import segmentation_models_pytorch as smp
from safetensors.torch import load_file

from config import VIS_LABEL_MAP as LABEL_COLORS_LIST               # paleta de culori pentru vizualizare
from utils import draw_segmentation_map, image_overlay   # funcții de vizualizare

def load_config(config_path):
    """
    Din config.json aflăm:
     - numărul de clase (din id2label sau din câmpul classes/int sau num_labels)
     - encoder_name și encoder_weights (dacă există)
    """
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    if "id2label" in cfg:
        num_classes = len(cfg["id2label"])
    elif "classes" in cfg and isinstance(cfg["classes"], list):
        num_classes = len(cfg["classes"])
    elif "classes" in cfg and isinstance(cfg["classes"], int):
        num_classes = cfg["classes"]
    elif "num_labels" in cfg:
        num_classes = cfg["num_labels"]
    else:
        raise ValueError(
            "config.json trebuie să conțină 'id2label', 'classes' (listă sau int) sau 'num_labels'"
        )

    encoder_name    = cfg.get("encoder_name", "resnet34")
    encoder_weights = cfg.get("encoder_weights", "imagenet")
    return num_classes, encoder_name, encoder_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True,
        help='fișierul video de intrare pentru inferență'
    )
    parser.add_argument(
        '--output',
        default='out/outputs_unet/video_inference',
        help='director de ieșire pentru fișierul video cu overlay'
    )
    parser.add_argument(
        '--config',
        default='out/outputs_unet/final_model/config.json',
        help='calea către config.json'
    )
    parser.add_argument(
        '--weights',
        default='out/outputs_unet/final_model/model.safetensors',
        help='calea către model.safetensors'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='cuda:0 sau cpu'
    )
    parser.add_argument(
        '--imgsz',
        nargs=2,
        type=int,
        default=None,
        help='resize cadru: --imgsz LĂȚIME ÎNĂȚIME'
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Încarcăm din config: câte clase și ce encoder
    num_classes, encoder_name, encoder_weights = load_config(args.config)

    # Instanțiem UNet direct, cu num_classes ca int
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None
    )

    # 1) Încărcăm state_dict pe CPU
    state_dict = load_file(args.weights)
    model.load_state_dict(state_dict)
    # 2) Mutăm modelul pe dispozitivul ales
    model.to(device)
    model.eval()

    # Pregătim calea de ieșire
    out_arg = args.output
    ext = os.path.splitext(out_arg)[1].lower()
    if ext in {'.mp4', '.avi', '.mov', '.mkv'}:
        output_path = out_arg
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        out_dir = out_arg
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(out_dir, f'{base}_unet.mp4')

    # Deschidem video-ul de intrare
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Nu pot deschide fișierul video {args.input}")

    # Preluăm FPS și dimensiuni cadru
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determinăm dimensiunea de ieșire
    if args.imgsz is not None:
        out_width, out_height = args.imgsz
    else:
        out_width, out_height = width, height

    # Pregătim VideoWriter pentru fișierul de ieșire
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # Deschidem fereastra pentru afișare
    cv2.namedWindow('UNet Inference', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('UNet Inference', out_width, out_height)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Resize dacă e cazul
        if args.imgsz is not None:
            frame_bgr = cv2.resize(frame_bgr, (out_width, out_height))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Preprocesare CHW + normalizare
        inp = torch.from_numpy(frame_rgb.transpose(2,0,1)).float() / 255.0
        inp = inp.unsqueeze(0).to(device)

        # Inferență
        with torch.no_grad():
            logits = model(inp)  # [1, num_classes, H, W]

        # Argmax și vizualizare
        labels   = logits.argmax(dim=1).squeeze().cpu().numpy()
        seg_map  = draw_segmentation_map(labels, LABEL_COLORS_LIST)
        output_bgr = image_overlay(frame_rgb, seg_map)  # already BGR :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

        # Afișăm rezultatul într-o fereastră
        cv2.imshow('UNet Inference', output_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Scriem cadrul în fișierul de ieșire
        writer.write(output_bgr)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Inferență video completă. Rezultatul a fost salvat în {output_path}")

if __name__ == '__main__':
    main()
