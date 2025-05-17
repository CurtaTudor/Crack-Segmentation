# python infer_image.py --input input/train/images/ --device cpu --imgsz 448 448

import argparse
import os
import glob
import json

import cv2
import torch
import segmentation_models_pytorch as smp
from safetensors.torch import load_file

from config import VIS_LABEL_MAP as LABEL_COLORS_LIST                   # paleta de culori pentru vizualizare
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
    parser.add_argument('--input',  default='input/inference_data/images',
                        help='director cu imagini de inferență')
    parser.add_argument('--config', default='out/outputs/model_iou/config.json',
                        help='calea către config.json')
    parser.add_argument('--weights',default='out/outputs/model_iou/model.safetensors',
                        help='calea către model.safetensors')
    parser.add_argument('--device', default='cuda:0',
                        help='cuda:0 sau cpu')
    parser.add_argument('--imgsz', nargs=2, type=int, default=None,
                        help='resize: --imgsz LĂȚIME ÎNĂLȚIME')
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

    # 1) Încărcăm state_dict pe CPU (fără a specifica device la load_file)
    state_dict = load_file(args.weights)
    # 2) Îl punem în model
    model.load_state_dict(state_dict)
    # 3) Mutăm modelul pe dispozitivul ales
    model.to(device)
    model.eval()

    out_dir = 'out/outputs/inference_results_image_unet'
    os.makedirs(out_dir, exist_ok=True)

    for image_path in glob.glob(os.path.join(args.input, '*')):
        # 1) Citire și resize
        img_bgr = cv2.imread(image_path)
        if args.imgsz is not None:
            img_bgr = cv2.resize(img_bgr, tuple(args.imgsz))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2) Preprocesare CHW + normalizare
        inp = torch.from_numpy(img_rgb.transpose(2,0,1)).float() / 255.0
        inp = inp.unsqueeze(0).to(device)  # batch + mutare pe device

        # 3) Inferență
        with torch.no_grad():
            logits = model(inp)  # [1, num_classes, H, W]

        # 4) Argmax și vizualizare
        labels = logits.argmax(dim=1).squeeze().cpu().numpy()
        seg_map = draw_segmentation_map(labels, LABEL_COLORS_LIST)
        output = image_overlay(img_rgb, seg_map)

        # 5) Afișare și salvare
        cv2.imshow('UNet Inference', output)
        cv2.waitKey(1)

        fname = os.path.basename(image_path)
        cv2.imwrite(os.path.join(out_dir, f'unet_{fname}'), output)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
