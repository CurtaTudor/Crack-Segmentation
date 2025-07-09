from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp

def segformer_model(classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b1',
        num_labels=len(classes),
    )
    return model

def unet_model(classes, encoder_name='resnet34', encoder_weights='imagenet'):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=len(classes),
        activation=None  # logits
    )
    return model