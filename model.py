from transformers import SegformerForSemanticSegmentation

def segformer_model(classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b1',
        num_labels=len(classes),
    )
    return model