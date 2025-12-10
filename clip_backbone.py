from transformers import CLIPProcessor, CLIPModel
import torch

from config import DEVICE


def load_clip(device: str = DEVICE):
    """Load CLIP image encoder and processor."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor
