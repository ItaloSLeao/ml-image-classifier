"""Peças compartilhadas por treino, avaliação e inferência."""

from __future__ import annotations

import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor

BACKBONE = "facebook/convnext-tiny-224"
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def build_processor() -> ViTImageProcessor:
    """Redimensiona direto para 224x224, sem corte central.

    Os recortes já estão justos no alimento, então o corte central padrão do ConvNeXt
    jogaria fora as bordas. O mesmo processador é salvo com o modelo e usado na
    validação, no teste e na inferência.
    """
    return ViTImageProcessor(
        do_resize=True,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        resample=3,  # bicúbica
        do_rescale=True,
        do_normalize=True,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
    )


def build_model(labels: list[str], backbone: str = BACKBONE):
    return AutoModelForImageClassification.from_pretrained(
        backbone,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        ignore_mismatched_sizes=True,  # troca a cabeça de 1000 classes do ImageNet
    )


def eval_transform(processor):
    return lambda image: processor(image, return_tensors="pt")["pixel_values"][0]


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_probs(model, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Probabilidades e rótulos de um DataLoader inteiro."""
    model.eval()
    probs, targets = [], []
    for pixels, target in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(pixel_values=pixels.to(device)).logits
        probs.append(logits.float().softmax(-1).cpu())
        targets.append(target)
    return torch.cat(probs), torch.cat(targets)
