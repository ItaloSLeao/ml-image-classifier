"""Inferência com rejeição por baixa confiança.

Uso em código:
    from src.predict import predict
    predict("prato.jpg")
    # {'classe': 'arroz', 'confianca': 0.97, 'top3': [...], 'rejeitado': False}

Uso em linha de comando:
    python -m src.predict prato.jpg --model checkpoints/convnext-tiny-finetune

Quando a confiança fica abaixo do limiar calibrado na validação (limiar.json, gravado
por src/evaluate.py), a resposta vem com 'classe': None e 'rejeitado': True. O limiar
mede baixa confiança; ele não é um detector de "não é comida" (ver docs/decisoes.md).
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_MODEL = "aitalo/convnext-tiny-comida-brasileira"
THRESHOLD_FILE = "limiar.json"


def _load_threshold(model_id: str) -> float:
    local = Path(model_id) / THRESHOLD_FILE
    if local.exists():
        path = local
    elif Path(model_id).exists():
        raise FileNotFoundError(f"{local} não existe: rode src.evaluate para calibrar o limiar")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, THRESHOLD_FILE)
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["limiar"])


class Classifier:
    def __init__(self, model_id: str = DEFAULT_MODEL, threshold: float | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.threshold = _load_threshold(model_id) if threshold is None else threshold

    @torch.no_grad()
    def predict(self, image) -> dict:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        pixels = self.processor(image.convert("RGB"), return_tensors="pt")["pixel_values"].to(self.device)
        probs = self.model(pixel_values=pixels).logits[0].float().softmax(-1).cpu()
        conf, ids = probs.topk(min(3, len(probs)))
        id2label = self.model.config.id2label
        top3 = [{"classe": id2label[int(i)], "confianca": float(c)} for c, i in zip(conf, ids)]
        rejected = top3[0]["confianca"] < self.threshold
        return {
            "classe": None if rejected else top3[0]["classe"],
            "confianca": top3[0]["confianca"],
            "top3": top3,
            "rejeitado": rejected,
        }


@lru_cache(maxsize=2)
def _classifier(model_id: str) -> Classifier:
    return Classifier(model_id)


def predict(image, model: str = DEFAULT_MODEL) -> dict:
    """Classifica uma imagem (caminho ou PIL.Image). O modelo fica em cache entre chamadas."""
    return _classifier(str(model)).predict(image)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("image")
    p.add_argument("--model", default=DEFAULT_MODEL, help="pasta local ou repositório no Hugging Face Hub")
    args = p.parse_args(argv)
    print(json.dumps(predict(args.image, args.model), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
