import json

import pytest
import torch
from PIL import Image
from transformers import ConvNextConfig, ConvNextForImageClassification

from src.modeling import build_processor
from src.predict import THRESHOLD_FILE, Classifier, predict

LABELS = ["arroz", "feijão", "farofa", "salada"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """ConvNeXt minúsculo com pesos aleatórios: testa o caminho de código, sem download."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("modelo")
    config = ConvNextConfig(
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    ConvNextForImageClassification(config).save_pretrained(path)
    build_processor().save_pretrained(path)
    (path / THRESHOLD_FILE).write_text(json.dumps({"limiar": 0.0}))
    return path


def test_predict_returns_class_confidence_and_top3(model_dir):
    out = predict(Image.new("RGB", (300, 180), "orange"), model=str(model_dir))
    assert out["classe"] in LABELS and out["rejeitado"] is False
    assert out["classe"] == out["top3"][0]["classe"]
    assert out["confianca"] == out["top3"][0]["confianca"]
    assert len(out["top3"]) == 3
    confs = [t["confianca"] for t in out["top3"]]
    assert confs == sorted(confs, reverse=True) and 0 < sum(confs) <= 1 + 1e-6


def test_predict_accepts_path_and_non_rgb_images(model_dir, tmp_path):
    path = tmp_path / "cinza.png"
    Image.new("L", (64, 64), 128).save(path)
    assert predict(path, model=str(model_dir))["classe"] in LABELS


def test_low_confidence_is_rejected(model_dir):
    out = Classifier(str(model_dir), threshold=1.01).predict(Image.new("RGB", (64, 64)))
    assert out["rejeitado"] is True and out["classe"] is None
    assert out["top3"][0]["confianca"] == out["confianca"]


def test_local_model_without_threshold_file_fails_clearly(model_dir, tmp_path):
    for f in model_dir.iterdir():
        if f.name != THRESHOLD_FILE:
            (tmp_path / f.name).write_bytes(f.read_bytes())
    with pytest.raises(FileNotFoundError, match="limiar"):
        Classifier(str(tmp_path))
