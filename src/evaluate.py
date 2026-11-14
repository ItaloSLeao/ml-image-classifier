"""Avaliação no conjunto de teste e calibração do limiar de rejeição.

Uso:
    python -m src.evaluate --majority
    python -m src.evaluate --name convnext-tiny-finetune

Grava em results/: <nome>.json, <nome>_confusao.png e <nome>_erros.png. Para modelos
treinados, grava também limiar.json na pasta do checkpoint (usado por src/predict.py).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import data

ROOT = data.ROOT
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
MAJORITY_NAME = "baseline-majoritaria"
KEEP_CORRECT = 0.95  # fração dos acertos de validação que o limiar deve preservar
N_WORST = 12


def compute_metrics(probs: np.ndarray, targets: np.ndarray, labels: list[str]) -> dict:
    preds = probs.argmax(1)
    ids = list(range(len(labels)))
    top3 = np.argsort(-probs, axis=1)[:, :3]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=ids, zero_division=0
    )
    return {
        "n_teste": int(len(targets)),
        "acuracia": float(accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, labels=ids, average="macro", zero_division=0)),
        "acuracia_top3": float(np.mean([t in row for t, row in zip(targets, top3)])),
        "por_classe": {
            label: {
                "precisao": float(precision[i]),
                "revocacao": float(recall[i]),
                "f1": float(f1[i]),
                "suporte": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "matriz_confusao": confusion_matrix(targets, preds, labels=ids).tolist(),
    }


def calibrate_threshold(probs: np.ndarray, targets: np.ndarray, keep_correct: float = KEEP_CORRECT) -> dict:
    """Limiar de confiança escolhido na validação.

    É o quantil (1 - keep_correct) da confiança das predições corretas: aceita cerca de
    95% dos acertos de validação e rejeita o que ficar abaixo disso.
    """
    conf, correct = probs.max(1), probs.argmax(1) == targets
    if not correct.any():
        raise ValueError("Nenhum acerto na validação: não há como calibrar o limiar")
    threshold = float(np.quantile(conf[correct], 1 - keep_correct))
    return {
        "limiar": threshold,
        "metodo": f"quantil {1 - keep_correct:.2f} da confiança dos acertos de validação",
        "validacao": rejection_stats(probs, targets, threshold),
    }


def rejection_stats(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    conf, correct = probs.max(1), probs.argmax(1) == targets
    accepted = conf >= threshold
    wrong = ~correct
    return {
        "n": int(len(targets)),
        "cobertura": float(accepted.mean()),
        "acuracia_nos_aceitos": float(correct[accepted].mean()) if accepted.any() else None,
        "erros_rejeitados": float((~accepted & wrong).sum() / wrong.sum()) if wrong.any() else None,
        "acertos_rejeitados": float((~accepted & correct).sum() / correct.sum()) if correct.any() else None,
    }


def plot_confusion(matrix: list[list[int]], labels: list[str], path: Path, title: str) -> None:
    m = np.array(matrix)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def worst_errors(probs: np.ndarray, targets: np.ndarray, items: list[dict], labels: list[str]) -> list[dict]:
    preds, conf = probs.argmax(1), probs.max(1)
    wrong = np.nonzero(preds != targets)[0]
    order = wrong[np.argsort(-conf[wrong])][:N_WORST]
    return [
        {"path": items[i]["path"], "real": labels[targets[i]], "predito": labels[preds[i]], "confianca": float(conf[i])}
        for i in order
    ]


def plot_errors(errors: list[dict], path: Path, crops_dir: Path = data.CROPS_DIR) -> None:
    if not errors:
        return
    cols = 4
    rows = -(-len(errors) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, e in zip(axes.flat, errors):
        ax.imshow(Image.open(crops_dir / e["path"]).convert("RGB"))
        ax.set_title(f"real: {e['real']}\npredito: {e['predito']} ({e['confianca']:.0%})", fontsize=9)
    fig.suptitle("Erros mais confiantes no teste")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def majority_probs(train_items: list[dict], labels: list[str], n: int) -> np.ndarray:
    """Baseline: prevê sempre as classes mais frequentes do treino, na mesma ordem."""
    counts = Counter(r["label"] for r in train_items)
    prior = np.array([counts[label] for label in labels], dtype=float)
    return np.tile(prior / prior.sum(), (n, 1))


def model_probs(checkpoint: Path, splits: dict, split: str, batch_size: int, workers: int, limit=None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from .modeling import eval_transform, pick_device, predict_probs

    device = pick_device()
    model = AutoModelForImageClassification.from_pretrained(checkpoint).to(device)
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    items = splits["splits"][split][:limit]
    dataset = data.CropDataset(items, splits["labels"], eval_transform(processor))
    probs, targets = predict_probs(model, DataLoader(dataset, batch_size=batch_size, num_workers=workers), device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs.numpy(), targets.numpy(), items


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", help="nome da rodada treinada (pasta em checkpoints/)")
    p.add_argument("--majority", action="store_true", help="avalia o baseline de classe majoritária")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="só os N primeiros exemplos, para teste rápido")
    p.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    if args.majority == bool(args.name):
        p.error("use --majority ou --name, um dos dois")

    splits = data.load_splits()
    labels = splits["labels"]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.majority:
        name = MAJORITY_NAME
        test_items = splits["splits"]["test"]
        targets = np.array([labels.index(r["label"]) for r in test_items])
        probs = majority_probs(splits["splits"]["train"], labels, len(test_items))
        result = {"nome": name, "tipo": "classe majoritária do treino", **compute_metrics(probs, targets, labels)}
    else:
        name = args.name
        checkpoint = args.checkpoints_dir / name
        val_probs, val_targets, _ = model_probs(checkpoint, splits, "val", args.batch_size, args.workers, args.limit)
        probs, targets, test_items = model_probs(checkpoint, splits, "test", args.batch_size, args.workers, args.limit)

        threshold = calibrate_threshold(val_probs, val_targets)
        (checkpoint / "limiar.json").write_text(json.dumps(threshold, ensure_ascii=False, indent=1), encoding="utf-8")

        errors = worst_errors(probs, targets, test_items, labels)
        plot_errors(errors, args.results_dir / f"{name}_erros.png")
        result = {
            "nome": name,
            "tipo": "modelo treinado",
            **compute_metrics(probs, targets, labels),
            "rejeicao": {**threshold, "teste": rejection_stats(probs, targets, threshold["limiar"])},
            "erros_mais_confiantes": errors,
        }

    plot_confusion(result["matriz_confusao"], labels, args.results_dir / f"{name}_confusao.png", f"Matriz de confusão: {name}")
    out = args.results_dir / f"{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"{name}: acurácia {result['acuracia']:.4f}  F1 macro {result['f1_macro']:.4f}  "
        f"top-3 {result['acuracia_top3']:.4f}  (n={result['n_teste']})  ->  {out}"
    )


if __name__ == "__main__":
    main()
