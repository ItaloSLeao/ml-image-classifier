"""Fine-tuning do ConvNeXt-Tiny nos recortes de alimentos.

Uso:
    python -m src.train --name convnext-tiny-finetune
    python -m src.train --name convnext-tiny-linear-probe --linear-probe

Com --linear-probe o backbone fica congelado e só a cabeça linear é treinada: é o
baseline "mesmo backbone, sem fine-tuning".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import transformers
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from . import data
from .modeling import (
    BACKBONE,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    build_model,
    build_processor,
    eval_transform,
    pick_device,
    predict_probs,
)

ROOT = data.ROOT
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transform():
    """Aumento leve: os recortes são pequenos e a cor é pista importante da classe."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", required=True, help="nome da rodada (pasta do checkpoint e do log)")
    p.add_argument("--backbone", default=BACKBONE)
    p.add_argument("--linear-probe", action="store_true", help="congela o backbone")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6, help="épocas sem melhora antes de parar")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="padrão: 1e-4 (fine-tuning) ou 1e-3 (cabeça)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=data.SEED)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-train-samples", type=int, default=None, help="subconjunto, para teste rápido")
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    lr = args.lr if args.lr is not None else (1e-3 if args.linear_probe else 1e-4)
    set_seed(args.seed)
    device = pick_device()
    use_amp = device.type == "cuda"

    splits = data.load_splits()
    labels = splits["labels"]
    train_items = splits["splits"]["train"]
    val_items = splits["splits"]["val"]
    if args.max_train_samples:
        train_items = random.Random(args.seed).sample(train_items, args.max_train_samples)
    if args.max_val_samples:
        val_items = random.Random(args.seed).sample(val_items, args.max_val_samples)

    processor = build_processor()
    model = build_model(labels, args.backbone)
    if args.linear_probe:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")
    model.to(device)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=use_amp)
    train_loader = DataLoader(
        data.CropDataset(train_items, labels, train_transform()),
        shuffle=True,
        drop_last=len(train_items) > args.batch_size,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_args,
    )
    val_loader = DataLoader(data.CropDataset(val_items, labels, eval_transform(processor)), **loader_args)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = len(train_loader)  # uma época de aquecimento, depois cosseno

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    out_dir = args.checkpoints_dir / args.name
    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        # com o backbone congelado, o modelo fica em modo de avaliação (sem drop path)
        model.train(not args.linear_probe)
        running, seen = 0.0, 0
        for pixels, target in train_loader:
            pixels, target = pixels.to(device), target.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(pixel_values=pixels).logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(target)
            seen += len(target)

        probs, targets = predict_probs(model, val_loader, device)
        preds = probs.argmax(-1)
        val_acc = (preds == targets).float().mean().item()
        val_f1 = f1_score(targets, preds, average="macro", labels=range(len(labels)), zero_division=0)
        history.append(
            {"epoca": epoch, "loss_treino": running / seen, "acuracia_val": val_acc, "f1_macro_val": val_f1}
        )
        print(f"época {epoch:2d}  loss {running / seen:.4f}  val acc {val_acc:.4f}  val F1 macro {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Parada antecipada: {args.patience} épocas sem melhora.")
                break

    log = {
        "nome": args.name,
        "backbone": args.backbone,
        "modo": "linear-probe" if args.linear_probe else "fine-tuning",
        "hiperparametros": {
            "epocas_max": args.epochs,
            "paciencia": args.patience,
            "batch_size": args.batch_size,
            "lr": lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
            "fp16": use_amp,
        },
        "n_treino": len(train_items),
        "n_val": len(val_items),
        "parametros_treinaveis": sum(p.numel() for p in params),
        "melhor_epoca": best_epoch,
        "melhor_f1_macro_val": best_f1,
        "epocas_rodadas": len(history),
        "tempo_treino_s": round(time.time() - start, 1),
        "dispositivo": torch.cuda.get_device_name(0) if use_amp else "cpu",
        "versoes": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
        },
        "historico": history,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.results_dir / f"treino_{args.name}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Melhor época: {best_epoch} (F1 macro val {best_f1:.4f}). Checkpoint em {out_dir}; log em {log_path}")


if __name__ == "__main__":
    main()
