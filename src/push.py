"""Publica o modelo no Hugging Face Hub.

Uso:
    HF_TOKEN=... python -m src.push --name convnext-tiny-finetune

O token vem só da variável de ambiente HF_TOKEN (no Kaggle: Add-ons > Secrets).
Envia os pesos, o processador, limiar.json, o model card, os resultados e alguns
recortes de teste como exemplos para o demo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

from . import data
from .predict import DEFAULT_MODEL, THRESHOLD_FILE
from .report import FINETUNE_NAME

ROOT = data.ROOT
N_EXAMPLES = 6


def example_items(splits: dict, n: int = N_EXAMPLES) -> list[dict]:
    """Um recorte de teste por classe, das primeiras n classes em ordem alfabética."""
    chosen = {}
    for item in splits["splits"]["test"]:
        chosen.setdefault(item["label"], item)
    return [chosen[label] for label in sorted(chosen)[:n]]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", default=FINETUNE_NAME, help="rodada em checkpoints/")
    p.add_argument("--repo", default=DEFAULT_MODEL)
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Defina HF_TOKEN no ambiente (no Kaggle: Add-ons > Secrets).")

    checkpoint = ROOT / "checkpoints" / args.name
    results = ROOT / "results"
    required = [checkpoint / "config.json", checkpoint / THRESHOLD_FILE, results / f"{args.name}.json", ROOT / "model_card.md"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Arquivos ausentes (treine, avalie e rode src.report antes): {missing}")

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=checkpoint, commit_message="Pesos, processador e limiar")
    api.upload_file(repo_id=args.repo, path_or_fileobj=ROOT / "model_card.md", path_in_repo="README.md", commit_message="Model card")
    for path in sorted(results.glob(f"{args.name}*")) + [results / "comparacao.md"]:
        if path.exists():
            api.upload_file(repo_id=args.repo, path_or_fileobj=path, path_in_repo=f"results/{path.name}", commit_message=f"Resultado: {path.name}")
    for item in example_items(data.load_splits()):
        api.upload_file(
            repo_id=args.repo,
            path_or_fileobj=data.CROPS_DIR / item["path"],
            path_in_repo=f"exemplos/{Path(item['path']).name}",
            commit_message="Exemplo para o demo",
        )
    print(f"Publicado em https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
