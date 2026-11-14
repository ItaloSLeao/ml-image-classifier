"""Monta as tabelas de resultado a partir de results/*.json.

Uso:
    python -m src.report

Grava results/comparacao.md e substitui o trecho entre os marcadores
<!-- RESULTADOS:INICIO --> e <!-- RESULTADOS:FIM --> em README.md e model_card.md.
Rodada sem arquivo em results/ aparece como pendente: nenhum número é escrito à mão.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import data
from .evaluate import MAJORITY_NAME

ROOT = data.ROOT
RESULTS_DIR = ROOT / "results"
FINETUNE_NAME = "convnext-tiny-finetune"
PROBE_NAME = "convnext-tiny-linear-probe"
RUNS = [
    (MAJORITY_NAME, "Classe majoritária"),
    (PROBE_NAME, "ConvNeXt-Tiny congelado + cabeça linear"),
    (FINETUNE_NAME, "ConvNeXt-Tiny com fine-tuning"),
]
START, END = "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIM -->"
PENDING = "pendente"


def _load(results_dir: Path, name: str) -> dict | None:
    path = results_dir / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(value: float | None) -> str:
    return PENDING if value is None else f"{100 * value:.1f}%".replace(".", ",")


def comparison_table(results_dir: Path = RESULTS_DIR) -> str:
    lines = [
        "| Modelo | Acurácia | F1 macro | Acurácia top-3 | Exemplos de teste |",
        "|---|---|---|---|---|",
    ]
    for name, title in RUNS:
        r = _load(results_dir, name)
        if r is None:
            lines.append(f"| {title} | {PENDING} | {PENDING} | {PENDING} | {PENDING} |")
        else:
            lines.append(
                f"| {title} | {_pct(r['acuracia'])} | {_pct(r['f1_macro'])} | {_pct(r['acuracia_top3'])} | {r['n_teste']} |"
            )
    return "\n".join(lines)


def finetune_details(results_dir: Path = RESULTS_DIR) -> str:
    r = _load(results_dir, FINETUNE_NAME)
    if r is None:
        return "Detalhes do modelo com fine-tuning: pendente (aguardando a rodada de treino)."
    parts = ["### F1 por classe (modelo com fine-tuning)", "", "| Classe | Precisão | Revocação | F1 | Exemplos de teste |", "|---|---|---|---|---|"]
    for label, m in sorted(r["por_classe"].items(), key=lambda kv: kv[1]["f1"]):
        parts.append(f"| {label} | {_pct(m['precisao'])} | {_pct(m['revocacao'])} | {_pct(m['f1'])} | {m['suporte']} |")
    labels = list(r["por_classe"])
    pairs = sorted(
        ((n, labels[i], labels[j]) for i, row in enumerate(r["matriz_confusao"]) for j, n in enumerate(row) if i != j and n),
        reverse=True,
    )[:5]
    if pairs:
        parts += ["", "### Confusões mais frequentes no teste", "", "| Classe real | Predita como | Vezes |", "|---|---|---|"]
        parts += [f"| {real} | {pred} | {n} |" for n, real, pred in pairs]
    rej, test, val = r["rejeicao"], r["rejeicao"]["teste"], r["rejeicao"]["validacao"]
    parts += [
        "",
        "### Rejeição por baixa confiança",
        "",
        f"Limiar de confiança: {rej['limiar']:.3f}".replace(".", ",") + f" ({rej['metodo']}).",
        "",
        "| Conjunto | Cobertura | Acurácia nos aceitos | Erros rejeitados | Acertos rejeitados |",
        "|---|---|---|---|---|",
        f"| Validação (n={val['n']}) | {_pct(val['cobertura'])} | {_pct(val['acuracia_nos_aceitos'])} | {_pct(val['erros_rejeitados'])} | {_pct(val['acertos_rejeitados'])} |",
        f"| Teste (n={test['n']}) | {_pct(test['cobertura'])} | {_pct(test['acuracia_nos_aceitos'])} | {_pct(test['erros_rejeitados'])} | {_pct(test['acertos_rejeitados'])} |",
    ]
    log = _load(results_dir, f"treino_{FINETUNE_NAME}")
    if log is not None:
        parts += [
            "",
            "### Treino",
            "",
            f"{log['epocas_rodadas']} épocas rodadas (melhor: {log['melhor_epoca']}), "
            f"{log['tempo_treino_s'] / 60:.1f} min".replace(".", ",")
            + f" em {log['dispositivo']}, fp16 {'ligado' if log['hiperparametros']['fp16'] else 'desligado'}, "
            f"seed {log['hiperparametros']['seed']}, uma única rodada.",
        ]
    return "\n".join(parts)


def build(results_dir: Path = RESULTS_DIR) -> str:
    return comparison_table(results_dir) + "\n\n" + finetune_details(results_dir)


def inject(path: Path, block: str) -> bool:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + ".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        return False
    path.write_text(pattern.sub(lambda _: f"{START}\n{block}\n{END}", text), encoding="utf-8")
    return True


def main() -> None:
    block = build()
    (RESULTS_DIR / "comparacao.md").write_text(block + "\n", encoding="utf-8")
    for name in ("README.md", "model_card.md"):
        path = ROOT / name
        if path.exists() and inject(path, block):
            print(f"Atualizado: {name}")
    print(block)


if __name__ == "__main__":
    main()
