"""Dados: download, recorte por máscara e split fixo.

O dataset "Brazilian food images" (Mendeley Data, CC BY 4.0) traz fotos de pratos com
vários alimentos. Cada pasta de classe guarda pares (foto, máscara de um alimento), e a
mesma foto aparece em várias classes. Aqui cada par vira um recorte do alimento, e o
split é feito por foto de origem para que um mesmo prato nunca fique em dois conjuntos.

Uso:
    python -m src.data            # baixa, recorta e grava data/splits.json
    python -m src.data --check    # só confere o split já gravado contra os recortes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_ID = "7n36jtcpv3"
DATASET_VERSION = 2
API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CROPS_DIR = ROOT / "data" / "crops"
SPLITS_PATH = ROOT / "data" / "splits.json"

SEED = 42
MASK_THRESHOLD = 128  # as máscaras são JPEG: binariza para tirar o ruído de compressão
MIN_MASK_FRACTION = 0.001  # máscara menor que 0,1% da foto é tratada como vazia
CROP_MARGIN = 0.05  # folga em volta da caixa da máscara, em fração do lado
MIN_CROP_SIDE = 32

# pasta no dataset -> rótulo em português usado pelo modelo
CLASSES = {
    "beans": "feijão",
    "beef": "carne bovina",
    "beet": "beterraba",
    "cabbage": "repolho roxo",
    "carrot": "cenoura",
    "cassava": "mandioca",
    "chicken": "frango",
    "corn": "milho",
    "egg": "ovo",
    "farofa": "farofa",
    "fries": "batata frita",
    "mashed": "purê",
    "mayo(potato salad)": "maionese de batata",
    "milanesesteak": "bife à milanesa",
    "noodles": "macarrão",
    "pastaredsauce": "macarrão ao molho vermelho",
    "pumpkin": "abóbora",
    "rice": "arroz",
    "salad": "salada",
    "tomato": "tomate",
}
LABELS = sorted(CLASSES.values())


# --------------------------------------------------------------------------- download


def _get_json(url: str):
    return json.loads(_fetch(url))


def _fetch(url: str, attempts: int = 6) -> bytes:
    """GET com novas tentativas: a API da Mendeley devolve 502 de vez em quando."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(3 * (attempt + 1))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _folder_paths(folders: list[dict]) -> dict[str, str]:
    by_id = {f["id"]: f for f in folders}

    def path(folder_id):
        parts = []
        while folder_id in by_id:
            parts.append(by_id[folder_id]["name"])
            folder_id = by_id[folder_id].get("parent_id")
        return "/".join(reversed(parts))

    return {folder_id: path(folder_id) for folder_id in by_id}


def list_remote_files() -> list[dict]:
    """Pares foto/máscara de training_set, a partir da API pública da Mendeley."""
    meta = _get_json(API)
    if meta["version"] != DATASET_VERSION:
        raise RuntimeError(f"Versão inesperada do dataset: {meta['version']}")
    paths = _folder_paths(_get_json(f"{API}/folders/{DATASET_VERSION}"))
    files = []
    for f in meta["files"]:
        folder = paths.get(f["folder_id"], "")
        parts = folder.split("/")
        # o upload original inclui pastas .git e um test_set sem rótulos: ficam de fora
        if len(parts) != 2 or parts[0] != "training_set" or parts[1] not in CLASSES:
            continue
        if not f["filename"].lower().endswith(".jpg"):
            continue
        files.append(
            {
                "relpath": f"{parts[1]}/{f['filename']}",
                "url": f["content_details"]["download_url"],
                "sha256": f["content_details"]["sha256_hash"],
            }
        )
    return files


def _download_one(item: dict, raw_dir: Path) -> None:
    dest = raw_dir / item["relpath"]
    if dest.exists() and _sha256(dest) == item["sha256"]:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        data = _fetch(item["url"])
        if data.startswith(b'{"url"'):
            # às vezes a API responde com um JSON apontando para o arquivo, em vez do arquivo
            data = _fetch(json.loads(data)["url"])
        if hashlib.sha256(data).hexdigest() == item["sha256"]:
            dest.write_bytes(data)
            return
        time.sleep(3 * (attempt + 1))  # sob carga a API devolve corpos que não são o arquivo
    raise OSError(f"Hash diferente do publicado: {item['relpath']}")


def download(raw_dir: Path = RAW_DIR, workers: int = 4) -> int:
    files = list_remote_files()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda item: _download_one(item, raw_dir), files))
    return len(files)


# ---------------------------------------------------------------------------- recorte


def mask_bbox(mask: Image.Image, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Caixa (esq, cima, dir, baixo) da região marcada, com folga; None se vazia."""
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    binary = np.asarray(mask.convert("L")) >= MASK_THRESHOLD
    if binary.mean() < MIN_MASK_FRACTION:
        return None
    ys, xs = np.nonzero(binary)
    left, right, top, bottom = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = int((right - left) * CROP_MARGIN), int((bottom - top) * CROP_MARGIN)
    width, height = size
    box = (max(left - mx, 0), max(top - my, 0), min(right + mx, width), min(bottom + my, height))
    if box[2] - box[0] < MIN_CROP_SIDE or box[3] - box[1] < MIN_CROP_SIDE:
        return None
    return tuple(int(v) for v in box)


def build_crops(raw_dir: Path = RAW_DIR, crops_dir: Path = CROPS_DIR) -> tuple[list[dict], list[str]]:
    """Gera um recorte por par (foto, máscara). Devolve os registros e os pares pulados."""
    records, skipped = [], []
    for folder, label in sorted(CLASSES.items()):
        for mask_path in sorted((raw_dir / folder).glob("*_.jpg")):
            photo_path = mask_path.with_name(mask_path.stem[:-1] + ".jpg")
            name = f"{folder}/{photo_path.name}"
            if not photo_path.exists():
                skipped.append(f"{name}: foto ausente")
                continue
            photo = Image.open(photo_path).convert("RGB")
            box = mask_bbox(Image.open(mask_path), photo.size)
            if box is None:
                skipped.append(f"{name}: máscara vazia ou pequena demais")
                continue
            out = crops_dir / folder / photo_path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            photo.crop(box).save(out, quality=95)
            records.append(
                {
                    "path": out.relative_to(crops_dir).as_posix(),
                    "label": label,
                    # mesma foto de origem = mesmo grupo, mesmo que esteja em outra pasta
                    "group": _sha256(photo_path)[:16],
                }
            )
    return records, skipped


# ------------------------------------------------------------------------------ split


def make_splits(records: list[dict], seed: int = SEED) -> dict:
    """Split estratificado por classe e agrupado por foto: cerca de 70/15/15."""
    from sklearn.model_selection import StratifiedGroupKFold

    records = sorted(records, key=lambda r: r["path"])
    labels = np.array([r["label"] for r in records])
    groups = np.array([r["group"] for r in records])
    index = np.arange(len(records))

    def hold_out(idx: np.ndarray, n_splits: int, split_seed: int):
        folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        keep, held = next(folds.split(idx, labels[idx], groups[idx]))
        return idx[keep], idx[held]

    rest, test = hold_out(index, 7, seed)  # 1/7 ~ 14% para teste
    train, val = hold_out(rest, 6, seed + 1)  # 1/6 do restante ~ 14% para validação
    return {
        "seed": seed,
        "dataset": f"doi:10.17632/{DATASET_ID}.{DATASET_VERSION}",
        "labels": LABELS,
        "splits": {
            name: [records[i] for i in idx]
            for name, idx in (("train", train), ("val", val), ("test", test))
        },
    }


def summarize(splits: dict) -> dict:
    counts = {name: Counter(r["label"] for r in items) for name, items in splits["splits"].items()}
    return {
        "total": {name: len(items) for name, items in splits["splits"].items()},
        "por_classe": {
            label: {name: counts[name].get(label, 0) for name in counts} for label in splits["labels"]
        },
    }


def check_splits(splits: dict, crops_dir: Path | None = CROPS_DIR) -> None:
    """Falha se houver vazamento de foto entre conjuntos ou classe ausente em algum deles."""
    seen = defaultdict(set)
    for name, items in splits["splits"].items():
        for r in items:
            seen[r["group"]].add(name)
            if crops_dir is not None and not (crops_dir / r["path"]).exists():
                raise FileNotFoundError(f"Recorte ausente: {r['path']}")
    leaked = [g for g, names in seen.items() if len(names) > 1]
    if leaked:
        raise ValueError(f"{len(leaked)} fotos de origem aparecem em mais de um conjunto")
    for label, per_split in summarize(splits)["por_classe"].items():
        missing = [name for name, n in per_split.items() if n == 0]
        if missing:
            raise ValueError(f"Classe '{label}' sem exemplos em: {missing}")


def load_splits(path: Path = SPLITS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(raw_dir: Path = RAW_DIR, crops_dir: Path = CROPS_DIR, splits_path: Path = SPLITS_PATH) -> dict:
    """Baixa e recorta. Reusa data/splits.json se existir; só cria um split novo se não houver."""
    n = download(raw_dir)
    print(f"{n} arquivos em {raw_dir}")
    records, skipped = build_crops(raw_dir, crops_dir)
    print(f"{len(records)} recortes em {crops_dir}; {len(skipped)} pares pulados")
    for line in skipped:
        print("  pulado:", line)
    if splits_path.exists():
        splits = load_splits(splits_path)
        print(f"Split existente reutilizado: {splits_path}")
    else:
        splits = make_splits(records)
        splits_path.write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Split gravado em {splits_path}")
    check_splits(splits, crops_dir)
    return splits


# ---------------------------------------------------------------------------- dataset


class CropDataset:
    """Dataset no formato do PyTorch: devolve (tensor da imagem, índice da classe)."""

    def __init__(self, items: list[dict], labels: list[str], transform, crops_dir: Path = CROPS_DIR):
        self.items = items
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.transform = transform
        self.crops_dir = Path(crops_dir)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        item = self.items[i]
        image = Image.open(self.crops_dir / item["path"]).convert("RGB")
        return self.transform(image), self.label_to_id[item["label"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="só valida o split existente")
    args = parser.parse_args()
    if args.check:
        splits = load_splits()
        check_splits(splits)
    else:
        splits = prepare()
    summary = summarize(splits)
    print(json.dumps(summary["total"]))
    for label, per_split in summary["por_classe"].items():
        print(f"  {label:28s} {per_split}")


if __name__ == "__main__":
    main()
