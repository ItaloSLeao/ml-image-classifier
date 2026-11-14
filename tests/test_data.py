import numpy as np
import pytest
from PIL import Image

from src import data


def _mask(box, size=(200, 100)):
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    left, top, right, bottom = box
    arr[top:bottom, left:right] = 255
    return Image.fromarray(arr)


def test_mask_bbox_adds_margin_and_stays_inside_image():
    box = data.mask_bbox(_mask((50, 20, 150, 80)), (200, 100))
    assert box == (45, 17, 155, 83)
    assert data.mask_bbox(_mask((0, 0, 200, 100)), (200, 100)) == (0, 0, 200, 100)


def test_mask_bbox_ignores_jpeg_noise_and_empty_masks():
    noisy = np.asarray(_mask((50, 20, 150, 80))).copy()
    noisy[noisy == 0] = 40  # ruído de compressão abaixo do limiar
    assert data.mask_bbox(Image.fromarray(noisy), (200, 100)) == (45, 17, 155, 83)
    assert data.mask_bbox(_mask((0, 0, 0, 0)), (200, 100)) is None
    assert data.mask_bbox(_mask((10, 10, 20, 20)), (200, 100)) is None  # menor que o lado mínimo


def test_build_crops_pairs_photo_with_mask(tmp_path):
    raw, crops = tmp_path / "raw", tmp_path / "crops"
    (raw / "rice").mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(raw / "rice" / "rice0000.jpg")
    _mask((50, 20, 150, 80)).save(raw / "rice" / "rice0000_.jpg")
    _mask((50, 20, 150, 80)).save(raw / "rice" / "rice0001_.jpg")  # máscara sem foto
    records, skipped = data.build_crops(raw, crops)
    assert [r["path"] for r in records] == ["rice/rice0000.jpg"]
    assert records[0]["label"] == "arroz"
    assert Image.open(crops / "rice" / "rice0000.jpg").size == (110, 66)
    assert len(skipped) == 1


def _records(n_groups=120, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for g in range(n_groups):  # cada foto aparece em 1 a 3 classes, como no dataset real
        for label in rng.choice(data.LABELS[:5], size=rng.integers(1, 4), replace=False):
            records.append({"path": f"{label}/{g}.jpg", "label": str(label), "group": f"g{g}"})
    return records


def test_make_splits_is_deterministic_and_has_no_group_leak():
    records = _records()
    a, b = data.make_splits(records), data.make_splits(list(reversed(records)))
    assert a == b
    assert sum(len(v) for v in a["splits"].values()) == len(records)
    groups = {name: {r["group"] for r in items} for name, items in a["splits"].items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_check_splits_detects_leak():
    splits = data.make_splits(_records())
    splits["labels"] = data.LABELS[:5]
    data.check_splits(splits, crops_dir=None)
    splits["splits"]["test"].append(splits["splits"]["train"][0])
    with pytest.raises(ValueError, match="mais de um conjunto"):
        data.check_splits(splits, crops_dir=None)


def test_committed_split_is_valid():
    if not data.SPLITS_PATH.exists():
        pytest.skip("data/splits.json ainda não foi gerado")
    splits = data.load_splits()
    assert splits["labels"] == data.LABELS
    assert splits["seed"] == data.SEED
    data.check_splits(splits, crops_dir=None)
