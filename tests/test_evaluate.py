import json

import numpy as np

from src import evaluate, report

LABELS = ["a", "b", "c", "d"]


def test_compute_metrics_on_known_case():
    probs = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.8, 0.1], [0.4, 0.3, 0.2, 0.1]])
    targets = np.array([0, 0, 2, 3])
    m = evaluate.compute_metrics(probs, targets, LABELS)
    assert m["acuracia"] == 0.5
    assert m["acuracia_top3"] == 0.5  # o exemplo 1 tem a classe certa em 4º; o 3 também
    assert m["por_classe"]["c"]["f1"] == 1.0
    assert m["por_classe"]["b"]["suporte"] == 0
    assert np.array(m["matriz_confusao"]).sum() == 4


def test_majority_baseline_always_predicts_most_frequent_class():
    train = [{"label": "b"}] * 5 + [{"label": "a"}] * 3 + [{"label": "c"}]
    probs = evaluate.majority_probs(train, LABELS, 6)
    targets = np.array([0, 1, 1, 2, 3, 3])
    m = evaluate.compute_metrics(probs, targets, LABELS)
    assert m["acuracia"] == 2 / 6
    assert m["acuracia_top3"] == 4 / 6


def test_threshold_keeps_most_correct_predictions():
    rng = np.random.default_rng(0)
    n = 400
    targets = rng.integers(0, 4, n)
    conf = rng.uniform(0.3, 1.0, n)
    correct = rng.uniform(size=n) < conf  # mais confiança, mais acerto
    pred = np.where(correct, targets, (targets + 1) % 4)
    probs = np.full((n, 4), 0.0)
    probs[np.arange(n), pred] = conf
    probs += (1 - conf)[:, None] / 4 * 0.99
    cal = evaluate.calibrate_threshold(probs, targets)
    stats = cal["validacao"]
    assert 0.03 <= stats["acertos_rejeitados"] <= 0.07
    assert stats["erros_rejeitados"] > stats["acertos_rejeitados"]
    assert evaluate.rejection_stats(probs, targets, 0.0)["cobertura"] == 1.0


def test_report_marks_missing_runs_as_pending(tmp_path):
    result = {"nome": evaluate.MAJORITY_NAME, "acuracia": 0.125, "f1_macro": 0.01, "acuracia_top3": 0.3, "n_teste": 8}
    (tmp_path / f"{evaluate.MAJORITY_NAME}.json").write_text(json.dumps(result))
    table = report.comparison_table(tmp_path)
    assert "| Classe majoritária | 12,5% | 1,0% | 30,0% | 8 |" in table
    assert table.count("pendente") == 8
    page = tmp_path / "README.md"
    page.write_text(f"antes\n{report.START}\nvelho\n{report.END}\ndepois")
    assert report.inject(page, table)
    assert "velho" not in page.read_text() and "12,5%" in page.read_text()
