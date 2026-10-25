"""Testy jednostkowe modułu data_prep."""

from __future__ import annotations

import pandas as pd
import pytest

from data_prep import load_data, split_data
from evaluate import evaluate_saved_model
from train import train_and_save


def sample_frame(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": range(rows),
            "feature_b": [value / 10 for value in range(rows)],
            "target": [value % 2 for value in range(rows)],
        }
    )


def test_load_data_reads_csv(tmp_path):
    source = tmp_path / "sample.csv"
    expected = sample_frame(10)
    expected.to_csv(source, index=False)
    loaded = load_data(source)
    pd.testing.assert_frame_equal(loaded, expected)


def test_split_data_returns_expected_sizes():
    x_train, x_test, y_train, y_test = split_data(sample_frame(), 0.25, 42)
    assert len(x_train) == len(y_train) == 15
    assert len(x_test) == len(y_test) == 5


def test_split_sizes_sum_to_original_size():
    frame = sample_frame()
    x_train, x_test, y_train, y_test = split_data(frame, 0.2, 42)
    assert len(x_train) + len(x_test) == len(frame)
    assert len(y_train) + len(y_test) == len(frame)


def test_split_data_is_reproducible():
    first = split_data(sample_frame(), 0.3, 7)
    second = split_data(sample_frame(), 0.3, 7)
    assert all(left.equals(right) for left, right in zip(first, second))


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (pd.DataFrame(), ValueError),
        (pd.DataFrame({"feature": [1, 2]}), ValueError),
        ("not-a-dataframe", TypeError),
    ],
)
def test_split_data_rejects_invalid_input(frame, error):
    with pytest.raises(error):
        split_data(frame, 0.2, 42)


def test_evaluate_creates_missing_results_directory(tmp_path, monkeypatch):
    data_path = tmp_path / "data" / "sample.csv"
    model_path = tmp_path / "models" / "sample.joblib"
    results_path = tmp_path / "reports" / "nested" / "results.txt"
    data_path.parent.mkdir(parents=True)
    sample_frame(30).to_csv(data_path, index=False)

    settings = {
        "DATA_PATH": str(data_path),
        "TEST_SIZE": "0.2",
        "RANDOM_SEED": "42",
        "TARGET_COLUMN": "target",
        "MODEL_TYPE": "logistic_regression",
        "MAX_ITER": "1000",
        "CLASSIFIER_C": "1.0",
        "MODEL_PATH": str(model_path),
        "RESULTS_PATH": str(results_path),
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)

    train_and_save()
    accuracy, report, written_path = evaluate_saved_model()

    assert written_path == results_path
    assert results_path.is_file()
    assert f"Accuracy: {accuracy:.4f}" in results_path.read_text(encoding="utf-8")
    assert "precision" in report
