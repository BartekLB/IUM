"""Niezależna ewaluacja zapisanego modelu dla laboratorium 3."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from data_prep import load_data, split_data


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def required_setting(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Brak wymaganej zmiennej środowiskowej: {name}")
    return value.strip()


def evaluate_saved_model() -> tuple[float, str, Path]:
    """Wczytaj artefakt modelu, oceń holdout i zapisz raport tekstowy."""
    model_path = ROOT / required_setting("MODEL_PATH")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {model_path}. Najpierw uruchom train.py."
        )

    frame = load_data(required_setting("DATA_PATH"))
    _, x_test, _, y_test = split_data(
        frame,
        float(required_setting("TEST_SIZE")),
        int(required_setting("RANDOM_SEED")),
        required_setting("TARGET_COLUMN"),
    )
    # Model jest wyłącznie wczytywany; ewaluacja nie uruchamia treningu.
    model = joblib.load(model_path)
    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, digits=4, zero_division=0)

    results_path = ROOT / required_setting("RESULTS_PATH")
    output = f"Accuracy: {accuracy:.4f}\n\nClassification report:\n{report}"
    results_path.write_text(output, encoding="utf-8")
    return accuracy, report, results_path


def main() -> None:
    accuracy, _, results_path = evaluate_saved_model()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Wyniki zapisano w: {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
