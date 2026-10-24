"""Trening i zapis pipeline'u scikit-learn dla laboratorium 3."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_data, split_data


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def required_setting(name: str) -> str:
    """Pobierz wymaganą, niepustą wartość konfiguracji."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Brak wymaganej zmiennej środowiskowej: {name}")
    return value.strip()


def build_model(model_type: str, max_iter: int, classifier_c: float, seed: int) -> Pipeline:
    """Utwórz konfigurowalny pipeline skalowania i klasyfikacji."""
    if model_type != "logistic_regression":
        raise ValueError(f"Nieobsługiwany MODEL_TYPE: {model_type}")
    if max_iter < 1 or classifier_c <= 0:
        raise ValueError("MAX_ITER i CLASSIFIER_C muszą być dodatnie.")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=max_iter, C=classifier_c, random_state=seed),
            ),
        ]
    )


def train_and_save() -> Path:
    """Wczytaj dane, wytrenuj model i zapisz artefakt joblib."""
    # Każdy parametr wykonania pochodzi z .env albo ze środowiska procesu.
    data_path = required_setting("DATA_PATH")
    test_size = float(required_setting("TEST_SIZE"))
    seed = int(required_setting("RANDOM_SEED"))
    target_column = required_setting("TARGET_COLUMN")
    model_path = ROOT / required_setting("MODEL_PATH")

    frame = load_data(data_path)
    x_train, _, y_train, _ = split_data(frame, test_size, seed, target_column)
    model = build_model(
        required_setting("MODEL_TYPE"),
        int(required_setting("MAX_ITER")),
        float(required_setting("CLASSIFIER_C")),
        seed,
    )
    # Pipeline zapobiega dopasowaniu skalera do danych testowych.
    model.fit(x_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def main() -> None:
    model_path = train_and_save()
    print(f"Model zapisano w: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
