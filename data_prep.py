"""Wczytywanie i podział danych dla laboratorium 3."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def load_data(path: str | Path) -> pd.DataFrame:
    """Wczytaj niepusty plik CSV i zwróć DataFrame."""
    # Ścieżki względne są interpretowane zawsze względem katalogu projektu.
    data_path = Path(path).expanduser()
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {data_path}")

    frame = pd.read_csv(data_path)
    if frame.empty:
        raise ValueError("Zbiór danych jest pusty.")
    return frame


def split_data(
    frame: pd.DataFrame,
    test_size: float,
    seed: int,
    target_column: str = "target",
):
    """Podziel dane na X_train, X_test, y_train i y_test bez efektów ubocznych."""
    # Walidacja wejścia daje czytelny błąd przed wywołaniem scikit-learn.
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dane wejściowe muszą być obiektem pandas.DataFrame.")
    if frame.empty:
        raise ValueError("Nie można podzielić pustego zbioru danych.")
    if target_column not in frame.columns:
        raise ValueError(f"Brak kolumny docelowej: {target_column}")
    if not 0 < test_size < 1:
        raise ValueError("test_size musi należeć do przedziału (0, 1).")

    features = frame.drop(columns=[target_column])
    target = frame[target_column]
    if features.empty:
        raise ValueError("Zbiór nie zawiera kolumn cech.")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Zbiór danych zawiera brakujące wartości.")

    # Stratyfikacja zachowuje proporcje klas w obu częściach zbioru.
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )


def _required_setting(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Brak wymaganej zmiennej środowiskowej: {name}")
    return value.strip()


def main() -> None:
    """Sprawdź przygotowanie danych z konfiguracji .env."""
    frame = load_data(_required_setting("DATA_PATH"))
    split = split_data(
        frame,
        test_size=float(_required_setting("TEST_SIZE")),
        seed=int(_required_setting("RANDOM_SEED")),
        target_column=_required_setting("TARGET_COLUMN"),
    )
    x_train, x_test, _, _ = split
    print(f"Wczytano {len(frame)} rekordów; train={len(x_train)}, test={len(x_test)}")


if __name__ == "__main__":
    main()
