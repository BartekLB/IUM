"""Reproducible Iris classification pipeline for laboratory 1.

The script loads configuration from .env, explores the Iris data, splits it
into train/test subsets, trains logistic regression, evaluates it, and saves
the metrics to wyniki.txt and the fitted model to models/.
"""

from pathlib import Path
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / os.getenv("DATA_PATH", "data/iris.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


def load_data() -> pd.DataFrame:
    """Read a CSV when available; otherwise create it from sklearn's Iris set."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    iris = load_iris(as_frame=True)
    frame = iris.frame.rename(columns={"target": "target"})
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_PATH, index=False)
    return frame


def main() -> None:
    """Run exploration, training, evaluation, and artifact export."""
    df = load_data()
    feature_columns = [column for column in df.columns if column != "target"]
    X = df[feature_columns]
    y = df["target"]

    # Zadanie 3: eksploracja danych.
    head = df.head().to_string(index=False)
    shape = str(df.shape)
    describe = df.describe().to_string()
    print("Pierwsze wiersze:\n", head)
    print("\nRozmiar zbioru:", shape)
    print("\nStatystyki opisowe:\n", describe)

    # Zadanie 4: losowy, powtarzalny podział oraz klasyfikator.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Zadanie 5: ocena i zapis wyniku.
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["setosa", "versicolor", "virginica"])
    results = (
        "Konfiguracja\n"
        f"DATA_PATH={DATA_PATH.relative_to(ROOT).as_posix()}\n"
        f"TEST_SIZE={TEST_SIZE}\n"
        f"RANDOM_SEED={RANDOM_SEED}\n\n"
        "Eksploracja danych\n"
        f"head():\n{head}\n\nshape: {shape}\n\ndescribe():\n{describe}\n\n"
        "Ocena modelu\n"
        f"accuracy: {accuracy:.4f}\n\nclassification_report:\n{report}"
    )
    (ROOT / "wyniki.txt").write_text(results, encoding="utf-8")

    # Artefakt modelu jest lokalny i może być odtworzony z kodu oraz danych.
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "iris_logistic_regression.joblib")

    print(f"\naccuracy: {accuracy:.4f}")
    print("\nclassification_report:\n", report)
    print(f"\nZapisano wyniki do {ROOT / 'wyniki.txt'}")


if __name__ == "__main__":
    main()

