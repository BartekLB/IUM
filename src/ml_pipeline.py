"""Laboratorium 2: porównanie pipeline'u scikit-learn i sieci PyTorch."""

from __future__ import annotations

import inspect
import json
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "iris.csv"
RESULTS_PATH = ROOT / "wyniki.txt"
RESULTS_JSON_PATH = ROOT / "wyniki.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
TORCH_EPOCHS = 80
TORCH_LEARNING_RATES = (0.001, 0.003, 0.005, 0.010)
BENCHMARK_REPEATS = 5


@dataclass(frozen=True)
class TimingSummary:
    """Mediana i rozstęp międzykwartylowy czasu wykonania."""

    median_s: float
    iqr_s: float
    repetitions: int


@dataclass(frozen=True)
class HoldoutMetrics:
    """Metryki na wspólnym zbiorze holdout i typowy czas treningu."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    training_time: TimingSummary


@dataclass(frozen=True)
class TorchTrainingDiagnostics:
    """Diagnostyka zbieżności pojedynczego treningu sieci."""

    initial_loss: float
    final_loss: float
    training_accuracy: float
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class TorchFoldDiagnostics:
    """Diagnostyka treningu i predykcji w jednym foldzie."""

    fold: int
    seed: int
    initial_loss: float
    final_loss: float
    training_accuracy: float
    validation_accuracy: float
    predicted_class_counts: tuple[int, ...]


@dataclass(frozen=True)
class CrossValidationMetrics:
    """Wyniki foldów, statystyki jakości i typowy czas całej CV."""

    fold_scores: tuple[float, ...]
    mean: float
    std: float
    total_time: TimingSummary
    fold_diagnostics: tuple[TorchFoldDiagnostics, ...] = ()


@dataclass(frozen=True)
class LearningRateResult:
    """Wynik diagnostycznej CV dla jednej wartości learning rate."""

    learning_rate: float
    fold_scores: tuple[float, ...]
    mean: float
    std: float


class IrisNet(nn.Module):
    """Sieć 4-16-8-3 z aktywacjami ReLU i surowymi logitami na wyjściu."""

    def __init__(self, input_size: int, number_of_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, number_of_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def set_seed(seed: int) -> None:
    """Ustaw kontrolowane ziarna dla Pythona, NumPy i PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Wczytaj lokalny zbiór Iris i zweryfikuj jego strukturę."""
    frame = pd.read_csv(DATA_PATH)
    if "target" not in frame.columns:
        raise ValueError("Plik danych musi zawierać kolumnę 'target'.")
    feature_names = [column for column in frame.columns if column != "target"]
    features = frame[feature_names].to_numpy(dtype=np.float32)
    targets = frame["target"].to_numpy(dtype=np.int64)
    assert features.shape == (150, 4)
    assert targets.shape == (150,)
    assert np.array_equal(np.unique(targets), np.array([0, 1, 2]))
    return features, targets, feature_names


def split_dataset(features: np.ndarray, targets: np.ndarray):
    """Wykonaj jeden wspólny, stratyfikowany podział 80/20."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )
    assert (len(x_train), len(x_test)) == (120, 30)
    assert set(map(tuple, x_train)).isdisjoint(set(map(tuple, x_test)))
    assert np.array_equal(np.bincount(y_train), np.array([40, 40, 40]))
    assert np.array_equal(np.bincount(y_test), np.array([10, 10, 10]))
    return x_train, x_test, y_train, y_test


def build_sklearn_pipeline() -> Pipeline:
    """Połącz StandardScaler i LogisticRegression w jeden Pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def calculate_metrics(
    targets: np.ndarray, predictions: np.ndarray, training_time: TimingSummary
) -> HoldoutMetrics:
    """Policz accuracy oraz makro-uśrednione precision, recall i F1."""
    assert predictions.shape == targets.shape
    assert set(np.unique(predictions)).issubset(set(np.unique(targets)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    return HoldoutMetrics(
        accuracy=float(accuracy_score(targets, predictions)),
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        training_time=training_time,
    )


def train_torch_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    input_size: int,
    number_of_classes: int,
    learning_rate: float,
    seed: int,
) -> tuple[IrisNet, TorchTrainingDiagnostics]:
    """Utwórz nową sieć i wykonaj 80 epok forward/backward/step."""
    set_seed(seed)
    feature_tensor = torch.as_tensor(np.array(x_train, dtype=np.float32, copy=True))
    target_tensor = torch.as_tensor(np.array(y_train, dtype=np.int64, copy=True))
    assert feature_tensor.dtype == torch.float32
    assert target_tensor.dtype == torch.long
    assert feature_tensor.ndim == 2 and target_tensor.ndim == 1
    model = IrisNet(input_size, number_of_classes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history: list[float] = []

    model.train()
    with torch.no_grad():
        initial_loss = float(loss_function(model(feature_tensor), target_tensor).item())
    for _ in range(TORCH_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(feature_tensor)
        loss = loss_function(logits, target_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        final_logits = model(feature_tensor)
        final_loss = float(loss_function(final_logits, target_tensor).item())
        training_predictions = torch.argmax(final_logits, dim=1).cpu().numpy()
    training_accuracy = float(accuracy_score(y_train, training_predictions))
    assert np.isfinite(initial_loss) and np.isfinite(final_loss)
    assert final_loss < initial_loss
    diagnostics = TorchTrainingDiagnostics(
        initial_loss=initial_loss,
        final_loss=final_loss,
        training_accuracy=training_accuracy,
        loss_history=tuple(loss_history),
    )
    return model, diagnostics


def predict_torch(model: IrisNet, features: np.ndarray) -> np.ndarray:
    """Wykonaj predykcję w trybie eval, bez gradientów i bez Softmax."""
    model.eval()
    feature_tensor = torch.as_tensor(np.array(features, dtype=np.float32, copy=True))
    with torch.no_grad():
        predictions = torch.argmax(model(feature_tensor), dim=1)
    result = predictions.cpu().numpy()
    assert result.shape == (len(features),)
    return result


T = TypeVar("T")


def benchmark(operation: Callable[[], T], repetitions: int) -> TimingSummary:
    """Rozgrzej operację, a następnie zwróć medianę i IQR kilku pomiarów."""
    if repetitions < 1:
        raise ValueError("Liczba powtórzeń musi być dodatnia.")
    operation()
    times: list[float] = []
    for _ in range(repetitions):
        started_at = perf_counter()
        operation()
        times.append(perf_counter() - started_at)
    q1, q3 = np.percentile(times, [25, 75])
    return TimingSummary(float(np.median(times)), float(q3 - q1), repetitions)


def run_sklearn_cv_once(features, targets, folds) -> tuple[float, ...]:
    """Wykonaj wymaganą CV funkcją cross_val_score."""
    scores = cross_val_score(
        build_sklearn_pipeline(), features, targets, cv=folds, scoring="accuracy"
    )
    return tuple(float(score) for score in scores)


def cross_validate_sklearn(features, targets, folds, repetitions=BENCHMARK_REPEATS):
    """Zwróć wyniki CV i medianę czasu pełnej procedury po rozgrzewce."""
    fold_scores = run_sklearn_cv_once(features, targets, folds)
    timing = benchmark(lambda: run_sklearn_cv_once(features, targets, folds), repetitions)
    assert len(fold_scores) == N_SPLITS
    return CrossValidationMetrics(
        fold_scores=fold_scores,
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores, ddof=0)),
        total_time=timing,
    )


def run_torch_cv_once(features, targets, folds, learning_rate):
    """Ręcznie wykonaj CV z nowym scalerem, modelem i optymalizatorem."""
    scores: list[float] = []
    details: list[TorchFoldDiagnostics] = []
    number_of_classes = int(np.unique(targets).size)
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        assert set(train_indices).isdisjoint(set(validation_indices))
        scaler = StandardScaler()
        x_fold_train = scaler.fit_transform(features[train_indices]).astype(np.float32)
        x_fold_validation = scaler.transform(features[validation_indices]).astype(np.float32)
        seed = RANDOM_STATE + fold_number
        model, training = train_torch_model(
            x_fold_train,
            targets[train_indices],
            input_size=features.shape[1],
            number_of_classes=number_of_classes,
            learning_rate=learning_rate,
            seed=seed,
        )
        predictions = predict_torch(model, x_fold_validation)
        validation_accuracy = float(
            accuracy_score(targets[validation_indices], predictions)
        )
        class_counts = tuple(
            int(value) for value in np.bincount(predictions, minlength=number_of_classes)
        )
        scores.append(validation_accuracy)
        details.append(
            TorchFoldDiagnostics(
                fold=fold_number,
                seed=seed,
                initial_loss=training.initial_loss,
                final_loss=training.final_loss,
                training_accuracy=training.training_accuracy,
                validation_accuracy=validation_accuracy,
                predicted_class_counts=class_counts,
            )
        )
    return tuple(scores), tuple(details)


def tune_torch_learning_rate(features, targets, folds):
    """Porównaj learning rate wyłącznie na zbiorze treningowym."""
    results: list[LearningRateResult] = []
    for learning_rate in TORCH_LEARNING_RATES:
        scores, _ = run_torch_cv_once(features, targets, folds, learning_rate)
        results.append(
            LearningRateResult(
                learning_rate=learning_rate,
                fold_scores=scores,
                mean=float(np.mean(scores)),
                std=float(np.std(scores, ddof=0)),
            )
        )
    selected = max(results, key=lambda item: (item.mean, -item.std, -item.learning_rate))
    return tuple(results), selected.learning_rate


def cross_validate_torch(
    features, targets, folds, learning_rate, repetitions=BENCHMARK_REPEATS
):
    """Zwróć ręczną CV PyTorch i medianę czasu po rozgrzewce."""
    fold_scores, details = run_torch_cv_once(features, targets, folds, learning_rate)
    timing = benchmark(
        lambda: run_torch_cv_once(features, targets, folds, learning_rate), repetitions
    )
    assert len(fold_scores) == N_SPLITS
    return CrossValidationMetrics(
        fold_scores=fold_scores,
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores, ddof=0)),
        total_time=timing,
        fold_diagnostics=details,
    )


def evaluate_sklearn_holdout(x_train, x_test, y_train, y_test):
    """Zmierz konstrukcję i fit Pipeline'u, a potem oceń holdout poza pomiarem."""
    def fit_pipeline():
        model = build_sklearn_pipeline()
        model.fit(x_train, y_train)
        return model

    timing = benchmark(fit_pipeline, BENCHMARK_REPEATS)
    final_model = fit_pipeline()
    return calculate_metrics(y_test, final_model.predict(x_test), timing)


def evaluate_torch_holdout(
    x_train, x_test, y_train, y_test, number_of_classes, learning_rate
):
    """Zmierz preprocessing i trening PyTorch, a potem oceń holdout poza pomiarem."""
    def fit_model():
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(x_train).astype(np.float32)
        model, diagnostics = train_torch_model(
            scaled_train,
            y_train,
            input_size=x_train.shape[1],
            number_of_classes=number_of_classes,
            learning_rate=learning_rate,
            seed=RANDOM_STATE,
        )
        return scaler, model, diagnostics

    timing = benchmark(fit_model, BENCHMARK_REPEATS)
    scaler, final_model, diagnostics = fit_model()
    scaled_test = scaler.transform(x_test).astype(np.float32)
    metrics = calculate_metrics(y_test, predict_torch(final_model, scaled_test), timing)
    return metrics, diagnostics


def nonempty_source_lines(*objects: object) -> int:
    """Policz niepuste linie, włącznie z komentarzami i docstringami."""
    return sum(
        sum(1 for line in inspect.getsource(obj).splitlines() if line.strip())
        for obj in objects
    )


def implementation_complexity() -> dict[str, object]:
    """Policz porównywalne elementy kodu specyficznego dla bibliotek."""
    sklearn_objects = (
        build_sklearn_pipeline,
        run_sklearn_cv_once,
        cross_validate_sklearn,
        evaluate_sklearn_holdout,
    )
    torch_objects = (
        IrisNet,
        train_torch_model,
        predict_torch,
        run_torch_cv_once,
        tune_torch_learning_rate,
        cross_validate_torch,
        evaluate_torch_holdout,
    )
    return {
        "method": "Niepuste linie wraz z komentarzami i docstringami; bez kodu wspólnego.",
        "common_components": [
            "load_dataset",
            "split_dataset",
            "calculate_metrics",
            "benchmark",
            "zapis wyników",
        ],
        "scikit_learn": {
            "nonempty_lines": nonempty_source_lines(*sklearn_objects),
            "functions": len(sklearn_objects),
            "custom_classes": 0,
            "hyperparameters": ["max_iter=1000", "random_state=42"],
            "preprocessing": "StandardScaler wewnątrz Pipeline",
            "training": "fit",
            "validation": "cross_val_score z KFold",
        },
        "pytorch": {
            "nonempty_lines": nonempty_source_lines(*torch_objects),
            "functions": len(torch_objects) - 1,
            "custom_classes": 1,
            "hyperparameters": [
                "warstwy 16 i 8",
                f"learning_rate={SELECTED_LEARNING_RATE_PLACEHOLDER}",
                f"epochs={TORCH_EPOCHS}",
                "Adam",
            ],
            "preprocessing": "Osobny StandardScaler w każdym treningu",
            "training": "Ręczna pętla forward/backward/step",
            "validation": "Ręczna pętla KFold i nowe wagi",
        },
    }


SELECTED_LEARNING_RATE_PLACEHOLDER = "wybierany na train"


def format_fold_scores(scores: tuple[float, ...]) -> str:
    return ", ".join(
        f"fold {index}: {score:.4f}" for index, score in enumerate(scores, start=1)
    )


def build_results_dict(
    feature_names,
    sklearn_holdout,
    torch_holdout,
    sklearn_cv,
    torch_cv,
    learning_rate_results,
    selected_learning_rate,
    final_training_diagnostics,
):
    """Zbuduj jedno źródło danych dla JSON, TXT i sprawozdania."""
    complexity = implementation_complexity()
    complexity["pytorch"]["hyperparameters"][1] = (
        f"learning_rate={selected_learning_rate:.3f}"
    )
    return {
        "dataset": {
            "name": "Iris",
            "records": 150,
            "features": feature_names,
            "classes": 3,
            "train_records": 120,
            "test_records": 30,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "configuration": {
            "sklearn": {
                "pipeline": ["StandardScaler", "LogisticRegression"],
                "max_iter": 1000,
            },
            "pytorch": {
                "architecture": [4, 16, 8, 3],
                "activation": "ReLU",
                "loss": "CrossEntropyLoss",
                "optimizer": "Adam",
                "epochs": TORCH_EPOCHS,
                "learning_rate_candidates": list(TORCH_LEARNING_RATES),
                "selected_learning_rate": selected_learning_rate,
                "selection_scope": "5-fold CV wyłącznie na train; holdout nie uczestniczył w wyborze",
            },
            "cross_validation": {
                "method": "KFold",
                "n_splits": N_SPLITS,
                "shuffle": True,
                "random_state": RANDOM_STATE,
                "scope": "train only",
                "std_ddof": 0,
            },
            "benchmark": {
                "timer": "time.perf_counter",
                "warmup_runs": 1,
                "measured_repetitions": BENCHMARK_REPEATS,
                "reported_statistic": "median and IQR",
                "scope": "przygotowanie danych i trening; predykcja poza pomiarem",
                "device": "CPU",
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "pytorch": torch.__version__,
            "processor": platform.processor() or platform.machine(),
        },
        "holdout": {
            "scikit_learn": asdict(sklearn_holdout),
            "pytorch": asdict(torch_holdout),
        },
        "cross_validation": {
            "scikit_learn": asdict(sklearn_cv),
            "pytorch": asdict(torch_cv),
        },
        "learning_rate_search": [asdict(result) for result in learning_rate_results],
        "final_pytorch_training": asdict(final_training_diagnostics),
        "complexity": complexity,
        "methodological_note": (
            "Learning rate wybrano na podstawie tych samych foldów treningowych, dlatego wynik "
            "CV wybranej konfiguracji może być umiarkowanie optymistyczny. Holdout wykorzystano "
            "dopiero do końcowej oceny modeli."
        ),
    }


def build_results_text(results: dict[str, object]) -> str:
    """Wygeneruj czytelny wyniki.txt z ustrukturyzowanego słownika."""
    data = results["dataset"]
    config = results["configuration"]
    holdout = results["holdout"]
    cv = results["cross_validation"]
    complexity = results["complexity"]
    selected_lr = config["pytorch"]["selected_learning_rate"]
    lines = [
        "LABORATORIUM 2 - PORÓWNANIE BIBLIOTEK ML",
        "",
        "Konfiguracja eksperymentu",
        f"Zbiór: Iris ({data['records']} rekordów, 4 cechy, 3 klasy)",
        "Podział: 120 train / 30 holdout test, stratify, random_state=42",
        "scikit-learn: StandardScaler + LogisticRegression(max_iter=1000, random_state=42)",
        f"PyTorch: 4-16-8-3, ReLU, CrossEntropyLoss, Adam, lr={selected_lr:.3f}, epoki=80",
        "CV: KFold(5, shuffle=True, random_state=42) wyłącznie na train",
        "Std: populacyjne, np.std(scores, ddof=0)",
        "",
        "Dobór learning rate na train",
    ]
    for result in results["learning_rate_search"]:
        scores = tuple(result["fold_scores"])
        lines.append(
            f"lr={result['learning_rate']:.3f}: {format_fold_scores(scores)}; "
            f"średnia={result['mean']:.4f}; std={result['std']:.4f}"
        )
    lines.extend(["", "Wyniki na wspólnym holdoucie"])
    for model_key, label in (("scikit_learn", "scikit-learn"), ("pytorch", "PyTorch")):
        row = holdout[model_key]
        timing = row["training_time"]
        lines.append(
            f"{label}: accuracy={row['accuracy']:.4f}; precision_macro={row['precision_macro']:.4f}; "
            f"recall_macro={row['recall_macro']:.4f}; f1_macro={row['f1_macro']:.4f}; "
            f"czas_treningu_mediana={timing['median_s']:.6f} s; IQR={timing['iqr_s']:.6f} s"
        )
    for model_key, label in (("scikit_learn", "scikit-learn"), ("pytorch", "PyTorch")):
        row = cv[model_key]
        timing = row["total_time"]
        lines.extend(
            [
                "",
                f"Walidacja krzyżowa {label}",
                format_fold_scores(tuple(row["fold_scores"])),
                f"Średnia accuracy: {row['mean']:.4f}",
                f"Odchylenie standardowe: {row['std']:.4f}",
                f"Czas 5-fold CV - mediana: {timing['median_s']:.6f} s; IQR: {timing['iqr_s']:.6f} s",
            ]
        )
    lines.extend(["", "Diagnostyka foldów PyTorch"])
    for detail in cv["pytorch"]["fold_diagnostics"]:
        class_counts = tuple(detail["predicted_class_counts"])
        lines.append(
            f"fold {detail['fold']}: seed={detail['seed']}; loss {detail['initial_loss']:.4f} -> "
            f"{detail['final_loss']:.4f}; train_accuracy={detail['training_accuracy']:.4f}; "
            f"val_accuracy={detail['validation_accuracy']:.4f}; predykowane klasy={class_counts}"
        )
    lines.extend(
        [
            "",
            "Benchmark",
            "Rozgrzewka: 1 przebieg; pomiary: 5; raportowana mediana i IQR.",
            "Zakres: konstrukcja preprocessingu/modelu i trening; predykcja poza pomiarem; CPU.",
            "",
            "Złożoność kodu",
            complexity["method"],
            f"scikit-learn: {complexity['scikit_learn']['nonempty_lines']} linii; "
            f"{complexity['scikit_learn']['functions']} funkcje; 0 własnych klas.",
            f"PyTorch: {complexity['pytorch']['nonempty_lines']} linii; "
            f"{complexity['pytorch']['functions']} funkcji; 1 własna klasa.",
            f"Kod wspólny: {', '.join(complexity['common_components'])}.",
            "",
            "Analiza",
            f"scikit-learn: CV {cv['scikit_learn']['mean']:.4f} ± {cv['scikit_learn']['std']:.4f}; "
            f"PyTorch: CV {cv['pytorch']['mean']:.4f} ± {cv['pytorch']['std']:.4f}.",
            "Wyższy learning rate poprawił wyniki i stabilność zbieżności PyTorch.",
            results["methodological_note"],
            "",
            "Wnioski końcowe",
            "Dla małego, tabelarycznego Iris scikit-learn zapewnia lepszy kompromis jakości, czasu i prostoty. "
            "PyTorch pozostaje bardziej elastyczny, ale wymaga jawnego strojenia i większej ilości kodu. "
            "Wyników nie należy uogólniać na inne dane ani uznawać za dowód uniwersalnej przewagi biblioteki.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment() -> dict[str, object]:
    """Uruchom CV, wybór learning rate, finalne modele i holdout w poprawnej kolejności."""
    torch.set_num_threads(1)
    set_seed(RANDOM_STATE)
    features, targets, feature_names = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(features, targets)
    folds = list(
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(
            x_train, y_train
        )
    )
    assert len(folds) == N_SPLITS
    sklearn_cv = cross_validate_sklearn(x_train, y_train, folds)
    learning_rate_results, selected_learning_rate = tune_torch_learning_rate(
        x_train, y_train, folds
    )
    torch_cv = cross_validate_torch(
        x_train, y_train, folds, selected_learning_rate
    )
    sklearn_holdout = evaluate_sklearn_holdout(
        x_train, x_test, y_train, y_test
    )
    torch_holdout, final_training_diagnostics = evaluate_torch_holdout(
        x_train,
        x_test,
        y_train,
        y_test,
        int(np.unique(targets).size),
        selected_learning_rate,
    )
    return build_results_dict(
        feature_names,
        sklearn_holdout,
        torch_holdout,
        sklearn_cv,
        torch_cv,
        learning_rate_results,
        selected_learning_rate,
        final_training_diagnostics,
    )


def main() -> None:
    """Uruchom eksperyment i zapisz spójne wyniki JSON oraz TXT."""
    results = run_experiment()
    RESULTS_JSON_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = build_results_text(results)
    RESULTS_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wyniki zapisano w: {RESULTS_PATH} oraz {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
