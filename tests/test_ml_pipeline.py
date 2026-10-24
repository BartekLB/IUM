import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml_pipeline import (  # noqa: E402
    DATA_PATH,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_JSON_PATH,
    RESULTS_PATH,
    TORCH_EPOCHS,
    IrisNet,
    TimingSummary,
    build_results_text,
    build_sklearn_pipeline,
    calculate_metrics,
    cross_validate_sklearn,
    cross_validate_torch,
    load_dataset,
    predict_torch,
    run_torch_cv_once,
    split_dataset,
    train_torch_model,
    tune_torch_learning_rate,
)


def prepared_data():
    features, targets, feature_names = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(features, targets)
    folds = list(
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(
            x_train, y_train
        )
    )
    return features, targets, feature_names, x_train, x_test, y_train, y_test, folds


def test_load_dataset_shape_types_and_classes():
    features, targets, names = load_dataset()
    assert DATA_PATH.exists()
    assert features.shape == (150, 4)
    assert targets.shape == (150,)
    assert features.dtype == np.float32
    assert targets.dtype == np.int64
    assert len(names) == 4
    assert np.array_equal(np.unique(targets), np.array([0, 1, 2]))


def test_split_dataset_is_reproducible_balanced_and_disjoint():
    features, targets, _ = load_dataset()
    first = split_dataset(features, targets)
    second = split_dataset(features, targets)
    x_train, x_test, y_train, y_test = first
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert (len(x_train), len(x_test)) == (120, 30)
    assert np.array_equal(np.bincount(y_train), np.array([40, 40, 40]))
    assert np.array_equal(np.bincount(y_test), np.array([10, 10, 10]))
    assert set(map(tuple, x_train)).isdisjoint(set(map(tuple, x_test)))


def test_sklearn_pipeline_contains_scaler_and_classifier():
    pipeline = build_sklearn_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert pipeline.named_steps["classifier"].max_iter == 1000


def test_iris_net_architecture_and_output_shape():
    model = IrisNet(4, 3)
    linear_layers = [layer for layer in model.layers if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (4, 16),
        (16, 8),
        (8, 3),
    ]
    assert not any(isinstance(layer, torch.nn.Softmax) for layer in model.modules())
    assert model(torch.zeros((5, 4), dtype=torch.float32)).shape == (5, 3)


def test_train_and_predict_torch_are_valid():
    _, _, _, x_train, _, y_train, _, _ = prepared_data()
    scaler = StandardScaler().fit(x_train)
    scaled = scaler.transform(x_train).astype(np.float32)
    model, diagnostics = train_torch_model(
        scaled,
        y_train,
        input_size=4,
        number_of_classes=3,
        learning_rate=0.010,
        seed=RANDOM_STATE,
    )
    predictions = predict_torch(model, scaled[:10])
    assert predictions.shape == (10,)
    assert set(predictions).issubset({0, 1, 2})
    assert len(diagnostics.loss_history) == TORCH_EPOCHS
    assert diagnostics.final_loss < diagnostics.initial_loss
    assert 0 <= diagnostics.training_accuracy <= 1


def test_independent_seeds_produce_independent_initializations():
    torch.manual_seed(RANDOM_STATE)
    first = IrisNet(4, 3)
    first_state = [parameter.detach().clone() for parameter in first.parameters()]
    torch.manual_seed(RANDOM_STATE + 1)
    second = IrisNet(4, 3)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first_state, second.parameters())
    )


def test_folds_are_shared_and_disjoint():
    *_, folds = prepared_data()
    assert len(folds) == 5
    assert all(set(train).isdisjoint(set(validation)) for train, validation in folds)
    assert all(len(train) == 96 and len(validation) == 24 for train, validation in folds)


def test_scaler_is_fit_only_on_fold_training_data():
    _, _, _, x_train, _, _, _, folds = prepared_data()
    train_idx, validation_idx = folds[0]
    scaler = StandardScaler().fit(x_train[train_idx])
    assert np.allclose(scaler.mean_, x_train[train_idx].mean(axis=0))
    assert not np.allclose(scaler.mean_, x_train.mean(axis=0))
    assert set(train_idx).isdisjoint(set(validation_idx))


def test_cross_validation_statistics_and_fold_count():
    _, _, _, x_train, _, y_train, _, folds = prepared_data()
    sklearn_cv = cross_validate_sklearn(x_train, y_train, folds, repetitions=1)
    torch_cv = cross_validate_torch(
        x_train, y_train, folds, learning_rate=0.010, repetitions=1
    )
    assert len(sklearn_cv.fold_scores) == len(torch_cv.fold_scores) == 5
    assert np.isclose(sklearn_cv.mean, np.mean(sklearn_cv.fold_scores))
    assert np.isclose(torch_cv.std, np.std(torch_cv.fold_scores, ddof=0))
    assert len(torch_cv.fold_diagnostics) == 5
    assert all(sum(row.predicted_class_counts) == 24 for row in torch_cv.fold_diagnostics)


def test_learning_rate_selection_uses_training_folds():
    _, _, _, x_train, _, y_train, _, folds = prepared_data()
    results, selected = tune_torch_learning_rate(x_train, y_train, folds)
    assert [result.learning_rate for result in results] == [0.001, 0.003, 0.005, 0.010]
    expected = max(results, key=lambda item: (item.mean, -item.std, -item.learning_rate))
    assert selected == expected.learning_rate == 0.010


def test_torch_cv_is_reproducible():
    _, _, _, x_train, _, y_train, _, folds = prepared_data()
    first, first_details = run_torch_cv_once(x_train, y_train, folds, 0.010)
    second, second_details = run_torch_cv_once(x_train, y_train, folds, 0.010)
    assert first == second
    assert first_details == second_details


def test_calculate_metrics_values():
    timing = TimingSummary(0.1, 0.01, 5)
    metrics = calculate_metrics(
        np.array([0, 1, 2, 2]), np.array([0, 1, 1, 2]), timing
    )
    assert metrics.accuracy == 0.75
    assert 0 <= metrics.precision_macro <= 1
    assert metrics.training_time == timing


def test_generated_txt_matches_json_source():
    results = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
    assert RESULTS_PATH.read_text(encoding="utf-8") == build_results_text(results)
    assert results["configuration"]["pytorch"]["selected_learning_rate"] == 0.010
    assert np.isclose(results["holdout"]["pytorch"]["accuracy"], 0.9666666667)
    assert np.isclose(results["cross_validation"]["pytorch"]["mean"], 0.925)
