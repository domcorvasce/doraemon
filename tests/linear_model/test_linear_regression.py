import numpy as np

from pydoraemon.linear_model import LinearRegression


def test_linear_regression_model_initialization():
    NUM_FEATURES = 5
    model = LinearRegression(features=NUM_FEATURES)
    assert len(model._weights) == NUM_FEATURES
    assert model._weights.sum() == 0.0
    assert model._bias == 0.0


def test_linear_regression_model_prediction():
    model = LinearRegression(features=2)
    model._weights = np.array([1, 2])
    model._bias = 3

    assert model.predict(np.array([[3, 5]])) == np.array([16])


def test_linear_regression_model_mse():
    model = LinearRegression(features=2)
    model._weights = np.array([1, 2])
    model._bias = 3

    assert model.mse(np.array([[3, 5]]), np.array([16])) == 0.0
    assert model.mse(np.array([[3, 5]]), np.array([15])) == 2.0


def test_basic_linear_regression_training():
    weights = np.array([1, 2])
    bias = 3.0

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, weights) + bias

    model = LinearRegression(features=2)
    model.train(X, y, max_epochs=1000)

    assert weights.sum() - model._weights.sum() < 0.01
    assert bias - model._bias < 0.01
