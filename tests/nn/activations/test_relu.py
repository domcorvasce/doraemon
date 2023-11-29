import numpy as np

from src.nobiru import nn


def test_relu_activation_on_positive_output():
    layer = nn.Linear(in_features=2, out_features=2)
    relu = nn.ReLU()

    layer.weights = np.array([[0.044631, 0.7956511], [0.62903345, 0.28984344]])
    layer.bias = np.array([0.0, 0.0])

    activation = relu.forward(layer.forward(np.array([[200.0, 17.0]])))

    assert activation.shape == (1, 2)
    assert round(activation[0][0], 2) == 19.62
    assert round(activation[0][1], 2) == 164.06


def test_relu_activation_on_negative_output():
    layer = nn.Linear(in_features=2, out_features=2)
    relu = nn.ReLU()

    layer.weights = np.array([[-0.044631, -0.7956511], [-0.62903345, -0.28984344]])
    layer.bias = np.array([-0.01, -0.02])

    activation = relu.forward(layer.forward(np.array([[200.0, 17.0]])))

    assert activation.shape == (1, 2)
    assert np.min(activation) == 0.0
