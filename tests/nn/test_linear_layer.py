import numpy as np

from pydoraemon import nn


def test_linear_layer_initialization():
    layer = nn.Linear(in_features=2, out_features=3)
    assert layer.weights.shape == (3, 2)
    assert layer.bias.shape == (3,)


def test_linear_layer_forward_propagation():
    layer = nn.Linear(in_features=2, out_features=2)
    layer.weights = np.array([[0.044631, 0.7956511], [0.62903345, 0.28984344]])
    layer.bias = np.array([0.0, 0.0])

    activation = layer.forward(np.array([[200.0, 17.0]]))
    assert activation.shape == (1, 2)
    assert round(activation[0][0], 2) == 19.62
    assert round(activation[0][1], 2) == 164.06
