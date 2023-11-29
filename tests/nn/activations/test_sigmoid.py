import numpy as np

from src.nobiru import nn


def test_sigmoid_activation_on_linear_layer():
    layer = nn.Linear(in_features=2, out_features=2)
    sigmoid = nn.Sigmoid()

    layer.weights = np.array([[0.044631, 0.7956511], [0.62903345, 0.28984344]])
    layer.bias = np.array([0.0, 0.0])

    activation = sigmoid.forward(layer.forward(np.array([[200.0, 17.0]])))

    assert activation.shape == (1, 2)
    assert round(activation[0][0], 2) == 1.0
    assert round(activation[0][1], 2) == 1.0
