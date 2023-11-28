import numpy as np


class Linear:
    """Applies a linear transformation on the incoming data: $Ax + b$."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self._in_features = in_features
        self._out_features = out_features

        self.weights = np.random.rand(out_features, self._in_features)
        self.bias = 0.0 if not bias else np.random.rand(self._out_features)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagates the inputs through the linear layer.
        We assume that the `inputs` matrix has shape (*, in_features).
        """
        return inputs @ self.weights + self.bias
