import numpy as np


class Softmax:
    """Applies the softmax activation function"""

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.exp(data) / np.exp(data).sum()
