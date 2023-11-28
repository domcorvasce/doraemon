import numpy as np


class Sigmoid:
    """Applies the sigmoid activation function"""

    def forward(self, data: np.array) -> np.array:
        return 1 / (1 + np.exp(-data))
