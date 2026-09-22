import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    denom = (1 + np.exp(-1 * np.array(x))) ** -1
    return 1 * denom