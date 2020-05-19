import numpy as np

"""  mean squared error

    Parameters
    ----------
    actual           : actual values
    predicted        : predicted values

    Output
    ------
    mean squared error
"""


def MSE(actual, predicted):
    return np.sqrt(np.mean((predicted - actual) ** 2))
