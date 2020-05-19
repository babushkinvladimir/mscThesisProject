import numpy as np

""" mean absolute error

    Parameters
    ----------
    actual           : actual values
    predicted        : predicted values
    
    Output
    ------
    mean absolute error
"""


def MAE(actual, predicted):
    return (1.0 / len(actual)) * np.sum(np.abs(predicted - actual))
