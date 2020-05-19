import numpy as np

"""  mean relative error

    Parameters
    ----------
    actual           : actual values
    predicted        : predicted values

    Output
    ------
    mean relative error
"""


def MRE(actual, predicted):
    return (100.0 / len(actual)) * np.sum(np.abs(predicted - actual) / np.mean(actual))
