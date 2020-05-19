""" Implementation of patterns sequence forecasting algorithm with k-medoids
    (for details see  Martinez-Alvarez, F., Troncoso, A., Riquelme, J. C, & Aguilar-Ruiz, J. S. (2008, December).
    LBF: A labeled-based forecasting algorithm and its application to electricity price time series.
    In Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on (pp. 453-461). IEEE.)

    autor: Vladimir Babushkin
        Parameters
        ----------
        dataset        : time series
        numClusters    : number of clusters
        windowSize     : length of sliding window
        outliersArray  : array of outliers -- these days are skipped while averaging

        Output
        ------
        pred          : predicted day
        indecesFound  : indeces of the days with similar sequences
        """
import numpy as np
from minisom import MiniSom

from searchSequenceNumpy import searchSequenceNumpy


def psfSom(dataset, numClusters, windowSize, outliersArray):
    w = windowSize
    # cluster the data
    # Initialization and training
    som_shape = (1, numClusters)
    som = MiniSom(som_shape[0], som_shape[1], 24, sigma=.35, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=10)

    som.pca_weights_init(dataset)
    som.train_batch(dataset, 3000, verbose=False)

    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in dataset]).T

    # get labels
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    labels = np.ravel_multi_index(winner_coordinates, som_shape)

    # find sequence of windowSize days preceding the day of interest
    sequenceToLook = labels[len(dataset) - w:len(dataset)]

    # find the first indeces of the sequences
    indecesFound = searchSequenceNumpy(labels, sequenceToLook)
    indecesFound = [i for i in indecesFound if i not in outliersArray]

    # if no sequence has been found reduce the window size and check again
    if len(indecesFound) == 0:
        while len(indecesFound) == 0:
            w = w - 1
            if w == 0:
                break
            else:
                sequenceToLook = labels[len(dataset) - w:len(dataset)]
                indecesFound = searchSequenceNumpy(labels, sequenceToLook)
                indecesFound = [i for i in indecesFound if i not in outliersArray]
        # go back to the original windows size
        w = windowSize
        # if still no patterns were found use the last day as prediction
        if len(indecesFound) == 0:
            pred = dataset[len(dataset) - 1, :]
        else:
            pred = np.mean(dataset[indecesFound, :], 0)
    else:
        pred = np.mean(dataset[indecesFound, :], 0)

    return (pred, indecesFound)
