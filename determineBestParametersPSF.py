""" module for determining best parameters for PSF algorithm
    with grid search

    author: Vladimir Babushkin

    Parameters
    ----------
    region: NSW ot NYC
    isPrice: if False predicts load
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# import errors
from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE
from loadData import loadData
from psfFuzzy import psfFuzzy
from psfHierarchical import psfHierarchical
from psfKMedoids import psfKMedoids
from psfKmeans import psfKmeans
from psfSom import psfSom


def predictWithPsf(dataset, dates, dateToPredict, numDaysToTrain, windowSize, numClusters, algorithm):
    Q1 = np.quantile(dataset, 0.25)
    Q3 = np.quantile(dataset, 0.75)
    IQR = Q3 - Q1
    lowerMargin = Q1 - 1.5 * IQR
    upperMargin = Q3 + 1.5 * IQR
    a = dataset < lowerMargin
    b = dataset > upperMargin

    outliersArray = []
    for k in range(dataset.shape[0]):
        if a[k].any() or b[k].any():
            outliersArray.append(1)
        else:
            outliersArray.append(0)

    # save norms values to reconstruct back to loads
    daysBeforeDate = len(dates[dates < dateToPredict])

    # we train on preceeding 12 months
    trainDataset = dataset[daysBeforeDate - numDaysToTrain:daysBeforeDate]
    actualDay = dataset[daysBeforeDate]

    outliersArrayTrain = outliersArray[daysBeforeDate - numDaysToTrain:daysBeforeDate]
    outliersArrayTrain = np.array(outliersArrayTrain)
    outliersArrayTrain = np.where(outliersArrayTrain == 1)
    outliersArrayTrain = outliersArrayTrain[0]

    # normalize the data
    datasetNorm = (trainDataset - np.min(trainDataset)) / (np.max(trainDataset) - np.min(trainDataset))

    if algorithm == 'Kmeans':
        (predicetdDayNorm, indecesFound) = psfKmeans(datasetNorm, numClusters, windowSize, outliersArrayTrain)
    if algorithm == 'Kmedoids':
        (predicetdDayNorm, indecesFound) = psfKMedoids(datasetNorm, numClusters, windowSize, outliersArrayTrain)
    if algorithm == 'Fuzzy':
        (predicetdDayNorm, indecesFound) = psfFuzzy(datasetNorm, numClusters, windowSize, outliersArrayTrain)
    if algorithm == 'Hierarchical':
        (predicetdDayNorm, indecesFound) = psfHierarchical(datasetNorm, numClusters, windowSize, outliersArrayTrain)
    if algorithm == 'Som':
        (predicetdDayNorm, indecesFound) = psfSom(datasetNorm, numClusters, windowSize, outliersArrayTrain)

    # denormalize the prediction
    predictedDay = predicetdDayNorm * (np.max(trainDataset) - np.min(trainDataset)) + np.min(trainDataset)
    return (predictedDay, actualDay)


########################################################################################################################
# define the region NSW or NYC
region = 'NSW'

yearToPredict = 2015
########################################################################################################################
# define what to predict -- price or load
isPrice = True

########################################################################################################################
# define results directory
directory = os.getcwd()

if region == 'NYC':
    if isPrice:
        datapath = '/DATA/NYISO_PRICE'
    else:
        datapath = '/DATA/NYISO'
if region == 'NSW':
    datapath = '/DATA/NSW'
########################################################################################################################
# specify algorithms
algorithmsArray = ['Kmeans','Kmedoids','Fuzzy','Hierarchical','Som']

for algorithm in algorithmsArray:
    if isPrice:
        resultsDirectory = directory + "/results/BEST_PARAMETERS/psfPrice" + algorithm + region + "_" + str(yearToPredict) + "/"
        np.seterr(divide='ignore')
    else:
        resultsDirectory = directory + "/results/BEST_PARAMETERS/psfLoad" + algorithm + region + "_" + str(yearToPredict) + "/"
    if not os.path.exists(resultsDirectory):
        os.makedirs(resultsDirectory)
    ########################################################################################################################
    # load data
    (datasetLoad, datesLoad) = loadData(datapath, region, isPrice)

    ########################################################################################################################
    # first check on the preceding 365 days:
    numDaysToTrain = 365 * 2  # two years

    trainDates = [d for d in datesLoad if d.year == yearToPredict]
    avgResults = np.zeros((20, 20, 4))

    for numClusters in range(2, 22):
        print("calculating for number of clusters " + str(numClusters))
        for windowSize in range(2, 22):
            measuresPsfTrain = np.zeros((12, 4))
            print("calculating for window size " + str(windowSize))
            for m in range(1, 13):
                currentMonthDates = [d for d in trainDates if d.month == m]
                tmpMeasuresArray = np.zeros((len(currentMonthDates), 4))
                for d in range(len(currentMonthDates)):
                    (predictedDay, actualDay) = predictWithPsf(datasetLoad, datesLoad, trainDates[d], numDaysToTrain,
                                                               numClusters, windowSize, algorithm)
                    mae = MAE(actualDay, predictedDay)
                    if not np.isfinite(mae) or np.isnan(mae):
                        mae = 0
                    mape = MAPE(actualDay, predictedDay)
                    if not np.isfinite(mape) or np.isnan(mape):
                        mape = 0
                    mre = MRE(actualDay, predictedDay)
                    if not np.isfinite(mre) or np.isnan(mre):
                        mre = 0
                    mse = MSE(actualDay, predictedDay)
                    if not np.isfinite(mse) or np.isnan(mse):
                        mse = 0
                    tmpMeasuresArray[d, :] = [mae, mape, mre, mse]

                measuresPsfTrain[m - 1, :] = np.mean(tmpMeasuresArray, axis=0)
                avgResults[numClusters - 2, windowSize - 2, :] = np.mean(measuresPsfTrain, axis=0)
    np.save(resultsDirectory + "/bestParameters" + algorithm, avgResults)

    ########################################################################################################################
    # avgResults = np.load(resultsDirectory+"bestParameters"+algorithm+".npy")

    plt.figure()
    maeData = avgResults[:, :, 0]
    color_map = plt.imshow(maeData)
    color_map.set_cmap("Blues_r")
    plt.gca().invert_yaxis()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 20, step=1))  # Set label locations.
    rightTicksArray = list(range(2, 22))
    plt.xticks(np.arange(20), rightTicksArray, fontsize=8)  # Set text labels.
    plt.yticks(np.arange(0, 20, step=1))  # Set label locations.
    rightTicksArray = list(range(2, 22))
    plt.yticks(np.arange(20), rightTicksArray, fontsize=8)  # Set text labels.
    plt.xlabel('Size of sliding window', fontsize=13)
    plt.ylabel('Number of clusters', fontsize=13)
    if algorithm == 'Kmeans':
        plt.title('Mean absolute error, K-means', fontsize=14, pad=20)
    if algorithm == 'Kmedoids':
        plt.title('Mean absolute error, K-medoids', fontsize=14, pad=20)
    if algorithm == 'Fuzzy':
        plt.title('Mean absolute error, Fuzzy C-means', fontsize=14, pad=20)
    if algorithm == 'Hierarchical':
        plt.title('Mean absolute error, Hierarchical', fontsize=14, pad=20)
    if algorithm == 'Som':
        plt.title('Mean absolute error, SOM', fontsize=14, pad=20)
    plt.colorbar()
    plt.savefig(resultsDirectory + "/avgMae" + algorithm + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

    plt.figure()
    mapeData = avgResults[:, :, 1]
    color_map = plt.imshow(mapeData)
    color_map.set_cmap("Blues_r")
    plt.gca().invert_yaxis()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 20, step=1))  # Set label locations.
    rightTicksArray = list(range(2, 22))
    plt.xticks(np.arange(20), rightTicksArray, fontsize=8)  # Set text labels.
    plt.yticks(np.arange(0, 20, step=1))  # Set label locations.
    rightTicksArray = list(range(2, 22))
    plt.yticks(np.arange(20), rightTicksArray, fontsize=8)  # Set text labels.
    plt.ylabel('Number of clusters', fontsize=13)
    plt.xlabel('Size of sliding window', fontsize=13)
    if algorithm == 'Kmeans':
        plt.title('Mean absolute percentage error, K-means', fontsize=14, pad=20)
    if algorithm == 'Kmedoids':
        plt.title('Mean absolute percentage error, K-medoids', fontsize=14, pad=20)
    if algorithm == 'Fuzzy':
        plt.title('Mean absolute percentage error, Fuzzy C-means', fontsize=14, pad=20)
    if algorithm == 'Hierarchical':
        plt.title('Mean absolute percentage error, Hierarchical', fontsize=14, pad=20)
    if algorithm == 'Som':
        plt.title('Mean absolute percentage error, SOM', fontsize=14, pad=20)
    plt.colorbar()
    plt.savefig(resultsDirectory + "/avgMape" + algorithm + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

    plt.figure()
    mreData = avgResults[:, :, 2]
    color_map = plt.imshow(mreData)
    color_map.set_cmap("Blues_r")
    plt.gca().invert_yaxis()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 20, step=1))  # Set label locations.
    rightTicksArray = list(range(2, 22))
    plt.xticks(np.arange(20), rightTicksArray, fontsize=8)  # Set text labels.
    plt.yticks(np.arange(0, 20, step=1))  # Set label locations.
    rightTicksArray = list(range(2, 22))
    plt.yticks(np.arange(20), rightTicksArray, fontsize=8)  # Set text labels.
    plt.ylabel('Number of clusters', fontsize=13)
    plt.xlabel('Size of sliding window', fontsize=13)
    # plt.clim(20, 80)
    if algorithm == 'Kmeans':
        plt.title('Mean relative error, K-means', fontsize=14, pad=20)
    if algorithm == 'Kmedoids':
        plt.title('Mean relative error, K-medoids', fontsize=14, pad=20)
    if algorithm == 'Fuzzy':
        plt.title('Mean relative error, Fuzzy C-means', fontsize=14, pad=20)
    if algorithm == 'Hierarchical':
        plt.title('Mean relative error, Hierarchical', fontsize=14, pad=20)
    if algorithm == 'Som':
        plt.title('Mean relative error, SOM', fontsize=14, pad=20)
    plt.colorbar()
    plt.savefig(resultsDirectory + "/avgMre" + algorithm + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

    plt.figure()
    mseData = avgResults[:, :, 3]
    color_map = plt.imshow(mseData)
    color_map.set_cmap("Blues_r")
    plt.gca().invert_yaxis()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 20, step=1))  # Set label locations.
    rightTicksArray = list(range(2, 22))
    plt.xticks(np.arange(20), rightTicksArray, fontsize=8)  # Set text labels.
    plt.yticks(np.arange(0, 20, step=1))  # Set label locations.
    rightTicksArray = list(range(2, 22))
    plt.yticks(np.arange(20), rightTicksArray, fontsize=8)  # Set text labels.
    plt.ylabel('Number of clusters', fontsize=13)
    plt.xlabel('Size of sliding window', fontsize=13)
    if algorithm == 'Kmeans':
        plt.title('Root mean square error, K-means', fontsize=14, pad=20)
    if algorithm == 'Kmedoids':
        plt.title('Root mean square error, K-medoids', fontsize=14, pad=20)
    if algorithm == 'Fuzzy':
        plt.title('Root mean square error, Fuzzy C-means', fontsize=14, pad=20)
    if algorithm == 'Hierarchical':
        plt.title('Root mean square error, Hierarchical', fontsize=14, pad=20)
    if algorithm == 'Som':
        plt.title('Root mean square error, SOM', fontsize=14, pad=20)
    plt.colorbar()
    plt.savefig(resultsDirectory + "/avgRmse" + algorithm + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
    plt.close('all')
