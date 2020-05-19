""" module for prediction with PSF-style Hierarchical classifier
    author: Vladimir Babushkin

    Parameters
    ----------
    numClusters : number fo clusters
    windowSize  : size of the sliding window
    numDaysToTrain : days, preceeding the day of interest to process with hierarchical
    region: NSW ot NYC
    isPrice: if False predicts load
"""
import os

import numpy as np
import pandas as pd

# import errors
from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE
from loadData import loadData
from psfHierarchical import psfHierarchical


def predictWithPsf(dataset, dates, dateToPredict, numDaysToTrain):
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

    # predict with psf
    (predicetdDayNorm, indecesFound) = psfHierarchical(datasetNorm, numClusters, windowSize, outliersArrayTrain)

    # denormalize the prediction
    predictedDay = predicetdDayNorm * (np.max(trainDataset) - np.min(trainDataset)) + np.min(trainDataset)
    return (predictedDay, actualDay)


########################################################################################################################
# define the region NSW or NYC
region = 'NYC'

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

if isPrice:
    resultsDirectory = directory + "/results/hierarchicalPrice" + region + "/"
else:
    resultsDirectory = directory + "/results/hierarchicalLoad" + region + "/"

if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)
########################################################################################################################
# load data
(datasetLoad, datesLoad) = loadData(datapath, region, isPrice)

########################################################################################################################
#  parameters
numClusters = 5
windowSize = 11
numDaysToTrain = 365 * 2

####################################
year = 2019
datesHierarchical = [d for d in datesLoad if d.year == year]

measuresHierarchical = np.zeros((len(datesHierarchical), 4))
actualDaysHierarchical = np.zeros((len(datesHierarchical), 24))
predictedDaysHierarchical = np.zeros((len(datesHierarchical), 24))

###################################################################################

for d in range(len(datesHierarchical)):
    (predictedDay, actualDay) = predictWithPsf(datasetLoad, datesLoad, datesHierarchical[d], numDaysToTrain)

    mae = MAE(actualDay, predictedDay)
    mape = MAPE(actualDay, predictedDay)
    mre = MRE(actualDay, predictedDay)
    mse = MSE(actualDay, predictedDay)

    measuresHierarchical[d, :] = [mae, mape, mre, mse]
    actualDaysHierarchical[d, :] = actualDay
    predictedDaysHierarchical[d, :] = predictedDay

########################################################################################################################
# save results

# convert measures to readible format and save as csv
measuresDf = pd.DataFrame(columns=['date', 'mae', 'mape', 'mre', 'mse'])
measuresDf['date'] = datesHierarchical
measuresDf['mae'] = measuresHierarchical[:, 0]
measuresDf['mape'] = measuresHierarchical[:, 1]
measuresDf['mre'] = measuresHierarchical[:, 2]
measuresDf['mse'] = measuresHierarchical[:, 3]

measuresDf.to_csv(resultsDirectory + "measuresPsfHierarchical" + str(year) + ".csv")

np.save(resultsDirectory + "predictedDaysArrayHierarchical" + str(year), predictedDaysHierarchical)
np.save(resultsDirectory + "actualDaysArrayHierarchical" + str(year), actualDaysHierarchical)
np.save(resultsDirectory + "actualDatesHierarchical" + str(year), datesHierarchical)
