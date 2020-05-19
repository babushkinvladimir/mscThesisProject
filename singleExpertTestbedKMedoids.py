""" module for prediction with PSF-style kMedoids classifier
    author: Vladimir Babushkin

    Parameters
    ----------
    numClusters : number fo clusters
    windowSize  : size of the sliding window
    numDaysToTrain : days, preceeding the day of interest to process with k-medoids
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
from psfKMedoids import psfKMedoids


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
    (predicetdDayNorm, indecesFound) = psfKMedoids(datasetNorm, numClusters, windowSize, outliersArrayTrain)

    # denormalize the prediction
    predictedDay = predicetdDayNorm * (np.max(trainDataset) - np.min(trainDataset)) + np.min(trainDataset)
    return (predictedDay, actualDay)


########################################################################################################################
# define the region NSW or NYC
region = 'NYC'

########################################################################################################################
# define what to predict -- price or load
isPrice = False

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
    resultsDirectory = directory + "/results/kMedoidsPrice" + region + "/"
else:
    resultsDirectory = directory + "/results/kMedoidsLoad" + region + "/"

if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)
########################################################################################################################
# load data

(datasetLoad, datesLoad) = loadData(datapath, region, isPrice)
########################################################################################################################

#  parameters
numClusters = 7
windowSize = 3
numDaysToTrain = 365 * 2

###################################################################################
year = 2020
datesKMedoids = [d for d in datesLoad if d.year == year]

measuresKMedoids = np.zeros((len(datesKMedoids), 4))
actualDaysKMedoids = np.zeros((len(datesKMedoids), 24))
predictedDaysKMedoids = np.zeros((len(datesKMedoids), 24))

###################################################################################
# run the prediction for 365 days:

measuresPsfKMedoids = np.zeros((len(datesKMedoids), 4))
actualDaysArrayKMedoids = np.zeros((len(datesKMedoids), 24))
predictedDaysArrayKMedoids = np.zeros((len(datesKMedoids), 24))

for d in range(len(datesKMedoids)):
    (predictedDay, actualDay) = predictWithPsf(datasetLoad, datesLoad, datesKMedoids[d], numDaysToTrain)

    mae = MAE(actualDay, predictedDay)
    mape = MAPE(actualDay, predictedDay)
    mre = MRE(actualDay, predictedDay)
    mse = MSE(actualDay, predictedDay)

    measuresPsfKMedoids[d, :] = [mae, mape, mre, mse]
    actualDaysArrayKMedoids[d, :] = actualDay
    predictedDaysArrayKMedoids[d, :] = predictedDay

########################################################################################################################
# save results

# convert measures to readible format and save as csv
measuresDf = pd.DataFrame(columns=['date', 'mae', 'mape', 'mre', 'mse'])
measuresDf['date'] = datesKMedoids
measuresDf['mae'] = measuresPsfKMedoids[:, 0]
measuresDf['mape'] = measuresPsfKMedoids[:, 1]
measuresDf['mre'] = measuresPsfKMedoids[:, 2]
measuresDf['mse'] = measuresPsfKMedoids[:, 3]

measuresDf.to_csv(resultsDirectory + "measuresPsfKmedoids" + str(year) + ".csv")

np.save(resultsDirectory + "predictedDaysArrayKmedoids" + str(year), predictedDaysArrayKMedoids)
np.save(resultsDirectory + "actualDaysArrayKmedoids" + str(year), actualDaysArrayKMedoids)
np.save(resultsDirectory + "actualDatesKmedoids" + str(year), datesKMedoids)
