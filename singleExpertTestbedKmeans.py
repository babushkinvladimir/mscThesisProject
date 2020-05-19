""" module for prediction with PSF-style kMeans classifier
    author: Vladimir Babushkin

    Parameters
    ----------
    numClusters : number fo clusters
    windowSize  : size of the sliding window
    numDaysToTrain : days, preceeding the day of interest to process with k-means
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
from psfKmeans import psfKmeans


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

    outliersArray = outliersArray[daysBeforeDate - numDaysToTrain:daysBeforeDate]
    outliersArray = np.array(outliersArray)
    outliersArray = np.where(outliersArray == 1)
    outliersArray = outliersArray[0]

    # normalize the data
    datasetNorm = (trainDataset - np.min(trainDataset)) / (np.max(trainDataset) - np.min(trainDataset))
    
    # predict with psf
    (predicetdDayNorm, indecesFound) = psfKmeans(datasetNorm, numClusters, windowSize, outliersArray)

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
    resultsDirectory = directory + "/results/kMeansPrice" + region + "/"
else:
    resultsDirectory = directory + "/results/kMeansLoad" + region + "/"

if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)
########################################################################################################################
# load data

(datasetLoad, datesLoad) = loadData(datapath, region, isPrice)
########################################################################################################################


########################################################################################################################
#  parameters
numClusters = 2
windowSize = 13
numDaysToTrain = 365 * 2

###################################################################################
# predict for given year
year = 2019
datesKmeans = [d for d in datesLoad if d.year == year]

measuresKmeans = np.zeros((len(datesKmeans), 4))
actualDaysKmeans = np.zeros((len(datesKmeans), 24))
predictedDaysKmeans = np.zeros((len(datesKmeans), 24))

###################################################################################
# run the prediction for 365 days:

measuresPsfKmeans = np.zeros((len(datesKmeans), 4))
actualDaysArrayKmeans = np.zeros((len(datesKmeans), 24))
predictedDaysArrayKmeans = np.zeros((len(datesKmeans), 24))

for d in range(len(datesKmeans)):
    (predictedDay, actualDay) = predictWithPsf(datasetLoad, datesLoad, datesKmeans[d], numDaysToTrain)

    mae = MAE(actualDay, predictedDay)
    mape = MAPE(actualDay, predictedDay)
    mre = MRE(actualDay, predictedDay)
    mse = MSE(actualDay, predictedDay)

    measuresPsfKmeans[d, :] = [mae, mape, mre, mse]
    actualDaysArrayKmeans[d, :] = actualDay
    predictedDaysArrayKmeans[d, :] = predictedDay

########################################################################################################################
# save results

# convert measures to readable format and save as csv
measuresDf = pd.DataFrame(columns=['date', 'mae', 'mape', 'mre', 'mse'])
measuresDf['date'] = datesKmeans
measuresDf['mae'] = measuresPsfKmeans[:, 0]
measuresDf['mape'] = measuresPsfKmeans[:, 1]
measuresDf['mre'] = measuresPsfKmeans[:, 2]
measuresDf['mse'] = measuresPsfKmeans[:, 3]

measuresDf.to_csv(resultsDirectory + "measuresPsfKmeans" + str(year) + ".csv")

np.save(resultsDirectory + "predictedDaysArrayKmeans" + str(year), predictedDaysArrayKmeans)
np.save(resultsDirectory + "actualDaysArrayKmeans" + str(year), actualDaysArrayKmeans)
np.save(resultsDirectory + "actualDatesKmeans" + str(year), datesKmeans)
