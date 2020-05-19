""" module for prediction with ARIMA
    author: Vladimir Babushkin

    Parameters
    ----------

    numDaysToTrain : days, preceeding the day of interest to train the ARIMA
    pVal: lag order
    dVal: degree of differencing
    qVal: order of moving average
    region: NSW ot NYC
    isPrice: if False predicts load
"""
import os
import numpy as np

from loadData import loadData

# import errors
from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd

########################################################################################################################
# define the region NSW or NYC
region = 'NSW'

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
    resultsDirectory = directory + "/results/arimaPrice" + region + "/"
else:
    resultsDirectory = directory + "/results/arimaLoad" + region + "/"

if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)
########################################################################################################################
# load data
(datasetLoad, datesLoad) = loadData(datapath, region, isPrice)

########################################################################################################################
# then test network on the preceding 365 days:
testYear = 2020
testDatesNetwork = [d for d in datesLoad if d.year == testYear]

numDaysToTrain = 30

(pVal, dVal, qVal) = (5, 2, 1)  # put d = 1 to avoid invertibility issue

measuresArimaTest = np.zeros((len(testDatesNetwork), 4))
actualDaysArimaTest = np.zeros((len(testDatesNetwork), 24))
predictedDaysArimaTest = np.zeros((len(testDatesNetwork), 24))

for d in range(len(testDatesNetwork)):
    # enter the date to predict
    dateToPredict = testDatesNetwork[d]
    print("predicting for " + str(dateToPredict))
    # save norms values to reconstruct back to loads
    daysBeforeDate = len(datesLoad[datesLoad < dateToPredict])

    # we train on preceeding 12 months
    trainDataset = datasetLoad[daysBeforeDate - numDaysToTrain:daysBeforeDate]
    actualDay = datasetLoad[daysBeforeDate]
    test = actualDay
    history = list(trainDataset.flatten())
    predictions = list()
    indicesSkipped = []
    for t in range(len(test)):
        model = ARIMA(history, order=(pVal, dVal, qVal))
        try:
            model_fit = model.fit(disp=0, start_ar_lags=2)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat[0])
            obs = test[t]
            history = history[1:]
            history.append(obs)
        except:
            pass
            indicesSkipped.append(t)

    if len(indicesSkipped) == 24:
        predictions.append(0)

    for k in indicesSkipped:
        if k == 0:
            predictions[k] = predictions[0]
        else:
            predictions.append(predictions[k - 1])
    error = mean_squared_error(test, predictions)
    mae = MAE(test, predictions)
    mape = MAPE(test, predictions)
    mre = MRE(test, predictions)
    mse = MSE(test, predictions)
    actualDaysArimaTest[d, :] = actualDay
    predictedDaysArimaTest[d, :] = predictions
    measuresArimaTest[d, :] = [mae, mape, mre, mse]
    print('Test MSE: %.3f' % error)
    print('Test MAE: %.3f' % mae)
    print('Test MAPE: %.3f' % mape)
    print('Test MRE: %.3f' % mre)
    print('Test RMSE: %.3f' % mse)
measuresDfTest = pd.DataFrame(columns=['date', 'mae', 'mape', 'mre', 'mse'])
measuresDfTest['date'] = testDatesNetwork
measuresDfTest['mae'] = measuresArimaTest[:, 0]
measuresDfTest['mape'] = measuresArimaTest[:, 1]
measuresDfTest['mre'] = measuresArimaTest[:, 2]
measuresDfTest['mse'] = measuresArimaTest[:, 3]

measuresDfTest.to_csv(
    resultsDirectory + "measuresArimaTest_" + str(pVal) + "_" + str(dVal) + "_" + str(qVal) + "_" + str(
        testYear) + ".csv")

np.save(
    resultsDirectory + "predictedDaysArimaTest_" + str(pVal) + "_" + str(dVal) + "_" + str(qVal) + "_" + str(testYear),
    predictedDaysArimaTest)
np.save(resultsDirectory + "actualDaysArimaTest_" + str(pVal) + "_" + str(dVal) + "_" + str(qVal) + "_" + str(testYear),
        actualDaysArimaTest)
np.save(resultsDirectory + "actualDatesTest_" + str(pVal) + "_" + str(dVal) + "_" + str(qVal) + "_" + str(testYear),
        testDatesNetwork)
