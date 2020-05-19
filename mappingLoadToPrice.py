""" mappping load to price
    author: Vladimir Babushkin

    Parameters
    ----------
    trainYear           : which year to train
    testyear            : which year to test
    region              : region where we want to analyze data ('NYC', 'NSW')
    lstm                : define which load prediction to use: LSTM or ARIMA (True/False)

"""
import os

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential

# import errors
from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE
from loadData import loadData

########################################################################################################################
# define the region NSW or NYC
region = 'NYC'
# define which load prediction to use: LSTM or ARIMA
lstm = False
numEpochs = 100
########################################################################################################################
# define results directory
directory = os.getcwd()

if region == 'NYC':
    datapathLoad = '/DATA/NYISO'
    datapathPrice = '/DATA/NYISO_PRICE'
if region == 'NSW':
    datapathLoad = '/DATA/NSW'
    datapathPrice = '/DATA/NSW'

if lstm:
    resultsDirectory = directory + "/results/loadToPriceLstm" + region + "/"
else:
    resultsDirectory = directory + "/results/loadToPriceArima" + region + "/"

if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

########################################################################################################################
# load data
(datasetLoad, datesLoad) = loadData(datapathLoad, region, False)
########################################################################################################################
# price data
(datasetPrice, datesPrice) = loadData(datapathPrice, region, True)
########################################################################################################################

trainYear = 2018
trainDates = [d for d in datesLoad if d.year == trainYear]
trainDatasetLoad = [datasetLoad[i] for i in range(len(datesLoad)) if datesLoad[i].year == trainYear]
trainDatasetPrice = [datasetPrice[i] for i in range(len(datesLoad)) if datesLoad[i].year == trainYear]

########################################################################################################################
# train network
dnnData = []
output = []

for i in range(6, len(trainDatasetLoad)):
    loadData = [trainDatasetPrice[i - 6], trainDatasetPrice[i - 1], trainDatasetLoad[i - 6], trainDatasetLoad[i - 1],
                trainDatasetLoad[i]]
    output.append(trainDatasetPrice[i])
    dnnData.append(np.array(loadData))
dnnData = np.array(dnnData)
output = np.array(output)
n_steps_in, n_steps_out = 5, 24
n_features = dnnData.shape[2]

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(dnnData, output, epochs=numEpochs, verbose=1)  # 50 epochs
########################################################################################################################
# save model
# serialize model to JSON
model_json = model.to_json()
with open(resultsDirectory + "modelTrain" + str(trainYear) + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(resultsDirectory + "modelTrain" + str(trainYear) + ".h5")
print("Saved model to disk")

########################################################################################################################
# predict
testYear = 2019
testDates = [d for d in datesLoad if d.year == testYear]
actualTestDatasetLoad = [datasetLoad[i] for i in range(len(datesLoad)) if datesLoad[i].year == testYear]
testDatasetPrice = [datasetPrice[i] for i in range(len(datesLoad)) if datesLoad[i].year == testYear]
# load results for Lstm
if lstm:
    actualDatesLstmTest = np.load(
        directory + "/results/lstmLoad" + region + "/actualDatesTest_500_700_100_" + str(testYear) + ".npy",
        allow_pickle=True)
    actualDaysArrayLstmTest = np.load(
        directory + "/results/lstmLoad" + region + "/actualDaysLstmTest_500_700_100_" + str(testYear) + ".npy")
    testDatasetLoad = np.load(
        directory + "/results/lstmLoad" + region + "/predictedDaysLstmTest_500_700_100_" + str(testYear) + ".npy")
else:
    actualDatesLstmTest = np.load(
        directory + "/results/arimaLoad" + region + "/actualDatesTest_5_2_1_" + str(testYear) + ".npy",
        allow_pickle=True)
    actualDaysArrayLstmTest = np.load(
        directory + "/results/arimaLoad" + region + "/actualDaysArimaTest_5_2_1_" + str(testYear) + ".npy")
    testDatasetLoad = np.load(
        directory + "/results/arimaLoad" + region + "/predictedDaysArimaTest_5_2_1_" + str(testYear) + ".npy")

measuresTest = np.zeros((len(testDatasetPrice), 4))
actualPricesTest = np.zeros((len(testDatasetPrice), 24))
predictedPricesTest = np.zeros((len(testDatasetPrice), 24))

for d in range(len(testDatasetPrice)):

    print('testing for ' + str(actualDatesLstmTest[d]))
    if d == 0:
        testData = np.array([[trainDatasetPrice[len(trainDatasetLoad) - 6],
                              trainDatasetPrice[len(trainDatasetLoad) - 1], trainDatasetLoad[len(trainDatasetLoad) - 6],
                              trainDatasetLoad[len(trainDatasetLoad) - 1], testDatasetLoad[d]]])
    elif d < 6:
        testData = np.array([[trainDatasetPrice[len(trainDatasetLoad) - 6 + d], testDatasetPrice[d - 1],
                              trainDatasetLoad[len(trainDatasetLoad) - 6 + d], actualTestDatasetLoad[d - 1],
                              testDatasetLoad[d]]])
    else:
        testData = np.array([[testDatasetPrice[d - 6], testDatasetPrice[d - 1], actualTestDatasetLoad[d - 6],
                              actualTestDatasetLoad[d - 1], testDatasetLoad[d]]])

    if d == 0:
        actualTestData = np.array([[trainDatasetPrice[len(trainDatasetLoad) - 6],
                                    trainDatasetPrice[len(trainDatasetLoad) - 1],
                                    trainDatasetLoad[len(trainDatasetLoad) - 6],
                                    trainDatasetLoad[len(trainDatasetLoad) - 1], actualTestDatasetLoad[d]]])
    elif d < 6:
        actualTestData = np.array([[trainDatasetPrice[len(trainDatasetLoad) - 6 + d], testDatasetPrice[d - 1],
                                    trainDatasetLoad[len(trainDatasetLoad) - 6 + d], actualTestDatasetLoad[d - 1],
                                    actualTestDatasetLoad[d]]])
    else:
        actualTestData = np.array([[testDatasetPrice[d - 6], testDatasetPrice[d - 1], actualTestDatasetLoad[d - 6],
                                    actualTestDatasetLoad[d - 1], actualTestDatasetLoad[d]]])

    predictedPriceCNN = model.predict(testData)
    predictedPriceCNN = predictedPriceCNN.flatten()

    actualPrice = testDatasetPrice[d]

    mae = MAE(actualPrice, predictedPriceCNN)
    mape = MAPE(actualPrice, predictedPriceCNN)
    mre = MRE(actualPrice, predictedPriceCNN)
    mse = MSE(actualPrice, predictedPriceCNN)

    actualPricesTest[d, :] = actualPrice
    predictedPricesTest[d, :] = predictedPriceCNN
    measuresTest[d, :] = [mae, mape, mre, mse]
    # add the test instance to the inputs and retrain model after every 7 days
    dnnData = np.append(dnnData, actualTestData, axis=0)
    output = np.append(output, np.array([actualPrice]), axis=0)
    model.fit(dnnData, output, epochs=numEpochs, verbose=0)

normMeasuresTest = np.zeros((len(predictedPricesTest), 4))
for k in range(len(predictedPricesTest)):
    predictedPriceCNN = predictedPricesTest[k]
    actualPrice = testDatasetPrice[k]
    normPredPriceCNN = np.abs(
        predictedPriceCNN)
    normActualPrice = np.abs(actualPrice)
    mae = MAE(normActualPrice, normPredPriceCNN)
    mape = MAPE(normActualPrice, normPredPriceCNN)
    mre = MRE(normActualPrice, normPredPriceCNN)
    mse = MSE(normActualPrice, normPredPriceCNN)
    normMeasuresTest[k, :] = [mae, mape, mre, mse]

print(np.mean(measuresTest, 0))
measuresDfTest = pd.DataFrame(columns=['date', 'mae', 'mape', 'mre', 'mse'])
measuresDfTest['date'] = testDates
measuresDfTest['mae'] = measuresTest[:, 0]
measuresDfTest['mape'] = measuresTest[:, 1]
measuresDfTest['mre'] = measuresTest[:, 2]
measuresDfTest['mse'] = measuresTest[:, 3]
measuresDfTest.to_csv(resultsDirectory + "measuresCnnLstmTest" + str(testYear) + ".csv")
np.save(resultsDirectory + "actualDatesTest" + str(testYear), testDates)
np.save(resultsDirectory + "actualPricesTest" + str(testYear), actualPricesTest)
np.save(resultsDirectory + "predictedPricesTest" + str(testYear), predictedPricesTest)

# save model
# serialize model to JSON
model_json = model.to_json()
with open(resultsDirectory + "modelTest" + str(testYear) + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(resultsDirectory + "modelTest" + str(testYear) + ".h5")
print("Saved model to disk")
