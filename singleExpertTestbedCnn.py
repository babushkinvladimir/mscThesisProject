""" module for prediction with CNN
    author: Vladimir Babushkin

    Parameters
    ----------
    trainYear  : which year use for training
    testYear  : which year use for testing
    n_steps_in  : size of the input vector in days
    numDaysToTrain : days, preceeding the day of interest to process with k-means
    region: NSW ot NYC
    isPrice: if False predicts load
"""
import os
import numpy as np
import pandas as pd

from loadData import loadData

# import errors
from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import model_from_json


# split a univariate sequence into samples
def split_sequences_cnn(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.extend(seq_y)
    return np.array(X), np.array(y)


def predictWithCnnModel(model, numEpochs, n_steps_in, n_steps_out, n_features, dataset, dates, dateToPredict,
                        numDaysToTrain):
    # save norms values to reconstruct back to loads
    daysBeforeDate = len(dates[dates < dateToPredict])

    # we train on preceeding 12 months
    trainDataset = dataset[daysBeforeDate - numDaysToTrain:daysBeforeDate]
    actualDay = dataset[daysBeforeDate]

    # normalize the data
    datasetNorm = (trainDataset - np.min(trainDataset)) / (np.max(trainDataset) - np.min(trainDataset))

    # convert into input/output
    daysToTrainX, y = split_sequences_cnn(datasetNorm, n_steps_in, n_steps_out)

    model.fit(daysToTrainX, y, epochs=numEpochs, verbose=0)

    # add a new day to daysToTrainX and predict for this sequence of 5 days:
    daysToTest = datasetLoad[daysBeforeDate - n_steps_in:daysBeforeDate]
    daysToTestNorm = (daysToTest - np.min(trainDataset)) / (np.max(trainDataset) - np.min(trainDataset))

    daysToTestX = []
    for i in range(n_steps_in):
        daysToTestX.append(daysToTestNorm[i])

    daysToTestX = np.array(daysToTestX)
    daysToTestX = daysToTestX.reshape((1, n_steps_in, n_features))

    # using the trained model to predict
    predictedDayNorm = model.predict(daysToTestX, verbose=1).flatten()
    predictedDay = predictedDayNorm * (np.max(trainDataset) - np.min(trainDataset)) + np.min(trainDataset)

    return (model, predictedDay, actualDay)


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
    resultsDirectory = directory + "/results/cnnPrice" + region + "/"
else:
    resultsDirectory = directory + "/results/cnnLoad" + region + "/"

if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)
########################################################################################################################
# load data
(datasetLoad, datesLoad) = loadData(datapath, region, isPrice)
########################################################################################################################


########################################################################################################################
# parameters
n_steps_in = 5
n_features = 24
n_steps_out = 1

# for deep learning networks we train on one year
numDaysToTrain = 365
numFilters = 15
numNeuronsLayer1 = 1500
numEpochs = 200
########################################################################################################################
# define model
model = Sequential()
model.add(Conv1D(filters=numFilters, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(numNeuronsLayer1, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

# from keras.utils import plot_model
# plot_model(model, to_file=directory+"/results/cnn/modelCnn.pdf",show_shapes=True, show_layer_names=True)

########################################################################################################################
# first train network on the preceding 365 days:
trainYear = 2018
trainDatesNetwork = [d for d in datesLoad if d.year == trainYear]

measuresCnnTrain = np.zeros((len(trainDatesNetwork), 4))
actualDaysCnnTrain = np.zeros((len(trainDatesNetwork), 24))
predictedDaysCnnTrain = np.zeros((len(trainDatesNetwork), 24))

for d in range(len(trainDatesNetwork)):
    # enter the date to predict
    dateToPredict = trainDatesNetwork[d]
    print("Training for " + str(dateToPredict))
    (model, predictedDay, actualDayTrain) = predictWithCnnModel(model, numEpochs, n_steps_in, n_steps_out, n_features,
                                                                datasetLoad, datesLoad, dateToPredict, numDaysToTrain)

    mae = MAE(actualDayTrain, predictedDay)
    mape = MAPE(actualDayTrain, predictedDay)
    mre = MRE(actualDayTrain, predictedDay)
    mse = MSE(actualDayTrain, predictedDay)

    actualDaysCnnTrain[d, :] = actualDayTrain
    predictedDaysCnnTrain[d, :] = predictedDay
    measuresCnnTrain[d, :] = [mae, mape, mre, mse]

measuresDfTrain = pd.DataFrame(columns=['date', 'mae', 'mape', 'mre', 'mse'])
measuresDfTrain['date'] = trainDatesNetwork
measuresDfTrain['mae'] = measuresCnnTrain[:, 0]
measuresDfTrain['mape'] = measuresCnnTrain[:, 1]
measuresDfTrain['mre'] = measuresCnnTrain[:, 2]
measuresDfTrain['mse'] = measuresCnnTrain[:, 3]

measuresDfTrain.to_csv(
    resultsDirectory + "measuresCnnTrain_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
        numEpochs) + "_" + str(trainYear) + ".csv")

np.save(resultsDirectory + "predictedDaysCnnTrain_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
    numEpochs) + "_" + str(trainYear), predictedDaysCnnTrain)
np.save(resultsDirectory + "actualDaysCnnTrain_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
    numEpochs) + "_" + str(trainYear), actualDaysCnnTrain)
np.save(resultsDirectory + "actualDatesCnnTrain_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
    numEpochs) + "_" + str(trainYear), trainDatesNetwork)

# save model
# serialize model to JSON
model_json = model.to_json()
with open(resultsDirectory + "modelTrain_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
        numEpochs) + "_" + str(trainYear) + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(
    resultsDirectory + "modelTrain_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(numEpochs) + "_" + str(
        trainYear) + ".h5")
print("Saved model to disk")

# # load model
# # load json and create model
# json_file = open(resultsDirectory+"modelTrain_"+str(numFilters)+"_"+str(numNeuronsLayer1)+"_"+str(numEpochs)+"_"+str(trainYear)+".json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights(resultsDirectory+"modelTrain_"+str(numFilters)+"_"+str(numNeuronsLayer1)+"_"+str(numEpochs)+"_"+str(trainYear)+".h5")
# print("Loaded model from disk")


########################################################################################################################
# then test network on the preceding 365 days:
testYear = 2020
testDatesNetwork = [d for d in datesLoad if d.year == testYear]

measuresCnnTest = np.zeros((len(testDatesNetwork), 4))
actualDaysCnnTest = np.zeros((len(testDatesNetwork), 24))
predictedDaysCnnTest = np.zeros((len(testDatesNetwork), 24))

for d in range(len(testDatesNetwork)):
    # enter the date to predict
    dateToPredict = testDatesNetwork[d]
    print("Testing for " + str(dateToPredict))
    (model, predictedDay, actualDayTest) = predictWithCnnModel(model, numEpochs, n_steps_in, n_steps_out, n_features,
                                                               datasetLoad, datesLoad, dateToPredict, numDaysToTrain)

    mae = MAE(actualDayTest, predictedDay)
    mape = MAPE(actualDayTest, predictedDay)
    mre = MRE(actualDayTest, predictedDay)
    mse = MSE(actualDayTest, predictedDay)

    actualDaysCnnTest[d, :] = actualDayTest
    predictedDaysCnnTest[d, :] = predictedDay
    measuresCnnTest[d, :] = [mae, mape, mre, mse]

# save model
# serialize model to JSON
model_json = model.to_json()
with open(resultsDirectory + "modelTest_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
        numEpochs) + "_" + str(testYear) + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(
    resultsDirectory + "modelTest_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(numEpochs) + "_" + str(
        testYear) + ".h5")
print("Saved model to disk")

# # load model
# # load json and create model
# json_file = open(resultsDirectory+"modelTest_"+str(numFilters)+"_"+str(numNeuronsLayer1)+"_"+str(numEpochs)+"_"+str(testYear)+".json", 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights(resultsDirectory+"modelTest_"+str(numFilters)+"_"+str(numNeuronsLayer1)+"_"+str(numEpochs)+"_"+str(testYear)+".h5")
# print("Loaded model from disk")

measuresDfTest = pd.DataFrame(columns=['date', 'mae', 'mape', 'mre', 'mse'])
measuresDfTest['date'] = testDatesNetwork
measuresDfTest['mae'] = measuresCnnTest[:, 0]
measuresDfTest['mape'] = measuresCnnTest[:, 1]
measuresDfTest['mre'] = measuresCnnTest[:, 2]
measuresDfTest['mse'] = measuresCnnTest[:, 3]

measuresDfTest.to_csv(resultsDirectory + "measuresCnnTest_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
    numEpochs) + "_" + str(testYear) + ".csv")

np.save(resultsDirectory + "predictedDaysCnnTest_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
    numEpochs) + "_" + str(testYear), predictedDaysCnnTest)
np.save(resultsDirectory + "actualDaysCnnTest_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
    numEpochs) + "_" + str(testYear), actualDaysCnnTest)
np.save(resultsDirectory + "actualDatesCnnTest_" + str(numFilters) + "_" + str(numNeuronsLayer1) + "_" + str(
    numEpochs) + "_" + str(testYear), testDatesNetwork)
