""" seraching for the optimal number of epochs for CNN
    author: Vladimir Babushkin

    Parameters
    ----------
    numDaysToTrain      : days, preceeding the day of interest
    region              : region where we want to analyze data ('NYC', 'NSW')

"""
import os

import numpy as np
from keras import backend
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential

from loadData import loadData

# redefine error metrics for keras:

def rmse_ud(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=0))


def mre_ud(y_true, y_pred):
    return 100 / 24 * backend.sum(backend.abs(y_pred - y_true) / backend.mean(y_true, axis=0), axis=0)


def mae_ud(y_true, y_pred):
    return 1 / 24 * backend.sum(backend.abs(y_pred - y_true), axis=0)


def mape_ud(y_true, y_pred):
    return 100 / 24 * backend.sum(backend.abs(y_pred - y_true) / y_true, axis=0)


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


########################################################################################################################
# set main parameters
region = 'NYC'
isPrice = False

########################################################################################################################
# define results directory
directory = os.getcwd()

# load data
if region == 'NYC':
    if isPrice:
        datapath = '/DATA/NYISO_PRICE'
    else:
        datapath = '/DATA/NYISO'
if region == 'NSW':
    datapath = '/DATA/NSW'

if isPrice:
    resultsDirectory = directory + "/results/epochsCnnPrice" + region + "/"
else:
    resultsDirectory = directory + "/results/epochsCnnLoad" + region + "/"

########################################################################################################################
# load data
(datasetLoad, datesLoad) = loadData(datapath, region, isPrice)

########################################################################################################################
# define model
########################################################################################################################
# parameters
n_features = 24
n_steps_in = 5
n_steps_out = 1
# for deep learning networks we train on one year
numDaysToTrain = 365

########################################################################################################################
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape', 'acc', rmse_ud, mre_ud, mae_ud, mape_ud])

########################################################################################################################
# first train network on the preceding 365 days:

numDaysToTrain = 365 * 2  # two years

trainDates = [d for d in datesLoad if d.year == 2015 or d.year == 2014]

# we train on preceeding 12 months
trainDataset = []
for d in trainDates:
    trainDataset.append(list(datasetLoad[np.where(datesLoad == d)[0][0]]))  # take data for 720 days
trainDataset = np.asarray(trainDataset)
# normalize the data
# datasetNorm = (trainDataset - np.min(trainDataset)) / (np.max(trainDataset) - np.min(trainDataset))
datasetNorm = trainDataset
# convert into input/output
daysToTrainX, y = split_sequences_cnn(datasetNorm, n_steps_in, n_steps_out)

# half of the data use for train and the other half to validate
x_val = daysToTrainX[:365]
partial_x_train = daysToTrainX[365:]

y_val = y[:365]
partial_y_train = y[365:]

# fit model

# fit model
history = model.fit(partial_x_train, partial_y_train, epochs=2000, verbose=1, validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

history_dict = history.history

ud_mse = history_dict['mean_squared_error']
val_ud_mse = history_dict['val_mean_squared_error']

ud_mape = history_dict['mean_absolute_percentage_error']
val_ud_mape = history_dict['val_mean_absolute_percentage_error']

ud_mae = history_dict['mean_absolute_error']
val_ud_mae = history_dict['val_mean_absolute_error']

ud_rmse = history_dict['rmse_ud']
val_ud_rmse = history_dict['val_rmse_ud']

ud_mre = history_dict['mre_ud']
val_ud_mre = history_dict['val_mre_ud']

ud_mae = history_dict['mae_ud']
val_ud_mae = history_dict['val_mae_ud']

ud_mape = history_dict['mape_ud']
val_ud_mape = history_dict['val_mape_ud']

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

numEpochsToDisplay = 25
plt.figure()
epochs = range(1, numEpochsToDisplay)
plt.plot(epochs, loss_values[:len(epochs)], 'b', label='Training loss')
plt.plot(epochs, val_loss_values[:len(epochs)], 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig("lossEpochs" + str(numEpochsToDisplay) + '.pdf',
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure()
plt.plot(epochs, acc_values[:len(epochs)], 'b', label='Training accuracy')
plt.plot(epochs, val_acc_values[:len(epochs)], 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig("accEpochs" + str(numEpochsToDisplay) + '.pdf',
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure()
plt.plot(epochs, ud_rmse[:len(epochs)], 'b', label='Training rmse')
plt.plot(epochs, val_ud_rmse[:len(epochs)], 'r', label='Validation rmse')
plt.title('Training and validation root mean square error')
plt.xlabel('Epochs')
plt.ylabel('root mean square error')
plt.legend()
plt.show()
plt.savefig("rmseEpochs" + str(numEpochsToDisplay) + '.pdf',
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure()
plt.plot(epochs, ud_mae[:len(epochs)], 'b', label='Training mae')
plt.plot(epochs, val_ud_mae[:len(epochs)], 'r', label='Validation mae')
plt.title('Training and validation mean absolute error')
plt.xlabel('Epochs')
plt.ylabel('mean absolute error')
plt.legend()
plt.show()
plt.savefig("maeEpochs" + str(numEpochsToDisplay) + '.pdf',
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure()
plt.plot(epochs, ud_mape[:len(epochs)], 'b', label='Training mape')
plt.plot(epochs, val_ud_mape[:len(epochs)], 'r', label='Validation mape')
plt.title('Training and validation mean absolute percentage error')
plt.xlabel('Epochs')
plt.ylabel('mean absolute percentage error')
plt.legend()
plt.show()
plt.savefig("mapeEpochs" + str(numEpochsToDisplay) + '.pdf',
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure()
plt.plot(epochs, ud_mre[:len(epochs)], 'b', label='Training mape')
plt.plot(epochs, val_ud_mre[:len(epochs)], 'r', label='Validation mape')
plt.title('Training and validation mean relative error')
plt.xlabel('Epochs')
plt.ylabel('mean relative error')
plt.legend()
plt.show()
plt.savefig("mreEpochs" + str(numEpochsToDisplay) + '.pdf',
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure()
mse = history_dict['mean_squared_error']
val_mse = history_dict['val_mean_squared_error']
plt.plot(epochs, mse[:len(epochs)], 'b', label='Training mse')
plt.plot(epochs, val_mse[:len(epochs)], 'r', label='Validation mse')
plt.title('Training and validation mean squared error')
plt.xlabel('Epochs')
plt.ylabel('mean squared error')
plt.legend()
plt.show()
plt.savefig("mseEpochs" + str(numEpochsToDisplay) + '.pdf',
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

np.save(resultsDirectory+"/maeNorm", ud_mae)
np.save(resultsDirectory+"/val_maeNorm", val_ud_mae)

np.save(resultsDirectory+"/mapeNorm", ud_mape)
np.save(resultsDirectory+"/val_mapeNorm", val_ud_mape)

np.save(resultsDirectory+"/rmseNorm", ud_rmse)
np.save(resultsDirectory+"/val_rmseNorm", val_ud_rmse)

np.save(resultsDirectory+"/mseNorm", ud_mse)
np.save(resultsDirectory+"/val_mseNorm", val_ud_mse)

np.save(resultsDirectory+"/ud_mreNorm", ud_mre)
np.save(resultsDirectory+"/val_ud_mreNorm", val_ud_mre)

np.save(resultsDirectory+"/ud_maeNorm", ud_mae)
np.save(resultsDirectory+"/val_ud_maeNorm", val_ud_mae)

np.save(resultsDirectory+"/ud_mapeNorm", ud_mape)
np.save(resultsDirectory+"/val_ud_mapeNorm", val_ud_mape)

np.save(resultsDirectory+"/loss_valuesNorm", loss_values)
np.save(resultsDirectory+"/val_loss_valuesNorm", val_loss_values)

np.save(resultsDirectory+"/acc_valuesNorm", acc_values)
np.save(resultsDirectory+"/val_acc_valuesNorm", val_acc_values)
########################################################################################################################
#
########################################################################################################################
# first train network on the preceding 365 days:
dataset = datasetLoad
dates = datesLoad

trainDates = [d for d in datesLoad if d.year == 2015 or d.year == 2014]

# we train on preceeding 12 months
trainDataset = []
for d in trainDates:
    trainDataset.append(list(datasetLoad[np.where(datesLoad == d)[0][0]]))  # take data for 720 days
trainDataset = np.asarray(trainDataset)
# normalize the data
# datasetNorm = (trainDataset - np.min(trainDataset)) / (np.max(trainDataset) - np.min(trainDataset))

datasetNorm = trainDataset

# convert into input/output
daysToTrainX, y = split_sequences_cnn(datasetNorm, n_steps_in, n_steps_out)

# half of the data use for train and the other half to validate
x_val = daysToTrainX[:365]
partial_x_train = daysToTrainX[365:]

y_val = y[:365]
partial_y_train = y[365:]

measuresArrayTrain = np.zeros((15, 15, 9))
measuresArrayVal = np.zeros((15, 15, 9))
numFilters = list(range(5, 105, 5))
numLayers1 = list(range(50, 1050, 50))
for i in range(15):
    for j in range(15):
        print('processing layer1 ' + str(i) + ' layer2 ' + str(j))
        model = Sequential()
        model = Sequential()
        model.add(Conv1D(filters=numFilters[i], kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(numLayers1[j], activation='relu'))
        model.add(Dense(n_features))
        model.compile(optimizer='adam', loss='mse',
                      metrics=['mse', 'mae', 'mape', 'acc', rmse_ud, mre_ud, mae_ud, mape_ud])
        # fit model
        history = model.fit(partial_x_train, partial_y_train, epochs=220, verbose=0, validation_data=(x_val, y_val))
        history_dict = history.history

        ud_mse = history_dict['mean_squared_error']
        val_ud_mse = history_dict['val_mean_squared_error']

        ud_mape = history_dict['mean_absolute_percentage_error']
        val_ud_mape = history_dict['val_mean_absolute_percentage_error']

        ud_mae = history_dict['mean_absolute_error']
        val_ud_mae = history_dict['val_mean_absolute_error']

        ud_rmse = history_dict['rmse_ud']
        val_ud_rmse = history_dict['val_rmse_ud']

        ud_mre = history_dict['mre_ud']
        val_ud_mre = history_dict['val_mre_ud']

        ud_mae = history_dict['mae_ud']
        val_ud_mae = history_dict['val_mae_ud']

        ud_mape = history_dict['mape_ud']
        val_ud_mape = history_dict['val_mape_ud']

        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']

        measuresArrayTrain[i, j, 0] = np.mean(ud_mse)
        measuresArrayVal[i, j, 0] = np.mean(val_ud_mse)
        measuresArrayTrain[i, j, 1] = np.mean(ud_mape)
        measuresArrayVal[i, j, 1] = np.mean(val_ud_mape)
        measuresArrayTrain[i, j, 2] = np.mean(ud_mae)
        measuresArrayVal[i, j, 2] = np.mean(val_ud_mae)
        measuresArrayTrain[i, j, 3] = np.mean(ud_rmse)
        measuresArrayVal[i, j, 3] = np.mean(val_ud_rmse)
        measuresArrayTrain[i, j, 4] = np.mean(ud_mre)
        measuresArrayVal[i, j, 4] = np.mean(val_ud_mre)
        measuresArrayTrain[i, j, 5] = np.mean(ud_mae)
        measuresArrayVal[i, j, 5] = np.mean(val_ud_mae)
        measuresArrayTrain[i, j, 6] = np.mean(ud_mape)
        measuresArrayVal[i, j, 6] = np.mean(val_ud_mape)
        measuresArrayTrain[i, j, 7] = np.mean(loss_values)
        measuresArrayVal[i, j, 7] = np.mean(val_loss_values)
        measuresArrayTrain[i, j, 8] = np.mean(acc_values)
        measuresArrayVal[i, j, 8] = np.mean(val_acc_values)
        del (model)
np.save(resultsDirectory+"/measuresArrayVal_epochs_220", measuresArrayVal)
np.save(resultsDirectory+"/measuresArrayTrain_epochs_220", measuresArrayTrain)

##########################################################################################################################
measuresArrayVal = np.load(resultsDirectory+"/measuresArrayVal_epochs_220.npy")

maeValues = measuresArrayVal[:, :, 5]
mapeValues = measuresArrayVal[:, :, 6]
rmseValues = measuresArrayVal[:, :, 3]
mseValues = measuresArrayVal[:, :, 0]
mreValues = measuresArrayVal[:, :, 4]
accValues = measuresArrayVal[:, :, 8]
lossValues = measuresArrayVal[:, :, 7]

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.figure(figsize=(8, 9))
pltAx = plt.gca()
color_map = plt.imshow(maeValues)
color_map.set_cmap("Greens_r")
plt.gca().invert_yaxis()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(100, 1600, 100))
plt.xticks(np.arange(15), rightTicksArray, fontsize=14, rotation=75)  # Set text labels.
plt.yticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(5, 105, 5))
plt.yticks(np.arange(15), rightTicksArray, fontsize=14)  # Set text labels.
plt.xlabel('Number of neurons in Dense Layer', fontsize=18, labelpad=10)
plt.ylabel('Number of filters', fontsize=18, labelpad=10)
plt.title('Mean absolute error (220 epochs)', fontsize=24, pad=20)
divider = make_axes_locatable(pltAx)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(color_map, cax=cax)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=14)
cbar = plt.colorbar(color_map, cax=cax)
plt.savefig(resultsDirectory+"/avgMaeCnn.pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure(figsize=(8, 9))
pltAx = plt.gca()
color_map = plt.imshow(mreValues)
color_map.set_cmap("Greens_r")
plt.gca().invert_yaxis()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(100, 1600, 100))
plt.xticks(np.arange(15), rightTicksArray, fontsize=14, rotation=75)  # Set text labels.
plt.yticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(5, 105, 5))
plt.yticks(np.arange(15), rightTicksArray, fontsize=14)  # Set text labels.
plt.xlabel('Number of neurons in Dense Layer', fontsize=18, labelpad=10)
plt.ylabel('Number of filters', fontsize=18, labelpad=10)
plt.title('Mean relative error (220 epochs)', fontsize=24, pad=20)
divider = make_axes_locatable(pltAx)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(color_map, cax=cax)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=14)
cbar = plt.colorbar(color_map, cax=cax)
plt.savefig(resultsDirectory+"/avgMreCnn.pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure(figsize=(8, 9))
pltAx = plt.gca()
color_map = plt.imshow(mapeValues)
color_map.set_cmap("Greens_r")
plt.gca().invert_yaxis()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(100, 1600, 100))
plt.xticks(np.arange(15), rightTicksArray, fontsize=14, rotation=75)  # Set text labels.
plt.yticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(5, 105, 5))
plt.yticks(np.arange(15), rightTicksArray, fontsize=14)  # Set text labels.
plt.xlabel('Number of neurons in Dense Layer', fontsize=18, labelpad=10)
plt.ylabel('Number of filters', fontsize=18, labelpad=10)
plt.title('Mean absolute percentage error (220 epochs)', fontsize=24, pad=20)
divider = make_axes_locatable(pltAx)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(color_map, cax=cax)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=14)
cbar = plt.colorbar(color_map, cax=cax)
plt.savefig(resultsDirectory+"/avgMapeCnn.pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure(figsize=(8, 9))
pltAx = plt.gca()
color_map = plt.imshow(rmseValues)
color_map.set_cmap("Greens_r")
plt.gca().invert_yaxis()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(100, 1600, 100))
plt.xticks(np.arange(15), rightTicksArray, fontsize=14, rotation=75)  # Set text labels.
plt.yticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(5, 105, 5))
plt.yticks(np.arange(15), rightTicksArray, fontsize=14)  # Set text labels.
plt.xlabel('Number of neurons in Dense Layer', fontsize=18, labelpad=10)
plt.ylabel('Number of filters', fontsize=18, labelpad=10)
plt.title('Root mean square error (220 epochs)', fontsize=24, pad=20)
divider = make_axes_locatable(pltAx)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(color_map, cax=cax)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=14)
cbar = plt.colorbar(color_map, cax=cax)
plt.savefig(resultsDirectory+"/avgRmseCnn.pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure(figsize=(8, 9))
pltAx = plt.gca()
color_map = plt.imshow(mseValues)
color_map.set_cmap("Greens_r")
plt.gca().invert_yaxis()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(100, 1600, 100))
plt.xticks(np.arange(15), rightTicksArray, fontsize=14, rotation=75)  # Set text labels.
plt.yticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(5, 105, 5))
plt.yticks(np.arange(15), rightTicksArray, fontsize=14)  # Set text labels.
plt.xlabel('Number of neurons in Dense Layer', fontsize=18, labelpad=10)
plt.ylabel('Number of filters', fontsize=18, labelpad=10)
plt.title('Mean square error (220 epochs)', fontsize=24, pad=20)
divider = make_axes_locatable(pltAx)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(color_map, cax=cax)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=14)
cbar = plt.colorbar(color_map, cax=cax)
plt.savefig(resultsDirectory+"/avgMseCnn.pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure(figsize=(8, 9))
pltAx = plt.gca()
color_map = plt.imshow(accValues)
color_map.set_cmap("Greens_r")
plt.gca().invert_yaxis()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(100, 1600, 100))
plt.xticks(np.arange(15), rightTicksArray, fontsize=14, rotation=75)  # Set text labels.
plt.yticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(5, 105, 5))
plt.yticks(np.arange(15), rightTicksArray, fontsize=14)  # Set text labels.
plt.xlabel('Number of neurons in Dense Layer', fontsize=18, labelpad=10)
plt.ylabel('Number of filters', fontsize=18, labelpad=10)
plt.title('Accuracy (220 epochs)', fontsize=24, pad=20)
divider = make_axes_locatable(pltAx)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(color_map, cax=cax)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=14)
cbar = plt.colorbar(color_map, cax=cax)
plt.savefig(resultsDirectory+"/avgAccCnn.pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

plt.figure(figsize=(8, 9))
pltAx = plt.gca()
color_map = plt.imshow(lossValues)
color_map.set_cmap("Greens_r")
plt.gca().invert_yaxis()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(100, 1600, 100))
plt.xticks(np.arange(15), rightTicksArray, fontsize=14, rotation=75)  # Set text labels.
plt.yticks(np.arange(0, 15, step=1))  # Set label locations.
rightTicksArray = list(range(5, 105, 5))
plt.yticks(np.arange(15), rightTicksArray, fontsize=14)  # Set text labels.
plt.xlabel('Number of neurons in Dense Layer', fontsize=18, labelpad=10)
plt.ylabel('Number of filters', fontsize=18, labelpad=10)
plt.title('Loss (220 epochs)', fontsize=24, pad=20)
divider = make_axes_locatable(pltAx)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = plt.colorbar(color_map, cax=cax)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=14)
cbar = plt.colorbar(color_map, cax=cax)
plt.savefig(resultsDirectory+"/avgLossCnn.pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)
