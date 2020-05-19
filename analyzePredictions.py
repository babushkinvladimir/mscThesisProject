""" plotting the best/worst predictions
    author: Vladimir Babushkin

    Parameters
    ----------
    monthToPlot     : for which month to plot the data
    testYear        : which testYear to analyze
    region          : region where we want to analyze data ('NYC', 'NSW')
    method          : algorithm for which to plot the data ('arima','kmeans','kmedoids','fuzzy','hierarchical','som','lstm','cnn')
    type            : which prediction to show, worst or best
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import errors
from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE

# set parameters
region = 'NYC'
monthToPlot = 4
isPrice = False
testYear = 2019
method = 'lstm'
type = "Best"

# figure parameters
width = 10
height = 8

# define results directory
directory = os.getcwd()

if isPrice:
    cond = "Price"
else:
    cond = "Load"
predictedDaysArray = []
actualDaysArray = []
actualDatesArray = []
measuresArray = []
algArray = ['arima', 'kmeans', 'kmedoids', 'fuzzy', 'hierarchical', 'som', 'lstm', 'cnn']

########################################################################################################################
for alg in algArray:
    resDirectory = directory + "/results/" + alg + cond + region
    actualDatesFiles = sorted(glob.glob(resDirectory + '/' + "actualDates" + '*' + str(testYear) + '.npy'))
    actualDaysArrayFiles = sorted(glob.glob(resDirectory + '/' + "actualDays" + '*' + str(testYear) + '.npy'))
    predictedDaysArrayFiles = sorted(glob.glob(resDirectory + '/' + "predictedDays" + '*' + str(testYear) + '.npy'))
    measuresFiles = sorted(glob.glob(resDirectory + '/' + "measures" + '*' + str(testYear) + '.csv'))
    print(actualDatesFiles)
    print(actualDaysArrayFiles)
    print(predictedDaysArrayFiles)
    print(measuresFiles)
    actualDates = np.load(actualDatesFiles[0], allow_pickle=True)
    actualDays = np.load(actualDaysArrayFiles[0])
    predictedDays = np.load(predictedDaysArrayFiles[0])
    measures = pd.read_csv(measuresFiles[0])
    predictedDaysArray.append(predictedDays)
    actualDaysArray.append(actualDays)
    actualDatesArray.append(actualDates)
    measuresArray.append(measures)

if method == 'arima':
    idx = 0
if method == 'kmeans':
    idx = 1
if method == 'kmedoids':
    idx = 2
if method == 'fuzzy':
    idx = 3
if method == 'hierarchical':
    idx = 4
if method == 'som':
    idx = 5
if method == 'lstm':
    idx = 6
if method == 'cnn':
    idx = 7

if not isPrice:
    ####################################################################################################################
    #
    # PLOT BEST/WORST LOAD PREDICTION
    #
    ####################################################################################################################
    mreArray = np.array(measuresArray[idx]['mre'])
    index_min = np.argmin(mreArray)
    index_max = np.argmax(mreArray)

    if type == "Worst":
        d = index_max
    if type == "Best":
        d = index_min

    actualDay = actualDaysArray[idx][d]
    predictedDay = predictedDaysArray[idx][d]
    actualDate = actualDatesArray[idx][d]

    mae = MAE(actualDay, predictedDay)
    mape = MAPE(actualDay, predictedDay)
    mre = MRE(actualDay, predictedDay)
    mse = MSE(actualDay, predictedDay)

    ####################################################################################################################
    maxY = max(np.max(actualDay), np.max(predictedDay), np.max(predictedDaysArray[0][d]),
               np.max(predictedDaysArray[1][d])) + 20
    minY = min(np.min(actualDay), np.min(predictedDay), np.min(predictedDaysArray[0][d]),
               np.min(predictedDaysArray[1][d])) - 20
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    plt.plot(actualDay, 'k-', label="actual", linewidth=2.5)
    plt.plot(predictedDay, 'r--', label=method, linewidth=2.5)
    plt.xticks(np.arange(0, 24, 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0, 23])
    plt.xlabel('Hours', fontsize=18)
    if isPrice:
        plt.ylabel('Price, $', fontsize=18)
    else:
        plt.ylabel('Load, MW', fontsize=18)
    plt.title(str(actualDate.strftime('%d %b, %Y')) + ", " + method, fontsize=24, y=1.03)
    plt.legend(loc='upper left', fontsize=16)
    ax.text(0.02, 0.80, 'mae = ' + str(np.round(mae, 3)), transform=ax.transAxes, fontsize=14)
    ax.text(0.02, 0.75, 'mape = ' + str(np.round(mape, 3)), transform=ax.transAxes, fontsize=14)
    ax.text(0.02, 0.70, 'mre = ' + str(np.round(mre, 3)), transform=ax.transAxes, fontsize=14)
    ax.text(0.02, 0.65, 'rmse = ' + str(np.round(mse, 3)), transform=ax.transAxes, fontsize=14)
    plt.savefig(directory + "/results/FIGURES/" + method + cond + region + type + str(testYear) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

    ####################################################################################################################
    maeLstm = MAE(actualDay, predictedDaysArray[6][d])
    mapeLstm = MAPE(actualDay, predictedDaysArray[6][d])
    mreLstm = MRE(actualDay, predictedDaysArray[6][d])
    mseLstm = MSE(actualDay, predictedDaysArray[6][d])
    maeCnn = MAE(actualDay, predictedDaysArray[7][d])
    mapeCnn = MAPE(actualDay, predictedDaysArray[7][d])
    mreCnn = MRE(actualDay, predictedDaysArray[7][d])
    mseCnn = MSE(actualDay, predictedDaysArray[7][d])
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    plt.plot(actualDay, 'k-', label="actual", linewidth=2.5)
    plt.plot(predictedDaysArray[6][d], 'm', label="LSTM", linewidth=2, alpha=0.5)
    plt.plot(predictedDaysArray[7][d], 'b', label="CNN", linewidth=2, alpha=0.5)
    plt.xticks(np.arange(0, 24, 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0, 23])
    plt.ylim([minY, maxY])
    plt.xlabel('Hours', fontsize=18)
    if isPrice:
        plt.ylabel('Price, $', fontsize=18)
    else:
        plt.ylabel('Load, MW', fontsize=18)
    plt.title(str(actualDate.strftime('%d %b, %Y')) + ", LSTM/CNN", fontsize=24, y=1.03)
    plt.legend(loc='upper left', fontsize=16)
    ax.text(0.02, 0.75, 'mae = ' + str(np.round(maeLstm, 3)), transform=ax.transAxes, fontsize=14, color='m')
    ax.text(0.02, 0.70, 'mape = ' + str(np.round(mapeLstm, 3)), transform=ax.transAxes, fontsize=14, color='m')
    ax.text(0.02, 0.65, 'mre = ' + str(np.round(mreLstm, 3)), transform=ax.transAxes, fontsize=14, color='m')
    ax.text(0.02, 0.60, 'rmse = ' + str(np.round(mseLstm, 3)), transform=ax.transAxes, fontsize=14, color='m')
    ax.text(0.02, 0.55, 'mae = ' + str(np.round(maeCnn, 3)), transform=ax.transAxes, fontsize=14, color='b')
    ax.text(0.02, 0.50, 'mape = ' + str(np.round(mapeCnn, 3)), transform=ax.transAxes, fontsize=14, color='b')
    ax.text(0.02, 0.45, 'mre = ' + str(np.round(mreCnn, 3)), transform=ax.transAxes, fontsize=14, color='b')
    ax.text(0.02, 0.40, 'rmse = ' + str(np.round(mseCnn, 3)), transform=ax.transAxes, fontsize=14, color='b')
    plt.savefig(directory + "/results/FIGURES/" + method + cond + region + type + "LstmCnn" + str(testYear) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

    ####################################################################################################################
    datesForPrediction = [d for d in actualDatesArray[idx] if d.month == monthToPlot]
    predictedLoadArrayMonth = []
    actualLoadArrayMonth = []
    ticksStr = []
    for d in datesForPrediction:
        k = next((i for i, j in enumerate(actualDatesArray[idx]) if j == d), None)
        predictedLoadArrayMonth.extend(predictedDaysArray[idx][k])
        actualLoadArrayMonth.extend(actualDaysArray[idx][k])
        ticksStr.append(str(d.strftime(' %Y-%m-%d')))
    plt.figure(figsize=(16, 5))
    plt.plot(predictedLoadArrayMonth, 'r', label='predicted', alpha=0.9)
    plt.plot(actualLoadArrayMonth, 'k', label='actual')
    plt.xlim([0, len(predictedLoadArrayMonth)])
    if region == 'NSW':
        plt.title(
            "Predicting electricity load with " + method + " for" + str(d.strftime(' %B, %Y')) + ", New South Wales",
            fontsize=16, y=1.03)
    else:
        plt.title(
            "Predicting electricity load with " + method + " for" + str(d.strftime(' %B, %Y')) + ", New York City",
            fontsize=16, y=1.03)
    plt.ylabel('Load, MW', fontsize=15)
    plt.xticks(np.arange(0, len(predictedLoadArrayMonth) - 1, 24), ticksStr, fontsize=8, rotation=60)
    plt.subplots_adjust(top=0.88, bottom=0.2, right=0.98, left=0.06, hspace=0, wspace=0)
    plt.legend()
    plt.savefig(directory + "/results/FIGURES/" + method + cond + region + "_" + str(monthToPlot) + "_" + str(
        testYear) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
else:
    ####################################################################################################################
    #
    # PLOT  PRICE FOR ONE MONTH
    #
    ####################################################################################################################
    # price predictions
    actualDatesPriceTest = np.load(
        directory + "/results/loadToPriceLstm" + region + "/actualDatesTest" + str(testYear) + ".npy",
        allow_pickle=True)
    actualPricesTest = np.load(
        directory + "/results/loadToPriceLstm" + region + "/actualPricesTest" + str(testYear) + ".npy")
    predictedPricesTest = np.load(
        directory + "/results/loadToPriceLstm" + region + "/predictedPricesTest" + str(testYear) + ".npy")
    measuresPrices = pd.read_csv(
        directory + "/results/loadToPriceLstm" + region + "/measuresTest" + str(testYear) + ".csv")

    maeArray = np.array(measuresPrices['mae'])
    index_min = np.argmin(maeArray)
    index_max = np.argmax(maeArray)

    if type == "Worst":
        d = index_max
    if type == "Best":
        d = index_min

    actualDay = actualPricesTest[d]
    predictedDay = predictedPricesTest[d]
    actualDate = actualDatesPriceTest[d]

    mae = MAE(actualDay, predictedDay)
    mape = MAPE(actualDay, predictedDay)
    mre = MRE(actualDay, predictedDay)
    mse = MSE(actualDay, predictedDay)

    maxY = max(np.max(actualDay), np.max(predictedDay)) + 20
    minY = min(np.min(actualDay), np.min(predictedDay)) - 20
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    plt.plot(actualDay, 'k-', label="actual", linewidth=2.5)
    plt.plot(predictedDay, 'r--', label="predicted", linewidth=2.5)
    plt.xticks(np.arange(0, 24, 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0, 23])
    plt.ylim([minY, maxY])
    plt.xlabel('Hours', fontsize=15)
    plt.ylabel('Price, $', fontsize=15)
    plt.title("Electricity price for " + str(actualDate.strftime('%d %b, %Y')), fontsize=16, y=1.03)
    plt.legend(loc='upper left', fontsize=16)
    ax.text(0.02, 0.80, 'mae = ' + str(np.round(mae, 3)), transform=ax.transAxes, fontsize=14)
    ax.text(0.02, 0.75, 'mape = ' + str(np.round(mape, 3)), transform=ax.transAxes, fontsize=14)
    ax.text(0.02, 0.70, 'mre = ' + str(np.round(mre, 3)), transform=ax.transAxes, fontsize=14)
    ax.text(0.02, 0.65, 'rmse = ' + str(np.round(mse, 3)), transform=ax.transAxes, fontsize=14)
    plt.savefig(directory + "/results/FIGURES/" + "Prices" + region + type + str(testYear) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

    ####################################################################################################################
    datesForPrediction = [d for d in actualDatesPriceTest if d.month == monthToPlot]
    predictedPricesArrayMonth = []
    actualPricesArrayMonth = []
    actualLoadArrayMonth = []
    ticksStr = []

    for d in datesForPrediction:
        k = next((i for i, j in enumerate(actualDatesPriceTest) if j == d), None)
        predictedPricesArrayMonth.extend(predictedPricesTest[k])
        actualPricesArrayMonth.extend(actualPricesTest[k])
        actualLoadArrayMonth.extend(actualDaysArray[idx][k])
        ticksStr.append(str(d.strftime(' %Y-%m-%d')))
    plt.figure(figsize=(16, 7))

    ax1 = plt.subplot(211)
    plt.plot(actualLoadArrayMonth, 'k')
    plt.xlim([0, len(predictedPricesArrayMonth)])
    plt.ylabel('Load, MW', fontsize=15)
    if region == 'NSW':
        plt.title("Electricity load for " + str(d.strftime(' %B, %Y')) + ", New South Wales", fontsize=16, y=1.03)
    else:
        plt.title("Electricity load for " + str(d.strftime(' %B, %Y')) + ", New York City", fontsize=16, y=1.03)
    plt.xticks(np.arange(0, len(predictedPricesArrayMonth) - 1, 24), [])
    plt.subplot(212)
    plt.plot(predictedPricesArrayMonth, 'r', label='predicted', alpha=0.9)
    plt.plot(actualPricesArrayMonth, 'k', label='actual')
    plt.xlim([0, len(predictedPricesArrayMonth)])
    plt.ylabel('Price, $', fontsize=15)
    if region == 'NSW':
        plt.title("Electricity price for " + str(d.strftime(' %B, %Y')) + ", New South Wales", fontsize=16, y=1.03)
    else:
        plt.title("Electricity price for " + str(d.strftime(' %B, %Y')) + ", New York City", fontsize=16, y=1.03)
    plt.xticks(np.arange(0, len(predictedPricesArrayMonth) - 1, 24), ticksStr, fontsize=8, rotation=60)
    plt.legend()
    plt.savefig(directory + "/results/FIGURES/" + region + "PricesLoadOctober" + str(testYear) + ".pdf",
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
