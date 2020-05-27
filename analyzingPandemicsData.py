""" evaluating performance of classifiers during COVID-19 pandemics
    autor: Vladimir Babushkin
        Parameters
        ----------
        region: NSW ot NYC


        Output
        ------
        plots with predictions
"""
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

region = 'NSW'

path = os.getcwd()
os.chdir(path)
dataPath = path + '/DATA/COVID/'
figuresDir = path + '/results/FIGURES/'
testYear = 2020

if region == 'NYC':
    tmpDf = pd.read_csv(dataPath + 'dataNYC.csv')
    dateTime = [datetime.datetime.strptime(d, "%m/%d/%y") for d in tmpDf["DATE_OF_INTEREST"]]
if region == 'NSW':
    tmpDf = pd.DataFrame(columns=["DATE_OF_INTEREST", "Cases"])
    preTmpDf = pd.read_csv(dataPath + 'dataNSW.csv')
    tmpDateTime = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in preTmpDf["notification_date"]]
    tmpDateTimeNp = np.array(tmpDateTime)
    uniqueDates = np.unique(tmpDateTimeNp)

    dateTime = list(pd.date_range(start="2020-01-01", end="2020-05-12").to_pydatetime())
    count = np.zeros(len(dateTime))
    i = 0
    for d in tmpDateTime:
        if d == uniqueDates[i]:
            idx = dateTime.index(d)
            count[idx] += 1
        else:
            i += 1
    tmpDf["DATE_OF_INTEREST"] = dateTime
    tmpDf["Cases"] = count

########################################################################################################################

tmpDf["DATE_OF_INTEREST"] = dateTime
datesColumn = tmpDf["DATE_OF_INTEREST"].apply(lambda x: x.strftime('%d-%b'))
xTicksLabels = datesColumn.tolist()
casesData = tmpDf["Cases"]

########################################################################################################################


fig = plt.figure(figsize=(14, 4))
plt.bar(datesColumn, casesData)
plt.xlim([-1, len(casesData)])
plt.xticks(np.arange(0, len(casesData), 1), xTicksLabels, fontsize=6, rotation=-60, ha='left')
if region == 'NSW':
    plt.title("Number of daily COVID-19 cases, New South Wales, Australia, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title(
        "Number of daily COVID-19 cases, New York City, US, from " + xTicksLabels[0] + ', ' + str(testYear) + ' to ' +
        xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.05, hspace=0, wspace=0)
plt.savefig(figuresDir + "covidCases_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

# load arima data
########################################################################################################################
measuresDfTestArima = pd.read_csv(
    path + '/results/arimaLoad' + region + '/' + "measuresArimaTest_5_2_1_" + str(testYear) + ".csv")
actualDaysArimaTest = np.load(
    path + '/results/arimaLoad' + region + '/' + "actualDaysArimaTest_5_2_1_" + str(testYear) + ".npy")
predictedDaysArimaTest = np.load(
    path + '/results/arimaLoad' + region + '/' + "predictedDaysArimaTest_5_2_1_" + str(testYear) + ".npy")

actualDaysArrayArima = actualDaysArimaTest.flatten()
predictedDaysArrayArima = predictedDaysArimaTest.flatten()
fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayArima, label="actual")
plt.plot(predictedDaysArrayArima, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayArima) + 24])
# plt.ylim([np.min(actualDaysArrayArima)-50,np.max(actualDaysArrayArima)+50])
plt.xticks(np.arange(-1, len(actualDaysArrayArima), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title(
        "Forecasting electricity load with ARIMA, New South Wales, Australia, from " + xTicksLabels[0] + ', ' + str(
            testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title("Forecasting electricity load with ARIMA, New York City, US, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "arimaForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

########################################################################################################################
# load kmeans data
########################################################################################################################
measuresDfTestKmeans = pd.read_csv(
    path + '/results/kmeansLoad' + region + '/' + "measuresPsfKmeans" + str(testYear) + ".csv")
actualDaysKmeansTest = np.load(
    path + '/results/kmeansLoad' + region + '/' + "actualDaysArrayKmeans" + str(testYear) + ".npy")
predictedDaysKmeansTest = np.load(
    path + '/results/kmeansLoad' + region + '/' + "predictedDaysArrayKmeans" + str(testYear) + ".npy")
actualDaysArrayKmeans = actualDaysKmeansTest.flatten()
predictedDaysArrayKmeans = predictedDaysKmeansTest.flatten()

fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayKmeans, label="actual")
plt.plot(predictedDaysArrayKmeans, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayKmeans) + 24])
plt.xticks(np.arange(-1, len(actualDaysArrayKmeans), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title(
        "Forecasting electricity load with k-means, New South Wales, Australia, from " + xTicksLabels[0] + ', ' + str(
            testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title("Forecasting electricity load with k-means, New York City, US, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "kMeansForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

########################################################################################################################
# load kmedoids data
########################################################################################################################
measuresDfTestKmedoids = pd.read_csv(
    path + '/results/kmedoidsLoad' + region + '/' + "measuresPsfKmedoids" + str(testYear) + ".csv")
actualDaysKmedoidsTest = np.load(
    path + '/results/kmedoidsLoad' + region + '/' + "actualDaysArrayKmedoids" + str(testYear) + ".npy")
predictedDaysKmedoidsTest = np.load(
    path + '/results/kmedoidsLoad' + region + '/' + "predictedDaysArrayKmedoids" + str(testYear) + ".npy")
actualDaysArrayKmedoids = actualDaysKmedoidsTest.flatten()
predictedDaysArrayKmedoids = predictedDaysKmedoidsTest.flatten()

fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayKmedoids, label="actual")
plt.plot(predictedDaysArrayKmedoids, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayKmedoids) + 24])
plt.xticks(np.arange(-1, len(actualDaysArrayKmedoids), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title(
        "Forecasting electricity load with k-medoids, New South Wales, Australia, from " + xTicksLabels[0] + ', ' + str(
            testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title("Forecasting electricity load with k-medoids, New York City, US, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "kMedoidsForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

########################################################################################################################
# load fuzzy cmeans data
########################################################################################################################
measuresDfTestFuzzy = pd.read_csv(
    path + '/results/fuzzyLoad' + region + '/' + "measuresPsfFuzzy" + str(testYear) + ".csv")
actualDaysFuzzyTest = np.load(
    path + '/results/fuzzyLoad' + region + '/' + "actualDaysArrayFuzzy" + str(testYear) + ".npy")
predictedDaysFuzzyTest = np.load(
    path + '/results/fuzzyLoad' + region + '/' + "predictedDaysArrayFuzzy" + str(testYear) + ".npy")
actualDaysArrayFuzzy = actualDaysFuzzyTest.flatten()
predictedDaysArrayFuzzy = predictedDaysFuzzyTest.flatten()

fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayFuzzy, label="actual")
plt.plot(predictedDaysArrayFuzzy, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayFuzzy) + 24])
plt.xticks(np.arange(-1, len(actualDaysArrayFuzzy), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title("Forecasting electricity load with fuzzy C-means, New South Wales, Australia, from " + xTicksLabels[
        0] + ', ' + str(testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title(
        "Forecasting electricity load with fuzzy C-means, New York City, US, from " + xTicksLabels[0] + ', ' + str(
            testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "fuzzyForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

########################################################################################################################
# load Hierarchical data
########################################################################################################################
measuresDfTestHierarchical = pd.read_csv(
    path + '/results/hierarchicalLoad' + region + '/' + "measuresPsfHierarchical" + str(testYear) + ".csv")
actualDaysHierarchicalTest = np.load(
    path + '/results/hierarchicalLoad' + region + '/' + "actualDaysArrayHierarchical" + str(testYear) + ".npy")
predictedDaysHierarchicalTest = np.load(
    path + '/results/hierarchicalLoad' + region + '/' + "predictedDaysArrayHierarchical" + str(testYear) + ".npy")
actualDaysArrayHierarchical = actualDaysHierarchicalTest.flatten()
predictedDaysArrayHierarchical = predictedDaysHierarchicalTest.flatten()

fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayHierarchical, label="actual")
plt.plot(predictedDaysArrayHierarchical, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayHierarchical) + 24])
plt.xticks(np.arange(-1, len(actualDaysArrayHierarchical), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title("Forecasting electricity load with Hierarchical, New South Wales, Australia, from " + xTicksLabels[
        0] + ', ' + str(testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title("Forecasting electricity load with Hierarchical, New York City, US, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "hierarchicalForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

########################################################################################################################
# load SOM data
########################################################################################################################
measuresDfTestSom = pd.read_csv(path + '/results/somLoad' + region + '/' + "measuresPsfSom" + str(testYear) + ".csv")
actualDaysSomTest = np.load(path + '/results/somLoad' + region + '/' + "actualDaysArraySom" + str(testYear) + ".npy")
predictedDaysSomTest = np.load(
    path + '/results/somLoad' + region + '/' + "predictedDaysArraySom" + str(testYear) + ".npy")
actualDaysArraySom = actualDaysSomTest.flatten()
predictedDaysArraySom = predictedDaysSomTest.flatten()

fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArraySom, label="actual")
plt.plot(predictedDaysArraySom, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArraySom) + 24])
plt.xticks(np.arange(-1, len(actualDaysArraySom), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title("Forecasting electricity load with SOM, New South Wales, Australia, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title("Forecasting electricity load with SOM, New York City, US, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "somForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

########################################################################################################################
# load lstm data
########################################################################################################################
measuresDfTestLstm = pd.read_csv(
    path + '/results/lstmLoad' + region + '/' + "measuresLstmTest_500_700_100_" + str(testYear) + ".csv")
actualDaysLstmTest = np.load(
    path + '/results/lstmLoad' + region + '/' + "actualDaysLstmTest_500_700_100_" + str(testYear) + ".npy")
predictedDaysLstmTest = np.load(
    path + '/results/lstmLoad' + region + '/' + "predictedDaysLstmTest_500_700_100_" + str(testYear) + ".npy")
actualDaysArrayLstm = actualDaysLstmTest.flatten()
predictedDaysArrayLstm = predictedDaysLstmTest.flatten()
fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayLstm, label="actual")
plt.plot(predictedDaysArrayLstm, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayLstm) + 24])
plt.ylim([np.min(actualDaysArrayLstm) - 50, np.max(actualDaysArrayLstm) + 50])
plt.xticks(np.arange(-1, len(actualDaysArrayLstm), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title("Forecasting electricity load with LSTM, New South Wales, Australia, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title("Forecasting electricity load with LSTM, New York City, US, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "lstmForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

########################################################################################################################
# load cnn data
########################################################################################################################
measuresDfTestCnn = pd.read_csv(
    path + '/results/cnnLoad' + region + '/' + "measuresCnnTest_15_1500_200_" + str(testYear) + ".csv")
actualDaysCnnTest = np.load(
    path + '/results/cnnLoad' + region + '/' + "actualDaysCnnTest_15_1500_200_" + str(testYear) + ".npy")
predictedDaysCnnTest = np.load(
    path + '/results/cnnLoad' + region + '/' + "predictedDaysCnnTest_15_1500_200_" + str(testYear) + ".npy")
actualDaysArrayCnn = actualDaysCnnTest.flatten()
predictedDaysArrayCnn = predictedDaysCnnTest.flatten()
fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayCnn, label="actual")
plt.plot(predictedDaysArrayCnn, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayCnn) + 24])
plt.ylim([np.min(actualDaysArrayCnn) - 50, np.max(actualDaysArrayCnn) + 50])
plt.xticks(np.arange(-1, len(actualDaysArrayCnn), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title("Forecasting electricity load with CNN, New South Wales, Australia, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title("Forecasting electricity load with CNN, New York City, US, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "cnnForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

########################################################################################################################
# load to price  data
########################################################################################################################
measuresDfTestLstm = pd.read_csv(
    path + '/results/loadToPriceLstm' + region + '/' + "measuresTest" + str(testYear) + ".csv")
actualDaysLstmTest = np.load(
    path + '/results/loadToPriceLstm' + region + '/' + "actualPricesTest" + str(testYear) + ".npy")
predictedDaysLstmTest = np.load(
    path + '/results/loadToPriceLstm' + region + '/' + "predictedPricesTest" + str(testYear) + ".npy")
actualDaysArrayLstm = actualDaysLstmTest.flatten()
predictedDaysArrayLstm = predictedDaysLstmTest.flatten()
fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayLstm, label="actual")
plt.plot(predictedDaysArrayLstm, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayLstm) + 24])
plt.ylim([np.min(actualDaysArrayLstm) - 50, np.max(actualDaysArrayLstm) + 50])
plt.xticks(np.arange(-1, len(actualDaysArrayLstm), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Price, $')
plt.title("Forecasting electricity price from load, " + region + ", 2020")
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "loadToPriceForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

# load arima data
########################################################################################################################
measuresDfTestArima = pd.read_csv(
    path + '/results/arimaPrice' + region + '/' + "measuresArimaTest_5_2_1_" + str(testYear) + ".csv")
actualDaysArimaTest = np.load(
    path + '/results/arimaPrice' + region + '/' + "actualDaysArimaTest_5_2_1_" + str(testYear) + ".npy")
predictedDaysArimaTest = np.load(
    path + '/results/arimaPrice' + region + '/' + "predictedDaysArimaTest_5_2_1_" + str(testYear) + ".npy")

actualDaysArrayArima = actualDaysArimaTest.flatten()
predictedDaysArrayArima = predictedDaysArimaTest.flatten()
fig = plt.figure(figsize=(14, 4))
plt.plot(actualDaysArrayArima, label="actual")
plt.plot(predictedDaysArrayArima, 'r-', alpha=0.5, label="predicted")
plt.xlim([-24, len(actualDaysArrayArima) + 24])
# plt.ylim([np.min(actualDaysArrayArima)-50,np.max(actualDaysArrayArima)+50])
plt.xticks(np.arange(-1, len(actualDaysArrayArima), 24), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.ylabel('Load, MW')
if region == 'NSW':
    plt.title(
        "Forecasting electricity price with ARIMA, New South Wales, Australia, from " + xTicksLabels[0] + ', ' + str(
            testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
if region == 'NYC':
    plt.title("Forecasting electricity price with ARIMA, New York City, US, from " + xTicksLabels[0] + ', ' + str(
        testYear) + ' to ' + xTicksLabels[-1] + ', ' + str(testYear), fontsize=16)
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.06, hspace=0, wspace=0)
plt.legend()
plt.savefig(figuresDir + "arimaForecast_" + region + "_" + str(testYear) + ".pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

# draw all errors in one plot
########################################################################################################################
errorArrayArima = measuresDfTestArima['mre']
errorArrayKmeans = measuresDfTestKmeans['mre']
errorArrayLstm = measuresDfTestLstm['mre']
fig = plt.figure(figsize=(14, 6))
plt.plot(errorArrayArima, label="ARIMA")
plt.plot(errorArrayKmeans, label="k-means")
plt.plot(errorArrayLstm, label="LSTM")
plt.xlim([-1, len(casesData)])
plt.xticks(np.arange(0, len(casesData), 1), xTicksLabels, fontsize=6, rotation=-60, ha='left')
plt.title("Mean Relative Errors, " + region + ", 2020")
plt.subplots_adjust(top=0.88, bottom=0.11, right=0.98, left=0.05, hspace=0, wspace=0)
plt.legend()
