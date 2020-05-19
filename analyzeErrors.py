""" calculates average errors for each month
    autor: Vladimir Babushkin
        Parameters
        ----------
        year        : testing dataset's year to calculate average errors across months

        Output
        ------
        .csv file with average errors for all 12 months and the total
"""
import datetime
import glob
import os

import numpy as np
import pandas as pd

# get the data
year = 2019

isPrice = True
# define results directory
directory = os.getcwd()
alg = 'lstm'
region = 'NYC'
if isPrice:
    resDirectory = directory + "/results/" + alg + "Price" + region
else:
    resDirectory = directory + "/results/" + alg + "Load" + region
allCsvFiles = sorted(glob.glob(resDirectory + '/' + "measures" + '*' + str(year) + '.csv'))
print(allCsvFiles[0])
measures = pd.read_csv(allCsvFiles[0])

mae = measures["mae"].apply(pd.to_numeric)
mape = measures["mape"].apply(pd.to_numeric)
mre = measures["mre"].apply(pd.to_numeric)
mse = measures["mse"].apply(pd.to_numeric)
# this line converts the string object in Timestamp object
if region == 'NYC':
    dateTime = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in measures["date"]]
else:
    dateTime = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in measures["date"]]

newDf = pd.DataFrame(columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "total"])
for m in range(1, 13):
    maeForMonth = np.array([mae[i] for i in range(len(mae)) if dateTime[i].month == m])
    mapeForMonth = np.array([mape[i] for i in range(len(mape)) if dateTime[i].month == m])
    mreForMonth = np.array([mre[i] for i in range(len(mre)) if dateTime[i].month == m])
    mseForMonth = np.array([mse[i] for i in range(len(mse)) if dateTime[i].month == m])
    newDf[str(m)] = [np.round(np.mean(maeForMonth[np.isfinite(maeForMonth)]), 2),
                     np.round(np.mean(mapeForMonth[np.isfinite(mapeForMonth)]), 2),
                     np.round(np.mean(mreForMonth[np.isfinite(mreForMonth)]), 2),
                     np.round(np.mean(mseForMonth[np.isfinite(mseForMonth)]), 2)]

newDf["total"] = np.round(newDf.mean(1), 2)
if isPrice:
    newDf.to_csv(directory + "/results/" + alg + "Price" + region + "/forWord" + str(year) + ".csv")
else:
    newDf.to_csv(directory + "/results/" + alg + "Load" + region + "/forWord" + str(year) + ".csv")
