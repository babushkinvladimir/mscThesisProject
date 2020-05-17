""" for loading data
    author: Vladimir Babushkin

    Parameters
    ----------
    datapath            :path where CSV-s are stored
    region              : region where we want to analyze data ('NYC', 'NSW')
    price=False         : false by default, set to True if you want to get the price data
    Output
    ------
    (dataset, dates)   : array of hourly demand/price values and corresponding days
"""

import os
import glob
import pandas as pd
import numpy as np
import datetime

#http://mis.nyiso.com/public/P-58Blist.htm
def loadData(datapath, region, price=False):
    path = os.getcwd()
    os.chdir(path)
    if region == 'NYC':
        # first list all the csv files
        nyisoData = sorted(glob.glob(path + datapath + '/*/**.csv'))

        # to store aggregated data
        allLoadDf = pd.DataFrame(
            columns=['date', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                     '17',
                     '18', '19', '20', '21', '22', '23', '24'])
        daysSkipped = []
        for i in range(len(nyisoData)):
            # read csv files
            tmpDf = pd.read_csv(nyisoData[i])

            # select data for New York City only
            timeSeriesDfNYC = tmpDf.loc[tmpDf['Name'] == 'N.Y.C.']

            # this line converts the string object in Timestamp object
            dateTime = [datetime.datetime.strptime(d, "%m/%d/%Y %H:%M:%S") for d in timeSeriesDfNYC["Time Stamp"]]

            # extract only 24 hours of data (each record starting with 00 minute in timestamp)
            # this is a boolean mask, True only for timestams that contain 00 in minutes (start of the hour)
            dateTime24 = [d.minute == 0 and d.second == 0 for d in dateTime]

            # apply boolean mask to timeseries for NYC
            timeSeriesDfNyc24 = timeSeriesDfNYC.loc[dateTime24]

            if len(timeSeriesDfNyc24) == 24:
                # intialise data of lists.
                if price == False:
                    allLoadDf.loc[i] = [dateTime[0].date()] + list(timeSeriesDfNyc24['Load'].apply(pd.to_numeric))
                else:
                    allLoadDf.loc[i] = [dateTime[0].date()] + list(timeSeriesDfNyc24['LBMP ($/MWHr)'].apply(pd.to_numeric))
            else:  # skip
                daysSkipped.append(dateTime[0].date())

        # rows with nan also ignored
        allLoadDf = allLoadDf.dropna()

        cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                '20', '21', '22', '23', '24']
        dates = allLoadDf['date']

        dataset = np.array(allLoadDf[cols].apply(pd.to_numeric))
        np.argwhere(np.isnan(dataset))
        dates = dates.reset_index(drop=True)
        return (dataset, dates)
    
    if region == 'NSW':
        nswData = sorted(glob.glob(path + datapath + '/**.csv'))
        # to store aggregated data
        allLoadDf = pd.DataFrame(
            columns=['date', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                     '17',
                     '18', '19', '20', '21', '22', '23', '24'])

        for i in range(len(nswData)):
            currentLoadDf = pd.DataFrame(
                columns=['date', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                         '17',
                         '18', '19', '20', '21', '22', '23', '24'])
            # read csv files
            tmpDf = pd.read_csv(nswData[i])

            # this line converts the string object in Timestamp object
            dateTime = [datetime.datetime.strptime(d, "%Y/%m/%d %H:%M:%S") for d in tmpDf["SETTLEMENTDATE"]]

            # extract only 24 hours of data (each record starting with 00 minute in timestamp)
            # this is a boolean mask, True only for timestams that contain 00 in minutes (start of the hour)
            dateTime24 = [d.minute == 0 and d.second == 0 for d in dateTime]

            # apply boolean mask to timeseries for NSW
            timeSeriesDfNsw24 = tmpDf.loc[dateTime24]
            timeSeriesDfNsw24 = timeSeriesDfNsw24.reset_index()
            if price == False:
                x = np.reshape(list(timeSeriesDfNsw24['TOTALDEMAND'].apply(pd.to_numeric)), (int(len(timeSeriesDfNsw24)/24), 24)).T
            else:
                x = np.reshape(list(timeSeriesDfNsw24['RRP'].apply(pd.to_numeric)),(int(len(timeSeriesDfNsw24) / 24), 24)).T
            dates = []
            for  d in range(len(timeSeriesDfNsw24)):
                if d%24 == 0:
                    dates.append(datetime.datetime.strptime(timeSeriesDfNsw24["SETTLEMENTDATE"].loc[d], "%Y/%m/%d %H:%M:%S"))
            currentLoadDf['date'] = dates

            for k in range(1,25):
                currentLoadDf[str(k)] = x[k-1]

            allLoadDf = allLoadDf.append(currentLoadDf, ignore_index=True)

        # rows with nan also ignored
        allLoadDf = allLoadDf.dropna()

        cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                '20', '21', '22', '23', '24']
        dates = allLoadDf['date']

        dataset = np.array(allLoadDf[cols].apply(pd.to_numeric))
        np.argwhere(np.isnan(dataset))
        dates = dates.reset_index(drop=True)
        return (dataset, dates)