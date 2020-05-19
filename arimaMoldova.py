""" module for aplying ARIMA model for predicting the monthly electricity balance in Moldovan network
    author: Vladimir Babushkin
"""
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import datetime
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

from MAE import MAE
from MAPE import MAPE
from MRE import MRE
from MSE import MSE

########################################################################################################################
dataPath = 'DATA/MOLDOVA_LOAD/'

# first list all the csv files
import os
import glob
# sort files according the first integer prefix
def sortKeyFunc(s):
    return int(os.path.basename(s).split('_')[0])

moldovaData = sorted(glob.glob(dataPath+'**.xls'),key=sortKeyFunc)

# to store aggregated data
loadMoldovaDf = pd.DataFrame(columns=['year','monthRo','date', 'load'])
yearsArray = []
monthRoArray = []
dateArray = []
loadArray = []
for m in range(len(moldovaData)):
    month =  moldovaData[m].split('_')[4]
    year =  moldovaData[m].split('_')[5].split('.')[0]
    if month == 'ian':
        dateArray.append(datetime(int(year), 1,1))
    if month == 'feb':
        dateArray.append(datetime(int(year), 2,1))
    if month == 'mar':
        dateArray.append(datetime(int(year), 3,1))
    if month == 'apr':
        dateArray.append(datetime(int(year), 4,1))
    if month == 'mai':
        dateArray.append(datetime(int(year), 5,1))
    if month == 'iun' or month == 'iunie':
        if month == 'iunie': month= 'iun'
        dateArray.append(datetime(int(year), 6,1))
    if month == 'iul' or month == 'iulie':
        if month == 'iulie': month ='iul'
        dateArray.append(datetime(int(year), 7,1))
    if month == 'aug':
        dateArray.append(datetime(int(year), 8,1))
    if month == 'sept':
        dateArray.append(datetime(int(year), 9,1))
    if month == 'oct':
        dateArray.append(datetime(int(year), 10,1))
    if month == 'noi':
        dateArray.append(datetime(int(year), 11,1))
    if month == 'dec':
        dateArray.append(datetime(int(year), 12,1))
    monthRoArray.append(month)
    yearsArray.append(year)
    xls = pd.ExcelFile(moldovaData[m])
    sheetX = xls.parse(0) #0 is the sheet number
    var1 = sheetX['Unnamed: 5'] # the load is in this column
    loadArray.append(var1[10]) # this is the row that contains load

loadMoldovaDf['year'] = yearsArray
loadMoldovaDf['monthRo'] = monthRoArray
loadMoldovaDf['date'] = dateArray
loadMoldovaDf['load'] = loadArray
datesColumn = loadMoldovaDf['date'].apply(lambda x: x.strftime('%b-%Y'))
import numpy as np
series = pd.Series(loadMoldovaDf['load'].values, index=loadMoldovaDf.date)
series.plot()

autocorrelation_plot(series)


# fit model
model = ARIMA(series, order=(5,2,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


X = series.values
X=X/1000
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,2,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat[0])
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat[0], obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


mae = MAE(test, predictions)
mape = MAPE(test, predictions)
mre = MRE(test, predictions)
mse = MSE(test, predictions)

# plot
fig = plt.figure(figsize=(14, 6))
xTicksLabels = datesColumn.tolist()
ax = fig.add_subplot(111)
plt.plot(X,'k--')
plt.plot(list(range(size)),train,'k',marker = 'o', linestyle='dashed',label='train')
plt.plot(list(range(size, len(X))),test,'b',marker = 'o', label='test')
plt.plot(list(range(size, len(X))),predictions, color='r',label='predicted',marker = '^',alpha = 0.9)
plt.xlim([-1,50])
plt.xticks(np.arange(0,50,1),xTicksLabels, fontsize=8, rotation = -60,ha='left')
plt.ylabel("Electrical energy balance, MWh", fontsize=15)
plt.title("Energia electrică intrată în reţelele de transport în total",fontsize=15)
plt.legend(loc='upper left',frameon=False)
ax.text(0.008, 0.80, 'mae = ' + str(np.round(mae, 3)), transform=ax.transAxes,fontsize=10)
ax.text(0.008, 0.75, 'mape = ' + str(np.round(mape, 3)), transform=ax.transAxes,fontsize=10)
ax.text(0.008, 0.70, 'mre = ' + str(np.round(mre, 3)), transform=ax.transAxes,fontsize=10)
ax.text(0.008, 0.65, 'rmse = ' + str(np.round(mse, 3)), transform=ax.transAxes,fontsize=10)