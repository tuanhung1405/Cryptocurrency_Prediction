import math

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta

"""
Class of model Arima for the purpose of building an object-oriented Class.
Each object is an Arima model with input variables as data and period.

Method
------
    From the original data set, calculate profit based on log formula of "close" column.
    
    Based on the strategy to find the model with the best AIC, 
    with the input data being the "close" column profit value. 
        
    Then using auto_arima from pmdarima to find the best p,d,q.
    Predict the future value of coin base on the model then return the actual value.

"""


class ArimaModel:
    def __init__(self, data, period):
        self.data = data
        self.period = period
        self.result = None
        self.new_model = None
        self.dbReturn = None

    def checkData(self):
        maxday = self.data.index.max()
        minday = self.data.index.min()
        if maxday - minday <= timedelta(days=730):
            warn = "This coin is quite new. The data it creates is less than 2 years, so the model is not reliable enough"
        else:
            warn = "The length of data is oke"
        return warn

    def checkStationarity(self):
        result = adfuller(self.dbReturn)
        if result[1] >= 0.05:
            warn = "P-value > 0.05 => Yield series is non-stationary, the model is not good"
        else:
            warn = "P-value < 0.05 => Yield series is stationary"


        return warn, result[0], result[1]

    def createDataReturn(self):
        self.dbReturn = pd.DataFrame(np.log(self.data['close'] / self.data['close'].shift(1)))
        self.dbReturn = self.dbReturn.fillna(self.dbReturn.head().mean())
        return self.dbReturn

    def displaySummary(self):
        model = auto_arima(self.dbReturn, start_p=1, start_q=1,
                           max_p=10, max_q=10, m=1,
                           start_P=0, seasonal=False,
                           d=0, D=0, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False, max_order=10)

        self.new_model = SARIMAX(self.dbReturn, order=model.order)
        self.result = self.new_model.fit(disp=False)

        return self.result

    def predict(self, delta):
        dic = {"DAY": 1, "1WEEK": 7, "2WEEK": 14, "MONTH": 30}

        latest = self.data.index.max() + timedelta(days=dic.get(self.period))
        date_list = [latest + timedelta(days=x * dic.get(self.period)) for x in range(delta)]

        fc = self.result.get_prediction(start=int(self.new_model.nobs),
                                        end=self.new_model.nobs + delta - 1,
                                        full_reports=True)

        prediction = fc.predicted_mean
        prediction_ci = fc.conf_int()

        prediction = pd.DataFrame(prediction)
        prediction.index = date_list

        prediction_ci = pd.DataFrame(prediction_ci)
        prediction_ci.index = date_list

        prediction.columns = ['predicted_mean']
        lst_mean = self.actualPrice(list(prediction['predicted_mean']))
        lst_upper = self.actualPrice(list(prediction_ci['upper close']))
        lst_lower = self.actualPrice(list(prediction_ci['lower close']))

        date_list_predict = [self.data.index.max() + timedelta(days=x * dic.get(self.period)) for x in range(delta + 1)]

        data_predict = pd.DataFrame({"Price_mean": lst_mean,
                                     "Price_lower": lst_upper,
                                     "Price_upper": lst_lower}, index=date_list_predict)

        return data_predict

    def actualPrice(self, lst):
        l_lastprice = list(self.data['close'].iloc[[0]])
        l_exp = list(math.e ** self.dbReturn['close'].iloc[[0]])

        for i in lst:
            a = math.e ** i
            l_exp.append(a)

        for i in l_exp:
            x = l_lastprice[-1] / i
            l_lastprice.append(x)

        l_lastprice.pop()
        return l_lastprice
