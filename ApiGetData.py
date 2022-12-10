import requests
import pandas as pd
from datetime import datetime, timedelta

"""
The data get from Coinbase's API with link: https://api.pro.coinbase.com
Unfortunately, the data per crawl is limited to 300 record, the code have to crawl many times

------
The functions below are built to handle crawling data, cleaning, 
dividing by different time intervals,... 

-----
Through API: https://api.pro.coinbase.com/currencies, the list of coins is obtained based on USD 

"""


def getDataApi(sym, timeEnd):
    apiUrl = "https://api.pro.coinbase.com"
    barSize = "86400"

    delta = timedelta(hours=24)
    timeStart = timeEnd - delta * 300

    timeStart = timeStart.isoformat()
    timeEnd = timeEnd.isoformat()

    parameters = {
        "start": timeStart,
        "end": timeEnd,
        "granularity": barSize,
    }

    data = requests.get(f"{apiUrl}/products/{sym}/candles",
                        params=parameters,
                        headers={"content-type": "application/json"})
    return data


def formatData(data):
    df = pd.DataFrame(data.json(),
                      columns=["time", "low", "high", "open", "close", "volume"])
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index("date", inplace=True)
    df.drop(["time"], axis=1, inplace=True)
    return df


def getAllData(sym):
    timeEnd = datetime.now()
    delta = timedelta(hours=24)
    df_final = pd.DataFrame(columns=["low", "high", "open", "close", "volume"])

    while True:
        db = getDataApi(sym, timeEnd)
        db = formatData(db)
        if len(db.index) != 0:
            df_final = df_final.append(db)
            timeEnd = timeEnd - delta * 300
        else:
            break

    return df_final


def getFinalData(sym, period="DAY"):
    df_origin = getAllData(sym)

    if period == "DAY":
        return df_origin
    else:
        if period == "1WEEK":
            df = df_origin.groupby(pd.Grouper(freq='1W'))
        elif period == "2WEEK":
            df = df_origin.groupby(pd.Grouper(freq='2W'))
        elif period == "MONTH":
            df = df_origin.groupby(pd.Grouper(freq='M'))

        lst_C = []
        for i in df:
            dd = convertData(i)
            lst_C.append(dd)

        final = pd.concat(lst_C)
        final = final[::-1]

        return final


def convertData(tup):
    index = tup[0]
    dataf = tup[1]
    col = dataf.columns
    high = dataf['high'].max()
    low = dataf['low'].min()
    vol = dataf['volume'].sum()
    close = dataf['close'].iloc[-1]
    open = dataf['open'].iloc[0]
    df = pd.DataFrame([low, high, open, close, vol], columns=[index], index=col)
    return df.T


def getListCoins():
    url = "https://api.pro.coinbase.com/currencies"
    response = requests.get(url).json()
    newcoins = []
    dct = {}

    for i in range(len(response)):
        if response[i]['details']['type'] == 'crypto':
            s = response[i]['id'] + "-USD"
            newcoins.append(s)
            n = response[i]['name']
            dct[s] = n

    newcoins.sort()
    tup = tuple(newcoins)

    return tup, dct
