import yfinance as yf
import numpy as np
import pandas as pd

# Importing the Data
tickers = ["SPY", "QQQ", "TLT", "GLD"]
etf = yf.download(tickers, start="2010-01-01", auto_adjust = True)
vix = yf.download("^VIX", start="2010-01-01")
dxy = yf.download("DX-Y.NYB", start="2010-01-01")

# Creating the Features
returns = etf["Close"].pct_change()
spy_return = returns["SPY"]
spy_vol = spy_return.rolling(20).std()
spy_mom = spy_return.pct_change(60)

qqq_return = returns["QQQ"]
qqq_vol = qqq_return.rolling(20).std()
qqq_mom = qqq_return.pct_change(60)
qqq_vs_spy = etf["Close"]["QQQ"] /etf["Close"]["SPY"]

tlt_return = returns["TLT"]
tlt_vol = tlt_return.rolling(20).std()
tlt_mom = tlt_return.pct_change(60)

dxy_return = dxy["Close"].pct_change()
dxy_vol = dxy_return.rolling(20).std()
dxy_mom = dxy_return.pct_change(60)

vix_level = vix["Close"]

# Creating a Matrix/Table
X = pd.concat([spy_return, spy_vol, spy_mom, qqq_return, qqq_vol, qqq_mom, qqq_vs_spy, tlt_return, tlt_vol, tlt_mom, dxy_return, dxy_vol, dxy_mom, vix_level], axis = 1)
X.columns = ["spy_return", "spy_vol", "spy_mom", "qqq_return", "qqq_vol", "qqq_mom", "qqq_vs_spy", "tlt_return", "tlt_vol", "tlt_mom", "dxy_return", "dxy_vol", "dxy_mom", "vix_level"]
X = X.dropna()
print(X.head())
print(X.shape)

#Standardizing the Data/Matrix