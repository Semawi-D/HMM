import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


# Importing the Data
tickers = ["SPY", "QQQ", "TLT", "GLD"]
etf = yf.download(tickers, start="2010-01-01", auto_adjust = True)
vix = yf.download("^VIX", start="2010-01-01")
dxy = yf.download("DX-Y.NYB", start="2010-01-01")

# Creating the Features/Columns
close = etf["Close"]
returns = etf["Close"].pct_change()

spy_return = returns["SPY"]
spy_vol = spy_return.rolling(20).std()
spy_mom = close["SPY"].pct_change(60)

qqq_return = returns["QQQ"]
qqq_vol = qqq_return.rolling(20).std()
qqq_mom = close["QQQ"].pct_change(60)

tlt_return = returns["TLT"]
tlt_vol = tlt_return.rolling(20).std()
tlt_mom = close["TLT"].pct_change(60)

dxy_return = dxy["Close"].pct_change()
dxy_vol = dxy_return.rolling(20).std()
dxy_mom = dxy["Close"].squeeze().pct_change(60)

vix_level = vix["Close"]

# Creating a Matrix/Table
X = pd.concat([spy_return, spy_vol, spy_mom, qqq_return, qqq_vol, qqq_mom, tlt_return, tlt_vol, tlt_mom, dxy_return, dxy_vol, dxy_mom, vix_level], axis = 1)
X.columns = ["spy_return", "spy_vol", "spy_mom", "qqq_return", "qqq_vol", "qqq_mom", "tlt_return", "tlt_vol", "tlt_mom", "dxy_return", "dxy_vol", "dxy_mom", "vix_level"]
X = X.dropna()
print(X.head())
print(X.shape)

#Standardizing the Data/Matrix and training the Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = GaussianHMM(n_components = 3, random_state = 42, n_iter = 1000, covariance_type = "full")
model.fit(X_scaled)
states = model.predict(X_scaled)
X["state"] = states
print(X.groupby("state").mean())
#STATE 0 = High Vol Bull/Calm
#STATE 1 = Bull/Calm
#STATE 2 = Bear/Stress
probs =  model.predict_proba(X_scaled)[-1]
todays_prob = probs.max().round(3)
if probs.argmax() == 0:
 print("Today's state is: High Vol BULL ", todays_prob)
elif probs.argmax() == 1:
 print("Today's state is: BULL ", todays_prob)
elif probs.argmax() == 2:
 print("Today's state is: Bear ", todays_prob)
print("Today's regime probablities: ", probs.round(3))

tomorrows_probs = probs @ model.transmat_
tomorrows_prob = tomorrows_probs.max().round(3)
if tomorrows_probs.argmax() == 0:
 print("Tomorrow's state is: High Vol BULL ", tomorrows_prob)
elif tomorrows_probs.argmax() == 1:
 print("Tomorrow's state is: BULL ", tomorrows_prob)
elif tomorrows_probs.argmax() == 2:
 print("Tomorrow's state is: Bear ", tomorrows_prob)
print("Tomorrow's regime probablities: ", tomorrows_probs.round(3))


