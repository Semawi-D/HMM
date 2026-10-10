#Importing the libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


# Importing the Data
tickers = ["SPY", "QQQ", "TLT", "GLD", "^VIX", "DX-Y.NYB"]
raw_data = yf.download(tickers, start="2010-01-01", auto_adjust = True)

data_close = raw_data["Close"]
data_close = data_close[data_close["SPY"].notna()]
data_close = data_close.ffill().dropna()


# Creating the Features/Columns
returns = np.log(data_close.drop(columns = "^VIX")).diff()

spy_return = returns["SPY"]
spy_vol = spy_return.rolling(10).std()
spy_mom = np.log(data_close["SPY"]).diff(30)

tlt_return = returns["TLT"]
tlt_vol = tlt_return.rolling(10).std()
tlt_mom = np.log(data_close["QQQ"]).diff(30)

dxy_return = returns["DX-Y.NYB"]
dxy_vol = dxy_return.rolling(10).std()
dxy_mom = np.log(data_close["TLT"]).diff(30)

gld_return = returns["GLD"]
gld_vol = gld_return.rolling(10).std()
gld_mom = np.log(data_close["GLD"]).diff(30)

vix_level = np.log(data_close["^VIX"])


# Creating a Matrix/Table
features = pd.DataFrame(index = data_close.index)
features["spy_return"] = spy_return
features["spy_vol"] = spy_vol
features["spy_mom"] = spy_mom
features["tlt_return"] = tlt_return
features["tlt_vol"] = tlt_vol
features["tlt_mom"] = tlt_mom
features["dxy_return"] = dxy_return
features["dxy_vol"] = dxy_vol
features["dxy_mom"] = dxy_mom
features["gld_return"] = gld_return
features["gld_vol"] = gld_vol
features["gld_mom"] = gld_mom
features["vix_level"] = vix_level
features = features.ffill().dropna()


#Standardizing the Matrix and training the Model
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = GaussianHMM(n_components = 3, random_state = 42, n_iter = 1000, covariance_type = "full")
model.fit(features_scaled)

spy_vol_idx = list(features.columns).index("spy_vol")
order = np.argsort(model.means_[:, spy_vol_idx])
names = {order[0]: "CALM", order[1]: "TRANSITION", order[2]: "STRESS"}


#Prediction
state = model.predict(features_scaled)
todays_state = state[-1]
regime = names[todays_state]

probs =  model.predict_proba(features_scaled)
todays_probs = probs[-1]
todays_probs = (todays_probs.round(3))*100

tomorrows_probs = probs @ model.transmat_
regime_days = 1/(1-tomorrows_probs.max())
tomorrows_probs = tomorrows_probs.round(3)*100
print("   ")
print(f"N.B!! For the regime probabilities below, CALM-index = {order[0]}, TRANSITION-index = {order[1]}, STRESS-index = {order[2]}.")
print(f"N.B!! The model doesn't know how many days the market has been in the current regime.")
print("   ")
print(f"Today's regime is {regime}, with a probability of {todays_probs.max()}%.")
print(f"The market might stay in this regime for {regime_days.round(1)} days.")
print(f"Today's Regime probabilities are {todays_probs}.")
print(f"Tomorrow's Regime probabilities are {tomorrows_probs[-1]}.")

