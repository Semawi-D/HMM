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
tlt_mom = np.log(data_close["TLT"]).diff(30)

dxy_return = returns["DX-Y.NYB"]
dxy_vol = dxy_return.rolling(10).std()
dxy_mom = np.log(data_close["DX-Y.NYB"]).diff(30)

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


#Trainig different models and chosing the best one
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

best_score = -np.inf
for seed in range (50):
 model = GaussianHMM(n_components = 3, random_state = seed, n_iter = 1000, covariance_type = "full")
 model.fit(features_scaled)
 score = model.score(features_scaled)
 if score > best_score:
  best_score = score
  best_model = model

spy_vol_idx = list(features.columns).index("spy_vol")
order = np.argsort(best_model.means_[:, spy_vol_idx])
names = {order[0]: "CALM", order[1]: "TRANSITION", order[2]: "STRESS"}


#Predictions
state = best_model.predict(features_scaled)
todays_state = state[-1]
regime = names[todays_state]

probs =  best_model.predict_proba(features_scaled)
todays_probs = probs[-1]
todays_probs = (todays_probs.round(3))*100

tomorrows_probs = probs @ model.transmat_
regime_days = 1/(1-tomorrows_probs.max())
tomorrows_probs = tomorrows_probs.round(3)*100
features["states"] = state

print("   ")
print(f"N.B!! For the Regime probabilities below, CALM-index = {order[0]}, TRANSITION-index = {order[1]}, STRESS-index = {order[2]}.")
print(f"N.B!! The model doesn't know how many days have passed in the current regime.")
print("   ")
print(f"Today's Regime is {regime}, with a probability of {todays_probs.max()}%.")
print(f"This Regime might last for {regime_days.round(1)} days.")
print(f"Today's Regime probabilities are {todays_probs}.")
print(f"Tomorrow's Regime probabilities are {tomorrows_probs[-1]}.")
print("  ")