This is a 3 state Gaussian HMM Regime Detection Model used for predicting the current market's regime (which is basically the hidden state of a market). It classifies the market into one of 3 regimes based on probabilities. These states/regimes are Calm, Transition, and Stress. FYI, these types of models are generally used to manage portfolios depending on the current, predicted regime.

Please note that HMM models are nowhere near perefect and we should use them mindfully. 

Please note that HMM models do not know how many days have passed in the current regime, meaning it could say this Regime may last for 60 days even though 59 days have already passed, so use that detail mindfully and as a rough estimate.

For this model I used three features (Daily returns, 10-days Volatility & 30-days Momentum) of SPY, TLT, GLD and DXY and the VIX level.

Below I wrote a brief explanation of what each regime indicates about each of the 13 features I mentioned.

## STRESS Regime ##
This regime represents periods of elevated market stress, fear, and uncertainty.

SPY Return
Stock returns are the weakest among all regimes and tend to be negative.

SPY Volatility
Market volatility is at its highest level.

SPY Momentum
Market momentum is negative.

TLT Return
Treasury returns are positive but lower than the returns from the calm level.

TLT Volatility
Treasury volatility is at its highest level.

TLT Momentum
Treasury momentum is positive.

DXY Return
The U.S. Dollar experiences its strongest performance.

DXY Volatility
Dollar volatility is elevated.

DXY Momentum
Dollar momentum is strongest(more positive) in this regime.

GLD Return
Gold performance is relatively weak.

GLD Volatility
Gold experiences its highest volatility.

GLD Momentum
Gold momentum is strongest in this regime.

VIX Level
The VIX reaches its highest level.


## TRANSITION Regime ##
This regime represents a middle ground between stress and calm conditions. Markets are generally improving, but uncertainty has not fully disappeared.

SPY Return
Stock returns are positive but remain below those observed in the calm regime.

SPY Volatility
Volatility declines significantly from stress levels but remains above calm levels.

SPY Momentum
Momentum is positive, suggesting improving market conditions and a recovering trend.

TLT Return
Treasury are negative and the lowest at this regime.

TLT Volatility
Treasury volatility falls from stress levels but remains elevated relative to calm periods.

TLT Momentum
Treasury momentum is the weakest in this regime.

DXY Return
The Dollar remains positive but is no longer experiencing the strong gains observed during stress periods.

DXY Volatility
Dollar volatility is relatively moderate from the other regimes.

DXY Momentum
Dollar momentum remains positive but weaker than during stress periods.

GLD Return
Gold posts positive returns.

GLD Volatility
Gold volatility noticeably decreases from the stress level.

GLD Momentum
Gold momentum remains positive but is considerably weaker than during stress and calm periods.

VIX Level
The VIX falls from stress levels but remains above calm-market readings.


## CALM Regime ##
This regime represents stable market conditions, lower uncertainty, and strong risk appetite.

SPY Return
Stock returns are strongest in this regime.

SPY Volatility
Market volatility is at its lowest level.

SPY Momentum
Momentum is strongest.

TLT Return
Treasury returns are the highest at this regime.

TLT Volatility
Treasury volatility is at its lowest level.

TLT Momentum
Treasury momentum remains positive but is less important than equity strength in defining the regime.

DXY Return
Dollar performance is weakest in this regime.

DXY Volatility
Dollar volatility is at its lowest level.

DXY Momentum
Dollar momentum is at its weakest in regime.

GLD Return
Gold returns are the highest in this regime.

GLD Volatility
Gold volatility reaches its lowest level.

GLD Momentum
Gold momentum remains positive and strong, but not as strong as in the stress regime.

VIX Level
The VIX is at its lowest level.
