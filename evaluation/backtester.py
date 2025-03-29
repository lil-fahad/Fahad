
import pandas as pd
import numpy as np

class AdvancedBacktester:
    def __init__(self, initial_cash=100000, transaction_cost=0.001):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost

    def run_walkforward(self, df, prediction_col, price_col='Close', window=60):
        df = df.copy()
        df['Position'] = 0
        df['Returns'] = df[price_col].pct_change().shift(-1)
        df['Signal'] = np.where(df[prediction_col].shift(1) > df[price_col].shift(1), 1, -1)

        for i in range(window, len(df)):
            df.loc[df.index[i], 'Position'] = df['Signal'].iloc[i]

        df['Strategy_Returns'] = df['Returns'] * df['Position']
        df['Strategy_Returns'] -= self.transaction_cost * np.abs(df['Position'].diff().fillna(0))

        df['Equity'] = (1 + df['Strategy_Returns']).cumprod() * self.initial_cash
        df['BuyHold'] = (1 + df['Returns']).cumprod() * self.initial_cash
        return df

    def calculate_metrics(self, df):
        strategy = df['Strategy_Returns'].dropna()
        buyhold = df['Returns'].dropna()

        def sharpe(returns):
            return np.sqrt(252) * returns.mean() / returns.std() if returns.std() else 0

        def sortino(returns):
            downside = returns[returns < 0]
            return np.sqrt(252) * returns.mean() / downside.std() if downside.std() else 0

        max_dd = (df['Equity'].cummax() - df['Equity']).max()
        metrics = {
            'Total Return (%)': (df['Equity'].iloc[-1] / df['Equity'].iloc[0] - 1) * 100,
            'Buy & Hold Return (%)': (df['BuyHold'].iloc[-1] / df['BuyHold'].iloc[0] - 1) * 100,
            'Sharpe Ratio': sharpe(strategy),
            'Sortino Ratio': sortino(strategy),
            'Max Drawdown ($)': max_dd
        }
        return metrics
