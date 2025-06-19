
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, HierarchicalRiskParity
import plotly.graph_objects as go

def otimizar_carteira(precos, modelo, tickers, cenario):
    mu = expected_returns.mean_historical_return(precos)
    S = risk_models.CovarianceShrinkage(precos).ledoit_wolf()
    fig = go.Figure()
    df_risco_retorno = pd.DataFrame(columns=["Ticker", "Retorno", "Risco"])

    if modelo == "Equal Weight":
        pesos = [1/len(tickers)] * len(tickers)

    elif modelo == "Markowitz - Sharpe":
        ef = EfficientFrontier(mu, S)
        pesos = ef.max_sharpe()
        pesos = ef.clean_weights()

    elif modelo == "Risk Parity":
        inv_vol = 1 / np.sqrt(np.diag(S))
        pesos = inv_vol / np.sum(inv_vol)
        pesos = dict(zip(tickers, pesos))

    elif modelo == "HRP":
        hrp = HierarchicalRiskParity()
        hrp_weights = hrp.optimize(precos)
        pesos = hrp_weights

    else:
        pesos = [1/len(tickers)] * len(tickers)  # fallback

    retornos = mu.loc[tickers]
    riscos = np.sqrt(np.diag(S.loc[tickers, tickers]))

    df_risco_retorno["Ticker"] = tickers
    df_risco_retorno["Retorno"] = retornos.values
    df_risco_retorno["Risco"] = riscos

    fig.add_trace(go.Scatter(x=df_risco_retorno["Risco"], y=df_risco_retorno["Retorno"],
                             mode='markers+text', text=tickers))

    return list(pesos.values()), df_risco_retorno, fig
