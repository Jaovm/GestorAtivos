
import yfinance as yf
import pandas as pd

def carregar_dados(tickers):
    dados = yf.download(tickers, period="1y", interval="1d")["Adj Close"]
    if isinstance(dados, pd.Series):
        dados = dados.to_frame()
    return dados.dropna()

def validar_tickers(dados):
    return list(dados.columns)
