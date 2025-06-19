
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

def carregar_dados(tickers):
    dados = yf.download(tickers, period="1y", interval="1d")["Adj Close"]
    if isinstance(dados, pd.Series):
        dados = dados.to_frame()
    return dados.dropna()

def validar_tickers(dados):
    return list(dados.columns)

def obter_rentabilidade_fundo_google(nome_fundo):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        search_query = f"{nome_fundo} site:google.com"
        url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        spans = soup.find_all("span")
        for span in spans:
            texto = span.get_text()
            if "%" in texto and ("ano" in texto or "12 meses" in texto):
                return texto
        return "Rentabilidade não encontrada"
    except:
        return "Erro ao buscar fundo"

def obter_rendimento_renda_fixa(tipo, selic, ipca):
    # Retornos simulados com base no tipo de ativo
    if tipo == "Tesouro Selic":
        return round(selic - 0.15, 2)
    elif tipo == "Tesouro IPCA+":
        return round(ipca + 5.0, 2)
    elif tipo == "CDB":
        return round(selic - 0.1, 2)
    else:
        return None
