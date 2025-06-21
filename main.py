import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from io import BytesIO
import matplotlib.pyplot as plt
from pyportfolioopt import EfficientFrontier, risk_models, expected_returns, HRPOpt, CLA
from pyportfolioopt import objective_functions

# ========================== PARTE 1: MAPEAMENTO DE ATIVOS ==========================

ATIVOS_ACOES = [
    "ITUB3", "TOTS3", "MDIA3", "TAEE3", "BBSE3", "WEGE3", "PSSA3", "EGIE3", "B3SA3", "VIVT3",
    "AGRO3", "BBAS3", "PRIO3", "BPAC11", "SBSP3", "SAPR4", "CMIG3"
]
ATIVOS_FIIS = [
    "MXRF11", "XPIN11", "VISC11", "XPLG11", "HGLG11", "ALZR11", "BCRI11", "HGRU11", "VILG11",
    "VGHF11", "KNRI11", "HGRE11", "BRCO11", "HGCR11", "VGIA11", "MALL11", "BTLG11", "BTLG12",
    "XPML11", "LVBI11", "TRXF11"
]
ATIVOS_TESOURO = ["LFT", "LTN"]
ATIVOS_DEBENTURE = ["VINLAND INCENTIVADO INVESTIMENTO DEB INFRA ATIVO FIF RF CP LP RL"]
ATIVOS_ETF_INT = ["IVV", "QQQM", "QUAL", "XLRE"]

ALL_TICKERS = ATIVOS_ACOES + ATIVOS_FIIS + ATIVOS_TESOURO + ATIVOS_DEBENTURE + ATIVOS_ETF_INT

# Classes para classificação automática
def classificar_ativo(ticker):
    ticker = ticker.upper()
    if ticker in ATIVOS_ACOES:
        return "Ação"
    if ticker in ATIVOS_FIIS or ticker.endswith("11") or ticker.endswith("12"):
        return "FII"
    if ticker in ATIVOS_ETF_INT:
        return "ETF Internacional"
    if ticker in ATIVOS_TESOURO:
        return "Tesouro Direto"
    if ticker in ATIVOS_DEBENTURE or "DEB" in ticker:
        return "Debênture"
    return "Desconhecido"

def mapear_ativos(df):
    return df["Ticker"].apply(classificar_ativo)

# ========================== PARTE 2: FUNÇÕES DE DADOS ==========================

def get_historical_prices(tickers, start, end, moeda="BRL"):
    # yfinance suporta tickers internacionais, mas tickers brasileiros precisam de '.SA'
    data = {}
    for ticker in tickers:
        if classificar_ativo(ticker) == "Ação" or classificar_ativo(ticker) == "FII":
            yf_ticker = f"{ticker}.SA"
        elif classificar_ativo(ticker) == "ETF Internacional":
            yf_ticker = ticker  # Ex: IVV (EUA)
        else:
            continue  # Tesouro/Debênture: simula como renda fixa (aproximação)
        try:
            df = yf.download(yf_ticker, start=start, end=end)["Adj Close"]
            data[ticker] = df
        except Exception as e:
            st.warning(f"Erro ao baixar {ticker}: {e}")
    prices = pd.DataFrame(data)
    return prices

def get_simulated_fixed_income(tickers, start, end):
    # Simula Tesouro e Debênture com retornos fixos aproximados (anualizados)
    drange = pd.date_range(start, end, freq='B')
    data = {}
    for ticker in tickers:
        if "LFT" in ticker:
            rate = 0.1  # CDI
        elif "LTN" in ticker:
            rate = 0.11  # Pré
        elif "DEB" in ticker or "VINLAND" in ticker:
            rate = 0.115
        else:
            rate = 0.1
        # Monta uma curva acumulada (juros compostos)
        n = len(drange)
        vals = np.cumprod(np.ones(n) * (1 + rate/252))
        data[ticker] = pd.Series(vals, index=drange)
    return pd.DataFrame(data)

def unir_bases(prices_rv, prices_rf):
    df = pd.concat([prices_rv, prices_rf], axis=1)
    df = df.fillna(method="ffill").dropna()
    return df

# ========================== PARTE 3: OTIMIZAÇÃO DE CARTEIRA ==========================

def otimizar_markowitz(prices, pesos_iniciais=None):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    if pesos_iniciais is not None:
        ef.set_weights(pesos_iniciais)
    ef.add_objective(objective_functions.L2_reg, gamma=0.01)
    weights = ef.max_sharpe()
    cleaned = ef.clean_weights()
    return cleaned, ef

def otimizar_risk_parity(prices):
    S = risk_models.sample_cov(prices)
    w = 1 / np.diag(S)
    w = w / w.sum()
    return dict(zip(S.columns, w))

def otimizar_hrp(prices):
    hrp = HRPOpt(prices.pct_change().dropna())
    weights = hrp.optimize()
    return weights

# ========================== PARTE 4: ESTRATÉGIAS DE REBALANCEAMENTO ==========================

def buy_and_hold(pesos_atual):
    return pesos_atual.copy()  # Mantém igual

def cppi(valor_inicial, valor_minimo, m, rf_rate, prices):
    # CPPI: calcula alocação ideal para o portfolio
    cushion = (valor_inicial - valor_minimo) / valor_inicial
    risky_weight = m * cushion
    risky_weight = min(max(risky_weight, 0), 1)
    rf_weight = 1 - risky_weight
    ativos_rv = [c for c in prices.columns if classificar_ativo(c) in ["Ação", "FII", "ETF Internacional"]]
    ativos_rf = [c for c in prices.columns if classificar_ativo(c) in ["Tesouro Direto", "Debênture"]]
    pesos = {c: 0 for c in prices.columns}
    if ativos_rv:
        for c in ativos_rv:
            pesos[c] = risky_weight / len(ativos_rv)
    if ativos_rf:
        for c in ativos_rf:
            pesos[c] = rf_weight / len(ativos_rf)
    return pesos

def constant_mix(pesos_alvo):
    return pesos_alvo.copy()

# ========================== PARTE 5: SUGESTÃO DE APORTE ==========================

def sugerir_aporte(df, aporte, pesos_alvo):
    # Calcula o quanto colocar em cada ativo para caminhar para o ideal (sem venda, só compra)
    total_atual = df["Valor"].sum()
    total_final = total_atual + aporte
    valor_ideal_final = {k: v * total_final for k, v in pesos_alvo.items()}
    valores_atuais = dict(zip(df["Ticker"], df["Valor"]))
    valores_aporte = {}
    for ticker, ideal_final in valor_ideal_final.items():
        atual = valores_atuais.get(ticker, 0)
        aporte_necessario = max(0, ideal_final - atual)
        valores_aporte[ticker] = aporte_necessario
    # Normaliza para não ultrapassar o aporte disponível
    soma = sum(valores_aporte.values())
    if soma > 0:
        for k in valores_aporte:
            valores_aporte[k] = valores_aporte[k] * aporte / soma
    return valores_aporte

# ========================== PARTE 6: STREAMLIT APP ==========================

st.set_page_config(page_title="Balanceador de Carteira", layout="wide")

st.title("💼 Balanceador e Otimizador de Carteira de Investimentos")

with st.expander("ℹ️ O que este app faz? Clique para saber mais."):
    st.markdown("""
    Este aplicativo permite que você:
    - Cadastre sua carteira de Ações, FIIs, Tesouro Direto, Debêntures e ETFs Internacionais.
    - Classifique seus ativos automaticamente.
    - Otimize sua carteira por diferentes modelos (Fronteira Eficiente de Markowitz, Risk Parity, HRP).
    - Defina a estratégia de rebalanceamento (Buy and Hold, CPPI, Constant Mix).
    - Simule novos aportes e veja como distribuir seus investimentos.
    - Visualize gráficos e exporte seus resultados.

    **Aviso:** Os retornos de Tesouro e Debênture são simulados. Ativos internacionais NÃO consideram automaticamente o dólar; é recomendado atenção ao risco cambial.
    """)

# ======= Aba de Explicações =======
with st.sidebar.expander("📚 Entenda os Modelos"):
    st.markdown("""
    **Modelos de Otimização:**
    - **Markowitz (Fronteira Eficiente):** Busca a melhor relação risco/retorno para sua carteira.
    - **Risk Parity:** Tenta igualar a contribuição de risco de cada ativo.
    - **HRP (Hierarchical Risk Parity):** Agrupa ativos correlacionados e distribui risco entre clusters.

    **Modelos de Rebalanceamento:**
    - **Buy and Hold:** Não mexe na carteira após o investimento inicial.
    - **CPPI:** Protege parte do capital, expondo o restante ao risco.
    - **Constant Mix:** Mantém sempre o percentual desejado em cada classe de ativo.

    **Atenção:** ETFs como IVV, QQQM são cotados em dólar - há risco cambial.
    """)

# ======= Entrada da Carteira =======
st.header("1️⃣ Informe sua carteira atual")
st.markdown("Digite o ticker e o valor investido em cada ativo. Você pode adicionar linhas, remover ou editar.")

if "df_carteira" not in st.session_state:
    df_init = pd.DataFrame(
        {"Ticker": ["ITUB3", "MXRF11", "IVV", "LFT"],
         "Valor": [10000, 8000, 5000, 7000]}
    )
    st.session_state["df_carteira"] = df_init.copy()

df_carteira = st.data_editor(
    st.session_state["df_carteira"],
    num_rows="dynamic",
    use_container_width=True,
    key="edicao_carteira"
)
# Classifica cada linha
df_carteira["Classe"] = mapear_ativos(df_carteira)

# Checa tickers inválidos
tickers_validos = ALL_TICKERS
if not all([t in tickers_validos or classificar_ativo(t) != "Desconhecido" for t in df_carteira["Ticker"]]):
    st.warning("⚠️ Alguns tickers não são reconhecidos. Verifique a lista de ativos suportados no início do app.")

# Calcula os pesos atuais
total = df_carteira["Valor"].sum()
df_carteira["Peso Atual (%)"] = df_carteira["Valor"] / total * 100

st.write(df_carteira)

# ====================== OTIMIZAÇÃO ======================
st.header("2️⃣ Escolha o modelo de otimização de carteira")
modelo_otimizacao = st.selectbox("Modelo de Otimização", ["Markowitz (Fronteira Eficiente)", "Risk Parity", "Hierarchical Risk Parity (HRP)"])

st.markdown("Escolha o período histórico para calcular os retornos (recomendado: 3 anos).")
periodo = st.selectbox("Período", ["1 Ano", "2 Anos", "3 Anos", "5 Anos"], index=2)
anos = int(periodo.split()[0])
end = datetime.date.today()
start = end - datetime.timedelta(days=anos*365)

# Baixar cotação histórica
tickers_rv = [t for t in df_carteira["Ticker"] if classificar_ativo(t) in ["Ação", "FII", "ETF Internacional"]]
tickers_rf = [t for t in df_carteira["Ticker"] if classificar_ativo(t) in ["Tesouro Direto", "Debênture"]]

prices_rv = get_historical_prices(tickers_rv, start, end)
prices_rf = get_simulated_fixed_income(tickers_rf, start, end)
prices = unir_bases(prices_rv, prices_rf)

# Otimização
st.subheader("Alocação Ideal Sugerida")
if modelo_otimizacao == "Markowitz (Fronteira Eficiente)":
    pesos_otim, ef = otimizar_markowitz(prices)
elif modelo_otimizacao == "Risk Parity":
    pesos_otim = otimizar_risk_parity(prices)
elif modelo_otimizacao == "Hierarchical Risk Parity (HRP)":
    pesos_otim = otimizar_hrp(prices)
else:
    pesos_otim = dict(zip(prices.columns, [1/len(prices.columns)]*len(prices.columns)))

# Ajusta ordem dos ativos para tabela
pesos_ideais = pd.Series(pesos_otim).reindex(df_carteira["Ticker"]).fillna(0)
df_carteira["Peso Ideal (%)"] = pesos_ideais.values * 100
df_carteira["Desvio (%)"] = df_carteira["Peso Atual (%)"] - df_carteira["Peso Ideal (%)"]

# ====================== REBALANCEAMENTO ======================
st.header("3️⃣ Escolha o modelo de rebalanceamento")
modelo_reb = st.selectbox("Modelo de Rebalanceamento", ["Buy and Hold", "CPPI", "Constant Mix"])

if modelo_reb == "CPPI":
    st.markdown("Configure os parâmetros do CPPI:")
    valor_inicial = st.number_input("Valor inicial da carteira", min_value=1.0, value=float(total))
    valor_minimo = st.number_input("Valor mínimo garantido", min_value=0.0, value=float(total*0.8))
    multiplicador = st.slider("Multiplicador (m)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    rf_rate = 0.1
    pesos_rebal = cppi(valor_inicial, valor_minimo, multiplicador, rf_rate, prices)
elif modelo_reb == "Constant Mix":
    st.markdown("Defina os pesos alvo para cada ativo (em %):")
    pesos_mix = {}
    for t in df_carteira["Ticker"]:
        pesos_mix[t] = st.number_input(f"{t}", value=float(round(df_carteira[df_carteira['Ticker']==t]['Peso Ideal (%)'].values[0],2)), min_value=0.0, max_value=100.0)
    total_mix = sum(pesos_mix.values())
    if total_mix > 0:
        pesos_rebal = {k: v/total_mix for k, v in pesos_mix.items()}
    else:
        pesos_rebal = {k: 1/len(pesos_mix) for k in pesos_mix}
else:
    pesos_rebal = buy_and_hold(pesos_ideais)

# Normaliza para somar 1
soma_rebal = sum(pesos_rebal.values())
final_pesos = {k: v/soma_rebal for k, v in pesos_rebal.items()}

# ====================== SIMULAÇÃO DE APORTE ======================
st.header("4️⃣ Simule um novo aporte")
novo_aporte = st.number_input("Valor do novo aporte (R$)", min_value=0.0, value=0.0, step=100.0)
if novo_aporte > 0:
    sugestao_aporte = sugerir_aporte(df_carteira, novo_aporte, final_pesos)
    df_aporte = pd.DataFrame({
        "Ticker": list(sugestao_aporte.keys()),
        "Valor Sugerido": list(sugestao_aporte.values()),
    })
    st.success("Distribuição sugerida para o novo aporte, sem venda de ativos:")
    st.dataframe(df_aporte)

# ====================== VISUALIZAÇÕES ======================
st.header("5️⃣ Painel de Resultados e Visualização")

# Gráfico de pizza: atual vs ideal
col1, col2 = st.columns(2)
with col1:
    st.subheader("Alocação Atual")
    fig1, ax1 = plt.subplots()
    ax1.pie(df_carteira["Peso Atual (%)"], labels=df_carteira["Ticker"], autopct='%1.1f%%')
    st.pyplot(fig1)
with col2:
    st.subheader("Alocação Ideal")
    fig2, ax2 = plt.subplots()
    ax2.pie([final_pesos.get(t,0)*100 for t in df_carteira["Ticker"]], labels=df_carteira["Ticker"], autopct='%1.1f%%')
    st.pyplot(fig2)

# Tabela: atual, alvo, desvio
st.subheader("Tabela de Alocação")
df_out = df_carteira.copy()
df_out["Peso Rebalanceado (%)"] = [final_pesos.get(t,0)*100 for t in df_out["Ticker"]]
st.dataframe(df_out[["Ticker", "Classe", "Peso Atual (%)", "Peso Ideal (%)", "Peso Rebalanceado (%)", "Desvio (%)"]], use_container_width=True)

# Fronteira Eficiente (apenas Markowitz)
if modelo_otimizacao == "Markowitz (Fronteira Eficiente)":
    st.subheader("Fronteira Eficiente de Markowitz")
    try:
        cla = CLA(expected_returns.mean_historical_return(prices), risk_models.sample_cov(prices))
        cla.max_sharpe()
        frontier_returns, frontier_risks, _ = cla.efficient_frontier()
        fig, ax = plt.subplots()
        ax.plot(frontier_risks, frontier_returns, 'b--')
        ax.set_xlabel("Risco (Desvio Padrão)")
        ax.set_ylabel("Retorno Esperado")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Não foi possível plotar a fronteira eficiente: {e}")

# Exportação para CSV
st.header("6️⃣ Exportar Alocação Final")
csv = df_out[["Ticker", "Classe", "Peso Atual (%)", "Peso Ideal (%)", "Peso Rebalanceado (%)", "Desvio (%)"]].to_csv(index=False)
st.download_button("Baixar tabela em CSV", data=csv, file_name="nova_alocacao_carteira.csv", mime="text/csv")

# ====================== FIM DO APP ======================
st.info("""
Este app é apenas para fins educacionais e não constitui recomendação de investimento. Sempre consulte um profissional qualificado antes de tomar decisões financeiras.
""")
