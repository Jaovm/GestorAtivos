
import streamlit as st
from utils import carregar_dados, validar_tickers
from macro_analysis import classificar_cenario_macro
from optimizers import otimizar_carteira
from rebalance import simular_rebalanceamento
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide", page_title="Gestor de Carteira Inteligente")
st.title("📈 Gestor de Carteira Multimercado")

st.sidebar.header("Configuração da Carteira")
ativos_input = st.sidebar.text_area("Tickers (separados por vírgula)", "PETR4.SA, BOVA11.SA, IVVB11.SA")
pesos_input = st.sidebar.text_input("Pesos correspondentes (soma 100%)", "40, 30, 30")

modelo = st.sidebar.selectbox("Modelo de Alocação", ["Equal Weight", "Markowitz - Sharpe", "Risk Parity", "HRP", "Black-Litterman", "Sugestão Macro"])
rebalance = st.sidebar.selectbox("Modelo de Rebalanceamento", ["Buy and Hold", "Constant Mix", "CPPI"])
valor_aporte = st.sidebar.number_input("Novo aporte (opcional)", min_value=0.0, step=100.0)

if st.sidebar.button("Executar otimização"):
    try:
        tickers = [t.strip() for t in ativos_input.split(",")]
        pesos = [float(p)/100 for p in pesos_input.split(",")]

        dados_precos = carregar_dados(tickers)
        tickers_validos = validar_tickers(dados_precos)

        st.subheader("Cenário Macroeconômico Atual")
        cenario, indicadores = classificar_cenario_macro()
        st.write(f"**Cenário classificado como:** {cenario}")
        st.json(indicadores)

        st.subheader("Carteira Otimizada")
        pesos_otimizados, df_risco_retorno, fig_fronteira = otimizar_carteira(dados_precos, modelo, tickers, cenario)

        st.plotly_chart(px.pie(names=tickers, values=pesos, title="Alocação Atual"), use_container_width=True)
        st.plotly_chart(px.pie(names=tickers_validos, values=pesos_otimizados, title="Alocação Otimizada"), use_container_width=True)
        st.plotly_chart(fig_fronteira, use_container_width=True)

        st.subheader("Risco vs Retorno")
        st.plotly_chart(px.scatter(df_risco_retorno, x="Risco", y="Retorno", text="Ticker", title="Risco vs Retorno"), use_container_width=True)

        st.download_button("Exportar como CSV", df_risco_retorno.to_csv(index=False), file_name="analise_portfolio.csv")

        if valor_aporte > 0:
            st.subheader("Sugestão de Alocação do Novo Aporte")
            valor_por_ativo = [peso * valor_aporte for peso in pesos_otimizados]
            st.dataframe(pd.DataFrame({"Ticker": tickers_validos, "Aporte Sugerido (R$)": valor_por_ativo}))

        st.subheader("Simulação de Rebalanceamento")
        df_rebalanceado = simular_rebalanceamento(pesos, pesos_otimizados, modelo_rebalance=rebalance)
        st.dataframe(df_rebalanceado)

    except Exception as e:
        st.error(f"Erro na execução: {e}")
