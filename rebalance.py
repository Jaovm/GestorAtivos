
import pandas as pd

def simular_rebalanceamento(pesos_atuais, pesos_otimizados, modelo_rebalance):
    df = pd.DataFrame({
        "Peso Atual": pesos_atuais,
        "Peso Ideal": pesos_otimizados
    })
    df["Diferença"] = df["Peso Ideal"] - df["Peso Atual"]
    return df
