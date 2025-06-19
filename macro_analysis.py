
import requests

def obter_indicador_bcb(codigo_serie):
    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/ultimos/1?formato=json"
        r = requests.get(url)
        valor = float(r.json()[0]['valor'].replace(',', '.'))
        return valor
    except Exception as e:
        return None

def classificar_cenario_macro():
    indicadores = {
        "SELIC": obter_indicador_bcb(432),
        "Inflação (IPCA)": obter_indicador_bcb(433),
        "PIB": obter_indicador_bcb(7326),
        "Dólar": obter_indicador_bcb(1)
    }

    if indicadores["SELIC"] and indicadores["SELIC"] > 9:
        cenario = "Restritivo"
    elif indicadores["SELIC"] and indicadores["SELIC"] < 6.5 and indicadores["PIB"] and indicadores["PIB"] > 2:
        cenario = "Expansionista"
    else:
        cenario = "Neutro"

    return cenario, indicadores
