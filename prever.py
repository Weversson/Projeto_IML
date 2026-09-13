"""
Script de Previsao de Obitos Mensais no Brasil (LightGBM)
Projeto IML - SIM / DATASUS (1996-2025)

Uso:
    python prever.py
    python prever.py --uf SP --ano 2024 --mes 7
    python prever.py --uf RJ --ano 2025 --mes 3
    python prever.py --resumo
"""

import argparse
import os
import sys
import urllib.request
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

# Mapeamento de siglas das Unidades Federativas para codigos IBGE
UF_SIGLA_COD = {
    'RO': 11, 'AC': 12, 'AM': 13, 'RR': 14, 'PA': 15, 'AP': 16, 'TO': 17,
    'MA': 21, 'PI': 22, 'CE': 23, 'RN': 24, 'PB': 25, 'PE': 26, 'AL': 27,
    'SE': 28, 'BA': 29, 'MG': 31, 'ES': 32, 'RJ': 33, 'SP': 35, 'PR': 41,
    'SC': 42, 'RS': 43, 'MS': 50, 'MT': 51, 'GO': 52, 'DF': 53
}
UF_COD_SIGLA = {v: k for k, v in UF_SIGLA_COD.items()}

NOME_UF = {
    'SP': 'Sao Paulo', 'RJ': 'Rio de Janeiro', 'MG': 'Minas Gerais', 'RS': 'Rio Grande do Sul',
    'PR': 'Parana', 'BA': 'Bahia', 'SC': 'Santa Catarina', 'GO': 'Goias', 'PE': 'Pernambuco',
    'CE': 'Ceara', 'PA': 'Para', 'MA': 'Maranhao', 'ES': 'Espirito Santo', 'PB': 'Paraiba',
    'AM': 'Amazonas', 'MT': 'Mato Grosso', 'RN': 'Rio Grande do Norte', 'AL': 'Alagoas',
    'PI': 'Piaui', 'DF': 'Distrito Federal', 'MS': 'Mato Grosso do Sul', 'SE': 'Sergipe',
    'RO': 'Rondonia', 'TO': 'Tocantins', 'AC': 'Acre', 'AP': 'Amapa', 'RR': 'Roraima'
}

RELEASE_BASE = 'https://github.com/Weversson/Projeto_IML/releases/download/v1.0'
CSV_PATH_DEFAULT = 'serie_obitos_uf.csv'
CSV_PATH_NOTEBOOK = os.path.join('notebooks', 'serie_obitos_uf.csv')
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'previsao_obitos_uf_lgbm.pkl')


def garantir_arquivos():
    """Garante a presenca da serie historica e do modelo treinado."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Verificar modelo
    if not os.path.exists(MODEL_PATH):
        # Tenta verificar se esta dentro de notebooks/models
        alt_model = os.path.join('notebooks', 'models', 'previsao_obitos_uf_lgbm.pkl')
        if os.path.exists(alt_model):
            return alt_model, garantir_csv()
        print("[+] Baixando modelo treinado do GitHub Releases...")
        urllib.request.urlretrieve(f"{RELEASE_BASE}/previsao_obitos_uf_lgbm.pkl", MODEL_PATH)
        print("[v] Modelo baixado com sucesso.")

    csv_path = garantir_csv()
    return MODEL_PATH, csv_path


def garantir_csv():
    """Verifica e retorna o caminho do arquivo de dados agregados."""
    if os.path.exists(CSV_PATH_DEFAULT):
        return CSV_PATH_DEFAULT
    if os.path.exists(CSV_PATH_NOTEBOOK):
        return CSV_PATH_NOTEBOOK
    print("[+] Baixando serie historica do GitHub Releases...")
    urllib.request.urlretrieve(f"{RELEASE_BASE}/serie_obitos_uf.csv", CSV_PATH_DEFAULT)
    print("[v] Serie historica baixada com sucesso.")
    return CSV_PATH_DEFAULT


def preparar_features(csv_path):
    """Carrega a serie e constroi as 13 features exatamente como no treino."""
    df = pd.read_csv(csv_path)
    df['DATA'] = pd.to_datetime(df['ANO'].astype(str) + '-' + df['MES'].astype(str) + '-01')
    df = df.sort_values(['UF', 'DATA']).reset_index(drop=True)

    df['SENO'] = np.sin(2 * np.pi * df['MES'] / 12)
    df['COS'] = np.cos(2 * np.pi * df['MES'] / 12)
    df['TEND'] = df.groupby('UF').cumcount()
    df['LAG_1'] = df.groupby('UF')['OBITOS'].shift(1)
    df['LAG_2'] = df.groupby('UF')['OBITOS'].shift(2)
    df['LAG_6'] = df.groupby('UF')['OBITOS'].shift(6)
    df['LAG_12'] = df.groupby('UF')['OBITOS'].shift(12)
    df['ROLL_3'] = df.groupby('UF')['OBITOS'].transform(lambda s: s.rolling(3).mean()).shift(1)
    df['ROLL_12'] = df.groupby('UF')['OBITOS'].transform(lambda s: s.rolling(12).mean()).shift(1)
    df['COVID'] = ((df['DATA'] >= '2020-03-01') & (df['DATA'] <= '2022-12-01')).astype(float)
    df['POS'] = (df['DATA'] >= '2023-01-01').astype(float)
    df['NAIVE'] = df.groupby('UF')['OBITOS'].transform(lambda s: s.shift(12))

    df = df.dropna(subset=['NAIVE', 'LAG_1']).copy()
    return df


def normalizar_uf(uf_arg):
    """Converte entrada do usuario para codigo IBGE e Sigla."""
    uf_str = str(uf_arg).strip().upper()
    if uf_str in UF_SIGLA_COD:
        return UF_SIGLA_COD[uf_str], uf_str
    if uf_str.isdigit() and int(uf_str) in UF_COD_SIGLA:
        cod = int(uf_str)
        return cod, UF_COD_SIGLA[cod]
    raise ValueError(f"UF invalida: '{uf_arg}'. Use a sigla (ex: SP, RJ, MG) ou o codigo IBGE.")


def prever_ponto(model, df, uf_cod, uf_sigla, ano, mes):
    """Realiza a previsao para uma UF, ano e mes especificos."""
    data_alvo = pd.to_datetime(f"{ano}-{mes:02d}-01")
    filtro = (df['UF'] == uf_cod) & (df['DATA'] == data_alvo)
    
    fc = ['UF', 'SENO', 'COS', 'TEND', 'COVID', 'POS', 'LAG_1', 'LAG_2', 'LAG_6', 'LAG_12', 'ROLL_3', 'ROLL_12', 'MES']
    subset = df[filtro]

    if subset.empty:
        print(f"\n[!] Data {mes:02d}/{ano} nao encontrada na serie historica para {uf_sigla}.")
        print("    O periodo disponivel para teste vai de 1997 ate 2025.")
        return

    row = subset.iloc[0]
    X = subset[fc]
    pred = float(model.predict(X)[0])
    real = float(row['OBITOS'])
    naive = float(row['NAIVE'])

    erro_lgbm = abs(real - pred)
    pct_lgbm = (erro_lgbm / real) * 100
    erro_naive = abs(real - naive)
    pct_naive = (erro_naive / real) * 100
    ganho_pct = ((erro_naive - erro_lgbm) / erro_naive) * 100

    nome = NOME_UF.get(uf_sigla, uf_sigla)
    print("\n" + "=" * 60)
    print(f" PREVISAO DE OBITOS - {nome} ({uf_sigla}) - {mes:02d}/{ano}")
    print("=" * 60)
    print(f" * Obitos Reais Registrados:        {int(real):>10,}".replace(',', '.'))
    print(f" * Previsao LightGBM:               {int(round(pred)):>10,}".replace(',', '.'))
    print(f"   -> Erro Absoluto do Modelo:      {int(round(erro_lgbm)):>10,} ({pct_lgbm:.2f}%)".replace(',', '.'))
    print(f" * Linha de Base (Mesmo mes ano ant):{int(naive):>10,}".replace(',', '.'))
    print(f"   -> Erro Absoluto da Base:        {int(round(erro_naive)):>10,} ({pct_naive:.2f}%)".replace(',', '.'))
    print("-" * 60)
    if erro_lgbm < erro_naive:
        print(f" [VITORIA] LightGBM superou o baseline com erro {ganho_pct:.1f}% menor!")
    else:
        print(" [INFO] Neste mes atipico, o baseline sazonal esteve mais proximo.")
    print("=" * 60 + "\n")


def exibir_resumo_nacional(model, df):
    """Exibe a avaliacao comparativa agregada no periodo de teste (2024-2025)."""
    fc = ['UF', 'SENO', 'COS', 'TEND', 'COVID', 'POS', 'LAG_1', 'LAG_2', 'LAG_6', 'LAG_12', 'ROLL_3', 'ROLL_12', 'MES']
    df['PREV'] = model.predict(df[fc])

    te = df[df['DATA'] >= '2024-01-01']
    real_nac = te.groupby('DATA')['OBITOS'].sum()
    prev_nac = te.groupby('DATA')['PREV'].sum()
    base_nac = te.groupby('DATA')['NAIVE'].sum()

    mae_base = abs(real_nac - base_nac).mean()
    mae_lgbm = abs(real_nac - prev_nac).mean()
    reduc = (1 - mae_lgbm / mae_base) * 100
    media_mensal = real_nac.mean()
    pct_erro = (mae_lgbm / media_mensal) * 100

    print("\n" + "=" * 65)
    print(" RESUMO DA AVALIACAO NACIONAL (TESTE: 2024 - 2025 / 24 MESES)")
    print("=" * 65)
    print(f" Volume Medio de Obitos/Mes no Brasil: {int(media_mensal):>10,}".replace(',', '.'))
    print(f" Baseline Sazonal Ingenuo (MAE):       {int(mae_base):>10,} obitos/mes".replace(',', '.'))
    print(f" LightGBM Global (MAE):                {int(mae_lgbm):>10,} obitos/mes".replace(',', '.'))
    print("-" * 65)
    print(f" Reducao de Erro Absoluto (MAE):       {reduc:>10.1f}%")
    print(f" Erro Medio em Relacao ao Volume:      {pct_erro:>10.2f}%")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Previsao de Obitos Mensais no Brasil (LightGBM)")
    parser.add_argument('--uf', type=str, default='SP', help='Sigla da UF (ex: SP, RJ, MG) ou codigo IBGE')
    parser.add_argument('--ano', type=int, default=2024, help='Ano para previsao (ex: 2024)')
    parser.add_argument('--mes', type=int, default=7, help='Mes de 1 a 12 (ex: 7)')
    parser.add_argument('--resumo', action='store_true', help='Exibe metricas gerais do periodo de teste nacional')
    args = parser.parse_args()

    model_path, csv_path = garantir_arquivos()
    print(f"Carregando modelo: {model_path}")
    model = joblib.load(model_path)
    df = preparar_features(csv_path)

    if args.resumo:
        exibir_resumo_nacional(model, df)
    else:
        uf_cod, uf_sigla = normalizar_uf(args.uf)
        prever_ponto(model, df, uf_cod, uf_sigla, args.ano, args.mes)


if __name__ == '__main__':
    main()
