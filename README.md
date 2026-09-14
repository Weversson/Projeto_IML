# Previsao de Obitos Mensais no Brasil

Previsao do numero de obitos mensais registrados no Sistema de Informacao sobre Mortalidade (SIM/DATASUS), agregados por mes e unidade federativa, utilizando 30 anos de historico.

## 1. O problema e por que importa

O Brasil registra cerca de 1,5 milhao de obitos por ano. Prever o volume mensal de obitos permite ao Ministerio da Saude dimensionar leitos, farmacos, campanhas e vigilancia em saude de forma antecipada.

Uma previsao confiavel funciona como linha de base para detectar mudancas reais: surtos, desastres e outros eventos de saude publica aparecem como desvios entre o previsto e o observado.

## 2. Os dados

Fonte: **DATASUS** (Departamento de Informatica do SUS), sistema SIM.

Portal: https://dadosabertos.saude.gov.br/dataset/sim

Periodo: **1996 a 2025** (30 anos), obtidos em JSON diretamente do S3 oficial do Governo Federal. Volume total: ~40 GB (30 arquivos, um por ano, ~1,5 milhao de registros por ano).

Para este trabalho, os obitos foram agregados em **series mensais por unidade federativa**:

- 9.420 pontos (27 UFs x 360 meses dariam 9.720; faltam os 12 meses de 2012 em 25 UFs, ver secao 7)
- Series nacional: pico de **207.106 obitos em marco de 2021** (auge da COVID-19)
- Media de ~96.700 obitos por mes no periodo (~99.500 sem contar 2012, que esta incompleto)

## 3. Preprocessamento

Extracao e concatenacao dos ZIPs (particoes por ano), seguida da agregacao por (ano, mes, UF). Para cada linha (mes-UF) foram derivadas features:

- SENO/COS: posicao do mes no ciclo anual (sazonalidade)
- TEND: indice temporal por UF (tendencia)
- LAG_1, LAG_2, LAG_6, LAG_12: obitos de 1, 2, 6 e 12 meses atras
- ROLL_3, ROLL_12: medias moveis de 3 e 12 meses
- COVID: indicador do periodo de pandemia (mar/2020 a dez/2022)
- POS: indicador do periodo pos-pandemia (2023 em diante)

Baselines: prever cada mes como o mesmo mes do ano anterior (sazonal ingenuo) e repetir o valor do mes anterior (ingenuo), que usa a mesma informacao do LAG_1.

## 4. Modelos testados

Modelos testados em ordem cronológica ao longo do desenvolvimento do projeto, com o que foi tentado e o motivo da falha (ou sucesso):

| Ordem | Modelo | Descrição | Por que deu errado / Resultado |
|---|---|---|---|
| **1º** | **K-Means (Clustering)** | Agrupamento não supervisionado dos microdados em 6 clusters por perfil demográfico (idade, sexo, raça, escolaridade). | **Por que deu errado:** Não resolvia uma tarefa preditiva acionável. Apenas descreveu dados óbvios (ex: jovens morrem mais de causas externas), sem utilidade prática para planejamento. |
| **2º** | **Random Forest e Gradient Boosting** | Classificação binária para prever se o óbito era evitável ou não a partir do perfil demográfico, sem usar a causa da morte para não trapacear. | **Por que deu errado:** Teto nas features. Parou em AUC de 0,61 e acurácia de 58% (mesmo com 3 milhões de dados). Perfil demográfico isolado não explica se o óbito era evitável sem exame clínico. |
| **3º** | **Classificador Multiclasse** | Tentativa de prever qual das 21 categorias de doenças da CID-10 causou a morte a partir dos dados demográficos. | **Por que deu errado:** Acurácia de apenas 16%. Ausência de correlação demográfica suficiente para adivinhar a patologia específica. |
| **4º** | **Série Temporal Nacional Única** | Regressão temporal agregando todos os óbitos do Brasil em uma linha do tempo única nacional (324 meses). | **Por que deu errado:** Pior que a linha de base ingênua (MAE de 6.140 vs 5.568 do baseline sazonal). Ignorava as dinâmicas regionais e tinha pouca massa de dados de treino. |
| **5º** | **LightGBM Global por UF** *(Final)* | Gradient Boosted Trees treinado com as 27 UFs simultaneamente (9.420 pontos), com a UF como código numérico do IBGE e 13 features temporais/sazonais. | **Deu certo:** MAE de 3.591 óbitos/mês (**redução de 35,5% do erro** frente ao baseline sazonal e **37,4%** frente ao mês anterior), errando apenas 2,8% do volume mensal. |

### Configuração do Modelo Final (LightGBM Global)
- **Hiperparâmetros:** `n_estimators=800`, `learning_rate=0.05`, `num_leaves=127`, `max_depth=10`, `min_child_samples=30`, `subsample=0.8`, `colsample_bytree=0.7`, `random_state=42`. O `subsample=0.8` não tem efeito na prática: o LightGBM só sorteia linhas quando `subsample_freq` é maior que 0, e ele ficou no padrão (0).
- **Divisão temporal, sem embaralhamento:** Treino de jan/1997 a dez/2023 (8.448 linhas) e teste de jan/2024 a dez/2025 (648 linhas, 24 meses x 27 UFs). O ano de 1996 entra apenas no cálculo dos lags.

## 5. Resultados

| Modelo | MAE | RMSE |
|--------|-----|------|
| Baseline sazonal | 5.568 | 6.673 |
| Baseline mes anterior | 5.738 | 7.720 |
| **LightGBM global** | **3.591** | 4.412 |

O modelo reduziu o MAE em **35,5%** frente ao baseline sazonal e em **37,4%** frente ao baseline do mes anterior, com erro medio de **2,8% do volume mensal** de obitos. Essas metricas sao calculadas na serie nacional (soma das previsoes das 27 UFs em cada mes do teste). Por UF, o modelo venceu o baseline sazonal em **18 das 27 UFs** e perdeu em MG, MT, AM, MS, SE, RO, AC, AP e RR (detalhes na secao 3 do notebook).

![Previsao de obitos mensais](figuras/previsao_obitos.png)

## 6. O que nao funcionou

**Classificacao binaria evitavel/nao-evitavel**: tentamos prever evitabilidade a partir de dados demograficos (idade, sexo, raca, escolaridade, local do obito). Random Forest e Gradient Boosting pararam em accuracy de 58% e AUC de 0,61, com teto claro nas features disponiveis: sem conhecer a causa do obito, o perfil demografico nao explica evitabilidade. A tarefa foi abandonada.

**Classificacao multiclasse (21 capitulos CID-10)**: accuracy de 16%. O modelo nao consegue prever a causa da morte so com dados demograficos. Abandonada.

**Modelo de serie nacional unico**: a serie agregada do Brasil em um unico modelo resultou pior que o baseline (MAE 6.140 vs 5.568). O modelo global por UF resolveu o problema: usa mais dados e captura tendencias regionais.

**Definicao ampla de causas evitaveis**: a primeira versao do criterio (todos os capitulos A-J, V-Y) marcou 67% dos obitos como evitaveis, inutil para qualquer separacao. Descartada.

## 7. Limitacoes

O baseline sazonal e forte em series estaveis como esta; o ganho de 35,5% vem sobretudo dos estados com tendencia propria no periodo.

O modelo assume que os padroes demograficos e epidemiologicos do passado se mantem. Pandemias e desastres nao podem ser antecipados.

Os dados de 2025 ainda sao preliminares e podem ser revisados pelo DATASUS.

O erro agregado nacional (3.591 obitos/mes) mascara divergencias maiores em estados com volume pequeno de obitos. Na soma nacional, erros de sinais opostos entre UFs se compensam: somando o MAE de cada UF, o erro chega a 6.090 obitos/mes. Em RR, o erro medio e de 15,6% do volume mensal, e o modelo perde para o baseline sazonal em 9 UFs.

A serie agregada nao tem 2012 para 25 das 27 UFs (apenas SP e TO). O arquivo de 2012 existe no DATASUS, entao a falha ocorreu no processamento. Nessas UFs, os lags de 2013 ficam deslocados (o mes anterior de jan/2013 vira dez/2011), o que afeta o treino, mas nao diretamente o periodo de teste.

## Como reproduzir

```bash
git clone https://github.com/Weversson/Projeto_IML.git
cd Projeto_IML
python -m venv .venv

# No Linux / macOS:
source .venv/bin/activate
pip install -r requirements.txt

# No Windows (Git Bash ou PowerShell):
source .venv/Scripts/activate   # no Git Bash
# ou: .venv\Scripts\activate     # no PowerShell/CMD
pip install -r requirements.txt
```

Execucao de demonstracao rapida via linha de comando:

```bash
# Prever obitos para um estado e mes especificos:
python prever.py --uf SP --ano 2024 --mes 7

# Visualizar resumo de avaliacao nacional no periodo de teste:
python prever.py --resumo
```

Abrir `notebooks/notebook_mortalidade.ipynb` no Colab ou localmente. O notebook baixa a serie agregada e o modelo treinado diretamente do GitHub Releases:

https://github.com/Weversson/Projeto_IML/releases

Na secao 3 do notebook, a celula "Treinamento do modelo" treina o LightGBM de novo e confirma que as previsoes sao identicas as do modelo publicado (recomendamos rodar no Colab). A celula seguinte, "Comparacao detalhada", calcula as duas linhas de base, o erro por UF e a importancia das features.

Uso do modelo em outro ambiente (a partir da raiz do repositorio, reaproveitando as funcoes do `prever.py`):

```python
import joblib
from prever import garantir_arquivos, preparar_features

modelo_path, csv_path = garantir_arquivos()  # baixa o modelo e a serie, se faltarem
modelo = joblib.load(modelo_path)
df = preparar_features(csv_path)  # monta as 13 features como no treino

fc = ['UF', 'SENO', 'COS', 'TEND', 'COVID', 'POS', 'LAG_1', 'LAG_2', 'LAG_6', 'LAG_12', 'ROLL_3', 'ROLL_12', 'MES']
df['PREV'] = modelo.predict(df[fc])
print(df[['ANO', 'MES', 'UF', 'OBITOS', 'PREV']].tail())
```