# Previsao de Mortalidade Evitavel no SUS

Classificacao de obitos registrados no Sistema de Informacao sobre Mortalidade (SIM) como **evitaveis** ou **nao-evitaveis**, utilizando exclusivamente dados demograficos do falecido.

## 1. O problema e por que importa

Cada ano, o Brasil registra mais de 1,5 milhao de obitos no SIM. Uma parte significativa dessas mortes poderia ser evitada com politicas publicas adequadas: vacinacao, acesso a tratamento, seguranca no transito, atendimento materno-infantil.

Identificar **quais perfis demograficos** estao mais associados a mortes evitaveis permite direcionar recursos de saude de forma mais eficiente. O modelo treinado neste projeto faz essa classificacao: dado o perfil de um falecido (idade, sexo, raca, escolaridade, local do obito), ele prevê se a morte era evitavel ou nao.

## 2. Os dados

Fonte: **DATASUS** (Departamento de Informatica do SUS), sistema SIM (Sistema de Informacao sobre Mortalidade).

Portal: https://dadosabertos.saude.gov.br/dataset/sim

Periodo: **1996 a 2025** (30 anos). Formato: JSON, obtidos diretamente do S3 oficial do Governo Federal.

Volume:
- 30 arquivos JSON, um por ano
- Total extraido: ~40 GB
- Registros por ano recente: ~1,5 milhao de obitos
- Total no dataset de treino (2024+2025): **3.026.337 registros**

Cada registro contem 86 colunas originais, incluindo dados socioeonomicos, local de residencia e ocorrencia, causa basica do obito (CID-10) e declaracao de obito.

## 3. Preprocessamento

**Filtragem**: removidos registros com campos obrigatorios invalidos (idade nao numerica, sexo fora de 1-2, raca/cor fora de 1-5, causa basica com menos de 3 caracteres). Restaram 1.510.970 registros validos para 2024.

**Engenharia de features**: 11 variaveis derivadas dos campos originais:

- IDADE (numerica)
- SEXO (binaria: 1=masculino, 2=feminino)
- RACACOR (ordinal: 1=branca a 5=indigena)
- ESTCIV (ordinal: estado civil)
- ESC2010 (ordinal: anos de estudo)
- LOCOCOR (ordinal: local do obito)
- UF (categorical: unidade federativa)
- MES_OBITO (ordinal: mes do obito)
- HORA (numerica: hora do obito)
- FAIXA_ETARIA (ordinal: 8 faixas etarias)
- ANO (numerica: ano do registro)

**Target binario**: a causa basica do obito (CAUSABAS) foi classificada como **evitavel** ou **nao-evitavel** com base em criterios da OMS e do Ministerio da Saude, mapeando codigos CID-10 especificos em 7 grupos: infecciosas, neoplasias curaveis, endocrinas, circulatorios, respiratorios, externas e neonatais/maternas.

Distribuicao do target: **50,1% evitaveis**, 49,9% nao-evitaveis.

**Decisao critica**: a causa basica (CAUSABAS) **nao foi utilizada como feature**. O objetivo e prever evitabilidade a partir do perfil demografico, sem conhecer a causa da morte. Se CAUSABAS fosse incluida, a tarefa seria trivial.

**Segunda tarefa (serie temporal)**: agregamos os obitos por mes e UF para os 30 anos, gerando 9.420 pontos (360 meses x 27 UFs). A serie nacional mostra sazonalidade clara e um pico de 207.106 obitos em marco de 2021, no auge da COVID-19.

## 4. Modelos testados

| Modelo | Tarefa | Dataset |
|--------|--------|---------|
| Random Forest | Binaria (evitavel) | 2024 (500k amostra) |
| Gradient Boosting | Binaria (evitavel) | 2024 (500k amostra) |
| Random Forest | Binaria (evitavel) | 2024+2025 (3M registros) |
| Random Forest | Multiclasse (21 capitulos CID-10) | 2024 |
| KMeans | Clustering (k=6) | 2024 (100k amostra) |
| LightGBM | Serie temporal (obitos mensais por UF) | 1996-2025 |

Hiperparametros do Random Forest final: n_estimators=200, max_depth=15, min_samples_leaf=50, class_weight='balanced'.

Para a serie temporal, o LightGBM foi treinado como modelo global com UF como variavel categorica e features de sazonalidade (seno/cosseno do mes), tendencia, lags (1, 2, 6 e 12 meses), medias moveis (3 e 12 meses) e indicador do periodo COVID.

Divisao: 80% treino, 20% teste, estratificada por target. Para a serie temporal: treino 1996-2023, teste 2024-2025 (24 meses).

## 5. Resultados

**Linha de base**: classificacao aleatoria (50% de accuracy) ou prever sempre a classe majoritaria (50,1% de accuracy).

| Modelo | Accuracy | AUC-ROC |
|--------|----------|---------|
| Random Forest (500k) | 57% | 0,6235 |
| Gradient Boosting (500k) | 58% | 0,6260 |
| **Random Forest (3M)** | **58%** | **0,6149** |
| Multiclasse (21 classes) | 16% | N/A |

O modelo final (Random Forest com 3 milhoes de registros) alcanca **AUC-ROC de 0,6149**, acima da linha de base aleatoria (0,50). A accuracy de 58% e modesta, mas significativa considerando que o modelo opera sem conhecer a causa da morte.

**Serie temporal** (obitos mensais por UF, previsao 2024-2025):

| Modelo | MAE | RMSE |
|--------|-----|------|
| Baseline sazonal (mesmo mes do ano anterior) | 5.568 | 6.673 |
| **LightGBM global (27 UFs)** | **3.591** | 4.412 |

O modelo LightGBM reduziu o erro em **35,5%** frente ao baseline sazonal, com erro medio de **2,8% do volume mensal** de obitos. Na avaliacao por UF, o modelo ganhou do baseline em 25 de 27 unidades federativas.

![Previsao de obitos mensais](figuras/previsao_obitos.png)

**Importancia das features** (Random Forest):

- LOCOCOR (local do obito): 36,7%
- IDADE: 25,5%
- ESC2010 (escolaridade): 7,5%
- HORA: 6,9%
- UF: 6,7%
- SEXO: 6,2%

O local do obito e a variavel mais discriminante: obitos em via publica ou residencia tem maior associacao com causas evitaveis.

**Clusterizacao** (KMeans, k=6):

- Cluster 2 (jovens, 24 anos medios): 73% evitavel, dominado por causas neonatais
- Cluster 1 (masculinos, 60 anos): 64% evitavel, causas circulatorias
- Cluster 4 (feminino, 65 anos): 46% evitavel

## 6. O que nao funcionou

**Gradient Boosting com dataset grande**: com 3 milhoes de registros e 500 estimadores, o treinamento excedeu 10 minutos e foi interrompido. O Random Forest e significativamente mais rapido para datasets grandes por ser paralelizavel.

**Classificacao multiclasse (capitulo CID-10)**: accuracy de 16% com 21 classes. O modelo nao consegue prever a causa da morte so com dados demograficos. Essa tarefa foi abandonada em favor da classificacao binaria.

**Aumento de dados nao melhorou o AUC**: treinar com 500k vs 3 milhoes de registros manteve o AUC praticamente identico (0,62 vs 0,61). O limitante sao as features disponiveis, nao a quantidade de dados.

**Primeira definicao de evitabilidade**: a definicao ampla (todos os capitulos A-J, V-Y) resultou em 67% de evitaveis, praticamente inutil para classificacao. Foi necessario refinar com codigos CID-10 especificos.

## 7. Limitacoes

**AUC moderado**: 0,61 indica poder discriminativo real, mas limitado. O modelo identifica tendencias demograficas, nao causalidade.

**Features insuficientes**: dados demograficos sozinhos nao capturam a complexidade da evitabilidade. Faltam variaveis como condicoes previas, acesso a saude, IDHM municipal, cobertura vacinal e numero de leitos.

**Definicao simplificada**: a lista de causas evitaveis e uma aproximacao dos criterios oficiais da OMS. A classificacao real depende de contexto clinico e epidemiologico.

**Dados de 2025 preliminares**: podem sofrer revisao pelo DATASUS.

**Serie temporal dependente da estabilidade historica**: a previsao de obitos assume que os padroes demograficos e epidemiologicos do passado se mantem. Eventos imprevisiveis (pandemias, desastres) nao podem ser antecipados, e o erro no agregado nacional mascara divergencias maiores em UFs individuais.

## Como reproduzir

```bash
git clone https://github.com/Weversson/Projeto_IML.git
cd Projeto_IML
pip install -r requirements.txt
```

Abrir `notebooks/notebook_mortalidade.ipynb` no Colab ou localmente. O notebook contem todo o pipeline: download dos dados, preprocessamento, treinamento e avaliacao.

Os modelos treinados estao disponiveis como Releases no GitHub:
https://github.com/Weversson/Projeto_IML/releases

Para usar o modelo em outro ambiente:

```python
import joblib
import pandas as pd

rf = joblib.load('models/evitavel_rf_2024_2025.pkl')

novo = pd.DataFrame({
    'IDADE': [65], 'SEXO': [1], 'RACACOR': [4], 'ESTCIV': [3],
    'ESC2010': [3], 'LOCOCOR': [1], 'UF': [35], 'MES_OBITO': [6],
    'HORA': [14], 'FAIXA_ETARIA': [5], 'ANO': [2025]
})

print(rf.predict(novo))        # 0 = nao-evitavel, 1 = evitavel
print(rf.predict_proba(novo))  # probabilidades
```
