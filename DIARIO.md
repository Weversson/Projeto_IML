# Diario de Decisoes

## 09/09

**Aquisicao dos dados**

O ponto de partida foi um dataset do Kaggle (joovictorluz/brazilian-unified-health-system-deaths-1996-2023). Antes de usar, pesquisamos a fonte original e identificamos o **DATASUS** (Departamento de Informatica do SUS) como produtor dos dados. O sistema se chama **SIM** (Sistema de Informacao sobre Mortalidade) e disponibiliza dados abertos desde 1996.

Decisao: usar a fonte direta do DATASUS ao inves do Kaggle. Motivos: dados desatualizados no Kaggle, intermediacao desnecessaria, risco de alteracao.

O portal de dados abertos do SUS (dadosabertos.saude.gov.br/dataset/sim) fornece arquivos JSON no S3 oficial. Baixamos 30 arquivos ZIP, de 1996 a 2025, totalizando 2,4 GB comprimidos.

**Extracao e concatenacao**

Os arquivos de 2024 e 2025 vieram em unico JSON cada. Os de 1996 a 2023 vieram particionados em 5 a 10 arquivos por ano. A abordagem ingenua (json.load + json.dump) travou por mais de 10 minutos em anos grandes. Testamos concatenacao textual direta (stripping dos colchetes externos) e resolveu: menos de 30 segundos por ano. O resultado sao 30 arquivos JSON limpos, totalizando 40 GB.

Duvida na epoca: valia a pena manter os arquivos particionados originais? Decidimos nao, pois o projeto so precisa dos JSONs finais.

**Analise exploratoria**

Carregamos 2024 (1.532.015 registros, 86 colunas). As principais causas de morte sao doenças circulatorias (I219, I10, I64), respiratorias (J189) e neoplasias (C349, C509). Distribuicao por sexo: 55% masculino, 45% feminino. Distribuicao por raca/cor: 51% branca, 39% parda, 9% preta.

**Definicao da tarefa de ML**

Testamos cinco alternativas: classificacao multiclasse (capitulo CID-10), clusterizacao, previsao de serie temporal, deteccao de anomalias e classificacao binaria evitavel/nao-evitavel.

Decisao: classificacao binaria de obitos evitaveis. Motivos: impacto social direto, alinhamento com criterios da OMS, distribuicao natural balanceada (~50/50).

**Engenharia de features**

Definimos 11 features demograficas: IDADE, SEXO, RACACOR, ESTCIV, ESC2010, LOCOCOR, UF, MES_OBITO, HORA, FAIXA_ETARIA, ANO. A decisao critica foi **nao incluir CAUSABAS** (causa basica do obito) como feature. Se o modelo soubesse a causa, a tarefa seria trivial, pois a propria definicao de evitabilidade depende da causa. O objetivo e prever evitabilidade a partir do perfil demografico.

**Criterio de evitabilidade**

Mapeamos codigos CID-10 especificos em 7 grupos: infecciosas (A/B), neoplasias curaveis (C especificos), endocrinas (E10-E14), circulatorios (I especificos), respiratorios (J especificos), externas (V-Y) e neonatais/maternas (P especificos). Resultado: 50,1% evitaveis, 49,9% nao-evitaveis.

Primeira tentativa: definicao ampla (todos os capítulos A-J, V-Y). Deu 67% evitaveis, muito enviesado. Refinamos para a definicao acima.

**Treinamento e avaliacao**

Treinamos Random Forest e Gradient Boosting com 500 mil registros (amostra de 2024). AUC: 0,62 (RF) e 0,63 (GB). Accuracy: 57-58%. A feature mais importante foi LOCOCOR (local do obito, 36%), seguida de IDADE (25%).

Expandimos para 3 milhoes de registros (2024+2025). O Gradient Boosting travou por tempo. O Random Forest treinou em minutos e manteve AUC de 0,61. A conclusao e que o limite esta nas features, nao no volume de dados.

Tambem treinamos classificacao multiclasse (21 capitulos CID-10) com accuracy de 16% e clusterizacao KMeans com 6 clusters. O cluster de maior risco foi o de jovens (73% evitavel, causas neonatais).

**Limpeza**

Removemos 6 arquivos redundantes (modelos antigos, multiclasse com accuracy baixa, label encoder sem uso). Sobraram 3 arquivos: modelo final (170 MB), clustering (392 KB) e scaler (1,1 KB). Total: 171 MB.

## 10/09

**Documentacao e organizacao do repositorio**

Criamos o notebook Colab (notebook_mortalidade.ipynb) com a documentacao completa do pipeline: desde a aquisicao dos dados ate o treinamento. Linguagem pt-br, sem travessoes, sem emojis.

Configuramos o repositorio GitHub com .gitignore (exclui data/ de 40GB e models/*.pkl), requirements.txt, README.md e DIARIO.md. O README segue o template da atividade: problema, dados, preprocessamento, modelos, resultados, o que nao funcionou, limitacoes.

Decisao importante: os modelos .pkl nao podem ir pro GitHub (limite de 100 MB por arquivo). Foram disponibilizados via GitHub Releases. O notebook no Colab carrega os modelos diretamente do link de release, eliminando a necessidade de baixar arquivos grandes manualmente.

Geramos a previsao de obitos mensais (1996-2025) para serie temporal. Resultado: LightGBM global com features sazonais (seno/cosseno do mes), lags e medias moveis, avaliado em 2024-2025. Ganho de 35,5% no MAE vs baseline sazonal (3.591 vs 5.568 obitos/mes), errando 2,8% do volume mensal. Grafico salvo em figuras/previsao_obitos.png.
