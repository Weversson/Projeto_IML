
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

**Classificacao removida**

A classificacao binaria evitavel/nao-evitavel foi removida do projeto. Tentamos treinar Random Forest e Gradient Boosting com features demograficas, mas Accuracy e AUC pararam em 58% e 0,61 respectivamente. O teto estava nas features: sem conhecer a causa do obito, o perfil demografico nao explica evitabilidade.

A classificacao multiclasse (21 capitulos CID-10) se saiu pior: accuracy de 16%. Nenhum ajuste de modelo ou volume de dados mudou o resultado, confirmando que o problema era a tarefa, nao a tecnica. Decisao: abandonar a classificacao e manter apenas a serie temporal, que tem ganho de 35,5% sobre o baseline e faz sentido na pratica.

Tambem removemos do release os arquivos relacionados (modelo Random Forest de 170 MB, clustering e scaler).

## 13/09

**Auditoria de reprodutibilidade e preparacao do seminario**

Revisamos os requisitos formais de entrega do seminario (reprodutibilidade, documentacao e apresentacao em 15 minutos com o repositorio na tela).

Identificamos e corrigimos dois pontos criticos de reprodutibilidade:
1. O arquivo `requirements.txt` nao continha a biblioteca `lightgbm`, necessaria para desserializar e inferir com o modelo final salvo em `.pkl`. Dependencia adicionada com versao fixada (`lightgbm>=4.0`).
2. As instrucoes de criacao e ativacao da virtualenv no `README.md` consideravam apenas o padrao Unix (`.venv/bin/pip`), falhando no Windows (`.venv\Scripts\activate` ou `source .venv/Scripts/activate` no Git Bash). Atualizamos a documentacao com suporte explicito para ambos os ambientes.

Criamos o script CLI `prever.py` na raiz do repositorio para permitir demonstracao rapida e interativa durante a apresentacao. O script automatiza o download do modelo caso nao esteja presente, carrega a serie agregada, monta as 13 features exatamente como no treinamento e calcula na hora a predicao do LightGBM contra o baseline sazonal, exibindo o ganho percentual no terminal.

## 14/09

**Codigo de treino no notebook**

Revisamos o repositorio com foco na reprodutibilidade. O modelo foi treinado no Colab, mas o codigo de treino nao estava no repositorio: o notebook e o `prever.py` apenas baixam o `.pkl` pronto do Release v1.0 e fazem a previsao. Assim, ninguem conseguia gerar o modelo de novo, so avaliar o que ja estava pronto.

Duvida na epoca: treinar de novo com os parametros do README geraria exatamente o mesmo modelo do Release? Retreinamos com os meses ate dez/2023, UF como codigo numerico do IBGE e `random_state=42` (a semente registrada no `.pkl`), e comparamos com o modelo publicado. As previsoes ficaram identicas nas 9.096 linhas, com diferenca maxima de 0,0, as mesmas 800 arvores e os mesmos parametros. A unica diferenca foi de arredondamento, na casa de 1e-16, em alguns limiares das arvores, sem efeito nas previsoes.

Adicionamos no notebook, depois do grafico real vs previsao, uma celula que treina o modelo e compara com o do Release. O modelo novo e salvo como `previsao_obitos_uf_lgbm_treinado.pkl`, para nao sobrescrever o arquivo baixado. Rodamos no Colab: 8.448 linhas de treino (01/1997 a 12/2023), 648 de teste (01/2024 a 12/2025), previsoes identicas ao Release e MAE de 3.591 obitos/mes. O treino comeca em 1997 porque o primeiro ano de cada UF nao tem `LAG_12` e sai no `dropna`.

**Problemas encontrados na revisao**

- A serie agregada tem 9.420 linhas, e nao 9.720 (360 meses x 27 UFs). Falta o ano de 2012 inteiro em 25 UFs, so SP e TO tem dados. O ZIP de 2012 existe no DATASUS, entao a falha esta no nosso processamento.
- O README diz que o modelo venceu o baseline em 25 das 27 UFs. Na conferencia, comparando o MAE de cada UF nos 24 meses de teste, foram 18.
- A celula 6 do notebook e o README dizem que a UF entra como variavel categorica, mas o modelo usa o codigo numerico.
- O `subsample=0.8` nao tem efeito, porque o `subsample_freq` ficou em 0.
- Testamos tambem outra linha de base, repetir o mes anterior: MAE de 5.738. O modelo vence as duas linhas de base.

Proximo passo: descobrir por que 2012 se perdeu e combinar com o grupo antes de corrigir, porque os numeros do README vao mudar.

**Correcoes da revisao**

Com a celula de treino pronta, corrigimos os pontos da revisao que nao mudam o modelo nem os dados. No README, trocamos "25 das 27 UFs" por 18 das 27, listando as 9 em que o modelo perde, registramos que o `subsample` nao tem efeito e acrescentamos 2012 nas limitacoes. No README e no notebook, corrigimos a descricao da UF (codigo numerico do IBGE, nao categorica) e explicamos por que a serie tem 9.420 linhas e nao 9.720. A tabela de modelos testados passou a incluir as duas linhas de base e o modelo de serie nacional unica. Tambem tiramos o `seaborn` do `requirements.txt`, que nenhum arquivo usava.

Para esses numeros nao ficarem so no texto, adicionamos no notebook a celula "Comparacao detalhada", com as duas linhas de base, o MAE por UF e a importancia das features, usando as previsoes ja feitas, sem treinar de novo. O `LAG_1` responde por 68,9% do gain do modelo e, somado as medias moveis de 3 e 12 meses, passa de 95%. Na pratica, o modelo funciona como uma persistencia ajustada pelas medias recentes.

Na secao 4 do notebook, recolocamos as aspas que tinham sumido no codigo comentado de reconstrucao da serie, conferindo o download com a versao do primeiro commit. No `prever.py`, incluimos um aviso quando a data pedida esta no periodo de treino e uma mensagem para 2012.

Duvida: corrigir 2012 antes da apresentacao, mesmo mudando os numeros do README, ou apresentar como limitacao? Precisamos decidir com o grupo.
