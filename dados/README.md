# Dados do Projeto

Este diretorio documenta as fontes, o fluxo de dados e os links para download dos arquivos brutos e processados utilizados no projeto de previsao de obitos mensais no Brasil.

---

## 1. Dados Brutos (DATASUS / SIM)

Os dados brutos sao publicos e fornecidos pelo **Ministerio da Saude** atraves do **Sistema de Informacao sobre Mortalidade (SIM)**.

* **Portal oficial:** [dadosabertos.saude.gov.br/dataset/sim](https://dadosabertos.saude.gov.br/dataset/sim)
* **Origem:** Arquivos comprimidos em `.zip` contendo as Declaracoes de Obito individuais em formato JSON no bucket S3 oficial da Uniao (`https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/json/`).
* **Periodo:** 1996 a 2025 (30 anos de historico).
* **Volume total:** ~2,4 GB comprimidos em 30 arquivos `.zip` (~40 GB apos descompactacao, com mais de 45 milhoes de registros e 86 colunas).

Devido ao tamanho de 40 GB, esses arquivos brutos nao sao versionados diretamente no repositorio Git. O script de download e extracao automatizada dos arquivos brutos esta documentado na secao 4 do notebook (`notebooks/notebook_mortalidade.ipynb`).

---

## 2. Dados Processados (Serie Agregada)

Para a modelagem preditiva por series temporais, os registros individuais foram agregados pelo trio `(ANO, MES, UF)` contando a quantidade de obitos.

* **Arquivo:** `serie_obitos_uf.csv`
* **Dimensoes:** 9.420 linhas (360 meses x 27 Unidades Federativas).
* **Tamanho:** ~141 KB.
* **Download direto:** [GitHub Releases v1.0 - serie_obitos_uf.csv](https://github.com/Weversson/Projeto_IML/releases/download/v1.0/serie_obitos_uf.csv)

### Dicionario de Dados

| Coluna | Tipo | Descricao | Exemplo |
| :--- | :--- | :--- | :--- |
| `ANO` | Inteiro | Ano da ocorrencia do obito (1996 a 2025) | `2024` |
| `MES` | Inteiro | Mes da ocorrencia do obito (1 a 12) | `7` |
| `UF` | Inteiro | Codigo oficial do IBGE para a Unidade Federativa | `35` (SP) |
| `OBITOS`| Inteiro | Quantidade total de obitos registrados naquele mes e estado | `29540` |

---

## 3. Modelo Treinado

O modelo final de Machine Learning (LightGBM Global treinado com as 13 features temporais) esta publicado no GitHub Releases:

* **Arquivo:** `previsao_obitos_uf_lgbm.pkl`
* **Download direto:** [GitHub Releases v1.0 - previsao_obitos_uf_lgbm.pkl](https://github.com/Weversson/Projeto_IML/releases/download/v1.0/previsao_obitos_uf_lgbm.pkl)
