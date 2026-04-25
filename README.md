# ML Predict Churn - Telecom

Projeto de Machine Learning end-to-end para previsao de churn em telecom, desenvolvido para o Tech Challenge da FIAP.

## Objetivo

Identificar clientes com maior probabilidade de cancelamento para apoiar acoes de retencao, priorizacao comercial e reducao de perda de receita.

## Escopo da Entrega

O repositorio cobre:

- EDA e baselines
- MLP em PyTorch com batching e early stopping
- tracking de experimentos com MLflow
- API FastAPI para inferencia
- testes automatizados
- Model Card
- documentacao de arquitetura e monitoramento

## Estrutura do Projeto

```text
ML_PREDICT_CHURN/
|-- data/
|-- docs/
|-- mlruns/
|-- models/
|-- notebooks/
|-- src/
|-- tests/
|-- Makefile
|-- pyproject.toml
|-- requirements.txt
|-- requirements-dev.txt
```

## Modelos do Projeto

### Baselines

- Dummy Classifier
- Logistic Regression
- Random Forest

### Modelo central

- MLP em PyTorch

## Resultados de Referencia

As metricas abaixo refletem os artefatos mais recentes gerados localmente:

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.781 | 0.602 | 0.513 | 0.554 | 0.827 |
| MLP (PyTorch) | 0.772 | 0.557 | 0.695 | 0.618 | 0.841 |

Os baselines adicionais (`DummyClassifier` e `RandomForestClassifier`) ficam registrados no MLflow para comparacao historica.

## Modelo Selecionado para Producao

O baseline de producao continua sendo a Regressao Logistica, porque oferece:

- interpretabilidade maior
- operacao mais simples
- desempenho competitivo
- facilidade de manutencao

A MLP foi industrializada no projeto por ser o modelo central exigido pelo desafio e pode ser utilizada via API com `model_name=mlp`.

## Requisitos

- Python 3.11+
- ambiente virtual ativo
- arquivo `data/raw/Telco_customer_churn.xlsx` disponivel localmente
- dependencias instaladas

## Dataset

O projeto utiliza o dataset Telco Customer Churn em formato Excel.

Para reproduzir o treino do zero:

1. obtenha o arquivo `Telco_customer_churn.xlsx`
2. salve em `data/raw/`
3. mantenha exatamente esse nome de arquivo

O dataset bruto nao e versionado no Git por tamanho e por boas praticas de projetos de ML.

## Instalacao

### Opcao 1

```powershell
python -m pip install -e ".[dev,train]"
```

Instala o projeto com as dependencias de desenvolvimento e treinamento.

### Opcao 2

```powershell
pip install -r requirements.txt
```

Instala apenas o runtime da API a partir do `pyproject.toml`.

### Opcao 3

```powershell
pip install -r requirements-dev.txt
```

Arquivo de conveniencia para instalar o projeto com extras de desenvolvimento e treino.

## Dependencias

O `pyproject.toml` e a fonte canonica de dependencias, lint e configuracao de testes.

Arquivos auxiliares:

- `requirements.txt`: runtime da API em producao, incluindo suporte a inferencia com MLP
- `requirements-dev.txt`: instalacao local com extras de treino e desenvolvimento

## Comandos Principais

```powershell
make install
make install-runtime
make install-dev
make lint
make test
make run
make train-logistic
make train-baselines
make train-mlp
make mlflow-ui
```

## Treinamento

### Regressao Logistica

```powershell
python -m src.models.train_baseline
```

Artefato salvo em:

- `models/trained/logistic_pipeline.joblib`

### Baselines comparativos

```powershell
python -m src.models.train_all_baselines
```

Artefatos auxiliares salvos em:

- `models/artifacts/dummy_classifier.joblib`
- `models/artifacts/logistic_regression.joblib`
- `models/artifacts/random_forest.joblib`

### MLP

```powershell
python -m src.models.train_mlp
```

Artefatos salvos em:

- `models/trained/mlp_bundle.joblib`
- `models/artifacts/mlp_training_history.json`

## API

Suba a API com:

```powershell
python -m uvicorn src.api.main:app --reload
```

### Endpoints

- `GET /health`
- `POST /predict`

### Header de observabilidade

Todas as respostas incluem:

- `X-Process-Time-Ms`

### Exemplo de inferencia com o baseline logistico

```powershell
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"count\":1,\"country\":\"United States\",\"state\":\"California\",\"city\":\"Los Angeles\",\"zip_code\":\"90001\",\"lat_long\":\"33.973616, -118.24902\",\"latitude\":33.973616,\"longitude\":-118.24902,\"gender\":\"Male\",\"senior_citizen\":0,\"partner\":\"Yes\",\"dependents\":\"No\",\"tenure_months\":12,\"phone_service\":\"Yes\",\"multiple_lines\":\"No\",\"internet_service\":\"Fiber optic\",\"online_security\":\"No\",\"online_backup\":\"Yes\",\"device_protection\":\"No\",\"tech_support\":\"No\",\"streaming_tv\":\"Yes\",\"streaming_movies\":\"Yes\",\"contract\":\"Month-to-month\",\"paperless_billing\":\"Yes\",\"payment_method\":\"Electronic check\",\"Monthly Charges\":79.9,\"total_charges\":958.8}"
```

### Exemplo de inferencia com a MLP

```powershell
curl -X POST "http://127.0.0.1:8000/predict?model_name=mlp" ^
  -H "Content-Type: application/json" ^
  -d "{\"count\":1,\"country\":\"United States\",\"state\":\"California\",\"city\":\"Los Angeles\",\"zip_code\":\"90001\",\"lat_long\":\"33.973616, -118.24902\",\"latitude\":33.973616,\"longitude\":-118.24902,\"gender\":\"Male\",\"senior_citizen\":0,\"partner\":\"Yes\",\"dependents\":\"No\",\"tenure_months\":12,\"phone_service\":\"Yes\",\"multiple_lines\":\"No\",\"internet_service\":\"Fiber optic\",\"online_security\":\"No\",\"online_backup\":\"Yes\",\"device_protection\":\"No\",\"tech_support\":\"No\",\"streaming_tv\":\"Yes\",\"streaming_movies\":\"Yes\",\"contract\":\"Month-to-month\",\"paperless_billing\":\"Yes\",\"payment_method\":\"Electronic check\",\"Monthly Charges\":79.9,\"total_charges\":958.8}"
```

Observacao:

- a rota `model_name=mlp` exige que o artefato `models/trained/mlp_bundle.joblib` exista no ambiente
- no repositorio versionado, o artefato garantido para inferencia imediata e o baseline logistico

## Qualidade e Validacao

O projeto possui:

- validacao HTTP com Pydantic
- validacao tabular com Pandera
- smoke test
- testes de schema
- testes da API
- testes unitarios de preprocessamento
- lint com Ruff

## MLflow

Os runs sao registrados localmente em `mlruns/`.

Cada experimento registra:

- parametros do modelo
- metricas de avaliacao
- artefatos gerados
- metadados do dataset de treino

Para abrir a interface:

```powershell
python -m mlflow ui --backend-store-uri ./mlruns
```

## Documentacao

- Arquitetura: [docs/architecture.md](docs/architecture.md)
- Monitoramento: [docs/monitoring.md](docs/monitoring.md)
- Model Card: [docs/model_card.md](docs/model_card.md)
- ML Canvas: [docs/ml_canvas.md](docs/ml_canvas.md)

## Observacoes

- O backend local do MLflow em filesystem funciona para o desafio, mas pode ser migrado depois para SQLite ou backend remoto.
- O deploy em nuvem continua como extensao natural para o bonus da entrega.
- Para deploy da API em App Service, `requirements.txt` instala o runtime da aplicacao via `pyproject.toml`, incluindo o extra `serve-mlp`.
- O endpoint `/predict` em nuvem exige que os arquivos `models/trained/logistic_pipeline.joblib` e `models/trained/mlp_bundle.joblib` estejam versionados no repositorio.

## Autor

Braian Montoro - Analista de Sistemas | ML Engineer

- GitHub: https://github.com/BraianMontoro/
- LinkedIn: https://www.linkedin.com/in/braian-montoro-450ba6113/
