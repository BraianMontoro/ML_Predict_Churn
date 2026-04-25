# Arquitetura do Projeto

## 1. Visao Geral

O projeto segue uma arquitetura de ML end-to-end com quatro blocos principais:

1. Ingestao e limpeza de dados tabulares vindos do arquivo `data/raw/Telco_customer_churn.xlsx`.
2. Treinamento de modelos com pipelines reproduziveis e rastreamento no MLflow.
3. Persistencia de artefatos em `models/trained/` e `models/artifacts/`.
4. Serving online via FastAPI para inferencia em tempo real.

## 2. Fluxo de Dados

O fluxo principal e:

1. `src/data/load_data.py` carrega o dataset bruto.
2. `src/data/preprocess.py` normaliza colunas e trata tipos.
3. `src/data/schemas.py` valida as features e o target com Pandera.
4. `src/models/train_baseline.py` treina a Regressao Logistica com pipeline sklearn.
5. `src/models/train_mlp.py` treina a MLP em PyTorch com batching e early stopping.
6. Os artefatos finais sao salvos localmente e registrados no MLflow.
7. `src/api/main.py` expoe os endpoints `/health` e `/predict`.

## 3. Componentes de Treinamento

### Baseline de producao

- Modelo: Logistic Regression
- Pipeline: `ColumnTransformer + Pipeline`
- Artefato local: `models/trained/logistic_pipeline.joblib`
- Motivo de escolha: melhor equilibrio entre desempenho, interpretabilidade e simplicidade operacional

### Modelo central do desafio

- Modelo: MLP em PyTorch
- Arquitetura: `input -> 64 -> 32 -> 1`
- Ativacao: `ReLU` nas camadas ocultas e `Sigmoid` na saida
- Treinamento: `DataLoader`, `Adam`, `BCELoss`, `early stopping`
- Artefato local: `models/trained/mlp_bundle.joblib`

## 4. Camada de Serving

A API utiliza FastAPI em modo real-time.

### Ambientes de execucao

- ambiente local: `http://127.0.0.1:8000`
- ambiente publico Azure App Service: `https://ml-predict-churn-braian.azurewebsites.net`
- Swagger publico: `https://ml-predict-churn-braian.azurewebsites.net/docs#/`

### Endpoints

- `GET /health`: verifica disponibilidade da aplicacao
- `POST /predict`: realiza inferencia com `model_name=logistic` ou `model_name=mlp`

### Validacoes

- Pydantic valida o contrato HTTP
- Pandera valida o dataframe montado para inferencia antes do modelo

### Observabilidade

- Logging estruturado com `logging`
- Middleware de latencia adicionando o header `X-Process-Time-Ms`

### Publicacao em nuvem

- deploy em Azure App Service
- pipeline automatizado por GitHub Actions em `push` para `main`
- artefatos exigidos em producao: `models/trained/logistic_pipeline.joblib` e `models/trained/mlp_bundle.joblib`

## 5. Decisao de Deploy: Real-time vs Batch

### Opcao escolhida: real-time

A escolha por serving online foi feita porque o caso de uso descrito no Tech Challenge pede um modelo servido via API e uma experiencia de inferencia sob demanda para clientes individuais.

### Justificativa

- A diretoria precisa consultar risco de churn em nivel de cliente.
- FastAPI e suficiente para um endpoint leve e simples de demonstracao.
- A entrega obrigatoria valoriza API funcional e pacote reutilizavel.

### Quando batch faria sentido

Um pipeline batch tambem seria util para:

- score diario da base inteira de clientes
- priorizacao de campanhas em CRM
- exportacao de listas para marketing e retencao

Nesse caso, o mesmo modelo poderia ser reutilizado em uma rotina agendada separada da API.

## 6. Estrutura de Persistencia

- `models/trained/`: bundles e pipelines prontos para inferencia
- `models/artifacts/`: historicos auxiliares de treino
- `mlruns/`: tracking local do MLflow
- `docs/`: documentacao da arquitetura, model card e monitoramento

## 7. Riscos Arquiteturais

- Dependencia de arquivo Excel local como fonte de dados
- Ausencia de backend SQL para MLflow em producao
- Serving sincrono sem fila ou autoscaling
- Possivel drift de features sem monitoramento automatico

## 8. Evolucoes Recomendadas

- migrar o MLflow para `sqlite:///mlflow.db` ou backend remoto
- adicionar pipeline batch agendado
- expor metricas para Prometheus
- containerizar a API com Docker
- adicionar monitoramento gerenciado no ambiente Azure
