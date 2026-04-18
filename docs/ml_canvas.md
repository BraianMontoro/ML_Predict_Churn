# ML Canvas - Previsao de Churn em Telecom

## 1. Problema de Negocio

Uma operadora de telecom precisa reduzir o cancelamento de clientes com acoes de retencao mais precisas e sustentaveis.

## 2. Objetivo de Negocio

Identificar com antecedencia clientes com alto risco de churn para orientar ofertas, campanhas e atendimento preventivo.

## 3. Stakeholders

- diretoria de negocio
- time de CRM e marketing
- time de atendimento e retencao
- engenharia de dados e MLOps

## 4. Problema de Machine Learning

Classificacao binaria para prever se um cliente ira cancelar o servico (`churn_value` igual a `1`) ou permanecer (`0`).

## 5. Dados Disponiveis

- fonte: `data/raw/Telco_customer_churn.xlsx`
- volume: 7043 registros
- colunas originais: 33
- principais grupos: perfil, servicos, relacionamento, contrato, financeiro e geografia

## 6. Features Relevantes

- relacionamento: `tenure_months`, `partner`, `dependents`
- servicos: `internet_service`, `tech_support`, `streaming_tv`, `streaming_movies`
- contrato e pagamento: `contract`, `payment_method`, `paperless_billing`
- financeiro: `monthly_charges`, `total_charges`
- contexto geografico: `state`, `city`, `zip_code`

## 7. Saida Esperada

- `prediction`: classe binaria de churn
- `churn_probability`: probabilidade estimada para apoiar priorizacao

## 8. Metrica Tecnica

- principal: AUC-ROC
- complementares: recall, precision e F1-score

## 9. Metrica de Negocio

- clientes em risco identificados antes do cancelamento
- churn evitado por campanha
- custo de retencao por cliente acionado

## 10. Trade-off de Decisao

O projeto privilegia recall para reduzir falsos negativos, pois deixar de abordar um cliente realmente em risco tende a custar mais do que acionar uma oferta desnecessaria para um falso positivo.

## 11. SLOs Propostos

- disponibilidade da API acima de 99%
- latencia p95 do endpoint `/predict` abaixo de 500 ms
- taxa de erro inferior a 2%
- suite de testes e lint sempre verdes antes de publicar novos artefatos

## 12. Riscos

- drift de comportamento de clientes ao longo do tempo
- vies indiretos em variaveis geograficas e financeiras
- dependencia de dataset local para reprodutibilidade do treino

## 13. Solucao Tecnica Escolhida

- baselines com Scikit-Learn para referencia
- MLP em PyTorch como modelo central do desafio
- MLflow para rastreamento de experimentos
- FastAPI para inferencia em tempo real

## 14. Estado Atual

- EDA concluida em notebook
- baselines registrados no MLflow
- MLP treinada com early stopping e batching
- API funcional com `/health` e `/predict`
- validacao com Pydantic e Pandera
- documentacao de arquitetura, monitoramento e Model Card

## 15. Proximos Passos

- gravar o video STAR de 5 minutos
- opcionalmente publicar a API em nuvem para o bonus da entrega
- migrar o backend local do MLflow para SQLite em uma evolucao futura
