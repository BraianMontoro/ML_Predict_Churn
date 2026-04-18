# Model Card - Telco Churn Prediction

## 1. Visao Geral

Este projeto preve a probabilidade de churn de clientes de telecom para apoiar acoes preventivas de retencao.

O repositorio contem dois modelos relevantes:

- Regressao Logistica, adotada como baseline de producao
- MLP em PyTorch, implementada como modelo central do desafio

## 2. Modelos Disponiveis

### Logistic Regression

- biblioteca: scikit-learn
- pipeline: `ColumnTransformer + Pipeline`
- artefato: `models/trained/logistic_pipeline.joblib`
- uso principal: inferencia padrao via API
- decisao de classe: regra nativa do estimador

### MLP

- biblioteca: PyTorch
- arquitetura: `input -> 64 -> 32 -> 1`
- ativacoes: `ReLU` e `Sigmoid`
- artefato: `models/trained/mlp_bundle.joblib`
- uso principal: comparacao tecnica e inferencia opcional via API
- decisao de classe: threshold configurado em `0.3`

## 3. Dados Utilizados

- dataset: Telco Customer Churn
- volume: 7043 registros
- colunas originais: 33
- target: `churn_value`

Principais grupos de variaveis:

- demograficas
- contratuais
- servicos contratados
- financeiras
- geograficas

## 4. Pipeline de Dados

O pipeline inclui:

- limpeza e padronizacao de colunas
- tratamento de tipos para `senior_citizen`, `monthly_charges` e `total_charges`
- validacao tabular com Pandera
- imputacao de faltantes
- `OneHotEncoder` para categoricas
- `StandardScaler` para numericas

## 5. Metricas de Referencia

### Logistic Regression

- accuracy: 0.781
- precision: 0.602
- recall: 0.513
- f1: 0.554
- auc: 0.827

### MLP

- accuracy: 0.772
- precision: 0.557
- recall: 0.695
- f1: 0.618
- auc: 0.841

## 6. Saida da API

O contrato de resposta e:

```json
{
  "prediction": 0,
  "churn_probability": 0.42
}
```

## 7. Trade-offs de Negocio

### Falso Positivo

Impactos:

- acao de retencao desnecessaria
- custo operacional
- possivel concessao indevida de beneficio

### Falso Negativo

Impactos:

- cliente em risco nao abordado
- perda de receita recorrente
- cancelamento nao evitado

Por isso, o projeto acompanha recall com atencao especial, mesmo mantendo a Regressao Logistica como opcao principal de producao pela simplicidade operacional.

## 8. Limitacoes

- a Regressao Logistica nao captura relacoes nao lineares complexas
- a MLP e mais sensivel a tuning e custo operacional
- o dataset pode sofrer drift ao longo do tempo
- a origem dos dados ainda e um arquivo local

## 9. Riscos e Vieses

- vies geograficos por estado, cidade e coordenadas
- vies socioeconomicos indiretos via servicos e pagamentos
- dependencia de comportamento historico passado da empresa

## 10. Monitoramento Recomendado

- disponibilidade e latencia da API
- distribuicao das probabilidades previstas
- taxa de predicoes positivas
- AUC, recall e precision ao longo do tempo
- drift das features criticas

Detalhamento operacional em [monitoring.md](monitoring.md).

## 11. Uso Recomendado

- priorizacao de campanhas de retencao
- apoio a CRM e marketing
- segmentacao de clientes por risco

## 12. Uso Nao Recomendado

- decisao automatica sem validacao humana
- uso fora do dominio de telecom
- uso como unica fonte de decisao de negocio

## 13. Versao

- baseline de producao: `logistic_pipeline.joblib`
- modelo central do desafio: `mlp_bundle.joblib`
- versao do projeto: 1.0
- data de referencia: 2026-04-18

## 14. Responsavel

- autor: Braian Montoro
- projeto: ML Predict Churn
