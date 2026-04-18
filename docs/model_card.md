# 📊 Model Card — Telco Churn Prediction

## 1. Visão Geral

Este modelo foi desenvolvido para prever a probabilidade de cancelamento (churn) de clientes em uma empresa de telecomunicações.

O objetivo principal é apoiar estratégias de retenção, identificando clientes com maior risco de churn para ações preventivas.

---

## 2. Tipo de Modelo

- Algoritmo: Logistic Regression
- Biblioteca: Scikit-learn
- Pipeline: ColumnTransformer + Pipeline
- Framework de serving: FastAPI
- Tracking de experimentos: MLflow

---

## 3. Dados Utilizados

- Dataset: Telco Customer Churn
- Volume: 7043 registros
- Variáveis: 33 colunas originais

### Tipos de features:
- Demográficas (ex: gender, senior_citizen)
- Contratuais (ex: contract, tenure_months)
- Serviços contratados (ex: internet_service, streaming_tv)
- Financeiras (ex: Monthly Charges, total_charges)
- Geográficas (ex: country, state, latitude, longitude)

### Target:
- `churn_value` (0 = não cancelou, 1 = cancelou)

---

## 4. Pipeline de Dados

O pipeline inclui:

- Imputação de valores ausentes:
  - Numéricas → mediana
  - Categóricas → mais frequente
- Encoding:
  - OneHotEncoder para variáveis categóricas
- Normalização:
  - StandardScaler para variáveis numéricas
- Integração com modelo via Pipeline do Scikit-learn

---

## 5. Métricas do Modelo

### Validação Cruzada (Stratified K-Fold)

- AUC média: ~0.83
- F1-score médio: ~0.57
- Recall médio: ~0.53
- Precision média: ~0.61

### Holdout (teste final)

- Accuracy: ~0.78
- Precision: ~0.61
- Recall: ~0.53
- F1-score: ~0.57
- AUC: ~0.83

---

## 6. Interpretação do Modelo

O modelo retorna:

```json
{
  "prediction": 0 ou 1,
  "churn_probability": valor entre 0 e 1
}

```

---

## 7. Trade-offs de Negócio

### Falsos Positivos (FP)
Cliente identificado como risco, mas não cancelaria.

Impacto:
- Ações de retenção desnecessárias
- Custo operacional
- Possível concessão de benefícios indevidos

### Falsos Negativos (FN)
Cliente em risco não identificado.

Impacto:
- Perda de receita
- Cancelamento não evitado
- Impacto direto no churn real

### Estratégia adotada:
O modelo foi calibrado para equilibrar precisão e recall, com leve foco em recall, visando reduzir perdas de clientes.

---

## 8. Limitações

- O modelo é linear (Logistic Regression), podendo não capturar relações complexas entre variáveis
- Dependência de qualidade dos dados (ex: `Total Charges`)
- Features geográficas podem introduzir ruído ou pouca relevância
- Sensível a mudanças no comportamento do cliente ao longo do tempo (data drift)

---

## 9. Riscos e Vieses

- Possível viés geográfico (estado, cidade)
- Possível viés socioeconômico implícito (via serviços contratados)
- Dados históricos podem refletir decisões passadas da empresa

---

## 10. Monitoramento Recomendado

- Monitorar AUC ao longo do tempo
- Monitorar taxa de churn real vs previsto
- Monitorar drift nas features (ex: distribuição de tenure, charges)
- Re-treinamento periódico recomendado

---

## 11. Uso Recomendado

- Priorização de campanhas de retenção
- Segmentação de clientes por risco
- Apoio à tomada de decisão em marketing e CRM

---

## 12. Uso NÃO Recomendado

- Decisões automáticas sem validação humana
- Uso como única fonte de decisão
- Aplicação fora do domínio de telecom

---

## 13. Versão do Modelo

- Nome: logistic_pipeline.joblib
- Versão: 1.0
- Data: (preencher com data atual)

---

## 14. Contato

Responsável: Braian Montoro  
Projeto: ML Predict Churn (FIAP Pós-Tech)