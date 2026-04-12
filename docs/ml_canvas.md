# 📄 ML Canvas — Previsão de Churn em Telecom

---

## 🎯 1. Objetivo de Negócio
Reduzir a taxa de cancelamento de clientes (churn), permitindo ações proativas de retenção e aumentando o tempo de vida do cliente (CLTV).

---

## 📊 2. Problema de Machine Learning
Problema de **classificação binária**, onde o objetivo é prever se um cliente irá cancelar o serviço (`Churn = Yes/No`).

---

## 📦 3. Dados Disponíveis

### 📁 Fonte
Dataset de telecom com **7.043 registros e 33 variáveis**

### 🔑 Principais grupos de variáveis:
- **Perfil do cliente:** Gender, Senior Citizen, Partner, Dependents  
- **Relacionamento:** Tenure Months  
- **Serviços contratados:** Internet Service, Tech Support, Streaming, etc.  
- **Financeiro:** Monthly Charges, Total Charges  
- **Contrato:** Contract, Payment Method, Paperless Billing  
- **Target:** Churn Label / Churn Value  

---

## 🧹 4. Preparação dos Dados

- Tratamento de valores ausentes  
- Conversão de tipos de dados  
- Encoding de variáveis categóricas (OneHotEncoder)  
- Padronização de variáveis numéricas  
- Pipeline com `ColumnTransformer`  

---

## 🧠 5. Features (Variáveis de Entrada)

### 📌 Selecionadas:
- Tenure Months  
- Monthly Charges  
- Total Charges  
- Contract  
- Internet Service  
- Tech Support  
- Payment Method  
- Partner / Dependents  

### 💡 Possíveis melhorias futuras:
- Feature engineering (ex: custo médio por mês)  
- Segmentação de clientes  
- Variáveis comportamentais  

---

## 🤖 6. Modelos Utilizados

### 🧪 Baselines:
- DummyClassifier  
- Logistic Regression  
- Random Forest  

### 🏆 Modelo escolhido:
**Logistic Regression**  
- Melhor equilíbrio entre interpretabilidade e performance  
- AUC ≈ 0.84  

---

## 📏 7. Métricas de Avaliação

- AUC-ROC  
- Precision  
- Recall  
- F1-Score  

### 🎯 Foco principal:
👉 **Recall (sensibilidade)**  
Motivo: minimizar falsos negativos (clientes que vão churnar e não foram identificados)

---

## ⚙️ 8. Ajuste de Threshold

- Threshold padrão: 0.5  
- Threshold ajustado: **0.3**

### 🎯 Objetivo:
Aumentar recall, mesmo com perda controlada de precisão

---

## 💸 9. Impacto no Negócio

### ✔ Benefícios:
- Identificação antecipada de clientes em risco  
- Redução de churn  
- Aumento de receita recorrente  
- Melhor direcionamento de campanhas de retenção  

### ⚠ Riscos:
- Falsos positivos (clientes que não iriam churnar)  
- Custo de ações desnecessárias  

---

## 🔁 10. Pipeline de ML

- Pipeline reprodutível com Scikit-Learn  
- Separação treino/teste  
- Validação cruzada estratificada  
- Registro de experimentos com MLflow  

---

## 🚀 11. Próximos Passos

- Implementar modelo de **Rede Neural (MLP com PyTorch)**  
- Comparar performance com baseline  
- Criar API de inferência com FastAPI  
- Implementar testes automatizados  
- Construir Model Card  
- Monitoramento de modelo (drift e performance)  

---

## 📌 12. Considerações Finais

O modelo atual apresenta boa capacidade preditiva e já permite aplicação prática no negócio. A evolução com redes neurais e melhorias no pipeline tende a aumentar ainda mais a performance e robustez da solução.