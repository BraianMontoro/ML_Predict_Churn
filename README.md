# 📊 ML Predict Churn — Telecom

Projeto de Machine Learning end-to-end para previsão de churn em uma empresa de telecomunicações, desenvolvido como parte do Tech Challenge (FIAP).

---

## 🎯 Objetivo

Identificar clientes com alta probabilidade de cancelamento (churn), permitindo ações proativas de retenção e redução de perda de receita.

---

## 🧠 Problema de ML

Classificação binária:

- 1 → Cliente cancelou (churn)
- 0 → Cliente permaneceu

---

## 📦 Dataset

- **7043 observações**
- **33 variáveis**

### Principais features:
- Perfil do cliente (Gender, Dependents, Partner)
- Tempo de relacionamento (Tenure Months)
- Serviços contratados (Internet, Tech Support, etc.)
- Financeiro (Monthly Charges, Total Charges)
- Contrato e pagamento

---

## 🧹 Pipeline de Dados

- Limpeza e tratamento de tipos
- Remoção de leakage (`Churn Score`, `Churn Reason`)
- Encoding com `OneHotEncoder`
- Escalonamento com `StandardScaler`
- Pipeline com `ColumnTransformer`

---

## 🤖 Modelos Treinados

| Modelo                     | Accuracy | Precision | Recall | F1 | AUC |
|---------------------------|---------|----------|--------|----|-----|
| Logistic Regression (0.3) | 0.79    | 0.54     | 0.74   | 0.623 | **0.84** |
| MLP (PyTorch)             | 0.76    | 0.54     | 0.74   | **0.625** | 0.83 |
| Random Forest             | 0.80    | **0.68** | 0.48   | 0.56 | 0.84 |
| Dummy Classifier          | 0.73    | 0.00     | 0.00   | 0.00 | - |

---

## 🏆 Modelo Selecionado

### 👉 Logistic Regression (threshold = 0.3)

Motivos:
- Melhor equilíbrio entre precision e recall
- Alto recall, essencial para identificar clientes em risco de churn
- AUC-ROC elevado (~0.84)
- Desempenho equivalente à MLP com menor complexidade
- Maior interpretabilidade e facilidade de manutenção

Apesar da MLP apresentar desempenho muito próximo (com leve ganho em F1-score), não houve melhoria significativa que justificasse o aumento de complexidade do modelo.

Dessa forma, a Regressão Logística foi mantida como modelo principal.

---

## ⚙️ Ajuste de Threshold

- Threshold padrão: **0.5**
- Threshold ajustado: **0.3**

### 🎯 Resultado:
- Recall aumentado (captura mais churns)
- Melhor alinhamento com o objetivo de negócio

---

## 📈 Métricas Prioritárias

- AUC-ROC
- Recall (principal)
- F1-score

### ⚠ Trade-off:
- Falso negativo → perda de cliente (alto impacto)
- Falso positivo → custo de retenção (baixo impacto)

---

## 💸 Impacto de Negócio

- Redução de churn
- Aumento de retenção
- Otimização de campanhas de CRM
- Maior previsibilidade de receita

---

## 🧪 Experiment Tracking

Utilização do **MLflow** para:
- Registro de experimentos
- Comparação de modelos
- Versionamento de métricas

---

## 🛠 Tecnologias

- Python
- Pandas / NumPy
- Scikit-learn
- MLflow
- Matplotlib / Seaborn

---

## 📁 Estrutura do Projeto

```bash
ML_PREDICT_CHURN/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda_baseline.ipynb
│   └── 02_mlp.ipynb (em desenvolvimento)
├── src/
├── models/
├── tests/
├── docs/
│   └── ml_canvas.md
├── README.md
```
---

## 👨‍💻 Autor

Braian Montoro
Analista de Sistemas | ML Engineer

###GitHub: https://github.com/BraianMontoro/ML_Predict_Churn
###Linkedin: 
