# 📊 ML_PREDICT_CHURN

Projeto de Machine Learning para previsão de churn em uma operadora de telecomunicações, desenvolvido como parte do Tech Challenge (Pós-Tech FIAP).

---

## 🎯 Objetivo

Construir um pipeline completo de Machine Learning capaz de prever a probabilidade de um cliente cancelar o serviço (churn), utilizando:

- Modelos baseline (Scikit-Learn)
- Rede Neural MLP (PyTorch)
- Rastreamento de experimentos com MLflow
- API de inferência com FastAPI

---

## 🧠 Problema de Negócio

A empresa enfrenta alta taxa de cancelamento de clientes.  
O objetivo é identificar clientes com maior risco de churn para permitir ações de retenção.

---

## 📁 Estrutura do Projeto

ML_PREDICT_CHURN/
├── data/
├── models/
├── notebooks/
├── src/
├── tests/
├── docs/

---

## 📦 Dataset

O dataset NÃO está versionado no repositório.

Para executar o projeto:

1. Baixe o dataset (ex: IBM Telco Customer : https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset?resource=downloadChurn)
2. Coloque o arquivo em:

data/raw/

---

## ⚙️ Setup do Ambiente

Criar ambiente virtual:

python -m venv .venv

Ativar ambiente:

Windows:
.venv\Scripts\activate

Linux / Mac:
source .venv/bin/activate

Instalar dependências:

pip install -r requirements.txt

---

## 🚀 Executando o Projeto

Rodar notebooks:

jupyter notebook

Rodar API:

uvicorn src.api.main:app --reload

Acessar documentação:
http://127.0.0.1:8000/docs

Rodar testes:

pytest

---

## 📊 Etapas do Projeto

1. EDA e análise exploratória  
2. Modelos baseline  
3. Rede neural MLP (PyTorch)  
4. Comparação de modelos  
5. API de inferência  
6. Testes automatizados  
7. Documentação e Model Card  

---

## 🧪 Tecnologias

- Python
- Scikit-Learn
- PyTorch
- MLflow
- FastAPI
- Pandera
- Pytest
- Ruff

---

## 📌 Autor

Braian Montoro

---

## 📄 Licença

Uso educacional.
