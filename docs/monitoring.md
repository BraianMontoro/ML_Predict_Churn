# Plano de Monitoramento

## 1. Objetivo

Garantir que a API e os modelos continuem confiaveis apos a entrega, acompanhando desempenho tecnico, qualidade de dados e saude operacional.

## 2. O que monitorar

### Saude da API

- disponibilidade do endpoint `/health`
- latencia media e percentis do endpoint `/predict`
- taxa de erro HTTP 4xx e 5xx

### Qualidade da inferencia

- volume de requests por modelo
- distribuicao de `churn_probability`
- proporcao de predicoes positivas por janela de tempo

### Performance supervisionada

Quando houver rotulo real posterior:

- AUC
- recall
- precision
- F1-score
- taxa de churn real vs prevista

### Drift de dados

Monitorar mudancas de distribuicao em:

- `tenure_months`
- `monthly_charges`
- `total_charges`
- `contract`
- `internet_service`
- `payment_method`

## 3. Alertas Recomendados

- disponibilidade da API abaixo de 99%
- latencia p95 acima de 500 ms
- taxa de erro acima de 2%
- aumento abrupto de predicoes positivas acima de 20% da media historica
- queda de AUC maior que 5 pontos percentuais
- drift relevante nas features criticas

## 4. Playbook de Resposta

### Incidente de API

1. Verificar se o `/health` responde.
2. Conferir logs da aplicacao.
3. Identificar se o problema esta no carregamento do modelo ou no payload.
4. Reiniciar o servico se necessario.
5. Registrar causa raiz.

### Incidente de latencia

1. Verificar o header `X-Process-Time-Ms`.
2. Comparar latencia por modelo (`logistic` vs `mlp`).
3. Inspecionar carga da maquina e I/O do disco.
4. Se necessario, priorizar temporariamente o baseline logistico.

### Incidente de drift

1. Comparar a distribuicao atual das features com a referencia de treino.
2. Confirmar se houve mudanca de negocio ou coleta.
3. Reexecutar EDA rapida no periodo afetado.
4. Re-treinar e revalidar os modelos antes de promover nova versao.

### Queda de performance supervisionada

1. Confirmar se o calculo das metricas esta correto.
2. Verificar mudanca de threshold.
3. Inspecionar drift e desbalanceamento.
4. Re-treinar baseline e MLP.
5. Atualizar Model Card e registrar o novo run no MLflow.

## 5. Frequencia Recomendada

- API e latencia: monitoramento continuo
- distribuicao de predicoes: diario
- drift de dados: semanal
- performance supervisionada: mensal ou por ciclo de campanha

## 6. Acao Preventiva

- manter versoes dos modelos em `models/trained/`
- registrar todos os treinos no MLflow
- repetir a suite de testes antes de trocar artefatos
- revisar o Model Card a cada nova versao promovida
