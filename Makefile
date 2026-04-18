install:
	python -m pip install -e .

lint:
	python -m ruff check .

test:
	python -m pytest -q

run-api:
	python -m uvicorn src.api.main:app --reload

train-logistic:
	python -m src.models.train_baseline

train-baselines:
	python -m src.models.train_all_baselines

train-mlp:
	python -m src.models.train_mlp

mlflow-ui:
	python -m mlflow ui --backend-store-uri ./mlruns
