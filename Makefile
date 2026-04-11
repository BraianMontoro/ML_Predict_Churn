install:
	pip install -r requirements.txt

lint:
	ruff check .

test:
	pytest -q

run-api:
	uvicorn src.api.main:app --reload
