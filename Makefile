install:
	pip install -e .

test:
	pytest -q

lint:
	ruff check .
