.PHONY: run test eval fmt lint

run:
	python -m app.cli "Compare the top 3 open-source vector databases for a startup building RAG products."

test:
	pytest -q

eval:
	python -m evals.run_eval

fmt:
	ruff format .
	ruff check --fix .

lint:
	ruff check .

