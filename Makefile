.PHONY: install
install:
	poetry install

.PHONY: format
format:
	poetry run isort .
	poetry run black .

.PHONY: lint
lint:
	poetry run flake8 .

.PHONY: type-check
type-check:
	poetry run mypy .

.PHONY: docs-serve
docs-serve:
	poetry run mkdocs serve --dev-addr=0.0.0.0:1337

.PHONY:  test
test:
	poetry run pytest tests
