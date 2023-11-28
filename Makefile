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
	poetry run mypy . --explicit-package-bases

.PHONY:  test
test:
	poetry run pytest tests
