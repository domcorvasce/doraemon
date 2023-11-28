.PHONY: install
install:
	poetry install

.PHONY: format
format:
	poetry run isort .
	poetry run black .

.PHONY:  test
test:
	poetry run pytest tests
