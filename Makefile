.PHONY: install install-eda lint format test select train predict

install:  ## окружение + git-хуки
	poetry install
	poetry run pre-commit install

install-eda:  ## + jupyter, seaborn, phik, shap для ноутбука
	poetry install --with eda

lint:
	poetry run pre-commit run --all-files

format:
	poetry run isort .
	poetry run black .

test:
	poetry run pytest

select:
	poetry run churn select

train:
	poetry run churn train

predict:  ## make predict INPUT=data/raw/test.csv
	poetry run churn predict $(INPUT)
