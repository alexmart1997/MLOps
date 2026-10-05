# Case 1 — прогноз оттока клиентов банка

Бинарная классификация оттока (`Exited`) на данных
[Kaggle Playground Series S4E1](https://www.kaggle.com/competitions/playground-series-s4e1/data).
Ключевая метрика — **recall**: важно найти как можно больше уходящих клиентов.

Исследование (EDA, сравнение моделей, SHAP) — в [`notebooks/Case_1.ipynb`](notebooks/Case_1.ipynb).
Production-код — в пакете [`src/churn`](src/churn).

## Структура

```
├── configs/config.yaml     # колонки, сетки гиперпараметров, пути — всё, что было зашито в ноутбук
├── src/churn/
│   ├── config.py           # типизированная загрузка конфига
│   ├── data.py             # чтение CSV + проверка схемы, стратифицированный split
│   ├── features.py         # drop_columns, add_has_balance
│   ├── pipeline.py         # sklearn-пайплайны: logreg / decision_tree / random_forest / catboost
│   ├── train.py            # отбор моделей (GridSearchCV) и обучение финальной (Optuna)
│   ├── predict.py          # инференс сохранённой модели
│   └── cli.py              # точка входа `churn`
├── tests/                  # pytest на синтетических данных (данные не нужны)
├── notebooks/              # EDA, коммитится без выводов (nbstripout)
├── data/ models/ reports/  # содержимое не попадает в git
├── pyproject.toml          # зависимости (poetry) + настройки black / isort / mypy / pytest
├── poetry.lock             # зафиксированные версии всех зависимостей
├── poetry.toml             # .venv создаётся внутри проекта
├── .pre-commit-config.yaml # хуки: форматтеры, линтеры, проверки
├── .flake8
└── .gitlab-ci.yml          # lint + test
```

## Быстрый старт

Нужны Python 3.10–3.12 и [Poetry](https://python-poetry.org/) ≥ 2.0 (`pipx install poetry`).

```bash
make install            # poetry install + pre-commit install
# положить train.csv из Kaggle в data/raw/train.csv
make test               # тесты
make select             # сравнить кандидатов  -> reports/model_selection.csv
make train              # обучить CatBoost + Optuna -> models/model.joblib, reports/metrics.json
make predict INPUT=data/raw/test.csv   # -> reports/predictions.csv
```

Без make: `poetry run churn --help`. Другой конфиг: `poetry run churn -c my.yaml train`.
Уровень логов: переменная окружения `LOG_LEVEL` (по умолчанию `INFO`).

Для ноутбука: `make install-eda` (jupyter, seaborn, phik, shap — отдельная опциональная группа,
в production-окружение не ставятся).

## Виртуальное окружение и git

- `poetry.toml` задаёт `virtualenvs.in-project = true`: окружение живёт в `.venv/` рядом с кодом,
  IDE и pre-commit находят его без настройки.
- Сам `.venv/` **не коммитится** (`.gitignore`). В git лежат `pyproject.toml` и `poetry.lock` —
  по ним `poetry install` воссоздаёт окружение с теми же версиями на любой машине и в CI.
- Хук `poetry-check` не даст закоммитить `pyproject.toml`, рассинхронизированный с `poetry.lock`.
- Добавить зависимость: `poetry add <pkg>` (dev: `poetry add --group dev <pkg>`) — коммитить оба файла.

## Качество кода

| Инструмент | Что делает | Конфиг |
|---|---|---|
| black | форматирование, строка 99 | `pyproject.toml` |
| isort | порядок импортов (профиль black) | `pyproject.toml` |
| flake8 + bugbear | PEP8, ошибки, сложность ≤ 10 | `.flake8` |
| mypy | типы в `src/` | `pyproject.toml` |
| pytest + coverage | тесты и покрытие | `pyproject.toml` |
| nbstripout | вычищает выводы ноутбуков | `.pre-commit-config.yaml` |
| pre-commit-hooks | пробелы, YAML/TOML, большие файлы, приватные ключи, запрет коммита в `main` | `.pre-commit-config.yaml` |

Линтеры запускаются из poetry-окружения, поэтому версии одинаковы локально, в хуках и в CI.
Прогнать все хуки вручную: `make lint`.

## Ветвление (GitLab Flow)

Прямые коммиты в `main` запрещены хуком `no-commit-to-branch`. Работа идёт в feature-ветках
(`git checkout -b feature/<name>`) и попадает в `main` через Merge Request после зелёного CI.

## Что изменилось по сравнению с ноутбуком

- Код разнесён по модулям с типами и docstring'ами; глобальные переменные и копипаста
  (`DecisionTreeMetrics_info`, `RandomForestResults_info`, …) заменены одной функцией `run_selection`.
- Все параметры вынесены в `configs/config.yaml`.
- Удаление малоинформативных признаков (`HasCrCard`, `EstimatedSalary`, `Tenure`) перенесено
  внутрь пайплайна: сохранённая модель принимает сырые данные в исходном формате.
- `drop_columns` терпит отсутствие технических колонок — на инференсе не нужны `CustomerId`/`Surname`.
- Модель и метрики сохраняются как артефакты (`joblib`, JSON) вместо вывода в ячейки.
- `print` заменён на `logging`; данные читаются из локального пути с проверкой схемы.
- Зависимости зафиксированы в `poetry.lock` вместо `!pip install` в ячейке.
