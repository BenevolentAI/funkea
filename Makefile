POETRY = poetry run
PYTHON = $(POETRY) python
MAKE = $(POETRY) make

# Add local bin to path, so we can use poetry
PATH := $(PATH):$(HOME)/.local/bin
export PATH

.PHONY: test docs

setup:
	test "$$(command -v sbt)" || (echo "Please install sbt (Scala compiler; please see https://www.scala-lang.org/download/)" && exit 1)
	test "$$(command -v poetry)" || (curl -sSL https://install.python-poetry.org | python3 -)
	poetry install

test_ldsc: setup
	$(PYTHON) -m pytest -vv --cov=ldsc tests/ldsc

test_funkea: setup
	$(PYTHON) -m pytest -vv --cov=funkea tests/funkea

test_scala:
	cd scala/spark-udf && sbt clean test

test: test_funkea test_ldsc test_scala

docs: setup
	$(MAKE) -C docs/source html