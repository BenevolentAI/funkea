POETRY = poetry run
PYTHON = $(POETRY) python
MAKE = $(POETRY) make

# Add local bin to path, so we can use poetry
DOTLOCAL := $(HOME)/.local
POETRY_HOME := $(DOTLOCAL)/share/pypoetry
COURSIER_HOME := $(DOTLOCAL)/share/coursier
PATH := $(PATH):$(POETRY_HOME)/bin:$(COURSIER_HOME)

export PATH
export POETRY_HOME
export COURSIER_HOME

ifeq ($(OS),Windows_NT)
	error "Windows is not supported"
else
	UNAME_S := $(shell uname -s)
	UNAME_M := $(shell uname -m)
	ifeq ($(UNAME_S),Linux)
		# For Linux with AARCH64 / ARM64 we cannot install JAX yet, as they
		# do not provide wheels for this architecture. https://github.com/google/jax/issues/7097
		ifeq ($(UNAME_M),x86_64)
			# Linux 64-bit
			POETRY_FLAGS := --extras "jax"
		endif
	else
		POETRY_FLAGS := --extras "jax"
	endif
endif

.PHONY: test docs

setup:
	test "$$(command -v java)" || (echo "PySpark requires Java 8 or later" && exit 1)
	test "$$(command -v sbt)" || \
		(./scripts/install-scala.sh $(COURSIER_HOME) || \
			(echo "Unable to install Scala compiler. Please install sbt (Scala compiler; please see https://www.scala-lang.org/download/)" && \
			exit 1) \
		)
	test "$$(command -v poetry)" || (mkdir -p $(POETRY_HOME) && curl -sSL https://install.python-poetry.org | python3 -)
	poetry install --without docs $(POETRY_FLAGS)

test_ldsc: setup
	$(PYTHON) -m pytest -vv --cov=ldsc tests/ldsc

test_funkea: setup
	$(PYTHON) -m pytest -vv --cov=funkea tests/funkea

test_scala:
	cd scala/spark-udf && sbt clean test

test: test_funkea test_ldsc test_scala

docs: setup
	poetry install --only docs
	$(MAKE) -C docs/source html

publish:
	# we build the wheel first to ensure that we have the latest jars compiled
	# then we build the sdist for the actual distribution (containing the jar)
	# this is very hacky, but it works
	poetry build --format wheel && rm -rf dist && poetry build --format sdist
	poetry publish -u __token__
