PYTHON ?= python3
VENV_PYTHON := .venv/bin/python

.PHONY: setup notebook

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	$(VENV_PYTHON) -m pip install jupyter

notebook:
	$(VENV_PYTHON) -m jupyter notebook
