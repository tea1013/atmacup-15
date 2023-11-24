install:
	poetry install

setup: install
	poetry run pip install jupyterlab-vim

jupyter:
	poetry run jupyter-lab

fix:
	poetry run black exp
	poetry run isort exp

lint:
	poetry run pflake8 exp
	