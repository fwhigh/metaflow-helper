.PHONY: help clean dev docs package test

help:
	@echo "This project assumes that an active Python virtualenv is present."
	@echo "The following make targets are available:"
	@echo "	 dev 	install all deps for dev env"
	@echo "  docs	create pydocs for all relveant modules"
	@echo "	 test	run all tests with coverage"

clean:
	rm -rf dist build *.egg-info .pytest_cache htmlcov

dev:
	pip install --upgrade pip
	pip install --upgrade --upgrade-strategy eager -r requirements.txt
	pip install -e .

example:
	pip install --upgrade pip
	pip install -r example-requirements.txt
	jupyter labextension install jupyterlab-plotly

# docs:
# 	$(MAKE) -C docs html

package:
	pip install build
	python -m build --sdist --wheel --outdir dist/ .

test:
	pip install -r test-requirements.txt
	coverage run --omit 'venv/*' -m pytest
	coverage html -i

test_examples:
	pip install -r example-requirements.txt
	python examples/model-selection/train.py run --test_mode 1
