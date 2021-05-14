.PHONY: help clean dev docs package test

help:
	@echo "This project assumes that an active Python virtualenv is present."
	@echo "The following make targets are available:"
	@echo "	 dev 	install all deps for dev env"
	@echo "  docs	create pydocs for all relveant modules"
	@echo "	 test	run all tests with coverage"

clean:
	rm -rf dist/*

dev:
	pip install --upgrade pip
	pip install -e .

# docs:
# 	$(MAKE) -C docs html

package:
	python -m build --sdist --wheel --outdir dist/ .

test:
	pip install -r test-requirements.txt
	coverage run --omit 'venv/*' -m pytest
	coverage html -i
