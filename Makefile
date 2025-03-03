clean:
	rm -rf mmd-tst-variable-detector-*
	rm -rf dist/mmd-tst-variable-detector-*
	rm -rf build/
	rm -rf *egg-info
build:
	rm -rf mmd-tst-variable-detector-*
	rm -rf dist/mmd-tst-variable-detector-*
	poetry build --format sdist
	tar -xvf dist/*-`poetry version -s`.tar.gz
	mv mmd_tst_variable_detector-* mmd-tst-variable-detector-latest
install:
	pip install -e .
install-dev:
	poetry install --no-dev