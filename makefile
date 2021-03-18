SHELL = /usr/bin/env bash

_python ?= python
PYTESTPARALLEL ?= --workers auto
EXTRAS ?=
COVHTML ?= --cov-report html
# for bump2version, valid options are: major, minor, patch
PART ?= patch
N_MPI ?= 2

# Main Targets #################################################################

.PHONY: test

test:
	$(_python) -m pytest -vv $(PYTESTPARALLEL) \
		--cov=src --cov-report term $(COVHTML) --no-cov-on-fail --cov-branch \
		tests

test-mpi:
	mpirun -n $(N_MPI) $(_python) -m pytest -vv --with-mpia \
		--capture=no \
		tests

# maintenance ##################################################################

.PHONY: pypi pypiManual pep8 flake8 pylint
# Deploy to PyPI
## by CI, properly git tagged
pypi:
	git push origin v0.1.0
## Manually
pypiManual:
	rm -rf dist
	tox -e check
	poetry build
	twine upload dist/*

# check python styles
pep8:
	pycodestyle . --ignore=E501
flake8:
	flake8 . --ignore=E501
pylint:
	pylint numba_quaternion

print-%:
	$(info $* = $($*))

# poetry #######################################################################

# since poetry doesn't support editable, we can build and extract the setup.py,
# temporary remove pyproject.toml and ask pip to install from setup.py instead.
editable:
	poetry build
	cd dist; tar -xf numba_quaternion-0.1.0.tar.gz numba_quaternion-0.1.0/setup.py
	mv dist/numba_quaternion-0.1.0/setup.py .
	rm -rf dist/numba_quaternion-0.1.0
	mv pyproject.toml .pyproject.toml
	$(_python) -m pip install -e .$(EXTRAS); mv .pyproject.toml pyproject.toml

# releasing ####################################################################

bump:
	bump2version $(PART)
	git push --follow-tags
