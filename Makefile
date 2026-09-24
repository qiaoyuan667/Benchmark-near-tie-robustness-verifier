.PHONY: install test figures audit verify

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements-pinned.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:$(PYTHONPATH) $(PYTHON) -m pytest -q -p no:cacheprovider

figures:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/render_paper_figures.py

audit:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/audit_release.py

verify: test audit
