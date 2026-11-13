.PHONY: install eval test plots
install:
	pip install -e ".[dev]"
eval:            ## reproduce the results table (deterministic)
	python -m fab_yield.eval
test:
	python -m pytest -q
plots:
	python -m fab_yield.plots --level 3 --seed 1
