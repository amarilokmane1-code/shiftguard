# Thin wrapper around tasks.py so `make <task>` works on Linux/macOS/Kaggle.
# On Windows use: python tasks.py <task>
PY ?= python

.PHONY: install lint fmt test test-fast pipeline synthetic clean
install lint fmt test test-fast pipeline synthetic clean:
	$(PY) tasks.py $@
