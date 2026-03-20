.PHONY: dist
dist: build

.PHONY: fmt
fmt:
	bash fmt.sh

.PHONY: twine
twine:
	pixi run -e build --frozen python -m twine upload --config-file .pypirc dist/*

.PNONY: build
build:
	pixi run -e build --frozen python -m build
