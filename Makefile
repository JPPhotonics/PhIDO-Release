
install:
	@if ! command -v uv > /dev/null; then \
		echo "uv not found. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv is already installed."; \
	fi
	uv venv --python 3.12
	uv sync

dev:
	uv run pip install -e .[dev,docs]

run:
	streamlit run PhotonicsAI/Photon/webapp.py

test:
	pytest -s

update-pre:
	pre-commit autoupdate

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

build:
	rm -rf dist
	pip install build
	python -m build

docs:
	jb build docs

.PHONY: drc doc docs