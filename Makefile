VENV := ./.venv/bin

.PHONY: install run test lint replay preview

install:
	python3 -m venv .venv
	$(VENV)/pip install -e ".[dev]"

run:
	$(VENV)/uvicorn aigateway.main:app --reload --port 8080

test:
	$(VENV)/python -m pytest -q

lint:
	$(VENV)/ruff check src tests

# Offline: re-score recorded traffic against alternative routing policies.
replay:
	$(VENV)/python -m aigateway.replay.harness var/records.db

# Dry-run the router against a sample request; spends nothing.
preview:
	curl -s localhost:8080/admin/route/preview \
		-H 'content-type: application/json' \
		-d '{"model":"auto","messages":[{"role":"user","content":"Review this diff for security issues"}]}' \
		| python3 -m json.tool
