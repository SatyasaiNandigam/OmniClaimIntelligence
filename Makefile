setup:    ; pip install -e ".[dev]"
run-api:  ; uvicorn claimtower.api.main:app --reload
run-ui:   ; streamlit run src/claimtower/ui/app.py
test:     ; pytest
test-settlement: ; pytest --cov=claimtower.settlement --cov-branch --cov-report=term-missing --cov-fail-under=95
lint:     ; ruff check . && lint-imports
eval:     ; python evals/runners/run_eval.py
data:     ; python scripts/generate_policy_corpus.py && python scripts/generate_synthetic_docs.py && python scripts/seed_pas_mock.py && python scripts/build_fixtures.py
data-check: ; python scripts/generate_synthetic_docs.py --check && python scripts/seed_pas_mock.py --check && python scripts/build_fixtures.py --check
schemas:  ; python scripts/export_schemas.py
fixtures: ; python scripts/build_fixtures.py
freeze-check: ; python scripts/freeze_contracts.py --check
