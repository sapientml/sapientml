# SapientML — Agent Memory

## Repository Overview
- **Purpose**: AutoML library that generates scikit-learn pipelines from tabular data
- **Branch structure**: `main` + feature branches; PRs via GitHub
- **Python support**: 3.10, 3.11 (CI matrix); Python 3.13 compatible after fixes

## Key Dependencies & Constraints
- `sapientml-core 0.7.3` pins **`scikit-learn==1.6.1`** exactly
  - sklearn 1.6.0 removed `AdaBoostClassifier(algorithm='SAMME.R')`; only `'SAMME'` is valid
  - sapientml-core still generates `SAMME.R` internally — trials fail at WARNING level via optuna
- `numpy = ">=1.19.5,<2.0.0"` — upper bound required for fasttext-wheel 0.9.2 compatibility
- `poetry` for dependency management; `poetry.lock` pins numpy to 1.26.4

## CI / GitHub Actions
- Workflow: `.github/workflows/test.yml`
- Artifact name pattern: `py${{ matrix.version }}-${{ matrix.test }}` (avoids 409 conflicts when multiple matrix jobs upload coverage)
- Strategy `fail-fast: true` → one real failure cancels all sibling jobs (many X annotations = cascading cancels, not real failures)
- Coverage: `report_coverage` job runs after all test jobs

## Test Patterns
- `caplog` assertions for HPO tests: use `assert not any(r.levelno >= logging.ERROR for r in caplog.records)` **not** `assert "Error" not in caplog.text`
  - Reason: optuna logs failed HPO trials at WARNING level with messages that contain exception class names like `InvalidParameterError` — substring `"Error"` is a false positive
- Test files: `tests/sapientml/test_sapientml.py`

## Commit History (feature/lancedb-integration)
| SHA      | Description |
|----------|-------------|
| 780be97  | base (lancedb integration) |
| 279f99d  | fix: numpy<2.0.0 constraint (fasttext-wheel compat) |
| 6821ed9  | ci: prefix artifact names with Python version |
| 9713da7  | fix(tests): check log level instead of text for HPO trial errors |

## Common Commands
```bash
# Run a specific test group locally
poetry run pytest tests/sapientml/test_sapientml.py -k test_misc -v

# Check CI status
gh run list --repo sapientml/sapientml --branch feature/lancedb-integration --limit 5
gh run view <run-id> --repo sapientml/sapientml | grep -E "^✓|^X|^\*"
```
