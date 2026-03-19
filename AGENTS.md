# SapientML — Agent Memory

## Repository Overview
- **Purpose**: AutoML library that generates scikit-learn pipelines from tabular data
- **Branch structure**: `main` + feature branches; PRs via GitHub
- **Python support**: `>=3.9,<3.14`; `requires-python` extended from `<3.13` in PR #112

## Key Dependencies & Constraints
- `sapientml-core 0.7.3` pins **`scikit-learn==1.6.1`** exactly
  - sklearn 1.6.0 removed `AdaBoostClassifier(algorithm='SAMME.R')`; only `'SAMME'` is valid
  - sapientml-core still generates `SAMME.R` internally — trials fail at WARNING level via optuna
- `numpy = ">=1.19.5,<3.0.0"` — previously capped at `<2.0.0` for fasttext-wheel (commit 279f99d); cap removed in PR #112 after sapientml-core replaced fasttext-wheel with langdetect
- `poetry` for dependency management (but `sapientml-core` uses `uv`)

## Releases
| Version | Date | Notes |
|---------|------|-------|
| 0.4.16 | 2026-03-18 | LanceDB (#91), datetime fix (#111); requires sapientml-core 0.7.4 |

## CI / GitHub Actions
- Workflow: `.github/workflows/release.yml`
- **Artifact name pattern**: `${{ matrix.version }}-${{ matrix.test }}` — MUST include the Python version prefix; without it, 3.10 and 3.11 matrix jobs upload to the same artifact name and the second upload gets a 409 Conflict, triggering a cascade cancellation via `fail-fast`
- **Strategy**: `fail-fast: false` on both `test` and `additional_test` matrices; `overwrite: true` on Upload Coverage for clean reruns
- Coverage: `report_coverage` job runs after all test + additional_test jobs complete

## Test Patterns
- `caplog` assertions for HPO tests: use `assert not any(r.levelno >= logging.ERROR for r in caplog.records)` **not** `assert "Error" not in caplog.text`
  - Reason: optuna logs failed HPO trials at WARNING level with messages that contain exception class names like `InvalidParameterError` — substring `"Error"` is a false positive
- Test files: `tests/sapientml/test_sapientml.py`

## Key Functions in params.py
- `_normalize_mixed_datetime_columns(df)`: converts object columns containing `pd.Timestamp` values to `datetime64[ns]`; applied to training/validation/test DataFrames in `Dataset.__init__`. Branches:
  - `non_null.empty` → `continue` (all-NaN object column)
  - `any(isinstance(v, pd.Timestamp) for v in non_null)` → False (no Timestamps, skip)
  - `any(...)` → True → coerce with `pd.to_datetime(..., errors="coerce")`

## Test Naming Conventions for CI Groups
CI runs `pytest -k <group>`. Test function names must contain the group keyword.
| Group | Example test names |
|-------|--------------------|
| `test_misc` | `test_misc_json_util`, `test_misc_normalize_mixed_*` |
| `test_sapientml_works_with` | `test_sapientml_works_with_*` |
Unit tests for internal functions go in `tests/sapientml/test_util.py` with `test_misc_` prefix.

## Codecov Notes
- Patch coverage threshold enforced on every PR
- To cover `_normalize_mixed_datetime_columns`: unit tests in `test_util.py` cover all 3 branches
  without running the heavy `SapientML.fit()` pipeline

## Commit History (feature/lancedb-integration)
| SHA      | Description |
|----------|-------------|
| 780be97  | base (lancedb integration) |
| 279f99d  | fix: numpy<2.0.0 constraint (fasttext-wheel compat) |
| 6821ed9  | ci: prefix artifact names with Python version |
| 9713da7  | fix(tests): check log level instead of text for HPO trial errors |
| 85cdfb6  | docs: add AGENTS.md |
| 26d00ac  | test: add unit tests for _normalize_mixed_datetime_columns (coverage fix) |

## Common Commands
```bash
# Run a specific test group locally
poetry run pytest tests/sapientml/test_sapientml.py -k test_misc -v

# Check CI status
gh run list --repo sapientml/sapientml --branch feature/lancedb-integration --limit 5
gh run view <run-id> --repo sapientml/sapientml | grep -E "^✓|^X|^\*"
```
