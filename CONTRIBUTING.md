# Contributing to `qts`

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]" + pre-commit hooks (commit and push stages)
make help             # list all targets
```

Tooling (all configured in `pyproject.toml`):

| Tool | Command | Gate |
|---|---|---|
| ruff lint | `make lint` | blocking in CI |
| ruff format | `make format` | `--check` non-blocking in CI until the one-shot format commit lands |
| pyright | `make typecheck` | `qts/core` strict and blocking; rest of `qts` basic and non-blocking |
| pytest | `make test-fast` | blocking in CI (Python 3.10 and 3.12) |
| slow tests | `make test-slow` | nightly scheduled job |

Pre-commit runs ruff, whitespace/EOF fixers, YAML/TOML checks, private-key detection and a
500 kB large-file guard on every commit, and the fast test tier on `git push`.

## Branches and commits

- Branch from `main`: `feat/<topic>`, `fix/<topic>`, `refactor/<topic>`, `docs/<topic>`, `chore/<topic>`.
- Commit subject: imperative, at most 72 characters, optionally prefixed with the layer
  (`risk: shrink specific variance toward structural model`). Body explains *why*, and cites
  the paper/section for any methodology change.
- One logical change per commit; refactors that move code go in a separate commit from behaviour changes.
- **No AI co-author trailers** (`Co-Authored-By: <assistant>` and similar) in commit messages.
- Never commit data, credentials, `runs/` output or notebooks with outputs (see `.gitignore`).

## Test tiers

| Tier | Selector | Budget | When |
|---|---|---|---|
| fast | `-m "not slow"` | under ~5 min | every push, every PR |
| slow | `-m slow` | minutes to an hour | nightly; locally before merging changes to ML training, the pipeline or the simulator |
| integration | `-m integration` | varies | tests spanning several layers end-to-end |

Mark long tests with `@pytest.mark.slow` and cross-layer tests with `@pytest.mark.integration`.
Markers are strict (`--strict-markers`): an unregistered marker is an error. Property-based
tests use `hypothesis`; keep `max_examples` small in the fast tier.

### Golden file

The end-to-end regression test compares demo-pipeline output against a stored golden file.
When a change *intentionally* moves the numbers:

```bash
pytest --update-golden            # rewrites the golden file
git diff                          # inspect every changed number
```

Explain in the commit body why each metric moved. Never update the golden file to make an
unexplained diff go away.

## Point-in-time (PIT) rules for new features

1. Every new feature builder must pass `qts.validation.leakage.pit_shift_test`: values at
   date *t* must be identical whether the input data is truncated at *t* or not. Add the test
   alongside the feature.
2. Labels (forward returns, residualised targets) are **never** used as features, directly or
   through a transformation. Anything computed from data after *t* is a label.
3. No `shift(-k)`, `bfill`, centred rolling windows or full-sample normalisation in feature
   code; the AST tripwire in `qts.validation.leakage` will flag `shift(-k)`.
4. Fundamentals are keyed on filing/publication date, not period end; short interest carries
   its publication lag.
5. Walk-forward training must keep the label-horizon embargo; do not shorten it to "get more data".

## Review checklist for math changes

Any change to an estimator, objective, constraint, cost formula or statistic requires:

- [ ] **A hand computation or a reference implementation** in the test (small worked example
      computed by hand, or agreement with a trusted library/paper table to a stated tolerance).
- [ ] The docstring formula matches the code, with every symbol and its units defined
      (per-day vs per-horizon, bps vs fraction, annualised or not).
- [ ] The source (paper, section/equation) is cited in the docstring.
- [ ] PIT rules above still hold; `pit_shift_test` passes for affected features.
- [ ] Edge cases covered: NaNs, empty cross-sections, a single name, zero variance, delistings.
- [ ] No new magic numbers: use `SystemConfig` (e.g. `trading_days_per_year`) or a named constant.
- [ ] Golden-file diffs, if any, are explained in the commit body.
- [ ] `make lint typecheck test-fast` passes locally.
