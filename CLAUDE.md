# CLAUDE.md — ETFTrader

Guidance for Claude (and any AI assistant) working in this repo. Read this fully before
making non-trivial changes. Ask if anything here conflicts with what the user requests.

## What this repo is

A Python 3.10+ quantitative ETF / FX trading project. Real money flows through this code
via Interactive Brokers, so correctness and conservatism matter more than cleverness.

Layout:
- [src/](src/) — the package. All reusable logic lives here.
  - [src/backtesting/](src/backtesting/) — engine, metrics, cost & slippage models
  - [src/data_collection/](src/data_collection/) — yfinance, IB, Databento collectors
  - [src/factors/](src/factors/) — momentum / quality / value / volatility factors
  - [src/fx_options/](src/fx_options/) — FX option data and vol surface
  - [src/leveraged/](src/leveraged/) — leveraged ETF handling
  - [src/portfolio/](src/portfolio/) — position management, stop-loss, sizing
  - [src/utils/](src/utils/) — shared helpers (logging, ETF names)
- [notebooks/](notebooks/) — research and execution surfaces. They import from `src/`.
- [signals/](signals/) — FX signal generation
- [scripts/](scripts/) — daily cron shell scripts
- [tests/](tests/) — pytest suite

Config: [pyproject.toml](pyproject.toml), [pytest.ini](pytest.ini),
[requirements.txt](requirements.txt).

## Code style — Python

- **Type hints everywhere.** Annotate all public function signatures and dataclass fields.
  Match the style in [src/factors/base_factor.py](src/factors/base_factor.py).
- **Google-style docstrings** with a 1-line summary, then `Args:` / `Returns:` blocks.
  Don't paraphrase what the code does — explain shapes, units, and invariants
  (e.g. "prices: DataFrame with tickers as columns, dates as index, forward-filled").
- **Use the project logger**, never `print`, in `src/`:
  ```python
  from src.utils.logging_config import get_logger
  logger = get_logger(__name__)
  ```
  Prints inside notebooks are fine for exploration.
- **Dataclasses for config** (see `BacktestConfig`). **ABCs for extension points**
  (see `BaseFactor` in [src/factors/base_factor.py](src/factors/base_factor.py)).
- **No bare `except:`** and no silent swallowing — at minimum `logger.warning(...)` with the
  exception. Catch the narrowest exception type that makes sense.
- **No magic numbers in trading logic.** Promote them to module-level constants or fields on
  a config dataclass. A future reader needs to see what `0.02` means.
- **Don't refactor "while you're in there."** Bug fixes don't get cleanup; one-shot scripts
  don't get helpers. Three similar lines beat a premature abstraction.

## Code style — quantitative correctness

These rules exist because they have bitten this codebase or codebases like it. Treat them
as non-negotiable.

- **No look-ahead bias.** When computing a signal at time `t`, only use data with index
  `< t`. If you genuinely mean `<= t` (T+0 fill), say so in the docstring and add a test.
- **Explicit alignment** for time-series joins. Always use `.reindex(...)` / `.align(...)`
  on a known index. Never rely on positional joins of `pd.DataFrame` / `pd.Series`.
- **Returns vs. prices**: name variables unambiguously (`prices`, `returns_d`, `log_returns`,
  `excess_returns`) and document units in the docstring. Mixing the two silently is the
  classic backtest bug.
- **Costs and slippage** go through the cost model in [src/backtesting/](src/backtesting/).
  Don't recompute fees ad-hoc inside a strategy or notebook.
- **NaN handling** is explicit. Decide and document: drop, forward-fill (with a max gap),
  or treat as a signal of "no position." Don't let NaNs propagate silently.
- **Timezone-aware timestamps** for any code touching market data. Mixing naive and aware
  timestamps will hard-error in pandas — make the choice once, at the data boundary.

## Testing

- Framework: pytest, configured in [pytest.ini](pytest.ini). Coverage runs HTML +
  term-missing — keep it from regressing.
- **Markers** (apply them on every new test):
  - `unit` — pure logic, no I/O
  - `integration` — multiple modules, still local
  - `slow` — long-running
  - `requires_data` — needs an external dataset
- Quick-feedback loop: `pytest -m "not slow and not requires_data"`.
- **New code requires tests.** Bug fixes require a regression test that fails before the fix
  and passes after.
- **Mock IB.** Tests must never connect to a live broker — mock `ib_insync` clients at the
  module boundary.
- Minimum coverage targets for changes:
  - Factor calculations
  - Position sizing
  - Stop-loss / trailing-stop trigger conditions
  - Order construction (side, qty, type, tif, account)

## Notebooks

- **Import from `src/`.** Reusable logic does not live in a notebook. If you write
  something in a notebook that another notebook would also want, move it to `src/` and
  re-import.
- **Strip outputs before commit:**
  ```bash
  jupyter nbconvert --clear-output --inplace notebooks/<file>.ipynb
  ```
  Outputs leak data, balloon diffs, and sometimes contain credentials.
- **Never commit** account numbers, API keys, IB client IDs, or absolute machine-specific
  paths inside a notebook. Use environment variables or a gitignored config file.
- The execution notebooks ([02_execute_trades.ipynb](notebooks/02_execute_trades.ipynb),
  [03_leveraged_execute.ipynb](notebooks/03_leveraged_execute.ipynb)) are safety-critical
  — see the next section.

## Trading-safety rules (read carefully)

This codebase places real orders. Treat the following as safety-critical:

- **IB execution code** in [src/data_collection/](src/data_collection/) and the execution
  notebooks 02 / 03.
- **Order construction** anywhere — side, quantity, order type, time-in-force, account.
- **Stop-loss and trailing-stop logic** in [src/portfolio/](src/portfolio/).
- **Leveraged ETF rebalancing / decay** assumptions in [src/leveraged/](src/leveraged/).

Rules:

1. **Default to the paper-trading port.** Verify the IB connection target before running
   any execution cell. Never silently switch from paper to live — a switch is a deliberate,
   commented, user-confirmed action.
2. **Stop-loss and order-construction changes require explicit unit tests** that pin the
   trigger condition or the exact order payload. "Looks right" is not acceptable.
3. **Don't refactor execution code without explicit user approval**, even if it looks ugly.
4. **Fail loudly** in the trading path. Never `try/except: pass` around an order — if
   something is wrong, the right behavior is to halt, not to skip a fill silently.

## Workflow conventions for Claude

- **Plan non-trivial changes** before editing. For multi-file work, share the plan with the
  user first.
- **Run the relevant pytest subset** after edits (use markers; don't run the full suite for
  a one-line factor change).
- **Don't push, force-push, or open PRs** unless explicitly asked.
- **Don't widen scope.** This is a research repo; extra abstractions become foot-guns.
- **Match the existing style** before introducing new patterns. If you want to introduce
  one, propose it first.

## Recommended future guardrails (not yet enforced)

These are not configured today. If the user adopts them, the rules in this file should be
upgraded from "documented" to "enforced."

- **Linter / formatter:** [`ruff`](https://docs.astral.sh/ruff/) — single tool, replaces
  black + flake8 + isort. Suggested config: `select = ["E", "F", "I", "B", "UP", "SIM"]`,
  line length 100.
- **Type checker:** `mypy --strict` over `src/` (exclude `notebooks/`).
- **Pre-commit hooks:** `ruff`, `mypy`, and `nbstripout` so notebook outputs can't be
  committed by accident.
- **CI:** a `.github/workflows/test.yml` that runs `pytest -m "not slow and not requires_data"`
  plus the lint and type-check steps above.

If you (Claude) are asked to "set up tooling" or "add CI", this list is the starting point.
