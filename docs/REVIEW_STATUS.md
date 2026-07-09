# ETFTrader — review status & handoff to Claude Code

_Prepared from the Cowork migration session. This is a **structural / config-level triage**,
not a line-by-line correctness audit of the trading logic — that deep pass is what Claude Code
should do locally, with your `etftrader` env and test suite._

> **Scope (per ADR-0001): ETF smart-beta path only.** Refactor and audit
> `data_collection` (ETF), `factors`, `portfolio`, `backtesting`, `utils`. **Leave FX
> (`fx_options`, `fx_spot`, `signals`, `s8…s10`) and leveraged (`src/leveraged`, `s11`/`s12`)
> untouched** — they're deferred to later projects, which keeps the risky leveraged-execution
> scripts out of the refactor entirely.

## Migration status (Cowork session — done)

| Item | Status |
|---|---|
| Clean repo at `C:\Users\stuar\code\ETFTrader` (code only, cruft stripped) | ✅ |
| Full git history carried over (`main`, HEAD `b78a7db`) | ✅ |
| Data relocated to `C:\Users\stuar\trade_data\ETFTrader` (316 MB, verified) | ✅ |
| Miniconda (`%APPDATA%\miniconda3`) + `etftrader` env built | ✅ |
| IB Gateway installed (`C:\Jts`), **live** account, port 4001, Read-Only API | ✅ |
| `environment.yml`, `.env` (Windows paths), `.env.example`, `setup_windows.ps1`, `.vscode` | ✅ |
| `.env` auto-loader added to `src/__init__.py` | ✅ |
| **Import unification (Option A)** | ⛔ **deferred to Claude Code** (see below) |
| `pip install -e .` / `setup_windows.ps1` verification run | ⏳ pending refactor |

The repo currently runs the **original way** (mixed import styles held together by `sys.path`
inserts). Nothing is left in a half-broken state.

## Codebase shape

A genuinely substantial quant stack: `src/` package (factors, portfolio optimiser/rebalancer/
risk-manager, backtesting engine + cost models, data collection, fx-options, leveraged),
a staged `s1…s12` notebook-script pipeline, `daily_*` cron jobs, IB execution scripts, and a
pytest suite with markers. The `CLAUDE.md` is excellent and safety-aware.

## Strengths
- Clear separation (`src/` package vs notebooks vs scripts vs tests).
- Real test suite with sensible markers (`unit`/`integration`/`slow`/`requires_data`).
- Safety discipline already documented: paper-default, mock-IB-in-tests, fail-loud execution.
- No hardcoded machine paths in `.py` — everything flows through `.env`.

## Findings / things to review (prioritised)

**P1 — correctness-sensitive, verify with tests running (Claude Code):**
1. **Import inconsistency + `sys.path` sprawl (ETF-path files only).** Unify on `from src.…`,
   delete the redundant `sys.path` inserts, set `pyproject` → `include = ["src*"]`, then
   `pip install -e .`. Touch only ETF-path modules and `s1…s7`; **leave `s8…s12` (FX/leveraged)
   alone**. Each edit to an ETF execution script needs a test pinning the order payload.
2. **Order-construction & stop-loss logic — ETF path** (`scripts/ib_execute_trades.py`,
   `src/portfolio/*`, `s7`) — the actual audit you wanted: look-ahead bias, alignment,
   fee handling, exact order side/qty/tif/account. Not yet reviewed here. (Leveraged `s12` audit
   is deferred with its strategy.)
3. **Windows test-suite status unknown** — run `pytest -m "not slow and not requires_data"`
   in the `etftrader` env; fix whatever the move surfaced.

**P2 — hygiene:**
4. **`requirements.txt` is stale** (a `pip freeze` that *missed* real deps like `openpyxl`,
   `tabulate`, and carried unused `pandas-ta`). Superseded by `environment.yml` — delete or
   regenerate from the built env.
5. **Secrets in `.env`** (Databento key, Apple app-password) — parked at your request; rotate
   when convenient. `.env.example` is clean.
6. **Notebook outputs** — `CLAUDE.md` says strip before commit; several `notebooks/*.ipynb`
   show as modified. Consider `nbstripout`.

**P3 — deferred features:**
7. **databento** collectors — secondary project, dependency intentionally omitted.
8. Recommended guardrails from your own `CLAUDE.md` not yet enforced: `ruff`, `mypy --strict`,
   pre-commit, CI.

## Why finish this in Claude Code (not Cowork)
The refactor and the logic audit require editing code and **running the test suite on each
change** in your real Windows `etftrader` env. Claude Code does that locally in a tight loop,
is git-aware, and auto-reads your `CLAUDE.md`. Cowork (this session) can't run your Windows
env — so doing order-placing edits from here would be changing safety-critical code blind.

## Suggested Claude Code session agenda
1. `conda activate etftrader`; `pip install -e .`; run the fast test subset → capture baseline.
2. Do the import unification (P1.1) file-by-file, tests green after each.
3. Audit order-construction / stop-loss paths (P1.2) with pinned tests.
4. Hygiene sweep (P2): drop stale `requirements.txt`, `nbstripout`, optional `ruff`/`mypy`/CI.

## What Cowork keeps doing here
Live IBKR work that doesn't touch the repo: reading your account, the flagship
`/portfolio-review`, drafting rebalance proposals for your approval, and the weekly scheduled
report — all advisory / draft-only against your live, Read-Only-API account.
