"""Manual refresh of the ETF fundamentals cache.

Two modes:

  # Small live test — hit a handful of tickers, print the results, don't cache.
  python scripts/refresh_fundamentals.py --smoke

  # Full universe refresh — hit every ticker, write parquet, print coverage report.
  python scripts/refresh_fundamentals.py --full

The smoke test exists so schema breaks on any one issuer are caught before
they poison the cache. Run it first after any change to
`src/data_collection/issuer_fundamentals.py`, or if a refresh returns
suspicious coverage numbers.

Use --parallel/--no-parallel and --workers to tune throughput. The default
(5 workers, parallel on for --full, off for --smoke) is a good balance
between honest rate-limiting and not taking half a day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data_collection.issuer_fundamentals import (
    DEFAULT_CACHE_PATH,
    FundamentalsRouter,
    coverage_report,
    save_cache,
)

# Small smoke set — one from each of the big-5 issuers we plan to cover.
# Deliberately picks large, obvious equity ETFs whose fundamentals are
# guaranteed to exist and whose numbers we roughly know, so a schema break
# reveals itself immediately in the printed output.
SMOKE_TICKERS = ["VOO", "VTI", "VUG", "VYM"]

# Coverage thresholds — refuse to overwrite the cache below these.
# Rationale: a partial refresh that trashes the value factor for half the
# universe is worse than a stale cache.
MIN_COVERAGE_FRAC = 0.30  # loose while only Vanguard is implemented; tighten later


def _load_universe() -> list[str]:
    """Pull the full ETF universe from the comprehensive list."""
    from src.data_collection.comprehensive_etf_list import COMPREHENSIVE_ETF_UNIVERSE
    universe: set[str] = set()
    for tickers in COMPREHENSIVE_ETF_UNIVERSE.values():
        universe.update(tickers)
    return sorted(universe)


def _print_frame(df) -> None:
    cols = ["ticker", "pe_ratio", "pb_ratio", "dividend_yield", "as_of", "source"]
    print(df[cols].to_string(index=False))


def _print_coverage(report: dict) -> None:
    print()
    print("Coverage:")
    print(f"  total tickers:      {report['total']}")
    print(f"  any field populated: {report['any_field']} "
          f"({report.get('any_field_pct', 0.0):.1%})")
    print(f"  P/E populated:       {report['pe']}")
    print(f"  P/B populated:       {report['pb']}")
    print(f"  Div-yield populated: {report['dy']}")
    print("  by source:")
    for src, n in sorted(report.get("by_source", {}).items(), key=lambda kv: -kv[1]):
        print(f"    {src:12s} {n}")


def main() -> int:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help=f"Live-fetch smoke tickers ({', '.join(SMOKE_TICKERS)}) and print.")
    mode.add_argument("--full", action="store_true",
                      help="Full universe refresh; writes the parquet cache on success.")
    p.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=None,
                   help="Parallelise across issuer scrapers. Default: on for --full, off for --smoke.")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    args = p.parse_args()

    router = FundamentalsRouter()

    if args.smoke:
        parallel = False if args.parallel is None else args.parallel
        print(f"Smoke test: fetching {len(SMOKE_TICKERS)} tickers (parallel={parallel})")
        df = router.fetch_many(SMOKE_TICKERS, parallel=parallel, max_workers=args.workers)
        _print_frame(df)
        _print_coverage(coverage_report(df))
        return 0

    # --full
    universe = _load_universe()
    parallel = True if args.parallel is None else args.parallel
    print(f"Full refresh: fetching {len(universe)} tickers "
          f"(parallel={parallel}, workers={args.workers})")

    df = router.fetch_many(universe, parallel=parallel, max_workers=args.workers)
    report = coverage_report(df)
    _print_coverage(report)

    if report["any_field_pct"] < MIN_COVERAGE_FRAC:
        print(f"\nERROR: coverage {report['any_field_pct']:.1%} below threshold "
              f"{MIN_COVERAGE_FRAC:.0%} — refusing to overwrite cache.")
        print("Investigate before rerunning.")
        return 1

    save_cache(df, args.cache_path)
    print(f"\nOK — cache written to {args.cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
