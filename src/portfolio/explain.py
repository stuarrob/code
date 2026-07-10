"""Narrate the deterministic TradeProposal in plain English.

Two paths:

1. **Deterministic template** (`narrate_proposal`): reads the
   `TradeProposal` directly and produces a structured plain-English
   summary. No LLM. Always works, offline. Provides a defensible
   baseline that the LLM can then embellish.

2. **LLM narrator** (`narrate_with_claude`): sends the structured
   proposal + template + optional user question to the Anthropic API
   and returns the model's response. Requires ``ANTHROPIC_API_KEY``.

Design principles applied:

- **LLM narrates, does not decide.** The LLM never returns a number
  the code will act on. It receives the deterministic proposal as
  ground-truth and can only reword or answer questions about it.
- **The deterministic version is the reference.** If the LLM output
  ever disagrees with the deterministic version on a factual claim
  (e.g. "you are buying 500 SPY"), the deterministic version wins.
- **Structured input.** The proposal is serialised to a JSON-safe dict
  before being sent to the model, so the model can quote it verbatim
  when useful and there is no ambiguity about which trade is which.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

try:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)

from src.portfolio.proposal import (
    ACTION_BUY, ACTION_EXTEND, ACTION_SELL,
    TradeProposal,
)


DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"  # verified via API probe 2026-07-10; latest Opus


# ────────────────────────────────────────────────────────────────
# Deterministic narrator (always works)
# ────────────────────────────────────────────────────────────────

def narrate_proposal(proposal: TradeProposal,
                     policy_name: Optional[str] = None,
                     metadata: Optional[dict] = None) -> str:
    """Pithy plain-English summary. One headline, ≤ 6 bullets, no per-trade dump.

    Args:
        proposal: the ground-truth blotter.
        policy_name: optional label for the policy that produced it.
        metadata: optional {ticker → TickerMetadata} from
            `ticker_metadata.enrich_tickers`. When present, buys are
            grouped by geography and the dominant factor is quoted.

    Returns markdown. Designed to fit on one screen without scrolling.
    """
    n_trades = len(proposal.trades)

    if n_trades == 0:
        text = ("**No trades proposed.** Portfolio is inside the drift "
                "threshold and matches the target basket.")
        if proposal.warnings:
            text += "\n\n**Warnings:** " + "; ".join(proposal.warnings)
        return text

    buys = [t for t in proposal.trades if t.action == ACTION_BUY]
    sells = [t for t in proposal.trades if t.action == ACTION_SELL]
    extends = [t for t in proposal.trades if t.action == ACTION_EXTEND]

    # 1. Headline — one line
    lines = [
        f"**{n_trades} trades: {len(buys)} buy · {len(sells)} sell · "
        f"{len(extends)} extend.** "
        f"Turnover **${proposal.turnover_notional:,.0f}** "
        f"({proposal.turnover_pct_of_nav:.0%} NAV), cost "
        f"**${proposal.total_est_cost:,.0f}** "
        f"({proposal.total_est_cost / max(proposal.turnover_notional, 1) * 10_000:.1f} bps). "
        f"Ends with **{proposal.n_positions_after}** positions."
    ]

    # 2. Buying into — grouped by geography if metadata provided
    incoming = buys + extends
    if incoming:
        lines.append("")
        if metadata:
            by_geo: dict[str, float] = {}
            for t in incoming:
                m = metadata.get(t.ticker)
                geo = m.geography if m and m.geography else "Unknown"
                by_geo[geo] = by_geo.get(geo, 0.0) + t.delta_notional
            top_geo = sorted(by_geo.items(), key=lambda kv: -kv[1])[:4]
            geo_str = ", ".join(
                f"{g} ${v/1000:.0f}k" for g, v in top_geo
            )
            lines.append(f"**Buying into:** {geo_str}.")
        else:
            top_buys = sorted(incoming, key=lambda x: -x.delta_notional)[:4]
            lines.append(
                f"**Top buys:** " +
                ", ".join(
                    f"{t.ticker} ${t.delta_notional/1000:.0f}k"
                    for t in top_buys
                ) + "."
            )

    # 3. Selling out of — top few
    if sells:
        lines.append("")
        top_sells = sorted(sells, key=lambda x: -x.delta_notional)[:4]
        if metadata:
            sell_str = ", ".join(
                f"{t.ticker} ({metadata[t.ticker].geography or '?'}) "
                f"${t.delta_notional/1000:.0f}k"
                for t in top_sells if t.ticker in metadata
            )
        else:
            sell_str = ", ".join(
                f"{t.ticker} ${t.delta_notional/1000:.0f}k" for t in top_sells
            )
        lines.append(f"**Selling out of:** {sell_str}.")

    # 4. Factor tilt — one line, only significant deltas
    if proposal.factor_exposures:
        tilts = []
        for fe in proposal.factor_exposures:
            if fe.before != fe.before or fe.after != fe.after:
                continue
            if abs(fe.delta) < 0.02:
                continue
            arrow = "↑" if fe.delta > 0 else "↓"
            tilts.append(f"{fe.factor} {arrow} ({fe.delta:+.2f})")
        if tilts:
            lines.append("")
            lines.append(f"**Factor tilt:** {' · '.join(tilts)}.")

    # 5. Cash + warnings — one line each if present
    lines.append("")
    lines.append(
        f"**Cash after:** ${proposal.cash_after:,.0f} "
        f"(reserve target: policy-set)."
    )
    if proposal.warnings:
        lines.append("")
        lines.append(f"**⚠ {len(proposal.warnings)} warning(s).** "
                     f"First: {proposal.warnings[0]}")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# LLM-enhanced narrator (Anthropic Claude, optional)
# ────────────────────────────────────────────────────────────────

def anthropic_available() -> bool:
    """True if the Anthropic SDK is importable AND an API key is present."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def _proposal_to_dict(proposal: TradeProposal) -> dict:
    """Serialise the proposal to a JSON-safe dict for the LLM prompt."""
    return {
        "n_trades": len(proposal.trades),
        "turnover_notional": proposal.turnover_notional,
        "turnover_pct_of_nav": proposal.turnover_pct_of_nav,
        "total_est_cost": proposal.total_est_cost,
        "n_positions_after": proposal.n_positions_after,
        "cash_after": proposal.cash_after,
        "investable_nav": proposal.investable_nav,
        "trades": [asdict(t) for t in proposal.trades],
        "factor_exposures": [asdict(f) for f in proposal.factor_exposures],
        "warnings": list(proposal.warnings),
    }


_SYSTEM_PROMPT = """You are the narrator for a deterministic ETF smart-beta trading applet operated against a live account.

STRICT LENGTH: ≤ 4 short paragraphs, ≤ 200 words total. The operator reads this on-screen at rebalance time. Longer is worse.

STRUCTURE the initial narration as:
  1) One sentence — WHAT the portfolio is moving into (geography / asset-class shift).
  2) One sentence — WHY, via the factor tilt (which factors drove the picks).
  3) One sentence — cost + turnover in bps.
  4) One sentence — anything the operator should scrutinise (warnings, cash impact, unusual concentration). Skip if nothing.

For Q&A: answer in ≤ 3 sentences unless the question genuinely needs more.

STRICT SAFETY:
- Never invent trades or numbers. The TradeProposal is ground truth. Quote it verbatim when precision matters.
- Never compute "what if" scenarios. Refer the operator back to Steps 1–4 with new inputs.
- If a fact isn't in the proposal, say "not shown in the proposal".

Strategy context you may reference (do not restate):
35% momentum · 30% quality · 20% low-vol · 15% value (yield + expense-ratio blend). Weighted geometric mean of factor ranks. Top-30, exponential rank weights, 2–15% bounds, 10% trailing stops. Rebalance bimonthly with 5% drift threshold. Universe is 599-ticker curated smart-beta list (no leverage, inverse, commodity, currency, vol products)."""


def narrate_with_claude(
    proposal: TradeProposal,
    deterministic_narration: str,
    user_question: Optional[str] = None,
    model: str = DEFAULT_ANTHROPIC_MODEL,
    max_tokens: int = 800,
) -> str:
    """Send the proposal + deterministic narration to Claude and return the response.

    Args:
        proposal: the ground-truth trade proposal.
        deterministic_narration: what `narrate_proposal(proposal)` produced;
            passed as reference so the LLM cannot drift from the numbers.
        user_question: optional operator question (Q&A mode). If None,
            the LLM produces an initial narration.
        model: Anthropic model ID. Default is claude-sonnet-4-5.
        max_tokens: response cap.

    Returns:
        Model response as plain text (may include markdown).

    Raises:
        RuntimeError if the SDK isn't installed or the key is missing.
    """
    if not anthropic_available():
        raise RuntimeError(
            "Anthropic SDK not available. Install `anthropic` and set "
            "ANTHROPIC_API_KEY in .env, or use narrate_proposal() only."
        )

    import anthropic
    client = anthropic.Anthropic()

    proposal_json = _proposal_to_dict(proposal)

    if user_question:
        prompt = (
            "The trader is looking at the following proposal:\n\n"
            "```json\n" + _json_dump(proposal_json) + "\n```\n\n"
            "Here is the deterministic plain-English summary:\n\n"
            "---\n" + deterministic_narration + "\n---\n\n"
            "The trader asks:\n\n"
            f"> {user_question}\n\n"
            "Answer their question precisely, using only facts from the "
            "proposal above. Do not invent numbers or propose alternatives."
        )
    else:
        prompt = (
            "Explain the following proposal to the operator. Add colour where "
            "the deterministic summary is thin — e.g. which factors are driving "
            "the picks, how the portfolio's tilt is shifting, and any items "
            "that deserve scrutiny (warnings, cash-reserve impact, or turnover "
            "spikes). Keep it under six short paragraphs.\n\n"
            "Proposal (JSON):\n"
            "```json\n" + _json_dump(proposal_json) + "\n```\n\n"
            "Deterministic summary (already shown to the operator; do not repeat verbatim):\n\n"
            "---\n" + deterministic_narration + "\n---"
        )

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in resp.content if hasattr(block, "text")]
    return "\n\n".join(parts).strip()


def _json_dump(payload: dict) -> str:
    """Compact-ish JSON for the LLM prompt — floats rounded to 4 decimals."""
    import json

    def _default(o):
        if isinstance(o, float):
            return round(o, 4)
        return str(o)

    return json.dumps(payload, indent=2, default=_default)
