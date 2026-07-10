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


DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"  # highest quality; ~5x Sonnet cost, trivial on live NAV


# ────────────────────────────────────────────────────────────────
# Deterministic narrator (always works)
# ────────────────────────────────────────────────────────────────

def narrate_proposal(proposal: TradeProposal,
                     policy_name: Optional[str] = None) -> str:
    """Compose a plain-English summary of the proposal, deterministically.

    This is what a human reader sees on Step 5 when the LLM is not
    configured or has been disabled. It is also the exact same text the
    LLM receives as ground truth in its prompt.
    """
    lines: list[str] = []
    n_trades = len(proposal.trades)

    if n_trades == 0:
        lines.append(
            "**No trades proposed.** The current portfolio is inside the "
            "drift threshold and matches the target basket. There is "
            "nothing to do this cycle."
        )
        if proposal.warnings:
            lines.append("")
            lines.append("**Warnings:**")
            for w in proposal.warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)

    # Buckets
    buys = [t for t in proposal.trades if t.action == ACTION_BUY]
    sells = [t for t in proposal.trades if t.action == ACTION_SELL]
    extends = [t for t in proposal.trades if t.action == ACTION_EXTEND]

    lines.append(
        f"**Headline.** {n_trades} trade{'s' if n_trades != 1 else ''} "
        f"proposed — "
        f"**{len(buys)} buy{'s' if len(buys) != 1 else ''}**, "
        f"**{len(sells)} sell{'s' if len(sells) != 1 else ''}**, "
        f"**{len(extends)} extend{'s' if len(extends) != 1 else ''}**. "
        f"Total turnover **${proposal.turnover_notional:,.0f}** "
        f"({proposal.turnover_pct_of_nav:.1%} of NAV). "
        f"Estimated cost **${proposal.total_est_cost:,.0f}**. "
        f"Portfolio ends with **{proposal.n_positions_after} positions**."
    )

    # Sells — clean out first
    if sells:
        lines.append("")
        lines.append("**Sells** (reduce or exit; existing TRAILs on held positions are not touched by this proposal):")
        for t in sorted(sells, key=lambda x: -x.delta_notional):
            reason = "exit position" if t.target_shares == 0 else "reduce to target weight"
            lines.append(
                f"- **{t.ticker}** — sell {abs(t.delta_shares):,d} shares "
                f"@ ${t.market_price:.2f} = ${t.delta_notional:,.0f}. "
                f"{reason.capitalize()} ({t.current_weight_pct:.1%} → {t.target_weight_pct:.1%})."
            )

    # Extends — add to existing
    if extends:
        lines.append("")
        lines.append("**Extends** (add to existing positions; a new trailing stop covering only the new shares will be attached):")
        for t in sorted(extends, key=lambda x: -x.delta_notional):
            lines.append(
                f"- **{t.ticker}** — buy {t.delta_shares:,d} more shares "
                f"@ ${t.market_price:.2f} = ${t.delta_notional:,.0f}. "
                f"Weight moves from {t.current_weight_pct:.1%} → {t.target_weight_pct:.1%} "
                f"(gap {t.weight_gap_pct:+.1%})."
            )

    # Buys — fresh positions
    if buys:
        lines.append("")
        lines.append("**Buys** (fresh positions; each gets a full trailing stop):")
        for t in sorted(buys, key=lambda x: -x.delta_notional):
            lines.append(
                f"- **{t.ticker}** — buy {t.delta_shares:,d} shares "
                f"@ ${t.market_price:.2f} = ${t.delta_notional:,.0f}. "
                f"New position at {t.target_weight_pct:.1%} of NAV."
            )

    # Factor tilt narrative
    if proposal.factor_exposures:
        tilts: list[str] = []
        for fe in proposal.factor_exposures:
            if fe.before != fe.before or fe.after != fe.after:  # NaN guards
                continue
            if abs(fe.delta) < 0.02:
                continue
            direction = "increases" if fe.delta > 0 else "decreases"
            tilts.append(
                f"**{fe.factor}** {direction} ({fe.before:+.2f} → {fe.after:+.2f}, "
                f"Δ {fe.delta:+.2f})"
            )
        if tilts:
            lines.append("")
            lines.append("**Factor tilt (before → after).** " + "; ".join(tilts) + ".")

    # Cash + warnings
    lines.append("")
    lines.append(
        f"**Cash impact.** After settlement the account should hold "
        f"about **${proposal.cash_after:,.0f}** in cash "
        f"(policy reserve is included in the target math)."
    )

    if proposal.warnings:
        lines.append("")
        lines.append(f"**{len(proposal.warnings)} warning(s) — review before Applying:**")
        for w in proposal.warnings:
            lines.append(f"- {w}")

    if policy_name:
        lines.append("")
        lines.append(f"*Policy: `{policy_name}`. This narration is deterministic; "
                     f"nothing here proposes a trade the code did not already decide.*")

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


_SYSTEM_PROMPT = """You are the narrator for a deterministic ETF smart-beta trading applet.

STRICT RULES — violating any of these breaks the app's safety contract:

1. You never propose trades. The trades in the TradeProposal are decided by
   a deterministic pipeline (`src/portfolio/proposal.py`) and are the sole
   source of truth. Your job is to explain WHY they make sense given the
   strategy's factor tilts and the current portfolio state.

2. You never invent numbers. If a number is not in the TradeProposal or the
   deterministic-narration text you receive, do not include it. Quote the
   proposal verbatim when precision matters.

3. If asked "what if" questions (e.g. "what if I increased my cash budget?"),
   answer conceptually — do NOT compute new trades. Refer the operator back
   to Steps 1-4 with the new inputs.

4. Keep the language clean and precise. Retail-quality "market wisdom" is
   worse than none. The reader is a serious operator with real money on the
   line.

5. When you receive a question about a specific trade, use the ticker as
   the primary key. State the action (BUY / SELL / EXTEND), the size, and
   ONE reason (usually a factor tilt or a target-weight gap).

6. The strategy is 35% momentum, 30% quality, 20% low-volatility, 15% value.
   All positions are chosen by weighted-geometric-mean rank on those factors.
   Position sizing uses exponential rank weights, bounded 2%-15%. Trailing
   stops are 10%. This is background you may reference; do not restate the
   whole strategy every response."""


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
