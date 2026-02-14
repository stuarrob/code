# Stop-Loss Strategy Design

**Based on:** Academic Research (Han, Zhou, Zhu 2014) + ETF Momentum Literature

---

## Core Principle

**"Let Winners Run, Cut Losers Quickly"**

- Select ETFs based on multi-factor scores (momentum, quality, value, volatility)
- Hold positions as long as they're profitable (above purchase price)
- Sell immediately when they drop below purchase price (or trigger stop-loss)
- Weekly check, bi-weekly factor recalculation

---

## Stop-Loss Levels (Research-Based)

### 1. **Entry Price Protection** (Your Suggestion)
- **Rule**: Sell if position drops below purchase price
- **Check**: Weekly
- **Rationale**: Preserve capital, avoid holding losers
- **Academic Support**: Han et al. showed 15% stop-loss eliminated momentum crashes

### 2. **Trailing Stop for Winners**
- **Rule**: For positions with >10% gain, set trailing stop at -8% from peak
- **Purpose**: Lock in profits while allowing upside
- **Academic Support**: 20% trailing stops outperformed fixed stops by 27% over 11 years

### 3. **Time-Based Review**
- **Rule**: Every 2 weeks, recalculate factor scores
- **Purpose**: Identify positions that no longer rank highly
- **Action**: If position is profitable but no longer in top 30 by factors → hold but flag
- **Action**: If flagged position drops below peak by 5% → sell

### 4. **Maximum Position Age**
- **Rule**: If position held >90 days and not in top 20 factors → consider exit
- **Purpose**: Prevent "zombie" positions from occupying slots
- **Exception**: Keep if still above purchase price + 15%

---

## Algorithm Design

### Weekly Workflow

```python
FOR each position in portfolio:
    current_price = get_current_price(position)
    entry_price = position.entry_price
    peak_price = position.peak_price
    current_gain = (current_price - entry_price) / entry_price

    # Update peak price
    if current_price > peak_price:
        position.peak_price = current_price

    # Rule 1: Entry Price Protection
    if current_price < entry_price:
        SELL(position, reason="Below entry price")
        continue

    # Rule 2: Trailing Stop for Winners
    if current_gain > 0.10:  # Position has >10% gain
        drawdown_from_peak = (current_price - peak_price) / peak_price
        if drawdown_from_peak < -0.08:  # -8% from peak
            SELL(position, reason="Trailing stop hit")
            continue

    # Position survives - keep holding
    HOLD(position)
```

### Bi-Weekly Factor Rebalancing

```python
# Every 2 weeks (not weekly):
1. Calculate fresh factor scores for all ETFs
2. Rank all ETFs by integrated factor score
3. Identify target portfolio (top 20 ETFs)

FOR each current position:
    if position in target_portfolio_top_30:
        KEEP(position)  # Still highly ranked
    else:
        position.flagged = True
        # Don't sell yet - wait for price action or next cycle

FOR each empty slot (due to stop-losses):
    candidate = highest_ranked_ETF_not_in_portfolio()
    if candidate.factor_score > threshold:
        BUY(candidate)
```

---

## Key Differences from Original Strategy

| Aspect | Original | New Stop-Loss Strategy |
|--------|----------|----------------------|
| **Rebalancing** | Weekly, force to target portfolio | Bi-weekly factors, sell only on stops |
| **Exit Rule** | Drift-based, sell to match target weights | Price-based, sell only if losing money |
| **Turnover** | ~75% weekly | Expected ~15-25% (only losers exit) |
| **Position Duration** | 1-4 weeks average | Unlimited for winners, fast exit for losers |
| **Core Premise** | Optimizer knows best | Momentum + capital preservation |

---

## Expected Outcomes (Based on Literature)

### Performance Improvements
- **Avoid Momentum Crashes**: Han et al. showed -49% crash became +2% gain with stops
- **Higher Sharpe**: Reduced tail risk while maintaining upside
- **Lower Turnover**: Only trade when necessary (stop-loss or new opportunity)

### Risk Reduction
- **Maximum Single Position Loss**: Capped at ~0-2% (entry price protection)
- **Portfolio Drawdown**: Expected to reduce from -5.7% to -3-4%
- **Tail Risk**: Eliminate large negative months

### Trade-offs
- **May Exit Winners Early**: If momentum fades temporarily
- **Underperformance in Strong Trends**: Stop-losses may trigger on volatility
- **Complexity**: More state to track (entry prices, peaks, flags)

---

## Implementation Checklist

### Data Required per Position
- [x] Ticker
- [x] Entry Price (purchase price)
- [x] Entry Date
- [x] Shares
- [x] Peak Price (highest since entry)
- [x] Peak Date
- [x] Flagged (boolean - factor rank dropped)
- [x] Days Held

### Logic to Implement
1. [x] Entry price tracking on buy
2. [x] Peak price updating (check every day or week)
3. [x] Stop-loss checking (weekly)
4. [x] Factor recalculation (bi-weekly)
5. [x] Replacement logic (fill slots from stop-losses)

---

## Backtest Modifications

### Changes to `backtest_paper_trading.py`

1. **Position Tracking Enhancement**:
   ```python
   class Position:
       ticker: str
       shares: float
       entry_price: float
       entry_date: pd.Timestamp
       peak_price: float
       peak_date: pd.Timestamp
       flagged: bool = False
       days_held: int = 0
   ```

2. **Weekly Check** (every rebalance date):
   - Update peak prices
   - Check stop-losses
   - Execute sells if triggered

3. **Bi-Weekly Check** (every 2 weeks):
   - Recalculate factors
   - Identify new buy candidates
   - Fill empty slots

4. **No Forced Rebalancing**:
   - Remove drift threshold logic
   - Keep positions indefinitely if above entry price
   - Only exit on stop-loss trigger

---

## Parameter Recommendations

### Conservative (Recommended Start)
```python
ENTRY_PRICE_STOP = True              # Sell if below purchase price
TRAILING_STOP_THRESHOLD = 0.10       # Activate trailing stop at +10% gain
TRAILING_STOP_DISTANCE = 0.08        # Trail by 8% from peak
FACTOR_RECALC_FREQ = "bi-weekly"     # Every 2 weeks
REBALANCE_CHECK_FREQ = "weekly"      # Check stops weekly
MAX_POSITION_AGE_DAYS = 90           # Review old positions
MIN_FACTOR_SCORE_TO_BUY = 0.5        # Only buy strong candidates
```

### Aggressive (More Turnover)
```python
ENTRY_PRICE_STOP = True
TRAILING_STOP_THRESHOLD = 0.05       # Activate at +5%
TRAILING_STOP_DISTANCE = 0.10        # Wider trail (10%)
FACTOR_RECALC_FREQ = "weekly"
```

---

## Success Metrics

### Target Performance (vs Original)
- **CAGR**: Maintain ~15-17% (original: implied 17% from backtest)
- **Sharpe**: Improve from 1.29 → 1.5+ (better tail risk)
- **Max Drawdown**: Reduce from -5.7% → -3 to -4%
- **Turnover**: Reduce from 75% → 15-25% per period
- **Win Rate**: Target >60% of closed positions profitable

### Key Questions to Answer
1. Does entry price protection reduce crashes?
2. Do winners run long enough to generate alpha?
3. Is bi-weekly factor recalculation sufficient?
4. What's the actual turnover in practice?

---

**Next Step**: Implement in backtest and run full historical test (March-October 2025)
