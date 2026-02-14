#!/usr/bin/env python3
"""Update the backtest_stop_loss_strategy.py to add trailing stop option."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
script_path = PROJECT_ROOT / "scripts" / "backtest_stop_loss_strategy.py"

# Read the file
with open(script_path, "r") as f:
    content = f.read()

# 1. Update the __init__ parameters to add trailing stop
old_init_params = """        # Stop-loss parameters
        use_entry_price_stop: bool = True,
        trailing_stop_threshold: float = 0.10,  # Activate at +10% gain
        trailing_stop_distance: float = 0.08,   # Trail by 8%"""

new_init_params = """        # Stop-loss parameters
        use_entry_price_stop: bool = True,
        use_trailing_stop: bool = False,        # NEW: Pure trailing stop from peak
        trailing_stop_pct: float = 0.10,        # NEW: Trail distance (e.g., 0.10 = -10% from peak)
        trailing_stop_threshold: float = 0.10,  # Activate at +10% gain (for original trailing)
        trailing_stop_distance: float = 0.08,   # Trail by 8% (for original trailing)"""

if old_init_params in content:
    content = content.replace(old_init_params, new_init_params)
    print("Updated __init__ parameters")
else:
    print("Could not find __init__ parameters to update")

# 2. Update the instance variable assignments
old_instance_vars = """        # Stop-loss parameters
        self.use_entry_price_stop = use_entry_price_stop
        self.entry_stop_loss_pct = 0.12  # -12% stop from entry price
        self.trailing_stop_threshold = trailing_stop_threshold
        self.trailing_stop_distance = trailing_stop_distance"""

new_instance_vars = """        # Stop-loss parameters
        self.use_entry_price_stop = use_entry_price_stop
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.entry_stop_loss_pct = 0.12  # -12% stop from entry price
        self.trailing_stop_threshold = trailing_stop_threshold
        self.trailing_stop_distance = trailing_stop_distance"""

if old_instance_vars in content:
    content = content.replace(old_instance_vars, new_instance_vars)
    print("Updated instance variables")
else:
    print("Could not find instance variables to update")

# 3. Update the logger.info in __init__
old_logger = """        logger.info(f"  Entry stop-loss: {use_entry_price_stop} (-{self.entry_stop_loss_pct:.0%} from entry)")
        logger.info(f"  Trailing stop: {trailing_stop_threshold:.0%} gain, {trailing_stop_distance:.0%} trail")"""

new_logger = """        logger.info(f"  Entry stop-loss: {use_entry_price_stop} (-{self.entry_stop_loss_pct:.0%} from entry)")
        logger.info(f"  Pure trailing stop: {use_trailing_stop} (-{trailing_stop_pct:.0%} from peak)")
        logger.info(f"  Gain-triggered trailing: {trailing_stop_threshold:.0%} gain, {trailing_stop_distance:.0%} trail")"""

if old_logger in content:
    content = content.replace(old_logger, new_logger)
    print("Updated logger info")
else:
    print("Could not find logger info to update")

# 4. Update the _check_stop_losses method to add pure trailing stop
old_check = """            # Check entry price stop-loss (-12% from entry)
            loss_pct = position.current_gain(current_price)
            if self.use_entry_price_stop and loss_pct < -self.entry_stop_loss_pct:"""

new_check = """            # Check pure trailing stop (from peak, no gain threshold required)
            if self.use_trailing_stop:
                drawdown_from_peak = position.drawdown_from_peak(current_price)
                if drawdown_from_peak < -self.trailing_stop_pct:
                    to_sell.append((ticker, f"Trail stop (-{self.trailing_stop_pct:.0%} from peak)", current_price))
                    logger.info(f"  📉 TRAIL STOP: {ticker} at ${current_price:.2f} (peak: ${position.peak_price:.2f}, dd: {drawdown_from_peak:.1%})")
                    continue

            # Check entry price stop-loss (-12% from entry)
            loss_pct = position.current_gain(current_price)
            if self.use_entry_price_stop and loss_pct < -self.entry_stop_loss_pct:"""

if old_check in content:
    content = content.replace(old_check, new_check)
    print("Updated _check_stop_losses method")
else:
    print("Could not find _check_stop_losses to update")

# 5. Update argparse to add new options
old_argparse = """    # Stop-loss parameters
    parser.add_argument("--no-entry-stop", action="store_true", help="Disable entry price stop")
    parser.add_argument("--trailing-threshold", type=float, default=0.10, help="Trailing stop threshold")
    parser.add_argument("--trailing-distance", type=float, default=0.08, help="Trailing stop distance")"""

new_argparse = """    # Stop-loss parameters
    parser.add_argument("--no-entry-stop", action="store_true", help="Disable entry price stop")
    parser.add_argument("--trailing-stop", action="store_true", help="Enable pure trailing stop from peak")
    parser.add_argument("--trailing-pct", type=float, default=0.10, help="Trailing stop percentage from peak (e.g., 0.10 = -10%)")
    parser.add_argument("--trailing-threshold", type=float, default=0.10, help="Gain-triggered trailing stop threshold")
    parser.add_argument("--trailing-distance", type=float, default=0.08, help="Gain-triggered trailing stop distance")"""

if old_argparse in content:
    content = content.replace(old_argparse, new_argparse)
    print("Updated argparse")
else:
    print("Could not find argparse to update")

# 6. Update the StopLossBacktest instantiation to pass new params
old_instantiation = """        use_entry_price_stop=not args.no_entry_stop,
        trailing_stop_threshold=args.trailing_threshold,
        trailing_stop_distance=args.trailing_distance,"""

new_instantiation = """        use_entry_price_stop=not args.no_entry_stop,
        use_trailing_stop=args.trailing_stop,
        trailing_stop_pct=args.trailing_pct,
        trailing_stop_threshold=args.trailing_threshold,
        trailing_stop_distance=args.trailing_distance,"""

if old_instantiation in content:
    content = content.replace(old_instantiation, new_instantiation)
    print("Updated StopLossBacktest instantiation")
else:
    print("Could not find instantiation to update")

# Write the updated content
with open(script_path, "w") as f:
    f.write(content)

print("\nScript updated successfully!")
print("\nNew usage options:")
print("  --trailing-stop          Enable pure trailing stop from peak")
print("  --trailing-pct 0.10      Set trailing stop to -10% from peak")
print("\nExamples:")
print("  # Current strategy (entry stop only):")
print("  python scripts/backtest_stop_loss_strategy.py --start 2025-01-01")
print()
print("  # Pure trailing stop (-10% from peak):")
print("  python scripts/backtest_stop_loss_strategy.py --start 2025-01-01 --no-entry-stop --trailing-stop --trailing-pct 0.10")
print()
print("  # Hybrid (entry stop + trailing stop):")
print("  python scripts/backtest_stop_loss_strategy.py --start 2025-01-01 --trailing-stop --trailing-pct 0.08")
