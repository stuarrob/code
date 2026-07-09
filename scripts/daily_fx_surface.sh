#!/bin/bash
# Daily FX option surface collection + SABR calibration
# Run via cron: 0 18 * * 1-5 /home/stuar/code/ETFTrader/scripts/daily_fx_surface.sh
#
# Polls for IB Gateway — will wait up to 2 hours if not yet logged in.
#
# Log output goes to ~/trade_data/ETFTrader/logs/daily_fx.log

LOG_DIR="$HOME/trade_data/ETFTrader/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_fx.log"

echo "========================================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting daily FX collection" >> "$LOG_FILE"

cd /home/stuar/code/ETFTrader

python3 scripts/daily_fx_surface.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "$(date '+%Y-%m-%d %H:%M:%S') — Finished (exit=$EXIT_CODE)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Keep log file manageable (last 5000 lines)
tail -5000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
