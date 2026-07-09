#!/bin/bash
# Daily signal check — runs before NYSE open, emails combined alert
# Run via cron: 15 14 * * 1-5 (14:15 UTC = 9:15 AM ET in winter)
#
# NO IB Gateway required — reads cached data only.
#
# Log output goes to ~/trade_data/ETFTrader/logs/daily_signal_check.log

LOG_DIR="$HOME/trade_data/ETFTrader/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_signal_check.log"

echo "========================================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting daily signal check" >> "$LOG_FILE"

cd /home/stuar/code/ETFTrader

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Load environment variables (SMTP config etc.)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

python3 scripts/daily_signal_check.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "$(date '+%Y-%m-%d %H:%M:%S') — Finished (exit=$EXIT_CODE)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Keep log file manageable (last 5000 lines)
tail -5000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
