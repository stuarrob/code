@echo off
REM Weekly FMP refresh — fundamentals + SPY + VIX + backfill top-up.
REM
REM Complements the nightly IB collector (daily_etf_data_windows.cmd).
REM Runs via Windows Task Scheduler; see docs/WINDOWS_AUTOMATION.md for
REM the schtasks command that installs it.
REM
REM Chain:
REM   1. Refresh ETF fundamentals cache (yield + expense_ratio + AUM).
REM   2. Refresh SPY + VIX daily-close series (small, fast).
REM   3. Top-up any tickers where FMP has newer data than local cache.
REM   4. Send a Telegram summary (start + end) via telegram_send.py.
REM
REM Failure in any step logs to daily_etf.log and is reported in the
REM end-of-job Telegram summary; a step failure does NOT abort the
REM subsequent steps — a partial refresh is better than nothing.

setlocal
set PYTHON="C:\Users\stuar\AppData\Roaming\miniconda3\envs\etftrader\python.exe"
set REPO=C:\Users\stuar\code\ETFTrader
set LOGDIR=C:\Users\stuar\trade_data\ETFTrader\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOG=%LOGDIR%\weekly_fmp_refresh.log

echo ======================================== >> "%LOG%"
echo %DATE% %TIME% - Starting weekly FMP refresh >> "%LOG%"
cd /d "%REPO%"

REM Start notification.
%PYTHON% scripts\telegram_send.py --subject "ETFTrader weekly FMP refresh started" --body "Fundamentals + SPY + VIX + price top-up. Will report when done." >> "%LOG%" 2>&1

REM Step 1: fundamentals.
echo %DATE% %TIME% - Step 1: refresh_fundamentals >> "%LOG%"
%PYTHON% scripts\refresh_fundamentals.py --full --source union --workers 5 >> "%LOG%" 2>&1
set RC1=%ERRORLEVEL%

REM Step 2: SPY + VIX (small; if it fails we log and continue).
echo %DATE% %TIME% - Step 2: SPY + VIX top-up >> "%LOG%"
%PYTHON% scripts\refresh_market_indices.py >> "%LOG%" 2>&1
set RC2=%ERRORLEVEL%

REM Step 3: FMP backfill (skips tickers whose cache is already deep enough,
REM so on weekly runs it will only touch the recently-added universe entries
REM or fix earlier failures — fast in the steady state).
echo %DATE% %TIME% - Step 3: FMP price top-up >> "%LOG%"
%PYTHON% scripts\backfill_prices_fmp.py --workers 3 --delay 0.5 >> "%LOG%" 2>&1
set RC3=%ERRORLEVEL%

echo %DATE% %TIME% - Finished (fund=%RC1% spyvix=%RC2% backfill=%RC3%) >> "%LOG%"

REM End notification — inline a small body summarising each step's exit code.
set BODY=Fundamentals exit: %RC1%. SPY+VIX exit: %RC2%. FMP top-up exit: %RC3%. Log: %LOG%.
%PYTHON% scripts\telegram_send.py --subject "ETFTrader weekly FMP refresh finished" --body "%BODY%" >> "%LOG%" 2>&1

echo. >> "%LOG%"
REM Return the worst exit code so Task Scheduler shows the correct status.
if %RC1% neq 0 exit /b %RC1%
if %RC2% neq 0 exit /b %RC2%
exit /b %RC3%
