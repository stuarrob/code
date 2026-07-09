@echo off
REM Windows wrapper for daily_etf_data.py (see docs/WINDOWS_AUTOMATION.md).
REM Runs via Task Scheduler after IBC has brought IB Gateway up.
setlocal
set PYTHON="C:\Users\stuar\AppData\Roaming\miniconda3\envs\etftrader\python.exe"
set REPO=C:\Users\stuar\code\ETFTrader
set LOGDIR=C:\Users\stuar\trade_data\ETFTrader\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOG=%LOGDIR%\daily_etf.log

echo ======================================== >> "%LOG%"
echo %DATE% %TIME% - Starting daily ETF collection >> "%LOG%"
cd /d "%REPO%"
%PYTHON% scripts\daily_etf_data.py >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo %DATE% %TIME% - Finished (exit=%RC%) >> "%LOG%"
echo. >> "%LOG%"
exit /b %RC%
