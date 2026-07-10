# Windows automation — nightly ETF data collection

Goal: bring back the nightly `daily_etf_data.py` job that the WSL cron used to
run, adapted for native Windows. The chain is:

```
Windows Task Scheduler   (nightly trigger, 01:30)
        │
        ▼
IBC (Interactive Brokers Controller)
    starts IB Gateway
    auto-fills username / password
    dismisses the daily disclaimer
    handles the second-factor prompt
        │
        ▼
python scripts/daily_etf_data.py
    connects on 127.0.0.1:4001, client_id 22
    incrementally updates stale/missing tickers
    writes ~/trade_data/ETFTrader/processed/etf_prices_ib.parquet
        │
        ▼
(optional) task stops IB Gateway to release credentials
```

Once this is in place, tomorrow's applet run will find the cache current in seconds instead of proposing an 18-hour catch-up.

## Prerequisites — confirmed already present on this machine

| Component | Location |
|---|---|
| IB Gateway | `C:\Jts\ibgateway\1045` |
| Python env | `%APPDATA%\miniconda3\envs\etftrader` |
| Repo | `C:\Users\stuar\code\ETFTrader` |
| Data root | `C:\Users\stuar\trade_data\ETFTrader` |
| IB API port | `4001` (live) |
| Client ID for daily job | `22` |

Not yet installed:

| Component | Where it will live |
|---|---|
| IBC | `C:\IBC` (recommended — outside Program Files, no admin rights needed) |
| IBC config file | `C:\IBC\config.ini` (contains credentials — see security note below) |
| Wrapper batch file | `C:\Users\stuar\code\ETFTrader\scripts\daily_etf_data_windows.cmd` |
| Task Scheduler entry | `\ETFTrader\Daily ETF Data Collection` |

## Step 1 — Install IBC

1. Download the latest release for Windows from
   <https://github.com/IbcAlpha/IBC/releases> (file named
   `IBCWin-<version>.zip`).
2. Create `C:\IBC` and extract the archive contents into it. You should end
   up with `C:\IBC\IBC.jar`, `C:\IBC\StartGateway.bat`, `C:\IBC\config.ini`,
   and a small set of supporting files.
3. Verify Java can launch it. IB Gateway ships its own JRE — IBC can use it.
   Test from a PowerShell prompt:
   ```powershell
   & "C:\Jts\ibgateway\1045\jre\bin\java.exe" -version
   ```
   You should see a Java version banner.

## Step 2 — Configure IBC

Open `C:\IBC\config.ini` in a text editor. The template ships with hundreds
of lines of commented options; the ones that actually matter for our use are
below. Uncomment / edit these; leave the rest at defaults.

```ini
# --- Identity ---
IbLoginId=YOUR_IB_USERNAME
IbPassword=YOUR_IB_PASSWORD
# For Live: TradingMode=live ; for Paper: TradingMode=paper
TradingMode=live

# --- Where IB Gateway lives ---
IbDir=C:\Jts\ibgateway\1045
IbSecurityDeviceInteraction=false

# --- API + startup ---
OverrideTwsApiPort=4001
AcceptIncomingConnectionAction=accept
AcceptNonBrokerageAccountWarning=yes
DismissPasswordExpiryWarning=yes
DismissNSEComplianceNotice=yes
ReadOnlyApi=yes
# ADR-0001 requires Read-Only API by default: nightly cron + all
# read-only applet steps work fine. To place a trade you must flip
# ReadOnlyApi off in Gateway's UI first — that's the intended,
# conscious gate for order execution.

# --- Auto-restart daily so IB's forced logout doesn't kill us ---
AutoRestartTime=23:45
# IB force-disconnects around midnight — the AutoRestartTime nudges IBC
# to relaunch Gateway cleanly before that.

# --- 2FA handling ---
# If you use IBKR Mobile 2FA: leave the below at the defaults, IBC will
# handle the "Authenticated?" polling. If you use another method, see
# https://github.com/IbcAlpha/IBC/blob/master/userguide.md
SecondFactorDevice=
```

**Security note on credentials.** `config.ini` will contain your live IB
password in plaintext. Options in ascending robustness:

1. **Leave it in `config.ini`, protect the file.** Right-click `C:\IBC` →
   Properties → Security → remove permissions for everyone except your own
   Windows user + SYSTEM. This is what most IBC users do.
2. **Use Windows Credential Manager + a small wrapper.** IBC does not read
   Credential Manager directly, but a PowerShell wrapper can fetch the
   password and write a temporary `config.ini` before launch. If you want
   this, tell me and I'll draft it.
3. **Change your IB password to a dedicated one used only by IBC** (still
   plaintext, but limits blast radius).

## Step 3 — Wrapper batch file

Windows Task Scheduler runs `.cmd`/`.bat` files well; running Python
scripts directly is fussy about paths. Create the wrapper below —
`scripts\daily_etf_data_windows.cmd` — which the scheduler will invoke.

Contents (already drafted at `scripts/daily_etf_data_windows.cmd` when I land
the code changes in this branch):

```cmd
@echo off
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
```

## Step 4 — Task Scheduler entry

Open **Task Scheduler** (search for it in the Start menu). Create Task
(*not* "Create Basic Task"). Then:

**General tab**
- Name: `Daily ETF Data Collection`
- Location: `\ETFTrader\` (create the folder in the tree)
- Run whether user is logged on or not (this asks for your Windows password)
- Run with highest privileges: unchecked (not needed)
- Configure for: Windows 10 / 11

**Triggers tab** — click *New*
- Begin the task: On a schedule
- Weekly, Mon-Fri
- Start: today's date, `01:30:00` (well after IB's midnight force-logout)
- Enabled: yes

**Actions tab** — two actions, in order:

Action 1 — start IBC (which starts Gateway)
- Action: Start a program
- Program/script: `C:\Windows\System32\cmd.exe`
- Arguments: `/c start "" "C:\IBC\StartGateway.bat"`
- Start in: `C:\IBC`

Action 2 — wait 3 minutes then run the collector
- Action: Start a program
- Program/script: `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`
- Arguments:
  ```
  -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 180; & 'C:\Users\stuar\code\ETFTrader\scripts\daily_etf_data_windows.cmd'"
  ```
- Start in: `C:\Users\stuar\code\ETFTrader`

**Conditions tab**
- Uncheck "Start the task only if the computer is on AC power" (leave it
  enabled on battery too)
- Wake the computer to run this task: **check this** — otherwise the box
  will sleep through the trigger

**Settings tab**
- Allow task to be run on demand: yes
- Stop the task if it runs longer than: 20 hours (safety net)

## Step 5 — Verify

Once installed, right-click the task in Task Scheduler and pick **Run**.
Then, in a separate PowerShell:

```powershell
# Is IB Gateway up?
Get-Process ibgateway -ErrorAction SilentlyContinue

# Is the Python collector running?
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CPU -gt 5 }

# Watch the log
Get-Content "$env:USERPROFILE\trade_data\ETFTrader\logs\daily_etf.log" -Tail 20 -Wait
```

Tomorrow morning after the trigger fires:

```powershell
Get-ChildItem "$env:USERPROFILE\trade_data\ETFTrader\ib_historical" -Filter "*.parquet" `
    | Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-8) } | Measure-Object
```
The count should be roughly the number of stale + missing tickers as of
last night — anywhere from a handful (steady state) to thousands (catch-up
mode after downtime).

## Weekly FMP refresh (fundamentals + SPY + VIX + price top-up)

Runs alongside the nightly IB collector. Uses Financial Modeling Prep
(Premium tier — verified 2026-07-10) for:

- ETF fundamentals (yield + expense ratio + AUM, ~5000 tickers)
- SPY + VIX daily-close series (for the regime overlay)
- Price top-up for any ETFs where FMP has newer data than the IB cache

Wrapper: [scripts/weekly_fmp_refresh_windows.cmd](../scripts/weekly_fmp_refresh_windows.cmd).
Sends a Telegram start notice and an end summary via
[scripts/telegram_send.py](../scripts/telegram_send.py) (reads
`TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID` from `.env`).

Runtime: ~15–20 minutes steady-state (mostly fundamentals refresh + FMP
top-up for any newly-added universe members). First-run backfill (2010
onwards) is a one-off ~15 minute job — do that manually before
scheduling the weekly job.

### Installing the weekly schedule (one PowerShell command)

Runs Saturday 20:00 local (feel free to change). No admin required —
uses user-scope Task Scheduler. Replace `stuar` with the actual username
if this doc is being reused.

```powershell
schtasks /Create `
  /TN "ETFTrader\Weekly FMP Refresh" `
  /TR "C:\Users\stuar\code\ETFTrader\scripts\weekly_fmp_refresh_windows.cmd" `
  /SC WEEKLY /D SAT /ST 20:00 `
  /RL LIMITED /F
```

Verify installation:

```powershell
schtasks /Query /TN "ETFTrader\Weekly FMP Refresh" /V /FO LIST
```

Test-run once manually to confirm the Telegram notifications land and
the log gets written:

```powershell
& "C:\Users\stuar\code\ETFTrader\scripts\weekly_fmp_refresh_windows.cmd"
Get-Content "C:\Users\stuar\trade_data\ETFTrader\logs\weekly_fmp_refresh.log" -Tail 40
```

## Troubleshooting

**Gateway launches but IBC never enters credentials.** Usually a 2FA
configuration mismatch. Check the IBC log at `C:\IBC\ibc.log` (created on
first run). The IBC user guide's 2FA section is the authoritative reference.

**API disconnects mid-run.** IB force-disconnects the API after ~20 hours
of continuous use. The `AutoRestartTime=23:45` line in `config.ini` handles
this, but if you see it happening earlier, shorten the `AutoRestartTime`
window.

**Task Scheduler says "Task completed successfully" but no data was
collected.** `Start a program` returns success if the program was
*launched*, not if it succeeded. Always check `daily_etf.log` for the
`Finished (exit=…)` line — non-zero exit codes mean the collector actually
failed.

**Python process is stuck on qualification for hours.** See the
`ib_data_collector.py` qualify batching optimisation — worth landing before
the next big catch-up so a healthy nightly delta finishes in minutes rather
than an hour.
