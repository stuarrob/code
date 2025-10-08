# Instructions for Claude - ETF Portfolio Optimization System

**Last Updated:** 2025-10-04

---

## Purpose of This Document

This document provides clear instructions for continuing work on the ETF Portfolio Optimization System. Read this file at the start of each session to understand:
1. Where we are in the project
2. What task to work on next
3. How to update the plan after completing tasks

---

## Quick Status Check

**Current Phase:** Phase 1 - Data Infrastructure
**Current Task:** Task 1.1 - ETF Data Collection Module (NOT STARTED)
**Overall Progress:** 5% (Planning complete)

---

## How to Use the Plan Documents

### 1. PROJECT_PLAN.md
- **Master plan** with all phases, tasks, and deliverables
- Read the "Current Phase" section to see what we're working on
- Each task has checkboxes `[ ]` - mark as `[x]` when complete
- Update status from 🔲 to 🔄 (in progress) to ✅ (done)

### 2. PROGRESS_LOG.md
- **Daily progress tracking** - update this after each work session
- Add entry for each day worked
- Track completed tasks, blockers, decisions made
- Update metrics dashboard

### 3. INSTRUCTIONS.md (this file)
- **Operating instructions** for continuing the project
- Read this first each session

---

## Workflow for Each Session

### Step 1: Read Current Status
```bash
# Navigate to project
cd /home/stuar/code/ETFTrader/Plan

# Read PROJECT_PLAN.md to find current task
# Look for the task marked "Status: NOT STARTED" or "IN PROGRESS"
```

### Step 2: Work on Current Task
- Follow the subtask checklist in PROJECT_PLAN.md
- Create deliverables as specified
- Write tests where indicated
- Document decisions in PROGRESS_LOG.md

### Step 3: Validate Code Quality (MANDATORY)

**Before delivering any code or notebooks, ALWAYS run:**

1. **Black Code Formatting**
   ```bash
   source venv/bin/activate
   black src/ scripts/ tests/
   ```

2. **Static Code Analysis (Pylint)**
   ```bash
   source venv/bin/activate
   pylint src/ scripts/ tests/
   # Target: 8.0/10 minimum score
   ```

3. **Notebook Validation** (if notebooks created/modified)
   ```bash
   source venv/bin/activate
   python scripts/validate_notebook.py notebooks/<notebook_name>.ipynb
   ```

**All validations must pass before moving to Step 4.**

### Step 4: Update Plan After Completing Work
1. **Update PROJECT_PLAN.md:**
   - Mark completed subtasks with `[x]`
   - Update progress percentages
   - Change status (🔲 → 🔄 → ✅)
   - Update "Last Updated" date at top
   - Add validation results to deliverables section

2. **Update PROGRESS_LOG.md:**
   - Add daily entry with date, time spent, completed items
   - Note any blockers or decisions
   - Update metrics dashboard
   - Update phase completion tracking
   - Document validation results (Black/Pylint/Notebook scores)

### Step 5: Communicate Status
- Summarize what was completed
- Note any issues or questions
- State next task to work on

---

## Current Task Details

### Phase 1, Task 1.1: ETF Data Collection Module

**Goal:** Build scripts to collect ETF data from free sources and save to CSV files

**Duration:** 5-6 days
**Status:** NOT STARTED

**Subtasks (in order):**
1. [ ] Install required libraries
   ```bash
   pip install yfinance pandas numpy requests beautifulsoup4
   ```

2. [ ] Create project structure:
   ```bash
   cd /home/stuar/code/ETFTrader
   mkdir -p data/raw/prices data/processed data/indicators data/signals
   mkdir -p src/data_collection src/data_management
   mkdir -p tests notebooks config results logs docs
   ```

3. [ ] Create ETF universe scraper (`src/data_collection/etf_scraper.py`)
   - Scrape ETF list from free source (ETF Database, Yahoo Finance screener)
   - Filter out leveraged ETFs (names containing "2x", "3x", "Ultra", "Inverse")
   - Save to `data/raw/etf_universe.csv` with columns:
     - ticker, name, category, expense_ratio, aum, inception_date

4. [ ] Build price data downloader (`src/data_collection/price_downloader.py`)
   - Download OHLCV data using yfinance
   - Download 3+ years of daily data
   - Save each ETF to separate CSV: `data/raw/prices/{TICKER}.csv`
   - Columns: date, open, high, low, close, adj_close, volume

5. [ ] Add fundamental data collector (extend etf_scraper.py)
   - Collect expense ratios, AUM, sectors
   - Save to `data/raw/fundamentals.csv`

6. [ ] Create data validator (`src/data_collection/data_validator.py`)
   - Check for missing dates
   - Identify data quality issues
   - Generate validation report
   - Save report to `results/data_validation_report.csv`

**Deliverables:**
- `src/data_collection/etf_scraper.py`
- `src/data_collection/price_downloader.py`
- `src/data_collection/data_validator.py`
- `data/raw/etf_universe.csv` (target: 100-200 ETFs for pilot)
- `data/raw/prices/*.csv` (one per ETF)
- `data/raw/fundamentals.csv`
- `results/data_validation_report.csv`

**Success Criteria:**
- Minimum 100 ETFs collected
- Each ETF has at least 2 years of daily price data
- Less than 5% missing data across universe
- Data validation report shows no critical issues

**Testing:**
- Create `tests/test_data_collection.py` with unit tests
- Create `notebooks/01_data_validation.ipynb` to visualize data quality

---

## Important Principles

### 1. Keep Plans Updated
- **After every task:** Update PROJECT_PLAN.md checkboxes
- **After every session:** Update PROGRESS_LOG.md with daily entry
- **Before ending session:** Mark current status clearly

### 2. Follow the Deliverables List
- Each task lists specific files to create
- Don't skip deliverables
- Create them in the order specified

### 3. Test as You Go
- Write tests for each module
- Create validation notebooks
- Don't move to next phase until tests pass

### 4. Document Decisions
- Record any changes to approach in PROGRESS_LOG.md
- Note why you chose a particular library or method
- Track issues and how they were resolved

### 5. Ask Before Deviating
- If the plan doesn't make sense, ask the user
- If you encounter blockers, document them
- If you want to change approach, get confirmation

---

## File Naming Conventions

### Python Modules
- Use snake_case: `etf_scraper.py`, `price_downloader.py`
- Group by functionality in src/ subdirectories

### Data Files
- CSV files use lowercase: `etf_universe.csv`
- Individual ETF files use ticker: `SPY.csv`, `QQQ.csv`
- Dates in ISO format: `2025-10-04`

### Notebooks
- Number sequentially: `01_data_validation.ipynb`
- Use descriptive names: `02_signal_analysis.ipynb`

### Config Files
- Use JSON for parameters: `signal_parameters.json`
- Use .env for secrets (if needed later)

---

## Git Workflow (for later)

Currently not using git, but when we do:
- Commit after each task completion
- Use descriptive commit messages
- Tag phase completions

---

## Common Commands

### Check Current Status
```bash
cd /home/stuar/code/ETFTrader/Plan
cat PROJECT_PLAN.md | grep "Status:" | head -20
```

### See What's Not Started
```bash
grep "NOT STARTED" PROJECT_PLAN.md
```

### View Recent Progress
```bash
tail -50 PROGRESS_LOG.md
```

### Run Tests
```bash
cd /home/stuar/code/ETFTrader
pytest tests/
```

### Start Jupyter
```bash
cd /home/stuar/code/ETFTrader
jupyter notebook notebooks/
```

---

## Next Session Checklist

When you (Claude) start working on this project again:

1. [ ] Read this INSTRUCTIONS.md file
2. [ ] Check PROJECT_PLAN.md for current task
3. [ ] Review PROGRESS_LOG.md for recent context
4. [ ] Confirm current task with user (if unclear)
5. [ ] Begin work on current task
6. [ ] Update both plan files after completing work
7. [ ] Summarize progress to user

---

## Contact & Questions

If the user asks "where are we?" or "what's next?":
1. Read PROJECT_PLAN.md Current Phase section
2. Find the next unchecked `[ ]` task
3. Report current phase, task, and progress percentage
4. State what you'll work on next

If the user asks to "continue" or "keep going":
1. Start working on the current task
2. Follow the subtask checklist
3. Update plan as you go

If you're unsure about anything:
1. Document the question in PROGRESS_LOG.md
2. Ask the user for clarification
3. Don't guess or make major decisions without confirmation

---

## Summary

**To continue this project:**
1. Read PROJECT_PLAN.md → find current task
2. Work through subtasks in order
3. Create all deliverables listed
4. Update both plan files
5. Communicate progress

**Current Next Action:** Begin Phase 1, Task 1.1 - Create ETF data collection module

---

*This instruction file will be updated as the project evolves.*
