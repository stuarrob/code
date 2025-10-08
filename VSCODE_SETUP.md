# VSCode Setup Guide for ETF Trader

This guide explains how to use VSCode with the ETF Trader project.

## Quick Start

1. **Open the project in VSCode:**
   ```bash
   cd /home/stuar/code/ETFTrader
   code .
   ```

2. **Install recommended extensions:**
   - When you open the project, VSCode will prompt you to install recommended extensions
   - Click "Install All" or install them individually from the Extensions panel (Ctrl+Shift+X)

3. **Select Python interpreter:**
   - Press `Ctrl+Shift+P` to open the command palette
   - Type "Python: Select Interpreter"
   - Choose the interpreter at `./venv/bin/python`
   - This should happen automatically, but verify it shows in the bottom-left corner

## Recommended Extensions

### Essential Python Extensions
- **Python** (`ms-python.python`) - Core Python language support
- **Pylance** (`ms-python.vscode-pylance`) - Fast, feature-rich language server
- **Black Formatter** (`ms-python.black-formatter`) - Code formatting
- **Flake8** (`ms-python.flake8`) - Linting
- **isort** (`ms-python.isort`) - Import organization

### Jupyter Notebook Extensions
- **Jupyter** (`ms-toolsai.jupyter`) - Full Jupyter notebook support
- **Jupyter Keymap** - Jupyter keyboard shortcuts
- **Jupyter Renderers** - Enhanced output rendering

### Data Science Extensions
- **Rainbow CSV** (`mechatroner.rainbow-csv`) - Colorize and navigate CSV files
- **Edit CSV** (`janisdd.vscode-edit-csv`) - In-editor CSV editing
- **Data Preview** (`randomfractalsinc.vscode-data-preview`) - Preview data files

### Git Extensions
- **GitLens** (`eamodio.gitlens`) - Enhanced Git capabilities
- **Git Graph** (`mhutchie.git-graph`) - Visualize git history

### Productivity Extensions
- **Error Lens** (`usernamehw.errorlens`) - Inline error highlighting
- **Indent Rainbow** (`oderwat.indent-rainbow`) - Colorize indentation levels
- **Material Icon Theme** (`PKief.material-icon-theme`) - Better file icons

## Running Jupyter Notebooks

### Method 1: Interactive Window (Recommended)
1. Open any `.ipynb` file in the `notebooks/` folder
2. VSCode will automatically open it in the notebook editor
3. Click "Select Kernel" in the top-right corner
4. Choose the Python interpreter from `./venv/bin/python`
5. Run cells using:
   - `Shift+Enter` - Run cell and move to next
   - `Ctrl+Enter` - Run cell and stay
   - `Alt+Enter` - Run cell and insert below

### Method 2: Classic Jupyter Server
```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook notebooks/
```

## Running Python Scripts

### From Command Palette
1. Press `Ctrl+Shift+P`
2. Type "Python: Run Python File in Terminal"
3. The script will run with the correct PYTHONPATH automatically

### Using Debug Configurations
Press `F5` or click Run → Start Debugging, then choose:
- **Python: Current File** - Run the currently open file
- **Python: Flask App** - Run the Flask web server
- **Streamlit: Web App** - Run the Streamlit dashboard
- **Python: Pytest** - Run all tests
- **Python: Data Collection Script** - Run ETF data scraper
- **Python: Portfolio Optimizer** - Run optimization

## PYTHONPATH Configuration

**You don't need to worry about PYTHONPATH!** Everything is configured automatically:

1. **`.vscode/settings.json`** sets:
   ```json
   "python.analysis.extraPaths": ["${workspaceFolder}/src"]
   "terminal.integrated.env.linux": {
       "PYTHONPATH": "${workspaceFolder}/src"
   }
   ```

2. **`.env`** file contains:
   ```
   PYTHONPATH=/home/stuar/code/ETFTrader/src
   ```

3. **All debug configurations** include `PYTHONPATH` in their environment

### Importing Modules
In any Python file or notebook, you can now import directly from `src/`:

```python
# This works automatically
from data_collection.etf_scraper import scrape_etf_list
from signals.indicators import calculate_macd
from optimization.portfolio_optimizer import optimize_portfolio
```

## Working with CSV Data

### Rainbow CSV
- Open any `.csv` file
- Data will be automatically colorized by column
- Hover over columns to see alignment
- Right-click → "Rainbow CSV" for advanced options

### Data Preview
- Right-click any `.csv`, `.json`, or data file
- Select "Preview Data" to see formatted table view

## Testing

### Running Tests
1. Open the Testing panel (beaker icon in left sidebar)
2. Click "Configure Python Tests"
3. Select "pytest"
4. Tests in `tests/` will appear in the tree
5. Click play button next to any test to run it

### Debugging Tests
1. Set breakpoints in test files
2. Right-click test in Testing panel
3. Select "Debug Test"

## Code Formatting and Linting

### Auto-format on Save
Configured automatically! When you save any `.py` file:
- **Black** formats code (120 char line length)
- **isort** organizes imports
- **Flake8** shows linting errors

### Manual Formatting
- Right-click in file → "Format Document"
- Or press `Shift+Alt+F`

## IntelliSense and Autocomplete

**Pylance** provides:
- Auto-completion for all installed packages
- Type checking
- Auto-imports
- Function signatures
- Documentation on hover

**IntelliCode** provides:
- AI-powered suggestions
- Ranked completions based on common patterns

## Debugging Python Code

### Set Breakpoints
- Click in the gutter (left of line numbers) to set breakpoints
- Red dots appear where code will pause

### Debug Controls
- `F5` - Start debugging
- `F10` - Step over
- `F11` - Step into
- `Shift+F11` - Step out
- `F5` - Continue
- `Shift+F5` - Stop

### Debug Console
- Evaluate expressions while paused
- Inspect variables
- Access at bottom panel during debug session

## File Navigation

### Quick File Open
- `Ctrl+P` - Quick open file by name
- `Ctrl+Shift+F` - Search across all files
- `Ctrl+T` - Go to symbol in workspace

### Split Editor
- `Ctrl+\` - Split editor
- `Ctrl+1/2/3` - Focus editor group

## Terminal Integration

### Integrated Terminal
- Press `` Ctrl+` `` to open terminal
- Virtual environment activates automatically
- PYTHONPATH is set automatically

### Multiple Terminals
- Click `+` in terminal panel to open new terminals
- All terminals inherit the configured environment

## Workspace Settings

All configured in `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.analysis.extraPaths": ["${workspaceFolder}/src"],
    "python.terminal.activateEnvironment": true,
    "notebook.formatOnSave.enabled": true,
    // ... and more
}
```

## Troubleshooting

### "Module not found" errors
1. Verify Python interpreter is set to `./venv/bin/python`
2. Check bottom-left corner of VSCode
3. If wrong, press `Ctrl+Shift+P` → "Python: Select Interpreter"

### Jupyter kernel not found
1. Install ipykernel in virtual environment:
   ```bash
   source venv/bin/activate
   pip install ipykernel
   ```
2. Reload VSCode: `Ctrl+Shift+P` → "Developer: Reload Window"

### Imports not working in notebooks
1. Restart the Jupyter kernel: Click "Restart" in notebook toolbar
2. Verify kernel shows `./venv/bin/python`

### Linting errors not showing
1. Ensure Flake8 extension is installed
2. Check Output panel → Select "Python" from dropdown
3. Install flake8: `pip install flake8`

## Keyboard Shortcuts Cheat Sheet

### General
- `Ctrl+Shift+P` - Command Palette
- `Ctrl+P` - Quick Open File
- `Ctrl+,` - Settings
- `` Ctrl+` `` - Toggle Terminal

### Editing
- `Ctrl+/` - Toggle comment
- `Shift+Alt+F` - Format document
- `Ctrl+Space` - Trigger autocomplete
- `F12` - Go to definition
- `Alt+F12` - Peek definition

### Notebooks
- `Shift+Enter` - Run cell, select below
- `Ctrl+Enter` - Run cell
- `Alt+Enter` - Run cell, insert below
- `DD` - Delete cell (in command mode)
- `A` - Insert cell above
- `B` - Insert cell below

### Debugging
- `F9` - Toggle breakpoint
- `F5` - Start/Continue debugging
- `F10` - Step over
- `F11` - Step into

## Next Steps

1. Install the recommended extensions
2. Open `notebooks/` and create a test notebook
3. Try importing from `src/` modules
4. Run the debugger on a simple script

For project-specific usage, see [README.md](README.md) and [Plan/PROJECT_PLAN.md](Plan/PROJECT_PLAN.md).
