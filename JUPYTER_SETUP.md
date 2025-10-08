# Jupyter Notebook Setup Guide

## Quick Start

To run the ETF Data Validation notebook:

```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
jupyter notebook
```

Then navigate to `notebooks/01_data_validation.ipynb` in the browser interface.

---

## IPython Kernel Setup

The project has a dedicated IPython kernel called **"Python (ETFTrader)"** that automatically uses the correct virtual environment.

### Verify Kernel Installation

Check that the kernel is installed:

```bash
jupyter kernelspec list
```

You should see:
```
etftrader    /home/stuar/.local/share/jupyter/kernels/etftrader
```

### Using the Kernel

1. **Open Jupyter Notebook:**
   ```bash
   cd /home/stuar/code/ETFTrader
   jupyter notebook
   ```

2. **Select the Kernel:**
   - When you open `01_data_validation.ipynb`, the kernel should automatically be set to **"Python (ETFTrader)"**
   - If not, go to: **Kernel → Change Kernel → Python (ETFTrader)**

3. **Run All Cells:**
   - Click **Cell → Run All** to execute the entire notebook
   - Or run cells individually with **Shift+Enter**

---

## Troubleshooting

### Kernel Not Found

If the kernel is missing, reinstall it:

```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
python -m ipykernel install --user --name=etftrader --display-name="Python (ETFTrader)"
```

### Import Errors

If you get import errors (e.g., `ModuleNotFoundError`), ensure:

1. The virtual environment is activated
2. All dependencies are installed:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Plotting Issues

If plots don't render, ensure you have the notebook magic command:
```python
%matplotlib inline
```

This is already included in the notebook cells.

---

## Available Notebooks

### 1. **01_data_validation.ipynb**
   - **Purpose:** Validate ETF data collection quality
   - **Contents:**
     - ETF Universe Overview (298 ETFs)
     - Category & AUM Distribution
     - Data Quality Validation (90.2/100 score)
     - Missing Data Analysis (3.95% missing)
     - Price Data Visualization
     - Fundamental Data Review

   - **Runtime:** ~30-60 seconds
   - **Output:** Charts, statistics, and quality reports

---

## Notebook Features

### Data Visualizations

The notebook includes:
- ✅ Category distribution bar charts
- ✅ AUM distribution histograms
- ✅ Data quality pie charts
- ✅ Missing data analysis
- ✅ Sample price & volume charts (SPY)
- ✅ Expense ratio distributions

### Interactive Analysis

You can modify the notebook to:
- Change the sample ETF (default: SPY)
- Adjust quality thresholds
- Filter by category or AUM
- Export data to CSV

---

## Running from Command Line

Alternatively, run the notebook headless:

```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/01_data_validation.ipynb
```

This executes all cells and saves the output.

---

## Kernel Management

### Remove the Kernel

If you need to remove the kernel:

```bash
jupyter kernelspec uninstall etftrader
```

### Reinstall the Kernel

```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
python -m ipykernel install --user --name=etftrader --display-name="Python (ETFTrader)"
```

---

## Next Steps

After reviewing the data validation notebook:

1. ✅ Verify data quality meets criteria (>90/100 score achieved!)
2. ✅ Check ETF universe coverage (298 ETFs collected)
3. 🔄 Move to Phase 2: Signal Generation
4. 🔄 Create `02_signal_analysis.ipynb` for technical indicators

---

## Additional Resources

- **Jupyter Documentation:** https://jupyter.org/documentation
- **IPython Kernels:** https://ipython.readthedocs.io/en/stable/install/kernel_install.html
- **Matplotlib Gallery:** https://matplotlib.org/stable/gallery/index.html
- **Pandas Visualization:** https://pandas.pydata.org/docs/user_guide/visualization.html

---

## Support

If you encounter issues:

1. Check that the virtual environment is activated
2. Verify all dependencies are installed
3. Ensure the kernel is properly configured
4. Review error messages in the notebook output

**Last Updated:** 2025-10-04
