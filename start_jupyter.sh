#!/bin/bash
# Start JupyterLab for ETFTrader project

cd "$(dirname "$0")"
source venv/bin/activate
jupyter lab --no-browser --port=8888
