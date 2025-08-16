#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip wheel setuptools
python3 -m pip install streamlit pandas numpy joblib scikit-learn

echo "Minimális frontend függőségek telepítve."