@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated.
echo Starting Signal Analyzer with drag and drop support...
python run_with_hot_reload.py
pause
