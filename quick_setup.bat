@echo off
echo ================================================================
echo SIGNAL ANALYZER RELEASE BUILDER - QUICK SETUP
echo ================================================================
echo.

echo Step 1: Installing missing dependencies...
echo.

echo Installing PyInstaller...
pip install pyinstaller
if %ERRORLEVEL% NEQ 0 (
    echo Trying with --user flag...
    pip install --user pyinstaller
)

echo Installing requests for GitHub integration...
pip install requests
if %ERRORLEVEL% NEQ 0 (
    echo Trying with --user flag...
    pip install --user requests
)

echo.
echo Step 2: Checking installation...
python -c "import pyinstaller; print('PyInstaller OK')"
python -c "import requests; print('Requests OK')"

echo.
echo ================================================================
echo SETUP COMPLETE!
echo ================================================================
echo.
echo Now you can run: python release_builder.py
echo.
echo If you still get permission errors, try running this batch file
echo as Administrator (right-click -> Run as administrator)
echo.
pause