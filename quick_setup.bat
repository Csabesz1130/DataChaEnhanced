@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo SIGNAL ANALYZER RELEASE BUILDER - QUICK SETUP v2.0
echo ================================================================
echo.

echo Step 1: Installing missing dependencies...
echo.

echo Installing PyInstaller...
pip install pyinstaller
set PYINSTALLER_INSTALL_ERROR=!ERRORLEVEL!

if !PYINSTALLER_INSTALL_ERROR! NEQ 0 (
    echo Warning: pip install failed, trying with --user flag...
    pip install --user pyinstaller
    set PYINSTALLER_INSTALL_ERROR=!ERRORLEVEL!
    
    if !PYINSTALLER_INSTALL_ERROR! NEQ 0 (
        echo Warning: --user install also failed, trying --upgrade...
        pip install --upgrade pyinstaller
        set PYINSTALLER_INSTALL_ERROR=!ERRORLEVEL!
    )
)

echo.
echo Installing requests for GitHub integration...
pip install requests
set REQUESTS_INSTALL_ERROR=!ERRORLEVEL!

if !REQUESTS_INSTALL_ERROR! NEQ 0 (
    echo Warning: pip install failed, trying with --user flag...
    pip install --user requests
    set REQUESTS_INSTALL_ERROR=!ERRORLEVEL!
)

echo.
echo Installing other requirements...
if exist requirements.txt (
    pip install -r requirements.txt
    if !ERRORLEVEL! NEQ 0 (
        echo Warning: Some requirements may not have installed correctly
    )
) else (
    echo Note: requirements.txt not found, skipping
)

echo.
echo Step 2: Checking installation...
echo.

echo Checking PyInstaller command availability...
pyinstaller --version >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [OK] PyInstaller command found
    for /f "tokens=*" %%i in ('pyinstaller --version 2^>nul') do echo     Version: %%i
    set PYINSTALLER_OK=1
) else (
    echo [ERROR] PyInstaller command not found in PATH
    echo.
    echo Troubleshooting steps:
    echo 1. Try closing this window and opening a new Command Prompt
    echo 2. Check if PyInstaller is installed: pip list ^| findstr pyinstaller
    echo 3. Try running: pip install --force-reinstall pyinstaller
    echo 4. Add Python Scripts folder to PATH if needed
    set PYINSTALLER_OK=0
)

echo.
echo Checking requests module...
python -c "import requests; print('Requests OK - Version:', requests.__version__)" 2>nul
if !ERRORLEVEL! EQU 0 (
    set REQUESTS_OK=1
) else (
    echo [ERROR] Requests module not found
    python -c "import sys; print('Python path:', sys.path)" 2>nul
    set REQUESTS_OK=0
)

echo.
echo Checking other core dependencies...
python -c "import numpy; print('NumPy OK')" 2>nul || echo [WARNING] NumPy not found
python -c "import scipy; print('SciPy OK')" 2>nul || echo [WARNING] SciPy not found  
python -c "import matplotlib; print('Matplotlib OK')" 2>nul || echo [WARNING] Matplotlib not found
python -c "import pandas; print('Pandas OK')" 2>nul || echo [WARNING] Pandas not found
python -c "import tkinter; print('Tkinter OK')" 2>nul || echo [WARNING] Tkinter not found

echo.
echo ================================================================
echo SETUP SUMMARY
echo ================================================================

if !PYINSTALLER_OK! EQU 1 if !REQUESTS_OK! EQU 1 (
    echo [SUCCESS] All critical dependencies are ready!
    echo.
    echo You can now run: python release_builder.py
    echo.
    echo If you still get errors, check the troubleshooting section above.
) else (
    echo [WARNING] Some dependencies are missing:
    if !PYINSTALLER_OK! EQU 0 echo   - PyInstaller command not available
    if !REQUESTS_OK! EQU 0 echo   - Requests module not available
    echo.
    echo Try these fixes:
    echo 1. Close this window and open a new Command Prompt as Administrator
    echo 2. Run: pip install --force-reinstall pyinstaller requests
    echo 3. Check your Python installation and PATH environment variable
    echo 4. Try using: python -m pip install pyinstaller requests
)

echo.
echo ================================================================
echo.
pause