@echo off
rem Double-click this file to open the convlab desktop application.
rem
rem It sets the app up the first time it runs (virtual environment plus
rem dependencies), which takes a few minutes, and starts immediately on
rem every run after that.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up convlab. This takes a few minutes.
    echo.
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo.
        echo Could not create the environment. Is Python 3.10+ installed?
        echo Download it from https://www.python.org/downloads/
        echo Be sure to tick "Add python.exe to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e ".[semantic]"
    if errorlevel 1 (
        echo.
        echo Installation failed. See the messages above.
        pause
        exit /b 1
    )
    echo.
    echo Setup complete.
    echo.
)

start "" ".venv\Scripts\pythonw.exe" -m convlab.gui
