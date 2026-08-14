@echo off
rem Conversation Analyst -- first-run setup.
rem
rem Runs once, in a visible window, the first time the app is launched
rem after installation. Finds a usable Python or installs a private one,
rem then installs the analysis engine. Every later launch skips straight
rem to the app.

setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Conversation Analyst - first-run setup

echo.
echo  ============================================================
echo   Conversation Analyst - one-time setup
echo  ============================================================
echo.
echo  This installs the analysis engine (about 1.6 GB of scientific
echo  libraries). It runs once and takes 10-30 minutes depending on
echo  your connection. Later launches open in seconds.
echo.

rem ---- 1. find a usable Python (3.10+ with Tkinter) -------------------
set "PYEXE="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYEXE (
        py -%%V -c "import tkinter" >nul 2>nul
        if not errorlevel 1 (
            for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)"') do set "PYEXE=%%P"
        )
    )
)
if not defined PYEXE (
    python -c "import sys,tkinter; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)"') do set "PYEXE=%%P"
    )
)

if defined PYEXE (
    echo  Using the Python already on this computer:
    echo    !PYEXE!
    echo.
    echo  Creating the app's private environment...
    "!PYEXE!" -m venv "%~dp0pyenv"
    if errorlevel 1 goto :fail
    set "APPPY=%~dp0pyenv\Scripts\python.exe"
    set "APPPYW=%~dp0pyenv\Scripts\pythonw.exe"
    goto :deps
)

rem ---- 2. no Python: install a private runtime ------------------------
echo  Python was not found, so a private copy will be installed just
echo  for this app (about 27 MB). Downloading from python.org...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ProgressPreference='SilentlyContinue';" ^
  "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;" ^
  "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%~dp0runtime-setup.exe'"
if not exist "%~dp0runtime-setup.exe" goto :fail_download

echo  Installing the private runtime (no admin needed)...
"%~dp0runtime-setup.exe" /quiet InstallAllUsers=0 TargetDir="%~dp0runtime" ^
  AssociateFiles=0 Shortcuts=0 Include_launcher=0 Include_test=0 Include_doc=0 PrependPath=0
if not exist "%~dp0runtime\python.exe" goto :fail
set "APPPY=%~dp0runtime\python.exe"
set "APPPYW=%~dp0runtime\pythonw.exe"

:deps
echo.
echo  Installing the analysis engine. The long pause on large packages
echo  is normal -- this is the 1.6 GB part.
echo.
"%APPPY%" -m pip install --upgrade pip
"%APPPY%" -m pip install -e "%~dp0app[semantic]"
if errorlevel 1 goto :fail

rem ---- 3. record what to launch and go --------------------------------
> "%~dp0installed.flag" echo %APPPYW%
echo.
echo  ============================================================
echo   Setup complete. Conversation Analyst is opening now.
echo   Next time, it opens in seconds.
echo  ============================================================
timeout /t 3 >nul
start "" "%APPPYW%" -m conversation_analyst.gui
exit /b 0

:fail_download
echo.
echo  The download from python.org did not complete. Check your
echo  internet connection and start Conversation Analyst again.
pause
exit /b 1

:fail
echo.
echo  Setup did not complete. Read the messages above; starting
echo  Conversation Analyst again will retry from this point.
pause
exit /b 1
