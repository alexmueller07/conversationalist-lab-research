@echo off
rem Conversation Analyst -- first-run setup.
rem
rem Runs once, in a visible window, the first time the app is launched
rem after installation. Finds a usable Python or installs a private one,
rem then installs the analysis engine. Every later launch skips straight
rem to the app.
rem
rem Written to survive the messy world: a second copy started while one is
rem running backs off instead of corrupting the environment; a half-built
rem environment from an interrupted or failed run is deleted and rebuilt
rem rather than tripped over; and if one Python on the machine cannot
rem create the environment, the next candidate is tried before falling
rem back to installing a private runtime.

setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Conversation Analyst - first-run setup

rem ---- 0. one setup at a time ----------------------------------------
rem mkdir is atomic: exactly one process wins the right to run setup.
mkdir "%~dp0setup.lock" 2>nul
if errorlevel 1 (
    echo.
    echo  Another setup window appears to be running already.
    echo  Let it finish. If you are certain none is running, delete the
    echo  folder below and start Conversation Analyst again:
    echo    %~dp0setup.lock
    echo.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Conversation Analyst - one-time setup
echo  ============================================================
echo.
echo  This installs the analysis engine (about 1.6 GB of scientific
echo  libraries). It runs once and takes 10-30 minutes depending on
echo  your connection. Later launches open in seconds.
echo.

rem ---- 1. clear the wreckage of any earlier attempt -------------------
if exist "%~dp0pyenv" if not exist "%~dp0pyenv\Scripts\python.exe" (
    echo  Removing an incomplete environment from an earlier attempt...
    rmdir /s /q "%~dp0pyenv"
)
if exist "%~dp0pyenv\Scripts\python.exe" (
    set "APPPY=%~dp0pyenv\Scripts\python.exe"
    set "APPPYW=%~dp0pyenv\Scripts\pythonw.exe"
    echo  Reusing the environment from an earlier attempt.
    goto :deps
)

rem ---- 2. try each Python on the machine ------------------------------
rem A candidate must be 3.10+ with Tkinter AND able to actually create the
rem environment; a Python that fails either test is skipped, not fatal.
for %%V in (3.12 3.11 3.10) do (
    if not defined APPPY (
        py -%%V -c "import tkinter" >nul 2>nul
        if not errorlevel 1 (
            for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)"') do (
                call :try_venv "%%P"
            )
        )
    )
)
if not defined APPPY (
    python -c "import sys,tkinter; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)"') do (
            call :try_venv "%%P"
        )
    )
)
if defined APPPY goto :deps

rem ---- 3. no usable Python: install a private runtime -----------------
echo  No Python on this computer could create the environment, so a
echo  private copy will be installed just for this app (about 27 MB).
echo  Downloading from python.org...
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

rem ---- 4. record what to launch and go --------------------------------
> "%~dp0installed.flag" echo %APPPYW%
rmdir "%~dp0setup.lock" 2>nul
echo.
echo  ============================================================
echo   Setup complete. Conversation Analyst is opening now.
echo   Next time, it opens in seconds.
echo  ============================================================
timeout /t 3 >nul
start "" "%APPPYW%" -m conversation_analyst.gui
exit /b 0

rem ---- subroutine: attempt a venv with one candidate ------------------
:try_venv
if defined APPPY exit /b 0
echo  Trying the Python at:
echo    %~1
"%~1" -m venv "%~dp0pyenv" >nul 2>nul
if exist "%~dp0pyenv\Scripts\python.exe" (
    set "APPPY=%~dp0pyenv\Scripts\python.exe"
    set "APPPYW=%~dp0pyenv\Scripts\pythonw.exe"
    echo  Environment created.
    exit /b 0
)
echo  ...that one could not create the environment; trying the next.
if exist "%~dp0pyenv" rmdir /s /q "%~dp0pyenv"
exit /b 1

:fail_download
rmdir "%~dp0setup.lock" 2>nul
echo.
echo  The download from python.org did not complete. Check your
echo  internet connection and start Conversation Analyst again.
pause
exit /b 1

:fail
rmdir "%~dp0setup.lock" 2>nul
echo.
echo  Setup did not complete. Read the messages above; starting
echo  Conversation Analyst again will retry from this point.
pause
exit /b 1
