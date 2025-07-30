@echo off
rem Set character encoding to UTF-8
chcp 65001 >nul

rem --- Batch file for executing flow.py ---

rem Move to the directory where the batch file is located
cd /d "%~dp0"

rem Specify the path to the virtual environment activation script
set VENV_ACTIVATE_SCRIPT=.\venv\Scripts\activate.bat

rem Specify the Python script filename
set PYTHON_SCRIPT=flow.py

rem Check if the virtual environment activation script exists
if not exist "%VENV_ACTIVATE_SCRIPT%" (
    echo Error: Virtual environment activation script not found.
    echo Path: %VENV_ACTIVATE_SCRIPT%
    echo Please check if the virtual environment is properly created.
    echo.
    pause
    exit /b 1
)

rem Check if the Python script exists
if not exist "%PYTHON_SCRIPT%" (
    echo Error: Python script not found.
    echo Filename: %PYTHON_SCRIPT%
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
rem Activate the virtual environment using call command
call "%VENV_ACTIVATE_SCRIPT%"

rem Check if virtual environment activation was successful
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    echo.
    pause
    exit /b 1
)

rem Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Virtual environment may not be set up correctly.
    echo.
    pause
    exit /b 1
)

echo.
echo ----------------------------------------
echo Executing %PYTHON_SCRIPT%...
echo ----------------------------------------
echo.

rem Execute the Python script
python "%PYTHON_SCRIPT%"

rem Check the execution result of the Python script
if errorlevel 1 (
    echo.
    echo ----------------------------------------
    echo Error: An error occurred during script execution.
    echo Exit code: %ERRORLEVEL%
    echo ----------------------------------------
    echo.
    goto :cleanup
)

echo.
echo ----------------------------------------
echo Script execution completed successfully.
echo ----------------------------------------
echo.

:cleanup
rem Deactivate the virtual environment
if defined VIRTUAL_ENV (
    call deactivate
)

rem Pause to prevent the window from closing immediately
pause

rem Exit with appropriate exit code
if errorlevel 1 (
    exit /b %ERRORLEVEL%
) else (
    exit /b 0
)
