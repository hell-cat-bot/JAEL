@echo off
title 1 - Install Python Requirements
cd /d "%~dp0\.."
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
echo.
echo [DONE] Dependencies installed.
pause
