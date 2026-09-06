@echo off
title 3 - Regenerate All Charts
cd /d "%~dp0\.."
echo Regenerating all 8 presentation charts from audited models...
python PROPOSAL/make_charts.py
echo.
pause
