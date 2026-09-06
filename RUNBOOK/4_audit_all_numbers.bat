@echo off
title 4 - Audit All Numbers & Mathematical Reconciliation
cd /d "%~dp0\.."
echo Running independent arithmetic audit against Note 18, RBI macro data, and models...
python verify_audited_numbers.py
echo.
pause
