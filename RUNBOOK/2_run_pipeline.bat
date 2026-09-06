@echo off
title 2 - Full Pipeline Execution (SMOKE Profile)
cd /d "%~dp0\.."
echo Running SMOKE pipeline (generates features, graphs, models, and JSON audit files)...
python scripts/run_v1.py --profile SMOKE
echo.
pause
