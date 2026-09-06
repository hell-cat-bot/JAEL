@echo off
title JA-LE Master Reproducibility Runner
echo =======================================================================
echo   JA-LE: Reproducibility Runner -- Re-run every number in 30 seconds
echo   TVS Credit E.P.I.C 8.0 -- Problem (E) Swarm Intelligence Lending Network
echo =======================================================================
echo.
cd /d "%~dp0\.."
python RUNBOOK\run_all.py
echo.
echo Press any key to exit...
pause >nul
