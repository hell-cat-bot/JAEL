@echo off
title 5 - Open Slide Artefacts
cd /d "%~dp0\.."
echo Opening Slide 1 and Slide 2 in default browser...
python PROPOSAL\make_slide1.py --open
python PROPOSAL\make_slide2.py --open
