@echo off
echo ============================================
echo   FORCING CLEAN app.py TO MAIN (Live App)
echo ============================================
cd /d "C:\Users\pauls\OneDrive\Working Files\tattoo_quoting_app"

REM Switch to main branch
git checkout main

REM Add app.py
git add app.py

REM Commit changes
git commit -m "Force push clean app.py to main" || echo No changes to commit

REM Push changes to main
git push origin main

echo ============================================
echo   Clean app.py pushed to MAIN (Live App)
echo ============================================
pause
