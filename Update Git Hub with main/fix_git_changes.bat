@echo off
echo ============================================
echo   Staging all changes and pushing to MAIN
echo ============================================
cd /d "C:\Users\pauls\OneDrive\Working Files\tattoo_quoting_app"

REM Switch to main branch
git checkout main

REM Stage all changes (including deletions)
git add -A

REM Commit changes
git commit -m "Auto-fix: Commit all changes and deletions"

REM Push to main
git push origin main

echo ============================================
echo   All changes pushed to MAIN (Live App)
echo ============================================
pause
