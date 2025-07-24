@echo off
echo ============================================
echo   Cleaning repository and updating GitHub
echo ============================================

cd /d "C:\Users\pauls\OneDrive\Working Files\tattoo_quoting_app"

REM Switch to main branch
git checkout main

REM Create a .gitignore file
(
echo .venv/
echo *.bat
echo @echo off
echo __pycache__/
echo *.pyc
) > .gitignore

REM Stage deletions and .gitignore
if exist "@echo off" git rm "@echo off"
if exist "deploy_app.bat" git rm deploy_app.bat
if exist ".venv" git rm -r --cached .venv
git add .gitignore

REM Commit and push changes
git commit -m "Cleaned repository: removed unnecessary files and added .gitignore"
git push origin main

echo ============================================
echo   Cleanup complete! Repo updated on GitHub.
echo ============================================
pause
