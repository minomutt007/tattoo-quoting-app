@echo off
REM ——————— Update App on GitHub ———————

REM Change to script’s directory (your repo root)
cd /d "%~dp0"

echo.
echo --- Checking out main and pulling latest ---
git checkout main
git pull origin main

echo.
echo --- Staging changes ---
git add app.py settings.json

echo.
echo --- Committing updates ---
git commit -m "Update app with crop UI and performance tweaks"

echo.
echo --- Pushing to GitHub (main) ---
git push origin main

echo.
echo --- Merge into dev branch ---
git checkout dev
git merge main

echo.
echo --- Pushing to GitHub (dev) ---
git push origin dev

echo.
echo ✅  All done! Your app.py is now live on GitHub.
pause
