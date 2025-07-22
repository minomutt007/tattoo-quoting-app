@echo off
echo ============================================
echo   Updating GitHub (dev -> main)
echo ============================================
cd /d "C:\Users\pauls\OneDrive\Working Files\tattoo_quoting_app"

REM Commit changes to dev
git checkout dev
git add .
git commit -m "Auto-update: Committing latest changes to dev" || echo No changes to commit
git push origin dev

REM Merge dev into main
git checkout main
git merge dev
git push origin main

REM Switch back to dev
git checkout dev

echo ============================================
echo   GitHub update complete! Streamlit Cloud will redeploy.
echo ============================================
pause
