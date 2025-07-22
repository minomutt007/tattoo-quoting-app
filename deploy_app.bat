@echo off
echo =========================================
echo DEPLOYING LATEST APP TO STREAMLIT CLOUD
echo =========================================

cd /d "C:\Users\pauls\OneDrive\Working Files\tattoo_quoting_app"

REM Copy app_dev.py to app.py
echo Copying app_dev.py to app.py...
copy /Y app_dev.py app.py

REM Stage changes
echo Staging changes...
git add .

REM Commit changes
set /p commitmsg="Enter commit message (default: Deploy latest app): "
if "%commitmsg%"=="" set commitmsg=Deploy latest app
git commit -m "%commitmsg%"

REM Push changes to main branch
echo Pushing changes to main...
git push origin main

echo =========================================
echo DEPLOY COMPLETE! Streamlit Cloud will rebuild automatically.
echo =========================================

pause
