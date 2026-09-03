@echo off
echo Starting MediCare Healthcare App...
cd /d "%~dp0"
python -m streamlit run app.py
pause
