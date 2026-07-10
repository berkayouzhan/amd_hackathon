@echo off
:: =====================================================================
:: Adaptive Model Dispatcher — Local Automation Script
:: Runs the offline pipeline, generates reports, and copies them to the dashboard.
:: =====================================================================

echo [1/4] Setting environment variables for offline test...
set TASKS_INPUT_PATH=.\local_test\tasks.json
set RESULTS_OUTPUT_PATH=.\local_test\results.json
set USE_FAKE_FIREWORKS=1
set FIREWORKS_API_KEY=test-key
set FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
set ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/gemma-4-31b-it

echo.
echo [2/4] Running the main pipeline...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] main.py failed! Make sure Python is in your PATH and dependencies are installed.
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Copying run_report.json to the dashboard directory...
if not exist ".\dashboard" mkdir ".\dashboard"
copy /Y ".\local_test\run_report.json" ".\dashboard\run_report.json" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to copy run_report.json. Verify folder permissions.
) else (
    echo [SUCCESS] Report copied to dashboard folder!
)

echo.
echo [4/4] Setup complete!
echo.
echo To view your dashboard:
echo   1. Keep your running python server terminal open.
echo   2. Open the following URL in your web browser:
echo      http://localhost:8080/optiroute-ai/dashboard/index.html
echo.
pause
