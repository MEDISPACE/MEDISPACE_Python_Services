# start_ml.ps1 - Khởi động ML Recommendation Service (port 8002)

$env:PYTHONIOENCODING = "utf-8"

$ML_PATH = "D:\HCMUTE\MAJOR\1TLCN\MEDISPACE_Project\MEDISPACE_Python_Services\MEDISPACE_ML_Service"

Write-Host "[ML Service] Starting ML Recommendation Service (port 8002)..."
Set-Location $ML_PATH
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8002
