# UniView Services Startup Script
# Run this from the project root: .\start_services.ps1
#
# Optional parameters:
#   .\start_services.ps1 -DatasetPath "C:\path\to\MATRIX_30x30\MATRIX_30x30"
#   .\start_services.ps1 -Drones "3,4,6,7"
#   .\start_services.ps1 -DatasetPath "C:\..." -Drones "3,4,6,7"

param(
    [string]$DatasetPath = "",
    [string]$Drones = "3,4,6,7"
)

$Root = $PSScriptRoot

# Resolve dataset path to absolute so subprocesses in other directories can find it
if (-not $DatasetPath) {
    $DatasetPath = Join-Path $Root "MATRIX_30x30\MATRIX_30x30"
}
$DatasetPath = (Resolve-Path $DatasetPath).Path

Write-Host "Starting UniView Services..." -ForegroundColor Cyan

# 1. Backend (port 8000)
Write-Host "`nStarting Backend on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\backend'; python main.py"
Start-Sleep -Seconds 2

# 2. Gateway (port 8080)
Write-Host "Starting Gateway on port 8080..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\gateway'; python main.py"
Start-Sleep -Seconds 2

# 3. Algorithm (port 8001)
Write-Host "Starting Algorithm on port 8001 (drones: $Drones)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\algorithm'; python main.py --drone-ids $Drones"
Start-Sleep -Seconds 2

# 4. ENet Drone Streamer
Write-Host "Starting Mock Drone Streamer (drones: $Drones)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\mock_drone_streamer'; python run_all_drones.py --dataset '$DatasetPath' --drones $Drones"
Start-Sleep -Seconds 1

# 5. Frontend (port 5173)
Write-Host "Starting Frontend on port 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\frontend'; npm run dev"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "All services started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Backend:          http://localhost:8000" -ForegroundColor White
Write-Host "Algorithm:        http://localhost:8001" -ForegroundColor White
Write-Host "Gateway:          http://localhost:8080" -ForegroundColor White
Write-Host "Frontend:         http://localhost:5173" -ForegroundColor White
Write-Host "ENet streamers:   drones $Drones"         -ForegroundColor White
Write-Host "========================================`n" -ForegroundColor Green
