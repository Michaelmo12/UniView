# UniView Services Startup Script
# Run this from the project root: .\start_services.ps1

Write-Host "Starting UniView Services..." -ForegroundColor Cyan

# Start Backend (port 8000)
Write-Host "`nStarting Backend on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; python main.py"

# Wait a moment for backend to initialize
Start-Sleep -Seconds 2

# Start Gateway (port 8080)
Write-Host "Starting Gateway on port 8080..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd gateway; python main.py"

# Wait a moment for gateway to initialize
Start-Sleep -Seconds 2

# Start Frontend (port 5173)
Write-Host "Starting Frontend on port 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "All services started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "Gateway:  http://localhost:8080" -ForegroundColor White
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "========================================`n" -ForegroundColor Green
