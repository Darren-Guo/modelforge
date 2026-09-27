# ModelForge 一键启动：后端(8000) + 前端(5173)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[1/2] 启动后端 FastAPI (127.0.0.1:8000) ..." -ForegroundColor Cyan
Start-Process -WorkingDirectory "$root\backend" -WindowStyle Minimized `
    -FilePath "python" -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"

Start-Sleep -Seconds 3

Write-Host "[2/2] 启动前端 Vite (127.0.0.1:5173) ..." -ForegroundColor Cyan
Start-Process -WorkingDirectory "$root\frontend" -WindowStyle Minimized `
    -FilePath "npm" -ArgumentList "run", "dev"

Start-Sleep -Seconds 3
Write-Host "完成！打开浏览器访问 http://127.0.0.1:5173" -ForegroundColor Green
Start-Process "http://127.0.0.1:5173"
