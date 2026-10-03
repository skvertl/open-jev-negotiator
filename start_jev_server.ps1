# Helper script to launch Open-Jev-2B inference server
Write-Host "Starting Open-Jev-2B server on http://127.0.0.1:8791..." -ForegroundColor Cyan
& "C:\Users\user\Open-Jev\.venv\Scripts\python.exe" -m jev.server `
  --checkpoint "C:\Users\user\Open-Jev\models\Open-Jev-2B\package\checkpoint" `
  --device cuda:0 `
  --max-length 4096 `
  --batch-size 1 `
  --no-prefix-cache `
  --host 127.0.0.1 `
  --port 8791
