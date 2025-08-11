param(
  [string]$RepoUrl = "https://github.com/Zyos-Group-LLC/segmented-product-analytics.git"
)
Write-Host "Preparing repo for Zyos-Group-LLC/segmented-product-analytics..." -ForegroundColor Cyan

git init
git add .
git commit -m "Initial commit - Zyos Group segmented analytics"
git branch -M main
git remote remove origin 2>$null
git remote add origin $RepoUrl

# If user has credential manager, this will use it; otherwise they'll be prompted
git push -u origin main

Write-Host "Pushed to $RepoUrl" -ForegroundColor Green
Write-Host "Next: Deploy at https://share.streamlit.io -> select Zyos-Group-LLC/segmented-product-analytics, entry point app.py" -ForegroundColor Yellow