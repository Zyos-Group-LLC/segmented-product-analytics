#!/usr/bin/env bash
set -e
REPO_URL="https://github.com/Zyos-Group-LLC/segmented-product-analytics.git"

echo "Preparing repo for Zyos-Group-LLC/segmented-product-analytics..."
git init
git add .
git commit -m "Initial commit - Zyos Group segmented analytics"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

git push -u origin main

echo "Pushed to $REPO_URL"
echo "Next: Deploy at https://share.streamlit.io -> select Zyos-Group-LLC/segmented-product-analytics, entry point app.py"