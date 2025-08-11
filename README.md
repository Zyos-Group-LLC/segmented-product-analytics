
# Segmented Product Analytics — Zyos Group

A Streamlit app for revenue and customer segmentation with ABC tiers, Pareto 80/20, new vs returning, recency buckets, and cohort heatmaps.

## Quick Start (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## One‑click Push to GitHub (Windows PowerShell)
Run `./deploy.ps1` and paste a **classic Personal Access Token** with `repo` scope when prompted (or use cached credentials).

## One‑click Push to GitHub (macOS/Linux)
Run:
```bash
chmod +x deploy.sh
./deploy.sh
```

---

# Segmented Product Analytics — PRO

Enhancements added:
- New vs Returning toggle: Lifetime vs Within current filters
- ABC scope selector: Global, Account-within-Bundle, Bundle-within-Account
- Pareto 80/20 flags for each scope
- Recency buckets
- Cohort heatmap (first purchase quarter x activity quarter)

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Push to GitHub
```
git init
git add .
git commit -m "Initial PRO version"
git branch -M main
git remote add origin https://github.com/<your-org>/<repo>.git
git push -u origin main
```

## Deploy to Streamlit Cloud
- Connect the GitHub repo at https://share.streamlit.io
- Set entry point to `app.py`
- Deploy
