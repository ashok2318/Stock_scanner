# MarketPulse Pro — RSI + KPI Dashboard

Professional local Mac dashboard.

## Improvements
- RSI divergence with confirmed price swings, minimum separation and RSI difference, 30-bar signal expiry, and chart annotations
- RSI status colors: green positive, red weak, orange overbought, blue oversold
- MACD (12, 26, 9) with signal line and colored histogram
- Market cap formatted in M/B/T
- EPS Growth and Revenue Growth displayed clearly as percentages
- 52-week position displayed as 0–100% with interpretation
- KPI cards arranged in two rows of three with clear status labels
- Watchlist comparison: WDC, MU, STX, AMD, NVDA
- Candlestick, configurable moving averages and RSI
- CSV export

## Run on Mac
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```
Then open http://localhost:8501

## Commit changes

The project uses Yahoo Finance through `yfinance`; no API key is required.
Local virtual environments, Python caches, and secret files are excluded by
`.gitignore`.

Review and commit the project from this directory:

```bash
git status
git add .gitignore README.md app.py requirements.txt run_dashboard.sh
git diff --cached
git commit -m "Add MarketPulse dashboard with MACD and colored RSI statuses"
```
