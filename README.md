# ⚡ AI Stock God Mode

> **Professional AI-powered US stock analysis platform**  
> Random Forest · Gradient Boosting · Full Technical Suite · Real-time Data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🚀 Features

| Feature | Details |
|---|---|
| **AI Forecast** | Random Forest / Gradient Boosting / Ensemble — predict price N days ahead |
| **Technical Indicators** | RSI (9+14), MACD, Bollinger Bands, EMA 20/50/200, ATR, ADX, Stochastic |
| **Candlestick Chart** | Full OHLC with overlay indicators + AI confidence band |
| **Model Backtest** | Out-of-sample time-series split — real accuracy, no data leakage |
| **Feature Importance** | See which indicators drive the AI signal |
| **Fundamentals** | Market cap, P/E, EPS, Beta, 52W High/Low from Yahoo Finance |
| **Flexible Config** | Ticker, period (3M–2Y), forecast horizon (3–30d), model choice |

---

## 🛠 Setup (Local)

```bash
git clone https://github.com/YOUR_USERNAME/ai-stock-godmode.git
cd ai-stock-godmode
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **New app** → select your repo → `app.py` → **Deploy**
4. Done ✅ — no API keys needed, all data is free via Yahoo Finance

---

## 📐 Architecture

```
app.py
├── Data Layer       → yfinance (OHLCV + fundamentals)
├── Indicator Layer  → pandas-ta (RSI, MACD, BB, EMA, ATR, ADX, Stoch)
├── ML Layer         → scikit-learn (RandomForest / GradientBoosting)
│   ├── Features: 20 engineered features (momentum, vol, technicals)
│   ├── Split: TimeSeriesSplit (no lookahead leakage)
│   └── Output: price prediction + MAPE accuracy score
└── Viz Layer        → Plotly (candlestick, MACD, RSI, gauge, bar charts)
```

---

## ⚠️ Disclaimer

For informational and educational purposes only. Not financial advice.  
Past model accuracy does not guarantee future results.

---

## 📦 Requirements

```
streamlit>=1.35.0
yfinance>=0.2.40
pandas>=2.0.0
pandas-ta>=0.3.14b
scikit-learn>=1.4.0
numpy>=1.26.0
plotly>=5.20.0
```
