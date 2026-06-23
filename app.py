import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock God Mode",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Dark Terminal / Bloomberg Style
# ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080c14;
    color: #c8d0dc;
  }
  .stApp { background: #080c14; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2530;
  }
  section[data-testid="stSidebar"] * { color: #a0aab8 !important; }
  section[data-testid="stSidebar"] .stTextInput input,
  section[data-testid="stSidebar"] .stSelectbox select {
    background: #161b22 !important;
    border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }

  /* Header band */
  .god-header {
    background: linear-gradient(135deg, #0a1628 0%, #0f2040 50%, #0a1628 100%);
    border: 1px solid #1a2d4a;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
  }
  .god-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #0066ff, #00d4ff, transparent);
  }
  .god-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8f4ff;
    margin: 0;
    letter-spacing: 0.05em;
  }
  .god-header .subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #4a7fa5;
    margin-top: 4px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00ff88;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 8px #00ff88;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* Metric cards */
  .metric-card {
    background: #0d1117;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
  }
  .metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #00d4ff);
    opacity: 0.6;
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #e8f4ff;
    line-height: 1;
  }
  .metric-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    margin-top: 6px;
  }
  .delta-up { color: #00cc66; }
  .delta-down { color: #ff4466; }

  /* Signal badge */
  .signal-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    border: 1px solid;
  }
  .signal-buy   { background:#001a0d; color:#00ff88; border-color:#00cc44; }
  .signal-sell  { background:#1a0009; color:#ff6688; border-color:#cc2244; }
  .signal-hold  { background:#0d1200; color:#ffe066; border-color:#ccaa00; }

  /* Section titles */
  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 4px 0 12px;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 16px;
  }

  /* Indicator table */
  .ind-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #131820;
    font-size: 0.83rem;
  }
  .ind-key { color: #5a7090; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
  .ind-val { font-family: 'IBM Plex Mono', monospace; color: #c8d8e8; }
  .tag { 
    display:inline-block; padding:2px 8px; border-radius:4px;
    font-size:0.65rem; font-family:'IBM Plex Mono',monospace; font-weight:600;
  }
  .tag-bull { background:#001a0d; color:#00cc66; }
  .tag-bear { background:#1a0009; color:#ff4466; }
  .tag-neut { background:#0d1000; color:#ccaa00; }

  /* Accuracy badge */
  .accuracy-bar {
    background: #0d1117;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 10px;
  }

  /* Hide Streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  div[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ CONTROL PANEL")
    st.markdown("---")

    ticker = st.text_input("🔍 Ticker Symbol", value="NVDA",
                           help="US stock symbols: AAPL, TSLA, AMD, NVDA, MSFT...").upper().strip()

    period_map = {
        "3 Months": 90, "6 Months": 180,
        "1 Year": 365, "2 Years": 730
    }
    selected_period = st.selectbox("📅 Historical Period", list(period_map.keys()), index=1)
    days_back = period_map[selected_period]

    forecast_days = st.slider("🔮 Forecast Horizon (days)", min_value=3, max_value=30, value=7, step=1)

    st.markdown("---")
    st.markdown("**📐 Technical Indicators**")
    show_bb    = st.checkbox("Bollinger Bands", value=True)
    show_ema   = st.checkbox("EMA 20 / 50", value=True)
    show_vwap  = st.checkbox("VWAP", value=False)
    show_vol   = st.checkbox("Volume", value=True)

    st.markdown("---")
    st.markdown("**🤖 AI Model**")
    model_choice = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Ensemble (Both)"])
    n_estimators = st.slider("n_estimators", 50, 300, 100, 50)

    st.markdown("---")
    run_btn = st.button("⚡ ANALYZE", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("Data: Yahoo Finance · Model: Scikit-learn · Charts: Plotly")

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="god-header">
  <h1><span class="live-dot"></span>AI STOCK GOD MODE</h1>
  <div class="subtitle">Random Forest · Gradient Boosting · Technical Analysis · {ticker} // {datetime.now().strftime('%d %b %Y  %H:%M')}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_data(symbol: str, days: int) -> pd.DataFrame | None:
    end   = datetime.now()
    start = end - timedelta(days=days + 60)  # extra buffer for indicator warmup
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # ── Technical Indicators ──────────────────────────────────────────────
    df['RSI']   = ta.rsi(df['Close'], length=14)
    df['RSI_9'] = ta.rsi(df['Close'], length=9)

    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist']   = macd['MACDh_12_26_9']

    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Mid']   = bb['BBM_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200']= ta.ema(df['Close'], length=200)

    df['ATR']    = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ADX']    = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['STOCH_K'], df['STOCH_D'] = ta.stoch(df['High'], df['Low'], df['Close']).T.values[:2]

    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

    # Price momentum features
    for lag in [1, 3, 5, 10, 21]:
        df[f'Return_{lag}d'] = df['Close'].pct_change(lag)

    # Candle features
    df['Body_Size']  = (df['Close'] - df['Open']).abs() / df['Open']
    df['Upper_Wick'] = (df['High'] - df[['Open','Close']].max(axis=1)) / df['Open']
    df['Lower_Wick'] = (df[['Open','Close']].min(axis=1) - df['Low']) / df['Open']

    # Target: % return after N days (set later)
    df = df.tail(days + 10).copy()
    return df

# ─────────────────────────────────────────
# ML ENGINE
# ─────────────────────────────────────────
def build_and_predict(df: pd.DataFrame, forecast_days: int,
                      model_choice: str, n_est: int):
    df = df.copy()
    df['Target'] = df['Close'].shift(-forecast_days)

    FEATURES = [
        'RSI', 'RSI_9', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'EMA_20', 'EMA_50', 'ATR', 'ADX',
        'Vol_Ratio', 'Body_Size', 'Upper_Wick', 'Lower_Wick',
        'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d', 'Return_21d',
        'Close'
    ]

    ml_df = df.dropna(subset=FEATURES + ['Target']).copy()
    if len(ml_df) < 40:
        return None, None, None, None

    X = ml_df[FEATURES].values
    y = ml_df['Target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series aware split (no leakage)
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    models_to_run = []
    if model_choice == "Random Forest":
        models_to_run = [("RF", RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1))]
    elif model_choice == "Gradient Boosting":
        models_to_run = [("GB", GradientBoostingRegressor(n_estimators=n_est, random_state=42))]
    else:
        models_to_run = [
            ("RF", RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)),
            ("GB", GradientBoostingRegressor(n_estimators=n_est, random_state=42))
        ]

    preds_test = []
    preds_latest = []
    mape_scores = []
    fi_total = np.zeros(len(FEATURES))

    latest_X = scaler.transform(df[FEATURES].dropna().iloc[[-1]].values)

    for name, mdl in models_to_run:
        mdl.fit(X_train, y_train)
        p_test  = mdl.predict(X_test)
        p_lat   = mdl.predict(latest_X)[0]
        mape    = mean_absolute_percentage_error(y_test, p_test) * 100
        preds_test.append(p_test)
        preds_latest.append(p_lat)
        mape_scores.append(mape)
        if hasattr(mdl, 'feature_importances_'):
            fi_total += mdl.feature_importances_

    final_pred = float(np.mean(preds_latest))
    avg_mape   = float(np.mean(mape_scores))
    accuracy   = max(0, 100 - avg_mape)
    fi = fi_total / len(models_to_run)
    fi_df = pd.DataFrame({'Feature': FEATURES, 'Importance': fi}).sort_values('Importance', ascending=False)

    return final_pred, accuracy, fi_df, y_test, np.mean(preds_test, axis=0) if len(preds_test) > 1 else preds_test[0]

# ─────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────
with st.spinner(f"Loading {ticker} data..."):
    df = load_data(ticker, days_back)

if df is None or len(df) < 50:
    st.error("❌ ไม่พบข้อมูลหุ้น หรือข้อมูลน้อยเกินไป กรุณาตรวจสอบ Ticker Symbol")
    st.stop()

# ── Company Info ─────────────────────────
@st.cache_data(ttl=3600)
def get_info(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return {
            'name':    info.get('longName', symbol),
            'sector':  info.get('sector', '—'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio':   info.get('trailingPE', 0),
            'eps':        info.get('trailingEps', 0),
            'dividend':   info.get('dividendYield', 0),
            '52w_high':   info.get('fiftyTwoWeekHigh', 0),
            '52w_low':    info.get('fiftyTwoWeekLow', 0),
            'beta':       info.get('beta', 0),
        }
    except:
        return {'name': symbol, 'sector': '—'}

info = get_info(ticker)

# ── Run AI ─────────────────────────────
prediction, accuracy, fi_df, y_test, y_pred_test = build_and_predict(
    df, forecast_days, model_choice, n_estimators
)

current_price = float(df['Close'].iloc[-1])
prev_close    = float(df['Close'].iloc[-2])
price_chg     = (current_price - prev_close) / prev_close * 100

# ─────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────
rsi_val   = float(df['RSI'].iloc[-1])  if df['RSI'].notna().any()  else 0
macd_val  = float(df['MACD'].iloc[-1]) if df['MACD'].notna().any() else 0
adx_val   = float(df['ADX'].iloc[-1])  if df['ADX'].notna().any()  else 0
vol_ratio = float(df['Vol_Ratio'].iloc[-1]) if df['Vol_Ratio'].notna().any() else 1

pred_delta = ((prediction - current_price) / current_price * 100) if prediction else 0
delta_color = "#00cc66" if pred_delta >= 0 else "#ff4466"
price_color = "#00cc66" if price_chg  >= 0 else "#ff4466"

if prediction and pred_delta > 2 and rsi_val < 70:
    signal_cls  = "signal-buy"
    signal_text = "⬆ STRONG BUY"
elif prediction and pred_delta > 0.5:
    signal_cls  = "signal-buy"
    signal_text = "↗ BUY"
elif prediction and pred_delta < -2:
    signal_cls  = "signal-sell"
    signal_text = "⬇ STRONG SELL"
elif prediction and pred_delta < -0.5:
    signal_cls  = "signal-sell"
    signal_text = "↘ SELL / TAKE PROFIT"
else:
    signal_cls  = "signal-hold"
    signal_text = "→ HOLD / NEUTRAL"

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.markdown(f"""
<div class="metric-card" style="--accent:#00d4ff">
  <div class="metric-label">CURRENT PRICE</div>
  <div class="metric-value">${current_price:,.2f}</div>
  <div class="metric-delta" style="color:{price_color}">{price_chg:+.2f}% today</div>
</div>""", unsafe_allow_html=True)

mc2.markdown(f"""
<div class="metric-card" style="--accent:{delta_color}">
  <div class="metric-label">AI FORECAST (+{forecast_days}d)</div>
  <div class="metric-value">${prediction:,.2f}</div>
  <div class="metric-delta" style="color:{delta_color}">{pred_delta:+.2f}%</div>
</div>""", unsafe_allow_html=True)

mc3.markdown(f"""
<div class="metric-card" style="--accent:#8888ff">
  <div class="metric-label">MODEL ACCURACY</div>
  <div class="metric-value">{accuracy:.1f}%</div>
  <div class="metric-delta" style="color:#8888ff">{model_choice}</div>
</div>""", unsafe_allow_html=True)

mc4.markdown(f"""
<div class="metric-card" style="--accent:#ff9900">
  <div class="metric-label">RSI (14) / ADX</div>
  <div class="metric-value">{rsi_val:.1f}</div>
  <div class="metric-delta" style="color:#ff9900">ADX {adx_val:.1f}</div>
</div>""", unsafe_allow_html=True)

mc5.markdown(f"""
<div class="metric-card" style="--accent:{delta_color}">
  <div class="metric-label">AI SIGNAL</div>
  <div class="metric-value" style="font-size:1rem; padding-top:8px">
    <span class="signal-badge {signal_cls}">{signal_text}</span>
  </div>
  <div class="metric-delta" style="color:#4a6080; margin-top:10px">Confidence: {accuracy:.0f}%</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN CHART
# ─────────────────────────────────────────
n_rows = 3 if show_vol else 2
row_heights = [0.55, 0.25, 0.20] if show_vol else [0.65, 0.35]

subplot_titles = [f"{ticker} · {info.get('name','')}", "MACD", "Volume"] if show_vol \
                 else [f"{ticker} · {info.get('name','')}", "MACD"]

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    subplot_titles=subplot_titles,
    vertical_spacing=0.04
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'],  close=df['Close'],
    name='OHLC',
    increasing_line_color='#00cc66', decreasing_line_color='#ff4466',
    increasing_fillcolor='#00cc6622', decreasing_fillcolor='#ff446622'
), row=1, col=1)

# Bollinger Bands
if show_bb and 'BB_Upper' in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
        line=dict(color='#4a7fa5', width=1, dash='dot'), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
        line=dict(color='#4a7fa5', width=1, dash='dot'), opacity=0.7,
        fill='tonexty', fillcolor='rgba(74,127,165,0.06)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Mid',
        line=dict(color='#4a7fa5', width=1), opacity=0.4), row=1, col=1)

# EMAs
if show_ema:
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20',
        line=dict(color='#ffcc44', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50',
        line=dict(color='#ff8800', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='EMA 200',
        line=dict(color='#cc4488', width=1.5, dash='dot')), row=1, col=1)

# VWAP (approximate daily VWAP using cumulative)
if show_vwap:
    vwap = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    fig.add_trace(go.Scatter(x=df.index, y=vwap, name='VWAP',
        line=dict(color='#aa88ff', width=1.5, dash='dashdot')), row=1, col=1)

# AI Forecast line
if prediction:
    last_date   = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
    future_prices = np.linspace(current_price, prediction, forecast_days)

    # Confidence band (±ATR scaled)
    atr_now = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns and df['ATR'].notna().any() else current_price * 0.02
    upper_band = future_prices + atr_now * np.linspace(0.5, 2, forecast_days)
    lower_band = future_prices - atr_now * np.linspace(0.5, 2, forecast_days)

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper_band) + list(lower_band[::-1]),
        fill='toself', fillcolor='rgba(0,212,255,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence Band', showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[last_date] + list(future_dates),
        y=[current_price] + list(future_prices),
        name=f'AI Forecast ({forecast_days}d)',
        line=dict(color='#00d4ff', width=2.5, dash='dash'),
        mode='lines+markers',
        marker=dict(size=[0]*len(future_dates) + [8], color='#00d4ff')
    ), row=1, col=1)

    fig.add_annotation(
        x=future_dates[-1], y=prediction,
        text=f"  ${prediction:.2f}",
        showarrow=False, font=dict(color='#00d4ff', size=13, family='IBM Plex Mono'),
        xanchor='left', row=1, col=1
    )

# MACD
fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist',
    marker_color=np.where(df['MACD_Hist'] >= 0, '#00cc6688', '#ff446688')), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
    line=dict(color='#00d4ff', width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
    line=dict(color='#ff9900', width=1.5)), row=2, col=1)

# Volume
if show_vol:
    colors = ['#00cc6666' if df['Close'].iloc[i] >= df['Open'].iloc[i]
              else '#ff446666' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
        marker_color=colors, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Vol_MA20'], name='Vol MA20',
        line=dict(color='#8888ff', width=1.2)), row=3, col=1)

fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='#080c14',
    paper_bgcolor='#080c14',
    font=dict(family='IBM Plex Mono, monospace', color='#7a90a8', size=11),
    legend=dict(
        bgcolor='#0d1117', bordercolor='#1e2530', borderwidth=1,
        font=dict(size=10), orientation='h', y=1.02
    ),
    xaxis_rangeslider_visible=False,
    height=700,
    margin=dict(l=0, r=60, t=30, b=0),
    hovermode='x unified',
)

for i in range(1, n_rows + 1):
    fig.update_xaxes(gridcolor='#131820', showgrid=True, row=i, col=1)
    fig.update_yaxes(gridcolor='#131820', showgrid=True, row=i, col=1)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# LOWER SECTION: 3 columns
# ─────────────────────────────────────────
left, mid, right = st.columns([1.2, 1, 1])

# ── Left: Technical Indicators ─────────
with left:
    st.markdown('<div class="section-title">TECHNICAL INDICATORS</div>', unsafe_allow_html=True)

    def tag(val, bull_cond, bear_cond, bull_txt="Bullish", bear_txt="Bearish", neut_txt="Neutral"):
        if bull_cond: return f'<span class="tag tag-bull">▲ {bull_txt}</span>'
        if bear_cond: return f'<span class="tag tag-bear">▼ {bear_txt}</span>'
        return f'<span class="tag tag-neut">— {neut_txt}</span>'

    rsi_tag  = tag(rsi_val, rsi_val < 30, rsi_val > 70, "Oversold", "Overbought")
    macd_tag = tag(macd_val, macd_val > 0, macd_val < 0)
    adx_tag  = tag(adx_val, adx_val > 25, adx_val < 15, "Strong Trend", "Weak Trend")
    vol_tag  = tag(vol_ratio, vol_ratio > 1.5, vol_ratio < 0.7, "High Vol", "Low Vol")
    ema_cross = float(df['EMA_20'].iloc[-1]) > float(df['EMA_50'].iloc[-1]) if df['EMA_20'].notna().any() else False
    ema_tag  = tag(1, ema_cross, not ema_cross, "20>50 Bullish", "20<50 Bearish")
    bb_pos   = (current_price - float(df['BB_Lower'].iloc[-1])) / (float(df['BB_Upper'].iloc[-1]) - float(df['BB_Lower'].iloc[-1])) * 100 if df['BB_Lower'].notna().any() else 50

    indicators = [
        ("RSI (14)",         f"{rsi_val:.1f}",        rsi_tag),
        ("MACD",             f"{macd_val:.3f}",       macd_tag),
        ("ADX (14)",         f"{adx_val:.1f}",        adx_tag),
        ("Volume Ratio",     f"{vol_ratio:.2f}x",     vol_tag),
        ("EMA Cross",        f"20/{float(df['EMA_20'].iloc[-1]):.1f}" if df['EMA_20'].notna().any() else "—", ema_tag),
        ("BB Position",      f"{bb_pos:.0f}%",        tag(bb_pos, bb_pos < 20, bb_pos > 80, "Near Low BB","Near High BB")),
        ("ATR",              f"${atr_now:.2f}",       '<span class="tag tag-neut">— Volatility</span>'),
    ]
    for key, val, t in indicators:
        st.markdown(f"""
        <div class="ind-row">
          <span class="ind-key">{key}</span>
          <span>{t}&nbsp;<span class="ind-val">{val}</span></span>
        </div>""", unsafe_allow_html=True)

# ── Mid: AI Model Info ─────────────────
with mid:
    st.markdown('<div class="section-title">AI MODEL PERFORMANCE</div>', unsafe_allow_html=True)

    # Model accuracy gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy,
        number={'suffix': '%', 'font': {'color': '#00d4ff', 'size': 36, 'family': 'IBM Plex Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#4a6080', 'tickfont': {'size': 10}},
            'bar':  {'color': '#00d4ff', 'thickness': 0.22},
            'bgcolor': '#0d1117',
            'bordercolor': '#1e2530',
            'steps': [
                {'range': [0, 60],   'color': '#ff446622'},
                {'range': [60, 80],  'color': '#ffcc4422'},
                {'range': [80, 100], 'color': '#00cc6622'},
            ],
            'threshold': {'line': {'color': '#00ff88', 'width': 2}, 'value': accuracy}
        }
    ))
    gauge.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#7a90a8', family='IBM Plex Mono'),
        height=200, margin=dict(l=20, r=20, t=20, b=0)
    )
    st.plotly_chart(gauge, use_container_width=True)

    # Feature importance mini chart
    if fi_df is not None:
        st.markdown('<div style="font-family:IBM Plex Mono;font-size:0.65rem;color:#4a6080;letter-spacing:0.1em;margin-bottom:8px">TOP FEATURES</div>', unsafe_allow_html=True)
        top_fi = fi_df.head(7)
        fi_fig = go.Figure(go.Bar(
            x=top_fi['Importance'], y=top_fi['Feature'],
            orientation='h',
            marker=dict(
                color=top_fi['Importance'],
                colorscale=[[0,'#1a2d4a'],[1,'#00d4ff']],
                showscale=False
            )
        ))
        fi_fig.update_layout(
            paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            font=dict(family='IBM Plex Mono', color='#7a90a8', size=10),
            height=220, margin=dict(l=0, r=10, t=0, b=0),
            xaxis=dict(gridcolor='#131820'),
            yaxis=dict(gridcolor='#131820')
        )
        st.plotly_chart(fi_fig, use_container_width=True)

# ── Right: Fundamentals ───────────────
with right:
    st.markdown('<div class="section-title">COMPANY FUNDAMENTALS</div>', unsafe_allow_html=True)

    mc = info.get('market_cap', 0)
    mc_str = f"${mc/1e12:.2f}T" if mc > 1e12 else f"${mc/1e9:.2f}B" if mc > 1e9 else f"${mc/1e6:.2f}M" if mc else "—"

    fundamentals = [
        ("Company",       info.get('name', ticker)),
        ("Sector",        info.get('sector', '—')),
        ("Market Cap",    mc_str),
        ("P/E Ratio",     f"{info.get('pe_ratio',0):.2f}" if info.get('pe_ratio') else "—"),
        ("EPS (TTM)",     f"${info.get('eps',0):.2f}" if info.get('eps') else "—"),
        ("Dividend Yield",f"{info.get('dividend',0)*100:.2f}%" if info.get('dividend') else "0.00%"),
        ("Beta",          f"{info.get('beta',0):.2f}" if info.get('beta') else "—"),
        ("52W High",      f"${info.get('52w_high',0):.2f}" if info.get('52w_high') else "—"),
        ("52W Low",       f"${info.get('52w_low',0):.2f}" if info.get('52w_low') else "—"),
    ]
    for key, val in fundamentals:
        st.markdown(f"""
        <div class="ind-row">
          <span class="ind-key">{key}</span>
          <span class="ind-val">{val}</span>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# RSI CHART + Prediction Backtest
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📉 RSI + Stochastic", "🔬 Model Backtest (Out-of-Sample)"])

with tab1:
    rsi_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.5], vertical_spacing=0.06,
                            subplot_titles=["RSI (9 / 14)", "Stochastic (14,3)"])
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI 14',
        line=dict(color='#ff9900', width=1.8)), row=1, col=1)
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI_9'], name='RSI 9',
        line=dict(color='#ffcc44', width=1.2, dash='dot')), row=1, col=1)
    rsi_fig.add_hline(y=70, line_dash='dash', line_color='#ff446688', row=1, col=1)
    rsi_fig.add_hline(y=30, line_dash='dash', line_color='#00cc6688', row=1, col=1)
    rsi_fig.add_hrect(y0=30, y1=70, fillcolor='#ffffff08', line_width=0, row=1, col=1)

    if 'STOCH_K' in df.columns:
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_K'], name='Stoch %K',
            line=dict(color='#00d4ff', width=1.5)), row=2, col=1)
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_D'], name='Stoch %D',
            line=dict(color='#aa88ff', width=1.2, dash='dot')), row=2, col=1)
        rsi_fig.add_hline(y=80, line_dash='dash', line_color='#ff446666', row=2, col=1)
        rsi_fig.add_hline(y=20, line_dash='dash', line_color='#00cc6666', row=2, col=1)

    rsi_fig.update_layout(template='plotly_dark', paper_bgcolor='#080c14',
        plot_bgcolor='#080c14', height=380, margin=dict(l=0,r=0,t=30,b=0),
        font=dict(family='IBM Plex Mono', color='#7a90a8', size=11))
    st.plotly_chart(rsi_fig, use_container_width=True)

with tab2:
    if y_test is not None and y_pred_test is not None:
        bt_fig = go.Figure()
        bt_fig.add_trace(go.Scatter(y=y_test, name='Actual Price',
            line=dict(color='#00cc66', width=2)))
        bt_fig.add_trace(go.Scatter(y=y_pred_test, name='Model Prediction',
            line=dict(color='#00d4ff', width=2, dash='dash')))
        bt_fig.update_layout(
            template='plotly_dark', paper_bgcolor='#080c14', plot_bgcolor='#080c14',
            height=350, margin=dict(l=0,r=0,t=10,b=0),
            font=dict(family='IBM Plex Mono', color='#7a90a8', size=11),
            xaxis_title="Test Window (days)", yaxis_title="Price ($)",
            legend=dict(bgcolor='#0d1117', bordercolor='#1e2530', borderwidth=1)
        )
        st.plotly_chart(bt_fig, use_container_width=True)
        st.markdown(f"""
        <div class="accuracy-bar">
          <span class="ind-key">OUT-OF-SAMPLE MAPE</span>&nbsp;
          <span style="font-family:'IBM Plex Mono';color:#00d4ff">{100-accuracy:.2f}%</span>&nbsp;&nbsp;
          <span class="ind-key">ACCURACY SCORE</span>&nbsp;
          <span style="font-family:'IBM Plex Mono';color:#00cc66">{accuracy:.1f}%</span>&nbsp;&nbsp;
          <span class="ind-key">TEST SAMPLES</span>&nbsp;
          <span style="font-family:'IBM Plex Mono';color:#c8d0dc">{len(y_test)}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Not enough data for backtest visualization.")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#2a3a4a;padding:16px 0;border-top:1px solid #131820">
  ⚠️ FOR INFORMATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE · AI predictions carry inherent uncertainty
  <br>Data: Yahoo Finance · Models: Scikit-learn Random Forest & Gradient Boosting · Charts: Plotly
</div>
""", unsafe_allow_html=True)
