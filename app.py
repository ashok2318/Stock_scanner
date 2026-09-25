"""MarketPulse Pro: a local Streamlit stock research dashboard.

Fetch Yahoo Finance prices and fundamentals, calculate RSI and swing-point
divergence, and display a watchlist, focused charts, and company KPI cards.
The script runs from top to bottom whenever Streamlit reruns the page.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------------------------------------------------------
# Page configuration and visual styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title='MarketPulse Pro',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); border: 1px solid rgba(120,120,120,.18); padding: 14px; border-radius: 12px;}
[data-testid="stMetricLabel"] {font-size:.82rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(120,120,120,.15);}
h1 {letter-spacing:-.03em;} h2,h3 {letter-spacing:-.02em;}
.small-note {opacity:.72; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)
st.title('MarketPulse Pro')
st.caption('Technical momentum + fundamental KPI dashboard • Local on your Mac')


# ----------------------------------------------------------------------------
# Cached market data
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_hist(ticker, period='1y'):
    """Fetch daily, unadjusted price history; cache results for five minutes."""
    return yf.Ticker(ticker).history(period=period, interval='1d', auto_adjust=False)

@st.cache_data(ttl=300)
def get_info(ticker):
    """Fetch cached company fundamentals, returning an empty mapping on failure."""
    try:
        return yf.Ticker(ticker).info
    except:
        return {}


# ----------------------------------------------------------------------------
# Technical indicator calculation
# ----------------------------------------------------------------------------
def calc_rsi(close, n=14):
    """Calculate RSI using exponentially smoothed gains and losses.

    Require n price changes before producing a value. Zero average losses
    yield RSI 100 for gains, or neutral RSI 50 for a flat series.
    """
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((al == 0) & (ag > 0), 100)
    return rsi.mask((al == 0) & (ag == 0), 50)


# ----------------------------------------------------------------------------
# Display formatting and metric labels
# ----------------------------------------------------------------------------
def money_compact(v):
    """Format a dollar amount with M/B/T suffixes, or N/A when missing."""
    if v is None or pd.isna(v):
        return 'N/A'
    v = float(v)
    if abs(v) >= 1e12:
        return f'${v / 1e12:.2f}T'
    if abs(v) >= 1e9:
        return f'${v / 1e9:.2f}B'
    if abs(v) >= 1e6:
        return f'${v / 1e6:.1f}M'
    return f'${v:,.0f}'

def pct(v, digits=1):
    """Format a fractional rate as a percentage, or N/A when missing."""
    if v is None or pd.isna(v):
        return 'N/A'
    return f'{float(v) * 100:.{digits}f}%'

def rsi_label(v):
    """Describe RSI using the 30, 50, and 70 momentum thresholds."""
    if pd.isna(v):
        return 'N/A'
    if v >= 70:
        return 'Overbought zone'
    if v <= 30:
        return 'Oversold zone'
    if v >= 50:
        return 'Positive momentum'
    return 'Weak momentum'

def pos_label(v):
    """Describe a fractional position within the observed price range."""
    if pd.isna(v):
        return 'N/A'
    p = v * 100
    if p >= 80:
        return 'Near 52W high'
    if p >= 60:
        return 'Upper range'
    if p >= 40:
        return 'Mid range'
    if p >= 20:
        return 'Lower range'
    return 'Near 52W low'


# ----------------------------------------------------------------------------
# Swing points and RSI divergence
# ----------------------------------------------------------------------------
def calc_macd(close):
    """Standard MACD (12, 26, 9), with a warm-up before values are shown."""
    line = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(
        span=26, adjust=False, min_periods=26).mean()
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    return pd.DataFrame({'MACD': line, 'Signal': signal, 'Histogram': line - signal})


def rsi_style(label):
    colors = {
        'Positive momentum': ('#dcfce7', '#166534'),
        'Weak momentum': ('#fee2e2', '#991b1b'),
        'Overbought zone': ('#ffedd5', '#9a3412'),
        'Oversold zone': ('#dbeafe', '#1e40af'),
    }
    bg, fg = colors.get(label, ('#e5e7eb', '#374151'))
    return f'background-color: {bg}; color: {fg}; font-weight: 600;'


def pivots(series, left=3, right=3, mode='low'):
    """Confirmed pivots; retain only the last point of a tied plateau."""
    a = series.to_numpy()
    result = []
    for i in range(left, len(a) - right):
        before, after = a[i-left:i], a[i+1:i+right+1]
        if not np.isfinite(a[i-left:i+right+1]).all():
            continue
        if mode == 'low' and a[i] <= before.min() and a[i] < after.min():
            result.append(i)
        elif mode == 'high' and a[i] >= before.max() and a[i] > after.max():
            result.append(i)
    return result


def divergence_events(df, lookback=120, max_age=30):
    """Compare consecutive confirmed swings; age signals from confirmation.

    Require swings 5–60 bars apart and at least 2 RSI points of divergence.
    Use price lows/highs and RSI at those same bars, preserving bar positions.
    """
    x = df.tail(lookback)
    events = []
    for column, mode, label in [('Low', 'low', 'Bullish'), ('High', 'high', 'Bearish')]:
        points = pivots(x[column], mode=mode)
        for a, b in zip(points, points[1:]):
            if not 5 <= b - a <= 60 or len(x) - 1 - (b + 3) > max_age:
                continue
            ra, rb = x.RSI.iloc[a], x.RSI.iloc[b]
            if pd.isna(ra) or pd.isna(rb):
                continue
            pa, pb = x[column].iloc[a], x[column].iloc[b]
            matches = (pb < pa and rb - ra >= 2) if mode == 'low' else (pb > pa and ra - rb >= 2)
            if matches:
                events.append(dict(label=label, start=x.index[a], end=x.index[b],
                                   confirmed=x.index[b+3], column=column))
    return sorted(events, key=lambda event: event['confirmed'])


def divergence(df, lookback=120):
    events = divergence_events(df, lookback)
    if not events:
        return ('None detected', None)
    latest = events[-1]
    return latest['label'], latest['confirmed']


# ----------------------------------------------------------------------------
# Watchlist summary data
# ----------------------------------------------------------------------------
def snapshot(t):
    """Build one watchlist row from one year of prices and company fundamentals.

    Watchlist RSI always uses 14 periods, independently of chart settings.
    Return None when no price history is available.
    """
    h = get_hist(t, '1y')
    inf = get_info(t)
    if h.empty:
        return None
    h['RSI'] = calc_rsi(h.Close, 14)
    p = float(h.Close.iloc[-1])
    prev = float(h.Close.iloc[-2]) if len(h) > 1 else np.nan
    # Approximate a trading year with up to 252 available daily observations.
    lo = float(h.Low.tail(252).min())
    hi = float(h.High.tail(252).max())
    pos = (p - lo) / (hi - lo) if hi != lo else np.nan
    (div, _) = divergence(h)
    return {
        'Ticker': t,
        'Price': p,
        'Day %': p / prev - 1,
        'RSI': h.RSI.iloc[-1],
        'RSI Status': rsi_label(h.RSI.iloc[-1]),
        'Divergence': div,
        'Market Cap': money_compact(inf.get('marketCap')),
        'P/E': inf.get('trailingPE'),
        'Forward P/E': inf.get('forwardPE'),
        'EPS': inf.get('trailingEps'),
        'EPS Growth %': None if inf.get('earningsQuarterlyGrowth') is None else inf.get('earningsQuarterlyGrowth') * 100,
        'Revenue Growth %': None if inf.get('revenueGrowth') is None else inf.get('revenueGrowth') * 100,
        '52W Position %': pos * 100 if pd.notna(pos) else np.nan,
        '52W Status': pos_label(pos),
        'Beta': inf.get('beta')
    }


# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
st.sidebar.markdown('### Watchlist')
raw = st.sidebar.text_input('Tickers', 'WDC,MU,STX,AMD,NVDA')
watch = [x.strip().upper() for x in raw.split(',') if x.strip()]
ticker = st.sidebar.selectbox('Focus stock', watch if watch else ['WDC'])
st.sidebar.markdown('### Chart settings')
period = st.sidebar.selectbox('History', ['6mo', '1y', '2y', '5y'], 1)
rn = st.sidebar.slider('RSI period', 5, 30, 14)
ma1 = st.sidebar.slider('Fast MA', 5, 100, 20)
ma2 = st.sidebar.slider('Slow MA', 20, 250, 50)
st.sidebar.caption('Market data cache refresh: 5 minutes')


# ----------------------------------------------------------------------------
# Portfolio monitor and CSV export
# ----------------------------------------------------------------------------
rows = []
for t in watch:
    try:
        r = snapshot(t)
        if r:
            rows.append(r)
    except Exception as e:
        # Keep the remaining watchlist visible if one ticker fails.
        st.sidebar.warning(f'{t}: data unavailable')
df = pd.DataFrame(rows)
st.subheader('Portfolio Monitor')
if not df.empty:
    st.dataframe(
        df.style.format(
            {
                'Price': '${:,.2f}',
                'Day %': '{:+.2%}',
                'RSI': '{:.1f}',
                'P/E': '{:.1f}',
                'Forward P/E': '{:.1f}',
                'EPS': '${:.2f}',
                'EPS Growth %': '{:+.1f}%',
                'Revenue Growth %': '{:+.1f}%',
                '52W Position %': '{:.1f}%',
                'Beta': '{:.2f}',
            },
            na_rep='—',
        ).apply(lambda column: column.map(rsi_style), subset=['RSI Status']),
        use_container_width=True,
        hide_index=True,
        height=min(420, 38 * len(df) + 40)
    )
    st.download_button(
        'Export watchlist CSV',
        df.to_csv(index=False).encode(),
        'marketpulse_watchlist.csv',
        'text/csv'
    )


# ----------------------------------------------------------------------------
# Focused stock: prices, indicators, and fundamentals
# ----------------------------------------------------------------------------
h = get_hist(ticker, period)
inf = get_info(ticker)
if h.empty:
    st.error('No price history returned.')
    st.stop()
h['RSI'] = calc_rsi(h.Close, rn)
h[['MACD', 'Signal', 'Histogram']] = calc_macd(h.Close)
h[f'MA{ma1}'] = h.Close.rolling(ma1).mean()
h[f'MA{ma2}'] = h.Close.rolling(ma2).mean()
price = float(h.Close.iloc[-1])
prev = float(h.Close.iloc[-2]) if len(h) > 1 else np.nan
rv = h.RSI.iloc[-1]
# Shorter selected histories use the available range, which may be under a year.
lo = float(h.Low.tail(252).min())
hi = float(h.High.tail(252).max())
pos = (price - lo) / (hi - lo) if hi != lo else np.nan
(div, divdate) = divergence(h)
epsg = inf.get('earningsQuarterlyGrowth')
revg = inf.get('revenueGrowth')


# ----------------------------------------------------------------------------
# Focused stock: headline KPI cards
# ----------------------------------------------------------------------------
st.divider()
st.subheader(f"{inf.get('shortName', ticker)} ({ticker})")
a, b, c = st.columns(3)
a.metric('Price', f'${price:,.2f}', f'{price / prev - 1:+.2%}' if pd.notna(prev) else None)
b.metric(f'RSI ({rn})', f'{rv:.1f}' if pd.notna(rv) else 'N/A')
label = rsi_label(rv)
b.markdown(f'<span style="{rsi_style(label)} padding: 4px 10px; border-radius: 6px;">{label}</span>', unsafe_allow_html=True)
c.metric('RSI Divergence', div)
c.caption('Confirmed ' + divdate.strftime('%b %d, %Y') if divdate is not None else 'No confirmed signal in the last 30 bars')
d, e, f = st.columns(3)
d.metric('Market Cap', money_compact(inf.get('marketCap')))
e.metric('EPS Growth', pct(epsg))
e.caption('Quarterly year-over-year')
f.metric('52W Position', f'{pos * 100:.1f}%' if pd.notna(pos) else 'N/A')
f.caption(pos_label(pos))


# ----------------------------------------------------------------------------
# Detail tabs
# ----------------------------------------------------------------------------
tabs = st.tabs(['Technical Chart', 'Fundamentals', 'Metric Guide'])


# ----------------------------------------------------------------------------
# Technical chart: price and moving averages above RSI
# ----------------------------------------------------------------------------
with tabs[0]:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.23, 0.22]
    )
    fig.add_trace(
        go.Candlestick(x=h.index, open=h.Open, high=h.High, low=h.Low, close=h.Close, name='Price'),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(x=h.index, y=h[f'MA{ma1}'], name=f'MA {ma1}', line=dict(width=1.6)),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(x=h.index, y=h[f'MA{ma2}'], name=f'MA {ma2}', line=dict(width=1.6)),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(x=h.index, y=h.RSI, name=f'RSI {rn}', line=dict(width=2)),
        row=2,
        col=1
    )
    # Reference levels mark overbought, midpoint, and oversold RSI values.
    for (level, dash) in [(70, 'dash'), (50, 'dot'), (30, 'dash')]:
        fig.add_hline(y=level, line_dash=dash, line_width=1, row=2, col=1)
    fig.update_yaxes(range=[0, 100], title='RSI', row=2, col=1)
    for event in divergence_events(h):
        dates = [event['start'], event['end']]
        color = '#16a34a' if event['label'] == 'Bullish' else '#dc2626'
        for row, column in [(1, event['column']), (2, 'RSI')]:
            fig.add_trace(go.Scatter(
                x=dates, y=h.loc[dates, column], mode='lines+markers',
                name=event['label'] + ' divergence', showlegend=False,
                line=dict(color=color, width=3),
            ), row=row, col=1)
    fig.add_trace(go.Bar(x=h.index, y=h.Histogram, name='MACD histogram',
                        marker_color=['#16a34a' if v >= 0 else '#dc2626' for v in h.Histogram]), row=3, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h.MACD, name='MACD (12, 26)',
                            line=dict(color='#3b82f6', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=h.index, y=h.Signal, name='Signal (9)',
                            line=dict(color='#f59e0b', width=2)), row=3, col=1)
    fig.add_hline(y=0, line_width=1, row=3, col=1)
    fig.update_yaxes(title='MACD', row=3, col=1)
    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False,
        legend_orientation='h',
        legend_y=1.03,
        margin=dict(l=20, r=20, t=35, b=20),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Divergence lines connect price swings and matching RSI values. Signals confirm 3 trading bars after the swing; only confirmations within the last 30 bars are shown. Watchlist RSI uses 14 periods.')
    if div == 'Bullish':
        st.success(
            'Bullish RSI divergence detected: price made a lower swing low while RSI made a higher swing low. This can indicate weakening downside momentum; it is not a guaranteed reversal.'
        )
    elif div == 'Bearish':
        st.warning(
            'Bearish RSI divergence detected: price made a higher swing high while RSI made a lower swing high. This can indicate weakening upside momentum; it is not a guaranteed reversal.'
        )
    else:
        st.info(
            "No recent classic RSI divergence detected using the dashboard's swing-point method."
        )


# ----------------------------------------------------------------------------
# Fundamentals: valuation, growth, and price range
# ----------------------------------------------------------------------------
with tabs[1]:
    (k1, k2, k3, k4) = st.columns(4)
    k1.metric(
        'Trailing P/E',
        f"{inf.get('trailingPE'):.1f}" if isinstance(inf.get('trailingPE'), (int, float)) else 'N/A'
    )
    k2.metric(
        'Forward P/E',
        f"{inf.get('forwardPE'):.1f}" if isinstance(inf.get('forwardPE'), (int, float)) else 'N/A'
    )
    k3.metric(
        'EPS',
        f"${inf.get('trailingEps'):.2f}" if isinstance(inf.get('trailingEps'), (int, float)) else 'N/A'
    )
    k4.metric('Revenue Growth', pct(revg))
    (k5, k6, k7, k8) = st.columns(4)
    k5.metric('52W Low', f'${lo:,.2f}')
    k6.metric('52W High', f'${hi:,.2f}')
    k7.metric('52W Position', f'{pos * 100:.1f}%', pos_label(pos))
    k8.metric(
        'Beta',
        str(round(inf.get('beta'), 2)) if isinstance(inf.get('beta'), (int, float)) else 'N/A'
    )


# ----------------------------------------------------------------------------
# Metric guide
# ----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("""
**EPS Growth** — displayed as a percentage. For example, `0.438` from the data source is shown as **43.8%**. In this dashboard it is labeled as quarterly year-over-year earnings growth.

**Revenue Growth** — also displayed as a percentage. A raw value of `0.438` is displayed as **43.8%**, meaning reported revenue growth is approximately 43.8% for the source's comparison period.

**52W Position** — shows where today's price sits between the 52-week low and high. `0%` = at the 52-week low, `50%` = halfway between the low and high, and `100%` = at the 52-week high. So a raw value of `0.506` becomes **50.6%**, roughly the middle of its 52-week range.

**RSI colors** — green: positive momentum (50–below 70); red: weak momentum (above 30–below 50); orange: overbought (70+); blue: oversold (30 or below); gray: unavailable.

**MACD (12, 26, 9)** — the difference between the 12- and 26-period exponential moving averages. The signal is its 9-period exponential average; green/red histogram bars show MACD above/below the signal. Initial values remain blank during warm-up.

**Divergence filtering** — compares consecutive confirmed price lows/highs in the last 120 bars, 5–60 bars apart, with at least 2 RSI points of difference. Three bars on either side confirm each swing. Signals expire after 30 bars from confirmation. These filters reduce small fluctuations but cannot eliminate false signals.

**RSI Divergence** — bullish divergence occurs when price makes a lower swing low but RSI makes a higher swing low; bearish divergence is the reverse pattern at swing highs. Divergence is a momentum warning, not a guaranteed buy/sell signal.

**Market Cap** — automatically formatted as `$850.0M`, `$42.5B`, or `$1.20T` instead of a long integer.
""")
st.caption(
    'For research/education. Technical indicators and source fundamentals can be delayed or revised; verify important figures before making investment decisions.'
)
