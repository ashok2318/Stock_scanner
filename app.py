import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="MarketPulse Pro", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

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

st.title("MarketPulse Pro")
st.caption("Technical momentum + fundamental KPI dashboard • Local on your Mac")

@st.cache_data(ttl=300)
def get_hist(ticker, period="1y"):
    return yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)

@st.cache_data(ttl=300)
def get_info(ticker):
    try: return yf.Ticker(ticker).info
    except: return {}

def calc_rsi(close, n=14):
    d=close.diff()
    gain=d.clip(lower=0); loss=-d.clip(upper=0)
    ag=gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al=loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-(100/(1+rs))

def money_compact(v):
    if v is None or pd.isna(v): return "N/A"
    v=float(v)
    if abs(v)>=1e12: return f"${v/1e12:.2f}T"
    if abs(v)>=1e9: return f"${v/1e9:.2f}B"
    if abs(v)>=1e6: return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def pct(v, digits=1):
    if v is None or pd.isna(v): return "N/A"
    return f"{float(v)*100:.{digits}f}%"

def rsi_label(v):
    if pd.isna(v): return "N/A"
    if v>=70: return "Overbought zone"
    if v<=30: return "Oversold zone"
    if v>=50: return "Positive momentum"
    return "Weak momentum"

def pos_label(v):
    if pd.isna(v): return "N/A"
    p=v*100
    if p>=80: return "Near 52W high"
    if p>=60: return "Upper range"
    if p>=40: return "Mid range"
    if p>=20: return "Lower range"
    return "Near 52W low"

def pivots(series, left=3, right=3, mode="low"):
    a=series.values
    idx=[]
    for i in range(left,len(a)-right):
        window=a[i-left:i+right+1]
        if np.isnan(a[i]): continue
        if mode=="low" and a[i]==np.nanmin(window): idx.append(i)
        if mode=="high" and a[i]==np.nanmax(window): idx.append(i)
    return idx

def divergence(df, lookback=120):
    x=df.tail(lookback).copy().dropna(subset=["RSI"])
    if len(x)<20: return "None", None
    lows=pivots(x["Close"],3,3,"low")
    highs=pivots(x["Close"],3,3,"high")
    events=[]
    if len(lows)>=2:
        a,b=lows[-2],lows[-1]
        if x["Close"].iloc[b] < x["Close"].iloc[a] and x["RSI"].iloc[b] > x["RSI"].iloc[a]:
            events.append(("Bullish",x.index[b]))
    if len(highs)>=2:
        a,b=highs[-2],highs[-1]
        if x["Close"].iloc[b] > x["Close"].iloc[a] and x["RSI"].iloc[b] < x["RSI"].iloc[a]:
            events.append(("Bearish",x.index[b]))
    if not events: return "None detected", None
    return sorted(events,key=lambda z:z[1])[-1]

def snapshot(t):
    h=get_hist(t,"1y"); inf=get_info(t)
    if h.empty:return None
    h["RSI"]=calc_rsi(h.Close,14)
    p=float(h.Close.iloc[-1]); prev=float(h.Close.iloc[-2])
    lo=float(h.Low.tail(252).min()); hi=float(h.High.tail(252).max())
    pos=(p-lo)/(hi-lo) if hi!=lo else np.nan
    div,_=divergence(h)
    return {
        "Ticker":t,"Price":p,"Day %":p/prev-1,"RSI":h.RSI.iloc[-1],"RSI Status":rsi_label(h.RSI.iloc[-1]),
        "Divergence":div,"Market Cap":money_compact(inf.get("marketCap")),
        "P/E":inf.get("trailingPE"),"Forward P/E":inf.get("forwardPE"),"EPS":inf.get("trailingEps"),
        "EPS Growth %": None if inf.get("earningsQuarterlyGrowth") is None else inf.get("earningsQuarterlyGrowth")*100,
        "Revenue Growth %": None if inf.get("revenueGrowth") is None else inf.get("revenueGrowth")*100,
        "52W Position %":pos*100 if pd.notna(pos) else np.nan,"52W Status":pos_label(pos),"Beta":inf.get("beta")
    }

st.sidebar.markdown("### Watchlist")
raw=st.sidebar.text_input("Tickers", "WDC,MU,STX,AMD,NVDA")
watch=[x.strip().upper() for x in raw.split(",") if x.strip()]
ticker=st.sidebar.selectbox("Focus stock", watch if watch else ["WDC"])
st.sidebar.markdown("### Chart settings")
period=st.sidebar.selectbox("History",["6mo","1y","2y","5y"],1)
rn=st.sidebar.slider("RSI period",5,30,14)
ma1=st.sidebar.slider("Fast MA",5,100,20)
ma2=st.sidebar.slider("Slow MA",20,250,50)
st.sidebar.caption("Market data cache refresh: 5 minutes")

rows=[]
for t in watch:
    try:
        r=snapshot(t)
        if r:rows.append(r)
    except Exception as e: st.sidebar.warning(f"{t}: data unavailable")
df=pd.DataFrame(rows)

st.subheader("Portfolio Monitor")
if not df.empty:
    st.dataframe(
        df.style.format({
            "Price":"${:,.2f}","Day %":"{:+.2%}","RSI":"{:.1f}","P/E":"{:.1f}","Forward P/E":"{:.1f}",
            "EPS":"${:.2f}","EPS Growth %":"{:+.1f}%","Revenue Growth %":"{:+.1f}%",
            "52W Position %":"{:.1f}%","Beta":"{:.2f}"
        },na_rep="—"),
        use_container_width=True, hide_index=True, height=min(420,38*len(df)+40)
    )
    st.download_button("Export watchlist CSV",df.to_csv(index=False).encode(),"marketpulse_watchlist.csv","text/csv")

h=get_hist(ticker,period); inf=get_info(ticker)
if h.empty: st.error("No price history returned."); st.stop()
h["RSI"]=calc_rsi(h.Close,rn)
h[f"MA{ma1}"]=h.Close.rolling(ma1).mean(); h[f"MA{ma2}"]=h.Close.rolling(ma2).mean()
price=float(h.Close.iloc[-1]); prev=float(h.Close.iloc[-2]); rv=h.RSI.iloc[-1]
lo=float(h.Low.tail(252).min()); hi=float(h.High.tail(252).max()); pos=(price-lo)/(hi-lo) if hi!=lo else np.nan
div,divdate=divergence(h)
epsg=inf.get("earningsQuarterlyGrowth"); revg=inf.get("revenueGrowth")

st.divider()
st.subheader(f"{inf.get('shortName',ticker)} ({ticker})")
a,b,c,d,e,f=st.columns(6)
a.metric("Price",f"${price:,.2f}",f"{price/prev-1:+.2%}")
b.metric("RSI",f"{rv:.1f}",rsi_label(rv))
c.metric("RSI Divergence",div,divdate.strftime("%b %d, %Y") if divdate is not None else None)
d.metric("Market Cap",money_compact(inf.get("marketCap")))
e.metric("EPS Growth",pct(epsg), "quarterly YoY" if epsg is not None else None)
f.metric("52W Position",f"{pos*100:.1f}%" if pd.notna(pos) else "N/A",pos_label(pos))

tabs=st.tabs(["Technical Chart","Fundamentals","Metric Guide"])
with tabs[0]:
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.05,row_heights=[.68,.32])
    fig.add_trace(go.Candlestick(x=h.index,open=h.Open,high=h.High,low=h.Low,close=h.Close,name="Price"),row=1,col=1)
    fig.add_trace(go.Scatter(x=h.index,y=h[f"MA{ma1}"],name=f"MA {ma1}",line=dict(width=1.6)),row=1,col=1)
    fig.add_trace(go.Scatter(x=h.index,y=h[f"MA{ma2}"],name=f"MA {ma2}",line=dict(width=1.6)),row=1,col=1)
    fig.add_trace(go.Scatter(x=h.index,y=h.RSI,name=f"RSI {rn}",line=dict(width=2)),row=2,col=1)
    for level,dash in [(70,"dash"),(50,"dot"),(30,"dash")]:
        fig.add_hline(y=level,line_dash=dash,line_width=1,row=2,col=1)
    fig.update_yaxes(range=[0,100],title="RSI",row=2,col=1)
    fig.update_layout(height=720,xaxis_rangeslider_visible=False,legend_orientation="h",legend_y=1.03,
                      margin=dict(l=20,r=20,t=35,b=20),hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    if div=="Bullish": st.success("Bullish RSI divergence detected: price made a lower swing low while RSI made a higher swing low. This can indicate weakening downside momentum; it is not a guaranteed reversal.")
    elif div=="Bearish": st.warning("Bearish RSI divergence detected: price made a higher swing high while RSI made a lower swing high. This can indicate weakening upside momentum; it is not a guaranteed reversal.")
    else: st.info("No recent classic RSI divergence detected using the dashboard's swing-point method.")

with tabs[1]:
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Trailing P/E",f"{inf.get('trailingPE'):.1f}" if isinstance(inf.get("trailingPE"),(int,float)) else "N/A")
    k2.metric("Forward P/E",f"{inf.get('forwardPE'):.1f}" if isinstance(inf.get("forwardPE"),(int,float)) else "N/A")
    k3.metric("EPS",f"${inf.get('trailingEps'):.2f}" if isinstance(inf.get("trailingEps"),(int,float)) else "N/A")
    k4.metric("Revenue Growth",pct(revg))
    k5,k6,k7,k8=st.columns(4)
    k5.metric("52W Low",f"${lo:,.2f}"); k6.metric("52W High",f"${hi:,.2f}")
    k7.metric("52W Position",f"{pos*100:.1f}%",pos_label(pos)); k8.metric("Beta",str(round(inf.get("beta"),2)) if isinstance(inf.get("beta"),(int,float)) else "N/A")

with tabs[2]:
    st.markdown("""
**EPS Growth** — displayed as a percentage. For example, `0.438` from the data source is shown as **43.8%**. In this dashboard it is labeled as quarterly year-over-year earnings growth.

**Revenue Growth** — also displayed as a percentage. A raw value of `0.438` is displayed as **43.8%**, meaning reported revenue growth is approximately 43.8% for the source's comparison period.

**52W Position** — shows where today's price sits between the 52-week low and high. `0%` = at the 52-week low, `50%` = halfway between the low and high, and `100%` = at the 52-week high. So a raw value of `0.506` becomes **50.6%**, roughly the middle of its 52-week range.

**RSI Divergence** — bullish divergence occurs when price makes a lower swing low but RSI makes a higher swing low; bearish divergence is the reverse pattern at swing highs. Divergence is a momentum warning, not a guaranteed buy/sell signal.

**Market Cap** — automatically formatted as `$850.0M`, `$42.5B`, or `$1.20T` instead of a long integer.
""")

st.caption("For research/education. Technical indicators and source fundamentals can be delayed or revised; verify important figures before making investment decisions.")
