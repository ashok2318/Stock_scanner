"""Retrospective percentage-reversal swings, independent of Streamlit."""
import numpy as np
import pandas as pd


def analyze_cycles(close, reversal=5., spacing=3):
    """Confirm alternating extremes after a percentage reversal and minimum bars.

    The first extreme is a boundary seed, excluded from cycle metrics. The last
    unconfirmed leg is excluded. Confirmation dates expose detection lag.
    """
    if reversal <= 0 or spacing < 1:
        raise ValueError('Reversal and spacing must be positive.')
    close = pd.to_numeric(close, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0].sort_index()
    close = close[~close.index.duplicated(keep='last')]
    values = close.to_numpy()
    swings = []
    direction, extreme, low, high = 0, 0, 0, 0
    threshold = reversal / 100
    for i, price in enumerate(values):
        if direction == 0:
            if price <= values[low]: low = i
            if price >= values[high]: high = i
            if i-low >= spacing and price >= values[low]*(1+threshold):
                direction, extreme = 1, i
            elif i-high >= spacing and price <= values[high]*(1-threshold):
                direction, extreme = -1, i
            continue
        if (direction == 1 and price >= values[extreme]) or (direction == -1 and price <= values[extreme]):
            extreme = i
        reversed_price = price <= values[extreme]*(1-threshold) if direction == 1 else price >= values[extreme]*(1+threshold)
        if reversed_price and i-extreme >= spacing:
            swings.append(dict(position=extreme, date=close.index[extreme], price=values[extreme],
                               kind='High' if direction == 1 else 'Low', confirmed=close.index[i]))
            direction, extreme = -direction, i
    points = pd.DataFrame(swings, columns=['position', 'date', 'price', 'kind', 'confirmed'])
    rows = []
    for a, b, c in zip(swings, swings[1:], swings[2:]):
        if [a['kind'], b['kind'], c['kind']] != ['Low', 'High', 'Low']: continue
        recovery = next((j for j in range(c['position']+1, len(close)) if values[j] >= b['price']), None)
        row = {'Start low': a['date'], 'Peak': b['date'], 'Next low': c['date'],
               'Low price': a['price'], 'Peak price': b['price'], 'Next low price': c['price'],
               'Rally %': (b['price']/a['price']-1)*100,
               'Pullback %': (c['price']/b['price']-1)*100,
               'Recovery date': close.index[recovery] if recovery is not None else pd.NaT,
               'Recovery status': 'Recovered prior peak' if recovery is not None else 'Not recovered in sample'}
        for label, start, end in [('Cycle', a['position'], c['position']), ('Rally', a['position'], b['position']),
                                  ('Pullback', b['position'], c['position']), ('Recovery', c['position'], recovery)]:
            row[f'{label} trading days'] = end-start if end is not None else np.nan
            row[f'{label} calendar days'] = (close.index[end].date()-close.index[start].date()).days if end is not None else np.nan
        previous_peaks = [x for x in swings if x['kind'] == 'High' and x['position'] < b['position']]
        previous = previous_peaks[-1] if previous_peaks else None
        row['Peak-to-peak trading days'] = b['position']-previous['position'] if previous else np.nan
        row['Peak-to-peak calendar days'] = (b['date'].date()-previous['date'].date()).days if previous else np.nan
        rows.append(row)
    return close, points, pd.DataFrame(rows)


def repeated_zones(points, tolerance=2.):
    """Group same-kind confirmed extremes within a bounded percentage band."""
    zones = []
    for kind in ['High', 'Low']:
        group = []
        for row in points[points.kind == kind].sort_values('price').to_dict('records'):
            if group and row['price']/group[0]['price']-1 > tolerance/100:
                if len(group) >= 2: zones.append((kind, group))
                group = []
            group.append(row)
        if len(group) >= 2: zones.append((kind, group))
    return zones


def render_cycle_page():
    import streamlit as st
    import yfinance as yf
    import plotly.graph_objects as go

    st.header('1-Year Cycle Analysis')
    st.caption('Explore rallies, rejections and recoveries across the last year of daily prices.')
    controls = st.columns([2, 1, 1])
    symbol = controls[0].text_input('Symbol', 'AMD').strip().upper()
    reversal = controls[1].slider('Reversal threshold (%)', 1., 20., 5., .5)
    spacing = controls[2].slider('Minimum confirmation bars', 1, 15, 3)
    unit = st.radio('Interval units', ['Trading days', 'Calendar days'], horizontal=True)
    tolerance = st.slider('Repeated zone tolerance (%)', .5, 5., 2., .5)
    if not symbol or len(symbol) > 20 or not all(c.isalnum() or c in '.-^=/' for c in symbol):
        st.info('Enter a valid market symbol.'); return

    @st.cache_data(ttl=300)
    def history(symbol):
        return yf.Ticker(symbol).history(period='1y', interval='1d', auto_adjust=True)

    try:
        with st.spinner(f'Analyzing {symbol}…'):
            data = history(symbol)
        if data.empty or 'Close' not in data:
            st.warning('No daily prices returned. Check the symbol or try again later.'); return
        close, points, cycles = analyze_cycles(data.Close, reversal, spacing)
    except Exception:
        st.error('Price data could not be loaded. Please try again later.'); return
    if len(close) < 3:
        st.info('Insufficient valid daily prices for swing analysis.'); return
    st.caption(f'{symbol} • {close.index[0]:%b %d, %Y} – {close.index[-1]:%b %d, %Y} • {len(close)} daily observations • adjusted closing prices • Yahoo Finance')
    suffix = unit.lower()
    intervals = {}
    for kind, label in [('High', 'Peak-to-peak'), ('Low', 'Trough-to-trough')]:
        subset = points[points.kind == kind]
        intervals[label] = (subset.position.diff().dropna().to_numpy() if unit == 'Trading days' else
                            np.array([(b.date()-a.date()).days for a,b in zip(subset.date, subset.date.iloc[1:])]))
    columns = st.columns(5)
    columns[0].metric('Completed low → low cycles', len(cycles))
    for col, (label, vals) in zip(columns[1:3], intervals.items()):
        col.metric(label + ' median', f'{np.median(vals):.1f} days' if len(vals) else 'N/A')
        col.caption(f'{len(vals)} intervals; mean {np.mean(vals):.1f}' if len(vals) else 'Need two confirmed swings')
    duration = cycles[f'Cycle {suffix}'] if not cycles.empty else pd.Series(dtype=float)
    columns[3].metric('Average cycle interval', f'{duration.mean():.1f} days' if len(duration) else 'N/A')
    columns[4].metric('Median cycle interval', f'{duration.median():.1f} days' if len(duration) else 'N/A')
    fig = go.Figure(go.Scatter(x=close.index, y=close, name='Adjusted close', line=dict(color='#3b82f6', width=2)))
    if not points.empty:
        fig.add_trace(go.Scatter(x=points.date, y=points.price, mode='lines', name='Confirmed swings', line=dict(color='#94a3b8', dash='dot')))
    for kind, color, marker in [('High', '#ef4444', 'triangle-down'), ('Low', '#10b981', 'triangle-up')]:
        subset = points[points.kind == kind]
        fig.add_trace(go.Scatter(x=subset.date, y=subset.price, mode='markers', name='Swing '+kind.lower(),
                                marker=dict(color=color, size=12, symbol=marker), customdata=[str(x.date()) for x in subset.confirmed],
                                hovertemplate='%{x|%b %d, %Y}<br>Price %{y:.2f}<br>Confirmed %{customdata}<extra></extra>'))
    zones = repeated_zones(points, tolerance)
    for kind, group in zones:
        fig.add_hrect(y0=min(x['price'] for x in group)*.999, y1=max(x['price'] for x in group)*1.001,
                      fillcolor='#ef4444' if kind == 'High' else '#10b981', opacity=.1, line_width=0,
                      annotation_text=f'{"Rejection" if kind == "High" else "Support"} zone · {len(group)} touches')
    fig.update_layout(height=570, hovermode='x unified', legend_orientation='h', margin=dict(l=15,r=15,t=40,b=15), yaxis_title='Adjusted price', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Zones group at least two confirmed highs or lows within the chosen price tolerance. They describe historical clusters; they do not establish future support or resistance.')
    if cycles.empty:
        st.info('No completed low → high → low cycles at these settings. Try a smaller reversal threshold or shorter confirmation spacing.')
    else:
        cols = st.columns(4)
        for col, label in zip(cols, ['Rally', 'Pullback', 'Recovery']):
            vals = cycles[f'{label} {suffix}'].dropna()
            col.metric(f'Median {label.lower()} duration', f'{vals.median():.1f} days' if len(vals) else 'N/A')
            col.caption(f'{len(vals)} completed observations')
        cols[3].metric('Median pullback', f'{cycles["Pullback %"].median():.1f}%')
        st.subheader('Detected cycles')
        st.dataframe(cycles, hide_index=True, use_container_width=True)
        st.download_button('Export cycles CSV', cycles.to_csv(index=False).encode(), f'{symbol}_1year_cycles.csv', 'text/csv')
    with st.expander('Swing confirmations and interval observations'):
        st.dataframe(points, hide_index=True, use_container_width=True)
        for label, vals in intervals.items():
            st.write(f'{label} ({suffix}): ' + (', '.join(f'{v:g}' for v in vals) if len(vals) else 'Insufficient swings'))
    st.info('Historical observations, not deterministic forecasts. Trading days count available daily price bars; calendar days count elapsed dates including weekends and holidays. Average/median cycle interval means confirmed trough-to-trough duration. Recovery is the first close after the next trough at or above the prior peak; unrecovered cycles are excluded from recovery averages.')
    st.caption(f'Method: a running high/low confirms only after a {reversal:g}% closing-price reversal at least {spacing} bars later. Extremes alternate. The initial boundary seed and final unconfirmed leg are excluded. Markers are backdated to the extreme, with confirmation dates in hover/table; this is retrospective and cannot be used as a same-day signal. Settings affect results. Adjusted prices account for splits/dividends; recent daily bars may still be incomplete. Short histories and small samples reduce reliability.')
