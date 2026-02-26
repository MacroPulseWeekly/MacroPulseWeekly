import os
import requests
from fredapi import Fred
import pandas as pd
import yfinance as yf
from serpapi import GoogleSearch  # <--- Added for Narrative Heat
from plotly import graph_objects as go
import plotly.io as pio
from datetime import datetime

# ────────────────────────────────────────────────
# 1. Setup & Theme
# ────────────────────────────────────────────────

fred = Fred(api_key=os.environ.get("FRED_API_KEY"))
SERPAPI_KEY = os.environ.get("SERPAPI_KEY") # <--- Added SerpApi Key

def ensure_charts_dir():
    os.makedirs("charts", exist_ok=True)

def register_macro_theme():
    colors = {
        "mpw_blue":    "#4DA3FF",
        "mpw_orange":  "#FF8C42",
        "mpw_green":   "#4CAF50",
        "mpw_red":     "#FF5252",
        "mpw_gray":    "#AAAAAA",
        "mpw_cyan":    "#00D4FF",
    }
    macro_theme = {
        "layout": {
            "paper_bgcolor": "#111111",
            "plot_bgcolor":  "#111111",
            "font": {"family": "Arial", "color": "#FFFFFF", "size": 14},
            "title": {"font": {"size": 22, "color": "#FFFFFF"}, "x": 0.5},
            "xaxis": {"gridcolor": "#333333", "showgrid": True},
            "yaxis": {"gridcolor": "#333333", "showgrid": True},
            "margin": {"l": 50, "r": 50, "t": 80, "b": 50},
            "hovermode": "x unified",
        }
    }
    pio.templates["macro_pulse"] = macro_theme
    pio.templates.default = "macro_pulse"
    return colors

# ────────────────────────────────────────────────
# 2. Data Fetching & Math
# ────────────────────────────────────────────────

def compute_rsi(series, window: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    return 100 - (100 / (1 + rs))

def get_data():
    # Fetch Bitcoin Price
    btc = yf.download("BTC-USD", start="2018-01-01", progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    btc = btc[["Close"]].rename(columns={"Close": "Price"})
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    
    # Note: We are now using SerpApi for advanced trends in the new builder
    # but keeping this for your existing btc_vs_ai chart fallback
    return btc

# ────────────────────────────────────────────────
# 3. Chart Builders
# ────────────────────────────────────────────────

def build_narrative_heat_chart(colors: dict) -> go.Figure:
    """Fetches Google Trends via SerpApi for AI, AGI, and Productivity."""
    print("Fetching Narrative Heat data from SerpApi...")
    params = {
        "engine": "google_trends",
        "q": "AI, AGI, Productivity",
        "data_type": "TIMESERIES",
        "api_key": SERPAPI_KEY,
        "date": "today 12-m"
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        timeline = results["interest_over_time"]["timeline_data"]
        
        fig = go.Figure()

        # AI Line (The Hype)
        fig.add_trace(go.Scatter(
            x=[d['date'] for d in timeline], 
            y=[d['values'][0]['extracted_value'] for d in timeline],
            name="AI Search Interest",
            line=dict(color=colors["mpw_orange"], width=3)
        ))

        # AGI Line (The Speculation)
        fig.add_trace(go.Scatter(
            x=[d['date'] for d in timeline], 
            y=[d['values'][1]['extracted_value'] for d in timeline],
            name="AGI Search Interest",
            line=dict(color=colors["mpw_cyan"], width=2)
        ))

        # Productivity Line (The Reality)
        fig.add_trace(go.Scatter(
            x=[d['date'] for d in timeline], 
            y=[d['values'][2]['extracted_value'] for d in timeline],
            name="Productivity Interest",
            line=dict(color=colors["mpw_gray"], width=2, dash='dot')
        ))

        fig.update_layout(
            title="Narrative Heat: AI/AGI Hype vs. Productivity Reality",
            yaxis=dict(title="Relative Interest (0-100)"),
            hovermode="x unified"
        )
        return fig
    except Exception as e:
        print(f"Narrative Heat Error: {e}")
        return go.Figure().update_layout(title="Narrative Heat Data Unavailable")

def build_fg_rsi_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    rsi = compute_rsi(btc_price)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color=colors["mpw_blue"])))
    fig.add_hline(y=70, line=dict(color=colors["mpw_red"], dash="dash"))
    fig.add_hline(y=30, line=dict(color=colors["mpw_green"], dash="dash"))
    fig.update_layout(title="Bitcoin Fear & Greed RSI")
    fig.update_yaxes(range=[0, 100])
    return fig

def build_btc_m2_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    m2_raw = fred.get_series("WM2NS")
    df_m2 = m2_raw.to_frame(name="M2_Supply")
    df_m2.index = pd.to_datetime(df_m2.index).tz_localize(None)
    df_m2 = df_m2.resample('D').ffill()
    lagged_btc = btc_price.shift(70)
    merged = pd.concat([lagged_btc, df_m2], axis=1).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.index, y=merged["Price"], name="BTC Price (70D Lag)", line=dict(color=colors["mpw_orange"], width=2)))
    fig.add_trace(go.Scatter(x=merged.index, y=merged["M2_Supply"], name="US M2 Supply", line=dict(color=colors["mpw_blue"], width=2), yaxis="y2"))
    fig.update_layout(title="Liquidity Wave: BTC vs M2 Supply", yaxis=dict(type="log"), yaxis2=dict(overlaying="y", side="right"))
    return fig

def build_net_liquidity_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    df_liq = pd.DataFrame({
        "WALCL": fred.get_series("WALCL"),
        "TGA": fred.get_series("WTREGEN"),
        "RRP": fred.get_series("RRPONTSYD")
    }).ffill()
    net_liq = (df_liq["WALCL"] - df_liq["TGA"] - df_liq["RRP"]) / 1000
    merged = pd.concat([btc_price.shift(70), net_liq.rename("Net_Liq")], axis=1).ffill().dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.index, y=merged["Price"], name="BTC Price (70D Lag)", line=dict(color=colors["mpw_orange"])))
    fig.add_trace(go.Scatter(x=merged.index, y=merged["Net_Liq"], name="Net Liquidity", yaxis="y2", line=dict(color=colors["mpw_blue"], dash='dot')))
    fig.update_layout(title="Monetary Fuel: BTC vs Net Liquidity", yaxis=dict(type="log"), yaxis2=dict(overlaying="y", side="right"))
    return fig

def build_yield_unemployment_chart(colors: dict) -> go.Figure:
    df = pd.DataFrame({
        "Yield": fred.get_series("DGS10"),
        "Unemployment": fred.get_series("UNRATE"),
        "Recession": fred.get_series("USREC")
    }).ffill().dropna()
    fig = go.Figure()
    is_rec = df['Recession'] == 1
    starts = df.index[is_rec & ~is_rec.shift(1).fillna(False)]
    ends = df.index[is_rec & ~is_rec.shift(-1).fillna(False)]
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(150,150,150,0.2)", layer="below", line_width=0)
    fig.add_trace(go.Scatter(x=df.index, y=df["Yield"], name="10Y Yield", line=dict(color=colors["mpw_orange"])))
    fig.add_trace(go.Scatter(x=df.index, y=df["Unemployment"], name="Unemployment", yaxis="y2", line=dict(color=colors["mpw_blue"])))
    fig.update_layout(title="Economic Cycle", yaxis2=dict(overlaying="y", side="right"))
    return fig

def build_copper_gold_ratio_chart(colors: dict) -> go.Figure:
    copper = yf.download("HG=F", period="5y", progress=False)['Close']
    gold = yf.download("GC=F", period="5y", progress=False)['Close']
    if isinstance(copper, pd.DataFrame): copper = copper.iloc[:, 0]
    if isinstance(gold, pd.DataFrame): gold = gold.iloc[:, 0]
    ratio = (copper / gold).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Cu/Au Ratio", line=dict(color=colors["mpw_blue"])))
    fig.update_layout(title="Economic Barometer: Copper/Gold Ratio")
    return fig

def build_btc_etf_flow_chart(colors: dict) -> go.Figure:
    url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    header = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=header, timeout=15)
        df = next(t for t in pd.read_html(r.text) if 'IBIT' in t.columns)
        df = df[['Date', 'Total']].copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Total'] = pd.to_numeric(df['Total'].astype(str).str.replace(r'[$,()]', '', regex=True), errors='coerce')
        df = df.dropna().sort_values('Date')
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Date'], y=df['Total'], name="Daily Flow", marker_color=colors["mpw_blue"]))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Total'].cumsum(), name="Cumulative", line=dict(color=colors["mpw_orange"]), yaxis="y2"))
        fig.update_layout(title="Bitcoin ETF Flows", yaxis2=dict(overlaying="y", side="right"))
        return fig
    except:
        return go.Figure().update_layout(title="ETF Data Unavailable")

def build_gli_bes_change_chart(colors: dict) -> go.Figure:
    liq = fred.get_series("WALCL").resample('W').last().pct_change(periods=6)*100
    btc = yf.download("BTC-USD", period="2y", progress=False)['Close'].resample('W').last().pct_change(periods=6)*100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=liq.index, y=liq, name="GLI$ %ch", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=btc.index, y=btc, name="BTC %ch", line=dict(color=colors["mpw_orange"], width=2), yaxis="y2"))
    fig.update_layout(title="GLI$ vs BTC (6w %ch)", yaxis2=dict(overlaying="y", side="right"))
    return fig

# ────────────────────────────────────────────────
# 4. Deployment
# ────────────────────────────────────────────────

def build_dashboard_index(figs_dict: dict):
    with open("template.html", "r", encoding="utf-8") as f:
        content = f.read()
    for div_id, fig in figs_dict.items():
        snippet = fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})
        content = content.replace(f'<div id="{div_id}"></div>', f'<div id="{div_id}">{snippet}</div>')
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    ensure_charts_dir()
    colors = register_macro_theme()
    btc = get_data()

    # 1. Build Figures
    figs = {
        "fg-rsi": build_fg_rsi_chart(btc["Price"], colors),
        "btc-m2": build_btc_m2_chart(btc["Price"], colors),
        "net-liq": build_net_liquidity_chart(btc["Price"], colors),
        "yield-unemp": build_yield_unemployment_chart(colors),
        "copper-gold": build_copper_gold_ratio_chart(colors),
        "btc-etf-flows": build_btc_etf_flow_chart(colors),
        "gli-bes": build_gli_bes_change_chart(colors),
        "narrative-heat": build_narrative_heat_chart(colors) # <--- New Chart
    }

    # 2. Save Standalone HTMLs
    for name, fig in figs.items():
        fig.write_html(f"charts/{name.replace('-', '_')}.html", include_plotlyjs="cdn")

    # 3. Build Index
    build_dashboard_index(figs)
    print(f"Update Complete: {len(figs)} Charts generated.")

if __name__ == "__main__":
    main()
