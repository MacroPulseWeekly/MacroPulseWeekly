import os
import requests
from fredapi import Fred
import pandas as pd
import yfinance as yf
from serpapi import GoogleSearch 
from plotly import graph_objects as go
import plotly.io as pio
from datetime import datetime

# ────────────────────────────────────────────────
# 1. Setup & Theme
# ────────────────────────────────────────────────

fred = Fred(api_key=os.environ.get("FRED_API_KEY"))
SERPAPI_KEY = os.environ.get("SERPAPI_KEY") 

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
    # multi_level_index=False is required for modern yfinance compatibility
    btc = yf.download("BTC-USD", start="2018-01-01", progress=False, multi_level_index=False)
    btc = btc[["Close"]].rename(columns={"Close": "Price"})
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    return btc

# ────────────────────────────────────────────────
# 3. Chart Builders
# ────────────────────────────────────────────────

def build_narrative_heat_chart(colors: dict) -> go.Figure:
    print("Fetching Narrative Heat data...")
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
        fig.add_trace(go.Scatter(x=[d['date'] for d in timeline], y=[d['values'][0]['extracted_value'] for d in timeline], name="AI Interest", line=dict(color=colors["mpw_orange"], width=3)))
        fig.add_trace(go.Scatter(x=[d['date'] for d in timeline], y=[d['values'][1]['extracted_value'] for d in timeline], name="AGI Interest", line=dict(color=colors["mpw_cyan"], width=2)))
        fig.add_trace(go.Scatter(x=[d['date'] for d in timeline], y=[d['values'][2]['extracted_value'] for d in timeline], name="Productivity", line=dict(color=colors["mpw_gray"], width=2, dash='dot')))
        fig.update_layout(title="Narrative Heat: AI/AGI vs. Productivity")
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
    return fig

def build_fg_rsi_21_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    rsi_21 = compute_rsi(btc_price, window=21)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi_21.index, y=rsi_21, name="RSI (21)", line=dict(color=colors["mpw_blue"], width=2)))
    fig.add_hline(y=70, line=dict(color="#ff4b4b", dash="dash"))
    fig.add_hline(y=30, line=dict(color="#00ff88", dash="dash"))
    fig.update_layout(title="BTC Fear & Greed RSI (21 Period)")
    return fig

def build_btc_ai_fallback(btc: pd.DataFrame, colors: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc.index, y=btc["Price"], name="BTC Price", line=dict(color=colors["mpw_orange"])))
    fig.update_layout(title="BTC Price (Trend Overlay)")
    return fig

def build_btc_m2_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    m2_raw = fred.get_series("WM2NS")
    df_m2 = m2_raw.to_frame(name="M2_Supply").resample('D').ffill()
    df_m2.index = pd.to_datetime(df_m2.index).tz_localize(None)
    merged = pd.concat([btc_price.shift(70), df_m2], axis=1).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.index, y=merged["Price"], name="BTC Price (70D Lag)", line=dict(color=colors["mpw_orange"])))
    fig.add_trace(go.Scatter(x=merged.index, y=merged["M2_Supply"], name="US M2 Supply", line=dict(color=colors["mpw_blue"]), yaxis="y2"))
    fig.update_layout(title="Liquidity Wave: BTC vs M2 Supply", yaxis2=dict(overlaying="y", side="right"))
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
    fig.add_trace(go.Scatter(x=merged.index, y=merged["Price"], name="BTC (70D Lag)", line=dict(color=colors["mpw_orange"])))
    fig.add_trace(go.Scatter(x=merged.index, y=merged["Net_Liq"], name="Net Liq ($B)", yaxis="y2", line=dict(color=colors["mpw_blue"], dash='dot')))
    fig.update_layout(title="Monetary Fuel: BTC vs Net Liquidity", yaxis2=dict(overlaying="y", side="right"))
    return fig

def build_yield_unemployment_chart(colors: dict) -> go.Figure:
    df = pd.DataFrame({"Yield": fred.get_series("DGS10"), "Unemployment": fred.get_series("UNRATE"), "Recession": fred.get_series("USREC")}).ffill().dropna()
    fig = go.Figure()
    is_rec = df['Recession'] == 1
    starts, ends = df.index[is_rec & ~is_rec.shift(1).fillna(False)], df.index[is_rec & ~is_rec.shift(-1).fillna(False)]
    for s, e in zip(starts, ends): fig.add_vrect(x0=s, x1=e, fillcolor="rgba(150,150,150,0.2)", layer="below", line_width=0)
    fig.add_trace(go.Scatter(x=df.index, y=df["Yield"], name="10Y Yield", line=dict(color=colors["mpw_orange"])))
    fig.add_trace(go.Scatter(x=df.index, y=df["Unemployment"], name="Unemployment", yaxis="y2", line=dict(color=colors["mpw_blue"])))
    fig.update_layout(title="Economic Cycle (Yield vs Unemp)", yaxis2=dict(overlaying="y", side="right"))
    return fig

def build_delinquency_unemployment_chart(colors: dict) -> go.Figure:
    print("Fetching Delinquency data...")
    try:
        df = pd.DataFrame({
            "Delinquency": fred.get_series("DRBLACBS"),
            "Unemployment": fred.get_series("UNRATE")
        }).ffill().dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Delinquency"], name="Biz Delinquency %", line=dict(color=colors["mpw_orange"], width=3)))
        fig.add_trace(go.Scatter(x=df.index, y=df["Unemployment"], name="Unemployment %", yaxis="y2", line=dict(color=colors["mpw_blue"], width=2, dash='dot')))
        fig.update_layout(title="Credit Risk vs Labor Market", yaxis2=dict(overlaying="y", side="right"))
        return fig
    except: return go.Figure().update_layout(title="Delinquency Data Unavailable")

def build_copper_gold_ratio_chart(colors: dict) -> go.Figure:
    c = yf.download("HG=F", period="5y", progress=False, multi_level_index=False)['Close']
    g = yf.download("GC=F", period="5y", progress=False, multi_level_index=False)['Close']
    ratio = (c / g).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Cu/Au Ratio", line=dict(color=colors["mpw_blue"])))
    fig.update_layout(title="Copper/Gold Ratio")
    return fig

def build_cu_au_pmi_chart(colors: dict) -> go.Figure:
    print("Fetching Copper/Gold vs ISM PMI data...")
    try:
        c = yf.download("HG=F", period="5y", progress=False, multi_level_index=False)['Close']
        g = yf.download("GC=F", period="5y", progress=False, multi_level_index=False)['Close']
        ratio = (c / g).dropna()
        pmi = fred.get_series("MANPMI", observation_start="2019-01-01")
        pmi = pmi.resample('D').ffill()
        df = pd.concat([ratio.rename("Ratio"), pmi.rename("PMI")], axis=1).dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Ratio"], name="Cu/Au Ratio", line=dict(color=colors["mpw_blue"], width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["PMI"], name="ISM PMI", yaxis="y2", line=dict(color=colors["mpw_orange"], width=2, dash='dot')))
        fig.add_hline(y=50, line=dict(color="gray", dash="dash", width=1), yref="y2")
        fig.update_layout(title="<b>Global Growth:</b> Copper/Gold Ratio vs. ISM PMI",
            yaxis=dict(title="Cu/Au Ratio"),
            yaxis2=dict(title="ISM PMI (50 = Neutral)", overlaying="y", side="right", showgrid=False))
        return fig
    except Exception as e:
        print(f"Cu/Au PMI Error: {e}")
        return go.Figure().update_layout(title="ISM PMI Correlation Data Unavailable")

def build_btc_etf_flow_chart(colors: dict) -> go.Figure:
    url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        df = next(t for t in pd.read_html(r.text) if 'IBIT' in t.columns)
        df = df[['Date', 'Total']].copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Total'] = pd.to_numeric(df['Total'].astype(str).str.replace(r'[$,()]', '', regex=True), errors='coerce')
        df = df.dropna().sort_values('Date')
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Date'], y=df['Total'], name="Daily Flow", marker_color=colors["mpw_blue"]))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Total'].cumsum(), name="Cumulative", line=dict(color=colors["mpw_orange"]), yaxis="y2"))
        fig.update_layout(title="Bitcoin Spot ETF Flows", yaxis2=dict(overlaying="y", side="right"))
        return fig
    except: return go.Figure().update_layout(title="ETF Flows Unavailable")

def build_gli_bes_change_chart(colors: dict) -> go.Figure:
    liq = fred.get_series("WALCL").resample('W').last().pct_change(periods=6)*100
    btc = yf.download("BTC-USD", period="2y", progress=False, multi_level_index=False)['Close'].resample('W').last().pct_change(periods=6)*100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=liq.index, y=liq, name="GLI$ %ch", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=btc.index, y=btc, name="BTC %ch", line=dict(color=colors["mpw_orange"]), yaxis="y2"))
    fig.update_layout(title="GLI$ vs BTC (6w %ch)")
    return fig

def build_leading_liquidity_chart(colors: dict) -> go.Figure:
    print("Fetching Leading Liquidity data...")
    liq = fred.get_series("WALCL", observation_start="2020-01-01")
    btc = yf.download("BTC-USD", start="2020-01-01", progress=False, multi_level_index=False)['Close']
    liq_6w = liq.resample('W').last().pct_change(periods=6) * 100
    btc_6w = btc.resample('W').last().pct_change(periods=6) * 100
    df = pd.DataFrame({"GLI_Leading": liq_6w.shift(13), "BTC_Ch": btc_6w}).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["GLI_Leading"], name="GLI$ (Leading +13w)", line=dict(color="white", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BTC_Ch"], name="BTC %ch", line=dict(color=colors["mpw_orange"], width=2), yaxis="y2"))
    fig.update_layout(title="Leading Indicator: GLI$ (+13w) vs. BTC",
        yaxis=dict(title="GLI$ % Change"), yaxis2=dict(title="BTC % Change", overlaying="y", side="right"))
    return fig

def build_crash_sentiment_overlay(colors: dict) -> go.Figure:
    print("Fetching 'Bitcoin Crash' trends and Sentiment data...")
    params = {"engine": "google_trends", "q": "bitcoin crash", "data_type": "TIMESERIES", "api_key": SERPAPI_KEY, "date": "today 5-y"}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        trends_data = results["interest_over_time"]["timeline_data"]
        fng_url = "https://api.alternative.me/fng/?limit=0&format=json"
        fng_res = requests.get(fng_url).json()['data']
        fng_df = pd.DataFrame(fng_res)
        fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'], unit='s')
        fng_df['value'] = pd.to_numeric(fng_df['value'])
        fng_df.set_index('timestamp', inplace=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[d['date'] for d in trends_data], y=[d['values'][0]['extracted_value'] for d in trends_data], name="Search: 'Bitcoin Crash'", line=dict(color=colors["mpw_red"], width=2), fill='tozeroy', fillcolor='rgba(255, 82, 82, 0.1)'))
        fig.add_trace(go.Scatter(x=fng_df.index, y=fng_df['value'], name="Fear & Greed Index", yaxis="y2", line=dict(color=colors["mpw_cyan"], width=2, dash='dot')))
        fig.update_layout(title="<b>Panic Barometer:</b> 'Bitcoin Crash' Search vs. Sentiment",
            yaxis=dict(title="Search Interest (0-100)"),
            yaxis2=dict(title="Fear & Greed (0-100)", overlaying="y", side="right", range=[0, 100], showgrid=False))
        return fig
    except Exception as e:
        print(f"Panic Chart Error: {e}")
        return go.Figure().update_layout(title="Panic Indicator Data Unavailable")

# ────────────────────────────────────────────────
# 4. Deployment
# ────────────────────────────────────────────────

def build_dashboard_index(figs_dict: dict):
    print("Building final dashboard index...")
    try:
        with open("template.html", "r", encoding="utf-8") as f:
            content = f.read()
        for div_id, fig in figs_dict.items():
            snippet = fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})
            placeholder = "{{" + div_id.replace("-", "_") + "_chart" + "}}"
            if placeholder in content:
                content = content.replace(placeholder, snippet)
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        if "{{ last_updated }}" in content:
            content = content.replace("{{ last_updated }}", last_updated)
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(content)
        print("Success: index.html generated.")
    except FileNotFoundError:
        print("Error: template.html not found.")

def main():
    ensure_charts_dir()
    colors = register_macro_theme()
    btc = get_data()

    figs = {
        "crash-sentiment": build_crash_sentiment_overlay(colors),
        "fg-rsi": build_fg_rsi_chart(btc["Price"], colors),
        "fg-rsi-21": build_fg_rsi_21_chart(btc["Price"], colors),
        "gli-bes": build_gli_bes_change_chart(colors),
        "leading-liq": build_leading_liquidity_chart(colors),
        "btc-ai": build_btc_ai_fallback(btc, colors),
        "narrative-heat": build_narrative_heat_chart(colors),
        "btc-m2": build_btc_m2_chart(btc["Price"], colors),
        "net-liq": build_net_liquidity_chart(btc["Price"], colors),
        "yield-unemp": build_yield_unemployment_chart(colors),
        "delinquency-unemp": build_delinquency_unemployment_chart(colors),
        "copper-gold": build_copper_gold_ratio_chart(colors),
        "cu-au-pmi": build_cu_au_pmi_chart(colors),
        "btc-etf-flows": build_btc_etf_flow_chart(colors)
    }

    for name, fig in figs.items():
        safe_name = name.replace('-', '_')
        fig.write_html(f"charts/{safe_name}.html", include_plotlyjs="cdn")

    build_dashboard_index(figs)
    print(f"Update Complete: {len(figs)} charts generated.")

if __name__ == "__main__":
    main()
