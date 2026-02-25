import os
from fredapi import Fred
import pandas as pd
import yfinance as yf
from pytrends.request import TrendReq
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

fred = Fred(api_key=os.environ.get("FRED_API_KEY"))

# ────────────────────────────────────────────────
# 1. Setup & Theme
# ────────────────────────────────────────────────

def ensure_charts_dir():
    os.makedirs("charts", exist_ok=True)

def register_macro_theme():
    colors = {
        "mpw_blue":   "#4DA3FF",
        "mpw_orange": "#FF8C42",
        "mpw_green":  "#4CAF50",
        "mpw_red":    "#FF5252",
        "mpw_gray":   "#AAAAAA",
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
    # 1. Fetch Bitcoin Price
    btc = yf.download("BTC-USD", start="2018-01-01", progress=False)
    
    # FIX: Flatten the MultiIndex columns if they exist
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    btc = btc[["Close"]].rename(columns={"Close": "Price"})
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    
    # 2. Fetch Google Trends for "AI"
    pytrends = TrendReq(hl="en-US", tz=0)
    end_str = datetime.today().strftime("%Y-%m-%d")
    pytrends.build_payload(["ai"], timeframe=f"2018-01-01 {end_str}")
    trends = pytrends.interest_over_time()
    
    # Handle empty trends or weird Index issues
    if trends.empty:
        # Fallback empty dataframe to prevent crash
        trends = pd.DataFrame(columns=["AI_Searches"])
    else:
        trends = trends[["ai"]].rename(columns={"ai": "AI_Searches"})
        trends.index = pd.to_datetime(trends.index).tz_localize(None)
    
    return btc, trends

# ────────────────────────────────────────────────
# 3. Chart Builders
# ────────────────────────────────────────────────

def build_fg_rsi_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    rsi = compute_rsi(btc_price)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color=colors["mpw_blue"])))
    fig.add_hline(y=70, line=dict(color=colors["mpw_red"], dash="dash"))
    fig.add_hline(y=30, line=dict(color=colors["mpw_green"], dash="dash"))
    fig.update_layout(title="Bitcoin Fear & Greed RSI")
    fig.update_yaxes(range=[0, 100])
    return fig

def build_btc_vs_ai_chart(btc: pd.DataFrame, trends: pd.DataFrame, colors: dict) -> go.Figure:
    merged = btc.join(trends, how="inner")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.index, y=merged["Price"], name="BTC Price", line=dict(color=colors["mpw_orange"])))
    fig.add_trace(go.Scatter(x=merged.index, y=merged["AI_Searches"], name="AI Trends", yaxis="y2", line=dict(color=colors["mpw_blue"])))
    fig.update_layout(
        title="Bitcoin vs Google AI Trends",
        yaxis=dict(title="Price (USD)"),
        yaxis2=dict(title="Search Interest", overlaying="y", side="right")
    )
    return fig

def build_btc_m2_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    # 1. Fetch M2 Money Supply from FRED (Series: WM2NS)
    m2_raw = fred.get_series("WM2NS")
    m2_raw.name = "M2_Supply"
    
    # 2. Resample and "Forward Fill"
    df_m2 = m2_raw.to_frame()
    df_m2.index = pd.to_datetime(df_m2.index).tz_localize(None)
    df_m2 = df_m2.resample('D').ffill()
    
    # --- ADDED: Shift BTC Price forward by 70 days ---
    # This aligns past liquidity with future price action
    lagged_btc = btc_price.shift(70)
    
    # 3. Combine with Lagged BTC
    merged = pd.concat([lagged_btc, df_m2], axis=1).dropna()

    # 4. Create the Chart
    fig = go.Figure()
    
    # Bitcoin Price (Left Axis) - Notice name change to reflect lag
    fig.add_trace(go.Scatter(
        x=merged.index, y=merged["Price"], 
        name="BTC Price (70D Lag)", 
        line=dict(color=colors["mpw_orange"], width=2)
    ))
    
    # M2 Supply (Right Axis)
    fig.add_trace(go.Scatter(
        x=merged.index, y=merged["M2_Supply"], 
        name="US M2 Money Supply",
        line=dict(color=colors["mpw_blue"], width=2),
        yaxis="y2"
    ))

    fig.update_layout(
        title="Liquidity Wave: BTC (70-Day Lag) vs. M2 Money Supply",
        yaxis=dict(title="BTC Price (Shifted)", type="log"),
        yaxis2=dict(title="M2 Supply (Billions)", overlaying="y", side="right"),
        hovermode="x unified"
    )
    
    return fig

def build_net_liquidity_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    # 1. Fetch and align FRED data
    df_liq = pd.DataFrame({
        "WALCL": fred.get_series("WALCL"),
        "TGA": fred.get_series("WTREGEN"),
        "RRP": fred.get_series("RRPONTSYD")
    }).ffill() 
    
    # 2. Calculate Net Liquidity in Trillions
    net_liq = (df_liq["WALCL"] - df_liq["TGA"] - df_liq["RRP"]) / 1000
    net_liq.name = "Net_Liq"
    net_liq.index = pd.to_datetime(net_liq.index).tz_localize(None)
    
    # --- ADDED: Shift BTC Price forward by 70 days ---
    lagged_btc = btc_price.shift(70)
    
    # 3. Combine with Lagged BTC
    merged = pd.concat([lagged_btc, net_liq], axis=1).ffill().dropna()
    
    # 4. Create Figure
    fig = go.Figure()
    
    # Update name to reflect the 70D Lag
    fig.add_trace(go.Scatter(
        x=merged.index, 
        y=merged["Price"], 
        name="BTC Price (70D Lag)", 
        line=dict(color=colors["mpw_orange"], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=merged.index, 
        y=merged["Net_Liq"], 
        name="Net Liquidity", 
        yaxis="y2", 
        line=dict(color=colors["mpw_blue"], width=2, dash='dot')
    ))

    fig.update_layout(
        title="Monetary Fuel: BTC (70-Day Lag) vs. Net Liquidity",
        yaxis=dict(title="BTC Price (Shifted - Log)", type="log"), 
        yaxis2=dict(title="Net Liquidity (Trillions)", overlaying="y", side="right"), 
        hovermode="x unified"
    )
    return fig

def build_yield_unemployment_chart(colors: dict) -> go.Figure:
    # 1. Fetch only what we need
    df = pd.DataFrame({
        "Yield": fred.get_series("DGS10"),
        "Unemployment": fred.get_series("UNRATE"),
        "Recession": fred.get_series("USREC")
    }).ffill().dropna()

    fig = go.Figure()

    # 2. Optimized Shading (Single Pass)
    # This finds the actual start and end dates once
    is_rec = df['Recession'] == 1
    starts = df.index[is_rec & ~is_rec.shift(1).fillna(False)]
    ends = df.index[is_rec & ~is_rec.shift(-1).fillna(False)]

    # Draw boxes only if there are fewer than 50 (prevents hanging)
    if len(starts) < 50:
        for s, e in zip(starts, ends):
            fig.add_vrect(x0=s, x1=e, fillcolor="rgba(150,150,150,0.2)", layer="below", line_width=0)

    # 3. Add Lines
    fig.add_trace(go.Scatter(x=df.index, y=df["Yield"], name="10Y Yield", line=dict(color=colors["mpw_orange"])))
    fig.add_trace(go.Scatter(x=df.index, y=df["Unemployment"], name="Unemployment", yaxis="y2", line=dict(color=colors["mpw_blue"])))

    fig.update_layout(title="Economic Cycle", yaxis2=dict(overlaying="y", side="right"), hovermode="x unified")
    return fig

def build_copper_gold_ratio_chart(colors: dict) -> go.Figure:
    # 1. Fetch Data
    # We use auto_adjust=True and explicitly grab the 'Close' column
    copper = yf.download("HG=F", period="5y")['Close']
    gold = yf.download("GC=F", period="5y")['Close']
    
    # 2. Flatten the data (yfinance sometimes returns MultiIndex columns)
    if isinstance(copper, pd.DataFrame):
        copper = copper.iloc[:, 0]
    if isinstance(gold, pd.DataFrame):
        gold = gold.iloc[:, 0]

    # 3. Create the DataFrame explicitly with the indices
    df = pd.DataFrame(index=copper.index)
    df["Copper"] = copper
    df["Gold"] = gold
    df = df.dropna()
    
    # 4. Calculate Ratio
    df["Ratio"] = df["Copper"] / df["Gold"]
    
    # 5. Create the Figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["Ratio"], 
        name="Copper/Gold Ratio",
        line=dict(color=colors["mpw_blue"], width=2)
    ))

    fig.update_layout(
        title="Economic Barometer: Copper/Gold Ratio",
        yaxis=dict(title="Ratio Value"),
        template="plotly_dark", # Ensures it matches your theme
        hovermode="x unified"
    )
    
    return fig

def build_copper_gold_pmi_chart(colors: dict) -> go.Figure:
    # 1. Fetch Copper and Gold
    copper = yf.download("HG=F", period="5y")['Close']
    gold = yf.download("GC=F", period="5y")['Close']
    
    if isinstance(copper, pd.DataFrame): copper = copper.iloc[:, 0]
    if isinstance(gold, pd.DataFrame): gold = gold.iloc[:, 0]

    # 2. Fetch the "PMI Proxy" (Industrial Production: Manufacturing - IPMANS)
    # This is free, reliable, and matches ISM PMI trends
    pmi_proxy = fred.get_series("IPMANS")
    pmi_proxy.name = "Manufacturing_Output"
    
    # 3. Align Data
    df = pd.DataFrame(index=copper.index)
    df["Ratio"] = (copper / gold)
    df = df.join(pmi_proxy, how='left').ffill().dropna()
    
    # 4. Create Figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Ratio"], 
        name="Cu/Au Ratio",
        line=dict(color=colors["mpw_blue"], width=2)
    ))
    
    # PMI Proxy (Right Axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Manufacturing_Output"], 
        name="Mfg Output (PMI Proxy)",
        line=dict(color=colors["mpw_orange"], width=2, dash='dot'),
        yaxis="y2"
    ))

    fig.update_layout(
        title="Economic Engine: Cu/Au Ratio vs. Manufacturing Output",
        yaxis=dict(title="Cu/Au Ratio"),
        yaxis2=dict(title="Production Index", overlaying="y", side="right"),
        hovermode="x unified",
        template="plotly_dark"
    )
    return fig
# ────────────────────────────────────────────────
# 4. Deployment
# ────────────────────────────────────────────────

def build_dashboard_index(figs_dict: dict):
    with open("template.html", "r", encoding="utf-8") as f:
        content = f.read()

    # Load Plotly library ONCE in the head
    lib_script = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    content = content.replace('</head>', f'{lib_script}\n</head>')

    for div_id, fig in figs_dict.items():
        # Generate snippet for the index page
        snippet = fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})
        content = content.replace(f'<div id="{div_id}"></div>', f'<div id="{div_id}">{snippet}</div>')

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    ensure_charts_dir()
    colors = register_macro_theme()
    btc, trends = get_data()
    

    # Build Figures
    fg_rsi_fig = build_fg_rsi_chart(btc["Price"], colors)
    btc_ai_fig = build_btc_vs_ai_chart(btc, trends, colors)
    btc_m2_fig = build_btc_m2_chart(btc["Price"], colors)
    net_liq_fig = build_net_liquidity_chart(btc["Price"], colors)
    yield_unemp_fig = build_yield_unemployment_chart(colors)
    copper_gold_fig = build_copper_gold_ratio_chart(colors)
    cu_au_pmi_fig = build_copper_gold_pmi_chart(colors)

    # 1. Save Standalone HTMLs (For Framer Embedding)
    # These include the library so they work as individual links
    fg_rsi_fig.write_html("charts/fg_rsi.html", include_plotlyjs="cdn", config={'responsive': True})
    btc_ai_fig.write_html("charts/btc_ai.html", include_plotlyjs="cdn", config={'responsive': True})
    btc_m2_fig.write_html("charts/btc_m2.html", include_plotlyjs="cdn", config={'responsive': True})
    net_liq_fig.write_html("charts/net_liquidity.html", include_plotlyjs="cdn", config={'responsive': True})
    yield_unemp_fig.write_html("charts/yield_unemployment.html", include_plotlyjs="cdn", config={'responsive': True})
    copper_gold_fig.write_html("charts/copper_gold.html", include_plotlyjs="cdn", config={'responsive': True})
    cu_au_pmi_fig.write_html("charts/cu_au_pmi.html", include_plotlyjs="cdn")
    

    # 2. Build the Main Dashboard Index (For GitHub Pages)
    build_dashboard_index({
        # ... your other charts ...,
        "copper-gold": copper_gold_fig,
        "cu-au-pmi": cu_au_pmi_fig
})
    print("Update Complete: Charts and Index generated.")

if __name__ == "__main__":
    main()
