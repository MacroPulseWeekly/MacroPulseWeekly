import os
import requests
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

def build_copper_gold_pmi_chart(colors: dict) -> go.Figure:
    # 1. Fetch Copper and Gold (20 years)
    copper = yf.download("HG=F", period="20y")['Close']
    gold = yf.download("GC=F", period="20y")['Close']
    
    if isinstance(copper, pd.DataFrame): copper = copper.iloc[:, 0]
    if isinstance(gold, pd.DataFrame): gold = gold.iloc[:, 0]

    # 2. Fetch PMI Proxy (IPMAN)
    try:
        pmi_proxy = fred.get_series("IPMAN")
        pmi_proxy.name = "Manufacturing_Output"
    except Exception as e:
        pmi_proxy = fred.get_series("INDPRO") 
        pmi_proxy.name = "Industrial_Production"
    
    # 3. Align and Smooth Data
    df = pd.DataFrame(index=copper.index)
    df["Ratio"] = (copper / gold)
    df = df.join(pmi_proxy, how='left').ffill()
    
    # --- SMOOTHING LOGIC ---
    # We apply a 3-month rolling average to the proxy to remove monthly noise
    df["Smoothed_Proxy"] = df["Manufacturing_Output"].rolling(window=90).mean() # ~3 months of daily rows
    df = df.dropna()
    
    # 4. Create Figure
    fig = go.Figure()
    
    # Ratio (Left Axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Ratio"], 
        name="Cu/Au Ratio",
        line=dict(color=colors["mpw_blue"], width=1.5)
    ))
    
    # Smoothed Manufacturing Output (Right Axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Smoothed_Proxy"], 
        name="Mfg Output (3m Smooth)",
        # 'spline' makes the line curve naturally
        line=dict(color=colors["mpw_orange"], width=2, shape='spline'),
        yaxis="y2"
    ))

    fig.update_layout(
        title="20-Year Macro Engine: Cu/Au vs. Smoothed Manufacturing",
        yaxis=dict(title="Cu/Au Ratio"),
        yaxis2=dict(title="Production Index (Smoothed)", overlaying="y", side="right"),
        hovermode="x unified",
        template="plotly_dark"
    )
    return fig

def build_copper_gold_ratio_chart(colors: dict) -> go.Figure:
    # 1. Fetch Data
    copper = yf.download("HG=F", period="5y")['Close']
    gold = yf.download("GC=F", period="5y")['Close']
    
    # Handle yfinance MultiIndex formatting
    if isinstance(copper, pd.DataFrame): copper = copper.iloc[:, 0]
    if isinstance(gold, pd.DataFrame): gold = gold.iloc[:, 0]

    # 2. Create the DataFrame
    df = pd.DataFrame(index=copper.index)
    df["Copper"] = copper
    df["Gold"] = gold
    df = df.dropna()
    
    # 3. Calculate Ratio
    df["Ratio"] = df["Copper"] / df["Gold"]
    
    # 4. Create the Figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Ratio"], 
        name="Copper/Gold Ratio",
        line=dict(color=colors["mpw_blue"], width=2)
    ))

    fig.update_layout(
        title="Economic Barometer: Copper/Gold Ratio",
        yaxis=dict(title="Ratio Value"),
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig

def build_ibit_proxy_fallback(colors: dict) -> go.Figure:
    """Fallback function that uses stable Yahoo Finance data if scraping fails."""
    try:
        # Fetch IBIT (BlackRock) and BTC
        ibit = yf.download("IBIT", period="1y", progress=False)
        btc = yf.download("BTC-USD", period="1y", progress=False)

        # Handle potential MultiIndex columns
        if isinstance(ibit.columns, pd.MultiIndex): ibit.columns = ibit.columns.get_level_values(0)
        if isinstance(btc.columns, pd.MultiIndex): btc.columns = btc.columns.get_level_values(0)

        df = pd.DataFrame(index=ibit.index)
        df['Volume'] = ibit['Volume']
        df['BTC'] = btc['Close']
        df = df.dropna()

        fig = go.Figure()
        # Bars for Volume (Proxy for Institutional activity)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="IBIT Volume", marker_color=colors["mpw_blue"], opacity=0.6))
        # Line for BTC Price
        fig.add_trace(go.Scatter(x=df.index, y=df['BTC'], name="BTC Price", line=dict(color=colors["mpw_orange"]), yaxis="y2"))

        fig.update_layout(
            title="Institutional Demand Proxy (IBIT Volume)",
            yaxis=dict(title="Shares Traded"),
            yaxis2=dict(title="BTC Price", overlaying="y", side="right"),
            template="plotly_dark"
        )
        return fig
    except Exception as e:
        return go.Figure().update_layout(title=f"Critical Data Error: {str(e)[:30]}")

def build_btc_etf_flow_chart(colors: dict) -> go.Figure:
    """Primary scraper with automatic fallback to stable data."""
    url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        r = requests.get(url, headers=header, timeout=15)
        tables = pd.read_html(r.text)
        
        # Look for the main flow table
        df = next(t for t in tables if 'IBIT' in t.columns)
        df = df[['Date', 'Total']].copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Clean numeric strings
        df['Total'] = (df['Total'].astype(str)
                       .str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False)
                       .str.replace('(', '-', regex=False)
                       .str.replace(')', '', regex=False))
        
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
        df = df.dropna().sort_values('Date')

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Date'], y=df['Total'], name="Daily Flow ($M)",
                             marker_color=df['Total'].apply(lambda x: colors["mpw_blue"] if x > 0 else colors["mpw_red"])))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Total'].cumsum(), name="Cumulative", 
                                 line=dict(color=colors["mpw_orange"]), yaxis="y2"))

        fig.update_layout(title="Bitcoin Spot ETF Net Flows (Farside)", template="plotly_dark", yaxis2=dict(overlaying="y", side="right"))
        return fig

    except Exception as e:
        print(f"Scraper failed, using fallback: {e}")
        # THIS CALLS THE FUNCTION WE ADDED ABOVE
        return build_ibit_proxy_fallback(colors)

def build_fg_rsi_21_chart(btc_price: pd.Series, colors: dict) -> go.Figure:
    # Compute 21-period RSI
    rsi_21 = compute_rsi(btc_price, window=21)
    
    # Add a 9-period smoothing (Signal Line)
    signal_line = rsi_21.rolling(window=9).mean()
    
    fig = go.Figure()

    # Main RSI Line
    fig.add_trace(go.Scatter(
        x=rsi_21.index, y=rsi_21, 
        name="RSI (21)", 
        line=dict(color=colors["mpw_blue"], width=2)
    ))
    
    # Signal Line (Smoothed)
    fig.add_trace(go.Scatter(
        x=signal_line.index, y=signal_line, 
        name="Signal (9)", 
        line=dict(color=colors["mpw_orange"], width=1.5, dash='dot')
    ))

    # Overbought/Oversold Zones
    fig.add_hline(y=70, line=dict(color="#ff4b4b", dash="dash", width=1))
    fig.add_hline(y=30, line=dict(color="#00ff88", dash="dash", width=1))

    fig.update_layout(
        title="BTC Fear & Greed RSI (21 Period)", # <--- Updated Title
        yaxis=dict(title="RSI Value", range=[0, 100]),
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig
# ────────────────────────────────────────────────
# 4. Deployment
# ────────────────────────────────────────────────

# ────────────────────────────────────────────────
# 4. Deployment
# ────────────────────────────────────────────────

def build_dashboard_index(figs_dict: dict):
    with open("template.html", "r", encoding="utf-8") as f:
        content = f.read()

    for div_id, fig in figs_dict.items():
        # Generate snippet without the library (template has it)
        snippet = fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})
        # Find the div and drop the chart inside it
        content = content.replace(f'<div id="{div_id}"></div>', f'<div id="{div_id}">{snippet}</div>')

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    ensure_charts_dir()
    colors = register_macro_theme()
    btc, trends = get_data()

    # Build All Figures
    fg_rsi_fig = build_fg_rsi_chart(btc["Price"], colors)
    fg_rsi_21_fig = build_fg_rsi_21_chart(btc["Price"], colors)
    btc_ai_fig = build_btc_vs_ai_chart(btc, trends, colors)
    btc_m2_fig = build_btc_m2_chart(btc["Price"], colors)
    net_liq_fig = build_net_liquidity_chart(btc["Price"], colors)
    yield_unemp_fig = build_yield_unemployment_chart(colors)
    copper_gold_fig = build_copper_gold_ratio_chart(colors)
    cu_au_pmi_fig = build_copper_gold_pmi_chart(colors)
    btc_etf_fig = build_btc_etf_flow_chart(colors)

    # 1. Save Standalone HTMLs (For Framer)
    fg_rsi_fig.write_html("charts/fg_rsi.html", include_plotlyjs="cdn", config={'responsive': True})
    fg_rsi_21_fig.write_html("charts/fg_rsi_21.html", include_plotlyjs="cdn", config={'responsive': True})
    btc_ai_fig.write_html("charts/btc_ai.html", include_plotlyjs="cdn", config={'responsive': True})
    btc_m2_fig.write_html("charts/btc_m2.html", include_plotlyjs="cdn", config={'responsive': True})
    net_liq_fig.write_html("charts/net_liquidity.html", include_plotlyjs="cdn", config={'responsive': True})
    yield_unemp_fig.write_html("charts/yield_unemployment.html", include_plotlyjs="cdn", config={'responsive': True})
    copper_gold_fig.write_html("charts/copper_gold.html", include_plotlyjs="cdn", config={'responsive': True})
    cu_au_pmi_fig.write_html("charts/cu_au_pmi.html", include_plotlyjs="cdn")
    btc_etf_fig.write_html("charts/btc_etf_flows.html", include_plotlyjs="cdn")

    # 2. Build the Main Dashboard Index (MUST INCLUDE ALL KEYS)
    build_dashboard_index({
        "fg-rsi": fg_rsi_fig,
        "fg-rsi-21": fg_rsi_21_fig,
        "btc-ai": btc_ai_fig,
        "btc-m2": btc_m2_fig,
        "net-liq": net_liq_fig,
        "yield-unemp": yield_unemp_fig,
        "copper-gold": copper_gold_fig,
        "cu-au-pmi": cu_au_pmi_fig,
        "btc-etf-flows": btc_etf_fig
    })
    print("Update Complete: 9 Charts and Index generated.")

if __name__ == "__main__":
    main()
