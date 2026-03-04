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

# Initialize API Keys from GitHub Secrets / Environment Variables
fred = Fred(api_key=os.environ.get("FRED_API_KEY"))
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
FMP_API_KEY = os.environ.get('FMP_API_KEY')

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
        "mpw_gold":    "#FFD700"
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

def get_btc_data():
    # multi_level_index=False ensures flat columns for easier processing
    btc = yf.download("BTC-USD", start="2018-01-01", progress=False, multi_level_index=False)
    btc = btc[["Close"]].rename(columns={"Close": "Price"})
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    return btc

def get_fmp_cash_data(symbols):
    """Fetches latest cash and market cap from FMP for a list of tickers."""
    if not FMP_API_KEY:
        return pd.DataFrame()
    
    results = []
    base_url = "https://financialmodelingprep.com/api/v3"
    
    for symbol in symbols:
        try:
            # 1. Balance Sheet
            bs_url = f"{base_url}/balance-sheet-statement/{symbol}"
            bs_res = requests.get(bs_url, params={"limit": 1, "apikey": FMP_API_KEY}).json()
            # 2. Market Cap
            mc_url = f"{base_url}/market-capitalization/{symbol}"
            mc_res = requests.get(mc_url, params={"apikey": FMP_API_KEY}).json()
            
            if bs_res and mc_res:
                cash = bs_res[0].get('cashAndCashEquivalents', 0)
                mkt_cap = mc_res[0].get('marketCap', 1)
                results.append({
                    "Symbol": symbol,
                    "Cash_B": cash / 1e9,
                    "Mkt_Cap_B": mkt_cap / 1e9,
                    "Cash_to_Cap": (cash / mkt_cap) * 100
                })
        except Exception as e:
            print(f"FMP Error for {symbol}: {e}")
            
    return pd.DataFrame(results)

# ────────────────────────────────────────────────
# 3. Chart Builders
# ────────────────────────────────────────────────

def build_corporate_cash_chart(colors: dict) -> go.Figure:
    print("Fetching FMP Big Tech Cash Data...")
    tech_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    df = get_fmp_cash_data(tech_tickers)
    
    if df.empty:
        return go.Figure().update_layout(title="Corporate Cash Data Unavailable")
        
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Symbol"], y=df["Cash_B"], name="Cash ($B)", marker_color=colors["mpw_gold"]))
    fig.add_trace(go.Scatter(x=df["Symbol"], y=df["Cash_to_Cap"], name="Cash/Mkt Cap %", yaxis="y2", line=dict(color=colors["mpw_cyan"], width=3)))
    
    fig.update_layout(
        title="<b>The Cash Fortress:</b> Big Tech Cash Piles",
        yaxis=dict(title="Total Cash ($ Billions)"),
        yaxis2=dict(title="Cash as % of Mkt Cap", overlaying="y", side="right", showgrid=False)
    )
    return fig

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
        fig.update_layout(title="<b>Global Growth:</b> Copper/Gold Ratio vs. ISM PMI", yaxis2=dict(overlaying="y", side="right"))
        return fig
    except Exception as e:
        print(f"Cu/Au PMI Error: {e}")
        return go.Figure().update_layout(title="ISM PMI Correlation Data Unavailable")

# ────────────────────────────────────────────────
# 4. Deployment Logic
# ────────────────────────────────────────────────

def build_dashboard_index(figs_dict: dict):
    print("Generating index.html dashboard...")
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
        print("Error: template.html not found. Skipping web index generation.")

def main():
    ensure_charts_dir()
    colors = register_macro_theme()
    btc = get_btc_data()

    # Define all charts to build
    figs = {
        "corp-cash": build_corporate_cash_chart(colors),
        "fg-rsi": build_fg_rsi_chart(btc["Price"], colors),
        "net-liq": build_net_liquidity_chart(btc["Price"], colors),
        "cu-au-pmi": build_cu_au_pmi_chart(colors),
        "narrative-heat": build_narrative_heat_chart(colors),
    }

    # Save individual static HTML files
    for name, fig in figs.items():
        safe_name = name.replace('-', '_')
        fig.write_html(f"charts/{safe_name}.html", include_plotlyjs="cdn")

    # Combine into dashboard
    build_dashboard_index(figs)
    print(f"Update Complete: {len(figs)} macro charts generated.")

if __name__ == "__main__":
    main()
