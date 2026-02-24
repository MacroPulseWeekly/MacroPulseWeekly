import os
import pandas as pd
import yfinance as yf
from pytrends.request import TrendReq
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

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

    # 1. Save Standalone HTMLs (For Framer Embedding)
    # These include the library so they work as individual links
    fg_rsi_fig.write_html("charts/fg_rsi.html", include_plotlyjs="cdn", config={'responsive': True})
    btc_ai_fig.write_html("charts/btc_ai.html", include_plotlyjs="cdn", config={'responsive': True})

    # 2. Build the Main Dashboard Index (For GitHub Pages)
    build_dashboard_index({
        "fg-rsi": fg_rsi_fig,
        "btc-ai": btc_ai_fig
    })
    print("Update Complete: Charts and Index generated.")

if __name__ == "__main__":
    main()
