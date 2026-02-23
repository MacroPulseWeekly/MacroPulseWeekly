import os
from datetime import datetime
import pandas as pd
import yfinance as yf
from pytrends.request import TrendReq
import plotly.graph_objects as go
import plotly.io as pio
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])

def ensure_charts_dir():
    os.makedirs("charts", exist_ok=True)

def compute_rsi(series, window: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_fred_series(series_id: str) -> pd.Series:
    s = fred.get_series(series_id)
    s.name = series_id
    return s

# ────────────────────────────────────────────────
# Data Fetching
# ────────────────────────────────────────────────

def load_full_history_btc(start="2014-01-01"):
    btc = yf.download("BTC-USD", start=start, progress=False)
    btc = btc[["Close"]].rename(columns={"Close": "CBBTCUSD"})
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    btc.index.name = "Date"
    return btc

def get_google_ai_trends(start="2018-01-01"):
    pytrends = TrendReq(hl="en-US", tz=0)
    end_str = datetime.today().strftime("%Y-%m-%d")
    timeframe = f"{start} {end_str}"
    pytrends.build_payload(["ai"], timeframe=timeframe, geo="")
    trends = pytrends.interest_over_time()
    
    if trends.empty:
        raise ValueError("Google Trends returned no data for 'ai'.")
    
    # Rename and clean
    trends = trends.rename(columns={"ai": "AI_Searches"}).drop(columns=["isPartial"], errors="ignore")
    
    # FORCE a single-level date index — this handles any weird MultiIndex from pytrends
    trends = trends.reset_index()                     # Turn everything into columns
    if 'date' in trends.columns:                      # pytrends usually calls it 'date'
        trends = trends.set_index('date')
    elif 'Date' in trends.columns:
        trends = trends.set_index('Date')
    else:
        raise ValueError("No date column found in trends data")
    
    trends.index.name = "Date"
    
    # Clean up datetime (no timezone, ensure it's index)
    trends.index = pd.to_datetime(trends.index).tz_localize(None)
    
    # Keep only the value we care about
    return trends[["AI_Searches"]]

def get_sox_data(start="2018-01-01"):
    sox = yf.download("^SOX", start=start, progress=False)
    sox = sox[["Close"]].rename(columns={"Close": "SOX"})
    sox.index = pd.to_datetime(sox.index).tz_localize(None)
    sox.index.name = "Date"
    return sox

def get_btc_data(start="2018-01-01"):
    btc = yf.download("BTC-USD", start=start, progress=False)
    btc = btc[["Close"]].rename(columns={"Close": "CBBTCUSD"})
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    btc.index.name = "Date"
    return btc

# GLI components
walcl = get_fred_series("WALCL")
rrp   = get_fred_series("RRPONTSYD")
tga   = get_fred_series("WTREGEN")
gli_raw = pd.concat([walcl, rrp, tga], axis=1).dropna()
gli_z = (gli_raw - gli_raw.mean()) / gli_raw.std()
gli = gli_z.mean(axis=1)
gli.name = "GLI"

# ────────────────────────────────────────────────
# Theme & Layout
# ────────────────────────────────────────────────

def register_macro_theme():
    colors = {
        "mpw_blue":   "#4DA3FF",
        "mpw_orange": "#FF8C42",
        "mpw_green":  "#4CAF50",
        "mpw_red":    "#FF5252",
        "mpw_yellow": "#FFC857",
        "mpw_gray":   "#AAAAAA",
    }

    macro_theme = {
        "layout": {
            "paper_bgcolor": "#111111",
            "plot_bgcolor":  "#111111",
            "font": {"family": "Arial", "color": "#FFFFFF", "size": 14},
            "title": {"font": {"size": 26, "color": "#FFFFFF"}, "x": 0.5, "xanchor": "center"},
            "xaxis": {"gridcolor": "#333333", "zerolinecolor": "#333333", "showgrid": True, "showline": False,
                      "tickfont": {"size": 12, "color": "#AAAAAA"}},
            "yaxis": {"gridcolor": "#333333", "zerolinecolor": "#333333", "showgrid": True, "showline": False,
                      "tickfont": {"size": 12, "color": "#AAAAAA"}},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1,
                       "font": {"size": 12, "color": "#FFFFFF"}},
            "margin": {"l": 60, "r": 40, "t": 80, "b": 40},
            "hovermode": "x unified",
        }
    }
    pio.templates["macro_pulse"] = macro_theme
    pio.templates.default = "macro_pulse"
    return colors

# ────────────────────────────────────────────────
# Chart Builders (with resize fix)
# ────────────────────────────────────────────────

def add_resize_script(fig_id: str) -> str:
    """Injects a resize listener for responsiveness"""
    return f"""
    <script>
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize('{fig_id}');
        }});
    </script>
    """

def build_fg_rsi_chart(btc_close: pd.Series, colors: dict) -> go.Figure:
    fg_rsi = compute_rsi(btc_close)
    fg_rsi_df = pd.DataFrame({"FG-RSI": fg_rsi}).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fg_rsi_df.index, y=fg_rsi_df["FG-RSI"], name="FG-RSI",
                             line=dict(color=colors["mpw_blue"], width=1.5)))
    fig.add_hline(y=70, line=dict(color=colors["mpw_red"], dash="dash"), annotation_text="70", annotation_position="top left")
    fig.add_hline(y=30, line=dict(color=colors["mpw_green"], dash="dash"), annotation_text="30", annotation_position="bottom left")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title="Bitcoin Fear & Greed RSI<br><span style='font-size:14px; color:#AAAAAA;'>MacroPulseWeekly</span>")
    return fig

def build_btc_vs_ai_chart(merged: pd.DataFrame, colors: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.index, y=merged["CBBTCUSD"], name="Bitcoin Price",
                             line=dict(color=colors["mpw_orange"], width=2)))
    fig.add_trace(go.Scatter(x=merged.index, y=merged["AI_Searches"], name="Google AI Trend",
                             line=dict(color=colors["mpw_blue"], width=2), yaxis="y2"))
    fig.update_layout(
        title="Bitcoin vs Google AI Trends<br><span style='font-size:14px; color:#AAAAAA;'>MacroPulseWeekly</span>",
        yaxis=dict(title="BTC Price (USD)"),
        yaxis2=dict(title="Google AI Search Interest", overlaying="y", side="right")
    )
    return fig

def build_btc_vs_sox_chart(btc: pd.DataFrame, sox: pd.DataFrame, colors: dict) -> go.Figure:
    df = btc.join(sox, how="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping data between BTC and SOX")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["CBBTCUSD"], name="Bitcoin Price (USD)",
                             line=dict(color=colors["mpw_orange"], width=2), yaxis="y"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SOX"], name="SOX Index",
                             line=dict(color=colors["mpw_blue"], width=2), yaxis="y2"))
    fig.update_layout(
        title="Bitcoin vs SOX Index<br><span style='font-size:14px; color:#aaa;'>MacroPulseWeekly</span>",
        yaxis=dict(title="BTC Price (USD)"),
        yaxis2=dict(title="SOX Index", overlaying="y", side="right")
    )
    return fig

def build_gli_chart(gli: pd.Series, colors: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gli.index, y=gli.values, name="Global Liquidity Index (v1)",
                             line=dict(color="#F2C94C", width=2)))
    fig.update_layout(
        title="Global Liquidity Index (US‑centric v1)",
        xaxis_title="Date",
        yaxis_title="Z‑Score"
    )
    return fig

# ────────────────────────────────────────────────
# Dashboard Builder
# ────────────────────────────────────────────────

def build_dashboard_index(figs_dict: dict):
    # figs_dict = {"fg-rsi": fig, "btc-ai": fig, ...}
    with open("template.html", "r", encoding="utf-8") as f:
        content = f.read()

    for div_id, fig in figs_dict.items():
        # Use latest CDN + full config for interactivity
        html_snippet = fig.to_html(
            include_plotlyjs="https://cdn.plot.ly/plotly-latest.min.js",
            full_html=False,
            config={'responsive': True, 'displaylogo': False}
        )
        # Add resize handler
        html_snippet += add_resize_script(div_id)
        # Replace exactly
        placeholder = f'<div id="{div_id}"></div>'
        content = content.replace(placeholder, f'<div id="{div_id}">{html_snippet}</div>')

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(content)
    print("index.html updated with interactive charts.")

# ────────────────────────────────────────────────
# Main Execution
# ────────────────────────────────────────────────

def main():
    ensure_charts_dir()
    colors = register_macro_theme()

    btc_full = load_full_history_btc()
    btc = get_btc_data(start="2018-01-01")
    trends = get_google_ai_trends(start="2018-01-01")
    sox = get_sox_data(start="2018-01-01")

    # === DEBUG PRINTS START ===
    print("=== DEBUG: btc index type and levels ===")
    print("btc index type:", type(btc.index))
    print("btc levels:", btc.index.nlevels if isinstance(btc.index, pd.MultiIndex) else "single level")
    print("btc index sample:", btc.index[:3])

    print("\n=== DEBUG: trends index type and levels ===")
    print("trends index type:", type(trends.index))
    print("trends levels:", trends.index.nlevels if isinstance(trends.index, pd.MultiIndex) else "single level")
    print("trends index sample:", trends.index[:3])
    print("trends columns:", trends.columns.tolist())

# === DEBUG PRINTS END ===  # (keep this if you want, or remove)

    # Reset both to columns so we can merge on the 'Date' column explicitly
    btc_reset = btc.reset_index()
    trends_reset = trends.reset_index()

    # Merge on the date column (explicit and safe)
    merged_ai = pd.merge(
        btc_reset,
        trends_reset,
        on="Date",          # merge on the column named 'Date'
        how="inner"
    )

    # Now build the charts
    fg_rsi_fig   = build_fg_rsi_chart(btc["CBBTCUSD"], colors)
    btc_ai_fig   = build_btc_vs_ai_chart(merged_ai, colors)
    btc_sox_fig  = build_btc_vs_sox_chart(btc, sox, colors)
    gli_fig      = build_gli_chart(gli, colors)

    # Save individual HTML files if needed
    fg_rsi_fig.write_html("charts/fg_rsi.html", include_plotlyjs="cdn", config={'responsive': True})
    # ... similarly for others

    # Build main index
    figs = {
        "fg-rsi":  fg_rsi_fig,
        "btc-ai":  btc_ai_fig,
        "btc-sox": btc_sox_fig,
        "gli":     gli_fig,
    }
    build_dashboard_index(figs)

if __name__ == "__main__":
    main()
