import os
from datetime import datetime
import pandas as pd
import requests
import gzip
import io
import time
import yfinance as yf

from pytrends.request import TrendReq
import plotly.graph_objects as go
import plotly.io as pio

def load_full_history_btc():
    import pandas as pd
    import yfinance as yf

    # Load BTC from yfinance (2014 → now)
    btc = yf.download("BTC-USD", start="2014-01-01", progress=False)

    # --- CRITICAL FIX: Flatten MultiIndex ---
    if isinstance(btc.index, pd.MultiIndex):
        btc.index = btc.index.get_level_values(0)

    # Remove timezone if present
    btc.index = pd.to_datetime(btc.index).tz_localize(None)

    # Rename and clean
    btc = btc.rename(columns={"Close": "CBBTCUSD"})
    btc.index.name = "Date"

    return btc[["CBBTCUSD"]]

from pytrends.request import TrendReq
import plotly.graph_objects as go
import plotly.io as pio

# ================================
# Helpers
# ================================
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

# ================================
# Data fetching
# ================================

def get_btc_data(start="2018-01-01"):
    import pandas as pd
    import yfinance as yf

    btc = yf.download("BTC-USD", start=start, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    # Keep only Close price and rename
    btc = btc[["Close"]].rename(columns={"Close": "CBBTCUSD"})

    # Normalize index
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
    trends = trends.rename(columns={"ai": "AI_Searches"})
    trends = trends.drop(columns=["isPartial"], errors="ignore")
    trends.index = pd.to_datetime(trends.index).tz_localize(None)
    trends.index.name = "Date"
    return trends

def get_sox_data(start="2018-01-01"):
    sox = yf.download("^SOX", start=start, progress=False)
    if sox.empty:
        raise ValueError("No data returned for ^SOX")
    if isinstance(sox.columns, pd.MultiIndex):
        sox.columns = sox.columns.get_level_values(0)
    sox = sox[["Close"]].rename(columns={"Close": "SOX"})
    sox.index = pd.to_datetime(sox.index).tz_localize(None)
    sox.index.name = "Date"
    return sox

# ================================
# Theme
# ================================
def register_macro_theme():
    mpw_blue   = "#4DA3FF"
    mpw_orange = "#FF8C42"
    mpw_green  = "#4CAF50"
    mpw_red    = "#FF5252"
    mpw_yellow = "#FFC857"
    mpw_gray   = "#888888"

    macro_theme = {
        "layout": {
            "paper_bgcolor": "#111111",
            "plot_bgcolor": "#111111",
            "font": {"family": "Arial", "color": "#FFFFFF", "size": 14},
            "title": {"font": {"size": 26, "color": "#FFFFFF"}, "x": 0.5, "xanchor": "center"},
            "xaxis": {"gridcolor": "#333333", "zerolinecolor": "#333333", "showgrid": True, "showline": False, "tickfont": {"size": 12, "color": "#AAAAAA"}},
            "yaxis": {"gridcolor": "#333333", "zerolinecolor": "#333333", "showgrid": True, "showline": False, "tickfont": {"size": 12, "color": "#AAAAAA"}},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"size": 12, "color": "#FFFFFF"}},
            "margin": {"l": 60, "r": 40, "t": 80, "b": 40},
        }
    }
    pio.templates["macro_pulse"] = macro_theme
    pio.templates.default = "macro_pulse"

    return {
        "mpw_blue": mpw_blue, "mpw_orange": mpw_orange, "mpw_green": mpw_green,
        "mpw_red": mpw_red, "mpw_yellow": mpw_yellow, "mpw_gray": mpw_gray,
    }

# ================================
# Chart builders
# ================================
def build_fg_rsi_chart(btc_close: pd.Series, colors: dict) -> go.Figure:
    fg_rsi_series = compute_rsi(btc_close, window=14)
    fg_rsi_df = pd.DataFrame({"fg_rsi": fg_rsi_series}).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fg_rsi_df.index, y=fg_rsi_df["fg_rsi"], name="FG-RSI",
        line=dict(color=colors["mpw_blue"], width=1.5),
    ))
    fig.add_hline(y=70, line=dict(color=colors["mpw_red"], width=1, dash="dash"), annotation_text="70", annotation_position="top left", annotation_font_color="#AAAAAA")
    fig.add_hline(y=30, line=dict(color=colors["mpw_green"], width=1, dash="dash"), annotation_text="30", annotation_position="bottom left", annotation_font_color="#AAAAAA")
    fig.update_xaxes(tickformat="%Y-%m-%d", showticklabels=True, tickfont=dict(size=12, color="#AAAAAA"))
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title="Bitcoin Fear & Greed RSI<br><span style='font-size:14px; color:#AAAAAA;'>MacroPulseWeekly</span>")
    return fig

def build_btc_vs_ai_chart(merged: pd.DataFrame, colors: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.index, y=merged["CBBTCUSD"], name="Bitcoin Price", line=dict(color=colors["mpw_orange"], width=2)))
    fig.add_trace(go.Scatter(x=merged.index, y=merged["AI_Searches"], name="Google AI Trend", line=dict(color=colors["mpw_blue"], width=2), yaxis="y2"))
    fig.update_layout(
        title="Bitcoin vs Google AI Trends<br><span style='font-size:14px; color:#AAAAAA;'>MacroPulseWeekly</span>",
        yaxis=dict(title="BTC Price (USD)"),
        yaxis2=dict(title="Google AI Search Interest", overlaying="y", side="right"),
        xaxis=dict(tickformat="%Y-%m-%d", showticklabels=True),
    )
    return fig

def build_clean_rsi_chart(btc_close: pd.Series, colors: dict) -> go.Figure:
    rsi_series = compute_rsi(btc_close, window=14)
    rsi_df = pd.DataFrame({"rsi": rsi_series}).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi_df.index, y=rsi_df["rsi"], name="RSI", line=dict(color=colors["mpw_yellow"], width=2)))
    fig.add_hline(y=70, line=dict(color=colors["mpw_red"], width=1, dash="dash"), annotation_text="70", annotation_position="top left", annotation_font_color="#AAAAAA")
    fig.add_hline(y=30, line=dict(color=colors["mpw_green"], width=1, dash="dash"), annotation_text="30", annotation_position="bottom left", annotation_font_color="#AAAAAA")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title="Bitcoin RSI<br><span style='font-size:14px; color:#AAAAAA;'>MacroPulseWeekly</span>")
    return fig

def build_btc_vs_sox_chart(btc: pd.DataFrame, sox: pd.DataFrame, colors: dict) -> go.Figure:
    print("BTC index levels:", btc.index.nlevels)
    print("SOX index levels:", sox.index.nlevels)
    print("BTC index sample:", btc.index[:5])
    print("SOX index sample:", sox.index[:5])
    print("SOX columns:", sox.columns)

    # Join on index (safer than merge when indices are aligned)
    df = btc[["CBBTCUSD"]].join(sox[["SOX"]], how="inner")

    if df.empty:
        raise ValueError("Merged BTC-SOX DataFrame is empty — check date overlap!")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["CBBTCUSD"], mode="lines", name="Bitcoin Price (USD)",
        line=dict(color=colors["mpw_orange"], width=2), yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SOX"], mode="lines", name="SOX Index",
        line=dict(color=colors["mpw_blue"], width=2), yaxis="y2"
    ))

    fig.update_layout(
        title=dict(
            text="Bitcoin vs SOX Index<br><span style='font-size:14px; color:#aaa;'>MacroPulseWeekly</span>",
            x=0.5, xanchor="center"
        ),
        paper_bgcolor="#111", plot_bgcolor="#111", font=dict(color="white"),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=80, b=40),
        xaxis=dict(showgrid=True, gridcolor="#333", zeroline=False),
        yaxis=dict(title="BTC Price (USD)", showgrid=True, gridcolor="#333", zeroline=False),
        yaxis2=dict(title="SOX Index", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ================================
# Dashboard index builder
# ================================
def _inject_block(content: str, start_marker: str, end_marker: str, block_html: str) -> str:
    start = content.find(start_marker)
    end = content.find(end_marker)
    if start == -1 or end == -1:
        print(f"ERROR: Markers {start_marker} / {end_marker} not found in index.html")
        return content
    start_end = start + len(start_marker)
    return content[:start_end] + "\n" + block_html + "\n" + content[end:]

def build_dashboard_index(fg_rsi_fig: go.Figure, btc_ai_fig: go.Figure, btc_sox_fig: go.Figure) -> None:
    fg_rsi_html   = fg_rsi_fig.to_html(full_html=False, include_plotlyjs="cdn")
    btc_ai_html   = btc_ai_fig.to_html(full_html=False, include_plotlyjs=False)
    btc_sox_html  = btc_sox_fig.to_html(full_html=False, include_plotlyjs=False)

    with open("index.html", "r", encoding="utf-8") as f:
        content = f.read()

    content = _inject_block(content, "<!-- FG_RSI_START -->", "<!-- FG_RSI_END -->", fg_rsi_html)
    content = _inject_block(content, "<!-- BTC_AI_START -->", "<!-- BTC_AI_END -->", btc_ai_html)
    content = _inject_block(content, "<!-- BTC_SOX_START -->", "<!-- BTC_SOX_END -->", btc_sox_html)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(content)
    print("Dashboard index updated with embedded charts.")

# ================================
# Main
# ================================
def main():
    ensure_charts_dir()
    colors = register_macro_theme()

    btc = load_full_history_btc()
    #btc     = get_btc_data(start="2018-01-01")
    trends  = get_google_ai_trends(start="2018-01-01")
    sox     = get_sox_data(start="2018-01-01")

    merged = btc.join(trends, how="inner").dropna(subset=["CBBTCUSD", "AI_Searches"])

    fg_rsi_fig   = build_fg_rsi_chart(btc["CBBTCUSD"], colors)
    btc_ai_fig   = build_btc_vs_ai_chart(merged, colors)
    btc_sox_fig  = build_btc_vs_sox_chart(btc, sox, colors)
    

    fg_rsi_fig.write_html("charts/fg_rsi.html",   include_plotlyjs="cdn", full_html=False)
    btc_ai_fig.write_html("charts/btc_vs_google_ai.html", include_plotlyjs="cdn", full_html=False)
    btc_sox_fig.write_html("charts/btc_sox.html", include_plotlyjs="cdn", full_html=False)
    

    build_dashboard_index(fg_rsi_fig, btc_ai_fig, btc_sox_fig)

if __name__ == "__main__":
    main()
