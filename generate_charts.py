import os
from datetime import datetime
import pandas as pd
import yfinance as yf
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

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ================================
# Data fetching
# ================================

def get_btc_data(start="2018-01-01"):
    btc = yf.download("BTC-USD", start=start)
    btc = btc[["Close"]].rename(columns={"Close": "CBBTCUSD"})

    # ⭐ Flatten MultiIndex columns (GitHub Actions issue)
    btc.columns = btc.columns.get_level_values(0)

    btc.index = btc.index.to_flat_index()
    btc.index = pd.to_datetime(btc.index)
    btc.index = btc.index.tz_localize(None)
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

    # ⭐ FIX: flatten index BEFORE tz_localize
    trends.index = trends.index.to_flat_index()
    trends.index = pd.to_datetime(trends.index)
    trends.index = trends.index.tz_localize(None)
    trends.index.name = "Date"

    return trends
    
def get_china_deflator_from_worldbank():
    import requests
    import pandas as pd

    # World Bank API endpoints
    nominal_url = "https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.MKTP.KN.ZG?format=json&per_page=500"
    real_url = "https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.MKTP.KD.ZG?format=json&per_page=500"

    # Fetch nominal GDP growth (%)
    nominal_response = requests.get(nominal_url).json()
    nominal_data = nominal_response[1]
    nominal_df = pd.DataFrame(nominal_data)[["date", "value"]]
    nominal_df.columns = ["Year", "Nominal"]

    # Fetch real GDP growth (%)
    real_response = requests.get(real_url).json()
    real_data = real_response[1]
    real_df = pd.DataFrame(real_data)[["date", "value"]]
    real_df.columns = ["Year", "Real"]

    # Merge the two datasets
    df = nominal_df.merge(real_df, on="Year", how="inner")

    # Convert types
    df["Year"] = df["Year"].astype(int)
    df["Nominal"] = pd.to_numeric(df["Nominal"], errors="coerce")
    df["Real"] = pd.to_numeric(df["Real"], errors="coerce")

    # Compute GDP deflator = nominal growth - real growth
    df["Deflator"] = df["Nominal"] - df["Real"]

    # Sort oldest → newest
    df = df.sort_values("Year")

    return df[["Year", "Deflator"]]


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
            "title": {
                "font": {"size": 26, "color": "#FFFFFF"},
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis": {
                "gridcolor": "#333333",
                "zerolinecolor": "#333333",
                "showgrid": True,
                "showline": False,
                "tickfont": {"size": 12, "color": "#AAAAAA"},
            },
            "yaxis": {
                "gridcolor": "#333333",
                "zerolinecolor": "#333333",
                "showgrid": True,
                "showline": False,
                "tickfont": {"size": 12, "color": "#AAAAAA"},
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "font": {"size": 12, "color": "#FFFFFF"},
            },
            "margin": {"l": 60, "r": 40, "t": 80, "b": 40},
        }
    }

    pio.templates["macro_pulse"] = macro_theme
    pio.templates.default = "macro_pulse"

    return {
        "mpw_blue": mpw_blue,
        "mpw_orange": mpw_orange,
        "mpw_green": mpw_green,
        "mpw_red": mpw_red,
        "mpw_yellow": mpw_yellow,
        "mpw_gray": mpw_gray,
    }


# ================================
# Chart builders
# ================================

def build_fg_rsi_chart(btc_close: pd.Series, colors: dict) -> go.Figure:
    fg_rsi_series = compute_rsi(btc_close, window=14)
    fg_rsi_df = pd.DataFrame({
        "observation_date": fg_rsi_series.index,
        "fg_rsi": fg_rsi_series
    }).dropna()

    fg_rsi_df = fg_rsi_df.set_index("observation_date")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fg_rsi_df.index,
        y=fg_rsi_df["fg_rsi"],
        name="FG-RSI",
        line=dict(color=colors["mpw_blue"], width=1.5),
    ))

    fig.add_hline(
        y=70,
        line=dict(color=colors["mpw_red"], width=1, dash="dash"),
        annotation_text="70",
        annotation_position="top left",
        annotation_font_color="#AAAAAA",
    )

    fig.add_hline(
        y=30,
        line=dict(color=colors["mpw_green"], width=1, dash="dash"),
        annotation_text="30",
        annotation_position="bottom left",
        annotation_font_color="#AAAAAA",
    )

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        showticklabels=True,
        tickfont=dict(size=12, color="#AAAAAA"),
    )

    fig.update_yaxes(range=[0, 100])

    fig.update_layout(
        title="Bitcoin Fear & Greed RSI<br><span style='font-size:14px; color:#AAAAAA;'>MacroPulseWeekly</span>"
    )

    return fig


def build_btc_vs_ai_chart(merged: pd.DataFrame, colors: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged.index,
        y=merged["CBBTCUSD"],
        name="Bitcoin Price",
        line=dict(color=colors["mpw_blue"], width=2),
    ))

    fig.add_trace(go.Scatter(
        x=merged.index,
        y=merged["AI_Searches"],
        name="Google AI Trend",
        line=dict(color=colors["mpw_orange"], width=2),
        yaxis="y2",
    ))

    fig.update_layout(
        title="Bitcoin vs Google AI Trends<br><span style='font-size:14px; color:#AAAAAA;'>MacroPulseWeekly</span>",
        yaxis=dict(title="BTC Price (USD)"),
        yaxis2=dict(
            title="Google AI Search Interest",
            overlaying="y",
            side="right",
        ),
    )

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        showticklabels=True,
    )

    return fig

def build_china_deflation_chart(df, colors):
    fig = go.Figure()

    # Color bars: red for negative, accent color for positive
    bar_colors = [
        "#d62728" if val < 0 else colors["accent"]
        for val in df["Deflator"]
    ]

    fig.add_trace(go.Bar(
        x=df["Year"],
        y=df["Deflator"],
        marker_color=bar_colors
    ))

    fig.update_layout(
        title="China Grapples With Longest Deflation Streak in Decades",
        yaxis_title="GDP Deflator (%)",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig


# ================================
# Main
# ================================

def main():
    ensure_charts_dir()
    colors = register_macro_theme()

    # Fetch data
    btc = get_btc_data(start="2018-01-01")
    trends = get_google_ai_trends(start="2018-01-01")

    # Fix MultiIndex issue on GitHub Actions
    trends.columns = trends.columns.get_level_values(0)
    trends.index = trends.index.to_flat_index()
    trends.index = pd.to_datetime(trends.index)
    trends.index = trends.index.tz_localize(None)
    trends.index.name = "Date"

    # Align BTC and AI trends
    merged = btc.join(trends, how="inner")
    merged = merged.dropna(subset=["CBBTCUSD", "AI_Searches"])

    # Build charts
    fg_rsi_fig = build_fg_rsi_chart(btc["CBBTCUSD"], colors)
    btc_ai_fig = build_btc_vs_ai_chart(merged, colors)

    # Save charts
    fg_rsi_fig.write_html("charts/fg_rsi.html", include_plotlyjs="cdn", full_html=False)
    btc_ai_fig.write_html("charts/btc_vs_google_ai.html", include_plotlyjs="cdn", full_html=False)

    # === China GDP Deflator Chart ===
    china = get_china_deflator_from_worldbank()
    china_fig = build_china_deflation_chart(china, colors)

    china_fig.write_html(
        "charts/china_deflation.html",
        include_plotlyjs="cdn",
        full_html=False
    )


if __name__ == "__main__":
    main()



