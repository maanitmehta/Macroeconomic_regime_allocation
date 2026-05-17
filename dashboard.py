"""
Macro Regime Dashboard
======================
Reads latest_signal.json and signal_history.csv directly from the GitHub repo.
The remote routine commits fresh data on the 1st of each month; this dashboard
picks it up automatically within the hour via a background interval refresh.

Run:
    python dashboard.py
    open http://localhost:8053
"""

import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from dash import Dash, html, dcc, callback, Output, Input

# ── GitHub raw URLs ────────────────────────────────────────────────────────────
REPO_RAW = "https://raw.githubusercontent.com/maanitmehta/Macroeconomic_regime_allocation/main/regime_allocation_results"
SIGNAL_URL  = f"{REPO_RAW}/latest_signal.json"
HISTORY_URL = f"{REPO_RAW}/signal_history.csv"

REGIME_COLORS  = {"Stress": "#d62728", "Expansion": "#2ca02c", "Baseline": "#1f77b4"}
REGIME_BG      = {"Stress": "#fde8e8", "Expansion": "#e8f5e9", "Baseline": "#e3edf7"}
TICKER_NAMES   = {"SPY": "S&P 500", "IWM": "Russell 2000", "XLU": "Utilities",
                  "XLY": "Cons. Disc.", "XLK": "Technology", "XLF": "Financials"}

REFRESH_MS = 60 * 60 * 1000   # re-fetch from GitHub every hour


# ── Data fetchers ──────────────────────────────────────────────────────────────
def fetch_signal() -> Optional[dict]:
    try:
        r = requests.get(SIGNAL_URL, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_history() -> Optional[pd.DataFrame]:
    try:
        r = requests.get(HISTORY_URL, timeout=10)
        r.raise_for_status()
        return pd.read_csv(pd.io.common.StringIO(r.text))
    except Exception:
        return None


# ── Chart builders ─────────────────────────────────────────────────────────────
def make_proba_chart(signal: dict) -> go.Figure:
    probas = signal["regime_probabilities"]
    labels = list(probas.keys())
    values = [probas[l] * 100 for l in labels]
    colors = [REGIME_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in values],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        title="Regime Posterior Probabilities",
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=40, t=40, b=10),
        height=200, plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def make_weights_chart(signal: dict) -> go.Figure:
    weights = signal["weights_probabilistic"]
    tickers = list(weights.keys())
    values  = [weights[t] * 100 for t in tickers]
    names   = [TICKER_NAMES.get(t, t) for t in tickers]
    colors  = ["#4e79a7" if v > 0 else "#d3d3d3" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=[f"{t}  {n}" for t, n in zip(tickers, names)],
        orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        title="Recommended Weights (Probabilistic Blend)",
        xaxis=dict(range=[0, 120], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=40, t=40, b=10),
        height=260, plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def make_history_chart(hist: pd.DataFrame) -> go.Figure:
    if hist is None or hist.empty:
        return go.Figure()

    prob_cols = [c for c in hist.columns if c.startswith("p_")]
    regime_labels = [c.replace("p_", "") for c in prob_cols]

    # Parse to date so plotly shows YYYY-MM-DD ticks, not timestamps
    dates = pd.to_datetime(hist["run_date"]).dt.date.astype(str)

    fig = go.Figure()
    for col, label in zip(prob_cols, regime_labels):
        fig.add_trace(go.Scatter(
            x=dates, y=hist[col] * 100,
            name=label, mode="lines+markers",
            line=dict(color=REGIME_COLORS.get(label, "#888"), width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        title="Regime Probability History",
        yaxis=dict(title="Probability (%)", range=[0, 105], gridcolor="#f0f0f0"),
        xaxis=dict(title="Signal Date", type="category", tickangle=-30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=50),
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    return fig


# ── App layout ─────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Macro Regime Dashboard")

app.layout = html.Div(style={"fontFamily": "Inter, sans-serif", "maxWidth": 900,
                              "margin": "0 auto", "padding": "24px"}, children=[

    dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),

    html.H1("Macro Regime Dashboard",
            style={"fontSize": 26, "fontWeight": 700, "marginBottom": 4}),
    html.Div(id="subtitle", style={"color": "#666", "marginBottom": 24, "fontSize": 13}),

    # ── Current regime card ────────────────────────────────────────────────
    html.Div(id="regime-card", style={"marginBottom": 24}),

    # ── Probability + weights charts (side-by-side) ────────────────────────
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": 20,
                    "marginBottom": 24}, children=[
        dcc.Graph(id="proba-chart", config={"displayModeBar": False}),
        dcc.Graph(id="weights-chart", config={"displayModeBar": False}),
    ]),

    # ── Hard-label weights comparison ─────────────────────────────────────
    html.Details(style={"marginBottom": 24, "fontSize": 13, "color": "#444"}, children=[
        html.Summary("Hard-label weights (for reference)", style={"cursor": "pointer"}),
        html.Div(id="hard-weights"),
    ]),

    # ── History chart ──────────────────────────────────────────────────────
    dcc.Graph(id="history-chart", config={"displayModeBar": False}),

    # ── Signal history table ───────────────────────────────────────────────
    html.H3("Signal History", style={"marginTop": 28, "marginBottom": 8, "fontSize": 15}),
    html.Div(id="history-table"),

    html.Div(id="error-msg", style={"color": "red", "fontSize": 12, "marginTop": 16}),
])


# ── Callback ───────────────────────────────────────────────────────────────────
@callback(
    Output("subtitle",      "children"),
    Output("regime-card",   "children"),
    Output("proba-chart",   "figure"),
    Output("weights-chart", "figure"),
    Output("hard-weights",  "children"),
    Output("history-chart", "figure"),
    Output("history-table", "children"),
    Output("error-msg",     "children"),
    Input("refresh",        "n_intervals"),
)
def update(__n):
    signal = fetch_signal()
    history = fetch_history()

    if signal is None:
        empty = go.Figure()
        return ("", "", empty, empty, "", empty, "", "Could not fetch signal from GitHub.")

    regime  = signal["current_regime"]["label"]
    as_of   = signal["as_of"]
    run_dt  = signal["run_date"]
    color   = REGIME_COLORS[regime]
    bg      = REGIME_BG[regime]

    subtitle = f"Last updated {run_dt}  ·  Macro data as of {as_of}  ·  Auto-refreshes hourly"

    regime_card = html.Div(style={
        "background": bg, "border": f"2px solid {color}",
        "borderRadius": 10, "padding": "16px 24px",
        "display": "flex", "alignItems": "center", "gap": 16,
    }, children=[
        html.Div(style={
            "width": 16, "height": 16, "borderRadius": "50%",
            "background": color, "flexShrink": 0,
        }),
        html.Div([
            html.Div("Current Regime", style={"fontSize": 11, "color": "#555",
                                               "textTransform": "uppercase", "letterSpacing": 1}),
            html.Div(regime, style={"fontSize": 28, "fontWeight": 700, "color": color}),
        ]),
    ])

    proba_fig   = make_proba_chart(signal)
    weights_fig = make_weights_chart(signal)

    hard = signal.get("weights_hard_label", {})
    hard_div = html.Div(style={"marginTop": 8, "display": "flex", "gap": 16, "flexWrap": "wrap"},
                        children=[
        html.Span(f"{t}: {w*100:.1f}%", style={"background": "#f4f4f4", "borderRadius": 4,
                                                 "padding": "2px 8px", "fontSize": 12})
        for t, w in hard.items()
    ])

    hist_fig = make_history_chart(history)

    if history is not None and not history.empty:
        display_cols = ["run_date", "regime_label"] + \
                       [c for c in history.columns if c.startswith("p_")] + \
                       [c for c in history.columns if c.startswith("w_prob_")]
        display_cols = [c for c in display_cols if c in history.columns]
        hist_df = history[display_cols].copy().sort_values("run_date", ascending=False)

        header = html.Tr([html.Th(c.replace("p_", "P(").replace("w_prob_", "W: ")
                                   .rstrip(")") + (")" if c.startswith("p_") else ""),
                                  style={"padding": "6px 10px", "textAlign": "left",
                                         "fontSize": 11, "color": "#555", "background": "#f7f7f7"})
                          for c in display_cols])
        rows = [
            html.Tr([
                html.Td(str(row[c]) if not str(row[c]).replace(".", "").isdigit()
                        else f"{float(row[c]):.1%}" if c.startswith(("p_", "w_")) else str(row[c]),
                        style={"padding": "5px 10px", "fontSize": 12,
                               "borderBottom": "1px solid #eee"})
                for c in display_cols
            ], style={"background": "white" if i % 2 == 0 else "#fafafa"})
            for i, (_, row) in enumerate(hist_df.iterrows())
        ]
        table = html.Table([html.Thead(header), html.Tbody(rows)],
                           style={"width": "100%", "borderCollapse": "collapse",
                                  "border": "1px solid #e5e5e5", "borderRadius": 6})
    else:
        table = html.Div("No history yet.", style={"color": "#888", "fontSize": 13})

    return subtitle, regime_card, proba_fig, weights_fig, hard_div, hist_fig, table, ""


if __name__ == "__main__":
    app.run(debug=False, port=8053)
