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
REPO_RAW    = "https://raw.githubusercontent.com/maanitmehta/Macroeconomic_regime_allocation/main/regime_allocation_results"
SIGNAL_URL  = f"{REPO_RAW}/latest_signal.json"
HISTORY_URL = f"{REPO_RAW}/signal_history.csv"

REGIME_COLORS = {"Stress": "#ef4444", "Expansion": "#22c55e", "Baseline": "#3b82f6"}
TICKER_NAMES  = {"SPY": "S&P 500", "IWM": "Russell 2000", "XLU": "Utilities",
                 "XLY": "Cons. Disc.", "XLK": "Technology", "XLF": "Financials"}

REFRESH_MS = 60 * 60 * 1000

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#0f1117"       # page background
SURFACE = "#1a1d27"       # card / chart background
BORDER  = "#2a2d3a"       # card border
TEXT    = "#e2e8f0"       # primary text
MUTED   = "#64748b"       # secondary text
ACCENT  = "#6366f1"       # weight bars (indigo)


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


# ── Shared chart theme ─────────────────────────────────────────────────────────
CHART_THEME = dict(
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    title_font=dict(size=13, color=TEXT),
)


# ── Chart builders ─────────────────────────────────────────────────────────────
def make_proba_chart(signal: dict) -> go.Figure:
    probas = signal["regime_probabilities"]
    labels = list(probas.keys())
    values = [probas[l] * 100 for l in labels]
    colors = [REGIME_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.9,
                    line=dict(color=[c + "44" for c in colors], width=0)),
        text=[f"<b>{v:.1f}%</b>" for v in values],
        textposition="outside", cliponaxis=False,
        textfont=dict(color=TEXT, size=12),
    ))
    fig.update_layout(
        **CHART_THEME,
        title="Posterior Probabilities",
        xaxis=dict(range=[0, 125], showgrid=False, showticklabels=False,
                   zeroline=False, showline=False),
        yaxis=dict(autorange="reversed", showgrid=False,
                   tickfont=dict(size=13, color=TEXT)),
        margin=dict(l=12, r=48, t=44, b=12),
        height=210,
    )
    return fig


def make_weights_chart(signal: dict) -> go.Figure:
    weights = signal["weights_probabilistic"]
    tickers = list(weights.keys())
    values  = [weights[t] * 100 for t in tickers]
    names   = [TICKER_NAMES.get(t, t) for t in tickers]
    bar_colors = [ACCENT if v > 0 else BORDER for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=[f"{t}  {n}" for t, n in zip(tickers, names)],
        orientation="h",
        marker=dict(color=bar_colors, opacity=0.9),
        text=[f"<b>{v:.1f}%</b>" if v > 0 else "" for v in values],
        textposition="outside", cliponaxis=False,
        textfont=dict(color=TEXT, size=12),
    ))
    fig.update_layout(
        **CHART_THEME,
        title="Recommended Weights",
        xaxis=dict(range=[0, 125], showgrid=False, showticklabels=False,
                   zeroline=False, showline=False),
        yaxis=dict(autorange="reversed", showgrid=False,
                   tickfont=dict(size=12, color=TEXT)),
        margin=dict(l=12, r=48, t=44, b=12),
        height=270,
    )
    return fig


def make_history_chart(hist: pd.DataFrame) -> go.Figure:
    if hist is None or hist.empty:
        return go.Figure(layout=dict(**CHART_THEME, height=300,
                                     margin=dict(l=12, r=12, t=44, b=12)))

    prob_cols     = [c for c in hist.columns if c.startswith("p_")]
    regime_labels = [c.replace("p_", "") for c in prob_cols]
    dates         = pd.to_datetime(hist["run_date"]).dt.date.astype(str)

    fig = go.Figure()
    for col, label in zip(prob_cols, regime_labels):
        fig.add_trace(go.Scatter(
            x=dates, y=hist[col] * 100,
            name=label, mode="lines+markers",
            line=dict(color=REGIME_COLORS.get(label, MUTED), width=2.5),
            marker=dict(size=8, line=dict(width=1.5, color=BG)),
        ))
    fig.update_layout(
        **CHART_THEME,
        title="Regime Probability History",
        yaxis=dict(title="Probability (%)", range=[0, 108],
                   gridcolor=BORDER, zeroline=False,
                   tickfont=dict(color=MUTED, size=11)),
        xaxis=dict(title="Signal Date", type="category", tickangle=-30,
                   showgrid=False, tickfont=dict(color=MUTED, size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=12)),
        margin=dict(l=12, r=12, t=52, b=52),
        height=320,
        hovermode="x unified",
    )
    return fig


# ── Shared card wrapper ────────────────────────────────────────────────────────
def card(children, extra_style=None):
    style = {
        "background": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": 12,
        "padding": "20px 24px",
    }
    if extra_style:
        style.update(extra_style)
    return html.Div(style=style, children=children)


# ── App layout ─────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Macro Regime Dashboard")

app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body { background: ''' + BG + '''; color: ''' + TEXT + '''; font-family: Inter, sans-serif; }
      ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: ''' + BG + '''; }
      ::-webkit-scrollbar-thumb { background: ''' + BORDER + '''; border-radius: 3px; }
      details summary { list-style: none; } details summary::-webkit-details-marker { display: none; }
    </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''

app.layout = html.Div(style={"maxWidth": 960, "margin": "0 auto", "padding": "32px 24px"}, children=[

    dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),

    # ── Header ─────────────────────────────────────────────────────────────
    html.Div(style={"marginBottom": 28}, children=[
        html.H1("Macro Regime Dashboard",
                style={"fontSize": 24, "fontWeight": 700, "letterSpacing": "-0.5px",
                       "marginBottom": 6}),
        html.Div(id="subtitle", style={"color": MUTED, "fontSize": 12}),
    ]),

    html.Div(id="error-msg", style={"color": "#ef4444", "fontSize": 12, "marginBottom": 12}),

    # ── Regime card ─────────────────────────────────────────────────────────
    html.Div(id="regime-card", style={"marginBottom": 20}),

    # ── Probabilities + Weights ──────────────────────────────────────────────
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                    "gap": 16, "marginBottom": 16}, children=[
        card(dcc.Graph(id="proba-chart",   config={"displayModeBar": False}),
             {"padding": "12px 16px"}),
        card(dcc.Graph(id="weights-chart", config={"displayModeBar": False}),
             {"padding": "12px 16px"}),
    ]),

    # ── Hard-label disclosure ────────────────────────────────────────────────
    html.Details(style={"marginBottom": 20}, children=[
        html.Summary(style={
            "cursor": "pointer", "fontSize": 12, "color": MUTED,
            "padding": "6px 0", "userSelect": "none",
        }, children="▸  Hard-label weights (for reference)"),
        html.Div(id="hard-weights", style={"marginTop": 10}),
    ]),

    # ── History chart ────────────────────────────────────────────────────────
    card(dcc.Graph(id="history-chart", config={"displayModeBar": False}),
         {"padding": "12px 16px", "marginBottom": 16}),

    # ── Signal history table ─────────────────────────────────────────────────
    card([
        html.Div("Signal History",
                 style={"fontSize": 13, "fontWeight": 600, "marginBottom": 14}),
        html.Div(id="history-table"),
    ]),
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
    signal  = fetch_signal()
    history = fetch_history()

    if signal is None:
        empty = go.Figure(layout=dict(**CHART_THEME))
        return ("", "", empty, empty, "", empty, "", "Could not fetch signal from GitHub.")

    regime = signal["current_regime"]["label"]
    as_of  = signal["as_of"]
    run_dt = signal["run_date"]
    color  = REGIME_COLORS[regime]

    subtitle = f"Last updated {run_dt}  ·  Macro data as of {as_of}  ·  Refreshes hourly"

    regime_card = html.Div(style={
        "background": SURFACE,
        "border": f"1px solid {color}33",
        "borderLeft": f"4px solid {color}",
        "borderRadius": 12,
        "padding": "18px 24px",
        "display": "flex", "alignItems": "center", "gap": 16,
    }, children=[
        html.Div(style={
            "width": 12, "height": 12, "borderRadius": "50%",
            "background": color, "boxShadow": f"0 0 8px {color}88", "flexShrink": 0,
        }),
        html.Div([
            html.Div("CURRENT REGIME",
                     style={"fontSize": 10, "color": MUTED, "letterSpacing": 2,
                            "fontWeight": 600, "marginBottom": 2}),
            html.Div(regime, style={"fontSize": 30, "fontWeight": 700, "color": color,
                                    "letterSpacing": "-0.5px"}),
        ]),
        html.Div(style={"marginLeft": "auto", "textAlign": "right"}, children=[
            html.Div("MACRO DATA AS OF",
                     style={"fontSize": 10, "color": MUTED, "letterSpacing": 2,
                            "fontWeight": 600, "marginBottom": 2}),
            html.Div(as_of, style={"fontSize": 18, "fontWeight": 600, "color": TEXT}),
        ]),
    ])

    proba_fig   = make_proba_chart(signal)
    weights_fig = make_weights_chart(signal)

    hard = signal.get("weights_hard_label", {})
    hard_div = html.Div(
        style={"display": "flex", "gap": 10, "flexWrap": "wrap"},
        children=[
            html.Span(f"{t}: {w*100:.1f}%", style={
                "background": BORDER, "color": TEXT, "borderRadius": 6,
                "padding": "3px 10px", "fontSize": 12,
            })
            for t, w in hard.items()
        ],
    )

    hist_fig = make_history_chart(history)

    if history is not None and not history.empty:
        display_cols = (["run_date", "regime_label"]
                        + [c for c in history.columns if c.startswith("p_")]
                        + [c for c in history.columns if c.startswith("w_prob_")])
        display_cols = [c for c in display_cols if c in history.columns]
        hist_df = history[display_cols].copy().sort_values("run_date", ascending=False)

        def col_label(c):
            if c == "run_date":     return "Date"
            if c == "regime_label": return "Regime"
            if c.startswith("p_"):  return "P(" + c[2:] + ")"
            if c.startswith("w_prob_"): return c[7:]
            return c

        def fmt_cell(c, v):
            if c.startswith(("p_", "w_prob_")):
                try: return f"{float(v):.1%}"
                except: return str(v)
            return str(v)

        header = html.Tr([
            html.Th(col_label(c), style={
                "padding": "8px 12px", "textAlign": "left", "fontSize": 11,
                "color": MUTED, "fontWeight": 600, "letterSpacing": 0.5,
                "borderBottom": f"1px solid {BORDER}", "whiteSpace": "nowrap",
            }) for c in display_cols
        ])
        rows = [
            html.Tr([
                html.Td(fmt_cell(c, row[c]), style={
                    "padding": "7px 12px", "fontSize": 12, "color": TEXT,
                    "borderBottom": f"1px solid {BORDER}22",
                }) for c in display_cols
            ], style={"background": SURFACE if i % 2 == 0 else BG})
            for i, (_, row) in enumerate(hist_df.iterrows())
        ]
        table = html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        )
    else:
        table = html.Div("No history yet.", style={"color": MUTED, "fontSize": 13})

    return subtitle, regime_card, proba_fig, weights_fig, hard_div, hist_fig, table, ""


if __name__ == "__main__":
    app.run(debug=False, port=8053)
