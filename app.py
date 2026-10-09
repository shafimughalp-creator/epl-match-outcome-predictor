"""
Premier League Match Outcome Predictor  -  Streamlit app
=========================================================
Files this app expects in the same folder:
    app.py            <- this file
    team_logos.py     <- crest URLs / display names (edit this to swap logos)
    predict_logic.py  <- YOUR code: predict_fixture() and get_available_teams()
    (plus whatever predict_logic.py needs: saved model, historical CSVs, ...)

If predict_logic.py cannot be imported, the app switches to a clearly labelled
DEMO MODE with placeholder numbers so you can still preview the whole UI.

Layout of this file
    1. Config & constants
    2. CSS (heavily commented - this is where the look lives)
    3. Helpers (crests, model call, maths, HTML builders)
    4. Page sections (header, fixture card, results, history, about, footer)
    5. main()
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from html import escape
from threading import Thread

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from team_logos import DISPLAY_NAMES, TEAM_CODES, TEAM_LOGOS, TLA, canonical_team

# set_page_config must be the first Streamlit call.
st.set_page_config(
    page_title="Premier League Match Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Your prediction code. Wrapped so the UI still loads if something is missing.
try:
    from predict_logic import get_available_teams, predict_fixture

    MODEL_READY, MODEL_ERROR = True, ""
except Exception as exc:  # noqa: BLE001 - we want to show *any* import problem
    MODEL_READY, MODEL_ERROR = False, f"{type(exc).__name__}: {exc}"


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG & CONSTANTS  (edit these freely)
# ═════════════════════════════════════════════════════════════════════════════
APP_VERSION = "v1.0"
TEST_LOG_LOSS = 1.0238
UNIFORM_LOG_LOSS = math.log(3)  # 1.0986 - what you score by always guessing 1/3 each
GITHUB_MODEL_CARD_URL = "https://github.com/YOUR-USERNAME/YOUR-REPO#model-card"  # <- change me

# Palette (kept in Python too, for the Plotly chart)
GREEN, PURPLE, PINK = "#00ff87", "#38003c", "#e90052"
BG, BG2, BG3 = "#0a0e27", "#151b3d", "#1e2547"
TEXT, TEXT2, MUTED, BORDER = "#ffffff", "#b8c1d9", "#6b7593", "#2a3358"

# Facts shown in the "About the model" panel. Edit to match your real model card.
MODEL_CARD_FACTS = [
    ("Model", "Multinomial logistic regression"),
    ("Regularisation", "Lasso (L1) penalty"),
    ("Inputs", "9 delta features comparing the two sides"),
    ("Target", "Home win / Draw / Away win"),
    ("Test set", "2025/26 season, locked until final evaluation"),
    ("Test log loss", f"{TEST_LOG_LOSS:.4f} (lower is better)"),
]
MODEL_CARD_LIMITS = [
    "Built from historical match data, so it cannot see lineups, injuries or team news on the day.",
    "Draws are the hardest outcome to call; expect them to sit in the middle of most predictions.",
    "Probabilities are estimates, not guarantees. Football is a high-variance sport.",
]


# ═════════════════════════════════════════════════════════════════════════════
# 2. CSS  -  every visible element is styled here
# ═════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

/* ── 2.0 DESIGN TOKENS ────────────────────────────────────────────────────
   Change a colour here and it changes everywhere. */
:root {
  --bg: #0a0e27;           /* page background            */
  --bg2: #151b3d;          /* panels                     */
  --bg3: #1e2547;          /* cards                      */
  --green: #00ff87;        /* accent primary             */
  --purple: #38003c;       /* accent secondary           */
  --pink: #e90052;         /* accent tertiary            */
  --text: #ffffff;
  --text2: #b8c1d9;
  --muted: #6b7593;
  --border: #2a3358;
  --warn: #ffb800;
  --font: 'Manrope', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

/* ── 2.1 PAGE BASE ───────────────────────────────────────────────────────
   Background glow + font. Streamlit's own chrome is hidden. */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text);
}
.stApp {
  font-family: var(--font);
  background:
    radial-gradient(1100px 560px at 88% -8%, rgba(56, 0, 60, 0.60), transparent 62%),
    radial-gradient(900px 520px at -8% 38%, rgba(0, 255, 135, 0.055), transparent 58%),
    var(--bg) !important;
}
.stApp p, .stApp label, .stApp input, .stApp button, .stApp li,
.stApp td, .stApp th, .stApp textarea { font-family: var(--font) !important; }

#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display: none !important;
}
[data-testid="stAppViewContainer"] { overflow-x: hidden; }
.block-container {
  max-width: 1180px !important;
  padding: 0 1.5rem 3rem !important;
}
.stApp :focus-visible { outline: 2px solid var(--green); outline-offset: 2px; }

/* ── 2.2 HEADER BANNER (full-bleed) ──────────────────────────────────────
   width:100vw + negative margin lets the banner escape Streamlit's centred column. */
.hdr {
  width: 100vw;
  margin-left: calc(50% - 50vw);
  position: relative;
  overflow: hidden;
  background: linear-gradient(115deg, #38003c 0%, #1d0a47 40%, #0a0e27 100%);
  border-bottom: 1px solid var(--border);
}
.hdr::after {                       /* soft green glow top-right */
  content: "";
  position: absolute; right: -70px; top: -90px;
  width: 360px; height: 360px; border-radius: 50%;
  background: radial-gradient(circle, rgba(0, 255, 135, 0.20), transparent 65%);
  pointer-events: none;
}
.hdr-in {
  position: relative; z-index: 1;
  max-width: 1180px; margin: 0 auto; padding: 30px 1.5rem;
  display: flex; align-items: center; justify-content: space-between;
  gap: 20px; flex-wrap: wrap;
}
.hdr-left { display: flex; align-items: center; gap: 18px; }
.hdr-badge {                        /* circular football badge */
  width: 68px; height: 68px; border-radius: 50%; flex: none;
  display: flex; align-items: center; justify-content: center;
  font-size: 34px;
  background: radial-gradient(circle at 30% 25%, #2a1a5e, #0a0e27 72%);
  border: 2px solid var(--green);
  box-shadow: 0 0 0 6px rgba(0, 255, 135, 0.10), 0 10px 30px rgba(0, 255, 135, 0.25);
}
.hdr-kicker {
  font-size: 0.8rem; font-weight: 800; letter-spacing: 0.34em;
  text-transform: uppercase; color: var(--green);
}
.hdr-title {
  font-size: clamp(1.8rem, 4.2vw, 2.9rem); font-weight: 800;
  letter-spacing: -0.035em; line-height: 1.05; color: var(--text); margin-top: 4px;
}
.hdr-right { display: flex; gap: 10px; flex-wrap: wrap; }
.chip {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 8px 15px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em;
}
.chip-ver { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); color: var(--text); }
.chip-ll  { background: rgba(0,255,135,0.10); border: 1px solid rgba(0,255,135,0.45); color: var(--green); }
.chip-ll b { color: var(--text2); font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.68rem; }

/* ── 2.3 SECTION LABELS ──────────────────────────────────────────────────*/
.sec-label {
  display: flex; align-items: center; gap: 14px;
  font-size: 0.72rem; font-weight: 800; letter-spacing: 0.24em;
  text-transform: uppercase; color: var(--muted);
}
.sec-label::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.sec-wrap { margin: 34px 0 14px; }

/* ── 2.4 FIXTURE SELECTOR CARD ───────────────────────────────────────────
   st.container(key="fixture_card") gets the class .st-key-fixture_card */
.st-key-fixture_card {
  margin-top: 30px;
  padding: 28px 30px 34px;
  background: linear-gradient(180deg, var(--bg3) 0%, var(--bg2) 100%);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow: 0 26px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
}
[data-testid="stWidgetLabel"] p {
  font-size: 0.72rem !important; font-weight: 800 !important;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--text2) !important;
}
/* select boxes */
div[data-baseweb="select"] > div {
  background: var(--bg) !important; border: 1px solid var(--border) !important;
  border-radius: 14px !important; min-height: 58px;
  transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-baseweb="select"] > div:hover { border-color: rgba(0,255,135,0.55) !important; }
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--green) !important; box-shadow: 0 0 0 3px rgba(0,255,135,0.16);
}
div[data-baseweb="select"] span, div[data-baseweb="select"] input {
  font-size: 1.05rem; font-weight: 700; color: var(--text) !important;
}
div[data-baseweb="select"] svg { fill: var(--green); }
/* dropdown list */
div[data-baseweb="popover"] > div, div[data-baseweb="menu"] {
  background: var(--bg2) !important; border: 1px solid var(--border); border-radius: 14px !important;
}
li[role="option"] { color: var(--text2) !important; font-weight: 600; }
li[role="option"]:hover, li[aria-selected="true"] {
  background: rgba(0,255,135,0.10) !important; color: var(--text) !important;
}
/* date input */
div[data-baseweb="input"] {
  background: var(--bg) !important; border: 1px solid var(--border) !important;
  border-radius: 14px !important; min-height: 58px; overflow: hidden;
  transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-baseweb="input"]:hover { border-color: rgba(0,255,135,0.55) !important; }
div[data-baseweb="input"]:focus-within {
  border-color: var(--green) !important; box-shadow: 0 0 0 3px rgba(0,255,135,0.16);
}
div[data-baseweb="base-input"] { background: transparent !important; }
div[data-baseweb="input"] input {
  color: var(--text) !important; font-weight: 700; font-size: 1.05rem;
  text-align: center; letter-spacing: 0.04em;
}
div[data-baseweb="calendar"] { background: var(--bg2) !important; color: var(--text) !important; }
div[data-baseweb="calendar"] [aria-selected="true"] { color: var(--bg) !important; font-weight: 800; }

.sel-crest { display: flex; align-items: center; justify-content: center; height: 58px; margin-top: 30px; }
.vs-col { display: flex; justify-content: center; padding-top: 30px; }
.vs-badge {                          /* purple -> pink "VS" disc between the dropdowns */
  width: 58px; height: 58px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.9rem; letter-spacing: 0.1em; color: var(--text);
  background: linear-gradient(135deg, var(--purple), var(--pink));
  box-shadow: 0 8px 24px rgba(233,0,82,0.30);
}
.crest { object-fit: contain; display: block; filter: drop-shadow(0 6px 14px rgba(0,0,0,0.45)); }
.crest-pair { display: flex; align-items: center; justify-content: center; }
.crest-pair .crest + .crest { margin-left: -10px; }

/* ── 2.5 PREDICT BUTTON ──────────────────────────────────────────────────*/
.st-key-predict_btn, .st-key-predict_btn [data-testid="stButton"] { width: 100%; }
.st-key-predict_btn button {
  width: 100%; min-height: 66px; border: 0; border-radius: 16px;
  background: linear-gradient(135deg, #00ff87 0%, #00d9a0 100%);
  box-shadow: 0 12px 32px rgba(0,255,135,0.28);
  transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
.st-key-predict_btn button p {
  color: var(--bg) !important; font-weight: 800 !important; font-size: 1.1rem !important;
  letter-spacing: 0.18em; text-transform: uppercase;
}
.st-key-predict_btn button:hover {
  transform: translateY(-2px); filter: brightness(1.06);
  box-shadow: 0 18px 42px rgba(0,255,135,0.40);
}
.st-key-predict_btn button:active { transform: translateY(0); }
.st-key-predict_btn button:disabled {
  opacity: 0.35; box-shadow: none; cursor: not-allowed; transform: none; filter: none;
}

/* ── 2.6 ALERTS & EMPTY STATE ────────────────────────────────────────────*/
.alert {
  margin-top: 14px; padding: 12px 16px; border-radius: 12px;
  font-size: 0.9rem; font-weight: 600; line-height: 1.5;
  background: rgba(255,184,0,0.10); border: 1px solid rgba(255,184,0,0.40); color: var(--warn);
}
.alert-error { background: rgba(233,0,82,0.10); border-color: rgba(233,0,82,0.5); color: #ff7aa8; }
.demo-banner {
  margin-top: 14px; padding: 10px 16px; border-radius: 12px; font-size: 0.82rem; font-weight: 700;
  background: rgba(255,184,0,0.10); border: 1px dashed rgba(255,184,0,0.55); color: var(--warn);
}
.empty {
  margin-top: 30px; padding: 40px 24px; text-align: center;
  border: 1px dashed var(--border); border-radius: 20px; color: var(--text2);
  font-size: 1.02rem; line-height: 1.6;
}
.empty b { color: var(--text); }

/* ── 2.7 RESULT PANEL ────────────────────────────────────────────────────*/
.fade-in { animation: fadeIn .7s cubic-bezier(.2,.7,.2,1) both; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

.fixture-line { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.fixture-name { font-size: clamp(1.3rem, 2.8vw, 1.8rem); font-weight: 800; letter-spacing: -0.02em; }
.fixture-date { font-size: 0.85rem; font-weight: 700; color: var(--text2); letter-spacing: 0.06em; }

.res-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.res-card {                          /* the broadcast-style metric cards */
  --c: var(--green);
  position: relative; overflow: hidden; text-align: center;
  padding: 30px 18px 26px;
  background: linear-gradient(180deg, var(--bg3) 0%, var(--bg2) 100%);
  border: 1px solid var(--border); border-bottom: 5px solid var(--c);
  border-radius: 20px;
  transition: transform .25s ease, box-shadow .25s ease;
}
.res-home { --c: var(--green); }
.res-draw { --c: var(--muted); }
.res-away { --c: var(--pink); }
.res-card:hover { transform: translateY(-7px); box-shadow: 0 24px 44px rgba(0,0,0,0.5); }
.res-home:hover { box-shadow: 0 24px 44px rgba(0,0,0,0.5), 0 10px 40px rgba(0,255,135,0.18); }
.res-away:hover { box-shadow: 0 24px 44px rgba(0,0,0,0.5), 0 10px 40px rgba(233,0,82,0.20); }
.res-card.is-pick { border-color: var(--c); }
.res-pick {                          /* "MODEL PICK" tag on the favourite */
  position: absolute; top: 12px; right: 12px;
  padding: 4px 10px; border-radius: 999px;
  font-size: 0.62rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase;
  background: var(--c); color: var(--bg);
}
.res-draw .res-pick { color: var(--text); }
.res-visual { min-height: 88px; display: flex; align-items: center; justify-content: center; }
.res-label { margin-top: 14px; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.26em; text-transform: uppercase; color: var(--text2); }
.res-team { margin-top: 4px; font-size: 1.02rem; font-weight: 700; color: var(--text); min-height: 1.5em; }
.res-num {
  margin-top: 8px; font-size: clamp(3rem, 6vw, 4rem); font-weight: 800;
  letter-spacing: -0.045em; line-height: 1; color: var(--text);
}
.res-home .res-num { color: var(--green); }
.res-away .res-num { color: var(--pink); }
.res-num span { font-size: 0.42em; font-weight: 700; margin-left: 3px; letter-spacing: 0; }

/* probability bar card (Plotly lives inside) */
.st-key-bar_card {
  margin-top: 18px; padding: 20px 22px 18px;
  background: var(--bg2); border: 1px solid var(--border); border-radius: 20px;
}
.st-key-bar_card [data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
.bar-legend { display: flex; flex-wrap: wrap; gap: 10px 28px; margin-top: 12px; font-size: 0.85rem; font-weight: 600; color: var(--text2); }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }

/* model interpretation block */
.interp {
  margin-top: 18px; padding: 24px 28px;
  background: linear-gradient(135deg, rgba(56,0,60,0.55), rgba(21,27,61,0.92));
  border: 1px solid var(--border); border-left: 4px solid var(--green); border-radius: 16px;
}
.interp-head { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
.interp-title { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; }
.conf { padding: 4px 12px; border-radius: 999px; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }
.conf-high { background: rgba(0,255,135,0.12); color: var(--green); border: 1px solid rgba(0,255,135,0.45); }
.conf-mod  { background: rgba(255,184,0,0.10); color: var(--warn); border: 1px solid rgba(255,184,0,0.45); }
.conf-low  { background: rgba(184,193,217,0.10); color: var(--text2); border: 1px solid var(--border); }
.interp p { margin: 6px 0 0; color: var(--text2); line-height: 1.7; font-size: 0.98rem; max-width: 76ch; }

/* ── 2.8 LOADING STATE (skeleton cards) ──────────────────────────────────*/
.load-note {
  display: flex; align-items: center; justify-content: center; gap: 12px; margin: 30px 0 16px;
  color: var(--green); font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; font-size: 0.78rem;
}
.spinner {
  width: 20px; height: 20px; border-radius: 50%;
  border: 3px solid rgba(0,255,135,0.20); border-top-color: var(--green);
  animation: spin .8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.skel {
  height: 270px; border-radius: 20px; border: 1px solid var(--border);
  background: linear-gradient(100deg, var(--bg2) 30%, var(--bg3) 50%, var(--bg2) 70%);
  background-size: 200% 100%; animation: shimmer 1.4s linear infinite;
}
@keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }

/* ── 2.9 RECENT PREDICTIONS TABLE ────────────────────────────────────────*/
.tbl-wrap { overflow-x: auto; background: var(--bg2); border: 1px solid var(--border); border-radius: 16px; }
.tbl { width: 100%; border-collapse: collapse; font-size: 0.9rem; min-width: 640px; }
.tbl th {
  text-align: left; padding: 14px 16px; font-size: 0.68rem; font-weight: 800;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border);
}
.tbl td { padding: 13px 16px; color: var(--text2); font-weight: 600; border-bottom: 1px solid rgba(42,51,88,0.6); }
.tbl tr:last-child td { border-bottom: 0; }
.tbl tr:hover td { background: rgba(255,255,255,0.025); }
.tbl td.fx { color: var(--text); font-weight: 700; }
.tbl td.num { font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em; }
.pill-h { background: rgba(0,255,135,0.12); color: var(--green); }
.pill-d { background: rgba(107,117,147,0.25); color: var(--text2); }
.pill-a { background: rgba(233,0,82,0.15); color: #ff6b9c; }

/* ── 2.10 EXPANDER ("About the model") ───────────────────────────────────*/
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important; border-radius: 16px !important;
  background: var(--bg2); overflow: hidden;
}
[data-testid="stExpander"] summary { padding: 16px 20px; }
[data-testid="stExpander"] summary:hover { background: rgba(255,255,255,0.03); }
[data-testid="stExpander"] summary p {
  font-size: 0.8rem !important; font-weight: 800 !important;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--text) !important;
}
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 6px 0 18px; }
.fact { padding: 14px 16px; background: var(--bg3); border: 1px solid var(--border); border-radius: 12px; }
.fact-k { font-size: 0.66rem; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); }
.fact-v { margin-top: 6px; font-size: 0.95rem; font-weight: 700; color: var(--text); line-height: 1.4; }
.limits { margin: 0; padding-left: 18px; color: var(--text2); line-height: 1.8; font-size: 0.93rem; }

/* ── 2.11 FOOTER ─────────────────────────────────────────────────────────*/
.ftr {
  margin-top: 48px; padding-top: 24px; border-top: 1px solid var(--border);
  display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  color: var(--muted); font-size: 0.85rem; font-weight: 500;
}
.ftr a { color: var(--green); font-weight: 700; text-decoration: none; }
.ftr a:hover { text-decoration: underline; }

/* ── 2.12 RESPONSIVE + ACCESSIBILITY ─────────────────────────────────────*/
@media (max-width: 820px) {
  .res-grid { grid-template-columns: 1fr; }
  .skel { height: 150px; }
  .st-key-fixture_card { padding: 20px 16px 24px; }
  .vs-col { padding-top: 0; }
}
@media (prefers-reduced-motion: reduce) {
  .fade-in, .skel, .spinner { animation: none !important; }
  .res-card, .st-key-predict_btn button { transition: none !important; }
}
"""


# ═════════════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def compact(html: str) -> str:
    """Collapse indented multi-line HTML into one line.
    Streamlit's markdown treats 4-space-indented lines (and blank lines) as code blocks /
    block terminators, which would break our HTML. One line is always safe."""
    return " ".join(line.strip() for line in html.strip().splitlines() if line.strip())


def md(html: str) -> None:
    st.markdown(compact(html), unsafe_allow_html=True)


def dn(team: str) -> str:
    """Display name for a team (the model still receives the raw name)."""
    return DISPLAY_NAMES.get(canonical_team(team), team)


def fmt_date(iso: str) -> str:
    return dt.date.fromisoformat(iso).strftime("%a %d %b %Y")


# ── crests ──────────────────────────────────────────────────────────────────
# Crests are downloaded ONCE per server (not per visitor) and kept as data-URIs.
# A background thread warms all 20 crests the first time the app starts, so the
# first page render only waits for a crest if it is still downloading. If a
# download fails we fall back to a drawn crest and retry after 10 minutes.
_RETRY_AFTER_S = 600


def _download_logo(url: str) -> str:
    """Plain download -> data-URI. No Streamlit calls, so it is safe in a thread."""
    r = requests.get(
        url, timeout=(3, 5),  # (connect, read) seconds
        headers={"User-Agent": "Mozilla/5.0 (compatible; EPL-Predictor/1.0)"},
    )
    r.raise_for_status()
    ctype = r.headers.get("content-type", "image/png").split(";")[0]
    if not ctype.startswith("image/"):
        raise ValueError("URL did not return an image")
    return f"data:{ctype};base64," + base64.b64encode(r.content).decode()


def _fetch_into(store: dict, url: str) -> None:
    try:
        store["data"][url] = _download_logo(url)
    except Exception:  # noqa: BLE001
        store["failed"][url] = time.time()


@st.cache_resource(show_spinner=False)
def logo_store() -> dict:
    """Shared across all visitors. Created once; kicks off the background warm-up."""
    store: dict = {"data": {}, "failed": {}}

    def warm() -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for url in set(TEAM_LOGOS.values()):
                pool.submit(_fetch_into, store, url)

    Thread(target=warm, daemon=True).start()
    return store


def _fallback_crest(team: str) -> str:
    """Purple badge with the club's 3-letter code - used when a logo can't be loaded."""
    code = TLA.get(canonical_team(team), team[:3].upper())
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{PURPLE}"/><stop offset="1" stop-color="{BG3}"/></linearGradient></defs>'
        f'<circle cx="50" cy="50" r="46" fill="url(#g)" stroke="{GREEN}" stroke-width="3"/>'
        '<text x="50" y="58" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-weight="800" font-size="26" fill="{TEXT}">{escape(code)}</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def crest_src(team: str) -> str:
    url = TEAM_LOGOS.get(canonical_team(team))
    if not url:
        return _fallback_crest(team)
    store = logo_store()
    if url not in store["data"]:
        failed_at = store["failed"].get(url)
        if failed_at is None or time.time() - failed_at > _RETRY_AFTER_S:
            _fetch_into(store, url)  # not warmed yet (or retry window passed): fetch now
    return store["data"].get(url) or _fallback_crest(team)


def crest(team: str, px: int) -> str:
    return f'<img class="crest" src="{crest_src(team)}" width="{px}" height="{px}" alt="{escape(dn(team))} crest">'


# ── model access ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_teams() -> list[str]:
    if MODEL_READY:
        return sorted(str(t) for t in get_available_teams())
    return sorted(TEAM_CODES)


def _demo_prediction(home: str, away: str) -> dict:
    """Deterministic placeholder numbers. ONLY used when predict_logic.py is missing."""
    seed = int(hashlib.md5(f"{home}|{away}".encode()).hexdigest()[:8], 16)
    a, b = (seed % 1000) / 1000, ((seed // 1000) % 1000) / 1000
    p_h, p_d = 0.30 + 0.35 * a, 0.20 + 0.10 * b
    return {"p_H": p_h, "p_D": p_d, "p_A": 1 - p_h - p_d}


@st.cache_data(show_spinner=False, max_entries=256)
def run_prediction(date_iso: str, home: str, away: str) -> dict:
    """Call YOUR predict_fixture() and return clean, normalised probabilities.
    Cached, so repeating a fixture is instant (a fresh one takes ~2-3 s)."""
    if not MODEL_READY:
        raw = _demo_prediction(home, away)
    else:
        day = dt.date.fromisoformat(date_iso)
        try:
            raw = predict_fixture(pd.Timestamp(day), home, away)  # 1st try: pandas Timestamp
        except Exception as first_error:  # noqa: BLE001
            try:
                raw = predict_fixture(date_iso, home, away)  # 2nd try: ISO string "YYYY-MM-DD"
            except Exception:  # noqa: BLE001
                raise first_error  # both failed: show the original, most informative error
    p = [float(raw["p_H"]), float(raw["p_D"]), float(raw["p_A"])]
    total = sum(p)
    if total <= 0:
        raise ValueError("Model returned probabilities that do not add up.")
    p = [x / total for x in p]
    return {"p_H": p[0], "p_D": p[1], "p_A": p[2]}


# ── maths & copy ────────────────────────────────────────────────────────────
def pct_ints(probs: list[float]) -> list[int]:
    """Round to whole percentages that still add to exactly 100 (largest-remainder method)."""
    raw = [p * 100 for p in probs]
    out = [math.floor(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: 100 - sum(out)]:
        out[i] += 1
    return out


def interpretation(res: dict) -> tuple[str, str, str, list[str]]:
    """Returns (headline, confidence label, confidence css class, paragraphs)."""
    home, away = dn(res["home"]), dn(res["away"])
    outcomes = [("home", res["p_H"]), ("draw", res["p_D"]), ("away", res["p_A"])]
    (k1, p1), (_, p2) = sorted(outcomes, key=lambda t: t[1], reverse=True)[:2]
    gap = (p1 - p2) * 100
    names = {"home": f"A {home} win", "draw": "A draw", "away": f"An {away} win" if away[:1] in "AEIOU" else f"A {away} win"}
    second = sorted(outcomes, key=lambda t: t[1], reverse=True)[1][0]

    if gap < 5:
        headline = "Too close to call"
    elif k1 == "draw":
        headline = "The model leans towards a draw"
    else:
        headline = f"The model leans {home if k1 == 'home' else away}"

    if p1 >= 0.55:
        conf, cls = "High confidence", "conf-high"
    elif p1 >= 0.42:
        conf, cls = "Moderate confidence", "conf-mod"
    else:
        conf, cls = "Low confidence", "conf-low"

    gain = (1 - TEST_LOG_LOSS / UNIFORM_LOG_LOSS) * 100
    paras = [
        f"{names[k1]} is the single most likely result at {p1 * 100:.1f}%, "
        f"{gap:.1f} points clear of {names[second].lower()} ({p2 * 100:.1f}%).",
        f"These are probabilities, not promises: an outcome rated {p1 * 100:.0f}% still fails to happen "
        f"in roughly {100 - p1 * 100:.0f} out of 100 comparable fixtures.",
        f"On the locked 2025/26 test season the model scored a log loss of {TEST_LOG_LOSS:.4f}. "
        f"Guessing one-third for each outcome scores {UNIFORM_LOG_LOSS:.4f}, so the model beats a "
        f"no-information guess by about {gain:.0f}%.",
    ]
    return headline, conf, cls, paras


def pick_label(res: dict) -> tuple[str, str]:
    """(label, pill css class) for the most likely outcome."""
    i = max(range(3), key=lambda k: [res["p_H"], res["p_D"], res["p_A"]][k])
    return (["Home", "Draw", "Away"][i], ["pill-h", "pill-d", "pill-a"][i])


# ── Plotly stacked probability bar ──────────────────────────────────────────
def probability_bar(res: dict) -> go.Figure:
    probs = [res["p_H"], res["p_D"], res["p_A"]]
    ints = pct_ints(probs)
    names = [f"{dn(res['home'])} win", "Draw", f"{dn(res['away'])} win"]
    colours, text_colours = [GREEN, MUTED, PINK], [BG, TEXT, TEXT]
    fig = go.Figure()
    for name, p, label, col, tcol in zip(names, probs, ints, colours, text_colours):
        fig.add_trace(
            go.Bar(
                x=[p * 100], y=[""], orientation="h", name=name,
                marker=dict(color=col, line=dict(width=2, color=BG2)),
                text=[f"{label}%"], textposition="inside", insidetextanchor="middle",
                textfont=dict(color=tcol, size=20, family="Manrope, Inter, sans-serif"),
                hovertemplate=f"{name}: {p * 100:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack", height=84, bargap=0, showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 100], fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        hoverlabel=dict(bgcolor=BG3, bordercolor=BORDER, font=dict(color=TEXT, family="Manrope, sans-serif")),
    )
    return fig


def show_chart(fig: go.Figure) -> None:
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)  # current Streamlit
    except Exception:  # noqa: BLE001 - older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, config=cfg)


# ═════════════════════════════════════════════════════════════════════════════
# 4. PAGE SECTIONS
# ═════════════════════════════════════════════════════════════════════════════
def render_header() -> None:
    md(f"""
    <div class="hdr"><div class="hdr-in">
      <div class="hdr-left">
        <div class="hdr-badge">⚽</div>
        <div>
          <div class="hdr-kicker">Premier League</div>
          <div class="hdr-title">Match Outcome Predictor</div>
        </div>
      </div>
      <div class="hdr-right">
        <span class="chip chip-ver">{APP_VERSION}</span>
        <span class="chip chip-ll"><b>Test log loss</b> {TEST_LOG_LOSS:.4f}</span>
      </div>
    </div></div>
    """)


def render_fixture_card(teams: list[str]) -> tuple[str, str, dt.date, bool]:
    default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
    default_away = teams.index("Liverpool") if "Liverpool" in teams else min(1, len(teams) - 1)

    with st.container(key="fixture_card"):
        md('<div class="sec-label">Choose a fixture</div>')
        c_home, c_vs, c_away = st.columns([5, 1.1, 5], gap="medium")

        with c_home:
            logo_col, sel_col = st.columns([1, 3.2], gap="small")
            with sel_col:
                home = st.selectbox("Home team", teams, index=default_home, key="home_team", format_func=dn)
            with logo_col:
                md(f'<div class="sel-crest">{crest(home, 54)}</div>')

        with c_vs:
            md('<div class="vs-col"><div class="vs-badge">VS</div></div>')

        with c_away:
            sel_col, logo_col = st.columns([3.2, 1], gap="small")
            with sel_col:
                away = st.selectbox("Away team", teams, index=default_away, key="away_team", format_func=dn)
            with logo_col:
                md(f'<div class="sel-crest">{crest(away, 54)}</div>')

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            today = dt.date.today()
            match_date = st.date_input(
                "Match date", value=today, min_value=dt.date(2015, 8, 1),
                max_value=today + dt.timedelta(days=365), key="match_date", format="DD/MM/YYYY",
            )
            same_team = home == away
            if same_team:
                md('<div class="alert">Pick two different clubs to build a fixture.</div>')
            st.write("")
            clicked = st.button("Predict match", key="predict_btn", disabled=same_team)

        if not MODEL_READY:
            md(
                '<div class="demo-banner">DEMO MODE: predict_logic.py could not be imported, '
                "so the numbers below are placeholders, not real predictions. "
                f"Reason: {escape(MODEL_ERROR)}</div>"
            )
    return home, away, match_date, clicked


def skeleton_html() -> str:
    return compact("""
    <div class="load-note"><div class="spinner"></div>Building features and running the model</div>
    <div class="res-grid"><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>
    """)


def render_results(res: dict) -> None:
    home, away = res["home"], res["away"]
    probs = [res["p_H"], res["p_D"], res["p_A"]]
    ints = pct_ints(probs)
    fav = max(range(3), key=lambda i: probs[i])

    def card(i: int, kind: str, label: str, team_line: str, visual: str) -> str:
        tag = '<span class="res-pick">Model pick</span>' if i == fav else ""
        pick = " is-pick" if i == fav else ""
        return (
            f'<div class="res-card res-{kind}{pick}">{tag}'
            f'<div class="res-visual">{visual}</div>'
            f'<div class="res-label">{label}</div>'
            f'<div class="res-team">{team_line}</div>'
            f'<div class="res-num">{ints[i]}<span>%</span></div></div>'
        )

    cards = (
        card(0, "home", "Home win", escape(dn(home)), crest(home, 76))
        + card(1, "draw", "Draw", "Honours even", f'<div class="crest-pair">{crest(home, 42)}{crest(away, 42)}</div>')
        + card(2, "away", "Away win", escape(dn(away)), crest(away, 76))
    )

    md('<div class="sec-wrap"><div class="sec-label">Prediction</div></div>')
    md(
        f'<div class="fade-in"><div class="fixture-line">'
        f'<div class="fixture-name">{escape(dn(home))} vs {escape(dn(away))}</div>'
        f'<div class="fixture-date">{fmt_date(res["date"])}</div></div>'
        f'<div class="res-grid">{cards}</div></div>'
    )

    with st.container(key="bar_card"):
        show_chart(probability_bar(res))
        md(
            f'<div class="bar-legend">'
            f'<span><i class="dot" style="background:{GREEN}"></i>{escape(dn(home))} win</span>'
            f'<span><i class="dot" style="background:{MUTED}"></i>Draw</span>'
            f'<span><i class="dot" style="background:{PINK}"></i>{escape(dn(away))} win</span></div>'
        )

    headline, conf, cls, paras = interpretation(res)
    body = "".join(f"<p>{p}</p>" for p in paras)
    md(
        f'<div class="interp fade-in"><div class="interp-head">'
        f'<div class="interp-title">{escape(headline)}</div><span class="conf {cls}">{conf}</span></div>{body}</div>'
    )


def render_history() -> None:
    history = st.session_state.history
    if not history:
        return
    md('<div class="sec-wrap"><div class="sec-label">Recent predictions</div></div>')
    rows = ""
    for h in history:
        label, pill = pick_label(h)
        ints = pct_ints([h["p_H"], h["p_D"], h["p_A"]])
        rows += (
            f'<tr><td class="num">{h["time"]}</td>'
            f'<td class="fx">{escape(dn(h["home"]))} vs {escape(dn(h["away"]))}</td>'
            f'<td class="num">{dt.date.fromisoformat(h["date"]).strftime("%d %b %Y")}</td>'
            f'<td class="num">{ints[0]}%</td><td class="num">{ints[1]}%</td><td class="num">{ints[2]}%</td>'
            f'<td><span class="pill {pill}">{label}</span></td></tr>'
        )
    md(
        '<div class="tbl-wrap"><table class="tbl"><thead><tr>'
        "<th>Time</th><th>Fixture</th><th>Match date</th><th>Home</th><th>Draw</th><th>Away</th><th>Pick</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def render_about() -> None:
    md('<div class="sec-wrap"><div class="sec-label">Model card</div></div>')
    with st.expander("About the model"):
        facts = "".join(
            f'<div class="fact"><div class="fact-k">{escape(k)}</div><div class="fact-v">{escape(v)}</div></div>'
            for k, v in MODEL_CARD_FACTS
        )
        limits = "".join(f"<li>{escape(x)}</li>" for x in MODEL_CARD_LIMITS)
        md(
            f'<div class="facts">{facts}</div>'
            f'<div class="sec-label" style="margin-bottom:10px">Limitations</div><ul class="limits">{limits}</ul>'
        )


def render_footer() -> None:
    md(
        f'<div class="ftr"><span>Portfolio project for learning — not for betting or financial use.</span>'
        f'<a href="{GITHUB_MODEL_CARD_URL}" target="_blank" rel="noopener noreferrer">Model card on GitHub</a></div>'
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("error", None)

    st.markdown(f"<style>{compact(CSS)}</style>", unsafe_allow_html=True)
    render_header()

    teams = load_teams()
    home, away, match_date, clicked = render_fixture_card(teams)

    slot = st.empty()  # skeleton cards appear here while the model runs
    if clicked:
        slot.markdown(skeleton_html(), unsafe_allow_html=True)
        started = time.perf_counter()
        try:
            probs = run_prediction(match_date.isoformat(), home, away)
            time.sleep(max(0.0, 0.5 - (time.perf_counter() - started)))  # avoid a flash on cache hits
            res = {**probs, "home": home, "away": away, "date": match_date.isoformat()}
            st.session_state.result = res
            st.session_state.error = None
            st.session_state.history = (
                [{**res, "time": dt.datetime.now().strftime("%H:%M:%S")}] + st.session_state.history
            )[:8]
        except Exception as exc:  # noqa: BLE001
            st.session_state.error = f"{type(exc).__name__}: {exc}"
        slot.empty()

    if st.session_state.error:
        md(
            '<div class="alert alert-error">The prediction could not be completed. '
            f"{escape(st.session_state.error)}. Check the team names and date, then try again.</div>"
        )

    if st.session_state.result:
        render_results(st.session_state.result)
    elif not st.session_state.error:
        md(
            '<div class="empty"><b>Pick a home team, an away team and a date.</b><br>'
            "Press Predict match to see the chance of a home win, a draw and an away win.</div>"
        )

    render_history()
    render_about()
    render_footer()


main()
