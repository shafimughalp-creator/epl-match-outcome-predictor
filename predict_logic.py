"""
predict_logic.py — the bridge between the Streamlit UI and the frozen champion.

Loads:
  - models/champion_final.joblib
  - models/champion_features.json
  - data/matches_history.csv (all matches up to May 2026)

Provides:
  - predict_fixture(date, home_team, away_team) -> dict with A/D/H probabilities
  - get_available_teams() -> sorted list of team names
"""

import os, json
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(HERE, 'data', 'matches_history.csv')
MODEL_PATH = os.path.join(HERE, 'models', 'champion_final.joblib')
FEAT_PATH  = os.path.join(HERE, 'models', 'champion_features.json')

# ---------- import the feature builders ----------
import sys
sys.path.append(os.path.join(HERE, 'src'))
import features  # our frozen src_snapshot/features.py

# ---------- load frozen artifacts once at import ----------
_CHAMPION = joblib.load(MODEL_PATH)
with open(FEAT_PATH) as f:
    _META = json.load(f)
FEATURES = _META['features']          # the exact 9 delta feature names

# ---------- history ----------
_HISTORY_RAW = pd.read_csv(DATA_PATH, parse_dates=['Date'])
_HISTORY_RAW = _HISTORY_RAW.sort_values('Date').reset_index(drop=True)

def get_available_teams():
    teams = sorted(set(_HISTORY_RAW['HomeTeam']) | set(_HISTORY_RAW['AwayTeam']))
    return teams

# ---------- the pipeline to build features on history + one fixture ----------
PROMOTED_TABLE = {
    '2016/17': ['Burnley','Hull','Middlesbrough'],
    '2017/18': ['Brighton','Huddersfield','Newcastle'],
    '2018/19': ['Cardiff','Fulham','Wolves'],
    '2019/20': ['Aston Villa','Norwich','Sheffield United'],
    '2020/21': ['Fulham','Leeds','West Brom'],
    '2021/22': ['Brentford','Norwich','Watford'],
    '2022/23': ['Bournemouth','Fulham',"Nott'm Forest"],
    '2023/24': ['Burnley','Luton','Sheffield United'],
    '2024/25': ['Ipswich','Leicester','Southampton'],
    '2025/26': ['Burnley','Leeds','Sunderland'],
}

def _build_features_on(df):
    """Run every feature builder on the given history DataFrame.
    Returns df augmented with all feature columns + FTR_int."""
    combined = df.copy().sort_values('Date').reset_index(drop=True)

    promoted_for_combined = {k: v for k, v in PROMOTED_TABLE.items()
                             if k in combined['season_label'].unique()}

    roll   = features.build_rolling_form_filled(
                combined,
                features.compute_prev_season_form(combined),
                features.compute_promoted_baseline(combined, promoted_for_combined, 5),
                PROMOTED_TABLE, n=5)
    prev_form_v     = features.compute_prev_season_form_venue(combined)
    promoted_base_v = features.compute_promoted_baseline_venue(combined, promoted_for_combined, 5)
    venue  = features.build_venue_form_filled(combined, prev_form_v, promoted_base_v, PROMOTED_TABLE, n=5)
    prev_stats_v    = features.compute_prev_season_match_stats_venue(combined)
    promoted_base_ms= features.compute_promoted_baseline_match_stats(combined, promoted_for_combined, 5)
    stats  = features.build_match_stats_filled(combined, prev_stats_v, promoted_base_ms, PROMOTED_TABLE, n=5)
    prev_gd = features.compute_prev_season_gd_per_game_venue(combined)
    std    = features.build_season_to_date_filled(combined, prev_gd, promoted_base_v, PROMOTED_TABLE)
    rest   = features.build_rest_days(combined, cap=14)
    elo    = features.build_elo_features(combined)
    prev_ppg = features.build_prev_season_ppg_features(combined, prev_form_v, promoted_base_v, PROMOTED_TABLE)

    blocks = [roll, venue, stats, std, rest, elo, prev_ppg]
    feats = pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)
    feats = pd.concat([combined.reset_index(drop=True), feats], axis=1)
    deltas = features.build_deltas(feats)
    feats = pd.concat([feats.reset_index(drop=True), deltas.reset_index(drop=True)], axis=1)

    feats['FTR_int'] = feats['FTR'].map({"A": 0, "D": 1, "H": 2}).fillna(-1).astype(int)
    return feats

def predict_fixture(date, home_team, away_team, verbose=False):
    """
    Predict a future match.

    Args:
        date: pandas Timestamp or string 'YYYY-MM-DD'
        home_team, away_team: strings from get_available_teams()

    Returns:
        dict with keys 'p_A', 'p_D', 'p_H', and metadata.
    """
    date = pd.Timestamp(date)

    # Determine which season the fixture belongs to
    # If date's month >= 7 (July), season = year/next-year; else previous/this-year
    year = date.year
    if date.month >= 7:
        season_label = f"{year}/{str(year+1)[2:]}"
    else:
        season_label = f"{year-1}/{str(year)[2:]}"

    # Build a fixture row matching the raw schema
    fixture_row = {
        'Date': date,
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'FTHG': np.nan, 'FTAG': np.nan, 'FTR': np.nan,
        'HTHG': np.nan, 'HTAG': np.nan, 'HTR': np.nan,
        'Referee': 'Unknown',
        'HS': np.nan, 'AS': np.nan, 'HST': np.nan, 'AST': np.nan,
        'HF': np.nan, 'AF': np.nan, 'HC': np.nan, 'AC': np.nan,
        'HY': np.nan, 'AY': np.nan, 'HR': np.nan, 'AR': np.nan,
        'season': int(season_label.split('/')[0]),
        'season_label': season_label,
    }
    fixture_df = pd.DataFrame([fixture_row])

    # Combine history + fixture and build features
    full = pd.concat([_HISTORY_RAW, fixture_df], ignore_index=True)
    feats = _build_features_on(full)

    # Extract fixture row's feature vector
    fixture_pos = feats[feats['HomeTeam'] == home_team].index[-1]
    X = feats.loc[fixture_pos, FEATURES].values.astype(float).reshape(1, -1)

    # Predict
    proba = _CHAMPION.predict_proba(X)[0]  # [p_A, p_D, p_H]

    result = {
        'date': date,
        'home_team': home_team,
        'away_team': away_team,
        'p_A': float(proba[0]),
        'p_D': float(proba[1]),
        'p_H': float(proba[2]),
    }
    if verbose:
        result['features'] = {f: float(feats.loc[fixture_pos, f]) for f in FEATURES}
    return result