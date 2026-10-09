"""
features.py — shared preprocessing and feature-engineering utilities.

Rules:
  - No global state. Every function takes its inputs explicitly.
  - Every rolling feature is shifted by one match (no current-match leakage).
  - Rolling windows reset at season boundaries (season-scoped).
  - Early-season missing history is filled via weighted blend with
    previous-season form, or promoted baseline for promoted teams.
  - Assertions fail loudly if leakage is detected.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# 1. TARGET ENCODING
# ============================================================

TARGET_MAP = {"A": 0, "D": 1, "H": 2}
TARGET_INV = {v: k for k, v in TARGET_MAP.items()}


def encode_target(series: pd.Series) -> pd.Series:
    """Map FTR string labels to integers. A=0, D=1, H=2."""
    return series.map(TARGET_MAP).astype(int)


def decode_target(series: pd.Series) -> pd.Series:
    """Reverse mapping."""
    return series.map(TARGET_INV)


# ============================================================
# 2. LEAKAGE GUARDS
# ============================================================

def assert_no_current_match_leakage(features_df: pd.DataFrame,
                                    matches_df: pd.DataFrame) -> None:
    """Verify row counts align. Full check in Step 08I."""
    assert len(features_df) == len(matches_df), \
        f"Features rows ({len(features_df)}) != matches rows ({len(matches_df)})"


# ============================================================
# 3. PROMOTED-TEAM HANDLING
# ============================================================

def get_promoted_teams(promoted_table: dict, season_label: str) -> list:
    return list(promoted_table.get(season_label, []))


def compute_promoted_baseline(train_df: pd.DataFrame,
                              promoted_table: dict,
                              early_gameweeks: int = 5) -> dict:
    """
    Compute, from TRAIN ONLY, the average per-match stats of promoted teams
    during their first `early_gameweeks` matches of the season.

    Returns dict with keys:
      promoted_baseline_ppg, promoted_baseline_gf, promoted_baseline_ga
    """
    train = train_df.copy().sort_values("Date").reset_index(drop=True)

    rows = []
    for season_label, group in train.groupby("season_label"):
        promoted = get_promoted_teams(promoted_table, season_label)
        if not promoted:
            continue
        for team in promoted:
            team_matches = train[
                (train["season_label"] == season_label) &
                ((train["HomeTeam"] == team) | (train["AwayTeam"] == team))
            ].sort_values("Date").head(early_gameweeks)
            rows.append(team_matches)

    if not rows:
        return {"promoted_baseline_ppg": 1.0,
                "promoted_baseline_gf":  1.0,
                "promoted_baseline_ga":  1.7}

    bdf = pd.concat(rows, ignore_index=True)

    def promoted_pts(m):
        is_home = m["HomeTeam"] in get_promoted_teams(promoted_table, m["season_label"])
        gf = m["FTHG"] if is_home else m["FTAG"]
        ga = m["FTAG"] if is_home else m["FTHG"]
        if gf > ga: return 3
        if gf == ga: return 1
        return 0

    def promoted_gf(m):
        is_home = m["HomeTeam"] in get_promoted_teams(promoted_table, m["season_label"])
        return m["FTHG"] if is_home else m["FTAG"]

    def promoted_ga(m):
        is_home = m["HomeTeam"] in get_promoted_teams(promoted_table, m["season_label"])
        return m["FTAG"] if is_home else m["FTHG"]

    pts = bdf.apply(promoted_pts, axis=1).mean()
    gf  = bdf.apply(promoted_gf,  axis=1).mean()
    ga  = bdf.apply(promoted_ga,  axis=1).mean()

    return {
        "promoted_baseline_ppg": float(pts),
        "promoted_baseline_gf":  float(gf),
        "promoted_baseline_ga":  float(ga),
    }


# ============================================================
# 4. PREVIOUS-SEASON FORM (per team, per season)
# ============================================================

def compute_prev_season_form(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team and each TARGET season, compute the team's end-of-season
    stats in the PREVIOUS season.

    Columns: team, season_label (target), prev_season_label (source),
             prev_season_ppg, prev_season_gf, prev_season_ga
    """
    rows = []
    for (season_label, team), group in _iter_team_seasons(train_df):
        pts, gf, ga = 0, 0, 0
        for _, m in group.iterrows():
            if m["HomeTeam"] == team:
                gf += m["FTHG"]; ga += m["FTAG"]
                if m["FTHG"] > m["FTAG"]: pts += 3
                elif m["FTHG"] == m["FTAG"]: pts += 1
            else:
                gf += m["FTAG"]; ga += m["FTHG"]
                if m["FTAG"] > m["FTHG"]: pts += 3
                elif m["FTAG"] == m["FTHG"]: pts += 1
        n = len(group)
        rows.append({
            "team": team,
            "prev_season_label": season_label,
            "ppg": pts / max(n, 1),
            "gf_per_match": gf / max(n, 1),
            "ga_per_match": ga / max(n, 1),
        })
    end_of_season = pd.DataFrame(rows)

    season_order = sorted(end_of_season["prev_season_label"].unique())
    next_season = {s: season_order[i+1] for i, s in enumerate(season_order) if i+1 < len(season_order)}
    end_of_season["season_label"] = end_of_season["prev_season_label"].map(next_season)

    prev_lookup = (
        end_of_season
        .dropna(subset=["season_label"])
        .rename(columns={
            "ppg":          "prev_season_ppg",
            "gf_per_match": "prev_season_gf",
            "ga_per_match": "prev_season_ga",
        })
        [["team", "season_label", "prev_season_label",
          "prev_season_ppg", "prev_season_gf", "prev_season_ga"]]
        .reset_index(drop=True)
    )
    return prev_lookup


def _iter_team_seasons(df):
    for season_label, season_group in df.groupby("season_label"):
        teams = sorted(set(season_group["HomeTeam"]).union(season_group["AwayTeam"]))
        for team in teams:
            mask = (season_group["HomeTeam"] == team) | (season_group["AwayTeam"] == team)
            yield (season_label, team), season_group[mask]


# ============================================================
# 5. ROLLING FORM (season-scoped)
# ============================================================

def build_rolling_form(matches_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Build season-scoped rolling form features per team.

    For each match, produce 8 columns (4 per side):
        home_form_pts_last{n}, away_form_pts_last{n}
        home_form_gf_last{n},  away_form_gf_last{n}
        home_form_ga_last{n},  away_form_ga_last{n}
        home_form_gd_last{n},  away_form_gd_last{n}

    Rolling window resets at season boundaries: only matches within the
    SAME season that are strictly EARLIER than the current match count
    toward the rolling window. Early-season rows have NaN (filled later
    by weighted blend).

    Output DataFrame has the same index as input.
    """
    df = matches_df.copy().reset_index(drop=True)
    df["_row_id"] = df.index

    # ---------- Long-form per-team timeline ----------
    def make_side(side):
        if side == "home":
            d = pd.DataFrame({
                "row_id": df["_row_id"],
                "Date":   df["Date"],
                "season_label": df["season_label"],
                "team":   df["HomeTeam"],
                "goals_for":     df["FTHG"],
                "goals_against": df["FTAG"],
                "is_home": 1,
            })
        else:
            d = pd.DataFrame({
                "row_id": df["_row_id"],
                "Date":   df["Date"],
                "season_label": df["season_label"],
                "team":   df["AwayTeam"],
                "goals_for":     df["FTAG"],
                "goals_against": df["FTHG"],
                "is_home": 0,
            })
        d["gd"] = d["goals_for"] - d["goals_against"]
        d["pts"] = np.where(d["goals_for"] > d["goals_against"], 3,
                     np.where(d["goals_for"] == d["goals_against"], 1, 0))
        return d

    long_df = pd.concat([make_side("home"), make_side("away")], ignore_index=True)
    long_df = long_df.sort_values(["team", "season_label", "Date", "row_id"]).reset_index(drop=True)

    # ---------- Rolling within (team, season) groups ----------
    def roll_col(col):
        return (long_df.groupby(["team", "season_label"])[col]
                       .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()))

    long_df["form_pts_last"] = roll_col("pts")
    long_df["form_gf_last"]  = roll_col("goals_for")
    long_df["form_ga_last"]  = roll_col("goals_against")
    long_df["form_gd_last"]  = roll_col("gd")

    # ---------- Back to match-level ----------
    home_feats = long_df[long_df["is_home"] == 1].set_index("row_id")
    away_feats = long_df[long_df["is_home"] == 0].set_index("row_id")

    out = pd.DataFrame(index=df.index)
    mapping = {
        "form_pts_last": f"form_pts_last{n}",
        "form_gf_last":  f"form_gf_last{n}",
        "form_ga_last":  f"form_ga_last{n}",
        "form_gd_last":  f"form_gd_last{n}",
    }
    for src, dst in mapping.items():
        out[f"home_{dst}"] = home_feats[src].reindex(df.index).values
        out[f"away_{dst}"] = away_feats[src].reindex(df.index).values
    return out


# ============================================================
# 6. WEIGHTED BLEND WITH PREVIOUS-SEASON FORM
# ============================================================

def build_rolling_form_filled(matches_df: pd.DataFrame,
                              prev_form_df: pd.DataFrame,
                              promoted_baseline: dict,
                              promoted_table: dict,
                              n: int = 5) -> pd.DataFrame:
    """
    Build rolling form (season-scoped) and fill early-season NaN via
    weighted blend:

        value = (k * rolling_in_season + (n-k) * prev_season_form) / n
        where k = number of in-season matches so far.

    For rows where rolling is already populated (k = n), no change.

    For rows where a team is promoted OR has no previous-season entry,
    fill with promoted_baseline values.

    Returns a DataFrame with 8 columns (4 home + 4 away) + 2 promoted flags.
    """
    n_actual = n
    roll = build_rolling_form(matches_df, n=n_actual)

    # Lookup for previous-season form
    prev_map = prev_form_df.set_index(["team", "season_label"]).to_dict("index")

    # Promoted flags
    matches = matches_df.copy().reset_index(drop=True)
    matches["home_is_promoted"] = matches.apply(
        lambda r: int(r["HomeTeam"] in get_promoted_teams(promoted_table, r["season_label"])), axis=1)
    matches["away_is_promoted"] = matches.apply(
        lambda r: int(r["AwayTeam"] in get_promoted_teams(promoted_table, r["season_label"])), axis=1)

    # Compute k (number of prior in-season matches) per team per season
    def compute_k(df):
        df = df.copy().reset_index(drop=True)
        # For each team-season, matches played before current match
        def team_k(df, team_col):
            df = df.sort_values(["season_label", "Date"]).reset_index(drop=True)
            df["_k"] = df.groupby([team_col, "season_label"]).cumcount()
            return df[["_k"]].rename(columns={"_k": f"_k_{team_col}"})
        # We'll compute both home and away k
        return df

    matches = matches.sort_values(["season_label", "Date"]).reset_index(drop=True)
    matches["_k_home"] = matches.groupby(["HomeTeam", "season_label"]).cumcount()
    matches["_k_away"] = matches.groupby(["AwayTeam", "season_label"]).cumcount()

    # Because build_rolling_form expects original order, reorder matching
    # For simplicity, rebuild roll from sorted matches and reindex to original input.
    original_index = matches_df.index
    # Build roll on the sorted matches; then reorder back
    roll_sorted = build_rolling_form(matches, n=n_actual)
    roll_final  = pd.DataFrame(index=matches_df.index)
    for c in roll_sorted.columns:
        roll_final[c] = roll_sorted[c].values  # same length and order as `matches` (which equals original after sort)

    # Apply blend per side
    def blend_side(side, k_series):
        team_col = "HomeTeam" if side == "home" else "AwayTeam"
        for metric_src, metric_prev in [
            (f"form_pts_last{n_actual}", "prev_season_ppg"),
            (f"form_gf_last{n_actual}",  "prev_season_gf"),
            (f"form_ga_last{n_actual}",  "prev_season_ga"),
        ]:
            col = f"{side}_{metric_src}"
            values = roll_final[col].values.copy()
            for i, row in matches.iterrows():
                k = int(row[f"_k_{team_col.replace('Team','').lower()}"])
                if k >= n_actual:
                    continue  # already fully populated by in-season history
                rolling_val = values[i]
                if pd.isna(rolling_val):
                    # No in-season history at all
                    in_season_avg = 0.0
                else:
                    in_season_avg = rolling_val  # already an average over k matches
                team = row[team_col]
                season = row["season_label"]
                prev = prev_map.get((team, season))
                if prev is not None:
                    prev_val = prev[metric_prev]
                    blended = (k * in_season_avg + (n_actual - k) * prev_val) / n_actual
                else:
                    # Promoted team or no history: use promoted baseline
                    base_key = {
                        "form_pts_last" + str(n_actual): "promoted_baseline_ppg",
                        "form_gf_last"  + str(n_actual): "promoted_baseline_gf",
                        "form_ga_last"  + str(n_actual): "promoted_baseline_ga",
                    }[metric_src]
                    prev_val = promoted_baseline[base_key]
                    blended = (k * in_season_avg + (n_actual - k) * prev_val) / n_actual
                values[i] = blended
            roll_final[col] = values

        # GD derived after pts/gf/ga
        col_gd = f"{side}_form_gd_last{n_actual}"
        col_gf = f"{side}_form_gf_last{n_actual}"
        col_ga = f"{side}_form_ga_last{n_actual}"
        roll_final[col_gd] = roll_final[col_gf] - roll_final[col_ga]

    # Compute k series aligned with matches
    k_home = matches["_k_home"].reset_index(drop=True)
    k_away = matches["_k_away"].reset_index(drop=True)
    blend_side("home", k_home)
    blend_side("away", k_away)

    # Add promoted flags
    roll_final["home_is_promoted"] = matches["home_is_promoted"].reset_index(drop=True).values
    roll_final["away_is_promoted"] = matches["away_is_promoted"].reset_index(drop=True).values

    # Reindex back to original input index
    roll_final.index = matches_df.index
    return roll_final


# ============================================================
# 7. CLASS WEIGHTS
# ============================================================

def compute_class_weights(y_int: pd.Series) -> dict:
    counts = y_int.value_counts().sort_index()
    total  = counts.sum()
    n_cls  = len(counts)
    return {int(cls): float(total / (n_cls * cnt)) for cls, cnt in counts.items()}


def compute_sample_weights(y_int: pd.Series) -> pd.Series:
    cw = compute_class_weights(y_int)
    return y_int.map(cw).astype(float)


# ============================================================
# 8. TESTING UTILITY
# ============================================================

def test_feature_builder(df: pd.DataFrame) -> None:
    assert "Date" in df.columns, "Missing Date column."
    assert df["Date"].is_monotonic_increasing, "Rows are not sorted by Date."


print("features.py module loaded.")


# ============================================================
# 9. VENUE-SPECIFIC PREVIOUS-SEASON FORM (added at Step 08B)
# ============================================================

def compute_prev_season_form_venue(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Like compute_prev_season_form, but returns venue-specific stats
    (home-only and away-only for each team in each previous season).

    Columns:
      team, season_label (target), prev_season_label (source),
      prev_season_ppg_home, prev_season_gf_home, prev_season_ga_home,
      prev_season_ppg_away, prev_season_gf_away, prev_season_ga_away
    """
    rows = []
    for (season_label, team), group in _iter_team_seasons(train_df):
        home_group = group[group["HomeTeam"] == team]
        away_group = group[group["AwayTeam"] == team]

        def stats(sub, side):
            if len(sub) == 0:
                return (np.nan, np.nan, np.nan)
            if side == "home":
                gf_series = sub["FTHG"]; ga_series = sub["FTAG"]
            else:
                gf_series = sub["FTAG"]; ga_series = sub["FTHG"]
            pts = 0
            for _, m in sub.iterrows():
                gf = m["FTHG"] if side == "home" else m["FTAG"]
                ga = m["FTAG"] if side == "home" else m["FTHG"]
                if gf > ga: pts += 3
                elif gf == ga: pts += 1
            n = len(sub)
            return (pts / n, gf_series.mean(), ga_series.mean())

        ph, gfh, gah = stats(home_group, "home")
        pa, gfa, gaa = stats(away_group, "away")

        rows.append({
            "team": team,
            "prev_season_label": season_label,
            "ppg_home": ph, "gf_home": gfh, "ga_home": gah,
            "ppg_away": pa, "gf_away": gfa, "ga_away": gaa,
        })

    end_of_season = pd.DataFrame(rows)

    season_order = sorted(end_of_season["prev_season_label"].unique())
    next_season = {s: season_order[i+1] for i, s in enumerate(season_order) if i+1 < len(season_order)}
    end_of_season["season_label"] = end_of_season["prev_season_label"].map(next_season)

    prev_lookup = (
        end_of_season
        .dropna(subset=["season_label"])
        .rename(columns={
            "ppg_home": "prev_season_ppg_home",
            "gf_home":  "prev_season_gf_home",
            "ga_home":  "prev_season_ga_home",
            "ppg_away": "prev_season_ppg_away",
            "gf_away":  "prev_season_gf_away",
            "ga_away":  "prev_season_ga_away",
        })
        [["team", "season_label", "prev_season_label",
          "prev_season_ppg_home", "prev_season_gf_home", "prev_season_ga_home",
          "prev_season_ppg_away", "prev_season_gf_away", "prev_season_ga_away"]]
        .reset_index(drop=True)
    )
    return prev_lookup


# ============================================================
# 10. VENUE-SPECIFIC PROMOTED BASELINE (added at Step 08B)
# ============================================================

def compute_promoted_baseline_venue(train_df: pd.DataFrame,
                                    promoted_table: dict,
                                    early_gameweeks: int = 5) -> dict:
    """
    Same as compute_promoted_baseline but returns venue-specific values.
    """
    train = train_df.copy().sort_values("Date").reset_index(drop=True)

    home_rows, away_rows = [], []
    for season_label, group in train.groupby("season_label"):
        promoted = get_promoted_teams(promoted_table, season_label)
        if not promoted:
            continue
        for team in promoted:
            team_matches = train[
                (train["season_label"] == season_label) &
                ((train["HomeTeam"] == team) | (train["AwayTeam"] == team))
            ].sort_values("Date").head(early_gameweeks)
            home_rows.append(team_matches[team_matches["HomeTeam"] == team])
            away_rows.append(team_matches[team_matches["AwayTeam"] == team])

    def summarize(rows, side):
        non_empty = [r for r in rows if len(r) > 0]
        if not non_empty:
            return {"ppg": 1.0, "gf": 1.0, "ga": 1.7}
        bdf = pd.concat(non_empty, ignore_index=True)
        if side == "home":
            gf = bdf["FTHG"]; ga = bdf["FTAG"]
            pts = ((gf > ga).astype(int) * 3 + (gf == ga).astype(int) * 1).mean()
        else:
            gf = bdf["FTAG"]; ga = bdf["FTHG"]
            pts = ((gf > ga).astype(int) * 3 + (gf == ga).astype(int) * 1).mean()
        return {"ppg": float(pts), "gf": float(gf.mean()), "ga": float(ga.mean())}

    h = summarize(home_rows, "home")
    a = summarize(away_rows, "away")
    return {
        "promoted_baseline_ppg_home": h["ppg"],
        "promoted_baseline_gf_home":  h["gf"],
        "promoted_baseline_ga_home":  h["ga"],
        "promoted_baseline_ppg_away": a["ppg"],
        "promoted_baseline_gf_away":  a["gf"],
        "promoted_baseline_ga_away":  a["ga"],
    }


# ============================================================
# 11. VENUE-SPECIFIC ROLLING FORM (added at Step 08B)
# ============================================================

def build_venue_form_filled(matches_df: pd.DataFrame,
                            prev_form_venue_df: pd.DataFrame,
                            promoted_baseline_venue: dict,
                            promoted_table: dict,
                            n: int = 5) -> pd.DataFrame:
    """
    Compute venue-conditional rolling form:

      home_venue_*_last{n} = home team's last n HOME matches
      away_venue_*_last{n} = away team's last n AWAY matches

    Season-scoped. Early-season NaN filled via weighted blend with
    venue-specific previous-season form (or venue-specific promoted baseline).

    Returns 8 columns:
      home_venue_pts_last{n}, home_venue_gf_last{n}, home_venue_ga_last{n}, home_venue_gd_last{n}
      away_venue_pts_last{n}, away_venue_gf_last{n}, away_venue_ga_last{n}, away_venue_gd_last{n}
    """
    df = matches_df.copy().reset_index(drop=True)
    df = df.sort_values(["season_label", "Date"]).reset_index(drop=True)
    df["_row_id"] = df.index

    # ---- Long form, one row per (match, team) ----
    def make_side(side):
        if side == "home":
            d = pd.DataFrame({
                "row_id": df["_row_id"],
                "Date": df["Date"],
                "season_label": df["season_label"],
                "team": df["HomeTeam"],
                "goals_for": df["FTHG"],
                "goals_against": df["FTAG"],
                "is_home": 1,
            })
        else:
            d = pd.DataFrame({
                "row_id": df["_row_id"],
                "Date": df["Date"],
                "season_label": df["season_label"],
                "team": df["AwayTeam"],
                "goals_for": df["FTAG"],
                "goals_against": df["FTHG"],
                "is_home": 0,
            })
        d["gd"] = d["goals_for"] - d["goals_against"]
        d["pts"] = np.where(d["goals_for"] > d["goals_against"], 3,
                     np.where(d["goals_for"] == d["goals_against"], 1, 0))
        return d

    long_df = pd.concat([make_side("home"), make_side("away")], ignore_index=True)
    long_df = long_df.sort_values(["team", "is_home", "season_label", "Date", "row_id"]).reset_index(drop=True)

    # ---- Rolling within (team, is_home, season) ----
    gb_cols = ["team", "is_home", "season_label"]
    def roll_venue(col):
        return (long_df.groupby(gb_cols)[col]
                       .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()))

    long_df["venue_pts_last"] = roll_venue("pts")
    long_df["venue_gf_last"]  = roll_venue("goals_for")
    long_df["venue_ga_last"]  = roll_venue("goals_against")
    long_df["venue_gd_last"]  = roll_venue("gd")

    # ---- Match-level output frame, aligned with `df` ----
    home_feats = long_df[long_df["is_home"] == 1].set_index("row_id")
    away_feats = long_df[long_df["is_home"] == 0].set_index("row_id")

    out = pd.DataFrame(index=df.index)
    for src, dst in [
        ("venue_pts_last", f"pts_last{n}"),
        ("venue_gf_last",  f"gf_last{n}"),
        ("venue_ga_last",  f"ga_last{n}"),
        ("venue_gd_last",  f"gd_last{n}"),
    ]:
        out[f"home_venue_{dst}"] = home_feats[src].reindex(df.index).values
        out[f"away_venue_{dst}"] = away_feats[src].reindex(df.index).values

    # ---- Compute k (in-season matches so far, per team per venue) ----
    df["_k_home_venue"] = df.groupby(["HomeTeam", "season_label"]).cumcount()
    df["_k_away_venue"] = df.groupby(["AwayTeam", "season_label"]).cumcount()
    # Count only same-venue matches: recompute via long table
    long_df["k"] = long_df.groupby(gb_cols).cumcount()
    home_k = long_df[long_df["is_home"] == 1].set_index("row_id")["k"].reindex(df.index).values
    away_k = long_df[long_df["is_home"] == 0].set_index("row_id")["k"].reindex(df.index).values

    # ---- Blend with venue-specific previous-season form ----
    prev_map = prev_form_venue_df.set_index(["team", "season_label"]).to_dict("index")

    def blend_side(side, k_arr):
        team_col = "HomeTeam" if side == "home" else "AwayTeam"
        venue_key = "home" if side == "home" else "away"
        prev_prefix = f"prev_season_ppg_{venue_key}", f"prev_season_gf_{venue_key}", f"prev_season_ga_{venue_key}"
        base_prefix = (f"promoted_baseline_ppg_{venue_key}",
                       f"promoted_baseline_gf_{venue_key}",
                       f"promoted_baseline_ga_{venue_key}")
        for metric, prev_key, base_key in zip(
            ["pts", "gf", "ga"], prev_prefix, base_prefix
        ):
            col = f"{side}_venue_{metric}_last{n}"
            values = out[col].values.copy()
            for i in range(len(df)):
                k = int(k_arr[i])
                if k >= n:
                    continue
                rolling_val = values[i]
                in_season_avg = 0.0 if pd.isna(rolling_val) else rolling_val
                team = df.iloc[i][team_col]
                season = df.iloc[i]["season_label"]
                prev = prev_map.get((team, season))
                if prev is not None:
                    prev_val = prev[prev_key]
                else:
                    prev_val = promoted_baseline_venue[base_key]
                values[i] = (k * in_season_avg + (n - k) * prev_val) / n
            out[col] = values

        # Derive gd from gf - ga
        out[f"{side}_venue_gd_last{n}"] = (
            out[f"{side}_venue_gf_last{n}"] - out[f"{side}_venue_ga_last{n}"]
        )

    blend_side("home", home_k)
    blend_side("away", away_k)

    # ---- Restore original order ----
    out.index = df.index
    return out


# ============================================================
# 12. VENUE-SPECIFIC PREVIOUS-SEASON MATCH STATS (added at Step 08C)
# ============================================================

def compute_prev_season_match_stats_venue(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team and each TARGET season, compute venue-specific match
    stats from the PREVIOUS season: shots, shots on target, corners.

    Returns columns:
      team, season_label (target), prev_season_label (source),
      prev_season_shots_home, prev_season_sot_home, prev_season_corners_home,
      prev_season_shots_away, prev_season_sot_away, prev_season_corners_away
    """
    rows = []
    for (season_label, team), group in _iter_team_seasons(train_df):
        home_group = group[group["HomeTeam"] == team]
        away_group = group[group["AwayTeam"] == team]

        def mean_or_nan(sub, cols):
            if len(sub) == 0:
                return tuple([np.nan] * len(cols))
            return tuple(float(sub[c].mean()) for c in cols)

        sh_h, sot_h, cn_h = mean_or_nan(home_group, ["HS", "HST", "HC"])
        sh_a, sot_a, cn_a = mean_or_nan(away_group, ["AS", "AST", "AC"])

        rows.append({
            "team": team,
            "prev_season_label": season_label,
            "shots_home": sh_h,  "sot_home": sot_h,  "corners_home": cn_h,
            "shots_away": sh_a,  "sot_away": sot_a,  "corners_away": cn_a,
        })

    end_of_season = pd.DataFrame(rows)

    season_order = sorted(end_of_season["prev_season_label"].unique())
    next_season = {s: season_order[i+1] for i, s in enumerate(season_order) if i+1 < len(season_order)}
    end_of_season["season_label"] = end_of_season["prev_season_label"].map(next_season)

    return (
        end_of_season
        .dropna(subset=["season_label"])
        .rename(columns={
            "shots_home":   "prev_season_shots_home",
            "sot_home":     "prev_season_sot_home",
            "corners_home": "prev_season_corners_home",
            "shots_away":   "prev_season_shots_away",
            "sot_away":     "prev_season_sot_away",
            "corners_away": "prev_season_corners_away",
        })
        [["team", "season_label", "prev_season_label",
          "prev_season_shots_home", "prev_season_sot_home", "prev_season_corners_home",
          "prev_season_shots_away", "prev_season_sot_away", "prev_season_corners_away"]]
        .reset_index(drop=True)
    )


# ============================================================
# 13. VENUE-SPECIFIC PROMOTED BASELINE FOR MATCH STATS
# ============================================================

def compute_promoted_baseline_match_stats(train_df: pd.DataFrame,
                                          promoted_table: dict,
                                          early_gameweeks: int = 5) -> dict:
    """
    Promoted-team baseline for shots, SOT, corners — venue-specific.
    """
    train = train_df.copy().sort_values("Date").reset_index(drop=True)

    home_rows, away_rows = [], []
    for season_label, group in train.groupby("season_label"):
        promoted = get_promoted_teams(promoted_table, season_label)
        if not promoted:
            continue
        for team in promoted:
            team_matches = train[
                (train["season_label"] == season_label) &
                ((train["HomeTeam"] == team) | (train["AwayTeam"] == team))
            ].sort_values("Date").head(early_gameweeks)
            home_rows.append(team_matches[team_matches["HomeTeam"] == team])
            away_rows.append(team_matches[team_matches["AwayTeam"] == team])

    def summarize(rows, cols):
        non_empty = [r for r in rows if len(r) > 0]
        if not non_empty:
            return {c: 0.0 for c in cols}
        bdf = pd.concat(non_empty, ignore_index=True)
        return {c: float(bdf[c].mean()) for c in cols}

    home_stats = summarize(home_rows, ["HS", "HST", "HC"])
    away_stats = summarize(away_rows, ["AS", "AST", "AC"])

    return {
        "promoted_baseline_shots_home":   home_stats["HS"],
        "promoted_baseline_sot_home":     home_stats["HST"],
        "promoted_baseline_corners_home": home_stats["HC"],
        "promoted_baseline_shots_away":   away_stats["AS"],
        "promoted_baseline_sot_away":     away_stats["AST"],
        "promoted_baseline_corners_away": away_stats["AC"],
    }


# ============================================================
# 14. ROLLING MATCH STATS (venue-specific, season-scoped)
# ============================================================

def build_match_stats_filled(matches_df: pd.DataFrame,
                             prev_stats_df: pd.DataFrame,
                             promoted_baseline_stats: dict,
                             promoted_table: dict,
                             n: int = 5) -> pd.DataFrame:
    """
    Rolling last-5 match stats, per team per venue, season-scoped.

    Output columns (8):
      home_form_shots_last{n},     away_form_shots_last{n}
      home_form_sot_last{n},       away_form_sot_last{n}
      home_form_corners_last{n},   away_form_corners_last{n}
      home_form_sot_ratio_last{n}, away_form_sot_ratio_last{n}
    """
    df = matches_df.copy().reset_index(drop=True).sort_values(
        ["season_label", "Date"]).reset_index(drop=True)
    df["_row_id"] = df.index

    # ---- Long form, per (match, team) with venue ----
    def make_side(side):
        if side == "home":
            d = pd.DataFrame({
                "row_id": df["_row_id"],
                "Date": df["Date"],
                "season_label": df["season_label"],
                "team": df["HomeTeam"],
                "shots": df["HS"], "sot": df["HST"], "corners": df["HC"],
                "is_home": 1,
            })
        else:
            d = pd.DataFrame({
                "row_id": df["_row_id"],
                "Date": df["Date"],
                "season_label": df["season_label"],
                "team": df["AwayTeam"],
                "shots": df["AS"], "sot": df["AST"], "corners": df["AC"],
                "is_home": 0,
            })
        d["sot_ratio"] = np.where(d["shots"] > 0, d["sot"] / d["shots"], np.nan)
        return d

    long_df = pd.concat([make_side("home"), make_side("away")], ignore_index=True)
    long_df = long_df.sort_values(["team", "is_home", "season_label", "Date", "row_id"]).reset_index(drop=True)

    gb_cols = ["team", "is_home", "season_label"]
    def roll_stat(col):
        return (long_df.groupby(gb_cols)[col]
                       .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()))

    long_df["stat_shots_last"]   = roll_stat("shots")
    long_df["stat_sot_last"]     = roll_stat("sot")
    long_df["stat_corners_last"] = roll_stat("corners")
    long_df["stat_sot_ratio_last"] = roll_stat("sot_ratio")

    # ---- Back to match level ----
    home_feats = long_df[long_df["is_home"] == 1].set_index("row_id")
    away_feats = long_df[long_df["is_home"] == 0].set_index("row_id")

    out = pd.DataFrame(index=df.index)
    for src, dst in [
        ("stat_shots_last",     f"form_shots_last{n}"),
        ("stat_sot_last",       f"form_sot_last{n}"),
        ("stat_corners_last",   f"form_corners_last{n}"),
        ("stat_sot_ratio_last", f"form_sot_ratio_last{n}"),
    ]:
        out[f"home_{dst}"] = home_feats[src].reindex(df.index).values
        out[f"away_{dst}"] = away_feats[src].reindex(df.index).values

    # ---- k (in-season venue matches so far) ----
    long_df["k"] = long_df.groupby(gb_cols).cumcount()
    home_k = long_df[long_df["is_home"] == 1].set_index("row_id")["k"].reindex(df.index).values
    away_k = long_df[long_df["is_home"] == 0].set_index("row_id")["k"].reindex(df.index).values

    # ---- Blend with venue-specific previous-season stats ----
    prev_map = prev_stats_df.set_index(["team", "season_label"]).to_dict("index")

    stat_specs = [
        ("shots",     "form_shots_last{n}",     "prev_season_shots_{v}",     "promoted_baseline_shots_{v}"),
        ("sot",       "form_sot_last{n}",       "prev_season_sot_{v}",       "promoted_baseline_sot_{v}"),
        ("corners",   "form_corners_last{n}",   "prev_season_corners_{v}",   "promoted_baseline_corners_{v}"),
    ]

    def blend_side(side, k_arr):
        team_col = "HomeTeam" if side == "home" else "AwayTeam"
        venue = "home" if side == "home" else "away"
        for _, col_tmpl, prev_tmpl, base_tmpl in stat_specs:
            col = f"{side}_{col_tmpl.format(n=n)}"
            prev_key = prev_tmpl.format(v=venue)
            base_key = base_tmpl.format(v=venue)
            values = out[col].values.copy()
            for i in range(len(df)):
                k = int(k_arr[i])
                if k >= n:
                    continue
                rolling_val = values[i]
                in_season_avg = 0.0 if pd.isna(rolling_val) else rolling_val
                team = df.iloc[i][team_col]
                season = df.iloc[i]["season_label"]
                prev = prev_map.get((team, season))
                if prev is not None and not pd.isna(prev[prev_key]):
                    prev_val = prev[prev_key]
                else:
                    prev_val = promoted_baseline_stats[base_key]
                values[i] = (k * in_season_avg + (n - k) * prev_val) / n
            out[col] = values

        # sot_ratio: derive from blended shots and sot
        col_sot = f"{side}_form_sot_last{n}"
        col_sh  = f"{side}_form_shots_last{n}"
        col_ratio = f"{side}_form_sot_ratio_last{n}"
        safe_sh = out[col_sh].replace(0, np.nan)
        out[col_ratio] = (out[col_sot] / safe_sh).fillna(0.0)

    blend_side("home", home_k)
    blend_side("away", away_k)

    out.index = df.index
    # Restore original (pre-sort) row order
    out = out.reindex(matches_df.index)
    return out


# ============================================================
# 15. SEASON-TO-DATE FEATURES (added at Step 08D)
# ============================================================

def compute_prev_season_gd_per_game_venue(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extend the venue-specific previous-season form with GD per game.
    Small helper that just derives from the existing lookup.
    """
    prev = compute_prev_season_form_venue(train_df)
    prev = prev.copy()
    prev["prev_season_gd_per_game_home"] = prev["prev_season_gf_home"] - prev["prev_season_ga_home"]
    prev["prev_season_gd_per_game_away"] = prev["prev_season_gf_away"] - prev["prev_season_ga_away"]
    return prev


def build_season_to_date_filled(matches_df: pd.DataFrame,
                                prev_form_venue_df: pd.DataFrame,
                                promoted_baseline_venue: dict,
                                promoted_table: dict) -> pd.DataFrame:
    """
    Cumulative season-to-date features, computed as of BEFORE the current
    match. Season-scoped (resets at season start).

    Output columns (6):
      home_season_ppg,           away_season_ppg
      home_season_gd_per_game,   away_season_gd_per_game
      home_season_matches_played, away_season_matches_played

    Filling rules:
      ppg and gd_per_game: C3 weighted blend using venue-specific prev-season form
      matches_played: actual count (no fill)
    """
    df = matches_df.copy().reset_index(drop=True).sort_values(
        ["season_label", "Date"]).reset_index(drop=True)
    df["_row_id"] = df.index

    # ---- Long form per (match, team) with venue ----
    def make_side(side):
        if side == "home":
            d = pd.DataFrame({
                "row_id": df["_row_id"],
                "Date": df["Date"],
                "season_label": df["season_label"],
                "team": df["HomeTeam"],
                "goals_for":     df["FTHG"],
                "goals_against": df["FTAG"],
                "is_home": 1,
            })
        else:
            d = pd.DataFrame({
                "row_id": df["_row_id"],
                "Date": df["Date"],
                "season_label": df["season_label"],
                "team": df["AwayTeam"],
                "goals_for":     df["FTAG"],
                "goals_against": df["FTHG"],
                "is_home": 0,
            })
        d["gd"] = d["goals_for"] - d["goals_against"]
        d["pts"] = np.where(d["goals_for"] > d["goals_against"], 3,
                     np.where(d["goals_for"] == d["goals_against"], 1, 0))
        return d

    long_df = pd.concat([make_side("home"), make_side("away")], ignore_index=True)
    long_df = long_df.sort_values(["team", "season_label", "Date", "row_id"]).reset_index(drop=True)

    # ---- Cumulative season-to-date (shifted by one match) ----
    gb = long_df.groupby(["team", "season_label"])

    long_df["_cum_pts"]  = gb["pts"].transform(lambda s: s.shift(1).expanding().sum())
    long_df["_cum_gd"]   = gb["gd"].transform(lambda s: s.shift(1).expanding().sum())
    long_df["_cum_n"]    = gb.cumcount()  # matches played BEFORE current match

    # ppg and gd_per_game = cum / n (n = matches played before current match)
    safe_n = long_df["_cum_n"].replace(0, np.nan)
    long_df["std_ppg"]  = long_df["_cum_pts"] / safe_n
    long_df["std_gdg"]  = long_df["_cum_gd"]  / safe_n
    long_df["std_n"]    = long_df["_cum_n"]

    # ---- To match-level ----
    home_feats = long_df[long_df["is_home"] == 1].set_index("row_id")
    away_feats = long_df[long_df["is_home"] == 0].set_index("row_id")

    out = pd.DataFrame(index=df.index)
    out["home_season_ppg"]              = home_feats["std_ppg"].reindex(df.index).values
    out["away_season_ppg"]              = away_feats["std_ppg"].reindex(df.index).values
    out["home_season_gd_per_game"]      = home_feats["std_gdg"].reindex(df.index).values
    out["away_season_gd_per_game"]      = away_feats["std_gdg"].reindex(df.index).values
    out["home_season_matches_played"]   = home_feats["std_n"].reindex(df.index).values
    out["away_season_matches_played"]   = away_feats["std_n"].reindex(df.index).values

    # ---- Fill ppg / gd_per_game via C3 blend (matches_played NOT filled) ----
    prev_map = prev_form_venue_df.set_index(["team", "season_label"]).to_dict("index")

    def blend_side(side, k_arr):
        team_col = "HomeTeam" if side == "home" else "AwayTeam"
        venue    = "home" if side == "home" else "away"
        specs = [
            ("ppg",         f"{side}_season_ppg",         f"prev_season_ppg_{venue}",         f"promoted_baseline_ppg_{venue}"),
            ("gd_per_game", f"{side}_season_gd_per_game", f"prev_season_gd_per_game_{venue}", None),
        ]
        for metric, col, prev_key, base_key in specs:
            values = out[col].values.copy()
            for i in range(len(df)):
                k = int(k_arr[i])
                if k >= 1:
                    continue  # has in-season data → ppg is genuine
                team = df.iloc[i][team_col]
                season = df.iloc[i]["season_label"]
                prev = prev_map.get((team, season))
                if prev is not None and not pd.isna(prev[prev_key]):
                    values[i] = prev[prev_key]
                else:
                    # promoted: use promoted baseline
                    if base_key is None:
                        # gd_per_game: derive from gf - ga baseline if needed
                        base_val = (promoted_baseline_venue.get(f"promoted_baseline_gf_{venue}", 1.0)
                                    - promoted_baseline_venue.get(f"promoted_baseline_ga_{venue}", 1.7))
                    else:
                        base_val = promoted_baseline_venue.get(base_key, 0.0)
                    values[i] = base_val
            out[col] = values

    home_k = home_feats["std_n"].reindex(df.index).values
    away_k = away_feats["std_n"].reindex(df.index).values
    blend_side("home", home_k)
    blend_side("away", away_k)

    # ---- Restore original row order ----
    out = out.reindex(matches_df.index)
    return out


# ============================================================
# 16. REST DAYS (added at Step 08E)
# ============================================================

def build_rest_days(matches_df: pd.DataFrame,
                    cap: int = 14) -> pd.DataFrame:
    """
    For each match, compute days of rest for the home and away team
    since their previous match in the dataset.

    Rules:
      - Rest = (this match date) - (team's previous match date) in days.
      - Cap at `cap` (default 14) to tame the long tail.
      - First-ever match in dataset -> cap.
      - Season openers naturally resolve to cap.

    Output columns (2): home_rest_days, away_rest_days
    """
    df = matches_df.copy().reset_index(drop=True).sort_values(
        ["season_label", "Date"]).reset_index(drop=True)
    df["_row_id"] = df.index

    # Long form per (match, team)
    def make_side(side):
        team_col = "HomeTeam" if side == "home" else "AwayTeam"
        return pd.DataFrame({
            "row_id": df["_row_id"],
            "Date": df["Date"],
            "team": df[team_col],
            "is_home": 1 if side == "home" else 0,
        })

    long_df = pd.concat([make_side("home"), make_side("away")], ignore_index=True)
    long_df = long_df.sort_values(["team", "Date", "row_id"]).reset_index(drop=True)

    # Previous match date per team (shifted by one)
    long_df["prev_date"] = long_df.groupby("team")["Date"].shift(1)

    # Days since previous match
    long_df["rest_days"] = (long_df["Date"] - long_df["prev_date"]).dt.days

    # Fill NaN (first ever match) and cap
    long_df["rest_days"] = long_df["rest_days"].fillna(cap).clip(upper=cap)

    # Back to match-level
    home_feats = long_df[long_df["is_home"] == 1].set_index("row_id")
    away_feats = long_df[long_df["is_home"] == 0].set_index("row_id")

    out = pd.DataFrame(index=df.index)
    out["home_rest_days"] = home_feats["rest_days"].reindex(df.index).values
    out["away_rest_days"] = away_feats["rest_days"].reindex(df.index).values

    # Restore original order
    out = out.reindex(matches_df.index)
    return out


# ============================================================
# 17. ELO RATINGS (added at Step 08F)
# ============================================================

def build_elo_features(matches_df: pd.DataFrame,
                       k_factor: float = 20.0,
                       home_bonus: float = 60.0,
                       initial_elo: float = 1500.0,
                       regression: float = 0.25) -> pd.DataFrame:
    """
    Compute pre-match Elo for home and away teams.

    Rewritten to use purely positional iteration over a date-sorted copy,
    with no ambiguous index alignment.
    """
    df = matches_df.copy().reset_index(drop=True).sort_values(
        ["season_label", "Date"]).reset_index(drop=True)

    elo = {}
    last_season = None
    home_elos = [None] * len(df)
    away_elos = [None] * len(df)

    for pos in range(len(df)):
        row = df.iloc[pos]
        season = row["season_label"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Season boundary: regress every team once
        if season != last_season:
            if last_season is not None:
                for t in list(elo.keys()):
                    elo[t] = elo[t] + regression * (initial_elo - elo[t])
            last_season = season

        if home not in elo: elo[home] = initial_elo
        if away not in elo: elo[away] = initial_elo

        elo_h = elo[home]
        elo_a = elo[away]

        # Store PRE-match Elo
        home_elos[pos] = elo_h
        away_elos[pos] = elo_a

        # Update
        E_h = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h - home_bonus) / 400.0))
        E_a = 1.0 - E_h
        if row["FTHG"] > row["FTAG"]:
            S_h, S_a = 1.0, 0.0
        elif row["FTHG"] < row["FTAG"]:
            S_h, S_a = 0.0, 1.0
        else:
            S_h, S_a = 0.5, 0.5

        elo[home] = elo_h + k_factor * (S_h - E_h)
        elo[away] = elo_a + k_factor * (S_a - E_a)

    out = pd.DataFrame({"home_elo": home_elos, "away_elo": away_elos},
                       index=df.index)
    # Map back to matches_df original row order
    out = out.reindex(matches_df.index)
    return out


# ============================================================
# 18. PREVIOUS-SEASON PPG AS A DIRECT FEATURE (added at Step 08F)
# ============================================================

def build_prev_season_ppg_features(matches_df: pd.DataFrame,
                                   prev_form_venue_df: pd.DataFrame,
                                   promoted_baseline_venue: dict,
                                   promoted_table: dict) -> pd.DataFrame:
    """
    Direct feature: venue-specific previous-season PPG for both sides.

    For each match, look up the home team's previous-season home-venue PPG
    and the away team's previous-season away-venue PPG. Fall back to
    promoted baseline for promoted teams or teams without history.

    Output columns (2): home_prev_season_ppg, away_prev_season_ppg
    """
    df = matches_df.copy().reset_index(drop=True)
    prev_map = prev_form_venue_df.set_index(["team", "season_label"]).to_dict("index")

    def lookup(team, season, venue):
        prev = prev_map.get((team, season))
        if prev is not None and not pd.isna(prev[f"prev_season_ppg_{venue}"]):
            return prev[f"prev_season_ppg_{venue}"]
        return promoted_baseline_venue.get(f"promoted_baseline_ppg_{venue}", 1.0)

    out = pd.DataFrame(index=df.index)
    out["home_prev_season_ppg"] = df.apply(
        lambda r: lookup(r["HomeTeam"], r["season_label"], "home"), axis=1).values
    out["away_prev_season_ppg"] = df.apply(
        lambda r: lookup(r["AwayTeam"], r["season_label"], "away"), axis=1).values

    out.index = matches_df.index
    return out


# ============================================================
# 19. MATCHUP DELTAS (added at Step 08G)
# ============================================================

DELTA_SPECS = [
    ("delta_elo",                "home_elo",                "away_elo"),
    ("delta_form_pts_last5",     "home_form_pts_last5",     "away_form_pts_last5"),
    ("delta_form_gd_last5",      "home_form_gd_last5",      "away_form_gd_last5"),
    ("delta_venue_pts_last5",    "home_venue_pts_last5",    "away_venue_pts_last5"),
    ("delta_season_ppg",         "home_season_ppg",         "away_season_ppg"),
    ("delta_prev_season_ppg",    "home_prev_season_ppg",    "away_prev_season_ppg"),
    ("delta_form_shots_last5",   "home_form_shots_last5",   "away_form_shots_last5"),
    ("delta_form_sot_last5",     "home_form_sot_last5",     "away_form_sot_last5"),
    ("delta_rest_days",          "home_rest_days",          "away_rest_days"),
]


def build_deltas(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame that already contains home_* and away_* columns,
    compute the derived delta columns.

    delta_X = home_X - away_X

    Returns a DataFrame with only the delta columns, indexed like the input.
    """
    out = pd.DataFrame(index=feature_df.index)
    for name, home_col, away_col in DELTA_SPECS:
        assert home_col in feature_df.columns, f"Missing {home_col}"
        assert away_col in feature_df.columns, f"Missing {away_col}"
        out[name] = feature_df[home_col].values - feature_df[away_col].values
    return out


# ============================================================
# 20. MARGIN-OF-VICTORY ELO (added at Step 12c)
# ============================================================

def build_mov_elo_features(matches_df: pd.DataFrame,
                           k_factor: float = 20.0,
                           home_bonus: float = 60.0,
                           initial_elo: float = 1500.0,
                           regression: float = 0.25) -> pd.DataFrame:
    """
    Margin-of-victory Elo. Same as build_elo_features, but the K-factor
    is scaled by goal difference:

        mult = 1 + log(1 + goal_diff)   # 0 -> 1.0, 1 -> 1.69, 3 -> 2.39
        update = K * mult * (S - E)

    This makes convincing wins move the rating more than narrow wins.
    """
    import numpy as np
    df = matches_df.copy().reset_index(drop=True).sort_values(
        ["season_label", "Date"]).reset_index(drop=True)

    elo = {}
    last_season = None
    home_elos = [None] * len(df)
    away_elos = [None] * len(df)

    for pos in range(len(df)):
        row = df.iloc[pos]
        season = row["season_label"]
        home = row["HomeTeam"]; away = row["AwayTeam"]

        if season != last_season:
            if last_season is not None:
                for t in list(elo.keys()):
                    elo[t] = elo[t] + regression * (initial_elo - elo[t])
            last_season = season

        if home not in elo: elo[home] = initial_elo
        if away not in elo: elo[away] = initial_elo

        elo_h = elo[home]; elo_a = elo[away]
        home_elos[pos] = elo_h
        away_elos[pos] = elo_a

        E_h = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h - home_bonus) / 400.0))
        E_a = 1.0 - E_h

        if row["FTHG"] > row["FTAG"]:
            S_h, S_a = 1.0, 0.0
        elif row["FTHG"] < row["FTAG"]:
            S_h, S_a = 0.0, 1.0
        else:
            S_h, S_a = 0.5, 0.5

        gd = abs(int(row["FTHG"]) - int(row["FTAG"]))
        mult = 1.0 + np.log1p(gd)   # log(1) = 0 for a draw, log(2)=0.69 for 1-goal, etc.

        elo[home] = elo_h + k_factor * mult * (S_h - E_h)
        elo[away] = elo_a + k_factor * mult * (S_a - E_a)

    out = pd.DataFrame({
        "home_elo_mov": home_elos,
        "away_elo_mov": away_elos,
    }, index=df.index)
    out["delta_elo_mov"] = out["home_elo_mov"] - out["away_elo_mov"]
    out = out.reindex(matches_df.index)
    return out
