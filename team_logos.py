"""
team_logos.py  -  crest URLs, display names and name aliases for the 20 EPL clubs.

HOW TO SWAP IN YOUR OWN LOGOS
-----------------------------
1. Easiest: put a URL in LOGO_OVERRIDES below, keyed by the team name your model uses.
       LOGO_OVERRIDES = {"Arsenal": "https://my-cdn.com/arsenal.png"}
2. Or change PL_BADGE_URL if you find a better source for the whole set.
3. Keys must match the names returned by get_available_teams()
   (football-data.co.uk style: "Man City", "Man United", "Nott'm Forest", ...).
   Common alternative spellings are handled by ALIASES.

If a URL cannot be loaded, app.py automatically draws a purple crest with the
club's 3-letter code, so the UI never shows a broken image.
"""

# Premier League's public badge CDN. {code} is the club's Opta team code (the "t" number).
PL_BADGE_URL = "https://resources.premierleague.com/premierleague/badges/70/t{code}.png"

# Club -> Opta team code used in the URL above.
TEAM_CODES = {
    "Arsenal": 3,
    "Aston Villa": 7,
    "Bournemouth": 91,
    "Brentford": 94,
    "Brighton": 36,
    "Burnley": 90,
    "Chelsea": 8,
    "Crystal Palace": 31,
    "Everton": 11,
    "Fulham": 54,
    "Leeds": 2,
    "Liverpool": 14,
    "Man City": 43,
    "Man United": 1,
    "Newcastle": 4,
    "Nott'm Forest": 17,
    "Sunderland": 56,
    "Tottenham": 6,
    "West Ham": 21,
    "Wolves": 39,
}

# Team name -> logo URL. This is the dictionary the app reads.
TEAM_LOGOS = {team: PL_BADGE_URL.format(code=code) for team, code in TEAM_CODES.items()}

# Put your own URLs here; they win over the defaults above.
LOGO_OVERRIDES = {
    # "Arsenal": "https://example.com/arsenal.png",
}
TEAM_LOGOS.update(LOGO_OVERRIDES)

# Nicer names for the dropdowns and result cards (the model still receives the short name).
DISPLAY_NAMES = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "West Ham": "West Ham United",
    "Leeds": "Leeds United",
    "Brighton": "Brighton & Hove Albion",
}

# Three-letter codes for the fallback crest.
TLA = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Burnley": "BUR", "Chelsea": "CHE", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Leeds": "LEE", "Liverpool": "LIV",
    "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW", "Nott'm Forest": "NFO",
    "Sunderland": "SUN", "Tottenham": "TOT", "West Ham": "WHU", "Wolves": "WOL",
}

# Other spellings -> the canonical name used as the key everywhere above.
ALIASES = {
    "manchester city": "Man City", "man city": "Man City", "mci": "Man City",
    "manchester united": "Man United", "manchester utd": "Man United", "man utd": "Man United",
    "nottingham forest": "Nott'm Forest", "nottm forest": "Nott'm Forest", "nott'm forest": "Nott'm Forest",
    "nott'ham forest": "Nott'm Forest",
    "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
    "newcastle united": "Newcastle", "newcastle utd": "Newcastle",
    "west ham united": "West Ham", "leeds united": "Leeds",
    "brighton & hove albion": "Brighton", "brighton and hove albion": "Brighton",
    "afc bournemouth": "Bournemouth", "sunderland afc": "Sunderland",
}

_LOOKUP = {name.lower(): name for name in TEAM_CODES}
_LOOKUP.update(ALIASES)


def canonical_team(name: str) -> str:
    """Map any common spelling of a club to the key used in TEAM_LOGOS."""
    key = " ".join(str(name).replace("\u2019", "'").lower().split())
    return _LOOKUP.get(key, name)
