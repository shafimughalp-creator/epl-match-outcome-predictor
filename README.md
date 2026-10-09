# Premier League Match Outcome Predictor

Streamlit app for an EPL match outcome model (Home win / Draw / Away win).
Logistic Regression with Lasso regularisation on 9 delta features, test log loss 1.0238 on the locked 2025/26 season.

Portfolio project for learning. Not for betting or financial use.

## Folder structure

Create exactly this layout on your laptop before pushing to GitHub:

```
your-repo/
├── app.py                  # the Streamlit UI
├── team_logos.py           # TEAM_LOGOS mapping, display names, aliases
├── predict_logic.py        # get_available_teams() and predict_fixture()
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml         # MUST be inside the .streamlit/ folder
├── models/                 # saved model files (.joblib / .pkl)
├── data/                   # historical match data used to rebuild features
└── src/                    # helper code imported by predict_logic.py
```

Two rules that cause most deployment failures:

1. **`config.toml` must live at `.streamlit/config.toml`.** Streamlit does not read it from the repo root. The folder name starts with a dot and must be spelled exactly `.streamlit`.
2. **`models/`, `data/` and `src/` must be committed to GitHub.** `predict_logic.py` reads them at import time, so if they are missing the app falls back to DEMO MODE (amber banner, placeholder numbers) or crashes on start.

Also check:

- Your `.gitignore` does not exclude `*.pkl`, `*.joblib`, `*.csv` or whole folders such as `data/` or `models/`.
- If `predict_logic.py` does `from src.something import ...`, add an empty `src/__init__.py`.
- Paths inside `predict_logic.py` should be built from the file's own location, for example `Path(__file__).parent / "models" / "model.joblib"`, not from the working directory.
- GitHub rejects single files over 100 MB.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. **Pin your scikit-learn version.** Run `pip show scikit-learn` locally and put the exact version in `requirements.txt` (for example `scikit-learn==1.5.2`). A version mismatch is the most common reason a saved model fails to load.
2. **Push the repo to GitHub** (public repos work on the free tier):
   ```
   git init
   git add .
   git status        # check .streamlit/, models/, data/ and src/ are listed
   git commit -m "EPL match outcome predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. Go to **share.streamlit.io** and sign in with GitHub.
4. Click **Create app** and choose **Deploy a public app from GitHub**.
5. Select your repository, the `main` branch, and set **Main file path** to `app.py`.
6. Open **Advanced settings** and choose the same Python version you use locally.
7. Click **Deploy**. The first build takes a few minutes while dependencies install.
8. If you see the amber **DEMO MODE** banner, open **Manage app > Logs** and read the import error. It almost always means a missing folder or a wrong file path.
9. Replace `GITHUB_MODEL_CARD_URL` near the top of `app.py` with your real model-card link, then push again. Streamlit redeploys automatically on every push.

Free apps go to sleep after a period of inactivity and wake with one click, so open yours before showing it to anyone.

## Customising

- **Logos:** add URLs to `LOGO_OVERRIDES` in `team_logos.py`. Crests are downloaded once per server start and cached. If one cannot be loaded, a purple crest with the club's 3-letter code is drawn instead.
- **Colours:** change the CSS variables at the top of the `CSS` string in `app.py` (section 2.0).
- **Model card text:** edit `MODEL_CARD_FACTS` and `MODEL_CARD_LIMITS` near the top of `app.py`.
