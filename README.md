# AI Transformation Readiness — Streamlit Dashboard

An interactive descriptive-analytics dashboard for the synthetic AI-Enabled Digital
Transformation Readiness dataset (1,200 working-professional respondents, 83 columns).

Covers frequency tables, descriptive statistics, cross-tabs, a correlation heatmap,
segment/clustering exploration (with a live, re-runnable K-means demo), simple
classification (AI_Adopter_Flag) and regression (AI_Transformation_Readiness_Score)
signal checks, scatter/box plots, the full data dictionary, and a filterable raw-data
explorer with CSV download.

## Repo structure

```
.
├── app.py                  # the Streamlit app (single file, multi-page via sidebar)
├── requirements.txt
├── .streamlit/
│   └── config.toml         # theme
├── AI_Transformation_Readiness_Synthetic_Dataset.csv
├── Data_Dictionary.csv
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy for free on Streamlit Community Cloud

1. Push this folder to a **public** GitHub repository (or a private one if you're on
   a paid Streamlit Cloud plan).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick this repository and branch, and set the main file path to
   `app.py`.
4. Click **Deploy**. The first build takes a couple of minutes while it installs
   `requirements.txt`; after that it's live at a `*.streamlit.app` URL you can share.

No API keys or secrets are needed — the app reads the CSV files bundled in the
`data/` folder, so it works immediately after deployment.

## Notes

- `True_Segment` is a ground-truth persona label included for teaching/validation.
  The Segments & Clustering page deliberately does **not** use it as a clustering
  input — it only compares the discovered clusters back to it afterward, which is
  why the Adjusted Rand Index metric is shown rather than treating clustering as
  pre-labelled.
- The dataset is **synthetic**, generated for an academic Data Analytics in Decision
  Making project. It is realistic enough to demonstrate meaningful descriptive,
  diagnostic, classification, clustering, and regression analytics, but it is not
  real survey data and should not be cited as such.
- To refresh the data, replace the two CSVs at the repo root with new exports that
  share the same column names, then redeploy (Streamlit Cloud auto-redeploys on
  every push to the connected branch).
