import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Transformation Readiness Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "AI_Transformation_Readiness_Synthetic_Dataset.csv"
DICT_PATH = Path(__file__).parent / "Data_Dictionary.csv"

NAVY = "#1F3864"
BLUE = "#2E75B6"
ORANGE = "#E08E45"
GREY = "#666666"
SEGMENT_COLORS = {
    "AI Champion": "#2E7D32",
    "Transformation Leader": "#2E75B6",
    "Cautious Pragmatist": "#E0A458",
    "Change Fatigued": "#C0622D",
    "Digital Resister": "#A4262C",
}
COMPOSITE_COLS = [
    "Leadership_Readiness_Score", "Training_Quality_Score", "AI_Trust_Score",
    "Resistance_Score", "Change_Fatigue_Score", "Digital_Confidence_Score",
    "Behavioural_Adoption_Intent_Score", "AI_Readiness_Index", "Digital_Maturity_Index",
    "AI_Transformation_Readiness_Score",
]

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def load_dictionary():
    if DICT_PATH.exists():
        return pd.read_csv(DICT_PATH)
    return None

df = load_data()
dict_df = load_dictionary()

# -----------------------------------------------------------------------------
# SIDEBAR — NAVIGATION + GLOBAL FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("📊 AI Readiness Dashboard")
st.sidebar.caption("AI-Enabled Digital Transformation Readiness — Synthetic Respondent Data")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Frequency Tables",
        "Descriptives",
        "Cross-Tabs",
        "Correlation Heatmap",
        "Segments & Clustering",
        "Classification & Regression",
        "Scatter & Box Plots",
        "Data Dictionary",
        "Raw Data Explorer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

segments_available = sorted(df["True_Segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Segment", segments_available, default=segments_available
)

industries_available = sorted(df["Industry_Sector"].dropna().unique().tolist())
selected_industries = st.sidebar.multiselect(
    "Industry Sector", industries_available, default=industries_available
)

adopter_filter = st.sidebar.radio(
    "AI Adopter Flag", ["All", "Adopters only (1)", "Non-adopters only (0)"], index=0
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "True_Segment is a ground-truth label included for teaching/validation. "
    "Drop it before treating clustering as a blind exercise."
)

# apply filters
fdf = df[df["True_Segment"].isin(selected_segments) & df["Industry_Sector"].isin(selected_industries)].copy()
if adopter_filter == "Adopters only (1)":
    fdf = fdf[fdf["AI_Adopter_Flag"] == 1]
elif adopter_filter == "Non-adopters only (0)":
    fdf = fdf[fdf["AI_Adopter_Flag"] == 0]

st.sidebar.markdown(f"**Filtered respondents:** {len(fdf):,} / {len(df):,}")

# -----------------------------------------------------------------------------
# SHARED HELPERS
# -----------------------------------------------------------------------------
def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
if page == "Overview":
    st.title("AI-Enabled Digital Transformation Readiness")
    st.caption("Synthetic respondent dataset · descriptive analytics dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Respondents (filtered)", f"{len(fdf):,}")
    with c2:
        adopt_rate = fdf["AI_Adopter_Flag"].mean() * 100 if len(fdf) else 0
        kpi_card("Adopter Rate", f"{adopt_rate:.1f}%")
    with c3:
        avg_readiness = fdf["AI_Transformation_Readiness_Score"].mean() if len(fdf) else 0
        kpi_card("Avg. Transformation Readiness", f"{avg_readiness:.1f} / 100")
    with c4:
        avg_trust = fdf["AI_Trust_Score"].mean() if len(fdf) else 0
        kpi_card("Avg. AI Trust Score", f"{avg_trust:.1f} / 100")

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### Mean Transformation Readiness by Segment")
        seg_means = (
            fdf.groupby("True_Segment")["AI_Transformation_Readiness_Score"]
            .mean().reindex(["AI Champion","Transformation Leader","Cautious Pragmatist","Change Fatigued","Digital Resister"])
            .dropna().reset_index()
        )
        fig = px.bar(
            seg_means, x="True_Segment", y="AI_Transformation_Readiness_Score",
            color="True_Segment", color_discrete_map=SEGMENT_COLORS,
            text_auto=".1f",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Mean Readiness Score (0-100)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Segment Distribution")
        seg_counts = fdf["True_Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig2 = px.pie(
            seg_counts, names="Segment", values="Count",
            color="Segment", color_discrete_map=SEGMENT_COLORS, hole=0.45,
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### About this dataset")
    st.markdown(
        """
This is a **synthetic** dataset of 1,200 working-professional respondents built from a 63-question
AI Transformation Readiness research instrument. Respondents were generated from five hidden behavioural
personas (AI Champion, Transformation Leader, Cautious Pragmatist, Change Fatigued, Digital Resister) with
realistic noise, missing values, outliers, skew, and a small number of deliberately contradictory responses,
so the dataset behaves like real survey data rather than an idealised one.

Use the sidebar to navigate between descriptive analytics views, or filter by segment, industry, and adopter status.
"""
    )

# =============================================================================
# PAGE: FREQUENCY TABLES
# =============================================================================
elif page == "Frequency Tables":
    section_header("Frequency Tables", "Counts and percentages for key categorical variables")

    freq_cols = [
        "True_Segment", "Industry_Sector", "Org_Level", "Prior_AI_Experience",
        "Training_Received", "AI_Strategy_Communicated", "Scenario_Response",
        "Barrier_Category", "AI_Adopter_Flag",
    ]
    chosen = st.selectbox("Choose a variable", freq_cols)

    counts = fdf[chosen].value_counts(dropna=False)
    pct = fdf[chosen].value_counts(normalize=True, dropna=False) * 100
    table = pd.DataFrame({"Count": counts, "Percent": pct.round(1)}).reset_index()
    table.columns = [chosen, "Count", "Percent"]

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.dataframe(table, use_container_width=True, hide_index=True)
    with col2:
        fig = px.bar(table, x=chosen, y="Count", text="Percent", color=chosen)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, height=420, xaxis_title="", margin=dict(b=120))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: DESCRIPTIVES
# =============================================================================
elif page == "Descriptives":
    section_header("Descriptive Statistics", "Mean, standard deviation, and range for composite scores and usage hours")

    desc_cols = COMPOSITE_COLS + ["AI_Usage_Hours_Monthly"]
    desc = fdf[desc_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    desc.columns = ["Mean", "Std Dev", "Min", "P25", "Median", "P75", "Max"]
    st.dataframe(desc, use_container_width=True)

    st.markdown("#### Mean Composite Scores")
    means = fdf[COMPOSITE_COLS].mean().sort_values(ascending=True).reset_index()
    means.columns = ["Variable", "Mean"]
    fig = px.bar(means, x="Mean", y="Variable", orientation="h", text_auto=".1f", color="Mean", color_continuous_scale="Blues")
    fig.update_layout(height=450, coloraxis_showscale=False, xaxis_title="Mean Score (0-100)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Distribution of a Single Variable")
    chosen = st.selectbox("Choose a variable to inspect", desc_cols)
    fig2 = px.histogram(fdf, x=chosen, nbins=40, color_discrete_sequence=[BLUE])
    fig2.add_vline(x=fdf[chosen].mean(), line_dash="dash", line_color=ORANGE, annotation_text="Mean")
    fig2.add_vline(x=fdf[chosen].median(), line_dash="dot", line_color=NAVY, annotation_text="Median")
    skew_val = fdf[chosen].skew()
    fig2.update_layout(height=380, title=f"Skewness: {skew_val:.2f}")
    st.plotly_chart(fig2, use_container_width=True)
    if abs(skew_val) > 1:
        st.info(f"**{chosen}** is notably skewed (skewness = {skew_val:.2f}). Mean ({fdf[chosen].mean():.1f}) and median ({fdf[chosen].median():.1f}) diverge, a signal to use median-based summaries or winsorising in cleaning.")

# =============================================================================
# PAGE: CROSS-TABS
# =============================================================================
elif page == "Cross-Tabs":
    section_header("Cross-Tabulations", "Relationships between two categorical variables")

    cat_cols = [
        "True_Segment", "Industry_Sector", "Org_Level", "Prior_AI_Experience",
        "Training_Received", "AI_Strategy_Communicated", "Scenario_Response",
        "Barrier_Category", "AI_Adopter_Flag", "Gender", "Age_Band",
    ]
    col1, col2 = st.columns(2)
    with col1:
        row_var = st.selectbox("Row variable", cat_cols, index=0)
    with col2:
        col_var = st.selectbox("Column variable", cat_cols, index=8)

    normalize_opt = st.radio("Show as", ["Counts", "Row %"], horizontal=True)

    if row_var == col_var:
        st.warning("Choose two different variables.")
    else:
        if normalize_opt == "Counts":
            ct = pd.crosstab(fdf[row_var], fdf[col_var], margins=True, margins_name="Total")
        else:
            ct = (pd.crosstab(fdf[row_var], fdf[col_var], normalize="index") * 100).round(1)
        st.dataframe(ct, use_container_width=True)

        ct_chart = pd.crosstab(fdf[row_var], fdf[col_var], normalize="index" if normalize_opt == "Row %" else None)
        fig = px.bar(ct_chart, barmode="stack" if normalize_opt == "Row %" else "group", text_auto=".0f" if normalize_opt=="Row %" else True)
        fig.update_layout(height=450, yaxis_title="% of row" if normalize_opt == "Row %" else "Count", xaxis_title=row_var)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: CORRELATION HEATMAP
# =============================================================================
elif page == "Correlation Heatmap":
    section_header("Correlation Matrix", "Pearson correlations across the nine composite scores")

    corr = fdf[COMPOSITE_COLS].corr().round(2)
    short_labels = [c.replace("_Score", "").replace("_Index", "").replace("Behavioural_Adoption_Intent", "BAIS").replace("AI_Transformation_Readiness", "Transform_Readiness") for c in COMPOSITE_COLS]
    corr_display = corr.copy()
    corr_display.index = short_labels
    corr_display.columns = short_labels

    fig = px.imshow(
        corr_display, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Correlations are deliberately moderate (|r| ≈ 0.4–0.8), not near-perfect, to mirror realistic "
        "survey data. Positive-pole composites (Leadership, Training, Trust, Digital Confidence) correlate "
        "positively with Transformation Readiness; Resistance and Change Fatigue correlate negatively."
    )

    st.markdown("#### Strongest relationships")
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack().reset_index()
    )
    corr_pairs.columns = ["Variable A", "Variable B", "Correlation"]
    corr_pairs["Abs"] = corr_pairs["Correlation"].abs()
    top_pairs = corr_pairs.sort_values("Abs", ascending=False).head(10).drop(columns="Abs")
    st.dataframe(top_pairs, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: SEGMENTS & CLUSTERING
# =============================================================================
elif page == "Segments & Clustering":
    section_header("Segments & Clustering", "Behavioural segmentation across the five hidden personas")

    seg_order = ["AI Champion","Transformation Leader","Cautious Pragmatist","Change Fatigued","Digital Resister"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Composite score profile by segment")
        seg_profile = fdf.groupby("True_Segment")[COMPOSITE_COLS].mean().reindex(seg_order).dropna(how="all")
        fig = px.imshow(
            seg_profile, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
            labels=dict(color="Mean Score"),
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Adopter rate by segment")
        adopt_by_seg = fdf.groupby("True_Segment")["AI_Adopter_Flag"].mean().reindex(seg_order).dropna() * 100
        adopt_by_seg = adopt_by_seg.reset_index()
        adopt_by_seg.columns = ["Segment", "Adopter Rate (%)"]
        fig2 = px.bar(adopt_by_seg, x="Segment", y="Adopter Rate (%)", color="Segment",
                      color_discrete_map=SEGMENT_COLORS, text_auto=".0f")
        fig2.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Live K-means clustering (blind, on standardised composite scores)")
    st.caption("This re-runs clustering on the filtered data without using True_Segment as an input, then compares the discovered clusters back to the hidden ground truth for validation.")

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, adjusted_rand_score

        cluster_feats = ["Leadership_Readiness_Score","Training_Quality_Score","AI_Trust_Score",
                         "Resistance_Score","Change_Fatigue_Score","Digital_Confidence_Score"]
        k = st.slider("Number of clusters (k)", 2, 8, 5)

        if len(fdf) > k:
            X = StandardScaler().fit_transform(fdf[cluster_feats])
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            fdf_clustered = fdf.copy()
            fdf_clustered["Cluster"] = km.labels_.astype(str)

            sil = silhouette_score(X, km.labels_)
            ari = adjusted_rand_score(fdf_clustered["True_Segment"], km.labels_)

            c1, c2 = st.columns(2)
            c1.metric("Silhouette Score", f"{sil:.3f}", help="Higher = more separated clusters. 0.15-0.55 is realistic for survey data.")
            c2.metric("Adjusted Rand Index vs. True Segment", f"{ari:.3f}", help="Agreement with the hidden ground-truth persona. 0.4-0.7 means recoverable but not trivial.")

            fig3 = px.scatter(
                fdf_clustered, x="AI_Trust_Score", y="Resistance_Score", color="Cluster",
                hover_data=["True_Segment"], opacity=0.7,
                title="Discovered clusters (color) vs. underlying AI Trust / Resistance",
            )
            fig3.update_layout(height=450)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Not enough filtered rows to cluster. Widen your filters.")
    except ImportError:
        st.error("scikit-learn is not installed. Add it to requirements.txt to enable this section.")

# =============================================================================
# PAGE: CLASSIFICATION & REGRESSION
# =============================================================================
elif page == "Classification & Regression":
    section_header("Classification & Regression", "Predictive signal in AI_Adopter_Flag and AI_Transformation_Readiness_Score")

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, roc_curve

        clf_feats = ["Leadership_Readiness_Score","Training_Quality_Score","AI_Trust_Score",
                     "Resistance_Score","Change_Fatigue_Score","Digital_Confidence_Score","AI_Readiness_Index"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Classification: AI_Adopter_Flag")
            if fdf["AI_Adopter_Flag"].nunique() == 2 and len(fdf) > 30:
                Xc, yc = fdf[clf_feats].values, fdf["AI_Adopter_Flag"].values
                Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=42, stratify=yc)
                lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                probs = lr.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(yte, probs)
                st.metric("Logistic Regression AUC (held-out)", f"{auc:.3f}")

                fpr, tpr, _ = roc_curve(yte, probs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color=BLUE)))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(color="grey", dash="dash")))
                fig.update_layout(height=380, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve")
                st.plotly_chart(fig, use_container_width=True)

                coef_df = pd.DataFrame({"Feature": clf_feats, "Coefficient": lr.coef_[0]}).sort_values("Coefficient")
                fig2 = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale="RdBu")
                fig2.update_layout(height=320, coloraxis_showscale=False, title="Standardised driver weights")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("Need both adopter classes present and enough rows. Widen your filters.")

        with col2:
            st.markdown("#### Regression: AI_Transformation_Readiness_Score")
            reg_feats = ["Training_Quality_Score","AI_Trust_Score","Resistance_Score","Change_Fatigue_Score","Digital_Confidence_Score"]
            if len(fdf) > 30:
                Xr, yr = fdf[reg_feats].values, fdf["AI_Transformation_Readiness_Score"].values
                Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3, random_state=42)
                linreg = LinearRegression().fit(Xtr, ytr)
                r2 = linreg.score(Xte, yte)
                st.metric("Linear Regression R² (held-out)", f"{r2:.3f}")

                preds = linreg.predict(Xte)
                fig3 = px.scatter(x=yte, y=preds, labels={"x": "Actual Readiness Score", "y": "Predicted Readiness Score"},
                                   opacity=0.6, color_discrete_sequence=[BLUE])
                fig3.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines", line=dict(color="grey", dash="dash"), name="Perfect fit"))
                fig3.update_layout(height=380, title="Predicted vs. Actual")
                st.plotly_chart(fig3, use_container_width=True)

                coef_df2 = pd.DataFrame({"Feature": reg_feats, "Coefficient": linreg.coef_}).sort_values("Coefficient")
                fig4 = px.bar(coef_df2, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale="RdBu")
                fig4.update_layout(height=320, coloraxis_showscale=False, title="Regression coefficients")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("Not enough filtered rows to fit a regression. Widen your filters.")

        st.caption("Models are intentionally simple (logistic/linear regression) to demonstrate signal strength, not to be production models. AUC ~0.65-0.85 and R² ~0.30-0.65 indicate realistic, non-trivial predictive signal.")
    except ImportError:
        st.error("scikit-learn is not installed. Add it to requirements.txt to enable this section.")

# =============================================================================
# PAGE: SCATTER & BOX PLOTS
# =============================================================================
elif page == "Scatter & Box Plots":
    section_header("Scatter & Box Plots", "Visual relationships and distribution shape")

    tab1, tab2 = st.tabs(["Scatter Plot", "Box Plot"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("X axis", COMPOSITE_COLS + ["AI_Usage_Hours_Monthly"], index=2)
        with col2:
            y_var = st.selectbox("Y axis", COMPOSITE_COLS + ["AI_Usage_Hours_Monthly"], index=6)
        with col3:
            color_by = st.selectbox("Color by", ["True_Segment", "AI_Adopter_Flag", "Industry_Sector", "None"])

        color_arg = None if color_by == "None" else color_by
        color_map = SEGMENT_COLORS if color_by == "True_Segment" else None
        fig = px.scatter(fdf, x=x_var, y=y_var, color=color_arg, color_discrete_map=color_map, opacity=0.65,
                          trendline="ols" if st.checkbox("Show trendline") else None)
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        if x_var in fdf.columns and y_var in fdf.columns:
            r = fdf[[x_var, y_var]].corr().iloc[0, 1]
            st.caption(f"Pearson r = {r:.3f} between {x_var} and {y_var} (filtered data)")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            num_var = st.selectbox("Numeric variable", COMPOSITE_COLS + ["AI_Usage_Hours_Monthly"], index=9)
        with col2:
            group_var = st.selectbox("Group by", ["True_Segment", "AI_Adopter_Flag", "Training_Received", "Industry_Sector"])

        color_map2 = SEGMENT_COLORS if group_var == "True_Segment" else None
        fig2 = px.box(fdf, x=group_var, y=num_var, color=group_var, color_discrete_map=color_map2, points="outliers")
        fig2.update_layout(height=520, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Points shown are statistical outliers beyond 1.5×IQR from the box edges.")

# =============================================================================
# PAGE: DATA DICTIONARY
# =============================================================================
elif page == "Data Dictionary":
    section_header("Data Dictionary", "Column-by-column documentation")
    if dict_df is not None:
        search = st.text_input("Search columns")
        show_df = dict_df.copy()
        if search:
            mask = show_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
            show_df = show_df[mask]
        st.dataframe(show_df, use_container_width=True, hide_index=True, height=600)
    else:
        st.warning("Data_Dictionary.csv not found in the repo root.")

# =============================================================================
# PAGE: RAW DATA EXPLORER
# =============================================================================
elif page == "Raw Data Explorer":
    section_header("Raw Data Explorer", "Browse and download the filtered dataset")
    st.dataframe(fdf, use_container_width=True, height=600)
    st.download_button(
        "Download filtered data as CSV",
        fdf.to_csv(index=False).encode("utf-8"),
        file_name="AI_Readiness_filtered.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("AI Transformation Readiness Dashboard · Synthetic data for academic/demonstration purposes · Built with Streamlit")
