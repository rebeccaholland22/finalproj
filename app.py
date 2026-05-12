import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Centre Location Decision Tool", layout="wide")

st.title("Data Centre Location Suitability – Decision Support Tool")
st.caption("Multi-criteria scoring using energy, climate, and infrastructure indicators.")

DATA_PATH = "data_processed/country_indicators_global_2022_2024.csv"

YEAR_LABELS = {
    2024: "Present",
    2030: "2030",
    2040: "2040",
}
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def minmax(series: pd.Series) -> pd.Series:
    denom = (series.max() - series.min())
    if denom == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / denom


def apply_future_adjustments(df: pd.DataFrame, year: int, pathway: str) -> pd.DataFrame:
    """
    Apply forward-looking adjustments to carbon intensity and temperature.

    year: 2024, 2030, 2040
    pathway: 'Baseline' or 'Accelerated' (ignored for 2024)
    """
    if year == 2024:
        return df.copy()

    df_adj = df.copy()

    carbon_multipliers = {
        2030: {"Baseline": 0.80, "Accelerated": 0.70},
        2040: {"Baseline": 0.60, "Accelerated": 0.40},
    }

    temp_increases = {
        2030: 0.3,
        2040: 0.7,
    }

    df_adj["carbon_intensity_gco2_kwh"] = (
        df_adj["carbon_intensity_gco2_kwh"] * carbon_multipliers[year][pathway]
    )
    df_adj["avg_temp_c"] = df_adj["avg_temp_c"] + temp_increases[year]

    return df_adj


def build_normalised(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df_raw.copy()

    # Indicator directions:
    # lower is better -> invert after min-max
    lower_better = ["elec_price_usd_kwh", "carbon_intensity_gco2_kwh", "avg_temp_c", "t_d_losses_pct"]
    higher_better = ["renew_share_pct"]

    for c in lower_better:
        df[c + "_norm"] = 1 - minmax(df[c])

    for c in higher_better:
        df[c + "_norm"] = minmax(df[c])

    norm_cols = [c + "_norm" for c in lower_better + higher_better]
    return df, norm_cols


def entropy_weights(df_norm: pd.DataFrame, norm_cols: list[str]) -> pd.Series:
    X = df_norm[norm_cols].copy()
    eps = 1e-12

    P = X / (X.sum(axis=0) + eps)
    n = len(X)
    k = 1 / np.log(n)

    entropy = -k * (P * np.log(P + eps)).sum(axis=0)
    d = 1 - entropy
    w = d / d.sum()
    return w


def score(df_norm: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = pd.Series(weights)
    cols = list(w.index)
    return (df_norm[cols] * w.values).sum(axis=1)


def make_bar_top10(df_ranked: pd.DataFrame, score_col: str, title: str):
    top = df_ranked.head(10).copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top["country"][::-1], top[score_col][::-1])
    ax.set_xlabel("Suitability score")
    ax.set_title(title)
    st.pyplot(fig)


try:
    df_raw = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Cannot find {DATA_PATH}. Make sure you've saved the processed dataset in that location.")
    st.stop()

st.sidebar.header("Scenario settings")

year = st.sidebar.selectbox(
    "Select year",
    [2024, 2030, 2040],
    index=0,
    format_func=lambda y: YEAR_LABELS[y]
)
pathway = "Baseline"
if year != 2024:
    pathway = st.sidebar.radio("Decarbonisation pathway", ["Baseline", "Accelerated"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Controls")

# Apply scenario adjustments BEFORE normalisation/scoring
df_scenario = apply_future_adjustments(df_raw, year, pathway)
df_norm, norm_cols = build_normalised(df_scenario)

mode = st.sidebar.radio(
    "Weighting mode",
    ["Preset investor profile", "Custom weights", "Entropy weights (data-driven)"],
    index=0
)

preset = None
entropy_w = None

# reader-friendly labels for UI
label_map = {
    "elec_price_usd_kwh_norm": "Electricity price (lower is better)",
    "carbon_intensity_gco2_kwh_norm": "Carbon intensity (lower is better)",
    "renew_share_pct_norm": "Renewable share (higher is better)",
    "avg_temp_c_norm": "Average temperature (lower is better)",
    "t_d_losses_pct_norm": "Transmission & distribution losses (lower is better)",
}

missing_norm = [c for c in label_map.keys() if c not in df_norm.columns]
if missing_norm:
    st.error(f"Missing expected normalised columns: {missing_norm}")
    st.stop()

if mode == "Preset investor profile":
    preset = st.sidebar.selectbox(
        "Choose profile",
        ["Balanced", "Cost-focused", "Sustainability-focused", "Reliability-focused", "Cooling-focused"],
        index=0
    )

    presets = {
        "Balanced": {
            "elec_price_usd_kwh_norm": 0.20,
            "carbon_intensity_gco2_kwh_norm": 0.20,
            "renew_share_pct_norm": 0.20,
            "avg_temp_c_norm": 0.20,
            "t_d_losses_pct_norm": 0.20
        },
        "Cost-focused": {
            "elec_price_usd_kwh_norm": 0.45,
            "carbon_intensity_gco2_kwh_norm": 0.10,
            "renew_share_pct_norm": 0.10,
            "avg_temp_c_norm": 0.10,
            "t_d_losses_pct_norm": 0.25
        },
        "Sustainability-focused": {
            "elec_price_usd_kwh_norm": 0.10,
            "carbon_intensity_gco2_kwh_norm": 0.45,
            "renew_share_pct_norm": 0.35,
            "avg_temp_c_norm": 0.05,
            "t_d_losses_pct_norm": 0.05
        },
        "Reliability-focused": {
            "elec_price_usd_kwh_norm": 0.15,
            "carbon_intensity_gco2_kwh_norm": 0.10,
            "renew_share_pct_norm": 0.10,
            "avg_temp_c_norm": 0.10,
            "t_d_losses_pct_norm": 0.55
        },
        "Cooling-focused": {
            "elec_price_usd_kwh_norm": 0.15,
            "carbon_intensity_gco2_kwh_norm": 0.15,
            "renew_share_pct_norm": 0.10,
            "avg_temp_c_norm": 0.45,
            "t_d_losses_pct_norm": 0.15
        }
    }

    weights = presets[preset]

elif mode == "Custom weights":
    raw_w = {}
    for col, label in label_map.items():
        raw_w[col] = st.sidebar.slider(label, min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    total = sum(raw_w.values())
    if total == 0:
        st.sidebar.warning("All weights are zero. Please set at least one weight above zero.")
        weights = {k: 0.0 for k in raw_w.keys()}
    else:
        weights = {k: v / total for k, v in raw_w.items()}

    st.sidebar.caption("Normalised weights:")
    st.sidebar.json({label_map[k]: round(v, 3) for k, v in weights.items()})

else:  # Entropy weights
    entropy_w = entropy_weights(df_norm, list(label_map.keys()))
    weights = entropy_w.to_dict()
    st.sidebar.write("Entropy-derived weights:")
    st.sidebar.json({label_map[k]: round(v, 3) for k, v in weights.items()})

# Compute ranking (scenario-selected year/pathway)
df_out = df_norm.copy()
df_out["suitability_score"] = score(df_out, weights)
df_ranked = df_out.sort_values("suitability_score", ascending=False).reset_index(drop=True)
df_ranked["rank"] = df_ranked.index + 1

if year != 2024:
    df_base_norm, _ = build_normalised(df_raw)
    df_base = df_base_norm.copy()
    df_base["baseline_score"] = score(df_base, weights)
    df_base = df_base.sort_values("baseline_score", ascending=False).reset_index(drop=True)
    df_base["baseline_rank"] = df_base.index + 1

    df_ranked = df_ranked.merge(df_base[["country", "baseline_rank"]], on="country", how="left")
    # Positive means the country improved (moved up) in the future ranking
    df_ranked["rank_change"] = df_ranked["baseline_rank"] - df_ranked["rank"]

# main
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Ranked results")

    st.caption(
        f"Viewing: **{YEAR_LABELS[year]}**"
        + (f" (**{pathway}** decarbonisation)" if year != 2024 else "")
        + " • Indicators adjusted: carbon intensity & temperature only."
    )

    show_cols = [
        "rank",
        "country",
        "suitability_score",
    ]

    if year != 2024:
        show_cols += ["baseline_rank", "rank_change"]

    show_cols += [
        "elec_price_usd_kwh",
        "carbon_intensity_gco2_kwh",
        "renew_share_pct",
        "avg_temp_c",
        "t_d_losses_pct",
    ]

    fmt = {
        "suitability_score": "{:.3f}",
        "elec_price_usd_kwh": "{:.3f}",
        "carbon_intensity_gco2_kwh": "{:.1f}",
        "renew_share_pct": "{:.1f}",
        "avg_temp_c": "{:.1f}",
        "t_d_losses_pct": "{:.2f}",
    }
    if year != 2024:
        fmt.update({
            "baseline_rank": "{:.0f}",
            "rank_change": "{:+.0f}",
            "rank": "{:.0f}",
        })
    else:
        fmt.update({"rank": "{:.0f}"})

    st.dataframe(
        df_ranked[show_cols].style.format(fmt),
        use_container_width=True
    )

    csv_bytes = df_ranked[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ranked results (CSV)",
        data=csv_bytes,
        file_name=f"data_centre_suitability_ranking_{year}{('_' + pathway.lower()) if year != 2024 else ''}.csv",
        mime="text/csv"
    )

with col2:
    st.subheader("Top 10 visualisation")
    title = f"Top 10 – {mode}" + (f" ({preset})" if preset else "")
    title += f" • {YEAR_LABELS[year]}" + (f" ({pathway})" if year != 2024 else "")
    make_bar_top10(df_ranked, "suitability_score", title)

st.markdown("---")
st.subheader("Weights used")
weights_table = pd.DataFrame({
    "Indicator": [label_map[k] for k in label_map.keys()],
    "Weight": [weights[k] for k in label_map.keys()]
})
st.table(weights_table.style.format({"Weight": "{:.3f}"}))

# -----------------------------
# Scraped data centre counts
# -----------------------------

st.markdown("---")
st.subheader("Existing data centre facility counts by country")

try:
    dc_df = pd.read_csv("data_processed/datacentre_facility_counts.csv")

    st.write(
        "This section uses web-scraped data to show the number of listed "
        "data centre facilities by country. This is used for comparison only "
        "and is not included in the main suitability score."
    )

    # Clean and format table
    dc_rank = (
        dc_df[["country", "iso3", "dc_facility_count"]]
        .rename(columns={
            "country": "Country",
            "iso3": "ISO Code",
            "dc_facility_count": "Data Centre Count"
        })
        .sort_values("Data Centre Count", ascending=False)
        .reset_index(drop=True)
    )

    dc_rank["Rank"] = dc_rank.index + 1
    dc_rank = dc_rank[["Rank", "Country", "ISO Code", "Data Centre Count"]]

    st.dataframe(dc_rank, use_container_width=True)


except FileNotFoundError:
    st.warning("Scraped data centre facility file not found.")

st.markdown("---")
st.subheader("Infrastructure dataset: PeeringDB facility and interconnection coverage")

try:
    pdb = pd.read_csv("data_processed/scraped_peeringdb_facilities.csv")
    pdb_country = pd.read_csv("data_processed/peeringdb_country_summary.csv")

    pdb_country = pdb_country.rename(columns={
        "country": "Country",
        "peeringdb_facility_count": "Infrastructure Facilities",
        "total_networks": "Connected Networks",
        "total_ixs": "Internet Exchange Connections",
        "avg_networks_per_facility": "Average Networks per Facility"
    })

    st.write(
        "This section adds a large externally sourced infrastructure dataset from PeeringDB. "
        "The data provides country-level evidence of digital infrastructure maturity, including "
        "facility presence, network density, and internet exchange activity. These measures help "
        "contextualise whether a country has the interconnection ecosystem needed to support "
        "large-scale data centre deployment."
    )

    st.write(f"Facility records collected: {len(pdb)}")

    st.dataframe(
        pdb_country.head(30),
        use_container_width=True
    )

    st.caption("Countries ranked by PeeringDB-listed facility and interconnection coverage")

except FileNotFoundError:
    st.warning("PeeringDB scraped files not found. Run the PeeringDB scraper and aggregator first.")