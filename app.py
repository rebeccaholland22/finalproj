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

LOWER_BETTER = [
    "elec_price_usd_kwh",
    "carbon_intensity_gco2_kwh",
    "avg_temp_c",
    "t_d_losses_pct",
]

HIGHER_BETTER = ["renew_share_pct"]

NORM_COLS = [
    "elec_price_usd_kwh_norm",
    "carbon_intensity_gco2_kwh_norm",
    "avg_temp_c_norm",
    "t_d_losses_pct_norm",
    "renew_share_pct_norm",
]

EPS = 1e-12


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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


def compute_baseline_bounds(df_2024: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Compute min/max bounds from the 2024 baseline dataset.
    These same bounds are reused for future scenarios so scores remain comparable over time.
    """
    bounds = {}
    for col in LOWER_BETTER + HIGHER_BETTER:
        if col not in df_2024.columns:
            raise KeyError(f"Missing required indicator column: {col}")
        bounds[col] = (df_2024[col].min(), df_2024[col].max())
    return bounds


def minmax_with_bounds(series: pd.Series, vmin: float, vmax: float) -> pd.Series:
    denom = vmax - vmin
    if abs(denom) < EPS:
        return pd.Series([0.5] * len(series), index=series.index)

    clipped = series.clip(lower=vmin, upper=vmax)
    return (clipped - vmin) / denom


def normalise_with_fixed_bounds(
    df_raw: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalise indicators using fixed 2024 baseline bounds.
    Lower-is-better indicators are inverted so that higher normalised values always mean better suitability.
    """
    df = df_raw.copy()

    for col in LOWER_BETTER:
        vmin, vmax = bounds[col]
        scaled = minmax_with_bounds(df[col], vmin, vmax)
        df[col + "_norm"] = 1.0 - scaled

    for col in HIGHER_BETTER:
        vmin, vmax = bounds[col]
        df[col + "_norm"] = minmax_with_bounds(df[col], vmin, vmax)

    norm_cols = [col + "_norm" for col in LOWER_BETTER + HIGHER_BETTER]
    return df, norm_cols


def entropy_weights(df_norm: pd.DataFrame, norm_cols: list[str]) -> pd.Series:
    X = df_norm[norm_cols].copy()

    P = X / (X.sum(axis=0) + EPS)
    n = len(X)
    k = 1 / np.log(n)

    entropy = -k * (P * np.log(P + EPS)).sum(axis=0)
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
    st.error(f"Cannot find {DATA_PATH}. Make sure you have saved the processed dataset in that location.")
    st.stop()

# Data quality guard: renew_share_pct should be a true percentage, not a TWh generation value.
if "renew_share_pct" in df_raw.columns and df_raw["renew_share_pct"].max(skipna=True) > 100:
    st.warning(
        "The processed dataset has renew_share_pct values above 100. "
        "This usually means the column still contains renewable generation rather than renewable share. "
        "Run fix_renewable_share.py, then restart the app."
    )

# Fixed baseline scaling bounds used across all present and future scenarios.
bounds_2024 = compute_baseline_bounds(df_raw)

st.sidebar.header("Scenario settings")

year = st.sidebar.selectbox(
    "Select year",
    [2024, 2030, 2040],
    index=0,
    format_func=lambda y: YEAR_LABELS[y],
)

pathway = "Baseline"
if year != 2024:
    pathway = st.sidebar.radio("Decarbonisation pathway", ["Baseline", "Accelerated"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Controls")

# Apply scenario adjustments, then normalise using fixed 2024 bounds.
df_scenario = apply_future_adjustments(df_raw, year, pathway)
df_norm, norm_cols = normalise_with_fixed_bounds(df_scenario, bounds_2024)

mode = st.sidebar.radio(
    "Weighting mode",
    ["Preset investor profile", "Custom weights", "Entropy weights (data-driven)"],
    index=0,
)

preset = None
entropy_w = None

label_map = {
    "elec_price_usd_kwh_norm": "Electricity price (lower is better)",
    "carbon_intensity_gco2_kwh_norm": "Carbon intensity (lower is better)",
    "renew_share_pct_norm": "Renewable electricity share (%) (higher is better)",
    "avg_temp_c_norm": "Average temperature (lower is better)",
    "t_d_losses_pct_norm": "Transmission & distribution losses (lower is better)",
}

missing_norm = [col for col in label_map.keys() if col not in df_norm.columns]
if missing_norm:
    st.error(f"Missing expected normalised columns: {missing_norm}")
    st.stop()

if mode == "Preset investor profile":
    preset = st.sidebar.selectbox(
        "Choose profile",
        ["Balanced", "Cost-focused", "Sustainability-focused", "Reliability-focused", "Cooling-focused"],
        index=0,
    )

    presets = {
        "Balanced": {
            "elec_price_usd_kwh_norm": 0.20,
            "carbon_intensity_gco2_kwh_norm": 0.20,
            "renew_share_pct_norm": 0.20,
            "avg_temp_c_norm": 0.20,
            "t_d_losses_pct_norm": 0.20,
        },
        "Cost-focused": {
            "elec_price_usd_kwh_norm": 0.45,
            "carbon_intensity_gco2_kwh_norm": 0.10,
            "renew_share_pct_norm": 0.10,
            "avg_temp_c_norm": 0.10,
            "t_d_losses_pct_norm": 0.25,
        },
        "Sustainability-focused": {
            "elec_price_usd_kwh_norm": 0.10,
            "carbon_intensity_gco2_kwh_norm": 0.45,
            "renew_share_pct_norm": 0.35,
            "avg_temp_c_norm": 0.05,
            "t_d_losses_pct_norm": 0.05,
        },
        "Reliability-focused": {
            "elec_price_usd_kwh_norm": 0.15,
            "carbon_intensity_gco2_kwh_norm": 0.10,
            "renew_share_pct_norm": 0.10,
            "avg_temp_c_norm": 0.10,
            "t_d_losses_pct_norm": 0.55,
        },
        "Cooling-focused": {
            "elec_price_usd_kwh_norm": 0.15,
            "carbon_intensity_gco2_kwh_norm": 0.15,
            "renew_share_pct_norm": 0.10,
            "avg_temp_c_norm": 0.45,
            "t_d_losses_pct_norm": 0.15,
        },
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

else:
    entropy_w = entropy_weights(df_norm, list(label_map.keys()))
    weights = entropy_w.to_dict()
    st.sidebar.write("Entropy-derived weights:")
    st.sidebar.json({label_map[k]: round(v, 3) for k, v in weights.items()})

# Compute ranking for the selected year/pathway.
df_out = df_norm.copy()
df_out["suitability_score"] = score(df_out, weights)
df_ranked = df_out.sort_values("suitability_score", ascending=False).reset_index(drop=True)
df_ranked["rank"] = df_ranked.index + 1

if year != 2024:
    df_base_norm, _ = normalise_with_fixed_bounds(df_raw, bounds_2024)
    df_base = df_base_norm.copy()
    df_base["baseline_score"] = score(df_base, weights)
    df_base = df_base.sort_values("baseline_score", ascending=False).reset_index(drop=True)
    df_base["baseline_rank"] = df_base.index + 1

    df_ranked = df_ranked.merge(df_base[["country", "baseline_rank"]], on="country", how="left")
    # Positive values mean the country moved up compared with the 2024 baseline ranking.
    df_ranked["rank_change"] = df_ranked["baseline_rank"] - df_ranked["rank"]

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Ranked results")

    st.caption(
        f"Viewing: **{YEAR_LABELS[year]}**"
        + (f" (**{pathway}** decarbonisation)" if year != 2024 else "")
        + " • Indicators adjusted: carbon intensity and temperature only. "
        + "Future scenarios use fixed 2024 normalisation bounds."
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
        "rank": "{:.0f}",
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
        })

    st.dataframe(
        df_ranked[show_cols].style.format(fmt),
        use_container_width=True,
    )

    csv_bytes = df_ranked[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ranked results (CSV)",
        data=csv_bytes,
        file_name=f"data_centre_suitability_ranking_{year}{('_' + pathway.lower()) if year != 2024 else ''}.csv",
        mime="text/csv",
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
    "Weight": [weights[k] for k in label_map.keys()],
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

    dc_rank = (
        dc_df[["country", "iso3", "dc_facility_count"]]
        .rename(columns={
            "country": "Country",
            "iso3": "ISO Code",
            "dc_facility_count": "Data Centre Count",
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
        "avg_networks_per_facility": "Average Networks per Facility",
    })

    st.write(
        "This section adds externally sourced infrastructure evidence from PeeringDB. "
        "The data provides country-level context on facility presence, network density, "
        "and internet exchange activity. These measures are used to contextualise the "
        "sustainability rankings, not to calculate the main suitability score."
    )

    st.write(f"Facility records collected: {len(pdb)}")

    st.dataframe(
        pdb_country.head(30),
        use_container_width=True,
    )

    st.caption("Countries ranked by PeeringDB-listed facility and interconnection coverage")

except FileNotFoundError:
    st.warning("PeeringDB scraped files not found. Run the PeeringDB scraper and aggregator first.")
