import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, kendalltau, levene

DATA_PATH = "data_processed/country_indicators_global_2022_2024.csv"

BALANCED_WEIGHTS = {
    "elec_price_usd_kwh_norm": 0.20,
    "carbon_intensity_gco2_kwh_norm": 0.20,
    "renew_share_pct_norm": 0.20,
    "avg_temp_c_norm": 0.20,
    "t_d_losses_pct_norm": 0.20
}

SCENARIOS = [
    ("2024", None),
    ("2030", "Baseline"),
    ("2030", "Accelerated"),
    ("2040", "Baseline"),
    ("2040", "Accelerated")
]

LOWER_BETTER = [
    "elec_price_usd_kwh",
    "carbon_intensity_gco2_kwh",
    "avg_temp_c",
    "t_d_losses_pct"
]
HIGHER_BETTER = ["renew_share_pct"]

EPS = 1e-12


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_future_adjustments(df: pd.DataFrame, year: str, pathway: str | None) -> pd.DataFrame:
    """
    Apply forward-looking adjustments to carbon intensity and temperature.
    """
    if year == "2024":
        return df.copy()

    df_adj = df.copy()

    carbon_multipliers = {
        "2030": {"Baseline": 0.80, "Accelerated": 0.70},
        "2040": {"Baseline": 0.60, "Accelerated": 0.40},
    }
    temp_increases = {
        "2030": 0.3,
        "2040": 0.7,
    }

    df_adj["carbon_intensity_gco2_kwh"] = df_adj["carbon_intensity_gco2_kwh"] * carbon_multipliers[year][pathway]
    df_adj["avg_temp_c"] = df_adj["avg_temp_c"] + temp_increases[year]

    return df_adj


def compute_baseline_minmax_bounds(df_2024: pd.DataFrame) -> dict:
    """
    Compute min/max for each indicator on the 2024 baseline dataset.
    These bounds will be used for all future normalisation to make scores comparable over time.
    """
    bounds = {}
    for col in LOWER_BETTER + HIGHER_BETTER:
        bounds[col] = (df_2024[col].min(), df_2024[col].max())
    return bounds


def minmax_with_bounds(series: pd.Series, vmin: float, vmax: float) -> pd.Series:
    """
    Min-max scaling using fixed (baseline) bounds.
    Values outside bounds are clipped to [vmin, vmax] to avoid >1 or <0 scores.
    """
    denom = (vmax - vmin)
    if abs(denom) < EPS:
        return pd.Series([0.5] * len(series), index=series.index)

    clipped = series.clip(lower=vmin, upper=vmax)
    return (clipped - vmin) / denom


def normalise_with_fixed_bounds(df_raw: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Normalise each indicator using 2024 baseline bounds.
    Applies direction handling (invert for lower-is-better).
    """
    df = df_raw.copy()

    for col in LOWER_BETTER:
        vmin, vmax = bounds[col]
        scaled = minmax_with_bounds(df[col], vmin, vmax)
        df[col + "_norm"] = 1 - scaled  # invert: lower is better

    for col in HIGHER_BETTER:
        vmin, vmax = bounds[col]
        df[col + "_norm"] = minmax_with_bounds(df[col], vmin, vmax)  # higher is better

    return df


def compute_scores(df_norm: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(df_norm[col] * w for col, w in weights.items())


def scenario_name(year: str, pathway: str | None) -> str:
    return year if year == "2024" else f"{year}_{pathway}"


df_raw = load_data(DATA_PATH)

# 1) Compute baseline scaling bounds (fixed for all scenarios)
bounds_2024 = compute_baseline_minmax_bounds(df_raw)

# 2) Build scenario score matrix using fixed baseline scaling
scenario_scores = {}

for year, pathway in SCENARIOS:
    df_adj = apply_future_adjustments(df_raw, year, pathway)
    df_norm = normalise_with_fixed_bounds(df_adj, bounds_2024)
    scores = compute_scores(df_norm, BALANCED_WEIGHTS)

    scenario_scores[scenario_name(year, pathway)] = scores.values

scenario_df = pd.DataFrame(scenario_scores)

print("\nScenario Score Matrix (fixed 2024 scaling):")
print(scenario_df.head())

# 1. FRIEDMAN TEST

friedman_stat, friedman_p = friedmanchisquare(
    scenario_df["2024"],
    scenario_df["2030_Baseline"],
    scenario_df["2030_Accelerated"],
    scenario_df["2040_Baseline"],
    scenario_df["2040_Accelerated"]
)

print("\nFriedman Test Across Scenarios (fixed scaling)")
print("Statistic:", friedman_stat)
print("p-value:", friedman_p)

# 2. PAIRWISE WILCOXON TESTS

pairs = [
    ("2024", "2030_Baseline"),
    ("2024", "2030_Accelerated"),
    ("2024", "2040_Baseline"),
    ("2024", "2040_Accelerated"),
    ("2030_Baseline", "2030_Accelerated"),
    ("2040_Baseline", "2040_Accelerated"),
]

print("\nPairwise Wilcoxon Signed-Rank Tests (fixed scaling):")
for a, b in pairs:
    diffs = scenario_df[a] - scenario_df[b]
    if np.allclose(diffs.values, 0):
        print(f"{a} vs {b} -> identical (no differences)")
        continue

    stat, p = wilcoxon(scenario_df[a], scenario_df[b])
    print(f"{a} vs {b} -> statistic={stat}, p={p}")

# 3. VARIANCE COMPARISON (convergence / dispersion change)
# Brown–Forsythe uses median centring; scipy's levene(center='median') does that.

var_stat, var_p = levene(
    scenario_df["2024"],
    scenario_df["2040_Accelerated"],
    center="median"
)

print("\nVariance Comparison (2024 vs 2040 Accelerated; Brown–Forsythe)")
print("Statistic:", var_stat)
print("p-value:", var_p)

# 4. RANK STABILITY (Kendall tau) between baseline and worst-case future

rank_2024 = scenario_df["2024"].rank(ascending=False, method="average")
rank_2040 = scenario_df["2040_Accelerated"].rank(ascending=False, method="average")

tau, tau_p = kendalltau(rank_2024, rank_2040)

print("\nKendall Rank Correlation (2024 vs 2040 Accelerated; fixed scaling)")
print("Tau:", tau)
print("p-value:", tau_p)

print("\nMean score change vs 2024 (fixed scaling):")
for col in scenario_df.columns:
    if col == "2024":
        continue
    delta = (scenario_df[col] - scenario_df["2024"]).mean()
    print(f"{col}: {delta:+.4f}")