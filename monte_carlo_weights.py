import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data_processed/country_indicators_global_2022_2024.csv"
OUT_DIR = "outputs"
N_SAMPLES = 10000  # Monte Carlo weight draws

SCENARIO_TO_PLOT = ("2040", "Accelerated")

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
    "renew_share_pct_norm",
    "avg_temp_c_norm",
    "t_d_losses_pct_norm",
]

EPS = 1e-12

DIRICHLET_ALPHA = np.ones(len(NORM_COLS))


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_future_adjustments(df: pd.DataFrame, year: str, pathway: str | None) -> pd.DataFrame:
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


def compute_baseline_bounds(df_2024: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds = {}
    for col in LOWER_BETTER + HIGHER_BETTER:
        bounds[col] = (df_2024[col].min(), df_2024[col].max())
    return bounds


def minmax_with_bounds(series: pd.Series, vmin: float, vmax: float) -> pd.Series:
    denom = vmax - vmin
    if abs(denom) < EPS:
        return pd.Series([0.5] * len(series), index=series.index)
    clipped = series.clip(lower=vmin, upper=vmax)
    return (clipped - vmin) / denom


def normalise_with_fixed_bounds(df_raw: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df_raw.copy()

    for col in LOWER_BETTER:
        vmin, vmax = bounds[col]
        scaled = minmax_with_bounds(df[col], vmin, vmax)
        df[col + "_norm"] = 1.0 - scaled

    for col in HIGHER_BETTER:
        vmin, vmax = bounds[col]
        df[col + "_norm"] = minmax_with_bounds(df[col], vmin, vmax)

    return df


def scenario_label(year: str, pathway: str | None) -> str:
    return year if year == "2024" else f"{year}_{pathway}"


def score_matrix(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    X: (n_countries, n_features)
    W: (n_samples, n_features)
    returns: (n_samples, n_countries)
    """
    return W @ X.T


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    scores: (n_samples, n_countries)
    returns ranks: (n_samples, n_countries) where rank 1 is best
    """
    # argsort descending
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    # ranks[ sample, country_index ] = rank_position (0-based) then +1
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    return ranks + 1

os.makedirs(OUT_DIR, exist_ok=True)

df_raw = load_data(DATA_PATH)

# Fixed baseline scaling bounds (2024 dataset)
bounds_2024 = compute_baseline_bounds(df_raw)

# Build scenario-normalised feature matrix (fixed 2024 bounds)
year, pathway = SCENARIO_TO_PLOT
df_adj = apply_future_adjustments(df_raw, str(year), pathway)
df_norm = normalise_with_fixed_bounds(df_adj, bounds_2024)

required = ["country"] + NORM_COLS
missing = [c for c in required if c not in df_norm.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

countries = df_norm["country"].astype(str).tolist()
X = df_norm[NORM_COLS].to_numpy(dtype=float)  # (n_countries, n_features)

# Sample weights on the simplex
rng = np.random.default_rng(42)
W = rng.dirichlet(DIRICHLET_ALPHA, size=N_SAMPLES)  # (n_samples, n_features)

# Compute scores and ranks across the weight space
scores = score_matrix(X, W)           # (n_samples, n_countries)
ranks = ranks_from_scores(scores)     # (n_samples, n_countries)

# Summary metrics per country
mean_rank = ranks.mean(axis=0)
sd_rank = ranks.std(axis=0, ddof=1)

pr_top3 = (ranks <= 3).mean(axis=0)
pr_top5 = (ranks <= 5).mean(axis=0)
pr_top10 = (ranks <= 10).mean(axis=0)

results = pd.DataFrame({
    "country": countries,
    "mean_rank": mean_rank,
    "sd_rank": sd_rank,
    "pr_top3": pr_top3,
    "pr_top5": pr_top5,
    "pr_top10": pr_top10,
})

# Sort by Top-5 probability (then mean rank)
results = results.sort_values(["pr_top5", "mean_rank"], ascending=[False, True]).reset_index(drop=True)

# Save results table
label = scenario_label(str(year), pathway)
csv_path = os.path.join(OUT_DIR, f"mcda_weightspace_summary_{label}.csv")
results.to_csv(csv_path, index=False)

print(f"\nSaved summary table to: {csv_path}")
print(results.head(10).to_string(index=False))

# FIGURE 1: Top-10 countries by Pr(Top 5)

topn = 10
plot_df = results.head(topn).copy()

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(plot_df["country"][::-1], plot_df["pr_top5"][::-1])
ax.set_xlabel("Probability of being in Top 5")
ax.set_title(f"Ranking robustness under weight uncertainty ({label})")
ax.set_xlim(0, 1)

fig.tight_layout()
fig1_path = os.path.join(OUT_DIR, f"fig_pr_top5_{label}.png")
fig.savefig(fig1_path, dpi=300)
plt.close(fig)

print(f"Saved figure to: {fig1_path}")

# FIGURE 2: Rank uncertainty (boxplot) for Top-10 by mean rank

top_by_mean = results.sort_values("mean_rank", ascending=True).head(topn)
top_countries = top_by_mean["country"].tolist()
idx = [countries.index(c) for c in top_countries]

rank_samples = [ranks[:, i] for i in idx]

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(rank_samples, labels=top_countries, showfliers=False)
ax.set_ylabel("Rank (1 = best)")
ax.set_title(f"Rank uncertainty across weight space ({label})")
ax.invert_yaxis() 
plt.xticks(rotation=30, ha="right")

fig.tight_layout()
fig2_path = os.path.join(OUT_DIR, f"fig_rank_boxplot_{label}.png")
fig.savefig(fig2_path, dpi=300)
plt.close(fig)

print(f"Saved figure to: {fig2_path}")