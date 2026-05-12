# aggregate_peeringdb_by_country.py

import pandas as pd
from pathlib import Path

df = pd.read_csv("data_processed/scraped_peeringdb_facilities.csv")

country_summary = (
    df.groupby("country")
    .agg(
        peeringdb_facility_count=("id", "count"),
        total_networks=("net_count", "sum"),
        total_ixs=("ix_count", "sum"),
        avg_networks_per_facility=("net_count", "mean"),
    )
    .reset_index()
    .sort_values("peeringdb_facility_count", ascending=False)
)

out = Path("data_processed/peeringdb_country_summary.csv")
country_summary.to_csv(out, index=False)

print(f"Saved country summary to {out}")
print(country_summary.head(20))