# scrape_peeringdb_facilities.py

import requests
import pandas as pd
from pathlib import Path

url = "https://www.peeringdb.com/api/fac"

headers = {
    "User-Agent": "University student data centre location project"
}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()

data = response.json()["data"]

df = pd.DataFrame(data)

# Keep useful columns only
cols = [
    "id",
    "name",
    "city",
    "country",
    "zipcode",
    "address1",
    "latitude",
    "longitude",
    "website",
    "net_count",
    "ix_count",
]

df = df[[c for c in cols if c in df.columns]]

out = Path("data_processed/scraped_peeringdb_facilities.csv")
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)

print(f"Saved {len(df)} scraped facility rows to {out}")
print(df.head())