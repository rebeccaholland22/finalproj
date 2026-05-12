import math
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "data_processed" / "country_indicators_global_2022_2024.csv"
OUTPUT_CSV = PROJECT_ROOT / "data_processed" / "datacentre_facility_counts.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "datacentre_scrape_log.txt"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT = 20
REQUEST_DELAY_SECONDS = 1.5


COUNTRY_SLUG_OVERRIDES = {
    "United Kingdom": "united-kingdom",
    "United States": "united-states",
    "South Korea": "south-korea",
    "New Zealand": "new-zealand",
    "Czech Republic": "czech-republic",
    "Czechia": "czech-republic",
    "UAE": "united-arab-emirates",
    "United Arab Emirates": "united-arab-emirates",
}

def write_log(message: str) -> None:
    """Append a line to the scrape log and print it."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def slugify_country_name(country: str) -> str:
    """
    Convert a country name into the expected URL slug format.
    Falls back to a simple normalisation if there is no explicit override.
    """
    if country in COUNTRY_SLUG_OVERRIDES:
        return COUNTRY_SLUG_OVERRIDES[country]

    slug = country.strip().lower()
    slug = slug.replace("&", "and")
    slug = slug.replace(",", "")
    slug = slug.replace("'", "")
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def build_country_url(country: str) -> str:
    slug = slugify_country_name(country)
    return f"https://www.datacentermap.com/{slug}/"


def extract_facility_count(page_text: str):
    """
    Extract the total number of data centres from the page text.

    Tries a few patterns, because exact wording can vary.
    """
    patterns = [
        r"We currently have\s+(\d+)\s+data centers listed",
        r"We currently have\s+(\d+)\s+data centres listed",
        r"currently have\s+(\d+)\s+data centers listed",
        r"currently have\s+(\d+)\s+data centres listed",
        r"(\d+)\s+data centers listed",
        r"(\d+)\s+data centres listed",
    ]

    for pattern in patterns:
        match = re.search(pattern, page_text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def fetch_page(url: str) -> str:
    """
    Fetch page HTML and return response text.
    Raises an exception if the request fails.
    """
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def scrape_country(country: str) -> dict:
    url = build_country_url(country)

    try:
        html = fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(" ", strip=True)

        facility_count = extract_facility_count(page_text)

        if facility_count is None:
            status = "parsed_no_count_found"
        else:
            status = "ok"

        return {
            "country": country,
            "source_url": url,
            "dc_facility_count": facility_count,
            "scrape_status": status,
        }

    except requests.HTTPError as e:
        return {
            "country": country,
            "source_url": url,
            "dc_facility_count": None,
            "scrape_status": f"http_error_{getattr(e.response, 'status_code', 'unknown')}",
        }

    except requests.RequestException as e:
        return {
            "country": country,
            "source_url": url,
            "dc_facility_count": None,
            "scrape_status": f"request_failed: {str(e)}",
        }

    except Exception as e:
        return {
            "country": country,
            "source_url": url,
            "dc_facility_count": None,
            "scrape_status": f"unexpected_error: {str(e)}",
        }


def detect_country_column(df: pd.DataFrame) -> str:
    """
    Detect likely country column name from the input CSV.
    """
    candidate_columns = ["country", "Country", "country_name", "Country Name"]
    for col in candidate_columns:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a country column in the input CSV. "
        f"Available columns: {list(df.columns)}"
    )


def detect_iso3_column(df: pd.DataFrame):
    """
    Detect likely ISO3 column if present.
    """
    candidate_columns = ["iso3", "ISO3", "iso_code", "ISO Code", "country_code"]
    for col in candidate_columns:
        if col in df.columns:
            return col
    return None


def minmax_normalise(series: pd.Series) -> pd.Series:
    """
    Min-max normalise a numeric series.
    """
    series = pd.to_numeric(series, errors="coerce")
    valid = series.dropna()

    if valid.empty:
        return pd.Series([None] * len(series), index=series.index)

    min_val = valid.min()
    max_val = valid.max()

    if min_val == max_val:
        return pd.Series([0.5 if pd.notnull(x) else None for x in series], index=series.index)

    return (series - min_val) / (max_val - min_val)


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Datacentre scrape log\n")
        f.write("=" * 80 + "\n")

    write_log("Starting scrape.")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df_base = pd.read_csv(INPUT_CSV)
    write_log(f"Loaded input file: {INPUT_CSV}")

    country_col = detect_country_column(df_base)
    iso3_col = detect_iso3_column(df_base)

    write_log(f"Detected country column: {country_col}")
    write_log(f"Detected ISO3 column: {iso3_col if iso3_col else 'None'}")

    keep_cols = [country_col]
    if iso3_col:
        keep_cols.append(iso3_col)

    df_countries = df_base[keep_cols].drop_duplicates().copy()
    df_countries = df_countries.rename(columns={country_col: "country"})
    if iso3_col:
        df_countries = df_countries.rename(columns={iso3_col: "iso3"})

    df_countries["country"] = df_countries["country"].astype(str).str.strip()
    df_countries = df_countries.sort_values("country").reset_index(drop=True)

    write_log(f"Found {len(df_countries)} unique countries to scrape.")

    rows = []

    for i, country in enumerate(df_countries["country"], start=1):
        write_log(f"Scraping {i}/{len(df_countries)}: {country}")
        row = scrape_country(country)
        rows.append(row)
        write_log(
            f"Finished {country} | status={row['scrape_status']} | "
            f"count={row['dc_facility_count']}"
        )
        time.sleep(REQUEST_DELAY_SECONDS)

    df_scraped = pd.DataFrame(rows)

    df_out = df_countries.merge(df_scraped, on="country", how="left")

    # Derived fields for later modelling / validation
    df_out["dc_facility_count_log"] = df_out["dc_facility_count"].apply(
        lambda x: math.log1p(x) if pd.notnull(x) else None
    )
    df_out["dc_maturity_norm"] = minmax_normalise(df_out["dc_facility_count_log"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    write_log(f"Saved output CSV to: {OUTPUT_CSV}")

    success_count = (df_out["scrape_status"] == "ok").sum()
    failed_count = len(df_out) - success_count

    write_log(f"Successful pages with extracted counts: {success_count}")
    write_log(f"Pages without extracted counts or with errors: {failed_count}")
    write_log("Scrape complete.")

    print("\nPreview of scraped data:")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()