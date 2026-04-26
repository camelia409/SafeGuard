"""
Clean the Zomato Bangalore dataset and load it into SQLite.

Run from project root:
    python src/01_clean_data.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def load_csv_with_fallbacks(csv_path: Path) -> pd.DataFrame:
    """Load CSV robustly so one encoding mismatch does not block the pipeline."""
    # Why: real-world exports often mix or mislabel encodings; we try common
    # options in order so ingestion is resilient without manual intervention.
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]
    last_error: Exception | None = None

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(csv_path, encoding=encoding, encoding_errors="replace")
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc

    raise RuntimeError(
        f"Failed to read CSV at {csv_path} with fallback encodings: {encodings_to_try}"
    ) from last_error


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "data" / "zomato.csv"
    output_db = project_root / "data" / "surgeguard.db"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_csv}")

    # Why: we need a baseline to measure how much data quality filtering removed.
    df = load_csv_with_fallbacks(input_csv)
    original_row_count = len(df)

    # Why: normalized column names avoid quoting/escaping pain in SQL workflows.
    df = df.rename(
        columns={
            "approx_cost(for two people)": "approx_cost",
            "listed_in(type)": "listing_type",
            "listed_in(city)": "listed_city",
        }
    )

    # Why: model and SQL features need a numeric rating; text tokens like NEW/- are
    # intentional "unknown" markers, so we coerce them to NaN.
    df["rate"] = (
        df["rate"]
        .astype(str)
        .str.replace("/5", "", regex=False)
        .str.strip()
        .replace({"NEW": pd.NA, "-": pd.NA, "nan": pd.NA})
    )
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    # Why: costs with thousand separators are strings; remove punctuation so cost
    # can be used in numeric aggregations and thresholds.
    df["approx_cost"] = (
        df["approx_cost"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"nan": pd.NA, "": pd.NA})
    )
    df["approx_cost"] = pd.to_numeric(df["approx_cost"], errors="coerce")

    # Why: booleans as 1/0 are easier to aggregate in SQL than Yes/No strings.
    yes_no_map = {"yes": 1, "no": 0}
    df["online_order"] = (
        df["online_order"].astype(str).str.strip().str.lower().map(yes_no_map)
    )
    df["book_table"] = (
        df["book_table"].astype(str).str.strip().str.lower().map(yes_no_map)
    )

    # Why: votes should be numeric for popularity features; invalid entries become
    # missing instead of silently corrupting values.
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").astype("Int64")

    # Why: location normalization reduces duplicate categories caused by casing and
    # spacing inconsistencies (e.g., " indiranagar " vs "Indiranagar").
    df["location"] = (
        df["location"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.title()
        .replace({"": pd.NA, "Nan": pd.NA})
    )

    # Why: rows without location cannot support geo-level feature engineering.
    df = df.dropna(subset=["location"])

    # Why: duplicate restaurant name/location pairs inflate counts and bias
    # aggregates; keep first observed record as canonical.
    df = df.drop_duplicates(subset=["name", "location"], keep="first")

    # Why: downstream SQL layer should consume a compact, analysis-focused schema.
    final_columns = [
        "name",
        "location",
        "online_order",
        "book_table",
        "rate",
        "votes",
        "approx_cost",
        "rest_type",
        "cuisines",
        "listing_type",
    ]
    df = df[final_columns].copy()

    final_row_count = len(df)
    rows_dropped = original_row_count - final_row_count
    null_counts = df.isnull().sum()
    unique_locations = df["location"].nunique(dropna=True)

    # Why: SQLite table makes cleaned data immediately queryable for SQL features.
    output_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_db) as conn:
        df.to_sql("restaurants", conn, if_exists="replace", index=False)

    # Why: report provides quick validation that cleaning steps behaved as expected.
    print("=== Zomato Cleaning Report ===")
    print(f"Original row count : {original_row_count}")
    print(f"Final row count    : {final_row_count}")
    print(f"Rows dropped       : {rows_dropped}")
    print("\nNull counts per column:")
    for column_name, count in null_counts.items():
        print(f"  - {column_name}: {int(count)}")
    print(f"\nUnique locations found: {unique_locations}")
    print(f"SQLite output: {output_db}")
    print("Table written: restaurants")


if __name__ == "__main__":
    main()
