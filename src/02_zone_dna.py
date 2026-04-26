"""
Build zone-level demand DNA features for Bangalore delivery risk.

Run from project root:
    python src/02_zone_dna.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ZONE_MAPPING = {
    "Koramangala 5Th Block": "Koramangala",
    "Koramangala 6Th Block": "Koramangala",
    "Koramangala 7Th Block": "Koramangala",
    "Koramangala 4Th Block": "Koramangala",
    "Koramangala 1St Block": "Koramangala",
    "Hsr": "HSR Layout",
    "Hsr Layout": "HSR Layout",
    "Indiranagar": "Indiranagar",
    "Cv Raman Nagar": "Indiranagar",
    "Whitefield": "Whitefield",
    "Marathahalli": "Whitefield",
    "Jp Nagar": "JP Nagar",
    "Jp Nagar 6Th Phase": "JP Nagar",
    "Jayanagar": "Jayanagar",
    "Btm": "BTM Layout",
    "Btm Layout": "BTM Layout",
    "Bellandur": "Bellandur",
    "Sarjapur Road": "Bellandur",
    "Banashankari": "Banashankari",
    "Bannerghatta Road": "Banashankari",
    "Malleshwaram": "Malleshwaram",
    "Rajajinagar": "Rajajinagar",
    "Mg Road": "MG Road",
    "Brigade Road": "MG Road",
    "Lavelle Road": "MG Road",
    "Church Street": "MG Road",
    "Ulsoor": "MG Road",
    "Electronic City": "Electronic City",
    "Electronic City Phase 1": "Electronic City",
    "Hebbal": "Hebbal",
    "Rt Nagar": "Hebbal",
    "Yeshwanthpur": "Yeshwanthpur",
    "Basavanagudi": "Basavanagudi",
    "Richmond Road": "Richmond Road",
    "Residency Road": "Richmond Road",
    "Cunningham Road": "Richmond Road",
    "Yelahanka": "Yelahanka",
    "Rr Nagar": "RR Nagar",
    "Mysore Road": "RR Nagar",
    "Kengeri": "RR Nagar",
    "Brookefield": "Brookefield",
    "Mahadevapura": "Brookefield",
    "Domlur": "Domlur",
    "Old Airport Road": "Domlur",
}


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Scale a feature to 0-1 while handling constant-value edge cases."""
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series([0.0] * len(series), index=series.index, dtype="float64")
    return (series - min_value) / (max_value - min_value)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "surgeguard.db"
    sql_output_path = project_root / "sql" / "zone_features.sql"

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        # Why: start from the cleaned restaurant base so zone features are built
        # off the same canonical dataset used by downstream feature engineering.
        restaurants_df = pd.read_sql_query("SELECT * FROM restaurants", conn)

        # Why: consolidating micro-localities into delivery zones avoids sparse
        # signals and better reflects rider/logistics behavior in operations.
        restaurants_df["zone"] = (
            restaurants_df["location"].map(ZONE_MAPPING).fillna(restaurants_df["location"])
        )

        # Why: we materialize a zoned table in SQLite so feature definitions remain
        # pure SQL, auditable, and easy to test independently of Python logic.
        restaurants_df.to_sql("restaurants_zoned", conn, if_exists="replace", index=False)

        query_definitions = [
            (
                "restaurant_count",
                "-- 1) Demand density: total restaurants available in each zone.\n"
                "SELECT\n"
                "    zone,\n"
                "    COUNT(*) AS restaurant_count\n"
                "FROM restaurants_zoned\n"
                "GROUP BY zone",
            ),
            (
                "engagement_score",
                "-- 2) Delivery engagement: average votes among online-ordering restaurants.\n"
                "SELECT\n"
                "    zone,\n"
                "    CAST(SUM(votes) AS FLOAT) / COUNT(*) AS engagement_score\n"
                "FROM restaurants_zoned\n"
                "WHERE online_order = 1\n"
                "GROUP BY zone",
            ),
            (
                "volatility_index",
                "-- 3) Demand volatility proxy: share of Quick Bites restaurants in each zone.\n"
                "SELECT\n"
                "    zone,\n"
                "    CAST(SUM(CASE WHEN rest_type = 'Quick Bites' THEN 1 ELSE 0 END) AS FLOAT)\n"
                "        / COUNT(*) AS volatility_index\n"
                "FROM restaurants_zoned\n"
                "GROUP BY zone",
            ),
            (
                "affluence_proxy",
                "-- 4) Spending proxy: average cost for two where price data is present.\n"
                "SELECT\n"
                "    zone,\n"
                "    AVG(approx_cost) AS affluence_proxy\n"
                "FROM restaurants_zoned\n"
                "WHERE approx_cost IS NOT NULL\n"
                "GROUP BY zone",
            ),
            (
                "online_penetration",
                "-- 5) Digital readiness: share of restaurants accepting online orders.\n"
                "SELECT\n"
                "    zone,\n"
                "    CAST(SUM(online_order) AS FLOAT) / COUNT(*) AS online_penetration\n"
                "FROM restaurants_zoned\n"
                "GROUP BY zone",
            ),
        ]

        # Why: persisting SQL definitions improves governance and lets analysts
        # re-run/inspect the exact feature logic without reading Python code.
        sql_output_path.parent.mkdir(parents=True, exist_ok=True)
        with sql_output_path.open("w", encoding="utf-8") as sql_file:
            sql_file.write("-- Zone DNA feature queries\n")
            sql_file.write("-- Source table: restaurants_zoned\n\n")
            for _, query in query_definitions:
                sql_file.write(query)
                sql_file.write(";\n\n")

        feature_frames: list[pd.DataFrame] = []
        for feature_name, query in query_definitions:
            print(f"\n=== SQL for {feature_name} ===")
            print(query + ";")
            feature_frames.append(pd.read_sql_query(query, conn))

        # Why: each SQL query returns one zone-level metric; joining them builds
        # a single profile table for scoring and downstream model consumption.
        zone_dna = feature_frames[0]
        for frame in feature_frames[1:]:
            zone_dna = zone_dna.merge(frame, on="zone", how="outer")

        # Why: missing values typically mean feature unavailable in that zone
        # (e.g., no priced records), and zero is a conservative default.
        feature_columns = [
            "restaurant_count",
            "engagement_score",
            "volatility_index",
            "affluence_proxy",
            "online_penetration",
        ]
        zone_dna[feature_columns] = zone_dna[feature_columns].fillna(0)

        # Why: min-max scaling puts heterogeneous features on a common risk axis
        # before weighted blending into one volatility score.
        for column in feature_columns:
            normalized_column = f"{column}_norm"
            zone_dna[normalized_column] = min_max_normalize(zone_dna[column].astype(float))

        zone_dna["zone_volatility_score"] = (
            0.25 * zone_dna["restaurant_count_norm"]
            + 0.30 * zone_dna["engagement_score_norm"]
            + 0.25 * zone_dna["volatility_index_norm"]
            + 0.10 * zone_dna["affluence_proxy_norm"]
            + 0.10 * zone_dna["online_penetration_norm"]
        )

        zone_dna = zone_dna.sort_values(
            by="zone_volatility_score", ascending=False
        ).reset_index(drop=True)

        # Why: storing scored zones in SQLite keeps the analytics stack centered
        # in SQL so later risk models and dashboards can query directly.
        zone_dna.to_sql("zone_dna", conn, if_exists="replace", index=False)

    # Why: a ranked report gives immediate sanity-check visibility into whether
    # known high-demand neighborhoods surface at the top as expected.
    print("\n=== Zone DNA Report (Sorted by zone_volatility_score) ===")
    print(zone_dna[["zone", "zone_volatility_score"] + feature_columns].to_string(index=False))
    print(f"\nTotal zones profiled: {zone_dna['zone'].nunique()}")
    print(f"SQLite output table: {db_path} -> zone_dna")
    print(f"Saved SQL definitions: {sql_output_path}")


if __name__ == "__main__":
    main()
