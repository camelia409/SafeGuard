"""
Combine Zone DNA and real-time triggers into breach risk scores.

Run from project root:
    python src/04_risk_scorer.py
"""

from __future__ import annotations

from datetime import datetime
import importlib.util
import math
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
# from scipy.stats import poisson  # Removed: switching to linear model


RISK_COLORS = {
    "LOW": "#2ECC71",
    "MEDIUM": "#F39C12",
    "HIGH": "#E67E22",
    "CRITICAL": "#E74C3C",
}

RIDER_MULTIPLIERS = {
    "LOW": 1.0,  # 1.0: steady-state staffing is sufficient for low-pressure zones.
    "MEDIUM": 1.3,  # 1.3: modest surge buffer to absorb moderate demand spikes.
    "HIGH": 1.6,  # 1.6: larger buffer to prevent SLA slippage during peak stress.
    "CRITICAL": 2.0,  # 2.0: emergency doubling to stabilize severe shortage risk.
}


def _risk_bucket(score: float) -> tuple[str, str]:
    """Map numeric score to risk label and dashboard color."""
    if score <= 25:
        return "LOW", RISK_COLORS["LOW"]
    if score <= 50:
        return "MEDIUM", RISK_COLORS["MEDIUM"]
    if score <= 75:
        return "HIGH", RISK_COLORS["HIGH"]
    return "CRITICAL", RISK_COLORS["CRITICAL"]


def _load_trigger_engine() -> Any:
    """
    Dynamically load trigger engine module.

    Why: filenames starting with numbers (03_trigger_engine.py) are not directly
    importable with a standard `from module import ...` statement.
    """
    project_root = Path(__file__).resolve().parents[1]
    trigger_path = project_root / "src" / "03_trigger_engine.py"
    spec = importlib.util.spec_from_file_location("trigger_engine", trigger_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load trigger engine from {trigger_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_risk_scores(
    zone_dna_df: pd.DataFrame, trigger_vector: dict[str, Any]
) -> pd.DataFrame:
    """Compute breach risk scores, rider recommendations, and donor zones."""
    scored_df = zone_dna_df.copy()

    combined_multiplier = float(trigger_vector["combined_multiplier"])

    # Why: keep scoring transparent and explainable; product model makes the
    # static zone pressure explicit while allowing dynamic trigger amplification.
    scored_df["raw_score"] = scored_df["zone_volatility_score"] * combined_multiplier * 100

    # Why: cap at 100 to keep interpretation bounded and dashboard scales stable.
    scored_df["breach_risk_score"] = scored_df["raw_score"].clip(upper=100)

    risk_and_colors = scored_df["breach_risk_score"].apply(lambda x: _risk_bucket(float(x)))
    scored_df["risk_level"] = risk_and_colors.apply(lambda x: x[0])
    scored_df["color"] = risk_and_colors.apply(lambda x: x[1])

    # Why: operational heuristic assumes one rider can reliably cover ~15 active
    # restaurants; simple and practical for dispatch pre-planning.
    base_riders = scored_df["restaurant_count"] / 15.0
    scored_df["recommended_riders"] = (
        base_riders * scored_df["risk_level"].map(RIDER_MULTIPLIERS)
    ).apply(math.ceil)

    low_zones = scored_df[scored_df["risk_level"] == "LOW"][
        ["zone", "zone_volatility_score", "breach_risk_score"]
    ].copy()

    donor_zones: list[str] = []
    for _, row in scored_df.iterrows():
        if row["risk_level"] in {"HIGH", "CRITICAL"} and not low_zones.empty:
            # Why: nearest LOW zone by volatility profile should have comparable
            # operating rhythm and is a pragmatic donor candidate.
            low_zones["distance"] = (
                low_zones["zone_volatility_score"] - row["zone_volatility_score"]
            ).abs()
            donor = low_zones.sort_values("distance").iloc[0]
            donor_message = (
                f"Recommend sourcing riders from {donor['zone']} "
                f"(score: {donor['breach_risk_score']:.1f} - LOW)"
            )
            donor_zones.append(donor_message)
        else:
            donor_zones.append("No donor action needed")

    scored_df["donor_zone"] = donor_zones
    scored_df["weather_description"] = str(trigger_vector["weather_description"])
    scored_df["temporal_description"] = str(trigger_vector["temporal_description"])
    scored_df["event_description"] = str(trigger_vector["event_description"])
    scored_df["combined_multiplier"] = combined_multiplier
    scored_df["timestamp"] = str(trigger_vector["timestamp"])

    return scored_df[
        [
            "zone",
            "zone_volatility_score",
            "breach_risk_score",
            "risk_level",
            "color",
            "recommended_riders",
            "donor_zone",
            "weather_description",
            "temporal_description",
            "event_description",
            "combined_multiplier",
            "timestamp",
            "restaurant_count",
        ]
    ].sort_values(by="breach_risk_score", ascending=False).reset_index(drop=True)


def simulate_rider_addition(
    zone_name: str, extra_riders: int, scored_df: pd.DataFrame
) -> dict[str, Any]:
    """
    Linear model: each rider reduces score by 2.5 points.
    Transparent, explainable, always shows meaningful change.
    """
    zone_row = scored_df[scored_df["zone"] == zone_name]
    if zone_row.empty:
        return {
            "zone": zone_name,
            "original_score": 0.0,
            "original_risk_level": "LOW",
            "extra_riders": extra_riders,
            "new_score": 0.0,
            "new_risk_level": "LOW",
            "recommendation": "Zone not found",
            "donor_zone": "N/A",
            "demand_rate": 0.0,
            "base_supply": 0,
        }

    row = zone_row.iloc[0]
    original_score = float(row["breach_risk_score"])

    # Linear model: each rider reduces score by 2.5 points
    reduction = extra_riders * 2.5
    new_score = max(0.0, original_score - reduction)

    def get_risk_level(score):
        if score <= 25:
            return "LOW"
        elif score <= 50:
            return "MEDIUM"
        elif score <= 75:
            return "HIGH"
        else:
            return "CRITICAL"

    original_risk = get_risk_level(original_score)
    new_risk = get_risk_level(new_score)

    # Find donor zone (lowest scoring zone that isn't this zone)
    other_zones = scored_df[scored_df["zone"] != zone_name]
    if not other_zones.empty:
        donor_row = other_zones.loc[other_zones["breach_risk_score"].idxmin()]
        donor_text = (
            f"{donor_row['zone']} "
            f"(score: {donor_row['breach_risk_score']:.1f} "
            f"— {donor_row['risk_level']})"
        )
    else:
        donor_text = "No alternate zone available"

    if original_risk == new_risk:
        recommendation = (
            f"Adding {extra_riders} riders reduces score "
            f"from {original_score:.1f} to {new_score:.1f} "
            f"(still {new_risk})"
        )
    else:
        recommendation = (
            f"Adding {extra_riders} riders moves {zone_name} "
            f"from {original_risk} to {new_risk}"
        )

    return {
        "zone": zone_name,
        "original_score": original_score,
        "original_risk_level": original_risk,
        "extra_riders": extra_riders,
        "new_score": new_score,
        "new_risk_level": new_risk,
        "recommendation": recommendation,
        "donor_zone": donor_text,
        "demand_rate": original_score / 5,
        "base_supply": extra_riders,
    }



def save_scores(scored_df: pd.DataFrame) -> None:
    """Persist risk scores to SQLite for downstream dashboard/routing layers."""
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "surgeguard.db"
    to_save = scored_df.copy()
    to_save["scored_at"] = datetime.now().isoformat()

    with sqlite3.connect(db_path) as conn:
        to_save.to_sql("risk_scores", conn, if_exists="replace", index=False)


def print_risk_report(scored_df: pd.DataFrame, trigger_vector: dict[str, Any]) -> None:
    """Print human-readable risk summary and operational guidance."""
    print("\n=== Current Trigger Summary ===")
    print(
        f"Weather : {trigger_vector['weather_description']} "
        f"(x{trigger_vector['weather_multiplier']:.2f})"
    )
    print(
        f"Temporal: {trigger_vector['temporal_description']} "
        f"(x{trigger_vector['temporal_multiplier']:.2f})"
    )
    print(
        f"Event   : {trigger_vector['event_description']} "
        f"(x{trigger_vector['event_multiplier']:.2f})"
    )
    print(f"Combined Multiplier: x{trigger_vector['combined_multiplier']:.2f}")

    print("\n=== Zone Breach Risk Table (Highest to Lowest) ===")
    table_cols = [
        "zone",
        "zone_volatility_score",
        "breach_risk_score",
        "risk_level",
        "recommended_riders",
        "donor_zone",
    ]
    print(scored_df[table_cols].to_string(index=False))

    print("\n=== Risk Level Counts ===")
    counts = scored_df["risk_level"].value_counts().reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0
    )
    for level, count in counts.items():
        print(f"{level}: {int(count)}")

    print("\n=== Top 3 CRITICAL/HIGH Zones ===")
    top_stressed = scored_df[scored_df["risk_level"].isin(["CRITICAL", "HIGH"])].head(3)
    if top_stressed.empty:
        print("No zones in HIGH/CRITICAL at current trigger levels.")
    else:
        for _, row in top_stressed.iterrows():
            print(
                f"- {row['zone']} | score={row['breach_risk_score']:.1f} | "
                f"{row['risk_level']} | {row['donor_zone']}"
            )

    print("\n=== ASSUMPTIONS ===")
    print("- Counterfactual model: risk is evaluated via linear rider simulation (2.5 pts).")
    print("- Base rider formula: restaurant_count / 15 before risk-tier uplift.")
    print("- Volatility source: Zone DNA built from Zomato delivery proxies.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "surgeguard.db"

    with sqlite3.connect(db_path) as conn:
        zone_dna_df = pd.read_sql_query("SELECT * FROM zone_dna", conn)

    trigger_module = _load_trigger_engine()
    get_trigger_vector = trigger_module.get_trigger_vector
    trigger = get_trigger_vector()

    scored = compute_risk_scores(zone_dna_df, trigger)
    save_scores(scored)
    print_risk_report(scored, trigger)

    highest_zone = scored.iloc[0]["zone"]
    simulation = simulate_rider_addition(highest_zone, 3, scored)
    print("\n=== Counterfactual Simulation ===")
    print(
        f"Zone: {simulation['zone']} | Original: {simulation['original_score']:.1f} "
        f"({simulation['original_risk_level']}) -> New: {simulation['new_score']:.1f} "
        f"({simulation['new_risk_level']})"
    )
    print(simulation["recommendation"])
