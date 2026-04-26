"""
SurgeGuard Streamlit dashboard for Bangalore operations.

Install dependencies:
    pip install streamlit folium streamlit-folium streamlit-autorefresh
"""

from __future__ import annotations

from datetime import datetime
import math
import importlib.util
from pathlib import Path
import sqlite3
from typing import Any

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium


st.set_page_config(page_title="SurgeGuard | Bangalore", page_icon="🛡️", layout="wide")


# ── Zone coordinates — canonical mapped delivery zones (operations view) ───
ZONE_COORDS: dict[str, tuple[float, float]] = {
    "BTM Layout":                    (12.9166, 77.6101),
    "Banashankari":                  (12.9255, 77.5468),
    "Basavanagudi":                  (12.9422, 77.5757),
    "Bellandur":                     (12.9261, 77.6762),
    "Brookefield":                   (12.9540, 77.7108),
    "Domlur":                        (12.9609, 77.6387),
    "Electronic City":               (12.8399, 77.6770),
    "HSR Layout":                    (12.9116, 77.6389),
    "Hebbal":                        (13.0450, 77.5950),
    "Indiranagar":                   (12.9784, 77.6408),
    "JP Nagar":                      (12.9063, 77.5857),
    "Jayanagar":                     (12.9308, 77.5838),
    "Koramangala":                   (12.9352, 77.6245),
    "MG Road":                       (12.9756, 77.6057),
    "Malleshwaram":                  (12.9997, 77.5660),
    "RR Nagar":                      (12.9259, 77.5180),
    "Rajajinagar":                   (12.9955, 77.5530),
    "Richmond Road":                 (12.9641, 77.5985),
    "Whitefield":                    (12.9698, 77.7499),
    "Yelahanka":                     (13.1007, 77.5963),
    "Yeshwanthpur":                  (13.0275, 77.5460),
}

RISK_COLORS = {
    "LOW":      "#2ECC71",
    "MEDIUM":   "#F39C12",
    "HIGH":     "#E67E22",
    "CRITICAL": "#E74C3C",
}


def _dynamic_load_module(module_name: str, module_path: Path) -> Any:
    """Load modules whose filenames start with numbers (not standard import-safe)."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RISK_MODULE = _dynamic_load_module("risk_scorer", PROJECT_ROOT / "src" / "04_risk_scorer.py")
TRIGGER_MODULE = _dynamic_load_module(
    "trigger_engine", PROJECT_ROOT / "src" / "03_trigger_engine.py"
)
compute_risk_scores = RISK_MODULE.compute_risk_scores
simulate_rider_addition = RISK_MODULE.simulate_rider_addition
get_trigger_vector = TRIGGER_MODULE.get_trigger_vector
get_12hour_forecast = TRIGGER_MODULE.get_12hour_forecast
get_risk_level_from_score = TRIGGER_MODULE.get_risk_level_from_score


def _weather_icon(description: str) -> str:
    desc = description.lower()
    if "rain" in desc:
        return "🌧️"
    if "cloud" in desc:
        return "🌤️"
    return "☀️"


def _primary_driver_label(
    trigger_vector: dict[str, Any],
    zone_volatility_score: float,
) -> str:
    """
    BUG 2 FIX — deviation-from-baseline logic.
    Returns a human-readable primary driver label for a specific zone row.
    """
    weather_mult   = float(trigger_vector["weather_multiplier"])
    temporal_mult  = float(trigger_vector["temporal_multiplier"])
    event_mult     = float(trigger_vector["event_multiplier"])
    weather_desc   = str(trigger_vector["weather_description"])
    temporal_desc  = str(trigger_vector["temporal_description"])
    event_desc     = str(trigger_vector["event_description"])

    weather_dev  = abs(weather_mult - 1.0)
    temporal_dev = abs(temporal_mult - 1.0)
    event_dev    = abs(event_mult - 1.0)

    if max(weather_dev, temporal_dev, event_dev) < 0.15 and zone_volatility_score > 0.6:
        return "Zone Structure (High Volatility Area)"
    elif weather_dev >= temporal_dev and weather_dev >= event_dev:
        return f"Weather ({weather_desc}) at {weather_mult:.2f}x"
    elif temporal_dev >= event_dev:
        return f"Time Pattern ({temporal_desc}) at {temporal_mult:.2f}x"
    else:
        return f"Event ({event_desc}) at {event_mult:.2f}x"


def _short_weather(description: str) -> str:
    """Return a concise weather label that fits in a KPI card."""
    desc = description.strip()
    for prefix in ("Clear/Baseline", "Clear/", "Baseline"):
        if desc.startswith(prefix):
            desc = desc[len(prefix):].strip("()/- ")
    return desc[:22] or "OK"


# ── CSS injection ───────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    .block-container { padding-top: 1rem !important; }

    /* Base theme */
    .stApp { background-color: #1A1A2E; color: #EAEAEA; }
    [data-testid="stSidebar"] { background-color: #0F3460; }
    [data-testid="stSidebar"] * { color: #EAEAEA !important; }

    /* Global text */
    .stMarkdown, p, span, div, h1, h2, h3, h4, h5, h6, label { color: #EAEAEA !important; }

    /* st.metric card shell */
    [data-testid="stMetric"] {
        background: #16213E;
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    /* metric value — the big number */
    [data-testid="stMetricValue"] > div { color: #FFFFFF !important; font-size: 1.8rem !important; }
    /* metric label — the title above */
    [data-testid="stMetricLabel"] > div { color: #A0A0B0 !important; font-size: 0.85rem !important; }
    /* metric delta — small label below */
    [data-testid="stMetricDelta"] { color: #7EC8E3 !important; }
    [data-testid="stMetricDelta"] * { color: #7EC8E3 !important; }

    /* Progress bars */
    [data-testid="stProgressBar"] > div { background-color: #4A90D9 !important; }

    /* st.caption */
    [data-testid="stCaptionContainer"] p { color: #A0A0B0 !important; }

    /* Body/write text */
    [data-testid="stText"] { color: #EAEAEA !important; }
    .element-container p { color: #EAEAEA !important; }

    /* st.info */
    [data-testid="stAlert"] { background: #16213E !important; }
    [data-testid="stAlert"] p { color: #EAEAEA !important; }

    /* Selectbox */
    [data-testid="stSelectbox"] label { color: #EAEAEA !important; }
    [data-testid="stSelectbox"] span { color: #1A1A2E !important; }

    /* Slider */
    [data-testid="stSlider"] label { color: #EAEAEA !important; }
    [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    [data-testid="stSlider"] div[data-testid="stTickBarMax"] { color: #A0A0B0 !important; }

    /* Expanders */
    [data-testid="stExpander"] summary span { color: #EAEAEA !important; }
    [data-testid="stExpander"] div { color: #EAEAEA !important; }

    /* Tab styling */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #16213E;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #A0A0B0 !important;
        font-weight: 600;
        padding: 8px 16px;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #0F3460 !important;
        color: #FFFFFF !important;
    }

    /* Custom classes */
    .subtle { color: #A0A0B0 !important; font-size: 0.95rem; }
    .zone-card { background: #16213E; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }
    .info-card { background: #16213E; border-left: 4px solid #0F3460;
        border-radius: 10px; padding: 12px; color: #EAEAEA; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Auto-refresh & timing ───────────────────────────────────────────────────
refresh_count = st_autorefresh(interval=900_000, key="refresh")

if "last_refresh_count" not in st.session_state:
    st.session_state["last_refresh_count"] = refresh_count
    st.session_state["last_refresh_time"] = datetime.now()
elif st.session_state["last_refresh_count"] != refresh_count:
    st.session_state["last_refresh_count"] = refresh_count
    st.session_state["last_refresh_time"] = datetime.now()

# ── Data load ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def load_live_data() -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """
    Load Zone DNA and compute live risk scores.

    Cached for 15 minutes to align with dashboard refresh cycle and avoid
    excessive API calls while keeping operational visibility near-real-time.
    """
    db_path = PROJECT_ROOT / "data" / "surgeguard.db"
    with sqlite3.connect(db_path) as conn:
        zone_dna_df = pd.read_sql_query("SELECT * FROM zone_dna", conn)

    trigger_vector = get_trigger_vector()
    scored_df = compute_risk_scores(zone_dna_df, trigger_vector)
    forecast = get_12hour_forecast(
        float(trigger_vector["weather_multiplier"]),
        float(trigger_vector["event_multiplier"]),
    )
    scored_df = scored_df.merge(
        zone_dna_df[
            [
                "zone",
                "restaurant_count_norm",
                "engagement_score_norm",
                "volatility_index_norm",
                "affluence_proxy_norm",
                "online_penetration_norm",
            ]
        ],
        on="zone",
        how="left",
    )
    return scored_df, trigger_vector, forecast, zone_dna_df


try:
    scored_df, trigger_vector, forecast, zone_dna_df = load_live_data()
except Exception as exc:
    st.error(f"Unable to load live data pipeline: {exc}")
    st.stop()

if str(trigger_vector.get("weather_description", "")).strip().lower() == "api unavailable":
    st.warning("Weather API unavailable — using baseline weather multiplier (1.0).")

# BUG 3 FIX — filter scored_df to only zones present in ZONE_COORDS
mapped_zones_df = scored_df[
    scored_df["zone"].isin(ZONE_COORDS.keys())
].copy()

# Countdown timer
elapsed_seconds = int((datetime.now() - st.session_state["last_refresh_time"]).total_seconds())
remaining_seconds = max(0, 900 - elapsed_seconds)
minutes, seconds = divmod(remaining_seconds, 60)

# Shared derived values
weather_mult    = float(trigger_vector["weather_multiplier"])
temporal_mult   = float(trigger_vector["temporal_multiplier"])
event_mult      = float(trigger_vector["event_multiplier"])
weather_desc    = str(trigger_vector.get("weather_description", "N/A"))
temporal_desc   = str(trigger_vector.get("temporal_description", "N/A"))
event_desc      = str(trigger_vector.get("event_description", "N/A"))
temp_c          = float(trigger_vector.get("bangalore_temp_c", 0.0))
weather_icon    = _weather_icon(weather_desc)

# BUG 3 FIX — KPI counts use mapped zones only
critical_count = int((mapped_zones_df["risk_level"] == "CRITICAL").sum())
high_count     = int((mapped_zones_df["risk_level"] == "HIGH").sum())

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🛡️ SurgeGuard — Bangalore Live Operations")
st.markdown('<p class="subtle">Hyperlocal SLA Breach Early Warning System</p>', unsafe_allow_html=True)
st.markdown(
    f"""
<p class="subtle">
Last updated: {trigger_vector.get("timestamp", "N/A")} |
Next refresh in: {minutes:02d}:{seconds:02d} |
Weather: {weather_icon} {weather_desc} |
Temp: {temp_c:.1f}°C
</p>
""",
    unsafe_allow_html=True,
)

# ── Tab structure ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🗺️ Live Operations",
    "📈 Risk Forecast",
    "🔍 Zone Intelligence",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Operations
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI Strip ─────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            "Zones Monitored",
            len(mapped_zones_df),
            delta="Live",
            delta_color="off",
        )
    with kpi2:
        st.metric(
            "Critical Zones",
            critical_count,
            delta="Requires action" if critical_count > 0 else "All clear",
            delta_color="inverse" if critical_count > 0 else "off",
        )
    with kpi3:
        st.metric(
            "High Risk Zones",
            high_count,
            delta="Monitor closely" if high_count > 0 else "All clear",
            delta_color="inverse" if high_count > 0 else "off",
        )
    with kpi4:
        weather_short = _short_weather(weather_desc)
        st.metric(
            "Weather",
            f"{weather_icon} {weather_short}",
            delta=f"{temp_c:.1f}°C Bangalore",
        )

    # ── Map + Rankings ─────────────────────────────────────────────────────
    map_col, rank_col = st.columns([6, 4])

    with map_col:
        st.subheader("Bangalore Zone Risk Map")

        if "map_zoom" not in st.session_state:
            st.session_state["map_zoom"] = 12
        if "map_center" not in st.session_state:
            st.session_state["map_center"] = [12.9716, 77.5946]

        risk_map = folium.Map(
            location=st.session_state["map_center"],
            zoom_start=st.session_state["map_zoom"],
            max_zoom=16,
            min_zoom=10,
            tiles="CartoDB dark_matter",
        )

        # Top-5 highest-risk zones get text labels
        top5_zones = set(mapped_zones_df.sort_values("breach_risk_score", ascending=False).head(5)["zone"].tolist())

        matched = 0
        for _, row in mapped_zones_df.iterrows():
            zone_name = row["zone"]
            matched += 1
            lat, lon = ZONE_COORDS[zone_name]
            color = row["color"]
            zone_vol = float(row.get("zone_volatility_score", 0.0))
            driver_label = _primary_driver_label(trigger_vector, zone_vol)

            popup_html = f"""
            <div style='font-family:Arial; min-width:220px;'>
                <b>{zone_name}</b><br>
                Score: {row['breach_risk_score']:.1f}<br>
                <span style='background:{color};color:white;padding:3px 8px;border-radius:12px;'>
                    {row['risk_level']}
                </span><br><br>
                Primary Trigger: {driver_label}<br>
                Recommended Riders: {int(row['recommended_riders'])}
            </div>
            """
            tooltip = f"{zone_name} | {row['risk_level']}"
            folium.CircleMarker(
                location=(lat, lon),
                radius=12,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=tooltip,
            ).add_to(risk_map)

            if zone_name in top5_zones:
                folium.Marker(
                    location=(lat, lon),
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:9px;color:white;font-weight:bold;'
                            f'white-space:nowrap;text-shadow:1px 1px 2px black;">'
                            f'{zone_name}</div>'
                        ),
                        icon_size=(100, 20),
                        icon_anchor=(50, 0),
                    ),
                ).add_to(risk_map)

        map_state = st_folium(
            risk_map,
            width=None,
            height=520,
            returned_objects=["zoom", "center"],
            key="risk_map",
        )
        if map_state and map_state.get("zoom"):
            st.session_state["map_zoom"] = map_state["zoom"]
        if map_state and map_state.get("center"):
            st.session_state["map_center"] = [
                map_state["center"]["lat"],
                map_state["center"]["lng"],
            ]

        st.caption(f"ℹ️ {matched} mapped zones plotted.")

    with rank_col:
        st.subheader("Zone Risk Rankings")
        # BUG 3 FIX — use mapped_zones_df only
        sorted_df = mapped_zones_df.sort_values("breach_risk_score", ascending=False)
        cards_html = '<div style="max-height:520px;overflow-y:auto;padding-right:6px;">'
        for _, row in sorted_df.iterrows():
            color = RISK_COLORS.get(row["risk_level"], "#A0A0B0")
            cards_html += f"""
<div style="background:#16213E;border-radius:10px;padding:10px 12px;
        margin-bottom:10px;border-left:6px solid {color};">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <b style="color:#EAEAEA;">{row['zone']}</b>
    <span style="display:inline-block;padding:4px 10px;border-radius:999px;
                 font-size:0.8rem;font-weight:700;color:#FFF;
                 background:{color};">{row['risk_level']}</span>
  </div>
  <div style="color:#A0A0B0;margin-top:4px;">Score: {row['breach_risk_score']:.1f}</div>
</div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── System Status Banner ───────────────────────────────────────────────
    sorted_mapped = mapped_zones_df.sort_values("breach_risk_score", ascending=False)
    # Find next peak hour from real forecast data
    next_peak = next(
        (h for h in forecast if h["is_peak"]),
        None,
    )
    if next_peak:
        next_peak_str = (
            f"{next_peak['hour_label']} "
            f"({next_peak['temporal_description']})"
        )
    else:
        next_peak_str = "no peak window in next 12 hours"

    if critical_count > 0:
        banner_color = "#E74C3C"
        banner_icon  = "🔴"
        banner_msg   = f"CRITICAL: {critical_count} zone{'s' if critical_count > 1 else ''} require immediate rider deployment"
    elif high_count > 0:
        banner_color = "#E67E22"
        banner_icon  = "🟠"
        top_zone     = sorted_mapped.iloc[0]["zone"]
        banner_msg   = f"{high_count} zone{'s' if high_count > 1 else ''} elevated — monitor {top_zone} closely"
    else:
        banner_color = "#2ECC71"
        banner_icon  = "🟢"
        banner_msg   = f"All zones operating normally — next elevated risk window: {next_peak_str}"

    st.markdown(
        f'<div style="background:{banner_color}22;border-left:4px solid {banner_color};'
        f'padding:12px 16px;border-radius:6px;margin-top:16px;color:#EAEAEA;">'
        f'{banner_icon} {banner_msg}</div>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Risk Forecast
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    peak_hour = max(forecast, key=lambda x: x["combined_multiplier"])
    peak_val = float(peak_hour["combined_multiplier"])
    peak_label = str(peak_hour["hour_label"])
    peak_desc = str(peak_hour["temporal_description"])

    if peak_val >= 1.6:
        alert_color = "#E67E22"
        alert_icon = "⚡"
        alert_msg = f"Peak risk window at {peak_label}"
        sub_msg = (
            f"Temporal pattern: {peak_desc} "
            f"({peak_val:.2f}x combined multiplier) — "
            f"Pre-position riders before this window"
        )
    elif peak_val >= 1.2:
        alert_color = "#F39C12"
        alert_icon = "⚠️"
        alert_msg = f"Moderate elevation expected at {peak_label}"
        sub_msg = f"Combined multiplier: {peak_val:.2f}x"
    else:
        alert_color = "#2ECC71"
        alert_icon = "✅"
        alert_msg = "No significant risk peaks in next 12 hours"
        sub_msg = "Current conditions suggest calm delivery period"

    st.markdown(
        f"""
<div style="background:{alert_color}22;border-left:4px solid {alert_color};padding:16px 20px;
            border-radius:8px;margin-bottom:20px;">
  <div style="font-size:1.2rem;font-weight:700;color:{alert_color};">
    {alert_icon} {alert_msg}
  </div>
  <div style="color:#A0A0B0;margin-top:6px;">
    {sub_msg}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    top5_zones_forecast = (
        zone_dna_df[zone_dna_df["zone"].isin(ZONE_COORDS.keys())]
        .sort_values("zone_volatility_score", ascending=False)
        .head(5)
    )
    rows: list[dict[str, Any]] = []
    for _, zone_row in top5_zones_forecast.iterrows():
        zone_name = str(zone_row["zone"])
        zone_vol = float(zone_row["zone_volatility_score"])
        for h in forecast:
            # FIX 2: Removal of score cap for chart only
            proj_score = zone_vol * float(h["combined_multiplier"]) * 100
            risk_lvl = get_risk_level_from_score(min(proj_score, 100))
            rows.append(
                {
                    "Hour": h["hour_label"],
                    "Zone": zone_name,
                    "Relative Risk Score": round(proj_score, 1),
                    "Risk Level": risk_lvl,
                    "Is Peak": bool(h["is_peak"]),
                }
            )
    df_forecast = pd.DataFrame(rows)

    fig = px.line(
        df_forecast,
        x="Hour",
        y="Relative Risk Score",
        color="Zone",
        title="📈 Projected Breach Risk — Next 12 Hours",
        markers=True,
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#EAEAEA",
        title_font_size=16,
        legend_title="Zone",
        hovermode="x unified",
        height=450,
        margin=dict(t=50, b=40, l=40, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(color="#EAEAEA"),
        ),
    )
    fig.add_hline(
        y=25,
        line_dash="dot",
        line_color="#2ECC71",
        annotation_text="LOW/MEDIUM threshold",
        annotation_font_color="#2ECC71",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="#F39C12",
        annotation_text="MEDIUM/HIGH threshold",
        annotation_font_color="#F39C12",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=75,
        line_dash="dot",
        line_color="#E67E22",
        annotation_text="HIGH/CRITICAL threshold",
        annotation_font_color="#E67E22",
        annotation_position="bottom left",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Zone × Hour Risk Heatmap")
    st.caption("L=LOW | M=MEDIUM | H=HIGH | C=CRITICAL | number=score")

    all_mapped = zone_dna_df[zone_dna_df["zone"].isin(ZONE_COORDS.keys())].sort_values(
        "zone_volatility_score", ascending=False
    )

    table_rows: list[dict[str, Any]] = []
    for _, zr in all_mapped.iterrows():
        row: dict[str, Any] = {"Zone": zr["zone"]}
        for h in forecast:
            proj = min(float(zr["zone_volatility_score"]) * float(h["combined_multiplier"]) * 100, 100)
            risk = get_risk_level_from_score(proj)
            # FIX 3: Add score in parentheses
            row[h["hour_label"]] = f"{risk[0]}({proj:.0f})"
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows).set_index("Zone")

    def style_cell(val: str) -> str:
        first_char = str(val)[0] if val else "L"
        color_map = {
            "L": "background-color:#2ECC7122;color:#2ECC71;font-weight:600",
            "M": "background-color:#F39C1222;color:#F39C12;font-weight:600",
            "H": "background-color:#E67E2222;color:#E67E22;font-weight:600",
            "C": "background-color:#E74C3C22;color:#E74C3C;font-weight:600",
        }
        return color_map.get(first_char, "")

    styled_table = df_table.style.map(style_cell)
    st.dataframe(styled_table, use_container_width=True)

    st.subheader("🚴 Pre-Positioning Recommendations")
    st.caption(f"Based on peak window at {peak_label} ({peak_val:.2f}x combined multiplier)")

    peak_scores: list[dict[str, Any]] = []
    for _, zr in all_mapped.iterrows():
        proj = min(float(zr["zone_volatility_score"]) * peak_val * 100, 100)
        rest_count = float(zr.get("restaurant_count", 20))
        risk_lvl = get_risk_level_from_score(proj)
        risk_mult = {
            "LOW": 1.0,
            "MEDIUM": 1.3,
            "HIGH": 1.6,
            "CRITICAL": 2.0,
        }.get(risk_lvl, 1.0)
        base = min(rest_count, 150) / 15
        recommended = min(math.ceil(base * risk_mult), 50)
        peak_scores.append(
            {
                "zone": zr["zone"],
                "peak_score": round(proj, 1),
                "risk_level": risk_lvl,
                "recommended_riders": recommended,
            }
        )

    donor_zone = min(peak_scores, key=lambda x: x["peak_score"])["zone"]
    top3 = sorted(peak_scores, key=lambda x: x["peak_score"], reverse=True)[:3]

    cols = st.columns(3)
    risk_colors = {
        "LOW": "#2ECC71",
        "MEDIUM": "#F39C12",
        "HIGH": "#E67E22",
        "CRITICAL": "#E74C3C",
    }
    for i, zone_data in enumerate(top3):
        color = risk_colors[zone_data["risk_level"]]
        with cols[i]:
            st.markdown(
                f"""
<div style="background:#16213E;border-left:4px solid {color};border-radius:8px;padding:16px;height:140px;">
  <div style="font-weight:700;color:#EAEAEA;font-size:1rem;">
    {zone_data["zone"]}
  </div>
  <div style="color:{color};font-size:1.4rem;font-weight:700;margin:6px 0;">
    {zone_data["peak_score"]}
    <span style="font-size:0.8rem;">{zone_data["risk_level"]}</span>
  </div>
  <div style="color:#A0A0B0;font-size:0.85rem;">
    Deploy {zone_data["recommended_riders"]} additional riders by {peak_label}
  </div>
  <div style="color:#A0A0B0;font-size:0.8rem;margin-top:4px;border-top:1px solid #333;padding-top:4px;">
    Source from: {donor_zone}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Zone Intelligence
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    risk_colors = {
        "LOW": "#2ECC71",
        "MEDIUM": "#F39C12",
        "HIGH": "#E67E22",
        "CRITICAL": "#E74C3C",
    }

    st.subheader("🔍 Zone Investigation")
    st.caption(
        "Select a zone to understand its risk profile "
        "and simulate rider interventions"
    )

    selected_zone = st.selectbox(
        "Select zone to investigate",
        options=list(mapped_zones_df.sort_values("breach_risk_score", ascending=False)["zone"]),
        key="investigation_zone_select",
    )

    drill_row = scored_df[scored_df["zone"] == selected_zone].iloc[0]
    dna_row = zone_dna_df[zone_dna_df["zone"] == selected_zone]
    dna = dna_row.iloc[0] if not dna_row.empty else None

    risk_color = str(drill_row["color"])
    score = float(drill_row["breach_risk_score"])
    risk_level = str(drill_row["risk_level"])
    zone_vol = float(drill_row["zone_volatility_score"])

    weather_mult = float(trigger_vector["weather_multiplier"])
    temporal_mult = float(trigger_vector["temporal_multiplier"])
    event_mult = float(trigger_vector["event_multiplier"])
    weather_dev = abs(weather_mult - 1.0)
    temporal_dev = abs(temporal_mult - 1.0)
    event_dev = abs(event_mult - 1.0)

    if max(weather_dev, temporal_dev, event_dev) < 0.15 and zone_vol > 0.6:
        driver = "Zone Structure"
    elif weather_dev >= temporal_dev and weather_dev >= event_dev:
        driver = "Weather"
    elif temporal_dev >= event_dev:
        driver = "Time Pattern"
    else:
        driver = "Event"

    # Row 1 — KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Risk Score", f"{score:.1f}")
    with k2:
        st.metric("Risk Level", risk_level)
    with k3:
        st.metric("Primary Driver", driver)
    with k4:
        if dna is not None:
            rest_count = dna.get("restaurant_count", 20)
            base = min(float(rest_count), 150) / 15
            risk_mult = {
                "LOW": 1.0,
                "MEDIUM": 1.3,
                "HIGH": 1.6,
                "CRITICAL": 2.0,
            }.get(risk_level, 1.0)
            riders_needed = min(math.ceil(base * risk_mult), 50)
            st.metric("Riders Needed", riders_needed)
        else:
            st.metric("Riders Needed", "—")

    # Row 2 — score breakdown + radar
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("**Score Breakdown**")

        for label, value, max_val in [
            ("Zone Volatility", zone_vol, 1.0),
            ("Weather", weather_mult, 3.0),
            ("Temporal", temporal_mult, 3.0),
            ("Event", event_mult, 3.0),
        ]:
            bar_val = min(value / max_val, 1.0)
            bl, bm, br = st.columns([2, 5, 1])
            with bl:
                st.caption(label)
            with bm:
                st.progress(float(bar_val))
            with br:
                st.caption(f"{value:.2f}")

        driver_descriptions = {
            "Zone Structure": (
                f"This zone's structural demand density "
                f"(volatility: {zone_vol:.2f}) creates "
                f"baseline elevated risk independent of "
                f"weather or time."
            ),
            "Weather": (
                f"Current weather ({trigger_vector['weather_description']}) "
                f"is the dominant risk factor at "
                f"{weather_mult:.2f}x multiplier."
            ),
            "Time Pattern": (
                f"Current time pattern "
                f"({trigger_vector['temporal_description']}) "
                f"is driving risk at {temporal_mult:.2f}x."
            ),
            "Event": (
                f"Active calendar event "
                f"({trigger_vector['event_description']}) "
                f"is elevating risk at {event_mult:.2f}x."
            ),
        }
        driver_text = driver_descriptions.get(driver, "")
        st.markdown(
            f'<div style="background:#0F3460;padding:10px 14px;'
            f'border-radius:6px;color:#A0A0B0;font-size:0.85rem;'
            f'margin-top:8px;">'
            f'<b style="color:#EAEAEA;">Why this score?</b><br>'
            f"{driver_text}</div>",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("**Zone DNA Profile**")

        if dna is not None:
            import plotly.graph_objects as go

            feature_labels = [
                "Demand Density",
                "Engagement",
                "Volatility",
                "Affluence",
                "Online %",
            ]
            feature_cols = [
                "restaurant_count_norm",
                "engagement_score_norm",
                "volatility_index_norm",
                "affluence_proxy_norm",
                "online_penetration_norm",
            ]

            values = []
            for col in feature_cols:
                if col in zone_dna_df.columns:
                    values.append(float(dna.get(col, 0)))
                else:
                    values.append(0.0)

            values_closed = values + [values[0]]
            labels_closed = feature_labels + [feature_labels[0]]
            try:
                r = int(risk_color[1:3], 16)
                g = int(risk_color[3:5], 16)
                b = int(risk_color[5:7], 16)
                radar_fill = f"rgba({r},{g},{b},0.2)"
            except (ValueError, TypeError):
                radar_fill = "rgba(231,126,34,0.2)"

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=labels_closed,
                    fill="toself",
                    fillcolor=radar_fill,
                    line=dict(color=risk_color, width=2),
                    name=selected_zone,
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(color="#A0A0B0", size=9),
                        gridcolor="#333",
                    ),
                    angularaxis=dict(
                        tickfont=dict(color="#EAEAEA", size=10),
                        gridcolor="#333",
                    ),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#EAEAEA",
                showlegend=False,
                height=280,
                margin=dict(t=20, b=20, l=40, r=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Zone DNA data unavailable")

    # Row 3 — Rider Simulator
    st.markdown("---")
    st.markdown("**⚡ Rider Pre-Positioning Simulator**")

    sim_c1, sim_c2, sim_c3 = st.columns([2, 2, 3])

    with sim_c1:
        extra_riders = st.slider(
            "Extra riders",
            min_value=1,
            max_value=20,
            value=3,
            key="tab3_rider_slider",
        )

    sim_result = simulate_rider_addition(selected_zone, extra_riders, scored_df)

    with sim_c2:
        delta_val = float(sim_result["new_score"]) - float(sim_result["original_score"])
        st.metric(
            "Score Change",
            f"{sim_result['new_score']:.1f}",
            delta=f"{delta_val:.1f}",
            delta_color="inverse",
        )

    with sim_c3:
        other_zones = mapped_zones_df[mapped_zones_df["zone"] != selected_zone]
        if not other_zones.empty:
            donor = other_zones.loc[other_zones["breach_risk_score"].idxmin()]
            donor_text = (
                f"Source from: <b>{donor['zone']}</b> "
                f"(score: {donor['breach_risk_score']:.1f} "
                f"— {donor['risk_level']})"
            )
        else:
            donor_text = "Consider temporary cross-zone reallocation"

        st.markdown(
            f'<div style="background:#16213E;border-radius:8px;padding:12px 16px;">'
            f'<div style="color:#EAEAEA;font-weight:600;">{sim_result["recommendation"]}</div>'
            f'<div style="color:#A0A0B0;font-size:0.85rem;margin-top:6px;">{donor_text}</div>'
            f'<div style="color:#555;font-size:0.75rem;margin-top:8px;">'
            f"Linear model: each rider reduces breach risk score by 2.5 points (transparent, explainable)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ SurgeGuard")
st.sidebar.markdown("**Bangalore Ops Command Layer**")
st.sidebar.markdown("---")

# BUG 5 FIX — contextual labels instead of raw multipliers
wx = float(trigger_vector["weather_multiplier"])
tx = float(trigger_vector["temporal_multiplier"])
ex = float(trigger_vector["event_multiplier"])
cx = float(trigger_vector["combined_multiplier"])

weather_label_sidebar  = f"{_weather_icon(weather_desc)} {weather_desc} — {wx:.2f}x"
temporal_label_sidebar = f"⏰ {temporal_desc} — {tx:.2f}x"
event_label_sidebar    = f"📅 {event_desc} — {ex:.2f}x"
combined_label_sidebar = f"⚡ Combined: {cx:.2f}x"

st.sidebar.markdown("**Live Signal Multipliers**")
st.sidebar.markdown(
    f"""
| Signal | Status |
|--------|--------|
| **Weather** | {weather_label_sidebar} |
| **Temporal** | {temporal_label_sidebar} |
| **Event** | {event_label_sidebar} |
| **Combined** | {combined_label_sidebar} |
""",
    unsafe_allow_html=False,
)
st.sidebar.caption("< 1.0 = suppressed demand | > 1.0 = elevated risk")

if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.write(
    "SurgeGuard is an early warning dashboard for SLA breach risk in Bangalore quick "
    "commerce. It combines static zone demand DNA with live weather/time/event "
    "signals to support proactive rider positioning."
)
