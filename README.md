# SurgeGuard — Hyperlocal SLA Breach Early Warning System

A real-time decision-support tool for last-mile quick 
commerce operations in Bangalore. SurgeGuard predicts 
zone-level SLA breach risk 12+ hours ahead by combining 
static demand DNA (derived from 56,000+ Zomato restaurant 
records) with live trigger signals (weather, time-of-day, 
calendar events), and surfaces actionable rider 
pre-positioning recommendations before breaches occur.

**Live demo:** https://safeguard-ihtrpqqnajzhgcwpqglpkp.streamlit.app
**Built by:** Abinandida R  
**Stack:** Python · Streamlit · SQLite · Folium · 
           Plotly · OpenWeatherMap API

---

## The Problem

Quick commerce companies operating 10-minute delivery SLAs 
face a structural gap: demand surge detection happens 
reactively, after the breach has already occurred. 
The average detection-to-response latency is 6-10 minutes — 
longer than the entire delivery promise.

The root cause is not insufficient riders or dark stores. 
It is **decision latency** — operations teams lack a 
forward-looking signal to act before the surge materialises.

SurgeGuard reduces this latency by generating zone-level 
breach risk scores and a 12-hour forecast, giving operations 
teams actionable pre-positioning recommendations before 
the first order is delayed.

---

## Architecture
Stage 1: Zone DNA (Static — runs once)
56,000+ Zomato restaurant records
→ SQL feature engineering (5 features per zone)
→ Min-max normalisation → zone_volatility_score
→ Stored in SQLite
Stage 2: Trigger Engine (Live — every 15 minutes)
OpenWeatherMap API → weather_multiplier (1.0x – 3.0x)
Bangalore IST clock → temporal_multiplier (0.6x – 2.0x)
Indian calendar → event_multiplier (1.0x – 2.5x)
Combined multiplier capped at 5.0x
Stage 3: Risk Scorer
breach_risk_score = zone_volatility_score
× combined_multiplier × 100
Bands: LOW (0-25) | MEDIUM (26-50)
HIGH (51-75) | CRITICAL (76-100)
Stage 4: 12-Hour Forecast
Same formula applied to future time windows
using known temporal multiplier curve
Stage 5: Dashboard (3 tabs)
Tab 1 — Live Operations: risk map + zone rankings
Tab 2 — Risk Forecast: trajectory chart + heatmap
Tab 3 — Zone Intelligence: drill-down + simulator

---

## Zone DNA Features

Five features engineered via SQL on 56,000+ 
Zomato restaurant records, aggregated to 21 
canonical Bangalore delivery zones:

| Feature | Proxy for | SQL operation |
|---|---|---|
| restaurant_count | Demand density | COUNT(*) per zone |
| engagement_score | Active delivery demand | AVG(votes) WHERE online_order=1 |
| volatility_index | Demand spike tendency | Quick Bites share per zone |
| affluence_proxy | Consumer spending power | AVG(approx_cost) per zone |
| online_penetration | Delivery dependency | SUM(online_order)/COUNT(*) |

Weighted composite score:
zone_volatility_score =
0.25 × restaurant_count_norm

0.30 × engagement_score_norm   ← highest weight:
0.25 × volatility_index_norm     strongest delivery proxy
0.10 × affluence_proxy_norm
0.10 × online_penetration_norm


---

## Trigger Multipliers

**Weather** (OpenWeatherMap API):
- Heavy rain > 10mm/h → 3.0x
- Moderate rain 2-10mm/h → 2.4x  
- Light rain < 2mm/h → 1.8x
- Cloudy → 1.2x
- Clear → 1.0x

**Temporal** (Bangalore IST):
- Weekend Dinner 18:00-22:59 → 2.0x
- Weekend Lunch 11:00-14:59 → 1.8x
- Weekday Dinner 19:00-22:59 → 1.6x
- Weekday Lunch 12:00-14:59 → 1.4x
- Weekday Evening 15:00-18:59 → 1.2x
- Morning → 0.8x | Late Night → 0.7x

**Event** (hardcoded Indian calendar):
- Diwali → 2.5x | New Year Eve → 2.2x
- Christmas Eve → 1.9x | IPL nights → 1.8x

---

## Design Decisions

**Why transparent scoring over ML models:**  
Operations managers make real-time staffing decisions 
based on this system. A black-box model that cannot 
explain why a zone is HIGH risk will not be acted upon. 
Every risk score is traceable to a single SQL query 
or multiplier value.

**Why linear rider simulation over Poisson:**  
The Poisson arrival model, while statistically correct, 
returns near-0% probability at low-medium risk scores 
making it operationally useless for pre-positioning 
decisions at calm periods. The linear model 
(2.5 points per rider) shows consistent, explainable 
impact at all risk levels.

**Why synthetic temporal patterns:**  
Real Zepto order timestamps are proprietary. 
Temporal multipliers are based on published food 
delivery demand patterns. The model's forecast 
accuracy is validated structurally: our top-ranked 
zones (Whitefield, Koramangala, Indiranagar, HSR Layout) 
align exactly with known Zepto dark store locations in 
Bangalore — confirming the Zone DNA captures real 
deployment signal.

**Why Zomato restaurant data proxies delivery demand:**  
Restaurant density + engagement (votes) in a zone 
is a legitimate proxy for food delivery demand. 
High-voted online-ordering restaurants = established 
delivery demand patterns. This is the same logic 
dark store site selection teams use internally.

---

## Assumptions and Limitations

**Data assumptions:**
- Restaurant density proxies food delivery demand 
  (directional, not exact)
- Zomato votes proxy cumulative order engagement
- Zone boundaries follow Zomato location tags, 
  not official Bangalore administrative zones

**Model assumptions:**
- Each extra rider reduces breach risk by 2.5 points 
  (linear approximation — real relationship is non-linear)
- Base rider need = restaurant_count / 15 
  (industry rule of thumb, capped at 150 restaurants)
- Weather held constant over 12-hour forecast window 
  (no weather forecast API in V1)

**Known limitations:**
- No cold-start handling for zones with fewer than 
  10 restaurants
- No traffic signal integration
- No competitor promotion detection
- Forecast reliability degrades beyond 6 hours 
  as weather assumption weakens
- Not validated against actual SLA breach data — 
  validated structurally against known Zepto 
  dark store locations only

---

## Dark Store Location Validation

Our zone risk rankings align with known Zepto 
operational presence in Bangalore:

| Our Risk Rank | Zone | Known Zepto Presence |
|---|---|---|
| 1 | Whitefield | ✅ Confirmed |
| 2 | Koramangala | ✅ Confirmed |
| 3 | Indiranagar | ✅ Confirmed |
| 4 | MG Road | ✅ Confirmed |
| 5 | HSR Layout | ✅ Confirmed |

*Source: Public Zepto app store listings and 
media coverage — not proprietary data.*

This confirms the Zone DNA feature engineering 
captures real demand signal without access to 
proprietary order data.

---

## Setup

```bash
# 1. Clone and install
git clone <repo-url>
cd SurgeGuard
pip install streamlit streamlit-folium \
    streamlit-autorefresh folium pandas \
    plotly requests python-dotenv

# 2. Add OpenWeatherMap API key
echo "OPENWEATHER_API_KEY=your_key_here" > .env

# 3. Build the data pipeline (run once)
python src/01_clean_data.py
python src/02_zone_dna.py

# 4. Launch dashboard
streamlit run app/dashboard.py
```

**Note:** `data/zomato.csv` (574MB) is not included 
in the repository. Download from:  
[Kaggle — Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)

---

## Repository Structure
SurgeGuard/
├── app/
│   └── dashboard.py          # Streamlit dashboard
├── src/
│   ├── 01_clean_data.py      # Data cleaning pipeline
│   ├── 02_zone_dna.py        # Zone DNA feature engineering
│   ├── 03_trigger_engine.py  # Live trigger multipliers
│   └── 04_risk_scorer.py     # Risk scoring + simulator
├── sql/
│   └── zone_features.sql     # SQL queries (auditable)
├── data/
│   └── surgeguard.db         # SQLite database
└── .env                      # API key (gitignored)

---

## Future Work

- Poisson arrival model for simulator 
  (statistically rigorous breach probability)
- City-wide rider reallocation optimiser 
  (multi-zone coordinated dispatch)
- Play Store review correlation validation 
  (complaint timestamps vs. predicted risk windows)
- FastAPI wrapper for production deployment
- Multi-city generalisation beyond Bangalore
