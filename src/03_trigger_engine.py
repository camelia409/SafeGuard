"""
Real-time trigger engine for surge risk scoring.

Install dependencies:
    pip install python-dotenv requests
"""

from __future__ import annotations

from datetime import datetime, date, timedelta
try:
    from zoneinfo import ZoneInfo
    _tz = ZoneInfo("Asia/Kolkata")
except ImportError:
    import pytz as _pytz  # type: ignore
    _tz = _pytz.timezone("Asia/Kolkata")
import os
from typing import Any

import requests
from dotenv import load_dotenv


# Hardcoded high-demand calendar (India-focused) used by risk operations.
# Key format: YYYY-MM-DD, value: (multiplier, description)
EVENT_MULTIPLIERS: dict[str, tuple[float, str]] = {
    # 2.5: Diwali drives one of the sharpest annual home-order spikes.
    "2024-10-31": (2.5, "Diwali"),
    "2025-10-20": (2.5, "Diwali"),
    # 2.2: New Year's Eve has concentrated late-evening order bursts.
    "2024-12-31": (2.2, "New Year's Eve"),
    "2025-12-31": (2.2, "New Year's Eve"),
    # 1.6: Valentine's Day increases premium meal and dessert ordering.
    "2025-02-14": (1.6, "Valentine's Day"),
    # 1.8: Holi gatherings increase group ordering and snack demand.
    "2025-03-14": (1.8, "Holi"),
    # 1.4: Independence Day increases at-home meal ordering modestly.
    "2024-08-15": (1.4, "Independence Day"),
    "2025-08-15": (1.4, "Independence Day"),
    # 1.9: Christmas Eve has strong celebration-led delivery demand.
    "2024-12-24": (1.9, "Christmas Eve"),
    "2025-12-24": (1.9, "Christmas Eve"),
}


def _build_event_calendar() -> dict[str, tuple[float, str]]:
    """Extend the base event dictionary with IPL match-night seasonal effects."""
    calendar = dict(EVENT_MULTIPLIERS)

    # 1.8: IPL nights increase parallel snack/meal ordering during live matches.
    current = date(2025, 4, 1)
    end = date(2025, 5, 31)
    while current <= end:
        calendar[current.isoformat()] = (1.8, "IPL Match Night")
        current += timedelta(days=1)

    return calendar


def get_weather_multiplier(api_key: str) -> tuple[float, str, float]:
    """
    Fetch current Bangalore weather and convert to a demand multiplier.

    Returns:
        (weather_multiplier, weather_description, bangalore_temp_c)
    """
    if not api_key or api_key == "paste_your_new_key_here":
        return 1.0, "API Unavailable", 0.0

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": "Bangalore,IN", "appid": api_key, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        weather_data: dict[str, Any] = response.json()
        print("\n=== Raw Weather API Data ===")
        print(weather_data)
    except requests.RequestException:
        return 1.0, "API Unavailable", 0.0
    except ValueError:
        return 1.0, "API Unavailable", 0.0

    weather_main = (
        str(weather_data.get("weather", [{}])[0].get("main", "")).strip().title()
    )
    weather_desc = (
        str(weather_data.get("weather", [{}])[0].get("description", "")).strip().title()
    )
    clouds = float(weather_data.get("clouds", {}).get("all", 0) or 0)
    temp_c = float(weather_data.get("main", {}).get("temp", 0.0) or 0.0)
    rain_1h = weather_data.get("rain", {}).get("1h", None)

    # 3.0: heavy rain causes severe rider slowdown and sharp surge behavior.
    if weather_main == "Rain" and rain_1h is not None and float(rain_1h) > 10:
        return 3.0, f"Heavy Rain ({weather_desc or 'Rain'})", temp_c

    # 2.4: moderate rain materially disrupts supply and increases order spikes.
    if weather_main == "Rain" and rain_1h is not None and 2 <= float(rain_1h) <= 10:
        return 2.4, f"Moderate Rain ({weather_desc or 'Rain'})", temp_c

    # 1.8: light rain drives indoor preference and moderate demand uplift.
    if weather_main == "Rain" and rain_1h is not None and float(rain_1h) < 2:
        return 1.8, f"Light Rain ({weather_desc or 'Rain'})", temp_c

    # 1.8: rain without intensity data is treated as light-rain default uplift.
    if weather_main == "Rain":
        return 1.8, f"Rain ({weather_desc or 'No Intensity Data'})", temp_c

    # 1.2: cloudy dry weather nudges users toward ordering in.
    if weather_main != "Rain" and clouds >= 50:
        return 1.2, f"Cloudy ({weather_desc or 'Overcast'})", temp_c

    # 1.0: clear/no-rain baseline with no exceptional delivery pressure.
    return 1.0, f"Clear/Baseline ({weather_desc or 'No Rain'})", temp_c


def _get_temporal_for_hour(hour: int, weekday: int) -> tuple[float, str]:
    """
    Shared internal function — maps (hour, weekday) to (multiplier, description).

    Used by get_temporal_multiplier() for live scoring and by the forecast
    engine (7B) to generate hourly risk projections without calling datetime.

    Args:
        hour:    0-23 representing the hour of day (Bangalore IST)
        weekday: 0=Monday … 6=Sunday

    Returns:
        (multiplier, description)
    """
    is_weekend = weekday >= 5

    # 0.7: late-night demand exists but volume is generally lower and patchy.
    if hour >= 23 or hour < 6:
        return 0.7, "Late Night"

    # 0.8: mornings are active but usually below meal-rush intensity.
    if 6 <= hour < 12:
        return 0.8, "Morning Hours"

    if not is_weekend:
        # 1.4: weekday lunch extends into late lunch through 3 PM.
        if 12 <= hour < 15:
            return 1.4, "Weekday Lunch"
        # 1.2: weekday late-afternoon/early-evening uplift.
        if 15 <= hour < 19:
            return 1.2, "Weekday Evening"
        # 1.6: weekday dinner is a high-demand prime ordering period.
        if 19 <= hour < 23:
            return 1.6, "Weekday Dinner Rush"
    else:
        # 1.8: weekend lunch starts earlier and extends to mid-afternoon.
        if 11 <= hour < 15:
            return 1.8, "Weekend Lunch Rush"
        # 2.0: weekend dinner starts earlier with stronger evening demand.
        if 18 <= hour < 23:
            return 2.0, "Weekend Dinner Peak"

    # 0.6: non-peak hours form the low-demand baseline outside rush periods.
    return 0.6, "Off-Peak Hours"


def get_temporal_multiplier() -> tuple[float, str]:
    """Map Bangalore local time windows to delivery demand multipliers."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    except ImportError:
        try:
            import pytz
            now = datetime.now(pytz.timezone("Asia/Kolkata"))
        except ImportError:
            # Last resort: add 5.5 hours to UTC manually.
            from datetime import timezone, timedelta
            utc_now = datetime.now(timezone.utc)
            now = utc_now + timedelta(hours=5, minutes=30)

    hour = now.hour
    weekday = now.weekday()
    print(f"[TriggerEngine] Bangalore time: {now}, hour={hour}, weekday={weekday}")
    return _get_temporal_for_hour(hour, weekday)


def get_12hour_forecast(
    weather_multiplier: float,
    event_multiplier: float,
) -> list[dict[str, Any]]:
    """Build a 12-hour forward multiplier forecast in Bangalore local time."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    except ImportError:
        import pytz
        now = datetime.now(pytz.timezone("Asia/Kolkata"))

    forecasts: list[dict[str, Any]] = []
    for i in range(1, 13):
        future_time = now + timedelta(hours=i)
        hour = future_time.hour
        weekday = future_time.weekday()
        temporal_mult, temporal_desc = _get_temporal_for_hour(hour, weekday)
        combined = min(temporal_mult * weather_multiplier * event_multiplier, 5.0)
        forecasts.append(
            {
                "hour_label": future_time.strftime("%I %p"),
                "hour_24": hour,
                "datetime": future_time,
                "temporal_multiplier": temporal_mult,
                "weather_multiplier": weather_multiplier,
                "event_multiplier": event_multiplier,
                "combined_multiplier": combined,
                "temporal_description": temporal_desc,
                "is_peak": temporal_mult >= 1.6,
            }
        )
    return forecasts


def get_risk_level_from_score(score: float) -> str:
    """Map a projected 0-100 score to categorical risk bands."""
    if score <= 25:
        return "LOW"
    if score <= 50:
        return "MEDIUM"
    if score <= 75:
        return "HIGH"
    return "CRITICAL"


def get_event_multiplier() -> tuple[float, str]:
    """Return event-day multiplier for today's date in Bangalore."""
    try:
        from zoneinfo import ZoneInfo
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    except ImportError:
        import pytz
        now_ist = datetime.now(pytz.timezone("Asia/Kolkata"))
    today_key = now_ist.date().isoformat()
    calendar = _build_event_calendar()

    if today_key in calendar:
        return calendar[today_key]

    # 1.0: normal day baseline when no known high-demand event is present.
    return 1.0, "Normal Day"


def get_trigger_vector() -> dict[str, Any]:
    """Compute real-time trigger multipliers and combined score signal."""
    load_dotenv()
    api_key = os.getenv("OPENWEATHER_API_KEY", "")

    weather_multiplier, weather_description, bangalore_temp_c = get_weather_multiplier(
        api_key
    )
    temporal_multiplier, temporal_description = get_temporal_multiplier()
    event_multiplier, event_description = get_event_multiplier()

    combined_multiplier = weather_multiplier * temporal_multiplier * event_multiplier
    combined_multiplier = min(combined_multiplier, 5.0)

    try:
        from zoneinfo import ZoneInfo as _ZI
        _ts = datetime.now(_ZI("Asia/Kolkata")).isoformat()
    except ImportError:
        import pytz as _tz2
        _ts = datetime.now(_tz2.timezone("Asia/Kolkata")).isoformat()

    return {
        "weather_multiplier": float(weather_multiplier),
        "temporal_multiplier": float(temporal_multiplier),
        "event_multiplier": float(event_multiplier),
        "combined_multiplier": float(combined_multiplier),
        "weather_description": str(weather_description),
        "temporal_description": str(temporal_description),
        "event_description": str(event_description),
        "timestamp": _ts,
        "bangalore_temp_c": float(bangalore_temp_c),
    }


if __name__ == "__main__":
    vector = get_trigger_vector()
    print("\n=== Trigger Vector Report ===")
    print(f"Timestamp             : {vector['timestamp']}")
    print(
        f"Weather               : {vector['weather_description']} "
        f"(x{vector['weather_multiplier']:.2f})"
    )
    print(f"Bangalore Temperature : {vector['bangalore_temp_c']:.2f} C")
    print(
        f"Temporal              : {vector['temporal_description']} "
        f"(x{vector['temporal_multiplier']:.2f})"
    )
    print(
        f"Event                 : {vector['event_description']} "
        f"(x{vector['event_multiplier']:.2f})"
    )
    print("----------------------------------------")
    print(f"Combined Multiplier   : x{vector['combined_multiplier']:.2f}")
    print("\nTrigger Drivers:")
    print(f"- Weather driver : {vector['weather_description']}")
    print(f"- Time driver    : {vector['temporal_description']}")
    print(f"- Event driver   : {vector['event_description']}")
