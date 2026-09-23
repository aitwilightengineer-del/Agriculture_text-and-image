"""
Location-based farm insights API for the mobile app.

Two ways to pick a location:
  1. Current location - the app sends the phone's GPS latitude/longitude.
  2. Custom location  - the user searches a place (GET /locations/search)
                        and the app sends the chosen place name or its lat/lon.

POST /farm-insights returns, for that location:
  climate, soil health, water availability (dams), seeds that can be
  cultivated, fertilizer, government schemes and subsidies.

GET /schemes/detail returns eligibility, description and documents for one subsidy.

Data sources:
  - Geocoding   : OpenStreetMap Nominatim
  - Climate     : Open-Meteo (live weather + forecast)
  - Soil        : ISRIC SoilGrids values + Gemini soil-health interpretation
  - Water, seeds, fertilizer, schemes, subsidies :
                  Tamil Nadu Agriculture Department portals (see tn_agri.py)
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import requests
from flask import Blueprint, request, jsonify

import tn_agri


# ============================================================
# Configuration
# ============================================================
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"   ##Paid
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Nominatim requires an identifying User-Agent (max 1 request/second)
HTTP_HEADERS = {"User-Agent": "agri-advisory-app/1.0 (farm-insights API)"}

# Restrict place search to India (water, seeds, fertilizer and schemes cover Tamil Nadu)
COUNTRY_CODES = "in"

ALL_SECTIONS = ["climate", "soil", "water", "seeds", "fertilizer", "schemes", "subsidies"]
SUPPORTED_LANGUAGES = {"en": "English", "ta": "Tamil"}

# Cache lifetimes (seconds)
GEOCODE_TTL = 7 * 24 * 3600
CLIMATE_TTL = 30 * 60
SOIL_TTL = 30 * 24 * 3600

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Heavy showers",
    85: "Snow showers", 86: "Snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}


# ============================================================
# TTL cache - shared with tn_agri so /cache/refresh clears everything
# ============================================================
cache_get = tn_agri.cache_lookup
cache_set = tn_agri.cache_store


class LocationError(Exception):
    """Raised when a location cannot be resolved (returned as HTTP 404)."""


# ============================================================
# Location: search, geocode, reverse geocode
# ============================================================
def _format_place(item):
    """Convert a Nominatim result into the location object used by the API."""
    address = item.get("address", {})

    district = (
        address.get("state_district") or
        address.get("county") or
        address.get("city_district") or
        address.get("city") or
        ""
    )
    district = re.sub(r"\s+district$", "", district, flags=re.IGNORECASE).strip()

    name = (
        address.get("village") or
        address.get("hamlet") or
        address.get("suburb") or
        address.get("town") or
        address.get("city") or
        address.get("municipality") or
        district or
        item.get("name", "")
    )

    return {
        "name": name,
        "district": district,
        "state": address.get("state", ""),
        "country": address.get("country", ""),
        "postcode": address.get("postcode", ""),
        "display_name": item.get("display_name", ""),
        "latitude": round(float(item["lat"]), 6),
        "longitude": round(float(item["lon"]), 6),
    }


def search_places(query_text, limit=5):
    key = ("search", query_text.lower(), limit)
    cached = cache_get(key)
    if cached is not None:
        return cached

    response = requests.get(
        f"{NOMINATIM_URL}/search",
        params={
            "q": query_text,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": limit,
            "countrycodes": COUNTRY_CODES,
        },
        headers=HTTP_HEADERS,
        timeout=15,
    )
    response.raise_for_status()

    return cache_set(key, [_format_place(item) for item in response.json()], GEOCODE_TTL)


def reverse_geocode(lat, lon):
    key = ("reverse", round(lat, 3), round(lon, 3))
    cached = cache_get(key)
    if cached is not None:
        return cached

    response = requests.get(
        f"{NOMINATIM_URL}/reverse",
        params={"lat": lat, "lon": lon, "format": "jsonv2", "addressdetails": 1, "zoom": 14},
        headers=HTTP_HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    item = response.json()

    if "error" in item:
        raise LocationError(f"No address found for coordinates {lat}, {lon}.")

    place = _format_place(item)
    # Keep the exact coordinates the phone sent
    place["latitude"] = round(lat, 6)
    place["longitude"] = round(lon, 6)
    return cache_set(key, place, GEOCODE_TTL)


def resolve_location(data):
    """
    Resolve the request into a location object.
    Uses latitude/longitude when given (GPS or a picked search result),
    otherwise geocodes the 'location' text.
    """
    location_type = (data.get("location_type") or "").strip().lower()
    lat, lon = data.get("latitude"), data.get("longitude")

    if lat is not None and lon is not None:
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            raise ValueError("'latitude' and 'longitude' must be numbers.")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("'latitude' must be -90..90 and 'longitude' -180..180.")

        place = dict(reverse_geocode(lat, lon))
        place["type"] = location_type or "current"
        return place

    place_text = (data.get("location") or "").strip()
    if not place_text:
        raise ValueError(
            "Send 'latitude' and 'longitude' (current location) "
            "or 'location' (custom place name)."
        )

    results = search_places(place_text, limit=1)
    if not results:
        raise LocationError(f"Location '{place_text}' was not found.")

    place = dict(results[0])
    place["type"] = location_type or "custom"
    return place


# ============================================================
# Climate (Open-Meteo)
# ============================================================
def get_farming_season(today=None):
    """Indian cropping season for the given date."""
    month = (today or date.today()).month
    if 6 <= month <= 10:
        return "Kharif"
    if month >= 11 or month <= 2:
        return "Rabi"
    return "Zaid"


def get_climate(lat, lon):
    key = ("climate", round(lat, 2), round(lon, 2))
    cached = cache_get(key)
    if cached is not None:
        return cached

    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,"
                       "wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                     "precipitation_probability_max,weather_code",
            "past_days": 30,
            "forecast_days": 7,
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    current = data.get("current", {})
    daily = data.get("daily", {})
    today = current.get("time", "")[:10] or date.today().isoformat()

    forecast = []
    rainfall_last_30_days = 0.0

    for i, day in enumerate(daily.get("time", [])):
        rain = daily["precipitation_sum"][i] or 0.0

        if day < today:
            rainfall_last_30_days += rain
            continue

        forecast.append({
            "date": day,
            "temp_max_c": daily["temperature_2m_max"][i],
            "temp_min_c": daily["temperature_2m_min"][i],
            "rain_mm": rain,
            "rain_probability_percent": daily["precipitation_probability_max"][i],
            "condition": WEATHER_CODES.get(daily["weather_code"][i], "Unknown"),
        })

    climate = {
        "source": "Open-Meteo",
        "season": get_farming_season(date.fromisoformat(today)),
        "current": {
            "time": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "condition": WEATHER_CODES.get(current.get("weather_code"), "Unknown"),
        },
        "rainfall_last_30_days_mm": round(rainfall_last_30_days, 1),
        "rainfall_next_7_days_mm": round(sum(d["rain_mm"] for d in forecast), 1),
        "forecast": forecast,
    }

    return cache_set(key, climate, CLIMATE_TTL)


# ============================================================
# Soil (ISRIC SoilGrids)
# ============================================================
SOIL_PROPERTIES = {
    # SoilGrids name: (API field, unit)
    "phh2o": ("ph", ""),
    "soc": ("organic_carbon_g_per_kg", "g/kg"),
    "nitrogen": ("nitrogen_g_per_kg", "g/kg"),
    "clay": ("clay_percent", "%"),
    "sand": ("sand_percent", "%"),
    "silt": ("silt_percent", "%"),
    "cec": ("cec_cmol_per_kg", "cmol(c)/kg"),
}


def soil_texture_class(sand, silt, clay):
    """USDA soil texture class from sand/silt/clay percentages."""
    if None in (sand, silt, clay):
        return None
    if silt + 1.5 * clay < 15:
        return "Sand"
    if silt + 2 * clay < 30:
        return "Loamy sand"
    if clay >= 40 and silt >= 40:
        return "Silty clay"
    if clay >= 40 and sand <= 45:
        return "Clay"
    if clay >= 35 and sand > 45:
        return "Sandy clay"
    if clay >= 27 and sand <= 20:
        return "Silty clay loam"
    if clay >= 27:
        return "Clay loam"
    if clay >= 20 and silt < 28 and sand > 45:
        return "Sandy clay loam"
    if silt >= 80 and clay < 12:
        return "Silt"
    if silt >= 50:
        return "Silt loam"
    if clay >= 7 and silt >= 28 and sand <= 52:
        return "Loam"
    return "Sandy loam"


def _query_soilgrids(lat, lon):
    """Topsoil (0-15 cm) averages for one point, or None if SoilGrids has no data there."""
    params = [("lon", lon), ("lat", lat), ("value", "mean")]
    params += [("property", p) for p in SOIL_PROPERTIES]
    params += [("depth", "0-5cm"), ("depth", "5-15cm")]

    response = requests.get(SOILGRIDS_URL, params=params, timeout=25)
    response.raise_for_status()

    values = {}
    for layer in response.json().get("properties", {}).get("layers", []):
        field = SOIL_PROPERTIES.get(layer.get("name"), (None,))[0]
        if not field:
            continue
        d_factor = layer.get("unit_measure", {}).get("d_factor", 1) or 1
        means = [
            depth["values"]["mean"] / d_factor
            for depth in layer.get("depths", [])
            if depth.get("values", {}).get("mean") is not None
        ]
        values[field] = round(sum(means) / len(means), 2) if means else None

    return values if any(v is not None for v in values.values()) else None


def get_soil(lat, lon):
    key = ("soil", round(lat, 2), round(lon, 2))
    cached = cache_get(key)
    if cached is not None:
        return cached

    # SoilGrids has no data for built-up areas and water bodies,
    # so fall back to farmland points ~2-4 km around the location.
    candidates = [(lat, lon)] + [
        (lat + dlat, lon + dlon)
        for step in (0.02, 0.04)
        for dlat, dlon in ((step, 0), (-step, 0), (0, step), (0, -step))
    ]

    values, sampled, last_error = None, None, None
    for point in candidates:
        try:
            values = _query_soilgrids(*point)
        except requests.RequestException as e:
            last_error = e
            # Service down or timing out: don't try the other points
            if not isinstance(e, requests.HTTPError) or e.response.status_code >= 500:
                break
            continue
        if values:
            sampled = point
            break

    if not values:
        if last_error:
            raise RuntimeError(f"SoilGrids unavailable: {last_error}")
        raise RuntimeError("No soil data available near this location.")

    soil = {
        "source": "ISRIC SoilGrids (topsoil 0-15 cm)",
        **values,
        "texture": soil_texture_class(
            values.get("sand_percent"), values.get("silt_percent"), values.get("clay_percent")
        ),
        "sampled_at": {"latitude": round(sampled[0], 4), "longitude": round(sampled[1], 4)},
    }

    return cache_set(key, soil, SOIL_TTL)


# ============================================================
# Soil health (Gemini interpretation of the SoilGrids values)
# ============================================================
def parse_json_response(text):
    """Extract the JSON object from an LLM reply (handles ``` fences and extra text)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM reply did not contain a JSON object.")
    return json.loads(text[start:end + 1])


def _place_label(place):
    parts = [place.get("name"), place.get("district"), place.get("state"), place.get("country")]
    seen, label = set(), []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            label.append(part)
    return ", ".join(label)


def get_soil_health(query_llm, place, soil, language):
    """Soil health status, issues and improvements from the measured soil values."""
    key = ("soil_health", round(place["latitude"], 2), round(place["longitude"], 2), bool(soil), language)
    cached = cache_get(key)
    if cached is not None:
        return cached

    soil_text = "Not available - describe the typical soil of this district."
    if soil:
        soil_text = json.dumps({k: v for k, v in soil.items() if k not in ("source", "sampled_at")})

    prompt = f"""
You are a soil scientist advising farmers in India.

Location: {_place_label(place)}
Topsoil data (ISRIC SoilGrids, 0-15 cm): {soil_text}

Assess the soil health for farming. Return ONLY a JSON object:
{{
  "status": "Good | Moderate | Poor",
  "summary": "2-3 sentences on fertility, pH and organic matter",
  "ph_status": "Acidic | Neutral | Alkaline",
  "organic_matter_status": "Low | Medium | High",
  "issues": ["..."],
  "improvements": ["..."]
}}

Write every text value in {SUPPORTED_LANGUAGES[language]}, except the status words,
which must stay exactly as listed in English. Keep the JSON keys in English.
"""

    health = parse_json_response(query_llm(prompt, json_output=True))
    return cache_set(key, health, SOIL_TTL)


# ============================================================
# Section loading (shared by /farm-insights and /chat)
# ============================================================
_executor = ThreadPoolExecutor(max_workers=8)


def attach_tn_district(place):
    """Match the place to a Tamil Nadu district (None outside Tamil Nadu)."""
    district = tn_agri.find_district(
        place.get("district"), place.get("name"), *place.get("display_name", "").split(",")
    ) if place.get("state", "").lower() == "tamil nadu" else None
    place["tn_district"] = district["name"] if district else None
    return district


def build_insights(query_llm, place, sections, language):
    """
    Fetch the requested sections for a resolved place.
    Returns ({section: data}, {section: error message}).
    """
    lat, lon = place["latitude"], place["longitude"]
    district = attach_tn_district(place)
    executor = _executor

    # --------------------------------------------------------
    # 1. Fetch every requested section in parallel
    # --------------------------------------------------------
    tasks = {}
    if "climate" in sections or "water" in sections:
        tasks["climate"] = executor.submit(get_climate, lat, lon)
    if "soil" in sections:
        tasks["soil"] = executor.submit(get_soil, lat, lon)

    tn_tasks = {
        "water": lambda: tn_agri.get_water_for_location(lat, lon, district["name"]),
        "seeds": lambda: tn_agri.get_seeds_for_location(lat, lon, district, language),
        "fertilizer": lambda: tn_agri.get_fertilizer_for_location(lat, lon, district, language),
    }
    errors = {}
    for name, fn in tn_tasks.items():
        if name in sections:
            if district:
                tasks[name] = executor.submit(fn)
            else:
                errors[name] = TN_ONLY_MESSAGE

    if sections & {"schemes", "subsidies"}:
        tasks["schemes"] = executor.submit(tn_agri.get_schemes_and_subsidies, language)

    results = {}
    for name, future in tasks.items():
        try:
            results[name] = future.result()
        except Exception as e:
            print(f"farm-insights: {name} failed: {e}", flush=True)
            results[name] = None
            errors[name] = str(e)

    # --------------------------------------------------------
    # 2. Shape each section
    # --------------------------------------------------------
    result = {}

    climate = results.get("climate")
    if "climate" in sections:
        result["climate"] = climate
    elif "climate" in errors:
        errors.pop("climate")  # fetched only for the water section

    if "soil" in sections:
        soil = results.get("soil")
        try:
            health = get_soil_health(query_llm, place, soil, language)
        except Exception as e:
            health = None
            errors["soil_health"] = f"Soil health analysis failed: {e}"
        result["soil"] = None
        if soil or health:
            result["soil"] = dict(soil) if soil else {
                "source": None,
                "note": "Measured soil data unavailable; health is estimated for the district.",
            }
            result["soil"]["health"] = health
            result["soil"]["health_source"] = "Gemini (AI interpretation)" if health else None

    if "water" in sections:
        water = results.get("water")
        if water and climate:
            water["rainfall_last_30_days_mm"] = climate["rainfall_last_30_days_mm"]
            water["rainfall_next_7_days_mm"] = climate["rainfall_next_7_days_mm"]
        result["water"] = water

    for name in ("seeds", "fertilizer"):
        if name in sections:
            result[name] = results.get(name)

    schemes = results.get("schemes")
    if "schemes" in sections:
        result["schemes"] = {"source": schemes["source"], "items": schemes["schemes"]} if schemes else None
    if "subsidies" in sections:
        result["subsidies"] = {
            "source": schemes["source"],
            "count": len(schemes["subsidies"]),
            "items": schemes["subsidies"],
        } if schemes else None
    if schemes is None and "schemes" in errors and "subsidies" in sections:
        errors["subsidies"] = errors["schemes"]
        if "schemes" not in sections:
            errors.pop("schemes")

    return result, errors


# ============================================================
# Blueprint
# ============================================================
TN_ONLY_MESSAGE = "Available only for locations in Tamil Nadu (Tamil Nadu Agriculture Department data)."

# Cache entries behind each section (first element of the cache key)
CACHE_SECTIONS = {
    "climate": ["climate"],
    "soil": ["soil", "soil_health"],
    "water": ["reservoirs"],
    "seeds": ["seed_stock", "seed_blocks"],
    "fertilizer": ["fert_dealers", "fert_blocks", "fert_inspector", "fert_prices"],
    "schemes": ["scheme_catalogue", "scheme_detail"],
    "subsidies": ["scheme_catalogue", "scheme_detail"],
    "location": ["search", "reverse"],
}

# Statewide data reloaded right after a refresh
CACHE_RELOADERS = {
    "water": tn_agri.get_reservoirs,
    "fertilizer": tn_agri.get_fertilizer_prices,
    "schemes": tn_agri.get_scheme_catalogue,
    "subsidies": tn_agri.get_scheme_catalogue,
}

# When set, /cache/* requires the header  X-Admin-Token: <value>
CACHE_ADMIN_TOKEN = os.getenv("CACHE_ADMIN_TOKEN", "")


def create_location_blueprint(query_llm):
    """
    Build the Flask blueprint. query_llm is rag.query_llm (supports json_output=).
    """
    bp = Blueprint("location_api", __name__)
    tn_agri.warm_up()

    @bp.route("/locations/search", methods=["GET"])
    def location_search():
        """Place autocomplete for the 'choose your own location' screen."""
        query_text = (request.args.get("q") or "").strip()
        if len(query_text) < 2:
            return jsonify({"error": "Query parameter 'q' must be at least 2 characters."}), 400

        try:
            limit = max(1, min(int(request.args.get("limit", 5)), 10))
        except ValueError:
            limit = 5

        try:
            return jsonify({"results": search_places(query_text, limit)})
        except requests.RequestException as e:
            return jsonify({"error": f"Location search service unavailable: {e}"}), 502

    @bp.route("/schemes/detail", methods=["GET"])
    def scheme_detail():
        """Full details of one subsidy item (eligibility, description, documents)."""
        args = request.args
        required = ["department", "scheme_id", "input_type_id", "class_id"]
        missing = [k for k in required if not (args.get(k) or "").strip()]
        if missing:
            return jsonify({"error": f"Missing query parameters: {missing}"}), 400

        language = (args.get("language") or "en").strip().lower()
        if language not in SUPPORTED_LANGUAGES:
            return jsonify({"error": f"'language' must be one of {list(SUPPORTED_LANGUAGES)}."}), 400

        try:
            detail = tn_agri.get_subsidy_detail(*(args[k].strip() for k in required), language)
        except (requests.RequestException, ValueError) as e:
            return jsonify({"error": f"Scheme service unavailable: {e}"}), 502

        if not detail:
            return jsonify({"error": "Subsidy item not found."}), 404
        return jsonify(detail)

    def admin_denied():
        if CACHE_ADMIN_TOKEN and request.headers.get("X-Admin-Token") != CACHE_ADMIN_TOKEN:
            return jsonify({"error": "Invalid or missing X-Admin-Token header."}), 401
        return None

    @bp.route("/cache/refresh", methods=["POST"])
    def cache_refresh():
        """
        Clear cached data so the next request fetches fresh data.
        Body (optional): {"sections": ["water", "seeds"], "reload": true}
        Without sections, everything is cleared.
        """
        denied = admin_denied()
        if denied:
            return denied

        data = request.get_json(silent=True, force=True) or request.form.to_dict() or request.args.to_dict()

        sections = data.get("sections") or list(CACHE_SECTIONS)
        if isinstance(sections, str):
            sections = [s.strip() for s in sections.split(",")]
        unknown = [s for s in sections if s not in CACHE_SECTIONS]
        if unknown:
            return jsonify({"error": f"Unknown sections {unknown}. Allowed: {list(CACHE_SECTIONS)}."}), 400

        reload = str(data.get("reload", True)).lower() not in ("false", "0", "no")

        prefixes = {p for s in sections for p in CACHE_SECTIONS[s]}
        cleared = tn_agri.clear_cache(prefixes)

        reloading = []
        if reload:
            loaders = list(dict.fromkeys(CACHE_RELOADERS[s] for s in sections if s in CACHE_RELOADERS))
            if loaders:
                tn_agri.warm_up(loaders)
                reloading = [fn.__name__ for fn in loaders]

        return jsonify({
            "refreshed_sections": sections,
            "cleared_entries": cleared,
            "reloading_in_background": reloading,
            "note": "Location-specific data (seeds, fertilizer, soil, climate) is fetched fresh on the next request.",
        })

    @bp.route("/cache/status", methods=["GET"])
    def cache_status():
        """What is cached and when each part next expires."""
        denied = admin_denied()
        if denied:
            return denied

        status = tn_agri.cache_status()
        return jsonify({
            "sections": {
                section: {
                    "entries": sum(status.get(p, {}).get("entries", 0) for p in prefixes),
                    "next_expiry_seconds": min(
                        (status[p]["next_expiry_seconds"] for p in prefixes if p in status),
                        default=None,
                    ),
                }
                for section, prefixes in CACHE_SECTIONS.items()
            },
        })

# /farm-insights api starts here ========>>>>>
    @bp.route("/farm-insights", methods=["POST"])
    def farm_insights():
        # Accept JSON (even without a JSON Content-Type), form-data or URL params
        data = (
            request.get_json(silent=True, force=True) or
            request.form.to_dict() or
            request.args.to_dict()
        )

        language = (data.get("language") or "en").strip().lower()
        if language not in SUPPORTED_LANGUAGES:
            return jsonify({"error": f"'language' must be one of {list(SUPPORTED_LANGUAGES)}."}), 400

        sections = data.get("sections") or ALL_SECTIONS
        if isinstance(sections, str):
            sections = [s.strip() for s in sections.split(",")]
        unknown = [s for s in sections if s not in ALL_SECTIONS]
        if unknown:
            return jsonify({"error": f"Unknown sections {unknown}. Allowed: {ALL_SECTIONS}."}), 400
        sections = set(sections)

        # --------------------------------------------------------
        # 1. Resolve location
        # --------------------------------------------------------
        try:
            place = resolve_location(data)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except LocationError as e:
            return jsonify({"error": str(e)}), 404
        except requests.RequestException as e:
            return jsonify({"error": f"Location service unavailable: {e}"}), 502

        # --------------------------------------------------------
        # 2. Fetch every requested section in parallel
        # --------------------------------------------------------
        sections_data, errors = build_insights(query_llm, place, sections, language)

        # --------------------------------------------------------
        # 3. Build response
        # --------------------------------------------------------
        result = {
            "location": place,
            "language": language,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **sections_data,
        }
        result["errors"] = errors

        if all(result.get(s) is None for s in sections):
            return jsonify(result), 502

        return jsonify(result)

    return bp
