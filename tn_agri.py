"""
Tamil Nadu Agriculture Department data (AGRISNET "People App" portals).

  - Reservoirs : tnagriculture.in/people_app/more/reservoir      (daily dam levels)
  - Seeds      : tnagrisnet.tn.gov.in/people_app/Seed            (govt depot seed stock)
  - Fertilizer : tnagriculture.in/people_app/fertilizer          (dealer stock + prices)
  - Schemes    : tnagrisnet.tn.gov.in/people_app/Scheme          (subsidy schemes)

All functions return plain dicts/lists ready for JSON and cache their results,
so the government servers are called only when the cached data expires.
"""

import difflib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html import unescape
from urllib.parse import urlparse

import requests


AGRISNET_URL = "https://www.tnagrisnet.tn.gov.in/people_app/"
TNAGRI_URL = "https://tnagriculture.in/people_app/"

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (farm-insights API)"}
HTTP_TIMEOUT = (10, 40)  # (connect, read) seconds

# When a portal can't be reached, skip it for this long instead of waiting on every call
HOST_DOWN_SECONDS = 120
# After a failed refresh: serve the previous data (or fail fast) for this long, then retry
FAILURE_RETRY_SECONDS = 5 * 60

IST = timezone(timedelta(hours=5, minutes=30))

# Cache lifetimes (seconds)
RESERVOIR_TTL = 3 * 3600
STOCK_TTL = 6 * 3600
CATALOGUE_TTL = 24 * 3600

# Keep the load on the government servers low
_pool = ThreadPoolExecutor(max_workers=3)

# Slow-changing data (scheme catalogue, prices, block lists) is also saved here,
# so restarts don't re-download it and it is still served while a portal is down
DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tn_cache")


# ============================================================
# Cache: one computation per key at a time
# ============================================================
_cache = {}
_cache_lock = threading.Lock()
_key_locks = {}
_failures = {}  # key -> (retry_after_timestamp, error message)


def _disk_path(key):
    return os.path.join(DISK_CACHE_DIR, "_".join(str(k) for k in key) + ".json")


def _disk_load(key):
    """(expires, value) saved on disk, or None."""
    try:
        with open(_disk_path(key), encoding="utf-8") as f:
            saved = json.load(f)
        return saved["expires"], saved["value"]
    except (OSError, ValueError, KeyError):
        return None


def _disk_save(key, expires, value):
    try:
        os.makedirs(DISK_CACHE_DIR, exist_ok=True)
        tmp = _disk_path(key) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"expires": expires, "value": value}, f, ensure_ascii=False)
        os.replace(tmp, _disk_path(key))
    except OSError as e:
        print(f"TN data: could not save {key[0]} to disk: {e}", flush=True)


def cached(key, ttl, compute, persist=False):
    """
    Return the cached value, computing it when missing or expired.
    If the refresh fails, the previous value is served for FAILURE_RETRY_SECONDS;
    with no previous value the error is remembered so callers fail fast.
    persist=True also keeps the value on disk (survives restarts).
    """
    with _cache_lock:
        item = _cache.get(key)
        if item and item[0] > time.time():
            return item[1]
        if not item and persist:
            item = _disk_load(key)
            if item:
                _cache[key] = item
                if item[0] > time.time():
                    return item[1]
        lock = _key_locks.setdefault(key, threading.Lock())

    with lock:
        now = time.time()
        with _cache_lock:
            # Another thread may have filled it while we waited
            item = _cache.get(key)
            if item and item[0] > now:
                return item[1]
            failure = _failures.get(key)
            if failure and failure[0] > now:
                raise RuntimeError(failure[1])

        try:
            value = compute()
        except Exception as e:
            with _cache_lock:
                if item:
                    print(f"TN data: refresh of {key[0]} failed, serving previous data: {e}", flush=True)
                    _cache[key] = (now + FAILURE_RETRY_SECONDS, item[1])
                    return item[1]
                _failures[key] = (now + FAILURE_RETRY_SECONDS, str(e))
            raise

        expires = time.time() + ttl
        with _cache_lock:
            _cache[key] = (expires, value)
            _failures.pop(key, None)
        if persist:
            _disk_save(key, expires, value)
        return value


def cache_lookup(key):
    """Cached value, or None when missing/expired."""
    with _cache_lock:
        item = _cache.get(key)
        if item and item[0] > time.time():
            return item[1]
        _cache.pop(key, None)
        return None


def cache_store(key, value, ttl):
    with _cache_lock:
        _cache[key] = (time.time() + ttl, value)
    return value


def clear_cache(prefixes=None):
    """Remove cached entries whose key starts with one of prefixes (all when None)."""
    with _cache_lock:
        keys = [k for k in _cache if prefixes is None or k[0] in prefixes]
        for k in keys:
            del _cache[k]
        for k in [k for k in _failures if prefixes is None or k[0] in prefixes]:
            del _failures[k]
        _host_down.clear()
    return len(keys)


def cache_status():
    """{key prefix: {entries, next_expiry_seconds}} for the cached data."""
    now = time.time()
    status = {}
    with _cache_lock:
        for key, (expires, _) in _cache.items():
            item = status.setdefault(key[0], {"entries": 0, "next_expiry_seconds": None})
            item["entries"] += 1
            left = max(0, int(expires - now))
            if item["next_expiry_seconds"] is None or left < item["next_expiry_seconds"]:
                item["next_expiry_seconds"] = left
    return status


_host_down = {}  # host -> timestamp until which it is skipped


def _request(method, url, **kwargs):
    """HTTP call that skips a portal for a while once it stops responding."""
    host = urlparse(url).netloc
    if _host_down.get(host, 0) > time.time():
        raise requests.ConnectionError(f"{host} is not responding; retrying in a few minutes.")

    try:
        response = requests.request(method, url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT, **kwargs)
    except requests.ConnectionError:
        # Can't connect at all (includes connect timeouts). A slow reply (ReadTimeout)
        # is only that one request, so it doesn't mark the whole portal as down.
        _host_down[host] = time.time() + HOST_DOWN_SECONDS
        raise

    response.raise_for_status()
    return response


def _post_json(url, data=None):
    response = _request("POST", url, data=data or "")
    return response.json() if response.text.strip() else []


def _get_json(url):
    response = _request("GET", url)
    return response.json() if response.text.strip() else []


def _pick(item, en_key, ta_key, language):
    """Tamil value when requested and present, otherwise English."""
    if language == "ta" and (item.get(ta_key) or "").strip():
        return item[ta_key].strip()
    return (item.get(en_key) or "").strip()


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return round(2 * r * math.asin(math.sqrt(a)), 1)


# ============================================================
# Districts
# The portals use two different ID systems:
#   fert_id - Fertilizer portal (LGD codes)
#   seed_id - Seed portal
# ============================================================
DISTRICTS = [
    # (name, fert_id, seed_id, aliases)
    ("Ariyalur", 3317, 30, []),
    ("Chengalpattu", 3338, 35, ["chengalpet"]),
    ("Chennai", 3302, 32, ["madras"]),
    ("Coimbatore", 3333, 11, ["kovai"]),
    ("Cuddalore", 3318, 4, []),
    ("Dharmapuri", 3331, 12, []),
    ("Dindigul", 3313, 22, []),
    ("Erode", 3310, 10, []),
    ("Kallakurichi", 3337, 38, ["kallakuruchi"]),
    ("Kancheepuram", 3303, 2, ["kanchipuram", "kanchi"]),
    ("Kanniyakumari", 3330, 29, ["kanyakumari", "nagercoil"]),
    ("Karur", 3314, 15, []),
    ("Krishnagiri", 3332, 13, []),
    ("Madurai", 3324, 21, []),
    ("Mayiladuthurai", 3340, 39, ["mayiladuturai"]),
    ("Nagapattinam", 3319, 19, []),
    ("Namakkal", 3309, 8, []),
    ("Perambalur", 3316, 16, []),
    ("Pudukkottai", 3322, 20, ["pudukottai"]),
    ("Ramanathapuram", 3327, 25, ["ramnad"]),
    ("Ranipet", 3335, 36, []),
    ("Salem", 3308, 9, []),
    ("Sivaganga", 3323, 24, ["sivagangai"]),
    ("Tenkasi", 3339, 34, []),
    ("Thanjavur", 3321, 17, ["tanjore", "tanjavur"]),
    ("The Nilgiris", 3311, 31, ["nilgiris", "ooty", "udhagamandalam"]),
    ("Theni", 3325, 23, []),
    ("Thiruvallur", 3301, 3, ["tiruvallur"]),
    ("Thiruvarur", 3320, 18, ["tiruvarur"]),
    ("Thoothukudi", 3328, 28, ["thoothukkudi", "tuticorin"]),
    ("Tiruchirappalli", 3315, 14, ["trichy", "tiruchchirappalli", "tiruchirapalli", "thiruchirappalli"]),
    ("Tirunelveli", 3329, 27, ["thirunelveli", "nellai"]),
    ("Tirupathur", 3336, 37, ["tirupattur", "thirupathur"]),
    ("Tiruppur", 3334, 33, ["tirupur", "thiruppur"]),
    ("Tiruvannamalai", 3306, 7, ["thiruvannamalai"]),
    ("Vellore", 3304, 6, []),
    ("Villupuram", 3307, 5, ["viluppuram"]),
    ("Virudhunagar", 3326, 26, ["virudunagar"]),
]


def _norm(text):
    text = re.sub(r"\bdistrict\b", "", (text or "").lower())
    text = re.sub(r"^the\s+", "", text.strip())
    return re.sub(r"[^a-z]", "", text)


DISTRICT_NAMES_TA = {
    "Ariyalur": "அரியலூர்", "Chengalpattu": "செங்கல்பட்டு", "Chennai": "சென்னை",
    "Coimbatore": "கோயம்புத்தூர்", "Cuddalore": "கடலூர்", "Dharmapuri": "தருமபுரி",
    "Dindigul": "திண்டுக்கல்", "Erode": "ஈரோடு", "Kallakurichi": "கள்ளக்குறிச்சி",
    "Kancheepuram": "காஞ்சிபுரம்", "Kanniyakumari": "கன்னியாகுமரி", "Karur": "கரூர்",
    "Krishnagiri": "கிருஷ்ணகிரி", "Madurai": "மதுரை", "Mayiladuthurai": "மயிலாடுதுறை",
    "Nagapattinam": "நாகப்பட்டினம்", "Namakkal": "நாமக்கல்", "Perambalur": "பெரம்பலூர்",
    "Pudukkottai": "புதுக்கோட்டை", "Ramanathapuram": "இராமநாதபுரம்", "Ranipet": "இராணிப்பேட்டை",
    "Salem": "சேலம்", "Sivaganga": "சிவகங்கை", "Tenkasi": "தென்காசி", "Thanjavur": "தஞ்சாவூர்",
    "The Nilgiris": "நீலகிரி", "Theni": "தேனி", "Thiruvallur": "திருவள்ளூர்",
    "Thiruvarur": "திருவாரூர்", "Thoothukudi": "தூத்துக்குடி", "Tiruchirappalli": "திருச்சிராப்பள்ளி",
    "Tirunelveli": "திருநெல்வேலி", "Tirupathur": "திருப்பத்தூர்", "Tiruppur": "திருப்பூர்",
    "Tiruvannamalai": "திருவண்ணாமலை", "Vellore": "வேலூர்", "Villupuram": "விழுப்புரம்",
    "Virudhunagar": "விருதுநகர்",
}
# Short Tamil forms people type (without the final letters that change with suffixes)
_TAMIL_ALIASES = {"திருச்சி": "Tiruchirappalli", "நெல்லை": "Tirunelveli", "கோவை": "Coimbatore",
                  "தஞ்சை": "Thanjavur", "புதுகை": "Pudukkottai", "ஊட்டி": "The Nilgiris"}

_DISTRICT_INDEX = {}
_DISTRICT_BY_NAME = {}
for _name, _fert, _seed, _aliases in DISTRICTS:
    _info = {"name": _name, "name_ta": DISTRICT_NAMES_TA[_name], "fert_id": _fert, "seed_id": _seed}
    _DISTRICT_BY_NAME[_name] = _info
    for _alias in [_name] + _aliases:
        _DISTRICT_INDEX[_norm(_alias)] = _info

_TAMIL_INDEX = {ta: _DISTRICT_BY_NAME[en] for en, ta in DISTRICT_NAMES_TA.items()}
_TAMIL_INDEX.update({ta: _DISTRICT_BY_NAME[en] for ta, en in _TAMIL_ALIASES.items()})


def find_district_in_text(text):
    """A Tamil Nadu district mentioned anywhere in free text (English or Tamil), or None."""
    text = text or ""
    for tamil, info in _TAMIL_INDEX.items():
        # Tamil words take suffixes (மதுரையில்), so match on the stem;
        # very short names (தேனி) must match in full
        stem = tamil[:-1] if len(tamil) > 4 else tamil
        if stem in text:
            return info
    compact = _norm(text)
    for alias, info in _DISTRICT_INDEX.items():
        if len(alias) >= 5 and alias in compact:
            return info
    return None


def find_district(*candidates):
    """Match a Tamil Nadu district from place names (district, city, display name...)."""
    for text in candidates:
        key = _norm(text)
        if key in _DISTRICT_INDEX:
            return _DISTRICT_INDEX[key]
    for text in candidates:
        close = difflib.get_close_matches(_norm(text), _DISTRICT_INDEX.keys(), n=1, cutoff=0.85)
        if close:
            return _DISTRICT_INDEX[close[0]]
    return None


# ============================================================
# Reservoirs (dam storage & flow)
# ============================================================
CAUVERY_DELTA = [
    "Salem", "Erode", "Namakkal", "Karur", "Tiruchirappalli", "Thanjavur", "Thiruvarur",
    "Nagapattinam", "Mayiladuthurai", "Pudukkottai", "Cuddalore", "Ariyalur",
]
PAP_DISTRICTS = ["Coimbatore", "Tiruppur"]
VAIGAI_DISTRICTS = ["Theni", "Madurai", "Dindigul", "Sivaganga", "Ramanathapuram", "Virudhunagar"]
TAMIRABARANI_DISTRICTS = ["Tirunelveli", "Thoothukudi", "Tenkasi"]

# Approximate dam coordinates, river and the districts whose irrigation they feed
DAMS = {
    "mettur": (11.799, 77.808, "Cauvery", "Tamil Nadu", CAUVERY_DELTA),
    "krishnarajasagar": (12.423, 76.572, "Cauvery", "Karnataka", CAUVERY_DELTA),
    "kabini": (11.973, 76.351, "Kabini (Cauvery)", "Karnataka", CAUVERY_DELTA),
    "harangi": (12.492, 75.905, "Harangi (Cauvery)", "Karnataka", CAUVERY_DELTA),
    "hemavathy": (12.822, 76.058, "Hemavathi (Cauvery)", "Karnataka", CAUVERY_DELTA),
    "bhavanisagar": (11.472, 77.113, "Bhavani", "Tamil Nadu", ["Erode", "Tiruppur", "Karur"]),
    "amaravathi": (10.423, 77.261, "Amaravathi", "Tamil Nadu", ["Tiruppur", "Karur", "Dindigul"]),
    "periyar": (9.529, 77.144, "Periyar", "Kerala (TN operated)", VAIGAI_DISTRICTS),
    "vaigai": (10.054, 77.591, "Vaigai", "Tamil Nadu", VAIGAI_DISTRICTS),
    "papanasam": (8.708, 77.371, "Thamirabarani", "Tamil Nadu", TAMIRABARANI_DISTRICTS),
    "manimuthar": (8.646, 77.422, "Manimuthar", "Tamil Nadu", TAMIRABARANI_DISTRICTS),
    "pechiparai": (8.444, 77.311, "Kodayar", "Tamil Nadu", ["Kanniyakumari"]),
    "perunchani": (8.356, 77.353, "Paralayar", "Tamil Nadu", ["Kanniyakumari"]),
    "krishnagiri": (12.482, 78.182, "Thenpennai", "Tamil Nadu", ["Krishnagiri", "Dharmapuri"]),
    "sathanur": (12.182, 78.867, "Thenpennai", "Tamil Nadu",
                 ["Tiruvannamalai", "Villupuram", "Kallakurichi", "Cuddalore"]),
    "sholayar": (10.299, 76.874, "Sholayar", "Tamil Nadu", PAP_DISTRICTS),
    "parambikulam": (10.392, 76.775, "Parambikulam", "Kerala (TN operated)", PAP_DISTRICTS),
    "aliyar": (10.486, 76.967, "Aliyar", "Tamil Nadu", PAP_DISTRICTS),
    "thirumurthy": (10.458, 77.174, "Palar (PAP)", "Tamil Nadu", PAP_DISTRICTS),
}


DAM_NAMES_TA = {
    "mettur": "மேட்டூர்", "krishnarajasagar": "கிருஷ்ணராஜ சாகர்", "kabini": "கபினி", "harangi": "ஹாரங்கி",
    "hemavathy": "ஹேமாவதி", "bhavanisagar": "பவானிசாகர்", "amaravathi": "அமராவதி", "periyar": "பெரியாறு",
    "vaigai": "வைகை", "papanasam": "பாபநாசம்", "manimuthar": "மணிமுத்தாறு", "pechiparai": "பேச்சிப்பாறை",
    "perunchani": "பெருஞ்சாணி", "krishnagiri": "கிருஷ்ணகிரி", "sathanur": "சாத்தனூர்", "sholayar": "சோலையாறு",
    "parambikulam": "பரம்பிக்குளம்", "aliyar": "ஆழியாறு", "thirumurthy": "திருமூர்த்தி",
}


def _dam_key(name):
    key = re.sub(r"\(.*?\)", "", name.lower())
    return re.sub(r"[^a-z]", "", key)


def _fetch_reservoirs(day):
    html = _request("GET", f"{TNAGRI_URL}more/reservoir/en/{day.isoformat()}").text

    dams = []
    # Each dam: <h4>NAME</h4> ... 3 <ul> rows (labels, units, values)
    for block in re.split(r"<h4[^>]*>", html)[1:]:
        name = unescape(re.sub(r"<[^>]+>", "", block.split("</h4>", 1)[0])).strip()
        rows = re.findall(r"<ul[^>]*>([\s\S]*?)</ul>", block)
        if not name or len(rows) < 3:
            continue
        values = [re.sub(r"<[^>]+>", "", v).strip() for v in re.findall(r"<li[^>]*>([\s\S]*?)</li>", rows[2])]
        values += [""] * (4 - len(values))
        full, level, inflow, outflow = (_to_float(v) for v in values[:4])
        if full is None and level is None:
            continue

        clean_name = name.rstrip("*").strip()
        info = DAMS.get(_dam_key(clean_name))
        dams.append({
            "name": clean_name,
            "name_ta": DAM_NAMES_TA.get(_dam_key(clean_name), clean_name),
            "river": info[2] if info else "",
            "state": info[3] if info else "",
            "full_depth_ft": full,
            "current_level_ft": level,
            "level_percent": round(level / full * 100, 1) if full and level is not None else None,
            "inflow_cusecs": inflow,
            "outflow_cusecs": outflow,
            "_info": info,
        })
    return dams


def get_reservoirs():
    """Latest dam levels (today, or the most recent day with data)."""
    def compute():
        today = datetime.now(IST).date()
        last_error = None
        for back in range(4):
            day = today - timedelta(days=back)
            try:
                dams = _fetch_reservoirs(day)
            except requests.RequestException as e:
                last_error = e
                continue
            if dams:
                return {"as_on_date": day.isoformat(), "dams": dams}
        raise RuntimeError(f"Reservoir data unavailable: {last_error or 'no data for the last 4 days'}")

    return cached(("reservoirs",), RESERVOIR_TTL, compute)


def get_water_for_location(lat, lon, district_name):
    """Dams that feed this district first, then the rest by distance."""
    data = get_reservoirs()
    dams = []
    for dam in data["dams"]:
        info = dam["_info"]
        item = {k: v for k, v in dam.items() if k != "_info"}
        item["distance_km"] = haversine_km(lat, lon, info[0], info[1]) if info else None
        item["serves_your_district"] = bool(info and district_name in info[4])
        dams.append(item)

    dams.sort(key=lambda d: (not d["serves_your_district"], d["distance_km"] is None, d["distance_km"] or 0))

    serving = [d["level_percent"] for d in dams if d["serves_your_district"] and d["level_percent"] is not None]
    nearest = [d["level_percent"] for d in dams[:3] if d["level_percent"] is not None]
    levels = serving or nearest
    average = round(sum(levels) / len(levels), 1) if levels else None

    if average is None:
        availability = "Unknown"
    elif average >= 70:
        availability = "High"
    elif average >= 40:
        availability = "Medium"
    else:
        availability = "Low"

    return {
        "source": "Tamil Nadu Agriculture Department - Storage and Flow Data on Major Reservoirs",
        "as_on_date": data["as_on_date"],
        "availability": availability,
        "average_level_percent": average,
        "availability_basis": "dams serving your district" if serving else "nearest dams",
        "reservoirs": dams,
    }


# ============================================================
# Seeds (government Agricultural Extension Centre depots)
# ============================================================
def _seed_blocks(seed_district_id):
    return cached(
        ("seed_blocks", seed_district_id), CATALOGUE_TTL,
        lambda: _post_json(f"{AGRISNET_URL}Seed/getBlocks/{seed_district_id}"),
        persist=True,
    )


def _seed_block_stock(seed_district_id, block_id):
    def compute():
        data = _post_json(
            f"{AGRISNET_URL}Seed/getStock",
            {"district_id": seed_district_id, "block_id": block_id, "crop_id": ""},
        )
        return data.get("aec_groups") or [] if isinstance(data, dict) else []

    return cached(("seed_stock", seed_district_id, block_id), STOCK_TTL, compute)


def get_seeds_for_location(lat, lon, district, language, max_depots=5):
    blocks = _seed_blocks(district["seed_id"])
    groups = [
        g for result in _pool.map(lambda b: _seed_block_stock(district["seed_id"], b["id"]), blocks)
        for g in result
    ]

    depots, crops = [], {}
    for g in groups:
        profile = g.get("profile") or {}
        d_lat, d_lon = _to_float(profile.get("latitude")), _to_float(profile.get("longitude"))
        has_gps = bool(d_lat and d_lon)

        stocks = []
        for s in g.get("stocks") or []:
            is_nos = (_to_float(s.get("crop_id")) or 0) > 20
            stock = {
                "crop": _pick(s, "cropName", "tamil_stock_name", language),
                "variety": _pick(s, "varietyName", "tamil_variety_name", language),
                "seed_class": _pick(s, "className", "tamil_class_name", language),
                "price_rs": _to_float(s.get("price")),
                "price_per": "No" if is_nos else "Kg",
                "quantity": _to_float(s.get("quantity")),
                "unit": "Nos" if is_nos else "Kg",
            }
            stocks.append(stock)

            crop = crops.setdefault(stock["crop"], {"crop": stock["crop"], "varieties": {}})
            crop["varieties"][stock["variety"]] = (
                crop["varieties"].get(stock["variety"], 0) + (stock["quantity"] or 0)
            )

        depots.append({
            "depot_name": _pick(g, "aec_name", "tamil_aec_name", language),
            "block": _pick(g, "Block_Name", "tamil_block_name", language),
            "address": profile.get("address", ""),
            "manager_name": g.get("full_name", ""),
            "manager_designation": g.get("designation", ""),
            "phone": g.get("user_phone", ""),
            "latitude": d_lat if has_gps else None,
            "longitude": d_lon if has_gps else None,
            "distance_km": haversine_km(lat, lon, d_lat, d_lon) if has_gps else None,
            "maps_url": f"https://maps.google.com/?q={d_lat},{d_lon}" if has_gps else None,
            "depot_photo_url": profile.get("depoUrl") or None,
            "stock": stocks,
        })

    depots.sort(key=lambda d: (d["distance_km"] is None, d["distance_km"] or 0))

    crops_available = [
        {
            "crop": c["crop"],
            "varieties": [
                {"variety": v, "total_quantity_in_district": round(q, 1)}
                for v, q in sorted(c["varieties"].items(), key=lambda x: -x[1])
            ],
        }
        for c in sorted(crops.values(), key=lambda c: c["crop"])
    ]

    return {
        "source": "Tamil Nadu Agriculture Department - Seed Availability (AEC depots)",
        "district": district["name"],
        "crops_available": crops_available,
        "nearest_depots": depots[:max_depots],
        "total_depots_in_district": len(depots),
    }


# ============================================================
# Fertilizer (dealer stock + official prices)
# ============================================================
FERTILIZER_PRODUCTS = {
    1: "UREA", 2: "DAP", 3: "MOP", 4: "NP 16:20:0:13 Complex", 5: "NP 20:20:0:13 Complex",
    6: "Zincated 20:20:0:13 Complex", 7: "NPK 10:26:26 Complex", 8: "NPK 12:32:16 Complex",
    9: "NPK 14:35:14 Complex", 10: "NPK 15:15:15 Complex", 11: "NPK 16:16:16 Complex",
    12: "NPK 17:17:17 Complex", 13: "NPK 28:28:0 Complex", 14: "Super Phosphate",
    15: "Ammonium Sulphate", 16: "NPK 24:24:0 Complex", 17: "NPK 15:15:15.09 Complex",
    18: "NPK 19:19:19 Complex", 19: "CITY COMPOST", 20: "Ammonium Chloride",
    21: "Mono Ammonium Phosphate", 22: "SSP (Powder)", 23: "SSP (Granuals)", 24: "9:24:24",
    25: "TSP", 26: "NPK Complex (11:13:14)", 27: "FOM (Fermented Organic Manure)",
    28: "PROM (Phosphate Rich Organic Manure)", 29: "Organic Plus",
    30: "PDM (Potash Derived from Molasses)",
}


def _fert_blocks(fert_district_id):
    return cached(
        ("fert_blocks", fert_district_id), CATALOGUE_TTL,
        lambda: _post_json(f"{TNAGRI_URL}Fertilizer/getBlocks/{fert_district_id}"),
        persist=True,
    )


def _fert_block_dealers(fert_district_id, subdistrict_id):
    return cached(
        ("fert_dealers", fert_district_id, subdistrict_id), STOCK_TTL,
        lambda: _get_json(f"{TNAGRI_URL}Fertilizer/getResults/{fert_district_id}/{subdistrict_id}/en"),
    )


def _fert_inspector(fert_district_id):
    def compute():
        data = _get_json(f"{TNAGRI_URL}Fertilizer/getInspector/{fert_district_id}")
        return data.get("mobile") if isinstance(data, dict) else None

    return cached(("fert_inspector", fert_district_id), CATALOGUE_TTL, compute)


def get_fertilizer_prices():
    """Official printed bag prices (statewide)."""
    def fetch(item):
        product_id, name = item
        rows = _post_json(f"{TNAGRI_URL}fertilizer_price/fertDetails/{product_id}")
        return {
            "product": name,
            "bag_size_kg": 45 if product_id == 1 else 50,
            "prices": [
                {"company": r.get("company", ""), "price_rs": _to_float(r.get("price"))}
                for r in rows or []
            ],
        }

    def compute():
        return [p for p in _pool.map(fetch, FERTILIZER_PRODUCTS.items()) if p["prices"]]

    return cached(("fert_prices",), CATALOGUE_TTL, compute, persist=True)


def get_fertilizer_for_location(lat, lon, district, language, max_dealers=10):
    blocks = _fert_blocks(district["fert_id"])
    inspector_future = _pool.submit(_fert_inspector, district["fert_id"])

    raw_dealers = [
        d for result in _pool.map(
            lambda b: _fert_block_dealers(district["fert_id"], b["subdistrict_id"]), blocks
        )
        for d in result
    ]

    dealers, seen = [], set()
    for d in raw_dealers:
        if d.get("dealer_id") in seen:
            continue
        seen.add(d.get("dealer_id"))

        d_lat, d_lon = _to_float(d.get("latitude")), _to_float(d.get("longitude"))
        has_gps = bool(d_lat and d_lon)
        dealers.append({
            "dealer_name": _pick(d, "delar_name", "tamil_agency", language),
            "address": _pick(d, "address", "tamil_address", language).rstrip(", "),
            "phone": d.get("mobile_number", ""),
            "dealer_id": d.get("dealer_id", ""),
            "latitude": d_lat if has_gps else None,
            "longitude": d_lon if has_gps else None,
            "distance_km": haversine_km(lat, lon, d_lat, d_lon) if has_gps else None,
            "maps_url": f"https://maps.google.com/?q={d_lat},{d_lon}" if has_gps else None,
            # Portal reports tonnes; the app shows kilograms
            "stock_kg": {
                name: round(_to_float(qty) * 1000)
                for name, qty in (d.get("fert") or {}).items()
                if _to_float(qty)
            },
        })

    dealers.sort(key=lambda d: (d["distance_km"] is None, d["distance_km"] or 0))

    try:
        inspector = inspector_future.result()
    except Exception:
        inspector = None
    # Not run inside _pool: it fans out into _pool itself (cached / warmed up at startup)
    try:
        prices = get_fertilizer_prices()
    except Exception:
        prices = []

    return {
        "source": "Tamil Nadu Agriculture Department - Fertilizer Finder & Price List",
        "district": district["name"],
        "fertilizer_inspector_phone": inspector,
        "nearest_dealers": dealers[:max_dealers],
        "total_dealers_in_district": len(dealers),
        "prices": prices,
        "price_note": "Buy fertilizers at the price printed on the bag.",
    }


# ============================================================
# Schemes & subsidies (statewide catalogue)
# ============================================================
DEPARTMENTS = {
    "A": ("Agriculture", "வேளாண்மை"),
    "H": ("Horticulture", "தோட்டக்கலை"),
    "E": ("Agricultural Engineering", "வேளாண் பொறியியல்"),
    "SO": ("Seed Certification & Organic Certification", "விதைச் சான்று மற்றும் அங்ககச் சான்று"),
    "M": ("Marketing & Agri Business", "வேளாண் விற்பனை மற்றும் வணிகம்"),
    "S": ("Sugarcane", "சர்க்கரை"),
    "T": ("Tamil Nadu Watershed Development Agency", "தமிழ்நாடு நீர்வடிப்பகுதி மேம்பாட்டு முகமை"),
    "TU": ("Tamil Nadu Agricultural University", "தமிழ்நாடு வேளாண்மைப் பல்கலைக்கழகம்"),
}


def _build_scheme_catalogue():
    base = f"{AGRISNET_URL}Scheme/"

    failed = []

    def safe_post(url):
        for attempt in range(2):
            try:
                return _post_json(url)
            except (requests.RequestException, ValueError) as e:
                error = e
                if attempt == 0:
                    time.sleep(2)
        failed.append(f"{url}: {error}")
        return []

    scheme_lists = dict(zip(DEPARTMENTS, _pool.map(lambda d: safe_post(f"{base}schemeList/{d}"), DEPARTMENTS)))

    schemes = [
        {"dept": dept, "scheme_id": s["scheme_id"],
         "name": s.get("Scheme_Name", ""), "name_ta": s.get("Scheme_Name_Tamil", ""),
         "sub_schemes": []}
        for dept, items in scheme_lists.items() for s in items or []
    ]

    type_lists = _pool.map(lambda s: safe_post(f"{base}loadInputType/{s['scheme_id']}/{s['dept']}"), schemes)
    sub_schemes = []
    for scheme, types in zip(schemes, type_lists):
        for t in types or []:
            sub = {"input_type_id": t["input_type_id"],
                   "name": t.get("input_type_name", ""), "name_ta": t.get("input_type_name_tamil", ""),
                   "components": [], "_scheme": scheme}
            scheme["sub_schemes"].append(sub)
            sub_schemes.append(sub)

    class_lists = _pool.map(
        lambda s: safe_post(f"{base}getClass/{s['input_type_id']}/{s['_scheme']['dept']}"), sub_schemes
    )
    for sub, classes in zip(sub_schemes, class_lists):
        sub["components"] = [
            {"class_id": c["class_id"], "name": c.get("class_Name", ""), "name_ta": c.get("class_name_Tamil", "")}
            for c in classes or []
        ]
        del sub["_scheme"]

    if not schemes:
        raise RuntimeError(f"Scheme portal returned no schemes. {failed[0] if failed else ''}".strip())
    # A partial catalogue must not replace a complete one (cache keeps the previous copy)
    if failed:
        raise RuntimeError(f"Scheme portal: {len(failed)} requests failed, e.g. {failed[0]}")
    return schemes


def get_scheme_catalogue():
    return cached(("scheme_catalogue",), CATALOGUE_TTL, _build_scheme_catalogue, persist=True)


def _lang_name(item, language):
    return item["name_ta"].strip() if language == "ta" and item.get("name_ta", "").strip() else item["name"].strip()


def get_schemes_and_subsidies(language):
    catalogue = get_scheme_catalogue()
    schemes, subsidies = [], []

    for s in catalogue:
        dept_name = DEPARTMENTS[s["dept"]][1 if language == "ta" else 0]
        schemes.append({
            "department_code": s["dept"],
            "department": dept_name,
            "scheme_id": s["scheme_id"],
            "scheme_name": _lang_name(s, language),
            "sub_schemes": [
                {"input_type_id": sub["input_type_id"], "name": _lang_name(sub, language),
                 "subsidy_count": len(sub["components"])}
                for sub in s["sub_schemes"]
            ],
        })
        for sub in s["sub_schemes"]:
            for c in sub["components"]:
                subsidies.append({
                    "department_code": s["dept"],
                    "department": dept_name,
                    "scheme_id": s["scheme_id"],
                    "scheme_name": _lang_name(s, language),
                    "input_type_id": sub["input_type_id"],
                    "sub_scheme": _lang_name(sub, language),
                    "class_id": c["class_id"],
                    "component": _lang_name(c, language),
                    "detail_url": (
                        f"/schemes/detail?department={s['dept']}&scheme_id={s['scheme_id']}"
                        f"&input_type_id={sub['input_type_id']}&class_id={c['class_id']}&language={language}"
                    ),
                })

    return {
        "source": "Tamil Nadu Agriculture Department - Subsidy Schemes (AGRISNET)",
        "schemes": schemes,
        "subsidies": subsidies,
    }


def _html_to_text(value):
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", value or "", flags=re.IGNORECASE)
    text = unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def get_subsidy_detail(department, scheme_id, input_type_id, class_id, language):
    def compute():
        return _post_json(f"{AGRISNET_URL}Scheme/viewScheme/{department}/{scheme_id}/{input_type_id}/{class_id}")

    data = cached(("scheme_detail", department, scheme_id, input_type_id, class_id), CATALOGUE_TTL, compute)
    if not isinstance(data, dict) or not data:
        return None

    def field(key):
        if language == "ta":
            for ta_key in (f"{key}_tamil", f"{key}_Tamil", f"{key}_ta"):
                if (data.get(ta_key) or "").strip():
                    return data[ta_key]
        return data.get(key) or ""

    return {
        "source": "Tamil Nadu Agriculture Department - Subsidy Schemes (AGRISNET)",
        "department_code": department,
        "department": DEPARTMENTS.get(department, ("", ""))[1 if language == "ta" else 0],
        "scheme_name": field("Scheme_Name").strip(),
        "sub_scheme": field("input_type_name").strip(),
        "component": (data.get("class_name_Tamil") if language == "ta" and data.get("class_name_Tamil")
                      else data.get("class_Name", "")).strip(),
        "subsidy": field("pattern_of_subsidy").strip(),
        "eligibility": _html_to_text(field("eligibility")),
        "description": _html_to_text(field("scheme_desc")),
        "documents_required": _html_to_text(field("doc_req")),
        "image_url": f"http://tnagrisnet.tn.gov.in/ARS/scheme_photo/{data['image']}" if data.get("image") else None,
        "toll_free": "1800 425 4444",
    }


def warm_up(loaders=None):
    """Load the statewide catalogues in the background so the first request is fast."""
    loaders = loaders or (get_scheme_catalogue, get_fertilizer_prices, get_reservoirs)

    def run():
        for fn in loaders:
            try:
                fn()
            except Exception as e:
                print(f"TN data warm-up: {fn.__name__} failed: {e}", flush=True)

    threading.Thread(target=run, daemon=True).start()
