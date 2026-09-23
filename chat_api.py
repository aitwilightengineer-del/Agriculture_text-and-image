"""
Conversational farm assistant for the mobile app.

POST /chat

  1. User types a question, e.g. "What is the current weather in Pondicherry?"
       {"message": "...", "session_id": "(optional)", "language": "en"}
  2. The API finds the topic (weather, soil, dams, seeds, fertilizer, schemes)
     and the location, fetches the data (same sources as /farm-insights) and
     answers in plain language.
  3. The response carries follow-up suggestions (dams, seeds, fertilizer,
     schemes...). When the user taps one, the app sends
       {"session_id": "...", "suggestion_id": "seeds"}
     and the same location is used - the user doesn't have to repeat it.
"""

import json
import re
import threading
import time
import uuid

import requests
from flask import Blueprint, request, jsonify

import tn_agri
from location_api import (
    SUPPORTED_LANGUAGES,
    LocationError,
    attach_tn_district,
    build_insights,
    parse_json_response,
    resolve_location,
    search_places,
)


# ============================================================
# Topics
# ============================================================
TOPICS = {
    # id: (sections to fetch, English label, Tamil label, English question, Tamil question)
    "climate": (["climate"], "Current weather", "தற்போதைய வானிலை",
                "What is the current weather in {place}?", "{place} பகுதியின் தற்போதைய வானிலை என்ன?"),
    "soil": (["soil"], "Soil health", "மண் வளம்",
             "How is the soil health in {place}?", "{place} பகுதியின் மண் வளம் எப்படி உள்ளது?"),
    "water": (["water"], "Dams & water availability", "அணைகள் & நீர் இருப்பு",
              "What is the dam water level for {place}?", "{place} பகுதிக்கான அணை நீர்மட்டம் என்ன?"),
    "seeds": (["seeds"], "Seeds that can be cultivated", "பயிரிடக்கூடிய விதைகள்",
              "Which seeds are available to cultivate in {place}?", "{place} மாவட்டத்தில் பயிரிட எந்த விதைகள் கிடைக்கின்றன?"),
    "fertilizer": (["fertilizer"], "Fertilizer availability", "உரம் இருப்பு",
                   "Where can I get fertilizer near {place}?", "{place} அருகில் உரம் எங்கு கிடைக்கும்?"),
    "schemes": (["schemes", "subsidies"], "Schemes & subsidies", "திட்டங்கள் & மானியங்கள்",
                "What schemes and subsidies are available for farmers?",
                "விவசாயிகளுக்கு என்ன திட்டங்கள் மற்றும் மானியங்கள் உள்ளன?"),
}

# Follow-up buttons shown after an answer: every topic the user hasn't asked about yet
SUGGESTION_TOPICS = ["climate", "soil", "water", "seeds", "fertilizer", "schemes"]

# Topics that use Tamil Nadu Agriculture Department data
TN_ONLY_TOPICS = {"water", "seeds", "fertilizer", "schemes"}

# Keyword fallback when Gemini can't classify the message
TOPIC_KEYWORDS = {
    "climate": ["weather", "climate", "rain", "temperature", "forecast", "humid", "wind", "hot", "cold",
                "வானிலை", "மழை", "வெப்ப", "காலநிலை"],
    "soil": ["soil", "ph ", "nutrient", "organic carbon", "மண்"],
    "water": ["dam", "reservoir", "water", "irrigation", "inflow", "அணை", "நீர்", "தண்ணீர்", "பாசன"],
    "seeds": ["seed", "variety", "varieties", "cultivat", "sow", "விதை", "இரக", "பயிரிட"],
    "fertilizer": ["fertilizer", "fertiliser", "urea", "dap", "potash", "mop", "manure", "உரம்", "யூரியா"],
    "schemes": ["scheme", "subsid", "loan", "assistance", "yojana", "திட்ட", "மானிய"],
}

TOPIC_WORDS = {w for words in TOPIC_KEYWORDS.values() for w in words}
STOP_WORDS = {
    "what", "which", "where", "when", "how", "the", "for", "and", "are", "is", "in", "at", "of",
    "to", "a", "an", "me", "my", "i", "can", "get", "available", "give", "show", "tell", "about",
    "any", "there", "near", "current", "now", "today", "please", "farmers", "farmer", "details",
}

SESSION_TTL = 6 * 3600

NEEDS_LOCATION_TEXT = {
    "en": "Which location? Share your current location or type a district / town name.",
    "ta": "எந்த இடம்? உங்கள் தற்போதைய இருப்பிடத்தைப் பகிரவும் அல்லது மாவட்டம் / ஊர் பெயரைத் தட்டச்சு செய்யவும்.",
}
GREETING_TEXT = {
    "en": "I can help with weather, soil health, dam water levels, seeds, fertilizer, and government "
          "schemes & subsidies. What would you like to know{about}?",
    "ta": "வானிலை, மண் வளம், அணை நீர்மட்டம், விதைகள், உரம், அரசு திட்டங்கள் & மானியங்கள் பற்றி "
          "உதவ முடியும். {about}என்ன தெரிந்து கொள்ள விரும்புகிறீர்கள்?",
}
TN_ONLY_TEXT = {
    "en": "{label} information is available only for locations in Tamil Nadu. {place} is outside Tamil Nadu.",
    "ta": "{label} தகவல் தமிழ்நாட்டில் உள்ள இடங்களுக்கு மட்டுமே கிடைக்கும். {place} தமிழ்நாட்டிற்கு வெளியே உள்ளது.",
}
NOT_FOUND_TEXT = {
    "en": "I couldn't find the location '{location}'. Please check the spelling or try a nearby town.",
    "ta": "'{location}' என்ற இடத்தைக் கண்டுபிடிக்க முடியவில்லை. எழுத்துப்பிழையைச் சரிபார்க்கவும் அல்லது அருகிலுள்ள ஊரை முயற்சிக்கவும்.",
}
FAILED_TEXT = {
    "en": "Sorry, I couldn't get the {label} information right now. Please try again in a few minutes.",
    "ta": "மன்னிக்கவும், இப்போது {label} தகவலைப் பெற முடியவில்லை. சில நிமிடங்கள் கழித்து மீண்டும் முயற்சிக்கவும்.",
}


# ============================================================
# Sessions (in memory)
# ============================================================
_sessions = {}
_sessions_lock = threading.Lock()


def get_session(session_id):
    now = time.time()
    with _sessions_lock:
        for sid in [s for s, v in _sessions.items() if v["expires"] < now]:
            del _sessions[sid]

        session = _sessions.get(session_id) if session_id else None
        if not session:
            session_id = uuid.uuid4().hex
            session = {"place": None, "answered": [], "pending_topic": None, "language": None}
            _sessions[session_id] = session
        session["expires"] = now + SESSION_TTL
        return session_id, session


# ============================================================
# Understanding the message
# ============================================================
def detect_language(text):
    return "ta" if re.search(r"[஀-௿]", text or "") else "en"


def understand_with_llm(query_llm, message):
    prompt = f"""
Classify a farmer's chat message for an agriculture assistant in Tamil Nadu, India.

Topics:
- climate: weather, temperature, rain, forecast, humidity, climate
- soil: soil health, soil type, pH, soil nutrients
- water: dams, reservoirs, water availability, irrigation water
- seeds: seeds, crops or varieties to cultivate, seed availability
- fertilizer: fertilizer availability, dealers, prices (urea, DAP, potash...)
- schemes: government schemes, subsidies, financial assistance
- none: greetings or anything else

Return ONLY a JSON object:
{{
  "topic": "climate | soil | water | seeds | fertilizer | schemes | none",
  "location": "place name mentioned in the message, spelled in English, or empty string",
  "search_terms": ["English keywords describing what they want, e.g. drip irrigation, tractor, paddy"]
}}

Message: {message}
"""
    data = parse_json_response(query_llm(prompt, json_output=True))
    topic = (data.get("topic") or "none").strip().lower()
    return {
        "topic": topic if topic in TOPICS else None,
        "location": (data.get("location") or "").strip(),
        "search_terms": [t.strip().lower() for t in data.get("search_terms") or [] if t.strip()],
    }


def understand_with_keywords(message):
    text = f" {message.lower()} "

    scores = {
        topic: sum(1 for w in words if w in text)
        for topic, words in TOPIC_KEYWORDS.items()
    }
    # "subsidy for drip irrigation" is about schemes, not dams
    if scores["schemes"]:
        scores["schemes"] += 10
    best = max(scores, key=scores.get)
    topic = best if scores[best] else None

    # Location: a Tamil Nadu district (English or Tamil), or a Capitalised name after "in/at/near/for"
    district = tn_agri.find_district_in_text(message)
    location = district["name"] if district else ""
    if not location:
        match = re.search(r"\b(?:in|at|near|for|around)\s+([A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+){0,2})", message)
        if match and match.group(1).lower() not in STOP_WORDS | TOPIC_WORDS:
            location = match.group(1).strip(" .")

    terms = [
        w for w in re.findall(r"[a-z]{4,}", text)
        if w not in STOP_WORDS and not any(w.startswith(t.strip()) for t in TOPIC_WORDS)
    ]
    return {"topic": topic, "location": location, "search_terms": terms}


def understand(query_llm, message):
    try:
        return understand_with_llm(query_llm, message)
    except Exception as e:
        print(f"chat: Gemini intent failed, using keywords: {e}", flush=True)
        return understand_with_keywords(message)


# ============================================================
# Answers
# ============================================================
def district_name(place, language):
    """District (or place) name in the answer language."""
    if not place:
        return ""
    district = place.get("tn_district")
    if district and language == "ta":
        return tn_agri.DISTRICT_NAMES_TA.get(district, district)
    return district or place.get("name") or ""


def place_label(place, language="en"):
    if not place:
        return ""
    if language == "ta" and place.get("tn_district"):
        # Village / town names from the map are only in English
        name = place.get("name")
        parts = [name if name != place.get("tn_district") else None, district_name(place, "ta"), "தமிழ்நாடு"]
    else:
        parts = [place.get("name"), place.get("tn_district") or place.get("district"), place.get("state")]
    return ", ".join(dict.fromkeys(p for p in parts if p))


def match_subsidies(language, search_terms, limit=20):
    """Subsidy items matching the user's keywords (matched on the English catalogue)."""
    english = tn_agri.get_schemes_and_subsidies("en")["subsidies"]
    shown = tn_agri.get_schemes_and_subsidies(language)["subsidies"] if language != "en" else english

    terms = [t for t in search_terms if t not in STOP_WORDS and len(t) >= 3]
    if not terms:
        return []

    scored = []
    for i, item in enumerate(english):
        haystack = f"{item['component']} {item['sub_scheme']} {item['scheme_name']}".lower()
        score = sum(1 for t in terms if t in haystack)
        if score:
            scored.append((score, i))
    scored.sort(key=lambda x: -x[0])
    return [shown[i] for _, i in scored[:limit]]


def compact_data(topic, data, matched=None):
    """Small version of the section data for the answer prompt."""
    if topic == "climate":
        c = data["climate"]
        return {"season": c["season"], "current": c["current"],
                "rainfall_last_30_days_mm": c["rainfall_last_30_days_mm"],
                "rainfall_next_7_days_mm": c["rainfall_next_7_days_mm"], "forecast": c["forecast"][:3]}

    if topic == "soil":
        return {k: v for k, v in data["soil"].items() if k not in ("sampled_at", "source", "health_source")}

    if topic == "water":
        w = data["water"]
        return {
            "as_on_date": w["as_on_date"], "availability": w["availability"],
            "average_level_percent": w["average_level_percent"], "basis": w["availability_basis"],
            "rainfall_last_30_days_mm": w.get("rainfall_last_30_days_mm"),
            "reservoirs": [
                {k: d.get(k) for k in ("name", "name_ta", "river", "current_level_ft", "full_depth_ft", "level_percent",
                                   "inflow_cusecs", "outflow_cusecs", "serves_your_district", "distance_km")}
                for d in w["reservoirs"][:6]
            ],
        }

    if topic == "seeds":
        s = data["seeds"]
        return {
            "district": s["district"],
            "crops_available": [
                {"crop": c["crop"], "varieties": [v["variety"] for v in c["varieties"][:6]]}
                for c in s["crops_available"]
            ],
            "nearest_depots": [
                {"name": d["depot_name"], "distance_km": d["distance_km"], "phone": d["phone"],
                 "stock": [f"{x['crop']} {x['variety']} - Rs.{x['price_rs']}/{x['price_per']}, {x['quantity']} {x['unit']}"
                           for x in d["stock"][:8]]}
                for d in s["nearest_depots"][:3]
            ],
        }

    if topic == "fertilizer":
        f = data["fertilizer"]
        key_prices = [
            {"product": p["product"], "bag_kg": p["bag_size_kg"], "price_rs": p["prices"][0]["price_rs"]}
            for p in f["prices"] if p["product"] in ("UREA", "DAP", "MOP", "NPK 17:17:17 Complex")
        ]
        return {
            "district": f["district"], "fertilizer_inspector_phone": f["fertilizer_inspector_phone"],
            "nearest_dealers": [
                {k: d[k] for k in ("dealer_name", "distance_km", "phone", "stock_kg")}
                for d in f["nearest_dealers"][:5]
            ],
            "prices": key_prices,
        }

    if topic == "schemes":
        schemes = data["schemes"]["items"]
        return {
            "total_schemes": len(schemes),
            "total_subsidy_items": data["subsidies"]["count"],
            "matching_subsidies": [
                {"scheme": m["scheme_name"], "sub_scheme": m["sub_scheme"], "component": m["component"]}
                for m in (matched or [])[:10]
            ],
            "scheme_names": [] if matched else [f"{s['scheme_name']} ({s['department']})" for s in schemes],
        }

    return data


WEATHER_TA = {
    "Clear sky": "தெளிவான வானம்", "Mainly clear": "பெரும்பாலும் தெளிவு", "Partly cloudy": "பகுதி மேகமூட்டம்",
    "Overcast": "மேகமூட்டம்", "Fog": "மூடுபனி", "Light drizzle": "லேசான தூறல்", "Drizzle": "தூறல்",
    "Heavy drizzle": "கனமான தூறல்", "Light rain": "லேசான மழை", "Rain": "மழை", "Heavy rain": "கனமழை",
    "Light showers": "லேசான சாரல் மழை", "Showers": "சாரல் மழை", "Heavy showers": "கனமான சாரல் மழை",
    "Thunderstorm": "இடியுடன் கூடிய மழை", "Thunderstorm with hail": "ஆலங்கட்டியுடன் இடி மழை",
}
LEVEL_TA = {"High": "அதிகம்", "Medium": "நடுத்தரம்", "Low": "குறைவு", "Unknown": "தெரியவில்லை",
            "Good": "நல்லது", "Moderate": "மிதமானது", "Poor": "மோசம்"}

# Fixed answer text used when Gemini is unavailable: {language: {key: text}}
TEMPLATES = {
    "en": {
        "weather": "Current weather in {place}: {temp}°C, {condition}, humidity {humidity}%, wind {wind} km/h.",
        "today": "Today: {tmin}–{tmax}°C, rain chance {rain_chance}%.",
        "rain7": "Expected rain in the next 7 days: {rain} mm.",
        "soil": "Soil near {place}: {texture}, pH {ph}, organic carbon {oc} g/kg, nitrogen {n} g/kg.",
        "soil_health": "Health: {status} – {summary}",
        "water": "Water availability for {place}: {level} (average dam level {avg}% as on {date}).",
        "dam": "• {name}: {level} / {full} ft ({pct}%), inflow {inflow} cusecs, outflow {outflow} cusecs",
        "seeds": "Seeds available in {place} district:",
        "depot": "Nearest depot: {name} ({km} km), phone {phone}.",
        "dealers": "Nearest fertilizer dealers:",
        "inspector": "Fertilizer inspector: {phone}",
        "matching": "Matching subsidies:",
        "schemes": "{count} schemes are available, for example:",
    },
    "ta": {
        "weather": "{place} தற்போதைய வானிலை: {temp}°C, {condition}, ஈரப்பதம் {humidity}%, காற்று {wind} கி.மீ/மணி.",
        "today": "இன்று: {tmin}–{tmax}°C, மழை வாய்ப்பு {rain_chance}%.",
        "rain7": "அடுத்த 7 நாட்களில் எதிர்பார்க்கப்படும் மழை: {rain} மி.மீ.",
        "soil": "{place} மண்: {texture}, pH {ph}, கரிமக் கார்பன் {oc} கி/கிலோ, நைட்ரஜன் {n} கி/கிலோ.",
        "soil_health": "மண் வளம்: {status} – {summary}",
        "water": "{place} நீர் இருப்பு: {level} (சராசரி அணை நீர்மட்டம் {avg}%, {date} நிலவரப்படி).",
        "dam": "• {name}: {level} / {full} அடி ({pct}%), நீர்வரத்து {inflow} கன அடி/வி, வெளியேற்றம் {outflow} கன அடி/வி",
        "seeds": "{place} மாவட்டத்தில் கிடைக்கும் விதைகள்:",
        "depot": "அருகிலுள்ள கிடங்கு: {name} ({km} கி.மீ), தொலைபேசி {phone}.",
        "dealers": "அருகிலுள்ள உர விற்பனையாளர்கள்:",
        "inspector": "உர ஆய்வாளர்: {phone}",
        "matching": "பொருந்தும் மானியங்கள்:",
        "schemes": "{count} திட்டங்கள் உள்ளன, உதாரணமாக:",
    },
}


def template_answer(topic, data, place, language, matched=None):
    """Answer built from fixed text, used when Gemini is unavailable."""
    t = TEMPLATES[language]
    ta = language == "ta"
    name = place_label(place, language)
    district = district_name(place, language)

    if topic == "climate":
        c = data["climate"]["current"]
        f = data["climate"]["forecast"][0] if data["climate"]["forecast"] else {}
        condition = WEATHER_TA.get(c["condition"], c["condition"]) if ta else c["condition"]
        return "\n".join([
            t["weather"].format(place=name, temp=c["temperature_c"], condition=condition,
                                humidity=c["humidity_percent"], wind=c["wind_speed_kmh"]),
            t["today"].format(tmin=f.get("temp_min_c"), tmax=f.get("temp_max_c"),
                              rain_chance=f.get("rain_probability_percent")),
            t["rain7"].format(rain=data["climate"]["rainfall_next_7_days_mm"]),
        ])

    if topic == "soil":
        s = data["soil"]
        lines = [t["soil"].format(place=name, texture=s.get("texture"), ph=s.get("ph"),
                                  oc=s.get("organic_carbon_g_per_kg"), n=s.get("nitrogen_g_per_kg"))]
        if s.get("health"):
            status = s["health"].get("status", "")
            lines.append(t["soil_health"].format(status=LEVEL_TA.get(status, status) if ta else status,
                                                 summary=s["health"].get("summary", "")))
        return "\n".join(lines)

    if topic == "water":
        w = data["water"]
        level = LEVEL_TA.get(w["availability"], w["availability"]) if ta else w["availability"]
        lines = [t["water"].format(place=district, level=level, avg=w["average_level_percent"], date=w["as_on_date"])]
        for d in w["reservoirs"][:4]:
            dam = d.get("name_ta", d["name"]) if ta else d["name"]
            lines.append(t["dam"].format(name=dam, level=d["current_level_ft"], full=d["full_depth_ft"],
                                         pct=d["level_percent"], inflow=d["inflow_cusecs"], outflow=d["outflow_cusecs"]))
        return "\n".join(lines)

    if topic == "seeds":
        s = data["seeds"]
        lines = [t["seeds"].format(place=district)]
        for c in s["crops_available"][:6]:
            lines.append(f"• {c['crop']}: " + ", ".join(v["variety"] for v in c["varieties"][:4]))
        if s["nearest_depots"]:
            d = s["nearest_depots"][0]
            lines.append(t["depot"].format(name=d["depot_name"], km=d["distance_km"], phone=d["phone"]))
        return "\n".join(lines)

    if topic == "fertilizer":
        f = data["fertilizer"]
        lines = [t["dealers"]]
        for d in f["nearest_dealers"][:3]:
            stock = ", ".join(f"{k} {v} kg" for k, v in list(d["stock_kg"].items())[:3])
            lines.append(f"• {d['dealer_name']} ({d['distance_km']} km, {d['phone']}): {stock}")
        if f["fertilizer_inspector_phone"]:
            lines.append(t["inspector"].format(phone=f["fertilizer_inspector_phone"]))
        return "\n".join(lines)

    if topic == "schemes":
        if matched:
            lines = [t["matching"]] + [f"• {m['component']} ({m['scheme_name']})" for m in matched[:6]]
        else:
            schemes = data["schemes"]["items"]
            lines = [t["schemes"].format(count=len(schemes))] + [f"• {x['scheme_name']}" for x in schemes[:8]]
        return "\n".join(lines)

    return ""


LANGUAGE_RULES = {
    "en": "Write the whole answer in English.",
    "ta": ("Write the whole answer in Tamil (தமிழ்), in simple words a farmer understands. "
           "Write place, crop, dam and scheme names in Tamil where a common Tamil name exists. "
           "Keep numbers, units, phone numbers and variety codes (e.g. ADT 45, CO 51) as they are."),
}


def write_answer(query_llm, topic, data, place, question, language, matched=None):
    source = {
        "climate": "Open-Meteo", "soil": "ISRIC SoilGrids",
        "water": "Tamil Nadu Agriculture Department reservoir data",
        "seeds": "Tamil Nadu Agriculture Department seed depots",
        "fertilizer": "Tamil Nadu Agriculture Department fertilizer finder",
        "schemes": "Tamil Nadu Agriculture Department subsidy schemes",
    }[topic]

    prompt = f"""
You are a friendly agricultural assistant for farmers in Tamil Nadu.

Farmer's question: {question}
Location: {place_label(place, language)}
Data ({source}):
{json.dumps(compact_data(topic, data, matched), ensure_ascii=False)}

Answer in 3 to 6 short lines of plain text (you may use "•" bullets).
Use ONLY the data above and include the key numbers with units.
Do not invent anything. If something is missing, say so briefly.
Do not add greetings or mention the data source.
{LANGUAGE_RULES[language]}
"""
    try:
        return query_llm(prompt).strip()
    except Exception as e:
        print(f"chat: Gemini answer failed, using template: {e}", flush=True)
        return template_answer(topic, data, place, language, matched)


# ============================================================
# Suggestions
# ============================================================
def build_suggestions(place, answered, current_topic, language):
    # Outside Tamil Nadu only weather and soil data exist
    in_tn = bool(place and place.get("tn_district"))
    available = [t for t in SUGGESTION_TOPICS if in_tn or t not in TN_ONLY_TOPICS]

    remaining = [t for t in available if t not in answered and t != current_topic]
    if not remaining:
        remaining = [t for t in available if t != current_topic]

    name = district_name(place, language)
    suggestions = []
    for topic in remaining:
        _, label_en, label_ta, q_en, q_ta = TOPICS[topic]
        question = (q_ta if language == "ta" else q_en).format(place=name).replace(" in ?", "?")
        suggestions.append({"id": topic, "label": label_ta if language == "ta" else label_en, "question": question})
    return suggestions


# ============================================================
# Blueprint
# ============================================================
def create_chat_blueprint(query_llm):
    bp = Blueprint("chat_api", __name__)

    @bp.route("/chat", methods=["POST"])
    def chat():
        data = (
            request.get_json(silent=True, force=True) or
            request.form.to_dict() or
            request.args.to_dict()
        )

        message = (data.get("message") or "").strip()
        # "id" is accepted too: it is the key used in the suggestions list
        suggestion_id = str(data.get("suggestion_id") or data.get("id") or "").strip().lower()
        if not message and not suggestion_id:
            return jsonify({"error": "Send 'message' (typed question) or 'suggestion_id' (tapped suggestion)."}), 400
        if suggestion_id and suggestion_id not in TOPICS:
            return jsonify({"error": f"Unknown suggestion_id '{suggestion_id}'. Allowed: {list(TOPICS)}."}), 400

        session_id, session = get_session((data.get("session_id") or "").strip())

        # Reply in the language of the message: Tamil question -> Tamil, English -> English.
        # A tapped suggestion (no text) keeps the language of the conversation.
        if message:
            language = detect_language(message)
        else:
            language = (data.get("language") or session["language"] or "en").strip().lower()
        if language not in SUPPORTED_LANGUAGES:
            return jsonify({"error": f"'language' must be one of {list(SUPPORTED_LANGUAGES)}."}), 400
        session["language"] = language

        def reply(answer, topic=None, place=None, section_data=None, errors=None, needs_location=False):
            place = place or session["place"]
            return jsonify({
                "session_id": session_id,
                "language": language,
                "topic": topic,
                "location": place,
                "answer": answer,
                "data": section_data,
                "errors": errors or {},
                "needs_location": needs_location,
                "suggestions": [] if needs_location else build_suggestions(place, session["answered"], topic, language),
            })

        # --------------------------------------------------------
        # 1. Understand: topic + location
        # --------------------------------------------------------
        if suggestion_id:
            intent = {"topic": suggestion_id, "location": "", "search_terms": []}
        else:
            intent = understand(query_llm, message)

        topic = intent["topic"] or session["pending_topic"]

        # --------------------------------------------------------
        # 2. Location: message > explicit request > session > request GPS
        # --------------------------------------------------------
        place = None
        try:
            if intent["location"]:
                results = search_places(intent["location"], limit=1)
                if not results:
                    return reply(NOT_FOUND_TEXT[language].format(location=intent["location"]), topic)
                place = dict(results[0], type="custom")
            elif data.get("location_type") and (data.get("latitude") is not None or data.get("location")):
                place = resolve_location(data)
            elif session["place"]:
                place = session["place"]
            elif data.get("latitude") is not None or data.get("location"):
                place = resolve_location(data)
        except (ValueError, LocationError) as e:
            return jsonify({"error": str(e)}), 400
        except requests.RequestException as e:
            return jsonify({"error": f"Location service unavailable: {e}"}), 502

        if not place:
            session["pending_topic"] = topic
            if not topic:
                return reply(GREETING_TEXT[language].format(about=""), None)
            return reply(NEEDS_LOCATION_TEXT[language], topic, needs_location=True)

        session["place"] = place
        session["pending_topic"] = None

        attach_tn_district(place)

        if not topic:
            about = f" about {place_label(place)}" if language == "en" else f"{place_label(place, 'ta')} பற்றி "
            return reply(GREETING_TEXT[language].format(about=about), None, place)

        # --------------------------------------------------------
        # 3. Fetch the data for this topic
        # --------------------------------------------------------
        sections = set(TOPICS[topic][0])
        section_data, errors = build_insights(query_llm, place, sections, language)
        label = TOPICS[topic][2] if language == "ta" else TOPICS[topic][1]

        if topic in TN_ONLY_TOPICS and not place.get("tn_district"):
            return reply(TN_ONLY_TEXT[language].format(label=label, place=place_label(place, language)),
                         topic, place, None, errors)

        primary = section_data.get(TOPICS[topic][0][0])
        if not primary:
            return reply(FAILED_TEXT[language].format(label=label), topic, place, section_data, errors)

        # --------------------------------------------------------
        # 4. Answer + follow-up suggestions
        # --------------------------------------------------------
        matched = None
        if topic == "schemes":
            matched = match_subsidies(language, intent["search_terms"])
            # Keep the chat response small: only matching subsidy items
            section_data["subsidies"] = dict(section_data["subsidies"], items=matched)

        question = message or TOPICS[topic][4 if language == "ta" else 3].format(
            place=district_name(place, language))
        answer = write_answer(query_llm, topic, section_data, place, question, language, matched)

        if topic not in session["answered"]:
            session["answered"].append(topic)

        return reply(answer, topic, place, section_data, errors)

    return bp
