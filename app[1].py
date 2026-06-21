from flask import Flask, request, jsonify, render_template_string
import os
import json
import statistics
import difflib
import urllib.parse
import urllib.request
from anthropic import Anthropic
from optimization import solve_sensor_placement

# .env dosyasını app.py ile aynı klasörden zorla oku
try:
    from dotenv import load_dotenv
    import os 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ENV_PATH = os.path.join(BASE_DIR, ".env")
    
    load_dotenv(dotenv_path=ENV_PATH, override=True)
except ImportError:
    pass

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# =======================
#  UYGULAMA AYARLARI
# =======================
APP_NAME = "Smart Agriculture Assistant"

# Sabit sensör tipleri (f_ij matrisi için)
SENSOR_TYPES = ["pH", "Moisture", "EC", "SoilTemp", "NDVI", "NPK"]

# =======================
#  LLM (GPT-4.1-mini)
# =======================

def llm_answer(message: str, lang: str, state: dict, sensors: dict):
    system_msg = f"""
You are a Smart Agriculture Decision Support AI.

Rules:
- Answer only the user's current question.
- Do not repeat previous answers.
- Do not give long reports.
- Keep answers short and step-based.
- Use maximum 5 bullet points.
- Always follow the user's current intent.
- Language: {lang}

Context:
- Polygon: {state.get("polygon")}
- Targets: {state.get("targets")}
- Sensor data: {sensors}
"""


    print("🔥 [LLM] llm_answer fonksiyonuna girildi", flush=True)
    print("🔥 [LLM] API KEY var mı?:", bool(ANTHROPIC_API_KEY), flush=True)
    

    if not ANTHROPIC_API_KEY or not ANTHROPIC_API_KEY.startswith("sk-ant-"):
        print("❌ [LLM] ANTHROPIC_API_KEY yok veya geçersiz", flush=True)
        return None

    try:
        print("🚀 [LLM] Claude API çağrısı yapılacak...", flush=True)
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # ... senin mevcut kod aynı kalsın ...

        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=350,
            temperature=0.35,
            system=system_msg,
            messages=[{"role": "user", "content": message}],
        )

        if not resp.content or not resp.content[0].text.strip():
          return None

        text = resp.content[0].text.strip()
        return text

    except Exception as e:
        print("⚠️ [LLM] Claude API hatası:", repr(e), flush=True)
        return None



# =======================
#  Tiny agronomy helpers
# =======================

# ---------- In-memory state ----------
STATE = {
    "polygon": [],
    "targets": [],
    "location_label": "",
    "last_geocode": None,
    "sensor_matrix": None,
    "sensor_types": SENSOR_TYPES,
    "advisor_step": None,
    "recommended_sensors": None,
    "pending_optimization": False,
    "selected_sensors": None,

    # Chat step sistemi
    "chat_step": 0,
    "last_question": "",
    "last_answer": "",
}

CROPS_TR = [
    "armut",
    "elma",
    "zeytin",
    "pamuk",
    "mısır",
    "buğday",
    "arpa",
    "ayçiçeği",
    "soya",
    "çeltik",
    "üzüm",
    "kiraz",
    "şeftali",
    "fındık",
    "antepfıstığı",
    "patates",
    "domates",
    "biber",
]
CROPS_EN = [
    "pear",
    "apple",
    "olive",
    "cotton",
    "maize",
    "corn",
    "wheat",
    "barley",
    "sunflower",
    "soy",
    "soybean",
    "paddy",
    "rice",
    "grape",
    "cherry",
    "peach",
    "hazelnut",
    "pistachio",
    "potato",
    "tomato",
    "pepper",
]


def fuzzy_crop(q, lang):
    q = (q or "").lower()
    vocab = list(set(CROPS_TR + CROPS_EN))
    best, score = None, 0
    for tok in q.replace(",", " ").split():
        cand = difflib.get_close_matches(tok, vocab, n=1, cutoff=0.78)
        if cand:
            s = difflib.SequenceMatcher(None, tok, cand[0]).ratio()
            if s > score:
                best, score = cand[0], s
    return best


def state_centroid(state=None):
    st = state or STATE
    if st["polygon"]:
        lat = statistics.mean(p[0] for p in st["polygon"])
        lng = statistics.mean(p[1] for p in st["polygon"])
        return (lat, lng)
    if st["targets"]:
        lat = statistics.mean(t["lat"] for t in st["targets"])
        lng = statistics.mean(t["lng"] for t in st["targets"])
        return (lat, lng)
    return None


def fmt_coords(lat, lng):
    return f"{lat:.5f}, {lng:.5f}"


def soil_signals(ph, moist, temp, ndvi, lang):
    out = []
    if ph is not None:
        if ph < 6:
            out.append(
                "pH düşük → kireçleme/kompost."
                if lang == "tr"
                else "Low pH → consider liming/compost."
            )
        elif ph > 7.5:
            out.append(
                "pH yüksek → elementel kükürt/organik madde."
                if lang == "tr"
                else "High pH → elemental sulfur/organic matter."
            )
        else:
            out.append(
                "pH 6.0–7.5 aralığında." if lang == "tr" else "pH in 6.0–7.5 range."
            )
    if moist is not None:
        if moist < 30:
            out.append(
                "Nem <%30 → damla sulama + malç."
                if lang == "tr"
                else "Moisture <30% → drip + mulch."
            )
        elif moist > 70:
            out.append(
                "Nem yüksek → sulamayı azalt, drenaj."
                if lang == "tr"
                else "High moisture → reduce irrigation, improve drainage."
            )
    if temp is not None:
        if temp < 5:
            out.append(
                "Düşük sıcaklık → don riski."
                if lang == "tr"
                else "Low temperature → frost risk."
            )
        elif temp > 35:
            out.append(
                "Yüksek sıcaklık → gölgeleme."
                if lang == "tr"
                else "High temperature → shading."
            )
    if ndvi is not None:
        if ndvi < 0.3:
            out.append(
                "NDVI düşük → bitki örtüsü zayıf / stresli."
                if lang == "tr"
                else "Low NDVI → weak/stressed vegetation."
            )
        elif ndvi < 0.6:
            out.append(
                "NDVI orta → gelişim idare eder, izlenmeli."
                if lang == "tr"
                else "Medium NDVI → average vegetation, monitor."
            )
        else:
            out.append(
                "NDVI yüksek → bitki örtüsü güçlü."
                if lang == "tr"
                else "High NDVI → strong vegetation."
            )
    return out


def craft_answer(msg, sensors, lang):
    """
    LLM çalışmazsa devreye giren lokal açıklama motoru.
    """

    def fget(k):
        v = sensors.get(k)
        try:
            return float(v) if v not in (None, "") else None
        except Exception:
            return None

    ph = fget("ph")
    moist = fget("moisture")
    temp = fget("temperature")
    ndvi = fget("ndvi")

    crop = fuzzy_crop(msg, lang)
    c = state_centroid()
    tgt_cnt = len(STATE["targets"])
    has_poly = len(STATE["polygon"]) >= 3
    loc = STATE["location_label"] or ("—" if lang == "tr" else "—")
    sig = soil_signals(ph, moist, temp, ndvi, lang)

    corners = STATE["polygon"][:5]  # ilk 5 köşe

    ask_grow = ("yetişir mi" in (msg or "").lower()) or (
        "grow" in (msg or "").lower()
    )
    ask_rec = (
        ("ne önerirsin" in (msg or "").lower())
        or ("recommend" in (msg or "").lower())
        or ("best" in (msg or "").lower())
    )

    L = []

    if lang == "tr":
        L += [
            "— Tarımsal Değerlendirme (Yerel Motor) —",
            f"• Konum etiketi: {loc}",
            f"• Bölge merkezi: {fmt_coords(*c) if c else '—'}",
            f"• Bölge çokgeni: {'var' if has_poly else 'yok'}.",
            f"• Hedef noktalar: {tgt_cnt} adet.",
        ]
        if corners:
            L.append("• Bölge köşe noktaları (ilk 5):")
            for i, (la, ln) in enumerate(corners, 1):
                L.append(f"   {i}) {fmt_coords(la, ln)}")
        if STATE["targets"]:
            L.append("• Seçili hedefler:")
            for i, t in enumerate(STATE["targets"], 1):
                lab = t.get("label") or "(map)"
                L.append(f"   {i}) {lab} → {fmt_coords(t['lat'], t['lng'])}")
        if crop:
            crop_tr = (
                crop
                if crop in CROPS_TR
                else {
                    "corn": "mısır",
                    "maize": "mısır",
                    "wheat": "buğday",
                    "barley": "arpa",
                    "sunflower": "ayçiçeği",
                    "soybean": "soya",
                    "rice": "çeltik",
                    "paddy": "çeltik",
                }.get(crop, crop)
            )
            L.append(f"• Odak ürün: {crop_tr.title()}.")

        if sig:
            L += ["", "— Toprak/Sulama / Bitki Sinyalleri —"]
            L += [f"• {s}" for s in sig]

        L += ["", "— Eylem Planı —"]
        if ask_grow and crop:
            L += [
                "1) pH 6–7 ve iyi drenaj doğrulansın.",
                "2) Sezon başında %2–3 organik madde (kompost) + malç uygulaması.",
                "3) 2–3 hedef noktada nem sensörü ile günlük kayıt.",
                "4) NDVI düşük ise sulama / gübreleme stratejisini gözden geçir.",
            ]
        elif ask_rec:
            L += [
                "1) pH/EC/organik madde analizi ile ürün adaylarını netleştir (örnek: buğday, ayçiçeği, mısır).",
                "2) Mikro-topografyaya göre tarlayı alt bölgelere ayır ve sulama hat tasarımını buna göre yap.",
                "3) Haftalık sulama planı; nemi %40–60 aralığında tutmaya çalış.",
                "4) NDVI takibi ile verimi düşük kalan alt bölgeleri tespit et.",
            ]
        else:
            L += [
                "1) 0–30 cm toprak örneği: pH, EC, organik madde analizi.",
                "2) Haftalık sulama planı; nemi %40–60 aralığında tut.",
                "3) Yüzey akışı ve drenajı (taban suyu) kontrol et.",
                "4) Aylık sensör değerlendirmesi ve NDVI karşılaştırması ile planı güncelle.",
            ]

        if not (ph and moist and temp):
            L += [
                "",
                "Eksik veri algılandı → Daha hassas öneriler için lütfen şunları gir:",
                "• pH (örn: 6.5), Nem % (örn: 40), Sıcaklık °C (örn: 22), NDVI (örn: 0.55).",
            ]

        L += [
            "",
            "💡 İsterseniz bir sonraki adımda hangi sensörlerin nereye yerleştirileceğine bakabiliriz.",
        ]

        return "\n".join(L)

    # ---- English branch ----
    L += [
        "— Agronomic Assessment (Local Fallback) —",
        f"• Location label: {loc}",
        f"• Region centroid: {fmt_coords(*c) if c else '—'}",
        f"• Region polygon: {'set' if has_poly else 'not set'}.",
        f"• Target points: {tgt_cnt}.",
    ]
    if corners:
        L.append("• Region corner points (first 5):")
        for i, (la, ln) in enumerate(corners, 1):
            L.append(f"   {i}) {fmt_coords(la, ln)}")
    if STATE["targets"]:
        L.append("• Selected targets:")
        for i, t in enumerate(STATE["targets"], 1):
            lab = t.get("label") or "(map)"
            L.append(f"   {i}) {lab} → {fmt_coords(t['lat'], t['lng'])}")
    if crop:
        L.append(f"• Focus crop: {crop.title()}.")

    if sig:
        L += ["", "— Soil / Irrigation / Vegetation Signals —"]
        L += [f"• {s}" for s in sig]

    L += ["", "— Action Plan —"]
    if ask_grow and crop:
        L += [
            f"1) Verify pH 6–7 and good drainage for {crop.title()}.",
            "2) Apply 2–3% organic matter + mulch at season start.",
            "3) Daily moisture logging at 2–3 target points.",
            "4) If NDVI is low, revisit irrigation and fertiliser strategy.",
        ]
    elif ask_rec:
        L += [
            "1) Use pH/EC/OM to shortlist candidate crops (e.g., wheat, sunflower, maize).",
            "2) Subdivide the field by micro-topography and design irrigation laterals accordingly.",
            "3) Weekly irrigation plan; keep soil moisture around 40–60%.",
            "4) Track NDVI by sub-region to detect underperforming zones.",
        ]
    else:
        L += [
            "1) Take 0–30 cm soil samples for pH, EC and organic matter.",
            "2) Set a weekly irrigation plan; keep moisture in the 40–60% band.",
            "3) Check drainage (surface runoff and groundwater conditions).",
            "4) Review sensors and NDVI monthly and adjust the plan.",
        ]

    if not (ph and moist and temp):
        L += [
            "",
            "Missing data detected → For more precise advice, please provide:",
            "• pH (e.g., 6.5), Moisture % (e.g., 40), Temperature °C (e.g., 22), NDVI (e.g., 0.55).",
        ]

    L += [
        "",
        "💡 When ready, the next step covers which sensors to place and where.",
    ]

    return "\n".join(L)


# =======================
#  OpenStreetMap / Nominatim
# =======================
NOMINATIM = "https://nominatim.openstreetmap.org"
UA = "SmartAgriAssistant/3.0 (edu-demo)"


def nominatim_get(path, params):
    params = {**params, "format": "json"}
    url = f"{NOMINATIM}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------- Static countries list (A–Z) ----------
COUNTRIES = sorted(
    [
        "Afghanistan",
        "Albania",
        "Algeria",
        "Andorra",
        "Angola",
        "Argentina",
        "Armenia",
        "Australia",
        "Austria",
        "Azerbaijan",
        "Bahamas",
        "Bahrain",
        "Bangladesh",
        "Belarus",
        "Belgium",
        "Belize",
        "Benin",
        "Bhutan",
        "Bolivia",
        "Bosnia and Herzegovina",
        "Botswana",
        "Brazil",
        "Brunei",
        "Bulgaria",
        "Burkina Faso",
        "Burundi",
        "Cambodia",
        "Cameroon",
        "Canada",
        "Chile",
        "China",
        "Colombia",
        "Costa Rica",
        "Croatia",
        "Cuba",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "Dominican Republic",
        "Ecuador",
        "Egypt",
        "El Salvador",
        "Estonia",
        "Ethiopia",
        "Finland",
        "France",
        "Georgia",
        "Germany",
        "Ghana",
        "Greece",
        "Greenland",
        "Guatemala",
        "Honduras",
        "Hungary",
        "Iceland",
        "India",
        "Indonesia",
        "Iran",
        "Iraq",
        "Ireland",
        "Israel",
        "Italy",
        "Jamaica",
        "Japan",
        "Jordan",
        "Kazakhstan",
        "Kenya",
        "Kuwait",
        "Kyrgyzstan",
        "Laos",
        "Latvia",
        "Lebanon",
        "Libya",
        "Liechtenstein",
        "Lithuania",
        "Luxembourg",
        "Madagascar",
        "Malaysia",
        "Maldives",
        "Mali",
        "Malta",
        "Mexico",
        "Moldova",
        "Monaco",
        "Mongolia",
        "Montenegro",
        "Morocco",
        "Mozambique",
        "Namibia",
        "Nepal",
        "Netherlands",
        "New Zealand",
        "Nicaragua",
        "Nigeria",
        "North Macedonia",
        "Norway",
        "Oman",
        "Pakistan",
        "Palestine",
        "Panama",
        "Paraguay",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Qatar",
        "Romania",
        "Russia",
        "Rwanda",
        "Saudi Arabia",
        "Senegal",
        "Serbia",
        "Singapore",
        "Slovakia",
        "Slovenia",
        "Somalia",
        "South Africa",
        "South Korea",
        "Spain",
        "Sri Lanka",
        "Sudan",
        "Sweden",
        "Switzerland",
        "Syria",
        "Taiwan",
        "Tajikistan",
        "Tanzania",
        "Thailand",
        "Tunisia",
        "Turkey",
        "Turkmenistan",
        "Ukraine",
        "United Arab Emirates",
        "United Kingdom",
        "United States",
        "Uruguay",
        "Uzbekistan",
        "Venezuela",
        "Vietnam",
        "Yemen",
        "Zambia",
        "Zimbabwe",
    ]
)


# =======================
#  HTML (UI) – SENİN GÖRÜNÜMÜ KORUDUM 💚
# =======================
app = Flask(__name__)

HTML = r"""{% raw %}
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Smart Agriculture Assistant</title>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>

<style>
:root{
  --bg:#171411; --panel:#1f1b16; --sunken:#221e19; --line:#3b332c; --txt:#f3f0ea;
  --brand:#7cc98a; --brand2:#5aa974; --danger:#d06b6b; --muted:#c9c2b9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-family:Inter,ui-sans-serif,-apple-system,Segoe UI,Roboto,Arial}

header{position:sticky;top:0;z-index:900;display:flex;align-items:center;gap:12px;
  padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.05);background:linear-gradient(180deg,#1c1814,#171411)}
.brand{display:flex;align-items:center;gap:10px}
.brand .logo{font-size:20px}
.brand .title{
  color:var(--brand);
  font-weight:800;
  font-size:20px;
  text-shadow:0 0 12px rgba(124,201,138,0.35);
}
.badge{font-size:12px;background:#203324;border:1px solid #2d4736;color:#9fe2b0;border-radius:999px;padding:2px 8px}
.lang{margin-left:auto;display:flex;gap:6px}
.lang button{border:1px solid var(--line);background:#2a251f;color:#fff;border-radius:10px;padding:6px 10px;cursor:pointer}
.lang button.active{background:var(--brand);color:#111}

.container{padding:12px}

/* Map area */
.mapwrap{position:relative;border:1px solid var(--line);border-radius:22px;overflow:hidden;background:#000}
#map{height:56vh;width:100%}
.stack{
  position:absolute;
  left:12px;
  top:12px;
  display:flex;
  flex-direction:column;
  gap:10px;
  z-index:650;
  padding:6px;
  border-radius:16px;
  background:rgba(18,18,18,0.32);
  backdrop-filter:blur(14px);
  border:1px solid rgba(255,255,255,0.06);
  box-shadow:0 10px 30px rgba(0,0,0,0.35);
}
.stack{
  max-width:220px;
}

.stack .btn{
  min-width:190px;
}
.btn{
  border:none;
  border-radius:12px;
  padding:7px 11px;
  font-size:14px;
  font-weight:800;
  cursor:pointer;
  transition:background 0.15s, box-shadow 0.15s, transform 0.08s;
}

.btn:hover{
  transform:translateY(-2px) scale(1.01);
  box-shadow:0 6px 16px rgba(0,0,0,0.25);
}

.btn.mode-active{box-shadow:0 0 0 2px var(--brand2) inset;filter:brightness(1.08);}
/* 🌌 HOLOGRAM-GLOW MODE EFFECT */
.btn.mode-active {
  border:1px solid rgba(124,201,138,.75);
  background:linear-gradient(180deg,#0f5c38,#0a3d27);
  color:#ecfff1 !important;
  box-shadow:
    0 0 10px rgba(124,201,138,.42),
    0 0 18px rgba(124,201,138,.16),
    inset 0 0 10px rgba(124,201,138,.18);
}

.green{
  background:linear-gradient(180deg,#8fe29b,#6fcb82);
  color:#08110b;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.25);
}
.dark{background:#2b251f;color:#eee;border:1px solid var(--line)}
.red{background:var(--danger);color:#fff}

/* zoom control bottom-left */
.leaflet-control-zoom{
  bottom:2px !important;
  right:22px !important;
  left:auto !important;
  top:auto !important;

  display:flex !important;
  flex-direction:row !important;
  border:none !important;
  box-shadow:0 8px 20px rgba(0,0,0,0.35) !important;
}

.leaflet-control-zoom a{
  width:24px !important;
  height:24px !important;
  line-height:24px !important;
  font-size:14px !important;
  border:none !important;
}

.leaflet-control-zoom-in{
  border-radius:10px 0 0 10px !important;
}

.leaflet-control-zoom-out{
  border-radius:0 10px 10px 0 !important;
}
.leaflet-right .leaflet-control{margin-right:12px}
.leaflet-bottom .leaflet-control{margin-bottom:12px}

/* Query section */
.section{margin-top:14px;border:1px solid var(--line);border-radius:14px;background:linear-gradient(180deg,#1f1b16,#1a1713);padding:14px}
.section h3{margin:0 0 10px 0;color:var(--brand)}
.grid{display:grid;grid-template-columns:repeat(5,1fr) 240px;gap:10px}
@media (max-width:1200px){.grid{grid-template-columns:1fr 1fr 1fr}}
@media (max-width:700px){.grid{grid-template-columns:1fr}}
label{display:block;margin:6px 0 4px 2px;color:var(--muted);font-weight:700}
input[type=text]{width:100%;padding:10px;border:none;border-radius:12px;background:var(--sunken);color:var(--txt)}
.auto-wrap{position:relative}
.auto-wrap{position:relative}

.auto-list{
  /* Inputun hemen altında, kutulu dropdown */
  position:relative;
  top:0;
  margin-top:4px;

  background:#15120f;
  border:1px solid var(--line);
  border-radius:12px;
  padding:4px 0;

  box-shadow:0px 4px 12px rgba(0,0,0,0.35);
  max-height:220px;
  overflow-y:auto;

  display:none;
  z-index:1200;
}

.auto-item{
  padding:10px 12px;
  cursor:pointer;
  font-size:14px;
  border-bottom:1px solid rgba(255,255,255,0.08);
}

.auto-item:last-child{
  border-bottom:none;
}

.auto-item:hover{
  background:var(--brand);
  color:#111;
}

.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.chip{background:#172018;border:1px solid #2c4c39;color:#bdeacc;font-size:12px;border-radius:999px;padding:4px 8px}

/* Advisor */
.advisor-head{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
}

.expand-chat-btn{
  border:1px solid var(--line);
  background:#2b251f;
  color:#e8fff0;
  border-radius:10px;
  padding:7px 12px;
  font-weight:800;
  cursor:pointer;
}

.expand-chat-btn:hover{
  background:#123824;
  border-color:#347a55;
}

.chat-expanded{
  position:fixed;
  inset:24px;
  z-index:2000;
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:18px;
  padding:18px;
  box-shadow:0 20px 60px rgba(0,0,0,0.55);
}

.chat-expanded .messages{
  max-height:65vh;
  min-height:55vh;
}

.chat-expanded .footergrid{
  grid-template-columns:1fr 320px;
}
.messages{
  display:flex;
  flex-direction:column;
  gap:8px;
  background:#221e19;
  border:1px solid var(--line);
  padding:12px;
  border-radius:12px;
  max-height:300px;
  overflow:auto;
}

.quick-prompts{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  margin-top:10px;
}

.quick-btn{
  border:1px solid #2c4c39;
  background:#172018;
  color:#dff4e6;
  border-radius:999px;
  padding:8px 12px;
  font-weight:800;
  cursor:pointer;
  font-size:13px;
}

.quick-btn:hover{
  background:#123824;
  border-color:#7cc98a;
}
.msg{
  max-width:88%;
  padding:14px 16px;
  border-radius:14px;
  white-space:pre-wrap;
  line-height:1.55;
  font-size:15px;
}
.user{
  align-self:flex-end;
  background:#4a4036;
  color:#fff;
  border:1px solid #5b4f44;
}
.bot{
  align-self:flex-start;
  background:#123824;
  color:#e8fff0;
  border:1px solid #347a55;
  box-shadow:0 6px 18px rgba(0,0,0,0.18);
}
.typing{
  opacity:0.85;
  font-style:italic;
}

.typing::after{
  content:"";
  animation:dots 1.2s infinite;
}

@keyframes dots{
  0%{content:"";}
  33%{content:".";}
  66%{content:"..";}
  100%{content:"...";}
}
.footergrid{display:grid;grid-template-columns:1fr 260px;gap:10px;margin-top:8px}
textarea{min-height:90px;resize:vertical;background:#221e19;color:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}
.smallgrid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px}
.kit{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.kit .btn{padding:10px 14px}
.kit .warn{background:#d2a85a;color:#201a10}
.targets{margin-top:8px;font-size:13px}
.targets .item{display:flex;justify-content:space-between;gap:8px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:#211c17}
.targets .item button{border:none;background:#3a332c;color:#fff;border-radius:8px;padding:4px 8px;cursor:pointer}
.targets .empty{color:#b6ada3}
.farm-status{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:10px;
  margin-top:12px;
}

.fs-card{
  background:#172018;
  border:1px solid #2c4c39;
  border-radius:12px;
  padding:10px 12px;
}

.fs-card b{
  display:block;
  color:#9fe2b0;
  font-size:13px;
  margin-bottom:4px;
}

.fs-card span{
  color:#efece6;
  font-size:13px;
}

@media (max-width:700px){
  .farm-status{
    grid-template-columns:1fr;
  }
}

.side-guide{
  position:absolute;
  right:14px;
  top:14px;
  width:260px;
  max-height:calc(100% - 28px);
  overflow:auto;
  padding:10px;
  border-radius:14px;
  background:rgba(18,18,18,0.42);
  border:1px solid rgba(124,201,138,0.45);
  color:#efece6;
  box-shadow:0 10px 28px rgba(0,0,0,0.45);
  backdrop-filter:blur(18px);
  z-index:700;
}

.sg-head{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:8px;
  margin-bottom:10px;
}

.sg-title{
  font-weight:900;
  color:var(--brand);
  font-size:14px;
}

.sg-hide{
  border:none;
  background:#2b251f;
  color:#dff4e6;
  border-radius:8px;
  padding:4px 8px;
  cursor:pointer;
  font-size:12px;
}

.sg-hide:hover{
  background:#123824;
}

.sg-step{
  display:flex;
  gap:9px;
  padding:8px 0;
  border-top:1px solid rgba(255,255,255,0.08);
}

.sg-step:first-of-type{
  border-top:none;
}

.sg-num{
  min-width:24px;
  height:24px;
  border-radius:50%;
  background:#213827;
  color:#9fe2b0;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:900;
  font-size:12px;
  border:1px solid rgba(124,201,138,0.45);
}

.sg-step.active .sg-num{
  background:var(--brand);
  color:#101510;
}

.sg-step.done .sg-num{
  background:#2f5e49;
  color:#fff;
}

.sg-text b{
  display:block;
  font-size:13px;
  color:#fff;
}

.sg-text span{
  display:block;
  margin-top:2px;
  color:#c9c2b9;
  font-size:12px;
  line-height:1.3;
}

.side-guide.collapsed{
  width:160px;
  max-height:52px;
  overflow:hidden;
}

.side-guide.collapsed .sg-steps{
  display:none;
}

@media (max-width:900px){
  .side-guide{
    right:10px;
    top:auto;
    bottom:52px;
    width:260px;
    max-height:220px;
  }
}

</style>
</head>
<body>
<header>
  <div class="brand"><div class="logo">🌿</div><div class="title">Smart Agriculture Assistant</div><div class="badge">beta</div></div>
  <div class="lang">
    <button id="langTR" class="active">Türkçe</button>
    <button id="langEN">English</button>
  </div>
</header>

<div class="container">
  
<!-- MAP -->
<div class="mapwrap">

  <div id="map"></div>

  <div class="stack">
    <button class="btn green" id="btnDraw">Bölge Çiz</button>
    <button class="btn green" id="btnTarget">Hedef Modu</button>
    <button class="btn dark"  id="btnPan">Pan Modu</button>
    <button class="btn dark"  id="btnReset">Görünümü Sıfırla</button>
    <button class="btn dark"  id="btnDelLast" disabled>Son Hedefi Sil</button>
    <button class="btn red"   id="btnClear" disabled>Hepsini Temizle</button>
    <button class="btn red"   id="btnResetRegion" disabled>Bölgeyi Sıfırla</button>
  </div>

  <div id="sideGuide" class="side-guide">
    <div class="sg-head">
      <div class="sg-title" id="sgTitle">🌿 Nasıl Kullanılır?</div>
      <button class="sg-hide" id="btnGuideToggle">Gizle</button>
    </div>

    <div class="sg-steps">
  <div class="sg-step active" id="guideStep1">
    <div class="sg-num">1</div>
    <div class="sg-text">
      <b>Konum ara</b>
      <span>Ülke, şehir veya adres yazarak haritada konuma git.</span>
    </div>
  </div>

  <div class="sg-step" id="guideStep2">
    <div class="sg-num">2</div>
    <div class="sg-text">
      <b>Bölge çiz</b>
      <span>Analiz etmek istediğin tarım alanını haritada seç.</span>
    </div>
  </div>

  <div class="sg-step" id="guideStep3">
    <div class="sg-num">3</div>
    <div class="sg-text">
      <b>Hedef ekle</b>
      <span>Sensör yerleştirilecek hedef noktaları işaretle.</span>
    </div>
  </div>

  <div class="sg-step" id="guideStep4">
    <div class="sg-num">4</div>
    <div class="sg-text">
      <b>Danışmana sor</b>
      <span>Ürün, sensör ve sulama önerisi al.</span>
    </div>
  </div>
</div>

</div>

</div>

  <!-- ADDRESS QUERY -->
  <div class="section">
    <h3 id="geoTitle">📍 Geospatial Query</h3>
    <div class="grid">
      <div class="auto-wrap">
        <label id="lblCountry">Country</label>
        <input id="country" type="text" placeholder="e.g., Turkey">
        <div id="list-country" class="auto-list"></div>
      </div>
      <div class="auto-wrap">
        <label id="lblCity">City / Region</label>
        <input id="city" type="text" placeholder="e.g., Ankara" disabled>
        <div id="list-city" class="auto-list"></div>
      </div>
      <div class="auto-wrap">
        <label id="lblCounty">County / District</label>
        <input id="county" type="text" placeholder="e.g., Çankaya" disabled>
        <div id="list-county" class="auto-list"></div>
      </div>
      <div class="auto-wrap">
        <label id="lblNeighbourhood">Neighbourhood / Suburb</label>
        <input id="neigh" type="text" placeholder="e.g., Bahçelievler" disabled>
        <div id="list-neigh" class="auto-list"></div>
      </div>
      <div class="auto-wrap">
        <label id="lblStreet">Full Address (street/number)</label>
        <input id="street" type="text" placeholder="e.g., 48. Cadde 3/5" disabled>
        <div id="list-street" class="auto-list"></div>
      </div>

      <div>
        <button id="btnQuery" class="btn green" style="width:100%">Query & Show on Map</button>
        <button id="btnAddTarget" class="btn dark" style="width:100%;margin-top:8px" disabled>Add This Address as Target</button>
        <div class="chips">
          <span class="chip" id="chipLoc">Location: —</span>
          <span class="chip" id="chipPoly">Region: none</span>
          <span class="chip" id="chipTgt">Targets: 0</span>
        </div>
        <div style="font-size:12px;color:#bfb7ac;margin-top:6px" id="hint">Global coverage. Type at least 1 letter.</div>
      </div>
    </div>

    <div class="targets" id="targetsBox">
      <div class="empty" id="targetsEmpty">No targets yet. Search an address → “Add This Address as Target”, or use Target Mode on the map.</div>
    </div>
    <div class="farm-status" id="farmStatus">
  <div class="fs-card">
    <b id="soilTitle">🌱 Toprak Uygunluğu</b>
    <span id="soilStatus">Veri bekleniyor</span>
  </div>
  <div class="fs-card">
    <b id="waterTitle">💧 Sulama Riski</b>
    <span id="waterStatus">Veri bekleniyor</span>
  </div>
  <div class="fs-card">
    <b id="coverageTitle">📡 Kapsama</b>
    <span id="coverageStatus">Hedef yok</span>
  </div>
</div>
  </div>

  <!-- ADVISOR -->
  <div class="section">
    <div class="advisor-head">
      <h3 id="advTitle">💬 Smart Agriculture Advisor</h3>
      <button id="btnExpandChat" class="expand-chat-btn">⤢ Expand</button>
    </div>
    <div class="messages" id="msgs"></div>
    <div class="quick-prompts">
      <button class="quick-btn" data-tr="Bu bölgede en iyi ne yetişir?" data-en="What grows best in this area?">🌱 Ürün öner</button>
      <button class="quick-btn" data-tr="Bu bölge için sensör önerileri yapar mısın?" data-en="Can you recommend sensors for this area?">📡 Sensör öner</button>
      <button class="quick-btn" data-tr="Bu bölge için sulama önerisi verir misin?" data-en="Can you give irrigation advice for this area?">💧 Sulama öner</button>
      <button class="quick-btn" data-tr="Bu bölgenin iklim ve tarım uygunluğunu yorumlar mısın?" data-en="Can you analyze climate and agricultural suitability?">☀️ İklim analizi</button>
    </div>
    <div class="footergrid">
      <textarea id="userTxt" placeholder="e.g., What grows best here? irrigation/pH advice..."></textarea>
      <div>
        <div class="smallgrid">
          <input id="ph"    type="text" placeholder="pH (e.g., 6.7)">
          <input id="moist" type="text" placeholder="Moisture % (e.g., 42)">
          <input id="temp"  type="text" placeholder="Temperature °C (e.g., 22)">
          <input id="ndvi"  type="text" placeholder="NDVI (e.g., 0.55)">
        </div>
        <div class="kit">
          <button id="btnDictate" class="btn dark">🎙️ Dictate</button>
          <button id="btnSend"   class="btn green">Send</button>
          <button id="btnSpeak"  class="btn warn">🔊 Speak</button>
          <button id="btnOptimize" class="btn dark">💰 Cost Analysis</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
function updateSideGuide(step){
  for(let i=1; i<=4; i++){
    const el = document.getElementById("guideStep"+i);
    if(!el) continue;

    el.classList.remove("active","done");

    if(i < step){
      el.classList.add("done");
    }else if(i === step){
      el.classList.add("active");
    }
  }
}

/* ===== i18n ===== */
const i18n = {
  tr:{
    draw:"Bölge Çiz",
    target:"Hedef Modu",
    pan:"Pan Modu",
    reset:"Görünümü Sıfırla",
    delLast:"Son Hedefi Sil",
    clear:"Hepsini Temizle",
    resetRegion:"Bölgeyi Sıfırla",
    geo:"📍 Coğrafi Sorgu",
    country:"Ülke",
    city:"Şehir / Bölge",
    county:"İlçe",
    neigh:"Mahalle / Semt",
    street:"Açık Adres (sokak/kapı no)",
    query:"Sorgula & Haritada Göster",
    addTarget:"Bu Adresi Hedefe Ekle",
    hint:"Tüm dünya desteklenir. En az 1 harf yazın.",
    loc:"Konum: ",
    region:"Bölge: ",
    none:"yok",
    targets:"Hedef: ",
    advisor:"💬 Akıllı Tarım Danışmanı",
    placeholder:"Örn: Bu alanda ne yetişir? sulama/pH/NDVI tavsiyesi...",
    dictate:"🎙️ Dikte",
    send:"Gönder",
    speak:"🔊 Oku",
    noTgt:"Henüz hedef yok. Adres sorgula → “Bu Adresi Hedefe Ekle” veya Harita’da Hedef Modu ile tıklayın.",
    confirmUpd:"Hedef noktalar mevcut. Güncellensin mi?",
    selCoord:"Seçili koordinat: ",
    guideTitle:"Nasıl Kullanılır",
    guideSteps:[
      "1. Bölge çiz",
      "2. Hedef noktaları ekle",
      "3. Adres ara (isteğe bağlı)",
      "4. Aşağıdan danışmana sor"
    ]
  },

  en:{
    draw:"Draw Region",
    target:"Target Mode",
    pan:"Pan Mode",
    reset:"Reset View",
    delLast:"Delete Last Target",
    clear:"Clear All",
    resetRegion:"Reset Region",
    geo:"📍 Geospatial Query",
    country:"Country",
    city:"City / Region",
    county:"County / District",
    neigh:"Neighbourhood / Suburb",
    street:"Full Address (street/number)",
    query:"Query & Show on Map",
    addTarget:"Add This Address as Target",
    hint:"Global coverage. Type at least 1 letter.",
    loc:"Location: ",
    region:"Region: ",
    none:"none",
    targets:"Targets: ",
    advisor:"💬 Smart Agriculture Advisor",
    placeholder:"e.g., What grows best here? irrigation/pH/NDVI advice...",
    dictate:"🎙️ Dictate",
    send:"Send",
    speak:"🔊 Speak",
    noTgt:"No targets yet. Search an address → “Add This Address as Target”, or use Target Mode on the map.",
    confirmUpd:"Targets already exist. Do you want to update?",
    selCoord:"Selected coordinate: ",
    guideTitle:"How to Use",
    guideSteps:[
      "1. Draw region",
      "2. Add target points",
      "3. Search address (optional)",
      "4. Ask advisor below"
    ]
  }
};
let lang = localStorage.getItem("lang") || "tr";
const $ = (s)=>document.querySelector(s);
function T(k){ return i18n[lang][k]; }
function applyLang(){
  $("#btnDraw").textContent=T("draw"); $("#btnTarget").textContent=T("target"); $("#btnPan").textContent=T("pan");
  $("#btnReset").textContent=T("reset"); $("#btnDelLast").textContent=T("delLast"); $("#btnClear").textContent=T("clear");
  $("#btnResetRegion").textContent=T("resetRegion");
  $("#geoTitle").textContent=T("geo"); $("#lblCountry").textContent=T("country"); $("#lblCity").textContent=T("city");
  $("#lblCounty").textContent=T("county"); $("#lblNeighbourhood").textContent=T("neigh"); $("#lblStreet").textContent=T("street");
  $("#btnQuery").textContent=T("query"); $("#btnAddTarget").textContent=T("addTarget"); $("#hint").textContent=T("hint");
  $("#chipLoc").textContent=T("loc")+(window._loc||"—");
  $("#chipPoly").textContent=T("region")+(window._hasPoly?(lang==='tr'?'var':'set'):T("none"));
  $("#chipTgt").textContent=T("targets")+(window._tgt||0);
  $("#advTitle").textContent=T("advisor"); $("#userTxt").placeholder=T("placeholder");
  $("#btnDictate").textContent=T("dictate"); $("#btnSend").textContent=T("send"); $("#btnSpeak").textContent=T("speak");
  const expandBtn = $("#btnExpandChat");
  if(expandBtn){
    const advisorSection = $("#advTitle").closest(".section");
    const isOpen = advisorSection && advisorSection.classList.contains("chat-expanded");
    expandBtn.textContent = isOpen
      ? (lang==='tr' ? '⤡ Küçült' : '⤡ Collapse')
      : (lang==='tr' ? '⤢ Büyüt' : '⤢ Expand');
  }
  const te = document.getElementById("targetsEmpty");
if(te) te.textContent=T("noTgt");

if(lang === "tr"){
  $("#sgTitle").textContent = "🌿 Nasıl Kullanılır?";
  $("#btnGuideToggle").textContent = $("#sideGuide").classList.contains("collapsed") ? "Göster" : "Gizle";

  $("#guideStep1 .sg-text b").textContent = "Konum ara";
  $("#guideStep1 .sg-text span").textContent = "Ülke, şehir veya adres yazarak haritada konuma git.";

  $("#guideStep2 .sg-text b").textContent = "Bölge çiz";
  $("#guideStep2 .sg-text span").textContent = "Analiz etmek istediğin tarım alanını haritada seç.";

  $("#guideStep3 .sg-text b").textContent = "Hedef ekle";
  $("#guideStep3 .sg-text span").textContent = "Sensör yerleştirilecek hedef noktaları işaretle.";

  $("#guideStep4 .sg-text b").textContent = "Danışmana sor";
  $("#guideStep4 .sg-text span").textContent = "Ürün, sensör ve sulama önerisi al.";
}else{
  $("#sgTitle").textContent = "🌿 How to Use";
  $("#btnGuideToggle").textContent = $("#sideGuide").classList.contains("collapsed") ? "Show" : "Hide";

  $("#guideStep1 .sg-text b").textContent = "Search location";
  $("#guideStep1 .sg-text span").textContent = "Enter a country, city, or address to move on the map.";

  $("#guideStep2 .sg-text b").textContent = "Draw region";
  $("#guideStep2 .sg-text span").textContent = "Select the agricultural area you want to analyze.";

  $("#guideStep3 .sg-text b").textContent = "Add targets";
  $("#guideStep3 .sg-text span").textContent = "Mark target points for sensor placement.";

  $("#guideStep4 .sg-text b").textContent = "Ask advisor";
  $("#guideStep4 .sg-text span").textContent = "Get crop, sensor, and irrigation recommendations.";
}

document.querySelectorAll(".lang button").forEach(b=>b.classList.toggle("active", b.id==="lang"+lang.toUpperCase()));
setButtonMode("btnPan");
updateSideGuide(1);
updateFarmStatus();
}
$("#langTR").onclick=()=>{lang="tr";localStorage.setItem("lang","tr");applyLang();}
$("#langEN").onclick=()=>{lang="en";localStorage.setItem("lang","en");applyLang();}

/* ===== Map ===== */
let map=L.map('map',{
  zoomControl:false,
  center:[39,35],
  zoom:6,
  tap:true,
  touchZoom:true,
  dragging:true
});
L.control.zoom({position:'bottomright'}).addTo(map);

// 🌍 Uydu
var vivid = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { maxZoom:20, attribution:'Esri, Maxar, Earthstar, CNES/Airbus, USGS' }
).addTo(map);

// 🛣 Yol + POI + Yer isimleri
var overlayLabelRoads = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}',
  { maxZoom:20, opacity:1 }
).addTo(map);

var overlayPOI = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Places/MapServer/tile/{z}/{y}/{x}',
  { maxZoom:20, opacity:1 }
).addTo(map);

var overlayPlaces = L.tileLayer(
  'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  { maxZoom:20, opacity:1 }
).addTo(map);

let fg=L.featureGroup().addTo(map);
let drawn=null, targetMode=false, markers=[];
setTimeout(()=>map.invalidateSize(),300);


map.on("click",e=>{
  if(!targetMode) return;

  const currentCenter = map.getCenter();
  const currentZoom = map.getZoom();

  const m=L.marker(e.latlng).addTo(fg);
  markers.push(m);

  map.setView(currentCenter, currentZoom, {animate:false});
  enableTargetBtns();
  fetch('/targets/add',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({lat:e.latlng.lat,lng:e.latlng.lng,label:"(map)"})})
    .then(()=>refreshState());
});
function enableTargetBtns(){
  const noTarget = markers.length===0;
  $("#btnDelLast").disabled = noTarget;
  $("#btnClear").disabled = noTarget;
}

function updateFarmStatus(){
  const targetCount = markers.length;
  const hasRegion = window._hasPoly;

  if($("#soilTitle")){
    $("#soilTitle").textContent = lang === "tr"
      ? "🌱 Toprak Uygunluğu"
      : "🌱 Soil Suitability";
  }

  if($("#waterTitle")){
    $("#waterTitle").textContent = lang === "tr"
      ? "💧 Sulama Riski"
      : "💧 Irrigation Risk";
  }

  if($("#coverageTitle")){
    $("#coverageTitle").textContent = lang === "tr"
      ? "📡 Kapsama"
      : "📡 Coverage";
  }

  if($("#soilStatus")){
    $("#soilStatus").textContent = hasRegion
      ? (lang === "tr" ? "Analize hazır" : "Ready for analysis")
      : (lang === "tr" ? "Önce bölge seçin" : "Select a region first");
  }

  if($("#waterStatus")){
    $("#waterStatus").textContent = targetCount >= 2
      ? (lang === "tr" ? "Orta - izlenmeli" : "Medium - monitor")
      : (lang === "tr" ? "Sensör noktası gerekli" : "Sensor point needed");
  }

  if($("#coverageStatus")){
    if(targetCount === 0){
      $("#coverageStatus").textContent = lang === "tr"
        ? "Hedef yok"
        : "No targets";
    }else{
      const coverage = Math.min(95, 45 + targetCount * 15);
      $("#coverageStatus").textContent = lang === "tr"
        ? "%" + coverage + " tahmini kapsama"
        : coverage + "% estimated coverage";
    }
  }
}

function drawPoly(){
  const drawer=new L.Draw.Polygon(map,{shapeOptions:{color:'#7cc98a',weight:3,fillOpacity:.15},allowIntersection:false,showArea:true});
  drawer.enable();
  map.once(L.Draw.Event.CREATED, e=>{
    if(drawn) fg.removeLayer(drawn); drawn=e.layer; fg.addLayer(drawn);
    const pts=drawn.getLatLngs()[0].map(p=>[p.lat,p.lng]);
    fetch('/polygon/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({"points":pts})});
    // map.fitBounds(drawn.getBounds(),{padding:[20,20]});
    $("#btnResetRegion").disabled=false;
    window._hasPoly=true;
    applyLang();
    updateSideGuide(3);
  });
}

function setButtonMode(activeId){
  ["btnDraw","btnTarget","btnPan"].forEach(id=>{
    const el = $("#"+id);
    if(!el) return;
    if(id===activeId) el.classList.add("mode-active");
    else el.classList.remove("mode-active");
  });
}

$("#btnDraw").onclick=()=>{
  targetMode=false;
  map.getContainer().style.cursor="";
  setButtonMode("btnDraw");
  drawPoly();
};

$("#btnTarget").onclick=()=>{
  targetMode=true;
  map.getContainer().style.cursor="crosshair";
  setButtonMode("btnTarget");
  updateSideGuide(3);
};

$("#btnPan").onclick=()=>{
  targetMode=false;
  map.getContainer().style.cursor="";
  setButtonMode("btnPan");
};

$("#btnReset").onclick=()=>{ if(drawn) map.fitBounds(drawn.getBounds(),{padding:[20,20]}); else map.setView([39,35],6); };
$("#btnDelLast").onclick=async()=>{
  if(!markers.length) return;
  fg.removeLayer(markers.pop());
  await fetch('/targets/del_last',{method:'POST'});
  enableTargetBtns();
  refreshState();
};
$("#btnClear").onclick=async()=>{
  markers.forEach(m=>fg.removeLayer(m));
  markers=[];
  await fetch('/targets/clear',{method:'POST'});
  enableTargetBtns();
  refreshState();
};
$("#btnResetRegion").onclick=async()=>{
  if(drawn){fg.removeLayer(drawn); drawn=null;}
  markers.forEach(m=>fg.removeLayer(m)); markers=[];
  await fetch('/reset',{method:'POST'});
  enableTargetBtns();
  window._hasPoly=false;
  applyLang();
  updateSideGuide(1);
  map.setView([39,35],6);
};

async function refreshState(){
  const s=await (await fetch('/state')).json();
  window._loc=s.location_label||"—"; window._hasPoly=(s.polygon||[]).length>=3; window._tgt=(s.targets||[]).length;
  $("#chipLoc").textContent=T("loc")+window._loc;
  $("#chipPoly").textContent=T("region")+(window._hasPoly?(lang==='tr'?'var':'set'):T("none"));
  $("#chipTgt").textContent=T("targets")+window._tgt;
  renderTargets(s.targets||[]);
  updateFarmStatus();
}
refreshState();

/* ===== Autocomplete & Address search ===== */
async function suggest(level, q, ctx={}){
  const u=new URLSearchParams({level,q,...ctx}); const r=await fetch('/suggest?'+u.toString()); return await r.json();
}
function attachAuto(inp, list, level, deps){
  let t=null;

  inp.addEventListener('input', ()=>{
    const v=inp.value.trim();
    const min = (level==='country'?1:2);

    if(v.length < min){
      list.style.display="none";
      return;
    }

    clearTimeout(t);

    t=setTimeout(async()=>{
      const ctx={};
      deps.forEach(k=>ctx[k]=$('#'+k).value);

      list.innerHTML = '<div class="auto-item">⏳ Aranıyor...</div>';
     
      let data=[];
      try{
        data = await suggest(level,v,ctx);
      }catch(e){
        data = [];
      }

      list.innerHTML="";

      // Kullanıcının yazdığı şeyi her zaman seçenek olarak göster
      const manual=document.createElement('div');
      manual.className='auto-item';
      manual.textContent = `🔎 "${v}" olarak ara`;
      manual.onclick=()=>{
        inp.value=v;
        list.style.display='none';

        if(level==='country'){
          $('#city').disabled=false;
          $('#county').disabled=false;
          $('#neigh').disabled=false;
          $('#street').disabled=false;
        }
        if(level==='city'){
          $('#county').disabled=false;
          $('#neigh').disabled=false;
          $('#street').disabled=false;
        }
        if(level==='county'){
          $('#neigh').disabled=false;
          $('#street').disabled=false;
        }
      };
      list.appendChild(manual);

      data.forEach(it=>{
        const d=document.createElement('div');
        d.className='auto-item';
        d.textContent=it.name;

        d.onclick=()=>{
          inp.value=it.name;
          list.style.display='none';

          if(level==='country'){
            $('#city').disabled=false;
            $('#county').disabled=false;
            $('#neigh').disabled=false;
            $('#street').disabled=false;
          }
          if(level==='city'){
            $('#county').disabled=false;
            $('#neigh').disabled=false;
            $('#street').disabled=false;
          }
          if(level==='county'){
            $('#neigh').disabled=false;
            $('#street').disabled=false;
          }
        };

        list.appendChild(d);
      });

      list.style.display="block";
    },200);
  });

  inp.addEventListener('keydown', e=>{
  if(e.key === 'Enter'){
    e.preventDefault();
    list.style.display='none';
    $("#btnQuery").click();
  }
});
    

  document.addEventListener('click',e=>{
    if(!list.contains(e.target)&&e.target!==inp){
      list.style.display='none';
    }
  });
}
attachAuto($("#country"), $("#list-country"), "country", []);
attachAuto($("#city"), $("#list-city"), "city", ["country"]);
attachAuto($("#county"), $("#list-county"), "county", ["country","city"]);
attachAuto($("#neigh"), $("#list-neigh"), "neigh", ["country","city","county"]);
attachAuto($("#street"), $("#list-street"), "street", ["country","city","county","neigh"]);

$("#btnQuery").onclick = async ()=>{
  const q = [
    $("#street").value.trim(),
    $("#neigh").value.trim(),
    $("#county").value.trim(),
    $("#city").value.trim(),
    $("#country").value.trim()
  ].filter(Boolean).join(", ");

  if(!q){
    alert(lang === "tr" ? "Lütfen konum yazın." : "Please enter a location.");
    return;
  }

  const r = await (await fetch('/geo?q='+encodeURIComponent(q))).json();
    
    if(!r.lat){
    alert(lang === "tr" ? "Konum bulunamadı. Örn: Istanbul, Turkey yazın." : "Location not found. Try: Istanbul, Turkey");
    return;
  }
  
    if(r.lat){
    updateSideGuide(2);
    map.setView([r.lat,r.lon], 16);
    window._lastGeocode = {lat:r.lat, lon:r.lon, label:q};
    $("#btnAddTarget").disabled=false;
    await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({"label":q})});
    refreshState();
  }
};
$("#btnAddTarget").onclick = async ()=>{
  const g = window._lastGeocode; if(!g) return;
  await fetch('/targets/add',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({lat:g.lat,lng:g.lon,label:g.label})});
  markers.push(L.marker([g.lat,g.lon]).addTo(fg));
  enableTargetBtns(); refreshState();
};

/* Targets list UI */
function renderTargets(arr){
  const box=$("#targetsBox"); box.innerHTML="";
  if(!arr.length){ const d=document.createElement('div'); d.className='empty'; d.id='targetsEmpty'; d.textContent=T("noTgt"); box.appendChild(d); return; }
  arr.forEach((t,idx)=>{
    const row=document.createElement('div'); row.className='item';
    const left=document.createElement('div'); left.textContent=`${idx+1}) ${t.label||'(target)'} — ${t.lat.toFixed(5)}, ${t.lng.toFixed(5)}`;
    const rt=document.createElement('div');
    const btnGo=document.createElement('button'); btnGo.textContent='↦'; btnGo.title='Zoom';
    btnGo.onclick=()=>{ map.setView([t.lat,t.lng],17); };
    const btnX=document.createElement('button'); btnX.textContent='✕'; btnX.title='Delete';
    btnX.onclick=async()=>{ await fetch('/targets/del_index',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({"index":idx})}); refreshState(); };
    rt.appendChild(btnGo); rt.appendChild(btnX);
    row.appendChild(left); row.appendChild(rt); box.appendChild(row);
  });
}

/* ===== Chat ===== */
function push(role, text){
  const d=document.createElement('div'); d.className='msg '+(role==='user'?'user':'bot'); d.textContent=text;
  $("#msgs").appendChild(d); $("#msgs").scrollTop=$("#msgs").scrollHeight; return d;
}
function sendChat(){
  const txt=$("#userTxt").value.trim(); if(!txt) return;
  updateSideGuide(4);
  push('user',txt); $("#userTxt").value="";
  const loadingSteps = lang === "tr"
  ? [
      "🌱 Tarım koşulları analiz ediliyor...",
      "📡 Sensör kapsaması hesaplanıyor...",
      "💧 Sulama ihtiyacı değerlendiriliyor..."
    ]
  : [
      "🌱 Analyzing agricultural conditions...",
      "📡 Calculating sensor coverage...",
      "💧 Evaluating irrigation needs..."
    ];

let loadingIndex = 0;
const typingMsg = push('bot', loadingSteps[loadingIndex]);

const loadingTimer = setInterval(()=>{
  loadingIndex = (loadingIndex + 1) % loadingSteps.length;
  typingMsg.textContent = loadingSteps[loadingIndex];
},900);
  const payload={
    message:txt,
    sensors:{
      ph:$("#ph").value,
      moisture:$("#moist").value,
      temperature:$("#temp").value,
      ndvi:$("#ndvi").value
    },
    lang
  };
  fetch('/advice',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
    .then(r=>r.json())
    .then(d=>{
      clearInterval(loadingTimer);
      typingMsg.textContent = d.response || (lang==='tr'?'Yanıt alınamadı.':'No response.');
      window._lastBot = typingMsg;
    })
    .catch(err=>{
      console.error(err);
      clearInterval(loadingTimer);
      typingMsg.textContent = lang==='tr'?'Sunucu hatası.':'Server error.';
      window._lastBot = typingMsg;
    });
}
$("#btnSend").onclick=sendChat;
$("#userTxt").addEventListener('keydown',e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendChat(); }});
$("#btnDictate").onclick=()=>{
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition; if(!SR){alert('SpeechRecognition not supported');return;}
  const r=new SR(); r.lang=(lang==='tr'?'tr-TR':'en-US'); r.onresult=(e)=>{ $("#userTxt").value=( $("#userTxt").value?$("#userTxt").value+" ":"")+e.results[0][0].transcript; }; r.start();
};
/* TTS */
function speak(txt){ if(!('speechSynthesis' in window)) return; speechSynthesis.cancel(); const u=new SpeechSynthesisUtterance(txt); u.lang=(lang==='tr'?'tr-TR':'en-US'); u.rate=0.95; speechSynthesis.speak(u); }
$("#btnSpeak").onclick=()=>{ const last=window._lastBot?.textContent; if(last) speak(last); };
$("#btnOptimize").onclick=async()=>{
  const loading = push("bot", lang==="tr"
    ? "💰 Maliyet analizi hesaplanıyor..."
    : "💰 Calculating cost analysis..."
  );

  try{
    const r = await fetch("/optimize", {method:"POST"});
    const d = await r.json();

    if(d.status !== "Optimal"){
      loading.textContent = lang==="tr"
        ? "Uygun çözüm bulunamadı."
        : "No feasible solution found.";
      return;
    }

    const boxes = d.selected_boxes.join(", ");
    const assignments = d.sensor_assignments
      .map(a => `${a.location} → ${a.sensor}`)
      .join("\n");

    loading.textContent =
`💰 Cost Analysis Result

Target Count: ${d.target_count}
Sensing Range: ${d.sensing_range_m} m

Used Sensors:
${d.used_sensors.join(", ")}

Sensor Cost: ${d.sensor_cost} TL
Box Cost: ${d.box_cost} TL
Gateway Cost: ${d.gateway_cost} TL
Installation Cost: ${d.installation_cost} TL

--------------------------------

Total Cost: ${d.minimum_cost} TL

Selected Boxes:
${boxes}

Sensor Assignments:
${assignments}`;

  }catch(e){
    loading.textContent = lang==="tr"
      ? "Maliyet analizi sırasında hata oluştu."
      : "Cost analysis error.";
  }
};
$("#btnExpandChat").onclick=()=>{
  const advisorSection = $("#advTitle").closest(".section");
  advisorSection.classList.toggle("chat-expanded");

  const isOpen = advisorSection.classList.contains("chat-expanded");
  $("#btnExpandChat").textContent = isOpen
    ? (lang==='tr' ? '⤡ Küçült' : '⤡ Collapse')
    : (lang==='tr' ? '⤢ Büyüt' : '⤢ Expand');

  setTimeout(()=>{
    $("#msgs").scrollTop = $("#msgs").scrollHeight;
  },100);
};

document.addEventListener("keydown", e=>{
  if(e.key === "Escape"){
    const advisorSection = $("#advTitle").closest(".section");
    if(advisorSection.classList.contains("chat-expanded")){
      advisorSection.classList.remove("chat-expanded");
      $("#btnExpandChat").textContent = lang==='tr' ? '⤢ Büyüt' : '⤢ Expand';
    }
  }
});

$("#btnGuideToggle").onclick=()=>{
  const guide = $("#sideGuide");
  guide.classList.toggle("collapsed");

  const collapsed = guide.classList.contains("collapsed");
  $("#btnGuideToggle").textContent = collapsed
    ? (lang === "tr" ? "Göster" : "Show")
    : (lang === "tr" ? "Gizle" : "Hide");
};

document.querySelectorAll(".quick-btn").forEach(btn=>{
  btn.onclick=()=>{
    const text = lang === "tr"
      ? btn.dataset.tr
      : btn.dataset.en;

    $("#userTxt").value = text;
    $("#userTxt").focus();
  };
});

applyLang();

</script>
</body>
</html>
{% endraw %}
"""


# =======================
#  FLASK ROUTES
# =======================

@app.route("/")
def index():
    return render_template_string(HTML, app_name=APP_NAME)


@app.route("/state")
def get_state():
    return jsonify(
        {
            "polygon": STATE["polygon"],
            "targets": STATE["targets"],
            "location_label": STATE["location_label"],
            "sensor_matrix": STATE["sensor_matrix"],
            "sensor_types": STATE["sensor_types"],
        }
    )


@app.route("/label", methods=["POST"])
def set_label():
    data = request.get_json(silent=True) or {}
    STATE["location_label"] = (data.get("label") or "").strip()
    return jsonify({"ok": True})


@app.route("/polygon/save", methods=["POST"])
def polygon_save():
    data = request.get_json(silent=True) or {}
    pts = data.get("points") or []
    if len(pts) < 3:
        return jsonify({"ok": False, "error": "invalid"})
    STATE["polygon"] = [[float(a), float(b)] for a, b in pts]
    return jsonify({"ok": True})


@app.route("/targets/add", methods=["POST"])
def targets_add():
    data = request.get_json(silent=True) or {}
    lat, lng = data.get("lat"), data.get("lng")
    if lat is None or lng is None:
        return jsonify({"ok": False})
    STATE["targets"].append(
        {"lat": float(lat), "lng": float(lng), "label": (data.get("label") or "").strip()}
    )
    return jsonify({"ok": True, "count": len(STATE["targets"])})


@app.route("/targets/del_last", methods=["POST"])
def targets_del_last():
    if STATE["targets"]:
        STATE["targets"].pop()
    return jsonify({"ok": True, "count": len(STATE["targets"])})


@app.route("/targets/del_index", methods=["POST"])
def targets_del_index():
    data = request.get_json(silent=True) or {}
    idx = int(data.get("index", -1))
    if 0 <= idx < len(STATE["targets"]):
        STATE["targets"].pop(idx)
    return jsonify({"ok": True, "count": len(STATE["targets"])})


@app.route("/targets/clear", methods=["POST"])
def targets_clear():
    STATE["targets"] = []
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset_all():
    STATE["polygon"] = []
    STATE["targets"] = []
    STATE["location_label"] = ""
    STATE["sensor_matrix"] = None
    STATE["advisor_step"] = None        # BUG FIX: önceden return'dan sonraydı (dead code)
    STATE["recommended_sensors"] = None # BUG FIX: aynı şekilde ulaşılamıyordu
    STATE["pending_optimization"] = False
    STATE["selected_sensors"] = None
    STATE["chat_step"] = 0
    STATE["last_question"] = ""
    STATE["last_answer"] = ""
    return jsonify({"ok": True})


# ---- SENSOR MATRIX SAVE (LLM önerisini kaydetmek için istersen kullanırsın) ----
@app.route("/sensor_matrix", methods=["POST"])
def sensor_matrix():
    """
    Beklenen JSON örneği:
      {
        "matrix": [[1,0,1],[0,1,1]],
        "sensor_types": ["pH","Moisture","EC","SoilTemp","NDVI","NPK"],
        "targets": ["T1","T2"]
      }
    Şimdilik sadece STATE'e kaydediyoruz.
    """
    data = request.get_json(silent=True) or {}
    STATE["sensor_matrix"] = data
    return jsonify({"ok": True})


# ---- Autocomplete: country, city, county, neighbourhood, street/address ----
@app.route("/suggest")
def suggest_route():
    q = (request.args.get("q") or "").strip()
    level = (request.args.get("level") or "country").lower()
    country = (request.args.get("country") or "").strip()
    city = (request.args.get("city") or "").strip()
    county = (request.args.get("county") or "").strip()
    neigh = (request.args.get("neigh") or "").strip()

    if len(q) < 1 and level == "country":
        pass
    elif len(q) < 2 and level != "country":
        return jsonify([])

    if level == "country":
        return jsonify(
            [{"name": c} for c in COUNTRIES if q.lower() in c.lower()][:40]
        )

    parts = [q]
    if level in ("city", "county", "neigh", "street"):
        if county:
            parts.append(county)
        if city:
            parts.append(city)
        if country:
            parts.append(country)
    if level == "street" and neigh:
        parts.insert(1, neigh)

    query = ", ".join([p for p in parts if p])
    data = nominatim_get(
        "/search", {"q": query, "limit": 15, "addressdetails": 1}
    )
    seen, out = set(), []
    for r in data:
        name = (r.get("display_name", "").split(",")[0]).strip()
        if name and name.lower().startswith(q.lower()) and name not in seen:
            out.append({"name": name})
            seen.add(name)
    return jsonify(out[:20])


# ---- Geocode ----
@app.route("/geo")
def geo():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({})
    data = nominatim_get("/search", {"q": q, "limit": 1})
    if not data:
        return jsonify({})
    it = data[0]
    return jsonify({"lat": float(it["lat"]), "lon": float(it["lon"])})


# ---- Advisor ----
import math

SENSOR_TECH_INFO = {
    "pH": {
        "interval": "Every 6 hours",
        "critical": "Critical if pH < 5.5 or pH > 7.5",
        "range": "45 m",
        "cost": "1430 TL"
    },
    "Moisture": {
        "interval": "Every 15–30 minutes",
        "critical": "Critical if moisture < 30% or moisture > 70%",
        "range": "45 m",
        "cost": "1430 TL"
    },
    "SoilTemp": {
        "interval": "Every 30 minutes",
        "critical": "Critical if soil temperature < 10°C or > 35°C",
        "range": "45 m",
        "cost": "1430 TL"
    },
    "NDVI": {
        "interval": "Daily or every 2–3 days",
        "critical": "Critical if NDVI < 0.30",
        "range": "45 m",
        "cost": "2500 TL"
    },
    "EC": {
        "interval": "Every 6 hours",
        "critical": "Critical if EC is too high, indicating salinity/fertilizer stress",
        "range": "45 m",
        "cost": "1430 TL"
    },
    "NPK": {
        "interval": "Daily",
        "critical": "Critical if nitrogen, phosphorus or potassium is below crop requirement",
        "range": "45 m",
        "cost": "5149 TL"
    }
}


def haversine_distance_m(lat1, lng1, lat2, lng2):
    R = 6371000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_coverage_matrix_note(selected_sensors, sensing_range_m=180):
    targets = STATE.get("targets", [])

    if not targets:
        return "Coverage Matrix Note:\nNo target points selected yet."

    names = [f"T{i+1}" for i in range(len(targets))]
    lines = []
    lines.append("Coverage Matrix Note:")
    lines.append(f"Sensing range used: {sensing_range_m} m")
    lines.append("Rows = target points, Columns = candidate sensor box locations")
    lines.append("1 = covered, 0 = not covered")

    matrix = []

    for i, ti in enumerate(targets):
        row = []
        for j, tj in enumerate(targets):
            dist = haversine_distance_m(
                ti["lat"], ti["lng"],
                tj["lat"], tj["lng"]
            )
            row.append(1 if dist <= sensing_range_m else 0)
        matrix.append(row)

    header = "      " + " ".join(names)
    lines.append(header)

    for name, row in zip(names, matrix):
        lines.append(f"{name}:   " + " ".join(str(v) for v in row))

    lines.append("")
    lines.append(f"This same 0–1 coverage matrix is applied for selected sensors: {', '.join(selected_sensors)}")

    return "\n".join(lines)


def build_sensor_technical_note(selected_sensors):
    lines = []
    lines.append("Technical Sensor Note:")
    lines.append("The following technical parameters are automatically generated after sensor selection.")

    for s in selected_sensors:
        info = SENSOR_TECH_INFO.get(s)
        if not info:
            continue

        lines.append(
            f"- {s}: Measurement interval = {info['interval']}; "
            f"Critical range = {info['critical']}; "
            f"Detection range = {info['range']}; "
            f"Unit cost = {info['cost']}"
        )

    return "\n".join(lines)

def extract_recommended_sensors(text):
    import re

    sensor_patterns = {
        "pH": r"\bph\b",
        "Moisture": r"\bmoisture\b|\bsoil moisture\b|\bnem\b|\btoprak nemi\b",
        "EC": r"\bec\b|\belectrical conductivity\b|\biletkenlik\b",
        "SoilTemp": r"\bsoiltemp\b|\bsoil temperature\b|\btoprak sıcaklığı\b|\bsoil temp\b",
        "NDVI": r"\bndvi\b",
        "NPK": r"\bnpk\b|\bnitrogen\b|\bphosphorus\b|\bpotassium\b|\bazot\b|\bfosfor\b|\bpotasyum\b"
    }

    found = []
    lower = (text or "").lower()

    for sensor, pattern in sensor_patterns.items():
        if re.search(pattern, lower, flags=re.IGNORECASE):
            found.append(sensor)

    return found

def format_optimization_result_for_chat(result):
        if not result or result.get("status") != "Optimal":
            return "Optimization could not find a feasible solution."

        lines = []
        lines.append("Optimization completed.")
        lines.append("")
        
        lines.append("Recommended sensor box placement:")

        for item in result.get("box_plan", []):
            sensors = ", ".join(item.get("sensors", []))
            lines.append(
                f"- {item['box']} at {item['location']}: {sensors}"
            )

        lines.append("")
        lines.append(f"Number of sensor boxes: {result.get('box_count')}")
        lines.append(f"Sensor Cost: {result.get('sensor_cost')} TL")
        lines.append(f"Box Cost: {result.get('box_cost')} TL")
        lines.append(f"Gateway Cost: {result.get('gateway_cost')} TL")
        lines.append(f"Installation Cost: {result.get('installation_cost')} TL")
        lines.append("")
        lines.append(f"Total Cost: {result.get('minimum_cost')} TL")

        return "\n".join(lines)

@app.route("/advice", methods=["POST"])
def advice():
    d = request.get_json(silent=True) or {}

    lang = (d.get("lang") or "tr").lower()
    msg = (d.get("message") or "").strip()
    msg_lower = msg.lower()
    sensors = d.get("sensors") or {}

    if not msg:
            return jsonify({"response": "Lütfen bir soru yazın."})

    confirmation_words = [
            "yes", "okay", "ok", "confirm", "run", "start",
            "evet", "tamam", "onaylıyorum", "başlat", "çalıştır"
        ]

    if STATE.get("pending_optimization") and msg_lower in confirmation_words:
        selected_sensors = STATE.get("selected_sensors") or STATE.get("recommended_sensors")

        result = solve_sensor_placement(
            targets=STATE.get("targets", []),
            recommended_sensors=selected_sensors,
            sensing_range_m=45,
            box_capacity=2,
            volume_capacity=8
        )

        STATE["pending_optimization"] = False

        out = (
            "I am starting the optimization using the selected sensors.\n\n"
            + format_optimization_result_for_chat(result)
        )

        return jsonify({"response": out})

    # Her yeni kullanıcı sorusunda step otomatik artsın
    STATE["chat_step"] = STATE.get("chat_step", 0) + 1
    step_no = STATE["chat_step"]

    previous_question = STATE.get("last_question", "")
    previous_answer = STATE.get("last_answer", "")

    # Soru tipini anla
    sensor_keywords = [
        "sensör", "sensor", "ph", "ndvi", "moisture", "nem",
        "temperature", "sıcaklık", "ec", "npk"
    ]

    alternative_crop_keywords = [
        "başka ne yetişir", "başka hangi ürün", "alternatif",
        "başka ürün", "else can grow", "alternative crop"
    ]

    crop_keywords = [
        "ne yetişir", "hangi ürün", "en iyi ne yetişir",
        "ürün öner", "crop", "grow"
    ]

    sensor_selection_keywords = [
    "i choose",
    "i selected",
    "i select",
    "i would like to use",
    "i want to use",
    "i will use",
    "selected sensors",
    "use ph",
    "use pH",
    "şu sensörleri",
    "bu sensörleri",
    "sensörleri seçiyorum",
    "sensörleri seçtim",
    "sensörleri seçmek",
    "kullanmak istiyorum",
    "kullanacağım",
    "seçiyorum",
    "seçtim"
]

    if any(k.lower() in msg_lower for k in sensor_selection_keywords):
       task_type = "sensor_selection"
    elif any(k in msg_lower for k in sensor_keywords):
       task_type = "sensor_recommendation"
    elif any(k in msg_lower for k in alternative_crop_keywords):
       task_type = "alternative_crop"
    elif any(k in msg_lower for k in crop_keywords):
       task_type = "crop_recommendation"
    else:
        task_type = "general_question"

    # Görev talimatı
    if task_type == "crop_recommendation":
        instruction = """
Kullanıcı bu bölgede hangi ürünlerin yetişebileceğini soruyor.
Sadece ürün önerisi ver.
En fazla 3 ürün yaz.
Her ürün için tek kısa neden yaz.
İklim/toprak hakkında uzun rapor yazma.
Sensör önerisine geçme.
"""

    elif task_type == "alternative_crop":
        instruction = """
Kullanıcı önceki ürün önerilerine alternatif soruyor.
Önceki cevabı tekrar etme.
Sadece yeni/alternatif ürünleri yaz.
En fazla 3 alternatif ürün ver.
Her biri için tek kısa neden yaz.
Sensör önerisine geçme.
"""

    elif task_type == "sensor_selection":
       instruction = f"""
Kullanıcı artık kullanacağı sensörleri seçti.
Seçilen sensörleri onayla.
Teknik detaylara kullanıcı diliyle kısa giriş yap.
Ölçüm aralığı, kritik değer aralığı, detection range ve coverage matrix bilgisinin sistem tarafından otomatik hesaplandığını söyle.
Uzun açıklama yazma.
"""

    elif task_type == "sensor_recommendation":
        instruction = f"""
Kullanıcı sensör önerisi istiyor.
Sadece sensör önerisi ver.
Şu sensörlerden en uygun olanları seç: {SENSOR_TYPES}
En fazla 4 sensör yaz.
Her sensör için tek kısa neden yaz.
Ürün önerisini tekrar etme.
Uzun açıklama yazma.
"""

    else:
        instruction = """
Kullanıcının mevcut sorusuna kısa ve doğrudan cevap ver.
Önceki cevabı tekrar etme.
En fazla 4 madde kullan.
Konu dışına çıkma.
"""

    prompt = f"""
STEP {step_no}

Kullanıcı sorusu:
{msg}

Soru tipi:
{task_type}

Görev:
{instruction}

Bağlam:
Poligon: {STATE.get("polygon")}
Target noktaları: {STATE.get("targets")}
Sensör verileri: {sensors}

Önceki kullanıcı sorusu:
{previous_question}

Önceki cevap:
{previous_answer}

Cevap kuralları:
- Cevabın başında sadece şu başlığı yaz: STEP {step_no} —
- Markdown kullanma. Yıldız, kalın yazı, ## başlık kullanma.
- Sadece kullanıcının şu an sorduğu soruya cevap ver.
- Önceki cevabı tekrar etme.
- Uzun rapor yazma.
- En fazla 5 kısa madde kullan.
- "Devam etmek için devam yaz" gibi cümle yazma.
- Kullanıcı yeni soru sorarsa sonraki step zaten otomatik artacak.
- Dil: {lang}
"""

    out = llm_answer(prompt, lang, STATE, sensors)

    if not out:
        out = (
            f"STEP {step_no}\n\n"
            "API cevabı alınamadı. Terminalde görünen API hatasını kontrol edin."
        )

    STATE["last_question"] = msg
    STATE["last_answer"] = out

    detected_sensors = extract_recommended_sensors(out)

    if task_type == "sensor_selection":
        selected_sensors = extract_recommended_sensors(msg)
    else:
        selected_sensors = detected_sensors

    if task_type == "sensor_recommendation" and detected_sensors:
        STATE["recommended_sensors"] = detected_sensors
        print("✅ Recommended sensors saved after recommendation:", detected_sensors, flush=True)

    if task_type == "sensor_selection" and selected_sensors:
        STATE["recommended_sensors"] = selected_sensors
        print("✅ User selected sensors:", selected_sensors, flush=True)

        technical_note = build_sensor_technical_note(selected_sensors)
        coverage_note = build_coverage_matrix_note(selected_sensors, sensing_range_m=45)

        STATE["pending_optimization"] = True
        STATE["selected_sensors"] = selected_sensors

        out = (
            out
            + "\n\n---\n"
            + technical_note
            + "\n\nI can now run the optimization model with these selected sensors. Should I start the optimization?"
        )

    return jsonify({"response": out})


@app.route("/optimize", methods=["POST"])
def optimize_sensor_placement():
    targets = STATE.get("targets", [])

    recommended_sensors = STATE.get("recommended_sensors")
    if not recommended_sensors:
        recommended_sensors = ["pH", "Moisture", "EC", "SoilTemp", "NDVI"]

    result = solve_sensor_placement(
        targets=targets,
        recommended_sensors=recommended_sensors,
        sensing_range_m=45,
        box_capacity=3,
        volume_capacity=8

    )

    if result:
        result["used_sensors"] = recommended_sensors
        result["target_count"] = len(targets)
        result["sensing_range_m"] = 45
        result["price_note"] = "Costs are calculated using sensor_catalog.json unit prices."

    return jsonify(result)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5050,
        debug=False
    )
