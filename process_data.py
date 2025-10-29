# © 2025 Madhan Mohan | All Rights Reserved.
# Unauthorized copying, modification, or redistribution of this file, via any medium, is strictly prohibited.

import os 
import re
import sqlite3
import pandas as pd

# ---------------- Configuration ----------------
RAIN_FILE = "rainfall_area-wt_sd_1901-2015.csv"
AGRI_FILE = "Table_8.3-Statewise.csv"
DB_FILE = "samarth_data.db"

# ---------------- Canonicalization ----------------
ALIASES = {
    "orissa": "odisha",
    "pondicherry": "puducherry",
    "nct of delhi": "delhi",
    "uttaranchal": "uttarakhand",
    "jammu & kashmir": "jammu and kashmir",
    "jammu and kashmir": "jammu and kashmir",
    "dadra and nagar haveli and daman and diu": "daman and diu",
}

def canon_state(s: str) -> str:
    if s is None:
        return ""
    x = str(s)
    x = re.sub(r"\(.*?\)", "", x)
    x = x.replace("&", "and")
    x = re.sub(r"[^a-zA-Z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip().lower()
    x = ALIASES.get(x, x)
    return x

def pretty_state(s: str) -> str:
    return str(s).strip().title()

# ---------------- Subdivision-to-State Mapping ----------------
SUB_TO_STATES = {
    "COASTAL ANDHRA PRADESH": ["Andhra Pradesh"],
    "RAYALSEEMA": ["Andhra Pradesh"],
    "TELANGANA": ["Telangana"],

    "KONKAN & GOA": ["Maharashtra", "Goa"],
    "NORTH INTERIOR KARNATAKA": ["Karnataka"],
    "SOUTH INTERIOR KARNATAKA": ["Karnataka"],
    "COASTAL KARNATAKA": ["Karnataka"],

    "SUB HIMALAYAN WEST BENGAL & SIKKIM": ["West Bengal", "Sikkim"],
    "GANGETIC WEST BENGAL": ["West Bengal"],
    "ASSAM & MEGHALAYA": ["Assam", "Meghalaya"],
    "NAGALAND, MANIPUR, MIZORAM, TRIPURA": ["Nagaland", "Manipur", "Mizoram", "Tripura"],
    "HARYANA, CHANDIGARH & DELHI": ["Haryana", "Chandigarh", "Delhi"],

    "EAST RAJASTHAN": ["Rajasthan"],
    "WEST RAJASTHAN": ["Rajasthan"],
    "EAST MADHYA PRADESH": ["Madhya Pradesh"],
    "WEST MADHYA PRADESH": ["Madhya Pradesh"],
    "EAST UTTAR PRADESH": ["Uttar Pradesh"],
    "WEST UTTAR PRADESH": ["Uttar Pradesh"],
    "GUJARAT REGION": ["Gujarat"],
    "SAURASHTRA & KUTCH": ["Gujarat"],

    "BIHAR": ["Bihar"],
    "JHARKHAND": ["Jharkhand"],
    "ODISHA": ["Odisha"],
    "CHHATTISGARH": ["Chhattisgarh"],
    "HIMACHAL PRADESH": ["Himachal Pradesh"],
    "JAMMU & KASHMIR": ["Jammu and Kashmir"],
    "PUNJAB": ["Punjab"],
    "UTTARAKHAND": ["Uttarakhand"],
    "TAMIL NADU": ["Tamil Nadu"],
    "KERALA": ["Kerala"],
    "ARUNACHAL PRADESH": ["Arunachal Pradesh"],
    "ANDAMAN & NICOBAR ISLANDS": ["Andaman and Nicobar Islands"],
    "LAKSHADWEEP": ["Lakshadweep"],
}

def norm_sub(s: str) -> str:
    return str(s).strip().upper().replace("-", " ").replace("  ", " ")

# ---------------- Crop normalization (-> CropSimple + ProductionMT) ----------------
CROP_PATTERNS = [
    (r"(?i)\brice\b", "Rice", 1000),
    (r"(?i)\bwheat\b", "Wheat", 1000),
    (r"(?i)\bmaize\b", "Maize", 1000),
    (r"(?i)\bjowar\b", "Jowar", 1000),
    (r"(?i)\bbajra\b", "Bajra", 1000),
    (r"(?i)\bragi\b", "Ragi", 1000),
    (r"(?i)\bbarley\b", "Barley", 1000),
    (r"(?i)\bsmall millets\b", "Small millets", 1000),
    (r"(?i)\bgram\b", "Gram", 1000),
    (r"(?i)\btur\b", "Tur", 1000),
    (r"(?i)\bother pulses\b", "Other pulses", 1000),
    (r"(?i)\bgroundnuts?\b", "Groundnut", 1000),
    (r"(?i)\bcastor ?seed\b|castorseed", "Castor seed", 1000),
    (r"(?i)\blinseed\b", "Linseed", 1000),
    (r"(?i)\bsesamum\b", "Sesamum", 1000),
    (r"(?i)\brapeseed.*mustard\b|rapeseed and", "Rapeseed & Mustard", 1000),
    (r"(?i)\btotal cereals\b", "Total cereals", 1000),
    (r"(?i)\btotal pulses\b", "Total pulses", 1000),
    (r"(?i)\btotal food grains\b", "Total Food Grains", 1000),
    (r"(?i)\btotal oilseeds\*?\b", "Total Oilseeds", 1000),
    # Sugarcane variants
    (r"(?i)\bsugarcane\b.*\(000mt\)", "Sugarcane", 1000),
    (r"(?i)\bsugarcane\b.*\(th\.? tonnes\)", "Sugarcane", 1000),
    (r"(?i)\bsugarcane\b(?!.*\()", "Sugarcane", 1000),


    (r"(?i)cotton.*\(000 bales\)", "Cotton", 170.0), # 1 Bale = 170kg = 0.170 MT. 1 ('000 Bale) = 170 MT
    (r"(?i)jute.*\(000 bales\)", "Jute", 180.0), # 1 Bale = 180kg = 0.180 MT. 1 ('000 Bale) = 180 MT
    (r"(?i)coconut.*\(000mt\)", "Coconut", 1000), # Fix for Kerala
    (r"(?i)coconut.*\(million nuts\)", "Coconut-(Million Nuts)", 1.0), # Keep this data
]

def clean_crop_label(raw: str):
    s = str(raw)
    for pat, clean, scale in CROP_PATTERNS:
        if re.search(pat, s):
            return clean, scale
    return s, 1000  # fallback assume thousand-tonne style

# ---------------- Checks ----------------
if not os.path.exists(RAIN_FILE):
    raise FileNotFoundError(RAIN_FILE)
if not os.path.exists(AGRI_FILE):
    raise FileNotFoundError(AGRI_FILE)

# ---------------- RAINFALL ----------------
print("--- Processing Rainfall ---")
rf = pd.read_csv(RAIN_FILE)
need_cols_rf = {"SUBDIVISION", "YEAR", "ANNUAL"}
miss = need_cols_rf - set(rf.columns)
if miss:
    raise ValueError(f"Rainfall CSV missing columns: {sorted(miss)}")

rows = []
for _, row in rf.iterrows():
    sub = norm_sub(row["SUBDIVISION"])
    yr = int(row["YEAR"])
    try:
        val = float(row["ANNUAL"])
    except Exception:
        continue
    states = SUB_TO_STATES.get(sub, [str(row["SUBDIVISION"]).strip().title()])
    for st in states:
        st_clean = st.strip()
        rows.append({
            "State": st_clean,
            "StateNorm": canon_state(st_clean),
            "Year": yr,
            "Annual_Rainfall": val
        })
rf_df = pd.DataFrame(rows)
rf_final = (
    rf_df.groupby(["State", "StateNorm", "Year"], as_index=False)["Annual_Rainfall"]
         .mean()
)

# ---------------- AGRICULTURE ----------------
print("\n--- Processing Agriculture ---")
ag = pd.read_csv(AGRI_FILE)
if "State/ UT Name" not in ag.columns:
    raise ValueError("Missing 'State/ UT Name' in agriculture CSV")

id_col = ["State/ UT Name"]
value_cols = [c for c in ag.columns if c not in id_col]
long_ag = pd.melt(
    ag, id_vars=id_col, value_vars=value_cols,
    var_name="Crop_Year_Raw", value_name="Production"
)

def parse_crop_year(s: str):
    s = str(s)
    m = re.match(r"(.+?)-(\d{4})-(\d{2})", s)  # e.g., "Rice-2013-14"
    if m:
        crop = m.group(1).strip()
        year = int(m.group(2))
        return pd.Series([crop, year])
    return pd.Series([None, None])

long_ag[["Crop", "Year"]] = long_ag["Crop_Year_Raw"].apply(parse_crop_year)
long_ag = long_ag.dropna(subset=["Crop", "Year"])
long_ag["Production"] = pd.to_numeric(long_ag["Production"], errors="coerce")
long_ag = long_ag.dropna(subset=["Production"])

long_ag["State"] = long_ag["State/ UT Name"].astype(str).str.strip().str.title()
long_ag["StateNorm"] = long_ag["State"].apply(canon_state)

# crop normalization -> MT
long_ag["CropSimple"], long_ag["UnitScale"] = zip(*long_ag["Crop"].map(clean_crop_label))
long_ag["ProductionMT"] = long_ag["Production"] * long_ag["UnitScale"]

final_ag = long_ag[["State", "StateNorm", "Year", "Crop", "CropSimple", "Production", "ProductionMT"]]

# ---------------- Write DB ----------------
with sqlite3.connect(DB_FILE) as conn:
    rf_final.to_sql("rainfall", conn, if_exists="replace", index=False)
    final_ag.to_sql("crop_production", conn, if_exists="replace", index=False)

print(f"✓ Rainfall table created with {len(rf_final)} rows.")
print(f"✓ Agriculture table created with {len(final_ag)} rows.")
print(f"\nAll done! Database '{DB_FILE}' ready.")

# --- Optional optimization: Create indexes for faster queries ---
import sqlite3

with sqlite3.connect("samarth_data.db") as conn:
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rain_state_year ON rainfall(StateNorm, Year)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_crop_state_year ON crop_production(StateNorm, Year)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_crop_year ON crop_production(Year)")
    conn.commit()

print("✅ Database indexes created successfully.")
