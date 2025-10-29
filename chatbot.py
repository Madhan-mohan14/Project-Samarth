# © 2025 Madhan Mohan | Licensed under CC BY-NC-ND 4.0
import os
import re
import difflib
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

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
    x = str(s)
    x = re.sub(r"\(.*?\)", "", x)
    x = x.replace("&", "and")
    x = re.sub(r"[^a-zA-Z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip().lower()
    return ALIASES.get(x, x)

def pretty_state(s: str) -> str:
    # Converts 'odisha' or 'all-india' to 'Odisha' or 'All-India'
    if s.lower() == 'all-india':
        return 'All-India'
    return str(s).strip().title()


# ---------------- Known entities (Lazy Loaded) ----------------

_KNOWN_STATES_CACHE: Optional[List[str]] = None
def get_known_states() -> List[str]:
    """
    Lazy-loads and caches known PRETTY states from both tables.
    This FIX ensures we use 'Odisha', not 'Orissa'.
    """
    global _KNOWN_STATES_CACHE
    if _KNOWN_STATES_CACHE is not None:
        return _KNOWN_STATES_CACHE
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # Query the CLEAN StateNorm column
            s1 = pd.read_sql("SELECT DISTINCT StateNorm FROM rainfall", conn)
            s2 = pd.read_sql("SELECT DISTINCT StateNorm FROM crop_production", conn)
        
        # Union of canonical names (e.g., 'odisha', 'karnataka')
        all_norms = set(s1["StateNorm"].dropna()).union(set(s2["StateNorm"].dropna()))
        
        # Convert 'odisha' -> 'Odisha', etc. and sort
        pretty_states = sorted([pretty_state(s) for s in all_norms if s and s != 'all-india'])
        
        # Manually add 'All-India' as it's a valid concept
        pretty_states.append("All-India")
        _KNOWN_STATES_CACHE = pretty_states
        return _KNOWN_STATES_CACHE
    except Exception as e:
        print(f"Error loading states: {e}")
        # Fallback list MUST be the pretty versions
        return ["Odisha", "Karnataka", "Kerala", "Tamil Nadu", "Gujarat", "All-India"]

_KNOWN_CROPS_CACHE: Optional[List[str]] = None
def get_known_crops() -> List[str]:
    """Lazy-loads and caches known crops."""
    global _KNOWN_CROPS_CACHE
    if _KNOWN_CROPS_CACHE is not None:
        return _KNOWN_CROPS_CACHE
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = pd.read_sql("SELECT DISTINCT CropSimple AS Crop FROM crop_production", conn)
        _KNOWN_CROPS_CACHE = sorted(c["Crop"].dropna().unique().tolist())
        return _KNOWN_CROPS_CACHE
    except Exception as e:
        print(f"Error loading crops: {e}")
        return ["Rice", "Wheat", "Sugarcane", "Total Cereals"] # Fallback

# --- NEW HELPER ---
_STATE_NORM_TO_PRETTY_MAP: Optional[Dict[str, str]] = None
def get_pretty_state_map() -> Dict[str, str]:
    """Builds a map from 'odisha' -> 'Odisha'."""
    global _STATE_NORM_TO_PRETTY_MAP
    if _STATE_NORM_TO_PRETTY_MAP is None:
        _STATE_NORM_TO_PRETTY_MAP = {canon_state(s): s for s in get_known_states()}
    return _STATE_NORM_TO_PRETTY_MAP

# ---------------- Low-level queries ----------------
# (All queries are now updated to select StateNorm and map it to the pretty name)

def q_rainfall(states: List[str], years: List[int]) -> pd.DataFrame:
    years = sorted(set(int(y) for y in years)) if years else []
    
    def _fetch(query: str, params: tuple) -> pd.DataFrame:
        with sqlite3.connect(DB_FILE) as conn:
            df = pd.read_sql(query, conn, params=params)
        
        # FIX: Ensure 'State' column is the pretty name
        if not df.empty and "StateNorm" in df.columns:
            pretty_map = get_pretty_state_map()
            df["State"] = df["StateNorm"].map(pretty_map).fillna(df["State"])
            df = df.drop(columns=["StateNorm"])
        return df

    base_q = "SELECT StateNorm, State, Year, Annual_Rainfall FROM rainfall"
    
    if states:
        stn = [canon_state(s) for s in states]
        phs = ",".join("?"*len(stn))
        if years:
            phy = ",".join("?"*len(years))
            q = f"{base_q} WHERE StateNorm IN ({phs}) AND Year IN ({phy})"
            params = tuple(stn + years)
        else:
            q = f"{base_q} WHERE StateNorm IN ({phs})"
            params = tuple(stn)
    else:
        if years:
            phy = ",".join("?"*len(years))
            q = f"{base_q} WHERE Year IN ({phy})"
            params = tuple(years)
        else:
            q = base_q
            params = ()
            
    return _fetch(q, params)

def q_rainfall_avg(states: List[str], years: List[int]) -> pd.DataFrame:
    df = q_rainfall(states, years) # This now returns a df with pretty names
    if df.empty:
        return pd.DataFrame(columns=["State","Avg_Rainfall","N"])
    g = df.groupby(["State"], as_index=False).agg(Avg_Rainfall=("Annual_Rainfall","mean"), N=("Annual_Rainfall","count"))
    return g.sort_values(by="Avg_Rainfall", ascending=False)

def q_total_production(states: List[str], years: List[int]) -> pd.DataFrame:
    years = sorted(set(int(y) for y in years)) if years else []
    
    with sqlite3.connect(DB_FILE) as conn:
        if not states: # All-India
            stn = [canon_state("all-india")]
            phs = "?"
        else:
            stn = [canon_state(s) for s in states]
            phs = ",".join("?"*len(stn))
            
        base_q = f"""
            SELECT StateNorm, State, SUM(ProductionMT) AS Total_ProductionMT
            FROM crop_production
            WHERE StateNorm IN ({phs})
        """
        params_list: List[Any] = list(stn)
            
        if years:
            phy = ",".join("?"*len(years))
            base_q += f" AND Year IN ({phy})"
            params_list += years
        
        base_q += " GROUP BY StateNorm, State ORDER BY Total_ProductionMT DESC;"
        
        df = pd.read_sql(base_q, conn, params=tuple(params_list))

    if not df.empty:
        pretty_map = get_pretty_state_map()
        df["State"] = df["StateNorm"].map(pretty_map).fillna(df["State"])
        df = df.drop(columns=["StateNorm"])
    return df

def q_top_crops_by_state(states: List[str], years: List[int], top_n: int=3) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    
    # We need the PRETTY state name for the dict key
    states_pretty = [s for s in states]
    if not states_pretty:
        states_pretty = ["All-India"]
        
    years = sorted(set(int(y) for y in years)) if years else []
    
    with sqlite3.connect(DB_FILE) as conn:
        for st_pretty in states_pretty:
            stn = canon_state(st_pretty) # Query using 'odisha'
            
            exclude_crops = "('total food grains', 'total cereals', 'total pulses', 'total oilseeds')"
            
            if years:
                phy = ",".join("?"*len(years))
                q = f"""
                    SELECT CropSimple AS Crop, SUM(ProductionMT) AS Total_ProductionMT
                    FROM crop_production
                    WHERE StateNorm=? AND Year IN ({phy})
                      AND lower(CropSimple) NOT IN {exclude_crops}
                    GROUP BY CropSimple ORDER BY Total_ProductionMT DESC LIMIT ?;
                """
                df = pd.read_sql(q, conn, params=(stn, *years, top_n))
            else:
                y = 2015
                q = f"""
                    SELECT CropSimple AS Crop, SUM(ProductionMT) AS Total_ProductionMT
                    FROM crop_production 
                    WHERE StateNorm=? AND Year=?
                      AND lower(CropSimple) NOT IN {exclude_crops}
                    GROUP BY CropSimple ORDER BY Total_ProductionMT DESC LIMIT ?;
                """
                df = pd.read_sql(q, conn, params=(stn, y, top_n))
            
            out[st_pretty] = df # Use 'Odisha' as the key
    return out

def q_top_crops_all_india(year: int, top_n: int=5) -> pd.DataFrame:
    with sqlite3.connect(DB_FILE) as conn:
        exclude_crops = "('total food grains', 'total cereals', 'total pulses', 'total oilseeds')"
        df = pd.read_sql(f"""
          SELECT CropSimple AS Crop, SUM(ProductionMT) AS Total_ProductionMT
          FROM crop_production 
          WHERE Year = ? AND StateNorm = ?
            AND lower(CropSimple) NOT IN {exclude_crops}
          GROUP BY CropSimple ORDER BY Total_ProductionMT DESC LIMIT ?;
        """, conn, params=(year, canon_state("all-india"), top_n))
    return df

def q_crop_data(states: List[str], years: List[int], crop: Optional[str]) -> pd.DataFrame:
    years = sorted(set(int(y) for y in years)) if years else []
    with sqlite3.connect(DB_FILE) as conn:
        q = "SELECT StateNorm, State, Year, CropSimple AS Crop, ProductionMT FROM crop_production WHERE 1=1"
        params: List[Any] = []
        
        if not states: # Default to All-India if no states
            stn = [canon_state("all-india")]
            phs = "?"
            q += f" AND StateNorm IN ({phs})"
            params += stn
        else:
            stn = [canon_state(s) for s in states]
            phs = ",".join("?"*len(stn))
            q += f" AND StateNorm IN ({phs})"
            params += stn
            
        if years:
            phy = ",".join("?"*len(years))
            q += f" AND Year IN ({phy})"
            params += years
        if crop:
            q += " AND lower(CropSimple) = ?"
            params.append(crop.lower())
            
        q += " ORDER BY State, Year, Crop"
        df = pd.read_sql(q, conn, params=tuple(params))

    if not df.empty:
        pretty_map = get_pretty_state_map()
        df["State"] = df["StateNorm"].map(pretty_map).fillna(df["State"])
        df = df.drop(columns=["StateNorm"])
    return df

def q_rainfall_data(states: List[str], years: List[int]) -> pd.DataFrame:
    with sqlite3.connect(DB_FILE) as conn:
        base = "SELECT StateNorm, State, Year, Annual_Rainfall FROM rainfall WHERE 1=1"
        params = []
        if states:
            states_norm = [canon_state(s) for s in states if canon_state(s) != 'all-india']
            if states_norm:
                phs = ",".join("?"*len(states_norm))
                base += f" AND StateNorm IN ({phs})"
                params += states_norm
        if years:
            phy = ",".join("?"*len(years))
            base += f" AND Year IN ({phy})"
            params += years
        base += " ORDER BY Year, State"
        df = pd.read_sql(base, conn, params=params)

    if not df.empty:
        pretty_map = get_pretty_state_map()
        df["State"] = df["StateNorm"].map(pretty_map).fillna(df["State"])
        df = df.drop(columns=["StateNorm"])
    return df


def q_highest_rainfall_state(year: int) -> pd.DataFrame:
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql("""
          SELECT StateNorm, State, Year, Annual_Rainfall
          FROM rainfall WHERE Year = ? AND StateNorm != 'all-india'
          ORDER BY Annual_Rainfall DESC;
        """, conn, params=(year,))
        
    if not df.empty:
        pretty_map = get_pretty_state_map()
        df["State"] = df["StateNorm"].map(pretty_map).fillna(df["State"])
        df = df.drop(columns=["StateNorm"])
    return df

def q_correlation(state: str, years: Optional[List[int]]=None, last_n: Optional[int]=None) -> Tuple[pd.DataFrame, Optional[float]]:
    if not state or canon_state(state) == 'all-india':
        return pd.DataFrame(), None # Correlation must be per-state
        
    stn = canon_state(state)
    with sqlite3.connect(DB_FILE) as conn:
        if years:
            ymin, ymax = min(years), max(years)
        else:
            yr_r = pd.read_sql("SELECT MIN(Year) min, MAX(Year) max FROM rainfall WHERE StateNorm=?", conn, params=(stn,))
            yr_c = pd.read_sql("SELECT MIN(Year) min, MAX(Year) max FROM crop_production WHERE StateNorm=?", conn, params=(stn,))
            if yr_r.empty or yr_c.empty or pd.isna(yr_r.loc[0,"max"]) or pd.isna(yr_c.loc[0,"max"]):
                return pd.DataFrame(), None
            max_start = max(int(yr_r.loc[0,"min"]), int(yr_c.loc[0,"min"]))
            min_end = min(int(yr_r.loc[0,"max"]), int(yr_c.loc[0,"max"]))
            
            if last_n:
                ymin = max(max_start, min_end - last_n + 1)
                ymax = min_end
            else:
                ymin, ymax = max_start, min_end
        
        if ymin > ymax:
            return pd.DataFrame(), None

        rain = pd.read_sql("""
            SELECT Year, Annual_Rainfall AS rain
            FROM rainfall WHERE StateNorm=? AND Year BETWEEN ? AND ?
            ORDER BY Year
        """, conn, params=(stn, ymin, ymax))
        
        prod = pd.read_sql("""
            SELECT Year, SUM(ProductionMT) AS prod
            FROM crop_production 
            WHERE StateNorm=? AND Year BETWEEN ? AND ? AND lower(CropSimple) = 'total food grains'
            GROUP BY Year ORDER BY Year
        """, conn, params=(stn, ymin, ymax))
        
    if rain.empty or prod.empty:
        return pd.DataFrame(), None

    df = pd.merge(rain, prod, on="Year", how="inner").dropna()
    df.rename(columns={"rain": "AnnualRainfall_mm", "prod": "TotalFoodGrains_MT"}, inplace=True)
    
    if len(df) < 3:
        return df, None
        
    r = float(df["AnnualRainfall_mm"].corr(df["TotalFoodGrains_MT"]))
    return df, r

def q_trend_production(states: List[str], crop: Optional[str], years: Optional[List[int]]=None, last_n: Optional[int]=None) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    
    states_pretty = [s for s in states]
    if not states_pretty:
        states_pretty = ["All-India"]
        
    with sqlite3.connect(DB_FILE) as conn:
        for st_pretty in states_pretty:
            stn = canon_state(st_pretty)
            crop_to_query = (crop.lower() if crop else 'total food grains')
            
            if years:
                ymin, ymax = min(years), max(years)
            else:
                mx = pd.read_sql("SELECT MIN(Year) min, MAX(Year) max FROM crop_production WHERE StateNorm=?", conn, params=(stn,))
                if mx.empty or pd.isna(mx.loc[0,"max"]):
                    continue
                ymax = int(mx.loc[0,"max"])
                ymin_db = int(mx.loc[0,"min"])
                L = last_n or 10
                ymin = max(ymin_db, ymax - L + 1)

            df = pd.read_sql("""
                SELECT Year, SUM(ProductionMT) AS prod
                FROM crop_production WHERE StateNorm=? AND lower(CropSimple)=? AND Year BETWEEN ? AND ?
                GROUP BY Year ORDER BY Year
            """, conn, params=(stn, crop_to_query, ymin, ymax))
            
            if len(df) < 2:
                out[st_pretty] = {"years": (ymin, ymax), "n": len(df), "cagr": None, "slope": None, "first": None, "last": None}
                continue
                
            y0, y1 = int(df["Year"].iloc[0]), int(df["Year"].iloc[-1])
            v0, v1 = float(df["prod"].iloc[0]), float(df["prod"].iloc[-1])
            years_count = (y1 - y0) if (y1 - y0) > 0 else 1
            cagr = ((v1 / v0)**(1/years_count) - 1) if (v0 > 0 and v1 > 0) else None
            
            X = df["Year"].values.astype(float)
            Y = df["prod"].values.astype(float)
            slope = float(np.polyfit(X, Y, 1)[0]) if len(df) >= 2 else None
            out[st_pretty] = {"years": (y0, y1), "n": len(df), "cagr": cagr, "slope": slope, "first": v0, "last": v1}
            
    return out

# ---------------- Pretty helpers ----------------
def _fmt_mt(v: float) -> str:
    return f"{v:,.0f} MT"

# ---------------- Parsed holder ----------------
@dataclass
class Parsed:
    intents: List[str]
    states: List[str]
    years: List[int]
    last_n: Optional[int]
    top_n: Optional[int]
    crop: Optional[str]
    raw: str

# ---------------- Orchestrator ----------------
#
# This build_response function is now fully updated.
#
def build_response(user_q: str, forced: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not forced:
        return {"answer": "I'm having trouble understanding. Can you rephrase?", "artifacts": {}}

    p = Parsed(
        intents = forced.get("intents", []),
        states = [pretty_state(s) for s in forced.get("states", [])], # States are now 'Odisha'
        years = [int(y) for y in forced.get("years", [])],
        last_n = forced.get("last_n"),
        top_n = forced.get("top_n"),
        crop = forced.get("crop"),
        raw = user_q
    )

    blocks: List[str] = []
    answers: List[str] = []
    artifacts: Dict[str, pd.DataFrame] = {}

    rainfall_states = p.states
    if any(canon_state(s) == 'all-india' for s in p.states):
        rainfall_states = []

    intents_clean = sorted(list(set(p.intents)))

    for intent in intents_clean:

        if intent == "compare_rainfall":
            if len(p.states) < 2:
                answers.append("To compare rainfall, please provide at least two states.")
                continue
            
            df_comp = pd.DataFrame()
            lines = ["Compare Rainfall", "----------------"]
            
            if len(p.years) > 1:
                df_comp = q_rainfall_avg(rainfall_states, p.years)
                y_str = f"{min(p.years)}–{max(p.years)}"
                lines.append(f"Average Annual Rainfall ({y_str}):")
                for _, r in df_comp.iterrows():
                    lines.append(f"• {r['State']}: {r['Avg_Rainfall']:.1f} mm (n={r['N']} yrs)")
            
            elif len(p.years) == 1:
                df_comp = q_rainfall_data(rainfall_states, p.years)
                y_str = str(p.years[0])
                lines.append(f"Annual Rainfall ({y_str}):")
                for _, r in df_comp.iterrows():
                    lines.append(f"• {r['State']}: {r['Annual_Rainfall']:.1f} mm")
            
            else:
                answers.append("Please specify a year or year range for rainfall comparison.")
                continue
                
            if not df_comp.empty:
                artifacts["rainfall_comparison"] = df_comp
                lines.append("[Source: IMD Area Weighted Annual Rainfall (1901–2015)]")
                blocks.append("\n".join(lines))
                df_sorted = df_comp.sort_values(by=df_comp.columns[1], ascending=False)
                s1 = df_sorted.iloc[0][0]
                v1 = df_sorted.iloc[0][1]
                s2 = df_sorted.iloc[1][0]
                v2 = df_sorted.iloc[1][1]
                answers.append(f"In {y_str}, {s1} had higher rainfall ({v1:.1f} mm) than {s2} ({v2:.1f} mm).")
            else:
                blocks.append("Compare Rainfall\n----------------\nNo matching rainfall data.")

        elif intent == "avg_rainfall":
            df_avg = q_rainfall_avg(rainfall_states, p.years)
            if not df_avg.empty:
                lines = ["Average Rainfall", "----------------"]
                y_str = f"{min(p.years)}–{max(p.years)}" if p.years else "all available"
                for _, r in df_avg.iterrows():
                    lines.append(f"• {r['State']} ({y_str}, n={r['N']} yrs): {r['Avg_Rainfall']:.1f} mm")
                lines.append("[Source: IMD Area Weighted Annual Rainfall (1901–2015)]")
                blocks.append("\n".join(lines))
                artifacts["avg_rainfall_data"] = df_avg
                # FIX 2: Add specific answer
                answers.append(f"The average rainfall data for {y_str} is shown.")
            else:
                blocks.append("Average Rainfall\n----------------\nNo matching rows.")
                answers.append("I could not find average rainfall data for that request.")

        elif intent == "rainfall_data":
            df_rain = q_rainfall_data(rainfall_states, p.years)
            if not df_rain.empty:
                lines = ["Rainfall Data", "-------------"]
                artifacts["rainfall_data"] = df_rain
                
                # FIX 2: Add specific answer
                if len(df_rain) == 1:
                    r = df_rain.iloc[0]
                    answers.append(f"In {r['Year']}, rainfall in {r['State']} was {r['Annual_Rainfall']:.1f} mm.")
                elif len(df_rain["State"].unique()) == 1:
                    st = df_rain["State"].unique()[0]
                    answers.append(f"Here is the requested rainfall data for {st}.")
                else:
                    answers.append("The requested rainfall data for multiple states is shown.")
                
                for st in df_rain["State"].unique():
                    lines.append(f"• {st}:")
                    for _, r in df_rain[df_rain["State"] == st].iterrows():
                        lines.append(f"  - {r['Year']}: {r['Annual_Rainfall']:.1f} mm")
                lines.append("[Source: IMD Area Weighted Annual Rainfall (1901–2015)]")
                blocks.append("\n".join(lines))
            else:
                blocks.append("Rainfall Data\n-------------\nNo matching rows.")
                answers.append("I could not find any matching rainfall data for that request.")
        
        elif intent == "highest_rainfall_year":
            if not p.years:
                answers.append("Please provide a year to find the highest rainfall.")
                continue
            year = p.years[0]
            df_max = q_highest_rainfall_state(year)
            if df_max.empty:
                blocks.append(f"Highest Rainfall ({year})\n-----------------------\nNo rows.")
            else:
                top = df_max.iloc[0]
                blocks.append(
                    f"Highest Rainfall ({year})\n-----------------------\n"
                    f"• {top['State']}: {top['Annual_Rainfall']:.1f} mm\n"
                    "[Source: IMD Area Weighted Annual... (1901–2015)]"
                )
                answers.append(f"In {year}, {top['State']} had the highest rainfall ({top['Annual_Rainfall']:.1f} mm).")
        
        elif intent == "top_crops":
            n = p.top_n or 5
            res = q_top_crops_by_state(p.states, p.years, top_n=n)
            if res:
                lines = ["Top Crops", "---------"]
                y_str = f"({min(p.years)}–{max(p.years)})" if p.years else "(most recent)"
                summary_dfs = []
                for st, df_crops in res.items():
                    if df_crops.empty:
                        lines.append(f"• No data for {st} {y_str}.")
                        continue
                    lines.append(f"The top {len(df_crops)} crops in {st} {y_str} were:")
                    for i, r in df_crops.iterrows():
                        lines.append(f"  {i+1}. {r['Crop']} ({_fmt_mt(r['Total_ProductionMT'])})")
                    summary_dfs.append(df_crops.assign(State=st))
                lines.append("[Source: State/UT-wise Production of Principal Crops (2009–2015)]")
                blocks.append("\n".join(lines))
                if summary_dfs:
                    artifacts["top_crops_data"] = pd.concat(summary_dfs)
                # FIX 2: Add specific answer
                answers.append(f"The top crops for {', '.join(p.states) or 'All-India'} are listed.")
            else:
                blocks.append("Top Crops\n---------\nNo matching rows.")
                answers.append(f"I could not find top crop data for {', '.join(p.states)}.")

        elif intent == "total_production":
            if p.crop is None and (not p.states or "All-India" in p.states) and p.years:
                n = p.top_n or 100
                year = p.years[0]
                df_top = q_top_crops_all_india(year=year, top_n=n)
                if not df_top.empty:
                    lines = [f"Top {len(df_top)} Crops (All-India, {year})", "---------------------------------"]
                    for i, r in df_top.iterrows():
                         lines.append(f"  {i+1}. {r['Crop']} ({_fmt_mt(r['Total_ProductionMT'])})")
                    lines.append("[Source: State/UT-wise Production of Principal Crops (2009–2015)]")
                    blocks.append("\n".join(lines))
                    artifacts["top_crops_all_india"] = df_top
                    # FIX 2: Add specific answer
                    answers.append(f"Here is the full list of crop production for All-India in {year}.")
                else:
                    blocks.append(f"Top Crops (All-India, {year})\n----------------\nNo matching rows.")
                    answers.append(f"I could not find any crop production data for All-India in {year}.")
                continue 

            df_prod = q_total_production(p.states, p.years)
            if not df_prod.empty:
                lines = ["Total Crop Production", "---------------------"]
                y_str = f"({min(p.years)}–{max(p.years)})" if p.years else "(all years)"
                for _, r in df_prod.iterrows():
                    lines.append(f"• {r['State']} {y_str}: {_fmt_mt(r['Total_ProductionMT'])}")
                lines.append("[Source: State/UT-wise Production of Principal Crops (2009–2015)]")
                blocks.append("\n".join(lines))
                artifacts["total_production_data"] = df_prod
                
                # FIX 2: Add specific answer (if not a comparison)
                if len(df_prod) == 1:
                    r = df_prod.iloc[0]
                    answers.append(f"Total crop production for {r['State']} {y_str} was {_fmt_mt(r['Total_ProductionMT'])}.")
                elif len(df_prod) == 2:
                    df_sorted = df_prod.sort_values(by="Total_ProductionMT", ascending=False)
                    s1, v1 = df_sorted.iloc[0][0], df_sorted.iloc[0][1]
                    s2, v2 = df_sorted.iloc[1][0], df_sorted.iloc[1][1]
                    answers.append(f"{s1} had higher total crop production than {s2} by {_fmt_mt(v1-v2)} in {y_str}.")
                else:
                    answers.append(f"Total crop production data for {len(df_prod)} states is shown.")
            else:
                blocks.append("Total Crop Production\n---------------------\nNo matching rows.")
                answers.append("I could not find total production data for that request.")

        elif intent == "crop_data":
            df_crop = q_crop_data(p.states, p.years, p.crop)
            if not df_crop.empty:
                lines = ["Crop Production Data", "--------------------"]
                artifacts["crop_data"] = df_crop
                for st in df_crop["State"].unique():
                    lines.append(f"• {st}:")
                    for _, r in df_crop[df_crop["State"] == st].iterrows():
                        lines.append(f"  - {r['Year']} ({r['Crop']}): {_fmt_mt(r['ProductionMT'])}")
                lines.append("[Source: State/UT-wise Production of Principal Crops (2009–2015)]")
                blocks.append("\n".join(lines))
                
                # FIX 2: Add specific answer
                if len(df_crop) == 1:
                    r = df_crop.iloc[0]
                    answers.append(f"In {r['Year']}, {r['State']} produced {_fmt_mt(r['ProductionMT'])} of {r['Crop']}.")
                else:
                    crop_name = p.crop if p.crop else "crops"
                    state_name = ", ".join(p.states) if p.states else "all states"
                    answers.append(f"Here is the requested production data for {crop_name} in {state_name}.")
            else:
                blocks.append("Crop Production Data\n--------------------\nNo matching rows.")
                answers.append(f"I could not find production data for {p.crop or 'crops'} in {', '.join(p.states)}.")

        elif intent == "production_trend":
            res = q_trend_production(p.states, p.crop, p.years, p.last_n)
            if res:
                lines = ["Trend (Production)", "-------------------"]
                comp_rows = []
                for st, info in res.items():
                    y0, y1 = info["years"]
                    n = info["n"]; cagr = info["cagr"]; slope = info["slope"]
                    if n > 1 and cagr is not None and slope is not None:
                        lines.append(f"• {st} ({y0}–{y1}, n={n}): CAGR = {cagr*100:.2f}%/yr; slope = {slope:,.0f} MT/yr.")
                        answers.append(f"Overall, {st} shows a {'growing' if cagr > 0 else 'declining' if cagr < 0 else 'flat'} production trend ({y0}–{y1}).")
                        comp_rows.append({"State": st, "From": y0, "To": y1, "N": n, "CAGR_pct": cagr*100, "Slope_MT_per_year": slope})
                    else:
                        lines.append(f"• {st} ({y0}–{y1}): insufficient data (n={n}) for trend.")
                lines.append("[Source: State/UT-wise Production of Principal Crops (2009–2015)]")
                blocks.append("\n".join(lines))
                if comp_rows:
                    artifacts["trend_table"] = pd.DataFrame(comp_rows)
            else:
                answers.append("I could not find sufficient data to analyze that production trend.")

        elif intent == "correlation":
            if "correlation" in p.intents:
                state = p.states[0] if p.states else ""
                out, r = q_correlation(state, p.years, p.last_n)
                
                if out.empty or r is None:
                    blocks.append("Correlation\n-----------\nInsufficient overlapping years for correlation.")
                
                else:
                    lines = ["Rainfall ↔ Production Correlation", 
                            "-------------------------------"]
                    lines.append(f"• {out['Year'].min()}–{out['Year'].max()} (n={len(out)})")
                    lines.append(f"• Pearson r = {r:.2f}")
                    lines.append("[Source: IMD Area Weighted Annual Rainfall (1901–2015)]")
                    lines.append("[Source: State/UT-wise Production of Principal Crops (2009–2015)]")
                    blocks.append("\n".join(lines))
                    
                    a = abs(r)
                    strength = "very weak/none"
                    if a >= 0.7: 
                        strength = "strong"
                    elif a >= 0.4: 
                        strength = "moderate"
                    elif a >= 0.2: 
                        strength = "weak"
                        
                    sign = "positive" if r >= 0 else "negative"
                    if state:
                        answers.append(f"In {state}, rainfall and total crop production show a {strength} {sign} correlation (r={r:.2f}).")

    # --- Final Composition ---
    if not blocks and not answers:
        answer = "I'm sorry, I couldn't find specific data for your request."
    elif not answers:
        answer = "Here is the data I found." # This should rarely, if ever, be hit now
    else:
        answer = " ".join(answers)

    return {
        "tool_blocks": "\n\n".join(blocks),
        "answer": answer,
        "artifacts": artifacts,
    }