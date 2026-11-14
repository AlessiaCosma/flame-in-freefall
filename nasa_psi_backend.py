"""
Flame in Freefall - NASA Space Apps Challenge 2026
Complete ETL + FastAPI Backend for NASA PSI Microgravity Combustion Datasets
Processes raw NASA PSI CSVs or loads the unified master CSV (1,021 experiments).
Run with: uvicorn nasa_psi_backend:app --reload --port 8000
"""

import os
import re
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Flame in Freefall - NASA PSI Fire Safety API",
    description="Backend API for Microgravity Combustion Data Exploration, Partial-Gravity Simulation & AI Safety Insights",
    version="1.0.0"
)

# Enable CORS for React / Next.js / Vue / Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MASTER_CSV = "nasa_psi_master_1021_experiments.csv"
MASTER_JSON = "nasa_psi_master_1021_experiments.json"


# =====================================================================
# 1. ETL PIPELINE: RAW NASA PSI CSVs -> UNIFIED 1,021 EXPERIMENTS
# =====================================================================

def clean_material_name(text: str) -> str:
    t = str(text).lower()
    if "sibal" in t:
        return "SIBAL (Cotton-Fiberglass)"
    if "nomex" in t:
        return "Nomex"
    if "wax" in t or "candle" in t:
        return "Japan Wax"
    if "silicone" in t:
        return "Silicone"
    return "PMMA (Acrylic)"


def clean_geometry_name(text: str) -> str:
    t = str(text).lower()
    if "rod" in t:
        return "Cylinder / Rod"
    if "spher" in t:
        return "Sphere"
    if "sibal" in t or "nomex" in t:
        return "Fabric Sheet"
    if "film" in t or "micron" in t:
        return "Thin Film"
    if "candle" in t:
        return "Candle"
    return "Flat Slab"


def build_unified_dataset(data_dir: str = ".") -> pd.DataFrame:
    """
    If nasa_psi_master_1021_experiments.csv already exists, loads it directly.
    Otherwise, parses all raw NASA PSI CSV files in `data_dir` and builds it.
    """
    master_path = os.path.join(data_dir, MASTER_CSV)
    if os.path.exists(master_path):
        return pd.read_csv(master_path)

    records = []

    # 1. PSI-25: BASS-II
    f_b2 = os.path.join(data_dir, "PSI-25_Experimental table_BASS-II.csv")
    if os.path.exists(f_b2):
        df_b2 = pd.read_csv(f_b2)
        df_b2.columns = [c.strip() for c in df_b2.columns]
        for _, r in df_b2.iterrows():
            air_s = str(r.get("Air display", "")).lower()
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", air_s) if float(x) <= 55]
            init_flow = nums[0] if nums else 5.0
            min_flow = min(nums) if nums else 0.0

            if "no ign" in air_s or "would not stay lit" in air_s:
                outcome = "No Ignition / Self-Extinguished"
            elif "blowoff" in air_s:
                outcome = "High-Flow Blowoff"
            elif "ext" in air_s or "off" in air_s or "decr" in air_s:
                outcome = "Low-Flow Quenching Extinction"
            else:
                outcome = "Sustained Burn"

            o2_init = pd.to_numeric(r.get("Calibrated  initial O2 % by vol"), errors="coerce")
            co_init = pd.to_numeric(r.get("Initial CO (ppm)"), errors="coerce")
            co_fin = pd.to_numeric(r.get("Final CO (ppm)"), errors="coerce")
            co_delta = max(0.0, float(co_fin - co_init)) if pd.notnull(co_init) and pd.notnull(co_fin) else 0.0

            records.append({
                "exp_id": f"BASS2-{r['Test #']}",
                "dataset_id": "PSI-25",
                "investigation": "BASS-II",
                "doi": "10.60555/4qc4-de67",
                "fuel_state": "Solid",
                "material": clean_material_name(r["Fuel Sample Material"]),
                "geometry": clean_geometry_name(r["Fuel Sample Material"]),
                "raw_sample_desc": str(r["Fuel Sample Material"]).strip(),
                "o2_pct": round(float(o2_init), 2) if pd.notnull(o2_init) else 21.0,
                "pressure_kpa": 101.3,
                "airflow_cm_s": init_flow,
                "min_airflow_cm_s": min_flow,
                "burn_time_s": None,
                "co_generated_ppm": co_delta,
                "outcome": outcome
            })

    # 2. PSI-26: BASS
    f_b1 = os.path.join(data_dir, "PSI-26_Experimental table_BASS.csv")
    if os.path.exists(f_b1):
        df_b1 = pd.read_csv(f_b1)
        df_b1.columns = [c.strip() for c in df_b1.columns]
        for idx, r in df_b1.iterrows():
            mat_raw = str(r.get("Fuel Sample Material", ""))
            records.append({
                "exp_id": f"BASS1-{r['Test #']}-{idx+1}",
                "dataset_id": "PSI-26",
                "investigation": "BASS",
                "doi": "10.60555/58h4-dy80",
                "fuel_state": "Solid",
                "material": clean_material_name(mat_raw),
                "geometry": clean_geometry_name(mat_raw),
                "raw_sample_desc": mat_raw.strip(),
                "o2_pct": 21.0,
                "pressure_kpa": 101.3,
                "airflow_cm_s": 5.0,
                "min_airflow_cm_s": None,
                "burn_time_s": None,
                "co_generated_ppm": None,
                "outcome": "Sustained Burn"
            })

    # 3. PSI-98, PSI-99, PSI-100: SAFFIRE I, II, III
    f_s1 = os.path.join(data_dir, "PSI-98_Experimental table_SAFFIRE-1.csv")
    if os.path.exists(f_s1):
        s1 = pd.read_csv(f_s1)
        for _, r in s1.iloc[:2].iterrows():
            records.append({
                "exp_id": f"SAFFIRE1-{r['Sample Number '].strip()}",
                "dataset_id": "PSI-98",
                "investigation": "SAFFIRE-I",
                "doi": "NASA-PSI-98",
                "fuel_state": "Solid (Large Scale)",
                "material": "SIBAL (Cotton-Fiberglass)",
                "geometry": f"Large Panel ({r['Flow Direction '].strip()})",
                "raw_sample_desc": f"{r['Material']} {r['Sample Length (cm)']}x{r['Sample Width (cm)']} cm",
                "o2_pct": 21.6,
                "pressure_kpa": 101.3,
                "airflow_cm_s": float(r["Air Flow (cm/s)"]),
                "min_airflow_cm_s": float(r["Air Flow (cm/s)"]),
                "burn_time_s": float(r["Burn Time (s)"]),
                "co_generated_ppm": None,
                "outcome": "Sustained Burn"
            })

    f_s2 = os.path.join(data_dir, "PSI-99_Experimental table_SAFFIRE-2.csv")
    if os.path.exists(f_s2):
        s2 = pd.read_csv(f_s2)
        s2_mats = {
            "2-1": ("Silicone", "0.27 mm Slab"),
            "2-2": ("Silicone", "0.61 mm Slab"),
            "2-3": ("Silicone", "1.03 mm Slab"),
            "2-4": ("Silicone", "0.37 mm Slab (Opposed)"),
            "2-5": ("SIBAL (Cotton-Fiberglass)", "0.37 mm Fabric"),
            "2-6": ("SIBAL (Cotton-Fiberglass)", "0.37 mm Fabric"),
            "2-7": ("PMMA & Nomex", "Composite Panel"),
            "2-8": ("PMMA (Acrylic)", "Profiled Slab"),
            "2-9": ("PMMA (Acrylic)", "1 cm Thick Slab"),
        }
        for _, r in s2.iloc[:9].iterrows():
            snum = str(r["Sample Number"]).strip()
            mat_c, geom_c = s2_mats.get(snum, ("Solid", "Panel"))
            bt_raw = str(r["Burn Time (s)"])
            bt_nums = re.findall(r"\d+(?:\.\d+)?", bt_raw)
            bt_val = float(bt_nums[0]) if bt_nums else 0.0
            o2_nums = re.findall(r"\d+(?:\.\d+)?", str(r["Percent O2 (Note 1)"]))
            o2_val = float(o2_nums[0]) if o2_nums else 22.0
            out_c = "No Ignition / Self-Extinguished" if "Insignificant" in bt_raw else "Sustained Burn"
            records.append({
                "exp_id": f"SAFFIRE2-{snum}",
                "dataset_id": "PSI-99",
                "investigation": "SAFFIRE-II",
                "doi": "NASA-PSI-99",
                "fuel_state": "Solid (Large Scale)",
                "material": mat_c,
                "geometry": geom_c,
                "raw_sample_desc": f"{mat_c} ({geom_c}) 29x5 cm",
                "o2_pct": o2_val,
                "pressure_kpa": 101.3,
                "airflow_cm_s": 20.0,
                "min_airflow_cm_s": 20.0,
                "burn_time_s": bt_val,
                "co_generated_ppm": None,
                "outcome": out_c
            })

    f_s3 = os.path.join(data_dir, "PSI-100_Experimental table_SAFFIRE-3.csv")
    if os.path.exists(f_s3):
        s3 = pd.read_csv(f_s3)
        for _, r in s3.iloc[:2].iterrows():
            records.append({
                "exp_id": f"SAFFIRE3-{r['Sample Number '].strip()}",
                "dataset_id": "PSI-100",
                "investigation": "SAFFIRE-III",
                "doi": "NASA-PSI-100",
                "fuel_state": "Solid (Large Scale)",
                "material": "SIBAL (Cotton-Fiberglass)",
                "geometry": f"Large Panel ({r['Direction'].strip()})",
                "raw_sample_desc": f"SIBAL {r['Sample Length']}x{r['Sample Width']}",
                "o2_pct": 21.0,
                "pressure_kpa": 101.3,
                "airflow_cm_s": 30.0,
                "min_airflow_cm_s": 30.0,
                "burn_time_s": None,
                "co_generated_ppm": None,
                "outcome": "Sustained Burn"
            })

    # 4. PSI-69: FLEX & PSI-68: FLEX-2
    f_flex = os.path.join(data_dir, "PSI-69_Experimental table_FLEX.csv")
    if os.path.exists(f_flex):
        df_flex = pd.read_csv(f_flex, encoding="latin1")
        for _, r in df_flex.iterrows():
            p_mmhg = pd.to_numeric(r["Ambient pressure; mmHg"], errors="coerce")
            p_kpa = round(float(p_mmhg) * 0.133322, 2) if pd.notnull(p_mmhg) and float(p_mmhg) > 50 else 101.3
            o2_mol = pd.to_numeric(r["O initial ambient composition; mole fraction"], errors="coerce")
            o2_p = round(float(o2_mol) * 100.0, 2) if pd.notnull(o2_mol) else 21.0
            bt_s = pd.to_numeric(r["Burn time; s"], errors="coerce")
            tend = str(r["Test end"]).strip()
            if tend == "Extinction":
                out_f = "Radiative / Diffusive Extinction (Cool Flame Candidate)"
            elif tend == "Disruption":
                out_f = "Droplet Disruption"
            else:
                out_f = "Complete Burnout"
            records.append({
                "exp_id": f"FLEX-{r['FLEX Test #']}",
                "dataset_id": "PSI-69",
                "investigation": "FLEX",
                "doi": "NASA-PSI-69",
                "fuel_state": "Liquid Droplet",
                "material": str(r["Fuel"]).strip(),
                "geometry": f"Spherical Droplet ({r['Droplet initial diameter; mm']} mm)",
                "raw_sample_desc": f"{r['Fuel']} droplet ({r['FLEX Identifier']})",
                "o2_pct": o2_p,
                "pressure_kpa": p_kpa,
                "airflow_cm_s": 0.0,
                "min_airflow_cm_s": 0.0,
                "burn_time_s": float(bt_s) if pd.notnull(bt_s) else None,
                "co_generated_ppm": None,
                "outcome": out_f
            })

    f_f2 = os.path.join(data_dir, "PSI-68_Experimental table_Flex2.csv")
    if os.path.exists(f_f2):
        df_f2 = pd.read_csv(f_f2)
        for idx, r in df_f2.iterrows():
            records.append({
                "exp_id": f"FLEX2-{idx+1}",
                "dataset_id": "PSI-68",
                "investigation": "FLEX-2",
                "doi": "NASA-PSI-68",
                "fuel_state": "Liquid Droplet",
                "material": str(r["Sample (Fuel Type)"]).strip(),
                "geometry": f"Droplet ({r['Initial Droplet Diameter (mm)']} mm)",
                "raw_sample_desc": f"{r['Sample (Fuel Type)']} in {r['Ambient Gas Mixture ']}",
                "o2_pct": 21.0,
                "pressure_kpa": 101.3,
                "airflow_cm_s": 2.5,
                "min_airflow_cm_s": 0.1,
                "burn_time_s": None,
                "co_generated_ppm": None,
                "outcome": "Cool Flame / Radiative Extinction Study"
            })

    # 5. ACME family: PSI-159 (CFI-G), PSI-10 (Flame Design), PSI-21 (CLD), PSI-23 (s-Flame), PSI-22 (E-FIELD), PSI-20 (BRE)
    f_cfig = os.path.join(data_dir, "PSI-159_Experimental table_CFI-G_Normal Flames.csv")
    if os.path.exists(f_cfig):
        cfig = pd.read_csv(f_cfig)
        for _, r in cfig.iterrows():
            o2_b = pd.to_numeric(r[" XO2 before test"], errors="coerce")
            p_bar = pd.to_numeric(r["approx p (bar)"], errors="coerce")
            bt_g = pd.to_numeric(r["burn time (sec)"], errors="coerce")
            se_ft = str(r["Self-ext or flow term"]).strip()
            out_g = "Self-Extinguished" if se_ft == "SE" else ("Flow Terminated (Sustained)" if se_ft == "FT" else "Sustained Gas Flame")
            records.append({
                "exp_id": f"CFIG-{str(r['Test']).strip()}",
                "dataset_id": "PSI-159",
                "investigation": "ACME / CFI-G",
                "doi": "NASA-PSI-159",
                "fuel_state": "Gas Jet",
                "material": str(r["Fuel"]).strip().title(),
                "geometry": "Spherical / Coflow Burner",
                "raw_sample_desc": f"Gas flame: {r['Fuel']} (Peak T: {r['Peak burner T (C )']} C)",
                "o2_pct": round(float(o2_b) * 100.0, 2) if pd.notnull(o2_b) else 21.0,
                "pressure_kpa": round(float(p_bar) * 100.0, 2) if pd.notnull(p_bar) else 100.0,
                "airflow_cm_s": 0.0,
                "min_airflow_cm_s": 0.0,
                "burn_time_s": float(bt_g) if pd.notnull(bt_g) else None,
                "co_generated_ppm": None,
                "outcome": out_g
            })

    f_fd = os.path.join(data_dir, "PSI-10_Experimental table_Flame Design_Normal Flame.csv")
    if os.path.exists(f_fd):
        fd = pd.read_csv(f_fd)
        for _, r in fd.iterrows():
            o2_i = pd.to_numeric(r["XO2 (init)"], errors="coerce")
            p_atm = pd.to_numeric(r["pinit (atm)"], errors="coerce")
            records.append({
                "exp_id": f"FD-{str(r['Test']).strip()}",
                "dataset_id": "PSI-10",
                "investigation": "ACME / Flame Design",
                "doi": "NASA-PSI-10",
                "fuel_state": "Gas Jet",
                "material": "Ethylene (C2H4)" if "C2H4" in str(r["Fuel"]) else "Propane (C3H8)",
                "geometry": "Spherical Diffusion Burner",
                "raw_sample_desc": f"ACME Flame Design {r['Fuel']} (Xfuel={r['Xfuel']})",
                "o2_pct": round(float(o2_i) * 100.0, 2) if pd.notnull(o2_i) else 21.0,
                "pressure_kpa": round(float(p_atm) * 101.325, 2) if pd.notnull(p_atm) else 101.3,
                "airflow_cm_s": 0.0,
                "min_airflow_cm_s": 0.0,
                "burn_time_s": None,
                "co_generated_ppm": None,
                "outcome": "Sustained Gas Flame"
            })

    f_cld = os.path.join(data_dir, "PSI-21_Experimental table_CLD Flame.csv")
    if os.path.exists(f_cld):
        df_cld = pd.read_csv(f_cld)
        for idx, r in df_cld.iterrows():
            records.append({
                "exp_id": f"CLD-{r['Test Day']}-{idx+1}",
                "dataset_id": "PSI-21",
                "investigation": "ACME / CLD Flame",
                "doi": "NASA-PSI-21",
                "fuel_state": "Gas Jet",
                "material": f"Methane ({r['Fuel Bottle']})",
                "geometry": "Coflow Laminar Diffusion Burner",
                "raw_sample_desc": f"{r['Test Type']} ({r['Fuel conc. Range']}, {r['Flames ignited']} flames)",
                "o2_pct": 21.0,
                "pressure_kpa": 101.3,
                "airflow_cm_s": 0.0,
                "min_airflow_cm_s": 0.0,
                "burn_time_s": None,
                "co_generated_ppm": None,
                "outcome": "Soot & Extinction Limit Reached"
            })

    f_sf = os.path.join(data_dir, "PSI-23_Experimental table_ACME s-Flame.csv")
    if os.path.exists(f_sf):
        df_sf = pd.read_csv(f_sf)
        for idx, r in df_sf.iterrows():
            atm = str(r["Atmosphere*"])
            o2_val = 30.0 if "30%" in atm else 21.0
            fb = str(r["Fuel Bottle"]).strip()
            if "H2" in fb and "CH4" in fb:
                mat = f"Hydrogen-Methane Mix ({fb})"
            elif "H2" in fb:
                mat = f"Hydrogen ({fb})"
            elif "C2H4" in fb:
                mat = f"Ethylene ({fb})"
            else:
                mat = f"Methane ({fb})"
            records.append({
                "exp_id": f"SFLAME-{r['Test Day*']}-{idx+1}",
                "dataset_id": "PSI-23",
                "investigation": "ACME / s-Flame",
                "doi": "NASA-PSI-23",
                "fuel_state": "Gas Jet",
                "material": mat,
                "geometry": "Spherical Diffusion Burner",
                "raw_sample_desc": f"{r['Test Type']} | Atmosphere: {atm} | {int(r['Flames Ignited*'])} flames ignited",
                "o2_pct": o2_val,
                "pressure_kpa": 101.3,
                "airflow_cm_s": 0.0,
                "min_airflow_cm_s": 0.0,
                "burn_time_s": None,
                "co_generated_ppm": None,
                "outcome": "Spherical Flame Oscillations / Extinction Study"
            })

    f_ef = os.path.join(data_dir, "PSI-22_Experimental table_ACME E-FIELD Flames.csv")
    if os.path.exists(f_ef):
        df_ef = pd.read_csv(f_ef)
        df_ef_valid = df_ef[df_ef["Test Day"].notnull()].copy()
        for idx, r in df_ef_valid.iterrows():
            tday = int(r["Test Day"])
            fb = str(r["Fuel Bottle"]).strip()
            fconc = str(r["Fuel Conc"]).strip()
            pol = str(r["Mesh Polarity"]).strip()
            flames_ign = int(r["Flames Ignited"]) if pd.notnull(r["Flames Ignited"]) else 0
            mat = f"Methane ({fb})" if "CH4" in fb else f"Ethylene ({fb})"
            geom = "Coflow Burner (E-Field)" if "coflow" in str(r["Test Type"]).lower() else "Gas-Jet Burner (E-Field)"
            records.append({
                "exp_id": f"EFIELD-{tday}-{idx+1}",
                "dataset_id": "PSI-22",
                "investigation": "ACME / E-FIELD",
                "doi": "NASA-PSI-22",
                "fuel_state": "Gas Jet",
                "material": mat,
                "geometry": geom,
                "raw_sample_desc": f"E-Field ({pol} polarity), Fuel conc: {fconc}, {flames_ign} flames ignited",
                "o2_pct": 21.0,
                "pressure_kpa": 101.3,
                "airflow_cm_s": 0.0,
                "min_airflow_cm_s": 0.0,
                "burn_time_s": None,
                "co_generated_ppm": None,
                "outcome": "Ion-Driven Convection / Partial Gravity Simulation"
            })

    f_bre = os.path.join(data_dir, "PSI-20_Experimental table_ACME BRE.csv")
    if os.path.exists(f_bre):
        df_bre = pd.read_csv(f_bre)
        for idx, r in df_bre.iterrows():
            o2_raw = str(r["Oxygen Conc (percent)"]).replace("3400%", "34").replace("2100%", "21")
            o2_nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", o2_raw) if float(x) <= 100]
            o2_max = max(o2_nums) if o2_nums else 21.0
            fb = str(r["Fuel Bottle"]).strip()
            mat = f"Ethylene ({fb})" if "C2H4" in fb else f"Methane ({fb})"
            bsize = int(r["Burner Size (mm)"])
            flames_ign = int(r["Flames Ignited"])
            out_bre = "No Ignition / Self-Extinguished" if flames_ign == 0 else "Sustained Burner Radiative Extinction Study"
            records.append({
                "exp_id": f"BRE-{r['Year']}-{r['Month']}-{r['Date']}-{idx+1}",
                "dataset_id": "PSI-20",
                "investigation": "ACME / BRE",
                "doi": "NASA-PSI-20",
                "fuel_state": "Gas Jet",
                "material": mat,
                "geometry": f"Porous Spherical Burner ({bsize} mm)",
                "raw_sample_desc": f"BRE {bsize}mm burner, O2 tested: [{o2_raw}]%, {flames_ign} flames ignited",
                "o2_pct": o2_max,
                "pressure_kpa": 101.3,
                "airflow_cm_s": 0.0,
                "min_airflow_cm_s": 0.0,
                "burn_time_s": None,
                "co_generated_ppm": None,
                "outcome": out_bre
            })

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        df_out.to_csv(master_path, index=False)
        df_out.to_json(os.path.join(data_dir, MASTER_JSON), orient="records", indent=2)
    return df_out


# Load dataset at startup
df_master = build_unified_dataset(".")


# =====================================================================
# 2. FASTAPI ENDPOINTS FOR FRONTEND & AI ASSISTANT
# =====================================================================

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "total_experiments_loaded": len(df_master),
        "investigations_count": int(df_master["investigation"].nunique()) if not df_master.empty else 0
    }


@app.get("/summary")
def get_summary_stats():
    """Returns aggregated statistics for Dashboard KPI cards and charts."""
    if df_master.empty:
        return {"error": "No data loaded"}

    return {
        "total_experiments": len(df_master),
        "by_investigation": df_master["investigation"].value_counts().to_dict(),
        "by_fuel_state": df_master["fuel_state"].value_counts().to_dict(),
        "by_material": df_master["material"].value_counts().head(12).to_dict(),
        "by_outcome": df_master["outcome"].value_counts().to_dict(),
        "o2_range": {
            "min": float(df_master["o2_pct"].min()),
            "max": float(df_master["o2_pct"].max()),
            "mean": round(float(df_master["o2_pct"].mean()), 2)
        },
        "pressure_range_kpa": {
            "min": float(df_master["pressure_kpa"].min()),
            "max": float(df_master["pressure_kpa"].max()),
            "mean": round(float(df_master["pressure_kpa"].mean()), 2)
        },
        "airflow_range_cm_s": {
            "min": float(df_master["airflow_cm_s"].min()),
            "max": float(df_master["airflow_cm_s"].max()),
            "mean": round(float(df_master["airflow_cm_s"].mean()), 2)
        }
    }


@app.get("/filters")
def get_available_filters():
    """Returns unique dropdown values for Frontend filter components."""
    return {
        "investigations": sorted(df_master["investigation"].dropna().unique().tolist()),
        "fuel_states": sorted(df_master["fuel_state"].dropna().unique().tolist()),
        "materials": sorted(df_master["material"].dropna().unique().tolist()),
        "geometries": sorted(df_master["geometry"].dropna().unique().tolist()),
        "outcomes": sorted(df_master["outcome"].dropna().unique().tolist())
    }


@app.get("/experiments")
def query_experiments(
    investigation: Optional[str] = Query(None, description="Filter by investigation (e.g., BASS-II, FLEX, SAFFIRE)"),
    fuel_state: Optional[str] = Query(None, description="Solid, Liquid Droplet, Gas Jet"),
    material: Optional[str] = Query(None, description="Material substring (e.g., PMMA, SIBAL, Heptane, Methane)"),
    outcome: Optional[str] = Query(None, description="Outcome substring"),
    min_o2: float = Query(0.0, description="Minimum O2 %"),
    max_o2: float = Query(100.0, description="Maximum O2 %"),
    min_airflow: float = Query(0.0, description="Minimum airflow cm/s"),
    max_airflow: float = Query(100.0, description="Maximum airflow cm/s"),
    min_pressure: float = Query(0.0, description="Minimum pressure kPa"),
    max_pressure: float = Query(500.0, description="Maximum pressure kPa"),
    limit: int = Query(200, le=1050, description="Max records to return")
):
    """Filterable endpoint for 3D Scatter Plot, Heatmap, and Data Explorer Table."""
    data = df_master.copy()
    if investigation:
        data = data[data["investigation"].str.contains(investigation, case=False, na=False)]
    if fuel_state:
        data = data[data["fuel_state"].str.contains(fuel_state, case=False, na=False)]
    if material:
        data = data[data["material"].str.contains(material, case=False, na=False)]
    if outcome:
        data = data[data["outcome"].str.contains(outcome, case=False, na=False)]

    data = data[
        (data["o2_pct"] >= min_o2) & (data["o2_pct"] <= max_o2) &
        (data["airflow_cm_s"] >= min_airflow) & (data["airflow_cm_s"] <= max_airflow) &
        (data["pressure_kpa"] >= min_pressure) & (data["pressure_kpa"] <= max_pressure)
    ]

    total_matched = len(data)
    records_out = data.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return {
        "total_matched": total_matched,
        "returned": len(records_out),
        "experiments": records_out
    }


@app.get("/simulate-risk")
def simulate_habitat_fire_risk(
    environment: str = Query("ISS (Microgravity 0g)", description="ISS (Microgravity 0g), Lunar Habitat (0.16g), Martian Habitat (0.38g), Deep Space Transit"),
    o2_pct: float = Query(21.0, ge=10.0, le=50.0, description="Ambient Oxygen concentration (%)"),
    pressure_kpa: float = Query(101.3, ge=30.0, le=320.0, description="Cabin pressure (kPa), e.g., 101.3 for ISS, 56.5 for Lunar exploration atmosphere"),
    airflow_cm_s: float = Query(5.0, ge=0.0, le=55.0, description="HVAC ventilation flow speed (cm/s)"),
    material: str = Query("PMMA", description="Fuel/Material (PMMA, SIBAL, Nomex, Silicone, Methanol, Heptane, Methane, Ethylene, Hydrogen)"),
    top_k: int = Query(6, ge=1, le=20, description="Number of nearest NASA PSI experiments to retrieve")
):
    """
    What-If Fire Safety Simulator & Nearest-Experiment Similarity Engine.
    Calculates Flammability Score, Cool-Flame Probability, Toxicity Risk, and Actionable Crew Recommendations.
    """
    subset = df_master[df_master["material"].str.contains(material, case=False, na=False)].copy()
    if subset.empty:
        subset = df_master.copy()

    # Normalized Euclidean distance across O2 (%), Pressure (kPa), and Airflow (cm/s)
    subset["similarity_distance"] = np.sqrt(
        ((subset["o2_pct"] - o2_pct) / 10.0) ** 2 +
        ((subset["pressure_kpa"] - pressure_kpa) / 50.0) ** 2 +
        ((subset["airflow_cm_s"] - airflow_cm_s) / 15.0) ** 2
    )
    subset["similarity_score_pct"] = (100.0 / (1.0 + subset["similarity_distance"])).round(1)
    nearest = subset.sort_values("similarity_distance").head(top_k)

    # Physics-informed microgravity flammability model
    o2_risk = np.clip((o2_pct - 13.5) / 18.0, 0.0, 1.8) * 45.0

    env_lower = environment.lower()
    effective_flow = airflow_cm_s
    if "lunar" in env_lower or "0.16" in env_lower:
        effective_flow = max(airflow_cm_s, 6.5)  # Weak buoyant flow on Moon (~0.16g)
    elif "martian" in env_lower or "mars" in env_lower or "0.38" in env_lower:
        effective_flow = max(airflow_cm_s, 11.0)  # Moderate buoyant flow on Mars (~0.38g)

    if effective_flow < 0.8:
        flow_multiplier = 0.45  # Starved of oxygen in 0g stagnation
    elif 0.8 <= effective_flow <= 22.0:
        flow_multiplier = 1.25  # Optimal low-velocity forced flow feeding microgravity flame
    else:
        flow_multiplier = 0.90  # High flow cooling / potential blowoff regime

    sustained_mask = nearest["outcome"].str.contains("Sustained|Complete Burnout|Flow Terminated", case=False, na=False)
    empirical_burn_prob = float(sustained_mask.mean()) if len(nearest) > 0 else 0.5

    mat_lower = material.lower()
    if any(m in mat_lower for m in ["nomex", "silicone"]):
        mat_mod = 0.65
    elif any(m in mat_lower for m in ["hydrogen", "heptane", "ethylene", "sibal"]):
        mat_mod = 1.20
    else:
        mat_mod = 1.0

    raw_score = (o2_risk + empirical_burn_prob * 40.0) * flow_multiplier * mat_mod
    risk_score = int(np.clip(round(raw_score), 5, 99))

    cool_flame_risk = "HIGH" if any(x in mat_lower for x in ["heptane", "octane", "decane", "butane"]) else (
        "MODERATE" if "methanol" in mat_lower else "LOW"
    )

    recommendations = []
    if risk_score >= 70:
        risk_level = "CRITICAL"
        recommendations.append("IMMEDIATE ACTION: Shut down cabin HVAC ventilation fans (reduce airflow toward 0 cm/s) to starve the flame zone of fresh O2 and trigger radiative/diffusive quenching (validated in PSI-25 BASS-II).")
    elif risk_score >= 40:
        risk_level = "ELEVATED"
        recommendations.append("CAUTION: Material is within the microgravity flammability boundary. Avoid low-speed forced airflows (2-15 cm/s) which sustain flame spread better than in 1g.")
    else:
        risk_level = "LOW / SELF-EXTINGUISHING"
        recommendations.append("SAFE REGIME: Oxygen and flow conditions lie near or below the Limiting Oxygen Concentration (LOC) / quenching threshold observed in NASA ISS tests.")

    if o2_pct >= 26.0 and pressure_kpa < 75.0:
        recommendations.append("EXPLORATION ATMOSPHERE ALERT: High-O2 / low-pressure cabin atmosphere (Lunar/Martian transit protocol) accelerates flame spread rates; use Nomex III or Silicone barriers (validated in PSI-99 SAFFIRE-II).")

    if cool_flame_risk in ["HIGH", "MODERATE"]:
        recommendations.append("COOL FLAME HAZARD (PSI-69 FLEX / PSI-68 FLEX-2): After visible flame extinction, hydrocarbon/alcohol droplets can transition into invisible low-temperature 'cool flames' (~500-800 K) that vaporize toxic CO/formaldehyde and can re-ignite.")

    co_series = nearest["co_generated_ppm"].dropna()
    avg_co_ppm = round(float(co_series.mean()), 1) if not co_series.empty else None

    return {
        "scenario": {
            "environment": environment,
            "o2_pct": o2_pct,
            "pressure_kpa": pressure_kpa,
            "airflow_cm_s": airflow_cm_s,
            "effective_convective_flow_cm_s": round(effective_flow, 1),
            "material": material
        },
        "risk_assessment": {
            "flammability_score": risk_score,
            "risk_level": risk_level,
            "cool_flame_reignition_risk": cool_flame_risk,
            "estimated_co_spike_ppm": avg_co_ppm,
            "sustained_burn_ratio_in_nearest_tests": round(empirical_burn_prob * 100, 1)
        },
        "safety_recommendations": recommendations,
        "nearest_nasa_experiments": nearest.replace({np.nan: None}).to_dict(orient="records")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
