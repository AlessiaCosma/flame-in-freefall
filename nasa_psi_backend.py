"""
Flame in Freefall - NASA Space Apps Challenge 2026
ETL + FastAPI backend for NASA PSI microgravity combustion datasets.

Principles:
  * every value comes from a NASA PSI table; a value that is not in the table is None, never guessed
  * the only assumption kept (cabin pressure ~1 atm for ISS/Cygnus tests) is flagged in `pressure_source`
  * raw CSVs are re-processed at every start, so fixes to the ETL are never hidden by a stale cache

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
    description="Microgravity combustion data from NASA PSI, filterable, with a data-driven what-if explorer.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = "nasa_psi_master.csv"
MASTER_JSON = "nasa_psi_master.json"
CABIN_KPA = 101.3
MOON_BASE = {"o2_pct": 30.0, "pressure_kpa": 70.3}  # challenge text: >=30% O2 at <=10.2 psi

DOI = {
    "PSI-25": "10.60555/4qc4-de67", "PSI-26": "10.60555/58h4-dy80", "PSI-98": "10.60555/0t15-1z43",
    "PSI-99": "10.60555/2chp-m469", "PSI-100": "10.60555/040q-4h36", "PSI-69": "10.60555/mbq8-0451",
    "PSI-68": "10.60555/byp8-h018", "PSI-159": "10.60555/cyt1-kh64", "PSI-10": "10.60555/w4zy-bt45",
    "PSI-21": "10.60555/23e1-9950", "PSI-23": "10.60555/k57n-p030", "PSI-22": "10.60555/se5a-8c73",
    "PSI-20": "10.60555/vb93-3y72", "PSI-102": "10.60555/gd4k-8h62", "PSI-101": "10.60555/2np7-jv14",
}

COLUMNS = [
    "exp_id", "dataset_id", "investigation", "pillar", "doi", "fuel_state", "material", "geometry",
    "raw_sample_desc", "o2_pct", "o2_min_pct", "o2_max_pct", "o2_context", "pressure_kpa",
    "pressure_source", "airflow_cm_s", "min_airflow_cm_s", "diluent", "suppressant_pct",
    "burn_time_s", "co_generated_ppm", "flames_ignited", "outcome", "fire_sustained", "data_notes",
]


# =====================================================================
# 1. HELPERS
# =====================================================================

def read_table(path: str) -> pd.DataFrame:
    """Read a PSI CSV as text, with header whitespace normalised and empty rows dropped."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin1")
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    df = df.apply(lambda col: col.str.replace("�", "", regex=False).str.strip())
    return df[df.apply(lambda r: any(v != "" for v in r), axis=1)].reset_index(drop=True)


def find_file(data_dir: str, name: str) -> Optional[str]:
    for folder in (data_dir, os.path.join(data_dir, "raw"), BASE_DIR, os.path.join(BASE_DIR, "raw")):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def num(value) -> Optional[float]:
    """'~ 21.5' -> 21.5 ; '20 cm/s' -> 20.0 ; 'none' / '' -> None"""
    found = re.findall(r"-?\d+(?:\.\d+)?", str(value).replace(",", "."))
    return float(found[0]) if found else None


def nums(value) -> list:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]


def gas_name(text: str) -> str:
    t = str(text).upper()
    if "H2" in t and "CH4" in t:
        return "Hydrogen-Methane mix"
    if "H2" in t and "C2H" not in t and "CH" not in t:
        return "Hydrogen"
    if "C3H8" in t or "PROPANE" in t:
        return "Propane"
    if "C2H4" in t or "ETHYLENE" in t:
        return "Ethylene"
    if "C2H6" in t or "ETHANE" in t:
        return "Ethane"
    if "BUTANE" in t:
        return "Butane"
    if "CH4" in t or "METHANE" in t:
        return "Methane"
    return str(text).strip() or "Not specified"


def clean_material_name(text: str) -> str:
    t = str(text).lower()
    if "nomex" in t and "pmma" in t:
        return "PMMA & Nomex"
    if "sibal" in t or "spherical section" in t or "cotton" in t:
        return "SIBAL (Cotton-Fiberglass)"
    if "nomex" in t:
        return "Nomex"
    if "wax" in t or "candle" in t:
        return "Wax"
    if "silicone" in t:
        return "Silicone"
    if "ultem" in t:
        return "Ultem (PEI)"
    if "pmma" in t or "rod" in t or "film" in t or "slab" in t or "micron" in t:
        return "PMMA (Acrylic)"
    return "Not specified"


def clean_geometry_name(text: str) -> str:
    t = str(text).lower()
    if "rod" in t:
        return "Cylinder / Rod"
    if "spher" in t:
        return "Spherical Section"
    if "candle" in t:
        return "Candle"
    if "sibal" in t or "nomex" in t:
        return "Fabric Sheet"
    if "film" in t or "micron" in t:
        return "Thin Film"
    if "slab" in t or "thick" in t:
        return "Flat Slab"
    return "Not specified"


def record(**kw) -> dict:
    rec = {c: None for c in COLUMNS}
    rec.update(kw)
    rec["doi"] = DOI.get(rec["dataset_id"])
    if rec["o2_pct"] is not None:
        rec["o2_min_pct"] = rec["o2_min_pct"] if rec["o2_min_pct"] is not None else rec["o2_pct"]
        rec["o2_max_pct"] = rec["o2_max_pct"] if rec["o2_max_pct"] is not None else rec["o2_pct"]
    if rec["o2_context"] is None and rec["o2_max_pct"] is not None:
        rec["o2_context"] = "ambient"
    if rec["pressure_kpa"] is not None and rec["pressure_source"] is None:
        rec["pressure_source"] = "measured"
    return rec


# =====================================================================
# 2. ETL - one parser per PSI table
# =====================================================================

def parse_bass2(df):
    out = []
    for _, r in df.iterrows():
        air, fan = r["Air display"], r["Fan display"]
        text = f"{fan} {air}".lower()
        flows = [v for v in nums(air) if v <= 40]  # wind tunnel max is 40 cm/s
        if "no ign" in text or "would not stay lit" in text:
            outcome, sustained = "No Ignition", False
        elif "blowoff" in text:
            outcome, sustained = "Blowoff", False
        elif re.search(r"\bext\b|extinguish|\bout\b|\boff\b|0ext", text):
            outcome, sustained = "Extinction", False
        else:
            outcome, sustained = "Not reported", None
        o2_i, o2_f = num(r["Calibrated initial O2 % by vol"]), num(r["Calibrated final O2 % by vol"])
        co_i, co_f = num(r["Initial CO (ppm)"]), num(r["Final CO (ppm)"])
        out.append(record(
            exp_id=f"BASS2-{r['Test #']}", dataset_id="PSI-25", investigation="BASS-II", pillar="Prevention",
            fuel_state="Solid", material=clean_material_name(r["Fuel Sample Material"]),
            geometry=clean_geometry_name(r["Fuel Sample Material"]), raw_sample_desc=r["Fuel Sample Material"],
            o2_pct=o2_i, diluent="N2", pressure_kpa=CABIN_KPA,
            pressure_source="assumed: ISS cabin ~1 atm (not in table)",
            airflow_cm_s=flows[0] if flows else None, min_airflow_cm_s=min(flows) if flows else None,
            co_generated_ppm=max(0.0, co_f - co_i) if co_i is not None and co_f is not None else None,
            outcome=outcome, fire_sustained=sustained,
            data_notes=f"Air display: {air} | Fan: {fan} | final O2: {o2_f}",
        ))
    return out


def parse_bass(df):
    return [record(
        exp_id=f"BASS1-{r['Test #']}-{i + 1}", dataset_id="PSI-26", investigation="BASS", pillar="Prevention",
        fuel_state="Solid", material=clean_material_name(r["Fuel Sample Material"]),
        geometry=clean_geometry_name(r["Fuel Sample Material"]), raw_sample_desc=r["Fuel Sample Material"],
        outcome="Not reported", data_notes="PSI table lists only the sample; O2, flow and outcome are not reported",
    ) for i, r in df.iterrows()]


def parse_saffire1(df):
    out = []
    for _, r in df[df["Sample Number"].str.match(r"^S\d")].iterrows():
        o2 = nums(r["Percent O2"])
        out.append(record(
            exp_id=f"SAFFIRE1-{r['Sample Number']}", dataset_id="PSI-98", investigation="SAFFIRE-I",
            pillar="Prevention", fuel_state="Solid (Large Scale)", material=clean_material_name(r["Material"]),
            geometry=f"Large Panel ({r['Flow Direction']})",
            raw_sample_desc=f"{r['Material']} {r['Sample Thickness (cm)']} x {r['Sample Width (cm)']} x {r['Sample Length (cm)']} cm",
            o2_pct=o2[0] if o2 else None, o2_min_pct=min(o2) if o2 else None, o2_max_pct=max(o2) if o2 else None,
            pressure_kpa=CABIN_KPA, pressure_source="assumed: Cygnus cabin ~1 atm (not in table)",
            airflow_cm_s=num(r["Air Flow (cm/s)"]), min_airflow_cm_s=num(r["Air Flow (cm/s)"]),
            burn_time_s=num(r["Burn Time (s)"]), outcome="Burned", fire_sustained=True,
        ))
    return out


def parse_saffire2(df):
    out = []
    for _, r in df[df["Sample Number"].str.match(r"^2-\d")].iterrows():
        burn = r["Burn Time (s)"]
        insignificant = "insignificant" in burn.lower()
        out.append(record(
            exp_id=f"SAFFIRE2-{r['Sample Number']}", dataset_id="PSI-99", investigation="SAFFIRE-II",
            pillar="Prevention", fuel_state="Solid (Large Scale)",
            material=clean_material_name(r["Material"]) if r["Material"] else "Not specified",
            geometry=f"Panel {r['Samle Thickness']} thick ({r['Flow Direction']})",
            raw_sample_desc=f"{r['Material'] or 'material not given in PSI table'}, {r['Samle Thickness']}, "
                            f"{r['Sample Width (cm)']} x {r['Sample Length']}",
            o2_pct=num(r["Percent O2 (Note 1)"]), pressure_kpa=CABIN_KPA,
            pressure_source="assumed: Cygnus cabin ~1 atm (not in table)",
            airflow_cm_s=num(r["Air Flow (cm/s)"]), min_airflow_cm_s=num(r["Air Flow (cm/s)"]),
            burn_time_s=None if insignificant else num(burn),
            outcome="Insignificant Burn" if insignificant else "Burned", fire_sustained=not insignificant,
            data_notes=f"ug burn length: {r['?-g Burn Length']} | 1g burn: {r['1-g Burn Length']}, "
                       f"1g spread: {r['1-g Spread Rate']}",
        ))
    return out


def parse_saffire3(df):
    return [record(
        exp_id=f"SAFFIRE3-{r['Sample Number']}", dataset_id="PSI-100", investigation="SAFFIRE-III",
        pillar="Prevention", fuel_state="Solid (Large Scale)", material=clean_material_name(r["Material"]),
        geometry=f"Large Panel ({r['Direction']})",
        raw_sample_desc=f"{r['Material']} {r['Sample Thickness']}, {r['Sample Width']} x {r['Sample Length']}",
        pressure_kpa=CABIN_KPA, pressure_source="assumed: Cygnus cabin ~1 atm (not in table)",
        airflow_cm_s=num(r["Air Flow"]), min_airflow_cm_s=num(r["Air Flow"]),
        outcome="Not reported", data_notes="PSI table does not list O2 or outcome",
    ) for _, r in df[df["Sample Number"].str.match(r"^S\d")].iterrows()]


def parse_flex(df):
    out = []
    labels = {"Extinction": ("Extinction", False), "Disruption": ("Droplet Disruption", None),
              "Completion": ("Complete Burnout", True)}
    for _, r in df.iterrows():
        p = num(r["Ambient pressure; mmHg"])
        o2 = num(r["O initial ambient composition; mole fraction"])
        co2 = num(r["CO initial ambient composition; mole fraction"]) or 0.0
        he = num(r["He initial ambient composition; mole fraction"]) or 0.0
        diluent = "CO2" if co2 > 0 else ("He" if he > 0 else "N2")
        outcome, sustained = labels.get(r["Test end"], ("Not reported", None))
        out.append(record(
            exp_id=f"FLEX-{r['FLEX Test #']}", dataset_id="PSI-69", investigation="FLEX", pillar="Suppression",
            fuel_state="Liquid Droplet", material=r["Fuel"], geometry="Spherical Droplet",
            raw_sample_desc=f"{r['Fuel']} droplet d0={r['Droplet initial diameter; mm'] or '?'} mm ({r['FLEX Identifier']})",
            o2_pct=round(o2 * 100, 2) if o2 is not None else None,
            pressure_kpa=round(p * 0.133322, 2) if p else None, airflow_cm_s=0.0, min_airflow_cm_s=0.0,
            diluent=diluent, suppressant_pct=round(max(co2, he) * 100, 1),
            burn_time_s=num(r["Burn time; s"]), outcome=outcome, fire_sustained=sustained,
            data_notes=f"extinction diameter: {r['Visible flame extinction diameter; mm'] or '-'} mm"
                       + (" | pressure recorded as 0 in PSI table" if p == 0 else ""),
        ))
    return out


def parse_flex2(df):
    out = []
    for i, r in df.iterrows():
        o2 = [v * 100 for v in nums(r["Oxygen Mole Fraction"])]
        p = nums(r["Ambient Pressure (atm)"])
        flow = nums(r["Flow Velocity (cm/s)"])
        out.append(record(
            exp_id=f"FLEX2-{i + 1}", dataset_id="PSI-68", investigation="FLEX-2", pillar="Suppression",
            fuel_state="Liquid Droplet", material=r["Sample (Fuel Type)"].replace(" (pure)", "").replace(" (mixture)", ""),
            geometry="Spherical Droplet",
            raw_sample_desc=f"{r['Sample (Fuel Type)']} in {r['Ambient Gas Mixture']}, d0 {r['Initial Droplet Diameter (mm)']} mm",
            o2_min_pct=min(o2) if o2 else None, o2_max_pct=max(o2) if o2 else None, o2_context="ambient",
            diluent=r["Ambient Gas Mixture"], airflow_cm_s=None, min_airflow_cm_s=min(flow) if flow else None,
            outcome="Test matrix (no per-test result)",
            data_notes=f"Test matrix row: O2 {r['Oxygen Mole Fraction']}; pressure {r['Ambient Pressure (atm)']} atm "
                       f"({min(p) * 101.325:.0f}-{max(p) * 101.325:.0f} kPa); flow {r['Flow Velocity (cm/s)']} cm/s" if p else "",
        ))
    return out


def _cfig(r, inverse: bool):
    hot = r["Hot ignition"].strip().lower()
    se = r.get("Self-ext or flow term", r.get("Self -ext or flow term", ""))
    if hot == "no":
        outcome, sustained = "No Ignition", False
    elif hot == "flash":
        outcome, sustained = "Flash (no sustained flame)", False
    elif se == "SE":
        outcome, sustained = "Self-Extinguished", False
    elif se == "FT":
        outcome, sustained = "Flow Terminated (Sustained)", True
    else:
        outcome, sustained = "Not reported", None
    o2 = num(r["XO2"] if inverse else r["XO2 before test"])
    p = num(r["approx p (bar)"])
    cool = r.get("Cool flame?", "")
    return outcome, sustained, o2, p, cool


def parse_cfig(df, inverse=False):
    out = []
    for _, r in df.iterrows():
        outcome, sustained, o2, p, cool = _cfig(r, inverse)
        kind = "Inverse" if inverse else "Normal"
        out.append(record(
            exp_id=f"CFIG-{'INV-' if inverse else ''}{r['Test'].replace(' ', '')}", dataset_id="PSI-159",
            investigation="ACME / CFI-G", pillar="Fundamental (high O2)", fuel_state="Gas Jet",
            material=gas_name(r["Fuel"]), geometry=f"Spherical Burner ({kind} flame)",
            raw_sample_desc=f"{r['Fuel']} {kind.lower()} flame, test {r['Test']}",
            o2_pct=round(o2 * 100, 2) if o2 is not None else None,
            o2_context="burner stream (inverse flame)" if inverse else "ambient",
            pressure_kpa=round(p * 100, 1) if p is not None else None, airflow_cm_s=0.0, min_airflow_cm_s=0.0,
            diluent="N2", burn_time_s=num(r["burn time (sec)"]), outcome=outcome, fire_sustained=sustained,
            data_notes="; ".join(x for x in [f"cool flame: {cool}" if cool else "", r.get("Comments", "")] if x),
        ))
    return out


def parse_flame_design(df):
    out = []
    for _, r in df.iterrows():
        o2, p = num(r["XO2 (init)"]), num(r["pinit (atm)"])
        out.append(record(
            exp_id=f"FD-{r['Test']}", dataset_id="PSI-10", investigation="ACME / Flame Design",
            pillar="Fundamental (high O2)", fuel_state="Gas Jet", material=gas_name(r["Fuel"]),
            geometry="Spherical Burner (Normal flame)", raw_sample_desc=f"{r['Fuel']} Xfuel={r['Xfuel']}",
            o2_pct=round(o2 * 100, 2) if o2 is not None else None,
            pressure_kpa=round(p * 101.325, 1) if p is not None else None, airflow_cm_s=0.0, min_airflow_cm_s=0.0,
            diluent="N2", outcome="Not reported", data_notes=f"Tad: {r['Tad (K)']} K",
        ))
    return out


def parse_cld(df):
    out = []
    for i, r in df[df["Test Day"] != ""].iterrows():
        tt, bottle = r["Test Type"].lower(), r["Fuel Bottle"].upper()
        if "ethylene" in tt or "C2H" in bottle:
            fuel = "Ethylene"
        elif "methane" in tt or "CH" in bottle:
            fuel = "Methane"
        else:
            fuel = "Not specified"
        out.append(record(
            exp_id=f"CLD-{r['Test Day']}-{i + 1}", dataset_id="PSI-21", investigation="ACME / CLD Flame",
            pillar="Fundamental", fuel_state="Gas Jet", material=fuel, geometry="Coflow Laminar Diffusion Burner",
            raw_sample_desc=f"{r['Test Type']}: {fuel} {r['Fuel conc. Range']} in N2",
            o2_pct=21.0, o2_context="ambient (air coflow, per PSI description)", outcome="Not reported",
            flames_ignited=num(r["Flames ignited"]),
            data_notes=f"PSI bottle label '{r['Fuel Bottle']}' (spreadsheet auto-fill typo corrected from test type)",
        ))
    return out


def parse_sflame(df):
    out = []
    for i, r in df[df["Test Day*"] != ""].iterrows():
        atm = r["Atmosphere*"]
        o2 = nums(atm.split("O2")[0])
        out.append(record(
            exp_id=f"SFLAME-{r['Test Day*']}-{i + 1}", dataset_id="PSI-23", investigation="ACME / s-Flame",
            pillar="Fundamental", fuel_state="Gas Jet", material=gas_name(r["Fuel Bottle"]),
            geometry="Spherical Burner", raw_sample_desc=f"{r['Fuel Bottle']} | {atm}",
            o2_pct=max(o2) if len(o2) == 1 else None, o2_min_pct=min(o2) if o2 else None,
            o2_max_pct=max(o2) if o2 else None, o2_context="ambient",
            diluent="He" if "He" in atm else "N2", flames_ignited=num(r["Flames Ignited*"]),
            outcome="Not reported", data_notes="Atmosphere ~1 atm; pressure rises and O2 drops with vitiation",
        ))
    return out


def parse_efield(df):
    out = []
    for i, r in df[df["Test Day"] != ""].iterrows():
        out.append(record(
            exp_id=f"EFIELD-{r['Test Day']}-{i + 1}", dataset_id="PSI-22", investigation="ACME / E-FIELD",
            pillar="Fundamental (partial-gravity analogue)", fuel_state="Gas Jet", material=gas_name(r["Fuel Bottle"]),
            geometry="Coflow Burner (E-Field)" if "coflow" in r["Test Type"].lower() else "Gas-Jet Burner (E-Field)",
            raw_sample_desc=f"{r['Fuel Bottle'] or 'bottle not given'}, conc {r['Fuel Conc'] or '?'}, "
                            f"mesh polarity {r['Mesh Polarity'] or '?'}",
            o2_pct=21.0, o2_context="ambient (synthetic air coflow, per PSI description)",
            flames_ignited=num(r["Flames Ignited"]), outcome="Not reported",
            data_notes="Electric field mimics a buoyancy-like force (ion-driven convection)",
        ))
    return out


def parse_bre(df):
    out = []
    for i, r in df.iterrows():
        raw = r["Oxygen Conc (percent)"]
        fixed = re.sub(r"\b(\d{2})00%", r"\1", raw)  # '3400%' / '2100%' are spreadsheet % formatting errors
        o2 = [v for v in nums(fixed) if v <= 100]
        flames = num(r["Flames Ignited"])
        out.append(record(
            exp_id=f"BRE-{r['Year']}-{r['Month']}-{r['Date']}-{i + 1}", dataset_id="PSI-20",
            investigation="ACME / BRE", pillar="Prevention", fuel_state="Gas Jet (emulates burning solid)",
            material=gas_name(r["Fuel Bottle"]), geometry=f"Porous Spherical Burner ({r['Burner Size (mm)']} mm)",
            raw_sample_desc=f"{r['Fuel Bottle']}, O2 levels tested that day: {fixed}",
            o2_pct=o2[0] if len(o2) == 1 else None, o2_min_pct=min(o2) if o2 else None,
            o2_max_pct=max(o2) if o2 else None, o2_context="ambient", diluent="N2",
            pressure_source="varied, not listed in PSI table", flames_ignited=flames,
            outcome="No Ignition" if flames == 0 else "Ignited", fire_sustained=False if flames == 0 else None,
            data_notes="One row = one test day with several O2 levels" + (f" | raw O2 '{raw}'" if raw != fixed else ""),
        ))
    return out


def parse_same(df):
    mat_col = "Sample Material"
    return [record(
        exp_id=f"SAME-{r['Sample #']}", dataset_id="PSI-102", investigation="SAME", pillar="Detection",
        fuel_state="Solid (heated, no flame)", material=r[mat_col].split(":")[0].strip(), geometry="Heated sample",
        raw_sample_desc=r[mat_col], airflow_cm_s=num(r["Air Flow (cm/s)"]),
        outcome="Smoke generated (no flame)",
        data_notes=f"weight loss {r['Weight loss rate (mg/min)'] or '-'} mg/min; aging {r['Aging Duration (minutes)'] or '-'} min",
    ) for _, r in df.iterrows()]


def parse_same_r(df):
    return [record(
        exp_id=f"SAMER-{r['SAME Test #']}-{i + 1}", dataset_id="PSI-101", investigation="SAME-R", pillar="Detection",
        fuel_state="Solid (heated, no flame)", material=r["Sample Type"], geometry="Heated sample",
        raw_sample_desc=f"{r['Sample Type']}: {r['Description']}", airflow_cm_s=num(r["Inlet velocity (cm/s)"]),
        outcome="Smoke generated (no flame)",
        data_notes=f"Rt/R0 {r['Rt/R0'] or '-'}; net aging {r['Net aging time (sec)'] or '-'} s",
    ) for i, r in df.iterrows() if r["Sample Type"]]


SOURCES = [
    ("PSI-25_Experimental table_BASS-II.csv", parse_bass2),
    ("PSI-26_Experimental table_BASS.csv", parse_bass),
    ("PSI-98_Experimental table_SAFFIRE-1.csv", parse_saffire1),
    ("PSI-99_Experimental table_SAFFIRE-2.csv", parse_saffire2),
    ("PSI-100_Experimental table_SAFFIRE-3.csv", parse_saffire3),
    ("PSI-69_Experimental table_FLEX.csv", parse_flex),
    ("PSI-68_Experimental table_Flex2.csv", parse_flex2),
    ("PSI-159_Experimental table_CFI-G_Normal Flames.csv", lambda d: parse_cfig(d, inverse=False)),
    ("PSI-159_Experimental table_CFI-G_Inverse Flames.csv", lambda d: parse_cfig(d, inverse=True)),
    ("PSI-10_Experimental table_Flame Design_Normal Flame.csv", parse_flame_design),
    ("PSI-21_Experimental table_CLD Flame.csv", parse_cld),
    ("PSI-23_Experimental table_ACME s-Flame.csv", parse_sflame),
    ("PSI-22_Experimental table_ACME E-FIELD Flames.csv", parse_efield),
    ("PSI-20_Experimental table_ACME BRE.csv", parse_bre),
    ("PSI-102_Experimental table_SAME.csv", parse_same),
    ("PSI-101_Experimental table_SAME-R.csv", parse_same_r),
]


# Did a flame establish itself (burn beyond the igniter)? None = the PSI table does not say.
FLAME_ESTABLISHED = {
    "Burned": True, "Complete Burnout": True, "Extinction": True, "Droplet Disruption": True, "Blowoff": True,
    "Self-Extinguished": True, "Flow Terminated (Sustained)": True, "Ignited": True,
    "No Ignition": False, "Flash (no sustained flame)": False, "Insignificant Burn": False,
}


def build_unified_dataset(data_dir: str = BASE_DIR) -> pd.DataFrame:
    """Re-parses every raw PSI CSV found. Falls back to the saved master only if no raw file exists."""
    records = []
    LOADED_FILES.clear()
    MISSING_FILES.clear()
    for name, parser in SOURCES:
        path = find_file(data_dir, name)
        if path is None:
            MISSING_FILES.append(name)
            continue
        records.extend(parser(read_table(path)))
        LOADED_FILES.append(name)

    master_path = os.path.join(data_dir, MASTER_CSV)
    if not records:
        return pd.read_csv(master_path) if os.path.exists(master_path) else pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(records, columns=COLUMNS)
    df["flame_established"] = df["outcome"].map(FLAME_ESTABLISHED)
    df["moon_base_relevant"] = (df["o2_max_pct"] >= MOON_BASE["o2_pct"]) & df["o2_context"].fillna("").str.startswith("ambient")
    df.to_csv(master_path, index=False)
    df.to_json(os.path.join(data_dir, MASTER_JSON), orient="records", indent=2)
    return df


LOADED_FILES: list = []
MISSING_FILES: list = []
df_master = build_unified_dataset()


# =====================================================================
# 3. API HELPERS
# =====================================================================

def to_records(data: pd.DataFrame) -> list:
    return data.astype(object).where(pd.notna(data), None).to_dict(orient="records")


def stat(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"min": None, "max": None, "mean": None, "known": 0}
    return {"min": float(s.min()), "max": float(s.max()), "mean": round(float(s.mean()), 2), "known": int(len(s))}


def mat_match(series: pd.Series, material: str) -> pd.Series:
    """Whole-word, case-insensitive match: 'Ethane' must not match 'Methane'."""
    return series.str.contains(r"\b" + re.escape(material), case=False, na=False, regex=True)


def in_range(low: pd.Series, high: pd.Series, lo: float, hi: float, include_unknown: bool) -> pd.Series:
    """Row matches if its [low, high] interval overlaps [lo, hi]; unknown values follow include_unknown."""
    known = low.notna() | high.notna()
    l, h = low.fillna(high), high.fillna(low)
    return (known & (h >= lo) & (l <= hi)) | (~known & include_unknown)


# =====================================================================
# 4. ENDPOINTS
# =====================================================================

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "total_experiments_loaded": len(df_master),
        "investigations_count": int(df_master["investigation"].nunique()) if not df_master.empty else 0,
        "files_loaded": LOADED_FILES,
        "files_missing": MISSING_FILES,
    }


@app.get("/summary")
def get_summary_stats():
    """Aggregated statistics for dashboard KPI cards and charts. Ranges are computed on known values only."""
    if df_master.empty:
        return {"error": "No data loaded"}
    return {
        "total_experiments": len(df_master),
        "by_investigation": df_master["investigation"].value_counts().to_dict(),
        "by_pillar": df_master["pillar"].value_counts().to_dict(),
        "by_fuel_state": df_master["fuel_state"].value_counts().to_dict(),
        "by_material": df_master["material"].value_counts().head(12).to_dict(),
        "by_outcome": df_master["outcome"].value_counts().to_dict(),
        "o2_range": stat(df_master["o2_pct"].fillna(df_master["o2_max_pct"])),
        "pressure_range_kpa": stat(df_master["pressure_kpa"]),
        "airflow_range_cm_s": stat(df_master["airflow_cm_s"]),
        "moon_base_relevant_tests": int(df_master["moon_base_relevant"].sum()),
        "data_completeness_pct": {
            c: round(100 * df_master[c].notna().mean(), 1) for c in ["o2_pct", "pressure_kpa", "airflow_cm_s", "burn_time_s"]
        },
    }


@app.get("/filters")
def get_available_filters():
    return {
        "investigations": sorted(df_master["investigation"].dropna().unique().tolist()),
        "pillars": sorted(df_master["pillar"].dropna().unique().tolist()),
        "fuel_states": sorted(df_master["fuel_state"].dropna().unique().tolist()),
        "materials": sorted(df_master["material"].dropna().unique().tolist()),
        "geometries": sorted(df_master["geometry"].dropna().unique().tolist()),
        "outcomes": sorted(df_master["outcome"].dropna().unique().tolist()),
        "diluents": sorted(df_master["diluent"].dropna().unique().tolist()),
    }


@app.get("/experiments")
def query_experiments(
    investigation: Optional[str] = Query(None, description="e.g. BASS-II, FLEX, SAFFIRE"),
    fuel_state: Optional[str] = Query(None, description="Solid, Liquid Droplet, Gas Jet"),
    material: Optional[str] = Query(None, description="Material substring (PMMA, SIBAL, Heptane, Methane...)"),
    outcome: Optional[str] = Query(None, description="Outcome substring"),
    pillar: Optional[str] = Query(None, description="Prevention, Detection, Suppression, Fundamental"),
    min_o2: float = Query(0.0), max_o2: float = Query(100.0),
    min_airflow: float = Query(0.0), max_airflow: float = Query(100.0),
    min_pressure: float = Query(0.0), max_pressure: float = Query(500.0),
    include_unknown: bool = Query(True, description="Keep tests where the filtered value is not reported by NASA"),
    limit: int = Query(200, ge=1, le=5000),
):
    data = df_master
    for col, val in [("investigation", investigation), ("fuel_state", fuel_state), ("material", material),
                     ("outcome", outcome), ("pillar", pillar)]:
        if val:
            data = data[data[col].str.contains(val, case=False, na=False, regex=False)]
    data = data[
        in_range(data["o2_min_pct"], data["o2_max_pct"], min_o2, max_o2, include_unknown)
        & in_range(data["min_airflow_cm_s"], data["airflow_cm_s"], min_airflow, max_airflow, include_unknown)
        & in_range(data["pressure_kpa"], data["pressure_kpa"], min_pressure, max_pressure, include_unknown)
    ]
    out = to_records(data.head(limit))
    return {"total_matched": len(data), "returned": len(out), "experiments": out}


@app.get("/simulate-risk")
def simulate_habitat_fire_risk(
    environment: str = Query("ISS (Microgravity 0g)", description="ISS (Microgravity 0g), Lunar Habitat (0.16g), Martian Habitat (0.38g), Deep Space Transit"),
    o2_pct: float = Query(21.0, ge=10.0, le=100.0),
    pressure_kpa: float = Query(101.3, ge=30.0, le=320.0, description="101.3 = ISS; Moon base plan ~70 kPa (10.2 psi)"),
    airflow_cm_s: float = Query(5.0, ge=0.0, le=55.0),
    material: str = Query("PMMA"),
    top_k: int = Query(6, ge=1, le=20),
):
    """
    Data-driven what-if explorer: finds the NASA PSI tests closest to the scenario and reports what happened in them.
    It is NOT a validated physical model; all limitations are returned in `limitations`.
    """
    limitations = ["Nearest-neighbour summary of NASA PSI tests, not a validated flammability model."]
    pool = df_master[df_master["o2_context"].fillna("").str.startswith("ambient")]
    subset = pool[mat_match(pool["material"], material)].copy()
    if subset.empty:
        limitations.append(f"No PSI test uses '{material}'; nearest tests are from other materials.")
        subset = pool.copy()

    o2_ref = subset["o2_pct"].fillna((subset["o2_min_pct"] + subset["o2_max_pct"]) / 2)
    subset = subset[o2_ref.notna()].copy()
    o2_ref = o2_ref[o2_ref.notna()]
    parts = pd.DataFrame({
        "o2": ((o2_ref - o2_pct) / 10.0) ** 2,
        "p": ((subset["pressure_kpa"] - pressure_kpa) / 50.0) ** 2,
        "flow": ((subset["airflow_cm_s"] - airflow_cm_s) / 15.0) ** 2,
    })
    subset["dims_compared"] = parts.notna().sum(axis=1)
    subset["similarity_distance"] = np.sqrt(parts.mean(axis=1, skipna=True))
    subset["similarity_score_pct"] = (100.0 / (1.0 + subset["similarity_distance"])).round(1)
    ranked = subset.sort_values(["similarity_distance", "dims_compared"], ascending=[True, False])
    nearest = ranked.head(top_k)
    evidence = ranked[ranked["flame_established"].notna()].head(top_k)  # only tests whose outcome PSI reports

    known = evidence["flame_established"].astype(bool)
    ratio = float(known.mean()) if len(known) else None
    if 0 < len(known) < 3:
        limitations.append(f"Score based on only {len(known)} test(s) with a reported outcome.")
    score = int(round(ratio * 100)) if ratio is not None else None
    if score is None:
        level = "UNKNOWN (no reported outcome in nearest tests)"
    elif score >= 67:
        level = "HIGH"
    elif score >= 34:
        level = "ELEVATED"
    else:
        level = "LOW"

    env = environment.lower()
    if any(k in env for k in ["lunar", "moon", "0.16", "mars", "martian", "0.38"]):
        limitations.append("PSI contains no partial-gravity combustion data (LUCI / FM2 pending); microgravity and "
                           "1g results only bound what happens on the Moon or Mars.")
    tested_max = subset["o2_max_pct"].max() if not subset.empty else None
    if tested_max is not None and o2_pct > tested_max:
        limitations.append(f"Requested O2 {o2_pct}% is above the highest O2 tested for this material "
                           f"({tested_max:.1f}%): extrapolation.")
    if pressure_kpa < 70 and not (subset["pressure_kpa"] < 75).any():
        limitations.append("No PSI test for this material at reduced pressure.")

    recs = []
    if score is not None:
        recs.append(f"A flame established itself in {int(known.sum())} of the {len(known)} most similar NASA tests "
                    f"({', '.join(evidence['exp_id'].head(4))}).")
    low_flow_ext = df_master[(df_master["investigation"] == "BASS-II")
                             & mat_match(df_master["material"], material)
                             & df_master["outcome"].isin(["Extinction"])]
    if not low_flow_ext.empty:
        recs.append(f"BASS-II: {len(low_flow_ext)} {material} tests ended in extinction when airflow was reduced "
                    f"(e.g. {', '.join(low_flow_ext['exp_id'].head(3))}) - reducing ventilation is supported by these tests.")
    if o2_pct >= MOON_BASE["o2_pct"]:
        recs.append("Planned Moon-base atmosphere (>=30% O2, ~70 kPa) is richer in O2 than Earth air; few PSI solid-material "
                    "tests reach it, so material screening at this O2 is a data gap.")
    sup = df_master[(df_master["investigation"] == "FLEX") & (df_master["diluent"].isin(["CO2", "He"]))]
    if not sup.empty:
        recs.append(f"FLEX: {int((sup['outcome'] == 'Extinction').sum())} of {len(sup)} droplet tests with added CO2/He ended "
                    "in extinction - evidence base for gaseous suppressants.")

    mat = material.lower()
    cfig_cool = df_master[(df_master["dataset_id"] == "PSI-159")
                          & df_master["data_notes"].fillna("").str.contains("cool flame: yes")
                          & mat_match(df_master["material"], material)]
    if not cfig_cool.empty:
        cool = "OBSERVED in PSI data (ACME CFI-G)"
    elif "heptane" in mat:
        cool = "REPORTED in FLEX publications for n-heptane (not flagged in the PSI table)"
    else:
        cool = "NO DATA"

    co = nearest["co_generated_ppm"].dropna()
    return {
        "scenario": {
            "environment": environment, "o2_pct": o2_pct, "pressure_kpa": pressure_kpa,
            "airflow_cm_s": airflow_cm_s, "effective_convective_flow_cm_s": airflow_cm_s, "material": material,
        },
        "risk_assessment": {
            "flammability_score": score,
            "risk_level": level,
            "cool_flame_reignition_risk": cool,
            "estimated_co_spike_ppm": round(float(co.mean()), 1) if not co.empty else None,
            "sustained_burn_ratio_in_nearest_tests": round(ratio * 100, 1) if ratio is not None else None,
            "method": "share of the most similar NASA PSI tests (with reported outcome) in which a flame established itself",
            "evidence_tests": evidence["exp_id"].tolist(),
        },
        "safety_recommendations": recs,
        "limitations": limitations,
        "nearest_nasa_experiments": to_records(nearest),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)v
