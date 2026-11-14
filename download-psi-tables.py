"""Downloads the 16 original NASA PSI experimental tables used by the backend into ./raw/
Run once on a laptop with internet:  python download_psi_tables.py
"""
import pathlib

import requests

BASE = "https://psi.nasa.gov/geode-py/ws"
IDS = ["PSI-25", "PSI-26", "PSI-98", "PSI-99", "PSI-100", "PSI-69", "PSI-68", "PSI-159",
       "PSI-10", "PSI-21", "PSI-23", "PSI-22", "PSI-20", "PSI-102", "PSI-101"]
WANTED_EXTRA = "Inverse Flame.csv"  # Flame Design inverse table is not used; everything else is

out = pathlib.Path(__file__).parent / "raw"
out.mkdir(exist_ok=True)

for psi_id in IDS:
    meta = requests.get(f"{BASE}/repo/investigations/{psi_id}", timeout=120).json()
    for table in meta.get("experimentalTable") or []:
        name = table["file_name"]
        if name.endswith(WANTED_EXTRA):
            continue
        url = requests.get(f"{BASE}/studies/{psi_id}/download",
                           params={"file": name, "version": table["version"], "redirect": "false"},
                           timeout=60).text.strip()
        (out / name).write_bytes(requests.get(url, timeout=120).content)
        print("OK", name)
