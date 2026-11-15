# Flame in Freefall · NASA Space Apps Challenge 2026

> *Team Star Sailors* (Alba Iulia, Romania)  
> SI-powered fire-safety insights from NASA microgravity combustion datasets (PSI).

---

## About the Project
*Flame in Freefall* is an interactive, Super-Intelligence (SI)-powered dashboard designed to help aerospace engineers, researchers, and students navigate, compare, and interpret decades of NASA microgravity combustion research. 

Built on top of official data from NASA's *Physical Sciences Informatics (PSI)* repository, our platform unifies 1,021 experimental records across 15 spaceflight investigations to address critical fire safety hazards in microgravity and future habitats on the Moon and Mars.

---

## Key Features

1. *Overview Dashboard:* Instant KPI metrics, oxygen ranges, and test distribution across NASA's 4 fire-safety pillars: Prevention, Detection, Suppression, and Fundamental Combustion.
2. *Interactive Explorer:* Filter and search through all 1,021 experiments by investigation, material, oxygen concentration, and outcome, complete with official DOI citations.
3. *Condition Map:* Visual scatter plots mapping Oxygen ($\text{O}_2$) vs. Pressure and Airflow, highlighting the planned *Moon-Base safety zone* ($\ge 30\% \text{ $\text{O}_2$}$ at $\le 10.2\text{ psi} / 70.3\text{ kPa}$) to expose data gaps.
4. *What-If Risk Simulator:* Nearest-neighbor multi-dimensional scoring engine that evaluates custom habitat scenarios (ISS, Lunar Habitat, Martian Habitat) against real NASA tests.
5. *Grounded AI Assistant (Ask AI):* Powered by *Google Gemini 3.5*, answering user questions strictly from the NASA PSI tables with fully traceable, clickable [EXP-ID] source citations.

---

## Tech Stack

- *Backend:* Python, FastAPI, Uvicorn, Pandas, NumPy, Pydantic.
- *AI Integration:* Google GenAI SDK (google-genai), Retrieval-Augmented Generation (RAG) grounded on NASA PSI records.
- *Frontend:* HTML5, CSS3, JavaScript (ES6+), Plotly.js for interactive charts.
- *Data Source:* NASA Physical Sciences Informatics (PSI) API (psi.nasa.gov), CC0 Public Domain.
