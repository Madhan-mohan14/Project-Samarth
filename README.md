# 🌾 Project Samarth — Agriculture × Climate Q&A

*Project Samarth* is an intelligent, data-driven chatbot that answers natural-language questions about Indian *agriculture and climate data*.  
It combines *rainfall data (IMD 1901–2015)* and *crop production data (MoA 2009–2015)* to enable interactive, explainable analysis — all through a *Streamlit-powered interface*.

---

## 🚀 Overview

Built as part of the **Bharat Digital Fellowship Challenge**, *Project Samarth* demonstrates how public data from [data.gov.in](https://data.gov.in) can be made accessible through conversational AI.

The system dynamically interprets user questions, queries a pre-processed SQLite database, and generates accurate, data-backed answers with traceable citations.

---

## 🧠 Features

✅ Conversational NLU-style chatbot (Groq Llama 3.1 API)  
✅ Integration of rainfall & crop datasets across Indian states  
✅ Correlation engine for rainfall vs. crop production trends  
✅ Streamlit dark-themed interactive UI  
✅ Local SQLite database — offline ready  
✅ Citation-aware, with dataset provenance  

---

## 🗂 Data Sources

| Dataset | Description | Years | Source |
|----------|--------------|-------|--------|
| IMD Area-Weighted Annual Rainfall | State-wise rainfall data (mm) | 1901–2015 | [data.gov.in - IMD](https://data.gov.in) |
| State/UT-wise Production of Principal Crops | Crop production by state (in metric tonnes) | 2009–2015 | [data.gov.in - MoA](https://data.gov.in) |

---

## 🧩 Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python 3, SQLite  
- **Data Processing:** Pandas, NumPy  
- **Statistical Analysis:** SciPy  
- **LLM:** Groq API (Llama 3.1 8B Instant)  
- **Deployment:** Streamlit Cloud  

##🧑‍💻 Authorship Note

This repository represents the **authentic and enhanced version** of *Project Samarth*, designed and developed solely by **Madhan Mohan**.  
Earlier versions were shared privately for feedback and subsequently misused without authorization.  
This current version is protected under **full copyright**, rebuilt independently, and contains multiple technical improvements — including a redesigned chatbot logic, efficient query pipeline, and improved UI.

---

## ⚖️ License

© 2025 *Madhan Mohan*. **All Rights Reserved.**  
No part of this project, including its code, datasets, or design, may be copied, modified, or redistributed without explicit written permission from the author.  
View-only access does not grant reuse or replication rights.

---

## 🪙 Credits

- Indian Government Open Data Platform — [data.gov.in](https://data.gov.in)  
- India Meteorological Department (IMD)  
- Ministry of Agriculture & Farmers Welfare  
- Groq API for Llama 3.1 Integration

## Links
Demo Video: https://drive.google.com/drive/folders/1yXIh_SZUQ89EhYe_bq5xO59pptYojNUR?usp=sharing

Live App: https://project-samarth14.streamlit.app
- ## 🛠 Local Setup

Clone and run locally:

```bash
git clone https://github.com/madhan-mohan14/Project-Samarth.git
cd Project-Samarth
pip install -r requirements.txt
streamlit run app.py
