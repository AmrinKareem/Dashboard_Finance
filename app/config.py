import os
import streamlit as st
DB_USER = st.secrets.get("DB_USER")
DB_PASSWORD = st.secrets.get("DB_PASSWORD")
DB_HOST = st.secrets.get("DB_HOST")
DB_NAME = st.secrets.get("DB_NAME")
DB_DRIVER = st.secrets.get("DB_DRIVER")
URL = st.secrets["LLM_API_URL"]
api_key = st.secrets["LLM_API_KEY"] or os.getenv("LLM_API_KEY")
TEMPERATURE = 0.4
MAX_TOKENS = 5000
FREQUENCY_PENALTY = 0.5
metric_map = {"forecast_costs_at_completion": "rForecast",
"ytd_actual": "rYearAct"
}
PROJECT_KEYS_DEFAULT = ["iProjNo", "iProjYear", "cProjDesc", "cClientDesc"]
projects_list = {
    "2035": 2035,
    "2121": 2121,
    "2171": [2171, 2172],
    "2172": [2171, 2172],
    "2222": 2222,
    "2300": [2300, 2301],
    "2301": [2300, 2301],
    "2302": [2302, 2303],
    "2303": [2302, 2303],
    "2369": 2369,
    "2377": [2377, 8353],
    "8353": [2377, 8353],
    "2412": [2412, 2413],
    "2413": [2412, 2413],
    "2462": 2462,
    "2523": 2523,
    "2543": 2543,
    "2631": 2631,
    "2642": [2642, 2883],
    "2883": [2642, 2883],
    "2682": 2682,
    "2683": 2683,
    "2705": 2705,
    "2706": 2706,
    "2722": 2722,
    "2734": [2734, 2735],
    "2735": [2734, 2735],
    "2745": 2745,
    "2790": [2790, 2791],
    "2791": [2790, 2791],
    "2792": [2792, 2793],
    "2793": [2792, 2793],
    "2800": 2800,
    "2820": 2820,
    "2824": 2824,
    "2859": 2859,
    "2891": 2891,
    "2913": 2913,
    "2993": 2993,
    "7279": 7279,
    "8288": 8288,
    "8405": 8405
}

# CONFIGS
# api_key = st.secrets["OPENAI_API_KEY"] or os.getenv("OPENAI_API_KEY") # not required for local
# client = OpenAI(api_key=api_key)  
# # MODEL = "gpt-4o"