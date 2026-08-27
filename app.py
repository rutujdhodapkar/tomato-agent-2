import base64
import io
import json
import os
import random
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import requests
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

def get_secret_or_env(key, default=None):
    """
    Safely read a value from Streamlit secrets or environment variables.
    Does not crash if secrets.toml does not exist.
    """
    try:
        value = st.secrets.get(key, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass

    value = os.getenv(key)

    if value not in (None, ""):
        return value

    return default


APP_TITLE = "Agri Super Agent"

MODEL_API_KEY = get_secret_or_env("MODEL_API_KEY")

MODEL_API_URL = get_secret_or_env(
    "MODEL_API_URL",
    ""
)

MODEL_NAME = get_secret_or_env(
    "MODEL_NAME",
    ""
)

REASONING_MODEL = get_secret_or_env(
    "REASONING_MODEL",
    MODEL_NAME
)

SARVAM_API_KEY = get_secret_or_env("SARVAM_API_KEY")

SARVAM_TRANSLATE_URL = get_secret_or_env(
    "SARVAM_TRANSLATE_URL",
    "https://api.sarvam.ai/translate"
)

USER_DB = "users.json"
EXPORT_DIR = "exports"

MAX_RETRIES = 5
REQUEST_TIMEOUT = 120


# ============================================================
# TRANSLATIONS
# ============================================================

TRANSLATIONS = {
    "English": {
        "home": "Home",
        "chat": "Chat",
        "shops": "Shop",
        "doctors": "Doctors",
        "contact": "Contact",

        "login": "Login",
        "username": "Username",
        "password": "Password",
        "continue": "Continue",
        "login_success": "Login successful",
        "account_created": "Account created",

        "upload": "Upload Leaf Image",
        "analyze": "Analyze",

        "agent_control": "Agent Control Panel",
        "select_language": "Select Language",

        "farm_location": "Farm location",

        "agent_status": "Agent Status",
        "idle": "Idle",

        "running_pipeline": "Running full farm intelligence pipeline...",
        "getting_location": "Getting location information...",
        "fetching_soil": "Fetching soil insights...",
        "fetching_water": "Fetching water and weather insights...",
        "analyzing_image": "Analyzing image...",
        "thinking": "Thinking...",

        "analysis_complete": "Analysis Complete!",
        "full_analysis": "Full Analysis Report",

        "crop_identified": "Crop Identified",
        "disease_status": "Disease Status",
        "condition_assessment": "Condition Assessment",
        "actionable_prescription": "Actionable Prescription",

        "soil_moisture": "Soil and Moisture Insights",
        "water_weather": "Water and Weather Outlook",

        "risk_urgency": "Risk and Urgency",
        "risk_level": "Risk Level",

        "fertilizer_recommendations": "Fertilizer Recommendations",

        "no_description": "No description available.",
        "no_solution": "No solution provided.",
        "no_soil": "No soil insights available.",
        "no_water": "No water forecast available.",
        "no_fertilizer": "No fertilizer recommendations available.",

        "quick_actions": "Quick Agent Actions",
        "select_analysis": "Select analysis",
        "run_analysis": "Run analysis",
        "do_all_analysis": "Do all analysis",
        "run_core_layers": "Run core layers",
        "queued": "Queued",
        "all_queued": "All analyses queued!",
        "layers_queued": "All layer analyses queued.",

        "cost_estimation": "Cost Estimation",
        "location_city": "Location (city/region)",
        "crop_name": "Crop name",
        "total_acres": "Total acres",
        "total_invested": "Total invested",
        "estimate_profit": "Estimate Cost & Profit",
        "fill_fields": "Please fill all fields correctly.",

        "chat_export": "Chat Export",
        "export_pdf": "Export chat as PDF",
        "download_pdf": "Download PDF",
        "saved": "Saved",

        "user": "User",
        "profile_menu": "Profile menu",
        "settings": "Settings",
        "logout": "Logout",

        "agent_chat": "Agent Chat",
        "chat_placeholder": (
            "Ask about farming, costs, irrigation, market, disease..."
        ),
        "agent_thinking": "Agent is thinking...",

        "shop_title": "Fertilizer Shop",
        "doctor_title": "Agricultural Doctors",
        "requirement": "Requirement",
        "search": "Search",
        "search_failed": "Search failed",
        "found": "Found",
        "show_nearby": "Show nearby options",
        "finding_options": "Finding the best options for you...",
        "listing_options": "Finding nearby options...",

        "contact_title": "Contact",
        "team": "AI Farm Agent Team",
        "email": "Email",
        "services": "Services",

        "generated_reports": "Generated Reports",
        "no_reports": "No reports yet. Run analyses from the left panel.",

        "cost_report": "Cost and Profit Estimation Report",
        "market_price": "Local Market Price",
        "price_trend": "Price Trend and Best Months",
        "best_months": "Best Months to Sell",
        "cost_revenue": "Cost and Revenue Breakdown",
        "production_cost": "Total Production Cost",
        "expected_revenue": "Expected Revenue",
        "profit_loss": "Profit or Loss",
        "travel_costs": "Travel Costs",
        "recommendation": "Recommendation",

        "unknown": "Unknown",
        "healthy": "Healthy",

        "api_missing": "API configuration is missing.",
        "api_busy": "Service is busy. Please try again shortly.",
        "queue_remaining": "Tasks remaining in queue",
    },

    "Hindi": {
        "home": "होम",
        "chat": "चैट",
        "shops": "दुकान",
        "doctors": "डॉक्टर्स",
        "contact": "संपर्क",

        "login": "लॉगिन",
        "username": "यूज़रनेम",
        "password": "पासवर्ड",
        "continue": "जारी रखें",
        "login_success": "लॉगिन सफल रहा",
        "account_created": "अकाउंट बनाया गया",

        "upload": "पत्ते की तस्वीर अपलोड करें",
        "analyze": "विश्लेषण करें",

        "agent_control": "एजेंट कंट्रोल पैनल",
        "select_language": "भाषा चुनें",

        "farm_location": "खेत का स्थान",

        "agent_status": "एजेंट की स्थिति",
        "idle": "निष्क्रिय",

        "running_pipeline": "पूर्ण कृषि इंटेलिजेंस पाइपलाइन चल रही है...",
        "getting_location": "स्थान की जानकारी प्राप्त की जा रही है...",
        "fetching_soil": "मिट्टी की जानकारी प्राप्त की जा रही है...",
        "fetching_water": "पानी और मौसम की जानकारी प्राप्त की जा रही है...",
        "analyzing_image": "तस्वीर का विश्लेषण किया जा रहा है...",
        "thinking": "सोच रहा है...",

        "analysis_complete": "विश्लेषण पूरा हुआ!",
        "full_analysis": "पूर्ण विश्लेषण रिपोर्ट",

        "crop_identified": "पहचानी गई फसल",
        "disease_status": "रोग की स्थिति",
        "condition_assessment": "स्थिति का आकलन",
        "actionable_prescription": "कार्रवाई योग्य समाधान",

        "soil_moisture": "मिट्टी और नमी की जानकारी",
        "water_weather": "पानी और मौसम का पूर्वानुमान",

        "risk_urgency": "जोखिम और प्राथमिकता",
        "risk_level": "जोखिम स्तर",

        "fertilizer_recommendations": "उर्वरक सिफारिशें",

        "no_description": "कोई विवरण उपलब्ध नहीं है।",
        "no_solution": "कोई समाधान उपलब्ध नहीं है।",
        "no_soil": "मिट्टी की जानकारी उपलब्ध नहीं है।",
        "no_water": "पानी का पूर्वानुमान उपलब्ध नहीं है।",
        "no_fertilizer": "उर्वरक सिफारिश उपलब्ध नहीं है।",

        "quick_actions": "त्वरित एजेंट कार्य",
        "select_analysis": "विश्लेषण चुनें",
        "run_analysis": "विश्लेषण चलाएं",
        "do_all_analysis": "सभी विश्लेषण चलाएं",
        "run_core_layers": "सभी मुख्य लेयर चलाएं",
        "queued": "कतार में जोड़ा गया",
        "all_queued": "सभी विश्लेषण कतार में हैं!",
        "layers_queued": "सभी लेयर विश्लेषण कतार में हैं।",

        "cost_estimation": "लागत अनुमान",
        "location_city": "स्थान (शहर/क्षेत्र)",
        "crop_name": "फसल का नाम",
        "total_acres": "कुल एकड़",
        "total_invested": "कुल निवेश",
        "estimate_profit": "लागत और लाभ का अनुमान",
        "fill_fields": "कृपया सभी जानकारी सही भरें।",

        "chat_export": "चैट एक्सपोर्ट",
        "export_pdf": "चैट को PDF में एक्सपोर्ट करें",
        "download_pdf": "PDF डाउनलोड करें",
        "saved": "सेव किया गया",

        "user": "उपयोगकर्ता",
        "profile_menu": "प्रोफाइल मेनू",
        "settings": "सेटिंग्स",
        "logout": "लॉगआउट",

        "agent_chat": "एजेंट चैट",
        "chat_placeholder": (
            "खेती, लागत, सिंचाई, बाजार या रोग के बारे में पूछें..."
        ),
        "agent_thinking": "एजेंट सोच रहा है...",

        "shop_title": "उर्वरक दुकान",
        "doctor_title": "कृषि डॉक्टर",
        "requirement": "आवश्यकता",
        "search": "खोजें",
        "search_failed": "खोज असफल",
        "found": "मिल गया",
        "show_nearby": "नजदीकी विकल्प दिखाएं",
        "finding_options": "आपके लिए सर्वोत्तम विकल्प खोजे जा रहे हैं...",
        "listing_options": "नजदीकी विकल्प खोजे जा रहे हैं...",

        "contact_title": "संपर्क",
        "team": "AI फार्म एजेंट टीम",
        "email": "ईमेल",
        "services": "सेवाएं",

        "generated_reports": "तैयार रिपोर्टें",
        "no_reports": "अभी कोई रिपोर्ट नहीं है।",

        "cost_report": "लागत और लाभ अनुमान रिपोर्ट",
        "market_price": "स्थानीय बाजार मूल्य",
        "price_trend": "मूल्य प्रवृत्ति और सर्वोत्तम महीने",
        "best_months": "बेचने के सर्वोत्तम महीने",
        "cost_revenue": "लागत और आय विवरण",
        "production_cost": "कुल उत्पादन लागत",
        "expected_revenue": "अपेक्षित आय",
        "profit_loss": "लाभ या हानि",
        "travel_costs": "यात्रा लागत",
        "recommendation": "सिफारिश",

        "unknown": "अज्ञात",
        "healthy": "स्वस्थ",

        "api_missing": "API कॉन्फ़िगरेशन उपलब्ध नहीं है।",
        "api_busy": "सेवा अभी व्यस्त है। कृपया थोड़ी देर बाद पुनः प्रयास करें।",
        "queue_remaining": "कतार में शेष कार्य",
    },

    "Marathi": {
        "home": "मुख्यपृष्ठ",
        "chat": "चॅट",
        "shops": "दुकान",
        "doctors": "डॉक्टर्स",
        "contact": "संपर्क",

        "login": "लॉगिन",
        "username": "वापरकर्तानाव",
        "password": "पासवर्ड",
        "continue": "पुढे जा",
        "login_success": "लॉगिन यशस्वी झाले",
        "account_created": "खाते तयार केले",

        "upload": "पानाचा फोटो अपलोड करा",
        "analyze": "विश्लेषण करा",

        "agent_control": "एजंट नियंत्रण पॅनेल",
        "select_language": "भाषा निवडा",

        "farm_location": "शेतीचे स्थान",

        "agent_status": "एजंटची स्थिती",
        "idle": "निष्क्रिय",

        "running_pipeline": "पूर्ण कृषी इंटेलिजेंस पाइपलाइन सुरू आहे...",
        "getting_location": "स्थानाची माहिती घेत आहे...",
        "fetching_soil": "मातीची माहिती घेत आहे...",
        "fetching_water": "पाणी आणि हवामानाची माहिती घेत आहे...",
        "analyzing_image": "प्रतिमेचे विश्लेषण सुरू आहे...",
        "thinking": "विचार करत आहे...",

        "analysis_complete": "विश्लेषण पूर्ण झाले!",
        "full_analysis": "संपूर्ण विश्लेषण अहवाल",

        "crop_identified": "ओळखलेले पीक",
        "disease_status": "रोगाची स्थिती",
        "condition_assessment": "स्थितीचे मूल्यांकन",
        "actionable_prescription": "कृतीयोग्य उपाय",

        "soil_moisture": "माती आणि ओलाव्याची माहिती",
        "water_weather": "पाणी आणि हवामान अंदाज",

        "risk_urgency": "धोका आणि तातडी",
        "risk_level": "धोका स्तर",

        "fertilizer_recommendations": "खतांच्या शिफारसी",

        "no_description": "कोणतेही वर्णन उपलब्ध नाही.",
        "no_solution": "कोणताही उपाय उपलब्ध नाही.",
        "no_soil": "मातीची माहिती उपलब्ध नाही.",
        "no_water": "पाण्याचा अंदाज उपलब्ध नाही.",
        "no_fertilizer": "खतांची शिफारस उपलब्ध नाही.",

        "quick_actions": "त्वरित एजंट क्रिया",
        "select_analysis": "विश्लेषण निवडा",
        "run_analysis": "विश्लेषण सुरू करा",
        "do_all_analysis": "सर्व विश्लेषण करा",
        "run_core_layers": "सर्व मुख्य स्तर सुरू करा",
        "queued": "रांगेत जोडले",
        "all_queued": "सर्व विश्लेषणे रांगेत आहेत!",
        "layers_queued": "सर्व स्तरांचे विश्लेषण रांगेत आहे.",

        "cost_estimation": "खर्चाचा अंदाज",
        "location_city": "स्थान (शहर/प्रदेश)",
        "crop_name": "पिकाचे नाव",
        "total_acres": "एकूण एकर",
        "total_invested": "एकूण गुंतवणूक",
        "estimate_profit": "खर्च आणि नफा अंदाज",
        "fill_fields": "कृपया सर्व माहिती योग्य भरा.",

        "chat_export": "चॅट एक्सपोर्ट",
        "export_pdf": "चॅट PDF म्हणून एक्सपोर्ट करा",
        "download_pdf": "PDF डाउनलोड करा",
        "saved": "जतन केले",

        "user": "वापरकर्ता",
        "profile_menu": "प्रोफाइल मेनू",
        "settings": "सेटिंग्ज",
        "logout": "लॉगआउट",

        "agent_chat": "एजंट चॅट",
        "chat_placeholder": (
            "शेती, खर्च, सिंचन, बाजार किंवा रोगाबद्दल विचारा..."
        ),
        "agent_thinking": "एजंट विचार करत आहे...",

        "shop_title": "खत दुकान",
        "doctor_title": "कृषी डॉक्टर",
        "requirement": "आवश्यकता",
        "search": "शोधा",
        "search_failed": "शोध अयशस्वी",
        "found": "सापडले",
        "show_nearby": "जवळील पर्याय दाखवा",
        "finding_options": "तुमच्यासाठी सर्वोत्तम पर्याय शोधत आहे...",
        "listing_options": "जवळील पर्याय शोधत आहे...",

        "contact_title": "संपर्क",
        "team": "AI फार्म एजंट टीम",
        "email": "ईमेल",
        "services": "सेवा",

        "generated_reports": "तयार केलेले अहवाल",
        "no_reports": "अजून कोणतेही अहवाल नाहीत.",

        "cost_report": "खर्च आणि नफा अंदाज अहवाल",
        "market_price": "स्थानिक बाजार भाव",
        "price_trend": "भावाचा कल आणि सर्वोत्तम महिने",
        "best_months": "विक्रीसाठी सर्वोत्तम महिने",
        "cost_revenue": "खर्च आणि उत्पन्न तपशील",
        "production_cost": "एकूण उत्पादन खर्च",
        "expected_revenue": "अपेक्षित उत्पन्न",
        "profit_loss": "नफा किंवा तोटा",
        "travel_costs": "प्रवास खर्च",
        "recommendation": "शिफारस",

        "unknown": "अज्ञात",
        "healthy": "निरोगी",

        "api_missing": "API कॉन्फिगरेशन उपलब्ध नाही.",
        "api_busy": "सेवा सध्या व्यस्त आहे. कृपया थोड्या वेळाने पुन्हा प्रयत्न करा.",
        "queue_remaining": "रांगेतील उर्वरित कामे",
    },
}


# ============================================================
# FONT SETTINGS
# ============================================================

FONT_MAP = {
    "English": "Arial, sans-serif",
    "Hindi": "'Nirmala UI', 'Mangal', sans-serif",
    "Marathi": "'Noto Sans Devanagari', 'Mangal', sans-serif",
}


# ============================================================
# STABLE NAVIGATION
# ============================================================

NAV_ITEMS = [
    ("home", "home"),
    ("chat", "chat"),
    ("shops", "shops"),
    ("doctors", "doctors"),
    ("contact", "contact"),
]


# ============================================================
# AGENT ACTIONS
# ============================================================

ACTION_MAP = {
    "Soil moisture modeling":
        "Analyze soil moisture and provide actionable irrigation guidance.",

    "Water requirement prediction":
        "Predict farm water requirements for the next 14 days.",

    "AI-driven irrigation schedule":
        "Create an irrigation schedule with recommended time windows.",

    "Drought early warning":
        "Generate drought early-warning indicators and preventive actions.",

    "Water waste optimization %":
        "Estimate water waste and optimization opportunities.",

    "NPK prediction":
        "Estimate nitrogen, phosphorus and potassium requirements.",

    "pH imbalance detection":
        "Detect possible soil pH imbalance and recommend treatment.",

    "Nutrient deficiency fusion":
        "Identify likely nutrient deficiencies using available farm context.",

    "Fertilizer recommendation":
        "Generate practical fertilizer recommendations for the farm.",

    "Long-term soil health score":
        "Estimate long-term soil health and create a yearly improvement plan.",

    "Insect classification":
        "Identify likely insects and estimate agricultural risk.",

    "Pest density estimation":
        "Estimate pest density and intervention thresholds.",

    "Swarm detection":
        "Detect possible swarm risk and generate an alert plan.",

    "Migration pattern prediction":
        "Predict possible pest migration patterns.",

    "Smart pesticide timing":
        "Recommend the safest and most effective pesticide timing.",

    "Satellite imagery integration":
        "Create a satellite imagery integration strategy.",

    "Growth stage tracking":
        "Track crop growth stages and next milestones.",

    "Production estimate per acre":
        "Estimate production per acre with confidence ranges.",

    "Profit forecast":
        "Generate a profit forecast using yield, costs and prices.",

    "Market price integration":
        "Analyze market trends and suggest selling timing.",

    "Camera to recommendation pipeline":
        "Design a camera-to-analysis-to-recommendation pipeline.",

    "Irrigation valve control":
        "Generate irrigation valve control logic and safety failsafes.",

    "Sprayer control":
        "Generate a smart sprayer control strategy.",

    "Drone-based spraying":
        "Plan a drone spraying route and timing.",

    "Automated farm reporting":
        "Create an automated farm reporting system.",

    "Multi-modal fusion model":
        "Design a fusion model using vision, weather, soil and time data.",

    "Disease risk 7-30 days":
        "Estimate disease risk for the next 7 to 30 days.",

    "Frost risk alerts":
        "Predict frost risk and preventive actions.",

    "Heat stress prediction":
        "Predict heat-stress windows and protection actions.",

    "Crop growth stage mapping":
        "Generate crop growth stage mapping.",

    "Price prediction AI":
        "Estimate crop production cost and expected market profit.",

    "Full Agent Pipeline":
        "Design an end-to-end agricultural AI agent pipeline.",
}


# ============================================================
# TRANSLATION HELPERS
# ============================================================

def tr(key, fallback=None):
    language = st.session_state.get("language", "English")
    language_data = TRANSLATIONS.get(
        language,
        TRANSLATIONS["English"],
    )

    return language_data.get(
        key,
        fallback if fallback is not None else key,
    )


def get_ai_language_instruction():
    language = st.session_state.get("language", "English")

    mapping = {
        "English": "Respond entirely in English.",
        "Hindi": "Respond entirely in Hindi using Devanagari script.",
        "Marathi": "Respond entirely in Marathi using Devanagari script.",
    }

    return mapping.get(
        language,
        mapping["English"],
    )


# ============================================================
# SESSION STATE
# ============================================================

def ensure_session_defaults():
    defaults = {
        "language": "English",
        "logged_in": False,
        "username": "",
        "photo_url": (
            "https://api.dicebear.com/8.x/adventurer/png?seed=Farmer"
        ),
        "agent_status": "Idle",
        "task_queue": [],
        "reports": [],
        "chat_history": [],
        "detection_result": None,
        "menu_choice": "home",
        "location": "",
        "cost_estimation": None,
        "pdf_data": None,
        "pdf_name": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    old_menu_labels = {
        "Home": "home",
        "होम": "home",
        "मुख्यपृष्ठ": "home",

        "Chat": "chat",
        "चैट": "chat",
        "चॅट": "chat",

        "Shop": "shops",
        "दुकान": "shops",

        "Doctors": "doctors",
        "डॉक्टर्स": "doctors",

        "Contact": "contact",
        "संपर्क": "contact",
    }

    st.session_state.menu_choice = old_menu_labels.get(
        st.session_state.menu_choice,
        st.session_state.menu_choice,
    )

    valid_pages = [
        "home",
        "chat",
        "shops",
        "doctors",
        "contact",
    ]

    if st.session_state.menu_choice not in valid_pages:
        st.session_state.menu_choice = "home"


# ============================================================
# FONT
# ============================================================

def apply_local_font():
    language = st.session_state.get("language", "English")
    font_family = FONT_MAP.get(
        language,
        FONT_MAP["English"],
    )

    st.markdown(
        f"""
        <style>
            html, body, [class*="css"], .stApp {{
                font-family: {font_family};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# API RESPONSE HELPERS
# ============================================================

def extract_message_content(data):
    try:
        content = data["choices"][0]["message"]["content"]

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))

            return "\n".join(parts).strip()

        return content

    except (KeyError, IndexError, TypeError):
        return None


def clean_json_text(text):
    if not isinstance(text, str):
        return text

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def safe_json_loads(text):
    text = clean_json_text(text)

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse valid JSON")


def is_error_response(text):
    if not isinstance(text, str):
        return False

    error_prefixes = (
        "API key is missing",
        "API configuration is missing",
        "API URL is missing",
        "Model is missing",
        "HTTP Error",
        "API Error",
        "Network Error",
        "Request timed out",
        "Service is currently busy",
    )

    return text.startswith(error_prefixes)


# ============================================================
# MODEL REQUEST WITH SMART RETRY
# ============================================================

def call_model(messages, model=None, temperature=0.2):
    if not API_KEY:
        return (
            "API key is missing. "
            "Add API_KEY to .streamlit/secrets.toml."
        )

    if not API_URL:
        return (
            "API URL is missing. "
            "Add API_URL to .streamlit/secrets.toml."
        )

    selected_model = model or MODEL

    if not selected_model:
        return (
            "Model is missing. "
            "Add MODEL to .streamlit/secrets.toml."
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
    }

    retryable_status_codes = {
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    }

    last_error = "Unknown error"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    return (
                        "API Error: invalid JSON response: "
                        f"{response.text[:500]}"
                    )

                content = extract_message_content(data)

                if content is not None:
                    return content

                if isinstance(data, dict) and "error" in data:
                    error = data["error"]

                    if isinstance(error, dict):
                        return (
                            "API Error: "
                            f"{error.get('message', str(error))}"
                        )

                    return f"API Error: {error}"

                return (
                    "API Error: unexpected response format: "
                    f"{str(data)[:800]}"
                )

            last_error = (
                f"HTTP Error {response.status_code}: "
                f"{response.text[:800]}"
            )

            if response.status_code not in retryable_status_codes:
                return last_error

            if attempt < MAX_RETRIES - 1:
                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 0
                else:
                    base_delay = min(
                        2 ** attempt,
                        20,
                    )

                    delay = (
                        base_delay
                        + random.uniform(0.5, 2.0)
                    )

                time.sleep(delay)
                continue

        except requests.exceptions.Timeout:
            last_error = "Request timed out."

        except requests.exceptions.RequestException as error:
            last_error = f"Network Error: {str(error)}"

        if attempt < MAX_RETRIES - 1:
            delay = (
                min(2 ** attempt, 20)
                + random.uniform(0.5, 1.5)
            )

            time.sleep(delay)

    return (
        "Service is currently busy after multiple retries. "
        f"Last error: {last_error}"
    )


# ============================================================
# VISION / PLANT ANALYSIS
# ============================================================

def analyze_plant_image(image_bytes, species_info):
    if not API_KEY:
        return {
            "error": "API key is missing."
        }

    base64_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    prompt = f"""
Analyze this plant image and metadata.

Metadata:
{json.dumps(species_info, ensure_ascii=False)}

{get_ai_language_instruction()}

Identify:

1. Specific crop or plant name.
2. Most likely disease or health issue.
3. Soil health observations.
4. Water and irrigation guidance.
5. Overall risk score.

Important:
- Do not invent precise local measurements unless supplied.
- Clearly treat uncertain conclusions as estimates.
- Use only information reasonably inferable from the image and metadata.
- All JSON values must be written in the selected language.
- Return ONLY valid JSON.

Required JSON:
{{
    "crop_name": "Crop name",
    "disease_name": "Disease name or Healthy",
    "description": "Condition description",
    "solution": "Step-by-step solution",
    "fertilizers": "Recommended nutrients or fertilizers",
    "soil_insights": "Soil health observations",
    "water_forecast": "Water and irrigation guidance",
    "risk_score": "Low, Medium or High"
}}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/jpeg;base64,"
                            f"{base64_image}"
                        )
                    },
                },
            ],
        }
    ]

    output = call_model(
        messages,
        model=VISION_MODEL,
    )

    if is_error_response(output):
        return {
            "error": output
        }

    try:
        return safe_json_loads(output)

    except Exception:
        return {
            "error": (
                "Model returned invalid analysis JSON."
            ),
            "raw_response": output[:2000],
        }


# ============================================================
# TASK QUEUE
# ============================================================

def queue_task(task_name, prompt, model=None):
    existing_tasks = {
        item["task"]
        for item in st.session_state.task_queue
    }

    if task_name in existing_tasks:
        return False

    st.session_state.task_queue.append(
        {
            "task": task_name,
            "prompt": prompt,
            "model": model or REASONING_MODEL,
        }
    )

    return True


def run_background_tasks():
    if not st.session_state.task_queue:
        return

    tasks_to_run = min(
        MAX_TASKS_PER_RUN,
        len(st.session_state.task_queue),
    )

    for _ in range(tasks_to_run):
        task = st.session_state.task_queue.pop(0)

        st.session_state.agent_status = (
            f"Running: {task['task']}"
        )

        system_prompt = f"""
You are an agricultural AI agent.

Provide a structured operational report with:
- assumptions
- risks
- recommended actions
- measurable outcomes

{get_ai_language_instruction()}
"""

        report = call_model(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": task["prompt"],
                },
            ],
            model=task["model"],
        )

        st.session_state.reports.insert(
            0,
            {
                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "title": task["task"],
                "content": report,
            },
        )

        time.sleep(1.5)

    if st.session_state.task_queue:
        st.session_state.agent_status = (
            f"Processed {tasks_to_run} task(s). "
            f"{len(st.session_state.task_queue)} remaining."
        )
    else:
        st.session_state.agent_status = (
            "All tasks completed"
        )


# ============================================================
# PDF EXPORT
# ============================================================

def export_chat_to_pdf():
    os.makedirs(
        EXPORT_DIR,
        exist_ok=True,
    )

    filename = (
        "chat_export_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    path = os.path.join(
        EXPORT_DIR,
        filename,
    )

    lines = [
        "AI Agent Chat Export",
        "",
    ]

    for message in st.session_state.chat_history:
        lines.append(
            f"[{message['time']}] "
            f"{message['role'].upper()}: "
            f"{message['text']}"
        )

    page_lines = 35

    with PdfPages(path) as pdf:
        total = max(len(lines), 1)

        for i in range(
            0,
            total,
            page_lines,
        ):
            fig = plt.figure(
                figsize=(8.27, 11.69)
            )

            chunk = "\n".join(
                lines[i:i + page_lines]
            )

            if not chunk:
                chunk = "No chat messages to export."

            fig.text(
                0.05,
                0.95,
                chunk,
                va="top",
                fontsize=9,
                family="sans-serif",
                wrap=True,
            )

            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    with open(path, "rb") as file:
        pdf_data = file.read()

    return filename, pdf_data


# ============================================================
# LOGIN
# ============================================================

def login_block():
    if not os.path.exists(USER_DB):
        with open(
            USER_DB,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump({}, file)

    try:
        with open(
            USER_DB,
            "r",
            encoding="utf-8",
        ) as file:
            users = json.load(file)

    except (
        json.JSONDecodeError,
        FileNotFoundError,
    ):
        users = {}

    if st.session_state.logged_in:
        return

    st.title(tr("login"))

    username = st.text_input(
        tr("username"),
        key="login_username",
    )

    password = st.text_input(
        tr("password"),
        type="password",
        key="login_password",
    )

    if st.button(
        tr("continue"),
        use_container_width=True,
    ):
        username = username.strip()

        if not username or not password:
            st.error(
                "Username and password are required."
            )
            st.stop()

        if username in users:
            if users[username] != password:
                st.error("Invalid password.")
                st.stop()

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success(tr("login_success"))
            st.rerun()

        else:
            users[username] = password

            with open(
                USER_DB,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    users,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success(tr("account_created"))
            st.rerun()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

def sidebar_controls():
    with st.sidebar:
        st.title(tr("agent_control"))

        languages = list(
            TRANSLATIONS.keys()
        )

        current_language = (
            st.session_state.language
        )

        language_index = languages.index(
            current_language
        )

        selected_language = st.selectbox(
            tr("select_language"),
            languages,
            index=language_index,
            key="language_selector",
        )

        if selected_language != current_language:
            st.session_state.language = selected_language
            st.rerun()

        st.markdown("---")

        # ---------------- COST ----------------

        st.subheader(
            tr("cost_estimation")
        )

        est_location = st.text_input(
            tr("location_city"),
            key="cost_location",
        )

        est_crop = st.text_input(
            tr("crop_name"),
            key="cost_crop",
        )

        est_acres = st.number_input(
            tr("total_acres"),
            min_value=0.0,
            step=0.1,
            key="cost_acres",
        )

        est_invested = st.number_input(
            tr("total_invested"),
            min_value=0.0,
            step=100.0,
            key="cost_invested",
        )

        if st.button(
            tr("estimate_profit"),
            use_container_width=True,
        ):
            if (
                not est_location
                or not est_crop
                or est_acres <= 0
            ):
                st.error(
                    tr("fill_fields")
                )

            else:
                cost_prompt = f"""
Location: {est_location}
Crop: {est_crop}
Acres: {est_acres}
Investment: {est_invested}

{get_ai_language_instruction()}

Provide a cost, revenue and profit analysis.

Use estimates when live market data is unavailable.

Return ONLY valid JSON:

{{
    "market_price": "...",
    "price_trend": "...",
    "best_months": ["..."],
    "total_cost": "...",
    "expected_revenue": "...",
    "profit_or_loss": "...",
    "travel_costs": "...",
    "recommendation": "..."
}}
"""

                with st.spinner(
                    tr("thinking")
                ):
                    st.session_state.cost_estimation = (
                        call_model(
                            [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an agricultural "
                                        "economic analyst. "
                                        + get_ai_language_instruction()
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": cost_prompt,
                                },
                            ],
                            model=REASONING_MODEL,
                        )
                    )

        st.markdown("---")

        # ---------------- QUICK ACTIONS ----------------

        st.subheader(
            tr("quick_actions")
        )

        selected_action = st.selectbox(
            tr("select_analysis"),
            list(ACTION_MAP.keys()),
            key="agent_action",
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                tr("run_analysis"),
                use_container_width=True,
            ):
                added = queue_task(
                    selected_action,
                    ACTION_MAP[selected_action],
                )

                if added:
                    st.success(
                        f"{tr('queued')}: "
                        f"{selected_action}"
                    )
                else:
                    st.info(
                        "This task is already queued."
                    )

        with col2:
            if st.button(
                tr("do_all_analysis"),
                use_container_width=True,
            ):
                for action, prompt in ACTION_MAP.items():
                    queue_task(
                        action,
                        prompt,
                    )

                st.success(
                    tr("all_queued")
                )

        if st.button(
            tr("run_core_layers"),
            use_container_width=True,
        ):
            layers = [
                "Vision Layer",
                "Climate Layer",
                "Soil Layer",
                "Water Layer",
                "Market Layer",
                "Execution Layer",
            ]

            for layer in layers:
                queue_task(
                    layer,
                    (
                        f"Generate an operational report "
                        f"for {layer} with metrics and actions."
                    ),
                )

            st.success(
                tr("layers_queued")
            )

        if st.session_state.task_queue:
            st.caption(
                f"{tr('queue_remaining')}: "
                f"{len(st.session_state.task_queue)}"
            )

        st.markdown("---")

        # ---------------- PDF ----------------

        st.subheader(
            tr("chat_export")
        )

        if st.button(
            tr("export_pdf"),
            use_container_width=True,
        ):
            filename, pdf_data = (
                export_chat_to_pdf()
            )

            st.session_state.pdf_name = filename
            st.session_state.pdf_data = pdf_data

            st.success(
                f"{tr('saved')}: {filename}"
            )

        if (
            st.session_state.pdf_data
            and st.session_state.pdf_name
        ):
            st.download_button(
                tr("download_pdf"),
                data=st.session_state.pdf_data,
                file_name=st.session_state.pdf_name,
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("---")

        # ---------------- USER ----------------

        st.subheader(
            tr("user")
        )

        try:
            st.image(
                st.session_state.photo_url,
                width=70,
            )
        except Exception:
            pass

        st.write(
            st.session_state.username
        )

        with st.expander(
            tr("profile_menu")
        ):
            st.button(
                tr("settings"),
                disabled=True,
            )

            if st.button(
                tr("logout"),
                use_container_width=True,
            ):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.menu_choice = "home"
                st.rerun()


# ============================================================
# HOME PAGE
# ============================================================

def home_page():
    st.title(APP_TITLE)

    st.session_state.location = st.text_input(
        tr("farm_location"),
        value=st.session_state.location,
        key="farm_location_input",
    )

    uploaded_image = st.file_uploader(
        tr("upload"),
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="leaf_upload",
    )

    if uploaded_image:
        image = Image.open(
            uploaded_image
        )

        st.image(
            image,
            caption=tr("upload"),
            use_container_width=True,
        )

        if st.button(
            tr("analyze"),
            key="analyze_leaf_button",
            use_container_width=True,
        ):
            try:
                buffer = io.BytesIO()

                rgb_image = image.convert(
                    "RGB"
                )

                rgb_image.save(
                    buffer,
                    format="JPEG",
                    quality=90,
                )

                image_bytes = buffer.getvalue()

                status = st.status(
                    tr("running_pipeline"),
                    expanded=True,
                )

                status.write(
                    tr("getting_location")
                )

                status.write(
                    tr("fetching_soil")
                )

                status.write(
                    tr("fetching_water")
                )

                status.write(
                    tr("analyzing_image")
                )

                species_info = {
                    "location":
                        st.session_state.location
                }

                result = analyze_plant_image(
                    image_bytes,
                    species_info,
                )

                status.write(
                    tr("thinking")
                )

                if "error" in result:
                    status.update(
                        label=result["error"],
                        state="error",
                    )

                    st.error(
                        result["error"]
                    )

                else:
                    st.session_state.detection_result = (
                        result
                    )

                    status.update(
                        label=tr("analysis_complete"),
                        state="complete",
                    )

                    st.success(
                        tr("analysis_complete")
                    )

            except Exception as error:
                st.error(
                    f"Analysis failed: {str(error)}"
                )

    # ---------------- RESULT ----------------

    result = (
        st.session_state.detection_result
    )

    if (
        result
        and isinstance(result, dict)
        and "error" not in result
    ):
        st.markdown("---")

        st.markdown(
            f"## {tr('full_analysis')}"
        )

        col_crop, col_disease = st.columns(2)

        with col_crop:
            st.markdown(
                f"### {tr('crop_identified')}"
            )

            st.write(
                result.get(
                    "crop_name",
                    tr("unknown"),
                )
            )

        with col_disease:
            st.markdown(
                f"### {tr('disease_status')}"
            )

            st.write(
                result.get(
                    "disease_name",
                    tr("healthy"),
                )
            )

        st.markdown(
            f"## {tr('condition_assessment')}"
        )

        st.write(
            result.get(
                "description",
                tr("no_description"),
            )
        )

        st.markdown(
            f"## {tr('actionable_prescription')}"
        )

        st.write(
            result.get(
                "solution",
                tr("no_solution"),
            )
        )

        st.markdown(
            f"## {tr('soil_moisture')}"
        )

        st.write(
            result.get(
                "soil_insights",
                tr("no_soil"),
            )
        )

        st.markdown(
            f"## {tr('water_weather')}"
        )

        st.write(
            result.get(
                "water_forecast",
                tr("no_water"),
            )
        )

        st.markdown(
            f"## {tr('risk_urgency')}"
        )

        risk = str(
            result.get(
                "risk_score",
                "Low",
            )
        )

        risk_lower = risk.lower()

        if (
            "high" in risk_lower
            or "उच्च" in risk_lower
            or "जास्त" in risk_lower
        ):
            st.error(
                f"**{tr('risk_level')}:** {risk}"
            )

        elif (
            "medium" in risk_lower
            or "मध्यम" in risk_lower
        ):
            st.warning(
                f"**{tr('risk_level')}:** {risk}"
            )

        else:
            st.success(
                f"**{tr('risk_level')}:** {risk}"
            )

        st.markdown(
            f"## {tr('fertilizer_recommendations')}"
        )

        st.write(
            result.get(
                "fertilizers",
                tr("no_fertilizer"),
            )
        )


# ============================================================
# CHAT PAGE
# ============================================================

def chat_page():
    st.title(
        tr("agent_chat")
    )

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"]
        ):
            st.caption(
                message["time"]
            )

            st.write(
                message["text"]
            )

    query = st.chat_input(
        tr("chat_placeholder")
    )

    if query:
        st.session_state.chat_history.append(
            {
                "time": datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "role": "user",
                "text": query,
            }
        )

        recent_history = (
            st.session_state.chat_history[-10:]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a practical agricultural AI agent. "
                    "Be useful, specific and honest about uncertainty. "
                    + get_ai_language_instruction()
                ),
            }
        ]

        for item in recent_history:
            messages.append(
                {
                    "role": item["role"],
                    "content": item["text"],
                }
            )

        with st.spinner(
            tr("agent_thinking")
        ):
            answer = call_model(
                messages,
                model=REASONING_MODEL,
            )

        st.session_state.chat_history.append(
            {
                "time": datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "role": "assistant",
                "text": answer,
            }
        )

        st.rerun()


# ============================================================
# SHOP / DOCTORS PAGE
# ============================================================

def shop_or_doctors_page(page_type):
    if page_type == "shop":
        title = tr("shop_title")
        actor = (
            "agricultural fertilizer supplier"
        )
    else:
        title = tr("doctor_title")
        actor = (
            "agricultural expert"
        )

    st.title(title)

    col1, col2 = st.columns(2)

    with col1:
        crop = st.text_input(
            tr("crop_name"),
            key=f"{page_type}_crop",
        )

    with col2:
        requirement = st.text_input(
            tr("requirement"),
            key=f"{page_type}_requirement",
        )

    location = (
        st.session_state.location
        or "the specified region"
    )

    col_search, col_nearby = st.columns(2)

    with col_search:
        if st.button(
            tr("search"),
            key=f"{page_type}_search",
            use_container_width=True,
        ):
            with st.spinner(
                tr("finding_options")
            ):
                prompt = f"""
Recommend useful {actor} options.

Crop: {crop}
Requirement: {requirement}
Location: {location}

{get_ai_language_instruction()}

Do not invent phone numbers or exact addresses.
If live local data is unavailable, clearly label the answer as general recommendations.
"""

                response = call_model(
                    [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=REASONING_MODEL,
                )

                st.write(response)

    with col_nearby:
        if st.button(
            tr("show_nearby"),
            key=f"{page_type}_nearby",
            use_container_width=True,
        ):
            with st.spinner(
                tr("listing_options")
            ):
                prompt = f"""
List general major categories and ways to find a nearby {actor}.

Crop: {crop}
Location: {location}

{get_ai_language_instruction()}

Do not fabricate real businesses, contacts or addresses.
Explain how the user should verify local availability.
"""

                response = call_model(
                    [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=REASONING_MODEL,
                )

                st.write(response)


# ============================================================
# CONTACT PAGE
# ============================================================

def contact_page():
    st.title(
        tr("contact_title")
    )

    st.markdown(
        f"""
### {tr('team')}

**{tr('email')}:** support@example.com

**{tr('services')}:**

Vision • Climate • Soil • Water • Market • Execution
"""
    )


# ============================================================
# REPORTS
# ============================================================

def show_reports_panel():
    st.markdown(
        f"## {tr('generated_reports')}"
    )

    if not st.session_state.reports:
        st.info(
            tr("no_reports")
        )
        return

    for report in st.session_state.reports[:12]:
        with st.expander(
            f"{report['time']} — {report['title']}"
        ):
            st.write(
                report["content"]
            )


# ============================================================
# COST REPORT
# ============================================================

def show_cost_report():
    estimation = (
        st.session_state.cost_estimation
    )

    if not estimation:
        return

    st.markdown("---")

    st.markdown(
        f"## {tr('cost_report')}"
    )

    if is_error_response(estimation):
        st.error(estimation)
        return

    try:
        data = safe_json_loads(
            estimation
        )

    except Exception:
        st.warning(
            "Could not parse structured estimation."
        )

        st.write(
            estimation
        )
        return

    st.markdown(
        f"### {tr('market_price')}"
    )

    st.write(
        data.get(
            "market_price",
            "N/A",
        )
    )

    st.markdown(
        f"### {tr('price_trend')}"
    )

    st.write(
        data.get(
            "price_trend",
            "",
        )
    )

    best_months = data.get(
        "best_months",
        [],
    )

    if isinstance(best_months, list):
        best_months_text = ", ".join(
            map(str, best_months)
        )
    else:
        best_months_text = str(
            best_months
        )

    st.write(
        f"**{tr('best_months')}:** "
        f"{best_months_text}"
    )

    st.markdown(
        f"### {tr('cost_revenue')}"
    )

    st.write(
        f"**{tr('production_cost')}:** "
        f"{data.get('total_cost', 'N/A')}"
    )

    st.write(
        f"**{tr('expected_revenue')}:** "
        f"{data.get('expected_revenue', 'N/A')}"
    )

    st.write(
        f"**{tr('profit_loss')}:** "
        f"{data.get('profit_or_loss', 'N/A')}"
    )

    st.markdown(
        f"### {tr('travel_costs')}"
    )

    st.write(
        data.get(
            "travel_costs",
            "N/A",
        )
    )

    st.markdown(
        f"### {tr('recommendation')}"
    )

    st.info(
        data.get(
            "recommendation",
            "",
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    ensure_session_defaults()
    apply_local_font()

    login_block()
    sidebar_controls()

    # Run only a small batch each rerun.
    # This prevents request explosions and greatly reduces 503 errors.
    run_background_tasks()

    st.markdown(
        f"### {tr('agent_status')}: "
        f"{st.session_state.agent_status}"
    )

    # --------------------------------------------------------
    # NAVIGATION
    # Internal IDs never change.
    # Only displayed labels are translated.
    # --------------------------------------------------------

    columns = st.columns(
        len(NAV_ITEMS)
    )

    for index, (
        page_id,
        translation_key,
    ) in enumerate(NAV_ITEMS):

        with columns[index]:
            if st.button(
                tr(translation_key),
                key=f"nav_{page_id}",
                use_container_width=True,
                type=(
                    "primary"
                    if st.session_state.menu_choice == page_id
                    else "secondary"
                ),
            ):
                st.session_state.menu_choice = (
                    page_id
                )
                st.rerun()

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    menu = (
        st.session_state.menu_choice
    )

    if menu == "home":
        home_page()

    elif menu == "chat":
        chat_page()

    elif menu == "shops":
        shop_or_doctors_page(
            "shop"
        )

    elif menu == "doctors":
        shop_or_doctors_page(
            "doctor"
        )

    elif menu == "contact":
        contact_page()

    else:
        st.session_state.menu_choice = "home"
        st.rerun()

    # --------------------------------------------------------
    # REPORTS
    # --------------------------------------------------------

    show_cost_report()
    show_reports_panel()


if __name__ == "__main__":
    main()
