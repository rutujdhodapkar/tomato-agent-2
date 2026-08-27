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

st.set_page_config(
    page_title="Agri Super Agent",
    page_icon="🌱",
    layout="wide",
)

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
VISION_MODEL = DEFAULT_MODEL
REASONING_MODEL = DEFAULT_MODEL

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

USER_DB = "users.json"
EXPORT_DIR = "exports"


# ============================================================
# LOAD SECRETS SAFELY
# ============================================================

def get_secret(name: str, default: str = "") -> str:
    """
    Reads secrets from:
    1. Streamlit secrets
    2. Environment variables

    Never hardcode production API keys in source code.
    """
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv(name, default).strip()


NVIDIA_API_KEY = get_secret("NVIDIA_API_KEY")
SARVAM_API_KEY = get_secret("SARVAM_API_KEY")


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
        "logout": "Logout",

        "app_title": "🌱 Agricultural Super AI Agent",
        "app_caption": (
            "Image analysis is AI-assisted. "
            "Recommendations should be verified before real-world treatment."
        ),

        "agent_panel": "🌱 Agent Control Panel",
        "agent_status": "Agent Status",

        "select_language": "Select Language",
        "apply_language": "Apply Language",

        "cost_estimation": "Cost Estimation",
        "location_city": "Location (city/region)",
        "crop_name_input": "Crop name",
        "total_acres": "Total acres",
        "total_invested": "Total invested (₹ or $)",
        "estimate_cost": "Estimate Cost & Profit",
        "fill_cost_fields": "Please fill location, crop, and acres.",

        "quick_actions": "Quick Agent Actions",
        "select_analysis": "Select analysis",
        "run_analysis": "Run analysis",
        "do_all_analysis": "Do all analysis",
        "run_core_layers": "Run all core layers",
        "queued": "Queued",
        "all_analyses_queued": "All analyses queued.",
        "core_layers_queued": "Core layers queued.",

        "chat_export": "Chat Export",
        "export_pdf": "Export chat as PDF",
        "download_pdf": "Download PDF",
        "pdf_failed": "PDF export failed",

        "user": "User",

        "farm_location": "Farm location",
        "farm_location_placeholder": "Example: Pune, Maharashtra",
        "upload": "Upload Leaf Image",
        "uploaded_leaf": "Uploaded Leaf",
        "analyze": "Analyze",

        "running_analysis": "Running plant analysis...",
        "processing_image": "Processing uploaded image...",
        "preparing_metadata": "Preparing farm metadata...",
        "sending_model": "Sending image to model...",
        "analysis_complete": "Analysis complete",
        "analysis_failed": "Analysis failed",
        "analysis_complete_success": "Analysis Complete!",

        "could_not_open_image": "Could not open image",
        "image_processing_failed": "Image processing failed",
        "api_key_missing": "API key is missing. Configure Streamlit secrets first.",

        "full_analysis": "Full Analysis Report",
        "crop_identified": "Crop Identified",
        "disease_status": "Disease Status",
        "ai_confidence": "AI Confidence",

        "condition_assessment": "Condition Assessment",
        "actionable_prescription": "Actionable Prescription",
        "soil_moisture": "Soil and Moisture Insights",
        "water_irrigation": "Water and Irrigation Guidance",
        "risk_urgency": "Risk and Urgency",
        "fertilizer_recommendations": "Fertilizer Recommendations",

        "risk_high": "Risk Level: High",
        "risk_medium": "Risk Level: Medium",
        "risk_low": "Risk Level: Low",

        "unknown": "Unknown",
        "no_description": "No description available.",
        "no_solution": "No solution provided.",
        "no_soil_insights": "No soil insights available.",
        "no_irrigation": "No irrigation guidance available.",
        "no_fertilizer": "No fertilizer recommendations available.",

        "chat_title": "💬 Agent Chat",
        "chat_placeholder": (
            "Ask about farming, irrigation, crops, disease, soil, or costs..."
        ),
        "agent_thinking": "Agent is thinking...",

        "shop_title": "🛒 Fertilizer & Agriculture Services",
        "doctors_title": "🩺 Agricultural Experts",

        "search_warning": (
            "This version does not use a verified local-business search API. "
            "Results generated by AI should not be treated as real businesses "
            "or real contact information."
        ),

        "shop_crop_name": "Shop: Crop name",
        "shop_requirement": "Shop: Requirement",
        "doctors_crop_name": "Doctors: Crop name",
        "doctors_requirement": "Doctors: Requirement",

        "generate_shop_search": "Generate Shop Search Criteria",
        "generate_doctors_search": "Generate Doctors Search Criteria",
        "preparing_search": "Preparing search criteria...",

        "contact_title": "Contact",
        "contact_intro": (
            "This application combines plant image analysis, "
            "agricultural reasoning, and farm-planning workflows."
        ),
        "contact_production": (
            "For production deployment, connect verified data sources for:"
        ),

        "weather": "Weather",
        "soil_sensors": "Soil sensors",
        "market_prices": "Market prices",
        "satellite_imagery": "Satellite imagery",
        "local_services": "Local agricultural services",

        "generated_reports": "Generated Reports",
        "no_reports": "No reports yet. Run analyses from the sidebar.",

        "cost_profit_report": "Cost and Profit Estimation Report",
        "could_not_parse_estimation": "Could not parse estimation as JSON.",

        "market_price": "Market Price",
        "price_trend_months": "Price Trend and Best Months",
        "best_months": "Best months to sell",
        "cost_revenue": "Cost and Revenue",
        "total_cost": "Total Cost",
        "expected_revenue": "Expected Revenue",
        "profit_loss": "Profit / Loss",
        "travel_costs": "Travel Costs",
        "recommendation": "Recommendation",

        "login_success": "Login successful.",
        "account_created": "Account created.",
        "invalid_password": "Invalid password.",
        "credentials_required": "Username and password are required.",

        "all_tasks_completed": "All tasks completed",
        "no_chat_messages": "No chat messages to export.",

        "na": "N/A",
        "no_recommendation": "No recommendation available.",

        "core_vision": "Vision Layer",
        "core_climate": "Climate Layer",
        "core_soil": "Soil Layer",
        "core_water": "Water Layer",
        "core_market": "Market Layer",
        "core_execution": "Execution Layer",
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
        "logout": "लॉगआउट",

        "app_title": "🌱 कृषि सुपर एआई एजेंट",
        "app_caption": (
            "छवि विश्लेषण एआई-सहायता प्राप्त है। "
            "वास्तविक उपचार से पहले सुझावों की पुष्टि करें।"
        ),

        "agent_panel": "🌱 एजेंट कंट्रोल पैनल",
        "agent_status": "एजेंट स्थिति",

        "select_language": "भाषा चुनें",
        "apply_language": "भाषा लागू करें",

        "cost_estimation": "लागत अनुमान",
        "location_city": "स्थान (शहर/क्षेत्र)",
        "crop_name_input": "फसल का नाम",
        "total_acres": "कुल एकड़",
        "total_invested": "कुल निवेश (₹ या $)",
        "estimate_cost": "लागत और लाभ का अनुमान लगाएं",
        "fill_cost_fields": "कृपया स्थान, फसल और एकड़ भरें।",

        "quick_actions": "त्वरित एजेंट कार्य",
        "select_analysis": "विश्लेषण चुनें",
        "run_analysis": "विश्लेषण चलाएं",
        "do_all_analysis": "सभी विश्लेषण चलाएं",
        "run_core_layers": "सभी मुख्य लेयर चलाएं",
        "queued": "कतार में जोड़ा गया",
        "all_analyses_queued": "सभी विश्लेषण कतार में जोड़ दिए गए।",
        "core_layers_queued": "मुख्य लेयर कतार में जोड़ दी गईं।",

        "chat_export": "चैट एक्सपोर्ट",
        "export_pdf": "चैट को PDF में एक्सपोर्ट करें",
        "download_pdf": "PDF डाउनलोड करें",
        "pdf_failed": "PDF एक्सपोर्ट विफल रहा",

        "user": "उपयोगकर्ता",

        "farm_location": "फार्म का स्थान",
        "farm_location_placeholder": "उदाहरण: पुणे, महाराष्ट्र",
        "upload": "पत्ते की छवि अपलोड करें",
        "uploaded_leaf": "अपलोड किया गया पत्ता",
        "analyze": "विश्लेषण करें",

        "running_analysis": "पौधे का विश्लेषण चल रहा है...",
        "processing_image": "अपलोड की गई छवि संसाधित की जा रही है...",
        "preparing_metadata": "फार्म मेटाडेटा तैयार किया जा रहा है...",
        "sending_model": "छवि मॉडल को भेजी जा रही है...",
        "analysis_complete": "विश्लेषण पूरा हुआ",
        "analysis_failed": "विश्लेषण विफल रहा",
        "analysis_complete_success": "विश्लेषण पूरा हुआ!",

        "full_analysis": "पूर्ण विश्लेषण रिपोर्ट",
        "crop_identified": "पहचानी गई फसल",
        "disease_status": "रोग की स्थिति",
        "ai_confidence": "एआई विश्वास",

        "condition_assessment": "स्थिति मूल्यांकन",
        "actionable_prescription": "व्यावहारिक उपचार योजना",
        "soil_moisture": "मिट्टी और नमी की जानकारी",
        "water_irrigation": "पानी और सिंचाई मार्गदर्शन",
        "risk_urgency": "जोखिम और तात्कालिकता",
        "fertilizer_recommendations": "उर्वरक सुझाव",

        "risk_high": "जोखिम स्तर: उच्च",
        "risk_medium": "जोखिम स्तर: मध्यम",
        "risk_low": "जोखिम स्तर: कम",

        "chat_title": "💬 एजेंट चैट",
        "chat_placeholder": (
            "खेती, सिंचाई, फसल, रोग, मिट्टी या लागत के बारे में पूछें..."
        ),
        "agent_thinking": "एजेंट सोच रहा है...",

        "shop_title": "🛒 उर्वरक और कृषि सेवाएं",
        "doctors_title": "🩺 कृषि विशेषज्ञ",

        "search_warning": (
            "यह संस्करण सत्यापित स्थानीय व्यवसाय खोज API का उपयोग नहीं करता। "
            "एआई द्वारा बनाए गए परिणामों को वास्तविक व्यवसाय या संपर्क जानकारी न मानें।"
        ),

        "shop_crop_name": "दुकान: फसल का नाम",
        "shop_requirement": "दुकान: आवश्यकता",
        "doctors_crop_name": "विशेषज्ञ: फसल का नाम",
        "doctors_requirement": "विशेषज्ञ: आवश्यकता",

        "generate_shop_search": "दुकान खोज मानदंड बनाएं",
        "generate_doctors_search": "विशेषज्ञ खोज मानदंड बनाएं",
        "preparing_search": "खोज मानदंड तैयार किए जा रहे हैं...",

        "contact_title": "संपर्क",
        "generated_reports": "बनाई गई रिपोर्ट",
        "no_reports": "अभी कोई रिपोर्ट नहीं है। साइडबार से विश्लेषण चलाएं।",

        "cost_profit_report": "लागत और लाभ अनुमान रिपोर्ट",
        "market_price": "बाजार मूल्य",
        "price_trend_months": "मूल्य प्रवृत्ति और सर्वोत्तम महीने",
        "best_months": "बेचने के सर्वोत्तम महीने",
        "cost_revenue": "लागत और आय",
        "total_cost": "कुल लागत",
        "expected_revenue": "अनुमानित आय",
        "profit_loss": "लाभ / हानि",
        "travel_costs": "परिवहन लागत",
        "recommendation": "सिफारिश",

        "login_success": "लॉगिन सफल रहा।",
        "account_created": "खाता बनाया गया।",
        "invalid_password": "अमान्य पासवर्ड।",
        "credentials_required": "यूज़रनेम और पासवर्ड आवश्यक हैं।",

        "all_tasks_completed": "सभी कार्य पूरे हुए",
        "no_chat_messages": "एक्सपोर्ट करने के लिए कोई चैट संदेश नहीं है।",

        "na": "लागू नहीं",
        "no_recommendation": "कोई सिफारिश उपलब्ध नहीं है।",
    },

    "Marathi": {
        "home": "मुख्यपृष्ठ",
        "chat": "चॅट",
        "shops": "दुकान",
        "doctors": "तज्ज्ञ",
        "contact": "संपर्क",

        "login": "लॉगिन",
        "username": "वापरकर्ता नाव",
        "password": "पासवर्ड",
        "continue": "पुढे जा",
        "logout": "लॉगआउट",

        "app_title": "🌱 कृषी सुपर एआय एजंट",
        "app_caption": (
            "प्रतिमा विश्लेषण एआयच्या सहाय्याने केले जाते. "
            "प्रत्यक्ष उपचारापूर्वी शिफारसी तपासा."
        ),

        "agent_panel": "🌱 एजंट कंट्रोल पॅनेल",
        "agent_status": "एजंट स्थिती",

        "select_language": "भाषा निवडा",
        "apply_language": "भाषा लागू करा",

        "cost_estimation": "खर्चाचा अंदाज",
        "location_city": "स्थान (शहर/प्रदेश)",
        "crop_name_input": "पिकाचे नाव",
        "total_acres": "एकूण एकर",
        "total_invested": "एकूण गुंतवणूक (₹ किंवा $)",
        "estimate_cost": "खर्च आणि नफ्याचा अंदाज",
        "fill_cost_fields": "कृपया स्थान, पीक आणि एकर भरा.",

        "quick_actions": "जलद एजंट क्रिया",
        "select_analysis": "विश्लेषण निवडा",
        "run_analysis": "विश्लेषण चालवा",
        "do_all_analysis": "सर्व विश्लेषण चालवा",
        "run_core_layers": "सर्व मुख्य स्तर चालवा",
        "queued": "रांगेत जोडले",
        "all_analyses_queued": "सर्व विश्लेषणे रांगेत जोडली आहेत.",
        "core_layers_queued": "मुख्य स्तर रांगेत जोडले आहेत.",

        "chat_export": "चॅट एक्सपोर्ट",
        "export_pdf": "चॅट PDF म्हणून एक्सपोर्ट करा",
        "download_pdf": "PDF डाउनलोड करा",
        "pdf_failed": "PDF एक्सपोर्ट अयशस्वी झाला",

        "user": "वापरकर्ता",

        "farm_location": "शेताचे स्थान",
        "farm_location_placeholder": "उदाहरण: पुणे, महाराष्ट्र",
        "upload": "पानाची प्रतिमा अपलोड करा",
        "uploaded_leaf": "अपलोड केलेले पान",
        "analyze": "विश्लेषण करा",

        "running_analysis": "वनस्पतीचे विश्लेषण सुरू आहे...",
        "processing_image": "अपलोड केलेली प्रतिमा प्रक्रिया केली जात आहे...",
        "preparing_metadata": "शेताची माहिती तयार केली जात आहे...",
        "sending_model": "प्रतिमा मॉडेलकडे पाठवली जात आहे...",
        "analysis_complete": "विश्लेषण पूर्ण",
        "analysis_failed": "विश्लेषण अयशस्वी",
        "analysis_complete_success": "विश्लेषण पूर्ण झाले!",

        "full_analysis": "संपूर्ण विश्लेषण अहवाल",
        "crop_identified": "ओळखलेले पीक",
        "disease_status": "रोगाची स्थिती",
        "ai_confidence": "एआय आत्मविश्वास",

        "condition_assessment": "स्थितीचे मूल्यांकन",
        "actionable_prescription": "कृतीयोग्य उपचार योजना",
        "soil_moisture": "माती आणि आर्द्रतेची माहिती",
        "water_irrigation": "पाणी आणि सिंचन मार्गदर्शन",
        "risk_urgency": "धोका आणि तातडी",
        "fertilizer_recommendations": "खत शिफारसी",

        "risk_high": "धोका पातळी: उच्च",
        "risk_medium": "धोका पातळी: मध्यम",
        "risk_low": "धोका पातळी: कमी",

        "chat_title": "💬 एजंट चॅट",
        "chat_placeholder": (
            "शेती, सिंचन, पिके, रोग, माती किंवा खर्चाबद्दल विचारा..."
        ),
        "agent_thinking": "एजंट विचार करत आहे...",

        "shop_title": "🛒 खते आणि कृषी सेवा",
        "doctors_title": "🩺 कृषी तज्ज्ञ",

        "search_warning": (
            "ही आवृत्ती सत्यापित स्थानिक व्यवसाय शोध API वापरत नाही. "
            "एआयने तयार केलेले निकाल वास्तविक व्यवसाय किंवा संपर्क माहिती समजू नका."
        ),

        "shop_crop_name": "दुकान: पिकाचे नाव",
        "shop_requirement": "दुकान: आवश्यकता",
        "doctors_crop_name": "तज्ज्ञ: पिकाचे नाव",
        "doctors_requirement": "तज्ज्ञ: आवश्यकता",

        "generate_shop_search": "दुकान शोध निकष तयार करा",
        "generate_doctors_search": "तज्ज्ञ शोध निकष तयार करा",
        "preparing_search": "शोध निकष तयार केले जात आहेत...",

        "contact_title": "संपर्क",
        "generated_reports": "तयार केलेले अहवाल",
        "no_reports": "अद्याप कोणतेही अहवाल नाहीत. साइडबारमधून विश्लेषण चालवा.",

        "cost_profit_report": "खर्च आणि नफा अंदाज अहवाल",
        "market_price": "बाजारभाव",
        "price_trend_months": "भावाचा कल आणि सर्वोत्तम महिने",
        "best_months": "विक्रीसाठी सर्वोत्तम महिने",
        "cost_revenue": "खर्च आणि उत्पन्न",
        "total_cost": "एकूण खर्च",
        "expected_revenue": "अपेक्षित उत्पन्न",
        "profit_loss": "नफा / तोटा",
        "travel_costs": "वाहतूक खर्च",
        "recommendation": "शिफारस",

        "login_success": "लॉगिन यशस्वी.",
        "account_created": "खाते तयार केले.",
        "invalid_password": "चुकीचा पासवर्ड.",
        "credentials_required": "वापरकर्ता नाव आणि पासवर्ड आवश्यक आहेत.",

        "all_tasks_completed": "सर्व कामे पूर्ण झाली",
        "no_chat_messages": "एक्सपोर्ट करण्यासाठी चॅट संदेश नाहीत.",

        "na": "लागू नाही",
        "no_recommendation": "कोणतीही शिफारस उपलब्ध नाही.",
    },
}


FONT_MAP = {
    "English": "Arial, sans-serif",
    "Hindi": "'Nirmala UI', 'Mangal', sans-serif",
    "Marathi": "'Noto Sans Devanagari', 'Mangal', sans-serif",
}


LANGUAGE_CODE_MAP = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
}


# ============================================================
# AGENT ACTIONS
# ============================================================

ACTION_MAP = {
    "Soil moisture modeling":
        "Analyze soil moisture using available farm metadata. Clearly state assumptions and provide actionable irrigation guidance.",

    "Water requirement prediction":
        "Predict farm water requirement for the next 14 days. Clearly state assumptions and uncertainty.",

    "AI-driven irrigation schedule":
        "Create an AI-driven irrigation schedule with recommended time windows and estimated liters per acre.",

    "Drought early warning":
        "Generate drought early-warning indicators for the next 30 days based on supplied data and assumptions.",

    "Water waste optimization %":
        "Estimate current water waste percentage and identify optimization opportunities. State assumptions.",

    "NPK prediction":
        "Estimate nitrogen, phosphorus, and potassium status only from available evidence. Do not pretend laboratory measurements exist.",

    "pH imbalance detection":
        "Assess possible pH imbalance and recommend a treatment protocol. Clearly distinguish estimated versus measured values.",

    "Nutrient deficiency fusion":
        "Use available leaf and soil evidence to identify likely nutrient deficiencies and confidence levels.",

    "Fertilizer recommendation":
        "Build a fertilizer recommendation based on crop, growth stage, symptoms, and available soil evidence.",

    "Long-term soil health score":
        "Estimate a long-term soil health score with assumptions and a yearly improvement plan.",

    "Insect classification":
        "Classify likely insects and risk level based on supplied observations.",

    "Pest density estimation":
        "Estimate pest density only when sufficient evidence exists. Otherwise request required observations and give intervention thresholds.",

    "Swarm detection":
        "Assess swarm risk and provide an alert and mitigation plan.",

    "Migration pattern prediction":
        "Predict possible wind-based pest migration patterns over 7 days. Clearly state assumptions.",

    "Smart pesticide timing":
        "Recommend pesticide timing based on available weather and pest information. Include safety and label-compliance warnings.",

    "Satellite imagery integration":
        "Provide a practical satellite imagery integration architecture and explain which crop signals can be inferred.",

    "Growth stage tracking":
        "Estimate crop growth stage from available metadata and identify the next milestones.",

    "Production estimate per acre":
        "Estimate production per acre with a confidence range and clearly state assumptions.",

    "Profit forecast":
        "Generate a profit forecast using supplied yield, costs, and market-price assumptions.",

    "Market price integration":
        "Explain how to integrate verified market-price data and suggest sell timing. Do not invent live prices.",

    "Camera→Analyze→Recommend→Auto-execute":
        "Design a camera-to-analysis-to-recommendation-to-execution pipeline with safety gates and human approval.",

    "Irrigation valve control":
        "Generate irrigation valve control logic including sensor validation, timeout protection, and failsafes.",

    "Sprayer control":
        "Generate a smart sprayer control strategy with safety and manual override.",

    "Drone-based spraying":
        "Plan a drone-based spraying workflow including route logic, timing, weather constraints, and safety checks.",

    "Automated farm reporting":
        "Create an automated farm-reporting template with KPIs, alerts, and recommendations.",

    "Multi-modal fusion model":
        "Design a multimodal fusion architecture using Vision, Weather, Soil, Time, and Farm metadata.",

    "Disease risk 7-30 days":
        "Estimate disease risk for 7-30 days using actual available climate data or clearly stated assumptions.",

    "Frost risk alerts":
        "Assess frost risk and recommend preventive actions.",

    "Heat stress prediction":
        "Assess heat-stress risk and recommend protection actions.",

    "Crop growth stage mapping":
        "Design a crop growth-stage mapping system using multimodal farm data.",

    "Price prediction AI":
        "Calculate total crop production cost and expected profit using supplied costs and explicitly stated market assumptions.",

    "Full Agent Pipeline":
        "Build one complete end-to-end agricultural AI agent pipeline using Vision, Climate, Soil, Water, Market, and Execution layers.",
}


# ============================================================
# NAVIGATION
# IMPORTANT: INTERNAL IDs NEVER CHANGE WITH LANGUAGE
# ============================================================

NAV_ITEMS = [
    ("home", "home"),
    ("chat", "chat"),
    ("shops", "shops"),
    ("doctors", "doctors"),
    ("contact", "contact"),
]


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
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# TRANSLATION HELPERS
# ============================================================

def t(key: str) -> str:
    language = st.session_state.get("language", "English")

    value = TRANSLATIONS.get(language, {}).get(key)

    if value is not None:
        return value

    return TRANSLATIONS["English"].get(key, key)


def apply_local_font(language):
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


@st.cache_data(show_spinner=False, ttl=3600)
def translate_text(text: str, language: str) -> str:
    if language == "English":
        return text

    if not isinstance(text, str):
        return text

    if not SARVAM_API_KEY:
        return text

    stripped = text.strip()

    if not stripped:
        return text

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "source_language_code": "en-IN",
        "target_language_code": LANGUAGE_CODE_MAP.get(
            language,
            "en-IN",
        ),
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True,
        "numerals_format": "international",
        "input": stripped,
    }

    try:
        response = requests.post(
            SARVAM_TRANSLATE_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        translated = (
            data.get("translated_text")
            or data.get("translation")
            or data.get("output")
            or data.get("data", {}).get("translated_text")
        )

        if isinstance(translated, str) and translated.strip():
            return translated

    except Exception:
        pass

    return text


def translate_result_data(data, language):
    """
    Translates structured AI output for display.

    If the AI already generated output in the selected language,
    this function simply keeps it as returned when possible.
    """
    if language == "English":
        return data

    if isinstance(data, dict):
        result = {}

        for key, value in data.items():
            if isinstance(value, str):
                result[key] = translate_text(value, language)

            elif isinstance(value, list):
                result[key] = [
                    (
                        translate_text(item, language)
                        if isinstance(item, str)
                        else item
                    )
                    for item in value
                ]

            else:
                result[key] = value

        return result

    return data


# ============================================================
# JSON CLEANER
# ============================================================

def extract_json(text):
    """
    Extract JSON safely from:
    - plain JSON
    - markdown JSON blocks
    - surrounding model text
    """
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        raise ValueError("Model output is not a string")

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        return json.loads(candidate)

    raise ValueError("No valid JSON object found")


# ============================================================
# NVIDIA API
# ============================================================

def call_nvidia(
    messages,
    model=DEFAULT_MODEL,
    max_tokens=2500,
    max_retries=6,
):
    if not NVIDIA_API_KEY:
        return "API Configuration Error: API key is missing."

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    last_error = ""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                NVIDIA_API_URL,
                headers=headers,
                json=payload,
                timeout=180,
            )

        except requests.exceptions.Timeout:
            last_error = "Request timed out."

        except requests.exceptions.RequestException as e:
            last_error = f"Network Error: {str(e)}"

        else:
            if response.status_code == 200:
                try:
                    data = response.json()

                    return (
                        data["choices"][0]["message"]["content"]
                    )

                except (
                    ValueError,
                    KeyError,
                    IndexError,
                    TypeError,
                ):
                    return (
                        "Unexpected successful response: "
                        f"{response.text[:1000]}"
                    )

            if response.status_code in (
                429,
                500,
                502,
                503,
                504,
            ):
                last_error = (
                    f"Temporary server error "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            elif response.status_code in (401, 403):
                return (
                    f"Authorization Error "
                    f"{response.status_code}: "
                    f"{response.text[:1000]}"
                )

            else:
                return (
                    f"HTTP Error {response.status_code}: "
                    f"{response.text[:1000]}"
                )

        if attempt < max_retries - 1:
            wait_time = min(
                2 ** (attempt + 1),
                60,
            ) + random.uniform(0, 1.5)

            time.sleep(wait_time)

    return (
        "AI service is currently overloaded after "
        f"{max_retries} attempts. Last error: {last_error}"
    )


# ============================================================
# API TEST
# ============================================================

def test_nvidia_api():
    return call_nvidia(
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly: API connection successful"
                ),
            }
        ],
        model=DEFAULT_MODEL,
        max_tokens=50,
    )


# ============================================================
# VISION + REASONING
# ============================================================

def run_reasoning_model(image_bytes, species_info):
    if not NVIDIA_API_KEY:
        return {
            "error": t("api_key_missing")
        }

    base64_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    selected_language = st.session_state.language

    prompt = f"""
You are an agricultural AI assistant.

Analyze the uploaded plant/leaf image and the farm metadata.

Farm metadata:
{json.dumps(species_info, ensure_ascii=False)}

IMPORTANT LANGUAGE RULE:
Respond in {selected_language}.
Keep the JSON keys EXACTLY in English as specified below.
Translate ONLY the JSON values into {selected_language}.

Your job:

1. Identify the most likely crop or plant.
2. Identify the most likely disease, pest damage, nutrient deficiency,
   or health issue.
3. If the plant appears healthy, use "Healthy".
4. Provide a short explanation of visible evidence.
5. Provide safe, practical treatment or care steps.
6. Recommend nutrients or fertilizer categories when appropriate.
7. Assess soil and moisture only from available evidence.

CRITICAL RULES:

- Do NOT pretend you fetched weather, soil reports, satellite data,
  market data, or laboratory tests unless those values were explicitly provided.
- Clearly label estimates as estimates.
- Do NOT invent exact local measurements.
- If location information is missing, say that local conditions cannot
  be determined precisely.
- Include a confidence score from 0 to 100.

Return ONLY valid JSON.

Required structure:

{{
    "crop_name": "Name or Unknown",
    "disease_name": "Disease name, issue, or Healthy",
    "description": "Visible condition and evidence",
    "solution": "Step-by-step actionable guidance",
    "fertilizers": "Recommended nutrients or fertilizer categories",
    "soil_insights": "Evidence-based soil insight or what soil data is needed",
    "water_forecast": "Irrigation guidance based only on available data and assumptions",
    "risk_score": "Low, Medium, or High",
    "confidence": 0
}}
"""

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": VISION_MODEL,
        "messages": [
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
        ],
        "temperature": 0.1,
        "max_tokens": 2500,
    }

    try:
        response = requests.post(
            NVIDIA_API_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )

    except requests.exceptions.Timeout:
        return {
            "error": "AI request timed out."
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": f"Network Error: {str(e)}"
        }

    if response.status_code in (401, 403):
        return {
            "error": (
                f"Authorization Error "
                f"({response.status_code}). "
                "Check your API key and model access."
            ),
            "details": response.text[:1000],
        }

    if response.status_code != 200:
        return {
            "error": (
                f"HTTP Error {response.status_code}"
            ),
            "details": response.text[:1000],
        }

    try:
        result = response.json()

    except ValueError:
        return {
            "error": "API returned invalid JSON.",
            "raw": response.text[:1000],
        }

    if "error" in result:
        error = result["error"]

        return {
            "error": (
                error.get("message", str(error))
                if isinstance(error, dict)
                else str(error)
            ),
            "raw_response": result,
        }

    try:
        output_text = (
            result["choices"][0]["message"]["content"]
        )

        parsed = extract_json(output_text)

        defaults = {
            "crop_name": t("unknown"),
            "disease_name": t("unknown"),
            "description": "",
            "solution": "",
            "fertilizers": "",
            "soil_insights": "",
            "water_forecast": "",
            "risk_score": "Low",
            "confidence": 0,
        }

        for key, value in defaults.items():
            parsed.setdefault(key, value)

        return parsed

    except Exception as e:
        return {
            "error": (
                f"Could not parse model response: {str(e)}"
            ),
            "raw_response": result,
        }


# ============================================================
# TASK QUEUE
# ============================================================

def queue_task(
    task_name,
    prompt,
    model=REASONING_MODEL,
):
    st.session_state.task_queue.append(
        {
            "task": task_name,
            "prompt": prompt,
            "model": model,
        }
    )


def run_all_background_tasks():
    """
    Streamlit is synchronous.
    This queue runs tasks sequentially.
    """
    selected_language = st.session_state.language

    while st.session_state.task_queue:
        task = st.session_state.task_queue.pop(0)

        st.session_state.agent_status = (
            f"Running: {task['task']}"
        )

        system_prompt = f"""
You are an agricultural AI analyst.

Produce a structured operational report.

Respond in {selected_language}.

Rules:

- Never claim that live web, weather, satellite, market, or sensor data
  was accessed unless the data was actually provided.
- Explicitly state assumptions.
- Separate observations, estimates, and recommendations.
- Include risk level and actionable next steps.
"""

        report = call_nvidia(
            messages=[
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

    st.session_state.agent_status = t(
        "all_tasks_completed"
    )


# ============================================================
# PDF EXPORT
# ============================================================

def export_chat_to_pdf():
    os.makedirs(
        EXPORT_DIR,
        exist_ok=True,
    )

    path = os.path.join(
        EXPORT_DIR,
        (
            "chat_export_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        ),
    )

    lines = [
        "AI Agent Chat Export",
        "",
    ]

    for msg in st.session_state.chat_history:
        lines.append(
            f"[{msg['time']}] "
            f"{msg['role'].upper()}: "
            f"{msg['text']}"
        )

    if len(lines) <= 2:
        lines.append(t("no_chat_messages"))

    with PdfPages(path) as pdf:
        page_lines = 30

        for i in range(
            0,
            len(lines),
            page_lines,
        ):
            fig = plt.figure(
                figsize=(8.27, 11.69)
            )

            text_chunk = "\n".join(
                lines[i:i + page_lines]
            )

            fig.text(
                0.05,
                0.95,
                text_chunk,
                va="top",
                fontsize=9,
                family="sans-serif",
                wrap=True,
            )

            plt.axis("off")
            pdf.savefig(
                fig,
                bbox_inches="tight",
            )
            plt.close(fig)

    return path


# ============================================================
# LOGIN
# ============================================================

def load_users():
    if not os.path.exists(USER_DB):
        return {}

    try:
        with open(
            USER_DB,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}


def save_users(users):
    with open(
        USER_DB,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            users,
            file,
            indent=2,
        )


def login_block():
    if st.session_state.logged_in:
        return

    st.title(t("login"))

    username = st.text_input(
        t("username")
    )

    password = st.text_input(
        t("password"),
        type="password",
    )

    if st.button(t("continue")):
        username = username.strip()

        if not username or not password:
            st.error(
                t("credentials_required")
            )
            st.stop()

        users = load_users()

        if username in users:
            if users[username] != password:
                st.error(
                    t("invalid_password")
                )
                st.stop()

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success(
                t("login_success")
            )
            st.rerun()

        else:
            users[username] = password
            save_users(users)

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success(
                t("account_created")
            )
            st.rerun()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

def sidebar_controls():
    with st.sidebar:
        st.title(t("agent_panel"))

        # ====================================================
        # LANGUAGE
        # ====================================================

        current_index = list(
            TRANSLATIONS.keys()
        ).index(
            st.session_state.language
        )

        new_language = st.selectbox(
            t("select_language"),
            list(TRANSLATIONS.keys()),
            index=current_index,
        )

        if st.button(
            t("apply_language"),
            use_container_width=True,
        ):
            st.session_state.language = new_language
            st.rerun()

        st.markdown("---")

        # ====================================================
        # COST ESTIMATION
        # ====================================================

        st.subheader(
            t("cost_estimation")
        )

        est_location = st.text_input(
            t("location_city"),
            key="cost_location",
        )

        est_crop = st.text_input(
            t("crop_name_input"),
            key="cost_crop",
        )

        est_acres = st.number_input(
            t("total_acres"),
            min_value=0.0,
            step=0.1,
            key="cost_acres",
        )

        est_invested = st.number_input(
            t("total_invested"),
            min_value=0.0,
            step=100.0,
            key="cost_invested",
        )

        if st.button(
            t("estimate_cost"),
            use_container_width=True,
        ):
            if (
                not est_location
                or not est_crop
                or est_acres <= 0
            ):
                st.error(
                    t("fill_cost_fields")
                )

            else:
                selected_language = (
                    st.session_state.language
                )

                cost_prompt = f"""
Location: {est_location}
Crop: {est_crop}
Acres: {est_acres}
Investment: {est_invested}

Create an agricultural cost and profit analysis.

Respond in {selected_language}.

IMPORTANT:
You do not have verified live market access in this request.
Do not invent current market prices.

Use assumptions where required and clearly label them.

Return ONLY valid JSON.

Keep these JSON keys exactly in English.
Translate the JSON values into {selected_language}.

{{
    "market_price": "Verified data unavailable unless provided; otherwise estimated assumption",
    "price_trend": "Trend analysis with assumptions",
    "best_months": ["Month"],
    "total_cost": "Estimated cost",
    "expected_revenue": "Estimated revenue",
    "profit_or_loss": "Estimated profit/loss",
    "travel_costs": "Estimated transport costs",
    "recommendation": "Actionable recommendation"
}}
"""

                estimation = call_nvidia(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an agricultural "
                                "economic analyst. "
                                "Never fabricate live market data."
                            ),
                        },
                        {
                            "role": "user",
                            "content": cost_prompt,
                        },
                    ],
                    model=REASONING_MODEL,
                )

                st.session_state.cost_estimation = (
                    estimation
                )

        st.markdown("---")

        # ====================================================
        # QUICK ACTIONS
        # ====================================================

        st.subheader(
            t("quick_actions")
        )

        selected_action = st.selectbox(
            t("select_analysis"),
            list(ACTION_MAP.keys()),
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                t("run_analysis"),
                use_container_width=True,
            ):
                queue_task(
                    selected_action,
                    ACTION_MAP[selected_action],
                )

                st.success(
                    f"{t('queued')}: "
                    f"{selected_action}"
                )

        with col2:
            if st.button(
                t("do_all_analysis"),
                use_container_width=True,
            ):
                for action, prompt in ACTION_MAP.items():
                    queue_task(
                        action,
                        prompt,
                    )

                st.success(
                    t("all_analyses_queued")
                )

        if st.button(
            t("run_core_layers"),
            use_container_width=True,
        ):
            layers = [
                t("core_vision"),
                t("core_climate"),
                t("core_soil"),
                t("core_water"),
                t("core_market"),
                t("core_execution"),
            ]

            for layer in layers:
                queue_task(
                    layer,
                    (
                        f"Generate an operational report for "
                        f"{layer}. State all assumptions and "
                        f"do not claim unavailable data sources."
                    ),
                )

            st.success(
                t("core_layers_queued")
            )

        st.markdown("---")

        # ====================================================
        # PDF EXPORT
        # ====================================================

        st.subheader(
            t("chat_export")
        )

        if st.button(
            t("export_pdf"),
            use_container_width=True,
        ):
            try:
                pdf_path = export_chat_to_pdf()

                with open(
                    pdf_path,
                    "rb",
                ) as pdf_file:
                    st.download_button(
                        t("download_pdf"),
                        data=pdf_file.read(),
                        file_name=os.path.basename(
                            pdf_path
                        ),
                        mime="application/pdf",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(
                    f"{t('pdf_failed')}: {str(e)}"
                )

        st.markdown("---")

        # ====================================================
        # USER
        # ====================================================

        st.subheader(
            t("user")
        )

        st.image(
            st.session_state.photo_url,
            width=70,
        )

        st.write(
            st.session_state.username
        )

        if st.button(
            t("logout"),
            use_container_width=True,
        ):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()


# ============================================================
# HOME PAGE
# ============================================================

def home_page():
    st.title(
        t("app_title")
    )

    st.caption(
        t("app_caption")
    )

    st.session_state.location = st.text_input(
        t("farm_location"),
        value=st.session_state.location,
        placeholder=t(
            "farm_location_placeholder"
        ),
    )

    uploaded_image = st.file_uploader(
        t("upload"),
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_image:
        try:
            image = Image.open(
                uploaded_image
            )

            st.image(
                image,
                caption=t("uploaded_leaf"),
                use_container_width=True,
            )

        except Exception as e:
            st.error(
                f"{t('could_not_open_image')}: "
                f"{str(e)}"
            )
            return

        if st.button(
            t("analyze"),
            type="primary",
            use_container_width=True,
        ):
            if not NVIDIA_API_KEY:
                st.error(
                    t("api_key_missing")
                )
                return

            try:
                buffer = io.BytesIO()

                processed_image = image.convert(
                    "RGB"
                )

                processed_image.save(
                    buffer,
                    format="JPEG",
                    quality=90,
                )

                image_bytes = buffer.getvalue()

            except Exception as e:
                st.error(
                    f"{t('image_processing_failed')}: "
                    f"{str(e)}"
                )
                return

            status = st.status(
                t("running_analysis"),
                expanded=True,
            )

            status.write(
                t("processing_image")
            )

            status.write(
                t("preparing_metadata")
            )

            species_info = {
                "location": (
                    st.session_state.location
                    or "Not provided"
                )
            }

            status.write(
                t("sending_model")
            )

            result = run_reasoning_model(
                image_bytes,
                species_info,
            )

            st.session_state.detection_result = result

            if "error" not in result:
                status.update(
                    label=t("analysis_complete"),
                    state="complete",
                    expanded=False,
                )

                st.success(
                    t("analysis_complete_success")
                )

            else:
                status.update(
                    label=t("analysis_failed"),
                    state="error",
                    expanded=True,
                )

                st.error(
                    result["error"]
                )

                if result.get("details"):
                    st.code(
                        result["details"]
                    )

    # ========================================================
    # RESULTS
    # ========================================================

    result = st.session_state.detection_result

    if result and "error" not in result:
        st.markdown("---")

        st.markdown(
            f"## {t('full_analysis')}"
        )

        col_crop, col_disease, col_confidence = (
            st.columns(3)
        )

        with col_crop:
            st.metric(
                t("crop_identified"),
                result.get(
                    "crop_name",
                    t("unknown"),
                ),
            )

        with col_disease:
            st.metric(
                t("disease_status"),
                result.get(
                    "disease_name",
                    t("unknown"),
                ),
            )

        with col_confidence:
            st.metric(
                t("ai_confidence"),
                f"{result.get('confidence', 0)}%",
            )

        st.markdown(
            f"### {t('condition_assessment')}"
        )

        st.write(
            result.get(
                "description",
                t("no_description"),
            )
        )

        st.markdown(
            f"### {t('actionable_prescription')}"
        )

        st.write(
            result.get(
                "solution",
                t("no_solution"),
            )
        )

        st.markdown(
            f"### {t('soil_moisture')}"
        )

        st.write(
            result.get(
                "soil_insights",
                t("no_soil_insights"),
            )
        )

        st.markdown(
            f"### {t('water_irrigation')}"
        )

        st.write(
            result.get(
                "water_forecast",
                t("no_irrigation"),
            )
        )

        st.markdown(
            f"### {t('risk_urgency')}"
        )

        risk = str(
            result.get(
                "risk_score",
                "Low",
            )
        ).strip().lower()

        if risk in (
            "high",
            "उच्च",
        ):
            st.error(
                t("risk_high")
            )

        elif risk in (
            "medium",
            "मध्यम",
        ):
            st.warning(
                t("risk_medium")
            )

        else:
            st.success(
                t("risk_low")
            )

        st.markdown(
            f"### {t('fertilizer_recommendations')}"
        )

        st.write(
            result.get(
                "fertilizers",
                t("no_fertilizer"),
            )
        )


# ============================================================
# CHAT PAGE
# ============================================================

def chat_page():
    st.title(
        t("chat_title")
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(
            msg["role"]
        ):
            st.caption(
                msg["time"]
            )

            st.write(
                msg["text"]
            )

    query = st.chat_input(
        t("chat_placeholder")
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

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner(
                t("agent_thinking")
            ):
                selected_language = (
                    st.session_state.language
                )

                answer = call_nvidia(
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
You are a practical agricultural AI assistant.

Be clear and evidence-based.

Respond completely in {selected_language}.

Never claim you accessed live weather, market, soil, satellite,
or sensor data unless it was explicitly provided.

If the user asks for current local information, explain what
additional verified data source or API is needed.
""",
                        },
                        {
                            "role": "user",
                            "content": query,
                        },
                    ],
                    model=REASONING_MODEL,
                )

                st.write(
                    answer
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


# ============================================================
# SHOP / DOCTORS PAGE
# ============================================================

def shop_or_doctors_page(
    title,
    actor,
):
    st.title(title)

    st.warning(
        t("search_warning")
    )

    col_in1, col_in2 = st.columns(2)

    if actor == "Shop":
        crop_key = "shop_crop_name"
        requirement_key = "shop_requirement"
        button_key = "generate_shop_search"

    else:
        crop_key = "doctors_crop_name"
        requirement_key = "doctors_requirement"
        button_key = "generate_doctors_search"

    with col_in1:
        crop = st.text_input(
            t(crop_key)
        )

    with col_in2:
        requirement = st.text_input(
            t(requirement_key)
        )

    if st.button(
        t(button_key),
        use_container_width=True,
    ):
        location = (
            st.session_state.location
            or "Location not provided"
        )

        selected_language = (
            st.session_state.language
        )

        prompt = f"""
Create a practical search specification for finding real agricultural
{actor.lower()} services.

Location: {location}
Crop: {crop}
Requirement: {requirement}

Respond in {selected_language}.

Do NOT invent businesses, addresses, phone numbers, or prices.

Instead provide:

1. What type of provider to search for.
2. Important qualifications.
3. Questions to ask.
4. Warning signs.
5. Search keywords.
"""

        with st.spinner(
            t("preparing_search")
        ):
            response = call_nvidia(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=REASONING_MODEL,
            )

        st.markdown(
            response
        )


# ============================================================
# CONTACT PAGE
# ============================================================

def contact_page():
    st.title(
        t("contact_title")
    )

    st.markdown(
        f"""
**Agri Super Agent**

{t("contact_intro")}

{t("contact_production")}

- {t("weather")}
- {t("soil_sensors")}
- {t("market_prices")}
- {t("satellite_imagery")}
- {t("local_services")}
"""
    )


# ============================================================
# REPORTS PANEL
# ============================================================

def show_reports_panel():
    st.markdown(
        f"## {t('generated_reports')}"
    )

    if not st.session_state.reports:
        st.info(
            t("no_reports")
        )
        return

    for report in st.session_state.reports[:12]:
        with st.expander(
            f"{report['time']} — "
            f"{report['title']}"
        ):
            st.write(
                report["content"]
            )


# ============================================================
# COST REPORT
# ============================================================

def show_cost_report():
    est = st.session_state.cost_estimation

    if not est:
        return

    st.markdown("---")

    st.markdown(
        f"## {t('cost_profit_report')}"
    )

    try:
        est_json = extract_json(est)

    except Exception:
        st.warning(
            t("could_not_parse_estimation")
        )

        st.write(est)
        return

    st.markdown(
        f"### {t('market_price')}"
    )

    st.write(
        est_json.get(
            "market_price",
            t("na"),
        )
    )

    st.markdown(
        f"### {t('price_trend_months')}"
    )

    st.write(
        est_json.get(
            "price_trend",
            t("na"),
        )
    )

    best_months = est_json.get(
        "best_months",
        [],
    )

    st.write(
        f"{t('best_months')}: "
        f"{best_months}"
    )

    st.markdown(
        f"### {t('cost_revenue')}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        t("total_cost"),
        str(
            est_json.get(
                "total_cost",
                t("na"),
            )
        ),
    )

    col2.metric(
        t("expected_revenue"),
        str(
            est_json.get(
                "expected_revenue",
                t("na"),
            )
        ),
    )

    col3.metric(
        t("profit_loss"),
        str(
            est_json.get(
                "profit_or_loss",
                t("na"),
            )
        ),
    )

    st.markdown(
        f"### {t('travel_costs')}"
    )

    st.write(
        est_json.get(
            "travel_costs",
            t("na"),
        )
    )

    st.markdown(
        f"### {t('recommendation')}"
    )

    st.info(
        est_json.get(
            "recommendation",
            t("no_recommendation"),
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_session_defaults()

    apply_local_font(
        st.session_state.language
    )

    login_block()

    sidebar_controls()

    # ========================================================
    # RUN TASK QUEUE
    # ========================================================

    if st.session_state.task_queue:
        with st.spinner(
            t("running_analysis")
        ):
            run_all_background_tasks()

    st.markdown(
        f"### {t('agent_status')}: "
        f"{st.session_state.agent_status}"
    )

    # ========================================================
    # NAVIGATION
    # ========================================================

    cols = st.columns(5)

    for i, (menu_id, label_key) in enumerate(
        NAV_ITEMS
    ):
        button_type = (
            "primary"
            if st.session_state.menu_choice == menu_id
            else "secondary"
        )

        if cols[i].button(
            t(label_key),
            key=f"nav_{menu_id}",
            use_container_width=True,
            type=button_type,
        ):
            st.session_state.menu_choice = menu_id
            st.rerun()

    menu = st.session_state.menu_choice

    # ========================================================
    # PAGE ROUTING
    # ========================================================

    if menu == "home":
        home_page()

    elif menu == "chat":
        chat_page()

    elif menu == "shops":
        shop_or_doctors_page(
            t("shop_title"),
            "Shop",
        )

    elif menu == "doctors":
        shop_or_doctors_page(
            t("doctors_title"),
            "Doctors",
        )

    else:
        contact_page()

    # ========================================================
    # GLOBAL PANELS
    # ========================================================

    show_cost_report()
    show_reports_panel()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
