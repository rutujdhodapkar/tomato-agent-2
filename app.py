import base64
import io
import json
import os
import re
from datetime import datetime

import requests
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
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

# Put the model you have ACTUALLY confirmed is available
# for your NVIDIA API account.
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


NVIDIA_API_KEY = get_secret("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")
SARVAM_API_KEY = get_secret("sk_22eh8r36_QnYnweT7GxKSYa1Iccx7h189")


# ============================================================
# LANGUAGE CONFIG
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
        "upload": "Upload Leaf Image",
        "analyze": "Analyze",
        "btn_desc": "📄 Disease Description",
        "btn_sol": "💡 Get Solution",
        "btn_fert": "🧪 Get Fertilizers",
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
        "upload": "पत्ता अपलोड करें",
        "analyze": "विश्लेषण",
        "btn_desc": "📄 बीमारी का विवरण",
        "btn_sol": "💡 समाधान प्राप्त करें",
        "btn_fert": "🧪 उर्वरक प्राप्त करें",
    },
    "Marathi": {
        "home": "मुख्यपृष्ठ",
        "chat": "चॅट",
        "shops": "दुकान",
        "doctors": "डॉक्टर्स",
        "contact": "संपर्क",
        "login": "लॉगिन",
        "username": "वापरकर्ता नाव",
        "password": "पासवर्ड",
        "upload": "पान अपलोड करा",
        "analyze": "विश्लेषण",
        "btn_desc": "📄 रोगाचे वर्णन",
        "btn_sol": "💡 उपाय मिळवा",
        "btn_fert": "🧪 खते मिळवा",
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
# SESSION STATE
# ============================================================

def ensure_session_defaults():
    defaults = {
        "language": "English",
        "logged_in": False,
        "username": "",
        "photo_url": "https://api.dicebear.com/8.x/adventurer/png?seed=Farmer",
        "agent_status": "Idle",
        "task_queue": [],
        "reports": [],
        "chat_history": [],
        "detection_result": None,
        "menu_choice": "Home",
        "location": "",
        "cost_estimation": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# FONT
# ============================================================

def apply_local_font(language):
    font_family = FONT_MAP.get(language, FONT_MAP["English"])

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
# TRANSLATION
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def translate_text(text, language):
    if language == "English" or not isinstance(text, str):
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
        "target_language_code": LANGUAGE_CODE_MAP.get(language, "en-IN"),
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


def t(text):
    return translate_text(text, st.session_state.language)


def translate_result_data(data, language):
    if language == "English":
        return data

    if isinstance(data, dict):
        return {
            key: translate_result_data(value, language)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            translate_result_data(item, language)
            for item in data
        ]

    if isinstance(data, str):
        return translate_text(data, language)

    return data


# ============================================================
# JSON CLEANER
# ============================================================

def extract_json(text):
    """
    Extract JSON safely from:
    - plain JSON
    - ```json blocks
    - ``` blocks
    - surrounding model text
    """

    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        raise ValueError("Model output is not a string")

    cleaned = text.strip()

    # Remove markdown code fences
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    # First try direct JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        return json.loads(candidate)

    raise ValueError("No valid JSON object found")


# ============================================================
# NVIDIA API
# ============================================================

def call_nvidia(messages, model=DEFAULT_MODEL, max_tokens=2500):
    """
    Standard NVIDIA chat-completions request.
    """

    if not NVIDIA_API_KEY:
        return (
            "API Configuration Error: NVIDIA_API_KEY is missing. "
            "Add it to .streamlit/secrets.toml."
        )

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

    try:
        response = requests.post(
            NVIDIA_API_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )

    except requests.exceptions.Timeout:
        return "Network Error: NVIDIA API request timed out."

    except requests.exceptions.RequestException as e:
        return f"Network Error: {str(e)}"

    if response.status_code == 401:
        return (
            "NVIDIA Authentication Error (401): "
            "Your API key is invalid or expired."
        )

    if response.status_code == 403:
        return (
            "NVIDIA Authorization Error (403): "
            "Your API key was accepted by the server but is not authorized "
            "for this resource/model. Generate a fresh key and verify that "
            f"your account has access to model '{model}'. "
            f"Server response: {response.text}"
        )

    if response.status_code != 200:
        return (
            f"HTTP Error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    try:
        data = response.json()

    except ValueError:
        return (
            "API returned a non-JSON response: "
            f"{response.text[:1000]}"
        )

    if "error" in data:
        error = data["error"]

        if isinstance(error, dict):
            message = error.get("message", str(error))
        else:
            message = str(error)

        return f"NVIDIA API Error: {message}"

    try:
        content = data["choices"][0]["message"]["content"]

        if not content:
            return "NVIDIA API returned an empty response."

        return content

    except (KeyError, IndexError, TypeError):
        return f"Unexpected NVIDIA response format: {data}"


# ============================================================
# API TEST
# ============================================================

def test_nvidia_api():
    """
    Simple authentication and model-access test.
    """

    result = call_nvidia(
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: NVIDIA API connection successful"
            }
        ],
        model=DEFAULT_MODEL,
        max_tokens=50,
    )

    return result


# ============================================================
# VISION + REASONING
# ============================================================

def run_reasoning_model(image_bytes, species_info):
    """
    Sends plant image + farm metadata to NVIDIA.
    Returns parsed JSON dictionary.
    """

    if not NVIDIA_API_KEY:
        return {
            "error": (
                "NVIDIA_API_KEY is missing. "
                "Add it to .streamlit/secrets.toml."
            )
        }

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""
You are an agricultural AI assistant.

Analyze the uploaded plant/leaf image and the farm metadata.

Farm metadata:
{json.dumps(species_info, ensure_ascii=False)}

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
            "error": "NVIDIA API request timed out."
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": f"Network Error: {str(e)}"
        }

    if response.status_code == 401:
        return {
            "error": (
                "NVIDIA Authentication Error (401): "
                "API key is invalid or expired."
            )
        }

    if response.status_code == 403:
        return {
            "error": (
                "NVIDIA Authorization Error (403). "
                "The API key is not authorized for this endpoint or model. "
                f"Model requested: {VISION_MODEL}. "
                "Create a fresh NVIDIA API key and verify model access."
            ),
            "details": response.text[:1000],
        }

    if response.status_code != 200:
        return {
            "error": f"HTTP Error {response.status_code}",
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
        output_text = result["choices"][0]["message"]["content"]

        parsed = extract_json(output_text)

        # Ensure expected fields exist
        defaults = {
            "crop_name": "Unknown",
            "disease_name": "Unknown",
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

def queue_task(task_name, prompt, model=REASONING_MODEL):
    st.session_state.task_queue.append(
        {
            "task": task_name,
            "prompt": prompt,
            "model": model,
        }
    )


def run_all_background_tasks():
    """
    Note:
    Streamlit is synchronous. This is a task queue, not true background
    execution. It runs tasks sequentially during the current app run.
    """

    while st.session_state.task_queue:

        task = st.session_state.task_queue.pop(0)

        st.session_state.agent_status = (
            f"Running: {task['task']}"
        )

        system_prompt = """
You are an agricultural AI analyst.

Produce a structured operational report.

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

    st.session_state.agent_status = "All tasks completed"


# ============================================================
# PDF EXPORT
# ============================================================

def export_chat_to_pdf():
    os.makedirs(EXPORT_DIR, exist_ok=True)

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
        lines.append("No chat messages to export.")

    with PdfPages(path) as pdf:

        page_lines = 30

        for i in range(
            0,
            len(lines),
            page_lines
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
            pdf.savefig(fig, bbox_inches="tight")
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
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def save_users(users):
    with open(
        USER_DB,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            users,
            file,
            indent=2
        )


def login_block(lang_text):

    if st.session_state.logged_in:
        return

    st.title(lang_text["login"])

    username = st.text_input(
        lang_text["username"]
    )

    password = st.text_input(
        lang_text["password"],
        type="password",
    )

    if st.button("Continue"):

        username = username.strip()

        if not username or not password:
            st.error(
                "Username and password are required."
            )
            st.stop()

        users = load_users()

        if username in users:

            if users[username] != password:
                st.error("Invalid password.")
                st.stop()

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success("Login successful.")
            st.rerun()

        else:
            users[username] = password
            save_users(users)

            st.session_state.logged_in = True
            st.session_state.username = username

            st.success("Account created.")
            st.rerun()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

def sidebar_controls(lang_text):

    with st.sidebar:

        st.title("🌱 Agent Control Panel")

        # -----------------------------
        # API STATUS
        # -----------------------------

        st.subheader("API Status")

        if NVIDIA_API_KEY:
            st.success("NVIDIA API key loaded")
        else:
            st.error("NVIDIA API key missing")

        if st.button("Test NVIDIA API"):
            with st.spinner("Testing connection..."):
                result = test_nvidia_api()

            if (
                "Error" in result
                or "403" in result
                or "401" in result
            ):
                st.error(result)
            else:
                st.success(result)

        st.markdown("---")

        # -----------------------------
        # LANGUAGE
        # -----------------------------

        current_index = list(
            TRANSLATIONS.keys()
        ).index(
            st.session_state.language
        )

        new_language = st.selectbox(
            "Select Language",
            list(TRANSLATIONS.keys()),
            index=current_index,
        )

        if st.button("Apply Language"):
            st.session_state.language = new_language
            st.rerun()

        st.markdown("---")

        # -----------------------------
        # COST ESTIMATION
        # -----------------------------

        st.subheader("Cost Estimation")

        est_location = st.text_input(
            "Location (city/region)"
        )

        est_crop = st.text_input(
            "Crop name"
        )

        est_acres = st.number_input(
            "Total acres",
            min_value=0.0,
            step=0.1,
        )

        est_invested = st.number_input(
            "Total invested (₹ or $)",
            min_value=0.0,
            step=100.0,
        )

        if st.button("Estimate Cost & Profit"):

            if (
                not est_location
                or not est_crop
                or est_acres <= 0
            ):
                st.error(
                    "Please fill location, crop, and acres."
                )

            else:

                cost_prompt = f"""
Location: {est_location}
Crop: {est_crop}
Acres: {est_acres}
Investment: {est_invested}

Create an agricultural cost and profit analysis.

IMPORTANT:
You do not have verified live market access in this request.
Do not invent current market prices.

Use assumptions where required and clearly label them.

Return ONLY valid JSON:

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
                                "You are an agricultural economic analyst. "
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

                st.session_state.cost_estimation = estimation

        st.markdown("---")

        # -----------------------------
        # QUICK ACTIONS
        # -----------------------------

        st.subheader("Quick Agent Actions")

        selected_action = st.selectbox(
            "Select analysis",
            list(ACTION_MAP.keys()),
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Run analysis"):
                queue_task(
                    selected_action,
                    ACTION_MAP[selected_action],
                )
                st.success(
                    f"Queued: {selected_action}"
                )

        with col2:
            if st.button("Do all analysis"):
                for action, prompt in ACTION_MAP.items():
                    queue_task(action, prompt)

                st.success(
                    "All analyses queued."
                )

        if st.button("Run all core layers"):

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
                        f"Generate an operational report for "
                        f"{layer}. State all assumptions and "
                        f"do not claim unavailable data sources."
                    ),
                )

            st.success(
                "Core layers queued."
            )

        st.markdown("---")

        # -----------------------------
        # PDF EXPORT
        # -----------------------------

        st.subheader("Chat Export")

        if st.button("Export chat as PDF"):

            try:
                pdf_path = export_chat_to_pdf()

                with open(
                    pdf_path,
                    "rb"
                ) as pdf_file:

                    st.download_button(
                        "Download PDF",
                        data=pdf_file.read(),
                        file_name=os.path.basename(
                            pdf_path
                        ),
                        mime="application/pdf",
                    )

            except Exception as e:
                st.error(
                    f"PDF export failed: {str(e)}"
                )

        st.markdown("---")

        # -----------------------------
        # USER
        # -----------------------------

        st.subheader("User")

        st.image(
            st.session_state.photo_url,
            width=70,
        )

        st.write(
            st.session_state.username
        )

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()


# ============================================================
# HOME PAGE
# ============================================================

def home_page(lang_text):

    st.title("🌱 Agricultural Super AI Agent")

    st.caption(
        "Image analysis is AI-assisted. "
        "Recommendations should be verified before real-world treatment."
    )

    st.session_state.location = st.text_input(
        "Farm location",
        value=st.session_state.location,
        placeholder="Example: Pune, Maharashtra",
    )

    uploaded_image = st.file_uploader(
        lang_text["upload"],
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_image:

        try:
            image = Image.open(
                uploaded_image
            )

            st.image(
                image,
                caption="Uploaded Leaf",
                use_container_width=True,
            )

        except Exception as e:
            st.error(
                f"Could not open image: {str(e)}"
            )
            return

        if st.button(
            lang_text["analyze"],
            type="primary",
        ):

            if not NVIDIA_API_KEY:
                st.error(
                    "NVIDIA_API_KEY is missing. "
                    "Configure Streamlit secrets first."
                )
                return

            try:
                buffer = io.BytesIO()

                # Convert RGBA / PNG safely to JPEG
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
                    f"Image processing failed: {str(e)}"
                )
                return

            status = st.status(
                "Running plant analysis...",
                expanded=True,
            )

            status.write(
                "Processing uploaded image..."
            )

            status.write(
                "Preparing farm metadata..."
            )

            species_info = {
                "location": (
                    st.session_state.location
                    or "Not provided"
                )
            }

            status.write(
                "Sending image to NVIDIA model..."
            )

            result = run_reasoning_model(
                image_bytes,
                species_info,
            )

            st.session_state.detection_result = result

            if "error" not in result:

                status.update(
                    label="Analysis complete",
                    state="complete",
                    expanded=False,
                )

                st.success(
                    "Analysis Complete!"
                )

            else:

                status.update(
                    label="Analysis failed",
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

    result = (
        st.session_state.detection_result
    )

    if result and "error" not in result:

        res = translate_result_data(
            result,
            st.session_state.language,
        )

        st.markdown("---")

        st.markdown(
            "## Full Analysis Report"
        )

        col_crop, col_disease, col_confidence = (
            st.columns(3)
        )

        with col_crop:
            st.metric(
                "Crop Identified",
                res.get(
                    "crop_name",
                    "Unknown"
                ),
            )

        with col_disease:
            st.metric(
                "Disease Status",
                res.get(
                    "disease_name",
                    "Unknown"
                ),
            )

        with col_confidence:
            st.metric(
                "AI Confidence",
                f"{res.get('confidence', 0)}%",
            )

        st.markdown(
            "### Condition Assessment"
        )

        st.write(
            res.get(
                "description",
                "No description available.",
            )
        )

        st.markdown(
            "### Actionable Prescription"
        )

        st.write(
            res.get(
                "solution",
                "No solution provided.",
            )
        )

        st.markdown(
            "### Soil and Moisture Insights"
        )

        st.write(
            res.get(
                "soil_insights",
                "No soil insights available.",
            )
        )

        st.markdown(
            "### Water and Irrigation Guidance"
        )

        st.write(
            res.get(
                "water_forecast",
                "No irrigation guidance available.",
            )
        )

        st.markdown(
            "### Risk and Urgency"
        )

        risk = str(
            res.get(
                "risk_score",
                "Low"
            )
        ).strip().lower()

        if risk == "high":
            st.error("Risk Level: High")

        elif risk == "medium":
            st.warning("Risk Level: Medium")

        else:
            st.success("Risk Level: Low")

        st.markdown(
            "### Fertilizer Recommendations"
        )

        st.write(
            res.get(
                "fertilizers",
                "No fertilizer recommendations available.",
            )
        )


# ============================================================
# CHAT PAGE
# ============================================================

def chat_page():

    st.title("💬 Agent Chat")

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
        "Ask about farming, irrigation, crops, disease, soil, or costs..."
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
                "Agent is thinking..."
            ):

                answer = call_nvidia(
                    messages=[
                        {
                            "role": "system",
                            "content": """
You are a practical agricultural AI assistant.

Be clear and evidence-based.

Never claim you accessed live weather, market, soil, satellite,
or sensor data unless it was explicitly provided.

If the user asks for current local information, explain what
additional data source or verified API is needed.
""",
                        },
                        {
                            "role": "user",
                            "content": query,
                        },
                    ],
                    model=REASONING_MODEL,
                )

            st.write(answer)

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
    lang_text,
):

    st.title(title)

    st.warning(
        "This version does not use a verified local-business "
        "search API. Results generated by AI should not be treated "
        "as real businesses or real contact information."
    )

    col_in1, col_in2 = st.columns(2)

    with col_in1:
        crop = st.text_input(
            f"{actor}: Crop name"
        )

    with col_in2:
        requirement = st.text_input(
            f"{actor}: Requirement"
        )

    if st.button(
        f"Generate {actor} Search Criteria"
    ):

        location = (
            st.session_state.location
            or "Location not provided"
        )

        prompt = f"""
Create a practical search specification for finding real agricultural
{actor.lower()} services.

Location: {location}
Crop: {crop}
Requirement: {requirement}

Do NOT invent businesses, addresses, phone numbers, or prices.

Instead provide:
1. What type of provider to search for.
2. Important qualifications.
3. Questions to ask.
4. Warning signs.
5. Search keywords.
"""

        with st.spinner(
            "Preparing search criteria..."
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

        st.markdown(response)


# ============================================================
# CONTACT PAGE
# ============================================================

def contact_page():

    st.title("Contact")

    st.markdown(
        """
**AI Farm Agent**

This application combines plant image analysis,
agricultural reasoning, and farm-planning workflows.

For production deployment, connect verified data sources for:

- Weather
- Soil sensors
- Market prices
- Satellite imagery
- Local agricultural services
"""
    )


# ============================================================
# REPORTS PANEL
# ============================================================

def show_reports_panel():

    st.markdown(
        "## Generated Reports"
    )

    if not st.session_state.reports:

        st.info(
            "No reports yet. "
            "Run analyses from the sidebar."
        )

        return

    for report in (
        st.session_state.reports[:12]
    ):

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

    est = (
        st.session_state.cost_estimation
    )

    if not est:
        return

    st.markdown("---")

    st.markdown(
        "## Cost and Profit Estimation Report"
    )

    try:
        est_json = extract_json(est)

    except Exception:

        st.warning(
            "Could not parse estimation as JSON."
        )

        st.write(est)

        return

    est_json = translate_result_data(
        est_json,
        st.session_state.language,
    )

    st.markdown(
        "### Market Price"
    )

    st.write(
        est_json.get(
            "market_price",
            "N/A",
        )
    )

    st.markdown(
        "### Price Trend and Best Months"
    )

    st.write(
        est_json.get(
            "price_trend",
            "N/A",
        )
    )

    best_months = est_json.get(
        "best_months",
        [],
    )

    st.write(
        f"Best months to sell: {best_months}"
    )

    st.markdown(
        "### Cost and Revenue"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Cost",
        str(
            est_json.get(
                "total_cost",
                "N/A",
            )
        ),
    )

    col2.metric(
        "Expected Revenue",
        str(
            est_json.get(
                "expected_revenue",
                "N/A",
            )
        ),
    )

    col3.metric(
        "Profit / Loss",
        str(
            est_json.get(
                "profit_or_loss",
                "N/A",
            )
        ),
    )

    st.markdown(
        "### Travel Costs"
    )

    st.write(
        est_json.get(
            "travel_costs",
            "N/A",
        )
    )

    st.markdown(
        "### Recommendation"
    )

    st.info(
        est_json.get(
            "recommendation",
            "No recommendation available.",
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

    lang_text = TRANSLATIONS[
        st.session_state.language
    ]

    login_block(lang_text)

    sidebar_controls(lang_text)

    # Run queued tasks
    if st.session_state.task_queue:

        with st.spinner(
            "Running queued analyses..."
        ):
            run_all_background_tasks()

    st.markdown(
        f"### Agent Status: "
        f"{st.session_state.agent_status}"
    )

    # Navigation
    menu_items = [
        lang_text["home"],
        lang_text["chat"],
        lang_text["shops"],
        lang_text["doctors"],
        lang_text["contact"],
    ]

    cols = st.columns(5)

    for i, item in enumerate(menu_items):

        button_type = (
            "primary"
            if st.session_state.menu_choice == item
            else "secondary"
        )

        if cols[i].button(
            item,
            use_container_width=True,
            type=button_type,
        ):

            st.session_state.menu_choice = item
            st.rerun()

    menu = (
        st.session_state.menu_choice
    )

    if menu == lang_text["home"]:

        home_page(lang_text)

    elif menu == lang_text["chat"]:

        chat_page()

    elif menu == lang_text["shops"]:

        shop_or_doctors_page(
            "🛒 Fertilizer & Agriculture Services",
            "Shop",
            lang_text,
        )

    elif menu == lang_text["doctors"]:

        shop_or_doctors_page(
            "🩺 Agricultural Experts",
            "Doctors",
            lang_text,
        )

    else:

        contact_page()

    show_cost_report()

    show_reports_panel()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
