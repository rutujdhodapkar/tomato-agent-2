import base64
import io
import json
import os
import random
import re
import threading
import time
from datetime import datetime

import requests
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Agri Super Agent",
    layout="wide",
)

USER_DB = "users.json"
EXPORT_DIR = "exports"

REQUEST_LOCK = threading.Lock()

TEMPORARY_STATUS_CODES = {429, 500, 502, 503, 504}


def get_config(name, default=""):
    """
    Reads configuration from Streamlit secrets first,
    then environment variables.
    """
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


PRIMARY_CONFIG = {
    "api_key": get_config("AI_API_KEY"),
    "api_url": get_config("AI_API_URL"),
    "model": get_config("AI_MODEL"),
}

FALLBACK_CONFIG = {
    "api_key": get_config("AI_FALLBACK_API_KEY"),
    "api_url": get_config("AI_FALLBACK_API_URL"),
    "model": get_config("AI_FALLBACK_MODEL"),
}

TRANSLATE_API_KEY = get_config("TRANSLATE_API_KEY")
TRANSLATE_API_URL = get_config("TRANSLATE_API_URL")


# ============================================================
# LANGUAGE
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
    },
}

LANGUAGE_CODE_MAP = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
}


# ============================================================
# ACTIONS
# ============================================================

ACTION_MAP = {
    "Soil moisture modeling":
        "Analyze soil moisture modeling and provide actionable irrigation guidance.",

    "Water requirement prediction":
        "Predict farm water requirements for the next 14 days.",

    "AI-driven irrigation schedule":
        "Create an irrigation schedule with time windows and water quantities.",

    "Drought early warning":
        "Generate drought early warning indicators for the next 30 days.",

    "NPK prediction":
        "Estimate nitrogen, phosphorus, and potassium status and recommend corrective actions.",

    "pH imbalance detection":
        "Analyze possible soil pH imbalance and recommend a treatment protocol.",

    "Nutrient deficiency fusion":
        "Identify likely nutrient deficiencies using available crop and soil information.",

    "Fertilizer recommendation":
        "Generate a fertilizer recommendation plan for the farm.",

    "Disease risk 7-30 days":
        "Estimate disease risk for the next 7 to 30 days.",

    "Growth stage tracking":
        "Analyze crop growth stage and identify upcoming milestones.",

    "Production estimate per acre":
        "Estimate crop production per acre with uncertainty and key assumptions.",

    "Profit forecast":
        "Generate a practical profit forecast using production, cost, and market assumptions.",

    "Camera to recommendation pipeline":
        "Design an end-to-end camera, analysis, recommendation, and automation pipeline.",

    "Automated farm reporting":
        "Create an automated farm reporting structure with KPIs and recommended actions.",

    "Multi-modal fusion model":
        "Design a multimodal agricultural AI system using image, weather, soil, and time data.",

    "Full Agent Pipeline":
        "Build a complete end-to-end agricultural intelligence pipeline.",
}


# ============================================================
# SESSION STATE
# ============================================================

def ensure_session_defaults():

    defaults = {
        "language": "English",
        "logged_in": False,
        "username": "",
        "menu_choice": "Home",
        "location": "",
        "chat_history": [],
        "task_queue": [],
        "reports": [],
        "agent_status": "Idle",
        "detection_result": None,
        "cost_estimation": None,
        "processing_tasks": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# SAFE JSON PARSER
# ============================================================

def parse_json_response(text):

    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        raise ValueError("Response is not text.")

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

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if match:
        return json.loads(match.group())

    raise ValueError("Could not find valid JSON in response.")


# ============================================================
# GENERIC AI REQUEST LAYER
# ============================================================

def send_request(messages, config, max_tokens=1200):

    api_key = config.get("api_key", "")
    api_url = config.get("api_url", "")
    model = config.get("model", "")

    if not api_key:
        return None, "Missing API key."

    if not api_url:
        return None, "Missing API URL."

    if not model:
        return None, "Missing model name."

    headers = {
        "Authorization": f"Bearer {api_key}",
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

        # One request at a time per app process.
        with REQUEST_LOCK:

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=180,
            )

        return response, None

    except requests.exceptions.Timeout:
        return None, "Request timed out."

    except requests.exceptions.RequestException as error:
        return None, f"Network error: {str(error)}"


def extract_response_content(response):

    try:
        data = response.json()
    except ValueError:
        return None, f"Non-JSON response: {response.text[:500]}"

    try:
        content = (
            data["choices"][0]
            ["message"]
            ["content"]
        )

        if isinstance(content, str):
            return content, None

        return json.dumps(content), None

    except (KeyError, IndexError, TypeError):

        error_message = (
            data.get("error", {})
            .get("message", "")
        )

        if error_message:
            return None, error_message

        return None, f"Unexpected response format: {str(data)[:500]}"


def call_ai(
    messages,
    max_tokens=1200,
    retries_per_provider=2,
):
    """
    Generic AI call.

    Flow:
    Primary -> retry
    Primary -> retry
    Fallback -> retry
    Fallback -> retry
    """

    providers = [PRIMARY_CONFIG]

    if (
        FALLBACK_CONFIG["api_key"]
        and FALLBACK_CONFIG["api_url"]
        and FALLBACK_CONFIG["model"]
    ):
        providers.append(FALLBACK_CONFIG)

    last_error = "Unknown error."

    for provider_index, provider in enumerate(providers):

        for attempt in range(retries_per_provider):

            response, network_error = send_request(
                messages=messages,
                config=provider,
                max_tokens=max_tokens,
            )

            if network_error:
                last_error = network_error

            elif response.status_code == 200:

                content, extraction_error = (
                    extract_response_content(response)
                )

                if not extraction_error:
                    return content

                last_error = extraction_error

            elif response.status_code in (401, 403):

                # Authentication failures should not retry forever.
                last_error = (
                    f"Authorization failed "
                    f"({response.status_code}): "
                    f"{response.text[:300]}"
                )

                break

            elif response.status_code in TEMPORARY_STATUS_CODES:

                last_error = (
                    f"Temporary service error "
                    f"{response.status_code}: "
                    f"{response.text[:300]}"
                )

                error_text = response.text.lower()

                # Resource exhaustion deserves a longer cooldown.
                if (
                    "resourceexhausted" in error_text
                    or "request limit reached" in error_text
                ):
                    wait_time = random.uniform(8, 15)
                else:
                    wait_time = min(
                        2 ** (attempt + 1)
                        + random.uniform(0.5, 2),
                        20,
                    )

                if attempt < retries_per_provider - 1:
                    time.sleep(wait_time)

            else:

                last_error = (
                    f"HTTP Error {response.status_code}: "
                    f"{response.text[:500]}"
                )

                break

        # Small delay before provider fallback.
        if provider_index < len(providers) - 1:
            time.sleep(random.uniform(1, 3))

    return f"AI_ERROR: {last_error}"


# ============================================================
# VISION ANALYSIS
# ============================================================

def run_image_analysis(image_bytes, species_info):

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    prompt = f"""
Analyze this agricultural plant image.

Farm metadata:
{json.dumps(species_info, ensure_ascii=False)}

Identify the likely crop and its health condition.

Return ONLY valid JSON:

{{
    "crop_name": "Name of crop",
    "disease_name": "Disease name or Healthy",
    "description": "Short condition assessment",
    "solution": "Step-by-step actions",
    "fertilizers": "Recommended nutrients or fertilizers",
    "soil_insights": "Soil assumptions and recommendations",
    "water_forecast": "Irrigation guidance",
    "risk_score": "Low, Medium, or High"
}}

Important:
- Do not invent exact sensor readings.
- Clearly treat unavailable environmental data as assumptions.
- Do not output markdown.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an agricultural vision and crop health assistant. "
                "Be practical, conservative, and explicit about uncertainty."
            ),
        },
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
                            f"{encoded_image}"
                        )
                    },
                },
            ],
        },
    ]

    output = call_ai(
        messages,
        max_tokens=1500,
        retries_per_provider=2,
    )

    if output.startswith("AI_ERROR:"):
        return {
            "error": output
        }

    try:
        return parse_json_response(output)

    except Exception as error:
        return {
            "error": (
                f"Could not parse image analysis: {str(error)}"
            ),
            "raw_response": output[:3000],
        }


# ============================================================
# TRANSLATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    ttl=3600,
)
def translate_text(text, language):

    if language == "English":
        return text

    if not isinstance(text, str):
        return text

    if not text.strip():
        return text

    # Translation is optional.
    if not TRANSLATE_API_KEY or not TRANSLATE_API_URL:
        return text

    payload = {
        "source_language_code": "en-IN",
        "target_language_code":
            LANGUAGE_CODE_MAP.get(
                language,
                "en-IN",
            ),
        "input": text,
    }

    headers = {
        "api-subscription-key":
            TRANSLATE_API_KEY,
        "Content-Type":
            "application/json",
    }

    try:

        response = requests.post(
            TRANSLATE_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        return (
            data.get("translated_text")
            or data.get("translation")
            or data.get("output")
            or text
        )

    except Exception:
        return text


def t(text):
    return translate_text(
        text,
        st.session_state.language,
    )


def translate_result_data(data, language):

    if language == "English":
        return data

    if isinstance(data, dict):
        return {
            key: translate_result_data(
                value,
                language,
            )
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            translate_result_data(
                item,
                language,
            )
            for item in data
        ]

    if isinstance(data, str):
        return translate_text(
            data,
            language,
        )

    return data


# ============================================================
# TASK QUEUE
# ============================================================

def queue_task(task_name, prompt):

    # Prevent duplicate tasks.
    existing = [
        task["task"]
        for task in st.session_state.task_queue
    ]

    if task_name not in existing:
        st.session_state.task_queue.append(
            {
                "task": task_name,
                "prompt": prompt,
            }
        )


def run_next_task():

    if not st.session_state.task_queue:
        st.session_state.agent_status = "Idle"
        return False

    task = st.session_state.task_queue.pop(0)

    st.session_state.agent_status = (
        f"Running: {task['task']}"
    )

    report = call_ai(
        [
            {
                "role": "system",
                "content": (
                    "You are an agricultural intelligence agent. "
                    "Return a concise structured operational report "
                    "with assumptions, risks, actions, and expected impact."
                ),
            },
            {
                "role": "user",
                "content": task["prompt"],
            },
        ],
        max_tokens=1200,
        retries_per_provider=2,
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

    if st.session_state.task_queue:
        st.session_state.agent_status = (
            f"{len(st.session_state.task_queue)} tasks remaining"
        )
    else:
        st.session_state.agent_status = (
            "All tasks completed"
        )

    return True


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
        "chat_export_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".pdf",
    )

    document = SimpleDocTemplate(
        path,
        pagesize=A4,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "AI Agent Chat Export",
            styles["Title"],
        ),
        Spacer(1, 20),
    ]

    if not st.session_state.chat_history:

        story.append(
            Paragraph(
                "No chat messages available.",
                styles["BodyText"],
            )
        )

    for message in st.session_state.chat_history:

        safe_text = (
            str(message["text"])
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        story.append(
            Paragraph(
                (
                    f"<b>[{message['time']}] "
                    f"{message['role'].upper()}:</b><br/>"
                    f"{safe_text}"
                ),
                styles["BodyText"],
            )
        )

        story.append(
            Spacer(1, 10)
        )

    document.build(story)

    return path


# ============================================================
# LOGIN
# ============================================================

def login_block(lang_text):

    if st.session_state.logged_in:
        return

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

    except Exception:
        users = {}

    st.title(lang_text["login"])

    username = st.text_input(
        lang_text["username"]
    )

    password = st.text_input(
        lang_text["password"],
        type="password",
    )

    if st.button("Continue"):

        if not username.strip():
            st.error(
                "Username cannot be empty."
            )
            st.stop()

        if not password:
            st.error(
                "Password cannot be empty."
            )
            st.stop()

        # NOTE:
        # For production, use password hashing.
        if (
            username in users
            and users[username] == password
        ):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful.")

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
                )

            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(
                "Account created."
            )

        st.rerun()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

def sidebar_controls(lang_text):

    with st.sidebar:

        st.title(
            "Agent Control Panel"
        )

        languages = list(
            TRANSLATIONS.keys()
        )

        current_index = languages.index(
            st.session_state.language
        )

        new_language = st.selectbox(
            "Select Language",
            languages,
            index=current_index,
        )

        if st.button(
            "Apply Language"
        ):
            st.session_state.language = (
                new_language
            )
            st.rerun()

        st.divider()

        st.subheader(
            "Quick Agent Actions"
        )

        selected_action = st.selectbox(
            "Select analysis",
            list(ACTION_MAP.keys()),
        )

        if st.button(
            "Queue Analysis",
            use_container_width=True,
        ):

            queue_task(
                selected_action,
                ACTION_MAP[
                    selected_action
                ],
            )

            st.success(
                "Task added to queue."
            )

        if st.button(
            "Queue Core Analysis",
            use_container_width=True,
        ):

            core_actions = [
                "Soil moisture modeling",
                "Water requirement prediction",
                "NPK prediction",
                "Disease risk 7-30 days",
                "Production estimate per acre",
                "Profit forecast",
                "Full Agent Pipeline",
            ]

            for action in core_actions:

                queue_task(
                    action,
                    ACTION_MAP[action],
                )

            st.success(
                f"{len(core_actions)} tasks queued."
            )

        # IMPORTANT:
        # One task per click.
        # No 30-request API avalanche.
        if (
            st.session_state.task_queue
            and st.button(
                "Run Next Task",
                use_container_width=True,
            )
        ):

            with st.spinner(
                "Running one task..."
            ):
                run_next_task()

            st.rerun()

        st.divider()

        st.subheader(
            "Cost Estimation"
        )

        location = st.text_input(
            "Location"
        )

        crop = st.text_input(
            "Crop name"
        )

        acres = st.number_input(
            "Total acres",
            min_value=0.0,
            step=0.1,
        )

        invested = st.number_input(
            "Total invested",
            min_value=0.0,
            step=100.0,
        )

        if st.button(
            "Estimate Cost & Profit",
            use_container_width=True,
        ):

            if (
                not location
                or not crop
                or acres <= 0
            ):
                st.error(
                    "Fill location, crop, and acres."
                )

            else:

                prompt = f"""
Analyze the economics of this farm.

Location: {location}
Crop: {crop}
Acres: {acres}
Investment: {invested}

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

Do not claim live market data unless it was actually supplied.
State assumptions clearly.
"""

                with st.spinner(
                    "Calculating..."
                ):

                    result = call_ai(
                        [
                            {
                                "role": "system",
                                "content":
                                    "You are a conservative agricultural economic analyst.",
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        max_tokens=1200,
                    )

                st.session_state.cost_estimation = (
                    result
                )

        st.divider()

        st.subheader(
            "Chat Export"
        )

        if st.button(
            "Export Chat as PDF",
            use_container_width=True,
        ):

            path = export_chat_to_pdf()

            with open(
                path,
                "rb",
            ) as file:

                st.download_button(
                    "Download PDF",
                    data=file.read(),
                    file_name=os.path.basename(
                        path
                    ),
                    mime="application/pdf",
                )

        st.divider()

        st.write(
            f"User: {st.session_state.username}"
        )

        if st.button(
            "Logout",
            use_container_width=True,
        ):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()


# ============================================================
# HOME PAGE
# ============================================================

def home_page(lang_text):

    st.title(
        "Agricultural Super AI Agent"
    )

    st.session_state.location = (
        st.text_input(
            "Farm location",
            value=st.session_state.location,
        )
    )

    uploaded_image = st.file_uploader(
        lang_text["upload"],
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded_image:

        image = Image.open(
            uploaded_image
        )

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True,
        )

        if st.button(
            lang_text["analyze"],
            use_container_width=True,
        ):

            buffer = io.BytesIO()

            image.convert(
                "RGB"
            ).save(
                buffer,
                format="JPEG",
                quality=90,
            )

            image_bytes = (
                buffer.getvalue()
            )

            species_info = {
                "location":
                    st.session_state.location
                    or "Not provided"
            }

            with st.spinner(
                "Analyzing crop image..."
            ):

                result = (
                    run_image_analysis(
                        image_bytes,
                        species_info,
                    )
                )

            st.session_state.detection_result = (
                result
            )

    result = (
        st.session_state.detection_result
    )

    if not result:
        return

    if "error" in result:

        st.error(
            result["error"]
        )

        if result.get(
            "raw_response"
        ):
            with st.expander(
                "Debug response"
            ):
                st.code(
                    result[
                        "raw_response"
                    ]
                )

        return

    result = translate_result_data(
        result,
        st.session_state.language,
    )

    st.divider()

    st.header(
        "Full Analysis Report"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Crop Identified",
            result.get(
                "crop_name",
                "Unknown",
            ),
        )

    with col2:

        st.metric(
            "Disease Status",
            result.get(
                "disease_name",
                "Unknown",
            ),
        )

    st.subheader(
        "Condition Assessment"
    )

    st.write(
        result.get(
            "description",
            "No description available.",
        )
    )

    st.subheader(
        "Actionable Prescription"
    )

    st.write(
        result.get(
            "solution",
            "No solution provided.",
        )
    )

    st.subheader(
        "Soil Insights"
    )

    st.write(
        result.get(
            "soil_insights",
            "No soil insights available.",
        )
    )

    st.subheader(
        "Water and Irrigation"
    )

    st.write(
        result.get(
            "water_forecast",
            "No water forecast available.",
        )
    )

    risk = result.get(
        "risk_score",
        "Unknown",
    )

    st.subheader(
        "Risk Level"
    )

    if risk.lower() == "high":
        st.error(risk)

    elif risk.lower() == "medium":
        st.warning(risk)

    else:
        st.success(risk)

    st.subheader(
        "Fertilizer Recommendations"
    )

    st.write(
        result.get(
            "fertilizers",
            "No fertilizer recommendations available.",
        )
    )


# ============================================================
# CHAT PAGE
# ============================================================

def chat_page():

    st.title("Agent Chat")

    for message in (
        st.session_state.chat_history
    ):

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
        "Ask about farming, irrigation, diseases, costs..."
    )

    if query:

        st.session_state.chat_history.append(
            {
                "time":
                    datetime.now().strftime(
                        "%H:%M:%S"
                    ),
                "role": "user",
                "text": query,
            }
        )

        with st.spinner(
            "Thinking..."
        ):

            answer = call_ai(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a practical agricultural AI assistant. "
                            "Do not invent live data. "
                            "State uncertainty clearly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                max_tokens=1000,
            )

        st.session_state.chat_history.append(
            {
                "time":
                    datetime.now().strftime(
                        "%H:%M:%S"
                    ),
                "role": "assistant",
                "text": answer,
            }
        )

        st.rerun()


# ============================================================
# SHOP / DOCTORS
# ============================================================

def services_page(title, actor):

    st.title(title)

    crop = st.text_input(
        "Crop name"
    )

    requirement = st.text_input(
        "Requirement"
    )

    if st.button(
        f"Get {actor} Recommendations"
    ):

        prompt = f"""
Provide useful recommendations for agricultural {actor.lower()} services.

Crop: {crop}
Requirement: {requirement}
Farm location: {st.session_state.location}

Do not invent real business names, phone numbers, addresses, or live availability.
If exact local information is unavailable, provide categories and selection criteria.
"""

        with st.spinner(
            "Analyzing options..."
        ):

            response = call_ai(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=1000,
            )

        st.write(response)


# ============================================================
# CONTACT
# ============================================================

def contact_page():

    st.title("Contact")

    st.write(
        "AI Farm Agent"
    )

    st.write(
        "Services: Vision, Crop Health, Soil, Water, Market Analysis, and Automation"
    )


# ============================================================
# REPORTS
# ============================================================

def show_reports_panel():

    st.divider()

    st.header(
        "Generated Reports"
    )

    reports = (
        st.session_state.reports
    )

    if not reports:

        st.info(
            "No reports yet."
        )

        return

    for report in reports[:12]:

        with st.expander(
            f"{report['time']} — {report['title']}"
        ):

            content = (
                report["content"]
            )

            if content.startswith(
                "AI_ERROR:"
            ):
                st.error(content)
            else:
                st.write(content)


# ============================================================
# COST REPORT
# ============================================================

def show_cost_report():

    estimation = (
        st.session_state.cost_estimation
    )

    if not estimation:
        return

    st.divider()

    st.header(
        "Cost and Profit Estimation"
    )

    if estimation.startswith(
        "AI_ERROR:"
    ):
        st.error(estimation)
        return

    try:

        data = parse_json_response(
            estimation
        )

    except Exception:

        st.warning(
            "The response was not valid JSON."
        )

        st.write(estimation)

        return

    for label, key in [

        (
            "Market Price",
            "market_price",
        ),

        (
            "Price Trend",
            "price_trend",
        ),

        (
            "Best Months",
            "best_months",
        ),

        (
            "Total Cost",
            "total_cost",
        ),

        (
            "Expected Revenue",
            "expected_revenue",
        ),

        (
            "Profit or Loss",
            "profit_or_loss",
        ),

        (
            "Travel Costs",
            "travel_costs",
        ),

        (
            "Recommendation",
            "recommendation",
        ),
    ]:

        st.subheader(label)

        value = data.get(
            key,
            "Not available",
        )

        if isinstance(
            value,
            list,
        ):
            st.write(
                ", ".join(
                    map(str, value)
                )
            )
        else:
            st.write(value)


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_session_defaults()

    lang_text = TRANSLATIONS[
        st.session_state.language
    ]

    login_block(
        lang_text
    )

    sidebar_controls(
        lang_text
    )

    st.caption(
        f"Agent Status: "
        f"{st.session_state.agent_status}"
    )

    cols = st.columns(5)

    menu_items = [
        "Home",
        "Chat",
        "Shop",
        "Doctors",
        "Contact",
    ]

    for index, item in enumerate(
        menu_items
    ):

        button_type = (
            "primary"
            if st.session_state.menu_choice
            == item
            else "secondary"
        )

        if cols[index].button(
            item,
            use_container_width=True,
            type=button_type,
        ):

            st.session_state.menu_choice = (
                item
            )

            st.rerun()

    menu = (
        st.session_state.menu_choice
    )

    if menu == "Home":
        home_page(lang_text)

    elif menu == "Chat":
        chat_page()

    elif menu == "Shop":
        services_page(
            "Fertilizer Shop",
            "Shop",
        )

    elif menu == "Doctors":
        services_page(
            "Agricultural Doctors",
            "Doctors",
        )

    elif menu == "Contact":
        contact_page()

    show_cost_report()

    show_reports_panel()


if __name__ == "__main__":
    main()
