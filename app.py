import base64
import io
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ================= CONFIG ================= #

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
REASONING_MODEL = "deepseek/deepseek-r1-0528:free"

USER_DB = "users.json"
EXPORT_DIR = "exports"

executor = ThreadPoolExecutor(max_workers=4)

# ================= API ================= #


def call_openrouter(messages, model):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"model": model, "messages": messages}
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    response = r.json()

    if "choices" in response:
        return response["choices"][0]["message"]["content"]

    return str(response)


# ================= AI AGENT TASKS ================= #


def run_agent_report(task_name, farm_context):
    prompts = {
        "Full Agent Pipeline": """
Create a single integrated AI farm-agent report with clear sections:
1) Vision Layer
2) Climate Layer
3) Soil Layer
4) Water Layer
5) Market Layer
6) Execution Layer

Must include:
- Multi-modal fusion model (Vision + Weather + Soil + Time)
- Camera → analyze → recommend → auto-execute pipeline
- Automated farm reporting summary
- Current top 3 actions for farmer
""",
        "Water Intelligence": """
Generate water intelligence report with:
- Soil moisture modeling
- Water requirement prediction
- AI-driven irrigation schedule
- Drought early warning
- Water waste optimization percentage
""",
        "Soil & Nutrient Intelligence": """
Generate soil + nutrient report including:
- Nitrogen, phosphorus, potassium prediction
- pH imbalance detection
- Nutrient deficiency via leaf + soil fusion model
- Fertilizer recommendation engine
- Long-term soil health score
""",
        "Pest & Disease Intelligence": """
Generate pest and disease intelligence with:
- Insect classification
- Pest density estimation
- Swarm detection
- Migration pattern prediction
- Smart pesticide timing
- 7–30 day disease risk prediction (humidity + temperature)
- Frost risk alerts
- Heat stress prediction
- Wind-based pest migration modeling
- Crop growth stage mapping
""",
        "Yield & Market Intelligence": """
Generate production and market report containing:
- Satellite imagery integration plan
- Growth stage tracking
- Production estimate per acre
- Profit forecast
- Market price integration
""",
        "Automation & Control": """
Generate automation execution plan with:
- Irrigation valve control
- Sprayer control
- Drone-based spraying
- Automated farm reporting
- Safety checks + fallback manual mode
""",
        "Price Prediction AI": """
Build a farm economics report with:
- Total cost estimation to grow crop (seed, water, fertilizer, labor, pesticide, transport)
- Local market sale price forecast
- Net gain and profitability scenarios (best/base/worst)
- Cost optimization suggestions
""",
    }

    system = (
        "You are an advanced agricultural AI agent. Be practical, concise, and provide a"
        " structured report with bullets, tables, and a short action plan."
    )
    user = f"""
Task: {task_name}
Farm context from user:
{farm_context}

Instructions:
{prompts[task_name]}
"""
    return call_openrouter(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        REASONING_MODEL,
    )


# ================= TRANSLATIONS ================= #

translations = {
    "English": {
        "home": "Home",
        "chat": "Chat",
        "shops": "Shops",
        "doctors": "Doctors",
        "contact": "Contact Us",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "upload": "Upload Leaf Image",
        "analyze": "Analyze Crop",
    },
    "Hindi": {
        "home": "होम",
        "chat": "चैट",
        "shops": "दुकान",
        "doctors": "डॉक्टर्स",
        "contact": "संपर्क करें",
        "login": "लॉगिन",
        "username": "उपयोगकर्ता नाम",
        "password": "पासवर्ड",
        "upload": "पत्ता अपलोड करें",
        "analyze": "विश्लेषण करें",
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
        "analyze": "विश्लेषण करा",
    },
}

LANG_FONT = {
    "English": "Arial, sans-serif",
    "Hindi": "'Noto Sans Devanagari', sans-serif",
    "Marathi": "'Noto Sans Devanagari', sans-serif",
}


# ================= HELPERS ================= #


def apply_font(language):
    font_family = LANG_FONT.get(language, "Arial, sans-serif")
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"], .stMarkdown, .stTextInput label, .stButton button {{
            font-family: {font_family};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_session_defaults():
    defaults = {
        "language": "English",
        "logged_in": False,
        "username": "",
        "user_photo": "https://i.pravatar.cc/120?img=12",
        "agent_status": "Idle",
        "chat_history": [],
        "agent_jobs": {},
        "agent_reports": {},
        "theme": "Light",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _simple_pdf(lines):
    def esc(text):
        return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    content_lines = ["BT", "/F1 12 Tf", "50 780 Td", "14 TL"]
    for line in lines:
        safe = esc(line.encode("latin-1", errors="replace").decode("latin-1"))
        content_lines.append(f"({safe}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n")

    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj

    xref_pos = len(pdf)
    pdf += f"xref\n0 {len(offsets)}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode("latin-1")
    pdf += f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin-1")
    return pdf


def export_chat_to_pdf(username):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"chat_export_{username}_{timestamp}.pdf")

    lines = [f"Chat Export for {username}", f"Generated: {datetime.now()}", ""]
    if not st.session_state.chat_history:
        lines.append("No chat messages available.")
    else:
        for idx, item in enumerate(st.session_state.chat_history, start=1):
            lines.append(f"{idx}. User: {item['user']}")
            lines.append(f"   Agent: {item['assistant']}")
            lines.append("")

    with open(path, "wb") as f:
        f.write(_simple_pdf(lines))
    return path


def queue_agent_task(task_name, farm_context):
    st.session_state.agent_status = f"Running: {task_name}"
    future = executor.submit(run_agent_report, task_name, farm_context)
    st.session_state.agent_jobs[task_name] = future


def refresh_jobs():
    finished = []
    for task_name, future in st.session_state.agent_jobs.items():
        if future.done():
            try:
                st.session_state.agent_reports[task_name] = future.result()
                st.session_state.agent_status = f"Completed: {task_name}"
            except Exception as e:
                st.session_state.agent_reports[task_name] = f"Error: {e}"
                st.session_state.agent_status = f"Failed: {task_name}"
            finished.append(task_name)
    for item in finished:
        del st.session_state.agent_jobs[item]


# ================= APP START ================= #

ensure_session_defaults()
refresh_jobs()
apply_font(st.session_state.language)

lang_text = translations[st.session_state.language]

# ================= LOGIN SYSTEM ================= #

if not os.path.exists(USER_DB):
    with open(USER_DB, "w", encoding="utf-8") as f:
        json.dump({}, f)

with open(USER_DB, "r", encoding="utf-8") as f:
    users = json.load(f)

if not st.session_state.logged_in:
    st.title(lang_text["login"])
    username = st.text_input(lang_text["username"])
    password = st.text_input(lang_text["password"], type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login Successful")
            st.rerun()
        else:
            users[username] = password
            with open(USER_DB, "w", encoding="utf-8") as f:
                json.dump(users, f)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Account Created")
            st.rerun()
    st.stop()

# ================= SIDEBAR ================= #

with st.sidebar:
    st.header("⚙️ Control Panel")
    selected_lang = st.selectbox("Language", ["English", "Hindi", "Marathi"], index=["English", "Hindi", "Marathi"].index(st.session_state.language))
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

    st.subheader("AI Agent Reports")
    farm_context = st.text_area(
        "Farm Context",
        value="Crop: Wheat\nArea: 5 acre\nSoil: Loamy\nIrrigation: Drip\nLocation: Maharashtra",
        height=120,
    )

    report_buttons = [
        "Full Agent Pipeline",
        "Water Intelligence",
        "Soil & Nutrient Intelligence",
        "Pest & Disease Intelligence",
        "Yield & Market Intelligence",
        "Automation & Control",
        "Price Prediction AI",
    ]

    for label in report_buttons:
        if st.button(f"Run {label}", use_container_width=True):
            if label not in st.session_state.agent_jobs:
                queue_agent_task(label, farm_context)
                st.success(f"Started: {label}")

    if st.button("Export Chat to PDF", use_container_width=True):
        file_path = export_chat_to_pdf(st.session_state.username)
        st.success(f"PDF exported: {file_path}")

    st.markdown("---")
    st.image(st.session_state.user_photo, width=70)
    st.caption(f"User: {st.session_state.username}")

    with st.popover("User Menu"):
        st.write("Settings")
        with st.expander("Open Settings"):
            theme = st.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
            st.session_state.theme = theme
            st.button("More")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

# ================= TOP AGENT TITLE ================= #

st.title(f"🤖 Agent Status: {st.session_state.agent_status}")

if st.session_state.agent_jobs:
    with st.container(border=True):
        st.write("Running in background:")
        for running_task in st.session_state.agent_jobs:
            st.write(f"• {running_task}")

# ================= NAVIGATION ================= #

menu = st.radio(
    "",
    [
        lang_text["home"],
        lang_text["chat"],
        lang_text["shops"],
        lang_text["doctors"],
        lang_text["contact"],
    ],
    horizontal=True,
)

# ================= HOME ================= #

if menu == lang_text["home"]:
    st.header("🌾 Agricultural Intelligence")

    location = st.text_input("Farm Location")
    uploaded_image = st.file_uploader(lang_text["upload"], type=["jpg", "png"])

    if st.button(lang_text["analyze"]):
        if not uploaded_image:
            st.error("Upload image.")
            st.stop()

        st.session_state.agent_status = "Running: Leaf Analysis"

        image = Image.open(uploaded_image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        desc = call_openrouter(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this leaf in detail."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
            VISION_MODEL,
        )

        diagnosis = call_openrouter(
            [
                {"role": "system", "content": "You are plant pathologist."},
                {
                    "role": "user",
                    "content": f"Based on: {desc}. Identify crop, disease, severity, and recommendation for location {location}.",
                },
            ],
            REASONING_MODEL,
        )

        st.markdown("## Result")
        st.write(diagnosis)
        st.session_state.agent_status = "Completed: Leaf Analysis"

    if st.session_state.agent_reports:
        st.subheader("Generated Reports")
        selected_report = st.selectbox("View report", list(st.session_state.agent_reports.keys()))
        st.write(st.session_state.agent_reports[selected_report])

# ================= CHAT ================= #

elif menu == lang_text["chat"]:
    st.header("💬 AI Chat")
    user_query = st.text_input("Ask anything about agriculture")

    if st.button("Send") and user_query.strip():
        st.session_state.agent_status = "Running: Chat Analysis"
        response = call_openrouter(
            [
                {"role": "system", "content": "You are agricultural assistant."},
                {"role": "user", "content": user_query},
            ],
            REASONING_MODEL,
        )
        st.session_state.chat_history.append({"user": user_query, "assistant": response})
        st.session_state.agent_status = "Completed: Chat Analysis"

    for item in reversed(st.session_state.chat_history[-10:]):
        st.markdown(f"**You:** {item['user']}")
        st.markdown(f"**Agent:** {item['assistant']}")
        st.markdown("---")

# ================= SHOPS ================= #

elif menu == lang_text["shops"]:
    st.header("🛒 Fertilizer Shop Search")

    fertilizers = pd.DataFrame(
        [
            {"Product": "Urea Plus", "NPK": "46-0-0", "Price": "₹290", "Use": "Nitrogen boost"},
            {"Product": "DAP Gold", "NPK": "18-46-0", "Price": "₹1350", "Use": "Root growth"},
            {"Product": "NPK Balance", "NPK": "20-20-20", "Price": "₹900", "Use": "General purpose"},
        ]
    )

    crop = st.text_input("Crop Name")
    req = st.text_input("Specific Requirement")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Products"):
            result = call_openrouter(
                [
                    {"role": "system", "content": "You are fertilizer market analyst."},
                    {
                        "role": "user",
                        "content": f"Find best fertilizers for crop: {crop}. Requirement: {req}. Include product, NPK, price, reason.",
                    },
                ],
                REASONING_MODEL,
            )
            st.write(result)
    with col2:
        if st.button("Show All"):
            st.dataframe(fertilizers, use_container_width=True)

# ================= DOCTORS ================= #

elif menu == lang_text["doctors"]:
    st.header("🩺 Agri Doctors")

    doctors = pd.DataFrame(
        [
            {"Name": "Dr. Mehta", "Specialization": "Soil & Nutrients", "Contact": "+91-900000001"},
            {"Name": "Dr. Patil", "Specialization": "Plant Disease", "Contact": "+91-900000002"},
            {"Name": "Dr. Khan", "Specialization": "Irrigation & Water", "Contact": "+91-900000003"},
        ]
    )

    issue = st.text_input("Describe your issue")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Find Doctors"):
            result = call_openrouter(
                [
                    {"role": "system", "content": "You are an agriculture health triage assistant."},
                    {
                        "role": "user",
                        "content": f"Farmer issue: {issue}. Suggest specialist doctor type, urgency, and first actions.",
                    },
                ],
                REASONING_MODEL,
            )
            st.write(result)
    with c2:
        if st.button("Show All Doctors"):
            st.dataframe(doctors, use_container_width=True)

# ================= CONTACT ================= #

elif menu == lang_text["contact"]:
    st.header("📞 Contact Us")

    st.markdown(
        """
    **Name:** Rutuj Dhodapkar  
    **Email:** rutujdhodapkar@gmail.com  
    **Username:** rutujdhodapkar  
    **Portfolio:** https://rutujdhodapkar.vercel.app/  
    **Specialization:** Advanced AI, Deep Learning, Machine Learning, Big Data  
    **Location:** Los Angeles  
    """
    )
