import base64
import io
import json
import os
import time
from datetime import datetime

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

# ================= API ================= #


def call_openrouter(messages, model):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"model": model, "messages": messages}
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return str(result)
    except Exception as err:
        return f"Agent API error: {err}"


# ================= TRANSLATIONS ================= #

translations = {
    "English": {
        "home": "Home",
        "chat": "Chat",
        "shops": "Shop",
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

FONT_BY_LANGUAGE = {
    "English": "Arial, sans-serif",
    "Hindi": "'Noto Sans Devanagari', sans-serif",
    "Marathi": "'Noto Sans Devanagari', sans-serif",
}

# ================= AGENT MODULES ================= #

AGENT_MODULES = {
    "Soil moisture modeling": "Estimate soil moisture trend from crop type, weather, and field signals.",
    "Water requirement prediction": "Predict daily and weekly irrigation water requirement.",
    "AI-driven irrigation schedule": "Generate optimal irrigation timing for next 7 days.",
    "Drought early warning": "Detect drought risk in upcoming 10 days and suggest safeguards.",
    "Water waste optimization %": "Estimate avoidable water waste and optimization percentage.",
    "NPK prediction": "Predict nitrogen, phosphorus, and potassium levels in current soil profile.",
    "pH imbalance detection": "Detect soil pH imbalance and likely causes.",
    "Nutrient deficiency fusion": "Use leaf + soil fusion logic to identify nutrient deficiency.",
    "Fertilizer recommendation engine": "Recommend fertilizer type, dose, and schedule.",
    "Long-term soil health score": "Compute seasonal soil health score with repair plan.",
    "Insect classification": "Identify likely pest class from symptoms and climate.",
    "Pest density estimation": "Estimate pest density and severity zones.",
    "Swarm detection": "Predict probability of swarm behavior and urgency.",
    "Migration pattern prediction": "Model wind-based pest migration pattern.",
    "Smart pesticide timing": "Suggest safe and efficient pesticide timing.",
    "Satellite imagery integration": "Integrate satellite cues for crop vigor and stress.",
    "Growth stage tracking": "Track crop growth stage and current milestone.",
    "Production estimate per acre": "Estimate expected production per acre.",
    "Profit forecast": "Forecast expected margin based on cost and yield.",
    "Market price integration": "Blend local market trends into selling strategy.",
    "Camera→Analyze→Recommend→Auto-execute": "Create vision-to-action workflow for farm operations.",
    "Irrigation valve control": "Generate smart valve control strategy.",
    "Sprayer control": "Generate precision sprayer control plan.",
    "Drone-based spraying": "Create drone route and spray planning logic.",
    "Automated farm reporting": "Build automated periodic farm status report.",
    "Multi-modal fusion model": "Fuse Vision + Weather + Soil + Time for better decision confidence.",
    "7–30 day disease risk prediction": "Predict medium-term disease risk from humidity and temperature.",
    "Frost risk alerts": "Detect frost alert windows and mitigation actions.",
    "Heat stress prediction": "Forecast heat stress and adaptive planning.",
    "Wind-based pest migration modeling": "Analyze wind direction impact on pest movement.",
    "Crop growth stage mapping": "Map field status to growth stages.",
    "Price prediction AI": "Estimate total crop cost vs local market selling gain.",
    "Full Agent Pipeline": "Run Vision, Climate, Soil, Water, Market, Execution layers in one pipeline.",
}


def ensure_session_defaults():
    defaults = {
        "language": "English",
        "logged_in": False,
        "chat_history": [],
        "agent_jobs": [],
        "agent_reports": [],
        "current_agent_action": "Idle",
        "theme": "Light",
        "settings_open": False,
        "settings_inner_open": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_language_font(language):
    font_family = FONT_BY_LANGUAGE.get(language, "Arial, sans-serif")
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            font-family: {font_family};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def enqueue_agent_job(title, context_text=""):
    st.session_state.agent_jobs.append(
        {
            "title": title,
            "context": context_text,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
        }
    )


def run_next_job_if_needed():
    running = [j for j in st.session_state.agent_jobs if j["status"] == "running"]
    if running:
        return

    queued = [j for j in st.session_state.agent_jobs if j["status"] == "queued"]
    if queued:
        next_job = queued[0]
        next_job["status"] = "running"
        next_job["started_at"] = time.time()
        st.session_state.current_agent_action = f"Running: {next_job['title']}"


def build_agent_prompt(job_title, job_context):
    return [
        {
            "role": "system",
            "content": (
                "You are an advanced agriculture AI agent with layers: Vision, Climate, Soil, Water, Market, "
                "Execution. Provide concise but actionable farming report with clear headings."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {job_title}\n"
                f"Focus: {AGENT_MODULES.get(job_title, '')}\n"
                f"Context: {job_context or 'No extra context provided by user.'}\n"
                "Include risk score, recommendation, and expected benefit."
            ),
        },
    ]


def process_background_jobs():
    run_next_job_if_needed()
    for job in st.session_state.agent_jobs:
        if job["status"] == "running" and job["started_at"] and (time.time() - job["started_at"] > 0.3):
            report = call_openrouter(build_agent_prompt(job["title"], job["context"]), REASONING_MODEL)
            job["status"] = "completed"
            st.session_state.agent_reports.insert(
                0,
                {
                    "title": job["title"],
                    "report": report,
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            st.session_state.current_agent_action = f"Completed: {job['title']}"
            break
    if not any(j["status"] == "running" for j in st.session_state.agent_jobs):
        run_next_job_if_needed()
        if not any(j["status"] == "running" for j in st.session_state.agent_jobs):
            st.session_state.current_agent_action = "Idle"


def _escape_pdf_text(text):
    return text.replace("\\", "\\\\").replace("(", "\(").replace(")", "\)")


def _write_simple_pdf(path, lines):
    content_cmds = ["BT", "/F1 12 Tf", "50 800 Td"]
    first = True
    for raw in lines:
        line = _escape_pdf_text(raw[:140])
        if first:
            content_cmds.append(f"({line}) Tj")
            first = False
        else:
            content_cmds.append("0 -16 Td")
            content_cmds.append(f"({line}) Tj")
    content_cmds.append("ET")
    stream = "\n".join(content_cmds).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append((f"5 0 obj << /Length {len(stream)} >> stream\n").encode("latin-1") + stream + b"\nendstream endobj\n")

    pdf = bytearray(b"%PDF-1.4\n")
    xref = [0]
    for obj in objects:
        xref.append(len(pdf))
        pdf.extend(obj)

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(xref)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in xref[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    trailer = f"trailer\n<< /Size {len(xref)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
    pdf.extend(trailer.encode("latin-1"))

    with open(path, "wb") as f:
        f.write(pdf)


def export_chat_to_pdf(username):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"chat_export_{username}_{int(time.time())}.pdf")

    lines = [f"Farm AI Chat Export - {username}", ""]
    if not st.session_state.chat_history:
        lines.append("No chat history available.")
    else:
        for item in st.session_state.chat_history:
            lines.append(f"User: {item['user']}")
            lines.append(f"AI: {item['assistant']}")
            lines.append("")

    _write_simple_pdf(path, lines)
    return path


# ================= APP ================= #

ensure_session_defaults()

if not os.path.exists(USER_DB):
    with open(USER_DB, "w", encoding="utf-8") as f:
        json.dump({}, f)

with open(USER_DB, "r", encoding="utf-8") as f:
    users = json.load(f)

apply_language_font(st.session_state.language)

if not st.session_state.logged_in:
    st.title(translations[st.session_state.language]["login"])
    username = st.text_input(translations[st.session_state.language]["username"])
    password = st.text_input(translations[st.session_state.language]["password"], type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()
        else:
            users[username] = password
            with open(USER_DB, "w", encoding="utf-8") as f:
                json.dump(users, f)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Account created")
            st.rerun()
    st.stop()

process_background_jobs()

# ================= SIDEBAR (LEFT PANEL) ================= #

with st.sidebar:
    st.subheader("🌍 Language")
    selected_lang = st.selectbox("Select", ["English", "Hindi", "Marathi"], index=["English", "Hindi", "Marathi"].index(st.session_state.language))
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

    st.markdown("---")
    st.subheader("🤖 Agent Controls")
    quick_context = st.text_area("Context for analysis", placeholder="Crop, location, issue...")

    if st.button("Run Analysis"):
        enqueue_agent_job("Full Agent Pipeline", quick_context)

    for module_name in AGENT_MODULES:
        if st.button(module_name, key=f"btn_{module_name}"):
            enqueue_agent_job(module_name, quick_context)

    st.markdown("---")
    st.subheader("📄 Generated Reports")
    if not st.session_state.agent_reports:
        st.caption("No report yet.")
    else:
        for idx, rep in enumerate(st.session_state.agent_reports[:12]):
            with st.expander(f"{rep['title']} ({rep['created']})", expanded=idx == 0):
                st.write(rep["report"])

    if st.button("Export Chat to PDF"):
        output_path = export_chat_to_pdf(st.session_state.get("username", "user"))
        st.success(f"PDF saved: {output_path}")

    st.markdown("---")
    st.subheader("👤 User")
    uname = st.session_state.get("username", "user")
    st.image(f"https://api.dicebear.com/7.x/initials/png?seed={uname}", width=72)
    st.write(uname)

    if st.button("Open profile box"):
        st.session_state.settings_open = not st.session_state.settings_open

    if st.session_state.settings_open:
        st.info("Profile actions")
        if st.button("Settings"):
            st.session_state.settings_inner_open = not st.session_state.settings_inner_open

    if st.session_state.settings_inner_open:
        st.selectbox("Theme", ["Light", "Dark", "Green"], key="theme")
        st.button("More")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

lang_text = translations[st.session_state.language]

st.title("🌾 AI Farm Agent")
st.caption(f"Agent status: {st.session_state.current_agent_action}")

menu = st.radio(
    "",
    [lang_text["home"], lang_text["chat"], lang_text["shops"], lang_text["doctors"], lang_text["contact"]],
    horizontal=True,
)

if menu == lang_text["home"]:
    st.header("Vision + Multi-Modal Crop Analysis")
    location = st.text_input("Farm Location")
    uploaded_image = st.file_uploader(lang_text["upload"], type=["jpg", "jpeg", "png"])

    if st.button(lang_text["analyze"]):
        if not uploaded_image:
            st.error("Upload image first.")
        else:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded leaf")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            desc = call_openrouter(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this leaf in detail and mention visible stress signs."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        ],
                    }
                ],
                VISION_MODEL,
            )

            fusion_report = call_openrouter(
                [
                    {"role": "system", "content": "Use multi-modal fusion model (Vision + Weather + Soil + Time)."},
                    {
                        "role": "user",
                        "content": (
                            f"Leaf vision summary: {desc}\n"
                            f"Location: {location}\n"
                            "Provide disease risk, growth stage mapping, and recommended next actions."
                        ),
                    },
                ],
                REASONING_MODEL,
            )
            st.subheader("Analysis result")
            st.write(fusion_report)

elif menu == lang_text["chat"]:
    st.header("💬 Agricultural AI Chat")
    user_query = st.text_input("Ask anything about your farm")
    if st.button("Send") and user_query.strip():
        response = call_openrouter(
            [
                {"role": "system", "content": "You are an agricultural assistant AI agent."},
                {"role": "user", "content": user_query},
            ],
            REASONING_MODEL,
        )
        st.session_state.chat_history.append({"user": user_query, "assistant": response})

    for item in reversed(st.session_state.chat_history[-20:]):
        st.markdown(f"**You:** {item['user']}")
        st.markdown(f"**Agent:** {item['assistant']}")
        st.markdown("---")

elif menu == lang_text["shops"]:
    st.header("🛒 Fertilizer Shop")
    crop = st.text_input("Crop name")
    requirement = st.text_input("Requirement")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Products"):
            result = call_openrouter(
                [
                    {"role": "system", "content": "You are fertilizer market analyst."},
                    {
                        "role": "user",
                        "content": f"Find best fertilizers for crop: {crop}, need: {requirement}. Include price and why useful.",
                    },
                ],
                REASONING_MODEL,
            )
            st.write(result)

    with col2:
        if st.button("Show All"):
            all_products = call_openrouter(
                [
                    {"role": "system", "content": "Provide a broad fertilizer catalog useful for Indian farming."},
                    {"role": "user", "content": "Show all major fertilizer options with NPK and price band."},
                ],
                REASONING_MODEL,
            )
            st.write(all_products)

elif menu == lang_text["doctors"]:
    st.header("🩺 Doctors")
    crop = st.text_input("Crop", key="doc_crop")
    issue = st.text_input("Issue", key="doc_issue")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Find Doctors"):
            doc_list = call_openrouter(
                [
                    {"role": "system", "content": "You are agri doctor directory assistant."},
                    {
                        "role": "user",
                        "content": f"Suggest agriculture experts for crop {crop} and issue {issue}. Include specialization and fee estimate.",
                    },
                ],
                REASONING_MODEL,
            )
            st.write(doc_list)
    with col2:
        if st.button("Show All", key="doc_show_all"):
            all_docs = call_openrouter(
                [
                    {"role": "system", "content": "You list agri doctor categories."},
                    {"role": "user", "content": "Show all agriculture doctor categories and support areas."},
                ],
                REASONING_MODEL,
            )
            st.write(all_docs)

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
