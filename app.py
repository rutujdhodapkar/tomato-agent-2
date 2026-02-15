import base64
import io
import json
import os
from datetime import datetime

import requests
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image

# ================= CONFIG ================= #
# API key intentionally kept in-code per requirement.
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
REASONING_MODEL = "deepseek/deepseek-r1-0528:free"
USER_DB = "users.json"
EXPORT_DIR = "exports"

# ================= LANGUAGE / FONT ================= #
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
FONT_MAP = {
    "English": "Arial, sans-serif",
    "Hindi": "'Nirmala UI', 'Mangal', sans-serif",
    "Marathi": "'Noto Sans Devanagari', 'Mangal', sans-serif",
}


def call_openrouter(messages, model=REASONING_MODEL):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages}
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return f"Model response: {data}"
    except Exception as exc:
        return f"Service unavailable, fallback used. Error: {exc}"


def ensure_session_defaults():
    defaults = {
        "language": "English",
        "theme": "Light",
        "logged_in": False,
        "username": "",
        "photo_url": "https://api.dicebear.com/8.x/adventurer/png?seed=Farmer",
        "agent_status": "Idle",
        "task_queue": [],
        "reports": [],
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


def queue_task(task_name, prompt, model=REASONING_MODEL):
    st.session_state.task_queue.append({"task": task_name, "prompt": prompt, "model": model})


def run_background_task_once():
    if not st.session_state.task_queue:
        if st.session_state.agent_status != "Idle":
            st.session_state.agent_status = "Idle"
        return

    task = st.session_state.task_queue.pop(0)
    st.session_state.agent_status = f"Running: {task['task']}"
    report = call_openrouter(
        [
            {
                "role": "system",
                "content": (
                    "You are an agricultural super-agent. Give practical steps, risk score, "
                    "timeline, and clear recommendation bullets."
                ),
            },
            {"role": "user", "content": task["prompt"]},
        ],
        task["model"],
    )
    st.session_state.reports.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": task["task"],
            "content": report,
        },
    )
    st.session_state.agent_status = f"Completed: {task['task']}"


def export_chat_to_pdf():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    lines = ["AI Agent Chat Export", ""]
    for msg in st.session_state.chat_history:
        lines.append(f"[{msg['time']}] {msg['role'].upper()}: {msg['text']}")

    page_lines = 35
    with PdfPages(path) as pdf:
        for i in range(0, max(len(lines), 1), page_lines):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor('white')
            text_chunk = "\n".join(lines[i:i + page_lines]) or "No chat messages to export."
            fig.text(0.05, 0.95, text_chunk, va='top', fontsize=9, family='sans-serif', wrap=True)
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

    return path


def login_block(lang_text):
    if not os.path.exists(USER_DB):
        with open(USER_DB, "w", encoding="utf-8") as file:
            json.dump({}, file)

    with open(USER_DB, "r", encoding="utf-8") as file:
        users = json.load(file)

    if st.session_state.logged_in:
        return

    st.title(lang_text["login"])
    username = st.text_input(lang_text["username"])
    password = st.text_input(lang_text["password"], type="password")

    if st.button("Continue"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()
        else:
            users[username] = password
            with open(USER_DB, "w", encoding="utf-8") as file:
                json.dump(users, file)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Account created")
            st.rerun()
    st.stop()


def sidebar_controls(lang_text):
    with st.sidebar:
        st.title("🤖 Agent Control Panel")

        st.session_state.language = st.selectbox("Language", list(TRANSLATIONS.keys()), index=list(TRANSLATIONS.keys()).index(st.session_state.language))
        st.session_state.theme = st.selectbox("Theme", ["Light", "Dark"])

        st.markdown("---")
        st.subheader("Quick Agent Actions")

        action_map = {
            "Soil moisture modeling": "Analyze soil moisture modeling with sensor + weather assumptions and actionable irrigation guidance.",
            "Water requirement prediction": "Predict farm water requirement for next 14 days by crop stage and weather uncertainty.",
            "AI-driven irrigation schedule": "Create AI-driven irrigation schedule with time windows and liters/acre.",
            "Drought early warning": "Generate drought early warning indicators for 30 days.",
            "Water waste optimization %": "Estimate current water waste percentage and optimization opportunities.",
            "NPK prediction": "Predict nitrogen, phosphorus, potassium levels and corrective plan.",
            "pH imbalance detection": "Detect pH imbalance and recommend treatment protocol.",
            "Nutrient deficiency fusion": "Use leaf + soil fusion assumptions to identify nutrient deficiencies.",
            "Fertilizer recommendation": "Build fertilizer recommendation engine output for this farm.",
            "Long-term soil health score": "Estimate long-term soil health score and yearly action plan.",
            "Insect classification": "Classify likely insects and risk level by season.",
            "Pest density estimation": "Estimate pest density per acre with intervention threshold.",
            "Swarm detection": "Detect swarm risk and alert plan.",
            "Migration pattern prediction": "Predict wind-based pest migration pattern over 7 days.",
            "Smart pesticide timing": "Recommend ideal pesticide application timing.",
            "Satellite imagery integration": "Provide satellite imagery integration plan and inferred crop signals.",
            "Growth stage tracking": "Track crop growth stage and next milestones.",
            "Production estimate per acre": "Estimate production per acre with confidence range.",
            "Profit forecast": "Generate profit forecast using yield, costs, and market price assumptions.",
            "Market price integration": "Integrate market price trend and suggest sell timing.",
            "Camera→Analyze→Recommend→Auto-execute": "Design camera-to-execution pipeline with automation gates.",
            "Irrigation valve control": "Generate irrigation valve control logic and failsafe.",
            "Sprayer control": "Generate smart sprayer control strategy.",
            "Drone-based spraying": "Plan drone-based spraying route and timing.",
            "Automated farm reporting": "Create automated farm reporting template and KPI plan.",
            "Multi-modal fusion model": "Design fusion model: Vision + Weather + Soil + Time.",
            "Disease risk 7-30 days": "Predict disease risk for 7-30 days using humidity + temperature.",
            "Frost risk alerts": "Predict frost risk and preventive actions.",
            "Heat stress prediction": "Predict heat stress windows and protection actions.",
            "Crop growth stage mapping": "Generate crop growth stage map from multimodal data.",
            "Price prediction AI": "Calculate total crop production cost and expected local market gain/profit.",
            "Full Agent Pipeline": "Build one proper end-to-end AI agent pipeline using Vision/Climate/Soil/Water/Market/Execution layers.",
        }

        selected_action = st.selectbox("Select analysis", list(action_map.keys()))
        if st.button("Run analysis"):
            queue_task(selected_action, action_map[selected_action])
            st.success(f"Queued: {selected_action}")

        if st.button("Run all core layers"):
            for layer_task in [
                "Vision Layer", "Climate Layer", "Soil Layer", "Water Layer", "Market Layer", "Execution Layer"
            ]:
                queue_task(layer_task, f"Generate operational report for {layer_task} with metrics and actions.")
            st.success("All layer analyses queued.")

        st.markdown("---")
        st.subheader("Chat Export")
        if st.button("Export chat as PDF"):
            pdf_path = export_chat_to_pdf()
            st.success(f"Saved: {pdf_path}")

        st.markdown("---")
        st.subheader("User")
        st.image(st.session_state.photo_url, width=70)
        st.write(st.session_state.username)

        with st.expander("Profile menu"):
            if st.button("Settings"):
                st.info("Theme / Logout / More available below")
            st.write("• Theme")
            st.write("• Logout")
            st.write("• More")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.rerun()


def home_page(lang_text):
    st.title("🌾 Agricultural Super AI Agent")
    location = st.text_input("Farm location")
    uploaded_image = st.file_uploader(lang_text["upload"], type=["jpg", "jpeg", "png"])

    if st.button(lang_text["analyze"]):
        if not uploaded_image:
            st.error("Upload an image first.")
            return
        image = Image.open(uploaded_image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        queue_task(
            "Leaf disease analysis",
            (
                f"Location: {location}. Analyze uploaded crop leaf image and provide diagnosis, confidence, treatment, "
                "and prevention plan."
            ),
        )

        vision_text = call_openrouter(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this crop leaf and possible stress signs."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                }
            ],
            model=VISION_MODEL,
        )
        st.info("Vision layer summary:")
        st.write(vision_text)
        st.success("Leaf pipeline queued. Agent continues processing in background.")


def chat_page():
    st.title("💬 Agent Chat")
    query = st.text_input("Ask about farming, costs, irrigation, market, disease...")
    if st.button("Send") and query.strip():
        st.session_state.chat_history.append(
            {"time": datetime.now().strftime("%H:%M:%S"), "role": "user", "text": query}
        )
        answer = call_openrouter(
            [
                {"role": "system", "content": "You are a practical agricultural AI agent."},
                {"role": "user", "content": query},
            ]
        )
        st.session_state.chat_history.append(
            {"time": datetime.now().strftime("%H:%M:%S"), "role": "assistant", "text": answer}
        )

    for msg in st.session_state.chat_history[-20:]:
        st.markdown(f"**{msg['role'].title()} ({msg['time']}):** {msg['text']}")


def shop_or_doctors_page(title, actor):
    st.title(title)
    crop = st.text_input(f"{actor}: Crop name")
    requirement = st.text_input(f"{actor}: Requirement")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Search {actor}"):
            queue_task(
                f"{actor} Search",
                f"For crop {crop}, requirement {requirement}, provide top options with price, reason, and availability.",
            )
            st.success(f"{actor} search queued.")
    with col2:
        if st.button("Show all"):
            queue_task(
                f"{actor} Show All",
                f"For crop {crop}, list all major {actor.lower()} options with pricing and usage summary.",
            )
            st.success("Show-all report queued.")


def contact_page():
    st.title("📞 Contact")
    st.markdown(
        """
        **AI Farm Agent Team**  
        Email: support@aifarmagent.local  
        Services: Vision, Climate, Soil, Water, Market, Execution  
        """
    )


def show_reports_panel():
    st.markdown("## 📊 Generated Reports")
    if not st.session_state.reports:
        st.info("No reports yet. Run analyses from the left panel.")
        return
    for report in st.session_state.reports[:12]:
        with st.expander(f"{report['time']} — {report['title']}"):
            st.write(report["content"])


def main():
    st.set_page_config(page_title="Agri Super Agent", layout="wide")
    ensure_session_defaults()
    apply_local_font(st.session_state.language)

    lang_text = TRANSLATIONS[st.session_state.language]
    login_block(lang_text)

    sidebar_controls(lang_text)
    apply_local_font(st.session_state.language)

    run_background_task_once()
    st.markdown(f"### 🧠 Agent Status: {st.session_state.agent_status}")

    menu = st.radio(
        "Navigation",
        [lang_text["home"], lang_text["chat"], lang_text["shops"], lang_text["doctors"], lang_text["contact"]],
        horizontal=True,
        label_visibility="collapsed",
    )

    if menu == lang_text["home"]:
        home_page(lang_text)
    elif menu == lang_text["chat"]:
        chat_page()
    elif menu == lang_text["shops"]:
        shop_or_doctors_page("🛒 Fertilizer Shop", "Shop")
    elif menu == lang_text["doctors"]:
        shop_or_doctors_page("🩺 Doctors", "Doctors")
    else:
        contact_page()

    show_reports_panel()


if __name__ == "__main__":
    main()
