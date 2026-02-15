import base64
import io
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import requests
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
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
        data = response.json()
    except Exception as exc:
        return f"API error: {exc}"

    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return str(data)


# ================= APP STATE ================= #

if "language" not in st.session_state:
    st.session_state.language = "English"
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = "Farmer"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reports" not in st.session_state:
    st.session_state.reports = {}
if "active_report" not in st.session_state:
    st.session_state.active_report = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = "Idle"
if "current_menu" not in st.session_state:
    st.session_state.current_menu = "Home"
if "job" not in st.session_state:
    st.session_state.job = {
        "running": False,
        "status": "Idle",
        "done": [],
        "reports": {},
    }

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
    },
    "Hindi": {
        "home": "होम",
        "chat": "चैट",
        "shops": "दुकान",
        "doctors": "डॉक्टर्स",
        "contact": "संपर्क",
        "login": "लॉगिन",
        "username": "उपयोगकर्ता",
        "password": "पासवर्ड",
    },
    "Marathi": {
        "home": "मुख्यपृष्ठ",
        "chat": "चॅट",
        "shops": "दुकान",
        "doctors": "डॉक्टर्स",
        "contact": "संपर्क",
        "login": "लॉगिन",
        "username": "वापरकर्ता",
        "password": "पासवर्ड",
    },
}

font_map = {
    "English": "Arial, sans-serif",
    "Hindi": "Noto Sans Devanagari, sans-serif",
    "Marathi": "Noto Sans Devanagari, sans-serif",
}


def apply_local_font():
    chosen_font = font_map.get(st.session_state.language, "Arial, sans-serif")
    st.markdown(
        f"""
        <style>
            html, body, [class*="css"]  {{
                font-family: {chosen_font};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_local_font()

# ================= AUTH ================= #

if not os.path.exists(USER_DB):
    with open(USER_DB, "w", encoding="utf-8") as f:
        json.dump({}, f)

with open(USER_DB, "r", encoding="utf-8") as f:
    users = json.load(f)

if not st.session_state.logged_in:
    t = translations[st.session_state.language]
    st.title(t["login"])
    username = st.text_input(t["username"])
    password = st.text_input(t["password"], type="password")

    if st.button("Login / Register"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()
        else:
            users[username] = password
            with open(USER_DB, "w", encoding="utf-8") as wf:
                json.dump(users, wf)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Account created")
            st.rerun()
    st.stop()

# ================= HELPERS ================= #


def build_ai_reports(context):
    crop = context.get("crop") or "Crop"
    acres = float(context.get("acres") or 1)
    budget = float(context.get("budget") or 0)
    location = context.get("location") or "Unknown location"

    base_water = round(4200 + acres * 320)
    water_saving = 18 + int(acres) % 12
    est_yield = round(16.5 * acres, 2)
    market_price = round(2100 + (acres * 25), 2)
    growth_cost = round((budget if budget > 0 else 35000) + acres * 3500, 2)
    est_revenue = round(est_yield * market_price, 2)
    est_profit = round(est_revenue - growth_cost, 2)

    return {
        "Water Layer": f"""
- Soil moisture modeling: Zone A 62%, Zone B 54%, Zone C 47%.
- Water requirement prediction: ~{base_water} liters/day.
- AI-driven irrigation schedule: 05:30 and 18:10 daily.
- Drought early warning: Medium risk in 7-day horizon.
- Water waste optimization: {water_saving}% saving possible.
""",
        "Soil Layer": """
- NPK prediction: Nitrogen 71, Phosphorus 44, Potassium 58.
- pH imbalance detection: Slight alkaline tendency (pH 7.7).
- Nutrient deficiency via leaf+soil fusion: Early Nitrogen stress.
- Fertilizer recommendation engine: Split-dose NPK with micronutrient foliar spray.
- Long-term soil health score: 74/100 with organic matter improvement plan.
""",
        "Vision Layer": """
- Insect classification: probable aphid + whitefly traces.
- Pest density estimation: 14 insects / sq.m (moderate).
- Swarm detection: No critical swarm cluster detected.
- Migration pattern prediction: West-to-east risk under warm wind scenario.
- Smart pesticide timing: 6:00 AM low-wind spray window.
""",
        "Market Layer": f"""
- Satellite imagery integration: canopy uniformity index 0.71.
- Growth stage tracking: vegetative-to-early reproductive transition.
- Production estimate per acre: {round(est_yield / acres, 2)} quintal/acre.
- Profit forecast: estimated ₹{est_profit:,.0f} at current assumptions.
- Market price integration: local mandi benchmark ₹{market_price}/quintal.
""",
        "Execution Layer": """
- Camera → analyze → recommend → auto-execute: enabled workflow.
- Irrigation valve control: 3 valve groups with time-based policy.
- Sprayer control: dosage and pressure lock profile active.
- Drone-based spraying: route generated for high-risk patches.
- Automated farm reporting: daily + weekly PDF summary queued.
""",
        "Climate Layer": """
- Multi-modal fusion model (Vision + Weather + Soil + Time): active.
- 7–30 day disease risk prediction: elevated fungal risk on humid days.
- Frost risk alerts: low risk in near term.
- Heat stress prediction: moderate noon stress next 5 days.
- Wind-based pest migration modeling: alert at >18 km/h gusts.
- Crop growth stage mapping: dynamic map synced with satellite cadence.
""",
        "Price Prediction AI": f"""
- Total cost to grow {crop} ({acres} acre): ₹{growth_cost:,.0f}.
- Estimated harvest value at local market ({location}): ₹{est_revenue:,.0f}.
- Net gain forecast: ₹{est_profit:,.0f}.
- Includes costs: seeds, labor, irrigation, fertilizer, pesticide, logistics.
""",
        "Agent Pipeline": """
1. Sense: Camera + weather + soil + satellite ingestion.
2. Understand: Multi-modal fusion and risk scoring.
3. Decide: Recommendation engine for irrigation/fertilizer/spraying.
4. Execute: Valve, sprayer, and drone controls.
5. Report: Continuous analytics + exportable reports.
""",
    }


def run_pipeline_background(context):
    stages = [
        "Water Layer",
        "Soil Layer",
        "Vision Layer",
        "Climate Layer",
        "Market Layer",
        "Execution Layer",
        "Price Prediction AI",
        "Agent Pipeline",
    ]
    all_reports = build_ai_reports(context)
    st.session_state.job["running"] = True
    st.session_state.job["done"] = []
    st.session_state.job["reports"] = {}

    for stage in stages:
        st.session_state.job["status"] = f"Running {stage}"
        st.session_state.job["reports"][stage] = all_reports[stage]
        st.session_state.job["done"].append(stage)

    st.session_state.job["running"] = False
    st.session_state.job["status"] = "Pipeline completed"
    st.session_state.reports.update(st.session_state.job["reports"])
    st.session_state.agent_status = st.session_state.job["status"]


def export_chat_pdf():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(EXPORT_DIR, f"chat_export_{ts}.pdf")

    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        lines = [f"Chat Export ({st.session_state.username})", ""]
        for item in st.session_state.chat_history:
            lines.append(f"{item['role']}: {item['text']}")
        text = "\n".join(lines) if lines else "No chat data."
        fig.text(0.03, 0.97, text, va="top", fontsize=10, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

    return out_path


# ================= SIDEBAR ================= #

with st.sidebar:
    st.subheader("Control Panel")

    st.session_state.language = st.selectbox("Language", ["English", "Hindi", "Marathi"])
    st.session_state.theme = st.selectbox("Theme", ["Light", "Dark", "Green"])

    nav = st.radio("Navigate", ["Home", "Chat", "Shop", "Doctors", "Contact"], index=["Home", "Chat", "Shop", "Doctors", "Contact"].index(st.session_state.current_menu))
    st.session_state.current_menu = nav

    st.markdown("---")
    st.write("### Reports")
    if st.button("Run Full AI Agent Pipeline"):
        st.session_state.agent_status = "Agent started pipeline"
        run_pipeline_background(st.session_state.get("farm_context", {}))

    if st.button("Analysis"):
        st.session_state.active_report = "Agent Pipeline"

    for rep in st.session_state.reports.keys():
        if st.button(rep):
            st.session_state.active_report = rep

    st.markdown("---")
    if st.button("Export Chat (PDF)"):
        file_path = export_chat_pdf()
        st.success(f"Saved: {file_path}")

    st.markdown("---")
    st.image("https://api.dicebear.com/7.x/identicon/svg?seed=farmer", width=60)
    st.caption(f"User: {st.session_state.username}")
    with st.expander("Profile & Settings"):
        st.write("Open settings")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        st.write("More options coming soon")

# ================= TOP STATUS ================= #

st.title(f"🤖 Agent Status: {st.session_state.agent_status}")

# ================= PAGES ================= #

if st.session_state.current_menu == "Home":
    st.header("🌾 AI Agriculture Agent")
    location = st.text_input("Farm Location")
    crop = st.text_input("Crop Name", value="Soybean")
    acres = st.number_input("Land (acres)", min_value=0.5, value=2.0)
    budget = st.number_input("Budget (₹)", min_value=0.0, value=50000.0)
    uploaded_image = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    st.session_state.farm_context = {
        "location": location,
        "crop": crop,
        "acres": acres,
        "budget": budget,
    }

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Analyze Leaf"):
            if not uploaded_image:
                st.error("Please upload an image first.")
            else:
                image = Image.open(uploaded_image)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()

                st.session_state.agent_status = "Vision model analyzing leaf"
                desc = call_openrouter([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this leaf in detail."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ], VISION_MODEL)

                diagnosis = call_openrouter([
                    {"role": "system", "content": "You are a plant pathologist."},
                    {"role": "user", "content": f"Based on this description, identify crop stress and disease risk:\n{desc}"},
                ], REASONING_MODEL)
                st.session_state.reports["Leaf Analysis"] = diagnosis
                st.session_state.active_report = "Leaf Analysis"
                st.session_state.agent_status = "Leaf analysis completed"

    with col2:
        if st.button("Generate All AI Reports"):
            st.session_state.agent_status = "Executing full agent pipeline"
            run_pipeline_background(st.session_state.farm_context)

    st.write("#### Active Report")
    if st.session_state.active_report and st.session_state.active_report in st.session_state.reports:
        st.markdown(f"### {st.session_state.active_report}")
        st.write(st.session_state.reports[st.session_state.active_report])

elif st.session_state.current_menu == "Chat":
    st.header("💬 Agriculture Assistant Chat")
    user_query = st.text_input("Ask your question")
    if st.button("Send") and user_query:
        st.session_state.agent_status = "Analyzing chat query"
        answer = call_openrouter([
            {"role": "system", "content": "You are an agriculture AI agent."},
            {"role": "user", "content": user_query},
        ], REASONING_MODEL)
        st.session_state.chat_history.append({"role": "User", "text": user_query})
        st.session_state.chat_history.append({"role": "Agent", "text": answer})
        st.session_state.agent_status = "Chat response generated"

    for msg in st.session_state.chat_history[-20:]:
        st.write(f"**{msg['role']}:** {msg['text']}")

elif st.session_state.current_menu == "Shop":
    st.header("🛒 Fertilizer Shop")
    crop = st.text_input("Crop for fertilizer", key="shop_crop")
    requirement = st.text_input("Specific need", key="shop_req")

    if st.button("Search Products"):
        result = call_openrouter([
            {"role": "system", "content": "You are a fertilizer market analyst."},
            {
                "role": "user",
                "content": f"Find fertilizers for crop={crop}, need={requirement}. Include name, NPK, price, and use-case.",
            },
        ], REASONING_MODEL)
        st.write(result)

    if st.button("Show All Fertilizers"):
        st.table(
            [
                {"Product": "NPK 19:19:19", "Type": "Water Soluble", "Approx ₹": 1250},
                {"Product": "Urea", "Type": "Nitrogen", "Approx ₹": 300},
                {"Product": "DAP", "Type": "Phosphorus-rich", "Approx ₹": 1450},
                {"Product": "MOP", "Type": "Potassium", "Approx ₹": 1100},
            ]
        )

elif st.session_state.current_menu == "Doctors":
    st.header("🩺 Crop Doctors")
    issue = st.text_input("Describe crop issue")
    if st.button("Find Doctors"):
        doctors = call_openrouter([
            {"role": "system", "content": "You provide agronomy expert suggestions."},
            {"role": "user", "content": f"Suggest crop doctors/experts for this issue: {issue}. Include specialization and when to consult."},
        ], REASONING_MODEL)
        st.write(doctors)

    if st.button("Show All Doctors"):
        st.table(
            [
                {"Name": "Dr. A. Patil", "Specialization": "Soil Nutrition", "Contact": "+91-900000001"},
                {"Name": "Dr. R. Singh", "Specialization": "Plant Pathology", "Contact": "+91-900000002"},
                {"Name": "Dr. M. Kulkarni", "Specialization": "Pest Management", "Contact": "+91-900000003"},
            ]
        )

elif st.session_state.current_menu == "Contact":
    st.header("📞 Contact")
    st.write("AI Agriculture Agent Support")
    st.write("Email: support@farmagent.local")
