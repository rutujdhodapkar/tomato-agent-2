import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
import os
import pandas as pd

# ================= CONFIG ================= #

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
REASONING_MODEL = "deepseek/deepseek-r1-0528:free"

USER_DB = "users.json"

# ================= API ================= #

def call_openrouter(messages, model):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {"model": model, "messages": messages}
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    response = r.json()

    if "choices" in response:
        return response["choices"][0]["message"]["content"]

    return str(response)

# ================= TRANSLATIONS ================= #

translations = {
    "English": {
        "home": "Home",
        "chat": "Chat",
        "shops": "Shops",
        "contact": "Contact Us",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "upload": "Upload Leaf Image",
        "analyze": "Analyze Crop"
    },
    "Hindi": {
        "home": "होम",
        "chat": "चैट",
        "shops": "दुकान",
        "contact": "संपर्क करें",
        "login": "लॉगिन",
        "username": "उपयोगकर्ता नाम",
        "password": "पासवर्ड",
        "upload": "पत्ता अपलोड करें",
        "analyze": "विश्लेषण करें"
    },
    "Marathi": {
        "home": "मुख्यपृष्ठ",
        "chat": "चॅट",
        "shops": "दुकान",
        "contact": "संपर्क",
        "login": "लॉगिन",
        "username": "वापरकर्ता नाव",
        "password": "पासवर्ड",
        "upload": "पान अपलोड करा",
        "analyze": "विश्लेषण करा"
    }
}

# ================= SESSION INIT ================= #

if "language" not in st.session_state:
    st.session_state.language = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ================= LANGUAGE SELECT ================= #

if not st.session_state.language:
    st.title("🌍 Select Language")
    lang = st.selectbox("Language", ["English", "Hindi", "Marathi"])
    if st.button("Continue"):
        st.session_state.language = lang
        st.rerun()

lang_text = translations[st.session_state.language]

# ================= LOGIN SYSTEM ================= #

if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

with open(USER_DB, "r") as f:
    users = json.load(f)

if not st.session_state.logged_in:

    st.title(lang_text["login"])
    username = st.text_input(lang_text["username"])
    password = st.text_input(lang_text["password"], type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            users[username] = password
            with open(USER_DB, "w") as f:
                json.dump(users, f)
            st.session_state.logged_in = True
            st.success("Account Created")
            st.rerun()

# ================= NAVIGATION ================= #

menu = st.radio(
    "",
    [lang_text["home"], lang_text["chat"], lang_text["shops"], lang_text["contact"]],
    horizontal=True
)

# ================= HOME ================= #

if menu == lang_text["home"]:

    st.title("🌾 Agricultural Intelligence")

    location = st.text_input("Farm Location")
    uploaded_image = st.file_uploader(lang_text["upload"], type=["jpg", "png"])

    if st.button(lang_text["analyze"]):

        if not uploaded_image:
            st.error("Upload image.")
            st.stop()

        image = Image.open(uploaded_image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Vision describe
        desc = call_openrouter([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this leaf in detail."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }], VISION_MODEL)

        # Diagnose
        diagnosis = call_openrouter([
            {"role": "system", "content": "You are plant pathologist."},
            {"role": "user", "content": f"Based on: {desc} Identify Crop and Disease."}
        ], REASONING_MODEL)

        st.markdown("## Result")
        st.write(diagnosis)

# ================= CHAT ================= #

elif menu == lang_text["chat"]:

    st.title("💬 AI Chat")
    user_query = st.text_input("Ask anything about agriculture")

    if st.button("Send"):
        response = call_openrouter([
            {"role": "system", "content": "You are agricultural assistant."},
            {"role": "user", "content": user_query}
        ], REASONING_MODEL)

        st.write(response)

# ================= SHOPS ================= #

elif menu == lang_text["shops"]:

    st.title("🛒 Fertilizer Shop Search")

    crop = st.text_input("Crop Name")
    req = st.text_input("Specific Requirement")

    if st.button("Search Products"):

        result = call_openrouter([
            {"role": "system", "content": "You are fertilizer market analyst."},
            {"role": "user",
             "content": f"""
             Find best fertilizers online for crop: {crop}
             Requirement: {req}
             Provide:
             - Product Name
             - NPK Ratio
             - Approx Price
             - Usage Reason
             - Online availability
             """}
        ], REASONING_MODEL)

        st.write(result)

# ================= CONTACT ================= #

elif menu == lang_text["contact"]:

    st.title("📞 Contact Us")

    st.markdown("""
    **Name:** Rutuj Dhodapkar  
    **Email:** rutujdhodapkar@gmail.com  
    **Username:** rutujdhodapkar  
    **Portfolio:** https://rutujdhodapkar.vercel.app/  
    **Specialization:** Advanced AI, Deep Learning, Machine Learning, Big Data  
    **Location:** Los Angeles  
    """)
