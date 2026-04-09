import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime
import pandas as pd
import os

st.set_page_config(
    page_title="Car Damage Fraud Detector",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_cnn():
    return load_model("cnn_model_final.h5")

model   = load_cnn()
IMG_SIZE = 64

def init_db():
    conn   = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp  TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(image_name, prediction, confidence):
    conn   = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions
        (image_name, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (image_name, prediction, confidence,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect("predictions.db")
    try:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 20", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

init_db()

def predict(img_array):
    img_bgr     = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_norm    = img_rgb / 255.0
    img_input   = np.expand_dims(img_norm, axis=0)
    prob  = model.predict(img_input, verbose=0)[0][0]
    pred  = 1 if prob > 0.5 else 0
    conf  = round(float(prob*100) if pred==1 else float((1-prob)*100), 2)
    label = "FAKE — Insurance Fraud Detected" if pred==1 else "REAL — Genuine Damage"
    return label, conf, pred

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:10px 0'>
    <h1 style='color:#1E2761; margin-bottom:0'>
        Car Damage Fraud Detector
    </h1>
    <p style='color:#888; font-size:16px'>
        AI-powered system to detect fake vs real car damage images
        for insurance fraud prevention
    </p>
</div>
<hr style='border-color:#1E2761'>
""", unsafe_allow_html=True)

# ─── Stats ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("CNN Accuracy",  "95.30%")
c2.metric("KNN Accuracy",  "81.50%")
c3.metric("Dataset Size",  "4,035 Images")
c4.metric("Classes",       "Real vs Fake")

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Main Layout ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    st.subheader("Upload Car Image")
    uploaded = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

        if st.button("Analyze Image", type="primary",
                     use_container_width=True):
            with st.spinner("Analyzing..."):
                img_array         = np.array(image)
                label, conf, pred = predict(img_array)
                save_to_db(uploaded.name, label, conf)

            st.markdown("<hr>", unsafe_allow_html=True)

            if pred == 1:
                st.error(f"RESULT: {label}")
                st.error(f"Confidence: {conf}%")
                st.warning(
                    "This image appears to be AI-generated or "
                    "digitally manipulated. Possible insurance "
                    "fraud detected."
                )
            else:
                st.success(f"RESULT: {label}")
                st.success(f"Confidence: {conf}%")
                st.info(
                    "This image appears to be a genuine "
                    "car damage photograph."
                )

with right:
    st.subheader("How It Works")
    st.markdown(
        """
        <div style='background:#1E2761; padding:20px;
                    border-radius:10px; color:white;
                    font-size:15px; line-height:2'>
            <b>Step 1</b> — Upload a car damage image<br>
            <b>Step 2</b> — CNN model analyzes the image<br>
            <b>Step 3</b> — Result: Real or Fake with confidence<br>
            <b>Step 4</b> — Prediction saved to database
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("About the Model")
    st.markdown(
        """
        <div style='background:#1E2761; padding:20px;
                    border-radius:10px; color:white;
                    font-size:15px; line-height:2'>
            <b>CNN Architecture:</b><br>
            3 Conv blocks — 32, 64, 128 filters<br>
            BatchNormalization + Dropout layers<br>
            EarlyStopping to prevent overfitting<br><br>
            <b>Dataset:</b><br>
            2,535 real damaged car images (Kaggle)<br>
            1,500 AI-generated fake images (Stable Diffusion)<br><br>
            <b>KNN Features:</b> HOG + DCT + LBP combined<br><br>
            <b>Group 51 | SIT Pune | AI & ML 2024-28</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Model Performance")
    st.markdown(
        """
        <div style='background:#196B24; padding:20px;
                    border-radius:10px; color:white;
                    font-size:15px; line-height:2'>
            <b>CNN Accuracy  : 95.30%</b><br>
            <b>KNN Accuracy  : 81.50%</b><br>
            Precision : 0.95<br>
            Recall    : 0.95<br>
            F1-Score  : 0.95<br><br>
            Real image confidence  : 99.88%<br>
            Fake image confidence  : 100.0%
        </div>
        """,
        unsafe_allow_html=True
    )

# ─── History ─────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Prediction History — Database Log")

df = get_history()
if len(df) == 0:
    st.info("No predictions yet. Upload an image above to get started.")
else:
    st.dataframe(df, use_container_width=True)
    st.caption("Showing last 20 predictions from SQLite database")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#888; font-size:12px'>
    Symbiosis Institute of Technology, Pune |
    Department of AI & ML | Group 51 |
    Animesh Khare | Sejal Jain | Ayush Das |
    Guide: Ms. Shruti Sunnad
</p>
""", unsafe_allow_html=True)