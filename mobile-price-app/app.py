import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱")


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load():
    return (
        joblib.load(os.path.join(BASE_DIR, "model.pkl")),
        joblib.load(os.path.join(BASE_DIR, "columns.pkl")),
    )


model, columns = load()
labels = {0: "Low cost", 1: "Medium cost", 2: "High cost", 3: "Very high cost"}

st.title("📱 Mobile Price Range Predictor")
st.write("Enter the phone's specifications to predict its price range.")

col1, col2 = st.columns(2)
with col1:
    battery_power = st.slider("Battery power (mAh)", 500, 2000, 1200)
    ram = st.slider("RAM (MB)", 256, 4000, 2000)
    int_memory = st.slider("Internal memory (GB)", 2, 64, 32)
    clock_speed = st.slider("Processor clock speed (GHz)", 0.5, 3.0, 1.5, 0.1)
    n_cores = st.slider("Number of processor cores", 1, 8, 4)
    mobile_wt = st.slider("Weight (g)", 80, 200, 140)
    m_dep = st.slider("Thickness (cm)", 0.1, 1.0, 0.5, 0.1)
    talk_time = st.slider("Talk time (hours)", 2, 20, 10)
    fc = st.slider("Front camera (MP)", 0, 19, 5)
    pc = st.slider("Primary camera (MP)", 0, 20, 10)
with col2:
    px_height = st.slider("Screen resolution height (px)", 0, 1960, 900)
    px_width = st.slider("Screen resolution width (px)", 500, 1998, 1200)
    sc_h = st.slider("Screen height (cm)", 5, 19, 12)
    sc_w = st.slider("Screen width (cm)", 0, 18, 7)
    blue = st.checkbox("Bluetooth", True)
    dual_sim = st.checkbox("Dual SIM", True)
    four_g = st.checkbox("4G", True)
    three_g = st.checkbox("3G", True)
    touch_screen = st.checkbox("Touch screen", True)
    wifi = st.checkbox("WiFi", True)

if st.button("Predict price range"):
    values = {
        "battery_power": battery_power, "blue": int(blue), "clock_speed": clock_speed,
        "dual_sim": int(dual_sim), "fc": fc, "four_g": int(four_g),
        "int_memory": int_memory, "m_dep": m_dep, "mobile_wt": mobile_wt,
        "n_cores": n_cores, "pc": pc, "px_height": px_height, "px_width": px_width,
        "ram": ram, "sc_h": sc_h, "sc_w": sc_w, "talk_time": talk_time,
        "three_g": int(three_g), "touch_screen": int(touch_screen), "wifi": int(wifi),
    }
    row = pd.DataFrame([values])[columns]
    pred = int(model.predict(row)[0])
    prob = model.predict_proba(row)[0][pred]
    st.success(f"Predicted price range: **{labels[pred]}** (confidence {prob:.0%})")
