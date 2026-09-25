# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌍",
    layout="centered"
)
# -----------------------------
# Load model and encoders
# -----------------------------

# model = joblib.load("aqi_model_final.pkl")
# state_encoder = joblib.load("state_encoder.pkl")
# area_encoder = joblib.load("area_encoder.pkl")


# -----------------------------
# Load dataset
# -----------------------------

# df = pd.read_csv("aqi_preprocessed_3.csv")

@st.cache_resource
def load_model():
    return joblib.load("aqi_model_final.pkl")


@st.cache_resource
def load_encoders():
    state_encoder = joblib.load("state_encoder.pkl")
    area_encoder = joblib.load("area_encoder.pkl")
    return state_encoder, area_encoder


@st.cache_data
def load_data():
    return pd.read_csv("aqi_preprocessed_3.csv")


model = load_model()

state_encoder, area_encoder = load_encoders()

df = load_data()
# -----------------------------
# Create State → Area mapping
# -----------------------------

state_names = state_encoder.classes_

state_area_map = {}

for state_code, state_name in enumerate(state_names):

    area_codes = df[df["state"] == state_code]["area"].unique()

    areas = area_encoder.inverse_transform(
        area_codes.astype(int)
    )

    state_area_map[state_name] = sorted(areas)


# -----------------------------
# Streamlit Page
# -----------------------------

st.title("🌍 Air Quality Index Prediction")

st.write(
    "Predict AQI using State, City/Area and Date."
)


# -----------------------------
# State selection
# -----------------------------

selected_state = st.selectbox(
    "Select State",
    sorted(state_names)
)


# -----------------------------
# Area selection
# -----------------------------

selected_area = st.selectbox(
    "Select City / Area",
    state_area_map[selected_state]
)


# -----------------------------
# Date selection
# -----------------------------

selected_date = st.date_input(
    "Select Date"
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict AQI"):

    state_code = state_encoder.transform(
        [selected_state]
    )[0]

    area_code = area_encoder.transform(
        [selected_area]
    )[0]

    day = selected_date.day
    month = selected_date.month
    year = selected_date.year

    input_data = np.array([[
        state_code,
        area_code,
        day,
        month,
        year
    ]])

    prediction = model.predict(input_data)[0]

    # AQI category
    if prediction <= 50:
        status = "Good"

    elif prediction <= 100:
        status = "Satisfactory"

    elif prediction <= 200:
        status = "Moderate"

    elif prediction <= 300:
        status = "Poor"

    elif prediction <= 400:
        status = "Very Poor"

    else:
        status = "Severe"


    # Display results

    st.success(
        f"Predicted AQI: {prediction:.2f}"
    )

    st.info(
        f"Air Quality: {status}"
    )