from pycaret.regression import load_model, predict_model
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np

# load pretrained model
model = load_model("catboost_final")

# predict the price of a residential property
def predict(model, df):
    predictions_data = predict_model(estimator=model, data=df)
    return predictions_data["Label"][0]

# app title and description
st.title("Washington D.C. Residential Properties Price Prediction 🏠")
st.write(
    """
![GitHub](https://img.shields.io/github/watchers/elvanselvano/purwadhika-final-project?style=social)
[![Status](https://img.shields.io/badge/status-active-success.svg)]() 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
"""
)
image = Image.open("./assets/wallpaper.jpg")
st.image(image)

st.markdown(
    """
This web app uses actual transaction data and machine learning models to predict the prices of housing in Washington, D.C., which is the capital of the United States. The actual transaction data contain attributes and transaction prices of real estate that respectively serve as independent variables and dependent variables for machine learning models.
"""
)
# st.write("---")

# interior details
st.sidebar.subheader("Interior Details")
st.sidebar.write("")
BEDRM = st.sidebar.slider(label="Bedroom", min_value=1, max_value=6, value=3, step=1)
BATHRM = st.sidebar.slider(label="Bathroom", min_value=0, max_value=6, value=2, step=1)
HF_BATHRM = st.sidebar.slider(
    label="Bahtroom (without Bathtub)", min_value=0, max_value=2, value=1, step=1
)
KITCHENS = st.sidebar.slider(
    label="Kitchens", min_value=0, max_value=4, value=1, step=1
)
HEAT = st.sidebar.radio(
    "Heating", ("Forced Air", "Hot Water Rad", "Warm Cool", "Ht Pump")
)

# additional features
st.sidebar.write("---")
st.sidebar.subheader("Additional Features")
st.sidebar.write("")
ac_check = st.sidebar.checkbox("AC", value=False)
AC = 1 if ac_check else 0

fireplaces_check = st.sidebar.checkbox("Fireplace", value=False)
FIREPLACES = 1 if fireplaces_check else 0

remodeled_check = st.sidebar.checkbox("Remodeled", value=False)
RMDL = 1 if remodeled_check else 0

# construction details
st.sidebar.write("---")
st.sidebar.subheader("Construction Details")
st.sidebar.write("")
STYLE = st.sidebar.selectbox(
    "Style", ("1 Story", "1.5 Story Fin", "2 Story", "2.5 Story Fin", "3 Story")
)
INTWALL = st.sidebar.selectbox(
    "Interior Wall", ("Hardwood", "Hardwood/Carp", "Wood Floor", "Carpet")
)
EXTWALL = st.sidebar.selectbox(
    "Exterior Wall",
    (
        "Common Brick",
        "Brick/Siding",
        "Vinyl Siding",
        "Wood Siding",
        "Stucco",
        "Face Brick",
    ),
)
ROOF = st.sidebar.selectbox(
    "Roof", ("Built Up", "Metal- Sms", "Comp Shingle", "Slate", "Neopren", "Shake")
)
STRUCT = st.sidebar.selectbox(
    "Structure", ("Row Inside", "Single", "Semi-Detached", "Row End", "Multi")
)
st.sidebar.write("---")
st.sidebar.subheader("Property Details")
st.sidebar.write("")
AYB = st.sidebar.slider(label="AYB", min_value=1914, max_value=2018, value=1935, step=1)
EYB = st.sidebar.slider(label="EYB", min_value=1964, max_value=2018, value=1972, step=1)
GBA = st.sidebar.slider(label="GBA", min_value=1204, max_value=1800, value=1577, step=2)
LANDAREA = st.sidebar.slider(
    label="Land Area", min_value=1425, max_value=3460, value=2736, step=2
)
CNDTN = st.sidebar.slider(label="Condition", min_value=1, max_value=6, value=4, step=1)
SALEYEAR = st.sidebar.slider(
    label="Sale Year", min_value=2010, max_value=2018, value=2017, step=1
)
GRADE = st.sidebar.selectbox(
    "Grade",
    (
        "Average",
        "Above Average",
        "Good Quality",
        "Very Good",
        "Superior",
        "Excellent",
        "Exceptional-A",
    ),
)
WARD = st.sidebar.selectbox(
    "Ward",
    ("Ward 1", "Ward 2", "Ward 3", "Ward 4", "Ward 5", "Ward 6", "Ward 7", "Ward 8"),
)

st.subheader("Property Information")
st.markdown(
    "👈  In the sidebar, there are 21 features that you can adjust. The features are shown\
    in the table below. After you have adjusted the \
    features, click on the **Predict** button to see the prediction of the machine\
        learning model."
)

# mapping features with user's input
features = {
    "BATHRM": BATHRM,
    "HF_BATHRM": HF_BATHRM,
    "HEAT": HEAT,
    "AC": AC,
    "BEDRM": BEDRM,
    "AYB": AYB,
    "EYB": EYB,
    "GBA": GBA,
    "STYLE": STYLE,
    "STRUCT": STRUCT,
    "GRADE": GRADE,
    "CNDTN": CNDTN,
    "EXTWALL": EXTWALL,
    "ROOF": ROOF,
    "INTWALL": INTWALL,
    "KITCHENS": KITCHENS,
    "FIREPLACES": FIREPLACES,
    "LANDAREA": LANDAREA,
    "WARD": WARD,
    "SALEYEAR": SALEYEAR,
    "RMDL": RMDL,
}

# Converting Features into DataFrame

features_df = pd.DataFrame([features])
st.dataframe(features_df)

# predict button
if st.button("Predict"):
    prediction = predict(model, features_df)
    st.success(
        "Based on the features, the price of the property is $"
        + str(int(prediction))
        + "."
    )

st.write("---")
# Data Section
expander_bar = st.beta_expander("Data Description")
expander_bar.markdown(
    """
|            Feature            |                         Description                          | 
| :-----------------------------: | -------------- |
| BATHRM | Number of Bathroom     |
| HF_BATHRM | Number of half bathroom (no bathtub or shower) |
| HEAT | Heating Type |
| AC | Has AC? (Y/N) |
| BEDRM | Number of bedroom |
| AYB | 	The earliest time the main portion of the building was built |
| EYB | The year an improvement was built more recent than actual year built |
| GBA | Gross building area in square feet |
| STYLE | Type of story |
| STRUCT | 	Building structure |
| GRADE | Property Grade |
| CNDTN | Property Condition |
| EXTWALL | Exterior wall type |
| ROOF | Roof type |
| INTWALL | Interior wall type |
| KITCHENS | Number of kitchens |
| FIREPLACES | Number of fireplaces |
| LANDAREA | Land area of property in square feet |
| WARD | Ward (district is divided into eight wards, each with approximately 75,000 residents) |
| SALEYEAR | Year of most recent sale |
| RMDL | Has remodeled? (Y/N)

"""
)
expander_bar.write("")
expander_bar.info(
    "All data is available at [Open Data D.C.](https://opendata.dc.gov/). The residential and address point data is managed by the [Office of the Chief Technology Officer](https://octo.dc.gov/)."
)
