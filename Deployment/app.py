import streamlit as st

st.set_page_config(page_title="AegisLens: Vehicle Object Classifier", layout="wide")

import eda
import appProfile
import inference

st.sidebar.title("What do you want to see?")
page = st.sidebar.radio("Pick One :)", ("About Project", "EDA", "Prediction"), index=0)


st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
**Team Member:**

1. Kemal  
2. Robi  
3. Mas Bibi  
4. Ojan
""")

if page == "EDA":
    eda.app()
elif page == "Prediction":
    inference.app()
else:
    appProfile.app()