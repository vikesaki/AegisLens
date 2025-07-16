import streamlit as st

st.set_page_config(page_title="AegisLens: Vehicle Object Classifier", layout="wide")

import eda
import appProfile
import inference

st.sidebar.title("What do you want to see?")
page = st.sidebar.radio("Choose one below:", ("About Project", "Exploratory Data Analysis", "Test our model!"), index=0)


st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
**Team Member:**
1. Muhammad Faishal Kemal Jauhar Arifin
2. Muhammad Rafi Abhinaya
3. Ma'ruf Habibie Siregar
4. Fauzan Rahmat Farghani
""")

if page == "Exploratory Data Analysis":
    eda.app()
elif page == "Test our model!":
    inference.app()
else:
    appProfile.app()
