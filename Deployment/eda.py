import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# st.set_page_config(page_title="EDA Page", layout="wide")

def app():
    st.title("Exploratory Data Analysis (EDA)")
    st.write("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    """)

    st.header("Sample Data")
    df = pd.DataFrame({
        "A": np.random.randn(50),
        "B": np.random.rand(50),
        "C": np.random.randint(0, 100, 50)
    })
    st.dataframe(df)

    st.header("Histogram Example")
    fig, ax = plt.subplots()
    ax.hist(df["A"], bins=15, color="skyblue", edgecolor="black")
    ax.set_title("Distribution of Column A")
    st.pyplot(fig)

    st.header("Scatter Plot Example")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["A"], df["B"], c=df["C"], cmap="viridis")
    ax2.set_xlabel("A")
    ax2.set_ylabel("B")
    ax2.set_title("Scatter Plot of A vs B")
    st.pyplot(fig2)

    st.write("""
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    """)