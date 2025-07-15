import streamlit as st

#st.set_page_config(page_title="App Profile", layout="centered")

def app():
    st.title("Welcome to AegisLens")
    st.write("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Vestibulum euismod, nisi vel consectetur cursus, nisl erat dictum urna, 
    a cursus enim erat nec enim. 
    """)

    st.header("What is AegisLens?")
    st.write("""
    Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, 
    totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.
    """)

    st.header("Contact Information")
    st.write("""
    Email: example@email.com  
    Phone: +62 812-3456-7890
    """)