import streamlit as st

#st.set_page_config(page_title="App Profile", layout="centered")

def app():
    # HEADER SECTION
    st.markdown("<h1 style='font-size: 62px;text-align: center;'>Welcome to AegisLens!</h1>", unsafe_allow_html=True)
    #foto
    st.image("Deployment/AegisLensLogo.png", use_container_width=True)
    paragraph1 = "AegisLens is an application designed to enhance traffic incident response and road monitoring by leveraging computer vision technology. AegisLens processes dashcam footage to automatically detect vehicles, recognize license plates, and extract critical information such as license plate numbers and expiry dates. This tool aims to streamline post-incident evidence collection and improve the efficiency of traffic investigations."
    st.markdown(
    """
    <style>
    .paragraph {{
        text-align: justify;
        font-size: 16px;  /* Optional: adjust font size */
        line-height: 1.6; /* Optional: improve readability */
    }}
    </style>
    <p class="paragraph">{}</p>
    """.format(paragraph1),
    unsafe_allow_html=True
    )

    # PROJECT DEFINITION
    st.header("What is AegisLens?")
    paragraph2 = "AegisLens is a model that can detect vehicle types and read their license plate. It utilizes an object detection model and an Optical Character Recognition (OCR) model to output it’s results."
    st.markdown(
    """
    <style>
    .paragraph {{
        text-align: justify;
        font-size: 16px;  /* Optional: adjust font size */
        line-height: 1.6; /* Optional: improve readability */
    }}
    </style>
    <p class="paragraph">{}</p>
    """.format(paragraph2),
    unsafe_allow_html=True
    )

    # HOW IT WORKS APP SECTION
    st.write("<h1 style='font-size: 20px;'>How does it work?</h1>", unsafe_allow_html=True)
    st.write("""
    1. **Object Detection**: The model detects vehicles in the video footage and identifies their bounding boxes.
    2. **License Plate Recognition**: For each detected vehicle, the model extracts the license plate region and applies OCR to read the text.
    3. **Expiry Date Extraction**: The model processes the license plate text to extract the expiry date, if available.
    4. **Output**: The model displays the detected vehicle type, license plate number, and expiry date on the video footage.
    """)

    # TOOLS SECTIOM
    st.write("<h1 style='font-size: 20px;'>Tools for model development:</h1>", unsafe_allow_html=True)
    st.write("""
    - **YOLOv8**: A object detection model used for detecting vehicles in the video footage.
    - **EasyOCR**: An OCR model used for reading text from the detected license plates.
    - **Streamlit**: A web application framework used to create the user interface for the application.
    """)
    # FEATURES SECTION
    st.write("<h1 style='font-size: 20px;'>Features:</h1>", unsafe_allow_html=True)
    st.write("""
    - **Exploratory Data Analysis**: View insights and statistics about the dataset used for training the model.
    - **Prediction**: Upload a video file to detect vehicles and read license plates.
    """)

    #HOW TO USE APP SECTION
    st.write("<h1 style='font-size: 20px;'>How to Use AegisLens?</h1>", unsafe_allow_html=True)
    paragraph3 = "To try our features, simply open the sidebar and select the feature you want to pick. You can view the exploratory data analysis related to our project and upload a video file to try the inference."
    st.markdown(
    """
    <style>
    .paragraph {{
        text-align: justify;
        font-size: 16px;  /* Optional: adjust font size */
        line-height: 1.6; /* Optional: improve readability */
    }}
    </style>
    <p class="paragraph">{}</p>
    """.format(paragraph3),
    unsafe_allow_html=True
    )

    # TEAM PROFILE SECTION
    st.markdown("<h1 style='font-size: 48px; text-align: center;'>MEET OUR TEAM</h1>", unsafe_allow_html=True)
    # Create four columns
    col1, col2, col3, col4, col5 = st.columns(5)

    # Display each image in its respective column
    with col1:
        st.image("Deployment/FotoKakLis.jpg", use_container_width=True)
        st.write("<p style='text-align: center;'>Lis Wahyuni</p>", unsafe_allow_html=True)
        st.write("<p style='text-align: center;font-weight: bold;'>Mentor</p>", unsafe_allow_html=True)

    with col2:
        st.image("Deployment/FotoKemalRevised.png", use_container_width=True)
        st.write("<p style='text-align: center;'>M. Faishal Kemal J. A.</p>", unsafe_allow_html=True)
        st.write("<p style='text-align: center;font-weight: bold;'>Data Scientist & Data Annotator</p>", unsafe_allow_html=True)

    with col3:
        st.image("Deployment/FotoRobiRevised.png", use_container_width=True)
        st.write("<p style='text-align: center;'>M. Rafi Abhinaya</p>", unsafe_allow_html=True)
        st.write("<p style='text-align: center;font-weight: bold;'>Data Scientist & Data Annotator</p>", unsafe_allow_html=True)

    with col4:
        st.image("Deployment/FotoMasBibiRevised.png", use_container_width=True)
        st.write("<p style='text-align: center;'>Ma'ruf Habibie Siregar</p>", unsafe_allow_html=True)
        st.write("<p style='text-align: center;font-weight: bold;'>Data Analyst & Data Annotator</p>", unsafe_allow_html=True)

    with col5:
        st.image("Deployment/FotoOjanRevised.jpeg", use_container_width=True)
        st.write("<p style='text-align: center;'>Fauzan Rahmat Farghani</p>", unsafe_allow_html=True)
        st.write("<p style='text-align: center;font-weight: bold;'>Data Engineer & Data Annotator</p>", unsafe_allow_html=True)
