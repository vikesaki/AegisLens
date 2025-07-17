import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def app():
    st.title("Exploratory Data Analysis of AegisLens")
    paragraph_eda1 = "Below is the exploratory data analysis of the dataset used for training the AegisLens model. This section provides insights into the data distribution and other statistical information."
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
    """.format(paragraph_eda1),
    unsafe_allow_html=True
    )
    # CLASS NAMES INFORMATION
    st.header("Class Names Information")
    st.write("Here are the class names of the dataset:")
    data = {
        "Class Name": ["Bus", "Car", "LicensePlate", "Motorcycle", "Truck"]
    }
    df = pd.DataFrame(data, index=range(1, 6))
    st.table(df)
    st.write("There are 5 Classes. These classes are types of vehicles we classify, along with license plates whose numbers will be detected.")
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    #EDA QUESTION 1
    st.header("1. Distribution of Objects per Class")
    st.image("Deployment/FotoEDA1.1.png", use_container_width=True)
    st.write("Based on the insight that Class 0 (Bus) is underrepresented, we will now look at example images containing objects from this class.")
    st.image("Deployment/FotoEDA1.2.png", use_container_width=True)
    st.write("""
    **Insight: Class Distribution**
    - The bar chart above illustrates the class distribution in the training dataset.
    - Class 0 (Bus) has very few samples compared to the other four classes, indicating a data imbalance.
    - Class 2 (Motorcycle) has the highest number of samples, followed by Class 1 (Car) and Class 3 (License Plate), with a difference of about 300 samples between Class 2 and 3.
    - Class 4 (Truck) also has relatively fewer samples, though still significantly more than Class 0 (Bus).
    - Overall, this suggests that the data collection environment is dominated by cars and motorcycles.
    - The number of License Plate labels is relatively low compared to the total number of vehicle objects, meaning not all vehicles had readable license plates.
    - To address the class imbalance—especially for Class 0 (Bus) and Class 4 (Truck)—additional sampling is recommended. For buses, this can be done in areas near bus terminals, and for trucks, in industrial zones or logistic warehouses.
    """)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # EDA QUESTION 2
    st.header("2. Distribution of the Number of Objects per Image")
    st.write("This section explores how many labeled objects typically appear in each image.")
    st.write("""
    Understanding this distribution helps us:
    - Identify whether the dataset is suited for single or multi-object detection.
    - Understand the complexity of each image in terms of object density.
    - Determine how well the model needs to handle crowded versus sparse scenes.
    """)
    st.image("Deployment/FotoEDA2.1.png", use_container_width=True)
    st.write("""
    **Insight: Number of Labeled Objects per Image**
    - The chart shows the distribution of labeled objects per image.
    - Most images contain more than one object, with the majority having between 4 to 8 objects.
    - There are also images with 15–20 objects, and even some with up to 30 objects, although these are fewer than 10 images in total.
    - The distribution is uneven, with a dominance of images containing a moderate number of objects (around 4–5). This could affect the model’s performance, as it may become more optimized for detecting objects within that range.
    """)
    st.write("""
    After analyzing the object count per image, we also explore the following aspects:
    1. Average number of objects per image
    2. Percentage of images with only 1 object
    3. Percentage of images containing both vehicles and license plates
    """)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.write("**Summary of Object Count Analysis:**")
    st.write("""
    - Total number of images                     : 1137
    - Average number of objects per image        : 6.91
    - Images with only 1 object                  : 25 (2.20%)
    - Images with both vehicle(s) and license plate : 1005 (88.39%)
    """)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.write("""
    **Insight: Object Statistics per Image**
    - The average number of objects per image is 6.91, indicating that most images contain multiple objects.
    - About 2.2% of images contain only one object, confirming that this dataset is suitable for multi-object detection models.
    - Around 1005 images (88.39%) contain a combination of vehicle and license plate, meaning license plates are typically present alongside the vehicle. This co-occurrence can help the model better learn the positioning of license plates.
    """)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # EDA QUESTION 3
    st.header("3. Bounding Box Size Distribution Pattern by Class")
    st.write("In this section, we examine how bounding box sizes vary across different object classes.")
    st.write("""
    Key objectives:
    - Understand whether certain classes tend to have consistently larger or smaller bounding boxes.
    - Identify challenges related to small object detection (e.g. License Plates).
    - Adjust model strategies accordingly based on object scale differences.
    """)
    st.image("Deployment/FotoEDA3.1.png", use_container_width=True)
    st.write("""
    **Insight: Bounding Box Size Distribution**
    - The chart illustrates the distribution of bounding box sizes in the dataset.
    - Most bounding boxes are relatively small, likely dominated by license plates or objects far from the camera.
    - The small size of license plate bounding boxes could present a challenge for the OCR model, especially for reading license numbers and expiry dates.
    - There are also large bounding boxes, likely belonging to buses, trucks, or cars close to the camera.
    - Since small bounding boxes are the most common, the model must be optimized to detect small objects effectively.
    - Conversely, due to the lower occurrence of large bounding boxes, the model should still be capable of recognizing large-sized objects when present.
    """)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # EDA QUESTION 4
    st.header("4. Bounding Box Aspect Ratio Distribution (Width-to-Height)")
    st.write("This section analyzes the distribution of bounding box aspect ratios, calculated as width divided by height.")
    st.write("""
    Purpose of this analysis:
    - Identify shape patterns specific to each class (e.g., horizontal vs. vertical rectangles).
    - Understand the spatial characteristics of objects like License Plates (typically wide) and Motorcycles (typically tall).
    - Help inform model design to better handle varying object shapes.
    """)
    st.image("Deployment/FotoEDA4.1.png", use_container_width=True)
    st.write("""
    **Insight : Bounding Box Aspect Ratio Distribution**
    - The chart above shows the distribution of bounding box aspect ratios (width/height).
    - Smaller ratios (less than 1) likely represent License Plates.
    - Larger ratios (greater than 1) may correspond to Buses, Cars, Trucks, or objects that appear closer to the camera.
    - License plates generally have a wider width (x) than height (y), making them horizontal rectangles.
    - Motorcycles, on the other hand, typically have a greater height (y) than width (x), resulting in vertical rectangle shapes.
    - Therefore, license plates are usually horizontally shaped, while motorcycles tend to be vertically shaped.
    """)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # EDA QUESTION 5
    st.header("5. Sample Visualizations of Labeled Images")
    st.write("This section presents sample images from the dataset with their corresponding bounding boxes and class labels for visual inspection.")
    st.write("""
    The purpose of this visualization is to:
    - Verify the accuracy and consistency of labeling.
    - Observe variations in object size, shape, and positioning.
    - Understand real-world image conditions such as lighting, occlusion, and viewing angles.
    """)
    st.image("Deployment/FotoEDA5.1.png", use_container_width=True)
    st.image("Deployment/FotoEDA5.2.png", use_container_width=True)
    st.image("Deployment/FotoEDA5.3.png", use_container_width=True)
    st.write("""
    **Insight : Visualizations of Labeled Images**
    - The images above are examples of labeled visualizations from the dataset.
    - Most objects are captured from the rear, with only a few labeled from the front or side.
    - There is a clear variation in bounding box sizes across classes. For example, cars tend to have more symmetrical width and height, compared to motorcycles.
    - The label shape for motorcycles appears taller and narrower than other classes.
    - Trucks have bounding boxes that are significantly larger compared to other objects.
    - The License Plate is the smallest object among all classes.
    - Given these distinct shape differences, the model should be able to learn and distinguish between classes effectively.
    """)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # EDA QUESTION 6
    st.header("6. Distribution of Bounding Box Centroid Positions")
    st.write("This section visualizes the spatial distribution of bounding box centroids to identify patterns in object placement within the images.")
    st.write("""
    Key insights we aim to uncover:
    - Identify whether objects are mostly centered or scattered across the frame.
    - Detect any positional bias that may affect model learning.
    - Understand common placements of certain object types, such as License Plates, which often appear in the lower-middle region of the image.
    """)
    st.image("Deployment/FotoEDA6.1.png", use_container_width=True)
    st.write("""
    **Insight : Heatmap of Bounding Box Centroid Positions**
    - The heatmap above shows the distribution of bounding box centroid positions.
    - From the heatmap, we can conclude that most objects are centered around (0.4–0.6, 0.6–0.7). This is indicated by the darker regions in that area.
    - The centroids tend to cluster around the center of the image, suggesting that most objects appear near the middle of the frame.
    - It is also likely that license plates are located around the y = 0.6 region, which corresponds to the lower-middle part of the image.
    """)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    #EDA Summary
    st.header("Summary of Exploratory Data Analysis")
    st.write("""
    This section summarizes key insights gathered from the Exploratory Data Analysis (EDA) phase.
    - The dataset contains 5 classes, with noticeable class imbalance in the Bus and Truck categories.
    - Most images contain **more than one object**, often featuring a combination of **vehicles and license plates**.
    - Bounding box sizes and aspect ratios vary significantly — from **small, horizontally shaped license plates** to **taller, vertically shaped motorcycles**.
    - The centroids of bounding boxes are **concentrated near the center of the image**, indicating that most objects appear in central regions.
    """)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
