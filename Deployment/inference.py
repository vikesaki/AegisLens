import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import tempfile
import time

def app():
    # --- HELPER FUNCTIONS ---
    def preprocess_expiry_crop(crop):
        """Preprocess expiry date crop for OCR."""
        if crop is None or crop.shape[0] < 5 or crop.shape[1] < 10:
            return None
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = max(2.0, 50.0 / gray.shape[0])
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def extract_and_validate_date(text):
        """Validate and format MM.YY date."""
        if not text:
            return "UNKNOWN"
        
        digits = re.sub(r'\D', '', text)
        if len(digits) == 4:
            month_str, year_str = digits[:2], digits[2:]
            try:
                if 1 <= int(month_str) <= 12:
                    return f"{month_str}.{year_str}"
            except ValueError:
                pass
        return "INCORRECT LENGTH"

    # Set page configuration
    # st.set_page_config(page_title="License Plate Detection with YOLO", layout="wide")
    # Streamlit app
    st.title("Let's test our model!")

    # Model configuration
    MODEL_PATH = "latest.pt"
    ALLOWED_EXTENSIONS = ['.mov', '.mp4']
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB in bytes
    DEFAULT_CONFIDENCE = 0.5
    RESOLUTION = "1280x720"  # Default resolution

    # Load model and OCR reader
    @st.cache_resource
    def load_model():
        return YOLO(MODEL_PATH, task='detect')

    @st.cache_resource
    def load_ocr_reader():
        return easyocr.Reader(['en'], gpu=False)  # Set GPU to False for compatibility

    model = load_model()
    reader = load_ocr_reader()
    labels = model.names

    # Set bounding box colors (Tableau 10 color scheme)
    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
                (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    # File uploader
    uploaded_file = st.file_uploader("Upload a video (.mov, .mp4)", type=['mov', 'mp4'])

    if uploaded_file is not None:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File size exceeds 200MB limit. Please upload a smaller file.")
        else:
            # Check file extension
            _, ext = os.path.splitext(uploaded_file.name)
            if ext.lower() not in ALLOWED_EXTENSIONS:
                st.error(f"Unsupported file format. Please upload a .mov or .mp4 file.")
            else:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Parse resolution
                resW, resH = map(int, RESOLUTION.split('x'))
                
                # Video processing
                cap = cv2.VideoCapture(tmp_file_path)
                output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
                recorder = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (resW, resH))
                
                stframe = st.empty()
                fps_display = st.empty()
                object_count_display = st.empty()
                
                frame_rate_buffer = []
                fps_avg_len = 200
                avg_frame_rate = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    t_start = time.perf_counter()

                    # Resize frame
                    frame = cv2.resize(frame, (resW, resH))

                    # Run inference
                    results = model(frame, verbose=False)
                    detections = results[0].boxes
                    object_count = 0

                    # Process detections
                    for i in range(len(detections)):
                        xyxy_tensor = detections[i].xyxy.cpu()
                        xyxy = xyxy_tensor.numpy().squeeze()
                        xmin, ymin, xmax, ymax = xyxy.astype(int)
                        classidx = int(detections[i].cls.item())
                        classname = labels[classidx]
                        conf = detections[i].conf.item()

                        if conf > DEFAULT_CONFIDENCE:
                            color = bbox_colors[classidx % 10]
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                            if classidx == 2:  # License Plate
                                plate_text = "..."
                                expiry_text = "EMPTY"
                                try:
                                    box_height = ymax - ymin
                                    split_point_upper = ymin + int(box_height * 0.70)
                                    split_point_lower = ymin + int(box_height * 0.55)

                                    plate_crop = frame[ymin:split_point_upper, xmin:xmax]
                                    expiry_crop = frame[split_point_lower:ymax, xmin:xmax]

                                    plate_text_results = reader.readtext(plate_crop, detail=0, paragraph=False)
                                    plate_text = "".join(plate_text_results).upper().replace(" ", "") if plate_text_results else "..."

                                    processed_expiry = preprocess_expiry_crop(expiry_crop)
                                    if processed_expiry is not None:
                                        ocr_results = reader.readtext(
                                            processed_expiry,
                                            detail=0,
                                            paragraph=False,
                                            allowlist='0123456789.'
                                        )
                                        raw_text = "".join(ocr_results) if ocr_results else ""
                                        expiry_text = extract_and_validate_date(raw_text) if raw_text else "EMPTY"
                                    else:
                                        expiry_text = "EMPTY"

                                    display_text_plate = f"Plate: {plate_text}"
                                    display_text_expiry = f"Expires: {expiry_text}"

                                    (w_plate, h_plate), _ = cv2.getTextSize(display_text_plate, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    (w_expiry, h_expiry), _ = cv2.getTextSize(display_text_expiry, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    max_w = max(w_plate, w_expiry)

                                    cv2.rectangle(frame, (xmin, ymin - (h_plate + h_expiry) - 20), (xmin + max_w + 10, ymin), color, -1)
                                    cv2.putText(frame, display_text_plate, (xmin + 5, ymin - h_expiry - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                    cv2.putText(frame, display_text_expiry, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                except Exception as e:
                                    plate_text = "ERROR"
                                    expiry_text = "ERROR"

                            else:
                                label = f'{classname}: {int(conf*100)}%'
                                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                label_ymin = max(ymin, labelSize[1] + 10)
                                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                            object_count += 1

                    # Display FPS and object count
                    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                    # Convert frame for Streamlit display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, use_container_width=True)
                    recorder.write(frame)

                    # Update FPS
                    t_stop = time.perf_counter()
                    frame_rate_calc = float(1/(t_stop - t_start))
                    if len(frame_rate_buffer) >= fps_avg_len:
                        frame_rate_buffer.pop(0)
                    frame_rate_buffer.append(frame_rate_calc)
                    avg_frame_rate = np.mean(frame_rate_buffer)

                    fps_display.text(f"Average FPS: {avg_frame_rate:.2f}")
                    object_count_display.text(f"Objects Detected: {object_count}")

                # Clean up
                cap.release()
                recorder.release()
                os.unlink(tmp_file_path)

                # Provide download link for processed video
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="output.mp4",
                        mime="video/mp4"
                    )
                os.unlink(output_path)
