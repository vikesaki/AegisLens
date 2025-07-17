# Vehicle Object Detection and License Plate Recognition (LPR) System

![Project Banner](banner.png)

## What is AegisLens?
AegisLens is a model that can detect vehicle types and read their license plate. It utilizes an object detection model and an Optical Character Recognition (OCR) model to output it’s results.

## How does it work?
1. **Object Detection**: The model detects vehicles in the video footage and identifies their bounding boxes.
2. **License Plate Recognition**: For each detected vehicle, the model extracts the license plate region and applies OCR to read the text.
3. **Expiry Date Extraction**: The model processes the license plate text to extract the expiry date, if available.
4. **Output**: The model displays the detected vehicle type, license plate number, and expiry date on the video footage.

## Tools for model development:
- **YOLOv8**: A object detection model used for detecting vehicles in the video footage.
- **EasyOCR**: An OCR model used for reading text from the detected license plates.
- **Streamlit**: A web application framework used to create the user interface for the application.

## Features:
- **Exploratory Data Analysis**: View insights and statistics about the dataset used for training the model.
- **Prediction**: Upload a video file to detect vehicles and read license plates.

## Repository Overview

```
.
├── AEGISLENS_EDA_ENG.ipynb           # Exploratory Data Analysis in English
├── AEGISLENS_EDA_ID.ipynb            # Exploratory Data Analysis in Indonesian
├── Archive/                          # Archived files or previous versions
├── data.yaml                         # YOLO dataset configuration file
├── demo1.avi                         # Sample demo video for testing
├── Deployment/                       # Scripts or configs related to model deployment
├── inference.py                      # Inference script using trained YOLO model
├── latest.pt                         # Trained YOLO model weights
├── metrics/                          # Evaluation metrics and visualizations
├── ocr_plate.ipynb                   # OCR pipeline for plate recognition
├── README.md                         # Project overview and instructions
├── References/                       # External references or citations
├── requirements.txt                  # Python dependencies
├── splitting.py                      # Script to split dataset/video frames
├── take_frames_from_video.ipynb      # Notebook to extract frames from video
├── test.MOV                          # Test video sample
├── test2.MOV                         # Another test video sample
└── yolo_implementation.ipynb         # YOLOv5 implementation and training notebook

```

## Project Description

This system combines YOLO (You Only Look Once) object detection with license plate recognition to:

1. Detect vehicles in images/video streams
2. Identify vehicle classes (car, truck, motorcycle, etc.)
3. Optionally recognize license plate details including expiration date

## How to Run Inference

### 1. Initial Setup

#### Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

#### Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running Detection

#### Basic Command Structure:
```bash
python inference.py \
    --model <model_path> \
    --source <input_source> \
    [--thresh <confidence_threshold>] \
    [--resolution <WxH>] \
    [--record]
```

### 3. Detailed Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--model` | Path to YOLO model file (e.g., "runs/detect/train/weights/best.pt") | Yes | - |
| `--source` | Input source: <br>- Image file ("test.jpg") <br>- Image folder ("test_dir") <br>- Video file ("testvid.mp4") <br>- USB camera ("usb0") <br>- Picamera ("picamera") | Yes | - |
| `--thresh` | Minimum confidence threshold (0.0 to 1.0) | No | 0.5 |
| `--resolution` | Display resolution in WxH format (e.g., "640x480") | No | Source resolution |
| `--record` | Record and save output as "demo.avi" (requires --resolution) | No | False |

### 4. Usage Examples

**Basic image detection:**
```bash
python inference.py --model latest.pt --source test.jpg
```

**Webcam detection with custom threshold:**
```bash
python inference.py --model latest.pt --source usb0 --thresh 0.7
```

**Video processing with recording:**
```bash
python inference.py --model latest.pt --source traffic.mp4 --resolution 1280x720 --record
```

**Batch process image folder:**
```bash
python inference.py --model latest.pt --source images/ --thresh 0.6
```

### 5. Important Notes

1. For recording (`--record`), you must specify `--resolution`
2. Camera indices start from 0 (e.g., "usb0" for first USB camera)
3. Confidence threshold (`--thresh`) filters out low-confidence detections
4. Supported image formats: JPG, PNG, BMP
5. Supported video formats: MP4, AVI, MOV

### 6. Troubleshooting

**Webcam not working:**
- Try different indices ("usb0", "usb1", etc.)
- Verify camera permissions
- Check if OpenCV can access the camera:
  ```python
  import cv2
  cap = cv2.VideoCapture(0)
  print(cap.isOpened())
  ```

# Graphical User Interface (GUI) Application
Run the interactive desktop application with:
```
python UIApp/main_app.py
```

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended)
- CUDA/cuDNN (for GPU acceleration)

### Setup
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/vehicle-plate-detection.git
   cd vehicle-plate-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Applications

- Automated vehicle verification systems
- Parking management
- Traffic monitoring
- Insurance fraud detection
- Expired plate detection 

Project Link: [click here](https://github.com/vikesaki/AegisLens#)

Deployment Link: [click here](https://aegislens.streamlit.app/)

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.   

**You must**:  
- Include original license  
- Maintain copyright notice

**No warranty** is provided. Use at your own risk.

## Contributors
- [Lis Wahyuni](https://github.com/liswahyuni) - Data Scientist (liswyuni@gmail.com)  
- [Faishal Kemal](https://github.com/vikesaki) - Data Scientist & Data Annotator (faishalkemal68@gmail.com)
- [Muhammad Rafi Abhinaya](https://github.com/RafiAbhinaya) - Data Scientist & Data Annotator (mr.abhinaya26@gmail.com)
- [Ma'ruf Habibie Siregar](https://github.com/HbbSiregar) - Data Analyst & Data Annotator  (maruf.habibie.siregar@gmail.com)  
- [Fauzan Rahmat Farghani](https://github.com/fauzanfarghani) - Data Engineer & Data Annotator (fauzanf78@rocketmail.com)    
