import sys
import threading
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QMessageBox, QCheckBox, QSpinBox, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon
from yolo import YOLOApp
from window import DetectionWindow, ImageResultWindow

class SourceSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AegisLens")
        self.setWindowIcon(QIcon("Logo.png"))
        self.setGeometry(100, 100, 450, 550)  # Increased window size

        self.video_path = None
        self._should_stop = False
        self.yolo_window = None
        self.yolo_thread = None

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ===== Header Section =====
        try:
            logo_label = QLabel()
            logo_pixmap = QPixmap("Logo.png") 
            zoomed = logo_pixmap.scaled(300, 300)
            logo_label.setPixmap(zoomed)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(logo_label)
        except:
            logo_label = QLabel("YOLO DETECTION")
            logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            main_layout.addWidget(logo_label)

        # Program Name
        title_label = QLabel("AegisLens")
        title_label.setStyleSheet("font-size: 18px; color: #3498db;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Real-time vehicle object detection using YOLO with OCR for the license plate")
        desc_label.setStyleSheet("font-size: 14px; color: #7f8c8d;")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(desc_label)

        # Horizontal line separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #bdc3c7;")
        main_layout.addWidget(line)

        # ===== Configuration Section =====
        # Input source
        main_layout.addWidget(QLabel("Select Input Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Camera", "USB Camera", "Video File", "Picture"])
        self.source_combo.currentIndexChanged.connect(self.on_source_change)
        main_layout.addWidget(self.source_combo)

        # Video browse
        self.video_label = QLabel("No file selected.")
        self.video_label.setVisible(False)
        self.browse_button = QPushButton("Browse Video")
        self.browse_button.clicked.connect(self.browse_video)
        self.browse_button.setVisible(False)
        main_layout.addWidget(self.browse_button)
        main_layout.addWidget(self.video_label)

        # Resolution selector
        main_layout.addWidget(QLabel("Select Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1600x900"])
        main_layout.addWidget(self.resolution_combo)

        # Threshold
        main_layout.addWidget(QLabel("Confidence Threshold:"))
        self.thresh_spinbox = QSpinBox()
        self.thresh_spinbox.setRange(1, 100)
        self.thresh_spinbox.setValue(50)
        self.thresh_spinbox.setSuffix(" %")
        main_layout.addWidget(self.thresh_spinbox)

        # Checkboxes
        self.ocr_checkbox = QCheckBox("Enable OCR")
        self.ocr_checkbox.setChecked(True)
        main_layout.addWidget(self.ocr_checkbox)

        self.record_checkbox = QCheckBox("Record Output")
        main_layout.addWidget(self.record_checkbox)

        # Start button
        self.start_button = QPushButton("Start Inference")
        self.start_button.clicked.connect(self.start_inference)
        main_layout.addWidget(self.start_button)

        self.setLayout(main_layout)

    def on_source_change(self):
        source_text = self.source_combo.currentText()
        if source_text == "Video File":
            self.browse_button.setVisible(True)
            self.video_label.setVisible(True)
            self.browse_button.setText("Browse Video")
        elif source_text == "Picture":
            self.browse_button.setVisible(True)
            self.video_label.setVisible(True)
            self.browse_button.setText("Browse Image")
        else:
            self.browse_button.setVisible(False)
            self.video_label.setVisible(False)
            self.video_path = None
            self.video_label.setText("No file selected.")
            
    def browse_video(self):
        source_text = self.source_combo.currentText()
        if source_text == "Video File":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
        elif source_text == "Picture":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image File",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp)"
            )
        
        if file_path:
            self.video_path = file_path
            self.video_label.setText(f"Selected: {file_path}")
        else:
            self.video_label.setText("No file selected.")

    def start_inference(self):
        source_text = self.source_combo.currentText()
        
        if source_text == "Camera":
            src = 0
        elif source_text == "USB Camera":
            src = 1
        elif source_text == "Video File":
            if not self.video_path:
                QMessageBox.warning(self, "Warning", "Please select a video file.")
                return
            src = self.video_path
        elif source_text == "Picture":
            if not self.video_path:
                QMessageBox.warning(self, "Warning", "Please select an image file.")
                return
            self.process_image()
            return
        else:
            QMessageBox.warning(self, "Warning", "Invalid source.")
            return
        
        resolution_text = self.resolution_combo.currentText()
        width, height = map(int, resolution_text.split("x"))

        thresh = self.thresh_spinbox.value() / 100.0
        use_ocr = self.ocr_checkbox.isChecked()
        record = self.record_checkbox.isChecked()

        # Start the YOLO window
        self._should_stop = False
        self.detection_popup = DetectionWindow(
            model_path="latest.pt",
            source=src,
            resolution=(width, height),
            record=record,
            thresh=thresh,
            disable_ocr=not use_ocr
        )
        self.detection_popup.exec()

    def closeEvent(self, event):
        if hasattr(self, 'yolo_thread') and self.yolo_thread is not None:
            self.yolo_thread.stop()
        if hasattr(self, 'detection_popup') and self.detection_popup:
            self.detection_popup.close()
        event.accept()

    def process_image(self):
        image_path = self.video_path
        if not image_path:
            QMessageBox.warning(self, "Warning", "Please select an image file first.")
            return
    
        if self.yolo_thread is None:
            self.yolo_thread = YOLOApp(
                model_path="latest.pt",
                source=None, 
                resolution=None,
                record=False,
                thresh=self.thresh_spinbox.value() / 100.0,
                disable_ocr=not self.ocr_checkbox.isChecked()
            )
        
        result = self.yolo_thread.predict_image(image_path)
        
        if result:
            rgb_frame, object_count = result
            self.image_result_window = ImageResultWindow(self)
            self.image_result_window.show_results(rgb_frame, object_count)
        else:
            QMessageBox.warning(self, "Error", "Failed to process the image.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SourceSelector()
    window.show()
    ret = app.exec()
    
    if hasattr(window, 'detection_popup') and window.detection_popup:
        window.detection_popup.close()
    
    sys.exit(ret)
