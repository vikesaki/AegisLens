from PyQt6.QtWidgets import (QDialog, QLabel, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFrame, QSizePolicy, QFileDialog, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import Qt, QSize
import cv2
from yolo import YOLOApp

class DetectionWindow(QDialog):
    def __init__(self, model_path, source, resolution, record, thresh, disable_ocr):
        super().__init__()
        self.setWindowTitle("AegisLens - Detection Viewer")
        self.setWindowIcon(QIcon("Logo.png"))
        
        # Store the original resolution
        self.source_resolution = resolution
        self.video_ended = False
        
        # Set window size with some padding
        self.setMinimumSize(600, 500)  # Larger minimum size
        self.resize(resolution[0] + 150, resolution[1] + 200)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # ===== Header Section =====
        header = QHBoxLayout()
        
        # Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap("Logo.png")
        logo_pixmap = logo_pixmap.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        header.addWidget(logo_label)
        
        # Title
        title = QLabel("AegisLens Detection")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.addWidget(title)
        
        header.addStretch()
        
        main_layout.addLayout(header)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #ddd;")
        main_layout.addWidget(line)
        
        # ===== Video Display Section =====
        video_container = QHBoxLayout()
        video_container.addStretch()
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(resolution[0] // 2, resolution[1] // 2)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #444;")
        video_container.addWidget(self.video_label)
        
        video_container.addStretch()
        main_layout.addLayout(video_container, stretch=1)
        
        # ===== Controls Section =====
        controls = QHBoxLayout()
        
        # Stats panel
        stats = QVBoxLayout()
        self.fps_label = QLabel("FPS: -")
        self.fps_label.setFont(QFont("Arial", 10))
        self.count_label = QLabel("Objects Detected: -")
        self.count_label.setFont(QFont("Arial", 10))
        stats.addWidget(self.fps_label)
        stats.addWidget(self.count_label)
        controls.addLayout(stats)
        
        controls.addStretch()
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.snapshot_btn = QPushButton("Take Snapshot")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        btn_layout.addWidget(self.snapshot_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_layout.addWidget(self.pause_btn)
        
        controls.addLayout(btn_layout)
        main_layout.addLayout(controls)
        
        self.setLayout(main_layout)

        # Initialize YOLO thread
        self.yolo_thread = YOLOApp(
            model_path=model_path,
            source=source,
            resolution=resolution,
            record=record,
            thresh=thresh,
            disable_ocr=disable_ocr
        )
        self.yolo_thread.frame_data.connect(self.update_frame)
        self.yolo_thread.finished.connect(self.on_thread_finished)
        self.yolo_thread.start()

    def update_frame(self, frame, fps, count):
        if self.video_ended:
            return
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.count_label.setText(f"Objects Detected: {count}")

    def resizeEvent(self, event):
        new_size = event.size()
        available_width = new_size.width() - 100
        available_height = new_size.height() - 200
        
        aspect_ratio = self.source_resolution[0] / self.source_resolution[1]
        
        if available_width / available_height > aspect_ratio:
            height = available_height
            width = int(height * aspect_ratio)
        else:
            width = available_width
            height = int(width / aspect_ratio)
        
        self.video_label.setFixedSize(width, height)
        super().resizeEvent(event)

    def on_thread_finished(self):
        self.video_ended = True
        self.count_label.setText(f"{self.count_label.text()} | Video Ended")
        self.pause_btn.setEnabled(False)

    def take_snapshot(self):
        if hasattr(self.video_label, 'pixmap') and self.video_label.pixmap():
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Snapshot", "", "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)")
            if file_path:
                self.video_label.pixmap().save(file_path)

    def toggle_pause(self):
        if self.pause_btn.isChecked():
            self.yolo_thread.pause()
            self.pause_btn.setText("Resume")
        else:
            self.yolo_thread.resume()
            self.pause_btn.setText("Pause")

    def closeEvent(self, event):
        if self.yolo_thread.isRunning():
            self.yolo_thread.stop()
            self.yolo_thread.wait()
        event.accept()
        
# ------------------------------------------------------------------------------------
# Image implementation
class ImageResultWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AegisLens - Detection Viewer")
        self.setWindowIcon(QIcon("Logo.png"))

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # ===== Header Section =====
        header = QHBoxLayout()
        
        # Logo
        logo_label = QLabel()
        logo_pixmap = QPixmap("Logo.png")
        logo_pixmap = logo_pixmap.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        header.addWidget(logo_label)
        
        # Title
        title = QLabel("AegisLens Detection")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.addWidget(title)
        
        header.addStretch()
        
        main_layout.addLayout(header)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #ddd;")
        main_layout.addWidget(line)
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.info_label)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        self.layout.addLayout(button_layout)
        
        self.image_data = None

    def show_results(self, rgb_frame, object_count):
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.info_label.setText(f"Objects detected: {object_count}")
        self.image_data = rgb_frame
        self.exec()
    
    def save_image(self):
        if self.image_data is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Processed Image",
                "",
                "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)"
            )
            
            if file_path:
                # Convert back to BGR for OpenCV saving
                bgr_frame = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr_frame)
                QMessageBox.information(self, "Success", "Image saved successfully!")