import time
import cv2
import numpy as np
import re
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import easyocr

def preprocess_expiry_crop(crop):
    if crop is None or crop.shape[0] < 5 or crop.shape[1] < 10:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(2.0, 50.0 / gray.shape[0])
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_and_validate_date(text):
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
    return "UNKNOWN"

class YOLOApp(QThread):
    frame_data = pyqtSignal(np.ndarray, float, int)  # frame, fps, object count

    def __init__(
        self,
        model_path,
        source,
        resolution=None,
        record=False,
        thresh=0.5,
        disable_ocr=False
    ):
        super().__init__()
        self.model = YOLO(model_path)
        self.labels = self.model.names
        self.disable_ocr = disable_ocr
        self.reader = None if disable_ocr else easyocr.Reader(['en'], gpu=True)
        self.cap = cv2.VideoCapture(source)

        # Thread control flags
        self.running = True
        self.paused = False

        if resolution:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Requested: {resolution}, Actual: ({actual_width}, {actual_height})")

        self.thresh = thresh
        self.record = record
        self.video_writer = None

        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = 'output_' + time.strftime('%Y%m%d_%H%M%S') + '.mp4'
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        self.bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133),
                          (88,159,106), (96,202,231), (159,124,168), (169,162,241),
                          (98,118,150), (172,176,184)]
        self.fps_buffer = []
        self.fps_avg_len = 100

    def run(self):
        while self.running and self.cap.isOpened():
            if self.paused:
                time.sleep(0.1)
                continue
                
            t_start = time.perf_counter()
            ret, frame = self.cap.read()
            
            if not ret:
                print("Video ended or capture failed")
                # Emit a black frame when video ends
                black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                self.frame_data.emit(black_frame, 0.0, 0)
                break

            results = self.model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0

            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                classname = self.labels[classidx]
                conf = detections[i].conf.item()

                if conf > self.thresh:
                    color = self.bbox_colors[classidx % 10]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    if classidx == 2 and not self.disable_ocr:
                        try:
                            box_height = ymax - ymin
                            split_upper = ymin + int(box_height * 0.70)
                            split_lower = ymin + int(box_height * 0.55)
                            plate_crop = frame[ymin:split_upper, xmin:xmax]
                            expiry_crop = frame[split_lower:ymax, xmin:xmax]

                            plate_text = "".join(self.reader.readtext(plate_crop, detail=0)).upper().replace(" ", "")
                            processed_expiry = preprocess_expiry_crop(expiry_crop)

                            if processed_expiry is not None:
                                expiry_ocr = self.reader.readtext(processed_expiry, detail=0, allowlist='0123456789.')
                                expiry_text = extract_and_validate_date("".join(expiry_ocr))
                            else:
                                expiry_text = "EMPTY"
                        except Exception as e:
                            print(f"OCR Error: {str(e)}")
                            plate_text = expiry_text = "ERROR"

                        cv2.rectangle(frame, (xmin, ymin - 40), (xmin + 200, ymin), color, -1)
                        cv2.putText(frame, f"Plate: {plate_text}", (xmin + 5, ymin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(frame, f"Expires: {expiry_text}", (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    else:
                        label = f'{classname}: {int(conf*100)}%'
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    object_count += 1

            fps = 1 / (time.perf_counter() - t_start)
            self.fps_buffer.append(fps)
            if len(self.fps_buffer) > self.fps_avg_len:
                self.fps_buffer.pop(0)
            avg_fps = np.mean(self.fps_buffer)

            if self.record and self.video_writer:
                self.video_writer.write(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_data.emit(rgb_frame, avg_fps, object_count)

        self.cleanup()

    def pause(self):
        """Pause the video processing"""
        self.paused = True

    def resume(self):
        """Resume the video processing"""
        self.paused = False

    def stop(self):
        """Stop the thread gracefully"""
        self.running = False
        if not self.wait(500):  # Wait up to 500ms for thread to finish
            print("Warning: Thread did not stop gracefully")
        self.cleanup()

    def cleanup(self):
        """Release all resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'video_writer') and self.record and self.video_writer:
            self.video_writer.release()
        print("YOLO thread cleanup complete")
        
# ---------------------------------------------------------------------------------------------
# Picture

    def predict_image(self, image_path):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: Could not read image {image_path}")
                return None

            results = self.model(frame, verbose=False)
            detections = results[0].boxes
            object_count = 0

            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                classidx = int(detections[i].cls.item())
                classname = self.labels[classidx]
                conf = detections[i].conf.item()

                if conf > self.thresh:
                    color = self.bbox_colors[classidx % 10]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    if classidx == 2 and not self.disable_ocr:
                        try:
                            box_height = ymax - ymin
                            split_upper = ymin + int(box_height * 0.70)
                            split_lower = ymin + int(box_height * 0.55)
                            plate_crop = frame[ymin:split_upper, xmin:xmax]
                            expiry_crop = frame[split_lower:ymax, xmin:xmax]

                            plate_text = "".join(self.reader.readtext(plate_crop, detail=0)).upper().replace(" ", "")
                            processed_expiry = preprocess_expiry_crop(expiry_crop)

                            if processed_expiry is not None:
                                expiry_ocr = self.reader.readtext(processed_expiry, detail=0, allowlist='0123456789.')
                                expiry_text = extract_and_validate_date("".join(expiry_ocr))
                            else:
                                expiry_text = "EMPTY"
                        except Exception as e:
                            print(f"OCR Error: {str(e)}")
                            plate_text = expiry_text = "ERROR"

                        cv2.rectangle(frame, (xmin, ymin - 40), (xmin + 200, ymin), color, -1)
                        cv2.putText(frame, f"Plate: {plate_text}", (xmin + 5, ymin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(frame, f"Expires: {expiry_text}", (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    label = f'{classname}: {int(conf*100)}%'
                    cv2.putText(frame, label, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    object_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb_frame, object_count

        except Exception as e:
            print(f"Image prediction error: {str(e)}")
            return None