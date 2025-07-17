import os
import sys
import glob
import time

import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import re

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
    return "INCORRECT LENGTH"

def run_detection(model_path, source, resolution=None, record=False, thresh=0.5, stop_flag=lambda: False):
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    model = YOLO(model_path, task='detect')
    reader = easyocr.Reader(['en'], gpu=True)
    labels = model.names

    img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
    vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

    if os.path.isdir(source):
        source_type = 'folder'
    elif os.path.isfile(source):
        _, ext = os.path.splitext(source)
        if str.lower(ext) in img_ext_list:
            source_type = 'image'
        elif str.lower(ext) in vid_ext_list:
            source_type = 'video'
        else:
            print(f'File extension {ext} is not supported.')
            return
    elif 'usb' in source:
        source_type = 'usb'
        usb_idx = int(source[3:])
    elif 'picamera' in source:
        source_type = 'picamera'
        picam_idx = int(source[8:])
    else:
        print(f'Input {source} is invalid.')
        return

    resize = False
    if resolution:
        resW, resH = map(int, resolution.lower().split('x'))
        resize = True

    if record:
        if source_type not in ['video','usb']:
            print('Recording only works for video and camera sources.')
            return
        if not resolution:
            print('Please specify resolution to record video at.')
            return
        recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW,resH))

    if source_type == 'image':
        imgs_list = [source]
    elif source_type == 'folder':
        imgs_list = [f for f in glob.glob(source + '/*') if os.path.splitext(f)[1] in img_ext_list]
    elif source_type == 'video' or source_type == 'usb':
        cap_arg = source if source_type == 'video' else usb_idx
        cap = cv2.VideoCapture(cap_arg)
        if resize:
            cap.set(3, resW)
            cap.set(4, resH)
    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
        cap.start()

    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
                   (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    avg_frame_rate = 0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0

    while not stop_flag():
        t_start = time.perf_counter()

        if source_type in ['image', 'folder']:
            if img_count >= len(imgs_list):
                print('All images processed.')
                break
            frame = cv2.imread(imgs_list[img_count])
            img_count += 1
        elif source_type in ['video', 'usb']:
            ret, frame = cap.read()
            if not ret:
                print('End of video or camera stream.')
                break
        elif source_type == 'picamera':
            frame = cap.capture_array()
            if frame is None:
                break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        results = model(frame, verbose=False)
        detections = results[0].boxes
        object_count = 0

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf > float(thresh):
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                if classidx == 2:
                    try:
                        box_height = ymax - ymin
                        split_upper = ymin + int(box_height * 0.70)
                        split_lower = ymin + int(box_height * 0.55)
                        plate_crop = frame[ymin:split_upper, xmin:xmax]
                        expiry_crop = frame[split_lower:ymax, xmin:xmax]

                        plate_text = "".join(reader.readtext(plate_crop, detail=0, paragraph=False)).upper().replace(" ", "")
                        processed_expiry = preprocess_expiry_crop(expiry_crop)

                        if processed_expiry is not None:
                            expiry_ocr = reader.readtext(processed_expiry, detail=0, paragraph=False, allowlist='0123456789.')
                            expiry_text = extract_and_validate_date("".join(expiry_ocr))
                        else:
                            expiry_text = "EMPTY"
                    except:
                        plate_text = expiry_text = "ERROR"

                    cv2.rectangle(frame, (xmin, ymin - 40), (xmin + 200, ymin), color, -1)
                    cv2.putText(frame, f"Plate: {plate_text}", (xmin + 5, ymin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(frame, f"Expires: {expiry_text}", (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                else:
                    label = f'{classname}: {int(conf*100)}%'
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                object_count += 1

        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.imshow('YOLO Detection', frame)
        if record: recorder.write(frame)
        
        if cv2.getWindowProperty('YOLO Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(5 if source_type != 'image' else 0)
        if key in [ord('q'), ord('Q')]: break
        elif key in [ord('s'), ord('S')]: cv2.waitKey()
        elif key in [ord('p'), ord('P')]: cv2.imwrite('capture.png', frame)

        frame_rate_calc = 1 / (time.perf_counter() - t_start)
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)

    if source_type in ['video', 'usb']: cap.release()
    elif source_type == 'picamera': cap.stop()
    if record: recorder.release()
    cv2.destroyAllWindows()

    print(f'Average FPS: {avg_frame_rate:.2f}')
