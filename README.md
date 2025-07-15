# eak

## How to run inference
1. with license plate detail + expiration date
`python inference.py --model=35epoch.pt --source=*filename*`

2. only classes
`python yolo_detect.py --model=runs/detect/train/weights/best.pt --source=*filename*`

use `--source=usb0` for using webcam

use `latest.pt` for the newest model
