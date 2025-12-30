import cv2
import torch
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import argparse

# add yolov5 repo to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project root: Ambulance-firetruck-detection-yolov5-main
if str(ROOT) not in os.sys.path:
    os.sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.dataloaders import LoadImages, LoadStreams

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights' / 'best.pt', help='model path(s)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    opt = parser.parse_args()
    return opt

opt = parse_opt()

# device
device = select_device('')  # '' = auto, cpu if no GPU

# load model
model = attempt_load(str(opt.weights), device=device)
model.conf = opt.conf_thres  # confidence threshold

LANE_COUNT = 2
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']

# Dataloader
if opt.source == '0':
    # Webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        lane_width = width // LANE_COUNT

        # Process image
        img = frame[:, :, ::-1]  # BGR -> RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 0–1
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # (1, H, W, 3) -> (1, 3, H, W)

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres=model.conf, iou_thres=0.45)[0]

        detections = pred
        labels = model.names

        lane_counts = [0] * LANE_COUNT
        ambulance_detected = [False] * LANE_COUNT
        total_vehicle_count = 0
        type_count = defaultdict(int)

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            class_id = int(cls)
            class_name = labels[class_id]

            # Calculate bounding box area percentage
            bbox_area = (x2 - x1) * (y2 - y1)
            total_area = width * height
            area_percentage = (bbox_area / total_area) * 100

            lane_idx = min(cx // lane_width, LANE_COUNT - 1)

            if class_name in ['ambulance', 'firetruck']:
                ambulance_detected[lane_idx] = True
            elif class_name in vehicle_classes:
                lane_counts[lane_idx] += 1
                total_vehicle_count += 1
                type_count[class_name] += 1

                # ✅ Print and speak vehicle type
                print(f"Detected a {class_name}")
                # os.system(f"say Detected a {class_name}")  # Commented out for Windows

            # Draw bounding box and label with confidence and area percentage
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f} ({area_percentage:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw 2 lanes and lane counts
        green_lane = None
        for i in range(LANE_COUNT):
            x_start = i * lane_width
            x_end = (i + 1) * lane_width
            color = (0, 0, 255)  # Red

            if ambulance_detected[i]:
                green_lane = i
                color = (0, 255, 0)

            cv2.rectangle(frame, (x_start, 0), (x_end, height), color, 2)
            cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (x_start + 10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Decision logic
        if green_lane is not None:
            decision = f"🚨 GREEN to Lane {green_lane+1} (Emergency)"
            red_lanes = [str(i+1) for i in range(LANE_COUNT) if i != green_lane]
        else:
            max_count = max(lane_counts)
            if max_count > 0:
                green_lane = lane_counts.index(max_count)
                decision = f"🚗 GREEN to Lane {green_lane+1} (Most Traffic)"
                red_lanes = [str(i+1) for i in range(LANE_COUNT) if i != green_lane]
            else:
                decision = "🟡 No traffic detected"
                red_lanes = []

        # Display GREEN/RED logic
        y_text = height - 90
        if green_lane is not None:
            cv2.putText(frame, f"🟢 GREEN: Lane {green_lane+1}", (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_text += 30
            if red_lanes:
                cv2.putText(frame, f"🔴 RED: Lane(s) {', '.join(red_lanes)}", (20, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, decision, (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show total vehicle count
        cv2.putText(frame, f"Total Vehicles: {total_vehicle_count}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Type-wise count
        y_offset = height - 100
        for vtype, count in type_count.items():
            cv2.putText(frame, f"{vtype.capitalize()}: {count}", (width - 220, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset -= 25

        # Show live frame
        cv2.imshow("Smart Traffic (2 Lanes)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

else:
    # For images/videos
    dataset = LoadImages(opt.source, img_size=640, stride=32, auto=True)
    for path, im, im0s, vid_cap, s in dataset:
        frame = im0s
        height, width, _ = frame.shape
        lane_width = width // LANE_COUNT

        # Preprocess image like in detect.py
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        with torch.no_grad():
            pred = model(im)[0]
            pred = non_max_suppression(pred, conf_thres=model.conf, iou_thres=0.45)[0]

        detections = pred
        detections[:, :4] = scale_boxes(im.shape[2:], detections[:, :4], frame.shape).round()
        labels = model.names

        lane_counts = [0] * LANE_COUNT
        ambulance_detected = [False] * LANE_COUNT
        type_count = defaultdict(int)

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            class_id = int(cls)
            class_name = labels[class_id]

            # Calculate bounding box area percentage
            bbox_area = (x2 - x1) * (y2 - y1)
            total_area = width * height
            area_percentage = (bbox_area / total_area) * 100

            lane_idx = min(cx // lane_width, LANE_COUNT - 1)

            if class_name in ['ambulance', 'firetruck']:
                ambulance_detected[lane_idx] = True
            elif class_name in vehicle_classes:
                lane_counts[lane_idx] += 1
                total_vehicle_count += 1
                type_count[class_name] += 1

                # ✅ Print and speak vehicle type
                print(f"Detected a {class_name}")
                # os.system(f"say Detected a {class_name}")  # Commented out for Windows

            # Draw bounding box and label with confidence and area percentage
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f} ({area_percentage:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw 2 lanes and lane counts
        green_lane = None
        for i in range(LANE_COUNT):
            x_start = i * lane_width
            x_end = (i + 1) * lane_width
            color = (0, 0, 255)  # Red

            if ambulance_detected[i]:
                green_lane = i
                color = (0, 255, 0)

            cv2.rectangle(frame, (x_start, 0), (x_end, height), color, 2)
            cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (x_start + 10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Decision logic
        if green_lane is not None:
            decision = f"🚨 GREEN to Lane {green_lane+1} (Emergency)"
            red_lanes = [str(i+1) for i in range(LANE_COUNT) if i != green_lane]
        else:
            max_count = max(lane_counts)
            if max_count > 0:
                green_lane = lane_counts.index(max_count)
                decision = f"🚗 GREEN to Lane {green_lane+1} (Most Traffic)"
                red_lanes = [str(i+1) for i in range(LANE_COUNT) if i != green_lane]
            else:
                decision = "🟡 No traffic detected"
                red_lanes = []

        # Display GREEN/RED logic
        y_text = height - 90
        if green_lane is not None:
            cv2.putText(frame, f"🟢 GREEN: Lane {green_lane+1}", (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_text += 30
            if red_lanes:
                cv2.putText(frame, f"🔴 RED: Lane(s) {', '.join(red_lanes)}", (20, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, decision, (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show total vehicle count
        cv2.putText(frame, f"Total Vehicles: {total_vehicle_count}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Type-wise count
        y_offset = height - 100
        for vtype, count in type_count.items():
            cv2.putText(frame, f"{vtype.capitalize()}: {count}", (width - 220, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset -= 25

        # Show frame
        cv2.imshow("Smart Traffic (2 Lanes)", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
