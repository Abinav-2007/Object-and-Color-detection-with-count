from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import sys
import streamlit as st
import logging
import os
import urllib.request

# --- Configuration ---
MODEL_URL = "https://your-link-to-yolov11n.pt"  # 🔥 <-- Replace this with your real model link
MODEL_PATH = "yolo11n.pt" # Using YOLOv11 model path
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KMEANS_CLUSTERS = 3
CONFIDENCE_THRESHOLD = 0.4
# --- End Configuration ---

# --- Download model if needed ---
def download_model():
    """Downloads the YOLO model if it's not already present."""
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Model file '{MODEL_PATH}' not found. Downloading from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logging.info(f"Model '{MODEL_PATH}' downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download model: {e}", exc_info=True)
            st.error(f"Fatal Error: Could not download YOLO model from {MODEL_URL}. Error: {e}")
            sys.exit(1)

@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model from the specified path."""
    try:
        download_model()  # 👈 Automatically ensure model is present
        logging.info(f"Attempting to load model: {model_path}...")
        model = YOLO(model_path)
        logging.info(f"Model '{model_path}' loaded successfully.")
        class_names = model.names
        logging.info(f"Model classes: {list(class_names.values())}")
        if not hasattr(model, 'predict') or not class_names:
            logging.error(f"Model loaded from {model_path} might not be a detection model or is missing class names.")
            st.error(f"Failed to initialize detection model from {model_path}. Check model compatibility.")
            sys.exit(1)
        return model, class_names
    except FileNotFoundError:
        logging.error(f"Error: Model file not found at {model_path}. Please ensure the file exists.")
        st.error(f"Fatal Error: Model file not found at {model_path}. Download or place the file correctly.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading YOLO model from {model_path}: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not load YOLO model from {model_path}. Check compatibility and file integrity. Error: {e}")
        sys.exit(1)

def get_dominant_color(image, k=KMEANS_CLUSTERS):
    """Finds the dominant color in an image using K-Means clustering."""
    if image is None or image.size == 0:
        logging.warning("get_dominant_color received empty image.")
        return np.array([0, 0, 0])

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] != 3:
        logging.warning(f"get_dominant_color received image with unexpected shape: {image.shape}")
        return np.array([0, 0, 0])

    try:
        pixels = image.reshape((-1, 3))
        if pixels.shape[0] == 0:
            logging.warning("Image became empty after reshape in get_dominant_color.")
            return np.array([0,0,0])

        pixels = np.float32(pixels)
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(pixels)
        counts = np.bincount(kmeans.labels_)

        if not counts.size > 0:
            logging.warning("KMeans resulted in empty counts.")
            return np.array([0, 0, 0])

        dominant_bgr = kmeans.cluster_centers_[np.argmax(counts)]
        return np.uint8(dominant_bgr)

    except cv2.error as cv_err:
        logging.error(f"OpenCV error in get_dominant_color: {cv_err}", exc_info=True)
        return np.array([0, 0, 0])
    except Exception as e:
        logging.error(f"Error in K-Means clustering for dominant color: {e}", exc_info=True)
        return np.array([0, 0, 0])

def classify_color(bgr_color):
    """Classifies a BGR color into a common color name using HSV thresholds."""
    if not isinstance(bgr_color, (np.ndarray, list, tuple)) or len(bgr_color) != 3:
        logging.warning(f"Invalid input to classify_color: {bgr_color}")
        return "unknown"

    pixel = np.uint8([[bgr_color]])
    try:
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]
    except Exception as e:
        logging.error(f"Error converting BGR to HSV in classify_color: {e}")
        return "unknown"

    if v < 60: return "black"
    if s < 50:
        return "white" if v > 190 else "gray"
    if h < 10 or h >= 170: return "red"
    elif 10 <= h < 25: return "orange"
    elif 25 <= h < 35: return "yellow"
    elif 35 <= h < 85: return "green"
    elif 85 <= h < 100: return "cyan"
    elif 100 <= h < 135: return "blue"
    elif 135 <= h < 160: return "purple"
    elif 160 <= h < 170: return "pink"
    else: return "unknown"

def analyze_image(image, model, class_names_map):
    """Analyzes an image using the YOLO model to detect objects and their colors."""
    if image is None:
        logging.error("analyze_image received None image.")
        return None, [], defaultdict(int)

    individual_detections = []
    object_counts = defaultdict(int)
    annotated_image = image.copy()

    logging.info(f"Starting analysis with confidence threshold: {CONFIDENCE_THRESHOLD}")
    detection_count_total = 0
    detection_count_threshold = 0

    try:
        results = model(image, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for r in results:
            boxes = r.boxes
            detection_count_total += len(boxes)

            for box in boxes:
                detection_count_threshold += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = class_names_map.get(cls, f"Unknown Class ({cls})")

                pad = 5
                crop_y1 = max(0, y1 - pad)
                crop_y2 = min(annotated_image.shape[0], y2 + pad)
                crop_x1 = max(0, x1 - pad)
                crop_x2 = min(annotated_image.shape[1], x2 + pad)

                cropped_object = image[crop_y1:crop_y2, crop_x1:crop_x2]

                if cropped_object.size == 0:
                    logging.warning(f"Skipping empty crop for {class_name} at [{x1},{y1},{x2},{y2}]")
                    continue

                dominant_bgr = get_dominant_color(cropped_object)
                color_name = classify_color(dominant_bgr)

                detection_info = {
                    "Color": color_name.capitalize(),
                    "Object": class_name.capitalize(),
                    "Confidence": f"{conf:.2f}"
                }
                individual_detections.append(detection_info)

                object_counts[(color_name, class_name)] += 1

                box_color_bgr = (int(dominant_bgr[0]), int(dominant_bgr[1]), int(dominant_bgr[2]))
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color_bgr, 2)

                label = f"{color_name.capitalize()} {class_name.capitalize()} ({conf:.2f})"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
                cv2.rectangle(annotated_image, (x1, label_y - h - 5), (x1 + w + 5, label_y), box_color_bgr, -1)
                cv2.putText(annotated_image, label, (x1 + 5, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        logging.info(f"Analysis complete. Found {detection_count_threshold} objects meeting threshold (out of {detection_count_total} potentials).")
        if detection_count_threshold == 0:
            logging.info("No objects detected meeting the confidence threshold.")

        return annotated_image, individual_detections, object_counts

    except Exception as e:
        logging.error(f"Error during image analysis: {e}", exc_info=True)
        return image, [], defaultdict(int)
