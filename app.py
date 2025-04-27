import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import io

# Import functions and variables from object_detector.py
from object_detector import (
    load_yolo_model,
    analyze_image,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD
)

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')  # Set WARNING level for production

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Object Detector UI", layout="wide")

st.title("📦 Object Color & Type Detector")
st.write("""
Use your webcam or upload an image file. Click **Analyze** to detect objects,
identify their dominant color, and see the summary.
Providing a new image (capture or upload) clears previous analysis results.
""")
st.markdown("---")

# --- Session State Initialization ---
for key in [
    'captured_image', 'annotated_image', 'individual_detections',
    'object_counts', 'error_message', 'image_source_caption'
]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.image_source_caption is None:
    st.session_state.image_source_caption = "No image provided yet."
# --- End Session State ---


# --- Load Model ---
try:
    model, class_names = load_yolo_model(MODEL_PATH)
except SystemExit:
    st.error("Critical error: Model could not be loaded. Application halted.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading model: {e}")
    st.stop()
# --- End Load Model ---


# --- Sidebar Controls ---
st.sidebar.header("⚙️ Input & Actions")

# Webcam Input
st.sidebar.subheader("Webcam Input")
device_index = st.sidebar.number_input("Camera Device Index", min_value=0, max_value=10, value=0, step=1)
capture_button = st.sidebar.button("📸 Capture Image", key="capture")

# File Upload Input
st.sidebar.markdown("---")
st.sidebar.subheader("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

# Analysis Action
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis")
analyze_button = st.sidebar.button(
    "🔍 Analyze Image",
    key="analyze",
    type="primary",
    disabled=(st.session_state.captured_image is None)
)

# Model Info
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Info")
st.sidebar.caption(f"Using Model: `{MODEL_PATH.split('/')[-1]}`")
st.sidebar.caption(f"Confidence Threshold: `{CONFIDENCE_THRESHOLD}`")
# --- End Sidebar Controls ---


# --- Input Handling Logic ---

def clear_analysis_results():
    """Clear previous analysis results."""
    for key in ['annotated_image', 'individual_detections', 'object_counts', 'error_message']:
        st.session_state[key] = None


# Webcam Capture
if capture_button:
    clear_analysis_results()
    st.session_state.captured_image = None

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        st.session_state.error_message = f"Error: Could not open camera device {device_index}."
    else:
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            st.session_state.error_message = f"Error: Failed to grab frame from device {device_index}."
        else:
            st.session_state.captured_image = frame
            st.session_state.image_source_caption = f"Image captured from Webcam {device_index}"
            st.toast("Image captured successfully!", icon="📸")
            st.rerun()

# File Upload
if uploaded_file is not None:
    if 'last_uploaded_filename' not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
        clear_analysis_results()
        st.session_state.captured_image = None

        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if opencv_image is None:
                st.session_state.error_message = f"Error: Could not decode uploaded file '{uploaded_file.name}'."
                st.session_state.last_uploaded_filename = None
            else:
                st.session_state.captured_image = opencv_image
                st.session_state.image_source_caption = f"Image uploaded: {uploaded_file.name}"
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.toast("Image uploaded successfully!", icon="📄")
                st.rerun()

        except Exception as e:
            st.session_state.error_message = f"Error processing uploaded file: {e}"
            st.session_state.last_uploaded_filename = None


# Analyze Button
if analyze_button:
    if st.session_state.captured_image is not None:
        st.session_state.error_message = None

        with st.spinner("🧠 Analyzing image... Please wait."):
            annotated_img, ind_detect, obj_counts = analyze_image(
                st.session_state.captured_image, model, class_names
            )

            if annotated_img is None:
                st.session_state.error_message = "Image analysis failed unexpectedly."
            else:
                st.session_state.annotated_image = annotated_img
                st.session_state.individual_detections = ind_detect
                st.session_state.object_counts = obj_counts
                st.toast("Analysis complete!", icon="✨")
                st.rerun()
    else:
        st.warning("Cannot analyze - no image provided.")


# --- Main Area Display ---

# Display Errors
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# Display Images
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Image Preview")
    if st.session_state.captured_image is not None:
        st.image(
            cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB),
            caption=st.session_state.image_source_caption,
            use_container_width=True
        )
    else:
        st.info("Provide an image via Webcam Capture or File Upload.")

with col2:
    st.subheader("🎨 Analysis Result")
    if st.session_state.annotated_image is not None:
        st.image(
            cv2.cvtColor(st.session_state.annotated_image, cv2.COLOR_BGR2RGB),
            caption="Detections from Last Analysis",
            use_container_width=True
        )
    elif st.session_state.captured_image is not None:
        st.info("Image provided, but not analyzed yet.")
    else:
        st.info("Provide and analyze an image.")

st.markdown("---")

# Display Detection Details and Summary
col3, col4 = st.columns(2)

with col3:
    st.subheader("📋 Detected Objects List")
    if st.session_state.individual_detections:
        df_detections = pd.DataFrame(st.session_state.individual_detections)
        st.dataframe(df_detections, use_container_width=True, height=300)
    else:
        st.info("No detection results available yet.")

with col4:
    st.subheader("📊 Detection Summary")
    if st.session_state.object_counts:
        summary_list = []
        sorted_counts = sorted(
            st.session_state.object_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1])
        )
        for (color, name), count in sorted_counts:
            plural = "s" if count > 1 else ""
            summary_list.append(f"- **{count}** {color.capitalize()} {name.capitalize()}{plural}")
        st.markdown("\n".join(summary_list))
    else:
        st.info("No detection summary available yet.")
# --- End Main Area Display ---
