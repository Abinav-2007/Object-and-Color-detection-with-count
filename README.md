
Object Color and Type Detector
Overview

This project is a web application that detects objects in images and identifies the main color of each object. You can use your webcam or upload an image file as input. The application uses the YOLOv11n model for detection and K-Means clustering for color analysis, with a user interface built using Streamlit.

Features

Detects objects in images using the YOLOv11n model.

Determines the dominant color for each detected object.

Uses K-Means clustering on pixel data (BGR format) to find the dominant color value.

Names the dominant color (e.g., Red, Blue, Green) using HSV color space rules.

Accepts input from either a webcam or an uploaded image file (JPG, PNG, etc.).

Provides an interactive web interface built with Streamlit.

Shows results clearly:

A preview of the input image.

The image with colored boxes drawn around detected objects.

A list detailing each detected object, its color, and confidence score.

A summary counting the detected objects by color and type.

How it Works

Input: The user provides an image through the Streamlit interface, either via webcam or file upload.

Model Loading: The YOLOv11n model is loaded using the Ultralytics library.

Analysis: The user clicks the "Analyze Image" button.

Object Detection: The YOLOv11n model processes the image to find objects, their classes, and bounding boxes.

Color Processing: For each detected object:

The area inside the bounding box is cropped.

K-Means clustering is applied to the BGR pixel colors in the crop to find the dominant BGR color value.

This BGR color is converted to HSV format.

The HSV values are used to determine a common color name.

Output Generation:

An annotated image is created with colored bounding boxes.

A list of detection details is generated.

A summary count is calculated.

Display: The Streamlit interface shows the annotated image, the details list, and the summary.

Technologies Used

Programming Language: Python 3

Object Detection Model: YOLOv11n (via Ultralytics)

Libraries:

Streamlit: For the web application interface.

Ultralytics: For loading and running the YOLO model.

OpenCV (opencv-python): For image capture, processing, color conversions, and drawing.

NumPy: For numerical operations and handling image data arrays.

Scikit-learn (sklearn): For the KMeans clustering algorithm.

Pandas: For displaying the detection list in a table.

Collections (defaultdict): For counting detections.

Logging: For status messages during execution.

io: For handling file uploads.

Setup

Get the Code:
Download or clone the project files to a folder on your computer.

Install Dependencies:
Open your terminal or command prompt, navigate to the project folder, and install the required libraries:

pip install streamlit opencv-python ultralytics numpy scikit-learn pandas Pillow


Download YOLOv11n Model:

Important: You MUST download the yolov11n.pt model file yourself. This program will not download it automatically.

Find the source where you obtained the YOLOv11n model information (e.g., a specific website or repository).

Download the yolov11n.pt file.

Place this downloaded yolov11n.pt file directly into the main project folder (where app.py and object_detector.py are located).

Run the Application:
In your terminal or command prompt, while inside the project folder, run:

streamlit run app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Usage

After running the command, Streamlit will provide a local URL (like http://localhost:8501). Open this URL in your web browser.

Use the sidebar to provide an image:

Webcam: Select the camera index and click "Capture Image".

File Upload: Click "Browse files" and choose an image from your computer.

Once an image shows in the "Image Preview", click the "Analyze Image" button in the sidebar.

View the results displayed in the main area of the application.

Project Structure

object_detector.py - Contains code for loading the model, detecting objects, and analyzing colors.

app.py - Contains the Streamlit user interface code and handles user interactions.

yolov11n.pt - The YOLOv11n model file you need to download and place here.

README.md - This file.
