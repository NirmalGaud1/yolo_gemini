#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path

# Configure Streamlit page
st.set_page_config(page_title="YOLOv8 + Gemini Vision", layout="wide")
st.title("Object Detection & Scene Understanding")
st.markdown("Upload an image/video - YOLOv8 detects objects, Gemini describes the content")

# Initialize models
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemini-pro-vision')

yolo_model = load_yolo()
gemini_model = load_gemini()

# File upload section
uploaded_file = st.file_uploader("Choose a file...", 
                               type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
                               accept_multiple_files=False)

def process_image(image):
    """Process single image with YOLOv8 and Gemini"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    # YOLOv8 Detection
    with st.spinner("Detecting objects..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = yolo_model.predict(tmp.name)
            os.unlink(tmp.name)

    # Process results
    detected_objects = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            conf = float(box.conf[0])
            detected_objects.append(f"{label} ({conf:.2f})")

    # Display detection
    with col1:
        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption="Detected Objects", use_column_width=True)

    # Gemini Analysis
    with col2:
        st.subheader("Scene Understanding")
        if detected_objects:
            prompt = f"""Analyze this image and its detected objects: {', '.join(detected_objects)}.
            Describe the scene in detail including objects, activities, and context."""
            
            try:
                response = gemini_model.generate_content([prompt, image])
                st.write(response.text)
            except Exception as e:
                st.error(f"Gemini API Error: {str(e)}")
        else:
            st.warning("No objects detected - Analysis skipped")

def process_video(video_path):
    """Process video file with YOLOv8 and Gemini"""
    st.subheader("Video Analysis Results")
    
    # Process video with YOLOv8
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_video:
        results = yolo_model.predict(video_path, 
                                   save=True, 
                                   project=tempfile.gettempdir(),
                                   name="",
                                   exist_ok=True)
        
        # Get processed video path
        processed_video = Path(results[0].save_dir) / f"{Path(video_path).stem}.mp4"
    
    # Display processed video
    st.video(str(processed_video))

    # Collect detection data from all frames
    all_objects = []
    for result in results:
        frame_objects = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            frame_objects.append(label)
        all_objects.extend(frame_objects)
    
    # Generate video summary with Gemini
    if all_objects:
        with st.spinner("Generating video summary..."):
            unique_objects = list(set(all_objects))
            object_counts = {obj: all_objects.count(obj) for obj in unique_objects}
            
            prompt = f"""Analyze this video content based on detected objects:
            Total frames: {len(results)}
            Object counts: {object_counts}
            Describe the main activities, key objects, and overall context of the video."""
            
            try:
                response = gemini_model.generate_content(prompt)
                st.subheader("Video Summary")
                st.write(response.text)
            except Exception as e:
                st.error(f"Gemini API Error: {str(e)}")
    else:
        st.warning("No objects detected in the video")

if uploaded_file is not None:
    # Determine file type
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        process_image(image)
        
    elif uploaded_file.type.startswith('video'):
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        
        # Process video
        process_video(video_path)
        os.unlink(video_path)

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    yolo_model.conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.markdown("---")
    st.markdown("**Supported Formats:**")
    st.markdown("- Images: JPG, PNG")
    st.markdown("- Videos: MP4, AVI, MOV")
    st.markdown("---")
    st.markdown("Powered by YOLOv8 & Gemini Pro Vision")

