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
st.set_page_config(page_title="Vision Analyzer", layout="wide")
st.title("AI Vision Analyzer")
st.markdown("Upload media for object detection (YOLOv8) and contextual analysis (Gemini)")

# Initialize models
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_gemini():
    # 🔒 Replace with your actual API key (TEMPORARY USE ONLY)
    API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
    
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel('gemini-pro-vision')

yolo_model = load_yolo()
gemini_model = load_gemini()

# File upload section
uploaded_file = st.file_uploader(
    "Choose an image/video...", 
    type=["jpg", "jpeg", "png", "mp4", "avi"],
    accept_multiple_files=False
)

def process_image(uploaded_image):
    """Handle image processing pipeline"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # YOLO Detection
    with st.spinner("Analyzing objects..."):
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
                uploaded_image.save(tmp_file.name)
                results = yolo_model.predict(tmp_file.name)
                
            res_plotted = results[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption="Detection Results", use_column_width=True)
            
            detected_objects = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                conf = float(box.conf[0])
                detected_objects.append(f"{label} ({conf:.2f})")
                
            return detected_objects
            
        except Exception as e:
            st.error(f"Detection Failed: {str(e)}")
            return []

def process_video(video_path):
    """Handle video processing pipeline"""
    st.subheader("Video Analysis Results")
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Process video with YOLO
            results = yolo_model.predict(
                source=video_path,
                project=tmp_dir,
                save=True,
                exist_ok=True
            )
            
            # Display processed video
            processed_video = Path(results[0].save_dir) / Path(video_path).name
            st.video(str(processed_video))
            
            # Collect detection data
            all_objects = [
                yolo_model.names[int(box.cls[0])]
                for result in results
                for box in result.boxes
            ]
            
            return all_objects
            
    except Exception as e:
        st.error(f"Video Processing Error: {str(e)}")
        return []

def generate_gemini_analysis(content, media_type):
    """Generate contextual analysis with Gemini"""
    try:
        prompt = f"""
        Analyze this {media_type} content containing: {', '.join(content)}.
        Provide detailed analysis including:
        1. Main objects/activities
        2. Contextual relationships
        3. Potential implications
        4. Safety considerations
        """
        return gemini_model.generate_content(prompt).text
        
    except Exception as e:
        return f"Analysis failed: {str(e)}"

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        detections = process_image(image)
        
        if detections:
            with st.spinner("Generating AI analysis..."):
                analysis = generate_gemini_analysis(detections, "image")
                st.subheader("Contextual Analysis")
                st.write(analysis)
                
    elif uploaded_file.type.startswith('video'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        
        detections = process_video(video_path)
        os.unlink(video_path)
        
        if detections:
            with st.spinner("Generating video summary..."):
                analysis = generate_gemini_analysis(list(set(detections)), "video")
                st.subheader("Video Summary")
                st.write(analysis)

# Security warning in sidebar
with st.sidebar:
    st.warning("""
    ⚠️ Security Notice:
    - This implementation exposes API keys
    - Only use for temporary testing
    - Never commit to version control
    - Rotate keys if exposed
    """)
    st.markdown("---")
    st.markdown("**Controls:**")
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5)
    yolo_model.conf = confidence

