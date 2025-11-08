# -- Import depencies ---
import streamlit as st
import cv2
import os

# --- Pinecone and Application Settings ---

# ---Fetch from streamlit secrets ---
if "PINECONE_API_KEY" not in st.secrets:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it before running the script.")
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1") 
FACE_INDEX_NAME = st.secrets.get("FACE_INDEX_NAME", "face-and-name-data")
ATTENDANCE_INDEX_NAME = st.secrets.get("ATTENDANCE_INDEX_NAME", "attendance-data")

VECTOR_DIMENSION = int(st.secrets['VECTOR_DIMENSION']) # 50 * 50 * 3 (for resized RGB face)
IMAGE_SIZE = (50, 50)
SCORE_THRESHOLD = float(st.secrets['SCORE_THRESHOLD']) # Minimum score for a face match

# --- OpenCV/Utilities ---

try:
    
    CASCADE_CLASSIFIER = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    print(f"Loading Cascade Classifier from: {CASCADE_CLASSIFIER}")
    CASCADE_CLASSIFIER = cv2.CascadeClassifier(CASCADE_CLASSIFIER)
    
    if CASCADE_CLASSIFIER.empty():
        print("ERROR: Failed to load haarcascade_frontalface_default.xml. Check file path.")
except Exception as e:
    print(f"Error loading face detector: {e}")