#--- Import dependencies ---
import streamlit as st
import cv2
import pandas as pd
import pinecone_service
import config
from datetime import datetime
from enroll_site import enroll_page
from mark_attendance_page import mark_attendance
from view_attendance_page import view_attendance
from manage_students_page import manage_students
# --- Session State Management ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'name' not in st.session_state:
    st.session_state.name = ""
if 'roll_no' not in st.session_state:
    st.session_state.roll_no = ""
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'stop_enrollment' not in st.session_state:
    st.session_state.stop_enrollment = False
if 'stop_marking' not in st.session_state:
    st.session_state.stop_marking = False
if 'recognized_name' not in st.session_state:
    st.session_state.recognized_name = "Unknown"
if 'recognized_roll_no' not in st.session_state:
    st.session_state.recognized_roll_no = ""

# Check for Pinecone health early
if pinecone_service.FACE_INDEX is None or pinecone_service.ATTENDANCE_INDEX is None:
    st.error("🚨 Pinecone initialization failed. Please check your PINECONE_API_KEY and PINECONE_ENVIRONMENT variables.")


# --- Page Rendering Logic ---

def home_page():
    st.title("👨‍🎓 Face Recognition & Attendance System")
    st.markdown("---")
    st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">
            <p style="font-size:20px;text-align:center;">
                Welcome to the automated attendance system!
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3,col4 = st.columns(4)

    with col1:
        if st.button("Enroll New Student", use_container_width=True):
            st.session_state.page = 'Enroll'
            st.rerun()
    with col2:
        if st.button("Mark My Attendance", use_container_width=True):
            st.session_state.page = 'Mark'
            st.rerun()
    with col3:
        if st.button("View Attendance", use_container_width=True):
            st.session_state.page = 'View'
            st.rerun()
    with col4:
        if st.button("Manage Students", use_container_width=True):
            st.session_state.page = 'Manage'
            st.rerun()




# --- Main App Logic ---
if __name__ == "__main__":
    
    # Page Router
    if st.session_state.page == 'Home':
        home_page()
    elif st.session_state.page == 'Enroll':
        enroll_page(session_state=st.session_state)
    elif st.session_state.page == 'Mark':
        mark_attendance(session_state=st.session_state)
    elif st.session_state.page == 'View':
        view_attendance(session_state=st.session_state)
    elif st.session_state.page == 'Manage':
        manage_students(session_state = st.session_state)
