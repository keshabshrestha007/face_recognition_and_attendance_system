# --- importing dependencies ---
import streamlit as st
import cv2
import numpy as np
from PIL import Image

import config
import pinecone_service

def mark_attendance(session_state):
    st.header("Mark Attendance 📸")  
    camera_image = st.camera_input("Take a photo for attendance", key="camera_attendance")
    if camera_image:
        
        bytes_data = camera_image.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        faces = config.CASCADE_CLASSIFIER.detectMultiScale(frame, 1.3, 5)
   
        for (x, y, w, h) in faces:
            cropped_face = frame[y:y+h,x:x+w]
            face_vector = pinecone_service.process_face_to_vector(cropped_face)

            current_recognized_name, current_roll_no, match_score = pinecone_service.recognize_face(face_vector)
                
                
            session_state.recognized_name = current_recognized_name
            session_state.recognized_roll_no = current_roll_no
                
            display_text = f"{session_state.recognized_name} ({match_score:.2f})"
            color = (0, 255, 0) if session_state.recognized_name != "Unknown" else (0, 0, 255)
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.rectangle(frame,(x,y-40),(x+w,y),color,-1)
            cv2.putText(frame, display_text, (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, session_state.recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
        result_image = Image.fromarray(frame)
        st.image(result_image, caption=f"Recognition Result: {session_state.recognized_name if session_state.recognized_name else 'No face detected'}")
        
        
        if session_state.recognized_name and st.button(f"Confirm Attendance for {session_state.recognized_name}"):
            
            if pinecone_service.mark_attendance(session_state.recognized_name, session_state.recognized_roll_no):
                st.success(f"Attendance marked for {session_state.recognized_name}.")
            else:
                st.error(f"Failed to mark attendance for {session_state.recognized_name}.")           
    
    if st.button("Back to Home", use_container_width=True, key="back_mark"):
        session_state.page = 'Home'
        session_state.camera_on = False
        session_state.stop_marking = False
        session_state.recognized_name = "Unknown"
        st.rerun()
