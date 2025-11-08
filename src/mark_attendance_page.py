# --- importing dependencies ---
import streamlit as st
import cv2

import config
import pinecone_service

def mark_attendance(session_state):
    st.header("Mark Attendance ⏳")
    st.markdown("---")

    placeholder_col, action_col = st.columns([2, 1])
    
    recognized_name_placeholder = st.empty()
    recognized_name_placeholder.info(f"Recognized: **{session_state.recognized_name}** (Roll No: {session_state.recognized_roll_no})" if session_state.recognized_name != "Unknown" else "")

    col_start, col_stop = action_col.columns(2)

    with col_start:
        if st.button("Start Camera", use_container_width=True, key="start_mark", disabled=session_state.camera_on):
            session_state.camera_on = True
            session_state.stop_marking = False
            st.rerun()

    with col_stop:
        if st.button("Stop Camera", use_container_width=True, key="stop_mark", disabled=not session_state.camera_on):
            session_state.stop_marking = True
            session_state.camera_on = False
            st.rerun()

    if session_state.camera_on and not session_state.stop_marking:
        
        frame_placeholder = placeholder_col.empty()
        
        cap = cv2.VideoCapture(0)
        
        while session_state.camera_on and not session_state.stop_marking:
            ret, frame = cap.read()
            if not ret:
                continue
            
            faces = config.CASCADE_CLASSIFIER.detectMultiScale(frame, 1.3, 5)
            current_recognized_name = "Unknown"
            match_score = 0.0

            for (x,y,w,h) in faces:
                
                cropped_face = frame[y:y+h,x:x+w]
                face_vector = pinecone_service.process_face_to_vector(cropped_face)

                current_recognized_name, current_roll_no, match_score = pinecone_service.recognize_face(face_vector)
                
                # Update session state for the button below the loop
                session_state.recognized_name = current_recognized_name
                session_state.recognized_roll_no = current_roll_no
                
                display_text = f"{current_recognized_name} ({match_score:.2f})"
                color = (0, 255, 0) if current_recognized_name != "Unknown" else (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                cv2.rectangle(frame,(x,y-40),(x+w,y),color,-1)
                cv2.putText(frame, display_text, (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)

            recognized_name_placeholder.info(f"Recognized: **{session_state.recognized_name}** (Roll No: {session_state.recognized_roll_no}) (Score: {match_score:.2f})")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        cap.release()
        session_state.camera_on = False
        session_state.stop_marking = True
        st.rerun()

    if st.button("MARK ATTENDANCE NOW", use_container_width=True, key="mark_final", 
                 disabled=session_state.recognized_name == "Unknown" or session_state.camera_on):
        if pinecone_service.mark_attendance(session_state.recognized_name, session_state.recognized_roll_no):
            st.success(f"Attendance marked for {session_state.recognized_name}.")
        else:
            st.error(f"Failed to mark attendance for {session_state.recognized_name}.")
            
        session_state.recognized_name = "Unknown" 
        session_state.recognized_roll_no = ""
        st.rerun()

    if st.button("Back to Home", use_container_width=True, key="back_mark"):
        session_state.page = 'Home'
        session_state.camera_on = False
        session_state.stop_marking = False
        session_state.recognized_name = "Unknown"
        st.rerun()