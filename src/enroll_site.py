# --- importing dependencies
import streamlit as st
import cv2
import config
import pinecone_service

def enroll_page(session_state):

    st.header("Enroll New Student 🎓 ")
    st.markdown("---")

    name = st.text_input("Enter Student Name:", value=session_state.name, help="Use Name_Surname format")
    session_state.name = name
    roll_no = st.text_input("Enter Student Roll No:", value=session_state.roll_no)
    session_state.roll_no = roll_no


    col_start, col_stop, _ = st.columns([1, 1, 3])

    with col_start:
        if st.button("Start Enrollment", use_container_width=True, key="start_enroll", disabled=not name or session_state.camera_on):
            session_state.camera_on = True
            session_state.stop_enrollment = False
            st.rerun()

    with col_stop:
        if st.button("Stop Enrollment", use_container_width=True, key="stop_enroll", disabled=not session_state.camera_on):
            session_state.stop_enrollment = True
            session_state.camera_on = False
            st.rerun()


    if session_state.camera_on and not session_state.stop_enrollment:
        
        vectors_to_upload = []
        
        frame_placeholder = st.empty()
        st.subheader(f"Capturing Faces for **{name}**...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        cap = cv2.VideoCapture(0)
        i = 0 # Frame counter
        count = 0 # Successful sample counter
        
        while count < 100 and not session_state.stop_enrollment:
            ret, frame = cap.read()
            if not ret:
                continue
            
            faces = config.CASCADE_CLASSIFIER.detectMultiScale(frame, 1.3, 5)

            for (x, y, w, h) in faces:
                cropped_face_bgr = frame[y:y+h, x:x+w]
                
                if count < 100 and i % 10 == 0:
                    vector = pinecone_service.process_face_to_vector(cropped_face_bgr)
                    vectors_to_upload.append(vector)
                    count += 1
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Samples: {count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            i += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            progress_bar.progress(count / 100)
            status_text.write(f"Collecting facial data: {count}/100")
        
        cap.release()
        
        session_state.camera_on = False
        session_state.stop_enrollment = True
        
        if count == 100:
            st.success("Facial data collection complete (100 samples)! Uploading...")
            if pinecone_service.enroll_face_batch(name, roll_no, vectors_to_upload):
                st.success(f"Student {name} enrolled successfully!")
            else:
                st.error("Enrollment failed due to a Pinecone error.")
        else:
            st.warning(f"Enrollment stopped early. Only {count} samples collected. No data uploaded.")

    if st.button("Back to Home", use_container_width=True, key="back_enroll"):
        session_state.page = 'Home'
        session_state.camera_on = False
        session_state.stop_enrollment = False
        st.rerun()
