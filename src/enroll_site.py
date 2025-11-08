# --- importing dependencies ---
import streamlit as st
import cv2
import numpy as np
import os
import pinecone_service 
import config 
from PIL import Image

from pinecone_service import FACE_INDEX

def enroll_page(session_state):
    
    if 'CASCADE_CLASSIFIER' not in dir(config):
        st.error("Configuration Error: CASCADE_CLASSIFIER not found in config.py.")
        return
        
    st.header("Enroll New Student 🎓 ")
    st.markdown("---")

    
    name = st.text_input("Enter Student Name:", key="input_name")
    roll_no = st.text_input("Enter Student Roll No:", key="input_roll_no")
    
    st.info("Enrollment requires one clear, high-quality picture. The system will create 100 near-identical samples from this picture to train the model in Pinecone.")

    if name and roll_no:
        query_results = FACE_INDEX.query(
                vector = [1.0] * config.VECTOR_DIMENSION,
                filter={"student_name":name},
                top_k=1,
                include_metadata=True
            )
    

        if query_results.matches == []:
        # Streamlit's camera input for image capture
            camera_image = st.camera_input(
                "Take a clear photo of the student's face", 
                key="camera_enrollment",
                disabled=not name or not roll_no
            )

            if camera_image is not None:
                
                # 1. Read and Prepare Image
                try:
                    # Convert the uploaded file buffer to OpenCV format
                    bytes_data = camera_image.getvalue()
                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    # Convert BGR to RGB for processing/display
                    frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                    
                except Exception as e:
                    st.error(f"Error processing image file: {e}")
                    return

                # 2. Face Detection
                faces = config.CASCADE_CLASSIFIER.detectMultiScale(frame, 1.3, 5)
                
                if len(faces) == 0:
                    st.warning("No face detected in the image. Please take a clearer picture.")
                    
                elif len(faces) > 1:
                    st.warning("Multiple faces detected. Please ensure only one person is in the frame.")
                    
                else:
                    # Only proceed if exactly one face is detected
                    (x, y, w, h) = faces[0]
                    
                    # Crop the face from the original BGR image before vector processing
                    cropped_face_bgr = cv2_img[y:y+h, x:x+w]
                    
                    # Draw rectangle on the RGB frame for display back to the user
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    st.image(frame, channels="RGB", caption="Detected Face", use_container_width=True)
                    
                    if st.button(f"Confirm & Enroll {name} ({roll_no})", key="confirm_upload"):
                        with st.spinner(f"Processing and uploading 100 samples for {name}..."):
                            
                            # 3. Generate Vector and Batch
                            
                            # Process the single BGR face to get the base feature vector
                            # NOTE: Assuming pinecone_service.process_face_to_vector expects BGR or handles conversion internally.
                            base_vector = pinecone_service.process_face_to_vector(cropped_face_bgr)
                            
                            # Create a batch of 100 identical vectors to simulate the 100 samples required by the original logic.
                            vectors_to_upload = [base_vector] * 100
                            
                            # 4. Upload to Pinecone
                            if pinecone_service.enroll_face_batch(name, roll_no, vectors_to_upload):
                                st.success(f"Student **{name}** (Roll No: **{roll_no}**) enrolled successfully with 100 vectors in Pinecone!")
                                
                        
                                
                                
                            else:
                                st.error("Enrollment failed due to a Pinecone error. Check console for details.")
        else:
            st.error(f"Faces data are available for Name: {name} and Roll No,: {roll_no}.")
         
    if st.button("Back to Home", use_container_width=True, key="back_enroll"):
        session_state.page = 'Home'
        st.rerun()
