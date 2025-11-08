import numpy as np
import cv2
import uuid
import os
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException

import config

# --- Initialization and Connection ---

def initialize_pinecone():
    """Initializes Pinecone client and connects/creates necessary indexes."""
    if not config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing. Please set it in your environment variables.")

    try:
        pc = Pinecone(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENVIRONMENT)
        
        existing_index_names = [index.name for index in pc.list_indexes()]

        # 1. Setup Face Data Index (Enrollment)
        if config.FACE_INDEX_NAME not in existing_index_names:
            print(f"Index '{config.FACE_INDEX_NAME}' not found. Creating...")
            pc.create_index(
                name=config.FACE_INDEX_NAME,
                dimension=config.VECTOR_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region=config.PINECONE_ENVIRONMENT)
            )
        face_data_index = pc.Index(config.FACE_INDEX_NAME)
        
        # 2. Setup Attendance Index
        if config.ATTENDANCE_INDEX_NAME not in existing_index_names:
            print(f"Index '{config.ATTENDANCE_INDEX_NAME}' not found. Creating...")
            pc.create_index(
                name=config.ATTENDANCE_INDEX_NAME, 
                dimension=config.VECTOR_DIMENSION, 
                metric='euclidean',
                spec=ServerlessSpec(cloud="aws", region=config.PINECONE_ENVIRONMENT)
            ) 
        attendance_index = pc.Index(config.ATTENDANCE_INDEX_NAME)
        
        return face_data_index, attendance_index

    except Exception as e:
        print(f"Pinecone Initialization Error: {e}")
        raise e


try:
    FACE_INDEX, ATTENDANCE_INDEX = initialize_pinecone()
except Exception:
    
    FACE_INDEX = None
    ATTENDANCE_INDEX = None


# --- Utility Functions ---

def process_face_to_vector(face_image_bgr):
    """Converts a BGR face image to the 7500-dim float vector for Pinecone.
        Args:
            face_image_bgr (np.ndarray): The cropped face image in BGR format.
        Returns:
            list: The flattened face vector as a list of floats.
    """
    resized_face = cv2.resize(face_image_bgr, config.IMAGE_SIZE)
    rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    vector = rgb_face.flatten().astype(np.float32).tolist()
    return vector

def enroll_face_batch(name, roll_no, vectors_to_upload):
    """Uploads a batch of face vectors to the FACE_INDEX.
        Args:
            name (str): The name of the person.
            roll_no (str): The roll number of the person.
            vectors_to_upload (list): List of face vectors to upload.
        Returns:
            bool: True if upload successful, False otherwise.
    """
    if FACE_INDEX is None:
        print("Pinecone FACE_INDEX not initialized.")
        return False
        
    if not vectors_to_upload:
        print("No vectors to upload.")
        return True # Successful, but nothing uploaded

    vectors_with_metadata = []
    for vector in vectors_to_upload:
        vector_id = f"{name}_{uuid.uuid4()}"
        metadata = {"student_name": name, "roll_no": roll_no}
        vectors_with_metadata.append((vector_id, vector, metadata))

    try:
        batch_size = 32
        for i in range(0, len(vectors_with_metadata), batch_size):
            batch = vectors_with_metadata[i:i + batch_size]
            FACE_INDEX.upsert(vectors=batch)
        print(f"Successfully uploaded {len(vectors_with_metadata)} vectors for {name}.")
        return True
    except PineconeApiException as e:
        print(f"ERROR: Pinecone Upload Failed: {e}")
        return False

def recognize_face(face_vector):
    """Queries the FACE_INDEX to recognize a face vector.
        Args:
            face_vector (list): The face vector to recognize.
        Returns:
            tuple: (recognized_name (str), recognized_roll_no (str), match_score (float))
    """
    if FACE_INDEX is None:
        return "Unknown", "", 0.0
        
    try:
        query_results = FACE_INDEX.query(
            vector=face_vector,
            top_k=1,
            include_metadata=True
        )
        
        if query_results.matches and query_results.matches[0].score > config.SCORE_THRESHOLD:
            best_match = query_results.matches[0]
            name = best_match.metadata.get("student_name", "Unknown")
            roll_no = best_match.metadata.get("roll_no", "")
            return name, roll_no, best_match.score
        
        return "Unknown", "", 0.0
    except PineconeApiException as e:
        print(f"ERROR: Pinecone Query Failed: {e}")
        return "Unknown", "", 0.0

def mark_attendance(name, roll_no):
    """Records attendance in the ATTENDANCE_INDEX.
        Args:
            name (str): The name of the person.
            roll_no (str): The roll number of the person.
        Returns:
            bool: True if attendance marked successfully, False otherwise."""
    if ATTENDANCE_INDEX is None:
        print("Pinecone ATTENDANCE_INDEX not initialized.")
        return False
        
    current_date = datetime.now().strftime('%d-%m-%Y')
    current_time = datetime.now().strftime('%H:%M:%S')
    attendance_id = f"{name}_{current_date}"
    
    # 1. Check if attendance already exists
    try:
        fetch_result = ATTENDANCE_INDEX.fetch(ids=[attendance_id])
        if attendance_id in fetch_result.vectors:
            print(f"Attendance for {name} on {current_date} already recorded.")
            return True # Already marked
    except PineconeApiException as e:
        print(f"Warning: Pinecone Fetch failed ({e}). Proceeding to upsert.")

    # 2. Upsert the new attendance record
    metadata = {
        "student_name": name,
        "roll_no": roll_no,
        "date": current_date,
        "time": current_time,
    }
    
    placeholder_vector = [1.0] * config.VECTOR_DIMENSION 

    try:
        ATTENDANCE_INDEX.upsert(
            vectors=[(attendance_id, placeholder_vector, metadata)]
        )
        print(f"Attendance recorded for: {name} at {current_time}.")
        return True
    except PineconeApiException as e:
        print(f"ERROR: Pinecone upsert failed for attendance record: {e}")
        return False
        
def get_all_attendance_records():
    """Fetches all attendance records from the index.
        Returns:
            list: List of attendance records as dictionaries.
    """
    if ATTENDANCE_INDEX is None:
        return []
        
    try:
        placeholder_vector = [1.0] * config.VECTOR_DIMENSION 
        
        query_results = ATTENDANCE_INDEX.query(
            vector=placeholder_vector,
            top_k=10000, 
            include_metadata=True
        )
        
        records = []
        for match in query_results.matches:
            meta = match.metadata
            records.append({
                'Roll No': meta.get('roll_no', 'N/A'),
                'Name': meta.get('student_name', 'N/A'),
                'Date': meta.get('date', 'N/A'),
                'Time': meta.get('time', 'N/A'),
                'Record ID': match.id
            })
        return records

    except PineconeApiException as e:
        print(f"ERROR: Error querying Pinecone attendance: {e}")
        return []
    
def delete_student_data(name_to_delete):
    """Deletes ALL facial vectors for a given student name from FACE_INDEX.
        Args:
            name_to_delete (str): The name of the student whose data is to be deleted.
        Returns:
            bool: True if deletion successful, False otherwise.
    """
    if FACE_INDEX is None:
        print("Pinecone FACE_INDEX not initialized.")
        return False
        
    try:
        
        FACE_INDEX.delete(
            filter={"student_name": name_to_delete},
            delete_all=False 
        )
        
        
        ATTENDANCE_INDEX.delete(
            filter={"student_name": name_to_delete},
            delete_all=False
        )

        print(f"Successfully deleted all enrollment and attendance data for: {name_to_delete}")
        return True
    except PineconeApiException as e:
        print(f"ERROR: Pinecone Deletion Failed: {e}")
        return False
    

def update_student_roll_no(name, new_roll_no):
    """
    Updates the 'roll_no' metadata field for ALL vectors belonging to a specific student 
    in the FACE_INDEX and ATTENDANCE_INDEX.
    """
    if FACE_INDEX is None or ATTENDANCE_INDEX is None:
        print("Pinecone indexes not initialized.")
        return False

    print(f"Starting roll number update for {name} to {new_roll_no}...")
    
    
    placeholder_vector = [1.0] * config.VECTOR_DIMENSION

    
    try:
     
        face_results = FACE_INDEX.query(
            vector=placeholder_vector,
            top_k=1000, 
            filter={"student_name": name},
            include_values=False,
            include_metadata=False 
        )
        
        face_ids_to_update = [match.id for match in face_results.matches]
        
        if not face_ids_to_update:
            print(f"No face vectors found for student: {name}.")
        
        
        for vector_id in face_ids_to_update:
            FACE_INDEX.update(
                id=vector_id,
                set_metadata={"student_name": name, "roll_no": new_roll_no}
                
            )

        print(f"Updated roll no for {len(face_ids_to_update)} face vectors.")
        
    except Exception as e:
        print(f"ERROR: Face Index Roll No Update Failed: {e}")
        return False

   
    try:
        attendance_results = ATTENDANCE_INDEX.query(
            vector=placeholder_vector,
            top_k=10000, 
            filter={"student_name": name},
            include_values=False,
            include_metadata=True 
        )

        attendance_updates_count = 0
        for match in attendance_results.matches:
            
            updated_meta = {
                "student_name": name,
                "roll_no": new_roll_no,
                "date": match.metadata.get("date"),
                "time": match.metadata.get("time")
            }
            
        
            ATTENDANCE_INDEX.update(
                id=match.id,
                set_metadata=updated_meta
            )
            attendance_updates_count += 1
            
        print(f"Updated roll no for {attendance_updates_count} attendance records.")

    except Exception as e:
        print(f"ERROR: Attendance Index Roll No Update Failed: {e}")
        return False
        
    return True