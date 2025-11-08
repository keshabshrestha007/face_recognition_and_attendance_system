# --- importing dependencies
import streamlit as st

import config
import pinecone_service
def manage_students(session_state):
    st.header("Manage Student Data ")
    st.markdown("---")
    
    # --- 1. UPDATE ROLL NUMBER SECTION ---
    st.subheader("✏️ Update Student Roll Number")
    st.info("This updates the Roll Number across all facial vectors and attendance records.")
    
    col1, col2 = st.columns(2)
    name_to_update = col1.text_input("Enter Student Name (Exact Match):", key="update_name_input")
    new_roll_no = col2.text_input("Enter New Roll No:", key="new_roll_no_input")
    
    if st.button("UPDATE Roll Number", use_container_width=True, 
                 key="update_roll_button", disabled=not (name_to_update and new_roll_no)):
        
        with st.spinner(f"Updating Roll No for {name_to_update}..."):
            if pinecone_service.update_student_roll_no(name_to_update, new_roll_no):
                st.success(f"Successfully updated Roll No to **{new_roll_no}** for **{name_to_update}** across all data.")
            else:
                st.error("Roll Number update failed.")
        
    st.markdown("---")
    
    # --- 2. DELETE STUDENT DATA SECTION ---
    st.subheader("🗑️ Delete Student Data")
    st.warning("Deleting a student will remove all their facial data and all attendance records permanently.")
    
    name_to_delete = st.text_input("Enter Student Name to Delete:", key="delete_name_input")
    
    # Button logic for deletion confirmation (as defined in the previous response)
    if st.button(f"DELETE Data for {name_to_delete if name_to_delete else 'Student'}", 
                 use_container_width=True, 
                 key="final_delete_button",
                 disabled=not name_to_delete):
        
        if session_state.get('confirm_delete', False):
            with st.spinner(f"Deleting data for {name_to_delete}..."):
                if pinecone_service.delete_student_data(name_to_delete):
                    st.success(f"Successfully deleted ALL data for **{name_to_delete}**.")
                    
                else:
                    st.error(f"Failed to delete data for {name_to_delete}.")
            session_state.confirm_delete = False
        else:
            session_state.confirm_delete = True
            
            


    if st.button("Back to Home", use_container_width=True, key="back_manage"):
        session_state.page = 'Home'
        session_state.confirm_delete = False
        st.rerun()
