# --- importing dependencies
import cv2
import streamlit as st
import pandas as pd
from datetime import datetime

import config
import pinecone_service

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to CSV format for download."""
    # Use index=False to exclude the DataFrame index from the CSV file
    return df.to_csv(index=False).encode('utf-8')

def view_attendance(session_state):
    
    st.header("View Attendance 📊 ")
    st.markdown("---")
    
    st.info("Fetches all attendance records from the Pinecone index.")

    if st.button("Fetch and Display Attendance", use_container_width=True, key="fetch_attendance"):
   
        try:
            with st.spinner('Fetching records from Pinecone...'):
                
                
                records = pinecone_service.get_all_attendance_records()
        except NameError:
             st.error("Error: pinecone_service is not defined. Please ensure it is imported.")
             return
        
        if not records:
            st.info("No attendance records found in Pinecone.")
            return

        df = pd.DataFrame(records)
        
        if not df.empty:
            
            
            df['Roll No'] = pd.to_numeric(df['Roll No'], errors='coerce').astype('Int64')
            
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            
  
            latest_date = df['Date'].max()
            daily_df = df[df['Date'] == latest_date].sort_values(by='Roll No').reset_index(drop=True).drop(columns=['Record ID'])
            st.subheader(f"Attendance Details for Latest Date: {latest_date.strftime('%Y-%m-%d')}")
            st.dataframe(daily_df, use_container_width=True)

            
            monthly_counts = df.groupby(['Roll No', 'Name']).size().reset_index(name='Total Attendance').sort_values(by='Roll No')
            st.subheader("Monthly Attendance")
            st.dataframe(monthly_counts, use_container_width=True)
            
            
            st.download_button(
                label="Download Full Attendance Data as CSV",
                data=monthly_counts.to_csv(index=False).encode('utf-8'),
                file_name= f"attendance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                key='download_csv'
            )

        else:
            st.info("No valid records found after processing.")
        
    if st.button("Back to Home", use_container_width=True, key="back_view_2"):
        session_state.page = 'Home'
        st.rerun()