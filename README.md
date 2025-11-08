# 👁️ Automated Face Recognition & Attendance System 

This project features a hybrid system for student attendance: a core CV application for real-time recognition, and Streamlit web application integrated with Pinecone for robust data management and analysis.

---

## Features

- Real-Time Recognition: Uses Haar Cascade classifiers and the cosine similarity algorithm for rapid and accurate face identification.

- Enrollment: Collects 100 face samples per student via webcam and stores embeddings along with (name and roll no) in pinecone vectorstore.

- Voice Confirmation: Provides audible confirmation (via pywin32) when attendance is successfully marked.

- Scalable Vector Search: Utilizes Pinecone to store and manage student face embeddings and attendance records.

- Student Management: Web interface to Delete entire student profiles (vectors + history) and perform mass Roll Number updates.
- Attendance Analysis: Displays daily and cumulative attendance summaries, sortable by Roll Number.

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+

webcam connected to your machine.

Pinecone account.
```
---

### ⚙️ Project Structure
```bash
.
├── src/
│   ├── config.py                                        # Global settings and constants
│   ├── pinecone_service.py                              # Pinecone API interactions
│   ├── streamlit_app.py                                 # Main Streamlit execution file
│   ├── enroll_site.py                                   # Streamlit page for student Enrollment
│   ├── view_attendance_page.py                          # Streamlit page for retrieving attendance
│   ├── make_attendance_page.py                          # Streamlit page for marking via webcam
│   ├── manage_students_page.py                          # Streamlit page for deletion/update
├── .streamlit/
│   └── secrets.toml                                     # Secure credentials for Streamlit app
├── requirements.txt                                     # All required Python packages
├── venv                                                 # virtual environment setup
├── secrets.toml.example                                 # Template for configuration
├── README.md                                            # README.md file
└── .gitignore                                           # Standard exclusion list
```
---

### Setup

 #### 1. Clone the Repository:
```bash
git clone https://github.com/keshabshrestha007/face_recognition_and_attendance_system
```
```bash
cd face_recognition_and_attendance_system
```
#### 2. Create a Virtual Environment
```bash
python -m venv venv
```
On Linux/Mac
```bash
source venv/bin/activate   
```
On Windows
```bash
venv\Scripts\activate       
```
#### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```


#### 4.Configure Credentials:

```bash
cp secrets.toml.example .streamlit/secrets.
```

#### 5.Usage 
```bash

streamlit run src/streamlit_app.py
```

---

- Access the application in your browser and use the sidebar navigation:

- Enroll: Upload face embeddings directly to Pinecone.

- Take Attendance: Mark attendance directly into the Pinecone ATTENDANCE_INDEX.

- View Attendance: See real-time data analysis, download summaries, and correct individual records.

- Manage Students: Delete or update student profiles system-wide.
