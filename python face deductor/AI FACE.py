import cv2
import os
import numpy as np
import sqlite3
import winsound  # For system beep
from datetime import datetime

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path for known faces
KNOWN_FACES_DIR = r"S:\old code\python face deductor\known_faces"
known_faces = []
known_names = []

# Database setup
DB_PATH = "face_log.db"

# Create database table
def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Function to check if a face is already in the database
def is_face_logged(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM face_log WHERE name = ?", (name,))
    result = cursor.fetchone()
    conn.close()
    return result is not None  # True if the face is already logged

# Function to insert detected face into database (only if it's new)
def log_face(name):
    if not is_face_logged(name):  # Only log if face is new
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO face_log (name, timestamp) VALUES (?, ?)",
                       (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        print(f"[INFO] {name} detected and logged into database.")

# Load known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        continue  # Skip if the file is not a valid image

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Take the first detected face
        face_roi = gray_image[y:y+h, x:x+w]  # Extract face region
        face_roi = cv2.resize(face_roi, (100, 100))  # Resize to standard size
        known_faces.append(face_roi)
        known_names.append(os.path.splitext(filename)[0])  # Store name from filename

# Initialize the database
initialize_database()

# Set to track already detected faces in the current session
detected_faces = set()

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from webcam
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        name = "Unknown"
        detected_face = gray_frame[y:y+h, x:x+w]
        detected_face = cv2.resize(detected_face, (100, 100))  # Resize to match stored images

        # Compare detected face with known faces using Histogram Comparison (Chi-Square)
        best_match_score = float("inf")  # Lower is better
        best_match_index = None

        for i, known_face in enumerate(known_faces):
            hist1 = cv2.calcHist([detected_face], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([known_face], [0], None, [256], [0, 256])
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

            if score < best_match_score:
                best_match_score = score
                best_match_index = i

        # Set a recognition threshold (lower score means better match)
        if best_match_index is not None and best_match_score < 100:
            name = known_names[best_match_index]

        # Draw a rectangle and label the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Check if this face has already been detected in the current session
        if name not in detected_faces:
            detected_faces.add(name)  # Mark this face as detected
            winsound.Beep(1000, 500)  # Play beep sound once
            log_face(name)  # Log to database only once

    # Display the video feed
    cv2.imshow("Face Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
