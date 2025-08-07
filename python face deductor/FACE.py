import cv2
import os
import numpy as np

# Load the pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load known face images from a folder
KNOWN_FACES_DIR = r"S:\old code\python face deductor\known_faces"
known_faces = []
known_names = []

# Read and store known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    image = cv2.imread(os.path.join(KNOWN_FACES_DIR, filename))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        known_faces.append(gray_image)
        known_names.append(os.path.splitext(filename)[0])  # Save name from filename

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

        # Compare detected face with known faces using SSIM (Structural Similarity)
        best_match_score = 0.0
        best_match_index = None

        for i, known_face in enumerate(known_faces):
            resized_face = cv2.resize(detected_face, (known_face.shape[1], known_face.shape[0]))
            score = np.mean(cv2.absdiff(resized_face, known_face))  # Simple pixel-wise difference
            if score < best_match_score or best_match_index is None:
                best_match_score = score
                best_match_index = i

        if best_match_index is not None:
            name = known_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Face Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()