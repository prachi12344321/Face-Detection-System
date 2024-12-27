import cv2
import streamlit as st
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw green rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def main():
    st.title("Face Detection System")
    st.write("This app detects your face and draws a green square around it.")

    # Add a button to start/stop the camera
    start_camera = st.button("Start Camera", key="start")

    if start_camera:
        # Open the webcam
        cap = cv2.VideoCapture(0)
        stframe = st.empty()  # Placeholder for video frames
        stop_button = st.button("Stop Camera", key="stop")  # Unique key

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the camera.")
                break
            # Detect faces
            frame = detect_faces(frame)
            # Display the frame
            stframe.image(frame, channels="BGR")

            # Stop capturing when user clicks "Stop Camera"
            if stop_button:
                break

        cap.release()
        stframe.empty()  # Clear the video frame

if __name__ == "__main__":
    main()
