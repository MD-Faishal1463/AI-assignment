import cv2
import face_recognition
import pickle
import os
import csv
from datetime import datetime
from utils import preprocess_frame, check_spoof

# Paths
ENCODINGS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'encodings.pkl')
ATTENDANCE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'attendance.csv')

def load_encodings():
    """Load registered face encodings."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def mark_attendance(name, action):
    """Mark punch-in or punch-out in CSV."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, action, now])

def main():
    data = load_encodings()
    if not data["encodings"]:
        print("No registered faces. Please register first.")
        return

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not access webcam.")
        return

    # Track attendance state 
    attendance_state = {}

    print("Starting attendance system. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Preprocess for lighting
        frame = preprocess_frame(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using HOG + linear classifier
        face_locations = face_recognition.face_locations(rgb_frame)
        # Generate embeddings using pretrained ResNet-34 model
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check spoof (basic movement check)
            if not check_spoof(face_locations, frame):
                continue  # Skip if spoof detected

            # Match face using Euclidean distance with tolerance 0.6
            matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = data["names"][matched_idx]

                # Handle punch-in/out
                if name not in attendance_state or attendance_state[name] == "Punch-out":
                    mark_attendance(name, "Punch-in")
                    attendance_state[name] = "Punch-in"
                    print(f"{name} punched in at {datetime.now().strftime('%H:%M:%S')}")
                elif attendance_state[name] == "Punch-in":
                    mark_attendance(name, "Punch-out")
                    attendance_state[name] = "Punch-out"
                    print(f"{name} punched out at {datetime.now().strftime('%H:%M:%S')}")

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow('Attendance System - Press q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()