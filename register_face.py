import cv2
import face_recognition
import pickle
import os

ENCODINGS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'encodings.pkl')

def register_face(name):
    """
    Registers a user's face by capturing from webcam, encoding it, and saving to a pickle file.
    This involves generating a face embedding using the pretrained dlib ResNet model.
    """
    data = {"encodings": [], "names": []}
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print("Warning: Encodings file is corrupted or empty. Starting fresh.")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not access webcam.")
        return

    print(f"Registering {name}. Look at the camera and press 's' to capture.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        
        cv2.imshow('Register Face - Press s to capture', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect faces using HOG + linear classifier
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 1:
                # Generate face embedding using pretrained ResNet-34 model
                encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                data["encodings"].append(encoding)
                data["names"].append(name)
                with open(ENCODINGS_FILE, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Face registered for {name}.")
                break
            else:
                print("Error: Exactly one face must be visible. Try again.")

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter name to register: ")
    register_face(name)