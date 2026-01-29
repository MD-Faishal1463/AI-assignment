# Face Authentication Attendance System

## Project Overview
This is a simple Face Authentication Attendance System built for an AI/ML intern role. It allows users to register their faces via a webcam and then perform real-time face recognition for attendance tracking (punch-in and punch-out). The system uses OpenCV for webcam input and the face_recognition library (based on dlib) for face encoding and matching. No custom model training is required, as it leverages pretrained models.

The project demonstrates practical ML understanding, including face detection, encoding, and basic preprocessing for lighting variations. It includes a rudimentary spoof prevention mechanism and stores attendance in a CSV file.

## Features
- **Face Registration**: Capture and encode a user's face from a webcam.
- **Real-Time Recognition**: Identify registered faces in real-time and mark attendance.
- **Attendance Tracking**: Punch-in and punch-out with timestamps stored in CSV.
- **Lighting Handling**: Basic preprocessing (histogram equalization) to mitigate varying lighting.
- **Spoof Prevention**: Simple liveness check based on face position and size (not foolproof).
- **Local Demo**: Fully functional with a real webcam; no cloud dependencies.

## Tech Stack
- **Python**: Core language.
- **OpenCV**: For webcam capture and image processing.
- **face_recognition (dlib-based)**: For face detection, encoding, and comparison.
- **Pickle**: To store face encodings.
- **CSV**: For attendance logs.

## System Workflow
1. **Registration**: Run `register_face.py` to capture a face encoding and save it.
2. **Attendance**: Run `attendance_system.py` for real-time recognition.
   - Detects faces in webcam feed.
   - Matches against registered encodings.
   - Marks punch-in on first match, punch-out on subsequent matches.
   - Displays recognized name on screen.
3. **Spoof Prevention**: Checks for face movement/size as a basic liveness indicator.
4. **Lighting Handling**: Applies histogram equalization before processing.

## Spoof Prevention Explanation
Spoof prevention is implemented via a simple heuristic in `utils.py`:
- Ensures exactly one face is detected.
- Checks if the face is reasonably sized and centered.
- This is basic and can be bypassed (e.g., with photos or masks). For production, integrate advanced liveness detection (e.g., blink or depth analysis).

## Accuracy Expectations
- **High Accuracy**: ~90-95% in good lighting with frontal faces and no occlusions, using face_recognition's default tolerance (0.6).
- **Factors Affecting Accuracy**: Lighting, angle, distance, and expressions. Preprocessing helps with lighting but is not advanced.

## Known Failure Cases
- Poor lighting: Extreme shadows or low light reduce detection accuracy.
- Non-frontal faces: Side profiles or tilted heads may not match.
- Occlusions: Glasses, masks, or hair covering parts of the face.
- Multiple faces: System assumes one face; crowded scenes may confuse it.
- Spoofing: Basic checks may fail against determined attacks (e.g., high-quality photos).
- Hardware: Low-quality webcams or fast movements can cause frame drops.

## How to Run the Project Locally
1. **Prerequisites**: Python 3.8+, webcam.
2. **Install Dependencies**: `pip install -r requirements.txt`.
3. **Register a Face**:
   - Run `python src/register_face.py`.
   - Enter a name and press 's' to capture.
4. **Run Attendance System**:
   - Run `python src/attendance_system.py`.
   - Faces will be recognized; press 'q' to quit.
5. **View Attendance**: Check `data/attendance.csv` for logs.

## Notes on Limitations and Real-World Constraints
- **Simplicity**: This is an internship-level project; production systems need robust security, multi-user handling, and advanced spoof detection.
- **Performance**: Runs on CPU; may lag on low-end hardware.
- **Privacy**: Face data is stored locally in pickle—ensure compliance with data protection laws.
- **Scalability**: Designed for small-scale use; large datasets would require optimization.
- **Ethical Considerations**: Face recognition has biases; test for fairness.
- **Time Constraint**: Built within 24 hours, focusing on core functionality over perfection.

This project showcases practical ML application with honesty about limitations. For feedback, contact [your email].