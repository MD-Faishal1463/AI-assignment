import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Basic preprocessing to handle varying lighting: Apply histogram equalization to improve contrast.
    This is a simple approach and may not work perfectly in extreme conditions.
    """
    # Convert to YUV for equalization on luminance channel
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def check_spoof(face_locations, frame):
    """
    Basic spoof prevention: Check for face movement across frames by comparing face positions.
    This is a simple heuristic; real liveness detection would require more advanced methods.
    Assumes a global variable for previous face location (not ideal, but simple for demo).
    """
    # Note: In a real system, track movement over multiple frames. Here, we use a static check.
    if len(face_locations) != 1:
        return False  

    # Simple check: If face is too small or not centered, flag as potential spoof
    top, right, bottom, left = face_locations[0]
    face_width = right - left
    face_height = bottom - top
    frame_height, frame_width = frame.shape[:2]

    # Check if face is reasonably sized and positioned (not too small or off-center)
    if face_width < 100 or face_height < 100 or abs((left + right)/2 - frame_width/2) > frame_width/4:
        return False  # Potential spoof

    return True  # Assume live if checks pass