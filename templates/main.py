from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import math as m
import time
from threading import Lock

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global variables
current_feedback = "Analyzing posture..."
feedback_lock = Lock()

# Initialize counters and times
good_frames = 0
bad_frames = 0
start_time = time.time()

# Colors
BLUE = (255, 127, 0)
RED = (50, 50, 255)
GREEN = (127, 255, 0)
DARK_BLUE = (127, 20, 0)
LIGHT_GREEN = (127, 233, 100)
YELLOW = (0, 255, 255)
PINK = (255, 0, 255)
WHITE = (255, 255, 255)

def findDistance(x1, y1, x2, y2):
    """Calculate distance between two points"""
    return m.sqrt((x2-x1)**2 + (y2-y1)**2)

def findAngle(x1, y1, x2, y2):
    """Calculate angle with respect to vertical"""
    theta = m.acos((y2-y1)*(-y1) / (m.sqrt((x2-x1)**2 + (y2-y1)**2) * y1))
    degree = int(180/m.pi) * theta
    return degree

def analyze_posture(neck_angle, torso_angle):
    """Analyze posture based on neck and torso angles"""
    if neck_angle < 40 and torso_angle < 10:
        return True, "Good posture! Keep it up!"
    elif neck_angle >= 40:
        return False, "High Risk: Your neck is too bent forward!"
    elif torso_angle >= 10:
        return False, "High Risk: Your back is not straight!"
    return False, "Poor posture detected! Please adjust your position."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feedback')
def get_feedback():
    global current_feedback
    with feedback_lock:
        return jsonify({'feedback': current_feedback})

def gen_frames():
    global current_feedback, good_frames, bad_frames
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for better performance
        frame = cv2.resize(frame, (800, 600))
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get key points
            l_shldr = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            r_shldr = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            l_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * h))
            l_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))

            # Calculate alignment
            shoulder_offset = findDistance(l_shldr[0], l_shldr[1], r_shldr[0], r_shldr[1])
            
            # Draw alignment status
            alignment_text = f"Offset: {int(shoulder_offset)}"
            if shoulder_offset < 100:
                alignment_text += " (Aligned)"
                cv2.putText(frame, alignment_text, (w - 200, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            else:
                alignment_text += " (Not Aligned)"
                cv2.putText(frame, alignment_text, (w - 200, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

            # Calculate angles
            neck_inclination = findAngle(l_shldr[0], l_shldr[1], l_ear[0], l_ear[1])
            torso_inclination = findAngle(l_hip[0], l_hip[1], l_shldr[0], l_shldr[1])

            # Analyze posture
            is_good_posture, feedback = analyze_posture(neck_inclination, torso_inclination)
            
            with feedback_lock:
                current_feedback = feedback

            # Update frame counters
            if is_good_posture:
                good_frames += 1
                bad_frames = 0
                color = GREEN
            else:
                bad_frames += 1
                good_frames = 0
                color = RED

            # Draw skeleton
            cv2.circle(frame, l_shldr, 5, YELLOW, -1)
            cv2.circle(frame, l_ear, 5, YELLOW, -1)
            cv2.circle(frame, l_hip, 5, YELLOW, -1)
            
            # Draw posture lines
            cv2.line(frame, l_shldr, l_ear, color, 2)
            cv2.line(frame, l_shldr, (l_shldr[0], l_shldr[1] - 100), color, 2)
            cv2.line(frame, l_hip, l_shldr, color, 2)
            cv2.line(frame, l_hip, (l_hip[0], l_hip[1] - 100), color, 2)

            # Display angles
            cv2.putText(frame, f"Neck: {int(neck_inclination)}°", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Torso: {int(torso_inclination)}°", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display posture time
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30  # Fallback if FPS detection fails
            
            good_time = (1/fps) * good_frames
            bad_time = (1/fps) * bad_frames
            
            if is_good_posture:
                time_text = f"Good Posture Time: {round(good_time, 1)}s"
            else:
                time_text = f"Bad Posture Time: {round(bad_time, 1)}s"
            
            cv2.putText(frame, time_text, (10, h - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)