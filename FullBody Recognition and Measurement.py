import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def pixel_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

LANDMARK_PAIRS = {
    "Shoulder Width": (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_SHOULDER),
    "Hip Width": (mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.RIGHT_HIP),
    "Torso Length": (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_HIP),
    "Left Arm Length": (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_WRIST),
    "Right Arm Length": (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_WRIST),
    "Left Leg Length": (mp_holistic.PoseLandmark.LEFT_HIP, mp_holistic.PoseLandmark.LEFT_ANKLE),
    "Right Leg Length": (mp_holistic.PoseLandmark.RIGHT_HIP, mp_holistic.PoseLandmark.RIGHT_ANKLE),
    "Height (Nose to Ankle)": (mp_holistic.PoseLandmark.NOSE, mp_holistic.PoseLandmark.LEFT_ANKLE),
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    ih, iw, _ = frame.shape

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        landmarks = results.pose_landmarks.landmark
        measurements = {}
        for label, (idx1, idx2) in LANDMARK_PAIRS.items():
            p1 = landmarks[idx1.value]
            p2 = landmarks[idx2.value]
            if p1.visibility > 0.5 and p2.visibility > 0.5:
                point1 = (int(p1.x * iw), int(p1.y * ih))
                point2 = (int(p2.x * iw), int(p2.y * ih))
                dist = pixel_distance(point1, point2)
                measurements[label] = dist
                cv2.line(frame, point1, point2, (0, 255, 255), 2)
                cv2.putText(frame, label, ((point1[0]+point2[0])//2, (point1[1]+point2[1])//2-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y0 = 30
        for label, value in measurements.items():
            text = f"{label}: {int(value)} px"
            cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y0 += 22

    if results.face_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_draw.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )

    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
        )

    cv2.imshow("All-in-One Human Recognition (Body, Face, Hands, Measurements)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()