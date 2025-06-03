import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155, 133, 7, 163, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382, 263, 249, 390, 373, 374, 380, 381, 382]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = frame.shape

            for idx in LEFT_EYE + LEFT_IRIS:
                x = int(face_landmarks.landmark[idx].x * iw)
                y = int(face_landmarks.landmark[idx].y * ih)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for idx in RIGHT_EYE + RIGHT_IRIS:
                x = int(face_landmarks.landmark[idx].x * iw)
                y = int(face_landmarks.landmark[idx].y * ih)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow('Eye Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
