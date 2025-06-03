import cv2
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    cv2.imshow("Face Features Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
