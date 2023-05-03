import cv2
import mediapipe as mp
import time

# Inicjalizacja MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Konfiguracja rysowania punktów na twarzy
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Użycie kamery
cap = cv2.VideoCapture(0)
width, height = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Inicjalizacja zmiennej do przechowywania czasu
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Przekształcenie obrazu na format RGB i przeskalowanie
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    # Wykrywanie punktów na twarzy
    results = face_mesh.process(frame_rgb)

    # Rysowanie punktów na twarzy
    frame.flags.writeable = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    # Putting FPS on frame
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Wyświetlanie obrazu
    cv2.imshow('Face Landmarks', frame)

    # Wyjście z pętli po naciśnięciu klawisza 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów kamery i zamknięcie okna
cap.release()
cv2.destroyAllWindows()
