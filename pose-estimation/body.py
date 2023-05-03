import cv2
import mediapipe as mp
import time

# Inicjalizacja modelu MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_frame(frame):
    # Przetwarzanie obrazu
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Rysowanie pozy na obrazie
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame

def main():
    # Inicjalizacja kamery
    cap = cv2.VideoCapture(0)

    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Przetwarzanie obrazu
        processed_frame = process_frame(frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        # Putting FPS on frame
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Wyświetlanie obrazu
        cv2.imshow('Pose Detection', processed_frame)

        # Wyjście z programu po naciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zwalnianie zasobów i zamykanie okna
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
