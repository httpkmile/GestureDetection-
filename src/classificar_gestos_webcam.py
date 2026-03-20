import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time

MODEL_FILENAME = "models/modelo_gestos.pkl"
HAND_MODEL_PATH = "models/hand_landmarker.task"

if not os.path.exists(MODEL_FILENAME):
    print(f"Erro: O modelo '{MODEL_FILENAME}' não foi encontrado. Execute 'treinar_modelo_gestos.py' primeiro.")
    exit()

model = joblib.load(MODEL_FILENAME)
print(f"Modelo {MODEL_FILENAME} carregado!")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

def draw_hand_landmarks(frame, hand_landmarks):
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
    ]
    h, w, _ = frame.shape
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        p1 = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
        p2 = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
        cv2.line(frame, p1, p2, (255, 255, 255), 2)
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

cap = cv2.VideoCapture(0)
last_timestamp_ms = 0

print("Iniciando reconhecimento de gestos... Pressione 'q' para sair.")

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):

                    handedness = result.handedness[i][0].category_name

                    mao_tipo = "Direita" if handedness == "Left" else "Esquerda"

                    draw_hand_landmarks(frame, hand_landmarks)

                    features = []
                    for lm in hand_landmarks:
                        features.extend([lm.x, lm.y, lm.z])

                    features = np.array(features).reshape(1, -1)
                    prediction = model.predict(features)[0]

                    probs = model.predict_proba(features)
                    confidence = np.max(probs)

                    txt = f"{mao_tipo}: {prediction} ({confidence:.2%})"

                    px, py = int(hand_landmarks[0].x * frame.shape[1]), int(hand_landmarks[0].y * frame.shape[0])
                    cv2.putText(frame, txt, (px, py + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Mao nao detectada", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Classificador de Gestos em Tempo Real', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
