import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
    7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M",
    13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S",
    19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

while True:

    data_aux = []
    x_ = []
    y_ = []
    rect, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        x1 = int(min(x_)*W)
        y1 = int(min(y_)*H)

        x2 = int(max(x_)*W)
        y2 = int(max(y_)*H)
        data_aux_np = np.asarray(data_aux)
        if data_aux_np.shape[0] == 42: 
            prediction = model.predict([data_aux_np])[0]
            probs = model.predict_proba([data_aux_np])[0]

            confidence = np.max(probs)
            predicted_character = labels_dict[int(prediction)]

            text = f"{predicted_character} ({confidence*100:.1f}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)