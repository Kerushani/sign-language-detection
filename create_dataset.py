import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "./data"

# array of x, y coords of all hand landmarks
data = []
# records the folder that the image came from
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        # !!normalizes data so results that have exactly 21 landmarks are stored - was causing issues when training classifier
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            if len(hand_landmarks.landmark) == 21:
                for landmark in hand_landmarks.landmark:
                    data_aux.extend([landmark.x, landmark.y])

                    if len(data_aux) == 42:
                        data.append(data_aux)
                        labels.append(dir_)

f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()