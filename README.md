# Sign Language Detector

A simple sign language detector built with Python, MediaPipe, OpenCV, and scikit-learn. It uses hand landmarks from MediaPipe to classify ASL letters using a RandomForestClassifier.

![b162b9e28ef3d23d4b58ad74fbd6d0f8](https://github.com/user-attachments/assets/8264c09d-47ca-46f3-bbe8-374d4c1dd4b2)

https://github.com/user-attachments/assets/e0a92529-932c-4e3c-8893-27996e4d8690

## How It Works

- MediaPipe detects hand landmarks from webcam input.
- Landmark coordinates (x, y) are extracted and used as features.
- A dataset is built by collecting 100 images per ASL letter.
- The dataset is split into 80% training and 20% testing.
- A `RandomForestClassifier` is trained on the training set and evaluated on the test set.
- The trained model predicts letters in real-time from webcam input.

