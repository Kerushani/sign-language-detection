# Sign Language Detector

A simple sign language detector built with Python, MediaPipe, OpenCV, and scikit-learn. It uses hand landmarks from MediaPipe to classify ASL letters using a RandomForestClassifier.

## How It Works

- MediaPipe detects hand landmarks from webcam input.
- Landmark coordinates (x, y) are extracted and used as features.
- A dataset is built by collecting 100 images per ASL letter.
- The dataset is split into 80% training and 20% testing.
- A `RandomForestClassifier` is trained on the training set and evaluated on the test set.
- The trained model predicts letters in real-time from webcam input.

