from ultralytics import YOLO
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2

# Load model
model = YOLO("yolov8n-cls.pt")  # Use the classification version (not detection)
# If only "yolov8n.pt" is available, it won't be usable directly for classification

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess images
def preprocess(images, size=64):  # yolov8n-cls expects 64x64 RGB
    result = []
    for img in images:
        img_resized = cv2.resize(img, (size, size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        result.append(img_rgb)
    return np.array(result)

x_test_proc = preprocess(x_test[:100])  # Test on 100 samples
y_test_true = y_test[:100]

# Predict
correct = 0
for img, true_label in zip(x_test_proc, y_test_true):
    results = model.predict(img, verbose=False)
    probs = results[0].probs
    if probs is not None:
        predicted_class = int(np.argmax(probs.data))
        if predicted_class == true_label:
            correct += 1

# Accuracy
accuracy = correct / len(x_test_proc)
print(f"YOLOv8n classification accuracy on MNIST (100 samples): {accuracy * 100:.2f}%")
