import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Path to test images and model
TEST_DIR = 'test_images'
MODEL_PATH = 'kibo_mobilenetv2_multitask_best.h5'
IMG_HEIGHT = 224  # Change if your model uses a different size
IMG_WIDTH = 224

# Get class names from test image filenames (e.g., coin.png -> coin)
class_names = [os.path.splitext(f)[0] for f in os.listdir(TEST_DIR) if f.endswith('.png') or f.endswith('.jpg')]
class_names.sort()

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

correct = 0
total = 0

for fname in os.listdir(TEST_DIR):
    if not (fname.endswith('.png') or fname.endswith('.jpg')):
        continue
    img_path = os.path.join(TEST_DIR, fname)
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)  # Match model preprocessing
    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds)]
    true_class = os.path.splitext(fname)[0]
    is_correct = (pred_class == true_class)
    print(f"Image: {fname} | Predicted: {pred_class} | True: {true_class} | Correct: {is_correct}")
    correct += int(is_correct)
    total += 1

print(f"\nAccuracy: {correct}/{total} = {correct/total:.2f}")
