import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
H5_MODEL_PATH = 'kibo_mobilenetv2_multitask.h5'
TFLITE_MODEL_PATH = 'kibo_mobilenetv2_multitask.tflite'

def load_and_preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img, dtype=np.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Get class_names from the training folder structure
class_names = sorted([d for d in os.listdir('augmented_images') if os.path.isdir(os.path.join('augmented_images', d))])

# Use a real image from item_template for testing
try:
    test_img = load_and_preprocess_image('item_template/coin.png')
except FileNotFoundError as e:
    print(e)
    exit(1)

print('Testing Keras .h5 model...')
# Load and test the Keras model
keras_model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
class_pred, orient_pred = keras_model.predict(test_img)
keras_class_idx = np.argmax(class_pred)
keras_class_name = class_names[keras_class_idx] if keras_class_idx < len(class_names) else 'Unknown'
print('Keras model outputs:')
print('  Class probabilities:', class_pred)
print(f'  Predicted class: {keras_class_name} (index {keras_class_idx})')
print('  Orientation (yaw, pitch, roll):', orient_pred)

print('\nTesting TFLite model...')
# Load and test the TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], test_img)
interpreter.invoke()

# Get outputs and fix order if needed
out0 = interpreter.get_tensor(output_details[0]['index'])
out1 = interpreter.get_tensor(output_details[1]['index'])
if out0.shape[1] == class_pred.shape[1]:
    class_pred_tflite = out0
    orient_pred_tflite = out1
else:
    class_pred_tflite = out1
    orient_pred_tflite = out0
tflite_class_idx = np.argmax(class_pred_tflite)
tflite_class_name = class_names[tflite_class_idx] if tflite_class_idx < len(class_names) else 'Unknown'
print('TFLite model outputs:')
print('  Class probabilities:', class_pred_tflite)
print(f'  Predicted class: {tflite_class_name} (index {tflite_class_idx})')
print('  Orientation (yaw, pitch, roll):', orient_pred_tflite)

# Check output shapes
def check_shapes():
    assert class_pred.shape == class_pred_tflite.shape, f"Class output shape mismatch: {class_pred.shape} vs {class_pred_tflite.shape}"
    assert orient_pred.shape == orient_pred_tflite.shape, f"Orientation output shape mismatch: {orient_pred.shape} vs {orient_pred_tflite.shape}"
    print('\nOutput shapes match between Keras and TFLite models.')

check_shapes()

# Evaluate on all images in item_template
def evaluate_on_directory(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    correct_keras = 0
    correct_tflite = 0
    total = 0
    for fname in image_files:
        img_path = os.path.join(directory, fname)
        try:
            img = load_and_preprocess_image(img_path)
        except FileNotFoundError:
            print(f"Warning: {img_path} not found, skipping.")
            continue
        # Keras prediction
        k_pred, _ = keras_model.predict(img)
        keras_class = np.argmax(k_pred)
        # TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        out0 = interpreter.get_tensor(output_details[0]['index'])
        out1 = interpreter.get_tensor(output_details[1]['index'])
        if out0.shape[1] == k_pred.shape[1]:
            tflite_class_pred = out0
        else:
            tflite_class_pred = out1
        tflite_class = np.argmax(tflite_class_pred)
        # True label from filename (e.g., coin.png -> class_names.index('coin'))
        label_name = fname.split('.')[0]
        if label_name in class_names:
            true_label = class_names.index(label_name)
            if keras_class == true_label:
                correct_keras += 1
            if tflite_class == true_label:
                correct_tflite += 1
            total += 1
    keras_acc = correct_keras / total if total > 0 else 0
    tflite_acc = correct_tflite / total if total > 0 else 0
    print(f'Keras model accuracy on {directory}: {keras_acc:.3f} ({correct_keras}/{total})')
    print(f'TFLite model accuracy on {directory}: {tflite_acc:.3f} ({correct_tflite}/{total})')

evaluate_on_directory('item_template')

print('\nTest complete.')
