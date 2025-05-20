import tensorflow as tf
from models.mobilenetv2_multitask import build_mobilenetv2_multitask

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 11  # Update if needed

# Load trained model
model = tf.keras.models.load_model('kibo_mobilenetv2_multitask.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('kibo_mobilenetv2_multitask.tflite', 'wb') as f:
    f.write(tflite_model)

print('Export to TFLite complete!')
