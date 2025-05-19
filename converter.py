# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('kibo_mobilenetv2_multitask.tflite', 'wb') as f:
    f.write(tflite_model)

print('Training and export complete!')