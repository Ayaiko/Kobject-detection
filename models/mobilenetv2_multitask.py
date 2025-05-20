import tensorflow as tf
from tensorflow.keras import layers, models

def build_mobilenetv2_multitask(img_height, img_width, num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
    x = base_model(x, training=False)
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    orient_output = layers.Dense(3, activation='linear', name='orient_output')(x)
    model = models.Model(inputs=inputs, outputs=[class_output, orient_output])
    return model, base_model
