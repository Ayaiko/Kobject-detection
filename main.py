import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set parameters
data_dir = 'augmented_images'
img_height = 224
img_width = 224
batch_size = 32
num_epochs = 3

# Prepare the dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=42
)

class_names = dataset.class_names
num_classes = len(class_names)

# Prefetch for performance
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Split into train/val
total_batches = tf.data.experimental.cardinality(dataset).numpy()
val_batches = int(0.2 * total_batches)
train_ds = dataset.skip(val_batches)
val_ds = dataset.take(val_batches)

# Data augmentation (optional, since you already augment offline)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Build the model with transfer learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False  # Freeze base model

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
x = base_model(x, training=False)

# Classification head
class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
# Orientation head (yaw, pitch, roll)
orient_output = layers.Dense(3, activation='linear', name='orient_output')(x)

model = models.Model(inputs=inputs, outputs=[class_output, orient_output])

model.compile(
    optimizer='adam',
    loss={
        'class_output': 'categorical_crossentropy',
        'orient_output': 'mse'
    },
    metrics={'class_output': 'accuracy'}
)

# Dummy orientation labels (replace with real orientation data if available)
def add_dummy_orientation(ds):
    def add_orientation(image, label):
        # Replace np.zeros(3) with your real orientation data
        orientation = tf.zeros((3,), dtype=tf.float32)
        return image, {'class_output': label, 'orient_output': orientation}
    return ds.map(add_orientation, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = add_dummy_orientation(train_ds)
val_ds = add_dummy_orientation(val_ds)

# Callbacks for saving and early stopping
checkpoint_cb = ModelCheckpoint(
    'kibo_mobilenetv2_multitask_best.h5',
    monitor='val_class_output_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)
earlystop_cb = EarlyStopping(
    monitor='val_class_output_accuracy',
    patience=5,
    mode='max',
    restore_best_weights=True,
    verbose=1
)

# Train the model with callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Optionally, unfreeze and fine-tune
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss={
        'class_output': 'categorical_crossentropy',
        'orient_output': 'mse'
    },
    metrics={'class_output': 'accuracy'}
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Save the model
model.save('kibo_mobilenetv2_multitask.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('kibo_mobilenetv2_multitask.tflite', 'wb') as f:
    f.write(tflite_model)

print('Training and export complete!')