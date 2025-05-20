import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data.dataset import get_datasets
from models.mobilenetv2_multitask import build_mobilenetv2_multitask

# Parameters
DATA_DIR = 'augmented_images'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_EPOCHS = 3

train_ds, val_ds, class_names, num_classes = get_datasets(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

def add_dummy_orientation(ds):
    def add_orientation(image, label):
        orientation = tf.zeros((3,), dtype=tf.float32)
        return image, {'class_output': label, 'orient_output': orientation}
    return ds.map(add_orientation, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = add_dummy_orientation(train_ds)
val_ds = add_dummy_orientation(val_ds)

model, base_model = build_mobilenetv2_multitask(IMG_HEIGHT, IMG_WIDTH, num_classes)
model.compile(
    optimizer='adam',
    loss={
        'class_output': tf.keras.losses.CategoricalCrossentropy(),
        'orient_output': tf.keras.losses.MeanSquaredError()
    },
    metrics={'class_output': 'accuracy'}
)

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

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Optionally, unfreeze and fine-tune
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss={
        'class_output': tf.keras.losses.CategoricalCrossentropy(),
        'orient_output': tf.keras.losses.MeanSquaredError()
    },
    metrics={'class_output': 'accuracy'}
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[checkpoint_cb, earlystop_cb]
)

model.save('kibo_mobilenetv2_multitask.h5')
print('Training complete!')
