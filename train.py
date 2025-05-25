import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data.dataset import get_datasets
from models.mobilenetv2_multitask import build_mobilenetv2_multitask
from tensorflow.keras.optimizers import AdamW

# Parameters
DATA_DIR = 'augmented_images'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-6

train_ds, val_ds, class_names, num_classes = get_datasets(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

model, base_model = build_mobilenetv2_multitask(IMG_HEIGHT, IMG_WIDTH, num_classes)
model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

checkpoint_cb = ModelCheckpoint(
    'kibo_mobilenetv2_multitask_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)
earlystop_cb = EarlyStopping(
    monitor='val_accuracy',
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
    optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint_cb, earlystop_cb]
)

model.save('kibo_mobilenetv2_multitask.h5')
print('Training complete!')
