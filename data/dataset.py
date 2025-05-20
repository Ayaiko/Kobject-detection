import tensorflow as tf
import os

def get_datasets(data_dir, img_height, img_width, batch_size, seed=42):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=seed
    )
    class_names = dataset.class_names
    num_classes = len(class_names)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    val_batches = int(0.2 * total_batches)
    train_ds = dataset.skip(val_batches)
    val_ds = dataset.take(val_batches)
    return train_ds, val_ds, class_names, num_classes
