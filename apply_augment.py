import os
from augmentation.data_augmentation import *
import tensorflow as tf

# Configurable probabilities for each augmentation
AUGMENT_PROBS = {
    'random_rotation': 0.5,
    'random_scaling_and_cropping': 0.5,
    'random_brightness_and_contrast': 0.5,
    'random_color_jitter': 0.5,
    'motion_blur': 0.5,
    'simulate_distortion': 0.5,
    'add_gaussian_noise': 0.5,
}

def preprocess_and_save(image_path, output_path):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Assuming PNG, RGB

    # Apply the preprocessing steps with configurable probabilities
    image = resize_image(image)
    # image = convert_to_rgb(image)
    if tf.random.uniform([]) < AUGMENT_PROBS['random_rotation']:
        image = random_rotation(image)
    if tf.random.uniform([]) < AUGMENT_PROBS['random_scaling_and_cropping']:
        image = random_scaling_and_cropping(image)
    if tf.random.uniform([]) < AUGMENT_PROBS['random_brightness_and_contrast']:
        image = random_brightness_and_contrast(image)
    if tf.random.uniform([]) < AUGMENT_PROBS['random_color_jitter']:
        image = random_color_jitter(image)
    if tf.random.uniform([]) < AUGMENT_PROBS['motion_blur']:
        image = motion_blur(image)
    if tf.random.uniform([]) < AUGMENT_PROBS['simulate_distortion']:
        image = simulate_distortion(image)
    image = normalize(image)
    if tf.random.uniform([]) < AUGMENT_PROBS['add_gaussian_noise']:
        image = add_gaussian_noise(image)

    # Save the processed image
    tf.io.write_file(output_path, tf.image.encode_png(tf.cast((image + 1) * 127.5, tf.uint8)))  # Denormalize and save as PNG

image_folder = 'item_template'
output_folder = 'augmented_images'
num_augmented = 10  # Number of augmented images per original

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    base, ext = os.path.splitext(image_name)
    class_folder = os.path.join(output_folder, base)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    for i in range(num_augmented):
        out_name = f"{base}_aug_{i}{ext}"
        out_path = os.path.join(class_folder, out_name)
        preprocess_and_save(image_path, out_path)
