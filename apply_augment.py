import os
from augmentation.data_augmentation import *

def preprocess_and_save(image_path, output_folder):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Assuming PNG, RGB
    
    # Apply the preprocessing steps
    image = resize_image(image)
    #image = convert_to_rgb(image)
    image = random_rotation(image)
    image = random_scaling_and_cropping(image)
    image = random_brightness_and_contrast(image)
    image = random_color_jitter(image)
    image = motion_blur(image)
    #image = add_gaussian_noise(image)
    image = simulate_distortion(image)
    image = normalize(image)
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the processed image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    tf.io.write_file(output_path, tf.image.encode_png(tf.cast((image + 1) * 127.5, tf.uint8)))  # Denormalize and save as PNG

# Example usage
image_folder = 'item_template/dummy_class'
output_folder = 'augmented_images'

# Process and save images
for image_path in os.listdir(image_folder):
    preprocess_and_save(os.path.join(image_folder, image_path), output_folder)
