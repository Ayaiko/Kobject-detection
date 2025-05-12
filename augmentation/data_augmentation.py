import tensorflow as tf
import numpy as np

# Resize function to resize images to expected input size
def resize_image(image, target_height=480, target_width=640):
    return tf.image.resize(image, [target_height, target_width])

# Convert to RGB
def convert_to_rgb(image):
    return tf.image.grayscale_to_rgb(image) if image.shape[-1] == 1 else image

# Optionally convert to grayscale (for AR markers detection)
def convert_to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)

# Apply random rotation within ±15 degrees
def random_rotation(image):
    return tf.image.rot90(image, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))

# Apply random scaling (±10%) and crop
def random_scaling_and_cropping(image):
    scale_factor = tf.random.uniform([], 0.9, 1.1)
    image = tf.image.resize(image, [int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor)])
    return tf.image.resize_with_crop_or_pad(image, 480, 640)

# Adjust brightness and contrast
def random_brightness_and_contrast(image):
    image = tf.image.random_brightness(image, max_delta=0.3)
    return tf.image.random_contrast(image, lower=0.7, upper=1.3)

# Apply random color jitter (change hue, saturation, etc.)
def random_color_jitter(image):
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_flip_left_right(image)
    return image

def motion_blur(image, size=5):
    # Create a motion blur kernel (size x size)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[int(size / 2), :] = np.ones(size, dtype=np.float32)
    kernel = kernel / size  # Normalize the kernel

    # Convert the kernel to a TensorFlow tensor
    kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)

    # Ensure the kernel has the correct shape for depthwise convolution (height, width, in_channels, 1)
    kernel = kernel[..., np.newaxis, np.newaxis]  # Add depth and channel dimensions
    kernel = tf.tile(kernel, [1, 1, 3, 1])  # Tile the kernel to match the 3 channels of the image

    # Add batch dimension to the image (batch_size, height, width, channels)
    image = image[None, ...]  # Add a batch dimension
    blurred_image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding="SAME")

    return tf.squeeze(blurred_image, axis=0)  # Remove the batch dimension

# Add Gaussian noise
def add_gaussian_noise(image, stddev=0.1):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
    return tf.clip_by_value(image + noise, 0.0, 1.0)

# Simulate distortion (using barrel or pincushion)
def simulate_distortion(image, factor=0.2):
    distortion_matrix = tf.keras.layers.RandomZoom(height_factor=factor, width_factor=factor)
    return distortion_matrix(image)

# Normalize pixel values to [-1, 1]
def normalize(image):
    return (image / 127.5) - 1.0  # Assuming input is [0, 255]

# If quantized model, scale between -128 and 127
def normalize_quantized(image):
    return tf.cast(image, tf.int8) - 128
