import os
import numpy as np
import imageio.v3 as iio
import imgaug.augmenters as iaa

# Define augmentation pipeline using imgaug
AUGMENT_PIPELINE = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Affine(rotate=(-30, 30))),
    iaa.Sometimes(0.5, iaa.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1))),
    iaa.Sometimes(0.5, iaa.AddToBrightness((-30, 30))),
    iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-20, 20))),
    iaa.Sometimes(0.5, iaa.MotionBlur(k=3)),
    iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.03))),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),
    iaa.Sometimes(0.5, iaa.GammaContrast((0.7, 1.5))),
    iaa.Sometimes(0.3, iaa.CoarseDropout((0.02, 0.1), size_percent=(0.02, 0.05))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.01, 0.10))),
    iaa.Sometimes(0.3, iaa.JpegCompression(compression=(70, 99))),
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 1.0))),
])

def preprocess_and_save(image_path, output_path):
    # Load image as numpy array
    image = iio.imread(image_path)
    # Ensure image is uint8 and 3 channels
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2 or image.shape[-1] == 1:
        image = np.stack([image.squeeze()] * 3, axis=-1)
    # Resize to 224x224 (if needed)
    image = iaa.Resize((224, 224))(image=image)
    # Apply augmentations
    aug_image = AUGMENT_PIPELINE(image=image)
    # Save the processed image
    iio.imwrite(output_path, aug_image)

image_folder = 'item_template'
output_folder = 'augmented_images'
num_augmented = 1000  # Number of augmented images per original

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
