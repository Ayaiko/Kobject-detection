# Kobject-detection

A repository for experimenting with object detection models, including data augmentation for training custom classifiers (e.g., for robotics or Kibo-RPC challenges).

## Setup

1. **Clone the repository:**
   ```powershell
   git clone <your-repo-url>
   cd Kobject-detection
   ```

2. **(Optional) Create a virtual environment:**
   ```powershell
   python -m venv env
   .\env\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Data Preparation

- Place your original class images in the `item_template/` folder.  
  Each image should be named after its class (e.g., `diamond.png`, `coin.png`).

- To generate augmented images for training, run:
   ```powershell
   python apply_augment.py
   ```
  This will create an `augmented_images/` folder with a subfolder for each class, containing multiple augmented images.

## Training

- Use the images in `augmented_images/` for training your object detection or classification model.
- You can use `tf.keras.utils.image_dataset_from_directory` or similar utilities to load the dataset, as the folder structure is compatible.

## Customization

- Edit `augmentation/data_augmentation.py` to adjust or add augmentation techniques.
- Change the number of augmentations per image by modifying `num_augmented` in `apply_augment.py`.

## Notes

- The project is designed for easy extension to new classes: just add new images to `item_template/` and re-run the augmentation script.
- Model training scripts (e.g., `main.py`, `yolo.py`) should be adapted to your specific use case.
