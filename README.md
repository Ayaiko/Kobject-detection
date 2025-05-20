# Kobject-detection

A repository for experimenting with object detection models, including data augmentation for training custom classifiers (e.g., for robotics or Kibo-RPC challenges).

## Requirements

- **Python 3.10 is required.**
- All dependencies are listed in `requirements.txt` (generated via `pip freeze` for full reproducibility).

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Kobject-detection
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python3.10 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

- Place your original class images in the `item_template/` folder.  
  Each image should be named after its class (e.g., `diamond.png`, `coin.png`).

- To generate augmented images for training, run:
   ```bash
   python apply_augment.py
   ```
  This will create an `augmented_images/` folder with a subfolder for each class, containing multiple augmented images.

## Training

- Use the images in `augmented_images/` for training your object detection or classification model.
- You can use `tf.keras.utils.image_dataset_from_directory` or similar utilities to load the dataset, as the folder structure is compatible.
- Training scripts are modularized: see `train.py`, `data/augmentation.py`, `data/dataset.py`, and `models/mobilenetv2_multitask.py` for details.

## Customization

- Edit `data/augmentation.py` to adjust or add augmentation techniques.
- Change the number of augmentations per image by modifying `num_augmented` in `apply_augment.py`.
- Augmentation probabilities are configurable via the `AUGMENT_PROBS` dictionary.

## Notes

- The project is designed for easy extension to new classes: just add new images to `item_template/` and re-run the augmentation script.
- Model training and evaluation scripts (e.g., `main.py`, `train.py`, `test_model.py`) are provided and should be adapted to your specific use case.
- All models are designed for compatibility with TensorFlow Lite and real-time inference on resource-constrained hardware (e.g., Kibo RPC Astrobee).
- See `kibo_context.txt` and the competition rules for further details on requirements and scoring.
