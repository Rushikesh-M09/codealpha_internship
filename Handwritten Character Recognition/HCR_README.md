# Handwritten Character Recognition (HCR)

A collection of experiments for recognizing handwritten characters — starting from simple digit classification and building up to full letter recognition and real-time handwriting-to-text extraction from video.

## Overview

This notebook is organized as a progression of increasingly capable handwriting-recognition approaches, each in its own cell/script:

1. **CNN digit recognizer** — a Keras/TensorFlow convolutional network trained on the full MNIST dataset (28x28 digits)
2. **CNN letter recognizer** — the same CNN approach extended to the EMNIST "letters" dataset (a–z)
3. **Video/webcam handwriting-to-text** — an OpenCV pipeline that locates handwritten characters in video frames, classifies them with the trained letter model, and reconstructs lines of text
4. **Single-image inference script** — a quick test harness to run the trained letter model on one image file
5. **EMNIST Balanced dataset exploration** — loads and visualizes the 47-class EMNIST Balanced split (digits + uppercase + selected lowercase)

## Components

### 1. CNN Digit Recognizer (MNIST)
Trains a CNN on the standard MNIST dataset (60,000 train / 10,000 test, 28x28 grayscale digits).

- **Architecture:** `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dropout(0.5) → Dense(10, softmax)`
- **Training:** Adam optimizer, categorical cross-entropy loss, 10 epochs, batch size 128, 10% validation split
- **Outputs:** saves the trained model to `digit_recognizer.keras` and training curves to `training_history.png`

### 2. CNN Letter Recognizer (EMNIST Letters)
Extends the digit model to the EMNIST "letters" split (~145,000 images, 26 case-insensitive classes a–z), loaded via `tensorflow_datasets`. Images are transposed to correct EMNIST's stored orientation.

- **Architecture:** a deeper CNN — `Conv2D(32) → BatchNorm → MaxPool → Conv2D(64) → BatchNorm → MaxPool → Conv2D(128) → BatchNorm → Flatten → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(27, softmax)`
- **Training:** Adam optimizer, categorical cross-entropy, 10 epochs
- **Outputs:** saves the trained model to `letter_recognizer.keras` and training curves to `letter_training_history.png`

### 3. Video/Webcam Handwriting Recognition
A classical computer-vision pipeline (no end-to-end video-to-text model) that:

1. Reads frames from a video file or webcam
2. Thresholds each frame (Otsu's method) to separate ink from background
3. Finds contours as candidate character blobs, filtering by area
4. Groups blobs into text lines by vertical position, sorted left-to-right
5. Crops, pads, and resizes each character to 28x28 and classifies it with the trained letter model
6. Reconstructs lines of text, inserting spaces at large horizontal gaps
7. Draws bounding boxes and predictions on the video and prints recognized text

**Known limitations** (documented in the script itself): works best on clean, high-contrast, print handwriting filmed straight-on; cursive/joined-up writing will not segment correctly, since the method assumes each character is a separate connected blob. For production OCR on messy real-world input, the script suggests text-detection + sequence-recognition approaches (e.g., EAST + CRNN with CTC loss) or an existing OCR engine (Tesseract, EasyOCR).

### 4. Single-Image Inference
A minimal script that loads `letter_recognizer.keras` and a class mapping from `emnist_classes.txt`, reads a single image (`letters_img.jpg`), inverts/resizes/normalizes it, and prints the predicted character with confidence. Displays the preprocessed image in a window.

### 5. EMNIST Balanced Exploration
Loads the EMNIST "balanced" split (47 classes: digits 0–9, uppercase A–Z, and a set of visually distinct lowercase letters) via `tensorflow_datasets`, applies normalization and orientation correction, and visualizes a batch of labeled sample images in a grid.

## Datasets Used

| Dataset | Source | Classes | Image Size |
|---|---|---|---|
| MNIST | `keras.datasets.mnist` (auto-downloaded, ~11MB) | 10 (digits) | 28x28 |
| EMNIST Letters | `tensorflow_datasets` (`emnist/letters`, auto-downloaded, ~500MB) | 26 (a–z) | 28x28 |
| EMNIST Balanced | `tensorflow_datasets` (`emnist/balanced`) | 47 (digits + upper + select lower) | 28x28 |

## Requirements

```
tensorflow            # or tensorflow-macos + tensorflow-metal on Apple Silicon
tensorflow-datasets
numpy
matplotlib
opencv-python
```

Install with:
```bash
pip install tensorflow tensorflow-datasets numpy matplotlib opencv-python
```

## Usage

Run the notebook cells in order, or extract individual sections as standalone scripts:

```bash
jupyter notebook HCR.ipynb
```

For the video/webcam recognition step (once a letter model has been trained and saved):
```bash
python video_text_recognition.py --source my_video.mp4   # from a file
python video_text_recognition.py --source 0               # from a webcam
```

**Note:** The video-recognition and single-image inference sections expect artifacts produced by the earlier training sections to already exist on disk:
- `letter_recognizer.keras` (falls back to `digit_recognizer.keras` if missing)
- `emnist_classes.txt` (class-index-to-character mapping)
- `letters_img.jpg` (for the single-image test script)

Train the corresponding model first, or supply your own matching files, before running these sections.

## Project Structure

```
.
├── HCR.ipynb                     # Main notebook: SVM/CNN training, EMNIST exploration, video OCR pipeline
├── digit_recognizer.keras        # (generated) trained MNIST CNN
├── letter_recognizer.keras       # (generated) trained EMNIST letters CNN
├── emnist_classes.txt            # (expected) class index -> character mapping
├── training_history.png          # (generated) MNIST training curves
├── letter_training_history.png   # (generated) EMNIST training curves
└── README.md                     # Project documentation
```

## Notes & Caveats

- The digit and letter CNNs are trained with fixed hyperparameters and no cross-validation or tuning.
- EMNIST images require a transpose to correct orientation — this is handled in the preprocessing steps but is easy to miss if adapting the code.
- The video pipeline is a classical segmentation + per-character classification approach, not a true handwriting recognition (HTR) sequence model — it will not handle cursive or heavily connected script.

## Possible Extensions

- Replace the contour-based video pipeline with a proper text-detection + sequence model (e.g., CRNN + CTC) for cursive support
- Combine the digit and letter models into a single unified classifier
- Add data augmentation (rotation, shear, noise) to improve robustness on real-world handwriting
- Package the trained models behind a simple inference API or web demo
