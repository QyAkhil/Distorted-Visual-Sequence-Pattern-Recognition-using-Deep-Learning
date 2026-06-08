# Distorted Visual Sequence Pattern Recognition

A deep learning pipeline to recognize and reconstruct text sequences from visually distorted grayscale images using a CRNN architecture with CTC loss.

## Folder Structure

```
Distorted Visual Sequence Pattern Recognition/
├── data/
│   └── cig_ps/
│       ├── train_images/        # labeled training images
│       ├── test_images/         # unlabeled test images
│       └── train-labels.csv     # image filename → text label
├── pipeline.ipynb               # full training and inference notebook
├── best_crnn.pth                # saved best model checkpoint
├── submission_Akhil_24117009.csv# final predictions
├── requirements.txt             # dependencies
└── .gitignore
```

## Model Architecture

**CRNN (Convolutional Recurrent Neural Network)**
- CNN backbone — extracts spatial features, compresses height while preserving width
- BiLSTM — reads character sequence in both directions
- CTC Loss — handles variable length outputs without needing character segmentation

## Preprocessing & Augmentation

- CLAHE contrast enhancement on all images
- Training augmentation: Gaussian blur
- Normalized to `[-1, 1]`

## Evaluation

Submissions are evaluated using **Character Error Rate (CER)** based on Levenshtein distance. Lower is better.
Best validation CER achieved on 20% split of test data is 0.0003.

## How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Place dataset in `data/cig_ps/`

3. Run `pipeline.ipynb` end to end

## Requirements

See `requirements.txt` and also make sure you have appropriate cuda version installed along with a Nvidea GPU for a faster model training. 
