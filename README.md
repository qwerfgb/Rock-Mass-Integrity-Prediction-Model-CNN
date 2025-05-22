# Rock Mass Integrity Prediction Model - Multimodal Deep Learning Approach

## Project Introduction

This project is a deep learning model based on multimodal data, specifically designed for predicting rock mass integrity
classification. The model integrates numerical features and image features, employing a convolutional neural network (
CNN) deep learning architecture to achieve accurate classification of rock mass integrity.

## Model Features

- **Multimodal Fusion**: Processes numerical data and image data simultaneously to improve prediction accuracy
- **Automated Feature Extraction**: Automatically extracts key features through image preprocessing
- **High-Precision Classification**: Utilizes deep learning models to identify different levels of rock mass integrity
- **Visualization Analysis**: Provides feature importance and sensitivity analysis

## System Requirements

- Python 3.8
- Dependencies listed in the `requirements.txt` file

## Installation Guide

1. Clone the repository to your local machine

```bash
git clone https://github.com/qwerfgb/Rock-Mass-Integrity-Prediction-Model-CNN.git
cd Rock-Mass-Integrity-Prediction-Model-CNN
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Data Input

The model requires two types of input data, both in Excel format:

1. **Numerical Feature Data**: Contains quantitative indicators related to rock mass
2. **Image Feature Data**: Features extracted from raw images using `data/PreprocessImage.py`

### Image Preprocessing Workflow

Raw image data is saved in different folders according to integrity level, with folder names serving as category labels.
Use `data/PreprocessImage.py` for feature extraction:

1. Read images from classification folders
2. Extract image features
3. Save features as Excel files for subsequent model input

## Project Structure

```
Rock-Mass-Integrity-Prediction-Model-CNN/
│
├── data/                          # Data processing module
│   ├── __init__.py                # Initialization file
│   └── PreprocessImage.py         # Image feature extraction script
│
├── model/                         # Model definition module
│   ├── __init__.py                # Initialization file
│   └── CNN.py                     # CNN architecture classification model
│
├── README.md                      # Project documentation
└── requirements.txt               # Dependency list
```

## Results Output

After training, the model will generate the following outputs:

1. Training and validation accuracy and loss curves
2. Detailed classification performance report (accuracy, precision, recall, F1 score)
3. Feature importance ranking and visualization
4. Feature sensitivity analysis

## Notes

- Required Python version: 3.8
- Required dependencies are listed in the requirements.txt file
- During data preprocessing, images should be correctly classified into different folders according to integrity level
- The format of Excel files after feature extraction must meet the model input requirements
- Label values should be integers from 1 to n, representing different rock mass integrity levels
