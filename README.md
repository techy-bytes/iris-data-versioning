# IRIS Dataset Versioning with DVC

This project demonstrates data versioning using DVC with the existing IRIS dataset.

## File Descriptions:

### `train.py`
Main training script that builds a Random Forest classifier on the IRIS dataset. The script:
- Loads the IRIS dataset from `data/iris.csv`
- Performs train-test split with 70-30 ratio
- Trains a Random Forest classifier with 100 estimators
- Evaluates model performance and saves metrics to `metrics.csv`
- Prints dataset statistics including shape, accuracy, and class distribution

### `double_iris.py`
Data preprocessing script that doubles the size of the IRIS dataset by concatenating the original dataset with itself. This creates version 2.0 of the dataset with 300 samples instead of the original 150 samples.

### `requirements.txt`
Contains all Python dependencies required for this project:
- `dvc` - Data Version Control for tracking dataset versions
- `scikit-learn` - Machine learning library for model training
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing support

### `commands.md`
Comprehensive step-by-step guide containing all DVC and Git commands needed to:
- Set up the project environment
- Initialize DVC repository
- Download and track data with DVC
- Create different dataset versions (v1.0 and v2.0)
- Switch between versions using Git tags and DVC checkout
- Verify dataset changes and model performance

## Versions:
- v1.0: Original IRIS dataset (150 samples, accuracy: ~0.8889)
- v2.0: Doubled IRIS dataset (300 samples, accuracy: ~0.9333)
