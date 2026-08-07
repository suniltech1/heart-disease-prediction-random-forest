# Heart Disease Prediction with Random Forest

A Streamlit-based heart disease prediction app built as a university final-year project. The application uses a Random Forest machine learning model to predict the likelihood of heart disease from patient medical features.

## Project Overview

- `app/app.py`: Streamlit frontend for entering patient data and generating heart disease predictions.
- `model/train_model.py`: Training script that preprocesses the dataset, trains multiple classifiers, evaluates them, and saves the best Random Forest model.
- `model/predict.py`: Loads the saved model and makes predictions for new patient records.
- `dataset/heart_cleveland_upload-selected-columns.csv`: Included dataset for training and testing.
- `saved_model/random_forest.pkl`: Serialized trained Random Forest model.
- `images/`: Generated visualizations including feature importance and model evaluation charts.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The project uses:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib

## Setup

1. Ensure your working directory is the project root.
2. Confirm the dataset file exists in `dataset/`.
   - The training code expects `dataset/heart.csv` by default.
   - If your dataset file is named `heart_cleveland_upload-selected-columns.csv`, rename it or copy it as `dataset/heart.csv`.

## Train the Model

Train the Random Forest model and generate evaluation plots:

```bash
python model/train_model.py
```

This script will:

- load and preprocess the dataset
- train multiple classifiers
- evaluate performance on a test split
- save the best Random Forest model to `saved_model/random_forest.pkl`
- generate plots in `images/`

## Run the Streamlit App

Launch the web app:

```bash
streamlit run app/app.py
```

Then open the local URL shown in the console (typically `http://localhost:8501`).

## Using the App

1. Enter patient features in the form.
2. Click **Predict Heart Disease**.
3. Review the diagnosis, confidence score, risk level, and the feature importance chart.

> Note: This project is for educational purposes and is not a medical diagnostic tool.

## File Structure

```text
README.md
requirements.txt
app/
  app.py
dataset/
  heart_cleveland_upload-selected-columns.csv
images/
  feature_importance_random_forest.png
  confusion_matrix_random_forest.png
  roc_curve_random_forest.png
model/
  train_model.py
  predict.py
saved_model/
  random_forest.pkl
```

## Notes

- If the Streamlit app fails to load, verify `saved_model/random_forest.pkl` exists.
- If the dataset file is missing or has a different name, update the path in `model/train_model.py` and `app/app.py` or rename the file to `heart.csv`.
- The model uses a Random Forest classifier and feature importance plots are generated during training.

## License

This repository is provided for educational use.
