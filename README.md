# AI-Based Heart Disease Prediction System Using Random Forest

## Project Description
This is a final year university AI coursework project focused on predicting the presence or absence of heart disease in patients based on a set of medical features. It uses machine learning, primarily a Random Forest Classifier, alongside several baseline models for performance comparison. A modern and simple Streamlit web application allows users to input patient details and receive real-time predictions, risk probabilities, and confidence scores.

## Objectives
- Automatically preprocess various medical datasets (handle missing values, encode categorical variables, scale numerical features).
- Train and evaluate multiple machine learning baseline models (Logistic Regression, Decision Tree, Naive Bayes).
- Implement a robust primary algorithm (Random Forest Classifier).
- Create a clean, modular, and beginner-friendly codebase following PEP8 standards.
- Develop an interactive web interface using Streamlit to showcase predictions with dynamic patient input forms.

## Project Structure
```text
HeartDiseasePrediction/
│
├── dataset/             # Contains the heart.csv dataset (provided later)
├── model/               # Machine learning scripts
│      train_model.py    # Code to load, preprocess, train, and evaluate models
│      predict.py        # Code to load the saved model and make new predictions
├── app/                 # Streamlit web application
│      app.py            # Main application interface
├── saved_model/         # Directory for storing the trained model artifact
│      random_forest.pkl # Pickled Random Forest model pipeline
├── images/              # Evaluation charts and plots
├── report/              # Final coursework reports and documentation
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Installation
1. Ensure you have Python 3.12+ installed.
2. Clone or download this repository.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Dataset**: Place your dataset named `heart.csv` into the `dataset/` directory. The codebase is designed to automatically detect features, target columns, and data types.
2. **Training**: Run the training script to evaluate models and save the Random Forest model:
   ```bash
   python -m model.train_model
   ```
3. **Application**: Launch the Streamlit web interface to make predictions:
   ```bash
   streamlit run app/app.py
   ```

## Dataset
The dataset (`heart.csv`) should be placed in the `dataset/` directory. The project will dynamically adapt to the dataset by detecting numerical and categorical columns. The target variable should ideally be at the end of the dataset.

## Algorithms
- **Primary Algorithm**: Random Forest Classifier
- **Baseline Models**: Logistic Regression, Decision Tree Classifier, Gaussian Naive Bayes

## Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix

Charts for Feature Importance and ROC Curves are saved in the `images/` directory.

## Future Improvements
- Integrate Deep Learning models if necessary.
- Connect to an SQL/NoSQL database to store patient history.
- Implement an authentication system for healthcare professionals.
- Provide explainable AI (XAI) interpretations like SHAP values in the dashboard.
