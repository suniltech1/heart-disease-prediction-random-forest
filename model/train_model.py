import os
import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'heart.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_model', 'random_forest.pkl')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads the dataset from the given filepath."""
    if not os.path.exists(filepath):
        logging.error(f"Dataset not found at {filepath}.")
        logging.info("Please place the 'heart.csv' dataset in the 'dataset/' directory before running this script.")
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    logging.info(f"Loading dataset from {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame):
    """
    Automatically detects columns, handles missing values, 
    and applies necessary encoding and scaling.
    """
    logging.info("Preprocessing data...")
    # Assume target is the last column
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Automatically detect numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logging.info(f"Detected numerical features: {numeric_features}")
    logging.info(f"Detected categorical features: {categorical_features}")

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor, numeric_features, categorical_features

def evaluate_model(y_true, y_pred, y_prob=None):
    """Calculates and returns evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        try:
            metrics['ROC AUC'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['ROC AUC'] = None
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Generates and saves the confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_DIR, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'))
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name: str):
    """Generates and saves the ROC curve."""
    if y_prob is None:
        return
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='orange', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f'roc_curve_{model_name.replace(" ", "_").lower()}.png'))
        plt.close()
    except Exception as e:
        logging.error(f"Could not plot ROC curve: {e}")

def plot_feature_importance(model, feature_names: list):
    """Generates and saves feature importance chart for Random Forest."""
    if not hasattr(model, 'feature_importances_'):
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance_random_forest.png'))
    plt.close()

def main():
    # 1. Load Dataset
    df = load_dataset(DATASET_PATH)
    
    # 2 & 3. Preprocess Data
    X, y, preprocessor, num_cols, cat_cols = preprocess_data(df)
    
    # 4. Train/Validation/Test Split (60/20/20, random_state=42)
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)
    
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = []
    trained_rf_pipeline = None
    feature_names_out = []

    logging.info("Training and evaluating models...")
    for name, clf in models.items():
        # Create a pipeline that combines preprocessor and the classifier
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', clf)])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict on validation set
        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Probability for ROC AUC
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
            
        # 7. Evaluate on test set
        test_metrics = evaluate_model(y_test, y_pred, y_prob)
        
        # Combine metrics
        metrics = {'Model': name, 'Val Accuracy': val_acc}
        metrics.update(test_metrics)
        
        results.append(metrics)
        
        # Save plots for Random Forest
        if name == "Random Forest":
            trained_rf_pipeline = pipeline
            plot_confusion_matrix(y_test, y_pred, name)
            plot_roc_curve(y_test, y_prob, name)
            
            # Extract feature names after preprocessing
            try:
                # Preprocessor creates new feature names for one-hot encoded columns
                num_features = num_cols
                
                # Get category names from OneHotEncoder
                if 'cat' in preprocessor.named_transformers_:
                    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
                    cat_features = ohe.get_feature_names_out(cat_cols).tolist()
                else:
                    cat_features = []
                    
                feature_names_out = num_features + cat_features
                
                # Plot Feature Importance
                rf_model = pipeline.named_steps['classifier']
                plot_feature_importance(rf_model, feature_names_out)
            except Exception as e:
                logging.error(f"Could not extract feature names or plot importance: {e}")

    # 8. Print comparison table
    results_df = pd.DataFrame(results).set_index('Model')
    print("\nModel Comparison Table:")
    print(results_df.to_string())

    # 9. Save Random Forest model & components
    if trained_rf_pipeline:
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        
        # Save pipeline and metadata as a dictionary
        model_data = {
            'pipeline': trained_rf_pipeline,
            'feature_names_in': X.columns.tolist(),
            'numerical_columns': num_cols,
            'categorical_columns': cat_cols
        }
        
        joblib.dump(model_data, MODEL_SAVE_PATH)
        logging.info(f"Saved Random Forest model and metadata to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
