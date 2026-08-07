---
# PRELIMINARY PAGES
---

<div align="center">
  
# Title of the Project
**AI-Based Heart Disease Prediction System Using Random Forest**

**Student Name(s) and ID(s):** [Your Name] - [Your ID]  
**Course Title and Code:** [Course Title] - [Course Code]  
**Institution Name:** [Institution Name]  
**Submission Date:** [Date]  

</div>

<div style="page-break-after: always;"></div>

## Acknowledgement
*(Page ii)*  
[Write your acknowledgements here. Thank your supervisor, institution, and anyone who helped you.]

<div style="page-break-after: always;"></div>

## Abstract
*(Page iii)*  
Heart disease remains one of the leading causes of mortality worldwide. Early detection is critical for effective treatment and prevention. This project develops an Artificial Intelligence-based system to predict the likelihood of heart disease in patients using medical records. We implemented a robust machine learning pipeline, primarily utilizing the Random Forest Classifier, alongside baseline models including Logistic Regression, Decision Trees, and Naive Bayes for comparative analysis. The system automatically preprocesses data, handling missing values and scaling features, before making predictions. A user-friendly Streamlit web application was developed to allow healthcare professionals or patients to input medical details and receive real-time risk assessments, probabilities, and confidence scores.

<div style="page-break-after: always;"></div>

## Table of Contents
*(Page iv)*  
1. Chapter 1: Introduction
2. Chapter 2: Literature Review
3. Chapter 3: Methodology
4. Chapter 4: Results and Evaluation
5. Chapter 5: Conclusion
6. References
7. Appendix

<div style="page-break-after: always;"></div>

## List of Figures
*(Page v)*  
- Figure 1: Random Forest Architecture  
- Figure 2: Confusion Matrix for Random Forest  
- Figure 3: ROC Curve Comparison  
- Figure 4: Feature Importance Chart  

## List of Abbreviations
*(Page vi)*  
- **AI:** Artificial Intelligence  
- **ML:** Machine Learning  
- **RF:** Random Forest  
- **ROC:** Receiver Operating Characteristic  
- **AUC:** Area Under the Curve  
- **RMSE:** Root Mean Square Error  

<div style="page-break-after: always;"></div>

---
# MAIN CHAPTERS
---

<div align="center">
  
# Chapter 1: Introduction
  
</div>

### 1.1 Background and Motivation
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early diagnosis of heart disease can significantly reduce the mortality rate by enabling timely medical interventions. However, diagnosing heart disease based on clinical parameters is a complex task that requires high expertise. The motivation behind this project is to leverage Artificial Intelligence (AI) and Machine Learning (ML) to analyze complex medical data and provide an automated, highly accurate prediction system that can act as a decision-support tool for medical practitioners.

### 1.2 Problem Statement and Scope
Despite advancements in healthcare, misdiagnosis of heart disease remains a significant challenge. Medical professionals must analyze multiple factors such as blood pressure, cholesterol levels, age, and electrocardiographic results simultaneously. The problem addressed in this coursework is the need for a reliable, automated predictive model that can assess these multi-dimensional factors accurately. The scope of this project is limited to tabular medical datasets and focuses on classifying patients into two categories: presence or absence of heart disease. It includes building a web-based prototype for ease of use but does not extend to real-time integration with hospital hardware systems.

### 1.3 Objective
1. To develop a machine learning pipeline that automatically preprocesses medical data and trains predictive models.
2. To implement and compare a Random Forest Classifier against baseline models (Logistic Regression, Decision Tree, Naive Bayes) to achieve high predictive accuracy.
3. To design and deploy a user-friendly, interactive web application that provides real-time risk probability and model interpretability.

<div style="page-break-after: always;"></div>

<div align="center">
  
# Chapter 2: Literature Review
  
</div>

*[Note: Write this section in paragraphs. Summarize 2–4 related research papers. Discuss approaches they used (e.g., SVM, Neural Networks, k-NN) and identify gaps (e.g., lack of interpretability, poor UI) that inspired your project. Remember to use APA 7 citation format consistently.]*

Heart disease prediction using machine learning has been heavily researched in recent years. For instance, [Author 1] (Year) utilized Support Vector Machines to classify clinical records, achieving high accuracy but lacking interpretability for medical staff. Similarly, [Author 2] (Year) explored Deep Neural Networks for cardiovascular prediction; however, their approach required massive computational resources and was prone to overfitting on smaller datasets. 

A common gap identified in previous studies is the lack of deployable, user-friendly interfaces accompanying the models, making them inaccessible to non-technical users. Furthermore, many studies do not provide feature importance analysis, which is critical in healthcare to build trust. This project addresses these gaps by utilizing a Random Forest model—known for its balance of high accuracy and high interpretability—and wrapping it in an accessible Streamlit web application.

<div style="page-break-after: always;"></div>

<div align="center">
  
# Chapter 3: Methodology
  
</div>

### 3.1 Dataset and Preprocessing
- **Dataset name and source:** Cleveland Heart Disease Dataset (or specify your chosen dataset), sourced from the UCI Machine Learning Repository / Kaggle.
- **Number of records and features:** [Fill in total rows], utilizing [Fill in number] features (e.g., age, sex, cholesterol, blood pressure) to predict the binary target variable.
- **Data cleaning and normalization:** The preprocessing pipeline is entirely automated. Numerical features containing missing values are imputed using the median strategy to avoid outlier bias, followed by standardization using `StandardScaler` to ensure all numerical features have a mean of 0 and standard deviation of 1. Categorical features are imputed using the most frequent value (mode) and encoded using `OneHotEncoder` to transform string attributes into machine-readable numerical formats.
- **Splitting into training, testing, and validation sets:** To ensure robust evaluation, the data is split into 60% Training Data, 20% Validation Data, and 20% Testing Data using a randomized split with a fixed random state for reproducibility. 

### 3.2 Algorithm Explanation
**Algorithm Name:** Random Forest Classifier

**Step-by-step explanation:**
1. **Bootstrapping:** The algorithm selects random subsets of the training data with replacement.
2. **Decision Tree Creation:** For each subset, a Decision Tree is built. At each node in the tree, only a random subset of features is considered for splitting the data.
3. **Voting:** Once all trees (e.g., 100 trees) are built, they independently predict the outcome for a given patient.
4. **Aggregation:** The final prediction is determined by majority voting across all trees.

**Intuition and logic:**
Random Forest is an ensemble learning method. A single Decision Tree is prone to overfitting (memorizing the training data). By creating a "forest" of many varied trees and averaging their predictions, the Random Forest minimizes variance and improves generalization. 

**Example:**
If Patient A is inputted into a forest of 100 trees: 80 trees predict "Heart Disease" and 20 trees predict "No Heart Disease". The model outputs "Heart Disease" with an 80% probability (Confidence).

### 3.3 Comparative Survey of AI Approaches
| Algorithm | Accuracy Potential | Interpretability | Scalability | Uncertainty Handling |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Moderate | Very High (Weights) | High | Outputs probabilities natively |
| **Decision Tree** | Moderate (Overfits) | High (Visualizable) | High | Poor (confident but wrong) |
| **Naive Bayes** | Moderate | Moderate | Very High | Generative probabilities |
| **Random Forest** | **High** | **Moderate (Feature Importance)** | **Moderate** | **High (Voting ensemble)** |

**Justified Choice:** Random Forest was chosen as the primary algorithm because healthcare data often contains non-linear relationships. While Neural Networks offer high accuracy, they act as "black boxes." Random Forest provides a perfect equilibrium—offering state-of-the-art accuracy while maintaining interpretability through feature importance charts, which is essential for medical professionals.

### 3.4 Model Training and Hyperparameter Tuning
**Key Parameters:**
- `n_estimators`: The number of trees in the forest (e.g., 100). More trees increase performance but slow down computation.
- `max_depth`: The maximum depth of each tree. Constraining this prevents the model from overfitting.
- `random_state`: Set to 42 to ensure reproducibility.

**Selection Method:**
Hyperparameters were evaluated using the Validation set (20% of the data). We utilized manual testing (or describe GridSearchCV if you add it later) by iterating through different values of `n_estimators` (e.g., 50, 100, 200) and observing the impact on Validation Accuracy to select the optimal parameters before final evaluation on the unseen Test set.

<div style="page-break-after: always;"></div>

<div align="center">
  
# Chapter 4: Results and Evaluation
  
</div>

### 4.1 Metrics Used
- **Accuracy:** The ratio of correctly predicted observations to total observations. Used as a general overview of performance.
- **Precision:** The ratio of correctly predicted positive observations to total predicted positives. Critical in healthcare to avoid false alarms.
- **Recall (Sensitivity):** The ratio of correctly predicted positive observations to all actual positives. Highly crucial in heart disease prediction to ensure we do not miss a sick patient (minimizing false negatives).
- **F1-Score:** The weighted average of Precision and Recall. Used because medical datasets are often imbalanced.
- **ROC AUC:** Represents the model's ability to distinguish between the two classes across different thresholds.

### 4.2 Results Table
*[Note: Run train_model.py on your final dataset to get these numbers, then fill in this table]*

| Model | Val Accuracy | Test Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.85 | 0.84 | 0.82 | 0.88 | 0.85 | 0.90 |
| Decision Tree | 0.76 | 0.75 | 0.74 | 0.78 | 0.76 | 0.75 |
| Naive Bayes | 0.81 | 0.80 | 0.83 | 0.77 | 0.80 | 0.86 |
| **Random Forest** | **0.88** | **0.87** | **0.86** | **0.90** | **0.88** | **0.93** |

### 4.3 Visualizations
*(Note: Insert the images saved in your `images/` folder here using Word. Label them as Figure 1, Figure 2, etc., below the image).*
- **Confusion Matrix:** Shows exactly how many True Positives, True Negatives, False Positives, and False Negatives the Random Forest generated.
- **ROC Curve:** Demonstrates the trade-off between the true positive rate and false positive rate.
- **Feature Importance Chart:** A bar chart revealing which medical attributes (e.g., age, cholesterol) heavily influenced the model's decision-making process.

### 4.4 Interpretation
The Random Forest model outperformed the baseline algorithms across all metrics. Notably, the high Recall score indicates that the model is highly capable of identifying patients who actually have heart disease, minimizing dangerous false negatives. The feature importance analysis revealed that [Feature 1] and [Feature 2] were the most critical indicators, which aligns with existing medical literature. 

<div style="page-break-after: always;"></div>

<div align="center">
  
# Chapter 5: Conclusion
  
</div>

### 5.1 Social and Legal Implications
The deployment of AI in healthcare raises important social and legal questions. Socially, this tool democratizes access to preliminary cardiovascular risk assessments, potentially saving lives in remote areas with limited access to cardiologists. Legally, the system must adhere to data privacy laws (e.g., GDPR, HIPAA) ensuring patient data processed by the Streamlit application is not maliciously intercepted. Furthermore, there is the issue of liability: if the AI misdiagnoses a patient, determining responsibility remains a complex legal gray area.

### 5.2 Human Factors
For an AI system to be adopted in healthcare, human-computer interaction must be seamless. The Streamlit interface was designed specifically with non-technical users in mind, abstracting the complex Random Forest mathematical backend into a simple form. Furthermore, by categorizing risk into "High", "Moderate", and "Low" alongside a confidence percentage, the system communicates uncertainty effectively, preventing automation bias where doctors blindly trust the machine.

### 5.3 Summary and Recommendation
**Summary:** This project successfully developed an AI-based Heart Disease Prediction system using a 60-20-20 data split. The Random Forest Classifier proved to be the most robust approach, outperforming baseline models while retaining interpretability. 
**Strengths:** Highly modular codebase, robust dynamic preprocessing pipeline, and an interactive UI.
**Limitations:** The model relies solely on tabular clinical data and does not account for lifestyle changes over time. 
**Recommendations:** Future work should integrate time-series patient data, implement Deep Learning techniques (if massive datasets become available), and connect the application to a secure hospital SQL database.

**Links:**
- **GitHub Repository:** [Insert Link]
- **Project Presentation Video:** [Insert Link]

<div style="page-break-after: always;"></div>

<div align="center">
  
# References
  
</div>

*(Note: Replace with your actual sources in APA 7 format)*
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., ... & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology*, 64(5), 304-310.
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

<div style="page-break-after: always;"></div>

<div align="center">
  
# Appendix
  
</div>

### Appendix A: Selected Code Snippet (Data Splitting)
```python
# 4. Train/Validation/Test Split (60/20/20)
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)
```

### Appendix B: Selected Code Snippet (Model Training Pipeline)
```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])
pipeline.fit(X_train, y_train)
```

### Appendix C: UI Screenshot Placeholder
*(Insert screenshot of the Streamlit app interface here)*
