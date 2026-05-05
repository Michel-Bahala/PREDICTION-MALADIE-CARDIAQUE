🚀 Heart Disease Prediction App (ML + Streamlit)

🌐 Live Demo

🚀 Try the app here:

👉 Streamlit Web App Link : https://prediction-maladie-cardiaque-j4kgpzbbv35uasc8dejf4o.streamlit.app/

🧠 Overview

This project focuses on the early detection of heart disease risk by leveraging real-world health data from the 2015 BRFSS dataset (229,781 records).

In the healthcare sector, failing to identify a high-risk patient can have severe consequences. Consequently, this model is optimized for high Recall (Sensitivity) to minimize false negatives.

🎯 Key Features

End-to-End ML Pipeline: From data cleaning to cloud deployment.

Advanced Feature Engineering: Creation of specialized medical indicators.

Class Imbalance Management: Implementation of balanced dataset strategies.

Hyperparameter Tuning: Systematic optimization using Optuna.

🧪 Model Performance (Test Set)

The model demonstrates robust results, particularly in its detection capability:

MetricScore
Recall81.84% ✅
Accuracy74.82%
Precision71.60%
F1-Score76.38%
ROC-AUC0.827

📌 Key Insight: The Age × BMI interaction emerged as the most influential feature, highlighting the power of domain-specific feature engineering.

🔬 Feature Engineering Highlights

The model's success is driven by synthetic variables that better reflect medical realities:

Age × BMI: The impact of weight increases significantly with age.

Alcohol × Smoking: Analyzing the synergistic effect of high-risk behaviors.

Income × Hypertension: Exploring correlations between socio-economic factors and cardiovascular health.

BMI Transformations: Using logarithmic and squared transformations to capture non-linear relationships.

🖥️ Streamlit Interactive Demo

An interactive web application developed with Streamlit allows users to easily engage with the model.

Patient Metrics Input: User-friendly form for health indicators.

Risk Prediction: Real-time heart disease risk calculation.

Probability & Interpretation: Clear display of results and medical context.

⚙️ Technologies Used

Language: Python 🐍

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-learn, LogisticRegression

Optimization: Optuna

Visualization: Matplotlib, Seaborn

Deployment: Streamlit

▶️ Local Execution

Bash



# Clone the repository

git clone https://github.com/Michel-Bahala/PREDICTION-MALADIE-CARDIAQUE.git# Navigate to the project foldercd PREDICTION-MALADIE-CARDIAQUE# Install dependencies

pip install -r requirements.txt# Run the application

streamlit run streamlit_app.py
